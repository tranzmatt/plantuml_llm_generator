"""
Cross-service message flows, traced deterministically from the code.

Starting from an operation a user or client invokes (a UI event, an HTTP
route, a CLI command), the trace follows the handler's internal calls in
source order to each publish, then through the broker to every deployed
consumer of that queue or exchange, and on from each consumer's callback.
Every step is a fact the extractor found: a call, a publish, a consume. The
trace stops where the code does: a publish nobody consumes is recorded as
such, with any consumer in the repo that the deployment leaves out.

Limits, stated in every diagram's legend: the order of steps within one
function is source order, not run-time order (branches, retries and loops
are not modelled); a direct exchange with a run-time routing key is assumed
to reach every queue bound to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .deployment import deployed_modules
from .interfaces import Interface, Operation
from .ir import Deployment, RepoIR
from .messaging import RUNTIME, Channel, End, channels

ENTRY_KINDS = ("ui", "http", "cli")
MAX_DEPTH = 10
MAX_STEPS = 80


@dataclass
class Step:
    kind: str                          # invoke | publish | deliver | unconsumed | par | else | end | truncated
    src: Optional[str] = None          # participant (service name), "@client" or "@broker"
    dst: Optional[str] = None
    text: str = ""
    channel: Optional[str] = None      # "queue x" / "exchange y"
    fields: List[str] = field(default_factory=list)
    evidence: Optional[str] = None     # operation or function id the step comes from


@dataclass
class Flow:
    entry: Tuple[Interface, Operation]
    steps: List[Step] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)   # services in order of appearance
    broker: Optional[str] = None


class _Tracer:
    def __init__(self, ir: RepoIR, dep: Optional[Deployment]):
        self.ir = ir
        self.fns = {f.id: f for f in ir.iter_functions()}
        self.chans = [c for c in channels(ir) if c.name != RUNTIME]
        self.placed = deployed_modules(dep) if dep else None
        groups = {s.build: s.name for s in dep.services if s.build} if dep else {}
        self.paths = {m.id: m.path for m in ir.modules}
        self.groups = groups
        # publishes by the function that performs them
        self.publishes: Dict[str, List[Tuple[End, Channel]]] = {}
        for c in self.chans:
            for e in c.publishers:
                if e.handler:
                    self.publishes.setdefault(e.handler, []).append((e, c))
        self.broker_service = None
        if dep:
            brokers = {c.broker.lower() for c in self.chans}
            self.broker_service = next((s.name for s in dep.services
                                        if s.image and any(b in s.image.lower() for b in brokers)), None)

    def service(self, module: str) -> Optional[str]:
        """Service running `module`; None if a deployment is given and leaves it out."""
        if self.placed is None:
            return module.split(".")[0]
        return self.placed.get(module)

    def repo_service(self, module: str) -> str:
        """Service whose build context holds `module`, deployed or not (for notes)."""
        p = self.paths.get(module, "")
        hits = [c for c in self.groups if c == "." or p.startswith(c.rstrip("/") + "/")]
        return self.groups[max(hits, key=len)] if hits else module.split(".")[0]

    def events(self, fn: str, seen: Set[str], depth: int = 0) -> List[Tuple[End, Channel]]:
        """Publishes reachable from `fn`, in source order through its internal calls."""
        if fn in seen or depth > MAX_DEPTH or fn not in self.fns:
            return []
        seen.add(fn)
        f = self.fns[fn]
        items: List[Tuple[int, int, object]] = []
        for e, c in self.publishes.get(fn, []):
            items.append((e.line or 0, 0, (e, c)))
        for callee in f.calls:
            if callee in self.fns:
                items.append((f.call_lines.get(callee, 0), 1, callee))
        out: List[Tuple[End, Channel]] = []
        for _, _, item in sorted(items, key=lambda t: (t[0], t[1])):
            if isinstance(item, str):
                out += self.events(item, seen, depth + 1)
            else:
                out.append(item)
        return out


def trace(ir: RepoIR, iface: Interface, op: Operation, dep: Optional[Deployment] = None) -> Flow:
    t = _Tracer(ir, dep)
    flow = Flow((iface, op), broker=t.broker_service or (t.chans[0].broker if t.chans else None))
    owner_svc = t.service(iface.owner) or iface.owner
    seen_participants: List[str] = []

    def see(p: Optional[str]) -> None:
        if p and p not in seen_participants:
            seen_participants.append(p)

    def add(step: Step) -> bool:
        if len(flow.steps) >= MAX_STEPS:
            if not flow.steps or flow.steps[-1].kind != "truncated":
                flow.steps.append(Step("truncated", text=f"trace stopped after {MAX_STEPS} steps"))
            return False
        flow.steps.append(step)
        return True

    see(owner_svc)
    add(Step("invoke", "@client", owner_svc, op.name, evidence=op.id))

    def walk(fn: str, svc: str, active: Set[str], depth: int) -> None:
        if depth > MAX_DEPTH:
            add(Step("truncated", text=f"trace depth limit {MAX_DEPTH} reached"))
            return
        for pub, chan in t.events(fn, set()):
            where = f"{chan.target} {chan.name}"
            label = f"publish {chan.name}" + (f" ({chan.exchange_type} exchange)" if chan.exchange_type else "")
            if pub.routing_key == RUNTIME:
                label += " [routing key set per message]"
            elif pub.routing_key:
                label += f" [{pub.routing_key}]"
            if not add(Step("publish", svc, "@broker", label, where, pub.fields, pub.operation)):
                return
            consumers = [(e, t.service(e.module)) for e in chan.consumers]
            live = [(e, s) for e, s in consumers if s]
            if not live:
                left_out = sorted({t.repo_service(e.module) for e, s in consumers})
                note = f"{chan.name}: no consumer" + (" deployed" if dep else "")
                if left_out:
                    note += f" (consumed in {', '.join(left_out)}, not deployed)"
                add(Step("unconsumed", "@broker", None, note, where))
                continue
            grouped = len(live) > 1
            if grouped:
                add(Step("par", text=f"{chan.name} delivered to each consumer"))
            for i, (e, s) in enumerate(live):
                if grouped and i:
                    add(Step("else"))
                see(s)
                if not add(Step("deliver", "@broker", s, chan.name, where, e.fields, e.operation)):
                    return
                if e.handler and e.handler not in active:
                    walk(e.handler, s, active | {e.handler}, depth + 1)
            if grouped:
                add(Step("end"))

    if op.handler:
        walk(op.handler, owner_svc, {op.handler}, 0)
    flow.participants = seen_participants
    return flow


def flow_entries(ir: RepoIR, dep: Optional[Deployment] = None) -> List[Tuple[Interface, Operation]]:
    """Operations a user or client invokes whose handler leads to at least one publish."""
    t = _Tracer(ir, dep)
    out = []
    for iface in ir.interfaces:
        if iface.kind not in ENTRY_KINDS or iface.direction != "provided":
            continue
        if dep is not None and t.service(iface.owner) is None:
            continue
        for op in iface.operations:
            if op.handler and t.events(op.handler, set()):
                out.append((iface, op))
    return out

