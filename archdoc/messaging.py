"""
Message channels across the whole repo: which modules publish to and consume
from each queue or exchange, joined from the per-module `message` interfaces.

Used by the component diagram (broker nodes with publisher/consumer arrows)
and by coverage, where a channel with no consumer, a channel with no
publisher, or a consumer reading fields no publisher sends is reported: in a
rebuild, each of those is either dead code or an interface outside the repo,
and the documentation has to say which (DESIGN.md 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir import RepoIR

RUNTIME = "(set at run time)"


@dataclass
class End:
    module: str
    operation: str                     # Operation id
    handler: Optional[str]
    routing_key: Optional[str]
    fields: List[str]


@dataclass
class Channel:
    broker: str
    target: str                        # queue | exchange
    name: str
    key: str                           # unique id, e.g. "mq:RabbitMQ:queue:image_queue"
    exchange_type: Optional[str] = None
    publishers: List[End] = field(default_factory=list)
    consumers: List[End] = field(default_factory=list)


def _broker(protocol: Optional[str]) -> str:
    if protocol and "RabbitMQ" in protocol:
        return "RabbitMQ"
    return protocol or "message broker"


def channels(ir: RepoIR) -> List[Channel]:
    found: Dict[str, Channel] = {}
    for iface in ir.interfaces:
        if iface.kind != "message":
            continue
        broker = _broker(iface.protocol)
        for op in iface.operations:
            target = "exchange" if "exchange" in op.meta else "queue"
            name = op.meta.get(target, RUNTIME)
            # a queue named at run time cannot be matched across modules
            key = f"mq:{broker}:{target}:{name}" + (f":{iface.owner}" if name == RUNTIME else "")
            ch = found.setdefault(key, Channel(broker, target, name, key, op.meta.get("exchange_type")))
            ch.exchange_type = ch.exchange_type or op.meta.get("exchange_type")
            end = End(iface.owner, op.id, op.handler, op.meta.get("routing_key"), [p.name for p in op.params])
            (ch.publishers if op.meta.get("role") == "publish" else ch.consumers).append(end)
    return [found[k] for k in sorted(found)]


def wiring_issues(ir: RepoIR) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for c in channels(ir):
        where = f"{c.target} {c.name}"
        if c.name == RUNTIME:
            continue
        if c.publishers and not c.consumers:
            issues.append({"channel": where, "issue": "published but never consumed in this repo",
                           "modules": ", ".join(sorted({e.module for e in c.publishers}))})
        elif c.consumers and not c.publishers:
            issues.append({"channel": where, "issue": "consumed but never published in this repo",
                           "modules": ", ".join(sorted({e.module for e in c.consumers}))})
        sent = {f for e in c.publishers for f in e.fields}
        if not sent or any(not e.fields for e in c.publishers):
            continue                   # some publisher's body is not a literal: fields unknown
        for e in c.consumers:
            missing = [f for f in e.fields if f not in sent]
            if missing:
                issues.append({"channel": where,
                               "issue": f"consumer reads field(s) no publisher sends: {', '.join(missing)}",
                               "modules": e.module})
    return issues
