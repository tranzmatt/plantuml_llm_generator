"""
Accuracy evaluation (DESIGN.md 17.1): how well the generated documentation
matches a reference, and how faithful each model's output is to the code.

Two parts:

  structure  Compares the service-to-service flows archdoc extracts with a
             reference diagram (Mermaid `graph`), node names mapped to
             services by name or by the spec. Each flow is matched, missing
             or extra, with the code evidence (channel, and whether it is
             deployed). No model is involved.

  models     Runs the same sequence and narrative requests on each backend
             and measures, against the code rather than a reference: answers
             valid on the first attempt, after one repair, or not at all;
             share of proposed messages that verification kept; share of the
             call graph's edges each diagram covers; names corrected; tokens
             and time. Same entries, same payload policy, every backend.

The reference is evidence too, not an oracle: a flow it shows that the code
lacks may be a documentation error, an unfinished refactor, or code outside
the repo. The report lists each one so a person can decide.
"""

from __future__ import annotations

import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .deployment import deployed_modules
from .egress import participant_of
from .interfaces import Interface
from .ir import Deployment, RepoIR
from .messaging import RUNTIME, channels

BROKER = "@broker"
EXTERNAL = "@external"

# ---------------------------------------------------------------------------
# Reference diagrams
# ---------------------------------------------------------------------------

_NODE_RE = re.compile(r"([A-Za-z_][\w]*)\s*(?:\[\(?\"?([^\]\")]*)\"?\)?\]|\(\(?\"?([^)\"]*)\"?\)?\)|\{\"?([^}\"]*)\"?\})?")
_EDGE_RE = re.compile(r"^\s*(.+?)\s*(-->|---|-.->|==>)\s*(?:\|([^|]*)\|\s*)?(.+?)\s*;?\s*$")


@dataclass
class Graph:
    nodes: Dict[str, str] = field(default_factory=dict)            # id -> label
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (from id, to id, label)


def _node(text: str, g: Graph) -> str:
    m = _NODE_RE.match(text.strip())
    if not m:
        return text.strip()
    nid = m.group(1)
    lab = next((x for x in m.groups()[1:] if x), None)
    if lab:
        g.nodes[nid] = lab.strip()
    else:
        g.nodes.setdefault(nid, nid)
    return nid


def parse_mermaid(text: str) -> Graph:
    """Nodes and edges of a Mermaid flowchart (`graph` / `flowchart`)."""
    g = Graph()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%%") or re.match(r"^(graph|flowchart)\b", s):
            continue
        m = _EDGE_RE.match(s)
        if m:
            a, b = _node(m.group(1), g), _node(m.group(4), g)
            g.edges.append((a, b, (m.group(3) or "").strip()))
        else:
            _node(s, g)
    return g


def _words(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"^\s*\d+\.\s*", "", text.lower()))
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def match_nodes(labels: Dict[str, str], services: List[str], explicit: Dict[str, str]) -> Dict[str, Optional[str]]:
    """reference node id -> service name, BROKER, EXTERNAL, or None (unmatched)."""
    out: Dict[str, Optional[str]] = {}
    for nid, lab in labels.items():
        if lab in explicit or nid in explicit:
            out[nid] = explicit.get(lab, explicit.get(nid))
            continue
        lw = _words(lab)
        best, score = None, 0.0
        for s in services:
            sw = _words(s.replace("_", " ").replace("-", " "))
            j = len(lw & sw) / len(lw | sw) if lw | sw else 0.0
            if j > score:
                best, score = s, j
        out[nid] = best if score >= 0.5 else None
    return out


def reference_flows(g: Graph, mapping: Dict[str, Optional[str]],
                    extra: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Service -> service flows the reference shows.

    Edges through the broker node are paired by shared words in their labels
    ("Celebrity Name" in, "Name" out); the pairing is reported so it can be
    checked, and `extra` adds flows whose labels share no word.
    """
    notes: List[str] = []
    flows: List[Dict[str, str]] = []
    hubs = {n for n, s in mapping.items() if s == BROKER}
    ins = [(a, lab) for a, b, lab in g.edges if b in hubs]
    outs = [(b, lab) for a, b, lab in g.edges if a in hubs]
    for a, b, lab in g.edges:
        if a not in hubs and b not in hubs:
            flows.append({"from": a, "to": b, "label": lab, "how": "direct edge"})
    used_in, used_out = set(), set()
    for a, la in ins:
        for b, lb in outs:
            shared = _words(la) & _words(lb)
            if a != b and shared:
                used_in.add((a, la))
                used_out.add((b, lb))
                flows.append({"from": a, "to": b, "label": f"{la} / {lb}",
                              "how": f"paired through the broker on {', '.join(sorted(shared))!r}"})
    by_label = {lab: nid for nid, lab in g.nodes.items()}
    for f in extra:
        a, b = by_label.get(f["from"], f["from"]), by_label.get(f["to"], f["to"])
        flows.append({"from": a, "to": b, "label": f.get("label", ""), "how": "from the spec"})
    stated_in = {f["from"] for f in flows if f["how"] == "from the spec"}
    stated_out = {f["to"] for f in flows if f["how"] == "from the spec"}
    for a, la in ins:
        if (a, la) not in used_in and a not in stated_in:
            notes.append(f"broker input from {g.nodes.get(a, a)!r} ({la!r}) pairs with no output")
    for b, lb in outs:
        if (b, lb) not in used_out and b not in stated_out:
            notes.append(f"broker output to {g.nodes.get(b, b)!r} ({lb!r}) pairs with no input")
    return flows, notes


# ---------------------------------------------------------------------------
# Structure: reference flows vs code flows
# ---------------------------------------------------------------------------

def _service_of_path(dep: Deployment) -> Dict[str, str]:
    return {s.build: s.name for s in dep.services if s.build}


def _service_resolver(ir: RepoIR, dep: Deployment):
    placed = deployed_modules(dep)
    contexts = _service_of_path(dep)
    paths = {m.id: m.path for m in ir.modules}

    def service(mod: str) -> Optional[str]:
        if mod in placed:
            return placed[mod]
        p = paths.get(mod, "")
        hits = [c for c in contexts if c == "." or p.startswith(c.rstrip("/") + "/")]
        return contexts[max(hits, key=len)] if hits else None
    return service


def _endpoints(ir: RepoIR, dep: Deployment, svc_name: str, role: str) -> List[str]:
    """What one service publishes to or consumes from, noting channels with no other end."""
    service = _service_resolver(ir, dep)
    out = []
    for c in channels(ir):
        mine = c.publishers if role == "publish" else c.consumers
        other = c.consumers if role == "publish" else c.publishers
        if any(service(e.module) == svc_name for e in mine):
            lonely = "" if other else (" (no consumer)" if role == "publish" else " (no publisher)")
            out.append(f"{c.target} {c.name}{lonely}")
    return out


def code_flows(ir: RepoIR, dep: Deployment) -> List[Dict[str, Any]]:
    """Publisher service -> consumer service per channel, for code in the repo,
    each marked deployed or not (a module is attributed to the service whose
    build context holds it, even when the image leaves it out)."""
    placed = deployed_modules(dep)
    service = _service_resolver(ir, dep)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in channels(ir):
        if c.name == RUNTIME:
            continue
        for p in c.publishers:
            for q in c.consumers:
                a, b = service(p.module), service(q.module)
                if not a or not b or a == b:
                    continue
                f = out.setdefault((a, b), {"from": a, "to": b, "channels": [], "deployed": False, "modules": set()})
                f["channels"].append(f"{c.target} {c.name}")
                f["modules"].update({p.module, q.module})
                if p.module in placed and q.module in placed:
                    f["deployed"] = True
    for f in out.values():
        f["channels"] = sorted(set(f["channels"]))
        f["modules"] = sorted(f["modules"])
    return [out[k] for k in sorted(out)]


def evaluate_structure(ir: RepoIR, repo_root: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    truth = spec.get("truth", {})
    with open(os.path.join(repo_root, truth["mermaid"]), encoding="utf-8") as fh:
        g = parse_mermaid(fh.read())
    deps = ir.deployments
    if truth.get("deployment"):
        deps = [d for d in deps if d.file == truth["deployment"]]
    if not deps:
        raise ValueError("no deployment file to compare against (set truth.deployment)")
    dep = max(deps, key=lambda d: len(d.services))
    services = [s.name for s in dep.services]
    mapping = match_nodes(g.nodes, services, truth.get("nodes", {}))
    ref, notes = reference_flows(g, mapping, truth.get("flows", []))

    def svc(nid: str) -> Optional[str]:
        return mapping.get(nid)

    code = code_flows(ir, dep)
    code_by_pair = {(f["from"], f["to"]): f for f in code}
    rows, ref_pairs = [], set()
    for f in ref:
        a, b = svc(f["from"]), svc(f["to"])
        if a in (None, EXTERNAL, BROKER) or b in (None, EXTERNAL, BROKER):
            rows.append({**f, "status": "not comparable",
                         "why": "an end is external or unmatched", "from_service": a, "to_service": b})
            continue
        ref_pairs.add((a, b))
        cf = code_by_pair.get((a, b))
        if cf and cf["deployed"]:
            status, why = "matched", ", ".join(cf["channels"])
        elif cf:
            status, why = "in code, not deployed", f"{', '.join(cf['channels'])} via {', '.join(cf['modules'])}"
        else:
            pubs = _endpoints(ir, dep, a, "publish") or ["nothing"]
            subs = _endpoints(ir, dep, b, "consume") or ["nothing"]
            status, why = "missing", (f"no channel connects them: {a} publishes to {', '.join(pubs)}; "
                                      f"{b} consumes {', '.join(subs)}")
        rows.append({**f, "from_service": a, "to_service": b, "status": status, "why": why})
    extra = [{"from_service": f["from"], "to_service": f["to"], "why": ", ".join(f["channels"]),
              "deployed": f["deployed"]}
             for f in code if (f["from"], f["to"]) not in ref_pairs and f["deployed"]]
    comparable = [r for r in rows if r["status"] != "not comparable"]
    matched = sum(1 for r in comparable if r["status"] == "matched")
    deployed_code = sum(1 for f in code if f["deployed"])
    return {
        "reference": truth["mermaid"],
        "deployment": dep.file,
        "nodes": {g.nodes[n]: m for n, m in mapping.items()},
        "pairing_notes": notes,
        "flows": rows,
        "extra_in_code": extra,
        "recall": matched / len(comparable) if comparable else None,
        "precision": matched / deployed_code if deployed_code else None,
    }


# ---------------------------------------------------------------------------
# Models: faithfulness of each backend's output to the code
# ---------------------------------------------------------------------------

def call_coverage(ir: RepoIR, plan: Dict[str, Any], ids: List[str]) -> Tuple[int, int]:
    """(call-graph edges the diagram shows, call-graph edges in the slice)."""
    fns = {f: ir.find_function(f) for f in ids}
    edges = {(f, c) for f, fn in fns.items() if fn for c in fn.calls if c in fns and c != f}
    shown = set()
    for m in plan["messages"]:
        call = m.get("call")
        if not call:
            continue
        for f, c in edges:
            if c == call and participant_of(f) == m["from"]:
                shown.add((f, c))
    return len(shown), len(edges)


@dataclass
class ModelRun:
    backend: str
    model: str
    tier: str
    requests: int = 0
    first_attempt: int = 0
    repaired: int = 0
    failed: int = 0
    proposed_messages: int = 0
    kept_messages: int = 0
    removed_groups: int = 0
    edges_shown: int = 0
    edges_total: int = 0
    corrected_names: int = 0
    cleared_labels: int = 0
    narrative_items: int = 0
    narrative_slots: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        pct = lambda a, b: round(100 * a / b) if b else None   # noqa: E731
        return {
            "backend": self.backend, "model": self.model, "tier": self.tier,
            "requests": self.requests,
            "valid_first_attempt_pct": pct(self.first_attempt, self.requests),
            "valid_after_repair": self.repaired, "failed": self.failed,
            "messages_kept_pct": pct(self.kept_messages, self.proposed_messages),
            "call_edges_shown_pct": pct(self.edges_shown, self.edges_total),
            "removed_groups": self.removed_groups,
            "corrected_names": self.corrected_names, "cleared_labels": self.cleared_labels,
            "narrative_filled_pct": pct(self.narrative_items, self.narrative_slots),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "minutes": round(self.seconds / 60, 1),
        }


def _count(result, run: ModelRun) -> None:
    run.requests += 1
    run.prompt_tokens += result.prompt_tokens
    run.completion_tokens += result.completion_tokens
    if not result.ok:
        run.failed += 1
    elif result.attempts == 1:
        run.first_attempt += 1
    else:
        run.repaired += 1
    for w in result.warnings:
        if w.startswith("corrected misspelled name"):
            run.corrected_names += 1
        m = re.match(r"cleared (\d+) label", w)
        if m:
            run.cleared_labels += int(m.group(1))


def evaluate_model(ctx, entries, interfaces: List[Interface], tier: str) -> ModelRun:
    from .plan.artifacts import plan_narrative, plan_sequence
    from .plan.entries import call_slice

    run = ModelRun(ctx.backend.name, ctx.backend.model or "?", tier)
    t_start = time.monotonic()
    for e in entries:
        t0 = time.monotonic()
        result = plan_sequence(ctx, e)
        _count(result, run)
        detail: Dict[str, Any] = {"artifact": f"sequence {e.handler}", "ok": result.ok,
                                  "attempts": result.attempts, "seconds": round(time.monotonic() - t0, 1)}
        if result.ok:
            removed = result.report.get("removed", [])
            removed_msgs = sum(1 for r in removed if r.startswith("removed messages"))
            run.removed_groups += sum(1 for r in removed if r.startswith("removed groups"))
            kept = result.report.get("messages", 0)
            run.kept_messages += kept
            run.proposed_messages += kept + removed_msgs
            shown, total = call_coverage(ctx.ir, result.plan, call_slice(ctx.ir, e.handler, ctx.depth, ctx.max_nodes))
            run.edges_shown += shown
            run.edges_total += total
            detail.update(kept=kept, removed=removed, edges=f"{shown}/{total}", warnings=result.warnings)
        else:
            detail["errors"] = result.errors[:3]
        run.details.append(detail)
    for iface in interfaces:
        t0 = time.monotonic()
        result = plan_narrative(ctx, iface)
        _count(result, run)
        slots = 1 + len(iface.operations) + sum(len(o.params) for o in iface.operations)
        run.narrative_slots += slots
        filled = 0
        if result.ok:
            plan = result.plan
            filled = bool(plan.get("purpose")) + sum(
                bool(o.get("summary")) + sum(1 for p in o.get("params", []) if p.get("description"))
                for o in plan.get("operations", []))
        run.narrative_items += min(filled, slots)
        run.details.append({"artifact": f"narrative {iface.id}", "ok": result.ok, "attempts": result.attempts,
                            "filled": f"{filled}/{slots}", "seconds": round(time.monotonic() - t0, 1)})
    run.seconds = time.monotonic() - t_start
    return run


# ---------------------------------------------------------------------------
# Spec and report
# ---------------------------------------------------------------------------

def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def report_markdown(structure: Optional[Dict[str, Any]], runs: List[Dict[str, Any]], repo: str) -> str:
    out = [f"# archdoc evaluation: {repo}", ""]
    if structure:
        s = structure
        pct = lambda x: "n/a" if x is None else f"{round(100 * x)}%"   # noqa: E731
        out += ["## Structure against the reference", "",
                f"Reference `{s['reference']}`, compared with the code deployed by `{s['deployment']}`.", "",
                f"- **Recall** (reference flows found deployed in the code): {pct(s['recall'])}",
                f"- **Precision** (deployed code flows the reference shows): {pct(s['precision'])}", "",
                "| Reference node | Mapped to |", "|---|---|"]
        out += [f"| {k} | {v or '**unmatched**'} |" for k, v in s["nodes"].items()]
        out += ["", "| From | To | Reference label | Status | Evidence |", "|---|---|---|---|---|"]
        for r in s["flows"]:
            out.append(f"| {r.get('from_service') or r['from']} | {r.get('to_service') or r['to']} | "
                       f"{r['label']} | {r['status']} | {r['why']} |")
        if s["extra_in_code"]:
            out += ["", "Flows in the deployed code that the reference does not show:", ""]
            out += [f"- {e['from_service']} -> {e['to_service']} ({e['why']})" for e in s["extra_in_code"]]
        if s["pairing_notes"]:
            out += ["", "Reference edges through the broker that could not be paired (add them under "
                        "`[[truth.flows]]` if they are flows):", ""]
            out += [f"- {n}" for n in s["pairing_notes"]]
        how = sorted({r["how"] for r in s["flows"]})
        out += ["", "How reference flows were read: " + "; ".join(how) + "."]
    if runs:
        cols = [("backend", "Backend"), ("model", "Model"), ("tier", "Tier"), ("requests", "Requests"),
                ("valid_first_attempt_pct", "Valid 1st try %"), ("valid_after_repair", "Repaired"),
                ("failed", "Failed"), ("messages_kept_pct", "Messages kept %"),
                ("call_edges_shown_pct", "Call edges shown %"), ("removed_groups", "Groups removed"),
                ("corrected_names", "Names corrected"), ("cleared_labels", "Labels cleared"),
                ("narrative_filled_pct", "Narrative filled %"), ("completion_tokens", "Output tokens"),
                ("minutes", "Minutes")]
        out += ["", "## Models", "",
                "Measured against the code: kept = messages that passed verification; call edges = "
                "share of the call graph each diagram shows.", "",
                "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
        for r in runs:
            out.append("| " + " | ".join("—" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"
