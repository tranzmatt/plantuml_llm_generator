"""
Sequence diagram from a validated, restored plan (see archdoc.plan.schemas).

The plan's refs are real ids by now. Names shown come from the IR, never from
the model; the model contributes order, labels, kinds and grouping only.
Groups that would overlap without nesting are dropped (PlantUML needs proper
nesting), and reported back as warnings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...ir import RepoIR
from .common import Aliases, Diagram, label

CLIENT = "client"


def participant_name(ir: RepoIR, ref: str, client_name: str) -> str:
    if ref == CLIENT:
        return client_name
    if ":" in ref:
        cls = ir.find_class(ref)
        return cls.name if cls else ref.split(":", 1)[1]
    return ref                                       # module id


def _message_text(ir: RepoIR, msg: Dict[str, Any]) -> str:
    text = (msg.get("label") or "").strip()
    if not text and msg.get("call"):
        fn = ir.find_function(msg["call"])
        text = f"{fn.name}()" if fn else msg["call"].rsplit(".", 1)[-1]
    return " ".join(text.replace(":", " -").split())   # a ':' would end the label


def nested_groups(groups: List[Dict[str, Any]], n_messages: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Keep groups that nest properly (outer first); drop crossing ones."""
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for g in sorted(groups, key=lambda g: (g["first"], -g["last"])):
        if not (0 <= g["first"] <= g["last"] < n_messages):
            dropped.append(f"group {g['label']!r}: out of range")
            continue
        crosses = any(k["first"] < g["first"] <= k["last"] < g["last"] for k in kept)
        if crosses:
            dropped.append(f"group {g['label']!r}: overlaps another group without nesting")
            continue
        kept.append(g)
    return kept, dropped


def render_sequence_diagram(ir: RepoIR, plan: Dict[str, Any], name: str, client_name: str = "Client",
                            title: Optional[str] = None) -> Tuple[str, List[str]]:
    """`title` is used when the plan has none. An empty message label falls
    back to the called function's name."""
    aliases = Aliases()
    d = Diagram(name, plan.get("title") or title or None,
                settings=["autonumber", "skinparam responseMessageBelowArrow true"])
    declared = set()
    implicit = [{"ref": r, "kind": "participant"}
                for m in plan["messages"] for r in (m["from"], m["to"])]
    for p in list(plan["participants"]) + implicit:
        ref, kind = p["ref"], p.get("kind") or "participant"
        if ref in declared:
            continue
        declared.add(ref)
        d.add(f"{kind} {label(participant_name(ir, ref, client_name))} as {aliases(ref, 'P')}")
    d.add("")

    messages = plan["messages"]
    groups, warnings = nested_groups(plan.get("groups", []), len(messages))
    opens: Dict[int, List[Dict[str, Any]]] = {}
    closes: Dict[int, int] = {}
    for g in groups:
        opens.setdefault(g["first"], []).append(g)
        closes[g["last"]] = closes.get(g["last"], 0) + 1

    for i, m in enumerate(messages):
        for g in opens.get(i, []):
            d.add(f"{g['kind']} {_message_text(ir, {'label': g['label']}) or g['kind']}")
        src, dst = aliases(m["from"], "P"), aliases(m["to"], "P")
        arrow = "-->" if m.get("reply") else "->"
        text = _message_text(ir, m)
        d.add(f"{src} {arrow} {dst}" + (f" : {text}" if text else ""))
        for _ in range(closes.get(i, 0)):
            d.add("end")
    return d.text(), warnings
