"""State diagram from archdoc.states: every arrow is an assignment in the code."""

from __future__ import annotations

from typing import Optional

from ...states import Machine
from .common import Aliases, Diagram, label

ANY_LIMIT = 6        # an "any state" transition is drawn from every state up to this many


def render_state_diagram(m: Machine, name: str, title: Optional[str] = None) -> str:
    d = Diagram(name, title or f"State: {m.name}", settings=["hide empty description"])
    a = Aliases()
    for s in m.states:
        d.add(f"state {label(s)} as {a(s, 'S')}")
    for s, where in m.initial:
        d.edge(f"[*] --> {a(s, 'S')}")
    any_alias = None
    for t in m.transitions:
        fn = t.function.split(":")[1]
        if t.src is None and len(m.states) > ANY_LIMIT:
            if any_alias is None:
                any_alias = "S__any"
                d.add(f'state "any state" as {any_alias} #line.dashed')
            d.edge(f"{any_alias} --> {a(t.dst, 'S')} : {label(fn)[1:-1]}")
        elif t.src is None:
            for s in m.states:
                if s != t.dst:
                    d.edge(f"{a(s, 'S')} --> {a(t.dst, 'S')} : {label(fn)[1:-1]}")
        else:
            d.edge(f"{a(t.src, 'S')} --> {a(t.dst, 'S')} : {label(fn)[1:-1]}")
    notes = []
    if m.fields:
        notes.append("held in " + ", ".join(m.fields))
    if m.unreached:
        notes.append("no code assigns " + ", ".join(m.unreached))
    if any(t.src is None for t in m.transitions):
        notes.append("an arrow from every state: the assignment does not check the current state")
    if notes:
        d.add("legend bottom")
        for n in notes:
            d.add(n.replace("\n", " "))
        d.add("endlegend")
    return d.text()


__all__ = ["render_state_diagram"]
