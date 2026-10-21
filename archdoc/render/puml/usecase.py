"""Use-case diagram from archdoc.usecases."""

from __future__ import annotations

from typing import Dict, Optional

from ...usecases import UseCaseModel
from .common import Aliases, Diagram, label


def render_usecase_diagram(model: UseCaseModel, name: str, title: Optional[str] = None,
                           goals: Optional[Dict[str, str]] = None,
                           actor_names: Optional[Dict[str, str]] = None) -> str:
    goals, actor_names = goals or {}, actor_names or {}
    d = Diagram(name, title or f"{model.system}: use cases", settings=["left to right direction"])
    a = Aliases()
    for act in model.actors:
        stereo = "" if act.primary else " <<system>>"
        d.add(f"actor {label(actor_names.get(act.key) or act.name)} as {a(act.key, 'A')}{stereo}")
    d.add(f"rectangle {label(model.system)} {{")
    by_iface: Dict[str, list] = {}
    for c in model.cases:
        by_iface.setdefault(c.interface, []).append(c)
    for iface, cases in by_iface.items():
        nested = len(by_iface) > 1
        if nested:
            d.add(f"  rectangle {label(iface)} {{")
        for c in cases:
            text = label(c.name)
            if goals.get(c.id):            # the goal, with the operation under it
                text = label(goals[c.id])[:-1] + "\\n" + label(f"({c.name})")[1:]
            d.add(f"{'    ' if nested else '  '}usecase {text} as {a(c.id, 'U')}")
        if nested:
            d.add("  }")
    d.add("}")
    for c in model.cases:
        d.edge(f"{a(c.actor, 'A')} --> {a(c.id, 'U')}")
        for k in c.reaches:
            d.edge(f"{a(c.id, 'U')} --> {a(k, 'A')}")
    return d.text()


__all__ = ["render_usecase_diagram"]
