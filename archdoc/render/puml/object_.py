"""Object diagram from archdoc.objects: the objects module import creates."""

from __future__ import annotations

from typing import Dict, List, Optional

from ...objects import ObjectModel
from .common import Aliases, Diagram, label


def render_object_diagram(model: ObjectModel, name: str, title: Optional[str] = None) -> str:
    d = Diagram(name, title, settings=["hide empty members"])
    a = Aliases()
    by_mod: Dict[str, List] = {}
    for o in model.objects:
        by_mod.setdefault(o.module, []).append(o)
    for mid, objs in by_mod.items():
        nested = len(by_mod) > 1
        if nested:
            d.add(f"package {label(mid)} {{")
        for o in objs:
            cls = o.cls.split(":", 1)[1] if o.internal else o.cls
            d.add(f"{'  ' if nested else ''}object {label(f'{o.name} : {cls}')} as {a(o.key, 'O')}")
        if nested:
            d.add("}")
    for o in model.objects:
        for k, v in o.attrs:
            d.add(f"{a(o.key, 'O')} : {k} = {v.replace(chr(10), ' ')}")
    for lk in model.links:
        d.edge(f"{a(lk.src, 'O')} --> {a(lk.dst, 'O')} : {label(lk.label)[1:-1]}")
    return d.text()


__all__ = ["render_object_diagram"]
