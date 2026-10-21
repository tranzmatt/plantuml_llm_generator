"""
Design-description rendering: the context from archdoc.design laid out by a
Jinja2 template, templates/<name>/design.md.j2 or a path to a .j2 file.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..icd import environment

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _evidence(row: Dict[str, Any]) -> str:
    ev = row.get("evidence")
    if ev and not row.get("generated"):
        return "*(from the code)*"
    if ev == "signature":
        return "*(generated; inferred from signature)*"
    return f"*(generated from {ev})*" if ev else ""


def render_design(context: Dict[str, Any], template: str = "lightweight") -> str:
    if os.path.isfile(template):
        env = environment(os.path.dirname(os.path.abspath(template)))
        tmpl = env.get_template(os.path.basename(template))
    else:
        if not os.path.isdir(os.path.join(TEMPLATE_DIR, template)):
            raise ValueError(f"unknown design template {template!r}")
        env = environment(TEMPLATE_DIR)
        tmpl = env.get_template(f"{template}/design.md.j2")
    env.filters["evidence"] = _evidence
    text = tmpl.render(**context)
    lines, blank = [], 0
    for ln in text.splitlines():
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            lines.append(ln.rstrip())
    return "\n".join(lines).strip() + "\n"


__all__ = ["render_design"]
