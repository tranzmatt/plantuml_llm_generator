"""
ICD rendering: deterministic context (model.py) laid out by a Jinja2 template.

Templates are looked up by name under templates/<name>/icd.md.j2, or as a
path to a user-supplied .j2 file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jinja2

from .model import anchor, apply_narrative, build_icd_context, inventory_snapshot

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
DASH = "—"


def _cell(value: Any) -> str:
    if value is None or value == "" or value == []:
        return DASH
    text = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _code(value: Any) -> str:
    if value is None or value == "":
        return DASH
    text = str(value).replace("\n", " ").replace("|", "\\|")
    fence = "``" if "`" in text else "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _tbd(value: Any) -> str:
    return "**TBD**" if value in (None, "") else _cell(value)


def _yn(value: Optional[bool]) -> str:
    return DASH if value is None else ("Yes" if value else "No")


def _typecell(t: Optional[Dict[str, Any]]) -> str:
    if not t or not t.get("display"):
        return DASH
    code = _code(t["display"])
    return f"[{code}](#{t['anchor']})" if t.get("anchor") else code


def environment(template_dir: str = TEMPLATE_DIR) -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
        autoescape=False,
    )
    env.filters.update(cell=_cell, code=_code, tbd=_tbd, yn=_yn, typecell=_typecell)
    return env


def render_icd(context: Dict[str, Any], template: str = "lightweight") -> str:
    """Render an ICD context with a bundled template name or a path to a .j2 file."""
    if os.path.isfile(template):
        env = environment(os.path.dirname(os.path.abspath(template)))
        tmpl = env.get_template(os.path.basename(template))
    else:
        if not os.path.isdir(os.path.join(TEMPLATE_DIR, template)):
            raise ValueError(f"unknown ICD template {template!r}")
        tmpl = environment().get_template(f"{template}/icd.md.j2")
    text = tmpl.render(**context)
    # collapse runs of blank lines left by template control blocks
    lines, blank = [], 0
    for ln in text.splitlines():
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            lines.append(ln.rstrip())
    return "\n".join(lines).strip() + "\n"


__all__ = ["anchor", "apply_narrative", "build_icd_context", "inventory_snapshot", "render_icd"]
