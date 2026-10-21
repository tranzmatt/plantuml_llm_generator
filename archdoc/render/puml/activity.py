"""Activity diagram from archdoc.activity: the function's own control flow."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ...activity import Activity, Node, label_of
from .common import _NON_ALIAS


def _action(text: str) -> str:
    """`:text;` content: one line, and no `;` at the end, which would end it early."""
    return text.replace("\n", " ").replace(";", ",").strip() or "…"


def _cond(text: str) -> str:
    """Inside `if (...)`: parentheses would close the condition early."""
    return re.sub(r"[()]", lambda m: "[" if m.group(0) == "(" else "]", text.replace("\n", " ")).strip() or "?"


def _lines(nodes: List[Node], labels: Dict[str, str], pad: str) -> List[str]:
    out: List[str] = []
    for n in nodes:
        text = label_of(n, labels)
        if n.kind == "action":
            out.append(f"{pad}:{_action(text)};")
        elif n.kind == "def":
            out.append(f"{pad}:{_action(text)};")
        elif n.kind == "if":
            for i, b in enumerate(n.branches):
                if b.kind == "cond":
                    t = text if i == 0 else label_of(b, labels)
                    kw = "if" if i == 0 else "elseif"
                    out.append(f"{pad}{kw} ({_cond(t)}) then (yes)")
                else:
                    out.append(f"{pad}else (no)")
                out += _lines(b.body, labels, pad + "  ")
            out.append(f"{pad}endif")
        elif n.kind == "loop":
            out.append(f"{pad}while ({_cond(text)}) is (next)")
            out += _lines(n.body, labels, pad + "  ")
            out.append(f"{pad}endwhile (done)")
        elif n.kind == "with":
            out.append(f"{pad}partition {_partition(text)} {{")
            out += _lines(n.body, labels, pad + "  ")
            out.append(f"{pad}}}")
        elif n.kind == "try":
            out.append(f"{pad}partition {_partition(text)} {{")
            out += _lines(n.body, labels, pad + "  ")
            out.append(f"{pad}}}")
            handlers = [b for b in n.branches if b.kind in ("except", "else")]
            if handlers:
                for i, b in enumerate(handlers):
                    if b.kind == "except":
                        # the code's text is an exception type; a model label is already a question or phrase
                        t = labels[b.id].rstrip("?") + "?" if labels.get(b.id) else f"{b.text} raised?"
                        out.append(f"{pad}{'if' if i == 0 else 'elseif'} ({_cond(t)}) then (yes)")
                    else:
                        out.append(f"{pad}else (no exception)")
                    out += _lines(b.body, labels, pad + "  ")
                out.append(f"{pad}endif")
            for b in n.branches:
                if b.kind == "finally":
                    out.append(f"{pad}partition finally {{")
                    out += _lines(b.body, labels, pad + "  ")
                    out.append(f"{pad}}}")
        elif n.kind == "match":
            out.append(f"{pad}switch ({_cond(text)})")
            for b in n.branches:
                out.append(f"{pad}case ({_cond(label_of(b, labels))})")
                out += _lines(b.body, labels, pad + "  ")
            out.append(f"{pad}endswitch")
        elif n.kind == "return":
            out.append(f"{pad}:{_action(text)};")
            out.append(f"{pad}stop")
        elif n.kind == "raise":
            out.append(f"{pad}:{_action(text)};")
            out.append(f"{pad}end")
        elif n.kind == "break":
            out.append(f"{pad}break")
        elif n.kind == "continue":
            out.append(f"{pad}:{_action(text)};")
    return out


def _partition(text: str) -> str:
    return '"' + text.replace('"', "'").replace("\n", " ") + '"'


def render_activity_diagram(act: Activity, name: str, title: Optional[str] = None,
                            labels: Optional[Dict[str, str]] = None) -> str:
    labels = labels or {}
    out = [f"@startuml {_NON_ALIAS.sub('_', name)}"]
    if title:
        out.append(f"title {title.replace(chr(10), ' ')}")
    out.append("start")
    body = _lines(act.nodes, labels, "")
    out += body
    if not body or body[-1].strip() not in ("stop", "end"):
        out.append("stop")
    out.append("@enduml")
    return "\n".join(out) + "\n"


__all__ = ["render_activity_diagram"]
