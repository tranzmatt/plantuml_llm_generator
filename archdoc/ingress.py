"""
Ingress: everything an LLM (or MCP client) sends back passes through here.

  - a string that is exactly an alias ("@C17") is a reference and becomes the
    real key (an IR id, name or literal)
  - aliases embedded in prose are replaced with their display text
  - any alias not in the map is an error: the model referenced something it
    was never shown (hallucination check)

Dict keys are treated the same way as values, so narrative keyed by
operation ref ({"@O5": {...}}) comes back keyed by the real operation id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from .pseudonym import ALIAS_RE, AliasMap


@dataclass
class IngressResult:
    data: Any
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def accept(obj: Any, aliases: AliasMap) -> IngressResult:
    errors: List[str] = []

    def restore_str(s: str, path: str) -> str:
        if ALIAS_RE.fullmatch(s):
            ref = aliases.resolve_ref(s)
            if ref is None:
                errors.append(f"{path}: unknown reference {s}")
                return s
            return ref
        for a in aliases.unknown_aliases(s):
            errors.append(f"{path}: unknown reference {a}")
        return aliases.restore_text(s)

    def walk(o: Any, path: str) -> Any:
        if isinstance(o, str):
            return restore_str(o, path)
        if isinstance(o, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(o)]
        if isinstance(o, dict):
            return {restore_str(k, f"{path}.<key>") if isinstance(k, str) else k: walk(v, f"{path}.{k}")
                    for k, v in o.items()}
        return o

    return IngressResult(walk(obj, "$"), errors)
