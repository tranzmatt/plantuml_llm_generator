"""
Shared PlantUML building blocks.

Every element gets an alias made only of [A-Za-z0-9_], and every display
label is double-quoted with quotes and newlines neutralized. Renderers only
ever emit edges between aliases they have declared, so the classic LLM
failure modes (quoted aliases, undeclared targets, interleaving) cannot occur.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_NON_ALIAS = re.compile(r"[^A-Za-z0-9_]")


def label(text: str) -> str:
    """A double-quoted PlantUML label."""
    clean = text.replace("\\", "/").replace('"', "'").replace("\r", " ").replace("\n", " ").strip()
    return f'"{clean}"'


def member_text(text: str) -> str:
    """Text inside a class body; braces would close the body early."""
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")


class Aliases:
    """Stable, unique PlantUML aliases for arbitrary ids."""

    def __init__(self) -> None:
        self._by_key: Dict[str, str] = {}
        self._used: set = set()

    def __call__(self, key: str, prefix: str = "E") -> str:
        if key in self._by_key:
            return self._by_key[key]
        base = _NON_ALIAS.sub("_", key).strip("_") or "x"
        base = f"{prefix}_{base}"
        alias, n = base, 2
        while alias in self._used:
            alias, n = f"{base}_{n}", n + 1
        self._used.add(alias)
        self._by_key[key] = alias
        return alias

    def get(self, key: str) -> Optional[str]:
        return self._by_key.get(key)


class Diagram:
    """Collects declarations and edges, emitting declarations first."""

    def __init__(self, name: str, title: Optional[str] = None, settings: Optional[List[str]] = None):
        self.name = _NON_ALIAS.sub("_", name)
        self.title = title
        self.settings = settings or []
        self.body: List[str] = []
        self.edges: List[str] = []
        self._edge_set: set = set()

    def add(self, line: str = "") -> None:
        self.body.append(line)

    def edge(self, line: str) -> None:
        if line not in self._edge_set:
            self._edge_set.add(line)
            self.edges.append(line)

    def text(self) -> str:
        out = [f"@startuml {self.name}"]
        out += self.settings
        if self.title:
            out.append(f"title {label(self.title)[1:-1]}")
        out += self.body
        if self.edges:
            out.append("")
            out += self.edges
        out.append("@enduml")
        return "\n".join(out) + "\n"
