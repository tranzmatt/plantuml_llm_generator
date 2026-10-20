"""
Pseudonymization: reversible aliases for internal identifiers (cloud tier).

Aliases are sigil-prefixed (`@C17`) so they can be restored safely inside
free text, not only in structured reference fields. Numbering is sequential
and the map is persisted locally, so aliases are stable across runs and carry
no information about the real name (a plain hash of a guessable name would).

    kind  prefix  keyed by
    ----  ------  -------------------------------
    module    M   module id
    class     C   class id
    function  F   function id
    name      p   field / parameter name (shared across owners)
    iface     I   interface id
    op        O   operation id
    literal   L   literal text (route paths, flags, env names, ...)
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

PREFIXES = {
    "module": "M", "class": "C", "function": "F", "name": "p",
    "iface": "I", "op": "O", "literal": "L",
}
ALIAS_RE = re.compile(r"@(?:[MCFIOL]|p)\d+\b")


class AliasMap:
    def __init__(self) -> None:
        self._forward: Dict[Tuple[str, str], str] = {}    # (kind, key) -> alias
        self._reverse: Dict[str, Tuple[str, str, str]] = {}  # alias -> (kind, key, display)
        self._counters: Dict[str, int] = {k: 0 for k in PREFIXES}

    def __len__(self) -> int:
        return len(self._reverse)

    def alias(self, kind: str, key: str, display: Optional[str] = None) -> str:
        """Alias for (kind, key), creating one if needed. `display` is the text
        restored in prose (defaults to key)."""
        fk = (kind, key)
        if fk in self._forward:
            return self._forward[fk]
        self._counters[kind] += 1
        a = f"@{PREFIXES[kind]}{self._counters[kind]}"
        self._forward[fk] = a
        self._reverse[a] = (kind, key, display if display is not None else key)
        return a

    def lookup(self, kind: str, key: str) -> Optional[str]:
        return self._forward.get((kind, key))

    def is_known(self, alias: str) -> bool:
        return alias in self._reverse

    def resolve_ref(self, alias: str) -> Optional[str]:
        """Alias -> real key (an IR id, name or literal)."""
        entry = self._reverse.get(alias)
        return entry[1] if entry else None

    def restore_text(self, text: str) -> str:
        """Replace every known alias in `text` with its display form."""
        return ALIAS_RE.sub(lambda m: self._reverse[m.group()][2] if m.group() in self._reverse else m.group(), text)

    def unknown_aliases(self, text: str) -> List[str]:
        return sorted({a for a in ALIAS_RE.findall(text) if a not in self._reverse})

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> Dict:
        return {"version": 1,
                "entries": [{"alias": a, "kind": k, "key": key, "display": d}
                            for a, (k, key, d) in self._reverse.items()]}

    @classmethod
    def from_dict(cls, data: Dict) -> "AliasMap":
        m = cls()
        for e in data.get("entries", []):
            kind, a = e["kind"], e["alias"]
            m._forward[(kind, e["key"])] = a
            m._reverse[a] = (kind, e["key"], e["display"])
            m._counters[kind] = max(m._counters[kind], int(a[2:]))
        return m

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "AliasMap":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
