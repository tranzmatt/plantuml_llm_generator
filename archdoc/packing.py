"""
Fitting the context window (DESIGN.md 12): what a request carries, and how much.

Encoding: payloads are sent as compact text, Python-stub style for functions
and `key: value` lines for the rest, instead of JSON. The JSON stays the
internal, audited form (Payload.data); the text is rendered from it, so it
contains nothing the policy did not already let through.

Budget: the window is the backend's `num_ctx` (Ollama), configured `context` or
the server's `max_model_len` (vLLM). Reserved: the system prompt and
instructions, the answer (`max_tokens`), the previous answer echoed in a repair
round, and a 10% margin. Characters per token start at a conservative 3.0 and
are calibrated per model from the token counts the server reports.

Packing: every item starts as a stub (signature, docstring line, calls). Items
are upgraded in rank order, first all to `outline`, then to `source`, each
upgrade kept only if the request still fits. Nothing is cut mid-function: if
even all-stubs does not fit, the caller shrinks the work (a shallower slice).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DEFAULT_WINDOW = 8192
DEFAULT_CHARS_PER_TOKEN = 3.0
MARGIN = 0.9
UPGRADES = ("outline", "source")
POLICY_FOR_LEVEL = {"outline": "control_flow", "source": "source"}

# -- encoding -----------------------------------------------------------------------

_BODY_KEYS = ("outline", "source")
_FUNCTION_LISTS = ("functions", "handlers", "methods")


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v).replace("\n", " ")


def _is_scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list))


def _function(f: Dict[str, Any], pad: str) -> List[str]:
    notes = []
    for key in ("owner", "calls", "creates", "raises"):
        if f.get(key):
            v = f[key]
            notes.append(f"{key} {', '.join(v) if isinstance(v, list) else v}")
    note = f"  # {'; '.join(notes)}" if notes else ""
    if "source" in f:                              # the source carries its own def line and docstring
        out = [f"{pad}# {f['ref']}" + (f"; {'; '.join(notes)}" if notes else "")]
        out += [pad + line if line else line for line in f["source"].splitlines()]
        return out
    out = [f"{pad}@{d}" for d in f.get("decorators", [])]
    out.append(f"{pad}{'async ' if f.get('async') else ''}def {f['ref']}{f.get('sig', '()')}:{note}")
    inner = pad + "    "
    if f.get("doc"):
        out.append(f'{inner}"""{_scalar(f["doc"])}"""')
    if "outline" in f:
        out += [inner + line for line in f["outline"].splitlines()]
    elif not f.get("doc"):
        out.append(f"{inner}...")
    return out


def _flat_dict(v: Any) -> bool:
    return isinstance(v, dict) and all(_is_scalar(x) for x in v.values())


def _inline_dict(v: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={_scalar(x)}" for k, x in v.items())


def _item(d: Dict[str, Any], pad: str) -> List[str]:
    """One list element: scalars, scalar lists and flat dicts inline after "- ", the rest indented below."""
    parts = []
    for k, v in d.items():
        if _is_scalar(v):
            parts.append(f"{k}: {_scalar(v)}")
        elif isinstance(v, list) and v and all(_is_scalar(x) for x in v):
            parts.append(f"{k}: {', '.join(_scalar(x) for x in v)}")
        elif _flat_dict(v) and v:
            parts.append(f"{k}: {_inline_dict(v)}")
    out = [f"{pad}- {'; '.join(parts)}"]
    for k, v in d.items():
        if isinstance(v, dict) and not _flat_dict(v):
            out.append(f"{pad}  {k}:")
            out += _block(v, pad + "    ")
        elif isinstance(v, list) and v and not all(_is_scalar(x) for x in v):
            out.append(f"{pad}  {k}:")
            out += _list(k, v, pad + "    ")
    return out


def _list(key: str, items: List[Any], pad: str) -> List[str]:
    out: List[str] = []
    for x in items:
        if isinstance(x, dict) and (key in _FUNCTION_LISTS or "sig" in x):
            out += _function(x, pad)
        elif isinstance(x, dict):
            out += _item(x, pad)
        else:
            out.append(f"{pad}- {_scalar(x)}")
    return out


def _block(d: Dict[str, Any], pad: str = "") -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        if _is_scalar(v):
            out.append(f"{pad}{k}: {_scalar(v)}")
        elif isinstance(v, list) and all(_is_scalar(x) for x in v):
            out.append(f"{pad}{k}: {', '.join(_scalar(x) for x in v)}")
        elif _flat_dict(v):
            out.append(f"{pad}{k}: {_inline_dict(v)}")
        elif isinstance(v, list):
            out.append(f"{pad}{k}:")
            out += _list(k, v, pad + "  ")
        else:
            out.append(f"{pad}{k}:")
            out += _block(v, pad + "  ")
    return out


def encode(data: Dict[str, Any]) -> str:
    """Compact text form of a payload: what the model reads."""
    return "\n".join(_block(data))


def encode_payload(data: Dict[str, Any], encoding: str) -> str:
    if encoding == "json":
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return encode(data)


# -- short ids ---------------------------------------------------------------------------

# fields whose values are ids; nothing else (prose, signatures) is rewritten
REF_KEYS = {"ref", "owner", "handler", "calls", "creates", "raises", "bases", "imports"}
_SEP = re.compile(r"[.:]")


def _suffixes(full: str) -> List[str]:
    cuts = [m.end() for m in _SEP.finditer(full)]
    return [full[c:] for c in reversed(cuts)] + [full]


def short_forms(ids) -> Dict[str, str]:
    """full id -> shortest suffix (at a . or : boundary) no other id in the request shares.
    Aliases (@F1) and third-party ids (ext:...) are left as they are."""
    own = sorted({i for i in ids if i and not i.startswith(("@", "ext:")) and _SEP.search(i)})
    counts: Dict[str, int] = {}
    for i in own:
        for s in set(_suffixes(i)):
            counts[s] = counts.get(s, 0) + 1
    taken = set(own) | {"client"}
    out: Dict[str, str] = {}
    for i in own:
        short = next((s for s in _suffixes(i) if counts[s] == 1 and (s == i or s not in taken)
                      and not s[0].isdigit()), i)
        if short != i:
            out[i] = short
            taken.add(short)
    return out


def _ids_in(obj: Any, key: str = "") -> List[str]:
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in _ids_in(v, k)]
    if isinstance(obj, list):
        return [x for v in obj for x in _ids_in(v, key)]
    return [obj] if key in REF_KEYS and isinstance(obj, str) else []


def _rewrite(obj: Any, table: Dict[str, str], key: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite(v, table, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, table, key) for v in obj]
    if key in REF_KEYS and isinstance(obj, str):
        return table.get(obj, obj)
    return obj


def shorten(data: Dict[str, Any], allowed: Dict[str, List[str]]):
    """(data, allowed, short -> full) with ids in their per-request short forms."""
    ids = _ids_in(data) + [x for refs in allowed.values() for x in refs]
    table = short_forms(ids)
    if not table:
        return data, allowed, {}
    return (_rewrite(data, table), {role: [table.get(r, r) for r in refs] for role, refs in allowed.items()},
            {s: f for f, s in table.items()})


# -- window and calibration ------------------------------------------------------------

@dataclass
class Window:
    tokens: int = DEFAULT_WINDOW
    output_tokens: int = 2048
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    def input_chars(self, fixed_chars: int) -> int:
        """Characters left for the payload after the prompt, the answer and a repair echo."""
        usable = (self.tokens - 2 * self.output_tokens) * MARGIN
        return max(0, int(usable * self.chars_per_token) - fixed_chars)


@dataclass
class Calibration:
    """Measured characters per token, per model, kept in the state directory."""
    path: Optional[str] = None
    ratios: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str]) -> "Calibration":
        cal = cls(path)
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    cal.ratios = {k: float(v) for k, v in json.load(fh).items()}
            except (OSError, ValueError):
                pass
        return cal

    def get(self, model: Optional[str]) -> float:
        return self.ratios.get(model or "", DEFAULT_CHARS_PER_TOKEN)

    def observe(self, model: Optional[str], chars: int, tokens: Optional[int]) -> None:
        """Keep the lowest ratio seen (the most tokens per character): conservative."""
        if not model or not tokens or tokens < 50:
            return
        ratio = round(chars / tokens, 3)
        old = self.ratios.get(model)
        self.ratios[model] = ratio if old is None else min(old, ratio)
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.ratios, fh, indent=1, sort_keys=True)


# -- packing ---------------------------------------------------------------------------

@dataclass
class Packed:
    payload: Any                                   # egress.Payload
    levels: Dict[str, str]                         # item id -> stub | outline | source
    fits: bool
    budget_chars: int
    tried: int = 0


def pack(build: Callable[[Dict[str, str]], Any], ranked: List[str], allowed: List[str],
         budget_chars: int) -> Packed:
    """Upgrade items in rank order while the encoded payload stays within budget."""
    levels: Dict[str, str] = {}
    payload = build(levels)
    tried = 1
    if len(payload.text) > budget_chars:
        return Packed(payload, {i: "stub" for i in ranked}, False, budget_chars, tried)
    for level in (lv for lv in UPGRADES if lv in allowed):
        for item in ranked:
            trial = {**levels, item: level}
            p = build(trial)
            tried += 1
            if len(p.text) == len(payload.text):
                continue                           # nothing added (policy or no source)
            if len(p.text) <= budget_chars:
                levels, payload = trial, p
    full = {i: levels.get(i, "stub") for i in ranked}
    return Packed(payload, full, True, budget_chars, tried)


def allowed_levels(policy) -> List[str]:
    return [lv for lv in UPGRADES if policy.level_at_least(POLICY_FOR_LEVEL[lv])]


def fixed_chars(*texts: str) -> int:
    return sum(len(t) for t in texts)


def levels_summary(levels: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lv in levels.values():
        out[lv] = out.get(lv, 0) + 1
    return out


def shrink_steps(n: int) -> List[int]:
    """Smaller slice sizes to try when even all-stubs does not fit."""
    steps, k = [], n
    while k > 2:
        k = max(2, (k * 2) // 3)
        steps.append(k)
    return steps


__all__ = ["Calibration", "Packed", "Window", "allowed_levels", "encode", "encode_payload", "fixed_chars",
           "levels_summary", "pack", "shrink_steps"]
