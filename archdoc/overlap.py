"""
Verbatim-overlap guard (DESIGN.md 2.5): model-written text must describe the
code, not reproduce it.

The protected text is everything in the repo's Python files that is not
interface: function and class bodies, comments, and string literals, after
removing def/class headers (with decorators), docstrings and import lines.
Those are allowed: signatures and identifiers are the interface, and
docstrings are already quoted in the ICDs as facts.

Text and source are reduced to words (identifiers kept whole, lowercased). A
generated text overlaps when it shares a run of at least `min_words`
consecutive words with the protected text. The source is the whole file, not
just what the model was shown: a model can reproduce code it saw in training.

On overlap, the planner asks for a rewrite in its one repair round without
quoting the matched words, so nothing from the code is sent in a repair.
Whatever still overlaps is cleared (`mode = "reject"`, the default) or kept
and reported (`"flag"`).
"""

from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .ir import RepoIR

MODES = ("reject", "flag", "off")
DEFAULT_MIN_WORDS = 8
_WORD = re.compile(r"[A-Za-z0-9_]+")


def words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Overlap:
    words: int                         # length of the longest shared run
    path: str
    line: int

    def describe(self) -> str:
        """For warnings and sidecars; never includes the matched words."""
        return f"{self.words} consecutive words match {self.path}:{self.line}"


def _allowed_lines(tree: ast.Module) -> Set[int]:
    """Lines that are interface, not implementation: headers, decorators, docstrings, imports."""
    lines: Set[int] = set()

    def span(node: ast.AST) -> range:
        return range(node.lineno, (getattr(node, "end_lineno", None) or node.lineno) + 1)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in node.decorator_list:
                lines.update(span(d))
            first_body = node.body[0].lineno if node.body else node.lineno + 1
            lines.update(range(node.lineno, first_body))
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                lines.update(span(body[0]))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.update(span(node))
    return lines


def _protected_words(source: str) -> Iterable[Tuple[str, int]]:
    """(word, line) for the protected text of one file, in order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    allowed = _allowed_lines(tree) if tree else set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = []
    for tok in tokens:
        line = tok.start[0]
        if line in allowed:
            continue
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#")
        elif tok.type in (tokenize.NAME, tokenize.NUMBER):
            text = tok.string
        elif tok.type == tokenize.STRING:
            text = tok.string
        else:
            continue
        for w in words(text):
            yield w, line


class SourceIndex:
    def __init__(self, root: str, ir: RepoIR, min_words: int = DEFAULT_MIN_WORDS, mode: str = "reject"):
        if mode not in MODES:
            raise ValueError(f"overlap mode must be one of {MODES}")
        self.n = min_words
        self.mode = mode
        self.grams: Dict[int, Tuple[str, int]] = {}
        if mode == "off":
            return
        for m in ir.modules:
            path = os.path.join(root, m.path)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    seq = list(_protected_words(fh.read()))
            except OSError:
                continue
            for i in range(len(seq) - self.n + 1):
                key = hash(tuple(w for w, _ in seq[i:i + self.n]))
                self.grams.setdefault(key, (m.path, seq[i][1]))

    def check(self, text: str) -> Optional[Overlap]:
        """The longest run `text` shares with the protected source, if at least n words."""
        if self.mode == "off" or not text:
            return None
        ws = words(text)
        best: Optional[Overlap] = None
        run, start_hit = 0, None
        for i in range(len(ws) - self.n + 1):
            hit = self.grams.get(hash(tuple(ws[i:i + self.n])))
            if hit:
                run, start_hit = (run + 1, start_hit) if run else (1, hit)
                length = self.n + run - 1
                if best is None or length > best.words:
                    best = Overlap(length, start_hit[0], start_hit[1])
            else:
                run, start_hit = 0, None
        return best


# -- checking plans ------------------------------------------------------------------

def sequence_texts(plan: dict) -> List[Tuple[str, str]]:
    """(location, text) for every model-written string of a sequence plan."""
    out = [(f"messages[{i}].label", m.get("label") or "") for i, m in enumerate(plan.get("messages", []))]
    out += [(f"groups[{i}].label", g.get("label") or "") for i, g in enumerate(plan.get("groups", []))]
    out.append(("title", plan.get("title") or ""))
    return [(k, v) for k, v in out if v]


def narrative_texts(plan: dict) -> List[Tuple[str, str]]:
    out = [("purpose", plan.get("purpose") or "")]
    for i, o in enumerate(plan.get("operations", [])):
        out.append((f"operations[{i}].summary", o.get("summary") or ""))
        for j, p in enumerate(o.get("params", [])):
            out.append((f"operations[{i}].params[{j}].description", p.get("description") or ""))
    return [(k, v) for k, v in out if v]


def errors(index: Optional[SourceIndex], texts: List[Tuple[str, str]]) -> List[str]:
    """Repair messages: where the overlap is, never what it says."""
    if index is None:
        return []
    out = []
    for where, text in texts:
        hit = index.check(text)
        if hit:
            out.append(f"{where} repeats {hit.words} consecutive words of the source code verbatim; "
                       f"describe what it does in your own words instead")
    return out


def _set(plan: dict, where: str, value: str) -> None:
    obj = plan
    for part in re.findall(r"[a-z_]+|\[\d+\]", where):
        if part.startswith("["):
            obj = obj[int(part[1:-1])] if isinstance(obj, list) else obj
        elif isinstance(obj, dict) and isinstance(obj.get(part), (dict, list)):
            obj = obj[part]
        else:
            obj[part] = value
            return


def enforce(index: Optional[SourceIndex], plan: dict, texts: List[Tuple[str, str]]) -> List[str]:
    """After the repair round: clear (reject) or report (flag) what still overlaps."""
    if index is None or index.mode == "off":
        return []
    warnings = []
    for where, text in texts:
        hit = index.check(text)
        if not hit:
            continue
        if index.mode == "reject":
            _set(plan, where, "")
            warnings.append(f"overlap: cleared {where} ({hit.describe()})")
        else:
            warnings.append(f"overlap: kept {where} although {hit.describe()}")
    return warnings
