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

String literals are checked on their own as well: a distinctive literal
from a function body quoted verbatim is an overlap however short the run.
Distinctive means at least `MIN_LITERAL` characters and either structure (a
URL, a path, SQL, a format string: anything with characters beyond words,
spaces and plain punctuation) or at least `MIN_LITERAL_WORDS` words. Literals
shaped like identifiers (dict keys, field names, enum values) are names and
allowed, as identifiers are; so are URLs (where an external interface is)
and everything in the interface inventory (routes, CLI names, environment
variables, queue names, defaults in signatures).

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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .ir import RepoIR

MODES = ("reject", "flag", "off")
DEFAULT_MIN_WORDS = 8
MIN_LITERAL = 8
MIN_LITERAL_WORDS = 5
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLAIN = re.compile(r"[A-Za-z0-9_\s,.:;!?'\-]*")
_URL = re.compile(r"[a-z][a-z0-9+.-]*://", re.I)
_WORD = re.compile(r"[A-Za-z0-9_]+")


def words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Overlap:
    words: int                         # length of the longest shared run
    path: str
    line: int
    literal: bool = False              # a quoted string literal rather than a run of words

    def describe(self) -> str:
        """For warnings and sidecars; never includes the matched words."""
        if self.literal:
            return f"quotes a string literal from {self.path}:{self.line}"
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


def _protected_literals(source: str) -> Iterable[Tuple[str, int]]:
    """(string literal, line) outside interface lines, long enough to be content."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    allowed = _allowed_lines(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.lineno not in allowed:
            text = node.value.strip()
            if len(text) < MIN_LITERAL or not re.search(r"[A-Za-z]", text) or _IDENT.fullmatch(text):
                continue
            if _PLAIN.fullmatch(text) and len(text.split()) < MIN_LITERAL_WORDS:
                continue                      # a short everyday phrase identifies nothing
            if len(words(text)) < 2 or _URL.match(text):
                continue                      # "%s": no content; a URL: where an interface is
            yield text, node.lineno


def interface_literals(ir: RepoIR) -> Set[str]:
    """Strings the documents may quote: they are the interface, not its implementation."""
    out: Set[str] = set()

    def add(*vals):
        for v in vals:
            if isinstance(v, str) and v:
                out.add(v.strip().strip("'\""))

    for iface in ir.interfaces:
        add(iface.name, iface.address, iface.protocol)
        for op in iface.operations:
            add(op.name, *op.meta.values(), *op.errors)
            for f in op.params + op.outputs:
                add(f.name, f.default, *f.constraints.values())
    for d in ir.data_types:
        for f in d.fields:
            add(f.name, f.default, *f.constraints.values())
    for fn in ir.iter_functions():
        for p in fn.params:
            add(p.default)
        for d in fn.decorators:
            add(*d.args, *d.kwargs.values())
    for svc in (s for dep in ir.deployments for s in dep.services):
        add(svc.name, svc.image, *svc.env)
    return out


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
        self.literals: Dict[str, Tuple[str, int]] = {}
        self._literal_rx: Optional[re.Pattern] = None
        if mode == "off":
            return
        allowed = interface_literals(ir)
        for m in ir.modules:
            path = os.path.join(root, m.path)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            seq = list(_protected_words(text))
            for lit, line in _protected_literals(text):
                if lit not in allowed:
                    self.literals.setdefault(lit, (m.path, line))
            for i in range(len(seq) - self.n + 1):
                key = hash(tuple(w for w, _ in seq[i:i + self.n]))
                self.grams.setdefault(key, (m.path, seq[i][1]))

    def check(self, text: str) -> Optional[Overlap]:
        """The longest run `text` shares with the protected source, if at least n
        words, else a protected string literal it quotes."""
        if self.mode == "off" or not text:
            return None
        run = self._check_run(text)
        if run is not None:
            return run
        if self.literals and self._literal_rx is None:
            alts = sorted(self.literals, key=len, reverse=True)
            self._literal_rx = re.compile("|".join(rf"(?<!\w){re.escape(x)}(?!\w)" for x in alts))
        m = self._literal_rx.search(text) if self._literal_rx else None
        if m:
            path, line = self.literals[m.group(0)]
            return Overlap(len(words(m.group(0))), path, line, literal=True)
        return None

    def _check_run(self, text: str) -> Optional[Overlap]:
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


_NOT_PROSE = {"ref", "name", "kind", "from", "to", "call"}


def plan_texts(plan: Any, where: str = "") -> List[Tuple[str, str]]:
    """(location, text) for every model-written string of any plan; refs and names are not prose."""
    out: List[Tuple[str, str]] = []
    if isinstance(plan, dict):
        for k, v in plan.items():
            if k not in _NOT_PROSE:
                out += plan_texts(v, f"{where}.{k}" if where else k)
    elif isinstance(plan, list):
        for i, v in enumerate(plan):
            out += plan_texts(v, f"{where}[{i}]")
    elif isinstance(plan, str) and plan:
        out.append((where, plan))
    return out


def errors(index: Optional[SourceIndex], texts: List[Tuple[str, str]]) -> List[str]:
    """Repair messages: where the overlap is, never what it says."""
    if index is None:
        return []
    out = []
    for where, text in texts:
        hit = index.check(text)
        if hit and hit.literal:
            out.append(f"{where} quotes a string literal from the source code; describe what the value is "
                       f"for instead of quoting it")
        elif hit:
            out.append(f"{where} repeats {hit.words} consecutive words of the source code verbatim; "
                       f"describe what it does in your own words instead")
    return out


def _set(plan: Any, where: str, value: str) -> None:
    parts = [int(p[1:-1]) if p.startswith("[") else p for p in re.findall(r"[A-Za-z_]+|\[\d+\]", where)]
    obj = plan
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value


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
