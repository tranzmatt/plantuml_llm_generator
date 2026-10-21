"""
Function bodies at two detail levels, read from the files on demand
(DESIGN.md 12.2). The IR never stores bodies; these texts exist only inside a
local request that the policy allows (`control_flow` for outlines, `source`
for source) and always pass the secret scanner on the way out.

  outline  the nesting of if / for / while / try / with / match, the calls each
           statement makes (as written), raised exception types and returns;
           no conditions, arguments or other expressions
  source   the whole function, decorators included, dedented
"""

from __future__ import annotations

import ast
import functools
import os
import textwrap
from typing import List, Optional, Tuple

from ..ir import RepoIR
from .python_ast import dotted_name

FuncNode = (ast.FunctionDef, ast.AsyncFunctionDef)
# builtins add length, not information, to an outline
_NOISE = {"str", "int", "float", "bool", "len", "list", "dict", "set", "tuple", "print", "isinstance",
          "getattr", "hasattr", "setattr", "range", "enumerate", "zip", "sorted", "min", "max", "sum",
          "any", "all", "repr", "type", "iter", "next", "map", "filter", "super", "format", "abs", "round"}


@functools.lru_cache(maxsize=256)
def _parsed(path: str, mtime: float) -> Tuple[Optional[ast.Module], List[str]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        return ast.parse(text), text.splitlines()
    except SyntaxError:
        return None, text.splitlines()


def locate(root: str, ir: RepoIR, function_id: str) -> Optional[Tuple[ast.AST, List[str]]]:
    """(AST node, file lines) for a function id such as "pkg.mod:Class.method"."""
    mid, _, qual = function_id.partition(":")
    m = ir.module(mid)
    if m is None or not qual:
        return None
    path = os.path.join(root, m.path)
    try:
        tree, lines = _parsed(path, os.path.getmtime(path))
    except OSError:
        return None
    if tree is None:
        return None
    scope: List[ast.stmt] = tree.body
    node = None
    for part in qual.split("."):
        node = next((n for n in scope if isinstance(n, (*FuncNode, ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        scope = node.body
    return (node, lines) if isinstance(node, FuncNode) else None


def source_text(node: ast.AST, lines: List[str]) -> str:
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    return textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))


def _calls(node: ast.AST) -> List[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = dotted_name(n.func)
            if name and name not in out and name not in _NOISE:
                out.append(name)
    return out


def _header_calls(*exprs: Optional[ast.AST]) -> str:
    names = [c for e in exprs if e is not None for c in _calls(e)]
    return f"  # calls {', '.join(names)}" if names else ""


def _outline(body: List[ast.stmt], depth: int, out: List[str]) -> None:
    pad = "    " * depth
    for st in body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"{pad}def {st.name}(...): ...")
        elif isinstance(st, ast.ClassDef):
            out.append(f"{pad}class {st.name}: ...")
        elif isinstance(st, ast.If):
            out.append(f"{pad}if ...:{_header_calls(st.test)}")
            _outline(st.body, depth + 1, out)
            orelse = st.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                out.append(f"{pad}elif ...:{_header_calls(orelse[0].test)}")
                _outline(orelse[0].body, depth + 1, out)
                orelse = orelse[0].orelse
            if orelse:
                out.append(f"{pad}else:")
                _outline(orelse, depth + 1, out)
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            out.append(f"{pad}for ...:{_header_calls(st.iter)}")
            _outline(st.body, depth + 1, out)
            if st.orelse:
                out.append(f"{pad}else:")
                _outline(st.orelse, depth + 1, out)
        elif isinstance(st, ast.While):
            out.append(f"{pad}while ...:{_header_calls(st.test)}")
            _outline(st.body, depth + 1, out)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            out.append(f"{pad}with ...:{_header_calls(*[i.context_expr for i in st.items])}")
            _outline(st.body, depth + 1, out)
        elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
            out.append(f"{pad}try:")
            _outline(st.body, depth + 1, out)
            for h in st.handlers:
                t = dotted_name(h.type) if h.type is not None and dotted_name(h.type) else (
                    "(...)" if h.type is not None else "")
                out.append(f"{pad}except {t}:".replace("except :", "except:"))
                _outline(h.body, depth + 1, out)
            if st.orelse:
                out.append(f"{pad}else:")
                _outline(st.orelse, depth + 1, out)
            if st.finalbody:
                out.append(f"{pad}finally:")
                _outline(st.finalbody, depth + 1, out)
        elif isinstance(st, ast.Match):
            out.append(f"{pad}match ...:{_header_calls(st.subject)}")
            for case in st.cases:
                out.append(f"{pad}    case ...:")
                _outline(case.body, depth + 2, out)
        elif isinstance(st, ast.Raise):
            exc = st.exc.func if isinstance(st.exc, ast.Call) else st.exc
            out.append(f"{pad}raise {dotted_name(exc) or '...'}" if exc is not None else f"{pad}raise")
        elif isinstance(st, ast.Return):
            out.append(f"{pad}return{_header_calls(st.value)}")
        elif isinstance(st, (ast.Break, ast.Continue)):
            out.append(f"{pad}{type(st).__name__.lower()}")
        else:
            calls = _calls(st)
            prefix = "yield " if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(st)) else ""
            if calls or prefix:
                out.append(f"{pad}{prefix}{', '.join(c + '()' for c in calls)}".rstrip())
    if out and out[-1].endswith(":"):
        out.append(f"{pad}...")


def outline_text(node: ast.AST) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                               # the docstring is already in the stub
    out: List[str] = []
    _outline(body, 0, out)
    return "\n".join(out) or "..."
