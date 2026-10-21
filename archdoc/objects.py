"""
Object diagram: the objects that exist once a module is imported (DESIGN.md 9).

Not example instances invented by a model: the module-level objects the code
actually creates (`default_db = Database()`, `app = FastAPI()`, `with
gr.Blocks() as demo:` and the components created inside it), their simple
constructor arguments as attribute values, and the links the code makes
between them at module level: an object passed to another's constructor, an
object created inside another's `with` block, and calls such as
`app.include_router(router)` or `button.click(inputs=[...], outputs=[...])`.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .extract.bodies import _parsed
from .extract.python_ast import dotted_name
from .ir import Module, RepoIR
from .secrets_scan import redact

MAX_VALUE = 40


@dataclass
class Obj:
    name: str                  # variable name
    module: str
    cls: str                   # class as resolved (id) or as written
    internal: bool
    line: int
    attrs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.module}:{self.name}"


@dataclass
class Link:
    src: str                   # object keys
    dst: str
    label: str


@dataclass
class ObjectModel:
    objects: List[Obj]
    links: List[Link]


def _value(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)) or (
            isinstance(node, ast.Constant) and node.value is None):
        text = repr(node.value)
        text = redact(text)[0]
        return text if len(text) <= MAX_VALUE else text[: MAX_VALUE - 1] + "…'"
    return None


def _is_classlike(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return bool(last) and last[0].isupper()


class _ModuleObjects:
    def __init__(self, ir: RepoIR, m: Module):
        self.ir, self.m = ir, m
        self.by_name: Dict[str, List[str]] = {}
        for c in ir.iter_classes():
            self.by_name.setdefault(c.name, []).append(c.id)
        self.objs: Dict[str, Obj] = {}
        self.links: List[Link] = []

    def resolve(self, written: str) -> Tuple[str, bool]:
        last = written.rsplit(".", 1)[-1]
        cands = self.by_name.get(last, [])
        own = [c for c in cands if c.split(":")[0] == self.m.id]
        imported = [c for c in cands if c.split(":")[0] in self.m.imports]
        pick = own or imported or (cands if len(cands) == 1 else [])
        return (pick[0], True) if len(pick) == 1 else (written, False)

    def link(self, src: str, node: ast.AST, label: str) -> None:
        items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
        for it in items:
            if isinstance(it, ast.Name) and it.id in self.objs and it.id != src:
                self.links.append(Link(self.objs[src].key, self.objs[it.id].key, label))

    def create(self, name: str, call: ast.Call, line: int, parent: Optional[str]) -> None:
        written = dotted_name(call.func)
        if not written:
            return
        cls, internal = self.resolve(written)
        if not internal and not _is_classlike(written):
            return                                   # a factory call: not knowably an object of a class
        obj = Obj(name, self.m.id, cls, internal, line)
        for i, a in enumerate(call.args):
            v = _value(a)
            if v is not None:
                obj.attrs.append((f"arg{i + 1}", v))
        for kw in call.keywords:
            if kw.arg is None:
                continue
            v = _value(kw.value)
            if v is not None:
                obj.attrs.append((kw.arg, v))
        self.objs[name] = obj
        for i, a in enumerate(call.args):
            self.link(name, a, f"arg{i + 1}")
        for kw in call.keywords:
            if kw.arg:
                self.link(name, kw.value, kw.arg)
        if parent and parent in self.objs:
            self.links.append(Link(self.objs[parent].key, obj.key, "contains"))

    def walk(self, body: List[ast.stmt], parent: Optional[str] = None) -> None:
        for st in body:
            if isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(st.value, ast.Call):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.create(t.id, st.value, st.lineno, parent)
            elif isinstance(st, ast.With):
                inner = parent
                for item in st.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.optional_vars, ast.Name):
                            self.create(item.optional_vars.id, item.context_expr, st.lineno, parent)
                            inner = item.optional_vars.id
                self.walk(st.body, inner)
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Attribute):
                call = st.value
                owner = dotted_name(call.func.value)
                if owner in self.objs:
                    method = call.func.attr
                    for a in call.args:
                        self.link(owner, a, method)
                    for kw in call.keywords:
                        if kw.arg:
                            self.link(owner, kw.value, f"{method} {kw.arg}")


def build_objects(root: str, ir: RepoIR, modules: Optional[List[str]] = None) -> ObjectModel:
    objects: List[Obj] = []
    links: List[Link] = []
    for m in ir.modules:
        if modules is not None and m.id not in modules:
            continue
        path = os.path.join(root, m.path)
        try:
            tree, _ = _parsed(path, os.path.getmtime(path))
        except OSError:
            continue
        if tree is None:
            continue
        mo = _ModuleObjects(ir, m)
        mo.walk(tree.body)
        objects += list(mo.objs.values())
        seen = set()
        for lk in mo.links:
            if (lk.src, lk.dst, lk.label) not in seen:
                seen.add((lk.src, lk.dst, lk.label))
                links.append(lk)
    return ObjectModel(objects, links)


__all__ = ["Link", "Obj", "ObjectModel", "build_objects"]
