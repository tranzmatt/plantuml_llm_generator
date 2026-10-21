"""
State machines from the code (DESIGN.md 2.2, 9), without a model.

States are enum members, or the string values assigned to an attribute named
like a state (`status`, `state`, `phase`, `stage`). A transition is an
assignment of a state inside a function: the target is the assigned state; the
source is the state an enclosing `if` requires (`if self.status ==
Status.OPEN:`, `in (...)`), or any state when nothing is required. An
assignment in `__init__`, or a field default, is an initial state. So the
verification rules hold by construction: every state is an enum member or an
assigned value, and every transition names the function that assigns it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .extract.bodies import locate
from .extract.python_ast import dotted_name
from .ir import RepoIR

STATE_ATTR = re.compile(r"^_?(status|state|phase|stage|lifecycle)$", re.I)
ANY = None


@dataclass
class Transition:
    src: Optional[str]                 # None: from any state
    dst: str
    function: str                      # function id that assigns dst
    line: int
    attribute: str


@dataclass
class Machine:
    key: str                           # enum class id, or "<class id>.<attr>" for string states
    name: str
    states: List[str]
    kind: str = "enum"                 # enum | string
    initial: List[Tuple[str, str]] = field(default_factory=list)      # (state, where)
    transitions: List[Transition] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)                    # "Class.attr" holding the state

    @property
    def unreached(self) -> List[str]:
        reached = {s for s, _ in self.initial} | {t.dst for t in self.transitions}
        return [s for s in self.states if s not in reached]


class _Enums:
    def __init__(self, ir: RepoIR):
        self.by_name: Dict[str, List[Tuple[str, List[str]]]] = {}
        for c in ir.iter_classes():
            if c.kind == "enum" and c.enum_members:
                self.by_name.setdefault(c.name, []).append((c.id, list(c.enum_members)))

    def member(self, node: Optional[ast.AST]) -> Optional[Tuple[str, str]]:
        """(enum id, member) for `Status.OPEN` / `models.Status.OPEN`."""
        name = dotted_name(node) if node is not None else None
        if not name or "." not in name:
            return None
        parts = name.split(".")
        cands = [(cid, m) for cid, m in self.by_name.get(parts[-2], []) if parts[-1] in m]
        return (cands[0][0], parts[-1]) if len(cands) == 1 else None


def _target_attr(t: ast.AST) -> Optional[str]:
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Name):
        return t.id
    return None


def _guard(test: ast.AST, enums: _Enums, negated: bool = False) -> List[Tuple[str, Set[str], Optional[str]]]:
    """(attribute, required states, enum id or None for strings) from a condition;
    `negated`: what holds after `if <test>: raise/return` (`!=` and `not in` then require)."""
    out = []
    if negated:
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            parts = test.values               # not (a or b) == not a and not b
        else:
            parts = [test]
    else:
        parts = [test] if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And) else test.values
    for node in parts:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        attr = _target_attr(node.left)
        if attr is None:
            continue
        op, comp = node.ops[0], node.comparators[0]
        if negated:
            op = {ast.NotEq: ast.Eq(), ast.IsNot: ast.Is(), ast.NotIn: ast.In()}.get(type(op))
            if op is None:
                continue
        if isinstance(op, (ast.Eq, ast.Is)):
            em = enums.member(comp)
            if em:
                out.append((attr, {em[1]}, em[0]))
            elif isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                out.append((attr, {comp.value}, None))
        elif isinstance(op, ast.In) and isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
            ems = [enums.member(e) for e in comp.elts]
            if ems and all(ems) and len({e[0] for e in ems}) == 1:
                out.append((attr, {e[1] for e in ems}, ems[0][0]))
            elif comp.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in comp.elts):
                out.append((attr, {e.value for e in comp.elts}, None))
    return out


def _owner_class(fid: str) -> Optional[str]:
    mid, _, qual = fid.partition(":")
    return f"{mid}:{qual.rsplit('.', 1)[0]}" if "." in qual else None


def state_machines(root: str, ir: RepoIR, modules: Optional[Set[str]] = None) -> List[Machine]:
    enums = _Enums(ir)
    machines: Dict[str, Machine] = {}
    enum_names = {cid: cid.split(":")[1] for lst in enums.by_name.values() for cid, _ in lst}
    members = {cid: m for lst in enums.by_name.values() for cid, m in lst}

    def machine(key: str, kind: str, name: str) -> Machine:
        if key not in machines:
            machines[key] = Machine(key, name, list(members.get(key, [])), kind)
        return machines[key]

    # fields typed with an enum, and their defaults
    for c in ir.iter_classes():
        for f in c.fields:
            cid = f.type_id if f.type_id in members else next(
                (k for k, n in enum_names.items() if f.annotation and f.annotation.split(".")[-1] == n
                 and len(enums.by_name.get(n, [])) == 1), None)
            if cid is None:
                continue
            m = machine(cid, "enum", enum_names[cid])
            m.fields.append(f"{c.name}.{f.name}")
            em = enums.member(ast.parse(f.default, mode="eval").body) if f.default and f.default != "<expr>" \
                else None
            if em and em[0] == cid:
                m.initial.append((em[1], f"default of {c.name}.{f.name}"))

    for fn in ir.iter_functions():
        if modules is not None and fn.id.split(":")[0] not in modules:
            continue
        found = locate(root, ir, fn.id)
        if found is None:
            continue
        cls = _owner_class(fn.id)
        is_init = fn.name == "__init__"

        def visit(body: List[ast.stmt], guards: List[Tuple[str, Set[str], Optional[str]]]) -> None:
            guards = list(guards)
            for st in body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(st, (ast.Assign, ast.AnnAssign)):
                    targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                    values = [st.value.body, st.value.orelse] if isinstance(st.value, ast.IfExp) else [st.value]
                    for t in targets:
                        attr = _target_attr(t)
                        if attr is None:
                            continue
                        for v in values:
                            em = enums.member(v)
                            if em:
                                m, dst = machine(em[0], "enum", enum_names[em[0]]), em[1]
                            elif (isinstance(t, ast.Attribute) and STATE_ATTR.match(attr) and cls
                                  and isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value):
                                m = machine(f"{cls}.{attr}", "string", f"{cls.split(':')[1]}.{attr}")
                                dst = v.value
                                if dst not in m.states:
                                    m.states.append(dst)
                            else:
                                continue
                            if is_init:
                                m.initial.append((dst, f"{fn.id} (constructor)"))
                                continue
                            req = next((g for g in reversed(guards) if g[0] == attr
                                        and (g[2] == m.key or (g[2] is None and m.kind == "string"))), None)
                            for src in (sorted(req[1]) if req else [ANY]):
                                tr = Transition(src, dst, fn.id, st.lineno, attr)
                                if not any((x.src, x.dst, x.function) == (tr.src, tr.dst, tr.function)
                                           for x in m.transitions):
                                    m.transitions.append(tr)
                if isinstance(st, ast.If):
                    visit(st.body, guards + _guard(st.test, enums))
                    visit(st.orelse, guards)
                    if st.body and isinstance(st.body[-1], (ast.Raise, ast.Return)) and not st.orelse:
                        guards += _guard(st.test, enums, negated=True)    # early exit: the rest requires the opposite
                elif isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                    visit(st.body, guards)
                    visit(getattr(st, "orelse", []), guards)
                elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
                    visit(st.body, guards)
                    for h in st.handlers:
                        visit(h.body, guards)
                    visit(st.orelse, guards)
                    visit(st.finalbody, guards)
                elif isinstance(st, ast.Match):
                    for case in st.cases:
                        visit(case.body, guards)

        visit(list(found[0].body), [])

    out = [m for m in machines.values() if m.transitions or m.initial]
    for m in out:
        if m.kind == "enum":                        # keep the declared order, then any extras
            m.states = list(members.get(m.key, [])) + [s for s in m.states if s not in members.get(m.key, [])]
    return sorted(out, key=lambda m: m.key)


__all__ = ["Machine", "Transition", "state_machines"]
