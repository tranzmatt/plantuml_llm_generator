"""
Activity structure of one function, from its AST (DESIGN.md 2.2, 9).

The diagram's shape is the code's: every decision, loop, try, with, match,
return and raise, and every statement that calls something, in source order.
Nothing is added or dropped by a model. Each node has a deterministic label
(the calls as written, the condition as written, shortened and secret-scanned);
a model may only replace labels with prose, keyed by node id (archdoc.plan
`plan_activity`), so the verification rule "every action maps to a call or
control-flow node of the function" holds by construction.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .extract.bodies import _NOISE, locate
from .extract.python_ast import dotted_name
from .ir import RepoIR
from .secrets_scan import redact

MAX_TEXT = 60


@dataclass
class Node:
    kind: str                  # action | if | loop | try | with | match | return | raise | break | continue | def
    id: str = ""               # n1, n2 ... for labelable nodes
    text: str = ""             # deterministic label
    calls: List[str] = field(default_factory=list)    # as written
    line: int = 0
    body: List["Node"] = field(default_factory=list)
    branches: List["Branch"] = field(default_factory=list)


@dataclass
class Branch:
    kind: str                  # cond | else | except | finally | case
    id: str = ""
    text: str = ""
    line: int = 0
    calls: List[str] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class Activity:
    function: str
    nodes: List[Node]

    def labelable(self) -> List[object]:
        """Every node and branch with an id, in source order."""
        out: List[object] = []

        def walk(nodes: List[Node]) -> None:
            for n in nodes:
                if n.id:
                    out.append(n)
                walk(n.body)                      # a try's body comes before its handlers
                for b in n.branches:
                    if b.id:
                        out.append(b)
                    walk(b.body)
        walk(self.nodes)
        return out

    @property
    def decisions(self) -> int:
        return sum(1 for x in self.labelable() if getattr(x, "kind", "") in ("if", "loop", "match", "try"))


def _short(text: str) -> str:
    text = " ".join(redact(text)[0].split())
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return _short(ast.unparse(node))
    except Exception:              # noqa: BLE001 - unparse of odd nodes
        return "…"


def _calls(node: ast.AST) -> List[str]:
    out: List[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = dotted_name(n.func)
            if name and name not in out and name not in _NOISE:
                out.append(name)
    return out


class _Builder:
    def __init__(self) -> None:
        self.n = 0

    def nid(self) -> str:
        self.n += 1
        return f"n{self.n}"

    def block(self, body: List[ast.stmt]) -> List[Node]:
        out: List[Node] = []
        for st in body:
            node = self.stmt(st)
            if node is not None:
                out.append(node)
        return out

    def stmt(self, st: ast.stmt) -> Optional[Node]:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return Node("def", text=f"define {st.name}", line=st.lineno)
        if isinstance(st, ast.If):
            node = Node("if", id=self.nid(), line=st.lineno)
            node.text, node.calls = _unparse(st.test), _calls(st.test)
            node.branches.append(Branch("cond", text=node.text, line=st.lineno, calls=node.calls,
                                        body=self.block(st.body)))
            orelse = st.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                e = orelse[0]
                node.branches.append(Branch("cond", id=self.nid(), text=_unparse(e.test), line=e.lineno,
                                            calls=_calls(e.test), body=self.block(e.body)))
                orelse = e.orelse
            if orelse:
                node.branches.append(Branch("else", body=self.block(orelse)))
            return node
        if isinstance(st, (ast.For, ast.AsyncFor)):
            text = f"for {_unparse(st.target)} in {_unparse(st.iter)}"
            return Node("loop", id=self.nid(), text=_short(text), calls=_calls(st.iter), line=st.lineno,
                        body=self.block(st.body))
        if isinstance(st, ast.While):
            return Node("loop", id=self.nid(), text=f"while {_unparse(st.test)}", calls=_calls(st.test),
                        line=st.lineno, body=self.block(st.body))
        if isinstance(st, (ast.With, ast.AsyncWith)):
            exprs = [i.context_expr for i in st.items]
            calls = [c for e in exprs for c in _calls(e)]
            return Node("with", id=self.nid(), text=_short("with " + ", ".join(_unparse(e) for e in exprs)),
                        calls=calls, line=st.lineno, body=self.block(st.body))
        if isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
            node = Node("try", id=self.nid(), text="try", line=st.lineno, body=self.block(st.body))
            for h in st.handlers:
                t = _unparse(h.type) if h.type is not None else "any exception"
                node.branches.append(Branch("except", id=self.nid(), text=t, line=h.lineno, body=self.block(h.body)))
            if st.orelse:
                node.branches.append(Branch("else", text="no exception", body=self.block(st.orelse)))
            if st.finalbody:
                node.branches.append(Branch("finally", text="finally", body=self.block(st.finalbody)))
            return node
        if isinstance(st, ast.Match):
            node = Node("match", id=self.nid(), text=_unparse(st.subject), calls=_calls(st.subject), line=st.lineno)
            for case in st.cases:
                node.branches.append(Branch("case", id=self.nid(), text=_unparse(case.pattern),
                                            line=case.pattern.lineno, body=self.block(case.body)))
            return node
        if isinstance(st, ast.Return):
            return Node("return", id=self.nid(), text=_short("return " + _unparse(st.value)).strip(),
                        calls=_calls(st.value) if st.value else [], line=st.lineno)
        if isinstance(st, ast.Raise):
            exc = st.exc.func if isinstance(st.exc, ast.Call) else st.exc
            return Node("raise", id=self.nid(), text=f"raise {dotted_name(exc) or _unparse(exc)}" if exc else "re-raise",
                        line=st.lineno)
        if isinstance(st, ast.Break):
            return Node("break", line=st.lineno)
        if isinstance(st, ast.Continue):
            return Node("continue", text="next iteration", line=st.lineno)
        calls = _calls(st)
        yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(st))
        if not calls and not yields:
            return None                             # plain assignments: no activity of their own
        text = ", ".join(f"{c}()" for c in calls)
        return Node("action", id=self.nid(), text=_short(("yield " if yields else "") + text).strip(),
                    calls=calls, line=st.lineno)


def build_activity(root: str, ir: RepoIR, function_id: str) -> Optional[Activity]:
    found = locate(root, ir, function_id)
    if found is None:
        return None
    node = found[0]
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return Activity(function_id, _Builder().block(body))


def iter_nodes(nodes: List[Node]) -> Iterator[Node]:
    for n in nodes:
        yield n
        yield from iter_nodes(n.body)
        for b in n.branches:
            yield from iter_nodes(b.body)


def has_control_flow(act: Activity) -> bool:
    return any(n.kind in ("if", "loop", "try", "match", "with") for n in iter_nodes(act.nodes))


def label_of(item, labels: Dict[str, str]) -> str:
    return labels.get(getattr(item, "id", ""), "") or item.text


__all__ = ["Activity", "Branch", "Node", "build_activity", "has_control_flow", "iter_nodes", "label_of"]
