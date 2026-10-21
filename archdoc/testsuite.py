"""
Test inventory (DESIGN.md 2.3): the repo's tests read as acceptance criteria.

Tests are evidence, not components: they are extracted with the rest of the
code (even when the diagrams exclude them) and never drawn. For every test:
where it is, what code it exercises (calls and instantiations that resolve
to the documented code), which exceptions it expects (`pytest.raises`,
`assertRaises`), how many assertions it makes, its parametrized cases,
fixtures and markers. All of that comes from the AST. A model adds one
sentence per test stating the behavior it requires, written from the test's
source (local) or its signature.
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from .extract.bodies import locate
from .extract.python_ast import dotted_name
from .ir import EXTERNAL_PREFIX, Module, RepoIR

TEST_FILE_GLOBS = ("test_*.py", "*_test.py")
TEST_DIRS = {"tests", "test", "testing"}


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    base = parts[-1]
    if base == "conftest.py":
        return False
    return any(fnmatch.fnmatch(base, g) for g in TEST_FILE_GLOBS) or (
        any(p in TEST_DIRS for p in parts[:-1]) and base.startswith("test"))


def inventory_excludes(excludes: Iterable[str]) -> List[str]:
    """The user's excludes minus those that only hide tests: tests are read as evidence anyway."""
    return [p for p in excludes if "test" not in p.lower()]


@dataclass
class TestCase:
    __test__ = False                   # not a pytest test class

    id: str
    module: str
    path: str
    line: int
    name: str
    cls: Optional[str] = None
    doc: Optional[str] = None
    exercises: List[str] = field(default_factory=list)     # documented function / class ids
    expects: List[str] = field(default_factory=list)       # exception types it requires
    asserts: int = 0
    cases: Optional[int] = None                             # parametrized cases, when countable
    fixtures: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    criterion: Optional[str] = None                         # model-written acceptance criterion
    criterion_generated: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


_ASSERT_METHODS = ("assert", "fail")


def _body_facts(node: ast.AST) -> Dict[str, object]:
    asserts, expects = 0, []
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            asserts += 1
        elif isinstance(n, ast.Call):
            name = dotted_name(n.func) or ""
            tail = name.rsplit(".", 1)[-1]
            if name.startswith("self.") and tail.startswith(_ASSERT_METHODS):
                asserts += 1
            if tail in ("raises", "assertRaises", "assertRaisesRegex") and n.args:
                exc = n.args[0]
                types = exc.elts if isinstance(exc, ast.Tuple) else [exc]
                for t in types:
                    tn = dotted_name(t)
                    if tn and tn not in expects:
                        expects.append(tn)
                if tail == "raises":
                    asserts += 1
    return {"asserts": asserts, "expects": expects}


def _cases(decorators: List[ast.expr]) -> Optional[int]:
    total = None
    for d in decorators:
        if isinstance(d, ast.Call) and (dotted_name(d.func) or "").endswith("mark.parametrize") and len(d.args) >= 2:
            vals = d.args[1]
            if isinstance(vals, (ast.List, ast.Tuple)):
                total = (total or 1) * len(vals.elts)
    return total


def _markers(decorators: List[ast.expr]) -> List[str]:
    out = []
    for d in decorators:
        name = dotted_name(d.func if isinstance(d, ast.Call) else d) or ""
        if ".mark." in f".{name}" and not name.endswith("parametrize"):
            out.append(name.rsplit(".", 1)[-1])
        elif name.rsplit(".", 1)[-1] in ("skip", "skipIf", "skipUnless", "expectedFailure"):
            out.append(name.rsplit(".", 1)[-1])
    return out


def _parametrized_names(decorators: List[ast.expr]) -> List[str]:
    names = []
    for d in decorators:
        if isinstance(d, ast.Call) and (dotted_name(d.func) or "").endswith("mark.parametrize") and d.args:
            a = d.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                names += [x.strip() for x in a.value.split(",")]
            elif isinstance(a, (ast.List, ast.Tuple)):
                names += [e.value for e in a.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return names


def _is_test_class(cls) -> bool:
    return cls.name.startswith("Test") or any(b.split(".")[-1].endswith("TestCase") for b in cls.bases)


def _tests_of(m: Module):
    for f in m.functions:
        if f.name.startswith("test"):
            yield f, None
    for c in m.classes:
        if _is_test_class(c):
            for f in c.methods:
                if f.name.startswith("test"):
                    yield f, c.name


def build_inventory(root: str, ir_all: RepoIR, documented: RepoIR) -> List[TestCase]:
    """Test cases in `ir_all` (the repo including its tests); `exercises` keeps
    only ids that `documented` (what the documents describe) contains."""
    known = {f.id for f in documented.iter_functions()} | {c.id for c in documented.iter_classes()}
    out: List[TestCase] = []
    for m in ir_all.modules:
        if not is_test_path(m.path):
            continue
        for fn, cls in _tests_of(m):
            found = locate(root, ir_all, fn.id)
            node = found[0] if found else None
            facts = _body_facts(node) if node is not None else {"asserts": 0, "expects": []}
            decos = list(getattr(node, "decorator_list", []))
            exercised = [c for c in dict.fromkeys(fn.calls + fn.creates)
                         if not c.startswith(EXTERNAL_PREFIX) and c in known]
            params = [p.name for p in fn.params if p.name not in ("self", "cls")]
            pnames = set(_parametrized_names(decos))
            out.append(TestCase(
                id=fn.id, module=m.id, path=m.path, line=getattr(node, "lineno", 0), name=fn.name, cls=cls,
                doc=(fn.docstring or "").strip().split("\n", 1)[0] or None,
                exercises=exercised, expects=list(facts["expects"]), asserts=int(facts["asserts"]),
                cases=_cases(decos), fixtures=[p for p in params if p not in pnames], markers=_markers(decos),
            ))
    return out


def exercised_by(tests: List[TestCase]) -> Dict[str, List[str]]:
    """documented id -> test ids that exercise it directly."""
    out: Dict[str, List[str]] = {}
    for t in tests:
        for x in t.exercises:
            out.setdefault(x, []).append(t.id)
    return out


__all__ = ["TestCase", "build_inventory", "exercised_by", "is_test_path", "inventory_excludes"]
