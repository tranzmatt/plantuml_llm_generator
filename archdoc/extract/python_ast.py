"""
Python extractor: repository -> RepoIR, using only the standard-library `ast`.

Passes:
  A. parse every file; build Module / Class / Function shells and import tables
  B. index all symbols; resolve bases, class kinds, fields and attribute types
  C. per function: infer local variable types, resolve calls / creates / raises
  D. framework detectors build the interface inventory (archdoc.extract.frameworks)

Function bodies are read to find calls, raises and attribute types, but no
body text is ever copied into the IR.
"""

from __future__ import annotations

import ast
import builtins
import fnmatch
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..ir import (
    EXTERNAL_PREFIX, Class, Decorator, Field, Function, Module, Param, RepoIR,
)

# Directory names that are never part of a project's source code.
SKIP_DIRS: frozenset = frozenset({
    "venv", ".venv", "env", ".env", "virtualenv", "pyenv", ".pyenv", "conda-env",
    "site-packages", "dist-packages",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "build", "dist",
    ".git", ".hg", ".svn", ".tox", ".nox",
    "node_modules",
    ".idea", ".vscode",
})

MAX_FILE_BYTES = 2_000_000

_BUILTIN_NAMES = frozenset(dir(builtins))
_BUILTIN_EXCEPTIONS = frozenset(
    n for n in _BUILTIN_NAMES
    if isinstance(getattr(builtins, n), type) and issubclass(getattr(builtins, n), BaseException)
)

_ENUM_BASES = {f"{EXTERNAL_PREFIX}enum.{n}" for n in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag")}
_PYDANTIC_BASES = {
    f"{EXTERNAL_PREFIX}pydantic.BaseModel",
    f"{EXTERNAL_PREFIX}pydantic.main.BaseModel",
    f"{EXTERNAL_PREFIX}pydantic_settings.BaseSettings",
    f"{EXTERNAL_PREFIX}pydantic.BaseSettings",
}
_PROTOCOL_BASES = {f"{EXTERNAL_PREFIX}typing.Protocol", f"{EXTERNAL_PREFIX}typing_extensions.Protocol"}
_ABC_BASES = {f"{EXTERNAL_PREFIX}abc.ABC"}
_ORM_BASES = {f"{EXTERNAL_PREFIX}sqlalchemy.orm.DeclarativeBase"}
_DATACLASS_DECORATORS = {
    f"{EXTERNAL_PREFIX}dataclasses.dataclass",
    f"{EXTERNAL_PREFIX}pydantic.dataclasses.dataclass",
}
_FIELD_FUNCS = {
    f"{EXTERNAL_PREFIX}pydantic.Field",
    f"{EXTERNAL_PREFIX}pydantic.fields.Field",
    f"{EXTERNAL_PREFIX}dataclasses.field",
    f"{EXTERNAL_PREFIX}sqlmodel.Field",
}
FIELD_CONSTRAINT_KWARGS = {
    "gt", "ge", "lt", "le", "multiple_of", "min_length", "max_length", "pattern", "regex",
    "max_digits", "decimal_places", "min_items", "max_items", "alias", "strict",
}


# ===========================================================================
# Small AST helpers
# ===========================================================================

def dotted_name(node: ast.AST) -> Optional[str]:
    """Name / Attribute chain -> "a.b.c"; None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def literal_text(node: Optional[ast.AST]) -> Optional[str]:
    """Source text for simple literals, "<expr>" for anything else, None for no node."""
    if node is None:
        return None
    if _is_simple_literal(node):
        return ast.unparse(node)
    return "<expr>"


def _is_simple_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if dotted_name(node) is not None:
        return True     # a name such as Status.ACTIVE or DEFAULT_PORT, never its value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_simple_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_simple_literal(k) and _is_simple_literal(v)
                   for k, v in zip(node.keys, node.values))
    return False


def str_literal(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def visibility_of(name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        return "private"
    if name.startswith("_") and not name.startswith("__"):
        return "protected"
    return "public"


def _unparse(node: Optional[ast.AST]) -> Optional[str]:
    return ast.unparse(node) if node is not None else None


def parse_decorator(node: ast.expr) -> Optional[Decorator]:
    target = node.func if isinstance(node, ast.Call) else node
    name = dotted_name(target)
    if not name:
        return None
    dec = Decorator(name=name)
    if isinstance(node, ast.Call):
        dec.args = [literal_text(a) for a in node.args]
        dec.kwargs = {kw.arg: literal_text(kw.value) for kw in node.keywords if kw.arg}
    return dec


def parse_params(args: ast.arguments) -> List[Param]:
    params: List[Param] = []
    positional = [(a, "positional_only") for a in args.posonlyargs] + [(a, "positional") for a in args.args]
    defaults: List[Optional[ast.expr]] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for (a, kind), d in zip(positional, defaults):
        params.append(Param(a.arg, _unparse(a.annotation), literal_text(d), kind))
    if args.vararg:
        params.append(Param(args.vararg.arg, _unparse(args.vararg.annotation), None, "var_positional"))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        params.append(Param(a.arg, _unparse(a.annotation), literal_text(d), "keyword_only"))
    if args.kwarg:
        params.append(Param(args.kwarg.arg, _unparse(args.kwarg.annotation), None, "var_keyword"))
    return params


def _is_main_guard(node: ast.If) -> bool:
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name) and t.left.id == "__name__"
        and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
        and str_literal(t.comparators[0]) == "__main__"
    )


def _walk_no_nested_scopes(nodes: Iterable[ast.AST]):
    """ast.walk over statements, without descending into nested class bodies."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.ClassDef):
                stack.append(child)


# ===========================================================================
# File discovery and module ids
# ===========================================================================

def collect_python_files(root: str, exclude: Iterable[str] = ()) -> List[str]:
    """Repo-relative, "/"-separated paths of .py files, sorted."""
    exclude = list(exclude)
    out: List[str] = []
    for dirpath, dirnames, fnames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith((".egg-info", ".dist-info"))
        )
        for fname in fnames:
            if not fname.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fname), root).replace(os.sep, "/")
            if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                continue
            out.append(rel)
    return sorted(out)


def module_id_for(rel_path: str, strip_src: bool, root_package: str = "") -> str:
    """root_package: set when the repo root is itself a package (has __init__.py)."""
    parts = rel_path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if strip_src and parts and parts[0] == "src":
        parts = parts[1:]
    if root_package:
        parts = [root_package] + parts
    return ".".join(parts) or "__root__"


# ===========================================================================
# Pass A: parse
# ===========================================================================

@dataclass
class _FuncNode:
    fn: Function
    node: ast.AST                     # FunctionDef | AsyncFunctionDef
    class_id: Optional[str]
    is_static: bool = False


@dataclass
class ParsedModule:
    module: Module
    tree: ast.Module
    is_package: bool
    imports: Dict[str, str] = field(default_factory=dict)      # local name -> qualified name
    defined: Set[str] = field(default_factory=set)             # module-level class/function names
    var_values: Dict[str, ast.expr] = field(default_factory=dict)  # module-level NAME = <expr>
    var_types: Dict[str, str] = field(default_factory=dict)    # module-level NAME -> type id
    class_nodes: Dict[str, ast.ClassDef] = field(default_factory=dict)  # class id -> node
    funcs: List[_FuncNode] = field(default_factory=list)
    # (qualified name, module it came from) per imported name; turned into
    # Module.imports once all internal modules are known
    import_quals: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.module.id

    @property
    def package(self) -> str:
        return self.id if self.is_package else self.id.rpartition(".")[0]


def _resolve_relative(pm: ParsedModule, level: int, module: Optional[str]) -> str:
    base = pm.package.split(".") if pm.package else []
    if level > 1:
        base = base[: len(base) - (level - 1)]
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _parse_module(root: str, rel: str, module_id: str) -> ParsedModule:
    path = os.path.join(root, rel)
    mod = Module(id=module_id, path=rel)
    is_package = rel.endswith("__init__.py")
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise ValueError("file too large")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=rel)
    except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as exc:
        mod.parse_error = type(exc).__name__
        return ParsedModule(mod, ast.Module(body=[], type_ignores=[]), is_package)

    pm = ParsedModule(mod, tree, is_package)
    mod.docstring = ast.get_docstring(tree)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    pm.imports[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    pm.imports[head] = head
                pm.import_quals.append((a.name, a.name))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(pm, node.level, node.module) if node.level else (node.module or "")
            for a in node.names:
                if a.name == "*":
                    continue
                qual = f"{base}.{a.name}" if base else a.name
                pm.imports[a.asname or a.name] = qual
                pm.import_quals.append((qual, base or a.name))
        elif isinstance(node, ast.ClassDef):
            pm.defined.add(node.name)
            mod.classes.extend(_parse_class(pm, node, f"{module_id}:{node.name}"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pm.defined.add(node.name)
            fn = _parse_function(node, f"{module_id}:{node.name}")
            mod.functions.append(fn)
            pm.funcs.append(_FuncNode(fn, node, None))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            pm.var_values[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            pm.var_values[node.target.id] = node.value
        elif isinstance(node, ast.If) and _is_main_guard(node):
            mod.has_main = True
    mod.variables = sorted(pm.var_values)
    return pm


def _parse_function(node, fid: str) -> Function:
    return Function(
        id=fid,
        name=node.name,
        params=parse_params(node.args),
        returns=_unparse(node.returns),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=[d for d in map(parse_decorator, node.decorator_list) if d],
        visibility=visibility_of(node.name),
        docstring=ast.get_docstring(node),
    )


def _parse_class(pm: ParsedModule, node: ast.ClassDef, cid: str) -> List[Class]:
    """Returns the class plus any nested classes (ids "mod:Outer.Inner")."""
    cls = Class(
        id=cid,
        name=node.name,
        decorators=[d for d in map(parse_decorator, node.decorator_list) if d],
        docstring=ast.get_docstring(node),
    )
    pm.class_nodes[cid] = node
    out = [cls]
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = _parse_function(item, f"{cid}.{item.name}")
            cls.methods.append(fn)
            is_static = any(d.name in ("staticmethod",) for d in fn.decorators)
            pm.funcs.append(_FuncNode(fn, item, cid, is_static))
        elif isinstance(item, ast.ClassDef):
            out.extend(_parse_class(pm, item, f"{cid}.{item.name}"))
    return out


# ===========================================================================
# Pass B: symbol index and resolution
# ===========================================================================

class Resolver:
    """Resolves names to IR ids across the whole repo."""

    MAX_DEPTH = 8

    def __init__(self, parsed: Dict[str, ParsedModule]):
        self.modules = parsed
        self.classes: Dict[str, Class] = {}
        self.functions: Dict[str, Function] = {}
        self.module_of_class: Dict[str, str] = {}
        for pm in parsed.values():
            for c in pm.module.classes:
                self.classes[c.id] = c
                self.module_of_class[c.id] = pm.id
                for m in c.methods:
                    self.functions[m.id] = m
            for f in pm.module.functions:
                self.functions[f.id] = f
        self.top_packages = {mid.split(".")[0] for mid in parsed}
        # attribute name -> type id, per class (filled in pass B)
        self.attr_types: Dict[str, Dict[str, str]] = {}

    # -- qualified names ----------------------------------------------------

    def resolve_qualname(self, qual: str, depth: int = 0) -> Optional[str]:
        """'pkg.mod.Class.meth' -> 'pkg.mod:Class.meth'; external -> 'ext:...'."""
        if not qual or depth > self.MAX_DEPTH:
            return None
        parts = qual.split(".")
        for i in range(len(parts), 0, -1):
            mid = ".".join(parts[:i])
            if mid in self.modules:
                return self._resolve_in_module(mid, parts[i:], depth)
        if parts[0] in self.top_packages:
            return None
        return EXTERNAL_PREFIX + qual

    def _resolve_in_module(self, mid: str, rest: List[str], depth: int) -> Optional[str]:
        if not rest:
            return mid
        pm = self.modules[mid]
        head = f"{mid}:{rest[0]}"
        if head in self.classes:
            cur = head
            for j, part in enumerate(rest[1:], start=1):
                if f"{cur}.{part}" in self.classes:
                    cur = f"{cur}.{part}"
                    continue
                meth = self.find_method(cur, part)
                return meth if (meth and j == len(rest) - 1) else None
            return cur
        if head in self.functions:
            return head if len(rest) == 1 else None
        if rest[0] in pm.imports:
            return self.resolve_qualname(".".join([pm.imports[rest[0]]] + rest[1:]), depth + 1)
        return None

    def resolve_name(self, pm: ParsedModule, dotted: str) -> Optional[str]:
        """Resolve a dotted name as written inside module `pm`."""
        parts = dotted.split(".")
        head = parts[0]
        if head in pm.defined:
            return self.resolve_qualname(f"{pm.id}.{dotted}")
        if head in pm.imports:
            return self.resolve_qualname(".".join([pm.imports[head]] + parts[1:]))
        return None

    def resolve_var(self, pm: ParsedModule, name: str, depth: int = 0) -> Optional[Tuple[ParsedModule, str]]:
        """Find the module-level variable `name` refers to, following imports."""
        if name in pm.var_values:
            return pm, name
        if name in pm.imports and depth < self.MAX_DEPTH:
            mid, _, var = pm.imports[name].rpartition(".")
            if mid in self.modules:
                return self.resolve_var(self.modules[mid], var, depth + 1)
        return None

    # -- classes ------------------------------------------------------------

    def mro(self, class_id: str) -> List[str]:
        """The class followed by its internal ancestors, breadth-first."""
        seen: List[str] = []
        queue = [class_id]
        while queue:
            cid = queue.pop(0)
            if cid in seen or cid not in self.classes:
                continue
            seen.append(cid)
            queue.extend(self.classes[cid].bases)
        return seen

    def ancestors_all(self, class_id: str) -> Set[str]:
        """Internal ancestors plus every external base reachable from them."""
        out: Set[str] = set()
        for cid in self.mro(class_id):
            out.add(cid)
            out.update(self.classes[cid].bases)
        return out

    def find_method(self, class_id: str, name: str) -> Optional[str]:
        for cid in self.mro(class_id):
            mid = f"{cid}.{name}"
            if mid in self.functions:
                return mid
        return None

    def attr_type(self, class_id: str, attr: str) -> Optional[str]:
        for cid in self.mro(class_id):
            t = self.attr_types.get(cid, {}).get(attr)
            if t:
                return t
        return None

    # -- annotations --------------------------------------------------------

    def annotation_types(self, pm: ParsedModule, node: Optional[ast.AST]) -> List[str]:
        """Type ids (internal classes or ext:...) referenced by an annotation, in order."""
        if node is None:
            return []
        out: List[str] = []

        def visit(n: ast.AST) -> None:
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                try:
                    visit(ast.parse(n.value, mode="eval").body)
                except SyntaxError:
                    pass
                return
            dn = dotted_name(n)
            if dn is not None:
                tid = self.resolve_name(pm, dn)
                if tid and tid not in out and (tid in self.classes or tid.startswith(EXTERNAL_PREFIX)):
                    out.append(tid)
                return
            for child in ast.iter_child_nodes(n):
                visit(child)

        visit(node)
        return out

    def primary_type(self, pm: ParsedModule, node: Optional[ast.AST]) -> Optional[str]:
        """The most useful single type for a variable: first internal class, else
        the outermost external class (skipping typing wrappers like Optional)."""
        types = self.annotation_types(pm, node)
        internal = [t for t in types if t in self.classes]
        if internal:
            return internal[0]
        for t in types:
            if not t.startswith(f"{EXTERNAL_PREFIX}typing.") and t.rsplit(".", 1)[-1][:1].isupper():
                return t
        return None


def _call_result_type(res: Resolver, pm: ParsedModule, call: ast.Call,
                      env: Dict[str, str], cls_id: Optional[str]) -> Optional[str]:
    """Type produced by evaluating `call`: a class instantiation or an annotated return."""
    callee = _resolve_callable(res, pm, call.func, env, cls_id)
    if callee is None:
        return None
    if callee in res.classes:
        return callee
    if callee in res.functions:
        fn = res.functions[callee]
        if fn.returns:
            node = ast.parse(fn.returns, mode="eval").body
            fpm = res.modules[callee.split(":")[0]]
            return res.primary_type(fpm, node)
        return None
    if callee.startswith(EXTERNAL_PREFIX) and callee.rsplit(".", 1)[-1][:1].isupper():
        return callee        # ext class instantiation, e.g. ext:requests.Session
    return None


def _type_of(res: Resolver, pm: ParsedModule, expr: ast.AST,
             env: Dict[str, str], cls_id: Optional[str]) -> Optional[str]:
    if isinstance(expr, ast.Name):
        if expr.id in env:
            return env[expr.id]
        return pm.var_types.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _type_of(res, pm, expr.value, env, cls_id)
        if base and base in res.classes:
            return res.attr_type(base, expr.attr)
        return None
    if isinstance(expr, ast.Call):
        return _call_result_type(res, pm, expr, env, cls_id)
    return None


def _resolve_callable(res: Resolver, pm: ParsedModule, func: ast.AST,
                      env: Dict[str, str], cls_id: Optional[str]) -> Optional[str]:
    """Resolve the target of a call expression to an id, or None."""
    if isinstance(func, ast.Attribute):
        recv = func.value
        # super().method()
        if (isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name)
                and recv.func.id == "super" and cls_id):
            for base in res.classes[cls_id].bases:
                if base in res.classes:
                    found = res.find_method(base, func.attr)
                    if found:
                        return found
            return None
        recv_type = _type_of(res, pm, recv, env, cls_id)
        if recv_type:
            if recv_type in res.classes:
                return res.find_method(recv_type, func.attr)
            if recv_type.startswith(EXTERNAL_PREFIX):
                return f"{recv_type}.{func.attr}"
            return None
    dn = dotted_name(func)
    if dn is None:
        return None
    head = dn.split(".")[0]
    if head in env or head in pm.var_types:
        return None          # a variable whose type we could not determine
    target = res.resolve_name(pm, dn)
    if target is None or target in res.modules:
        return None
    return target


def _build_index(res: Resolver) -> None:
    """Pass B: bases, module var types, class kinds, fields, attribute types."""
    for pm in res.modules.values():
        for cid, node in pm.class_nodes.items():
            cls = res.classes[cid]
            for b in node.bases:
                dn = dotted_name(b) if not isinstance(b, ast.Subscript) else dotted_name(b.value)
                if dn:
                    cls.bases.append(res.resolve_name(pm, dn) or dn)

    for pm in res.modules.values():
        for name, value in pm.var_values.items():
            if isinstance(value, ast.Call):
                t = _call_result_type(res, pm, value, {}, None)
                if t:
                    pm.var_types[name] = t
        pm.module.imports = sorted({
            _import_target(res, qual, from_module) for qual, from_module in pm.import_quals
        } - {None, pm.id})

    for pm in res.modules.values():
        for cid, node in pm.class_nodes.items():
            _classify(res, pm, res.classes[cid], node)
    for pm in res.modules.values():
        for cid, node in pm.class_nodes.items():
            _collect_fields(res, pm, res.classes[cid], node)


def _import_target(res: Resolver, qual: str, from_module: str) -> Optional[str]:
    """Imported dotted name -> the internal module id it lives in, or ext:<module>."""
    resolved = res.resolve_qualname(qual)
    if resolved and not resolved.startswith(EXTERNAL_PREFIX):
        return resolved.split(":")[0]       # follows re-exports to the defining module
    parts = qual.split(".")
    for i in range(len(parts), 0, -1):
        mid = ".".join(parts[:i])
        if mid in res.modules:
            return mid
    if parts[0] in res.top_packages:
        return None
    return EXTERNAL_PREFIX + from_module


def _classify(res: Resolver, pm: ParsedModule, cls: Class, node: ast.ClassDef) -> None:
    ancestors = res.ancestors_all(cls.id)
    deco_ids = {res.resolve_name(pm, d.name) for d in cls.decorators} | {d.name for d in cls.decorators}
    has_tablename = any(
        isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in s.targets)
        for s in node.body
    )
    metaclass_abc = any(
        kw.arg == "metaclass" and (dotted_name(kw.value) or "").endswith("ABCMeta") for kw in node.keywords
    )
    if ancestors & _ENUM_BASES:
        cls.kind = "enum"
    elif ancestors & _PYDANTIC_BASES:
        cls.kind = "pydantic"
    elif has_tablename or ancestors & _ORM_BASES:
        cls.kind = "orm"
    elif deco_ids & _DATACLASS_DECORATORS or "dataclass" in deco_ids:
        cls.kind = "dataclass"
    elif ancestors & _PROTOCOL_BASES:
        cls.kind = "protocol"
    elif ancestors & _ABC_BASES or metaclass_abc:
        cls.kind = "abc"


def _parse_field_call(res: Resolver, pm: ParsedModule, value: ast.expr, f: Field) -> bool:
    """Fill default/constraints/description from Field(...)/field(...). Returns True if handled."""
    if not isinstance(value, ast.Call):
        return False
    dn = dotted_name(value.func)
    if not dn or res.resolve_name(pm, dn) not in _FIELD_FUNCS:
        return False
    default: Optional[ast.expr] = value.args[0] if value.args else None
    has_default = default is not None and not (isinstance(default, ast.Constant) and default.value is Ellipsis)
    for kw in value.keywords:
        if kw.arg == "default":
            default, has_default = kw.value, True
        elif kw.arg == "default_factory":
            f.default, has_default = "<factory>", True
        elif kw.arg == "description":
            f.docstring = str_literal(kw.value)
        elif kw.arg in FIELD_CONSTRAINT_KWARGS:
            f.constraints[kw.arg] = literal_text(kw.value)
    if has_default and f.default is None:
        f.default = literal_text(default)
    return True


def _collect_fields(res: Resolver, pm: ParsedModule, cls: Class, node: ast.ClassDef) -> None:
    structured = cls.kind in ("pydantic", "dataclass", "orm")
    attr_types: Dict[str, str] = {}
    body = node.body

    for idx, stmt in enumerate(body):
        if cls.kind == "enum" and isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    cls.enum_members.append(t.id)
            continue
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        ann = ast.unparse(stmt.annotation)
        if ann.startswith(("ClassVar", "typing.ClassVar")):
            continue
        f = Field(name=stmt.target.id, annotation=ann)
        handled = stmt.value is not None and _parse_field_call(res, pm, stmt.value, f)
        if stmt.value is not None and not handled:
            f.default = literal_text(stmt.value)
        if structured:
            f.required = f.default is None
        # PEP 257 attribute docstring: a bare string right after the field
        if f.docstring is None and idx + 1 < len(body):
            nxt = body[idx + 1]
            if isinstance(nxt, ast.Expr):
                f.docstring = str_literal(nxt.value)
        f.type_id = res.primary_type(pm, stmt.annotation)
        if f.type_id:
            attr_types[f.name] = f.type_id
        cls.fields.append(f)

    # Instance attributes assigned in methods: self.x = ... / self.x: T = ...
    known = {f.name for f in cls.fields}
    for item in body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        param_ann = {a.arg: a.annotation for a in item.args.args + item.args.kwonlyargs if a.annotation}
        for stmt in _walk_no_nested_scopes(item.body):
            target, value, ann = None, None, None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target, value, ann = stmt.target, stmt.value, stmt.annotation
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue
            name = target.attr
            tid: Optional[str] = None
            ann_text: Optional[str] = None
            if ann is not None:
                tid, ann_text = res.primary_type(pm, ann), ast.unparse(ann)
            elif isinstance(value, ast.Name) and value.id in param_ann:
                tid, ann_text = res.primary_type(pm, param_ann[value.id]), ast.unparse(param_ann[value.id])
            elif isinstance(value, ast.Call):
                tid = _call_result_type(res, pm, value, {"self": cls.id}, cls.id)
                if tid:
                    ann_text = tid.split(":")[-1].removeprefix(EXTERNAL_PREFIX)
            if tid:
                attr_types.setdefault(name, tid)
            if name not in known:
                known.add(name)
                cls.fields.append(Field(name=name, annotation=ann_text, type_id=tid if tid in res.classes else None))
    res.attr_types[cls.id] = attr_types


# ===========================================================================
# Pass C: calls, creates, raises
# ===========================================================================

def _function_env(res: Resolver, pm: ParsedModule, fnode: _FuncNode) -> Dict[str, str]:
    node = fnode.node
    env: Dict[str, str] = {}
    all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    if fnode.class_id and all_args and not fnode.is_static:
        env[all_args[0].arg] = fnode.class_id        # self / cls
        all_args = all_args[1:]
    for a in all_args:
        t = res.primary_type(pm, a.annotation)
        if t:
            env[a.arg] = t
    # Flow-insensitive: later assignments overwrite earlier ones; good enough for diagrams.
    for stmt in _walk_no_nested_scopes(node.body):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if isinstance(stmt.value, ast.Call):
                t = _call_result_type(res, pm, stmt.value, env, fnode.class_id)
                if t:
                    env[stmt.targets[0].id] = t
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            t = res.primary_type(pm, stmt.annotation)
            if t:
                env[stmt.target.id] = t
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if isinstance(item.optional_vars, ast.Name) and isinstance(item.context_expr, ast.Call):
                    t = _call_result_type(res, pm, item.context_expr, env, fnode.class_id)
                    if t:
                        env[item.optional_vars.id] = t
    return env


def _resolve_bodies(res: Resolver) -> None:
    for pm in res.modules.values():
        for fnode in pm.funcs:
            env = _function_env(res, pm, fnode)
            calls: Set[str] = set()
            creates: Set[str] = set()
            raises: Set[str] = set()
            for n in _walk_no_nested_scopes(fnode.node.body):
                if isinstance(n, ast.Call):
                    target = _resolve_callable(res, pm, n.func, env, fnode.class_id)
                    if target is None:
                        continue
                    if target in res.classes:
                        creates.add(target)
                    elif target != fnode.fn.id:
                        calls.add(target)
                elif isinstance(n, ast.Raise) and n.exc is not None:
                    exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
                    dn = dotted_name(exc)
                    if not dn:
                        continue
                    rid = res.resolve_name(pm, dn)
                    if rid and (rid in res.classes or rid.startswith(EXTERNAL_PREFIX)):
                        raises.add(rid)
                    elif dn in _BUILTIN_EXCEPTIONS:
                        raises.add(dn)
            fnode.fn.calls = sorted(calls)
            fnode.fn.creates = sorted(creates)
            fnode.fn.raises = sorted(raises)


# ===========================================================================
# Entry point
# ===========================================================================

@dataclass
class ExtractContext:
    """What framework detectors get to work with."""
    root: str
    ir: RepoIR
    resolver: Resolver

    @property
    def modules(self) -> Dict[str, ParsedModule]:
        return self.resolver.modules


def extract_python(root: str, exclude: Iterable[str] = (), detectors: Optional[list] = None) -> RepoIR:
    """Extract a RepoIR (structure + interface inventory) from a Python repo."""
    root = os.path.abspath(root)
    files = collect_python_files(root, exclude)
    strip_src = os.path.isdir(os.path.join(root, "src")) and not os.path.exists(os.path.join(root, "src", "__init__.py"))
    root_package = os.path.basename(root) if os.path.exists(os.path.join(root, "__init__.py")) else ""

    parsed: Dict[str, ParsedModule] = {}
    for rel in files:
        mid = module_id_for(rel, strip_src, root_package)
        if mid in parsed:       # e.g. both pkg.py and pkg/__init__.py; keep the first
            continue
        parsed[mid] = _parse_module(root, rel, mid)

    res = Resolver(parsed)
    _build_index(res)
    _resolve_bodies(res)

    ir = RepoIR(repo=os.path.basename(root), modules=[parsed[m].module for m in sorted(parsed)])
    ctx = ExtractContext(root=root, ir=ir, resolver=res)

    if detectors is None:
        from .frameworks import DEFAULT_DETECTORS
        detectors = DEFAULT_DETECTORS
    for detect in detectors:
        detect(ctx)
    ir.interfaces.sort(key=lambda i: i.id)
    ir.data_types.sort(key=lambda d: d.id)
    return ir
