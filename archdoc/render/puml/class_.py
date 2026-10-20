"""
Deterministic class diagram from the IR.

  - classes grouped into one package per module
  - inheritance (<|-- for classes, <|.. for protocols)
  - associations from typed fields (*-- for data classes, --> otherwise)
  - dependencies from methods: creates / uses another class
  - classes outside `scope` that are related to classes inside it are drawn
    as member-less stubs, so every edge has both ends declared
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ...ir import EXTERNAL_PREFIX, Class, Field, Function, RepoIR
from .common import Aliases, Diagram, label, member_text

_VIS = {"public": "+", "protected": "#", "private": "-"}
_COLLECTION_RE = re.compile(r"^(typing\.)?(list|List|set|Set|frozenset|FrozenSet|tuple|Tuple|Sequence|"
                            r"Iterable|Iterator|dict|Dict|Mapping|collections\.abc\.\w+)\[")
_KIND_STEREOTYPES = {"dataclass": "dataclass", "pydantic": "pydantic", "orm": "entity"}
# external/builtin bases already implied by the class kind
_IMPLIED_BASES = {"BaseModel", "BaseSettings", "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
                  "Protocol", "ABC", "object", "str", "int", "Generic", "DeclarativeBase"}
MAX_PARAMS_SHOWN = 4


def in_scope(module_id: str, scope: Optional[str]) -> bool:
    return scope is None or module_id == scope or module_id.startswith(scope + ".")


def partition_modules(ir: RepoIR, max_classes: int) -> List[Tuple[str, Set[str]]]:
    """Split modules into groups of at most `max_classes` classes.

    A package over the limit is split one level deeper, then its small
    sub-groups are packed back together (in name order) up to the limit, so
    diagrams are neither huge nor fragmented. Each module lands in exactly one
    group; a single module over the limit becomes its own group. Groups
    without classes are dropped."""
    counts = Counter(_module_of(c.id) for c in ir.iter_classes())

    def split(mods: List[str], depth: int) -> List[Tuple[str, Set[str], int]]:
        groups: Dict[str, List[str]] = {}
        for m in mods:
            groups.setdefault(".".join(m.split(".")[:depth]), []).append(m)
        leaves: List[Tuple[str, Set[str], int]] = []
        for g, members in sorted(groups.items()):
            n = sum(counts[m] for m in members)
            if n == 0:
                continue
            deeper = [m for m in members if len(m.split(".")) > depth]
            if n <= max_classes or not deeper:
                leaves.append((g, set(members), n))
            else:
                if g in members and counts[g]:
                    leaves.append((g, {g}, counts[g]))     # the package's own __init__
                leaves.extend(split(deeper, depth + 1))
        return leaves

    leaves = split(sorted(m.id for m in ir.modules), 1)

    # pack adjacent small leaves that share a parent package
    packed: List[Tuple[str, Set[str], int]] = []
    for label, mods, n in leaves:
        parent = label.rsplit(".", 1)[0] if "." in label else ""
        if packed:
            plabel, pmods, pn = packed[-1]
            pparent = plabel if plabel == parent else (plabel.rsplit(".", 1)[0] if "." in plabel else "")
            if pparent == parent and pn + n <= max_classes:
                packed[-1] = (parent or plabel, pmods | mods, pn + n)
                continue
        packed.append((label, mods, n))

    # disambiguate labels that packing made identical
    seen: Counter = Counter(label for label, _, _ in packed)
    index: Counter = Counter()
    out: List[Tuple[str, Set[str]]] = []
    for label, mods, _ in packed:
        if seen[label] > 1:
            index[label] += 1
            label = f"{label}_part{index[label]}"
        out.append((label, mods))
    return out


def _module_of(class_id: str) -> str:
    return class_id.split(":", 1)[0]


def _display_name(cls: Class) -> str:
    return cls.id.split(":", 1)[1]


def _short_base(base: str) -> str:
    return base[len(EXTERNAL_PREFIX):].rsplit(".", 1)[-1] if base.startswith(EXTERNAL_PREFIX) else base


def _include_member(name: str, visibility: str, members: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return name == "__init__"
    return members == "all" or visibility == "public"


def _method_line(fn: Function, is_static: bool) -> str:
    params = fn.params
    if not is_static and params and params[0].kind in ("positional", "positional_only"):
        params = params[1:]
    parts = []
    for p in params[:MAX_PARAMS_SHOWN]:
        prefix = "*" if p.kind == "var_positional" else "**" if p.kind == "var_keyword" else ""
        parts.append(f"{prefix}{p.name}: {p.annotation}" if p.annotation else f"{prefix}{p.name}")
    if len(params) > MAX_PARAMS_SHOWN:
        parts.append("...")
    mods = ""
    if any(d.name.endswith("abstractmethod") for d in fn.decorators):
        mods += "{abstract} "
    if is_static or any(d.name == "classmethod" for d in fn.decorators):
        mods += "{static} "
    ret = f" : {fn.returns}" if fn.returns else ""
    prefix = "async " if fn.is_async else ""
    return member_text(f"{_VIS[fn.visibility]}{mods}{prefix}{fn.name}({', '.join(parts)}){ret}")


def _field_line(f: Field) -> str:
    vis = "-" if f.name.startswith("__") else "#" if f.name.startswith("_") else "+"
    ann = f" : {f.annotation}" if f.annotation else ""
    return member_text(f"{vis}{f.name}{ann}")


def _declaration(cls: Class, alias: str, body: Optional[List[str]]) -> List[str]:
    if cls.kind == "enum":
        keyword = "enum"
    elif cls.kind == "protocol":
        keyword = "interface"
    elif cls.kind == "abc":
        keyword = "abstract class"
    else:
        keyword = "class"
    stereos = []
    if cls.kind in _KIND_STEREOTYPES:
        stereos.append(_KIND_STEREOTYPES[cls.kind])
    for b in cls.bases:
        if ":" in b and not b.startswith(EXTERNAL_PREFIX):
            continue                                   # internal base: drawn as an edge
        short = _short_base(b)
        if short not in _IMPLIED_BASES:
            stereos.append(short)
    stereo = "".join(f" <<{s}>>" for s in stereos)
    head = f"{keyword} {label(_display_name(cls))} as {alias}{stereo}"
    if body is None:
        return [f"  {head}"]
    return [f"  {head} {{"] + [f"    {ln}" for ln in body] + ["  }"]


def render_class_diagram(
    ir: RepoIR,
    name: str,
    scope: Optional[str] = None,
    modules: Optional[Iterable[str]] = None,
    members: str = "public",
    show_dependencies: bool = True,
    title: Optional[str] = None,
) -> str:
    """Classes in `modules` (exact module ids) or under `scope` (package prefix),
    or all classes when neither is given."""
    classes: Dict[str, Class] = {c.id: c for c in ir.iter_classes()}
    wanted = set(modules) if modules is not None else None
    focus = [c for c in classes.values()
             if (_module_of(c.id) in wanted if wanted is not None else in_scope(_module_of(c.id), scope))]
    focus_ids = {c.id for c in focus}
    aliases = Aliases()

    edges: List[Tuple[str, str, str]] = []          # (line, from id, to id)
    related: Set[str] = set()
    assoc_pairs: Set[Tuple[str, str]] = set()

    def rel(line_fmt: str, a: str, b: str) -> None:
        related.update((a, b))
        edges.append((line_fmt, a, b))

    for c in focus:
        for b in c.bases:
            if b in classes:
                arrow = "<|.." if classes[b].kind == "protocol" else "<|--"
                rel(f"{{{b}}} {arrow} {{{c.id}}}", b, c.id)
        for f in c.fields:
            if f.type_id and f.type_id in classes and f.type_id != c.id:
                many = bool(f.annotation and _COLLECTION_RE.match(f.annotation))
                arrow = "*--" if c.kind in ("dataclass", "pydantic", "orm") else "-->"
                mult = ' "*"' if many else ""
                rel(f"{{{c.id}}} {arrow}{mult} {{{f.type_id}}} : {f.name}", c.id, f.type_id)
                assoc_pairs.add((c.id, f.type_id))
        if not show_dependencies:
            continue
        deps: Dict[str, str] = {}
        for m in c.methods:
            for target in m.creates:
                if target in classes and target != c.id:
                    deps[target] = "creates"
            for call in m.calls:
                owner = call.rsplit(".", 1)[0] if ":" in call and "." in call.split(":", 1)[1] else None
                if owner in classes and owner != c.id:
                    deps.setdefault(owner, "uses")
        for target, kind in sorted(deps.items()):
            if (c.id, target) not in assoc_pairs and target not in c.bases:
                rel(f"{{{c.id}}} ..> {{{target}}} : {kind}", c.id, target)

    shown = focus_ids | {x for x in related if x in classes}
    by_module: Dict[str, List[Class]] = {}
    for cid in shown:
        by_module.setdefault(_module_of(cid), []).append(classes[cid])

    diagram = Diagram(name, title, settings=[
        "set namespaceSeparator none",
        "hide empty members",
        "skinparam classAttributeIconSize 0",
    ])
    for mid in sorted(by_module):
        palias = aliases(mid, "P")
        diagram.add(f"package {label(mid)} as {palias} {{")
        for cls in sorted(by_module[mid], key=lambda k: k.id):
            alias = aliases(cls.id, "C")
            if cls.id not in focus_ids:
                diagram.body.extend(_declaration(cls, alias, None))
                continue
            body: List[str] = []
            if cls.kind == "enum":
                body = list(cls.enum_members)
            else:
                body += [_field_line(f) for f in cls.fields
                         if _include_member(f.name, "protected" if f.name.startswith("_") else "public", members)]
                for m in cls.methods:
                    if _include_member(m.name, m.visibility, members):
                        if m.name == "__init__" and len(m.params) <= 1:
                            continue
                        is_static = any(d.name == "staticmethod" for d in m.decorators)
                        body.append(_method_line(m, is_static))
            diagram.body.extend(_declaration(cls, alias, body))
        diagram.add("}")

    for line_fmt, a, b in edges:
        diagram.edge(line_fmt.replace(f"{{{a}}}", aliases(a, "C")).replace(f"{{{b}}}", aliases(b, "C")))
    return diagram.text()
