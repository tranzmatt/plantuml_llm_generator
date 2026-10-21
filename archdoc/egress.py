"""
Egress: the single function that turns the local IR into what an LLM (or an
MCP client) is allowed to see.

    build_payload(ir, config, tier, aliases) -> Payload

The view is built field by field from an allowlist; nothing is copied from
the IR wholesale, so a new IR field cannot leak by accident. Every string
leaf passes through `_emit`, which applies secret scanning. References use
real ids when the symbol's effective policy is `pseudonymize = off`, and
aliases from the AliasMap otherwise.

Payload JSON shape (keys kept short to save tokens):

    {"repo", "tier", "level",
     "modules": [{"ref", "doc"?, "imports", "main"?,
                  "classes": [{"ref", "kind"?, "bases", "doc"?, "fields", "enum"?, "methods"}],
                  "functions": [{"ref", "sig", "async"?, "doc"?, "decorators"?,
                                 "calls"?, "creates"?, "raises"?}]}],
     "interfaces": [{"ref", "kind", "dir", "owner", "protocol"?, "address"?, "name"?,
                     "ops": [{"ref", "name"?, "params", "outputs"?, "request"?, "response"?,
                              "errors"?, "handler"?, "doc"?, "meta"?}]}]}
"""

from __future__ import annotations

import builtins
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .interfaces import Interface, Operation, TypeRef
from .extract.bodies import locate, outline_text, source_text
from .packing import encode_payload, shorten
from .ir import EXTERNAL_PREFIX, Class, Decorator, Field, Function, Module, RepoIR
from .policy import Config, EgressPolicy, stricter
from .pseudonym import AliasMap
from .secrets_scan import redact
from .trust import CLOUD

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
_QUOTED_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")

# Names that are never repo-specific and stay readable even under pseudonymize=all.
# Operation meta whose values are protocol vocabulary, not anything about the
# code: sent as is in every tier, so a model can tell a publisher from a consumer.
_PROTOCOL_META = {
    "role": {"publish", "consume"},
    "delivery": {"push (basic_consume)", "poll (basic_get)"},
    "ack": {"automatic", "manual"},
    "exchange_type": {"direct", "fanout", "topic", "headers"},
    "persistent": {"yes"}, "mandatory": {"yes"},
    "correlation_id": {"set"}, "reply_to": {"set"},
    "queue": {"exclusive, server-named"},
    "content_type": {"application/json", "text/plain", "application/octet-stream"},
}

_SAFE_TYPE_NAMES = frozenset(dir(builtins)) | {
    "Any", "Optional", "Union", "List", "Dict", "Set", "FrozenSet", "Tuple", "Type", "Callable",
    "Iterable", "Iterator", "Generator", "AsyncIterator", "AsyncGenerator", "Awaitable", "Coroutine",
    "Sequence", "Mapping", "MutableMapping", "Literal", "Annotated", "ClassVar", "Final", "Self",
    "typing", "collections", "abc", "None", "NoneType",
}
_SAFE_DECORATORS = {
    "property", "staticmethod", "classmethod", "dataclass", "dataclasses.dataclass",
    "abstractmethod", "abc.abstractmethod", "cached_property", "functools.cached_property",
    "functools.wraps", "functools.lru_cache", "lru_cache", "cache", "overload", "override",
    "contextmanager", "asynccontextmanager", "contextlib.contextmanager", "typing.overload",
}
_KEEP_DUNDERS = {"__init__", "__call__"}
_STRUCTURAL_CONSTRAINTS = {"in", "kind", "action"}     # values are vocabulary, not literals


class EgressAborted(Exception):
    """A secret was found and the effective policy says abort."""

    def __init__(self, findings: List[Tuple[str, str]]):
        self.findings = findings
        where = "; ".join(f"{kind} in {path}" for path, kind in findings[:10])
        super().__init__(f"egress aborted: {len(findings)} possible secret(s): {where}")


@dataclass
class Payload:
    tier: str
    artifact: str
    data: Dict[str, Any]
    text: str
    redactions: List[Tuple[str, str]] = field(default_factory=list)   # (location, kind)
    # refs exposed in this payload, by role, as the model sees them; planners turn
    # these into schema enums
    allowed: Dict[str, List[str]] = field(default_factory=dict)
    # per-request short id -> full id (archdoc.packing.shorten); ingress maps back
    short: Dict[str, str] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough, tokenizer-free estimate (~4 chars/token for code-ish JSON)."""
    return math.ceil(len(text) / 4)


# ===========================================================================
# Builder
# ===========================================================================

class _ViewBuilder:
    def __init__(self, ir: RepoIR, config: Config, tier: str, aliases: AliasMap):
        self.ir = ir
        self.config = config
        self.tier = tier
        self.aliases = aliases
        self.base = config.policy_for(tier)
        cloud = config.policy_for(CLOUD)
        self.mod_policy: Dict[str, EgressPolicy] = {
            m.id: stricter(self.base, cloud) if config.is_sensitive(m.path) else self.base
            for m in ir.modules
        }
        # True if anything at all is pseudonymized; then every annotation and
        # docstring is rewritten, because a non-aliased module may mention a
        # symbol owned by an aliased (e.g. sensitive) one.
        self.any_aliased = self.base.pseudonymize != "off" or any(
            p.pseudonymize != "off" for p in self.mod_policy.values())
        self.module_ids = {m.id for m in ir.modules}
        # first segment of internal dotted paths (pkg) and module basenames (connection)
        self.module_heads = {m.id.split(".")[0] for m in ir.modules} | {m.id.rsplit(".", 1)[-1] for m in ir.modules}
        self.internal_vars = {v for m in ir.modules for v in m.variables}
        self.class_ids = {c.id for c in ir.iter_classes()}
        self.class_by_name: Dict[str, List[str]] = {}
        for c in ir.iter_classes():
            self.class_by_name.setdefault(c.name, []).append(c.id)
        self.func_by_name: Dict[str, str] = {}
        for f in ir.iter_functions():
            self.func_by_name.setdefault(f.name, f.id)
        self.ext_roots: Dict[str, set] = {
            m.id: {i[len(EXTERNAL_PREFIX):].split(".")[0] for i in m.imports if i.startswith(EXTERNAL_PREFIX)}
            for m in ir.modules
        }
        self.redactions: List[Tuple[str, str]] = []
        self.abort_findings: List[Tuple[str, str]] = []
        # function id -> outline | source for function_view (archdoc.packing); needs `root`
        self.detail: Dict[str, str] = {}
        self.root: Optional[str] = None

    # -- policy helpers -----------------------------------------------------

    def pol(self, mid: Optional[str]) -> EgressPolicy:
        return self.mod_policy.get(mid or "", self.base)

    @staticmethod
    def owner_module(symbol_id: str) -> str:
        return symbol_id.split(":")[0]

    def _emit(self, text: Optional[str], mid: Optional[str], where: str) -> Optional[str]:
        """Final gate for every string leaf: secret scanning."""
        if text is None or not self.config.secrets_scan:
            return text
        clean, findings = redact(text)
        for f in findings:
            if self.pol(mid).on_secret == "abort":
                self.abort_findings.append((where, f.kind))
            else:
                self.redactions.append((where, f.kind))
        return clean

    # -- references ---------------------------------------------------------

    def _aliased(self, mid: Optional[str]) -> bool:
        return self.pol(mid).pseudonymize != "off"

    def module_ref(self, mid: str) -> str:
        return self.aliases.alias("module", mid) if self._aliased(mid) else mid

    def symbol_ref(self, sid: str, ctx_mid: Optional[str] = None) -> str:
        """Ref for a class id, function id, module id, ext:... or raw name."""
        if sid.startswith(EXTERNAL_PREFIX):
            if self.pol(ctx_mid).pseudonymize == "all":
                return self.aliases.alias("literal", sid)
            return sid
        if ":" not in sid:
            if sid in self.module_ids:
                return self.module_ref(sid)
            # unresolved name (e.g. a base like "str" or a module-level variable)
            if self._aliased(ctx_mid) and sid not in _SAFE_TYPE_NAMES:
                return self.aliases.alias("literal", sid)
            return sid
        mid = self.owner_module(sid)
        if not self._aliased(mid):
            return sid
        kind = "class" if sid in self.class_ids else "function"
        display = sid.split(":", 1)[1].rsplit(".", 1)[-1]
        return self.aliases.alias(kind, sid, display)

    def name(self, n: str, mid: Optional[str]) -> str:
        return self.aliases.alias("name", n) if self._aliased(mid) else n

    def literal(self, text: Optional[str], mid: Optional[str], where: str) -> Optional[str]:
        if text is None:
            return None
        mode = self.pol(mid).interface_literals
        if mode == "none":
            return None
        if mode == "alias":
            return self.aliases.alias("literal", text)
        return self._emit(text, mid, where)

    def type_text(self, annotation: Optional[str], mid: Optional[str], type_id: Optional[str] = None,
                  where: str = "") -> Optional[str]:
        if annotation is None:
            return None
        if not self.any_aliased:
            return self._emit(annotation, mid, where)

        aliased_here = self._aliased(mid)

        def sub(m: re.Match) -> str:
            token = m.group()
            if token.startswith("@"):
                return token                        # already an alias (from a quoted literal)
            head, last = token.split(".", 1)[0], token.rsplit(".", 1)[-1]
            candidates = self.class_by_name.get(last)
            if candidates:
                cid = type_id if type_id in candidates else candidates[0]
                return self.symbol_ref(cid, mid)
            if aliased_here and head not in self.ext_roots.get(mid, ()):
                internal_path = "." in token and (head in self.module_heads or last in self.internal_vars)
                if internal_path or token in self.internal_vars or (
                        self.pol(mid).pseudonymize == "all" and last not in _SAFE_TYPE_NAMES):
                    return self.aliases.alias("literal", token)
            return token

        def quoted(m: re.Match) -> str:
            content = m.group(1) if m.group(1) is not None else m.group(2)
            if _IDENT_RE.fullmatch(content) and (content.rsplit(".", 1)[-1] in self.class_by_name
                                                 or content in self.internal_vars):
                return content                      # forward reference: rewritten by sub()
            lit = self.literal(content, mid, where)  # a value, e.g. Literal['read']
            if lit is None:
                return "..."
            return m.group(0) if lit == content else lit   # keep quotes on real literals

        if aliased_here:
            annotation = _QUOTED_RE.sub(quoted, annotation)
        return self._emit(re.sub(r"@?" + _IDENT_RE.pattern, sub, annotation), mid, where)

    def prose(self, text: Optional[str], mid: Optional[str], where: str) -> Optional[str]:
        """Docstrings: trimmed per policy; internal class/function names aliased (best effort)."""
        if not text:
            return None
        mode = self.pol(mid).docstrings
        if mode == "none":
            return None
        if mode == "first_line":
            text = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if self.any_aliased:
            def sub(m: re.Match) -> str:
                w = m.group()
                if w in self.class_by_name:
                    return self.symbol_ref(self.class_by_name[w][0], mid)
                if w in self.func_by_name and ("_" in w or len(w) > 6):
                    return self.symbol_ref(self.func_by_name[w], mid)
                return w
            text = _WORD_RE.sub(sub, text)
        return self._emit(text, mid, where) or None

    def default_text(self, default: Optional[str], mid: Optional[str], where: str) -> Optional[str]:
        """None when the parameter/field has no default or defaults are dropped."""
        if default is None:
            return None
        mode = self.pol(mid).defaults
        if mode == "none":
            return None
        if mode == "presence":
            return "..."
        if self.any_aliased and _IDENT_RE.fullmatch(default) and default not in _SAFE_TYPE_NAMES:
            # a name such as Status.ACTIVE follows the same aliasing as types
            return self.type_text(default, mid, where=where)
        return self._emit(default, mid, where)

    # -- members ------------------------------------------------------------

    def _include_member(self, name: str, visibility: str, mid: str) -> bool:
        if self.pol(mid).members == "all":
            return True
        if name.startswith("__") and name.endswith("__"):
            return name in _KEEP_DUNDERS
        return visibility == "public"

    def field_view(self, f: Field, mid: str, where: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name(f.name, mid)}
        t = self.type_text(f.annotation, mid, f.type_id, where)
        if t:
            out["type"] = t
        if f.required is not None:
            out["req"] = f.required
        d = self.default_text(f.default, mid, where)
        if d is not None and not f.required:
            out["default"] = d
        cons = {}
        for k, v in f.constraints.items():
            if k in _STRUCTURAL_CONSTRAINTS or (v and (_NUMERIC_RE.match(v) or v in ("True", "False"))):
                cons[k] = v
            else:
                lit = self.literal(v, mid, where)
                if lit is not None:
                    cons[k] = lit
        if cons:
            out["constraints"] = cons
        doc = self.prose(f.docstring, mid, where)
        if doc:
            out["doc"] = doc
        return out

    def signature(self, fn: Function, mid: str, is_method: bool) -> str:
        params = list(fn.params)
        is_static = any(d.name in ("staticmethod",) for d in fn.decorators)
        if is_method and not is_static and params and params[0].kind in ("positional", "positional_only"):
            params = params[1:]
        parts: List[str] = []
        star_done = False
        for i, p in enumerate(params):
            if p.kind == "keyword_only" and not star_done:
                parts.append("*")
                star_done = True
            prefix = "*" if p.kind == "var_positional" else "**" if p.kind == "var_keyword" else ""
            if p.kind == "var_positional":
                star_done = True
            s = prefix + self.name(p.name, mid)
            t = self.type_text(p.annotation, mid, where=fn.id)
            if t:
                s += f": {t}"
            d = self.default_text(p.default, mid, fn.id)
            if d is not None:
                s += f" = {d}"
            parts.append(s)
            if p.kind == "positional_only" and (i + 1 == len(params) or params[i + 1].kind != "positional_only"):
                parts.append("/")
        sig = f"({', '.join(parts)})"
        ret = self.type_text(fn.returns, mid, where=fn.id)
        return f"{sig} -> {ret}" if ret else sig

    def decorator_view(self, d: Decorator, mid: str) -> Optional[str]:
        head = d.name.split(".")[0]
        keep_name = (not self._aliased(mid) or d.name in _SAFE_DECORATORS
                     or head in self.ext_roots.get(mid, ()))
        name = d.name if keep_name else self.aliases.alias("literal", d.name)
        if self.pol(mid).interface_literals == "real" and (d.args or d.kwargs):
            args = [a for a in d.args] + [f"{k}={v}" for k, v in d.kwargs.items()]
            # args can name classes (response_model=Item): same rewriting as annotations
            return self.type_text(f"{name}({', '.join(args)})", mid, where=f"{mid}:decorator")
        return name

    def function_view(self, fn: Function, mid: str, is_method: bool) -> Dict[str, Any]:
        pol = self.pol(mid)
        out: Dict[str, Any] = {"ref": self.symbol_ref(fn.id, mid), "sig": self.signature(fn, mid, is_method)}
        if fn.is_async:
            out["async"] = True
        doc = self.prose(fn.docstring, mid, fn.id)
        if doc:
            out["doc"] = doc
        decos = [v for v in (self.decorator_view(d, mid) for d in fn.decorators) if v]
        if decos:
            out["decorators"] = decos
        if pol.level_at_least("callgraph"):
            for key, ids in (("calls", fn.calls), ("creates", fn.creates), ("raises", fn.raises)):
                if ids:
                    out[key] = [self.symbol_ref(i, mid) for i in ids]
        level = self.detail.get(fn.id)
        body = self.body_view(fn.id, mid, level, self.root) if level else None
        if body is not None:
            out[level] = body
        return out

    def body_view(self, fid: str, mid: str, level: str, root: Optional[str]) -> Optional[str]:
        """A function's outline or source, if this request may carry it: a local
        (unaliased) module whose policy level allows it. Secret-scanned like any leaf."""
        need = {"outline": "control_flow", "source": "source"}.get(level)
        if need is None or root is None or self._aliased(mid) or not self.pol(mid).level_at_least(need):
            return None
        found = locate(root, self.ir, fid)
        if found is None:
            return None
        node, lines = found
        text = outline_text(node) if level == "outline" else source_text(node, lines)
        return self._emit(text, mid, fid)

    def class_view(self, c: Class, mid: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ref": self.symbol_ref(c.id, mid)}
        if c.kind != "class":
            out["kind"] = c.kind
        bases = [self.symbol_ref(b, mid) for b in c.bases]
        if bases:
            out["bases"] = bases
        doc = self.prose(c.docstring, mid, c.id)
        if doc:
            out["doc"] = doc
        fields_ = [self.field_view(f, mid, c.id) for f in c.fields
                   if self._include_member(f.name, "protected" if f.name.startswith("_") else "public", mid)]
        if fields_:
            out["fields"] = fields_
        if c.enum_members:
            out["enum"] = [self.name(m, mid) for m in c.enum_members]
        methods = [self.function_view(m, mid, True) for m in c.methods
                   if self._include_member(m.name, m.visibility, mid)]
        if methods:
            out["methods"] = methods
        return out

    def module_view(self, m: Module) -> Dict[str, Any]:
        mid = m.id
        out: Dict[str, Any] = {"ref": self.module_ref(mid)}
        doc = self.prose(m.docstring, mid, mid)
        if doc:
            out["doc"] = doc
        imports = [self.symbol_ref(i, mid) for i in m.imports]
        if imports:
            out["imports"] = imports
        if m.has_main:
            out["main"] = True
        classes = [self.class_view(c, mid) for c in m.classes
                   if self._include_member(c.name, "protected" if c.name.startswith("_") else "public", mid)]
        if classes:
            out["classes"] = classes
        funcs = [self.function_view(f, mid, False) for f in m.functions
                 if self._include_member(f.name, f.visibility, mid)]
        if funcs:
            out["functions"] = funcs
        return out

    # -- interfaces ---------------------------------------------------------

    def type_ref_view(self, t: Optional[TypeRef], mid: str, where: str) -> Optional[str]:
        if t is None:
            return None
        return self.type_text(t.display, mid, t.type_id, where)

    def op_view(self, iface: Interface, op: Operation, mid: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ref": self._ref_for("op", op.id, op.name, mid)}
        if iface.kind == "http" and " " in op.name:
            method, path = op.name.split(" ", 1)
            lit = self.literal(path, mid, op.id)
            out["name"] = f"{method} {lit}" if lit else method
        else:
            lit = self.literal(op.name, mid, op.id)
            if lit:
                out["name"] = lit
        params = [self.field_view(p, mid, op.id) for p in op.params]
        if params:
            out["params"] = params
        outputs = [self.field_view(p, mid, op.id) for p in op.outputs]
        if outputs:
            out["outputs"] = outputs
        for key, t in (("request", op.request), ("response", op.response)):
            v = self.type_ref_view(t, mid, op.id)
            if v:
                out[key] = v
        errors = [e if e.isdigit() else self.symbol_ref(e, mid) for e in op.errors]
        if errors:
            out["errors"] = errors
        if op.handler:
            out["handler"] = self.symbol_ref(op.handler, mid)
        doc = self.prose(op.docstring, mid, op.id)
        if doc:
            out["doc"] = doc
        meta = {}
        for k, v in op.meta.items():
            if k == "status_code" or (k in _PROTOCOL_META and v in _PROTOCOL_META[k]) or (
                    k == "component" and v.isidentifier()):     # a UI library class, e.g. Button
                meta[k] = v
            elif k in ("used_in", "required_in"):
                meta[k] = [self.module_ref(x.strip()) for x in v.split(",") if x.strip()]
            elif k == "default":
                d = self.default_text(v, mid, op.id)
                if d is not None:
                    meta[k] = d
            else:
                lit = self.literal(v, mid, op.id)
                if lit is not None:
                    meta[k] = lit
        if meta:
            out["meta"] = meta
        return out

    def _ref_for(self, kind: str, key: str, display: str, mid: str) -> str:
        return self.aliases.alias(kind, key, display) if self._aliased(mid) else key

    def interface_view(self, iface: Interface) -> Dict[str, Any]:
        mid = iface.owner if iface.owner in self.module_ids else None
        out: Dict[str, Any] = {
            "ref": self._ref_for("iface", iface.id, iface.name, mid),
            "kind": iface.kind,
            "dir": iface.direction,
            "owner": self.module_ref(mid) if mid else self.repo_name(),
        }
        if iface.protocol:
            out["protocol"] = iface.protocol
        addr = self.literal(iface.address, mid, iface.id)
        if addr:
            out["address"] = addr
        out["ops"] = [self.op_view(iface, op, mid) for op in iface.operations]
        return out

    def repo_name(self) -> str:
        if self.base.pseudonymize == "off":
            return self.ir.repo
        return self.aliases.alias("literal", self.ir.repo)

    # -- top level ----------------------------------------------------------

    def build(self, modules: Optional[Iterable[str]]) -> Dict[str, Any]:
        wanted = None if modules is None else set(modules)
        mods = [m for m in self.ir.modules if wanted is None or m.id in wanted]
        ifaces = [i for i in self.ir.interfaces if wanted is None or i.owner in wanted or i.kind == "env"]
        view: Dict[str, Any] = {"repo": self.repo_name(), "tier": self.tier, "level": self.base.level}
        if self.base.level == "none":
            view["modules"], view["interfaces"] = [], []
            return view
        view["modules"] = [self.module_view(m) for m in mods if not m.parse_error]
        view["interfaces"] = [self.interface_view(i) for i in ifaces]
        return view


def _finish(builder: _ViewBuilder, data: Dict[str, Any], tier: str, artifact: str,
            allowed: Optional[Dict[str, List[str]]] = None) -> Payload:
    if builder.abort_findings:
        raise EgressAborted(builder.abort_findings)
    short: Dict[str, str] = {}
    sent = data
    # planner requests (with `allowed` refs) use short ids; the whole-repo preview keeps full ids
    if builder.config.payload_encoding == "text" and allowed:
        sent, allowed, short = shorten(data, allowed)
    text = encode_payload(sent, builder.config.payload_encoding)
    return Payload(tier=tier, artifact=artifact, data=data, text=text,
                   redactions=builder.redactions, allowed=allowed or {}, short=short)


def participant_of(function_id: str) -> str:
    """Sequence participant for a function: its class for methods, else its module."""
    mid, _, qual = function_id.partition(":")
    return f"{mid}:{qual.rsplit('.', 1)[0]}" if "." in qual else mid


def build_sequence_payload(
    ir: RepoIR, config: Config, tier: str, aliases: AliasMap,
    entry_handler: str, function_ids: List[str], via: Optional[Tuple[Interface, Operation]] = None,
    detail: Optional[Dict[str, str]] = None, root: Optional[str] = None,
    repo_map: Optional[List[Dict[str, Any]]] = None,
) -> Payload:
    """Call-graph slice for one entry point. `function_ids` must start with the handler.
    `detail` maps function ids to outline/source (default stub); see archdoc.packing."""
    b = _ViewBuilder(ir, config, tier, aliases)
    if not b.base.level_at_least("callgraph"):
        raise EgressAborted([("policy", f"level={b.base.level} does not allow call graphs")])
    in_slice = set(function_ids)
    participants: List[str] = []
    functions = []
    for fid in function_ids:
        fn = ir.find_function(fid)
        if fn is None:
            continue
        owner = participant_of(fid)
        if owner not in participants:
            participants.append(owner)
        mid = b.owner_module(fid)
        view = b.function_view(fn, mid, is_method=owner != mid)
        view.pop("decorators", None)
        view["owner"] = b.symbol_ref(owner, mid)
        view["calls"] = [b.symbol_ref(c, mid) for c in fn.calls
                         if c in in_slice or c.startswith(EXTERNAL_PREFIX)]
        if not view["calls"]:
            view.pop("calls")
        level = (detail or {}).get(fid, "stub")
        body = b.body_view(fid, mid, level, root)
        if body is not None:
            view[level] = body
        functions.append(view)
    part_views = []
    for pid in participants:
        mid = b.owner_module(pid)
        pv: Dict[str, Any] = {"ref": b.symbol_ref(pid, mid), "type": "class" if pid in b.class_ids else "module"}
        cls = ir.find_class(pid)
        if cls is not None and cls.kind != "class":
            pv["kind"] = cls.kind
        doc = b.prose(cls.docstring if cls else (ir.module(pid).docstring if ir.module(pid) else None), mid, pid)
        if doc:
            pv["doc"] = doc
        part_views.append(pv)
    entry: Dict[str, Any] = {"handler": b.symbol_ref(entry_handler, b.owner_module(entry_handler))}
    if via is not None:
        iface, op = via
        mid = iface.owner if iface.owner in b.module_ids else None
        entry["interface"] = iface.kind
        opv = b.op_view(iface, op, mid)
        if "name" in opv:
            entry["operation"] = opv["name"]
    data = {"entry": entry, "participants": part_views, "functions": functions}
    if repo_map:
        data = {"system_map": repo_map, **data}
    return _finish(b, data, tier, f"sequence:{entry_handler}", {
        "participants": [pv["ref"] for pv in part_views],
        "functions": [f["ref"] for f in functions],
    })


def build_interface_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap,
                            iface: Interface, detail: Optional[Dict[str, str]] = None,
                            root: Optional[str] = None,
                            repo_map: Optional[List[Dict[str, Any]]] = None) -> Payload:
    """One interface, its handlers' signatures and the data types it exchanges.
    `detail` maps handler ids to outline/source (default stub)."""
    b = _ViewBuilder(ir, config, tier, aliases)
    iv = b.interface_view(iface)
    handlers, seen = [], set()
    for op in iface.operations:
        fn = ir.find_function(op.handler) if op.handler else None
        if fn is not None and fn.id not in seen:
            seen.add(fn.id)
            mid = b.owner_module(fn.id)
            hv = b.function_view(fn, mid, is_method=participant_of(fn.id) != mid)
            for k in ("decorators", "calls", "creates"):
                hv.pop(k, None)
            level = (detail or {}).get(fn.id, "stub")
            body = b.body_view(fn.id, mid, level, root)
            if body is not None:
                hv[level] = body
            handlers.append(hv)
    type_ids = []
    for op in iface.operations:
        for t in (op.request, op.response):
            if t and t.type_id and t.type_id not in type_ids:
                type_ids.append(t.type_id)
    types = []
    for tid in type_ids:
        cls = ir.find_class(tid)
        if cls is not None:
            cv = b.class_view(cls, b.owner_module(tid))
            cv.pop("methods", None)
            types.append(cv)
    data: Dict[str, Any] = {"system_map": repo_map} if repo_map else {}
    data["interface"] = iv
    if handlers:
        data["handlers"] = handlers
    if types:
        data["types"] = types
    params = sorted({p["name"] for o in iv["ops"] for p in o.get("params", [])})
    return _finish(b, data, tier, f"icd:{iface.id}", {
        "operations": [o["ref"] for o in iv["ops"]],
        "params": params,
    })


def build_repo_map(ir: RepoIR, config: Config, tier: str, aliases: AliasMap,
                   cap_chars: int) -> List[Dict[str, Any]]:
    """Ranked outline of the whole system, the same in every request of a run
    (DESIGN.md 12.4): modules by how many others import them and how many
    interfaces they own, each with its docstring line (as policy allows) and
    the kinds of interface it provides or uses. Cut at `cap_chars` encoded."""
    b = _ViewBuilder(ir, config, tier, aliases)
    if not b.base.level_at_least("interfaces"):
        return []
    importers: Dict[str, int] = {}
    for m in ir.modules:
        for imp in m.imports:
            importers[imp] = importers.get(imp, 0) + 1
    owned: Dict[str, List[Interface]] = {}
    for iface in ir.interfaces:
        owned.setdefault(iface.owner, []).append(iface)
    ranked = sorted(ir.modules, key=lambda m: (-(importers.get(m.id, 0) + 2 * len(owned.get(m.id, []))), m.id))
    out: List[Dict[str, Any]] = []
    for m in ranked:
        if m.parse_error:
            continue
        entry: Dict[str, Any] = {"ref": b.module_ref(m.id)}
        doc = b.prose((m.docstring or "").split("\n", 1)[0] or None, m.id, m.id)
        if doc:
            entry["doc"] = doc
        kinds = sorted({f"{i.kind} {i.direction}" for i in owned.get(m.id, [])})
        if kinds:
            entry["interfaces"] = kinds
        if len(encode_payload({"system_map": out + [entry]}, "text")) > cap_chars:
            break
        out.append(entry)
    if b.abort_findings:
        return []
    return out


def is_stdlib(ext_id: str) -> bool:
    root = ext_id[len(EXTERNAL_PREFIX):].split(".")[0] if ext_id.startswith(EXTERNAL_PREFIX) else ext_id
    return root in sys.stdlib_module_names or root == "__future__"


def _builder(ir: RepoIR, config: Config, tier: str, aliases: AliasMap,
             detail: Optional[Dict[str, str]], root: Optional[str]) -> _ViewBuilder:
    b = _ViewBuilder(ir, config, tier, aliases)
    b.detail, b.root = dict(detail or {}), root
    return b


def _with_map(data: Dict[str, Any], repo_map: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {"system_map": repo_map, **data} if repo_map else data


def _caught(ir: RepoIR, exc: str, caught: Iterable[str]) -> bool:
    """Whether an except clause for one of `caught` stops `exc`: the same type,
    a builtin base class of a builtin, or a base class in the repo's hierarchy."""
    caught = set(caught)
    if not caught:
        return False
    if "*" in caught or exc in caught:
        return True
    bexc = getattr(builtins, exc, None)
    if isinstance(bexc, type) and any(isinstance(getattr(builtins, c, None), type)
                                      and issubclass(bexc, getattr(builtins, c)) for c in caught):
        return True
    seen, todo = set(), [exc]
    while todo:
        cls = ir.find_class(todo.pop())
        if cls is None or cls.id in seen:
            continue
        seen.add(cls.id)
        if any(b in caught for b in cls.bases):
            return True
        todo += cls.bases
    return False


def build_function_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, function_id: str,
                           detail: Optional[Dict[str, str]] = None, root: Optional[str] = None,
                           repo_map: Optional[List[Dict[str, Any]]] = None) -> Payload:
    """One function to specify (first in "functions"), the functions it calls
    (stubs unless `detail` upgrades them) and, for a method, its class's fields."""
    b = _builder(ir, config, tier, aliases, detail, root)
    if not b.base.level_at_least("callgraph"):
        raise EgressAborted([("policy", f"level={b.base.level} does not allow call graphs")])
    fn = ir.find_function(function_id)
    if fn is None:
        raise KeyError(function_id)
    mid = b.owner_module(function_id)
    owner = participant_of(function_id)
    target = b.function_view(fn, mid, is_method=owner != mid)
    target.pop("decorators", None)
    callees = [c for c in dict.fromkeys(fn.calls) if not c.startswith(EXTERNAL_PREFIX) and ir.find_function(c)]
    views = [target]
    raises = list(fn.raises)
    for cid in callees:
        callee = ir.find_function(cid)
        cmid = b.owner_module(cid)
        cv = b.function_view(callee, cmid, is_method=participant_of(cid) != cmid)
        for k in ("decorators", "calls", "creates"):
            cv.pop(k, None)
        views.append(cv)
        raises += [r for r in callee.raises if r not in raises and not _caught(ir, r, fn.caught.get(cid, ()))]
    data: Dict[str, Any] = {"target": target["ref"]}
    cls = ir.find_class(owner) if owner != mid else None
    if cls is not None:
        cv = b.class_view(cls, mid)
        cv.pop("methods", None)
        data["owner_class"] = cv
    data["functions"] = views
    params = [p.name for p in fn.params if p.name not in ("self", "cls")]
    return _finish(b, _with_map(data, repo_map), tier, f"behavior:{function_id}", {
        "function": [target["ref"]],
        "params": [b.name(p, mid) for p in params],
        "calls": [b.symbol_ref(c, mid) for c in dict.fromkeys(fn.calls)],
        "raises": [b.symbol_ref(r, mid) for r in dict.fromkeys(raises)],
    })


def module_collaborators(ir: RepoIR, mid: str) -> List[str]:
    """Internal modules `mid` imports or calls into: the only valid collaborators of its summary."""
    m = ir.module(mid)
    ids = {m2.id for m2 in ir.modules}
    out = [i for i in m.imports if i in ids and i != mid]
    for fn in [*m.functions, *(f for c in m.classes for f in c.methods)]:
        for c in fn.calls:
            cm = c.split(":")[0]
            if cm in ids and cm != mid and cm not in out:
                out.append(cm)
    return out


def module_externals(ir: RepoIR, mid: str) -> List[str]:
    """Third-party libraries `mid` imports (the standard library is not an external system)."""
    m = ir.module(mid)
    return [i for i in m.imports if i.startswith(EXTERNAL_PREFIX) and not is_stdlib(i)]


def build_module_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, mid: str,
                         detail: Optional[Dict[str, str]] = None, root: Optional[str] = None,
                         repo_map: Optional[List[Dict[str, Any]]] = None) -> Payload:
    """One module for its summary: its members (stubs unless `detail` upgrades them)."""
    b = _builder(ir, config, tier, aliases, detail, root)
    m = ir.module(mid)
    view = b.module_view(m)
    members = [c["ref"] for c in view.get("classes", [])] + [f["ref"] for f in view.get("functions", [])]
    return _finish(b, _with_map({"module": view}, repo_map), tier, f"summary:{mid}", {
        "members": members,
        "collaborators": [b.module_ref(x) for x in module_collaborators(ir, mid)],
        "external": [b.symbol_ref(x, mid) for x in module_externals(ir, mid)],
    })


def build_group_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, group: str,
                        parts: List[Tuple[str, str]], collaborators: List[str], external: List[str],
                        repo_map: Optional[List[Dict[str, Any]]] = None) -> Payload:
    """A package (parts = its modules) or the system (parts = its packages), from
    the parts' summaries. The texts are the model's own earlier answers, as the
    model wrote them (aliased for an aliased tier); they are scanned like any leaf."""
    b = _ViewBuilder(ir, config, tier, aliases)
    rows = []
    for ref_id, text in parts:
        rows.append({"ref": b.module_ref(ref_id), "summary": b._emit(text, ref_id, ref_id) or ""})
    key = "packages" if group == "" else "modules"
    data: Dict[str, Any] = {"package": b.module_ref(group)} if group else {"system": b.repo_name()}
    data[key] = rows
    return _finish(b, _with_map(data, repo_map), tier, f"summary:{group or '<system>'}", {
        "parts": [r["ref"] for r in rows],
        "collaborators": [b.module_ref(c) for c in collaborators],
        "external": [b.symbol_ref(e) for e in external],
    })


def build_tests_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, mid: str,
                        test_ids: List[str], exercised: List[str], detail: Optional[Dict[str, str]] = None,
                        root: Optional[str] = None) -> Payload:
    """The tests of one test module (sources as `detail` allows) and the code they exercise (stubs)."""
    b = _builder(ir, config, tier, aliases, detail, root)
    tests, seen = [], []
    for tid in test_ids:
        fn = ir.find_function(tid)
        if fn is None:
            continue
        tmid = b.owner_module(tid)
        tv = b.function_view(fn, tmid, is_method=participant_of(tid) != tmid)
        tests.append(tv)
        seen.append(tv["ref"])
    code = []
    for fid in exercised:
        fn = ir.find_function(fid)
        if fn is None:
            continue
        cmid = b.owner_module(fid)
        cv = b.function_view(fn, cmid, is_method=participant_of(fid) != cmid)
        for k in ("decorators", "calls", "creates", "outline", "source"):
            cv.pop(k, None)
        code.append(cv)
    data: Dict[str, Any] = {"module": b.module_ref(mid), "tests": tests}
    if code:
        data["exercised"] = code
    return _finish(b, data, tier, f"tests:{mid}", {"tests": seen})


def _resolve_written(fn: Function, written: str) -> Optional[str]:
    """A call as written (`self.repo.get`) to the call id the extractor resolved, if any."""
    tail = written.rsplit(".", 1)[-1]
    for c in fn.calls:
        if re.split(r"[.:]", c)[-1] == tail:
            return c
    return None


def build_activity_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, act,
                           detail: Optional[Dict[str, str]] = None, root: Optional[str] = None) -> Payload:
    """One function's activity nodes (archdoc.activity) for labelling. Needs the
    module's policy to reach `control_flow`: the nodes are its control flow.
    Locally each node carries its deterministic text (the condition or calls as
    written); aliased, only its kind and the resolved calls."""
    b = _builder(ir, config, tier, aliases, detail, root)
    fid = act.function
    mid = b.owner_module(fid)
    if not b.pol(mid).level_at_least("control_flow"):
        raise EgressAborted([("policy", f"level={b.pol(mid).level} does not allow control flow")])
    fn = ir.find_function(fid)
    fv = b.function_view(fn, mid, is_method=participant_of(fid) != mid)
    fv.pop("decorators", None)
    nodes = []
    for item in act.labelable():
        nv: Dict[str, Any] = {"id": item.id, "kind": getattr(item, "kind", "")}
        refs = [r for r in (_resolve_written(fn, w) for w in getattr(item, "calls", [])) if r]
        if refs:
            nv["calls"] = [b.symbol_ref(r, mid) for r in dict.fromkeys(refs)]
        if not b._aliased(mid) and item.text:
            nv["at"] = b._emit(item.text, mid, fid)
        nodes.append(nv)
    data = {"functions": [fv], "nodes": nodes}
    return _finish(b, data, tier, f"activity:{fid}", {"nodes": [n["id"] for n in nodes]})


def build_usecase_payload(ir: RepoIR, config: Config, tier: str, aliases: AliasMap, model,
                          repo_map: Optional[List[Dict[str, Any]]] = None) -> Payload:
    """The use cases (operations as the ICD shows them) and actors of archdoc.usecases, for naming."""
    b = _ViewBuilder(ir, config, tier, aliases)
    ops = {o.id: (i, o) for i in ir.interfaces for o in i.operations}
    cases, actors = [], []
    for c in model.cases:
        iface, op = ops[c.id]
        mid = iface.owner if iface.owner in b.module_ids else None
        ov = b.op_view(iface, op, mid)
        cv: Dict[str, Any] = {"ref": ov["ref"], "interface": iface.kind}
        for k in ("name", "doc", "params"):
            if k in ov:
                cv[k] = ov[k]
        if op.handler:
            fn = ir.find_function(op.handler)
            doc = b.prose(fn.docstring, b.owner_module(op.handler), op.handler) if fn else None
            if doc:
                cv["handler_doc"] = doc
        cv["actor"] = c.actor
        if c.reaches:
            cv["reaches"] = c.reaches
        cases.append(cv)
    shown = {}
    for a in model.actors:
        name = a.name if a.primary or not a.source.startswith("ext:") else b.literal(a.name, None, a.key)
        shown[a.key] = name
        actors.append({"ref": a.key, "kind": "primary" if a.primary else "external system",
                       **({"name": name} if name else {})})
    data = _with_map({"system": b.repo_name(), "actors": actors, "use_cases": cases}, repo_map)
    return _finish(b, data, tier, "usecase", {"cases": [c["ref"] for c in cases],
                                              "actors": [a["ref"] for a in actors]})


def build_payload(
    ir: RepoIR,
    config: Config,
    tier: str,
    aliases: AliasMap,
    artifact: str = "repo",
    modules: Optional[Iterable[str]] = None,
) -> Payload:
    """Sanitized, optionally pseudonymized view of `ir` for `tier`.

    Raises EgressAborted if a secret is found where policy says abort.
    """
    builder = _ViewBuilder(ir, config, tier, aliases)
    return _finish(builder, builder.build(modules), tier, artifact)


# ===========================================================================
# Audit log
# ===========================================================================

class EgressLog:
    """Append-only JSONL record of everything sent (<out>/.archdoc/egress.jsonl)."""

    def __init__(self, path: str, include_payload: bool = True):
        self.path = path
        self.include_payload = include_payload

    def record(self, payload: Payload, backend: str, url: Optional[str], **extra: Any) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "backend": backend,
            "url": url,
            "tier": payload.tier,
            "artifact": payload.artifact,
            "sha256": payload.sha256,
            "tokens_est": payload.token_estimate,
            "redactions": len(payload.redactions),
            **extra,
        }
        if self.include_payload:
            entry["payload"] = payload.data
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
