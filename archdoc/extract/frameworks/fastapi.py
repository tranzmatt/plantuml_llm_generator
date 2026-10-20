"""FastAPI: apps and routers as provided HTTP interfaces."""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Tuple

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX, Field
from ..python_ast import (
    FIELD_CONSTRAINT_KWARGS, ExtractContext, ParsedModule, dotted_name, literal_text, str_literal,
)
from ._common import call_target, kwarg, type_ref

_ROUTER_TYPES = {
    f"{EXTERNAL_PREFIX}fastapi.FastAPI",
    f"{EXTERNAL_PREFIX}fastapi.APIRouter",
    f"{EXTERNAL_PREFIX}fastapi.routing.APIRouter",
    f"{EXTERNAL_PREFIX}fastapi.applications.FastAPI",
}
_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head", "trace"}
_PARAM_FUNCS = {"Query": "query", "Path": "path", "Header": "header", "Cookie": "cookie",
                "Body": "body", "Form": "form", "File": "file"}
_SKIP_PARAM_TYPES = {"Request", "Response", "WebSocket", "BackgroundTasks"}
_STATUS_RE = re.compile(r"(\d{3})")

RouterKey = Tuple[str, str]   # (module id, variable name)


def detect_fastapi(ctx: ExtractContext) -> None:
    routers: Dict[RouterKey, ast.Call] = {}
    for pm in ctx.modules.values():
        for name, t in pm.var_types.items():
            if t in _ROUTER_TYPES:
                routers[(pm.id, name)] = pm.var_values[name]
    if not routers:
        return

    mounts = _include_router_prefixes(ctx, routers)
    ops: Dict[RouterKey, List[Operation]] = {k: [] for k in routers}

    for pm in ctx.modules.values():
        for fnode in pm.funcs:
            for deco in fnode.node.decorator_list:
                if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
                    continue
                var = dotted_name(deco.func.value)
                if not var or "." in var:
                    continue
                found = ctx.resolver.resolve_var(pm, var)
                key = (found[0].id, found[1]) if found else None
                if key not in routers:
                    continue
                op = _route_operation(ctx, pm, fnode, deco)
                if op:
                    ops[key].append(op)

    for key, call in routers.items():
        if not ops[key]:
            continue
        mid, var = key
        own_prefix = str_literal(kwarg(call, "prefix")) or ""
        address = mounts.get(key, "") + own_prefix
        iface_id = f"if:http:{mid}:{var}"
        for op in ops[key]:
            op.id = f"{iface_id}:{op.name}"
        ctx.ir.interfaces.append(Interface(
            id=iface_id,
            name=f"{mid}.{var}",
            kind="http",
            direction="provided",
            owner=mid,
            protocol="HTTP/JSON",
            address=address or "/",
            operations=sorted(ops[key], key=lambda o: o.name),
            source_refs=sorted({o.handler for o in ops[key] if o.handler}),
        ))


def _include_router_prefixes(ctx: ExtractContext, routers: Dict[RouterKey, ast.Call]) -> Dict[RouterKey, str]:
    """app.include_router(router, prefix="/v1") -> {router key: "/v1"} (first mount wins)."""
    out: Dict[RouterKey, str] = {}
    for pm in ctx.modules.values():
        for node in ast.walk(pm.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "include_router" and node.args):
                continue
            arg = dotted_name(node.args[0])
            found = ctx.resolver.resolve_var(pm, arg) if arg and "." not in arg else None
            if found:
                key = (found[0].id, found[1])
                if key in routers:
                    out.setdefault(key, str_literal(kwarg(node, "prefix")) or "")
    return out


def _route_operation(ctx: ExtractContext, pm: ParsedModule, fnode, deco: ast.Call) -> Optional[Operation]:
    attr = deco.func.attr
    if attr in _HTTP_METHODS:
        methods = [attr.upper()]
    elif attr == "api_route":
        lit = kwarg(deco, "methods")
        methods = [str_literal(e) for e in lit.elts] if isinstance(lit, (ast.List, ast.Tuple)) else ["GET"]
        methods = [m.upper() for m in methods if m]
    elif attr == "websocket":
        methods = ["WS"]
    else:
        return None
    path = str_literal(deco.args[0]) if deco.args else str_literal(kwarg(deco, "path"))
    if path is None:
        path = "<dynamic>"

    node = fnode.node
    op = Operation(
        id="",
        name=f"{'|'.join(methods)} {path}",
        handler=fnode.fn.id,
        docstring=fnode.fn.docstring or str_literal(kwarg(deco, "description")) or str_literal(kwarg(deco, "summary")),
    )
    status = kwarg(deco, "status_code")
    if status is not None:
        op.meta["status_code"] = _status_text(status)
    tags = kwarg(deco, "tags")
    if isinstance(tags, (ast.List, ast.Tuple)):
        op.meta["tags"] = ", ".join(filter(None, (str_literal(t) for t in tags.elts)))

    path_params = set(re.findall(r"{(\w+)(?::\w+)?}", path))
    args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    defaults = [None] * (len(node.args.posonlyargs + node.args.args) - len(node.args.defaults)) \
        + list(node.args.defaults) + list(node.args.kw_defaults)
    data_type_ids = {d.id for d in ctx.ir.data_types}

    for a, default in zip(args, defaults):
        if a.arg in ("self", "cls"):
            continue
        ann = a.annotation
        ann_name = (dotted_name(ann) or "").rsplit(".", 1)[-1] if ann is not None else ""
        if ann_name in _SKIP_PARAM_TYPES:
            continue
        location, default_node, required = None, default, default is None
        constraints, description = {}, None
        if isinstance(default, ast.Call):
            fn_name = (dotted_name(default.func) or "").rsplit(".", 1)[-1]
            if fn_name in ("Depends", "Security"):
                continue
            if fn_name in _PARAM_FUNCS:
                location = _PARAM_FUNCS[fn_name]
                default_node = default.args[0] if default.args else kwarg(default, "default")
                required = default_node is None or (
                    isinstance(default_node, ast.Constant) and default_node.value is Ellipsis)
                constraints = {kw.arg: literal_text(kw.value) for kw in default.keywords
                               if kw.arg in FIELD_CONSTRAINT_KWARGS}
                description = str_literal(kwarg(default, "description"))
        ref = type_ref(ctx, pm, ann)
        if location in (None, "body") and ref and ref.type_id in data_type_ids and a.arg not in path_params:
            op.request = ref
            continue
        if location is None:
            location = "path" if a.arg in path_params else "query"
        op.params.append(Field(
            name=a.arg,
            annotation=ast.unparse(ann) if ann is not None else None,
            required=True if location == "path" else required,
            default=None if required else literal_text(default_node),
            constraints={"in": location, **constraints},
            docstring=description,
            type_id=ref.type_id if ref else None,
        ))

    response_model = kwarg(deco, "response_model")
    op.response = type_ref(ctx, pm, response_model if response_model is not None else node.returns)
    op.errors = sorted(_http_errors(ctx, pm, node))
    return op


def _status_text(node: ast.AST) -> str:
    """201 / status.HTTP_201_CREATED -> "201"."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    m = _STATUS_RE.search(dotted_name(node) or "")
    return m.group(1) if m else "<expr>"


def _http_errors(ctx: ExtractContext, pm: ParsedModule, func_node: ast.AST) -> set:
    out = set()
    for n in ast.walk(func_node):
        if not (isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)):
            continue
        target = call_target(ctx, pm, n.exc) or ""
        if not target.endswith("HTTPException"):
            continue
        status = kwarg(n.exc, "status_code")
        if status is None and n.exc.args:
            status = n.exc.args[0]
        if status is not None:
            out.add(_status_text(status))
    return out
