"""
RabbitMQ through pika: what each module publishes and consumes.

A module that consumes gets a provided `message` interface (it offers a
handler for those messages); one that publishes gets a required one. Each
operation names its destination as written in the code: a queue (published
through the default exchange, or consumed directly) or an exchange (with its
type when some module declares it). Message bodies built as
`json.dumps({...})` literals give the published field names; a consumer's
`json.loads(body)` subscripts give the field names it reads. Values are never
recorded, only names and string literals that address the broker.

pika calls go through channel objects whose type the resolver cannot follow,
so they are matched by method name, only in modules that import pika.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX, Field
from ..python_ast import ExtractContext, ParsedModule, dotted_name, str_literal
from ._common import kwarg, scope_nodes

_PIKA = f"{EXTERNAL_PREFIX}pika"
_METHODS = {"basic_publish", "basic_consume", "basic_get", "queue_bind", "exchange_declare", "queue_declare"}
from ...messaging import RUNTIME


def _arg(call: ast.Call, pos: int, name: str) -> Optional[ast.expr]:
    found = kwarg(call, name)
    if found is not None:
        return found
    return call.args[pos] if len(call.args) > pos else None


def _const(node: Optional[ast.expr]) -> Optional[str]:
    """String literal as written, RUNTIME for any other expression, None if absent."""
    if node is None:
        return None
    s = str_literal(node)
    return s if s is not None else RUNTIME


@dataclass
class _Endpoint:
    role: str                          # publish | consume
    target: str                        # queue | exchange
    name: str
    routing_key: Optional[str] = None
    handler: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)
    fields: List[str] = field(default_factory=list)
    line: Optional[int] = None

    @property
    def key(self) -> Tuple[str, str, str, Optional[str]]:
        return self.role, self.target, self.name, self.routing_key


def _uses_pika(pm: ParsedModule) -> bool:
    return any(i == _PIKA or i.startswith(_PIKA + ".") for i in pm.module.imports)


def _pika_calls(body: List[ast.stmt]) -> List[ast.Call]:
    calls = [n for n in scope_nodes(body)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _METHODS]
    return sorted(calls, key=lambda n: (n.lineno, n.col_offset))


def _dict_keys(node: Optional[ast.expr], assigned: Dict[str, ast.expr]) -> List[str]:
    """Keys of json.dumps({...}), directly or through a local variable holding the dict."""
    if not (isinstance(node, ast.Call) and dotted_name(node.func) in ("json.dumps", "dumps") and node.args):
        return []
    obj = node.args[0]
    if isinstance(obj, ast.Name):
        obj = assigned.get(obj.id, obj)
    if not isinstance(obj, ast.Dict):
        return []
    return [k for k in (str_literal(k) for k in obj.keys if k is not None) if k]


def _read_keys(fn_node: ast.AST) -> List[str]:
    """Field names a consumer reads from json.loads(...) results: x["k"] and x.get("k")."""
    loaded: Set[str] = set()
    for n in ast.walk(fn_node):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call) and dotted_name(n.value.func) in ("json.loads", "loads")):
            loaded.add(n.targets[0].id)
    keys: List[str] = []
    for n in ast.walk(fn_node):
        k = None
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in loaded:
            k = str_literal(n.slice)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
              and isinstance(n.func.value, ast.Name) and n.func.value.id in loaded and n.args):
            k = str_literal(n.args[0])
        if k and k not in keys:
            keys.append(k)
    return keys


def _publish_meta(call: ast.Call) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    props = kwarg(call, "properties")
    if isinstance(props, ast.Call):
        ct = str_literal(kwarg(props, "content_type"))
        if ct:
            meta["content_type"] = ct
        mode = kwarg(props, "delivery_mode")
        if isinstance(mode, ast.Constant) and mode.value == 2:
            meta["persistent"] = "yes"
        if kwarg(props, "correlation_id") is not None:
            meta["correlation_id"] = "set"
        if kwarg(props, "reply_to") is not None:
            meta["reply_to"] = "set"
    if isinstance(kwarg(call, "mandatory"), ast.Constant) and kwarg(call, "mandatory").value is True:
        meta["mandatory"] = "yes"
    return meta


def _scope_endpoints(ctx: ExtractContext, pm: ParsedModule, handler: Optional[str],
                     body: List[ast.stmt], fn_nodes: Dict[str, ast.AST]) -> List[_Endpoint]:
    assigned = {n.targets[0].id: n.value for n in scope_nodes(body)
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    binds: Dict[str, List[Tuple[str, Optional[str]]]] = {}      # queue expression -> [(exchange, key)]
    out: List[_Endpoint] = []
    for call in _pika_calls(body):
        m = call.func.attr
        start = len(out)
        if m == "queue_bind":
            queue, exchange = _arg(call, 0, "queue"), _const(_arg(call, 1, "exchange"))
            if queue is not None and exchange:
                binds.setdefault(ast.unparse(queue), []).append((exchange, _const(_arg(call, 2, "routing_key"))))
        elif m == "basic_publish":
            exchange = _const(_arg(call, 0, "exchange")) or ""
            key = _const(_arg(call, 1, "routing_key"))
            fields = _dict_keys(_arg(call, 2, "body"), assigned)
            if exchange == "":
                out.append(_Endpoint("publish", "queue", key or RUNTIME, None, handler, _publish_meta(call), fields))
            else:
                out.append(_Endpoint("publish", "exchange", exchange, key, handler, _publish_meta(call), fields))
        elif m in ("basic_consume", "basic_get"):
            queue = _arg(call, 0, "queue")
            cb_node = kwarg(call, "on_message_callback") if m == "basic_consume" else None
            if m == "basic_consume" and cb_node is None and len(call.args) > 1:
                cb_node = call.args[1]
            cb = None
            if cb_node is not None and dotted_name(cb_node):
                cb = ctx.resolver.resolve_name(pm, dotted_name(cb_node))
            reader = cb if cb in fn_nodes else handler
            fields = _read_keys(fn_nodes[reader]) if reader in fn_nodes else []
            meta = {"delivery": "push (basic_consume)" if m == "basic_consume" else "poll (basic_get)"}
            ack = kwarg(call, "auto_ack")
            if isinstance(ack, ast.Constant):
                meta["ack"] = "automatic" if ack.value else "manual"
            name = str_literal(queue) if queue is not None else None
            bound = binds.get(ast.unparse(queue), []) if queue is not None else []
            if name:
                out.append(_Endpoint("consume", "queue", name, None, reader, meta, fields))
            elif bound:
                for exchange, key in bound:
                    out.append(_Endpoint("consume", "exchange", exchange, key, reader,
                                         {**meta, "queue": "exclusive, server-named"}, fields))
            else:
                out.append(_Endpoint("consume", "queue", RUNTIME, None, reader, meta, fields))
        for ep in out[start:]:
            ep.line = call.lineno
    return out


def _exchange_types(ctx: ExtractContext) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for pm in ctx.modules.values():
        if not _uses_pika(pm):
            continue
        for n in ast.walk(pm.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "exchange_declare":
                name, kind = str_literal(_arg(n, 0, "exchange")), str_literal(kwarg(n, "exchange_type"))
                if name:
                    types.setdefault(name, kind or "direct")
    return types


def _operation(iface_id: str, ep: _Endpoint, exchange_types: Dict[str, str], by_handler: bool) -> Operation:
    where = f"{ep.target} {ep.name}"
    meta = {"role": ep.role, ep.target: ep.name, **ep.meta}
    if ep.target == "exchange" and ep.name in exchange_types:
        meta["exchange_type"] = exchange_types[ep.name]
    if ep.routing_key is not None and ep.routing_key != "":
        meta["routing_key"] = ep.routing_key
    key = f" [{ep.routing_key}]" if ep.routing_key not in (None, "", RUNTIME) else ""
    handler_name = (ep.handler or "<module>").rsplit(":", 1)[-1]
    suffix = f" (in {handler_name})" if by_handler else ""
    return Operation(
        id=f"{iface_id}:{ep.role}:{ep.target}:{ep.name}{key}" + (f":{handler_name}" if by_handler else ""),
        name=f"{ep.role} {where}{key}{suffix}",
        params=[Field(name=f) for f in ep.fields],
        handler=ep.handler,
        meta=meta,
        line=ep.line,
    )


def detect_pika(ctx: ExtractContext) -> None:
    exchange_types = _exchange_types(ctx)
    for pm in ctx.modules.values():
        if not _uses_pika(pm):
            continue
        fn_nodes = {f.fn.id: f.node for f in pm.funcs}
        scopes = [(None, pm.tree.body)] + [(f.fn.id, f.node.body) for f in pm.funcs]
        # one operation per destination and handler: the same queue used twice in a
        # function is one operation (fields unioned); used by two functions, two
        merged: Dict[tuple, _Endpoint] = {}
        for handler, body in scopes:
            for ep in _scope_endpoints(ctx, pm, handler, body, fn_nodes):
                seen = merged.get((ep.key, ep.handler))
                if seen is None:
                    merged[(ep.key, ep.handler)] = ep
                else:
                    seen.fields += [f for f in ep.fields if f not in seen.fields]
        shared = {k for k, _ in merged if sum(1 for k2, _ in merged if k2 == k) > 1}
        for role, direction in (("consume", "provided"), ("publish", "required")):
            eps = [e for e in merged.values() if e.role == role]
            if not eps:
                continue
            iface_id = f"if:message:{pm.id}:{role}"
            ctx.ir.interfaces.append(Interface(
                id=iface_id,
                name=f"{pm.id} {'consumes' if role == 'consume' else 'publishes'}",
                kind="message", direction=direction, owner=pm.id, protocol="AMQP (RabbitMQ, pika)",
                operations=[_operation(iface_id, e, exchange_types, e.key in shared) for e in eps],
                source_refs=[pm.module.path],
            ))
