"""
Gradio: a web UI as a provided `ui` interface.

Each UI event wired to a function (`button.click(fn=..., inputs=..., outputs=...)`,
`gr.Interface(fn=...)`) is one operation: its inputs are the parameters and its
outputs the outputs, both named by the components' labels. The port comes
from `launch(server_port=...)` when it is a literal.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX, Field
from ..python_ast import ExtractContext, ParsedModule, dotted_name, str_literal
from ._common import call_target, kwarg, scope_nodes

_GR = f"{EXTERNAL_PREFIX}gradio."
_EVENTS = {"click", "submit", "change", "upload", "select", "input", "load", "blur", "release"}


def _component(ctx: ExtractContext, pm: ParsedModule, node: ast.expr) -> Optional[Tuple[str, Optional[str]]]:
    """(component class, label) for a gr.<Component>(...) call."""
    if not isinstance(node, ast.Call):
        return None
    target = call_target(ctx, pm, node) or ""
    if not target.startswith(_GR):
        return None
    label = str_literal(kwarg(node, "label"))
    if label is None and target.endswith("Button"):        # gr.Button("Submit"): the value is the label
        label = str_literal(kwarg(node, "value")) or (str_literal(node.args[0]) if node.args else None)
    return target[len(_GR):], label


def _fields(node: Optional[ast.expr], comps: Dict[str, Tuple[str, Optional[str]]]) -> List[Field]:
    if node is None:
        return []
    elts = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    out = []
    for e in elts:
        if isinstance(e, ast.Name) and e.id in comps:
            kind, label = comps[e.id]
            out.append(Field(name=label or e.id, annotation=kind))
        elif isinstance(e, ast.Constant) and isinstance(e.value, str):
            out.append(Field(name=e.value, annotation=e.value))       # gr.Interface shorthand: "text"
    return out


def _handler(ctx: ExtractContext, pm: ParsedModule, node: Optional[ast.expr]) -> Optional[str]:
    dn = dotted_name(node) if node is not None else None
    return ctx.resolver.resolve_name(pm, dn) if dn else None


def detect_gradio(ctx: ExtractContext) -> None:
    for pm in ctx.modules.values():
        if not any(i.startswith(_GR) or i == _GR[:-1] for i in pm.module.imports):
            continue
        bodies = [pm.tree.body] + [f.node.body for f in pm.funcs]
        comps: Dict[str, Tuple[str, Optional[str]]] = {}
        ops: List[Operation] = []
        port: Optional[str] = None
        title: Optional[str] = None
        iface_id = f"if:ui:{pm.id}"
        for body in bodies:
            nodes = list(scope_nodes(body))
            for n in nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    c = _component(ctx, pm, n.value)
                    if c:
                        comps[n.targets[0].id] = c
                elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
                    c = _component(ctx, pm, n.context_expr)
                    if c and c[0] == "Blocks":
                        title = title or str_literal(kwarg(n.context_expr, "title"))
            calls = sorted((n for n in nodes if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
            for call in calls:
                target = call_target(ctx, pm, call) or ""
                if target == f"{_GR}Interface":
                    title = title or str_literal(kwarg(call, "title"))
                    ops.append(Operation(
                        id=f"{iface_id}:interface", name="submit",
                        params=_fields(kwarg(call, "inputs"), comps),
                        outputs=_fields(kwarg(call, "outputs"), comps),
                        handler=_handler(ctx, pm, kwarg(call, "fn") or (call.args[0] if call.args else None)),
                    ))
                    continue
                if not isinstance(call.func, ast.Attribute):
                    continue
                if call.func.attr == "launch":
                    p = kwarg(call, "server_port")
                    if isinstance(p, ast.Constant):
                        port = str(p.value)
                    continue
                owner = call.func.value
                if call.func.attr not in _EVENTS or not (isinstance(owner, ast.Name) and owner.id in comps):
                    continue
                kind, label = comps[owner.id]
                fn = kwarg(call, "fn") or (call.args[0] if call.args else None)
                handler = _handler(ctx, pm, fn)
                if handler is None:
                    continue
                name = f"{call.func.attr} {label or owner.id}"
                ops.append(Operation(
                    id=f"{iface_id}:{call.func.attr}:{label or owner.id}", name=name,
                    params=_fields(kwarg(call, "inputs") or (call.args[1] if len(call.args) > 1 else None), comps),
                    outputs=_fields(kwarg(call, "outputs") or (call.args[2] if len(call.args) > 2 else None), comps),
                    handler=handler, meta={"component": kind},
                ))
        if not ops:
            continue
        ctx.ir.interfaces.append(Interface(
            id=iface_id, name=title or f"{pm.id} UI", kind="ui", direction="provided", owner=pm.id,
            protocol="HTTP (Gradio web UI)", address=f"port {port}" if port else None,
            operations=ops, source_refs=[pm.module.path],
        ))
