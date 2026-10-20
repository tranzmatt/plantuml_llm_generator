"""Helpers shared by the framework detectors."""

from __future__ import annotations

import ast
from typing import Optional

from ...interfaces import TypeRef
from ..python_ast import ExtractContext, ParsedModule, dotted_name


def call_target(ctx: ExtractContext, pm: ParsedModule, call: ast.Call) -> Optional[str]:
    """Resolved id of what a call invokes, using module scope only."""
    dn = dotted_name(call.func)
    return ctx.resolver.resolve_name(pm, dn) if dn else None


def kwarg(call: ast.Call, name: str) -> Optional[ast.expr]:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def type_ref(ctx: ExtractContext, pm: ParsedModule, node: Optional[ast.AST]) -> Optional[TypeRef]:
    """TypeRef for an annotation / response_model, linked to a DataType when possible."""
    if node is None:
        return None
    data_type_ids = {d.id for d in ctx.ir.data_types}
    types = ctx.resolver.annotation_types(pm, node)
    linked = next((t for t in types if t in data_type_ids), None)
    return TypeRef(display=ast.unparse(node), type_id=linked)
