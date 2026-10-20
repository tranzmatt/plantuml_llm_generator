"""Environment variables read by the code, as one required 'env' interface."""

from __future__ import annotations

import ast
from typing import Dict, Set

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX
from ..python_ast import ExtractContext, dotted_name, literal_text, str_literal
from ._common import call_target, kwarg

_GETTERS = {f"{EXTERNAL_PREFIX}os.getenv", f"{EXTERNAL_PREFIX}os.environ.get"}
_ENVIRON = f"{EXTERNAL_PREFIX}os.environ"


def detect_env_vars(ctx: ExtractContext) -> None:
    used_in: Dict[str, Set[str]] = {}
    defaults: Dict[str, str] = {}
    for pm in ctx.modules.values():
        for node in ast.walk(pm.tree):
            name = None
            if isinstance(node, ast.Call) and node.args and call_target(ctx, pm, node) in _GETTERS:
                name = str_literal(node.args[0])
                default = node.args[1] if len(node.args) > 1 else kwarg(node, "default")
                if name and default is not None:
                    defaults.setdefault(name, literal_text(default))
            elif isinstance(node, ast.Subscript):
                dn = dotted_name(node.value)
                if dn and ctx.resolver.resolve_name(pm, dn) == _ENVIRON:
                    name = str_literal(node.slice)
            if name:
                used_in.setdefault(name, set()).add(pm.id)
    if not used_in:
        return

    iface_id = f"if:env:{ctx.ir.repo}"
    ops = []
    for name in sorted(used_in):
        op = Operation(id=f"{iface_id}:{name}", name=name, meta={"used_in": ", ".join(sorted(used_in[name]))})
        if name in defaults:
            op.meta["default"] = defaults[name]
        ops.append(op)
    ctx.ir.interfaces.append(Interface(
        id=iface_id, name="Environment", kind="env", direction="required",
        owner=ctx.ir.repo, protocol="process environment", operations=ops,
    ))
