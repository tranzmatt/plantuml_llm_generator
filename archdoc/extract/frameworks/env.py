"""
Environment variables read by the code, as one required 'env' interface.

Reads are found directly (os.getenv / os.environ.get / os.environ[...]) and
through wrappers: a function that passes one of its parameters to one of those
(`def required_env(name): ... os.environ.get(name)`) makes each call with a
literal argument a read of that variable. Whether a variable has a default is
recorded per module, since one module may require what another defaults.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Set, Tuple

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX
from ..python_ast import ExtractContext, ParsedModule, dotted_name, literal_text, str_literal
from ._common import call_target, kwarg

_GETTERS = {f"{EXTERNAL_PREFIX}os.getenv", f"{EXTERNAL_PREFIX}os.environ.get"}
_ENVIRON = f"{EXTERNAL_PREFIX}os.environ"

# (name expression, default expression or None) for one direct read
Read = Tuple[ast.expr, Optional[ast.expr]]


def _direct_read(ctx: ExtractContext, pm: ParsedModule, node: ast.AST) -> Optional[Read]:
    if isinstance(node, ast.Call) and node.args and call_target(ctx, pm, node) in _GETTERS:
        default = node.args[1] if len(node.args) > 1 else kwarg(node, "default")
        return node.args[0], default
    if isinstance(node, ast.Subscript):
        dn = dotted_name(node.value)
        if dn and ctx.resolver.resolve_name(pm, dn) == _ENVIRON:
            return node.slice, None
    return None


def _wrappers(ctx: ExtractContext) -> Dict[str, Tuple[int, str, bool]]:
    """function id -> (index of the name parameter, its name, whether it supplies a default)."""
    found: Dict[str, Tuple[int, str, bool]] = {}
    for pm in ctx.modules.values():
        for f in pm.funcs:
            params = [p.name for p in f.fn.params]
            for node in ast.walk(f.node):
                read = _direct_read(ctx, pm, node)
                if read and isinstance(read[0], ast.Name) and read[0].id in params:
                    name = read[0].id
                    found[f.fn.id] = (params.index(name), name, read[1] is not None)
                    break
    return found


def detect_env_vars(ctx: ExtractContext) -> None:
    wrappers = _wrappers(ctx)
    # var -> module -> set of default texts ("" when read without a default)
    reads: Dict[str, Dict[str, Set[str]]] = {}

    def add(name: Optional[str], module: str, default: Optional[ast.expr]) -> None:
        if name:
            reads.setdefault(name, {}).setdefault(module, set()).add(
                "" if default is None else (literal_text(default) or "<expr>"))

    for pm in ctx.modules.values():
        for node in ast.walk(pm.tree):
            read = _direct_read(ctx, pm, node)
            if read:
                add(str_literal(read[0]), pm.id, read[1])
                continue
            if isinstance(node, ast.Call):
                target = call_target(ctx, pm, node)
                if target in wrappers:
                    idx, pname, has_default = wrappers[target]
                    arg = kwarg(node, pname) or (node.args[idx] if len(node.args) > idx else None)
                    # a wrapper that supplies its own default is not "required"
                    add(str_literal(arg), pm.id, ast.Constant("<default in wrapper>") if has_default else None)
    if not reads:
        return

    iface_id = f"if:env:{ctx.ir.repo}"
    ops = []
    for name in sorted(reads):
        per_module = reads[name]
        op = Operation(id=f"{iface_id}:{name}", name=name, meta={"used_in": ", ".join(sorted(per_module))})
        required = sorted(m for m, d in per_module.items() if "" in d)
        defaults = {d for ds in per_module.values() for d in ds if d}
        if required:
            op.meta["required_in"] = ", ".join(required)
        if len(defaults) == 1:
            op.meta["default"] = next(iter(defaults))
        elif defaults:
            op.meta["defaults_differ"] = "yes"
        ops.append(op)
    ctx.ir.interfaces.append(Interface(
        id=iface_id, name="Environment", kind="env", direction="required",
        owner=ctx.ir.repo, protocol="process environment", operations=ops,
    ))
