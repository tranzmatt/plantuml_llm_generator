"""Command-line interfaces: argparse, Click and Typer."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ...interfaces import Interface, Operation
from ...ir import EXTERNAL_PREFIX, Field
from ..python_ast import ExtractContext, ParsedModule, dotted_name, literal_text, str_literal
from ._common import call_target, kwarg, scope_nodes

_ARGPARSE_PARSER = f"{EXTERNAL_PREFIX}argparse.ArgumentParser"


def _program_name(pm: ParsedModule) -> str:
    return pm.module.path.rsplit("/", 1)[-1]


# ===========================================================================
# argparse
# ===========================================================================

def detect_argparse(ctx: ExtractContext) -> None:
    for pm in ctx.modules.values():
        scopes: List[Tuple[str, Optional[str], List[ast.stmt]]] = [("<module>", None, pm.tree.body)]
        scopes += [(f.fn.name, f.fn.id, f.node.body) for f in pm.funcs]
        for scope_name, handler, body in scopes:
            for iface in _argparse_in_scope(ctx, pm, scope_name, handler, body):
                ctx.ir.interfaces.append(iface)


def _argparse_in_scope(ctx, pm, scope_name, handler, body) -> List[Interface]:
    # parser var -> operation it feeds; subparsers-action vars; root parser vars
    ops: Dict[str, Operation] = {}
    roots: Dict[str, Tuple[ast.Call, Operation]] = {}
    subactions: Dict[str, str] = {}          # subparsers var -> root parser var
    sub_ops: Dict[str, List[Operation]] = {}  # root var -> subcommand operations

    assigns = [n for n in scope_nodes(body)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)]
    for a in sorted(assigns, key=lambda n: (n.lineno, n.col_offset)):
        var, call = a.targets[0].id, a.value
        if call_target(ctx, pm, call) == _ARGPARSE_PARSER:
            prog = str_literal(kwarg(call, "prog")) or _program_name(pm)
            op = Operation(id="", name=prog, handler=handler,
                           docstring=str_literal(kwarg(call, "description")))
            ops[var] = op
            roots[var] = (call, op)
            sub_ops[var] = []
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        owner = dotted_name(call.func.value)
        if call.func.attr == "add_subparsers" and owner in ops:
            subactions[var] = _root_of(owner, roots, ops)
        elif call.func.attr == "add_parser" and owner in subactions and call.args:
            name = str_literal(call.args[0]) or "<dynamic>"
            op = Operation(id="", name=name,
                           docstring=str_literal(kwarg(call, "help")) or str_literal(kwarg(call, "description")))
            ops[var] = op
            sub_ops[subactions[owner]].append(op)
        elif call.func.attr in ("add_argument_group", "add_mutually_exclusive_group") and owner in ops:
            ops[var] = ops[owner]

    if not roots:
        return []

    for n in scope_nodes(body):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        owner = dotted_name(n.func.value)
        if owner not in ops:
            continue
        if n.func.attr == "add_argument":
            f = _argparse_field(n)
            if f:
                ops[owner].params.append(f)
        elif n.func.attr == "set_defaults":
            target = kwarg(n, "func")
            dn = dotted_name(target) if target is not None else None
            resolved = ctx.resolver.resolve_name(pm, dn) if dn else None
            if resolved in ctx.resolver.functions:
                ops[owner].handler = resolved

    out = []
    for var, (call, root_op) in roots.items():
        iface_id = f"if:cli:{pm.id}:{scope_name}:{var}"
        operations = [root_op] + sorted(sub_ops[var], key=lambda o: o.name)
        for op in operations:
            op.id = f"{iface_id}:{op.name}"
        out.append(Interface(
            id=iface_id,
            name=root_op.name,
            kind="cli",
            direction="provided",
            owner=pm.id,
            protocol="argv",
            address=root_op.name,
            operations=operations,
            source_refs=[handler] if handler else [],
        ))
    return out


def _root_of(var: str, roots, ops) -> str:
    """Which root parser a (sub)parser variable ultimately belongs to."""
    if var in roots:
        return var
    for root, (_, root_op) in roots.items():
        if ops.get(var) is root_op:
            return root
    return next(iter(roots))


def _argparse_field(call: ast.Call) -> Optional[Field]:
    names = [s for s in (str_literal(a) for a in call.args) if s]
    if not names:
        return None
    options = [s for s in names if s.startswith("-")]
    positional = not options
    name = max(options, key=len).lstrip("-") if options else names[0]
    dest = str_literal(kwarg(call, "dest"))

    action = str_literal(kwarg(call, "action"))
    type_node = kwarg(call, "type")
    if action in ("store_true", "store_false"):
        annotation = "bool"
    elif action == "count":
        annotation = "int"
    else:
        annotation = ast.unparse(type_node) if type_node is not None else "str"
    nargs = kwarg(call, "nargs")
    nargs_text = literal_text(nargs)
    if nargs is not None and nargs_text not in ("'?'", "1"):
        annotation = f"list[{annotation}]"

    required_node = kwarg(call, "required")
    if positional:
        required = nargs_text not in ("'?'", "'*'")
    else:
        required = isinstance(required_node, ast.Constant) and required_node.value is True

    constraints = {"kind": "positional" if positional else "option"}
    if options:
        constraints["flags"] = ", ".join(options)
    if dest:
        constraints["dest"] = dest
    choices = kwarg(call, "choices")
    if choices is not None:
        constraints["choices"] = literal_text(choices)
    if action and action not in ("store",):
        constraints["action"] = action
    default = kwarg(call, "default")
    return Field(
        name=name,
        annotation=annotation,
        required=required,
        default=literal_text(default) if default is not None else None,
        constraints=constraints,
        docstring=str_literal(kwarg(call, "help")),
    )


# ===========================================================================
# Click / Typer
# ===========================================================================

_CLICK_COMMAND = f"{EXTERNAL_PREFIX}click.command"
_CLICK_GROUP = f"{EXTERNAL_PREFIX}click.group"
_CLICK_OPTION = f"{EXTERNAL_PREFIX}click.option"
_CLICK_ARGUMENT = f"{EXTERNAL_PREFIX}click.argument"
_TYPER_APP = f"{EXTERNAL_PREFIX}typer.Typer"
_TYPER_OPTION = f"{EXTERNAL_PREFIX}typer.Option"
_TYPER_ARGUMENT = f"{EXTERNAL_PREFIX}typer.Argument"


def _cmd_name(deco: ast.AST, func_name: str) -> str:
    if isinstance(deco, ast.Call):
        explicit = (str_literal(deco.args[0]) if deco.args else None) or str_literal(kwarg(deco, "name"))
        if explicit:
            return explicit
    return func_name.replace("_", "-")


def detect_click_typer(ctx: ExtractContext) -> None:
    # root key -> (Interface, ops). Root key is a click group function id, a
    # standalone click command id, or a Typer app variable "mod:var".
    roots: Dict[str, Interface] = {}

    def root_iface(key: str, pm: ParsedModule, name: str, protocol: str) -> Interface:
        if key not in roots:
            roots[key] = Interface(
                id=f"if:cli:{key}", name=name, kind="cli", direction="provided",
                owner=pm.id, protocol=protocol, address=name,
            )
        return roots[key]

    for pm in ctx.modules.values():
        for fnode in pm.funcs:
            fn = fnode.fn
            for deco in fnode.node.decorator_list:
                target_node = deco.func if isinstance(deco, ast.Call) else deco
                dn = dotted_name(target_node)
                if not dn:
                    continue
                resolved = ctx.resolver.resolve_name(pm, dn)
                base, _, attr = dn.rpartition(".")

                if resolved in (_CLICK_GROUP, _CLICK_COMMAND):
                    iface = root_iface(fn.id, pm, _cmd_name(deco, fn.name), "argv (click)")
                    if resolved == _CLICK_COMMAND:
                        iface.operations.append(_click_operation(ctx, pm, fnode, iface.name))
                    iface.source_refs.append(fn.id)
                    break

                if attr in ("command", "group") and base:
                    base_id = ctx.resolver.resolve_name(pm, base)
                    var = ctx.resolver.resolve_var(pm, base) if "." not in base else None
                    if var and var[0].var_types.get(var[1]) == _TYPER_APP:
                        key = f"{var[0].id}:{var[1]}"
                        iface = root_iface(key, var[0], var[0].module.path.rsplit("/", 1)[-1], "argv (typer)")
                        iface.operations.append(_typer_operation(ctx, pm, fnode, _cmd_name(deco, fn.name)))
                        iface.source_refs.append(fn.id)
                        break
                    if base_id in ctx.resolver.functions and _is_click_group(ctx, base_id):
                        owner_pm = ctx.modules[base_id.split(":")[0]]
                        group_fn = ctx.resolver.functions[base_id]
                        iface = root_iface(base_id, owner_pm, group_fn.name.replace("_", "-"), "argv (click)")
                        iface.operations.append(_click_operation(ctx, pm, fnode, _cmd_name(deco, fn.name)))
                        iface.source_refs.append(fn.id)
                        break

    for iface in roots.values():
        for op in iface.operations:
            op.id = f"{iface.id}:{op.name}"
        iface.operations.sort(key=lambda o: o.name)
        iface.source_refs = sorted(set(iface.source_refs))
        ctx.ir.interfaces.append(iface)


def _is_click_group(ctx: ExtractContext, func_id: str) -> bool:
    pm = ctx.modules[func_id.split(":")[0]]
    fn = ctx.resolver.functions[func_id]
    for d in fn.decorators:
        if ctx.resolver.resolve_name(pm, d.name) == _CLICK_GROUP or d.name.endswith(".group"):
            return True
    return False


def _click_operation(ctx, pm, fnode, name: str) -> Operation:
    op = Operation(id="", name=name, handler=fnode.fn.id, docstring=fnode.fn.docstring)
    for deco in fnode.node.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        target = call_target(ctx, pm, deco)
        if target not in (_CLICK_OPTION, _CLICK_ARGUMENT):
            continue
        names = [s for s in (str_literal(a) for a in deco.args) if s]
        if not names:
            continue
        is_arg = target == _CLICK_ARGUMENT
        options = [s for s in names if s.startswith("-")]
        pname = names[0] if is_arg else max(options or names, key=len).lstrip("-")
        is_flag = kwarg(deco, "is_flag")
        type_node = kwarg(deco, "type")
        annotation = "bool" if (isinstance(is_flag, ast.Constant) and is_flag.value) else (
            ast.unparse(type_node) if type_node is not None else "str")
        required_node = kwarg(deco, "required")
        required = (required_node.value if isinstance(required_node, ast.Constant) else is_arg)
        default = kwarg(deco, "default")
        constraints = {"kind": "positional" if is_arg else "option"}
        if options:
            constraints["flags"] = ", ".join(options)
        op.params.append(Field(
            name=pname, annotation=annotation, required=bool(required),
            default=literal_text(default) if default is not None else None,
            constraints=constraints, docstring=str_literal(kwarg(deco, "help")),
        ))
    return op


def _typer_operation(ctx, pm, fnode, name: str) -> Operation:
    op = Operation(id="", name=name, handler=fnode.fn.id, docstring=fnode.fn.docstring)
    node = fnode.node
    args = node.args.args + node.args.kwonlyargs
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults) \
        + list(node.args.kw_defaults)
    for a, d in zip(args, defaults):
        is_arg = d is None
        default_node, help_text = d, None
        if isinstance(d, ast.Call) and call_target(ctx, pm, d) in (_TYPER_OPTION, _TYPER_ARGUMENT):
            is_arg = call_target(ctx, pm, d) == _TYPER_ARGUMENT
            default_node = d.args[0] if d.args else kwarg(d, "default")
            help_text = str_literal(kwarg(d, "help"))
        required = default_node is None or (
            isinstance(default_node, ast.Constant) and default_node.value is Ellipsis)
        constraints = {"kind": "positional" if is_arg else "option"}
        if not is_arg:
            constraints["flags"] = "--" + a.arg.replace("_", "-")
        op.params.append(Field(
            name=a.arg,
            annotation=ast.unparse(a.annotation) if a.annotation is not None else None,
            required=required,
            default=None if required else literal_text(default_node),
            constraints=constraints,
            docstring=help_text,
        ))
    return op
