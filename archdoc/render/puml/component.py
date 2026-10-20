"""
Deterministic component diagram from the IR.

  - components are module groups at a package depth (auto-chosen by default)
  - dependencies come from internal imports between groups
  - third-party libraries appear as <<external>> components
  - provided interfaces are lollipops (I - C); required ones are sockets (C -( I)
  - message interfaces are drawn through the broker instead: each queue and
    exchange the code names is a node, with publisher -> channel -> consumer
    arrows, so the diagram shows who talks to whom
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Set

from ...interfaces import Interface
from ...messaging import RUNTIME, channels
from ...ir import EXTERNAL_PREFIX, RepoIR
from .common import Aliases, Diagram, label

MAX_COMPONENTS = 30
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def group_of(module_id: str, depth: int) -> str:
    return ".".join(module_id.split(".")[:depth])


def auto_depth(module_ids: List[str], limit: int = MAX_COMPONENTS) -> int:
    """Smallest depth reaching the largest component count that stays within `limit`."""
    counts = {d: len({group_of(m, d) for m in module_ids}) for d in range(1, 7)}
    feasible = {d: n for d, n in counts.items() if n <= limit} or {1: counts[1]}
    best = max(feasible.values())
    return min(d for d, n in feasible.items() if n == best)


def interface_label(iface: Interface) -> str:
    if iface.kind == "http":
        return f"HTTP {iface.address or '/'}"
    if iface.kind == "cli":
        return f"CLI {iface.name}"
    if iface.kind == "env":
        return f"ENV ({len(iface.operations)} vars)"
    return f"{iface.kind.upper()} {iface.name}"


def render_component_diagram(
    ir: RepoIR,
    name: str,
    depth: Optional[int] = None,
    externals: bool = True,
    interfaces: bool = True,
    title: Optional[str] = None,
) -> str:
    module_ids = [m.id for m in ir.modules]
    depth = depth or auto_depth(module_ids)
    aliases = Aliases()
    groups: Dict[str, Set[str]] = {}
    for mid in module_ids:
        groups.setdefault(group_of(mid, depth), set()).add(mid)
    group_by_module = {mid: group_of(mid, depth) for mid in module_ids}

    deps: Dict[str, Set[str]] = {g: set() for g in groups}
    ext_deps: Dict[str, Set[str]] = {g: set() for g in groups}
    for m in ir.modules:
        g = group_by_module[m.id]
        for imp in m.imports:
            if imp.startswith(EXTERNAL_PREFIX):
                root = imp[len(EXTERNAL_PREFIX):].split(".")[0]
                if root not in _STDLIB:
                    ext_deps[g].add(root)
            elif imp in group_by_module and group_by_module[imp] != g:
                deps[g].add(group_by_module[imp])

    diagram = Diagram(name, title, settings=["skinparam componentStyle uml2"])

    # Components, nested in a package per top-level segment when depth > 1
    by_top: Dict[str, List[str]] = {}
    nested_tops = {g.split(".")[0] for g in groups if "." in g} if depth > 1 else set()
    for g in sorted(groups):
        top = g.split(".")[0]
        by_top.setdefault(top if top in nested_tops else "", []).append(g)
    for top in sorted(by_top):
        indent = ""
        if top:
            diagram.add(f"package {label(top)} as {aliases('pkg:' + top, 'P')} {{")
            indent = "  "
        for g in by_top[top]:
            diagram.add(f"{indent}component {label(g)} as {aliases(g, 'C')}")
        if top:
            diagram.add("}")

    used_ext = sorted(set().union(*ext_deps.values())) if externals and ext_deps else []
    if used_ext:
        diagram.add(f"package {label('External libraries')} as {aliases('pkg:external', 'P')} {{")
        for root in used_ext:
            diagram.add(f"  component {label(root)} as {aliases('ext:' + root, 'X')} <<external>>")
        diagram.add("}")

    iface_edges: List[str] = []
    if interfaces:
        iface_edges += _message_channels(ir, diagram, aliases, group_by_module)
        for iface in ir.interfaces:
            if iface.kind == "message":
                continue
            ialias = aliases(iface.id, "I")
            diagram.add(f"interface {label(interface_label(iface))} as {ialias}")
            if iface.direction == "provided" and iface.owner in group_by_module:
                iface_edges.append(f"{ialias} -- {aliases(group_by_module[iface.owner], 'C')}")
            else:
                users = {iface.owner} if iface.owner in group_by_module else set()
                for op in iface.operations:
                    users.update(u.strip() for u in op.meta.get("used_in", "").split(",") if u.strip())
                for g in sorted({group_by_module[u] for u in users if u in group_by_module}):
                    iface_edges.append(f"{aliases(g, 'C')} --( {ialias}")

    for g in sorted(deps):
        for target in sorted(deps[g]):
            diagram.edge(f"{aliases(g, 'C')} ..> {aliases(target, 'C')}")
    for g in sorted(ext_deps):
        if externals:
            for root in sorted(ext_deps[g]):
                diagram.edge(f"{aliases(g, 'C')} ..> {aliases('ext:' + root, 'X')}")
    for e in iface_edges:
        diagram.edge(e)
    return diagram.text()


def _message_channels(ir: RepoIR, diagram: Diagram, aliases: Aliases, group_by_module: Dict[str, str]) -> List[str]:
    chans = channels(ir)
    if not chans:
        return []
    edges: List[str] = []
    for broker in sorted({c.broker for c in chans}):
        diagram.add(f"node {label(broker)} as {aliases('broker:' + broker, 'B')} {{")
        for c in (c for c in chans if c.broker == broker):
            stereo = f"<<{c.exchange_type} exchange>>" if c.target == "exchange" and c.exchange_type else f"<<{c.target}>>"
            diagram.add(f"  queue {label(c.name)} as {aliases(c.key, 'Q')} {stereo}")
        diagram.add("}")
    for c in chans:
        q = aliases(c.key, "Q")
        for role, ends in (("publish", c.publishers), ("consume", c.consumers)):
            seen: Set[str] = set()
            for e in ends:
                g = group_by_module.get(e.module)
                if g is None or g in seen:
                    continue
                seen.add(g)
                key = f" : {'routing key set at run time' if e.routing_key == RUNTIME else e.routing_key}" if e.routing_key else ""
                edges.append(f"{aliases(g, 'C')} --> {q}{key}" if role == "publish" else f"{q} --> {aliases(g, 'C')}{key}")
    return edges
