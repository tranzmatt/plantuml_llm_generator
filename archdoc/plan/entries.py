"""
Entry points and call-graph slices for behavioral diagrams. Local only:
nothing here leaves the machine; egress.build_sequence_payload does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..interfaces import Interface, Operation
from ..ir import RepoIR


@dataclass
class Entry:
    handler: str                                   # function id
    label: str                                     # human description for logs / titles
    via: Optional[Tuple[Interface, Operation]] = None


def call_slice(ir: RepoIR, handler: str, depth: int = 3, max_nodes: int = 25) -> List[str]:
    """Internal functions reachable from `handler`, breadth-first in call order."""
    functions = {f.id: f for f in ir.iter_functions()}
    if handler not in functions:
        return []
    order = [handler]
    frontier = [handler]
    for _ in range(depth):
        nxt = []
        for fid in frontier:
            for c in functions[fid].calls:
                if c in functions and c not in order and len(order) < max_nodes:
                    order.append(c)
                    nxt.append(c)
        frontier = nxt
    return order


def discover_entries(ir: RepoIR) -> List[Entry]:
    """Interface operations with a handler, then `main` functions of runnable modules."""
    entries: List[Entry] = []
    seen = set()
    for iface in ir.interfaces:
        for op in iface.operations:
            if op.handler and op.handler not in seen:
                seen.add(op.handler)
                entries.append(Entry(op.handler, f"{iface.kind} {op.name}", (iface, op)))
    for m in ir.modules:
        fid = f"{m.id}:main"
        if m.has_main and fid not in seen and ir.find_function(fid):
            seen.add(fid)
            entries.append(Entry(fid, f"{m.id} main"))
    return entries


def select_entries(ir: RepoIR, wanted: List[str], limit: int, depth: int, max_nodes: int) -> List[Entry]:
    """`wanted` may name handler ids or operation ids; otherwise pick the entries
    with the largest call graphs (the most informative sequences)."""
    entries = discover_entries(ir)
    if wanted:
        by_key: Dict[str, Entry] = {}
        for e in entries:
            by_key[e.handler] = e
            if e.via:
                by_key[e.via[1].id] = e
        out = []
        for w in wanted:
            if w in by_key:
                out.append(by_key[w])
            elif ir.find_function(w):
                out.append(Entry(w, w))
            else:
                raise ValueError(f"unknown entry {w!r} (use a function id like pkg.mod:func or an operation id)")
        return out
    sized = sorted(entries, key=lambda e: (-len(call_slice(ir, e.handler, depth, max_nodes)), e.handler))
    return [e for e in sized if len(call_slice(ir, e.handler, depth, max_nodes)) > 1][:limit]
