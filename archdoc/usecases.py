"""
Use cases from the interface inventory (DESIGN.md 9), without a model.

Use cases are the operations the system provides to the outside (UI events,
HTTP routes, CLI commands); each interface kind has its primary actor. The
secondary actors are the external systems a use case reaches: following the
handler's calls, and messages through the broker to their consumers and on
(archdoc.flows), to every module that uses an outbound library (HTTP client,
database, cloud SDK). The external system is named by the hosts of URL
literals in those modules when there are any, otherwise by the library. A
model may only rename actors and give use cases goal names, keyed by id
(`plan_usecase`).
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

from .coverage import INTERFACE_LIBRARIES
from .extract.bodies import _parsed
from .flows import ENTRY_KINDS, _Tracer
from .ir import EXTERNAL_PREFIX, RepoIR

PRIMARY = {"ui": "User", "http": "API client", "cli": "Operator"}
MAX_REACH = 400


@dataclass
class Actor:
    key: str                   # neutral: "actor1" ... (safe to show any tier)
    name: str
    primary: bool
    source: str = ""           # what it stands for: "primary:ui", "ext:imagerecognize.com", "lib:requests"


@dataclass
class UseCase:
    id: str                    # operation id
    name: str                  # operation name
    interface: str             # interface name
    actor: str                 # primary actor key
    reaches: List[str] = field(default_factory=list)   # secondary actor keys
    via: List[str] = field(default_factory=list)       # modules that reach each external system


@dataclass
class UseCaseModel:
    system: str
    actors: List[Actor]
    cases: List[UseCase]


def _url_hosts(root: str, path: str) -> List[str]:
    full = os.path.join(root, path)
    try:
        tree, _ = _parsed(full, os.path.getmtime(full))
    except OSError:
        return []
    hosts: List[str] = []
    for node in ast.walk(tree) if tree else []:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.match(r"https?://", node.value):
            host = urlparse(node.value).hostname
            if host and "." in host and host not in hosts and host not in ("localhost",) and not host[0].isdigit():
                hosts.append(host)
    return hosts


def _outbound(ir: RepoIR, mid: str) -> List[str]:
    """Outbound libraries (required http, db, cloud) a module imports."""
    m = ir.module(mid)
    out = []
    for imp in m.imports if m else []:
        if not imp.startswith(EXTERNAL_PREFIX):
            continue
        name = imp[len(EXTERNAL_PREFIX):]
        for lib, (kind, direction, _desc) in INTERFACE_LIBRARIES.items():
            if (name == lib or name.startswith(lib + ".")) and kind in ("http", "db", "cloud") \
                    and direction == "required" and lib not in out:
                out.append(lib)
    return out


def _calls_lib(fn, lib: str) -> bool:
    return any(c.startswith(f"{EXTERNAL_PREFIX}{lib}.") or c == f"{EXTERNAL_PREFIX}{lib}" for c in fn.calls)


def _reach(ir: RepoIR, t: _Tracer, handler: str) -> Set[str]:
    """Functions reached from `handler`: calls, then message deliveries, and on."""
    fns = t.fns
    seen_fn: Set[str] = set()
    todo = [handler]
    while todo and len(seen_fn) < MAX_REACH:
        f = todo.pop()
        if f in seen_fn or f not in fns:
            continue
        seen_fn.add(f)
        todo += [c for c in fns[f].calls if c in fns]
        for _end, chan in t.publishes.get(f, []):
            todo += [e.handler for e in chan.consumers if e.handler]
    return seen_fn


def build_usecases(root: str, ir: RepoIR) -> Optional[UseCaseModel]:
    t = _Tracer(ir, None)
    actors: Dict[str, Actor] = {}            # by source
    cases: List[UseCase] = []

    def actor(source: str, name: str, primary: bool) -> str:
        if source not in actors:
            actors[source] = Actor(f"actor{len(actors) + 1}", name, primary, source)
        return actors[source].key

    hosts_cache: Dict[str, List[str]] = {}
    for iface in ir.interfaces:
        if iface.kind not in ENTRY_KINDS or iface.direction != "provided":
            continue
        pkey = actor(f"primary:{iface.kind}", PRIMARY[iface.kind], True)
        for op in iface.operations:
            uc = UseCase(op.id, op.name, iface.name, pkey)
            if op.handler:
                reached = _reach(ir, t, op.handler)
                for mid in sorted({f.split(":")[0] for f in reached}):
                    # only libraries a reached function of this module actually calls
                    libs = [x for x in _outbound(ir, mid)
                            if any(_calls_lib(t.fns[f], x) for f in reached if f.split(":")[0] == mid)]
                    if not libs:
                        continue
                    m = ir.module(mid)
                    if mid not in hosts_cache:
                        hosts_cache[mid] = _url_hosts(root, m.path) if any(
                            INTERFACE_LIBRARIES[x][0] == "http" for x in libs) else []
                    sources = [(f"ext:{h}", h) for h in hosts_cache[mid]] or [
                        (f"lib:{x}", INTERFACE_LIBRARIES[x][2]) for x in libs]
                    for src, name in sources:
                        k = actor(src, name, False)
                        if k not in uc.reaches:
                            uc.reaches.append(k)
                    if mid not in uc.via:
                        uc.via.append(mid)
            cases.append(uc)
    if not cases:
        return None
    used = {c.actor for c in cases} | {k for c in cases for k in c.reaches}
    return UseCaseModel(ir.repo, [a for a in actors.values() if a.key in used], cases)


__all__ = ["Actor", "UseCase", "UseCaseModel", "build_usecases"]
