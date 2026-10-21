"""
Cross-checks between deployment files and the code (DESIGN.md 2.4).

A deployment file is a second statement of the architecture, and it can
disagree with the code. Each disagreement is reported, because a rebuild from
the documentation has to know which statement holds:

  - code with an entry point or an interface that no image contains
  - a variable the deployed code requires that the service does not set
  - a variable the service sets that its code never reads
  - a credential written literally into the deployment file
  - a message channel whose other end is not deployed by this file
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from .ir import Deployment, RepoIR
from .messaging import RUNTIME, channels

_CREDENTIAL = re.compile(r"PASS|SECRET|TOKEN|API_?KEY|PRIVATE", re.I)
# set for the runtime rather than read by the code
_RUNTIME_VARS = re.compile(r"^(PYTHON\w*|PATH|TZ|LANG|LC_\w+)$")


def deployed_modules(dep: Deployment) -> Dict[str, str]:
    """module id -> service name, for every module some image of `dep` contains."""
    return {m: s.name for s in dep.services for m in s.modules}


def _env_reads(ir: RepoIR) -> Dict[str, Dict[str, bool]]:
    """var -> module -> required (read with no default)."""
    out: Dict[str, Dict[str, bool]] = {}
    for iface in ir.interfaces:
        if iface.kind != "env":
            continue
        for op in iface.operations:
            required = {m.strip() for m in op.meta.get("required_in", "").split(",") if m.strip()}
            for m in op.meta.get("used_in", "").split(","):
                if m.strip():
                    out.setdefault(op.name, {})[m.strip()] = m.strip() in required
    return out


def deployment_issues(ir: RepoIR) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    reads = _env_reads(ir)
    owners = {i.owner for i in ir.interfaces if i.kind not in ("env",)}
    for dep in ir.deployments:
        placed = deployed_modules(dep)
        built = [s for s in dep.services if s.build]

        def add(where: str, issue: str) -> None:
            issues.append({"file": dep.file, "where": where, "issue": issue})

        if built:
            for m in ir.modules:
                if m.id in placed or not (m.has_main or m.id in owners):
                    continue
                add(m.id, "has an entry point or interface but is in no image of this file")
        for s in dep.services:
            if s.build and s.entry_module is None:
                add(s.name, "the module this service runs could not be determined")
            mods = set(s.modules)
            if not mods:
                for name, how in sorted(s.env.items()):
                    if how == "literal" and _CREDENTIAL.search(name):
                        add(s.name, f"credential `{name}` is written literally in the file")
                continue
            for var, per_module in sorted(reads.items()):
                needed = [m for m, req in per_module.items() if req and m in mods]
                if needed and s.env.get(var) in (None, "unset"):
                    add(s.name, f"`{var}` is required by {', '.join(sorted(needed))} but not set")
            read_here = {v for v, per_module in reads.items() if mods & set(per_module)}
            for name, how in sorted(s.env.items()):
                if name not in read_here and not _RUNTIME_VARS.match(name):
                    add(s.name, f"`{name}` is set but not read by the code in this image")
                if how == "literal" and _CREDENTIAL.search(name):
                    add(s.name, f"credential `{name}` is written literally in the file")
        if built:
            for c in channels(ir):
                if c.name == RUNTIME:
                    continue
                pubs = {placed[e.module] for e in c.publishers if e.module in placed}
                subs = {placed[e.module] for e in c.consumers if e.module in placed}
                where = f"{c.target} {c.name}"
                if pubs and not subs and c.consumers:
                    add(where, f"published by {', '.join(sorted(pubs))}; its consumers are not deployed")
                elif subs and not pubs and c.publishers:
                    add(where, f"consumed by {', '.join(sorted(subs))}; its publishers are not deployed")
    return issues


def service_flows(ir: RepoIR, dep: Deployment) -> List[Dict[str, str]]:
    """Publisher service -> consumer service through each channel, for deployed code."""
    placed = deployed_modules(dep)
    flows: List[Dict[str, str]] = []
    seen: Set[tuple] = set()
    for c in channels(ir):
        for p in c.publishers:
            for q in c.consumers:
                a, b = placed.get(p.module), placed.get(q.module)
                if a and b and a != b and (a, b, c.key) not in seen:
                    seen.add((a, b, c.key))
                    flows.append({"from": a, "to": b, "channel": f"{c.target} {c.name}"})
    return flows
