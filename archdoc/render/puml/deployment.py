"""
Deterministic deployment diagram from one deployment file.

Each service is a node holding the modules its image contains (the one it
runs marked <<entry>>); image-only services show their image. Message flows
of the deployed code go through the broker's service when the file has one
(publisher -> broker -> consumer, labelled with the queue or exchange), else
directly between services. Nothing here is inferred: a flow is drawn only
when a deployed module publishes to a channel a deployed module consumes.
"""

from __future__ import annotations

from typing import List, Optional, Set

from ...deployment import deployed_modules
from ...ir import Deployment, RepoIR
from ...messaging import RUNTIME, channels
from .common import Aliases, Diagram, label


def _broker_service(dep: Deployment, broker: str) -> Optional[str]:
    for s in dep.services:
        if s.image and broker.lower() in s.image.lower():
            return s.name
    return None


def render_deployment_diagram(ir: RepoIR, dep: Deployment, name: str, title: Optional[str] = None) -> str:
    aliases = Aliases()
    diagram = Diagram(name, title, settings=["skinparam componentStyle uml2"])
    for s in dep.services:
        lines = [s.name]
        if s.ports:
            lines.append("ports " + ", ".join(s.ports))
        if s.placement:
            lines += s.placement
        stereo = f"<<image {s.image}>>" if s.image and not s.build else "<<container>>"
        node = aliases("svc:" + s.name, "S")
        if s.modules:
            diagram.add(f"node {label(chr(10).join(lines))} as {node} {stereo} {{")
            for m in s.modules:
                entry = " <<entry>>" if m == s.entry_module else ""
                diagram.add(f"  component {label(m)} as {aliases(s.name + '/' + m, 'C')}{entry}")
            diagram.add("}")
        else:
            diagram.add(f"node {label(chr(10).join(lines))} as {node} {stereo}")

    for s in dep.services:
        for d in s.depends_on:
            if aliases.get("svc:" + d):
                diagram.edge(f"{aliases('svc:' + s.name, 'S')} ..> {aliases('svc:' + d, 'S')} : depends on")

    placed = deployed_modules(dep)
    seen: Set[tuple] = set()
    for c in channels(ir):
        if c.name == RUNTIME:
            continue
        hub = _broker_service(dep, c.broker)
        pubs = sorted({placed[e.module] for e in c.publishers if e.module in placed})
        subs = sorted({placed[e.module] for e in c.consumers if e.module in placed})
        edges: List[str] = []
        if hub:
            edges += [f"{aliases('svc:' + p, 'S')} --> {aliases('svc:' + hub, 'S')} : {c.name}" for p in pubs]
            edges += [f"{aliases('svc:' + hub, 'S')} --> {aliases('svc:' + q, 'S')} : {c.name}" for q in subs]
        else:
            edges += [f"{aliases('svc:' + p, 'S')} --> {aliases('svc:' + q, 'S')} : {c.name}"
                      for p in pubs for q in subs if p != q]
        for e in edges:
            if e not in seen:
                seen.add(e)
                diagram.edge(e)
    return diagram.text()
