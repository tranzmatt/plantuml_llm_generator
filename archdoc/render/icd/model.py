"""
ICD context: every fact in an ICD table, built deterministically from the IR.

The template only lays these out. Narrative (purpose, semantics) is not
produced here; those slots are None and render as TBD until the Phase 3
planner fills them, and every TBD is listed in the gaps appendix.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Set

from ...interfaces import DataType, Interface, Operation, TypeRef
from ...ir import Field, RepoIR

KIND_LABELS = {
    "http": "HTTP API",
    "cli": "Command-line interface",
    "env": "Environment configuration",
    "message": "Messaging",
    "ui": "User interface",
    "rpc": "RPC",
    "library": "Library API",
    "file": "File interface",
    "db": "Database",
}

# Questions an ICD must answer that code cannot; listed per interface as TBD.
STANDARD_GAPS = {
    "http": ["Authentication / authorization", "Rate limits and timeouts", "API versioning policy"],
    "cli": ["Exit codes", "Required runtime environment"],
    "env": ["Value formats and units", "Which variables are secrets"],
    "message": ["Message schema versioning", "Delivery guarantees and retry / dead-letter handling"],
    "ui": ["Authentication / access control", "Supported browsers / clients"],
}


def anchor(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _first_paragraph(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    para = doc.strip().split("\n\n", 1)[0]
    return " ".join(ln.strip() for ln in para.splitlines())


def _constraints_text(f: Field, skip: Iterable[str] = ("in", "kind", "flags", "dest")) -> str:
    return ", ".join(f"{k}={v}" for k, v in f.constraints.items() if k not in skip)


class _Builder:
    def __init__(self, ir: RepoIR):
        self.ir = ir
        self.data_types: Dict[str, DataType] = {d.id: d for d in ir.data_types}
        self.referenced: List[str] = []
        self.gaps: List[Dict[str, str]] = []

    def gap(self, where: str, what: str, link: Optional[str] = None, key: Optional[str] = None) -> None:
        """`key` lets narrative or review close the gap: purpose:<iface>, op:<op>,
        param:<op>:<name>, type:<data type>, field:<data type>.<name>."""
        self.gaps.append({"where": where, "what": what, "anchor": link or "", "key": key or ""})

    def ref_type(self, type_id: Optional[str]) -> None:
        if type_id and type_id in self.data_types and type_id not in self.referenced:
            self.referenced.append(type_id)
            for f in self.data_types[type_id].fields:
                self.ref_type(f.type_id)

    def type_cell(self, display: Optional[str], type_id: Optional[str]) -> Dict[str, Any]:
        self.ref_type(type_id)
        return {
            "display": display,
            "anchor": anchor(f"dt-{type_id}") if type_id in self.data_types else None,
        }

    def type_ref(self, t: Optional[TypeRef]) -> Optional[Dict[str, Any]]:
        return self.type_cell(t.display, t.type_id) if t else None

    def param(self, f: Field, where: str, link: str, op_id: str = "") -> Dict[str, Any]:
        desc = f.docstring
        if not desc:
            self.gap(where, f"description of `{f.name}`", link, key=f"param:{op_id}:{f.name}")
        if not f.annotation:
            self.gap(where, f"type of `{f.name}`", link)
        return {
            "name": f.name,
            "flags": f.constraints.get("flags"),
            "location": f.constraints.get("in") or f.constraints.get("kind"),
            "type": self.type_cell(f.annotation, f.type_id),
            "required": f.required,
            "default": f.default,
            "constraints": _constraints_text(f),
            "description": desc,
            "description_generated": False,
        }

    def operation(self, iface: Interface, op: Operation) -> Dict[str, Any]:
        link = anchor(op.id)
        where = f"{iface.name} / {op.name}"
        summary = _first_paragraph(op.docstring)
        if not summary:
            self.gap(where, "operation description", link, key=f"op:{op.id}")
        out = {
            "id": op.id,
            "anchor": link,
            "name": op.name,
            "handler": op.handler,
            "summary": summary,
            "summary_generated": False,
            "params": [self.param(p, where, link, op.id) for p in op.params],
            "outputs": [{"name": f.name, "type": self.type_cell(f.annotation, f.type_id)} for f in op.outputs],
            "request": self.type_ref(op.request),
            "response": self.type_ref(op.response),
            "errors": op.errors,
            "meta": dict(op.meta),
        }
        if iface.kind == "http" and op.response is None:
            self.gap(where, "response type", link)
        if iface.kind == "env" and "default" not in op.meta and "required_in" not in op.meta:
            self.gap(where, "default value", link)
        return out

    def interface(self, iface: Interface) -> Dict[str, Any]:
        link = anchor(iface.id)
        self.gap(iface.name, "interface purpose", link, key=f"purpose:{iface.id}")
        for g in STANDARD_GAPS.get(iface.kind, []):
            self.gap(iface.name, g, link)
        return {
            "id": iface.id,
            "anchor": link,
            "name": iface.name,
            "kind": iface.kind,
            "kind_label": KIND_LABELS.get(iface.kind, iface.kind),
            "direction": iface.direction,
            "owner": iface.owner,
            "protocol": iface.protocol,
            "address": iface.address,
            "purpose": None,               # narrative: apply_narrative()
            "purpose_generated": False,
            "operations": [self.operation(iface, op) for op in iface.operations],
            "source_refs": iface.source_refs,
        }

    def data_type(self, d: DataType) -> Dict[str, Any]:
        link = anchor(f"dt-{d.id}")
        if not d.docstring:
            self.gap(d.name, "data type description", link, key=f"type:{d.id}")
        fields = []
        for f in d.fields:
            if d.source != "enum" and not f.docstring:
                self.gap(d.name, f"description of field `{f.name}`", link, key=f"field:{d.id}.{f.name}")
            fields.append({
                "name": f.name,
                "type": {"display": f.annotation,
                         "anchor": anchor(f"dt-{f.type_id}") if f.type_id in self.data_types else None},
                "required": f.required,
                "default": f.default,
                "constraints": _constraints_text(f),
                "description": f.docstring,
            })
        return {
            "id": d.id, "anchor": link, "name": d.name, "source": d.source,
            "description": _first_paragraph(d.docstring), "fields": fields,
        }


def build_icd_context(
    ir: RepoIR,
    interfaces: Optional[List[Interface]] = None,
    title: Optional[str] = None,
    diagrams: Optional[List[Dict[str, str]]] = None,
    generated: Optional[str] = None,
) -> Dict[str, Any]:
    """Context for one ICD document covering `interfaces` (default: all)."""
    from ... import __version__

    b = _Builder(ir)
    selected = ir.interfaces if interfaces is None else interfaces
    ifaces = [b.interface(i) for i in selected]
    dts = [b.data_type(b.data_types[t]) for t in b.referenced]
    provided = sum(1 for i in selected if i.direction == "provided")
    return {
        "title": title or f"{ir.repo} Interface Control Document",
        "repo": ir.repo,
        "version": __version__,
        "generated": generated,
        "summary": {
            "interfaces": len(selected),
            "provided": provided,
            "required": len(selected) - provided,
            "operations": sum(len(i.operations) for i in selected),
            "data_types": len(dts),
        },
        "interfaces": ifaces,
        "data_types": dts,
        "gaps": b.gaps,
        "generated_items": 0,
        "narrative_aliased": False,
        "coverage": None,                  # set by archdoc.coverage.icd_coverage after narrative
        "reviewed_items": 0,               # set by archdoc.review.apply_to_icd
        "diagrams": diagrams or [],
        "system_overview": None,           # the design description's system summary, when it ran first
    }


def apply_narrative(context: Dict[str, Any], interface_id: str, narrative: Dict[str, Any]) -> int:
    """Fill TBD slots of one interface from a restored narrative plan.

    Only empty slots are filled (code docstrings always win); filled values are
    flagged *_generated and their open items removed. Returns slots filled.
    """
    iface = next((i for i in context["interfaces"] if i["id"] == interface_id), None)
    if iface is None:
        return 0
    closed: Set[str] = set()
    purpose = (narrative.get("purpose") or "").strip()
    if purpose and not iface["purpose"] and not iface.get("purpose_rejected"):
        iface["purpose"], iface["purpose_generated"] = purpose, True
        closed.add(f"purpose:{interface_id}")
    ops = {o["id"]: o for o in iface["operations"]}
    for item in narrative.get("operations", []):
        op = ops.get(item.get("ref"))
        if op is None:
            continue
        summary = (item.get("summary") or "").strip()
        if summary and not op["summary"] and not op.get("summary_rejected"):
            op["summary"], op["summary_generated"] = summary, True
            closed.add(f"op:{op['id']}")
        params = {p["name"]: p for p in op["params"]}
        for pn in item.get("params", []):
            p = params.get(pn.get("name"))
            desc = (pn.get("description") or "").strip()
            if p is not None and desc and not p["description"] and not p.get("description_rejected"):
                p["description"], p["description_generated"] = desc, True
                closed.add(f"param:{op['id']}:{p['name']}")
    context["gaps"] = [g for g in context["gaps"] if not g.get("key") or g["key"] not in closed]
    context["generated_items"] = context.get("generated_items", 0) + len(closed)
    return len(closed)


def inventory_snapshot(ir: RepoIR) -> Dict[str, Any]:
    """Interface + data type inventory, saved alongside ICDs for the Phase 6 drift check."""
    return {
        "repo": ir.repo,
        "interfaces": [asdict(i) for i in ir.interfaces],
        "data_types": [asdict(d) for d in ir.data_types],
    }
