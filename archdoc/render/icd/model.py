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

    def gap(self, where: str, what: str, link: Optional[str] = None) -> None:
        self.gaps.append({"where": where, "what": what, "anchor": link or ""})

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

    def param(self, f: Field, where: str, link: str) -> Dict[str, Any]:
        desc = f.docstring
        if not desc:
            self.gap(where, f"description of `{f.name}`", link)
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
        }

    def operation(self, iface: Interface, op: Operation) -> Dict[str, Any]:
        link = anchor(op.id)
        where = f"{iface.name} / {op.name}"
        summary = _first_paragraph(op.docstring)
        if not summary:
            self.gap(where, "operation description", link)
        out = {
            "id": op.id,
            "anchor": link,
            "name": op.name,
            "handler": op.handler,
            "summary": summary,
            "semantics": None,             # narrative (Phase 3)
            "params": [self.param(p, where, link) for p in op.params],
            "request": self.type_ref(op.request),
            "response": self.type_ref(op.response),
            "errors": op.errors,
            "meta": dict(op.meta),
        }
        if iface.kind == "http" and op.response is None:
            self.gap(where, "response type", link)
        if iface.kind == "env" and "default" not in op.meta:
            self.gap(where, "default value (none in code: required?)", link)
        return out

    def interface(self, iface: Interface) -> Dict[str, Any]:
        link = anchor(iface.id)
        self.gap(iface.name, "interface purpose", link)
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
            "purpose": None,               # narrative (Phase 3)
            "operations": [self.operation(iface, op) for op in iface.operations],
            "source_refs": iface.source_refs,
        }

    def data_type(self, d: DataType) -> Dict[str, Any]:
        link = anchor(f"dt-{d.id}")
        if not d.docstring:
            self.gap(d.name, "data type description", link)
        fields = []
        missing = []
        for f in d.fields:
            if d.source != "enum" and not f.docstring:
                missing.append(f.name)
            fields.append({
                "name": f.name,
                "type": {"display": f.annotation,
                         "anchor": anchor(f"dt-{f.type_id}") if f.type_id in self.data_types else None},
                "required": f.required,
                "default": f.default,
                "constraints": _constraints_text(f),
                "description": f.docstring,
            })
        if missing:
            self.gap(d.name, f"field descriptions ({', '.join(f'`{m}`' for m in missing)})", link)
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
        "diagrams": diagrams or [],
    }


def inventory_snapshot(ir: RepoIR) -> Dict[str, Any]:
    """Interface + data type inventory, saved alongside ICDs for the Phase 6 drift check."""
    return {
        "repo": ir.repo,
        "interfaces": [asdict(i) for i in ir.interfaces],
        "data_types": [asdict(d) for d in ir.data_types],
    }
