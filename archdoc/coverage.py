"""
Coverage: what the documentation covers, and what it is known to miss.

The most dangerous inaccuracy is an interface that is missing without anything
saying so. `known_unknowns` flags every module that uses a library implying an
interface (a message broker, a web framework, an HTTP client, a database...)
where no interface of that kind was extracted from the module. Those become
explicit open items instead of silent omissions (DESIGN.md 2.4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

from .ir import EXTERNAL_PREFIX, RepoIR
from .deployment import deployment_issues
from .messaging import wiring_issues

# import root -> (interface kind, direction, human description)
INTERFACE_LIBRARIES: Dict[str, tuple] = {
    # messaging
    "pika": ("message", "both", "RabbitMQ (AMQP)"),
    "aio_pika": ("message", "both", "RabbitMQ (AMQP)"),
    "kombu": ("message", "both", "message broker (kombu)"),
    "celery": ("message", "both", "Celery tasks"),
    "kafka": ("message", "both", "Kafka"),
    "confluent_kafka": ("message", "both", "Kafka"),
    "aiokafka": ("message", "both", "Kafka"),
    "nats": ("message", "both", "NATS"),
    "paho": ("message", "both", "MQTT"),
    "zmq": ("message", "both", "ZeroMQ"),
    "google.cloud.pubsub": ("message", "both", "Google Pub/Sub"),
    # provided HTTP / UI
    "fastapi": ("http", "provided", "FastAPI"),
    "flask": ("http", "provided", "Flask"),
    "django": ("http", "provided", "Django"),
    "starlette": ("http", "provided", "Starlette"),
    "tornado": ("http", "provided", "Tornado"),
    "sanic": ("http", "provided", "Sanic"),
    "bottle": ("http", "provided", "Bottle"),
    "gradio": ("ui", "provided", "Gradio web UI"),
    "streamlit": ("ui", "provided", "Streamlit web UI"),
    "dash": ("ui", "provided", "Dash web UI"),
    # RPC
    "grpc": ("rpc", "both", "gRPC"),
    # outbound HTTP
    "requests": ("http", "required", "outbound HTTP (requests)"),
    "httpx": ("http", "required", "outbound HTTP (httpx)"),
    "aiohttp": ("http", "both", "HTTP (aiohttp)"),
    "urllib3": ("http", "required", "outbound HTTP (urllib3)"),
    "urllib.request": ("http", "required", "outbound HTTP (urllib.request)"),
    "http.client": ("http", "required", "outbound HTTP (http.client)"),
    # data stores and cloud services
    "sqlalchemy": ("db", "required", "SQL database (SQLAlchemy)"),
    "psycopg2": ("db", "required", "PostgreSQL"),
    "psycopg": ("db", "required", "PostgreSQL"),
    "pymysql": ("db", "required", "MySQL"),
    "pymongo": ("db", "required", "MongoDB"),
    "motor": ("db", "required", "MongoDB"),
    "redis": ("db", "required", "Redis"),
    "sqlite3": ("db", "required", "SQLite"),
    "boto3": ("cloud", "required", "AWS"),
    "google.cloud": ("cloud", "required", "Google Cloud"),
    "azure": ("cloud", "required", "Azure"),
    # CLI
    "argparse": ("cli", "provided", "command line (argparse)"),
    "click": ("cli", "provided", "command line (Click)"),
    "typer": ("cli", "provided", "command line (Typer)"),
}


@dataclass
class KnownUnknown:
    module: str
    path: str
    library: str
    kind: str
    direction: str
    description: str

    @property
    def text(self) -> str:
        return (f"uses {self.description} (`{self.library}`), but no {self.kind} interface was "
                f"extracted from this module: not documented")


def _library_of(import_id: str) -> Optional[str]:
    name = import_id[len(EXTERNAL_PREFIX):]
    for lib in sorted(INTERFACE_LIBRARIES, key=len, reverse=True):     # google.cloud.pubsub before google.cloud
        if name == lib or name.startswith(lib + "."):
            return lib
    return None


def _extracted_kinds(ir: RepoIR) -> Dict[str, Set[str]]:
    """module id -> interface kinds attributed to it (owner, or env 'used_in')."""
    out: Dict[str, Set[str]] = {}
    for iface in ir.interfaces:
        out.setdefault(iface.owner, set()).add(iface.kind)
        for op in iface.operations:
            for m in op.meta.get("used_in", "").split(","):
                if m.strip():
                    out.setdefault(m.strip(), set()).add(iface.kind)
    return out


def known_unknowns(ir: RepoIR) -> List[KnownUnknown]:
    extracted = _extracted_kinds(ir)
    found: List[KnownUnknown] = []
    for m in ir.modules:
        seen_libs: Set[str] = set()
        for imp in m.imports:
            if not imp.startswith(EXTERNAL_PREFIX):
                continue
            lib = _library_of(imp)
            if lib is None or lib in seen_libs:
                continue
            seen_libs.add(lib)
            kind, direction, desc = INTERFACE_LIBRARIES[lib]
            if kind in extracted.get(m.id, set()):
                continue
            found.append(KnownUnknown(m.id, m.path, lib, kind, direction, desc))
    return found


def _count(items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    fact = sum(1 for i in items if i.get(field) and not i.get(f"{field}_generated"))
    gen = sum(1 for i in items if i.get(field) and i.get(f"{field}_generated"))
    return {"total": len(items), "from_code": fact, "generated": gen, "tbd": len(items) - fact - gen}


def icd_coverage(context: Dict[str, Any], ir: RepoIR) -> Dict[str, Any]:
    """Coverage of one ICD context (after any narrative was applied)."""
    ifaces = context["interfaces"]
    ops = [o for i in ifaces for o in i["operations"]]
    params = [p for o in ops for p in o["params"]]
    fields = [f for d in context["data_types"] for f in d["fields"] if d["source"] != "enum"]
    kinds: Dict[str, int] = {}
    for i in ifaces:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    unknowns = known_unknowns(ir)
    return {
        "interfaces": {"total": len(ifaces), "by_kind": kinds, **{
            k: v for k, v in _count(ifaces, "purpose").items() if k != "total"}},
        "operations": _count(ops, "summary"),
        "parameters": _count(params, "description"),
        "data_fields": _count([{**f, "description_generated": False} for f in fields], "description"),
        "open_items": len(context["gaps"]),
        "known_unknowns": [asdict(u) | {"text": u.text} for u in unknowns],
        "message_wiring": wiring_issues(ir),
        "deployment": deployment_issues(ir),
    }


def coverage_lines(cov: Dict[str, Any]) -> List[str]:
    """Short console summary."""
    def pct(c):
        return f"{c['total'] - c['tbd']}/{c['total']}" if c["total"] else "0/0"
    lines = [
        f"coverage: operations described {pct(cov['operations'])}, parameters {pct(cov['parameters'])}, "
        f"data fields {pct(cov['data_fields'])}; {cov['open_items']} open items",
    ]
    if cov["known_unknowns"]:
        lines.append(f"  [GAP] {len(cov['known_unknowns'])} interface technolog(ies) used but not extracted:")
        for u in cov["known_unknowns"][:12]:
            lib = "" if u["library"] in u["description"] else f" ({u['library']})"
            lines.append(f"        {u['module']}: {u['description']}{lib}")
        if len(cov["known_unknowns"]) > 12:
            lines.append(f"        ... and {len(cov['known_unknowns']) - 12} more (see coverage.json)")
    wiring = cov.get("message_wiring") or []
    if wiring:
        lines.append(f"  [GAP] {len(wiring)} message channel(s) with a missing end:")
        for w in wiring[:12]:
            lines.append(f"        {w['channel']}: {w['issue']} ({w['modules']})")
        if len(wiring) > 12:
            lines.append(f"        ... and {len(wiring) - 12} more (see coverage.json)")
    dep = cov.get("deployment") or []
    if dep:
        lines.append(f"  [GAP] {len(dep)} disagreement(s) between deployment files and code:")
        for d in dep[:12]:
            lines.append(f"        {d['file']}: {d['where']}: {d['issue']}")
        if len(dep) > 12:
            lines.append(f"        ... and {len(dep) - 12} more (see coverage.json)")
    return lines
