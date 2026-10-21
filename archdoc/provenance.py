"""
Provenance sidecars (DESIGN.md 2.1): `<artifact>.provenance.json` beside every
diagram and ICD.

Every item is one of three classes:

  fact     extracted from the code; `evidence` names the IR elements
  derived  written by a model; `request` points at the record of what the
           model was shown (real ids, detail level, aliased or not), which
           model answered, prompt version and payload hash
  open     the code cannot answer it; rendered as TBD

Sequence messages are model choices checked against the call graph, so they
are `derived` with `verified: true`; their labels are `derived` when the
model wrote them and `fact` when they fall back to the function's name.

The header records the tool version, the analysed repo's git commit (and
whether the work tree had changes) and a hash of the IR, so a reader can tell
whether a document still describes the code in front of them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from typing import Any, Dict, List, Optional

from . import __version__
from .egress import participant_of
from .ir import RepoIR

SCHEMA = "archdoc.provenance/1"


def _git(repo: str, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def ir_sha256(ir: RepoIR) -> str:
    return hashlib.sha256(json.dumps(ir.to_dict(), sort_keys=True).encode()).hexdigest()


def run_header(repo_path: str, ir: RepoIR, dated: bool = True) -> Dict[str, Any]:
    commit = _git(repo_path, "rev-parse", "HEAD")
    status = _git(repo_path, "status", "--porcelain") if commit else None
    return {
        "schema": SCHEMA,
        "tool": f"archdoc {__version__}",
        "repo": ir.repo,
        "git_commit": commit,
        "git_dirty": bool(status) if commit else None,
        "ir_sha256": ir_sha256(ir),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds") if dated else None,
    }


def llm_header(ctx) -> Dict[str, Any]:
    pol = ctx.config.policy_for(ctx.tier)
    return {
        "backend": ctx.backend.name,
        "model": ctx.backend.model,
        "model_digest": getattr(ctx.backend, "digest", None),
        "url": ctx.backend.url,
        "tier": ctx.tier,
        "detail": {"level": pol.level, "pseudonymize": pol.pseudonymize, "docstrings": pol.docstrings},
    }


def request_record(result, ctx=None) -> Dict[str, Any]:
    """What one model request was based on and what came of it."""
    rec = {
        "artifact": result.artifact,
        "backend": result.backend,
        "payload_sha256": result.payload_sha256,
        "prompt_sha256": result.prompt_sha256,
        "aliased": result.payload_aliased,
        "shown": result.shown,
        "attempts": result.attempts,
        "ok": result.ok,
        "prompt_tokens": result.prompt_tokens or None,
        "completion_tokens": result.completion_tokens or None,
        "warnings": result.warnings,
        "errors": result.errors,
    }
    if ctx is not None:
        rec["tier"] = ctx.tier
        rec["detail"] = llm_header(ctx)["detail"]
    return rec


def sidecar(header: Dict[str, Any], artifact: str, producer: str, items: List[Dict[str, Any]],
            requests: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[it["class"]] = counts.get(it["class"], 0) + 1
    out = {**header, "artifact": artifact, "producer": producer, "counts": counts, **extra, "items": items}
    if requests:
        out["requests"] = requests
    return out


def write(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


# -- per artifact -----------------------------------------------------------------

def deterministic_items(what: str, evidence: List[str]) -> List[Dict[str, Any]]:
    return [{"id": what, "class": "fact", "evidence": sorted(set(evidence))}]


def flow_items(flow) -> List[Dict[str, Any]]:
    return [{"id": f"step[{i}]", "class": "fact", "what": st.kind, "text": st.text,
             "evidence": [st.evidence] if st.evidence else []}
            for i, st in enumerate(flow.steps) if st.kind in ("invoke", "publish", "deliver", "unconsumed")]


def sequence_items(ir: RepoIR, result, slice_ids: List[str]) -> List[Dict[str, Any]]:
    fns = {f: ir.find_function(f) for f in slice_ids}
    items = []
    for i, m in enumerate(result.plan["messages"]):
        call = m.get("call") or ""
        callers = sorted(f for f, fn in fns.items() if fn and call in fn.calls and participant_of(f) == m["from"])
        item = {"id": f"message[{i}]", "class": "derived", "verified": True, "from": m["from"], "to": m["to"],
                "call": call or None, "evidence": ([call] if call else []) + callers, "request": 0}
        if m.get("label"):
            item["label"] = {"class": "derived", "text": m["label"]}
        elif call:
            item["label"] = {"class": "fact", "text": call.rsplit(".", 1)[-1].rsplit(":", 1)[-1],
                             "how": "function name"}
        items.append(item)
    raising = sorted(f for f, fn in fns.items() if fn and fn.raises)
    for gi, g in enumerate(result.plan.get("groups", [])):
        inside = result.plan["messages"][g["first"]:g["last"] + 1]
        ev = sorted({m.get("call") for m in inside if m.get("call") in raising} |
                    {f for f in raising if participant_of(f) in {m["from"] for m in inside}})
        items.append({"id": f"group[{gi}]", "class": "derived", "verified": True, "kind": g["kind"],
                      "label": {"class": "derived" if g["label"] else "fact", "text": g["label"]},
                      "evidence": ev, "request": 0})
    title = result.plan.get("title") or ""
    items.append({"id": "title", "class": "derived" if title else "fact", "text": title,
                  "request": 0 if title else None})
    return items


def icd_items(context: Dict[str, Any], request_index: Dict[str, int]) -> List[Dict[str, Any]]:
    """Every describable slot of an ICD context, classed after narrative was applied."""
    items: List[Dict[str, Any]] = []

    def slot(item_id: str, value: Any, generated: bool, evidence: List[str], iface_id: str) -> None:
        if not value:
            cls = "open"
        elif generated:
            cls = "derived"
        else:
            cls = "fact"
        it: Dict[str, Any] = {"id": item_id, "class": cls, "evidence": [e for e in evidence if e]}
        if cls == "derived":
            it["request"] = request_index.get(iface_id)
        items.append(it)

    for i in context["interfaces"]:
        slot(f"purpose:{i['id']}", i.get("purpose"), i.get("purpose_generated"), [i["id"]], i["id"])
        for o in i["operations"]:
            slot(f"op:{o['id']}", o.get("summary"), o.get("summary_generated"), [o["id"], o.get("handler")], i["id"])
            for p in o["params"]:
                slot(f"param:{o['id']}:{p['name']}", p.get("description"), p.get("description_generated"),
                     [o["id"]], i["id"])
    for d in context["data_types"]:
        slot(f"type:{d['id']}", d.get("description"), False, [d["id"]], "")
        for f in d["fields"]:
            if d["source"] != "enum":
                slot(f"field:{d['id']}.{f['name']}", f.get("description"), False, [d["id"]], "")
    return items
