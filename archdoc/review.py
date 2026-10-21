"""
Review overlay (DESIGN.md 2.6): human decisions that regeneration respects.

`<out>/review/review.yaml` holds one entry per reviewable item, keyed by the
same ids as the provenance sidecars:

  purpose:<interface>   op:<operation>   param:<operation>:<name>
  type:<data type>      field:<data type>.<name>
  sequence:<handler function>

Each run adds new items as `pending` (with the current text, so a reviewer
sees what they are judging) and refreshes pending ones. The reviewer changes
`status`:

  approved   keep this text; the model is not asked again
  edited     use `text` instead (also how a TBD gets filled by hand)
  rejected   leave it TBD; the model is not asked again

Entries that are not pending are never changed by the tool. Each records
`evidence`, a hash of the code facts it describes (the operation's signature,
the field's type...). When the code changes, the hash no longer matches: the
reviewed text is still used, never silently replaced, but it is marked stale
in the document and listed as an open item until someone reviews it again.

Sequence diagrams are reviewed whole: an approved or edited diagram is
rendered from its plan in `review/plans/` (which the reviewer may edit), with
no model call, even when no backend is configured.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .ir import RepoIR

STATUSES = ("pending", "approved", "edited", "rejected")
DECIDED = ("approved", "edited", "rejected")

HEADER = """\
# archdoc review overlay. Change `status` on any entry:
#   approved  keep `text` as is (the model is not asked again)
#   edited    use your `text` instead; also fills a TBD by hand
#   rejected  leave it TBD (the model is not asked again)
# Entries that are not pending are never changed by archdoc. `evidence` is a
# hash of the code the entry describes: if the code changes, your text is kept
# but marked stale in the documents until you review it again (set `evidence`
# to the value in `current_evidence`, or change the status back to pending).
"""


def evidence_hash(*facts: Any) -> str:
    blob = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Review:
    def __init__(self, out_dir: str):
        self.dir = os.path.join(out_dir, "review")
        self.path = os.path.join(self.dir, "review.yaml")
        self.items: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for k, v in (data.get("items") or {}).items():
                if isinstance(v, dict) and v.get("status") in STATUSES:
                    self.items[str(k)] = v
        self.stale: List[str] = []

    def decided(self, item_id: str) -> Optional[Dict[str, Any]]:
        e = self.items.get(item_id)
        return e if e and e.get("status") in DECIDED else None

    def is_stale(self, item_id: str, current: str) -> bool:
        e = self.items[item_id]
        if e.get("evidence") == current:
            e.pop("current_evidence", None)
            return False
        e["current_evidence"] = current
        if item_id not in self.stale:
            self.stale.append(item_id)
        return True

    def offer(self, item_id: str, text: str, evidence: str, origin: str, where: str = "") -> None:
        """Add or refresh a pending entry; decided entries are left alone."""
        e = self.items.get(item_id)
        if e and e.get("status") in DECIDED:
            return
        self.items[item_id] = {"status": "pending", "origin": origin, "where": where,
                               "text": text or "", "evidence": evidence}

    def save(self) -> str:
        os.makedirs(self.dir, exist_ok=True)
        body = yaml.safe_dump({"items": dict(sorted(self.items.items()))}, sort_keys=False,
                              allow_unicode=True, width=100)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(HEADER + body)
        return self.path

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.items.values():
            out[e["status"]] = out.get(e["status"], 0) + 1
        if self.stale:
            out["stale"] = len(self.stale)
        return out

    # -- sequence plans ------------------------------------------------------------

    def plan_path(self, handler: str) -> str:
        slug = "".join(c if c.isalnum() else "_" for c in handler).strip("_")
        return os.path.join(self.dir, "plans", f"sequence_{slug}.json")

    def load_plan(self, handler: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self.plan_path(handler), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def offer_plan(self, handler: str, plan: Dict[str, Any], evidence: str) -> None:
        item_id = f"sequence:{handler}"
        if self.decided(item_id):
            return
        path = self.plan_path(handler)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=1, ensure_ascii=False)
        self.offer(item_id, plan.get("title", ""), evidence, "generated",
                   where=os.path.relpath(path, os.path.dirname(self.dir)))


# -- evidence: the code facts each reviewable item describes -------------------------

def _fn_signature(ir: RepoIR, fid: Optional[str]) -> Any:
    fn = ir.find_function(fid) if fid else None
    if fn is None:
        return None
    return {"params": [asdict(p) for p in fn.params], "returns": fn.returns, "doc": fn.docstring,
            "raises": fn.raises}


def icd_evidence(ir: RepoIR) -> Dict[str, Tuple[str, str]]:
    """item id -> (evidence hash, human location) for every ICD slot."""
    out: Dict[str, Tuple[str, str]] = {}
    for iface in ir.interfaces:
        ops = [{k: v for k, v in asdict(o).items() if k != "line"} for o in iface.operations]
        out[f"purpose:{iface.id}"] = (evidence_hash(iface.kind, iface.direction, iface.protocol, iface.address, ops),
                                      iface.name)
        for o in iface.operations:
            op = {k: v for k, v in asdict(o).items() if k != "line"}
            out[f"op:{o.id}"] = (evidence_hash(op, _fn_signature(ir, o.handler)), f"{iface.name} / {o.name}")
            for p in o.params:
                out[f"param:{o.id}:{p.name}"] = (evidence_hash(o.name, asdict(p)),
                                                 f"{iface.name} / {o.name} / {p.name}")
    for d in ir.data_types:
        out[f"type:{d.id}"] = (evidence_hash(asdict(d)), d.name)
        for f in d.fields:
            out[f"field:{d.id}.{f.name}"] = (evidence_hash(d.id, asdict(f)), f"{d.name}.{f.name}")
    return out


def sequence_evidence(ir: RepoIR, slice_ids: List[str]) -> str:
    facts = []
    for fid in slice_ids:
        fn = ir.find_function(fid)
        if fn:
            facts.append([fid, _fn_signature(ir, fid), fn.calls])
    return evidence_hash(facts)


# -- applying to an ICD context -----------------------------------------------------------

def _slots(context: Dict[str, Any]):
    """(item id, dict holding the slot, field name) for every describable ICD slot."""
    for i in context["interfaces"]:
        yield f"purpose:{i['id']}", i, "purpose"
        for o in i["operations"]:
            yield f"op:{o['id']}", o, "summary"
            for p in o["params"]:
                yield f"param:{o['id']}:{p['name']}", p, "description"
    for d in context["data_types"]:
        yield f"type:{d['id']}", d, "description"
        if d["source"] != "enum":
            for f in d["fields"]:
                yield f"field:{d['id']}.{f['name']}", f, "description"


def apply_to_icd(context: Dict[str, Any], review: Review, evidence: Dict[str, Tuple[str, str]]) -> int:
    """Apply decided entries before narrative, so the model never redoes them."""
    closed, stale_gaps, applied = set(), [], 0
    anchors = {g["key"]: (g["where"], g["anchor"]) for g in context["gaps"] if g.get("key")}
    for item_id, slot, fld in _slots(context):
        e = review.decided(item_id)
        if not e:
            continue
        current = evidence.get(item_id, ("", ""))[0]
        stale = review.is_stale(item_id, current)
        applied += 1
        if e["status"] == "rejected":
            slot[fld], slot[f"{fld}_generated"], slot[f"{fld}_rejected"] = None, False, True
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        slot[fld], slot[f"{fld}_generated"] = text, False
        slot[f"{fld}_review"] = "stale" if stale else "reviewed"
        closed.add(item_id)
        if stale:
            where, link = anchors.get(item_id, (evidence.get(item_id, ("", item_id))[1], ""))
            stale_gaps.append({"where": where, "what": "reviewed text predates a code change: re-review",
                               "anchor": link, "key": f"stale:{item_id}"})
    context["gaps"] = [g for g in context["gaps"] if not g.get("key") or g["key"] not in closed] + stale_gaps
    context["reviewed_items"] = sum(1 for _, s, f in _slots(context) if s.get(f"{f}_review"))
    return applied


def offer_icd(context: Dict[str, Any], review: Review, evidence: Dict[str, Tuple[str, str]]) -> None:
    """After narrative: offer every generated or still-open slot for review."""
    for item_id, slot, fld in _slots(context):
        if slot.get(f"{fld}_review") or slot.get(f"{fld}_rejected"):
            continue
        h, where = evidence.get(item_id, ("", ""))
        if slot.get(f"{fld}_generated"):
            review.offer(item_id, slot[fld], h, "generated", where)
        elif not slot.get(fld):
            review.offer(item_id, "", h, "open", where)
