"""Planner entry points per artifact: build the payload, schema and checks, run the loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from ..backends import LLMBackend
from ..egress import EgressLog, build_interface_payload, build_sequence_payload
from ..interfaces import Interface
from ..ir import RepoIR
from ..policy import Config
from ..pseudonym import ALIAS_RE, AliasMap
from . import prompts
from .entries import Entry, call_slice
from .planner import PlanResult, run_plan
from .schemas import CLIENT, narrative_schema, sequence_schema


@dataclass
class PlanContext:
    ir: RepoIR
    config: Config
    tier: str
    aliases: AliasMap
    backend: LLMBackend
    log: Optional[EgressLog] = None
    depth: int = 3
    max_nodes: int = 25
    alias_path: Optional[str] = None     # where the alias map is persisted after the run


def check_sequence(plan: Any) -> List[str]:
    errors: List[str] = []
    declared = [p["ref"] for p in plan["participants"]]
    dupes = sorted({r for r in declared if declared.count(r) > 1})
    if dupes:
        errors.append(f"participants listed more than once: {', '.join(dupes)}")
    # A message may use a participant the plan forgot to declare: the schema
    # already limited it to refs from the slice, so the renderer declares it.
    n = len(plan["messages"])
    for i, g in enumerate(plan["groups"]):
        if not (0 <= g["first"] <= g["last"] < n):
            errors.append(f"groups[{i}] range {g['first']}..{g['last']} is outside messages 0..{n - 1}")
    return errors


def normalize_sequence(ir: RepoIR, plan: Any, slice_ids: List[str]) -> List[str]:
    """Remove what a static call graph cannot support. Returns warnings.

    - a message that is a call is never a return arrow
    - groups need evidence: alt/opt only if a function in the slice raises;
      loops never (no control-flow data at this level)
    - declared participants that take part in no message are dropped
    """
    warnings: List[str] = []
    for m in plan["messages"]:
        if m.get("call") and m.get("reply"):
            m["reply"] = False
    raises = any(fn.raises for fn in (ir.find_function(f) for f in slice_ids) if fn)
    kept = []
    for g in plan["groups"]:
        if g["kind"] == "loop" or (g["kind"] in ("alt", "opt") and not raises):
            warnings.append(f"dropped {g['kind']} {g['label']!r}: no evidence in the call graph")
        else:
            kept.append(g)
    plan["groups"] = kept
    used = {r for m in plan["messages"] for r in (m["from"], m["to"])}
    unused = [p["ref"] for p in plan["participants"] if p["ref"] not in used]
    if unused:
        plan["participants"] = [p for p in plan["participants"] if p["ref"] in used]
        warnings.append(f"dropped {len(unused)} participant(s) with no messages")
    return warnings


def keep_grounded_labels(result: PlanResult) -> List[str]:
    """When the model saw aliases, keep only text it wrote *in terms of* those
    aliases (restored to real names by ingress). Text with no alias in it was
    invented: the model could not know what the code is about, so there is
    nothing to map back. Cleared labels fall back to the called function's name.
    """
    if not result.payload_aliased:
        return []
    plan, raw = result.plan, result.raw
    cleared = 0
    for m, rm in zip(plan["messages"], raw["messages"]):
        if m["label"] and not ALIAS_RE.search(rm["label"]):
            m["label"] = ""
            cleared += 1
    for g, rg in zip(plan["groups"], raw["groups"]):
        if g["label"] and not ALIAS_RE.search(rg["label"]):
            g["label"] = ""
            cleared += 1
    if plan["title"] and not ALIAS_RE.search(raw["title"]):
        plan["title"] = ""
        cleared += 1
    return [f"cleared {cleared} label(s) written without any reference (not reversible)"] if cleared else []


def plan_sequence(ctx: PlanContext, entry: Entry) -> PlanResult:
    ids = call_slice(ctx.ir, entry.handler, ctx.depth, ctx.max_nodes)
    payload = build_sequence_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, entry.handler, ids, entry.via)
    schema = sequence_schema(payload.allowed["participants"], payload.allowed["functions"])
    result = run_plan(ctx.backend, payload, schema, prompts.SEQUENCE_SYSTEM, prompts.SEQUENCE_INSTRUCTIONS,
                      ctx.aliases, validate=check_sequence, log=ctx.log)
    if result.ok:
        result.warnings += keep_grounded_labels(result)
        result.warnings += normalize_sequence(ctx.ir, result.plan, ids)
    return result


def check_narrative(plan: Any) -> List[str]:
    refs = [o["ref"] for o in plan["operations"]]
    dupes = sorted({r for r in refs if refs.count(r) > 1})
    return [f"operations described more than once: {', '.join(dupes)}"] if dupes else []


def plan_narrative(ctx: PlanContext, iface: Interface) -> PlanResult:
    payload = build_interface_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, iface)
    schema = narrative_schema(payload.allowed["operations"], payload.allowed["params"])
    return run_plan(ctx.backend, payload, schema, prompts.NARRATIVE_SYSTEM, prompts.NARRATIVE_INSTRUCTIONS,
                    ctx.aliases, validate=check_narrative, log=ctx.log)


__all__ = ["CLIENT", "PlanContext", "check_narrative", "check_sequence", "keep_grounded_labels",
           "normalize_sequence",
           "plan_narrative", "plan_sequence"]
