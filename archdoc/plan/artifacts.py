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
from .verify import prune_sequence, verify_sequence
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


def dedupe_participants(plan: Any) -> None:
    """A participant listed twice is a formatting slip, not a factual error: keep the first."""
    seen, kept = set(), []
    for p in plan["participants"]:
        if p["ref"] not in seen:
            seen.add(p["ref"])
            kept.append(p)
    plan["participants"] = kept


def check_sequence(plan: Any) -> List[str]:
    """Structural checks; facts are checked by verify.verify_sequence."""
    errors: List[str] = []
    dedupe_participants(plan)
    # A message may use a participant the plan forgot to declare: the schema
    # already limited it to refs from the slice, so the renderer declares it.
    n = len(plan["messages"])
    for i, g in enumerate(plan["groups"]):
        if not (0 <= g["first"] <= g["last"] < n):
            errors.append(f"groups[{i}] range {g['first']}..{g['last']} is outside messages 0..{n - 1}")
    return errors


def fix_reply_flags(plan: Any) -> None:
    """A message that is a call is never a return arrow."""
    for m in plan["messages"]:
        if m.get("call") and m.get("reply"):
            m["reply"] = False


def drop_unused_participants(plan: Any) -> List[str]:
    used = {r for m in plan["messages"] for r in (m["from"], m["to"])}
    unused = [p["ref"] for p in plan["participants"] if p["ref"] not in used]
    if not unused:
        return []
    plan["participants"] = [p for p in plan["participants"] if p["ref"] in used]
    return [f"dropped {len(unused)} participant(s) with no messages"]


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
    """Plan, verify against the call graph, repair once, then prune what still fails."""
    ids = call_slice(ctx.ir, entry.handler, ctx.depth, ctx.max_nodes)
    payload = build_sequence_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, entry.handler, ids, entry.via)
    schema = sequence_schema(payload.allowed["participants"], payload.allowed["functions"])

    def validate(plan: Any) -> List[str]:
        structural = check_sequence(plan)
        if structural:
            return structural
        fix_reply_flags(plan)
        return [str(x) for x in verify_sequence(ctx.ir, plan, ids, entry.handler)]

    result = run_plan(ctx.backend, payload, schema, prompts.SEQUENCE_SYSTEM, prompts.SEQUENCE_INSTRUCTIONS,
                      ctx.aliases, validate=validate, log=ctx.log)
    passed_unaided = result.ok          # every check passed without pruning (maybe after a repair)
    if not result.ok and result.candidate is not None and not check_sequence(result.candidate):
        # still failing verification after repair: keep what verifies, drop the rest
        result.plan, result.raw, result.errors = result.candidate, result.candidate_raw, []
    if not result.ok:
        return result

    fix_reply_flags(result.plan)
    result.warnings += keep_grounded_labels(result)          # indexes still match the raw answer
    removed, plan_issues = prune_sequence(ctx.ir, result.plan, ids, entry.handler)
    result.warnings += removed + [f"unresolved: {x}" for x in plan_issues]
    if not any(m.get("call") for m in result.plan["messages"]):
        result.plan, result.errors = None, ["no message survived verification against the call graph"]
        return result
    result.warnings += drop_unused_participants(result.plan)
    result.report = {
        "verified_first_time": passed_unaided and result.attempts == 1,
        "verified_after_repair": passed_unaided and result.attempts > 1,
        "attempts": result.attempts,
        "messages": len(result.plan["messages"]),
        "removed": removed,
        "slice": ids,
    }
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


__all__ = ["CLIENT", "PlanContext", "check_narrative", "check_sequence", "drop_unused_participants",
           "fix_reply_flags", "keep_grounded_labels",
           "plan_narrative", "plan_sequence"]
