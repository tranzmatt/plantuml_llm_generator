"""Planner entry points per artifact: build the payload, schema and checks, run the loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _osa_distance(a: str, b: str) -> int:
    """Edit distance counting an adjacent transposition as one edit."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _names_in_slice(ir: RepoIR, ids: List[str]) -> List[str]:
    names = set()
    for fid in ids:
        mod, _, qual = fid.partition(":")
        names.update({mod, mod.rsplit(".", 1)[-1]})
        if qual:
            names.add(qual)
            names.update(qual.split("."))
    return sorted(n for n in names if len(n) >= 4)


def correct_misspelled_names(ir: RepoIR, plan: Any, ids: List[str]) -> List[str]:
    """A label word one or two edits away from a name in the slice is a copying
    slip (llama4:scout wrote "Dtaabase.query" and "maxn"): a misspelled real
    name reads as authoritative and is worse than none, so it is replaced with
    the exact name. Only used when the model saw real names."""
    known = _names_in_slice(ir, ids)
    lower = {n.lower() for n in known}
    fixed: Dict[str, str] = {}

    def fix(text: str) -> str:
        def repl(m: "re.Match[str]") -> str:
            w = m.group(0)
            if len(w) < 4 or w.lower() in lower or w.lower().rstrip("s") in lower:
                return w
            limit = 1 if len(w) <= 6 else 2
            best = min(known, key=lambda n: (_osa_distance(w, n), n), default=None)
            if best is not None and _osa_distance(w, best) <= limit:
                fixed[w] = best
                return best
            return w
        return _WORD_RE.sub(repl, text)

    for item in plan["messages"] + plan["groups"]:
        item["label"] = fix(item["label"])
    plan["title"] = fix(plan["title"])
    return [f"corrected misspelled name {w!r} -> {n!r}" for w, n in sorted(fixed.items())]


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

    system, instructions = prompts.sequence_prompts(bool(ALIAS_RE.search(payload.text)))
    result = run_plan(ctx.backend, payload, schema, system, instructions, ctx.aliases, validate=validate, log=ctx.log)
    passed_unaided = result.ok          # every check passed without pruning (maybe after a repair)
    if not result.ok and result.candidate is not None and not check_sequence(result.candidate):
        # still failing verification after repair: keep what verifies, drop the rest
        result.plan, result.raw, result.errors = result.candidate, result.candidate_raw, []
    if not result.ok:
        return result

    fix_reply_flags(result.plan)
    result.warnings += keep_grounded_labels(result)          # indexes still match the raw answer
    if not result.payload_aliased:
        result.warnings += correct_misspelled_names(ctx.ir, result.plan, ids)
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
    system, instructions = prompts.narrative_prompts(bool(ALIAS_RE.search(payload.text)))
    return run_plan(ctx.backend, payload, schema, system, instructions, ctx.aliases, validate=check_narrative, log=ctx.log)


__all__ = ["CLIENT", "PlanContext", "check_narrative", "check_sequence", "drop_unused_participants",
           "fix_reply_flags", "keep_grounded_labels",
           "plan_narrative", "plan_sequence"]
