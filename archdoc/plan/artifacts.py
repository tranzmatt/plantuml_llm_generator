"""Planner entry points per artifact: build the payload, schema and checks, run the loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..backends import LLMBackend
from ..cache import ResponseCache
from ..egress import (
    EgressLog, build_activity_payload, build_function_payload, build_group_payload, build_interface_payload, build_module_payload,
    build_repo_map, build_sequence_payload, build_tests_payload, build_usecase_payload,
)
from ..interfaces import Interface
from ..ir import RepoIR
from ..policy import Config
from ..pseudonym import ALIAS_RE, AliasMap
from .. import overlap as overlap_guard
from ..overlap import SourceIndex
from ..packing import (
    DEFAULT_WINDOW, Calibration, Packed, Window, allowed_levels, levels_summary, pack, shrink_steps,
)
from . import prompts
from .entries import Entry, call_slice
from .planner import PlanResult, run_plan
from .verify import prune_sequence, verify_sequence
from .schemas import (
    CLIENT, activity_schema, behavior_schema, group_summary_schema, module_summary_schema, narrative_schema, outline,
    sequence_schema, tests_schema, usecase_schema,
)


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
    overlap: Optional[SourceIndex] = None   # verbatim-overlap guard (None: not checked)
    # context fitting (archdoc.packing): repo root for outline/source, the window
    # (default: the backend's, else 8k), measured chars per token, the shared repo map
    root: Optional[str] = None
    window: Optional[Window] = None
    calibration: Optional[Calibration] = None
    repo_map: Optional[List[Dict[str, Any]]] = None
    map_share: float = 0.15
    cache: Optional[ResponseCache] = None   # model responses by exact request (archdoc.cache)


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


# -- context fitting -----------------------------------------------------------------------

MAX_TOKENS = 2048          # the answer budget passed to run_plan


def _window(ctx: PlanContext) -> Window:
    w = ctx.window or Window(tokens=getattr(ctx.backend, "window", None) or DEFAULT_WINDOW)
    cpt = ctx.calibration.get(ctx.backend.model) if ctx.calibration else w.chars_per_token
    return Window(w.tokens, MAX_TOKENS, cpt)


def _budget(ctx: PlanContext, system: str, instructions: str, schema: Dict[str, Any]) -> int:
    fixed = len(system) + len(outline(schema)) + 120 + len(instructions)
    return _window(ctx).input_chars(fixed)


def _repo_map(ctx: PlanContext) -> List[Dict[str, Any]]:
    """Built once per run, so every request carries the identical map (prefix caching)."""
    if ctx.repo_map is None:
        cap = int(_window(ctx).input_chars(0) * ctx.map_share)   # a share of what payloads can use
        ctx.repo_map = build_repo_map(ctx.ir, ctx.config, ctx.tier, ctx.aliases, cap)
    return ctx.repo_map


def _ask(ctx: PlanContext, payload, schema, system, instructions, validate) -> PlanResult:
    return run_plan(ctx.backend, payload, schema, system, instructions, ctx.aliases, validate=validate,
                    log=ctx.log, max_tokens=MAX_TOKENS, cache=ctx.cache)


def _calibrate(ctx: PlanContext, result: PlanResult) -> None:
    if ctx.calibration is not None:
        ctx.calibration.observe(ctx.backend.model, result.first_prompt_chars, result.first_prompt_tokens)


def _fit(packed: Packed, ctx: PlanContext) -> Dict[str, Any]:
    w = _window(ctx)
    return {"window_tokens": w.tokens, "chars_per_token": w.chars_per_token, "budget_chars": packed.budget_chars,
            "payload_chars": len(packed.payload.text), "fits": packed.fits,
            "levels": levels_summary(packed.levels)}


def plan_sequence(ctx: PlanContext, entry: Entry) -> PlanResult:
    """Pack the slice into the window, plan, verify against the call graph, repair
    once, then prune what still fails."""
    policy = ctx.config.policy_for(ctx.tier)
    repo_map = _repo_map(ctx)

    def build_for(slice_ids: List[str], with_map: bool = True):
        return lambda detail: build_sequence_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, entry.handler,
                                                     slice_ids, entry.via, detail=detail, root=ctx.root,
                                                     repo_map=repo_map if with_map else None)

    ids = call_slice(ctx.ir, entry.handler, ctx.depth, ctx.max_nodes)
    stub = build_for(ids)({})
    schema = sequence_schema(stub.allowed["participants"], stub.allowed["functions"])
    system, instructions = prompts.sequence_prompts(bool(ALIAS_RE.search(stub.text)))
    notes: List[str] = []

    def fit(budget: int):
        nonlocal ids
        packed = pack(build_for(ids), ids, allowed_levels(policy), budget)
        if not packed.fits and repo_map:
            trial = pack(build_for(ids, False), ids, allowed_levels(policy), budget)
            if trial.fits:
                notes.append("repo map left out to fit the window")
                return trial
        if not packed.fits:
            full = len(ids)
            for n in shrink_steps(full):
                smaller = call_slice(ctx.ir, entry.handler, ctx.depth, n)
                trial = pack(build_for(smaller, False), smaller, allowed_levels(policy), budget)
                if trial.fits:
                    ids, packed = smaller, trial
                    notes.append(f"slice reduced from {full} to {len(ids)} functions to fit the window")
                    break
            else:
                if full > 2:                      # send the smallest attempt, not the original
                    ids = call_slice(ctx.ir, entry.handler, ctx.depth, 2)
                    packed = pack(build_for(ids, False), ids, [], budget)
                notes.append(f"the slice does not fit the window even at {len(ids)} functions; sent as stubs")
        return packed

    def validate(plan: Any) -> List[str]:
        structural = check_sequence(plan)
        if structural:
            return structural
        fix_reply_flags(plan)
        return [str(x) for x in verify_sequence(ctx.ir, plan, ids, entry.handler)] + \
            overlap_guard.errors(ctx.overlap, overlap_guard.sequence_texts(plan))

    budget = _budget(ctx, system, instructions, schema)
    packed = fit(budget)
    result = _ask(ctx, packed.payload, schema, system, instructions, validate)
    if result.overflow:
        _calibrate(ctx, result)
        notes.append("the prompt overflowed the window; repacked smaller and retried")
        packed = fit(min(int(budget * 0.6), _budget(ctx, system, instructions, schema)))
        result = _ask(ctx, packed.payload, schema, system, instructions, validate)
    _calibrate(ctx, result)
    result.levels, result.fit = packed.levels, _fit(packed, ctx)
    result.warnings = notes + result.warnings
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
    result.warnings += overlap_guard.enforce(ctx.overlap, result.plan, overlap_guard.sequence_texts(result.plan))
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


def _plan_packed(ctx: PlanContext, build, ranked: List[str], schema_of, prompts_of, structural,
                 texts=overlap_guard.plan_texts, what: str = "the input") -> PlanResult:
    """Pack `ranked` items into the window, plan, repair once. `structural` checks
    the plan's shape; the overlap guard runs on `texts`. A plan that fails only
    the overlap check after repair is kept with the overlapping text cleared."""
    policy = ctx.config.policy_for(ctx.tier)
    stub = build({})
    schema = schema_of(stub.allowed)
    system, instructions = prompts_of(bool(ALIAS_RE.search(stub.text)))

    def validate(plan: Any) -> List[str]:
        return structural(plan) + overlap_guard.errors(ctx.overlap, texts(plan))

    budget = _budget(ctx, system, instructions, schema)
    packed = pack(build, ranked, allowed_levels(policy), budget)
    notes = [] if packed.fits else [f"{what} does not fit the window even as stubs; sent anyway"]
    result = _ask(ctx, packed.payload, schema, system, instructions, validate)
    if result.overflow:
        _calibrate(ctx, result)
        notes.append("the prompt overflowed the window; repacked smaller and retried")
        packed = pack(build, ranked, allowed_levels(policy),
                      min(int(budget * 0.6), _budget(ctx, system, instructions, schema)))
        result = _ask(ctx, packed.payload, schema, system, instructions, validate)
    _calibrate(ctx, result)
    result.levels, result.fit = packed.levels, _fit(packed, ctx)
    result.warnings = notes + result.warnings
    if not result.ok and result.candidate is not None and not structural(result.candidate):
        result.plan, result.raw, result.errors = result.candidate, result.candidate_raw, []
    if result.ok:
        result.warnings += overlap_guard.enforce(ctx.overlap, result.plan, texts(result.plan))
    return result


def plan_narrative(ctx: PlanContext, iface: Interface) -> PlanResult:
    repo_map = _repo_map(ctx)
    handlers = list(dict.fromkeys(op.handler for op in iface.operations if op.handler))

    def build(detail: Dict[str, str]):
        return build_interface_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, iface, detail=detail,
                                       root=ctx.root, repo_map=repo_map)

    return _plan_packed(ctx, build, handlers,
                        lambda a: narrative_schema(a["operations"], a["params"]),
                        prompts.narrative_prompts, check_narrative, overlap_guard.narrative_texts,
                        what="the interface")


def check_activity(plan: Any) -> List[str]:
    _dedupe(plan["labels"], "node")
    for lab in plan["labels"]:
        lab["text"] = " ".join(lab["text"].split())[:80]
    return []


def plan_activity(ctx: PlanContext, act) -> PlanResult:
    """Labels for an activity diagram drawn from the code (archdoc.activity)."""
    fid = act.function
    ranked = [fid]

    def build(detail: Dict[str, str]):
        return build_activity_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, act, detail=detail,
                                      root=ctx.root)

    return _plan_packed(ctx, build, ranked, lambda a: activity_schema(a["nodes"]), prompts.activity_prompts,
                        check_activity, what="the function")


def check_usecase(plan: Any) -> List[str]:
    _dedupe(plan["use_cases"], "ref")
    _dedupe(plan["actors"], "ref")
    for c in plan["use_cases"]:
        c["goal"] = " ".join(c["goal"].split())[:70]
    return []


def plan_usecase(ctx: PlanContext, model) -> PlanResult:
    """Goal names for the use cases and clearer actor names (archdoc.usecases)."""
    repo_map = _repo_map(ctx)

    def build(detail: Dict[str, str]):
        return build_usecase_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, model, repo_map=repo_map)

    return _plan_packed(ctx, build, [], lambda a: usecase_schema(a["cases"], a["actors"]), prompts.usecase_prompts,
                        check_usecase, what="the use cases")


# -- rebuild content (4c) ---------------------------------------------------------------

def _dedupe(items: List[Dict[str, Any]], key: str) -> None:
    """An entry listed twice is a formatting slip, not a factual error: keep the
    first, adding any other wording the repeats carry."""
    first: Dict[str, Dict[str, Any]] = {}
    kept = []
    for it in items:
        if it[key] not in first:
            first[it[key]] = it
            kept.append(it)
            continue
        for k, v in it.items():
            if k != key and isinstance(v, str) and v and v not in first[it[key]].get(k, ""):
                first[it[key]][k] = (first[it[key]][k] + "; " + v).lstrip("; ")
    items[:] = kept


def check_behavior(plan: Any) -> List[str]:
    _dedupe(plan["inputs"], "name")
    _dedupe(plan["errors"], "ref")
    _dedupe(plan["side_effects"], "ref")
    return []


def plan_behavior(ctx: PlanContext, function_id: str) -> PlanResult:
    """Behavior specification of one function: the target first in rank (source
    where policy allows), then its callees. Inputs, side effects and errors can
    only name the target's parameters, its calls and the exception types it or a
    callee raises (schema enums), so a claim about an error the code cannot raise
    is not expressible."""
    repo_map = _repo_map(ctx)
    fn = ctx.ir.find_function(function_id)
    ranked = [function_id] + [c for c in dict.fromkeys(fn.calls) if c != function_id and ctx.ir.find_function(c)]

    def build(detail: Dict[str, str]):
        return build_function_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, function_id, detail=detail,
                                      root=ctx.root, repo_map=repo_map)

    result = _plan_packed(ctx, build, ranked,
                          lambda a: behavior_schema(a["function"], a["params"], a["calls"], a["raises"]),
                          prompts.behavior_prompts, check_behavior, what="the function")
    result.report = {"evidence": result.levels.get(function_id, "stub")}
    return result


def _member_rank(ir: RepoIR, mid: str) -> List[str]:
    """A module's functions, most called from elsewhere first (DESIGN.md 12.2)."""
    m = ir.module(mid)
    fns = [f for f in [*m.functions, *(x for c in m.classes for x in c.methods)]]
    callers: Dict[str, int] = {}
    for f in ir.iter_functions():
        for c in set(f.calls):
            callers[c] = callers.get(c, 0) + 1
    return [f.id for f in sorted(fns, key=lambda f: (f.visibility != "public", -callers.get(f.id, 0), f.id))]


def check_summary(plan: Any) -> List[str]:
    for key in ("key_operations", "parts", "collaborators", "external"):
        if key in plan:
            _dedupe(plan[key], "ref")
    return []


def plan_module_summary(ctx: PlanContext, mid: str) -> PlanResult:
    repo_map = _repo_map(ctx)

    def build(detail: Dict[str, str]):
        return build_module_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, mid, detail=detail,
                                    root=ctx.root, repo_map=repo_map)

    return _plan_packed(ctx, build, _member_rank(ctx.ir, mid),
                        lambda a: module_summary_schema(a["members"], a["collaborators"], a["external"]),
                        prompts.module_summary_prompts, check_summary, what="the module")


def plan_group_summary(ctx: PlanContext, group: str, parts: List[Any], collaborators: List[str],
                       external: List[str]) -> PlanResult:
    """A package summary from its modules' summaries, or (group "") the system
    summary from the packages'. `parts` is [(id, summary text as the model wrote it)]."""
    repo_map = _repo_map(ctx)

    def build(detail: Dict[str, str]):
        return build_group_payload(ctx.ir, ctx.config, ctx.tier, ctx.aliases, group, parts, collaborators,
                                   external, repo_map=repo_map)

    return _plan_packed(ctx, build, [],
                        lambda a: group_summary_schema(a["parts"], a["collaborators"], a["external"]),
                        prompts.group_summary_prompts, check_summary,
                        what="the package" if group else "the system")


def check_tests(plan: Any, expected: Optional[List[str]] = None) -> List[str]:
    _dedupe(plan["tests"], "ref")
    missing = [t for t in (expected or []) if t not in {x["ref"] for x in plan["tests"]}]
    return [f"no criterion for: {', '.join(missing)}"] if missing else []


def plan_tests(ctx: PlanContext, mid: str, test_ids: List[str], exercised: List[str],
               ir_all: Optional[RepoIR] = None) -> PlanResult:
    """Acceptance criteria for one test module. Tests live in `ir_all` (the repo
    including its tests), which the documented IR may exclude."""
    ir = ir_all or ctx.ir

    def build(detail: Dict[str, str]):
        return build_tests_payload(ir, ctx.config, ctx.tier, ctx.aliases, mid, test_ids, exercised,
                                   detail=detail, root=ctx.root)

    real = [t for t in test_ids if ir.find_function(t)]
    result = _plan_packed(ctx, build, real, lambda a: tests_schema(a["tests"]), prompts.tests_prompts,
                          lambda plan: check_tests(plan, real), what="the test module")
    if not result.ok and result.candidate is not None and not check_tests(result.candidate):
        # still missing some after the repair round: keep the criteria it did write
        missing = check_tests(result.candidate, real)
        result.plan, result.raw, result.errors = result.candidate, result.candidate_raw, []
        result.warnings += missing + overlap_guard.enforce(ctx.overlap, result.plan,
                                                           overlap_guard.plan_texts(result.plan))
    return result


__all__ = ["CLIENT", "PlanContext", "check_activity", "check_behavior", "check_narrative", "check_sequence", "check_summary",
           "check_tests", "drop_unused_participants", "fix_reply_flags", "keep_grounded_labels",
           "check_usecase", "plan_activity", "plan_behavior", "plan_group_summary", "plan_module_summary", "plan_narrative", "plan_sequence",
           "plan_tests", "plan_usecase"]
