"""
Deterministic verification of model output against the IR (DESIGN.md 2.2).

Every check here is about facts the extractor already knows: who owns a
function, who calls it, in what order, what raises. A plan that disagrees
with those facts is wrong, however plausible it looks. Issues are first sent
back to the model for one repair round; whatever still fails is removed and
reported, never silently kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..egress import participant_of
from ..ir import EXTERNAL_PREFIX, Function, RepoIR
from .schemas import CLIENT


@dataclass
class Issue:
    kind: str                  # message | group | plan
    index: Optional[int]       # message or group index; None for plan-level issues
    text: str

    def __str__(self) -> str:
        where = f"{self.kind}s[{self.index}]" if self.index is not None else "plan"
        return f"{where}: {self.text}"


def verify_sequence(ir: RepoIR, plan: Dict[str, Any], slice_ids: List[str], entry: str) -> List[Issue]:
    fns: Dict[str, Function] = {f: ir.find_function(f) for f in slice_ids if ir.find_function(f)}
    entry_owner = participant_of(entry)
    ext_callers: Set[str] = {participant_of(g) for g, fn in fns.items()
                             if any(c.startswith(EXTERNAL_PREFIX) for c in fn.calls)}
    issues: List[Issue] = []
    calls_made: List[Tuple[str, str]] = []           # (from, to) of call messages so far
    last_pos: Dict[str, Tuple[int, str]] = {}        # caller fn -> (position in its calls, callee)
    seen: Dict[str, Set[str]] = {}

    messages = plan["messages"]
    if not messages or messages[0]["from"] != CLIENT:
        issues.append(Issue("plan", None, f"the first message must come from {CLIENT} (the entry call)"))

    for i, m in enumerate(messages):
        src, dst, call = m["from"], m["to"], m.get("call") or ""
        if call:
            if call not in fns:
                issues.append(Issue("message", i, f"{call} is not part of this flow"))
                continue
            owner = participant_of(call)
            if dst != owner:
                issues.append(Issue("message", i, f"{call} belongs to {owner}, so the message must go to "
                                                  f"{owner}, not {dst}"))
                continue
            if src == CLIENT:
                if call != entry:
                    issues.append(Issue("message", i, f"{CLIENT} can only call the entry point {entry}, "
                                                      f"not {call}"))
                    continue
            else:
                callers = [g for g, fn in fns.items() if call in fn.calls and participant_of(g) == src]
                if not callers:
                    real = sorted({participant_of(g) for g, fn in fns.items() if call in fn.calls})
                    hint = f"; it is called from {', '.join(real)}" if real else ""
                    issues.append(Issue("message", i, f"nothing in {src} calls {call}{hint}"))
                    continue
                if len(callers) == 1:
                    g = callers[0]
                    if call not in seen.setdefault(g, set()):
                        pos = fns[g].calls.index(call)
                        prev_pos, prev = last_pos.get(g, (-1, ""))
                        if pos < prev_pos:
                            issues.append(Issue("message", i, f"out of order: {g} calls {call} before {prev}"))
                            continue
                        seen[g].add(call)
                        last_pos[g] = (pos, call)
            calls_made.append((src, dst))
        elif m.get("reply"):
            if (dst, src) not in calls_made:
                issues.append(Issue("message", i, f"return from {src} to {dst} answers no earlier call "
                                                  f"from {dst} to {src}"))
        elif src == CLIENT and dst == entry_owner:
            calls_made.append((src, dst))                # entry invocation written without "call"
        elif src == dst and src in ext_callers:
            pass                                         # a third-party call, drawn as a self-message
        else:
            issues.append(Issue("message", i, f"{src} -> {dst} is not a call, a return, or a "
                                              f"third-party call in the code"))

    # A raise is evidence of a conditional path in the function that raises. Inside a
    # group, that function is either called (a call to it) or already running (a
    # message sent by its owner, e.g. read_item raising HTTPException as a self-call).
    raising = {f for f, fn in fns.items() if fn.raises}
    raising_owners = {participant_of(f) for f in raising}
    for gi, g in enumerate(plan.get("groups", [])):
        if g["kind"] == "loop":
            issues.append(Issue("group", gi, "loops need control-flow data, which this request did not include"))
        elif g["kind"] in ("alt", "opt"):
            inside = messages[g["first"]:g["last"] + 1]
            if not any((m.get("call") or "") in raising or m["from"] in raising_owners for m in inside):
                issues.append(Issue("group", gi, f"{g['kind']} {g['label']!r} involves no function that raises, "
                                                 f"so the code shows no condition here"))
    return issues


def prune_sequence(ir: RepoIR, plan: Dict[str, Any], slice_ids: List[str], entry: str,
                   rounds: int = 4) -> Tuple[List[str], List[Issue]]:
    """Remove what fails verification, re-checking until stable (removing a call
    can orphan its return). Returns (warnings, remaining plan-level issues)."""
    removed: List[str] = []
    for _ in range(rounds):
        issues = verify_sequence(ir, plan, slice_ids, entry)
        bad_msgs = {x.index for x in issues if x.kind == "message"}
        bad_groups = {x.index for x in issues if x.kind == "group"}
        if not bad_msgs and not bad_groups:
            return removed, [x for x in issues if x.kind == "plan"]
        removed += [f"removed {x}" for x in issues if x.kind in ("message", "group")]
        keep = [i for i in range(len(plan["messages"])) if i not in bad_msgs]
        remap = {old: new for new, old in enumerate(keep)}
        plan["messages"] = [plan["messages"][i] for i in keep]
        groups = []
        for gi, g in enumerate(plan.get("groups", [])):
            if gi in bad_groups:
                continue
            inside = [remap[i] for i in range(g["first"], g["last"] + 1) if i in remap]
            if inside:
                groups.append({**g, "first": inside[0], "last": inside[-1]})
        plan["groups"] = groups
    issues = verify_sequence(ir, plan, slice_ids, entry)
    return removed, [x for x in issues if x.kind == "plan"]
