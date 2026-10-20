"""Verification of sequence plans against the call graph (DESIGN.md 2.2)."""

import copy
import json
import os
import re

import pytest

from archdoc.backends import ScriptedBackend
from archdoc.extract import extract_python
from archdoc.plan.artifacts import PlanContext, plan_sequence
from archdoc.plan.entries import call_slice, discover_entries
from archdoc.plan.verify import prune_sequence, verify_sequence
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap
from tests.test_egress import CLOUD_FORBIDDEN
from tests.test_plan import payload_of, sequence_responder

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")
ENTRY = "shop.api:read_item"            # read_item -> ItemRepo.get -> Database.query


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


@pytest.fixture
def slice_ids(ir):
    return call_slice(ir, ENTRY)


def msg(src, dst, call="", reply=False, label="x"):
    return {"from": src, "to": dst, "call": call, "label": label, "reply": reply}


GOOD = {
    "title": "t",
    "participants": [],
    "messages": [
        msg("client", "shop.api", ENTRY),
        msg("shop.api", "shop.repo:ItemRepo", "shop.repo:ItemRepo.get"),
        msg("shop.repo:ItemRepo", "shop.repo:Database", "shop.repo:Database.query"),
        msg("shop.repo:Database", "shop.repo:ItemRepo", reply=True),
        msg("shop.api", "client", reply=True),
    ],
    "groups": [],
}


def plan_with(**changes):
    p = copy.deepcopy(GOOD)
    p.update(changes)
    return p


def texts(issues):
    return [str(x) for x in issues]


def test_faithful_plan_passes(ir, slice_ids):
    assert verify_sequence(ir, GOOD, slice_ids, ENTRY) == []


def test_message_to_wrong_owner(ir, slice_ids):
    p = plan_with()
    p["messages"][1]["to"] = "shop.repo:Database"
    issues = texts(verify_sequence(ir, p, slice_ids, ENTRY))
    assert issues[0] == ("messages[1]: shop.repo:ItemRepo.get belongs to shop.repo:ItemRepo, so the message "
                         "must go to shop.repo:ItemRepo, not shop.repo:Database")


def test_caller_that_does_not_call(ir, slice_ids):
    p = plan_with()
    p["messages"][2]["from"] = "shop.api"           # read_item never calls Database.query itself
    issues = texts(verify_sequence(ir, p, slice_ids, ENTRY))
    assert "messages[2]: nothing in shop.api calls shop.repo:Database.query; " \
           "it is called from shop.repo:ItemRepo" in issues


def test_client_may_only_call_the_entry(ir, slice_ids):
    p = plan_with()
    p["messages"][0]["call"] = "shop.repo:ItemRepo.get"
    p["messages"][0]["to"] = "shop.repo:ItemRepo"
    assert any("client can only call the entry point" in t for t in texts(verify_sequence(ir, p, slice_ids, ENTRY)))


def test_out_of_source_order(ir):
    # main() calls only ping() internally; give it a second call to test ordering
    fn = ir.find_function("shop.cli:main")
    original = list(fn.calls)
    try:
        fn.calls = ["shop.repo:ping", "shop.repo:build_repo"]
        ids = ["shop.cli:main", "shop.repo:ping", "shop.repo:build_repo"]
        p = {"title": "", "participants": [], "groups": [], "messages": [
            msg("client", "shop.cli", "shop.cli:main"),
            msg("shop.cli", "shop.repo", "shop.repo:build_repo"),
            msg("shop.cli", "shop.repo", "shop.repo:ping"),
        ]}
        assert texts(verify_sequence(ir, p, ids, "shop.cli:main")) == [
            "messages[2]: out of order: shop.cli:main calls shop.repo:ping before shop.repo:build_repo"]
        p["messages"][1:] = [p["messages"][2], p["messages"][1]]
        assert verify_sequence(ir, p, ids, "shop.cli:main") == []
    finally:
        fn.calls = original


def test_orphan_return(ir, slice_ids):
    p = plan_with()
    p["messages"][3] = msg("shop.repo:Database", "shop.api", reply=True)
    assert any("answers no earlier call from shop.api to shop.repo:Database" in t
               for t in texts(verify_sequence(ir, p, slice_ids, ENTRY)))


def test_message_with_no_basis(ir, slice_ids):
    p = plan_with()
    p["messages"].insert(2, msg("shop.repo:ItemRepo", "shop.api", label="notify"))
    assert any("is not a call, a return, or a third-party call" in t
               for t in texts(verify_sequence(ir, p, slice_ids, ENTRY)))


def test_third_party_self_call_allowed_only_for_real_callers(ir):
    ids = call_slice(ir, "shop.cli:main")            # ping() calls requests
    base = [msg("client", "shop.cli", "shop.cli:main"), msg("shop.cli", "shop.repo", "shop.repo:ping")]
    ok = {"title": "", "participants": [], "groups": [],
          "messages": base + [msg("shop.repo", "shop.repo", label="requests.get")]}
    assert verify_sequence(ir, ok, ids, "shop.cli:main") == []
    # Database.query makes no third-party calls, so a self-message there has no basis
    bad = {**ok, "messages": base + [msg("shop.repo", "shop.repo:Database", "shop.repo:Database.query"),
                                     msg("shop.repo:Database", "shop.repo:Database", label="requests.get")]}
    assert any("not a call, a return, or a third-party call" in t
               for t in texts(verify_sequence(ir, bad, ids, "shop.cli:main")))


def test_groups_need_evidence(ir, slice_ids):
    p = plan_with(groups=[
        {"kind": "alt", "label": "missing", "first": 1, "last": 1},     # ItemRepo.get raises KeyError
        {"kind": "alt", "label": "invented", "first": 2, "last": 2},    # Database.query raises nothing
        {"kind": "loop", "label": "retry", "first": 1, "last": 2},
    ])
    issues = texts(verify_sequence(ir, p, slice_ids, ENTRY))
    assert any(t.startswith("groups[1]:") and "no call to a function that raises" in t for t in issues)
    assert any(t.startswith("groups[2]:") and "loops need control-flow data" in t for t in issues)
    assert not any(t.startswith("groups[0]:") for t in issues)


def test_first_message_must_come_from_client(ir, slice_ids):
    p = plan_with(messages=GOOD["messages"][1:])
    assert "plan: the first message must come from client (the entry call)" in \
        texts(verify_sequence(ir, p, slice_ids, ENTRY))


def test_prune_cascades_to_orphaned_returns(ir, slice_ids):
    p = plan_with(groups=[{"kind": "alt", "label": "missing", "first": 1, "last": 3}])
    p["messages"][2]["from"] = "shop.api"           # bad call; its return (message 3) becomes orphaned
    removed, plan_issues = prune_sequence(ir, p, slice_ids, ENTRY)
    assert [m["call"] or "reply" for m in p["messages"]] == [ENTRY, "shop.repo:ItemRepo.get", "reply"]
    assert len(removed) == 2 and plan_issues == []
    assert p["groups"] == [{"kind": "alt", "label": "missing", "first": 1, "last": 1}]   # remapped
    assert verify_sequence(ir, p, slice_ids, ENTRY) == []


# -- through the planner -----------------------------------------------------------

def _ctx(ir, tier, responder):
    return PlanContext(ir=ir, config=Config(), tier=tier, aliases=AliasMap(), backend=ScriptedBackend(responder))


def _entry(ir, handler):
    return next(e for e in discover_entries(ir) if e.handler == handler)


def misroute(system, user, schema):
    """Sends every non-entry call from the entry's owner: plausible, and wrong."""
    plan = json.loads(sequence_responder(system, user, schema))
    entry_owner = plan["messages"][0]["to"]
    for m in plan["messages"][1:]:
        if m["call"]:
            m["from"] = entry_owner
    return json.dumps(plan)


def test_repair_fixes_on_second_attempt(ir):
    answers = iter([misroute, sequence_responder])
    ctx = _ctx(ir, "local", lambda s, u, sc: next(answers)(s, u, sc))
    result = plan_sequence(ctx, _entry(ir, ENTRY))
    assert result.ok and result.attempts == 2 and result.warnings == []
    assert result.report["verified_first_time"] is False
    assert result.report["verified_after_repair"] is True and result.report["removed"] == []


def test_persistent_errors_are_pruned_not_kept(ir):
    result = plan_sequence(_ctx(ir, "local", misroute), _entry(ir, ENTRY))
    assert result.ok and result.attempts == 2
    calls = [m["call"] for m in result.plan["messages"] if m["call"]]
    assert calls == [ENTRY, "shop.repo:ItemRepo.get"]          # the misrouted query call is gone
    assert any("removed messages[" in w for w in result.warnings)
    assert verify_sequence(ir, result.plan, call_slice(ir, ENTRY), ENTRY) == []


def test_nothing_verifiable_fails(ir):
    def junk(system, user, schema):
        plan = json.loads(sequence_responder(system, user, schema))
        for m in plan["messages"]:
            m["call"], m["reply"] = "", False
            m["from"], m["to"] = m["to"], m["from"]
        return json.dumps(plan)

    result = plan_sequence(_ctx(ir, "local", junk), _entry(ir, ENTRY))
    assert not result.ok


def test_cloud_repair_prompt_uses_aliases_only(ir):
    """Verification messages are built from real ids; the repair prompt must not leak them."""
    backend = ScriptedBackend(misroute)
    ctx = PlanContext(ir=ir, config=Config(), tier="cloud", aliases=AliasMap(), backend=backend)
    plan_sequence(ctx, _entry(ir, ENTRY))
    repair = backend.calls[1]["user"]
    feedback = repair.split("IT WAS REJECTED FOR THESE REASONS:")[1]
    assert "belongs to @" in feedback or "nothing in @" in feedback       # errors were sent...
    leaked = [w for w in CLOUD_FORBIDDEN + ["shop.repo:ItemRepo", "shop.api"]
              if re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", feedback)]
    assert leaked == []                                                    # ...in alias terms only
    payload_of(repair)                                                     # payload still parseable
