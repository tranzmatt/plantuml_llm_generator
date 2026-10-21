"""Evaluation harness: reference parsing, flow comparison, model metrics."""

import os

import pytest

from archdoc.evaluate import (
    BROKER, call_coverage, evaluate_model, evaluate_structure, match_nodes, parse_mermaid, report_markdown,
)
from archdoc.extract import extract_python
from tests.test_plan import _ctx, narrative_responder, sequence_responder

HERE = os.path.dirname(__file__)
MQ = os.path.join(HERE, "fixtures", "mq_app")
SPEC = {"truth": {"mermaid": "docs/arch.mermaid", "nodes": {"Broker": BROKER}}}


def test_parse_mermaid():
    g = parse_mermaid(open(os.path.join(MQ, "docs", "arch.mermaid")).read())
    assert g.nodes == {"U": "Console UI", "B": "Broker", "W": "Worker", "S": "Store"}
    assert ("U", "B", "Question") in g.edges and ("W", "S", "") in g.edges


def test_match_nodes():
    m = match_nodes({"a": "1. Bio Retrieval", "b": "RabbitMQ", "c": "Upload Service"},
                    ["bio_retrieval", "gradio_interface"], {"RabbitMQ": BROKER})
    assert m == {"a": "bio_retrieval", "b": BROKER, "c": None}


def test_structure_against_reference():
    ir = extract_python(MQ)
    s = evaluate_structure(ir, MQ, SPEC)
    status = {(r["from_service"], r["to_service"]): r["status"] for r in s["flows"]}
    assert status[("ui", "worker")] == "matched"          # "Question" in and out of the broker
    assert status[("worker", "ui")] == "matched"          # "Answers" / "Answer": plural folded
    assert status[("worker", None)] == "not comparable"   # Store is not a service
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    assert any("Audit" in n for n in s["pairing_notes"])
    text = report_markdown(s, [], ir.repo)
    assert "| ui | worker | Question / Question | matched | queue jobs |" in text


def test_missing_flow_explains_both_ends():
    ir = extract_python(MQ)
    spec = {"truth": {**SPEC["truth"], "flows": [{"from": "Console UI", "to": "Worker", "label": "x"},
                                                   {"from": "Worker", "to": "Console UI", "label": "y"}]}}
    s = evaluate_structure(ir, MQ, spec)
    assert s["recall"] == 1.0
    # drop the code side: the same reference now has nothing to match
    for i in ir.interfaces:
        if i.kind == "message":
            i.operations = [o for o in i.operations if "jobs" not in o.name]
    s = evaluate_structure(ir, MQ, SPEC)
    miss = next(r for r in s["flows"] if (r["from_service"], r["to_service"]) == ("ui", "worker"))
    assert miss["status"] == "missing" and "worker consumes nothing" in miss["why"]


@pytest.fixture(scope="module")
def shop_ir():
    from tests.test_plan import FIXTURE
    return extract_python(FIXTURE)


def test_model_metrics(shop_ir):
    from archdoc.plan.entries import select_entries

    def respond(system, user, schema):
        return (sequence_responder if "sequence diagram" in system else narrative_responder)(system, user, schema)

    ctx = _ctx(shop_ir, "local", respond)
    entries = select_entries(shop_ir, [], 2, 3, 25)
    run = evaluate_model(ctx, entries, shop_ir.interfaces[:2], "local")
    s = run.summary()
    assert s["requests"] == 4 and s["valid_first_attempt_pct"] == 100 and s["failed"] == 0
    assert s["messages_kept_pct"] == 100
    assert 0 < s["call_edges_shown_pct"] <= 100
    assert len(run.details) == 4


def test_call_coverage(shop_ir):
    ids = ["shop.api:read_item", "shop.repo:ItemRepo.get", "shop.repo:Database.query"]
    plan = {"messages": [{"from": "shop.api", "call": "shop.repo:ItemRepo.get"}]}
    shown, total = call_coverage(shop_ir, plan, ids)
    assert shown == 1 and total >= 2
