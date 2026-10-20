"""pika and Gradio detectors, channel wiring, and their rendering."""

import json
import os
import re

import pytest

from archdoc.coverage import known_unknowns
from archdoc.extract import extract_python
from archdoc.egress import build_payload
from archdoc.messaging import RUNTIME, channels, wiring_issues
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap
from archdoc.render.puml.component import render_component_diagram

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mq_app")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _ops(ir, iface_id):
    iface = next(i for i in ir.interfaces if i.id == iface_id)
    return {o.name: o for o in iface.operations}


def test_publish_and_consume_operations(ir):
    worker_in = _ops(ir, "if:message:worker.app:consume")
    jobs = worker_in["consume queue jobs"]
    assert jobs.handler == "worker.app:handle"                     # the callback, not main
    assert [p.name for p in jobs.params] == ["request_id", "text"]  # fields the consumer reads
    assert jobs.meta["ack"] == "manual" and jobs.meta["delivery"].startswith("push")

    worker_out = _ops(ir, "if:message:worker.app:publish")
    assert set(worker_out) == {"publish exchange results", "publish queue audit"}
    res = worker_out["publish exchange results"]
    assert res.meta["exchange_type"] == "direct"                    # declared in the other module
    assert res.meta["routing_key"] == RUNTIME
    assert [p.name for p in res.params] == ["request_id", "answer"]


def test_exclusive_queue_consumer_is_tied_to_its_exchange(ir):
    ui_in = _ops(ir, "if:message:ui.app:consume")
    (op,) = ui_in.values()
    assert op.name == "consume exchange results"
    assert op.meta["queue"] == "exclusive, server-named" and op.meta["delivery"].startswith("poll")
    assert [p.name for p in op.params] == ["answer", "score"]
    ui_out = _ops(ir, "if:message:ui.app:publish")
    assert [p.name for p in ui_out["publish queue jobs"].params] == ["request_id", "text"]  # via a variable
    assert ui_out["publish queue jobs"].meta["persistent"] == "yes"


def test_gradio_ui(ir):
    ui = next(i for i in ir.interfaces if i.kind == "ui")
    assert (ui.name, ui.address, ui.direction) == ("Job Console", "port 7861", "provided")
    (op,) = ui.operations
    assert op.name == "click Ask" and op.handler == "ui.app:submit"
    assert [(p.name, p.annotation) for p in op.params] == [("Question", "Textbox")]
    assert [(p.name, p.annotation) for p in op.outputs] == [("Answer", "Textbox"), ("Score", "Number")]
    assert op.response is None


def test_no_known_unknowns_left(ir):
    assert known_unknowns(ir) == []


def test_channels_and_wiring(ir):
    chans = {c.name: c for c in channels(ir)}
    assert set(chans) == {"jobs", "results", "audit"}
    assert [e.module for e in chans["jobs"].publishers] == ["ui.app"]
    assert [e.module for e in chans["jobs"].consumers] == ["worker.app"]
    issues = wiring_issues(ir)
    assert {(w["channel"], w["issue"]) for w in issues} == {
        ("queue audit", "published but never consumed in this repo"),
        ("exchange results", "consumer reads field(s) no publisher sends: score"),
    }


def test_component_diagram_draws_the_broker(ir):
    text = render_component_diagram(ir, "mq")
    assert 'node "RabbitMQ"' in text
    assert 'queue "results"' in text and "<<direct exchange>>" in text
    assert "C_ui --> Q_mq_RabbitMQ_queue_jobs" in text
    assert "Q_mq_RabbitMQ_queue_jobs --> C_worker" in text
    assert "MESSAGE" not in text                                   # no per-module lollipops


# names chosen in this code: never sent to a cloud model as written
MQ_CLOUD_FORBIDDEN = ["jobs", "results", "audit", "request_id", "answer", "score", "text",
                      "Question", "Answer", "Score", "Job Console", "Ask", "7861", "submit", "handle"]


def test_cloud_payload_aliases_channels_fields_and_labels(ir):
    payload = build_payload(ir, Config(), "cloud", AliasMap())
    sent = payload.text
    leaked = [w for w in MQ_CLOUD_FORBIDDEN
              if re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", sent)]
    assert leaked == []
    # protocol vocabulary still gets through, so the model can tell the roles apart
    metas = [o.get("meta", {}) for i in json.loads(sent)["interfaces"] for o in i["ops"]]
    assert {"publish", "consume"} <= {m.get("role") for m in metas}
    assert "direct" in {m.get("exchange_type") for m in metas}
