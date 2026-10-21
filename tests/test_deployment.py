"""docker-compose extraction, env wrappers, deployment cross-checks and diagram."""

import os

import pytest

from archdoc.deployment import deployment_issues, service_flows
from archdoc.extract import extract_python
from archdoc.render.puml.deployment import render_deployment_diagram
from archdoc.validate import check_syntax, plantuml_command

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mq_app")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _svc(ir, name):
    (dep,) = ir.deployments
    return next(s for s in dep.services if s.name == name)


def test_services_images_and_entry_points(ir):
    (dep,) = ir.deployments
    assert dep.file == "docker-compose.yml"
    ui, worker, broker = _svc(ir, "ui"), _svc(ir, "worker"), _svc(ir, "broker")
    assert (ui.build, ui.dockerfile, ui.entry_module, ui.modules) == ("ui", "ui/Dockerfile", "ui.app", ["ui.app"])
    # COPY of one file: worker/extra/replay.py is in the repo but not in the image
    assert worker.modules == ["worker.app"] and worker.entry_module == "worker.app"   # ENTRYPOINT, continued line
    assert broker.image == "rabbitmq:3-management" and broker.modules == []
    assert ui.ports == ["7861:7861"] and ui.depends_on == ["broker"]


def test_env_sources_never_values(ir):
    assert _svc(ir, "ui").env == {"BROKER_HOST": "host", "BROKER_PASS": "literal"}
    assert _svc(ir, "worker").env["BROKER_HOST"] == "literal"
    assert "hunter2" not in repr(ir.deployments)


def test_env_wrapper_and_required_per_module(ir):
    env = next(i for i in ir.interfaces if i.kind == "env")
    ops = {o.name: o.meta for o in env.operations}
    assert ops["BROKER_HOST"]["required_in"] == "ui.app, worker.extra.replay"   # via required(), and os.environ[]
    assert ops["BROKER_PORT"] == {"used_in": "ui.app", "default": "'5672'"}


def test_deployment_issues(ir):
    found = {(d["where"], d["issue"]) for d in deployment_issues(ir)}
    assert found == {
        ("worker.extra.replay", "has an entry point or interface but is in no image of this file"),
        ("ui", "`BROKER_PASS` is set but not read by the code in this image"),
        ("ui", "credential `BROKER_PASS` is written literally in the file"),
        ("worker", "`BROKER_HOST` is set but not read by the code in this image"),
        ("worker", "`UNUSED_FLAG` is set but not read by the code in this image"),
    }


def test_service_flows(ir):
    (dep,) = ir.deployments
    flows = {(f["from"], f["to"], f["channel"]) for f in service_flows(ir, dep)}
    assert flows == {("ui", "worker", "queue jobs"), ("worker", "ui", "exchange results")}


def test_deployment_diagram(ir):
    (dep,) = ir.deployments
    text = render_deployment_diagram(ir, dep, "d")
    assert "<<image rabbitmq:3-management>>" in text and "<<entry>>" in text
    assert "S_svc_ui --> S_svc_broker : jobs" in text and "S_svc_broker --> S_svc_worker : jobs" in text
    assert "worker.extra.replay" not in text
    if plantuml_command():
        assert check_syntax(text).ok
