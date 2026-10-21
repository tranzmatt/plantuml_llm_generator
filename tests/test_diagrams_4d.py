"""
Activity, state, use-case and object diagrams (4d): drawn from the code; a
model may only label activity nodes and use cases. Model-free (ScriptedBackend).
"""

import os
import textwrap

import pytest

from archdoc.activity import build_activity, has_control_flow
from archdoc.extract import extract_python
from archdoc.objects import build_objects
from archdoc.plan.artifacts import plan_activity, plan_usecase
from archdoc.render.puml.activity import render_activity_diagram
from archdoc.render.puml.object_ import render_object_diagram
from archdoc.render.puml.state import render_state_diagram
from archdoc.render.puml.usecase import render_usecase_diagram
from archdoc.states import state_machines
from archdoc.usecases import build_usecases
from archdoc.validate import check_syntax, plantuml_command

from tests.test_design import _ctx, _payload_text
from tests.test_egress import CLOUD_FORBIDDEN, _word_hits

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")
needs_plantuml = pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not available")

APP = '''
from enum import Enum
from dataclasses import dataclass

import requests


class OrderState(Enum):
    NEW = 1
    PAID = 2
    SHIPPED = 3
    CANCELLED = 4


@dataclass
class Order:
    id: int
    state: OrderState = OrderState.NEW

    def pay(self):
        if self.state == OrderState.NEW:
            self.state = OrderState.PAID

    def ship(self):
        if self.state != OrderState.PAID:
            raise ValueError("not paid")
        self.state = OrderState.SHIPPED

    def cancel(self):
        if self.state in (OrderState.NEW, OrderState.PAID):
            self.state = OrderState.CANCELLED

    def reopen(self):
        self.state = OrderState.NEW


def process(orders, gateway):
    """Charge every new order."""
    for order in orders:
        if order.state == OrderState.NEW:
            try:
                gateway.charge(order.id)
                order.pay()
            except TimeoutError:
                continue
        elif order.state == OrderState.PAID:
            order.ship()
        else:
            break
    return len(orders)


def notify(order_id):
    return requests.post("https://hooks.example.com/notify", json={"id": order_id}).ok
'''


@pytest.fixture
def app(tmp_path):
    p = tmp_path / "shop" / "orders.py"
    p.parent.mkdir()
    p.write_text(textwrap.dedent(APP))
    return str(tmp_path), extract_python(str(tmp_path))


# -- activity --------------------------------------------------------------------------

def test_activity_structure_is_the_code(app):
    root, ir = app
    act = build_activity(root, ir, "shop.orders:process")
    kinds = [getattr(x, "kind", "") for x in act.labelable()]
    assert kinds == ["loop", "if", "try", "action", "action", "except", "cond", "action", "return"]
    assert has_control_flow(act)
    text = render_activity_diagram(act, "a", "process")
    assert "while (for order in orders) is (next)" in text
    assert "if (TimeoutError raised?) then (yes)" in text and "break" in text and "stop" in text
    assert "elseif (order.state == OrderState.PAID) then (yes)" in text


def test_activity_labels_replace_text_but_not_structure(app):
    root, ir = app
    act = build_activity(root, ir, "shop.orders:process")
    ctx = _ctx(ir, root=root)
    res = plan_activity(ctx, act)
    assert res.ok
    sent = _payload_text(ctx.backend.calls[0])
    assert "def process(orders, gateway):" in sent and "at: order.state == OrderState.NEW" in sent
    enum = ctx.backend.calls[0]["schema"]["properties"]["labels"]["items"]["properties"]["node"]["enum"]
    assert enum == [x.id for x in act.labelable()]
    labels = {x["node"]: x["text"] for x in res.plan["labels"]}
    plain = render_activity_diagram(act, "a")
    labelled = render_activity_diagram(act, "a", labels=labels)
    shape = lambda t: [ln.split("(")[0].split(":")[0].strip() for ln in t.splitlines()]   # noqa: E731
    assert shape(plain) == shape(labelled) and "does the thing" in labelled


def test_activity_is_not_sent_to_cloud_by_default(app):
    from archdoc.egress import EgressAborted
    root, ir = app
    act = build_activity(root, ir, "shop.orders:process")
    with pytest.raises(EgressAborted):
        plan_activity(_ctx(ir, tier="cloud", root=root), act)


@needs_plantuml
def test_activity_syntax(app):
    root, ir = app
    for fid in ("shop.orders:process", "shop.orders:Order.ship"):
        assert check_syntax(render_activity_diagram(build_activity(root, ir, fid), "a", fid)).ok


# -- state -------------------------------------------------------------------------------

def test_state_machine_from_assignments_and_guards(app):
    root, ir = app
    (m,) = state_machines(root, ir)
    assert m.key == "shop.orders:OrderState" and m.fields == ["Order.state"]
    assert m.initial == [("NEW", "default of Order.state")]
    got = {(t.src, t.dst, t.function.split(":")[1]) for t in m.transitions}
    assert got == {("NEW", "PAID", "Order.pay"), ("PAID", "SHIPPED", "Order.ship"),     # early-exit guard
                   ("NEW", "CANCELLED", "Order.cancel"), ("PAID", "CANCELLED", "Order.cancel"),
                   (None, "NEW", "Order.reopen")}
    assert m.unreached == []
    text = render_state_diagram(m, "s")
    assert "[*] --> S_NEW" in text and "S_SHIPPED --> S_NEW : Order.reopen" in text
    assert "does not check the current state" in text


def test_state_unreached_is_reported(ir_fixture):
    (m,) = state_machines(FIXTURE, ir_fixture)
    assert m.unreached == ["RETIRED"] and m.transitions == []
    assert "no code assigns RETIRED" in render_state_diagram(m, "s")


@pytest.fixture(scope="module")
def ir_fixture():
    return extract_python(FIXTURE)


# -- use cases -----------------------------------------------------------------------------

def test_usecases_and_external_systems(ir_fixture):
    model = build_usecases(FIXTURE, ir_fixture)
    names = {c.name: c for c in model.cases}
    assert {"GET /{item_id}", "POST /", "GET /health", "seed-data", "export-items"} <= set(names)
    actors = {a.key: a for a in model.actors}
    host = next(a for a in model.actors if a.source == "ext:internal.example")
    assert names["shopctl"].reaches == [host.key]                # main -> ping -> requests.post
    assert names["GET /{item_id}"].reaches == []                 # same module, but no outbound call on its path
    assert actors[names["GET /health"].actor].name == "API client"
    assert all(a.key.startswith("actor") for a in model.actors)


def test_usecase_cloud_payload_hides_hosts(ir_fixture):
    model = build_usecases(FIXTURE, ir_fixture)
    ctx = _ctx(ir_fixture, tier="cloud")
    res = plan_usecase(ctx, model)
    assert res.ok
    sent = _payload_text(ctx.backend.calls[0])
    assert "internal.example" not in sent and _word_hits({"t": sent}, CLOUD_FORBIDDEN) == []
    goals = {c["ref"]: c["goal"] for c in res.plan["use_cases"]}
    assert set(goals) <= {c.id for c in model.cases}             # restored to real operation ids
    text = render_usecase_diagram(model, "u", goals=goals)
    assert "does the thing\\n(" in text


# -- objects --------------------------------------------------------------------------------

def test_objects_created_at_import(tmp_path):
    (tmp_path / "ui.py").write_text(textwrap.dedent('''
        import gradio as gr
        from fastapi import FastAPI, APIRouter

        app = FastAPI(title="Shop")
        router = APIRouter(prefix="/items")
        app.include_router(router, prefix="/v1")
        name = "x".upper()

        with gr.Blocks() as demo:
            box = gr.Textbox(label="Name")
            btn = gr.Button("Go")
            btn.click(fn=str, inputs=[box], outputs=box)
    '''))
    ir = extract_python(str(tmp_path))
    model = build_objects(str(tmp_path), ir)
    objs = {o.name: o for o in model.objects}
    assert set(objs) == {"app", "router", "demo", "box", "btn"}                 # not the factory call
    assert ("title", "'Shop'") in objs["app"].attrs and ("arg1", "'Go'") in objs["btn"].attrs
    links = {(lk.src.split(":")[1], lk.dst.split(":")[1], lk.label) for lk in model.links}
    assert ("app", "router", "include_router") in links
    assert {("demo", "box", "contains"), ("btn", "box", "click inputs"), ("btn", "box", "click outputs")} <= links
    text = render_object_diagram(model, "o")
    assert 'object "router : APIRouter"' in text


@needs_plantuml
def test_state_usecase_object_syntax(app, ir_fixture, tmp_path):
    root, ir = app
    (m,) = state_machines(root, ir)
    assert check_syntax(render_state_diagram(m, "s")).ok
    assert check_syntax(render_usecase_diagram(build_usecases(FIXTURE, ir_fixture), "u")).ok
    assert check_syntax(render_object_diagram(build_objects(FIXTURE, ir_fixture), "o")).ok
