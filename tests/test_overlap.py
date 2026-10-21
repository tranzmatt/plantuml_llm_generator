"""Verbatim-overlap guard: generated text must describe the code, not copy it."""

import json

import pytest

from archdoc.extract import extract_python
from archdoc.overlap import SourceIndex
from archdoc.plan.artifacts import plan_narrative, plan_sequence
from archdoc.policy import PolicyError, load_config
from tests.test_plan import FIXTURE, _ctx, _entry, narrative_responder, sequence_responder

MODULE = '''\
"""Pricing rules for the basket service, with every adjustment applied in a fixed order."""
import os


def compute_total_price(items, discount_rate, tax_rate, currency_code, rounding_mode, region):
    """Compute the total price of all the items in the basket with tax applied."""
    # apply the regional discount before tax because the auditors require this exact ordering
    subtotal = sum(item.price for item in items) * (1 - discount_rate) + region.surcharge(currency_code)
    if not items:
        raise ValueError("the basket must contain at least one item before checkout can start")
    return subtotal
'''
COMMENT = "Applies the regional discount before tax because the auditors require this exact ordering."


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    root = tmp_path_factory.mktemp("pricing")
    (root / "pricing.py").write_text(MODULE)
    return SourceIndex(str(root), extract_python(str(root)))


def test_comments_bodies_and_literals_are_protected(index):
    hit = index.check(COMMENT)
    assert hit and hit.words >= 12 and (hit.path, hit.line) == ("pricing.py", 7)
    assert index.check("It fails when the basket must contain at least one item before checkout can start.")
    assert index.check("subtotal = sum(item.price for item in items) * (1 - discount_rate)")


def test_interface_text_is_allowed(index):
    assert index.check("compute_total_price(items, discount_rate, tax_rate, currency_code, rounding_mode, region)") is None
    assert index.check("Compute the total price of all the items in the basket with tax applied.") is None
    assert index.check("Pricing rules for the basket service, with every adjustment applied in a fixed order") is None


def test_short_shared_phrases_are_fine(index):
    assert index.check("Applies the regional discount before computing tax.") is None   # 5 shared words
    assert index.check("") is None


def test_describe_never_quotes_the_source(index):
    text = index.check(COMMENT).describe()
    assert "auditors" not in text and "pricing.py:7" in text


def _copying(responder, field, text, times=99):
    """Wrap a responder so its first `times` answers put `text` into `field`."""
    state = {"n": 0, "repairs": []}

    def respond(system, user, schema):
        plan = json.loads(responder(system, user, schema))
        if "YOUR PREVIOUS ANSWER" in user:
            state["repairs"].append(user.split("IT WAS REJECTED FOR THESE REASONS:", 1)[1])
        if state["n"] < times:
            if field == "purpose":
                plan["purpose"] = text
            else:
                plan["messages"][1]["label"] = text
        state["n"] += 1
        return json.dumps(plan)
    return respond, state


@pytest.fixture(scope="module")
def shop_ir():
    return extract_python(FIXTURE)


def test_narrative_repaired_without_sending_the_source(shop_ir, index):
    respond, state = _copying(narrative_responder, "purpose", COMMENT, times=1)
    ctx = _ctx(shop_ir, "local", respond)
    ctx.overlap = index
    result = plan_narrative(ctx, shop_ir.interfaces[0])
    assert result.ok and result.attempts == 2 and result.plan["purpose"] != COMMENT
    (repair,) = state["repairs"]
    assert "purpose repeats" in repair and "auditors" not in repair and "pricing.py" not in repair


def test_narrative_still_copying_is_cleared(shop_ir, index):
    respond, _ = _copying(narrative_responder, "purpose", COMMENT)
    ctx = _ctx(shop_ir, "local", respond)
    ctx.overlap = index
    result = plan_narrative(ctx, shop_ir.interfaces[0])
    assert result.ok and result.plan["purpose"] == ""
    assert any(w.startswith("overlap: cleared purpose") for w in result.warnings)


def test_flag_mode_keeps_and_reports(shop_ir, tmp_path):
    (tmp_path / "pricing.py").write_text(MODULE)
    flag = SourceIndex(str(tmp_path), extract_python(str(tmp_path)), mode="flag")
    respond, _ = _copying(sequence_responder, "label", COMMENT)
    ctx = _ctx(shop_ir, "local", respond)
    ctx.overlap = flag
    result = plan_sequence(ctx, _entry(shop_ir))
    assert result.ok and result.plan["messages"][1]["label"] == COMMENT
    assert any(w.startswith("overlap: kept messages[1].label") for w in result.warnings)


def test_sequence_label_cleared_in_reject_mode(shop_ir, index):
    respond, _ = _copying(sequence_responder, "label", COMMENT)
    ctx = _ctx(shop_ir, "local", respond)
    ctx.overlap = index
    result = plan_sequence(ctx, _entry(shop_ir))
    assert result.ok and result.plan["messages"][1]["label"] == ""          # renderer falls back to the name
    assert any("overlap: cleared messages[1].label" in w for w in result.warnings)


def test_config(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[overlap]\nmode = "flag"\nmin_words = 10\n')
    cfg = load_config(None, user_config=None, extra=str(cfg_file))
    assert (cfg.overlap_mode, cfg.overlap_min_words) == ("flag", 10)
    cfg_file.write_text('[overlap]\nmin_words = 2\n')
    with pytest.raises(PolicyError):
        load_config(None, user_config=None, extra=str(cfg_file))
