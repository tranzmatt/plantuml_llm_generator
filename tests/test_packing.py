"""Context fitting: encoding, detail levels, packing, calibration, overflow (DESIGN.md 12)."""

import dataclasses
import json

import pytest

from archdoc.backends import Completion, ScriptedBackend
from archdoc.egress import build_interface_payload, build_sequence_payload
from archdoc.extract import extract_python
from archdoc.packing import Calibration, Window, encode, pack
from archdoc.plan.artifacts import plan_sequence
from archdoc.plan.entries import call_slice
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap
from tests.test_egress import SECRET
from tests.test_plan import FIXTURE, _ctx, _entry, narrative_responder, sequence_responder

READ = "shop.api:read_item"
GET = "shop.repo:ItemRepo.get"


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _seq(ir, detail, tier="local", config=None, ids=None, handler=READ):
    ids = ids or call_slice(ir, handler, 3, 25)
    return build_sequence_payload(ir, config or Config(), tier, AliasMap(), handler, ids, detail=detail,
                                  root=FIXTURE)


def test_encoding_is_stub_text(ir):
    text = _seq(ir, {}).text
    assert "def get(item_id: int) -> Item:  # owner ItemRepo; calls query" in text     # short ids
    assert '"""Fetch one item."""' in text and "{" not in text.split("functions:")[1]


def test_outline_and_source_levels(ir):
    outline = _seq(ir, {GET: "outline"}).text
    assert "raise KeyError" in outline and "CANARY_SQL_a41c" not in outline     # structure, no expressions
    source = _seq(ir, {GET: "source"}).text
    assert 'self.db.query("CANARY_SQL_a41c")' in source                      # the whole body, locally
    assert "# get; owner ItemRepo" in source


def test_bodies_never_reach_cloud_or_sensitive_modules(ir):
    for tier, config in (("cloud", Config()), ("local", dataclasses.replace(Config(), sensitive=["src/shop/repo.py"]))):
        text = _seq(ir, {GET: "source", READ: "source"}, tier, config).text
        assert "CANARY_SQL_a41c" not in text and "raise KeyError" not in text, tier


def test_policy_ceiling(ir):
    pol = dataclasses.replace(Config().policies["local"], level="control_flow")
    cfg = dataclasses.replace(Config(), policies={**Config().policies, "local": pol})
    text = _seq(ir, {GET: "source"}, config=cfg).text
    assert "CANARY_SQL_a41c" not in text                                     # source above the ceiling


def test_secrets_redacted_inside_source(ir):
    text = _seq(ir, {"shop.repo:connect": "source"}, ids=["shop.repo:connect"], handler="shop.repo:connect").text
    assert "def connect(" in text and SECRET not in text


def test_pack_upgrades_in_rank_order_within_budget(ir):
    ids = call_slice(ir, READ, 3, 25)
    build = lambda d: _seq(ir, d, ids=ids)                                    # noqa: E731
    stub, full = len(build({}).text), len(build({i: "source" for i in ids}).text)
    roomy = pack(build, ids, ["outline", "source"], full + 10)
    assert roomy.fits and set(roomy.levels.values()) == {"source"}
    # exactly room for the top-ranked item's outline: it gets it, and nothing else fits after
    first = len(build({ids[0]: "outline"}).text)
    tight = pack(build, ids, ["outline", "source"], first)
    assert tight.levels == {ids[0]: "outline", **{i: "stub" for i in ids[1:]}}
    # an item too big to upgrade is skipped; smaller ones below it still are
    skip = pack(build, ids, ["outline", "source"], len(build({ids[-1]: "source"}).text))
    assert skip.levels[ids[-1]] != "stub"
    none = pack(build, ids, ["outline", "source"], stub - 1)
    assert not none.fits


def test_body_canaries_only_whole_and_only_where_packed(ir):
    ids = call_slice(ir, READ, 3, 25)
    build = lambda d: _seq(ir, d, ids=ids)                                    # noqa: E731
    for budget in range(len(build({}).text), len(build({i: "source" for i in ids}).text) + 1, 40):
        p = pack(build, ids, ["outline", "source"], budget)
        has = "CANARY_SQL_a41c" in p.payload.text
        assert has == (p.levels[GET] == "source")
        if has:                                                                # never a partial body
            assert 'row = self.db.query("CANARY_SQL_a41c")' in p.payload.text


def test_plan_shrinks_the_slice_for_a_small_window(ir):
    seen = {}
    for tokens in range(4800, 5600, 10):              # the prompt itself takes most of a window this small
        ctx = _ctx(ir, "local", sequence_responder)
        ctx.root, ctx.window = FIXTURE, Window(tokens=tokens)
        result = plan_sequence(ctx, _entry(ir))
        assert result.ok
        key = next((w for w in result.warnings if "slice" in w or "map" in w), "full")
        seen.setdefault(key.split(" to fit")[0].split(";")[0], result)
    assert "slice reduced from 3 to 2 functions" in seen               # shrinking happened at some size
    reduced = seen["slice reduced from 3 to 2 functions"]
    assert reduced.fit["fits"] and reduced.fit["payload_chars"] <= reduced.fit["budget_chars"]
    assert any(k.startswith("the slice does not fit") for k in seen)      # and below that, the minimum
    assert "full" in seen


def test_plan_uses_spare_room_for_source(ir):
    ctx = _ctx(ir, "local", sequence_responder)
    ctx.root = FIXTURE
    ctx.window = Window(tokens=32768)
    result = plan_sequence(ctx, _entry(ir))
    assert result.ok and set(result.levels.values()) == {"source"}
    assert result.fit["levels"] == {"source": 3}


class Truncating(ScriptedBackend):
    """Reports a prompt that filled the window the first time, like a truncating Ollama."""

    def complete_json(self, system, user, schema, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "schema": schema})
        tokens = self.window if len(self.calls) == 1 else len(system + user) // 4
        return Completion(self.respond(system, user, schema), prompt_tokens=tokens)


def test_overflow_is_repacked_and_retried(ir):
    ctx = _ctx(ir, "local", sequence_responder)
    ctx.backend = Truncating(sequence_responder)
    ctx.backend.window = 32768
    ctx.root = FIXTURE
    ctx.calibration = Calibration()
    result = plan_sequence(ctx, _entry(ir))
    assert result.ok and len(ctx.backend.calls) == 2
    assert any("overflowed the window" in w for w in result.warnings)
    assert len(ctx.backend.calls[1]["user"]) < len(ctx.backend.calls[0]["user"])
    assert ctx.calibration.get("scripted") < 3.0                             # learned from the overflow


def test_calibration_keeps_the_conservative_ratio(tmp_path):
    path = tmp_path / "cal.json"
    cal = Calibration.load(str(path))
    assert cal.get("m") == 3.0
    cal.observe("m", 4000, 1000)
    cal.observe("m", 3500, 1000)
    cal.observe("m", 5000, 1000)
    assert cal.get("m") == 3.5 and json.loads(path.read_text()) == {"m": 3.5}
    assert Calibration.load(str(path)).get("m") == 3.5
    cal.observe("m", 10, 2)                                                  # too small to measure
    assert cal.get("m") == 3.5


def test_window_budget():
    w = Window(tokens=8192, output_tokens=2048, chars_per_token=3.0)
    assert w.input_chars(0) == int((8192 - 4096) * 0.9 * 3.0)
    assert w.input_chars(10**6) == 0


def test_repo_map_is_identical_across_requests(ir):
    ctx = _ctx(ir, "cloud", lambda s, u, sc: (sequence_responder if "messages" in sc["properties"]
                                              else narrative_responder)(s, u, sc))
    plan_sequence(ctx, _entry(ir))
    from archdoc.plan.artifacts import plan_narrative
    plan_narrative(ctx, ir.interfaces[0])
    maps = [c["user"].split("INPUT:\n", 1)[1].split("\n", 1)[1].split("\n", 1)[0] for c in ctx.backend.calls]
    heads = [c["user"].split("INPUT:\n", 1)[1].split("\nentry:")[0].split("\ninterface:")[0]
             for c in ctx.backend.calls]
    assert heads[0].startswith("system_map:") and len(set(heads)) == 1 and maps
