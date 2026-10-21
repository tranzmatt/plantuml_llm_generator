"""Review overlay: decisions survive regeneration, settle the model's work, go stale with the code."""

import json
import shutil

import pytest
import yaml

import archdoc.cli as cli
from archdoc.backends import ScriptedBackend
from tests.test_plan import FIXTURE, narrative_responder, sequence_responder

POST = "op:if:http:shop.api:router:POST /"
PURPOSE = "purpose:if:http:shop.api:app"
PRICE = "field:shop.models:Item.price"
SEQ = "sequence:shop.api:read_item"


@pytest.fixture
def repo(tmp_path):
    dst = tmp_path / "src_copy" / "sample_app"
    shutil.copytree(FIXTURE, dst)
    return dst


class Model:
    """Scripted backend that counts what it is asked."""

    def __init__(self):
        self.sequence_calls = 0
        self.narrative_calls = 0

    def __call__(self, system, user, schema):
        if "messages" in schema["properties"]:
            self.sequence_calls += 1
            return sequence_responder(system, user, schema)
        self.narrative_calls += 1
        return narrative_responder(system, user, schema)


def _run(repo, out, monkeypatch, model=None, extra=()):
    args = ["all", str(repo), "-o", str(out), "--types", "sequence", "--no-validate", "--no-date",
            "--entry", "shop.api:read_item", *extra]
    if model is not None:
        monkeypatch.setattr(cli, "make_backend", lambda cfg: ScriptedBackend(model))
        args += ["--narrative", "--model", "m", "--trust", "local"]
    assert cli.main(args) == 0


def _review(out):
    return yaml.safe_load((out / "review" / "review.yaml").read_text())["items"]


def _decide(out, **changes):
    path = out / "review" / "review.yaml"
    data = yaml.safe_load(path.read_text())
    for item_id, update in changes.items():
        data["items"].setdefault(item_id, {}).update(update)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_first_run_offers_generated_and_open_items(repo, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _run(repo, out, monkeypatch, Model())
    items = _review(out)
    assert items[POST]["status"] == "pending" and items[POST]["origin"] == "generated" and items[POST]["text"]
    assert items[PRICE] == {"status": "pending", "origin": "open", "where": "Item.price", "text": "",
                            "evidence": items[PRICE]["evidence"]}
    assert items[SEQ]["status"] == "pending" and (out / "review" / "plans" / "sequence_shop_api_read_item.json").exists()
    assert "op:if:http:shop.api:router:GET /{item_id}" not in items       # from a docstring: nothing to review


def test_decisions_are_applied_and_settle_the_model(repo, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _run(repo, out, monkeypatch, Model())
    _decide(out, **{
        POST: {"status": "edited", "text": "Stores a new item and echoes it back."},
        PURPOSE: {"status": "rejected"},
        PRICE: {"status": "edited", "text": "Unit price in EUR."},
        SEQ: {"status": "approved"},
    })
    model = Model()
    _run(repo, out, monkeypatch, model)
    icd = (out / "icd" / "sample_app_icd.md").read_text()
    assert "Stores a new item and echoes it back. *(reviewed)*" in icd
    assert "Unit price in EUR. *(reviewed)*" in icd
    assert "descriptions marked *(reviewed)*" in icd
    app_purpose = icd.split("shop.api.app (HTTP API)", 1)[1].split("| Purpose |", 1)[1].split("\n", 1)[0]
    assert "**TBD**" in app_purpose                                        # rejected stays TBD
    assert model.sequence_calls == 0                                       # approved plan reused
    seq = json.loads((out / "diagrams" / "sample_app_sequence_shop_api_read_item.provenance.json").read_text())
    assert seq["producer"] == "review" and all(i["class"] == "reviewed" for i in seq["items"])
    prov = json.loads((out / "icd" / "sample_app_icd.provenance.json").read_text())
    by_id = {i["id"]: i for i in prov["items"]}
    assert by_id[POST]["class"] == "reviewed" and by_id[POST]["stale"] is False
    assert by_id[PURPOSE]["class"] == "open" and by_id[PURPOSE]["rejected_in_review"]
    items = _review(out)
    assert items[POST]["text"] == "Stores a new item and echoes it back."  # decided entries untouched
    assert items[PURPOSE]["status"] == "rejected"


def test_reviewed_sequence_renders_without_a_model(repo, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _run(repo, out, monkeypatch, Model())
    _decide(out, **{SEQ: {"status": "approved"}})
    (out / "diagrams" / "sample_app_sequence_shop_api_read_item.puml").unlink()
    _run(repo, out, monkeypatch)                                           # no backend at all
    assert (out / "diagrams" / "sample_app_sequence_shop_api_read_item.puml").exists()
    assert "except reviewed sequence diagrams" in capsys.readouterr().out


def test_rejected_sequence_is_not_drawn(repo, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _run(repo, out, monkeypatch, Model())
    _decide(out, **{SEQ: {"status": "rejected"}})
    (out / "diagrams" / "sample_app_sequence_shop_api_read_item.puml").unlink()
    model = Model()
    _run(repo, out, monkeypatch, model)
    assert not (out / "diagrams" / "sample_app_sequence_shop_api_read_item.puml").exists()
    assert model.sequence_calls == 0


def test_code_change_makes_reviews_stale_not_lost(repo, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _run(repo, out, monkeypatch, Model())
    _decide(out, **{POST: {"status": "edited", "text": "Stores a new item."},
                    SEQ: {"status": "approved"}})
    api = repo / "src" / "shop" / "api.py"
    api.write_text(api.read_text()
                   .replace("async def create_item(item: Item) -> Item:", "async def create_item(item: Item, dry_run: bool = False) -> Item:")
                   .replace("def read_item(item_id: int, verbose: bool = False,",
                            "def read_item(item_id: int, verbose: bool = False, trace: bool = False,"))
    _run(repo, out, monkeypatch, Model())
    printed = capsys.readouterr().out
    assert f"[STALE] {POST}" in printed
    icd = (out / "icd" / "sample_app_icd.md").read_text()
    assert "Stores a new item. *(reviewed; the code changed since: re-review)*" in icd
    assert "reviewed text predates a code change: re-review" in icd       # an open item
    items = _review(out)
    assert items[POST]["status"] == "edited" and items[POST]["text"] == "Stores a new item."
    assert items[POST]["current_evidence"] != items[POST]["evidence"]
    puml = (out / "diagrams" / "sample_app_sequence_shop_api_read_item.puml").read_text()
    assert "re-review" in puml                                              # the slice's code changed too

    # re-approving against the new code clears it
    _decide(out, **{POST: {"evidence": items[POST]["current_evidence"]}})
    _run(repo, out, monkeypatch, Model())
    assert "Stores a new item. *(reviewed)*" in (out / "icd" / "sample_app_icd.md").read_text()
    assert "current_evidence" not in _review(out)[POST]
