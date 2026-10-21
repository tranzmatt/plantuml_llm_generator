"""
Egress canary tests: the core privacy guarantees, per trust tier.

  cloud: no body/comment/docstring canary, no secret, and no real internal
         identifier or interface literal appears in any payload value
  local: no body/comment canary and no secret; real names are expected
"""

import dataclasses
import json
import os
import re

import pytest

from archdoc.egress import EgressAborted, EgressLog, build_payload
from archdoc.extract import extract_python
from archdoc.ingress import accept
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")

BODY_ONLY = ["CANARY_BODY_7f3a", "CANARY_COMMENT_19bd", "CANARY_SQL_a41c", "CANARY_DETAIL_55e0",
             "internal.example", "select 1"]
DOCSTRING_CANARY = "CANARY_DOC_88aa"
SECRET = "AKIAQWERTYUIOPASDFGH"

# Real names from the fixture that must not reach a cloud model. Checked as
# whole words in payload *values* (keys are the payload's own vocabulary).
CLOUD_FORBIDDEN = [
    "shop", "sample_app", "Item", "Order", "Status", "ItemRepo", "CachedRepo", "Database",
    "read_item", "create_item", "build_repo", "make_order", "cmd_serve", "seed_data", "export_items",
    "connect", "item_id", "verbose", "access_key", "order_id", "ACTIVE", "RETIRED",
    "shopctl", "serve", "seed-data", "export-items", "admin", "standalone",
    "SHOP_DB_URL", "SHOP_SECRET", "/v1/items", "/{item_id}", "/health", "--port", "--mode",
    "sqlite://", "Display name", "Shop control", "Fetch one item", "us-east-1",
]


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _values(obj):
    """All string values (not keys) in a JSON-like structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for v in obj:
            yield from _values(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _values(v)


def _word_hits(payload_data, words):
    vals = list(_values(payload_data))
    hits = []
    for w in words:
        rx = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])")
        if any(rx.search(v) for v in vals):
            hits.append(w)
    return hits


def _payload(ir, tier, config=None, aliases=None, **kw):
    return build_payload(ir, config or Config(), tier, aliases if aliases is not None else AliasMap(), **kw)


def _with_policy(tier, **changes):
    cfg = Config()
    cfg.policies = dict(cfg.policies)
    cfg.policies[tier] = dataclasses.replace(cfg.policies[tier], **changes)
    return cfg


# -- cloud ----------------------------------------------------------------------

def test_cloud_no_canaries_or_secret(ir):
    text = _payload(ir, "cloud").text
    assert [c for c in BODY_ONLY + [DOCSTRING_CANARY, SECRET] if c in text] == []


def test_cloud_no_real_identifiers(ir):
    assert _word_hits(_payload(ir, "cloud").data, CLOUD_FORBIDDEN) == []


def test_cloud_with_first_line_docstrings_still_aliases_names(ir):
    cfg = _with_policy("cloud", docstrings="first_line")
    p = _payload(ir, "cloud", cfg)
    assert DOCSTRING_CANARY in p.text       # opted in: docstrings flow...
    assert _word_hits(p.data, ["Item", "Order", "ItemRepo"]) == []   # ...but names in them don't


def test_cloud_keeps_third_party_and_structure(ir):
    p = _payload(ir, "cloud")
    assert "ext:fastapi" in p.text and "ext:requests.get" in p.text
    kinds = {i["kind"] for i in p.data["interfaces"]}
    assert kinds == {"http", "cli", "env"}
    router = next(i for i in p.data["interfaces"] if len(i["ops"]) == 2 and i["kind"] == "http")
    assert sorted(o["name"].split()[0] for o in router["ops"]) == ["GET", "POST"]
    get = next(o for o in router["ops"] if o["name"].startswith("GET"))
    assert get["errors"] == ["404"]


def test_cloud_secret_aborts_when_defaults_opted_in(ir):
    cfg = _with_policy("cloud", defaults="simple_literals")
    with pytest.raises(EgressAborted) as exc:
        _payload(ir, "cloud", cfg)
    assert SECRET not in str(exc.value)
    assert "aws_access_key" in str(exc.value)


def test_interface_literals_none_drops_them(ir):
    cfg = _with_policy("cloud", interface_literals="none")
    p = _payload(ir, "cloud", cfg)
    http = [i for i in p.data["interfaces"] if i["kind"] == "http"]
    assert all("address" not in i for i in http)
    assert all(o["name"] in ("GET", "POST") for i in http for o in i["ops"])


# -- local ----------------------------------------------------------------------

def test_local_no_canaries_or_secret(ir):
    p = _payload(ir, "local")
    assert [c for c in BODY_ONLY + [SECRET] if c in p.text] == []
    assert p.redactions and p.redactions[0][1] == "aws_access_key"


def test_local_uses_real_names(ir):
    p = _payload(ir, "local")
    assert DOCSTRING_CANARY in p.text                   # first-line docstrings by default
    assert "shop.repo:ItemRepo" in p.text and "/v1/items" in p.text and "SHOP_DB_URL" in p.text
    assert "@" not in "".join(v for v in _values(p.data) if v.startswith("@"))   # no aliases


def test_sensitive_paths_alias_even_locally_and_across_modules(ir):
    cfg = Config(sensitive=["src/shop/models.py"])
    p = _payload(ir, "local", cfg)
    # the sensitive module's symbols are aliased everywhere they are referenced...
    assert _word_hits(p.data, ["Item", "Order", "Status", "shop.models", "shop.models:Item"]) == []
    # ...while the rest of the repo keeps real names
    assert "shop.repo:ItemRepo" in p.text


# -- levels and members ---------------------------------------------------------

def test_level_interfaces_has_no_call_edges(ir):
    p = _payload(ir, "local", _with_policy("local", level="interfaces"))
    assert not any(k in p.text for k in ('"calls"', '"creates"', '"raises"'))


def test_level_none_sends_nothing(ir):
    p = _payload(ir, "cloud", _with_policy("cloud", level="none"))
    assert p.data["modules"] == [] and p.data["interfaces"] == []


def test_members_public_hides_protected(ir):
    public = _payload(ir, "local").text
    everything = _payload(ir, "local", _with_policy("local", members="all")).text
    assert not re.search(r"\b_cache\b", public) and re.search(r"\b_cache\b", everything)


def test_scope_limits_modules(ir):
    p = _payload(ir, "local", modules=["shop.repo"])
    assert [m["ref"] for m in p.data["modules"]] == ["shop.repo"]


# -- aliases ------------------------------------------------------------------

def test_alias_numbering_is_stable(ir, tmp_path):
    path = str(tmp_path / "alias_map.json")
    first = AliasMap()
    a = _payload(ir, "cloud", aliases=first).text
    first.save(path)
    b = _payload(ir, "cloud", aliases=AliasMap.load(path)).text
    c = _payload(ir, "cloud", aliases=AliasMap()).text
    assert a == b == c


def test_round_trip_through_ingress(ir):
    aliases = AliasMap()
    _payload(ir, "cloud", aliases=aliases)
    item = aliases.lookup("class", "shop.models:Item")
    get = aliases.lookup("function", "shop.repo:ItemRepo.get")
    plan = {
        "participants": [{"ref": item}, {"ref": get}],
        "label": f"{get} loads one {item}",
        item: {"summary": "narrative keyed by ref"},
    }
    result = accept(plan, aliases)
    assert result.ok
    assert result.data["participants"] == [{"ref": "shop.models:Item"}, {"ref": "shop.repo:ItemRepo.get"}]
    assert result.data["label"] == "get loads one Item"
    assert "shop.models:Item" in result.data


def test_ingress_rejects_unknown_aliases(ir):
    aliases = AliasMap()
    _payload(ir, "cloud", aliases=aliases)
    result = accept({"ref": "@C999", "text": "calls @F888 then @F1"}, aliases)
    assert result.errors == ["$.ref: unknown reference @C999", "$.text: unknown reference @F888"]


# -- audit log ----------------------------------------------------------------

def test_egress_log(ir, tmp_path):
    p = _payload(ir, "cloud")
    log = EgressLog(str(tmp_path / ".archdoc" / "egress.jsonl"), include_payload=False)
    log.record(p, backend="claude", url="https://api.example", diagram="sequence")
    entry = json.loads((tmp_path / ".archdoc" / "egress.jsonl").read_text())
    assert entry["tier"] == "cloud" and entry["sha256"] == p.sha256 and entry["diagram"] == "sequence"
    assert "payload" not in entry


def test_cloud_annotation_literals_and_internal_type_aliases(tmp_path):
    # regressions found on pip: Literal values and alias-qualified internal type aliases
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_types.py").write_text("from typing import Callable\nHookKind = Callable[[], None]\n")
    (pkg / "api.py").write_text(
        "from typing import Literal\nfrom . import _types as _t\n\n"
        "def run(mode: Literal['turbo_secret_mode'], hook: _t.HookKind) -> 'Engine':\n    pass\n\n"
        "class Engine:\n    pass\n")
    p = _payload(extract_python(str(tmp_path)), "cloud")
    assert _word_hits(p.data, ["turbo_secret_mode", "HookKind", "_types", "Engine", "pkg"]) == []
    assert "Literal[@L" in p.text
