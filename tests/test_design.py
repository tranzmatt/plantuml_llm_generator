"""
Rebuild content (4c): response cache, behavior specifications, hierarchical
summaries, dependency manifest, test inventory, design description.
Model-free (ScriptedBackend).
"""

import hashlib
import json
import os
import shutil
import textwrap

import pytest

from archdoc.backends import ScriptedBackend
from archdoc.cache import ResponseCache
from archdoc.design import DesignOptions, behavior_targets, build_design_context, design_items
from archdoc.extract import extract_python
from archdoc.manifest import build_manifest
from archdoc.overlap import SourceIndex
from archdoc.plan.artifacts import PlanContext, plan_behavior, plan_module_summary
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap
from archdoc.render.design import render_design
from archdoc.testsuite import build_inventory, is_test_path, inventory_excludes

from tests.test_egress import BODY_ONLY, CLOUD_FORBIDDEN, SECRET, _word_hits

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")
GET = "shop.repo:ItemRepo.get"


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def auto_answer(schema, text="does the thing", i=0):
    """The smallest valid answer to any planner schema: one item per array (or
    minItems, each with the next enum value)."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][min(i, len(schema["enum"]) - 1)]
    if t == "object":
        return {k: auto_answer(v, text, i) for k, v in schema["properties"].items()}
    if t == "array":
        n = 0 if schema.get("maxItems") == 0 else max(1, schema.get("minItems", schema.get("maxItems", 1)))
        return [auto_answer(schema["items"], text, j) for j in range(n)]
    if t == "boolean":
        return False
    if t == "integer":
        return 0
    if schema.get("maxLength") == 0:
        return ""
    return text


def responder(system, user, schema):
    return json.dumps(auto_answer(schema))


def _ctx(ir, tier="local", respond=responder, cache=None, root=FIXTURE):
    return PlanContext(ir=ir, config=Config(), tier=tier, aliases=AliasMap(),
                       backend=ScriptedBackend(respond), root=root, cache=cache)


def _payload_text(call):
    return call["user"].split("INPUT:\n", 1)[1]


# -- response cache ------------------------------------------------------------------

def test_rerun_makes_zero_model_calls(ir, tmp_path):
    cache = ResponseCache(str(tmp_path / "cache"))
    first = _ctx(ir, cache=cache)
    a = build_design_context(ir, ir, FIXTURE, first)
    sent = len(first.backend.calls)
    assert sent > 0 and cache.misses == sent

    again = _ctx(ir, cache=ResponseCache(str(tmp_path / "cache")))
    b = build_design_context(ir, ir, FIXTURE, again)
    assert again.backend.calls == []                              # exit criterion: nothing sent
    assert again.cache.hits == sent
    assert render_design(a) == render_design(b)


def varying(system, user, schema):
    """Summaries that differ whenever their input differs, so changes propagate upward."""
    a = auto_answer(schema)
    if "responsibility" in a:
        a["responsibility"] = "R" + hashlib.sha256(user.encode()).hexdigest()[:8]
    return json.dumps(a)


def test_cache_misses_only_what_changed(tmp_path):
    repo = tmp_path / "sample_app"
    shutil.copytree(FIXTURE, repo)
    ir1 = extract_python(str(repo))
    cache_dir = str(tmp_path / "cache")
    first = _ctx(ir1, respond=varying, cache=ResponseCache(cache_dir), root=str(repo))
    build_design_context(ir1, ir1, str(repo), first, DesignOptions(tests=False))

    cli = repo / "src" / "shop" / "cli.py"
    cli.write_text(cli.read_text() + "\n\ndef extra(flag: bool) -> int:\n    return 1 if flag else 0\n")
    ir2 = extract_python(str(repo))
    second = _ctx(ir2, respond=varying, cache=ResponseCache(cache_dir), root=str(repo))
    build_design_context(ir2, ir2, str(repo), second, DesignOptions(tests=False))
    sent = [_payload_text(c) for c in second.backend.calls]
    # the new function's spec, its module's summary, and the package summary built on it;
    # every other module, function and summary came from the cache
    assert len(sent) == 3
    assert any("target: extra" in t for t in sent)
    assert any("module:" in t and "def extra(" in t for t in sent)
    assert any("package:" in t for t in sent)
    assert second.cache.hits == len(first.backend.calls) - 2


def test_cache_disabled_always_asks(ir, tmp_path):
    for _ in range(2):
        ctx = _ctx(ir, cache=ResponseCache(str(tmp_path / "c"), enabled=False))
        plan_behavior(ctx, GET)
        assert len(ctx.backend.calls) == 1


def test_cached_answer_is_still_validated(ir, tmp_path):
    """The cache replays answers, not verdicts: a cached answer goes through every check."""
    cache = ResponseCache(str(tmp_path / "c"))
    ctx = _ctx(ir, cache=cache)
    plan_behavior(ctx, GET)
    for d in os.listdir(cache.path):
        for f in os.listdir(os.path.join(cache.path, d)):
            p = os.path.join(cache.path, d, f)
            entry = json.load(open(p))
            entry["text"] = "not json"
            json.dump(entry, open(p, "w"))
    again = _ctx(ir, cache=ResponseCache(str(tmp_path / "c")))
    res = plan_behavior(again, GET)
    assert res.ok and len(again.backend.calls) == 1               # repair round asked the model


# -- behavior specifications -------------------------------------------------------

def test_behavior_from_source_locally(ir):
    ctx = _ctx(ir)
    res = plan_behavior(ctx, GET)
    assert res.ok and res.report["evidence"] == "source"
    sent = _payload_text(ctx.backend.calls[0])
    assert "raise KeyError(item_id)" in sent                      # the target's own source
    schema = ctx.backend.calls[0]["schema"]
    assert schema["properties"]["errors"]["items"]["properties"]["ref"]["enum"] == ["KeyError"]
    assert schema["properties"]["inputs"]["items"]["properties"]["name"]["enum"] == ["item_id"]
    calls = schema["properties"]["side_effects"]["items"]["properties"]["ref"]["enum"]
    assert "query" in " ".join(calls)


def test_behavior_error_the_code_cannot_raise_is_rejected(ir):
    def answer(system, user, schema):
        a = auto_answer(schema)
        a["errors"] = [{"ref": "ValueError", "when": "always"}]
        return json.dumps(a)

    ctx = _ctx(ir, respond=answer)
    res = plan_behavior(ctx, GET)
    assert not res.ok and res.attempts == 2
    assert any("ValueError" in e and "not a ref from the input" in e for e in res.errors)


def test_behavior_cloud_is_signature_only_and_leaks_nothing(ir):
    ctx = _ctx(ir, tier="cloud")
    for fid in behavior_targets(ir):
        res = plan_behavior(ctx, fid)
        assert res.ok and res.report["evidence"] == "stub", fid
    for call in ctx.backend.calls:
        text = _payload_text(call)
        assert [c for c in BODY_ONLY + [SECRET] if c in text] == []
    payload = "\n".join(_payload_text(c) for c in ctx.backend.calls)
    assert _word_hits({"t": payload}, CLOUD_FORBIDDEN) == []


def test_behavior_overlap_is_cleared(ir):
    copied = "row self db query canary_sql_a41c if not row raise keyerror item_id"

    def answer(system, user, schema):
        a = auto_answer(schema)
        a["algorithm"] = ["Look the item up.", copied]
        return json.dumps(a)

    ctx = _ctx(ir, respond=answer)
    ctx.overlap = SourceIndex(FIXTURE, ir)
    res = plan_behavior(ctx, GET)
    assert res.ok and res.plan["algorithm"] == ["Look the item up.", ""]
    assert "copied" not in json.dumps(ctx.backend.calls[1]["user"].split("REJECTED")[1])
    assert "CANARY_SQL" not in ctx.backend.calls[1]["user"].split("REJECTED")[1]
    assert any("cleared algorithm[1]" in w for w in res.warnings)


# -- summaries ---------------------------------------------------------------------------

def test_module_summary_collaborators_are_import_edges(ir):
    ctx = _ctx(ir)
    plan_module_summary(ctx, "shop.api")
    props = ctx.backend.calls[0]["schema"]["properties"]
    collab = props["collaborators"]["items"]["properties"]["ref"]["enum"]
    external = props["external"]["items"]["properties"]["ref"]["enum"]
    assert set(collab) == {"models", "repo"} or set(collab) == {"shop.models", "shop.repo"}
    assert external == ["ext:fastapi"]                            # os is the standard library


def test_hierarchy_feeds_module_text_upward(ir):
    ctx = _ctx(ir, respond=varying)
    out = build_design_context(ir, ir, FIXTURE, ctx, DesignOptions(behavior=False, tests=False))
    pkg = next(_payload_text(c) for c in ctx.backend.calls if "package:" in _payload_text(c))
    assert all(m["summary"]["responsibility"] in pkg for m in out["modules"])
    assert out["system"] is not None                              # one package: its summary is the system's


# -- manifest ---------------------------------------------------------------------------

def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))


def test_manifest_declared_used_and_scoped(tmp_path):
    _write(tmp_path, "pyproject.toml", """
        [project]
        name = "x"
        dependencies = ["requests>=2", "PyYAML"]
        [project.optional-dependencies]
        dev = ["pytest"]
    """)
    _write(tmp_path, "app/main.py", "import requests\nimport yaml\nimport numpy\nimport os\n")
    _write(tmp_path, "svc/Dockerfile", "FROM python\nRUN pip install --no-cache-dir pika \\\n    pillow==10.0\n")
    _write(tmp_path, "svc/worker.py", "import pika\nimport requests\n")
    ir = extract_python(str(tmp_path))
    man = build_manifest(str(tmp_path), ir)
    names = {(r.name, r.spec, r.file, r.group) for r in man.requirements}
    assert ("requests", ">=2", "pyproject.toml", "") in names and ("pytest", "", "pyproject.toml", "dev") in names
    assert ("pillow", "==10.0", "svc/Dockerfile", "") in names
    issues = {(i["kind"], i["package"]) for i in man.issues}
    assert ("undeclared", "numpy") in issues                      # imported, declared nowhere
    assert ("unused", "pillow") in issues                         # svc installs it, svc never imports it
    assert ("unused", "pytest") in issues and ("undeclared", "yaml") not in issues   # yaml is PyYAML
    assert ("undeclared", "requests") not in issues               # the root pyproject covers svc/ too
    assert "os" not in man.imports and man.imports["pika"] == ["svc.worker"]


# -- test inventory -----------------------------------------------------------------------

def test_inventory_from_excluded_tests(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "app")
    _write(tmp_path / "app", "tests/test_repo.py", '''
        import pytest
        from shop.repo import ItemRepo, Database

        @pytest.mark.slow
        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_get_missing(n, tmp_path):
            """Unknown ids fail."""
            with pytest.raises(KeyError):
                ItemRepo(Database()).get(n)

        class TestThing:
            def test_ok(self):
                assert True
                assert 1 == 1
    ''')
    root = str(tmp_path / "app")
    documented = extract_python(root, exclude=["tests/*"])
    ir_all = extract_python(root, exclude=inventory_excludes(["tests/*"]))
    inv = {t.name: t for t in build_inventory(root, ir_all, documented)}
    t = inv["test_get_missing"]
    assert t.expects == ["KeyError"] and t.cases == 3 and t.fixtures == ["tmp_path"] and t.markers == ["slow"]
    assert "shop.repo:ItemRepo.get" in t.exercises and t.doc == "Unknown ids fail." and t.line == 7
    assert inv["test_ok"].asserts == 2 and inv["test_ok"].cls == "TestThing"
    assert is_test_path("tests/test_repo.py") and not is_test_path("tests/conftest.py")

    ctx = build_design_context(documented, ir_all, root)
    get = next(f for m in ctx["modules"] for f in m["functions"] if f["id"] == "shop.repo:ItemRepo.get")
    assert get["tested_by"] == ["tests.test_repo:test_get_missing"]
    assert ctx["coverage"]["tests"]["functions_exercised"] >= 1

    # criteria: the tests are sent (from the full IR) though the documents exclude them
    llm = _ctx(documented, root=root)
    out = build_design_context(documented, ir_all, root, llm, DesignOptions(behavior=False, summaries=False))
    sent = _payload_text(llm.backend.calls[0])
    assert "pytest.raises(KeyError)" in sent and "def test_ok(self)" in sent
    assert {t["name"]: t["criterion"] for t in out["tests"]} == {"test_get_missing": "does the thing",
                                                                 "test_ok": "does the thing"}
    assert out["coverage"]["tests"]["with_criterion"] == 2


def test_criteria_must_cover_every_test(ir):
    from archdoc.plan.artifacts import check_tests
    plan = {"tests": [{"ref": "m:test_a", "criterion": "x"}]}
    assert check_tests(plan, ["m:test_a", "m:test_b"]) == ["no criterion for: m:test_b"]


# -- document ---------------------------------------------------------------------------------

def test_design_document_labels_evidence(ir):
    local = build_design_context(ir, ir, FIXTURE, _ctx(ir), DesignOptions(tests=False))
    cloud = build_design_context(ir, ir, FIXTURE, _ctx(ir, tier="cloud"), DesignOptions(tests=False))
    assert "*(generated from source)*" in render_design(local)
    text = render_design(cloud)
    assert "inferred from signature" in text and "from source)*" not in text
    cov = cloud["coverage"]["behavior"]
    assert cov["signature"] + cov["source"] == cov["total"] and cov["none"] == 0
    assert cov["source"] == 9                                     # the trivial ones, from the AST
    items = design_items(local)
    assert {i["class"] for i in items} == {"derived", "fact"}
    spec = next(i for i in items if i["id"] == f"behavior:{GET}")
    assert spec["detail"] == "source" and GET in spec["evidence"]


def test_design_without_model(ir):
    ctx = build_design_context(ir, ir, FIXTURE)
    text = render_design(ctx)
    cov = ctx["coverage"]["behavior"]
    assert "Behavior: **TBD**" in text and cov["none"] == cov["total"] - 9
    assert all(i["class"] in ("open", "fact") for i in design_items(ctx))


def test_trivial_functions_are_specified_from_the_code(ir):
    """Small models invent behavior for empty or pass-through bodies; these need no model."""
    ctx = _ctx(ir)
    out = build_design_context(ir, ir, FIXTURE, ctx, DesignOptions(summaries=False, tests=False))
    rows = {f["id"]: f for m in out["modules"] for f in m["functions"]}
    assert rows["shop.click_cli:seed_data"]["purpose"].startswith("Has no behavior of its own")
    assert rows["shop.api:create_item"]["purpose"] == "Returns its argument `item` unchanged."
    assert rows["shop.repo:Database.query"]["purpose"] == "Always returns an empty dict."
    assert rows["shop.repo:CachedRepo.get"]["purpose"] == \
        "Calls `shop.repo:ItemRepo.get` with `item_id` and returns its result."
    assert not rows["shop.repo:Database.query"]["generated"]
    asked = [c["user"].split("target: ", 1)[1].split("\n", 1)[0] for c in ctx.backend.calls]
    assert "seed_data" not in asked and "get" in asked and len(asked) == 18 - 9
    assert "*(from the code)*" in render_design(out)


SQL_APP = """
import os

def lookup(db, item_id: int):
    \"\"\"Find one item.\"\"\"
    url = "https://api.example.com/v1/items"
    row = db.execute("SELECT name FROM items WHERE id = ?", (item_id,))
    if row is None:
        raise KeyError("no such item")
    return {"item_name": row[0], "region": os.environ.get("APP_REGION")}
"""


@pytest.fixture
def sql_app(tmp_path):
    _write(tmp_path, "app/store.py", SQL_APP)
    return str(tmp_path), extract_python(str(tmp_path))


def test_quoted_literal_is_an_overlap_but_names_and_urls_are_not(sql_app):
    root, ir = sql_app
    index = SourceIndex(root, ir)
    hit = index.check("Runs SELECT name FROM items WHERE id = ? against the database.")
    assert hit and hit.literal and "SELECT" not in hit.describe() and hit.path == "app/store.py"
    assert index.check("Returns item_name and region; reads APP_REGION.") is None     # names
    assert index.check("Calls https://api.example.com/v1/items.") is None              # where, not how
    assert index.check("Fails with no such item.") is None                             # everyday phrase
    assert index.check("Selects the item's name by its id.") is None


def test_behavior_literal_is_repaired_without_quoting_it(sql_app):
    root, ir = sql_app

    def answer(system, user, schema):
        a = auto_answer(schema)
        if "YOUR PREVIOUS ANSWER" not in user:
            a["algorithm"] = ["Execute SELECT name FROM items WHERE id = ? with the id."]
        return json.dumps(a)

    ctx = _ctx(ir, respond=answer, root=root)
    ctx.overlap = SourceIndex(root, ir)
    res = plan_behavior(ctx, "app.store:lookup")
    assert res.ok and res.attempts == 2 and res.plan["algorithm"] == ["does the thing"]
    reasons = ctx.backend.calls[1]["user"].split("REJECTED FOR THESE REASONS:")[1]
    assert "quotes a string literal" in reasons and "SELECT" not in reasons


def test_duplicate_entries_are_merged_not_rejected(ir):
    def answer(system, user, schema):
        a = auto_answer(schema)
        a["external"] = [{"ref": "ext:pydantic", "role": "models"}, {"ref": "ext:pydantic", "role": "validation"}]
        return json.dumps(a)

    res = plan_module_summary(_ctx(ir, respond=answer), "shop.models")
    assert res.ok and res.attempts == 1
    assert res.plan["external"] == [{"ref": "ext:pydantic", "role": "models; validation"}]


def test_cloud_hierarchy_passes_the_aliased_wording_upward(ir):
    """Module summaries are restored to real names for the document, but the
    package request gets them as the model wrote them, aliases intact."""
    import re as _re

    def answer(system, user, schema):
        a = auto_answer(schema)
        refs = _re.findall(r"ref: (@M\d+)", user.split("module:", 1)[-1])
        if "responsibility" in a and refs:
            a["responsibility"] = f"Handles {refs[0]} requests."
        return json.dumps(a)

    ctx = _ctx(ir, tier="cloud", respond=answer)
    out = build_design_context(ir, ir, FIXTURE, ctx, DesignOptions(behavior=False, tests=False))
    assert "shop" in out["modules"][1]["summary"]["responsibility"]          # restored for the reader
    pkg = next(_payload_text(c) for c in ctx.backend.calls if "package:" in _payload_text(c))
    assert "Handles @M" in pkg and _word_hits({"t": pkg}, CLOUD_FORBIDDEN) == []


def test_large_test_modules_are_batched(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "app")
    body = "\n".join(f"def test_case_{i}():\n    assert {i} == {i}\n" for i in range(30))
    _write(tmp_path / "app", "tests/test_many.py", body)
    root = str(tmp_path / "app")
    documented = extract_python(root, exclude=["tests/*"])
    ir_all = extract_python(root)
    llm = _ctx(documented, root=root)
    out = build_design_context(documented, ir_all, root, llm, DesignOptions(behavior=False, summaries=False))
    assert len(llm.backend.calls) == 3                            # 12 + 12 + 6
    assert out["coverage"]["tests"]["with_criterion"] == 30
    reqs = {i["request"] for i in design_items(out) if i["id"].startswith("criterion:")}
    assert reqs == {0, 1, 2}


def test_partial_criteria_are_kept(tmp_path):
    from archdoc.plan.artifacts import plan_tests
    shutil.copytree(FIXTURE, tmp_path / "app")
    _write(tmp_path / "app", "tests/test_two.py", "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n")
    root = str(tmp_path / "app")
    ir_all = extract_python(root)

    def lazy(system, user, schema):                  # always leaves test_b out
        return json.dumps({"tests": [{"ref": schema["properties"]["tests"]["items"]["properties"]["ref"]["enum"][0],
                                      "criterion": "A holds."}]})

    res = plan_tests(_ctx(ir_all, respond=lazy, root=root), "tests.test_two",
                     ["tests.test_two:test_a", "tests.test_two:test_b"], [], ir_all)
    assert res.ok and res.attempts == 2 and [t["ref"] for t in res.plan["tests"]] == ["tests.test_two:test_a"]
    assert res.warnings == ["no criterion for: tests.test_two:test_b"]
