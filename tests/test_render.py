"""
Deterministic renderers: PlantUML class/component diagrams and ICD Markdown.

Golden files live in tests/golden/. To accept intentional output changes:
    ARCHDOC_UPDATE_GOLDEN=1 python -m pytest tests/test_render.py
PlantUML syntax checks run when PlantUML is available (PLANTUML_JAR or
`plantuml` on PATH) and are skipped otherwise.
"""

import os
import re

import pytest

from archdoc.cli import main
from archdoc.extract import extract_python
from archdoc.ir import Class, Module, RepoIR
from archdoc.render.icd import build_icd_context, render_icd
from archdoc.render.puml import render_class_diagram, render_component_diagram
from archdoc.render.puml.class_ import partition_modules
from archdoc.render.puml.common import Aliases, label
from archdoc.render.puml.component import auto_depth
from archdoc.validate import check_syntax, plantuml_command

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "sample_app")
GOLDEN = os.path.join(HERE, "golden")
needs_plantuml = pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not available")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _golden(name: str, text: str) -> None:
    path = os.path.join(GOLDEN, name)
    if os.environ.get("ARCHDOC_UPDATE_GOLDEN") or not os.path.exists(path):
        os.makedirs(GOLDEN, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(path, encoding="utf-8") as fh:
        assert text == fh.read(), f"{name} differs from golden (ARCHDOC_UPDATE_GOLDEN=1 to accept)"


# -- common -------------------------------------------------------------------

def test_aliases_are_safe_and_unique():
    a = Aliases()
    assert a("shop.models:Item", "C") == "C_shop_models_Item"
    assert a("shop.models.Item", "C") == "C_shop_models_Item_2"     # collides after sanitizing
    assert a("shop.models:Item", "C") == "C_shop_models_Item"       # stable
    assert re.fullmatch(r"[A-Za-z0-9_]+", a("weird name/with\"chars", "X"))


def test_label_neutralizes_quotes_and_newlines():
    assert label('say "hi"\nthere') == "\"say 'hi' there\""


# -- class diagram --------------------------------------------------------------

def test_class_diagram_golden(ir):
    _golden("sample_app_class.puml", render_class_diagram(ir, "sample_app_class"))


def test_class_diagram_relationships(ir):
    text = render_class_diagram(ir, "x")
    assert "C_shop_repo_ItemRepo <|-- C_shop_repo_CachedRepo" in text
    assert 'C_shop_models_Order *-- "*" C_shop_models_Item : items' in text
    assert "C_shop_repo_ItemRepo --> C_shop_repo_Database : db" in text
    assert "C_shop_repo_ItemRepo ..> C_shop_models_Item : creates" in text
    assert 'enum "Status" as C_shop_models_Status {' in text
    assert "<<pydantic>>" in text and "<<dataclass>>" in text


def test_class_diagram_scope_draws_related_classes_as_stubs(ir):
    text = render_class_diagram(ir, "x", scope="shop.repo")
    assert 'class "ItemRepo" as C_shop_repo_ItemRepo {' in text          # in scope: with body
    assert re.search(r'^  class "Item" as C_shop_models_Item <<pydantic>>$', text, re.M)  # stub
    assert "+price" not in text


def test_members_all_shows_protected(ir):
    assert "#_cache" not in render_class_diagram(ir, "x")
    assert "#_cache" in render_class_diagram(ir, "x", members="all")


def _ir_with(counts):
    ir = RepoIR(repo="r")
    for mid, n in counts.items():
        m = Module(id=mid, path=mid.replace(".", "/") + ".py")
        m.classes = [Class(id=f"{mid}:K{i}", name=f"K{i}") for i in range(n)]
        ir.modules.append(m)
    return ir


def test_partition_splits_large_packages_and_packs_small_ones():
    ir = _ir_with({"big.a": 30, "big.b": 30, "big.c.x": 5, "big.c.y": 5, "big.d": 4, "small": 3, "empty": 0})
    parts = partition_modules(ir, max_classes=40)
    covered = [m for _, mods in parts for m in mods]
    assert sorted(covered) == sorted(m.id for m in ir.modules if m.classes)   # each once, no empties
    for _, mods in parts:
        assert sum(len(ir.module(m).classes) for m in mods) <= 40
    assert len(parts) < 6                                                     # small siblings packed


def test_partition_single_oversized_module_is_its_own_group():
    assert partition_modules(_ir_with({"huge": 99}), max_classes=10) == [("huge", {"huge"})]


# -- component diagram ----------------------------------------------------------

def test_component_diagram_golden(ir):
    _golden("sample_app_component.puml", render_component_diagram(ir, "sample_app_component"))


def test_component_diagram_interfaces(ir):
    text = render_component_diagram(ir, "x")
    assert 'interface "HTTP /v1/items" as I_if_http_shop_api_router' in text
    assert "I_if_http_shop_api_router -- C_shop_api" in text            # provided: lollipop
    assert "C_shop_api --( I_if_env_sample_app" in text                  # required: socket
    assert "C_shop_repo ..> X_ext_requests" in text
    assert "ext_os" not in text and "ext_argparse" not in text           # stdlib hidden


def test_auto_depth():
    assert auto_depth(["shop", "shop.api", "shop.repo"]) == 2
    assert auto_depth(["a", "b", "c"]) == 1
    many = [f"p.m{i}.x" for i in range(50)]
    assert auto_depth(many, limit=30) == 1


# -- PlantUML syntax ------------------------------------------------------------

@needs_plantuml
@pytest.mark.parametrize("render", [
    lambda ir: render_class_diagram(ir, "c"),
    lambda ir: render_class_diagram(ir, "c", members="all"),
    lambda ir: render_class_diagram(ir, "c", scope="shop.repo"),
    lambda ir: render_component_diagram(ir, "k"),
    lambda ir: render_component_diagram(ir, "k", depth=1),
])
def test_plantuml_accepts_output(ir, render):
    res = check_syntax(render(ir))
    assert res.ok, f"line {res.line}: {res.message}"


@needs_plantuml
def test_check_syntax_reports_errors():
    res = check_syntax("@startuml\nclass A\nA --> B :\n@enduml\n")
    assert not res.ok and res.line == 3


def test_check_syntax_without_plantuml(monkeypatch):
    monkeypatch.delenv("ARCHDOC_PLANTUML", raising=False)
    monkeypatch.delenv("PLANTUML_JAR", raising=False)
    monkeypatch.setenv("PATH", "")
    assert check_syntax("@startuml\n@enduml\n").available is False


# -- ICD ------------------------------------------------------------------------

def test_icd_golden(ir):
    _golden("sample_app_icd.md", render_icd(build_icd_context(ir)))


def test_icd_context_facts(ir):
    ctx = build_icd_context(ir)
    assert ctx["summary"] == {"interfaces": 7, "provided": 6, "required": 1, "operations": 11, "data_types": 2}
    # Status is reached only through Item.status: referenced transitively
    assert [d["id"] for d in ctx["data_types"]] == ["shop.models:Item", "shop.models:Status"]
    router = next(i for i in ctx["interfaces"] if i["id"] == "if:http:shop.api:router")
    get = router["operations"][0]
    assert get["response"]["anchor"] == "dt-shop-models-item"
    assert router["purpose"] is None and not get["summary_generated"]     # narrative comes later


def test_icd_gaps_cover_every_tbd(ir):
    ctx = build_icd_context(ir)
    text = render_icd(ctx)
    gaps = {(g["where"], g["what"]) for g in ctx["gaps"]}
    assert ("shop.api.router / POST /", "operation description") in gaps
    assert ("Environment / SHOP_SECRET", "default value (none in code: required?)") in gaps
    # every interface has a purpose gap, and the body has no raw None
    assert sum(1 for g in ctx["gaps"] if g["what"] == "interface purpose") == 7
    assert "None" not in text.replace("`None`", "")


def test_icd_escapes_table_cells(ir):
    ir2 = extract_python(FIXTURE)
    ir2.interfaces[0].operations[0].docstring = "a | b\nc"
    text = render_icd(build_icd_context(ir2))
    assert "a \\| b c" in text


def test_icd_custom_template(ir, tmp_path):
    tmpl = tmp_path / "mine.md.j2"
    tmpl.write_text("{{ title }}: {% for i in interfaces %}{{ i.name | code }} {% endfor %}\n")
    out = render_icd(build_icd_context(ir), str(tmpl))
    assert out.startswith("sample_app Interface Control Document: `shopctl`")


def test_icd_unknown_template(ir):
    with pytest.raises(ValueError):
        render_icd(build_icd_context(ir), "no-such-template")


# -- CLI end to end -------------------------------------------------------------

def test_cli_all(tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["all", FIXTURE, "-o", str(out), "--no-validate", "--no-date", "--types", "class,component,sequence"])
    assert rc == 0
    assert sorted(os.listdir(out / "diagrams")) == ["sample_app_class.puml", "sample_app_component.puml"]
    icd = (out / "icd" / "sample_app_icd.md").read_text()
    assert "](../diagrams/sample_app_class.puml)" in icd
    assert (out / ".archdoc" / "icd_snapshot.json").exists()
    assert "sequence: needs an LLM" in capsys.readouterr().out


def test_cli_icd_per_interface(tmp_path):
    out = tmp_path / "out"
    assert main(["icd", FIXTURE, "-o", str(out), "--scope", "interface", "--no-date"]) == 0
    assert len(os.listdir(out / "icd")) == 7


def test_cli_class_diagrams_split_when_large(tmp_path):
    out = tmp_path / "out"
    assert main(["diagrams", FIXTURE, "-o", str(out), "--types", "class", "--max-classes", "3",
                 "--no-validate"]) == 0
    files = sorted(os.listdir(out / "diagrams"))
    assert len(files) > 1 and all(f.startswith("sample_app_class_") for f in files)
