"""IDD and SDD templates, embedded diagrams, pandoc export (4d)."""

import os
import re

import pytest

import archdoc.cli as cli
from archdoc.design import build_design_context
from archdoc.export import figure_markdown, figures, pandoc
from archdoc.extract import extract_python
from archdoc.render.design import render_design
from archdoc.render.icd import build_icd_context, render_icd
from archdoc.validate import plantuml_command

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _diagram(tmp_path, kind, about=(), title=None):
    p = tmp_path / "diagrams" / f"x_{kind}.puml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(f"@startuml {kind}\nBob -> Alice\n@enduml\n")
    return {"title": title or f"{kind} diagram", "path": str(p), "kind": kind, "about": list(about)}


def test_figures_modes(tmp_path):
    d = _diagram(tmp_path, "component")
    img = str(tmp_path / "diagrams" / "x_component.svg")
    open(img, "w").write("<svg/>")
    doc_dir = str(tmp_path / "icd")
    (svg,) = figures([d], doc_dir, "svg", {d["path"]: img})
    assert figure_markdown(svg).startswith("![component diagram](../diagrams/x_component.svg)")
    (fallback,) = figures([d], doc_dir, "svg", {})                 # no PlantUML: the source instead
    assert "```plantuml\n@startuml component" in figure_markdown(fallback)
    (link,) = figures([d], doc_dir, "link")
    assert figure_markdown(link) == "Figure: [component diagram](../diagrams/x_component.puml)"


def test_idd_layout_and_per_interface_figures(ir, tmp_path):
    ds = [_diagram(tmp_path, "component"), _diagram(tmp_path, "sequence", ["shop.api:read_item"], "read item flow")]
    ctx = build_icd_context(ir, diagrams=figures(ds, str(tmp_path / "icd"), "link"))
    text = render_icd(ctx, "idd")
    for heading in ("## 1. Scope", "### 1.2 System overview", "## 3. Interface design",
                    "### 3.1 Interface identification and diagrams", "## 4. Data element assemblies",
                    "## 5. Requirements traceability", "## Appendix A. Open items"):
        assert heading in text
    assert "**c. Characteristics of individual data elements.**" in text and "**f. Protocol characteristics.**" in text
    # the sequence figure sits in the section of the interface that owns read_item, after the component one
    router = text.index("shop.api.router (")
    assert text.index("Figure: [read item flow]") > router > text.index("Figure: [component diagram]")
    assert text.count("Figure: [read item flow]") == 1
    assert "| `item_id` | path |" in text


def test_idd_system_overview(ir):
    ctx = build_icd_context(ir)
    ctx["system_overview"] = "A small shop."
    assert "A small shop. *(generated)*" in render_icd(ctx, "idd")


def test_sdd_layout(ir, tmp_path):
    ctx = build_design_context(ir, ir, FIXTURE)
    ctx["diagrams"] = figures([_diagram(tmp_path, "component"), _diagram(tmp_path, "activity", ["shop.api:read_item"]),
                               _diagram(tmp_path, "deployment")], str(tmp_path / "design"), "link")
    text = render_design(ctx, "sdd")
    order = ["## 1. Scope", "## 3. CSCI-wide design decisions", "Figure: [deployment diagram]",
             "### 4.1 CSCI components", "Figure: [component diagram]", "### 4.2 Concept of execution",
             "Figure: [activity diagram]", "### 4.3 Interface design", "## 5. CSCI detailed design",
             "## 6. Requirements traceability", "## Appendix A."]
    pos = [text.index(x) for x in order]
    assert pos == sorted(pos)
    assert re.search(r"#### 5\.\d+\.\d+ `ItemRepo.get`", text)
    assert "Functions without a specification:" in text


def test_cli_all_documents_without_model(tmp_path, monkeypatch):
    out = tmp_path / "out"
    rc = cli.main(["all", FIXTURE, "-o", str(out), "--no-date", "--no-validate", "--design", "--template", "idd",
                   "--design-template", "sdd", "--embed", "source"])
    assert rc == 0
    icd = (out / "icd" / "sample_app_icd.md").read_text()
    design = (out / "design" / "sample_app_design.md").read_text()
    assert "Interface Design Description (IDD)" in icd and "```plantuml" in icd
    assert "Software Design Description (SDD)" in design and "```plantuml\n@startuml sample_app_usecase" in design
    kinds = {f[len("sample_app_"):-len(".puml")].split("_")[0] for f in os.listdir(out / "diagrams")
             if f.endswith(".puml")}
    assert {"class", "component", "state", "object", "usecase", "activity"} <= kinds


@pytest.mark.skipif(pandoc() is None, reason="pandoc not installed")
def test_export_docx_and_html(tmp_path):
    out = tmp_path / "out"
    assert cli.main(["design", FIXTURE, "-o", str(out), "--no-date", "--export", "docx,html"]) == 0
    assert (out / "design" / "sample_app_design.docx").stat().st_size > 5000
    html = (out / "design" / "sample_app_design.html").read_text()
    assert '<h1 id="sample_app-design-description">sample_app Design Description</h1>' in html
    assert html.count("<h1") == 1
    assert cli.main(["export", str(out), "--to", "html"]) == 0


@pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not available")
def test_cli_embeds_rendered_svg(tmp_path):
    out = tmp_path / "out"
    assert cli.main(["all", FIXTURE, "-o", str(out), "--no-date", "--template", "idd", "--embed", "svg"]) == 0
    icd = (out / "icd" / "sample_app_icd.md").read_text()
    imgs = re.findall(r"!\[[^\]]*\]\((\.\./diagrams/[^)]+\.svg)\)", icd)
    assert imgs and all((out / "icd" / p).resolve().exists() for p in imgs)


@pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not available")
def test_rendered_svg_for_a_hyphenated_repo_name(tmp_path):
    # PlantUML names the image after `@startuml <name>`, which has no hyphens
    import shutil
    repo = tmp_path / "sample-app"
    shutil.copytree(FIXTURE, repo)
    out = tmp_path / "out"
    assert cli.main(["all", str(repo), "-o", str(out), "--no-date", "--embed", "svg"]) == 0
    pumls = sorted((out / "diagrams").glob("*.puml"))
    assert pumls and all(p.with_suffix(".svg").exists() for p in pumls)
    assert not list((out / "diagrams").glob("sample_app_*.svg"))
