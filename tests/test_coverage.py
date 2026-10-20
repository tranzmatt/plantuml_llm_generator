"""Coverage and known unknowns: missing interfaces must never be silent."""

import json
import os

import pytest

from archdoc.cli import main
from archdoc.coverage import icd_coverage, known_unknowns
from archdoc.extract import extract_python
from archdoc.render.icd import build_icd_context, render_icd

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def test_fixture_known_unknowns(ir):
    # FastAPI, argparse, Click and Typer interfaces were extracted, so are not flagged;
    # shop.repo's outbound HTTP calls (requests) were not
    assert [(u.module, u.library, u.kind, u.direction) for u in known_unknowns(ir)] == [
        ("shop.repo", "requests", "http", "required")]


def test_flags_each_technology_once_per_module(tmp_path):
    (tmp_path / "svc.py").write_text(
        "import pika\nfrom pika import BlockingConnection\nimport urllib.request\nimport gradio as gr\n")
    found = {(u.library, u.kind) for u in known_unknowns(extract_python(str(tmp_path)))}
    assert found == {("pika", "message"), ("urllib.request", "http"), ("gradio", "ui")}


def test_extracted_kind_suppresses_flag(tmp_path):
    (tmp_path / "cli.py").write_text(
        "import argparse\n\ndef main():\n    p = argparse.ArgumentParser()\n    p.add_argument('--x')\n")
    (tmp_path / "helper.py").write_text("import argparse\n")      # imports argparse, builds no parser
    flagged = [(u.module, u.library) for u in known_unknowns(extract_python(str(tmp_path)))]
    assert flagged == [("helper", "argparse")]


def test_icd_coverage_counts(ir):
    ctx = build_icd_context(ir)
    cov = icd_coverage(ctx, ir)
    assert cov["operations"] == {"total": 11, "from_code": 5, "generated": 0, "tbd": 6}
    assert cov["interfaces"]["by_kind"] == {"cli": 4, "env": 1, "http": 2}
    assert cov["open_items"] == len(ctx["gaps"]) and len(cov["known_unknowns"]) == 1


def test_icd_says_it_is_incomplete(ir):
    ctx = build_icd_context(ir)
    ctx["coverage"] = icd_coverage(ctx, ir)
    text = render_icd(ctx)
    assert "**Incomplete:** 1 use(s) of interface" in text
    assert "### A.2 Interfaces not extracted" in text and "| `shop.repo` | outbound HTTP (requests)" in text
    assert "## Appendix C. Coverage" in text


def test_cli_writes_coverage(tmp_path, capsys):
    assert main(["icd", FIXTURE, "-o", str(tmp_path), "--no-date"]) == 0
    cov = json.loads((tmp_path / "coverage.json").read_text())
    assert cov["known_unknowns"][0]["module"] == "shop.repo"
    assert "[GAP] 1 interface technolog(ies) used but not extracted" in capsys.readouterr().out
