"""
Rendered diagrams and document export (DESIGN.md 10.2).

  render_images  .puml -> .svg / .png beside them, one PlantUML run for all
  figures        what a document embeds for each diagram: the image, the
                 PlantUML source, or a link, relative to the document
  export         Markdown -> docx / html / pdf with pandoc (optional): docx
                 and pdf get PNG figures, html keeps SVG

Both tools are optional: without PlantUML, documents embed the diagram
source; without pandoc, export reports what is missing and writes nothing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional

from .validate import plantuml_command

EMBED_MODES = ("svg", "png", "source", "link")
EXPORT_FORMATS = ("docx", "html", "pdf")
PDF_ENGINES = ("xelatex", "lualatex", "pdflatex", "wkhtmltopdf", "weasyprint")


def render_images(puml_paths: List[str], fmt: str = "svg", timeout: int = 600) -> Dict[str, str]:
    """{puml path: image path} for the diagrams PlantUML rendered (empty without PlantUML)."""
    cmd = plantuml_command()
    if not cmd or not puml_paths:
        return {}
    proc = subprocess.run(cmd + [f"-t{fmt}", "-charset", "UTF-8"] + puml_paths, capture_output=True, text=True,
                          timeout=timeout)
    out = {}
    for p in puml_paths:
        img = p[: -len(".puml")] + f".{fmt}"
        if os.path.exists(img) and os.path.getmtime(img) >= os.path.getmtime(p) - 1:
            out[p] = img
    if proc.returncode and not out:
        raise RuntimeError(f"PlantUML failed: {proc.stderr.strip()[:300]}")
    return out


def figures(diagrams: List[Dict[str, object]], doc_dir: str, mode: str,
            images: Optional[Dict[str, str]] = None) -> List[Dict[str, object]]:
    """Diagram entries as a document in `doc_dir` refers to them."""
    out = []
    for d in diagrams:
        path = str(d["path"])
        img = (images or {}).get(path)
        source = None
        if mode == "source" or (mode in ("svg", "png") and img is None):
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
        out.append({**d, "path": os.path.relpath(path, doc_dir),
                    "image": os.path.relpath(img, doc_dir) if img and mode in ("svg", "png") else None,
                    "source": source, "mode": mode})
    return out


def figure_markdown(d: Dict[str, object]) -> str:
    """Markdown for one figure: image, PlantUML source, or link, with its title."""
    title = str(d.get("title", "")).replace("\n", " ")
    if d.get("image"):
        return f"![{title}]({d['image']})\n\n*Figure: {title}*"
    if d.get("source") and d.get("mode") != "link":
        return f"*Figure: {title}*\n\n```plantuml\n{str(d['source']).rstrip()}\n```"
    return f"Figure: [{title}]({d['path']})"


def pandoc() -> Optional[str]:
    return shutil.which("pandoc")


def pdf_engine() -> Optional[str]:
    return next((e for e in PDF_ENGINES if shutil.which(e)), None)


def export(markdown_path: str, fmt: str, png: bool = False) -> str:
    """Convert one Markdown document. For docx/pdf, SVG figures are swapped for
    PNGs rendered beside them (`png`): Word and LaTeX handle SVG poorly."""
    exe = pandoc()
    if exe is None:
        raise RuntimeError("pandoc not found: install it (https://pandoc.org) to export documents")
    doc_dir = os.path.dirname(os.path.abspath(markdown_path))
    with open(markdown_path, encoding="utf-8") as fh:
        text = fh.read()
    if png:
        text = re.sub(r"(\]\()([^)]+)\.svg\)", lambda m: m.group(1) + m.group(2) + ".png)" if os.path.exists(
            os.path.join(doc_dir, m.group(2) + ".png")) else m.group(0), text)
    out = os.path.splitext(markdown_path)[0] + f".{fmt}"
    heading = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")),
                   os.path.basename(markdown_path)[:-3])
    cmd = [exe, "-f", "gfm", "-o", out, "--resource-path", doc_dir, "--standalone", "--wrap=none",
           "--metadata", f"pagetitle={heading}"]
    if fmt == "html":
        cmd += ["--embed-resources"]
    if fmt == "pdf":
        engine = pdf_engine()
        if engine is None:
            raise RuntimeError(f"no PDF engine found (pandoc needs one of: {', '.join(PDF_ENGINES)})")
        cmd += [f"--pdf-engine={engine}", "-V", "geometry:margin=2cm"]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=doc_dir, timeout=600)
    if proc.returncode:
        raise RuntimeError(f"pandoc failed for {markdown_path}: {proc.stderr.strip()[:400]}")
    return out


__all__ = ["EMBED_MODES", "EXPORT_FORMATS", "export", "figure_markdown", "figures", "pandoc", "pdf_engine",
           "render_images"]
