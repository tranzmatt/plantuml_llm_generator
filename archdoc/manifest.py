"""
Dependency manifest (DESIGN.md 2.3): the third-party packages a rebuild needs,
with the versions the repo pins, and where declaration and use disagree.

Declared: requirements*.txt, pyproject.toml ([project] dependencies and
optional dependencies, Poetry dependencies), setup.cfg install_requires, and
`pip install` lines in Dockerfiles. A file declares dependencies for the code
under its directory, so a service's requirements.txt covers only that
service.

Used: the third-party imports of each module (the standard library is not a
dependency). Import names are matched to distribution names after
normalization, with a table for the common cases where they differ
(`yaml` is PyYAML, `cv2` is opencv-python).

Cross-checks: an import that no applicable file declares, and a declared
package nothing under the file's directory imports (which may still be right:
servers, plugins and tools are run, not imported).
"""

from __future__ import annotations

import configparser
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from .egress import is_stdlib
from .ir import EXTERNAL_PREFIX, RepoIR

# import name -> distribution names that provide it
IMPORT_TO_DIST = {
    "yaml": ["pyyaml"], "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "PIL": ["pillow"], "sklearn": ["scikit-learn"], "skimage": ["scikit-image"], "bs4": ["beautifulsoup4"],
    "dotenv": ["python-dotenv"], "dateutil": ["python-dateutil"], "jose": ["python-jose"],
    "jwt": ["pyjwt"], "magic": ["python-magic"], "multipart": ["python-multipart"],
    "google": ["protobuf", "google-cloud-storage", "google-api-python-client", "google-auth"],
    "grpc": ["grpcio"], "attr": ["attrs"], "Crypto": ["pycryptodome"], "OpenSSL": ["pyopenssl"],
    "serial": ["pyserial"], "usb": ["pyusb"], "zmq": ["pyzmq"], "docx": ["python-docx"],
    "pptx": ["python-pptx"], "fitz": ["pymupdf"], "faiss": ["faiss-cpu", "faiss-gpu"],
    "sentence_transformers": ["sentence-transformers"], "psycopg2": ["psycopg2", "psycopg2-binary"],
    "MySQLdb": ["mysqlclient"], "redis": ["redis"], "kafka": ["kafka-python"], "telegram": ["python-telegram-bot"],
    "face_recognition": ["face-recognition"], "mediawiki": ["pymediawiki"], "wikipedia": ["wikipedia"], "tomli": ["tomli"], "pkg_resources": ["setuptools"],
}
# run or loaded by name, rarely imported: "declared, not imported" is expected for these
TOOLS = {"gunicorn", "uvicorn", "hypercorn", "pytest", "pytest-cov", "black", "ruff", "flake8", "mypy",
         "pip", "setuptools", "wheel", "build", "twine", "pre-commit", "coverage", "tox", "isort", "pylint",
         "watchdog", "supervisor", "celery", "alembic", "honcho"}

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Requirement:
    name: str                          # as written
    spec: str = ""                     # version constraint and markers, as written
    file: str = ""                     # repo-relative
    line: int = 0
    group: str = ""                    # optional-dependency group, "" for main

    @property
    def key(self) -> str:
        return normalize(self.name)


@dataclass
class Manifest:
    requirements: List[Requirement] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    imports: Dict[str, List[str]] = field(default_factory=dict)   # import root -> modules
    issues: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"files": self.files, "requirements": [asdict(r) for r in self.requirements],
                "imports": self.imports, "issues": self.issues}


def _parse_line(text: str, file: str, line: int, group: str = "") -> Optional[Requirement]:
    text = text.split(" #", 1)[0].strip()
    if not text or text.startswith(("#", "-", "git+", "http:", "https:", ".", "/")):
        return None
    m = _NAME.match(text)
    if not m:
        return None
    return Requirement(m.group(1), (m.group(2) or "") + m.group(3).strip(), file, line, group)


def _requirements_txt(path: str, rel: str) -> List[Requirement]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            r = _parse_line(line, rel, i)
            if r:
                out.append(r)
    return out


def _line_of(lines: List[str], needle: str) -> int:
    return next((i for i, ln in enumerate(lines, 1) if needle in ln), 0)


def _pyproject(path: str, rel: str) -> List[Requirement]:
    with open(path, "rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError:
            return []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = []
    project = data.get("project", {})
    for spec in project.get("dependencies", []):
        r = _parse_line(spec, rel, _line_of(lines, spec))
        if r:
            out.append(r)
    for group, specs in project.get("optional-dependencies", {}).items():
        for spec in specs:
            r = _parse_line(spec, rel, _line_of(lines, spec), group)
            if r:
                out.append(r)
    poetry = data.get("tool", {}).get("poetry", {})
    for group, deps in [("", poetry.get("dependencies", {}))] + [
            (g, v.get("dependencies", {})) for g, v in poetry.get("group", {}).items()]:
        for name, spec in deps.items():
            if name.lower() == "python":
                continue
            version = spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
            out.append(Requirement(name, version, rel, _line_of(lines, name), group))
    return out


def _setup_cfg(path: str, rel: str) -> List[Requirement]:
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error:
        return []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = []
    for spec in cp.get("options", "install_requires", fallback="").splitlines():
        r = _parse_line(spec, rel, _line_of(lines, spec.strip()) if spec.strip() else 0)
        if r:
            out.append(r)
    return out


_PIP = re.compile(r"\bpip3?\s+install\s+(.*)")


def _dockerfile(path: str, rel: str) -> List[Requirement]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read().splitlines()
    out = []
    # join continuation lines, remembering where each logical line starts
    logical, start, buf = [], 0, ""
    for i, ln in enumerate(raw, 1):
        if not buf:
            start = i
        buf += ln.rstrip("\\").rstrip() + " "
        if not ln.rstrip().endswith("\\"):
            logical.append((start, buf))
            buf = ""
    for line_no, text in logical:
        for cmd in re.split(r"&&|;", text):
            m = _PIP.search(cmd)
            if not m:
                continue
            skip = False
            for tok in m.group(1).split():
                if skip:
                    skip = False
                    continue
                if tok in ("-r", "--requirement", "-c", "--constraint", "-i", "--index-url",
                           "--extra-index-url", "-f", "--find-links", "--target", "-t"):
                    skip = True
                    continue
                r = _parse_line(tok.strip("'\""), rel, line_no)
                if r:
                    out.append(r)
    return out


def _walk(root: str, skip_dirs: Iterable[str] = (".git", "node_modules", ".venv", "venv", "__pycache__",
                                                  ".tox", "build", "dist", ".archdoc", "archdoc_out")):
    skip = set(skip_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
        for f in sorted(filenames):
            yield os.path.join(dirpath, f)


def _dist_candidates(import_root: str) -> List[str]:
    return [normalize(d) for d in IMPORT_TO_DIST.get(import_root, [])] + [normalize(import_root)]


def build_manifest(root: str, ir: RepoIR) -> Manifest:
    root = os.path.abspath(root)
    man = Manifest()
    for path in _walk(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        base = os.path.basename(path)
        reqs: List[Requirement] = []
        try:
            if re.fullmatch(r"requirements.*\.(txt|in)", base):
                reqs = _requirements_txt(path, rel)
            elif base == "pyproject.toml":
                reqs = _pyproject(path, rel)
            elif base == "setup.cfg":
                reqs = _setup_cfg(path, rel)
            elif base == "Dockerfile" or base.startswith("Dockerfile.") or base.endswith(".Dockerfile"):
                reqs = _dockerfile(path, rel)
            else:
                continue
        except OSError:
            continue
        if reqs:
            man.files.append(rel)
            man.requirements += reqs
    internal_roots = {m.id.split(".")[0] for m in ir.modules}
    by_module: Dict[str, List[str]] = {}
    for m in ir.modules:
        roots = []
        for imp in m.imports:
            if not imp.startswith(EXTERNAL_PREFIX) or is_stdlib(imp):
                continue
            r = imp[len(EXTERNAL_PREFIX):].split(".")[0]
            if r in internal_roots or r in roots:
                continue
            roots.append(r)
            man.imports.setdefault(r, []).append(m.id)
        by_module[m.path] = roots

    def covers(file: str, module_path: str) -> bool:
        d = os.path.dirname(file)
        return not d or module_path.startswith(d + "/")

    undeclared: Dict[str, List[str]] = {}
    for m in ir.modules:
        applicable = [r for r in man.requirements if covers(r.file, m.path)]
        if not applicable:
            continue
        keys = {r.key for r in applicable}
        for imp in by_module[m.path]:
            if not keys & set(_dist_candidates(imp)):
                undeclared.setdefault(imp, []).append(m.path)
    for imp, paths in sorted(undeclared.items()):
        more = f" and {len(paths) - 3} more" if len(paths) > 3 else ""
        man.issues.append({"kind": "undeclared", "package": imp, "where": ", ".join(paths[:3]) + more,
                           "issue": "imported, but no requirements file covering it declares it"})
    for r in man.requirements:
        users = [m for m in ir.modules if covers(r.file, m.path)
                 and r.key in {c for i in by_module[m.path] for c in _dist_candidates(i)}]
        if not users and not any(covers(r.file, m.path) for m in ir.modules):
            continue                              # no Python code under that file: nothing to compare
        if not users:
            tool = r.key in TOOLS
            man.issues.append({"kind": "unused", "package": r.name, "where": f"{r.file}:{r.line}",
                               "issue": "declared, not imported" + (" (a tool or server: run, not imported)"
                                                                    if tool else "")})
    return man


__all__ = ["IMPORT_TO_DIST", "Manifest", "Requirement", "build_manifest", "normalize"]
