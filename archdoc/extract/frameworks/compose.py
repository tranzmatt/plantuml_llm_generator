"""
docker-compose files as deployments: which services exist, which code each
image contains and runs, ports, dependencies, and where each environment
variable's value comes from.

What an image contains is read from its Dockerfile (ADD/COPY, WORKDIR, CMD,
ENTRYPOINT), overridden by the compose `entrypoint` / `command`: a module that
is in the repo but copied into no image is not part of that deployment, which
a rebuild has to know. Environment values are never recorded, only whether
the value comes from the host, is written literally, or is missing.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from ...ir import Deployment, Service
from ..python_ast import ExtractContext

try:
    import yaml
except ImportError:                    # pragma: no cover - pyyaml is a dependency
    yaml = None

_COMPOSE_RE = re.compile(r"^(docker-)?compose([.-].*)?\.ya?ml$|.*[.-]docker-compose\.ya?ml$|.*[.-]compose\.ya?ml$")
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", "build", "dist"}
_HOST_REF = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*(:?[-?][^}]*)?\}?$")


def _compose_files(root: str) -> List[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            if _COMPOSE_RE.match(f):
                found.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    return found


def _env(spec: Any) -> Dict[str, str]:
    """name -> host | literal | unset, from list ("A=b", "A") or mapping form."""
    items: List[Tuple[str, Optional[str]]] = []
    if isinstance(spec, dict):
        items = [(str(k), None if v is None else str(v)) for k, v in spec.items()]
    elif isinstance(spec, list):
        for e in spec:
            name, eq, value = str(e).partition("=")
            items.append((name.strip(), value if eq else None))
    out = {}
    for name, value in items:
        if value is None:
            out[name] = "host"                      # passed through from the host environment
        elif value == "":
            out[name] = "unset"
        elif _HOST_REF.match(value.strip()):
            out[name] = "host"
        else:
            out[name] = "literal"
    return out


def _dockerfile_lines(path: str) -> List[Tuple[str, str]]:
    """(INSTRUCTION, arguments) with continuations joined and comments dropped."""
    lines, buf = [], ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            s = raw.rstrip("\n")
            if not buf and s.lstrip().startswith("#"):
                continue
            if s.endswith("\\"):
                buf += s[:-1] + " "
                continue
            s, buf = buf + s, ""
            if s.strip():
                instr, _, args = s.strip().partition(" ")
                lines.append((instr.upper(), args.strip()))
    return lines


def _command(args: Any) -> List[str]:
    if isinstance(args, list):
        return [str(a) for a in args]
    if isinstance(args, str):
        s = args.strip()
        if s.startswith("["):
            try:
                return [str(a) for a in json.loads(s)]
            except ValueError:
                pass
        try:
            return shlex.split(s)
        except ValueError:
            return s.split()
    return []


class _Image:
    """What a Dockerfile copies (container path -> repo path) and runs."""

    def __init__(self, context: str, dockerfile: Optional[str], root: str):
        self.context = context
        self.copies: List[Tuple[str, str]] = []       # (container dest, repo-relative src)
        self.workdir = "/"
        self.entrypoint: List[str] = []
        self.cmd: List[str] = []
        self.cmd_workdir = "/"
        if not dockerfile:
            return
        for instr, args in _dockerfile_lines(os.path.join(root, dockerfile)):
            if instr == "WORKDIR":
                self.workdir = posixpath.join(self.workdir, args)
            elif instr in ("ADD", "COPY"):
                parts = [p for p in _command(args) if not p.startswith("--")]
                if len(parts) < 2 or any("://" in p for p in parts) or "--from" in args:
                    continue
                raw_dest = parts[-1]
                dest = posixpath.join(self.workdir, raw_dest)
                into_dir = raw_dest.endswith("/") or raw_dest in (".", "./") or len(parts) > 2
                for src in parts[:-1]:
                    repo_src = posixpath.normpath(posixpath.join(context, src))
                    d = dest
                    # a directory's contents go to dest; a file goes to dest/<name> when dest is a directory
                    if into_dir and not os.path.isdir(os.path.join(root, repo_src)):
                        d = posixpath.join(dest, posixpath.basename(src.rstrip("/")))
                    self.copies.append((posixpath.normpath(d), repo_src))
            elif instr == "CMD":
                self.cmd, self.cmd_workdir = _command(args), self.workdir
            elif instr == "ENTRYPOINT":
                self.entrypoint, self.cmd_workdir = _command(args), self.workdir

    def repo_path(self, container_path: str, workdir: Optional[str] = None) -> Optional[str]:
        p = posixpath.normpath(posixpath.join(workdir or self.cmd_workdir, container_path))
        for dest, src in sorted(self.copies, key=lambda c: -len(c[0])):
            if p == dest:
                return src if src.endswith(".py") else None
            if p.startswith(dest.rstrip("/") + "/"):
                return posixpath.normpath(posixpath.join(src, p[len(dest.rstrip("/")) + 1:]))
        return None

    def contains(self, path: str) -> bool:
        if not self.copies:
            return path == self.context or path.startswith(self.context.rstrip("/") + "/") or self.context == "."
        return any(path == src or path.startswith(src.rstrip("/") + "/") or src == "." for _, src in self.copies)


def _entry(ctx: ExtractContext, image: _Image, argv: List[str]) -> Optional[str]:
    by_path = {m.path: m.id for m in ctx.ir.modules}
    for i, a in enumerate(argv):
        if a == "-m" and i + 1 < len(argv):
            mod = argv[i + 1]
            if any(m.id == mod or m.id.endswith("." + mod) for m in ctx.ir.modules):
                return next(m.id for m in ctx.ir.modules if m.id == mod or m.id.endswith("." + mod))
        if a.endswith(".py"):
            rp = image.repo_path(a) if image.copies else posixpath.normpath(posixpath.join(image.context, a))
            if rp in by_path:
                return by_path[rp]
    return None


def _service(ctx: ExtractContext, file_dir: str, name: str, spec: Dict[str, Any]) -> Service:
    svc = Service(name=name, image=spec.get("image"))
    build = spec.get("build")
    context, dockerfile = None, None
    if isinstance(build, str):
        context = build
    elif isinstance(build, dict):
        context, dockerfile = build.get("context", "."), build.get("dockerfile")
    image = None
    if context is not None:
        svc.build = posixpath.normpath(posixpath.join(file_dir, context))
        df = posixpath.normpath(posixpath.join(svc.build, dockerfile or "Dockerfile"))
        svc.dockerfile = df if os.path.isfile(os.path.join(ctx.root, df)) else None
        image = _Image(svc.build, svc.dockerfile, ctx.root)
        svc.modules = sorted(m.id for m in ctx.ir.modules if image.contains(m.path))
        argv = _command(spec["entrypoint"]) if "entrypoint" in spec else image.entrypoint
        argv = argv + (_command(spec["command"]) if "command" in spec else ([] if "entrypoint" in spec else image.cmd))
        svc.entry_module = _entry(ctx, image, argv)
    svc.ports = [str(p) if not isinstance(p, dict) else f"{p.get('published', '')}:{p.get('target', '')}"
                 for p in spec.get("ports", []) or []]
    deps = spec.get("depends_on") or []
    svc.depends_on = sorted(deps) if isinstance(deps, (list, dict)) else []
    svc.env = _env(spec.get("environment"))
    placement = ((spec.get("deploy") or {}).get("placement") or {}).get("constraints") or []
    svc.placement = [str(c) for c in placement]
    return svc


def detect_compose(ctx: ExtractContext) -> None:
    if yaml is None:
        return
    for rel in _compose_files(ctx.root):
        try:
            with open(os.path.join(ctx.root, rel), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        services = (doc or {}).get("services") if isinstance(doc, dict) else None
        if not isinstance(services, dict):
            continue
        file_dir = posixpath.dirname(rel) or "."
        dep = Deployment(file=rel)
        for name in sorted(services):
            if isinstance(services[name], dict):
                dep.services.append(_service(ctx, file_dir, str(name), services[name]))
        ctx.ir.deployments.append(dep)
