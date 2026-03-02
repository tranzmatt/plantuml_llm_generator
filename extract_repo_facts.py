#!/usr/bin/env python3
"""
extract_repo_facts.py

Walk a Python repo and write per-file "facts" to JSONL.
Facts are designed to support multiple UML diagram types:
- class/object: classes, bases, dataclasses/pydantic-ish models, methods, signatures
- component: imports (module deps), internal vs external libs
- sequence/activity: entrypoints, routes/tasks/commands, call targets, IO hints
- state: status/state field usage, enums/constants, transition-like methods
- deployment: server/worker/cli hints, env var usage, ports, infra SDK hints

Output: facts.jsonl (one JSON dict per file)
"""

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache",
    "site-packages",
}

# Common IO / integration libs (heuristics)
IO_IMPORT_HINTS = {
    "requests": "http",
    "httpx": "http",
    "aiohttp": "http",
    "urllib3": "http",
    "boto3": "aws",
    "botocore": "aws",
    "s3fs": "object_store",
    "sqlalchemy": "db",
    "psycopg2": "db",
    "pymysql": "db",
    "pymongo": "db",
    "redis": "cache",
    "celery": "queue",
    "kombu": "queue",
    "pika": "queue",
    "kafka": "queue",
    "confluent_kafka": "queue",
    "google.cloud": "gcp",
    "azure": "azure",
    "grpc": "rpc",
    "fastapi": "web",
    "flask": "web",
    "django": "web",
    "uvicorn": "server",
    "gunicorn": "server",
}

ROUTE_DECORATOR_PREFIXES = ("app.", "router.", "bp.", "blueprint.", "api.")
ROUTE_DECORATOR_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
CELERY_DECORATORS = {"task", "shared_task"}
CLICK_DECORATOR = "click.command"
ARGPARSE_HINTS = ("argparse", "ArgumentParser")

STATE_WORDS = {"state", "status", "phase", "stage", "mode"}


def is_excluded_path(path: str, exclude_dirs: Set[str]) -> bool:
    parts = path.split(os.sep)
    return any(p in exclude_dirs for p in parts)


def safe_read_text(path: str, max_chars: int) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "\n# ... (truncated) ...\n"
        return txt
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def dotted_name(node: ast.AST) -> Optional[str]:
    # Convert Name / Attribute chains to dotted string
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def get_str_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def get_int_literal(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def unparse_sig_args(args: ast.arguments) -> str:
    parts = []
    # positional-only (py3.8+)
    for a in getattr(args, "posonlyargs", []):
        parts.append(a.arg)
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        parts.append(f"{a.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


@dataclass
class FileFacts:
    path: str
    module: str
    defines: Dict[str, List[Dict]] = field(default_factory=lambda: {"classes": [], "functions": []})
    entrypoints: Dict[str, List[Dict]] = field(default_factory=lambda: {"routes": [], "tasks": [], "commands": [], "mains": []})
    imports: Dict[str, List[str]] = field(default_factory=lambda: {"modules": [], "from": [], "external": [], "internal": []})
    calls: Dict[str, List[str]] = field(default_factory=lambda: {"targets": []})
    io: Dict[str, List[str]] = field(default_factory=lambda: {"kinds": [], "endpoints": [], "queues": [], "buckets": [], "db": [], "files": []})
    env: List[str] = field(default_factory=list)
    state: Dict[str, List[str]] = field(default_factory=lambda: {"fields": [], "enums": [], "transitions": []})
    notes: List[str] = field(default_factory=list)


class FactVisitor(ast.NodeVisitor):
    def __init__(self, facts: FileFacts):
        self.facts = facts
        self.current_class: Optional[str] = None
        self.imported_modules: Set[str] = set()
        self.from_imports: Set[str] = set()
        self.external_imports: Set[str] = set()
        self.internal_imports: Set[str] = set()
        self._seen_io_kinds: Set[str] = set()
        self._string_literals: List[str] = []
        self._maybe_ports: List[int] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            self.imported_modules.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            mod = node.module
            self.from_imports.add(mod)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self._string_literals.append(node.value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # detect state/status fields via attribute targets (self.status = ...)
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                field = tgt.attr
                if field.lower() in STATE_WORDS:
                    self.facts.state["fields"].append(field)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # detect annotated state/status fields: self.status: str = ...
        tgt = node.target
        if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
            field = tgt.attr
            if field.lower() in STATE_WORDS:
                self.facts.state["fields"].append(field)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        prev = self.current_class
        self.current_class = node.name

        bases = []
        for b in node.bases:
            dn = dotted_name(b) or getattr(b, "id", None)
            if dn:
                bases.append(dn)

        decorators = [dotted_name(d) for d in node.decorator_list]
        decorators = [d for d in decorators if d]

        # Heuristics: dataclass / pydantic model / Enum
        is_dataclass = any(d.endswith("dataclass") for d in decorators)
        is_enum = any(b.endswith("Enum") or b == "Enum" for b in bases) or node.name.endswith("Enum")

        class_rec = {
            "name": node.name,
            "bases": bases,
            "decorators": decorators,
            "dataclass": bool(is_dataclass),
            "enum": bool(is_enum),
            "methods": [],
        }

        # methods collected as we visit FunctionDef within this class
        self.facts.defines["classes"].append(class_rec)

        self.generic_visit(node)
        self.current_class = prev

    def _add_function(self, node: ast.FunctionDef, is_async: bool):
        decorators = [dotted_name(d) for d in node.decorator_list]
        decorators = [d for d in decorators if d]

        sig = unparse_sig_args(node.args)

        rec = {
            "name": node.name,
            "signature": sig,
            "async": bool(is_async),
            "decorators": decorators,
        }

        if self.current_class:
            # append to last matching class record
            for c in reversed(self.facts.defines["classes"]):
                if c["name"] == self.current_class:
                    c["methods"].append(rec)
                    break
        else:
            self.facts.defines["functions"].append(rec)

        # entrypoint heuristics
        self._maybe_record_entrypoints(node, decorators)

        # transition-ish heuristics
        lname = node.name.lower()
        if any(k in lname for k in ("transition", "advance", "set_state", "set_status", "mark_", "move_to")):
            self.facts.state["transitions"].append(
                f"{self.current_class + '.' if self.current_class else ''}{node.name}{sig}"
            )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._add_function(node, is_async=True)
        self.generic_visit(node)

    def _maybe_record_entrypoints(self, node: ast.AST, decorators: List[str]):
        # Routes: @app.get("/x"), @router.post("/y")
        for d in decorators:
            if not d:
                continue
            # fastapi/flask style: app.get, router.post etc
            parts = d.split(".")
            if len(parts) >= 2 and parts[-1] in ROUTE_DECORATOR_METHODS:
                if parts[0] in {"app", "router", "bp", "api", "blueprint"} or any(
                    d.startswith(pfx) for pfx in ROUTE_DECORATOR_PREFIXES
                ):
                    # try to extract first arg string literal from decorator call if available
                    # but AST decorator_list stores Call nodes, so dotted_name() may lose args.
                    # We'll instead do heuristic later from string literals.
                    self.facts.entrypoints["routes"].append(
                        {"func": getattr(node, "name", "<lambda>"), "decorator": d}
                    )

            # Celery: @app.task, @shared_task
            if parts[-1] in CELERY_DECORATORS or d.endswith(".task") or d.endswith("shared_task"):
                self.facts.entrypoints["tasks"].append(
                    {"func": getattr(node, "name", "<lambda>"), "decorator": d}
                )

            # Click command
            if d == CLICK_DECORATOR or d.endswith(".command"):
                self.facts.entrypoints["commands"].append(
                    {"func": getattr(node, "name", "<lambda>"), "decorator": d}
                )

    def visit_If(self, node: ast.If):
        # main guard
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    # __name__ == "__main__"
                    for op, comp in zip(node.test.ops, node.test.comparators):
                        if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value == "__main__":
                            self.facts.entrypoints["mains"].append({"type": "__main__"})
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        dn = dotted_name(node.func)
        if dn:
            self.facts.calls["targets"].append(dn)

            # Env var usage: os.environ.get("X"), os.getenv("X")
            if dn in {"os.getenv", "os.environ.get"} and node.args:
                s = get_str_literal(node.args[0])
                if s:
                    self.facts.env.append(s)

            # common "port=" patterns in uvicorn.run(..., port=8000)
            if dn.endswith("uvicorn.run") or dn.endswith("app.run") or dn.endswith("run"):
                for kw in node.keywords:
                    if kw.arg == "port":
                        iv = get_int_literal(kw.value)
                        if iv is not None:
                            self._maybe_ports.append(iv)

        self.generic_visit(node)

    def finalize(self):
        # classify imports into internal/external based on repo module prefix
        # We treat anything without a dot and in IO_IMPORT_HINTS as external.
        # Internal modules are those that start with the repo module root prefix (facts.module root) or relative patterns.
        # This is heuristic; reducer will re-group.
        mods = sorted(self.imported_modules)
        frs = sorted(self.from_imports)

        self.facts.imports["modules"] = mods
        self.facts.imports["from"] = frs

        externals: Set[str] = set()
        for m in mods + frs:
            root = m.split(".")[0]
            if root in IO_IMPORT_HINTS:
                externals.add(root)
                kind = IO_IMPORT_HINTS[root]
                if kind not in self._seen_io_kinds:
                    self.facts.io["kinds"].append(kind)
                    self._seen_io_kinds.add(kind)

        self.facts.imports["external"] = sorted(externals)

        # IO hints from string literals (rough but useful)
        for s in self._string_literals:
            if isinstance(s, str):
                if s.startswith(("http://", "https://")):
                    self.facts.io["endpoints"].append(s)
                if "s3://" in s:
                    self.facts.io["buckets"].append(s)
                if any(k in s.lower() for k in ("queue", "topic", "exchange")) and len(s) < 120:
                    self.facts.io["queues"].append(s)

        # Ports
        for p in self._maybe_ports:
            self.facts.notes.append(f"detected_port:{p}")

        # Dedup lists
        for k in list(self.facts.calls.keys()):
            self.facts.calls[k] = sorted(set(self.facts.calls[k]))

        self.facts.env = sorted(set(self.facts.env))
        for k in list(self.facts.io.keys()):
            self.facts.io[k] = sorted(set(self.facts.io[k]))
        for k in list(self.facts.state.keys()):
            self.facts.state[k] = sorted(set(self.facts.state[k]))


def module_name_from_path(repo_root: str, rel_path: str) -> str:
    # Convert foo/bar/baz.py -> foo.bar.baz
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    return rel_path.replace(os.sep, ".")


def process_file(repo_root: str, file_path: str, max_chars: int) -> Optional[Dict]:
    rel = os.path.relpath(file_path, repo_root)
    mod = module_name_from_path(repo_root, rel)
    txt = safe_read_text(file_path, max_chars=max_chars)
    if txt is None:
        return None

    facts = FileFacts(path=rel, module=mod)
    try:
        tree = ast.parse(txt, filename=rel)
    except SyntaxError:
        facts.notes.append("syntax_error_parsing_file")
        # still return minimal info
        return facts.__dict__

    v = FactVisitor(facts)
    v.visit(tree)
    v.finalize()
    return facts.__dict__


def main():
    ap = argparse.ArgumentParser(description="Extract per-file architecture facts to JSONL.")
    ap.add_argument("--input", "-i", required=True, help="Repo root")
    ap.add_argument("--output", "-o", default="uml_out/facts.jsonl", help="Output JSONL path")
    ap.add_argument("--max-file-chars", type=int, default=300_000, help="Truncate very large files")
    ap.add_argument("--exclude-dir", action="append", default=[], help="Repeatable directory names to exclude")
    ap.add_argument("--ext", action="append", default=[".py"], help="Repeatable file extensions (default .py)")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.input)
    ex = set(DEFAULT_EXCLUDE_DIRS)
    ex.update(args.exclude_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    include_exts = tuple(args.ext)

    total_files = 0
    written = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(repo_root):
            if is_excluded_path(dirpath, ex):
                dirnames[:] = []
                continue

            dirnames[:] = [d for d in dirnames if d not in ex]

            for fn in sorted(filenames):
                if not fn.endswith(include_exts):
                    continue
                fp = os.path.join(dirpath, fn)
                total_files += 1
                rec = process_file(repo_root, fp, max_chars=args.max_file_chars)
                if rec is None:
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Processed files: {total_files}, wrote records: {written}")
    print(f"Facts output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
