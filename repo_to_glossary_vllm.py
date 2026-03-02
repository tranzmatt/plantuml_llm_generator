#!/usr/bin/env python3
"""
repo_to_glossary_vllm.py

Pre-pass: Build a compact repo index and ask vLLM to output a canonical naming glossary as JSON.

Goal: enforce consistent naming across multiple downstream diagram generations
without stuffing the entire repo into context.

Designed to pair with repo_to_diagrams_vllm_local.py. :contentReference[oaicite:1]{index=1}
"""

import os
import re
import json
import sys
import argparse
from typing import Dict, List, Tuple, Optional

# Keep vLLM env behavior consistent with your existing script (safe defaults).
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_NO_CUDA_GRAPH", "1")
os.environ.setdefault("VLLM_ENFORCE_EAGER", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_MAX_NUM_SEQS", "32")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.85")

from vllm import LLM, SamplingParams


DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache",
    "site-packages",
}


def should_skip_dir(dirpath: str, exclude_dirs: set) -> bool:
    parts = set(dirpath.split(os.sep))
    return any(p in exclude_dirs for p in parts)


def extract_repo_index_for_file(
    text: str,
    max_lines: int = 300,
) -> str:
    """
    Extract a compact "index" view:
    - module docstring header (if present)
    - import lines
    - class/def signatures
    - common web route decorators
    - argparse/click option definitions (heuristic)

    This is intentionally lossy but architecture-relevant.
    """
    lines = text.splitlines()
    out: List[str] = []

    # Grab leading docstring-ish header (first ~40 lines)
    header = lines[:40]
    for ln in header:
        if ln.strip().startswith(("import ", "from ", "class ", "def ", "@")):
            break
        out.append(ln)
    if out:
        out.append("")

    # Imports
    for ln in lines:
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            out.append(ln)

    out.append("")

    # Signatures + decorators of interest
    sig_pat = re.compile(r"^\s*(class|def)\s+[A-Za-z_][A-Za-z0-9_]*")
    deco_interest = (
        "@app.", "@router.", "@bp.", "@blueprint.", "@api.", "@fastapi.", "@flask.",
        "@click.", "@pytest.", "@celery.", "@shared_task", "@task",
    )

    count = 0
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(d) for d in deco_interest):
            out.append(ln)
            count += 1
        elif sig_pat.match(ln):
            out.append(ln)
            count += 1

        if count >= max_lines:
            out.append("... (truncated index for this file) ...")
            break

    return "\n".join(out).strip()


def build_repo_index(
    root: str,
    include_exts: Tuple[str, ...] = (".py",),
    exclude_dirs: Optional[set] = None,
    max_file_chars: int = 200_000,
    max_total_chars: int = 2_000_000,
    per_file_index_lines: int = 300,
) -> Tuple[str, Dict[str, int]]:
    """
    Returns:
      - repo_index_text: big string containing per-file index blocks
      - file_char_stats: per file raw size for visibility
    """
    if exclude_dirs is None:
        exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)

    chunks: List[str] = []
    stats: Dict[str, int] = {}

    total_chars = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if should_skip_dir(dirpath, exclude_dirs):
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for fname in sorted(filenames):
            if not fname.endswith(include_exts):
                continue

            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = f.read()
            except UnicodeDecodeError:
                continue

            stats[rel] = len(raw)
            if len(raw) > max_file_chars:
                # Keep a hint but avoid blowing the index with giant files
                raw = raw[:max_file_chars] + "\n... (file truncated due to size) ...\n"

            index_view = extract_repo_index_for_file(raw, max_lines=per_file_index_lines)

            block = f"===== FILE: {rel} =====\n{index_view}\n"
            if total_chars + len(block) > max_total_chars:
                chunks.append("... (repo index truncated due to max_total_chars) ...\n")
                return "\n".join(chunks), stats

            chunks.append(block)
            total_chars += len(block)

    return "\n".join(chunks), stats


def vllm_generate_json(llm: LLM, system_msg: str, user_msg: str, max_tokens: int) -> str:
    prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    out = llm.generate([prompt], sp)[0].outputs[0].text
    return out.strip()


def extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of a single top-level JSON object from model output.
    """
    # Find first '{' and match braces.
    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in output; model did not return JSON.")

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("Unbalanced JSON braces in output.")


def main():
    ap = argparse.ArgumentParser(description="Generate a canonical architecture glossary JSON for a repo (vLLM).")
    ap.add_argument("--input", "-i", default=".", help="Repo root")
    ap.add_argument("--output", "-o", default="uml_out/glossary.json", help="Output glossary JSON path")
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "4")))
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_LEN", "131072")))
    ap.add_argument("--max-tokens", type=int, default=4000, help="Max tokens for glossary JSON output")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")

    # Index sizing knobs
    ap.add_argument("--max-total-chars", type=int, default=2_000_000)
    ap.add_argument("--max-file-chars", type=int, default=200_000)
    ap.add_argument("--per-file-index-lines", type=int, default=300)
    ap.add_argument("--exclude-dir", action="append", default=[], help="Repeatable directory names to exclude")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.input)
    repo_name = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs.update(args.exclude_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=" * 70)
    print("Repo → Glossary (vLLM)")
    print("=" * 70)
    print(f"Repo:   {repo_root}")
    print(f"Model:  {args.model}")
    print(f"TP:     {args.tp}")
    print(f"MaxLen: {args.max_model_len}")
    print(f"Out:    {os.path.abspath(args.output)}")
    print("=" * 70)

    print("[1/3] Building compact repo index...")
    repo_index, stats = build_repo_index(
        root=repo_root,
        include_exts=(".py",),
        exclude_dirs=exclude_dirs,
        max_file_chars=args.max_file_chars,
        max_total_chars=args.max_total_chars,
        per_file_index_lines=args.per_file_index_lines,
    )
    if not repo_index.strip():
        raise RuntimeError("Repo index is empty (no .py files found?).")

    print(f"      Repo index chars: {len(repo_index):,}")
    if stats:
        biggest = sorted(stats.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("      Top 10 largest raw files (chars):")
        for rel, sz in biggest:
            print(f"        {sz:>10,}  {rel}")

    print("[2/3] Loading vLLM model...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )

    system_msg = (
        "You are a senior software architect. "
        "You must produce a canonical naming glossary for a codebase. "
        "Output MUST be valid JSON only (no markdown, no commentary)."
    )

    schema_hint = {
        "repo": repo_name,
        "naming_rules": [
            "Use exact identifiers found in code where possible.",
            "Prefer stable, consistent component names across diagrams.",
            "Avoid synonyms: pick one canonical term per concept."
        ],
        "components": [
            {"name": "string", "type": "service|module|worker|library|cli|api", "description": "string", "code_refs": ["path.py:Symbol"]}
        ],
        "classes": [
            {"name": "string", "module": "string", "role": "string", "relations": ["string"]}
        ],
        "external_systems": [
            {"name": "string", "type": "db|queue|cache|object_store|http_api|filesystem|other", "description": "string"}
        ],
        "data_entities": [
            {"name": "string", "description": "string", "code_refs": ["path.py:Symbol"]}
        ],
        "aliases": {
            "alias_or_variant": "canonical_name"
        },
        "main_flows": [
            {"name": "string", "steps": ["string", "string"]}
        ],
    }

    user_msg = (
        f"Repository name: {repo_name}\n\n"
        "Below is a compact index of the repository (imports, signatures, decorators, and key structure). "
        "Use it to infer the architecture and choose canonical names.\n\n"
        "Return ONE JSON object that follows this schema (you may omit empty lists):\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Repo index:\n"
        "```text\n"
        f"{repo_index}\n"
        "```\n"
    )

    print("[3/3] Generating glossary JSON...")
    raw = vllm_generate_json(llm, system_msg, user_msg, max_tokens=args.max_tokens)

    json_text = extract_first_json_object(raw)
    glossary = json.loads(json_text)  # validates JSON

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"✓ Wrote glossary: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
