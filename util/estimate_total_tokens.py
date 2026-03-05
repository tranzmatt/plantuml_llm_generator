#!/usr/bin/env python3
"""
estimate_total_tokens.py

Build the SAME prompt shape as repo_to_diagrams_vllm_local.py (repo + RAG examples)
but DO NOT run inference. Instead, estimate token usage and print a breakdown.

Breakdown includes:
- Repo code tokens
- RAG tokens (total + per diagram type)
- Wrapper/system/user framing tokens
- TOTAL prompt tokens
- Headroom vs max-model-len and planned output tokens

Based on the structure of:
- repo_to_diagrams_vllm_local.py (prompt wrapper + RAG via FAISS + SentenceTransformer)
- repo_to_diagrams_ollama.py (diagram ordering, prompt sections)
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


DIAGRAM_TYPES: List[Tuple[str, str]] = [
    ("class", "Class diagram describing main modules, services, data structures and their relationships."),
    ("sequence", "Sequence diagram describing the main runtime flow from inputs to outputs."),
    ("activity", "Activity diagram describing overall workflow and branching."),
    ("state", "State diagram for the most important stateful part of the system."),
    ("component", "Component diagram describing services, queues, and external APIs."),
    ("deployment", "Deployment diagram showing runtime nodes, processes, queues, and external systems."),
    ("usecase", "Use case diagram showing main actors and high-level use cases."),
    ("object", "Object diagram showing a runtime snapshot of key objects/instances."),
]


def walk_repo_collect_code(root: str, include_exts: Tuple[str, ...] = (".py",)) -> str:
    chunks: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.endswith(include_exts):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    code = f.read()
            except UnicodeDecodeError:
                continue
            rel = os.path.relpath(fpath, root)
            chunks.append(f"===== FILE: {rel} =====\n{code}\n")
    return "\n".join(chunks)


def load_faiss_and_meta(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    embed_model_name = meta.get("embed_model", "nomic-embed-text")
    return index, docs, embed_model_name


def get_rag_examples_for_type(diagram_type: str, index, docs, embed_model, top_k: int) -> str:
    query = f"PlantUML {diagram_type} diagram for a distributed Python microservice application."
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)

    examples: List[str] = []
    for i, idx in enumerate(indices[0]):
        d = docs[int(idx)]
        plantuml = d.get("output") or d.get("plantuml", "")
        if not plantuml:
            continue
        instr = d.get("instruction", "")
        examples.append(
            f"--- Example {i+1} (score={scores[0][i]:.4f}):\n"
            f"Instruction: {instr}\n\n"
            f"{plantuml}\n"
        )
    return "\n".join(examples)


def build_user_message(repo_name: str, repo_text: str, rag_examples: Dict[str, str]) -> str:
    user_parts: List[str] = []
    user_parts.append(
        f"The repository is a Python-based system named '{repo_name}'. "
        f"Below is the entire codebase (all .py files):"
    )
    user_parts.append("")
    user_parts.append("```python")
    user_parts.append(repo_text)
    user_parts.append("```")
    user_parts.append("")
    user_parts.append(
        "You will generate EIGHT PlantUML 1.2025.0 diagrams for this system in the following fixed order:"
    )
    user_parts.append("1. CLASS")
    user_parts.append("2. SEQUENCE")
    user_parts.append("3. ACTIVITY")
    user_parts.append("4. STATE")
    user_parts.append("5. COMPONENT")
    user_parts.append("6. DEPLOYMENT")
    user_parts.append("7. USE CASE")
    user_parts.append("8. OBJECT")
    user_parts.append("")
    user_parts.append(
        "For each diagram type, you are given style/syntax examples retrieved from a RAG index. "
        "Use these ONLY as reference, do NOT copy them verbatim."
    )
    user_parts.append("")

    for dtype, desc in DIAGRAM_TYPES:
        header = dtype.upper() if dtype != "usecase" else "USE CASE"
        examples = (rag_examples.get(dtype, "") or "").strip() or "(No examples available.)"
        user_parts.append(f"### RAG EXAMPLES FOR {header}")
        user_parts.append(f"Diagram description: {desc}")
        user_parts.append("")
        user_parts.append(examples)
        user_parts.append("")

    user_parts.append(
        "Now generate ALL eight diagrams in ONE response using the following format EXACTLY:\n"
        "For each section:\n\n"
        "### CLASS\n"
        "@startuml\n"
        "...class diagram here...\n"
        "@enduml\n\n"
        "### SEQUENCE\n"
        "@startuml\n"
        "...sequence diagram here...\n"
        "@enduml\n\n"
        "### ACTIVITY\n"
        "@startuml\n"
        "...activity diagram here...\n"
        "@enduml\n\n"
        "### STATE\n"
        "@startuml\n"
        "...state diagram here...\n"
        "@enduml\n\n"
        "### COMPONENT\n"
        "@startuml\n"
        "...component diagram here...\n"
        "@enduml\n\n"
        "### DEPLOYMENT\n"
        "@startuml\n"
        "...deployment diagram here...\n"
        "@enduml\n\n"
        "### USE CASE\n"
        "@startuml\n"
        "...use case diagram here...\n"
        "@enduml\n\n"
        "### OBJECT\n"
        "@startuml\n"
        "...object diagram here...\n"
        "@enduml\n\n"
        "Rules:\n"
        "- ALWAYS include @startuml and @enduml in each section.\n"
        "- Do NOT include any explanations or commentary outside of the diagrams.\n"
        "- Keep naming consistent across ALL diagrams (services, queues, APIs, classes, etc.)."
    )

    return "\n".join(user_parts)


def build_vllm_style_prompt(system_msg: str, user_msg: str) -> str:
    # Matches repo_to_diagrams_vllm_local.py’s prompt wrapper.
    return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def tok_len(tok: AutoTokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser(description="Estimate token usage for repo+RAG prompt (no inference).")
    ap.add_argument("--input", "-i", default=".", help="Repo root")
    ap.add_argument("--model", required=True, help="HF model id used for tokenizer (match vLLM model)")
    ap.add_argument("--faiss-index", required=True)
    ap.add_argument("--faiss-meta", required=True)
    ap.add_argument("--rag-k", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=131072, help="Context window you intend to use")
    ap.add_argument("--max-tokens", type=int, default=8000, help="Planned output tokens")
    ap.add_argument("--ext", action="append", default=[".py"], help="File extension to include; repeatable")
    ap.add_argument("--dump-prompt", default="", help="Optional path to write the final prompt text")
    ap.add_argument("--show-rag-breakdown", action="store_true", help="Print per-diagram RAG token counts")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.input)
    repo_name = os.path.basename(repo_root.rstrip(os.sep)) or "repo"
    include_exts = tuple(args.ext)

    print(f"[estimator] Repo: {repo_root}")
    print(f"[estimator] Include exts: {include_exts}")
    print(f"[estimator] rag-k: {args.rag_k}")

    print("[estimator] Collecting repo...")
    repo_text = walk_repo_collect_code(repo_root, include_exts=include_exts)
    if not repo_text.strip():
        raise SystemExit("No files collected. Check --input/--ext.")

    print("[estimator] Loading FAISS...")
    index, docs, embed_model_name = load_faiss_and_meta(args.faiss_index, args.faiss_meta)

    if embed_model_name == "nomic-embed-text":
        embed_model_id = "nomic-ai/nomic-embed-text-v1.5"
    else:
        embed_model_id = embed_model_name

    print(f"[estimator] Loading embed model: {embed_model_id}")
    embed_model = SentenceTransformer(embed_model_id, trust_remote_code=True)

    print("[estimator] Retrieving RAG examples...")
    rag_examples: Dict[str, str] = {}
    for dtype, _ in DIAGRAM_TYPES:
        rag_examples[dtype] = get_rag_examples_for_type(dtype, index, docs, embed_model, args.rag_k)

    system_msg = (
        "You are an expert in Python static analysis and UML architecture. "
        "You generate high-quality PlantUML 1.2025.0 diagrams for entire systems. "
        "You MUST keep names and semantics consistent across ALL diagrams. "
        "Always follow the requested output format exactly."
    )

    print("[estimator] Building prompt...")
    user_msg = build_user_message(repo_name, repo_text, rag_examples)
    prompt = build_vllm_style_prompt(system_msg, user_msg)

    if args.dump_prompt:
        with open(args.dump_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"[estimator] Prompt written to: {args.dump_prompt}")

    print("[estimator] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    # Token breakdown
    repo_tokens = tok_len(tok, repo_text)

    rag_tokens_by_type: Dict[str, int] = {}
    rag_tokens_total = 0
    for dtype, _ in DIAGRAM_TYPES:
        t = tok_len(tok, rag_examples.get(dtype, "") or "")
        rag_tokens_by_type[dtype] = t
        rag_tokens_total += t

    # Wrapper overhead: everything that isn't repo_text or raw RAG example blocks
    # Compute as: total - repo_tokens - rag_tokens_total
    total_tokens = tok_len(tok, prompt)
    overhead_tokens = total_tokens - repo_tokens - rag_tokens_total

    # Headroom
    remaining_for_output_max = args.max_model_len - total_tokens
    remaining_after_request = args.max_model_len - (total_tokens + args.max_tokens)

    print("\n" + "=" * 72)
    print("TOKEN BREAKDOWN")
    print("=" * 72)
    print(f"Model tokenizer:            {args.model}")
    print(f"Max model len:              {args.max_model_len}")
    print(f"Planned output tokens:      {args.max_tokens}")
    print("-" * 72)
    print(f"Repo code tokens:           {repo_tokens}")
    print(f"RAG tokens total:           {rag_tokens_total}   (rag-k={args.rag_k}, types={len(DIAGRAM_TYPES)})")
    if args.show_rag_breakdown:
        for dtype, _ in DIAGRAM_TYPES:
            print(f"  - {dtype:10s}: {rag_tokens_by_type[dtype]}")
    print(f"Wrapper/overhead tokens:    {overhead_tokens}")
    print("-" * 72)
    print(f"TOTAL input tokens:         {total_tokens}")
    print("=" * 72)

    print("\nHEADROOM")
    print("-" * 72)
    print(f"Remaining tokens for output (max): {remaining_for_output_max}")
    print(f"Remaining after requesting max_tokens={args.max_tokens}: {remaining_after_request}")

    if remaining_for_output_max < 0:
        print("\nSTATUS: ERROR (prompt exceeds max-model-len)")
    elif remaining_after_request < 0:
        print("\nSTATUS: WARNING (prompt fits, but prompt + requested output does NOT)")
    else:
        print("\nSTATUS: OK (prompt and requested output fit)")


if __name__ == "__main__":
    main()
