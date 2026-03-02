#!/usr/bin/env python3
"""
estimate_prompt_tokens.py

Standalone utility to estimate token counts for the repo_to_diagrams_* prompt
BEFORE you call vLLM/Ollama.

It can:
- Collect repo code (same approach as repo_to_diagrams scripts)
- Optionally retrieve RAG examples (FAISS + SentenceTransformer), like repo_to_diagrams_vllm_local.py
- Build the exact prompt string in the same "System: ... User: ... Assistant:" format used by vllm_generate()
- Tokenize using a HuggingFace tokenizer for your target model
- Report breakdown + safety headroom against --max-model-len and --max-tokens

Requirements (install if missing):
  pip install transformers sentencepiece
  pip install faiss-cpu sentence-transformers   # only needed if --with-rag

Notes:
- vLLM tokenization should match the model tokenizer closely. This uses AutoTokenizer.
- Some models have slight differences vs vLLM internals; this is still a very good preflight.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

# Keep identical diagram ordering/descriptions to your generator scripts
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
    """
    Collect files under root into one string with separators.
    Default matches your existing scripts (only .py), but you can add more extensions.
    """
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
    import faiss  # type: ignore

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs: List[Dict] = meta["documents"]
    embed_model_name: str = meta.get("embed_model", "nomic-embed-text")
    return index, docs, embed_model_name


def get_rag_examples_for_type(
    diagram_type: str,
    index,
    docs: List[Dict],
    embed_model,
    top_k: int,
) -> str:
    """
    Mirrors repo_to_diagrams_vllm_local.py retrieval style: query -> embed -> faiss search -> examples.
    """
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


def build_user_message(
    repo_name: str,
    full_repo_text: str,
    rag_examples_by_type: Dict[str, str],
) -> str:
    """
    Builds the same user_msg structure as your generator scripts.
    """
    user_parts: List[str] = []
    user_parts.append(
        f"The repository is a Python-based system named '{repo_name}'. "
        f"Below is the entire codebase (all .py files):"
    )
    user_parts.append("")
    user_parts.append("```python")
    user_parts.append(full_repo_text)
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
        examples = (rag_examples_by_type.get(dtype, "") or "").strip() or "(No examples available.)"
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
    """
    Matches repo_to_diagrams_vllm_local.py's vllm_generate() prompt wrapper.
    """
    return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def tokenize_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    p = argparse.ArgumentParser(description="Estimate prompt token usage for UML diagram generator.")
    p.add_argument("--input", "-i", default=".", help="Repo root (default: .)")
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
                   help="HF model id for tokenizer (default: VLLM_MODEL or Llama 3.1 8B Instruct)")
    p.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_LEN", "131072")),
                   help="Context window you intend to use (default: VLLM_MAX_LEN or 131072)")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("VLLM_MAX_TOKENS", "8000")),
                   help="Planned generation tokens (default: VLLM_MAX_TOKENS or 8000)")

    # RAG options
    p.add_argument("--with-rag", action="store_true", help="Include RAG examples (FAISS + SentenceTransformer).")
    p.add_argument("--faiss-index", default=os.environ.get("RAG_FAISS_INDEX", "rag/faiss.index"))
    p.add_argument("--faiss-meta", default=os.environ.get("RAG_FAISS_META", "rag/faiss_meta.json"))
    p.add_argument("--rag-k", type=int, default=int(os.environ.get("RAG_TOP_K", "20")))

    # Repo file types (default matches your scripts)
    p.add_argument("--ext", action="append", default=[".py"],
                   help="File extension to include; repeatable. Default: --ext .py")

    # Output
    p.add_argument("--dump-prompt", default="",
                   help="Optional path to write the full prompt text for inspection.")
    args = p.parse_args()

    repo_root = os.path.abspath(args.input)
    repo_name = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    include_exts = tuple(args.ext)

    print(f"[token_estimator] Repo: {repo_root}")
    print(f"[token_estimator] Include extensions: {include_exts}")

    full_repo_text = walk_repo_collect_code(repo_root, include_exts=include_exts)
    if not full_repo_text.strip():
        raise SystemExit("No files collected. Check --input/--ext settings.")

    # System message matches your generator scripts
    system_msg = (
        "You are an expert in Python static analysis and UML architecture. "
        "You generate high-quality PlantUML 1.2025.0 diagrams for entire systems. "
        "You MUST keep names and semantics consistent across ALL diagrams. "
        "Always follow the requested output format exactly."
    )

    rag_examples_by_type: Dict[str, str] = {dt: "" for dt, _ in DIAGRAM_TYPES}

    if args.with_rag:
        from sentence_transformers import SentenceTransformer  # type: ignore

        print(f"[token_estimator] Loading FAISS index: {args.faiss_index}")
        index, docs, embed_model_name = load_faiss_and_meta(args.faiss_index, args.faiss_meta)

        print(f"[token_estimator] Loading embedder: {embed_model_name}")
        if embed_model_name == "nomic-embed-text":
            model_id = "nomic-ai/nomic-embed-text-v1.5"
        else:
            model_id = embed_model_name
        embed_model = SentenceTransformer(model_id, trust_remote_code=True)

        print(f"[token_estimator] Retrieving RAG examples (top_k={args.rag_k})...")
        for dtype, _ in DIAGRAM_TYPES:
            rag_examples_by_type[dtype] = get_rag_examples_for_type(
                diagram_type=dtype,
                index=index,
                docs=docs,
                embed_model=embed_model,
                top_k=args.rag_k,
            )
    else:
        print("[token_estimator] RAG disabled (--with-rag not set).")

    user_msg = build_user_message(repo_name, full_repo_text, rag_examples_by_type)
    prompt = build_vllm_style_prompt(system_msg, user_msg)

    if args.dump_prompt:
        with open(args.dump_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"[token_estimator] Wrote prompt to: {args.dump_prompt}")

    # Tokenize
    print(f"[token_estimator] Loading tokenizer for: {args.model}")
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    # Breakdown
    sys_tokens = tokenize_len(tok, f"System: {system_msg}\n\n")
    user_tokens = tokenize_len(tok, f"User: {user_msg}\n\n")
    assistant_tokens = tokenize_len(tok, "Assistant:")
    total_tokens = tokenize_len(tok, prompt)

    repo_tokens = tokenize_len(tok, full_repo_text)

    rag_tokens_total = 0
    rag_tokens_by_type: Dict[str, int] = {}
    for dtype, _ in DIAGRAM_TYPES:
        t = tokenize_len(tok, rag_examples_by_type.get(dtype, "") or "")
        rag_tokens_by_type[dtype] = t
        rag_tokens_total += t

    print("")
    print("=" * 80)
    print("TOKEN ESTIMATE")
    print("=" * 80)
    print(f"Model tokenizer:        {args.model}")
    print(f"Max model len:          {args.max_model_len}")
    print(f"Planned output tokens:  {args.max_tokens}")
    print("-" * 80)
    print(f"System wrapper tokens:  {sys_tokens}")
    print(f"User wrapper tokens:    {user_tokens}")
    print(f"Assistant tag tokens:   {assistant_tokens}")
    print(f"Repo code tokens:       {repo_tokens}")
    print(f"RAG tokens total:       {rag_tokens_total}   (enabled={args.with_rag})")
    if args.with_rag:
        for dtype, _ in DIAGRAM_TYPES:
            print(f"  - {dtype:10s}: {rag_tokens_by_type[dtype]}")
    print("-" * 80)
    print(f"TOTAL prompt tokens:    {total_tokens}")
    print("=" * 80)

    # Headroom checks
    remaining_for_output = args.max_model_len - total_tokens
    remaining_after_request = args.max_model_len - (total_tokens + args.max_tokens)

    print("")
    print("HEADROOM")
    print("-" * 80)
    print(f"Remaining tokens for output (max): {remaining_for_output}")
    print(f"Remaining after requesting max_tokens={args.max_tokens}: {remaining_after_request}")

    if remaining_for_output < 0:
        print("")
        print("ERROR: Prompt exceeds max-model-len. Reduce input (repo, RAG, etc.) or raise max-model-len.")
    elif remaining_after_request < 0:
        print("")
        print("WARNING: Prompt fits, but prompt + requested output does NOT. Lower --max-tokens or shrink input.")
    else:
        print("")
        print("OK: Prompt and requested output fit within max-model-len.")


if __name__ == "__main__":
    main()
