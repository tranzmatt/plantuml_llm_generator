#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

from core.repo_scanner import scan_python_repo, load_repo_text
from core.rag_retriever import RagRetriever
from core.prompt_builder import build_system_prompt, build_user_prompt
from core.diagram_writer import write_diagrams
from core.utils import get_repo_name, safe_json_loads
from llm_backends.ollama_client import ollama_chat
from llm_backends.vllm_client import vllm_chat


def build_rag_context(
    retriever: RagRetriever,
    diagram_types: List[str],
    per_type: int = 4,
):
    ctx = {}
    for dtype in diagram_types:
        query = f"plantuml {dtype} diagram example with correct syntax"
        examples = retriever.search(query, top_k=per_type)
        ctx[dtype] = examples
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PlantUML diagrams for a Python repo using RAG + LLM."
    )
    parser.add_argument(
        "--input", "-i", default=".",
        help="Root folder of source code files (default: current directory)",
    )
    parser.add_argument(
        "--output", "-o", default="uml",
        help="Folder to store output .puml files (default: ./uml)",
    )
    parser.add_argument(
        "--backend", choices=["ollama", "vllm"], default="ollama",
        help="LLM backend to use (default: ollama)",
    )
    parser.add_argument(
        "--model", "-m", default=os.environ.get("PLANTUML_LLM_MODEL", "llama4:maverick"),
        help="Model name to use on the selected backend.",
    )
    parser.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (default: env OLLAMA_URL or http://localhost:11434)",
    )
    parser.add_argument(
        "--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000"),
        help="vLLM base URL for OpenAI-compatible API.",
    )
    parser.add_argument(
        "--faiss-index", required=True,
        help="Path to FAISS index file for RAG.",
    )
    parser.add_argument(
        "--faiss-meta", required=True,
        help="Path to metadata JSON file corresponding to the FAISS index.",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("PLANTUML_EMBED_MODEL", "nomic-embed-text"),
        help="Sentence-transformers embedding model name (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Disable local PlantUML validation (plantuml -checkonly).",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.input)
    repo_name = get_repo_name(repo_root)

    print(f"[repo_to_diagrams] Repo root: {repo_root}")
    print(f"[repo_to_diagrams] Repo name: {repo_name}")
    print(f"[repo_to_diagrams] Output dir: {args.output}")
    print(f"[repo_to_diagrams] Backend: {args.backend}")
    print(f"[repo_to_diagrams] Model: {args.model}")

    print("[repo_to_diagrams] Scanning Python files...")
    py_files = scan_python_repo(repo_root)
    print(f"[repo_to_diagrams] Found {len(py_files)} Python files.")
    repo_text = load_repo_text(py_files)

    print("[repo_to_diagrams] Initializing RAG retriever...")
    retriever = RagRetriever(
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        embed_model_name=args.embed_model,
    )

    diagram_types = [
        "class", "sequence", "activity", "state",
        "component", "deployment", "usecase", "object",
    ]
    print("[repo_to_diagrams] Retrieving RAG examples...")
    rag_ctx = build_rag_context(retriever, diagram_types)

    system_msg = build_system_prompt()
    user_msg = build_user_prompt(repo_name, repo_text, rag_ctx)

    print(f"[repo_to_diagrams] Calling backend '{args.backend}'...")
    if args.backend == "ollama":
        raw = ollama_chat(
            ollama_url=args.ollama_url,
            model=args.model,
            system_msg=system_msg,
            user_msg=user_msg,
        )
    else:
        raw = vllm_chat(
            base_url=args.vllm_url,
            model=args.model,
            system_msg=system_msg,
            user_msg=user_msg,
        )

    print("[repo_to_diagrams] Parsing JSON from LLM output...")
    diagrams = safe_json_loads(raw)

    if not isinstance(diagrams, dict):
        raise ValueError("LLM output JSON is not an object mapping diagram types to PlantUML diagrams.")

    # Ensure all eight keys exist
    missing = [k for k in diagram_types if k not in diagrams]
    if missing:
        raise ValueError(f"LLM JSON is missing keys: {missing}")

    print("[repo_to_diagrams] Writing diagrams to disk...")
    write_diagrams(
        output_dir=args.output,
        repo_name=repo_name,
        diagrams=diagrams,
        validate=not args.no_validate,
    )

    print("[repo_to_diagrams] Done.")


if __name__ == "__main__":
    main()
