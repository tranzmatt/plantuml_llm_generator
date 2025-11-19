#!/usr/bin/env python3
"""
PlantUML diagram generator using vLLM backend exclusively.

This is a streamlined version optimized for vLLM deployment with:
- Simplified configuration (no backend selection)
- vLLM-specific parameter tuning
- Better defaults for large context windows
- Optimized for high-throughput inference
"""

import argparse
import os
import sys
from typing import Dict, List

from core.repo_scanner import scan_python_repo, load_repo_text
from core.rag_retriever import RagRetriever
from core.prompt_builder import build_system_prompt, build_user_prompt
from core.diagram_writer import write_diagrams
from core.utils import get_repo_name, safe_json_loads
from llm_backends.vllm_client import vllm_chat


def build_rag_context(
    retriever: RagRetriever,
    diagram_types: List[str],
    per_type: int = 4,
) -> Dict[str, List[Dict]]:
    """Build RAG context by retrieving examples for each diagram type."""
    ctx = {}
    for dtype in diagram_types:
        query = f"plantuml {dtype} diagram example with correct syntax"
        examples = retriever.search(query, top_k=per_type)
        ctx[dtype] = examples
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PlantUML diagrams using vLLM backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local vLLM server with Llama-4-Maverick
  %(prog)s --input ./myrepo --model meta-llama/Llama-4-Maverick-17B-128E-Instruct

  # Remote vLLM server
  %(prog)s --input ./myrepo --vllm-url http://gpu-server:8000 --model local

  # With custom RAG index
  %(prog)s --input ./myrepo --faiss-index ./custom_rag/index.faiss

Environment Variables:
  VLLM_URL              Base URL for vLLM server (default: http://localhost:8000)
  PLANTUML_EMBED_MODEL  Embedding model for RAG (default: nomic-embed-text)
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i", 
        default=".",
        help="Root folder of source code repository (default: current directory)",
    )
    parser.add_argument(
        "--output", "-o", 
        default="uml",
        help="Output folder for .puml files (default: ./uml)",
    )
    
    # vLLM Configuration
    parser.add_argument(
        "--vllm-url", 
        default=os.environ.get("VLLM_URL", "http://localhost:8000"),
        help="vLLM server base URL (default: $VLLM_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        help="Model name/path on vLLM server (default: Llama-4-Maverick-17B-128E-Instruct)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens in LLM response (default: 8000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation (default: 0.1 for deterministic output)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Request timeout in seconds (default: 1800 = 30 minutes)",
    )
    
    # RAG Configuration
    parser.add_argument(
        "--faiss-index", 
        required=True,
        help="Path to FAISS index file for RAG retrieval",
    )
    parser.add_argument(
        "--faiss-meta", 
        required=True,
        help="Path to metadata JSON file for FAISS index",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("PLANTUML_EMBED_MODEL", "nomic-embed-text"),
        help="Sentence-transformers embedding model (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--rag-examples-per-type",
        type=int,
        default=4,
        help="Number of RAG examples to retrieve per diagram type (default: 4)",
    )
    
    # Validation
    parser.add_argument(
        "--no-validate", 
        action="store_true",
        help="Skip PlantUML validation with 'plantuml -checkonly'",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()

    # Resolve paths
    repo_root = os.path.abspath(args.input)
    if not os.path.isdir(repo_root):
        print(f"Error: Input directory does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)
    
    repo_name = get_repo_name(repo_root)

    # Display configuration
    print("=" * 70)
    print("PlantUML Diagram Generator (vLLM Backend)")
    print("=" * 70)
    print(f"Repository Root:  {repo_root}")
    print(f"Repository Name:  {repo_name}")
    print(f"Output Directory: {args.output}")
    print(f"vLLM Server:      {args.vllm_url}")
    print(f"Model:            {args.model}")
    print(f"Max Tokens:       {args.max_tokens}")
    print(f"Temperature:      {args.temperature}")
    print(f"Validation:       {'Disabled' if args.no_validate else 'Enabled'}")
    print("=" * 70)

    # Step 1: Scan repository
    print("\n[1/5] Scanning Python files...")
    py_files = scan_python_repo(repo_root)
    
    if not py_files:
        print(f"Error: No Python files found in {repo_root}", file=sys.stderr)
        sys.exit(1)
    
    print(f"      Found {len(py_files)} Python files")
    
    if args.verbose:
        for f in py_files[:10]:
            print(f"      - {f}")
        if len(py_files) > 10:
            print(f"      ... and {len(py_files) - 10} more")
    
    repo_text = load_repo_text(py_files)
    print(f"      Loaded {len(repo_text):,} characters of source code")

    # Step 2: Initialize RAG retriever
    print("\n[2/5] Initializing RAG retriever...")
    if not os.path.exists(args.faiss_index):
        print(f"Error: FAISS index not found: {args.faiss_index}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.faiss_meta):
        print(f"Error: FAISS metadata not found: {args.faiss_meta}", file=sys.stderr)
        sys.exit(1)
    
    try:
        retriever = RagRetriever(
            faiss_index_path=args.faiss_index,
            faiss_meta_path=args.faiss_meta,
            embed_model_name=args.embed_model,
        )
        print(f"      Loaded FAISS index with {retriever.index.ntotal} vectors")
    except Exception as e:
        print(f"Error initializing RAG retriever: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Retrieve RAG examples
    diagram_types = [
        "class", "sequence", "activity", "state",
        "component", "deployment", "usecase", "object",
    ]
    
    print(f"\n[3/5] Retrieving RAG examples ({args.rag_examples_per_type} per type)...")
    rag_ctx = build_rag_context(
        retriever, 
        diagram_types, 
        per_type=args.rag_examples_per_type
    )
    
    total_examples = sum(len(examples) for examples in rag_ctx.values())
    print(f"      Retrieved {total_examples} total examples")

    # Step 4: Build prompts and call vLLM
    print("\n[4/5] Generating diagrams with vLLM...")
    system_msg = build_system_prompt()
    user_msg = build_user_prompt(repo_name, repo_text, rag_ctx)
    
    if args.verbose:
        print(f"      System prompt: {len(system_msg)} chars")
        print(f"      User prompt: {len(user_msg)} chars")
        print(f"      Total prompt size: {len(system_msg) + len(user_msg):,} chars")
    
    print(f"      Calling vLLM at {args.vllm_url}...")
    print("      (This may take several minutes for large repositories)")
    
    try:
        raw_response = vllm_chat(
            base_url=args.vllm_url,
            model=args.model,
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"\nError calling vLLM: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print(f"  1. Verify vLLM is running at: {args.vllm_url}", file=sys.stderr)
        print(f"  2. Check model is loaded: {args.model}", file=sys.stderr)
        print("  3. Review vLLM server logs for errors", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"      Response length: {len(raw_response):,} chars")

    # Step 5: Parse and write diagrams
    print("\n[5/5] Parsing and writing diagrams...")
    
    try:
        diagrams = safe_json_loads(raw_response)
    except ValueError as e:
        print(f"\nError parsing LLM response: {e}", file=sys.stderr)
        print("\nRaw response (first 500 chars):", file=sys.stderr)
        print(raw_response[:500], file=sys.stderr)
        sys.exit(1)

    if not isinstance(diagrams, dict):
        print("Error: LLM output is not a JSON object", file=sys.stderr)
        sys.exit(1)

    # Validate all required diagram types are present
    missing = [k for k in diagram_types if k not in diagrams]
    if missing:
        print(f"Error: Missing diagram types in LLM output: {missing}", file=sys.stderr)
        print(f"Received keys: {list(diagrams.keys())}", file=sys.stderr)
        sys.exit(1)

    # Write diagrams to disk
    try:
        write_diagrams(
            output_dir=args.output,
            repo_name=repo_name,
            diagrams=diagrams,
            validate=not args.no_validate,
        )
    except Exception as e:
        print(f"\nError writing diagrams: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("SUCCESS: All diagrams generated")
    print("=" * 70)
    print(f"Output location: {os.path.abspath(args.output)}/")
    print(f"Files generated:")
    for dtype in diagram_types:
        filename = f"{repo_name}_{dtype}.puml"
        print(f"  - {filename}")
    print("\nTo render diagrams:")
    print(f"  plantuml {os.path.abspath(args.output)}/*.puml")
    print("=" * 70)


if __name__ == "__main__":
    main()
