#!/usr/bin/env python3
"""
Generate PlantUML diagrams using local vLLM (Python library).
Based on the working Ollama version but uses vLLM for inference.
"""
import argparse
import json
import os
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams


# Fixed order, as requested
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


def walk_repo_collect_code(root: str) -> str:
    """
    Collect all .py files under root into a single big string with separators.
    You can extend this later to include yaml / docker-compose, etc.
    """
    chunks: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    code = f.read()
            except UnicodeDecodeError:
                # Skip weird encodings
                continue
            rel = os.path.relpath(fpath, root)
            chunks.append(f"===== FILE: {rel} =====\n{code}\n")
    return "\n".join(chunks)


def vllm_generate(
    llm: LLM,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> str:
    """
    Generate text using vLLM.
    Formats messages and uses vLLM's generate method.
    """
    # Combine system and user messages
    # Most models expect a format like:
    # <|system|>system message<|user|>user message<|assistant|>
    # But we'll use a simple format that works well:
    prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def load_faiss_and_meta(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict], str]:
    """
    Load FAISS index and metadata.
    Expects metadata format: {"embed_model": "...", "documents": [...]}
    """
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs: List[Dict] = meta["documents"]
    embed_model: str = meta.get("embed_model", "nomic-embed-text")
    return index, docs, embed_model


def get_rag_examples_for_type(
    diagram_type: str,
    index: faiss.Index,
    docs: List[Dict],
    embed_model: SentenceTransformer,
    top_k: int,
) -> str:
    """
    Retrieve RAG examples for a specific diagram type.
    """
    query = f"PlantUML {diagram_type} diagram for a distributed Python microservice application."
    # Use the cached model directly
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)

    examples: List[str] = []
    for i, idx in enumerate(indices[0]):
        d = docs[idx]
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


def parse_multi_diagram_output(
    text: str,
    diagram_types: List[str],
) -> Dict[str, str]:
    """
    Parse LLM output that uses section headers like:

    ### CLASS
    @startuml
    ...
    @enduml

    ### SEQUENCE
    ...

    Returns a dict: { 'class': '...puml...', 'sequence': '...puml...', ... }
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Build a regex that matches sections starting with ### <NAME>
    # We'll map our known types to headers: CLASS, SEQUENCE, ACTIVITY, STATE, COMPONENT, DEPLOYMENT, USECASE, OBJECT
    header_map = {
        "class": "CLASS",
        "sequence": "SEQUENCE",
        "activity": "ACTIVITY",
        "state": "STATE",
        "component": "COMPONENT",
        "deployment": "DEPLOYMENT",
        "usecase": "USE CASE",
        "object": "OBJECT",
    }

    # We'll split on lines like: ### CLASS, ### SEQUENCE, etc.
    pattern = r"^###\s+([A-Z ]+)\s*$"
    sections = re.split(pattern, text, flags=re.MULTILINE)

    # re.split gives something like: [before, HEADER1, content1, HEADER2, content2, ...]
    # We ignore the first "before" element.
    parsed: Dict[str, str] = {t: "" for t in diagram_types}

    if len(sections) <= 1:
        # Fallback: maybe the model just output a single @startuml...@enduml
        # We cannot reliably split, so dump everything into 'class' as a last resort.
        parsed["class"] = text.strip()
        return parsed

    # sections[0] is preamble, then pairs of (HEADER, CONTENT)
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        header = sections[i].strip().upper()
        content = sections[i + 1].strip()

        # Normalize header to a diagram_type key
        for dtype, hname in header_map.items():
            if header == hname:
                # Extract just the @startuml...@enduml block if present
                m = re.search(r"@startuml.*?@enduml", content, flags=re.DOTALL)
                if m:
                    parsed[dtype] = m.group(0).strip()
                else:
                    parsed[dtype] = content
                break

    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALL system-level UML PlantUML .puml diagrams in a SINGLE LLM call using FAISS RAG + local vLLM."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=".",
        help="Root folder of the source repo (default: current directory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="uml_out",
        help="Output directory for .puml files (default: uml_out).",
    )
    parser.add_argument(
        "--faiss-index",
        default=os.environ.get("RAG_FAISS_INDEX", "rag/faiss.index"),
        help="FAISS index path (default: RAG_FAISS_INDEX or rag/faiss.index).",
    )
    parser.add_argument(
        "--faiss-meta",
        default=os.environ.get("RAG_FAISS_META", "rag/faiss_meta.json"),
        help="FAISS metadata JSON path (default: RAG_FAISS_META or rag/faiss_meta.json).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b"),
        help="vLLM model to use (default: VLLM_MODEL or openai/gpt-oss-120b).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.environ.get("VLLM_TP", "4")),
        help="Tensor parallel size (number of GPUs, default: VLLM_TP or 4).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_LEN", "32000")),
        help="Max model length (default: VLLM_MAX_LEN or 32000).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("VLLM_MAX_TOKENS", "8000")),
        help="Max tokens to generate (default: VLLM_MAX_TOKENS or 8000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("VLLM_TEMPERATURE", "0.0")),
        help="Sampling temperature (default: VLLM_TEMPERATURE or 0.0).",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", "20")),
        help="How many RAG examples per diagram type (default: 20 or RAG_TOP_K).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (default: 0.95).",
    )

    args = parser.parse_args()

    repo_root = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    repo_name = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    print("=" * 70)
    print("PlantUML Diagram Generator (Local vLLM)")
    print("=" * 70)
    print(f"Repo root:       {repo_root}")
    print(f"Repo name:       {repo_name}")
    print(f"Output dir:      {output_dir}")
    print(f"Model:           {args.model}")
    print(f"Tensor Parallel: {args.tp} GPUs")
    print(f"Max Model Len:   {args.max_model_len}")
    print(f"Max Tokens:      {args.max_tokens}")
    print(f"Temperature:     {args.temperature}")
    print(f"RAG Top-K:       {args.rag_k}")
    print("=" * 70)

    print("\n[1/5] Collecting Python code from repo...")
    full_repo_text = walk_repo_collect_code(repo_root)
    if not full_repo_text.strip():
        raise RuntimeError("No Python files found in the repo.")
    print(f"      Collected code from repository ({len(full_repo_text)} chars)")

    print(f"\n[2/5] Loading FAISS RAG from {args.faiss_index} and {args.faiss_meta}")
    index, docs, embed_model_name = load_faiss_and_meta(args.faiss_index, args.faiss_meta)
    print(f"      Loaded {len(docs)} documents, embedding model: {embed_model_name}")
    
    # Load embedding model once for all RAG queries
    print(f"      Loading embedding model...")
    if embed_model_name == "nomic-embed-text":
        model_id = "nomic-ai/nomic-embed-text-v1.5"
    else:
        model_id = embed_model_name
    embed_model = SentenceTransformer(model_id, trust_remote_code=True)
    print(f"      Embedding model loaded")

    # Get RAG examples for all diagram types up front
    print("\n[3/5] Retrieving RAG examples for each diagram type...")
    rag_examples_by_type: Dict[str, str] = {}
    for dtype, _desc in DIAGRAM_TYPES:
        print(f"      - Retrieving {dtype} examples...")
        try:
            rag_examples_by_type[dtype] = get_rag_examples_for_type(
                diagram_type=dtype,
                index=index,
                docs=docs,
                embed_model=embed_model,
                top_k=args.rag_k,
            )
        except Exception as e:
            print(f"      Warning: RAG retrieval for {dtype} failed: {e}")
            rag_examples_by_type[dtype] = ""

    print("\n[4/5] Loading vLLM model (this may take a few minutes)...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("      Model loaded successfully")

    system_msg = (
        "You are an expert in Python static analysis and UML architecture. "
        "You generate high-quality PlantUML 1.2025.0 diagrams for entire systems. "
        "You MUST keep names and semantics consistent across ALL diagrams. "
        "Always follow the requested output format exactly."
    )

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
        examples = rag_examples_by_type.get(dtype, "").strip() or "(No examples available.)"

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

    user_msg = "\n".join(user_parts)

    print("\n[5/5] Generating all diagrams in one LLM call...")
    raw_output = vllm_generate(
        llm=llm,
        system_msg=system_msg,
        user_msg=user_msg,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Parse the combined output into separate diagrams
    print("\n[6/6] Parsing output and writing diagram files...")
    diagram_types = [dt for dt, _ in DIAGRAM_TYPES]
    parsed = parse_multi_diagram_output(raw_output, diagram_types)

    # Write each diagram to its .puml file
    written_count = 0
    for dtype, _desc in DIAGRAM_TYPES:
        diag_text = parsed.get(dtype, "").strip()
        if not diag_text:
            print(f"      Warning: no diagram text parsed for type '{dtype}'. Skipping file.")
            continue
        out_name = f"{repo_name}_{dtype}.puml"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(diag_text)
        print(f"      ✓ Wrote {dtype} diagram to {out_path}")
        written_count += 1

    print("\n" + "=" * 70)
    print(f"✓ Done! Generated {written_count} diagrams")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
