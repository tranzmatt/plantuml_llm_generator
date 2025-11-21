#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
import requests


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


def ollama_embed(texts: List[str], embed_model: str, ollama_url: str) -> np.ndarray:
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": embed_model, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    embs = data.get("embeddings")
    if embs is None:
        raise RuntimeError(f"Ollama /api/embed returned no 'embeddings' field: {data}")
    return np.array(embs, dtype="float32")


def ollama_chat(model: str, ollama_url: str, system_msg: str, user_msg: str, num_ctx: int = 200000) -> str:
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "options": {"num_ctx": num_ctx},
        },
        timeout=1800,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    return content


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
    embed_model: str,
    ollama_url: str,
    top_k: int,
) -> str:
    """
    Retrieve RAG examples for a specific diagram type.
    """
    query = f"PlantUML {diagram_type} diagram for a distributed Python microservice application."
    q_emb = ollama_embed([query], embed_model, ollama_url)
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
        description="Generate ALL system-level UML PlantUML .puml diagrams in a SINGLE LLM call using FAISS RAG + Ollama."
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
        "--llm-model",
        default=os.environ.get("RAG_LLM_MODEL", "llama4:maverick"),
        help="LLM model to use via Ollama (default: RAG_LLM_MODEL or llama4:maverick).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (default: OLLAMA_URL or http://localhost:11434).",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", "20")),
        help="How many RAG examples per diagram type (default: 20 or RAG_TOP_K).",
    )

    args = parser.parse_args()

    repo_root = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    repo_name = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    print(f"[repo_to_diagrams] Repo root: {repo_root}")
    print(f"[repo_to_diagrams] Repo name: {repo_name}")
    print(f"[repo_to_diagrams] Output dir: {output_dir}")

    print("[repo_to_diagrams] Collecting Python code from repo...")
    full_repo_text = walk_repo_collect_code(repo_root)
    if not full_repo_text.strip():
        raise RuntimeError("No Python files found in the repo.")
    print(f"      Collected code from repository ({len(full_repo_text)} chars)")

    print(f"[repo_to_diagrams] Loading FAISS RAG from {args.faiss_index} and {args.faiss_meta}")
    index, docs, embed_model = load_faiss_and_meta(args.faiss_index, args.faiss_meta)
    print(f"      Loaded {len(docs)} documents, embedding model: {embed_model}")

    # Get RAG examples for all diagram types up front
    print("\n[repo_to_diagrams] Retrieving RAG examples for each diagram type...")
    rag_examples_by_type: Dict[str, str] = {}
    for dtype, _desc in DIAGRAM_TYPES:
        print(f"[repo_to_diagrams] Retrieving RAG examples for {dtype} diagrams...")
        try:
            rag_examples_by_type[dtype] = get_rag_examples_for_type(
                diagram_type=dtype,
                index=index,
                docs=docs,
                embed_model=embed_model,
                ollama_url=args.ollama_url,
                top_k=args.rag_k,
            )
        except Exception as e:
            print(f"[repo_to_diagrams] Warning: RAG retrieval for {dtype} failed: {e}")
            rag_examples_by_type[dtype] = ""

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

    print("[repo_to_diagrams] Calling Ollama once to generate ALL diagrams...")
    raw_output = ollama_chat(
        model=args.llm_model,
        ollama_url=args.ollama_url,
        system_msg=system_msg,
        user_msg=user_msg,
        num_ctx=200000,
    )

    # Parse the combined output into separate diagrams
    print("\n[print("[repo_to_diagrams] Calling Ollama once to generate ALL diagrams...")] Parsing output and writing diagram files...")
    diagram_types = [dt for dt, _ in DIAGRAM_TYPES]
    parsed = parse_multi_diagram_output(raw_output, diagram_types)

    # Write each diagram to its .puml file
    written_count = 0
    for dtype, _desc in DIAGRAM_TYPES:
        diag_text = parsed.get(dtype, "").strip()
        if not diag_text:
            print(f"[repo_to_diagrams] Warning: no diagram text parsed for type '{dtype}'. Skipping file.")
            continue
        out_name = f"{repo_name}_{dtype}.puml"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(diag_text)
        print(f"[repo_to_diagrams] Wrote {dtype} diagram to {out_path}")
        written_count += 1

    print("\n" + "=" * 70)
    print(f"✓ Done! Generated {written_count} diagrams")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
