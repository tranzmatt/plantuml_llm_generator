#!/usr/bin/env python3
"""
repo_to_diagrams_chunked_ollama.py

A context-window-safe variant for large codebases.

THREE-PASS MAP-REDUCE PIPELINE:

  Pass 1  EXTRACT  (per file, small prompt)
          Each .py file → compact JSON structural summary
          (classes, methods, relationships, actors, states)
          Tiny output (~1 KB each) always fits in any context window.

  Pass 2  NORMALIZE  (one call, all summaries)
          All summaries → one LLM call → canonical "Entity Registry"
          Authoritative names for every class, module, actor, component.
          This becomes the ground truth injected into every generation prompt.

  Pass 3  GENERATE  (per diagram type)
          Each diagram call receives:
            • The canonical Entity Registry  ← prevents name drift
            • Relevant file chunks (entry-points vs all-files strategy)
            • RAG PlantUML syntax examples
"""

import argparse, json, os, re, sys
from typing import Dict, List, Optional, Tuple
import faiss, numpy as np, requests

# ---------------------------------------------------------------------------
DIAGRAM_TYPES: List[Tuple[str, str]] = [
    ("class",      "Class diagram: modules, services, data structures and relationships."),
    ("sequence",   "Sequence diagram: main runtime flow from inputs to outputs."),
    ("activity",   "Activity diagram: overall workflow and branching logic."),
    ("state",      "State diagram: the most important stateful component."),
    ("component",  "Component diagram: services, queues, and external APIs."),
    ("deployment", "Deployment diagram: runtime nodes, processes, queues, external systems."),
    ("usecase",    "Use-case diagram: main actors and high-level use cases."),
    ("object",     "Object diagram: a runtime snapshot of key objects/instances."),
]

# Diagrams that need ALL files vs. just entry points
ALL_FILES_DIAGRAMS   = {"class", "component", "deployment", "object"}
ENTRY_POINT_DIAGRAMS = {"sequence", "activity", "state", "usecase"}

MAX_CHARS_PER_FILE = 6_000   # per-file budget inside generation prompts

# ===========================================================================
# Ollama helpers
# ===========================================================================

def ollama_embed(texts, embed_model, ollama_url):
    resp = requests.post(f"{ollama_url}/api/embed",
                         json={"model": embed_model, "input": texts}, timeout=600)
    resp.raise_for_status()
    embs = resp.json().get("embeddings")
    if embs is None:
        raise RuntimeError("Ollama /api/embed returned no 'embeddings' field")
    return np.array(embs, dtype="float32")


def ollama_chat(model, ollama_url, system_msg, user_msg, num_ctx=32_768, temperature=0.0):
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "system", "content": system_msg},
                           {"role": "user",   "content": user_msg}],
              "options": {"num_ctx": num_ctx, "temperature": temperature}},
        timeout=1800)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ===========================================================================
# FAISS / RAG helpers
# ===========================================================================

def load_faiss_and_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return index, meta["documents"], meta.get("embed_model", "nomic-embed-text")


def get_rag_examples(diagram_type, index, docs, embed_model, ollama_url, top_k):
    q = f"PlantUML {diagram_type} diagram for a Python application."
    scores, idxs = index.search(ollama_embed([q], embed_model, ollama_url), top_k)
    out = []
    for i, idx in enumerate(idxs[0]):
        d = docs[idx]
        puml = d.get("output") or d.get("plantuml", "")
        if puml:
            out.append(f"--- Example {i+1}:\nInstruction: {d.get('instruction','')}\n\n{puml}\n")
    return "\n".join(out)


# ===========================================================================
# File collection
# ===========================================================================

def collect_files(root):
    result = []
    for dirpath, _, fnames in os.walk(root):
        for fname in sorted(fnames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                code = open(fpath, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            result.append((os.path.relpath(fpath, root), code))
    return result


def is_entry_point(rel_path):
    return os.path.basename(rel_path).lower() in {
        "main.py","app.py","__main__.py","manage.py",
        "cli.py","run.py","server.py","wsgi.py","asgi.py",
    }


# ===========================================================================
# PASS 1 — per-file structural extraction
# ===========================================================================

EXTRACTION_SYSTEM = (
    "You are a senior Python architect doing static analysis. "
    "Extract structural information from a Python file and return ONLY raw JSON. "
    "No prose, no markdown fences."
)

EXTRACTION_TMPL = """\
Analyse this Python file and return a JSON object:

{{
  "file": "<relative path>",
  "classes": [
    {{"name": "<ClassName>", "bases": [], "methods": [], "attributes": []}}
  ],
  "functions": ["<top_level_fn>"],
  "imports":   ["<module_or_symbol>"],
  "relationships": [
    {{"from": "<A>", "to": "<B>", "type": "inherits|uses|creates|calls"}}
  ],
  "actors":  ["<human or external system>"],
  "states":  ["<state machine state>"],
  "summary": "<one sentence role>"
}}

Use exact Python names. Empty categories → []. Output ONLY the JSON.

===== FILE: {rel_path} =====
{source}
"""

def _parse_json(raw):
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\n?```$",        "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def extract_file_structure(rel_path, source, model, ollama_url, num_ctx):
    user_msg = EXTRACTION_TMPL.format(rel_path=rel_path, source=source[:MAX_CHARS_PER_FILE])
    raw = ollama_chat(model, ollama_url, EXTRACTION_SYSTEM, user_msg, num_ctx)
    result = _parse_json(raw)
    if result is None:
        print(f"      [WARN] JSON parse failed for {rel_path}")
    return result


# ===========================================================================
# PASS 2 — normalize extractions → canonical Entity Registry
# ===========================================================================

REGISTRY_SYSTEM = (
    "You are a senior Python architect. "
    "Consolidate per-file structural summaries into a single canonical Entity Registry. "
    "Output ONLY raw JSON — no prose, no markdown fences."
)

REGISTRY_TMPL = """\
Below are structural summaries for every file in the '{repo_name}' repository.
Produce a canonical Entity Registry JSON:

{{
  "repo_name": "{repo_name}",
  "classes": [
    {{
      "canonical_name": "<AuthoritativeName>",
      "defined_in": "<file.py>",
      "bases": [],
      "key_methods": [],
      "relationships": [{{"to": "<Other>", "type": "inherits|uses|creates|calls"}}]
    }}
  ],
  "modules":    [{{"canonical_name": "<Name>", "file": "<file.py>", "role": "<one-line>"}}],
  "actors":     ["<Human or external system>"],
  "components": ["<logical service/component>"],
  "states":     ["<state name>"],
  "entry_points":          ["<file.py>"],
  "top_level_functions":   ["<fn_name>"]
}}

Rules:
- Resolve name discrepancies across files — pick ONE authoritative name.
- Merge duplicates — never list the same class twice with different names.
- Mark entry points (main.py, app.py, __main__.py, etc.).
- Output ONLY the JSON.

===== PER-FILE SUMMARIES =====
{summaries_json}
"""

def build_entity_registry(extractions, repo_name, model, ollama_url, num_ctx):
    user_msg = REGISTRY_TMPL.format(
        repo_name=repo_name,
        summaries_json=json.dumps(extractions, indent=2),
    )
    print(f"      Registry prompt: {len(user_msg):,} chars")
    raw = ollama_chat(model, ollama_url, REGISTRY_SYSTEM, user_msg, num_ctx)
    registry = _parse_json(raw)
    if registry:
        return registry
    # Fallback: hand-craft from raw extractions
    print("      [WARN] Registry parse failed — building fallback registry")
    fb = {"repo_name": repo_name, "classes": [], "modules": [],
          "actors": [], "components": [], "states": [],
          "entry_points": [], "top_level_functions": []}
    for ex in extractions:
        for cls in ex.get("classes", []):
            fb["classes"].append({"canonical_name": cls.get("name","Unknown"),
                                   "defined_in": ex.get("file",""),
                                   "bases": cls.get("bases",[]),
                                   "key_methods": cls.get("methods",[])[:5],
                                   "relationships": []})
        fb["modules"].append({
            "canonical_name": os.path.splitext(os.path.basename(ex.get("file","")))[0],
            "file": ex.get("file",""), "role": ex.get("summary","")})
        fb["actors"].extend(ex.get("actors",[]))
        fb["states"].extend(ex.get("states",[]))
        if is_entry_point(ex.get("file","")):
            fb["entry_points"].append(ex["file"])
    fb["actors"] = list(set(fb["actors"]))
    fb["states"] = list(set(fb["states"]))
    return fb


# ===========================================================================
# PASS 3 — per-diagram generation (registry + file chunks)
# ===========================================================================

GENERATION_SYSTEM = (
    "You are an expert in Python static analysis and UML architecture. "
    "Generate high-quality PlantUML 1.2025.0 diagrams. "
    "You MUST use ONLY the canonical names from the Entity Registry. "
    "Never invent new class or component names. "
    "Output ONLY the @startuml...@enduml block."
)

GENERATION_TMPL = """\
Generate a PlantUML 1.2025.0 {dtype_upper} diagram for '{repo_name}'.

{description}

=== CANONICAL ENTITY REGISTRY (authoritative names — do not deviate) ===
{registry_json}

=== RELEVANT SOURCE CODE ===
{code_chunks}

=== PLANTUML SYNTAX EXAMPLES ===
{rag_examples}

Output ONLY the @startuml...@enduml block.
"""

def select_chunks(diagram_type, files, registry, budget=24_000):
    entry_set = set(registry.get("entry_points", []))
    if diagram_type in ENTRY_POINT_DIAGRAMS:
        ordered = ([(p,c) for p,c in files if p in entry_set or is_entry_point(p)]
                 + [(p,c) for p,c in files if p not in entry_set and not is_entry_point(p)])
    else:
        ordered = files

    chunks, total = [], 0
    for rel, src in ordered:
        block = f"===== FILE: {rel} =====\n{src[:MAX_CHARS_PER_FILE]}\n"
        if total + len(block) > budget:
            rem = budget - total
            if rem > 200:
                chunks.append(f"===== FILE: {rel} (truncated) =====\n{src[:rem]}\n")
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks)


def generate_diagram(dtype, desc, repo_name, registry, files, rag_examples,
                     model, ollama_url, num_ctx):
    user_msg = GENERATION_TMPL.format(
        dtype_upper=dtype.upper(), repo_name=repo_name, description=desc,
        registry_json=json.dumps(registry, indent=2),
        code_chunks=select_chunks(dtype, files, registry),
        rag_examples=rag_examples,
    )
    raw = ollama_chat(model, ollama_url, GENERATION_SYSTEM, user_msg, num_ctx)
    m = re.search(r"@startuml.*?@enduml", raw, re.DOTALL)
    if m:
        return m.group(0).strip()
    return f"@startuml\n{raw.strip()}\n@enduml" if raw.strip() else ""


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Chunked UML generation with coherent names.")
    p.add_argument("--input",  "-i", default=".")
    p.add_argument("--output", "-o", default="uml_out")
    p.add_argument("--faiss-index", default=os.environ.get("RAG_FAISS_INDEX", "rag/faiss.index"))
    p.add_argument("--faiss-meta",  default=os.environ.get("RAG_FAISS_META",  "rag/faiss_meta.json"))
    p.add_argument("--llm-model",   default=os.environ.get("RAG_LLM_MODEL",   "llama3.1:8b"))
    p.add_argument("--ollama-url",  default=os.environ.get("OLLAMA_URL",       "http://localhost:11434"))
    p.add_argument("--rag-k",    type=int, default=int(os.environ.get("RAG_TOP_K", "5")))
    p.add_argument("--num-ctx",  type=int, default=int(os.environ.get("OLLAMA_CTX", "32768")))
    p.add_argument("--save-registry",  action="store_true")
    p.add_argument("--registry-file",  help="Load an existing registry JSON (skips Pass 1+2).")
    args = p.parse_args()

    repo_root  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    repo_name  = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    print("=" * 70)
    print(f"  Chunked pipeline  |  repo: {repo_name}  |  model: {args.llm_model}")
    print("=" * 70)

    # --- Collect files ---
    print("\n[1] Collecting .py files...")
    files = collect_files(repo_root)
    if not files:
        print("ERROR: no .py files found.", file=sys.stderr); sys.exit(1)
    print(f"    {len(files)} files  ({sum(len(c) for _,c in files):,} total chars)")

    # --- Load RAG ---
    print("\n[2] Loading FAISS RAG...")
    index, docs, embed_model = load_faiss_and_meta(args.faiss_index, args.faiss_meta)
    print(f"    {len(docs)} docs, embed model: {embed_model}")

    print("\n[3] Fetching RAG examples per diagram type...")
    rag_by_type = {}
    for dtype, _ in DIAGRAM_TYPES:
        try:
            rag_by_type[dtype] = get_rag_examples(
                dtype, index, docs, embed_model, args.ollama_url, args.rag_k)
            print(f"    ✓ {dtype}")
        except Exception as e:
            print(f"    ✗ {dtype}: {e}"); rag_by_type[dtype] = ""

    # --- Pass 1 + 2 or load registry ---
    if args.registry_file:
        print(f"\n[4] Loading registry from {args.registry_file} (skipping extraction)...")
        registry = json.load(open(args.registry_file))
    else:
        # Pass 1
        print(f"\n[4] Pass 1 — per-file extraction ({len(files)} files)...")
        extractions = []
        for i, (rel, src) in enumerate(files, 1):
            print(f"    [{i:>3}/{len(files)}] {rel}", end="  ", flush=True)
            ex = extract_file_structure(rel, src, args.llm_model, args.ollama_url, args.num_ctx)
            if ex:
                extractions.append(ex)
                print(f"→ {len(ex.get('classes',[]))} classes, {len(ex.get('functions',[]))} fns")
            else:
                print("→ skipped")
        if not extractions:
            print("ERROR: extraction produced nothing.", file=sys.stderr); sys.exit(1)

        # Pass 2
        print(f"\n[5] Pass 2 — normalizing {len(extractions)} extractions → Entity Registry...")
        registry = build_entity_registry(
            extractions, repo_name, args.llm_model, args.ollama_url, args.num_ctx)
        print(f"    {len(registry.get('classes',[]))} classes  "
              f"{len(registry.get('modules',[]))} modules  "
              f"{len(registry.get('components',[]))} components")

    # Always persist registry
    reg_path = os.path.join(output_dir, f"{repo_name}_entity_registry.json")
    json.dump(registry, open(reg_path, "w"), indent=2)
    print(f"\n    Registry saved → {reg_path}")
    print(f"    TIP: --registry-file {reg_path} skips extraction on re-runs.")

    # --- Pass 3 ---
    print(f"\n[6] Pass 3 — generating {len(DIAGRAM_TYPES)} diagrams...")
    written = 0
    for dtype, desc in DIAGRAM_TYPES:
        print(f"\n    ── {dtype.upper()} ──")
        try:
            puml = generate_diagram(dtype, desc, repo_name, registry, files,
                                    rag_by_type.get(dtype,""),
                                    args.llm_model, args.ollama_url, args.num_ctx)
        except Exception as e:
            print(f"    ✗ {e}"); continue
        if not puml.strip():
            print("    ✗ empty output — skipping"); continue
        out_path = os.path.join(output_dir, f"{repo_name}_{dtype}.puml")
        open(out_path, "w").write(puml)
        print(f"    ✓ {out_path}  ({len(puml):,} chars)")
        written += 1

    print(f"\n{'='*70}\n  Done!  {written}/{len(DIAGRAM_TYPES)} diagrams\n{'='*70}")

if __name__ == "__main__":
    main()
