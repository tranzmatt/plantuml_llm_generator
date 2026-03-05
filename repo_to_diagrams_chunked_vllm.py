#!/usr/bin/env python3
"""
repo_to_diagrams_chunked_vllm.py

vLLM version of the three-pass chunked pipeline.

KEY OPTIMIZATION vs the Ollama version:
  - Model is loaded ONCE into GPU memory at startup and reused for all passes.
    No repeated load/unload cycles — critical for large models like llama4:scout.
  - Pass 1 (per-file extraction) is BATCHED: all file prompts are submitted to
    llm.generate() in a single call, letting vLLM fill all 4 GPUs in parallel.
  - Pass 2 (registry normalization) and Pass 3 (diagram generation) are also
    single calls — vLLM handles scheduling internally.

Usage:
    python repo_to_diagrams_chunked_vllm.py \
        --input  ~/Code/MyProject \
        --output uml_out \
        --model  meta-llama/Llama-4-Scout-17B-16E-Instruct \
        --tp     4 \
        --faiss-index rag/faiss.index \
        --faiss-meta  rag/faiss_meta.json
"""

# ---------------------------------------------------------------------------
# vLLM env vars MUST be set before importing vllm
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("VLLM_USE_V1",                   "1")
os.environ.setdefault("VLLM_NO_CUDA_GRAPH",            "1")
os.environ.setdefault("VLLM_ENFORCE_EAGER",            "1")
os.environ.setdefault("VLLM_USE_MODELSCOPE",           "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD",  "spawn")
os.environ.setdefault("VLLM_MAX_NUM_SEQS",             "32")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION",   "0.85")

import json, re, sys, argparse
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Diagram catalogue
# ---------------------------------------------------------------------------
DIAGRAM_TYPES: List[Tuple[str, str]] = [
    ("class",      "Class diagram: modules, services, data structures and their relationships."),
    ("sequence",   "Sequence diagram: main runtime flow from inputs to outputs."),
    ("activity",   "Activity diagram: overall workflow and branching logic."),
    ("state",      "State diagram: the most important stateful component."),
    ("component",  "Component diagram: services, queues, and external APIs."),
    ("deployment", "Deployment diagram: runtime nodes, processes, queues, external systems."),
    ("usecase",    "Use-case diagram: main actors and high-level use cases."),
    ("object",     "Object diagram: a runtime snapshot of key objects/instances."),
]

ALL_FILES_DIAGRAMS   = {"class", "component", "deployment", "object"}
ENTRY_POINT_DIAGRAMS = {"sequence", "activity", "state", "usecase"}

MAX_CHARS_PER_FILE       = 6_000
GENERATION_CHUNK_BUDGET  = 24_000

# ---------------------------------------------------------------------------
# Per-diagram syntax rules injected into every generation prompt
# ---------------------------------------------------------------------------
DIAGRAM_SYNTAX_HINTS: Dict[str, str] = {
    "activity": (
        "Use ONLY modern activity syntax:\n"
        "  start / stop\n"
        "  :Action label;\n"
        "  if (condition?) then (yes) / else (no) / endif\n"
        "  fork / fork again / end fork\n"
        "  while (condition?) / endwhile\n"
        "NEVER use (*) --> or --> (*) legacy syntax."
    ),
    "state": (
        "Correct transition syntax: StateName --> OtherState : label\n"
        "NEVER write -->|label| — that is Mermaid and will fail.\n"
        "Composite states: state StateName { [*] --> SubState }\n"
        "Entry/exit use [*]."
    ),
    "usecase": (
        'Rectangle names with spaces MUST be quoted: rectangle "My Service" { }\n'
        'Actor names with spaces must be quoted: actor "End User" as u\n'
        'Use case names with spaces must be quoted: usecase "Do Something" as UC1'
    ),
    "class": (
        "STRICT CLASS-ONLY OUTPUT (no mixing):\n"
        "- Use ONLY: class, abstract class, interface, enum, package, namespace, note, hide/show, skinparam.\n"
        "- Do NOT use: component, participant, node, queue, cloud, artifact, rectangle, frame, actor, object.\n"
        "- Do NOT use allowmixing.\n"
        "\n"
        "CRITICAL — class body declarations and relationship arrows MUST be on SEPARATE lines:\n"
        "  WRONG:  class LocalFileWriter <|-- FileWriter {\n"
        "  WRONG:  class A --|> IFoo\n"
        "  CORRECT:\n"
        "    class LocalFileWriter {\n"
        "    }\n"
        "    LocalFileWriter <|-- FileWriter\n"
        "\n"
        "NEVER use 'extends', 'implements', or 'inherits' keywords — use PlantUML arrows:\n"
        "  WRONG:  class Foo extends Bar\n"
        "  CORRECT: Foo <|-- Bar\n"
        "\n"
        "Inheritance:  Child <|-- Parent\n"
        "Realisation:  Class ..|> Interface\n"
        "Composition:  ClassA *-- ClassB\n"
        "Aggregation:  ClassA o-- ClassB\n"
        "Association:  ClassA --> ClassB\n"
        "Members: + public, - private, # protected\n"
        "Hard constraints:\n"
        "- Do not mix other diagram element families here (no component/queue/node/participant/object).\n"
        "- Relationship arrows go on their OWN lines, never on a class declaration line."
    ),
    "sequence": (
        "Declare all participants at the top before any arrows.\n"
        'participant "Name" as alias\n'
        "Sync call: A -> B : message\n"
        "Return:    B --> A : response\n"
        "activate B / deactivate B\n"
        "Groups: alt/else/end, loop, opt\n"
        "Hard constraints:\n"
        "- Every participant referenced in a message must be declared before messages.\n"
        "- If a participant name contains spaces, quote it and/or use an alias; message endpoints should be aliases."
    ),
    "component": (
        'Components/packages/frames with spaces must be quoted.\n'
        'component "My Component" as alias\n'
        'interface "My API" as api\n'
        'queue "message_queue" as mq\n'
        "\n"
        "Declare ALL elements before edges.\n"
        "Edges must use --> or ..> only.\n"
        "Never use .>.\n"
        "Never create new keywords like exchange or topic.\n"
        "Represent messaging exchanges as stereotypes:\n"
        'queue "celebrity_names" <<exchange>>\n'
        "Links: A --> B, A ..> B (dependency)\n"
        "Hard constraints (avoid common syntax traps):\n"
        "- Do NOT define/alias an element inline inside a link line. (INVALID: `A --> \"Thing\" as T`). Declare first, then connect.\n"
        "- One edge per line. Do NOT use comma-separated targets. (INVALID: `A --> B, C`).\n"
        "- Do not mix other diagram element families (e.g., class/object/participant) in a component diagram."
        "\n"
        "Do NOT declare elements inline on edges (e.g. A --> interface \"X\" as x). Declare first, then connect."
    ),
    "deployment": (
        'Nodes/clouds/frames with spaces must be quoted: node "My Node" { }\n'
        'cloud "AWS" { }\n'
        'frame "VPC" { }\n'
        "Artifacts: artifact \"app.jar\"\n"
        "Links: A --> B : label\n"
        "Hard constraints:\n"
        "- There is no `exchange` keyword. If you need to model a RabbitMQ exchange/topic, represent it as an `artifact` or `component`\n"
        "  with a descriptive label (e.g., `artifact \"exchange: celebrity_names\" as EX`).\n"
        "- Do NOT define/alias an element inline inside a link line. Declare first, then connect.\n"
        "- One edge per line. Do NOT use comma-separated targets."
    ),
    "object": (
        "Object diagrams should only use `object` declarations and links between objects.\n"
        'object "name:Type" as o1\n'
        "Links: o1 -- o2\n"
        "Hard constraints:\n"
        "- Do not include queue/component/class/node/participant elements.\n"
        "- If mixing is unavoidable, add `allowmixing`, but prefer separate diagrams."
        'Object instances: object "instanceName : ClassName" as alias\n'
        "Field values: alias : field = value\n"
        "Links same as class diagram."
    ),
}


def lint_plantuml(diagram_type: str, puml: str) -> List[str]:
    """Heuristic lint rules to prevent common PlantUML syntax errors."""

    issues: List[str] = []
    if not puml:
        return issues

    lines = [ln.rstrip() for ln in puml.splitlines()]

    # Mixing diagram element families (often triggers "use allowmixing").
    if diagram_type == "class":
        banned_prefixes = ("component ", "queue ", "node ", "participant ", "actor ", "object ")
        if any(ln.lstrip().startswith(banned_prefixes) for ln in lines):
            issues.append("Class diagram contains non-class elements (component/queue/node/participant/object). Keep to class syntax only.")

        # Detect class/interface/abstract declarations that incorrectly embed a
        # relationship arrow on the same line, e.g.:
        #   class Foo <|-- Bar {        ← INVALID
        #   abstract class A --|> B     ← INVALID
        # PlantUML requires: class body declaration and relationship arrows on
        # separate lines.
        _cls_decl_arrow_re = re.compile(
            r"^\s*(abstract\s+class|class|interface|enum)\s+\S.*"
            r"(<\|--|--\|>|\*--|--\*|o--|--o|<--|-+>|\.\.>|<\.\.)",
            re.IGNORECASE,
        )
        for ln in lines:
            if _cls_decl_arrow_re.match(ln):
                issues.append(
                    "Class declaration line contains a relationship arrow — PlantUML does NOT "
                    "allow combining 'class Foo <|-- Bar {' on one line. "
                    "Declare the class body separately, then add the arrow on its own line: "
                    "'class Foo { }\\nFoo <|-- Bar'"
                )
                break

        # Detect Java/Python-style inheritance keywords that are not valid PlantUML.
        _java_inherit_re = re.compile(
            r"^\s*(class|abstract\s+class|interface)\s+\S+\s+(extends|implements|inherits)\s+",
            re.IGNORECASE,
        )
        if any(_java_inherit_re.match(ln) for ln in lines):
            issues.append(
                "Class diagram uses Java/Python-style 'extends'/'implements' keywords. "
                "Use PlantUML arrows instead: Child <|-- Parent  or  Class ..|> Interface"
            )

    if diagram_type == "object":
        banned_prefixes = ("class ", "component ", "queue ", "node ", "participant ", "actor ")
        if any(ln.lstrip().startswith(banned_prefixes) for ln in lines):
            issues.append("Object diagram contains non-object elements (class/component/queue/node/participant). Keep to object syntax only.")

    # Inline alias/definition inside an edge (e.g., `A --> "X" as X`).
    arrow_pat = re.compile(r"(--?>|\.\.|-\?>|<--|<\.\.)")
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("'"):
            continue
        if arrow_pat.search(s) and re.search(r"\bas\s+\w+\b", s) and not s.startswith((
            "participant ", "actor ", "component ", "artifact ", "node ", "class ", "object ",
        )):
            issues.append("Edge line defines/aliases an element inline (contains `as ...`). Declare it on its own line, then connect.")
            break

    # Comma-separated targets in an edge (PlantUML expects one edge per line).
    for ln in lines:
        s = ln.strip()
        if arrow_pat.search(s) and "," in s:
            issues.append("Edge line targets multiple nodes with commas. Use one edge per line.")
            break

    # Sequence: message references to names with spaces should be quoted/aliased.
    if diagram_type == "sequence":
        for ln in lines:
            s = ln.strip()
            if re.search(r"-+>>", s) or re.search(r"-+>", s):
                m = re.search(r"-+>>?\s*([^:]+?)\s*:", s)
                if m:
                    rhs = m.group(1).strip()
                    if " " in rhs and not (rhs.startswith('"') and rhs.endswith('"')):
                        issues.append("Sequence message targets a name with spaces but without quotes/alias. Declare participants and use aliases.")
                        break

    # Deployment: `exchange` isn't a PlantUML keyword.
    if diagram_type == "deployment":
        if any(re.match(r"\s*exchange\b", ln) for ln in lines):
            issues.append("Deployment diagram uses `exchange` keyword. Model exchanges as `artifact` or `component` instead.")

    return issues

# ===========================================================================
# vLLM helpers — model loaded ONCE, reused everywhere
# ===========================================================================

def load_model(model: str, tp: int, max_model_len: int,
               gpu_memory_utilization: float, enforce_eager: bool) -> LLM:
    """Load the vLLM model once. Pass the returned object to all generate calls."""
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )


def make_sampling_params(max_tokens: int, temperature: float = 0.0) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=1.0,
        top_k=1,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
    )


def format_prompt(tokenizer, system_msg: str, user_msg: str) -> str:
    """
    Format a chat prompt using the model's own chat template if available,
    falling back to a plain System/User/Assistant format.
    """
    try:
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def vllm_generate_batch(
    llm: LLM,
    prompts: List[str],
    max_tokens: int,
    temperature: float = 0.0,
) -> List[str]:
    """
    Submit a batch of prompts in ONE llm.generate() call.
    vLLM schedules them across all GPUs automatically.
    Results are returned in the same order as prompts.
    """
    sampling = make_sampling_params(max_tokens, temperature)
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]


def vllm_generate_one(llm: LLM, prompt: str,
                      max_tokens: int, temperature: float = 0.0) -> str:
    return vllm_generate_batch(llm, [prompt], max_tokens, temperature)[0]


# ===========================================================================
# FAISS / RAG helpers  (embeddings via SentenceTransformer, not Ollama)
# ===========================================================================

def load_faiss_and_meta(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return index, meta["documents"], meta.get("embed_model", "nomic-embed-text")


def load_embed_model(embed_model_name: str) -> SentenceTransformer:
    model_id = (
        "nomic-ai/nomic-embed-text-v1.5"
        if embed_model_name == "nomic-embed-text"
        else embed_model_name
    )
    return SentenceTransformer(model_id, trust_remote_code=True)


def get_rag_examples(
    diagram_type: str,
    index: faiss.Index,
    docs: List[Dict],
    embed_model: SentenceTransformer,
    top_k: int,
) -> str:
    query = f"PlantUML {diagram_type} diagram for a Python application."
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    out = []
    for i, idx in enumerate(idxs[0]):
        d = docs[idx]
        puml = d.get("output") or d.get("plantuml", "")
        if puml:
            out.append(
                f"--- Example {i+1} (score={scores[0][i]:.4f}):\n"
                f"Instruction: {d.get('instruction','')}\n\n{puml}\n"
            )
    return "\n".join(out)


# ===========================================================================
# File collection
# ===========================================================================

def collect_files(root: str) -> List[Tuple[str, str]]:
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


def is_entry_point(rel_path: str) -> bool:
    return os.path.basename(rel_path).lower() in {
        "main.py","app.py","__main__.py","manage.py",
        "cli.py","run.py","server.py","wsgi.py","asgi.py",
    }


# ===========================================================================
# Prompt builders
# ===========================================================================

EXTRACTION_SYSTEM = (
    "You are a senior Python architect doing static analysis. "
    "Extract structural information and return ONLY raw JSON. No prose, no markdown fences."
)

EXTRACTION_TMPL = """\
Analyse this Python file and return a JSON object:

{{
  "file": "<relative path>",
  "classes": [
    {{"name": "<ClassName>", "bases": [], "methods": [], "attributes": []}}
  ],
  "functions":     ["<top_level_fn>"],
  "imports":       ["<module_or_symbol>"],
  "relationships": [{{"from": "<A>", "to": "<B>", "type": "inherits|uses|creates|calls"}}],
  "actors":        ["<human or external system>"],
  "states":        ["<state machine state>"],
  "summary":       "<one sentence role>"
}}

Use exact Python names. Empty categories → []. Output ONLY the JSON.

===== FILE: {rel_path} =====
{source}
"""

REGISTRY_SYSTEM = (
    "You are a senior Python architect. "
    "Consolidate per-file structural summaries into a canonical Entity Registry. "
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
  "modules":               [{{"canonical_name": "<Name>", "file": "<file.py>", "role": "<one-line>"}}],
  "actors":                ["<Human or external system>"],
  "components":            ["<logical service/component>"],
  "states":                ["<state name>"],
  "entry_points":          ["<file.py>"],
  "top_level_functions":   ["<fn_name>"]
}}

Rules:
- ONE authoritative name per class — resolve cross-file discrepancies.
- Never list the same class twice with different names.
- Output ONLY the JSON.

===== PER-FILE SUMMARIES =====
{summaries_json}
"""

GENERATION_SYSTEM = (
    "You are an expert in Python static analysis and UML architecture. "
    "Generate strictly valid PlantUML 1.2025.10 diagrams. "
    "Use ONLY canonical names from the Entity Registry — never invent new names. "
    "Output ONLY the @startuml...@enduml block.\n\n"
    "STRICT PLANTUML 1.2025.10 SYNTAX RULES:\n"
    "- Activity: modern syntax only (start/stop/:action;/fork). NEVER (*) -->.\n"
    "- State: transitions use '--> State : label'. NEVER -->|label|.\n"
    "- Use case/component/deployment: ALL multi-word names must be double-quoted.\n"
    "- Class: <|-- inherit, *-- compose, o-- aggregate, --> associate.\n"
    "- Sequence: declare all participants before first arrow."
    "PLANTUML VALIDATION CONTRACT:\n"
    "- Never output PlantUML error text.\n"
    "- Never reference undeclared elements/aliases.\n"
    "- Every element must be declared before it is used in an edge.\n"
    "- Do NOT declare elements inline on edges.\n"
    "- Forbidden legacy/mistyped tokens: 'exchange' keyword, '.>' arrows, '->>' arrows.\n\n"

    "ALLOWED KEYWORDS (global):\n"
    "- component, interface, database, cloud, node, artifact, actor, package, rectangle, frame, queue,\n"
    "  participant, object, class, enum, note, skinparam, title, legend, left to right direction.\n"
    "- Forbidden keywords: exchange, topic, fanout, pubsub, service, lambda.\n"
    "- If you need an exchange/topic concept, represent it as a queue with a stereotype:\n"
    "  queue \"X\" <<exchange>>\n\n"

    "EDGE RULES:\n"
    "- Allowed arrows: -->, <--> , ..> , <.. , ..|> , <|-- , *-- , o--.\n"
    "- NEVER use .> or ->> or =>.\n"
    "- Never declare new elements inline on an edge.\n"
    "- All elements must be declared first, then edges listed afterward.\n\n"

    "QUOTING RULES:\n"
    "- Names containing spaces MUST be quoted.\n"
    "- Aliases must be used consistently after declaration.\n\n"

    "If any generated line violates these rules, rewrite it before output."
)

GENERATION_TMPL = """\
Generate a PlantUML 1.2025.10 {dtype_upper} diagram for '{repo_name}'.

{description}

=== SYNTAX RULES FOR {dtype_upper} ===
{syntax_hint}

=== CANONICAL ENTITY REGISTRY (use ONLY these names) ===
{registry_json}

=== RELEVANT SOURCE CODE ===
{code_chunks}

=== PLANTUML SYNTAX EXAMPLES ===
{rag_examples}

Output ONLY the @startuml...@enduml block.
"""

REPAIR_TMPL = """\
Your previous PlantUML for diagram type {dtype_upper} failed validation for PlantUML 1.2025.10.

Validation errors:
{errors}

Rewrite the ENTIRE diagram to satisfy ALL constraints.
- Output ONLY one @startuml...@enduml block.
- Declare all elements first, then edges.
- Do not use 'exchange' keyword; use queue stereotypes instead.
- Never use '.>' or '->>' arrows.
- Do not mix unrelated element types for this diagram type.

Previous output:
{previous}
"""

# ===========================================================================
# JSON parsing helper
# ===========================================================================

def _parse_json(raw: str) -> Optional[Dict]:
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


# ===========================================================================
# PlantUML validation helpers (lightweight; targets the failures we've seen)
# ===========================================================================

_FORBIDDEN_LINE_RE = [
    re.compile(r"^\s*exchange\b", re.IGNORECASE),  # keyword 'exchange' (not stereotype)
]

_FORBIDDEN_SUBSTRINGS = [
    "->>",   # sequence invalid here per contract
    ".>",    # invalid arrow; PlantUML uses ..> not .>
]

# Lines that indicate inline declaration on edges (e.g. A --> component "X" as X)
_INLINE_DECL_RE = re.compile(
    r"(-->|<--|<-->|\.{2}>|<\.{2}|\*--|o--|<\|--|--\|>)\s*(component|interface|queue|node|cloud|artifact|database|participant|actor|object|class|rectangle|frame)\b",
    re.IGNORECASE,
)

# Class diagrams should not mix these element types (we want to avoid allowmixing)
_CLASS_MIXED_RE = re.compile(
    r"^\s*(component|participant|node|queue|cloud|artifact|rectangle|frame|actor|object)\b",
    re.IGNORECASE,
)

# Detect a class/interface/abstract declaration that also contains a relationship arrow —
# PlantUML does not permit combining body declaration and arrows on the same line.
# Example of INVALID syntax caught here:
#   class LocalFileWriter <|-- FileWriter {
#   abstract class A --|> Iface
_CLASS_DECL_ARROW_RE = re.compile(
    r"^\s*(abstract\s+class|class|interface|enum)\s+\S.*"
    r"(<\|--|--\|>|\*--|--\*|o--|--o|<\.\.|\.\.|>|<--|-+>)",
    re.IGNORECASE,
)

# Detect Java/Python-style inheritance keywords that are not valid PlantUML.
_JAVA_INHERIT_RE = re.compile(
    r"^\s*(abstract\s+class|class|interface)\s+\S+\s+(extends|implements|inherits)\s+",
    re.IGNORECASE,
)

def repair_class_diagram_lines(puml: str) -> str:
    """
    Post-process a class diagram to split lines that combine a class body
    declaration with a relationship arrow — a common LLM mistake that causes
    PlantUML syntax errors.

    Examples repaired:
      BEFORE: class LocalFileWriter <|-- FileWriter {
      AFTER:  class LocalFileWriter {
              }
              LocalFileWriter <|-- FileWriter

      BEFORE: abstract class A --|> IFoo
      AFTER:  abstract class A {
              }
              A --|> IFoo
    """
    if "@startuml" not in puml:
        return puml

    # Matches: (class-kw) (ClassName) (optional-stereotype) (arrow) (OtherName) (optional-{)
    _split_re = re.compile(
        r"^(\s*)(abstract\s+class|class|interface|enum)"   # group 1=indent, 2=kw
        r"\s+(\w+)"                                        # group 3=ClassName
        r"(\s+<<[^>]+>>)?"                                 # group 4=optional stereotype
        r"\s*(<\|--|--\|>|\*--|--\*|o--|--o|<\.\.|\.\.|>|<--|-+>)"  # group 5=arrow
        r"\s*(\w+)"                                        # group 6=OtherName
        r"\s*(\{?).*$",                                    # group 7=optional {
        re.IGNORECASE,
    )

    out_lines: List[str] = []
    deferred_arrows: List[str] = []
    skip_next_close_brace = False  # consume orphaned } after a split combined line

    for ln in puml.splitlines():
        # If the previous iteration split a "class Foo <|-- Bar {" line, the LLM's
        # matching closing "}" is now orphaned — skip exactly one such line.
        if skip_next_close_brace and ln.strip() == "}":
            skip_next_close_brace = False
            continue

        m = _split_re.match(ln)
        if m:
            indent, kw, cls_name, stereo, arrow, other_name, brace = m.groups()
            stereo = stereo or ""
            out_lines.append(f"{indent}{kw} {cls_name}{stereo} {{")
            out_lines.append(f"{indent}}}")
            skip_next_close_brace = bool(brace)
            deferred_arrows.append(f"{indent}{cls_name} {arrow} {other_name}")
        else:
            if ln.strip() == "@enduml" and deferred_arrows:
                out_lines.extend(deferred_arrows)
                deferred_arrows.clear()
            out_lines.append(ln)

    if deferred_arrows:
        out_lines.extend(deferred_arrows)

    return "\n".join(out_lines)


def extract_start_end_block(raw: str) -> str:
    m = re.search(r"@startuml.*?@enduml", raw, re.DOTALL)
    if m:
        return m.group(0).strip()
    if raw.strip():
        return f"@startuml\n{raw.strip()}\n@enduml"
    return ""

def normalize_startuml_name(puml: str, uml_name: str) -> str:
    """
    Ensure the first @startuml line uses a deterministic name that matches the output file stem.
    This prevents PlantUML from generating images with mismatched names and clobbering outputs.
    """
    if not puml:
        return puml
    lines = puml.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("@startuml"):
            lines[i] = f"@startuml {uml_name}"
            break
    return "\n".join(lines)

def validate_puml(diagram_type: str, puml: str) -> List[str]:
    errors: List[str] = []
    if not puml.strip():
        return ["empty output"]

    lines = puml.splitlines()
    for i, line in enumerate(lines, start=1):
        # ignore @startuml/@enduml and blank lines
        if not line.strip() or line.strip().startswith("@startuml") or line.strip().startswith("@enduml"):
            continue

        for rx in _FORBIDDEN_LINE_RE:
            if rx.search(line):
                errors.append(f"line {i}: forbidden keyword usage: {line.strip()}")
                break

        for s in _FORBIDDEN_SUBSTRINGS:
            if s in line:
                errors.append(f"line {i}: forbidden token '{s}': {line.strip()}")
                break

        if _INLINE_DECL_RE.search(line):
            errors.append(f"line {i}: inline element declaration on edge: {line.strip()}")

        if diagram_type == "class" and _CLASS_MIXED_RE.search(line):
            errors.append(f"line {i}: mixed non-class element in class diagram: {line.strip()}")

        if diagram_type == "class" and _CLASS_DECL_ARROW_RE.match(line):
            errors.append(
                f"line {i}: class declaration combined with relationship arrow "
                f"(not valid in PlantUML): {line.strip()}"
            )

        if diagram_type == "class" and _JAVA_INHERIT_RE.match(line):
            errors.append(
                f"line {i}: Java/Python-style inheritance keyword ('extends'/'implements') — "
                f"use PlantUML arrows instead: {line.strip()}"
            )

        if diagram_type == "sequence":
            # sequence must use -> not ->> (already caught), also discourage missing participant declarations,
            # but we keep this lightweight (we don't parse fully).
            pass

    return errors


# ===========================================================================
# PASS 1 — batched per-file extraction
# ===========================================================================

def pass1_extract_batch(
    files: List[Tuple[str, str]],
    llm: LLM,
    tokenizer,
    max_tokens: int,
) -> List[Dict]:
    """
    Build all extraction prompts at once, submit as a single batch to vLLM.
    vLLM fills all 4 GPUs across the batch — much faster than serial calls.
    """
    prompts = []
    for rel_path, source in files:
        user_msg = EXTRACTION_TMPL.format(
            rel_path=rel_path,
            source=source[:MAX_CHARS_PER_FILE],
        )
        prompts.append(format_prompt(tokenizer, EXTRACTION_SYSTEM, user_msg))

    print(f"      Submitting {len(prompts)} extraction prompts as one batch...")
    raw_outputs = vllm_generate_batch(llm, prompts, max_tokens=max_tokens)

    extractions = []
    for (rel_path, _), raw in zip(files, raw_outputs):
        result = _parse_json(raw)
        if result:
            n_cls = len(result.get("classes", []))
            n_fn  = len(result.get("functions", []))
            print(f"      ✓ {rel_path}  → {n_cls} classes, {n_fn} fns")
            extractions.append(result)
        else:
            print(f"      ✗ {rel_path}  → JSON parse failed, skipped")
    return extractions


# ===========================================================================
# PASS 2 — normalize to canonical Entity Registry
# ===========================================================================

def pass2_build_registry(
    extractions: List[Dict],
    repo_name: str,
    llm: LLM,
    tokenizer,
    max_tokens: int,
) -> Dict:
    user_msg = REGISTRY_TMPL.format(
        repo_name=repo_name,
        summaries_json=json.dumps(extractions, indent=2),
    )
    print(f"      Registry prompt: {len(user_msg):,} chars")
    prompt = format_prompt(tokenizer, REGISTRY_SYSTEM, user_msg)
    raw = vllm_generate_one(llm, prompt, max_tokens=max_tokens)
    registry = _parse_json(raw)
    if registry:
        return registry

    print("      [WARN] Registry JSON parse failed — building fallback registry")
    fb: Dict = {
        "repo_name": repo_name, "classes": [], "modules": [],
        "actors": [], "components": [], "states": [],
        "entry_points": [], "top_level_functions": [],
    }
    for ex in extractions:
        for cls in ex.get("classes", []):
            fb["classes"].append({
                "canonical_name": cls.get("name", "Unknown"),
                "defined_in": ex.get("file", ""),
                "bases": cls.get("bases", []),
                "key_methods": cls.get("methods", [])[:5],
                "relationships": [],
            })
        fb["modules"].append({
            "canonical_name": os.path.splitext(os.path.basename(ex.get("file","")))[0],
            "file": ex.get("file",""), "role": ex.get("summary",""),
        })
        fb["actors"].extend(ex.get("actors", []))
        fb["states"].extend(ex.get("states", []))
        if is_entry_point(ex.get("file", "")):
            fb["entry_points"].append(ex["file"])
    fb["actors"] = list(set(fb["actors"]))
    fb["states"] = list(set(fb["states"]))
    return fb


# ===========================================================================
# PASS 3 — diagram generation (also batched across all 8 diagram types)
# ===========================================================================

def select_chunks(
    diagram_type: str,
    files: List[Tuple[str, str]],
    registry: Dict,
) -> str:
    entry_set = set(registry.get("entry_points", []))
    if diagram_type in ENTRY_POINT_DIAGRAMS:
        ordered = (
            [(p, c) for p, c in files if p in entry_set or is_entry_point(p)]
          + [(p, c) for p, c in files if p not in entry_set and not is_entry_point(p)]
        )
    else:
        ordered = files

    chunks, total = [], 0
    for rel, src in ordered:
        block = f"===== FILE: {rel} =====\n{src[:MAX_CHARS_PER_FILE]}\n"
        if total + len(block) > GENERATION_CHUNK_BUDGET:
            rem = GENERATION_CHUNK_BUDGET - total
            if rem > 200:
                chunks.append(f"===== FILE: {rel} (truncated) =====\n{src[:rem]}\n")
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks)


def pass3_generate_all(
    diagram_types: List[Tuple[str, str]],
    repo_name: str,
    registry: Dict,
    files: List[Tuple[str, str]],
    rag_by_type: Dict[str, str],
    llm: LLM,
    tokenizer,
    max_tokens: int,
) -> Dict[str, str]:
    """
    Build all 8 diagram prompts, submit as one batch.
    If a diagram fails lightweight validation, retry that diagram once with a repair prompt.
    """
    dtypes  = [dt for dt, _ in diagram_types]
    prompts: List[str] = []
    user_msgs: Dict[str, str] = {}
    for dtype, desc in diagram_types:
        user_msg = GENERATION_TMPL.format(
            dtype_upper=dtype.upper(),
            repo_name=repo_name,
            description=desc,
            syntax_hint=DIAGRAM_SYNTAX_HINTS.get(dtype, ""),
            registry_json=json.dumps(registry, indent=2),
            code_chunks=select_chunks(dtype, files, registry),
            rag_examples=rag_by_type.get(dtype, ""),
        )
        prompts.append(format_prompt(tokenizer, GENERATION_SYSTEM, user_msg))

    print(f"      Submitting {len(prompts)} diagram prompts as one batch...")
    raw_outputs = vllm_generate_batch(llm, prompts, max_tokens=max_tokens)

    results: Dict[str, str] = {}
    raw_by_type: Dict[str, str] = {}
    for dtype, raw in zip(dtypes, raw_outputs):
        raw_by_type[dtype] = raw
        puml = extract_start_end_block(raw)
        # Auto-repair common class diagram structural mistakes before validation
        if dtype == "class":
            puml = repair_class_diagram_lines(puml)
        results[dtype] = puml

    # One retry per failing diagram type (only for the known failure patterns)
    retry_types: List[str] = []
    retry_prompts: List[str] = []

    for dtype in dtypes:
        puml = results.get(dtype, "")
        errs = validate_puml(dtype, puml)
        if errs:
            retry_types.append(dtype)
            user_msg = REPAIR_TMPL.format(
                dtype_upper=dtype.upper(),
                errors="\n".join(f"- {e}" for e in errs[:25]),
                previous=puml,
            )
            retry_prompts.append(format_prompt(tokenizer, GENERATION_SYSTEM, user_msg))

    if retry_prompts:
        print(f"      Retrying {len(retry_prompts)} diagram(s) after validation failures...")
        retry_raw = vllm_generate_batch(llm, retry_prompts, max_tokens=max_tokens)
        for dtype, raw in zip(retry_types, retry_raw):
            repaired = extract_start_end_block(raw)
            if dtype == "class":
                repaired = repair_class_diagram_lines(repaired)
            # If repair still fails, keep repaired anyway (often closer), but warn.
            errs = validate_puml(dtype, repaired)
            if errs:
                print(f"      [WARN] {dtype} still fails validation after retry (showing first issue): {errs[0]}")
            results[dtype] = repaired

    return results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked vLLM UML generator — model loaded once, all passes batched."
    )
    parser.add_argument("--input",  "-i", default=".")
    parser.add_argument("--output", "-o", default="uml_out")
    parser.add_argument("--faiss-index",
                        default=os.environ.get("RAG_FAISS_INDEX", "rag/faiss.index"))
    parser.add_argument("--faiss-meta",
                        default=os.environ.get("RAG_FAISS_META",  "rag/faiss_meta.json"))
    parser.add_argument("--model",
                        default=os.environ.get("VLLM_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct"))
    parser.add_argument("--tp",    type=int,
                        default=int(os.environ.get("VLLM_TP",   "4")))
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.environ.get("VLLM_MAX_LEN",    "32000")))
    parser.add_argument("--max-tokens",    type=int,
                        default=int(os.environ.get("VLLM_MAX_TOKENS",  "2048")))
    parser.add_argument("--extract-tokens", type=int,
                        default=int(os.environ.get("VLLM_EXTRACT_TOKENS", "1024")),
                        help="Max tokens for Pass 1 extraction (smaller = faster, default 1024).")
    parser.add_argument("--registry-tokens", type=int,
                        default=int(os.environ.get("VLLM_REGISTRY_TOKENS", "2048")),
                        help="Max tokens for Pass 2 registry (default 2048).")
    parser.add_argument("--temperature",   type=float,
                        default=float(os.environ.get("VLLM_TEMPERATURE", "0.0")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--rag-k",  type=int,
                        default=int(os.environ.get("RAG_TOP_K", "5")))
    parser.add_argument("--registry-file",
                        help="Load existing registry JSON — skips Pass 1 and Pass 2.")
    args = parser.parse_args()

    repo_root  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    repo_name  = os.path.basename(repo_root.rstrip(os.sep)) or "repo"

    print("=" * 70)
    print(f"  Chunked vLLM pipeline")
    print(f"  Repo    : {repo_name}  ({repo_root})")
    print(f"  Model   : {args.model}  (tp={args.tp})")
    print(f"  Output  : {output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # [1] Collect files
    # ------------------------------------------------------------------
    print("\n[1] Collecting .py files...")
    files = collect_files(repo_root)
    if not files:
        print("ERROR: no .py files found.", file=sys.stderr); sys.exit(1)
    print(f"    {len(files)} files  ({sum(len(c) for _,c in files):,} total chars)")

    # ------------------------------------------------------------------
    # [2] Load FAISS + embeddings (CPU — no GPU contention with vLLM)
    # ------------------------------------------------------------------
    print("\n[2] Loading FAISS RAG index and embedding model...")
    index, docs, embed_model_name = load_faiss_and_meta(args.faiss_index, args.faiss_meta)
    embed_model = load_embed_model(embed_model_name)
    print(f"    {len(docs)} docs, embed model: {embed_model_name}")

    print("\n[3] Fetching RAG examples per diagram type...")
    rag_by_type: Dict[str, str] = {}
    for dtype, _ in DIAGRAM_TYPES:
        try:
            rag_by_type[dtype] = get_rag_examples(dtype, index, docs, embed_model, args.rag_k)
            print(f"    ✓ {dtype}")
        except Exception as e:
            print(f"    ✗ {dtype}: {e}"); rag_by_type[dtype] = ""

    # ------------------------------------------------------------------
    # [4] Load vLLM model ONCE  ← the whole point
    # ------------------------------------------------------------------
    print(f"\n[4] Loading vLLM model (once — used for all passes)...")
    print(f"    {args.model}  |  tp={args.tp}  |  max_len={args.max_model_len}")
    llm = load_model(
        model=args.model,
        tp=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    # Grab tokenizer for chat template formatting
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    print("    ✓ Model loaded and ready")

    # ------------------------------------------------------------------
    # [5] Pass 1 + 2 or load registry
    # ------------------------------------------------------------------
    if args.registry_file:
        print(f"\n[5] Loading existing registry from {args.registry_file}...")
        registry = json.load(open(args.registry_file))
        print("    Skipping Pass 1 and Pass 2.")
    else:
        # Pass 1 — batched extraction
        print(f"\n[5] Pass 1 — batched extraction ({len(files)} files in one call)...")
        extractions = pass1_extract_batch(files, llm, tokenizer, args.extract_tokens)
        if not extractions:
            print("ERROR: extraction produced nothing.", file=sys.stderr); sys.exit(1)

        # Pass 2 — registry normalization
        print(f"\n[6] Pass 2 — normalizing {len(extractions)} extractions → Entity Registry...")
        registry = pass2_build_registry(extractions, repo_name, llm, tokenizer, args.registry_tokens)
        print(f"    {len(registry.get('classes',[]))} classes  "
              f"{len(registry.get('modules',[]))} modules  "
              f"{len(registry.get('components',[]))} components")

    # Always persist registry
    reg_path = os.path.join(output_dir, f"{repo_name}_entity_registry.json")
    json.dump(registry, open(reg_path, "w"), indent=2)
    print(f"\n    Registry saved → {reg_path}")
    print(f"    TIP: --registry-file {reg_path} skips extraction on re-runs.")

    # ------------------------------------------------------------------
    # [7] Pass 3 — all 8 diagrams in one batch
    # ------------------------------------------------------------------
    print(f"\n[7] Pass 3 — generating all {len(DIAGRAM_TYPES)} diagrams in one batch...")
    diagrams = pass3_generate_all(
        diagram_types=DIAGRAM_TYPES,
        repo_name=repo_name,
        registry=registry,
        files=files,
        rag_by_type=rag_by_type,
        llm=llm,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )

    print("\n[8] Writing diagram files...")
    written = 0
    for dtype, _ in DIAGRAM_TYPES:
        puml = diagrams.get(dtype, "").strip()
        if not puml:
            print(f"    ✗ {dtype} — empty output, skipped"); continue
        uml_name = f"{repo_name}_{dtype}"
        puml = normalize_startuml_name(puml, uml_name).strip()
        out_path = os.path.join(output_dir, f"{uml_name}.puml")
        open(out_path, "w").write(puml)
        print(f"    ✓ {out_path}  ({len(puml):,} chars)")
        written += 1

    print(f"\n{'='*70}\n  Done!  {written}/{len(DIAGRAM_TYPES)} diagrams\n{'='*70}")


if __name__ == "__main__":
    main()
