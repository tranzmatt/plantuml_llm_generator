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
# VLLM_WORKER_MULTIPROC_METHOD and VLLM_USE_MODELSCOPE are still valid.
# VLLM_USE_V1, VLLM_NO_CUDA_GRAPH, VLLM_ENFORCE_EAGER, VLLM_MAX_NUM_SEQS,
# VLLM_GPU_MEMORY_UTILIZATION were removed/renamed in newer vLLM releases —
# their equivalents are now passed directly as LLM() constructor arguments.
os.environ.setdefault("VLLM_USE_MODELSCOPE",           "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD",  "spawn")

import json, re, sys, argparse
from typing import Dict, List, Optional, Tuple

import torch
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

# Enable TF32 for float32 matmuls on Ampere/Ada/Hopper GPUs (A100, RTX 6000 Ada, H100…).
# Gives a meaningful throughput boost with negligible precision loss.
torch.set_float32_matmul_precision("high")

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
# Extra token headroom subtracted from every input budget check.
# _count_tokens() can disagree with vLLM's internal tokenizer by a few tokens
# (different BOS/EOS handling, special tokens, etc.).  A margin of 64 ensures
# our pre-check is always stricter than vLLM's hard limit.
CONTEXT_SAFETY_MARGIN    = 64

# ---------------------------------------------------------------------------
# Per-model context defaults
# Matched by substring (case-insensitive) against the --model argument.
# --max-model-len overrides these when explicitly provided.
# ---------------------------------------------------------------------------
MODEL_DEFAULTS: List[Tuple[str, int]] = [
    # Llama 4 Scout/Maverick (MoE — 10M native, 256k practical cap on 4×A100 80GB)
    # Use --max-model-len to push higher if your hardware allows
    ("llama-4",          256_000),
    # Llama 3.1 / 3.3 (dense 128k native)
    ("llama-3",          128_000),
    # Mistral Large 3 675B (NVFP4 quantized — 256k native)
    ("mistral-large-3",  262_144),
    # Mistral Large 2411 123B (dense — 128k native but KV cache heavy; 48k safe on 4×A100 80GB)
    ("mistral-large",     48_000),
    # Devstral 2 123B (code-focused dense — 256k native)
    ("devstral",         262_144),
    # Mixtral 8x22B (MoE — 64k native)
    ("mixtral",           64_000),
    # Mistral Small / Nemo / other Mistral variants
    ("mistral",           32_000),
    # gpt-oss-120b (128k native)
    ("gpt-oss-120b",     128_000),
]
DEFAULT_MAX_MODEL_LEN = 32_000   # safe fallback for unknown models


def resolve_max_model_len(model_name: str, override: Optional[int]) -> int:
    """Return effective max_model_len: explicit override > model table > fallback."""
    if override is not None:
        return override
    lower = model_name.lower()
    for pattern, length in MODEL_DEFAULTS:
        if pattern in lower:
            return length
    return DEFAULT_MAX_MODEL_LEN

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
        "NEVER use (*) --> or --> (*) legacy syntax.\n"
        "BLOCK MATCHING (every opener needs exactly one closer):\n"
        "  if → endif\n"
        "  fork → end fork  (use 'fork again' for parallel branches, then 'end fork')\n"
        "  while → endwhile\n"
        "  repeat → repeat while\n"
        "NEVER write endwhile unless there is a matching while above it.\n"
        "NEVER write endif unless there is a matching if above it.\n"
        "A fork block ends with 'end fork', not 'endwhile' or 'end'.\n"
        "\n"
        "repeat/repeat while:\n"
        "  repeat\n"
        "    :action;\n"
        "  repeat while (condition?) is (yes) not (no)\n"
        "\n"
        "Swimlanes (partition):\n"
        "  partition \"Service A\" {\n"
        "    :action;\n"
        "  }\n"
        "\n"
        "Notes:\n"
        "  note left: text\n"
        "  note right: text\n"
        "\n"
        "Hard constraints:\n"
        "- NEVER use (*) --> or --> (*) legacy transition syntax.\n"
        "- Every action line starts with : and ends with ; — never omit either.\n"
        "- detach and kill are valid terminators instead of stop."
    ),
    "state": (
        "ALLOWED elements: state, note, skinparam, hide/show.\n"
        "NEVER use: participant, class, component, node, actor, object.\n"
        "\n"
        "Transition syntax:\n"
        "  Correct:  StateName --> OtherState : label\n"
        "  WRONG:    -->|label|  (that is Mermaid syntax — will fail in PlantUML)\n"
        "\n"
        "State declaration:\n"
        "  Named state:    state \"Display Name\" as alias\n"
        "  Composite:      state \"Display Name\" as alias {\n"
        "                    [*] --> SubState\n"
        "                  }\n"
        "  Inline (simple): [*] --> Idle  (Idle is created implicitly — no quotes needed)\n"
        "\n"
        "CRITICAL — transition targets must be plain identifiers or declared aliases. NEVER quoted strings:\n"
        "  WRONG:   [*] --> \"Initializing\"\n"
        "  CORRECT: [*] --> Initializing           (plain identifier, implicitly declared)\n"
        "  CORRECT: state \"Initializing\" as init  then  [*] --> init\n"
        "\n"
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  state \"Idle\" as \"Idle\"\n"
        "  CORRECT: state \"Idle\" as idle\n"
        "\n"
        "Entry/exit points use [*]:\n"
        "  [*] --> FirstState   (entry)\n"
        "  LastState --> [*]    (exit)\n"
        "\n"
        "Concurrent (orthogonal) states — use -- as separator inside a composite state:\n"
        "  state \"Processing\" as processing {\n"
        "    state \"Receiving\" as receiving\n"
        "    --\n"
        "    state \"Sending\" as sending\n"
        "  }\n"
        "\n"
        "Pseudostate stereotypes (optional, for clarity):\n"
        "  state choice1 <<choice>>\n"
        "  state fork1   <<fork>>\n"
        "  state join1   <<join>>\n"
        "  state end1    <<end>>\n"
        "\n"
        "Notes:\n"
        "  note on link : text   (annotates the preceding transition)\n"
        "  note left of StateName : text\n"
        "  note right of StateName : text\n"
        "\n"
        "Hard constraints:\n"
        "- Transition endpoints are identifiers only — NEVER quoted strings.\n"
        "- Declare composite state blocks before referencing their children in transitions.\n"
        "- Every { must have exactly one matching }. Close state blocks before writing transitions."
    ),
    "usecase": (
        "ALLOWED elements: actor, usecase, rectangle, package, note.\n"
        "NEVER use: participant, node, cloud, component, queue, artifact.\n"
        'Rectangle names with spaces MUST be quoted: rectangle "My Service" { }\n'
        'Actor names with spaces must be quoted: actor "End User" as u\n'
        'Use case names with spaces must be quoted: usecase "Do Something" as UC1\n'
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  usecase \"Do Something\" as \"Do Something\"\n"
        "  CORRECT: usecase \"Do Something\" as do_something\n"
        "If a rectangle or package is referenced in edges, it MUST have an alias:\n"
        "  WRONG:  rectangle \"Imagery Pipeline\" { }  ...then later...  foo --> imagery_pipeline\n"
        "  CORRECT: rectangle \"Imagery Pipeline\" as imagery_pipeline { }\n"
        "Never reference an alias in an edge that was not explicitly declared with 'as'.\n"
        "\n"
        "Include/extend relationships (use dotted arrow with stereotype):\n"
        "  UC1 ..> UC2 : <<include>>\n"
        "  UC1 ..> UC3 : <<extend>>\n"
        "  Actor --|> GeneralActor    (actor generalisation)\n"
        "\n"
        "BRACE DISCIPLINE: every { must have exactly one matching }.\n"
        "Write ALL declarations and close ALL { } blocks BEFORE writing any edges."
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
        "Modifiers: {abstract}, {static} — place after method/field name:\n"
        "  + run() {abstract}\n"
        "  + instance : Foo {static}\n"
        "Suppress empty bodies: hide empty members\n"
        "Generics: class Foo<T>\n"
        "Hard constraints:\n"
        "- Do not mix other diagram element families here (no component/queue/node/participant/object).\n"
        "- Relationship arrows go on their OWN lines, never on a class declaration line.\n"
        "- Every { must have exactly one matching }. Never emit an orphaned closing }.\n"
        "- Every class/interface/enum declaration that opens a { MUST close it with } before ANY relationship arrows.\n"
        "  WRONG:  class ImageryRouter {\n"
        "  (arrows here before } closes the body)\n"
        "  CORRECT: class ImageryRouter {\n"
        "           }\n"
        "           Child <|-- Parent"
    ),
    "sequence": (
        "ALLOWED elements: participant, actor, boundary, control, entity, database, collections.\n"
        "NEVER use: node, component, queue, cloud, artifact, rectangle, object, class.\n"
        "Declare all participants at the top before any arrows.\n"
        'participant "Name" as alias\n'
        "Sync call: A -> B : message\n"
        "Return:    B --> A : response\n"
        "activate B / deactivate B\n"
        "Groups: alt/else/end, loop, opt\n"
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  participant \"My Server\" as \"My Server\"\n"
        "  CORRECT: participant \"My Server\" as my_server\n"
        "File paths: replace / with _ in display names:\n"
        "  WRONG:  participant \"ui/server.py\"\n"
        "  CORRECT: participant \"ui_server.py\" as server\n"
        "Notes:\n"
        "  note left of Alice : text\n"
        "  note right of Bob : text\n"
        "  note over Alice, Bob : text   (spans multiple participants)\n"
        "  note left: text               (shorthand for last participant)\n"
        "\n"
        "Grouping:\n"
        "  box \"External Systems\" #LightBlue\n"
        "    participant X\n"
        "  end box\n"
        "\n"
        "Dividers and delays:\n"
        "  == Phase 1 ==           (section separator)\n"
        "  ...                     (delay)\n"
        "  ... 5 minutes later ... (labelled delay)\n"
        "\n"
        "Hard constraints:\n"
        "- Every participant referenced in a message must be declared before messages.\n"
        "- If a participant name contains spaces, quote it and/or use an alias; message endpoints should be aliases.\n"
        "- Message arrows must have a label or omit the colon: WRONG  A -> B :   CORRECT  A -> B : call  or  A -> B"
    ),
    "component": (
        "ALLOWED elements: component, interface, queue, database, cloud, artifact, node, package, rectangle.\n"
        "NEVER use: participant, boundary, control, entity, collections.\n"
        'Components/packages/frames with spaces must be quoted: component "My Component" as my_component\n'
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  component \"MQTT Listener\" as \"MQTT Listener\"\n"
        "  CORRECT: component \"MQTT Listener\" as mqtt_listener\n"
        "Every alias must be unique within the diagram.\n"
        "Edge labels must have text, or omit the colon entirely:\n"
        "  WRONG:  A --> B :\n"
        "  CORRECT: A --> B : publishes  OR  A --> B\n"
        "Declare ALL elements before edges.\n"
        "Edges must use --> or ..> only. Never use .>.\n"
        "Never create new keywords like exchange or topic.\n"
        "Represent messaging exchanges as stereotypes:\n"
        'queue "celebrity_names" <<exchange>>\n'
        "Package/frame nesting:\n"
        "  package \"Group\" {\n"
        "    component \"A\" as a\n"
        "    component \"B\" as b\n"
        "  }\n"
        "Interface notation (both forms valid):\n"
        "  interface \"IFoo\" as ifoo\n"
        "  () \"IFoo\" as ifoo\n"
        "\n"
        "Hard constraints:\n"
        "- Do NOT define/alias an element inline inside a link line.\n"
        "- One edge per line. Do NOT use comma-separated targets.\n"
        "- Do not mix other diagram element families (e.g., class/object/participant).\n"
        "- Close ALL package/frame/node { } blocks before writing edges."
    ),
    "deployment": (
        "ALLOWED elements: node, component, artifact, database, cloud, queue, package.\n"
        "NEVER use: participant, boundary, control, entity, collections, actor.\n"
        'Nodes/clouds/frames with spaces must be quoted: node "My Node" { }\n'
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  node \"MongoDB Cluster\" as \"Mongo Cluster\"\n"
        "  CORRECT: node \"MongoDB Cluster\" as Mongo_Cluster\n"
        "Write ALL node/component declarations and close ALL { } blocks BEFORE writing edges.\n"
        "Additional container types: frame, stack, rectangle (all support { } nesting).\n"
        "  frame \"Kubernetes Cluster\" {\n"
        "    node \"Pod A\" as pod_a {\n"
        "      component \"App\" as app\n"
        "    }\n"
        "  }\n"
        "\n"
        "Hard constraints:\n"
        "- There is no `exchange` keyword. Represent exchanges as artifact or component with label.\n"
        "- Do NOT define/alias an element inline inside a link line. Declare first, then connect.\n"
        "- One edge per line. Do NOT use comma-separated targets.\n"
        "- ALL nested { } blocks must be fully closed before any edge lines."
    ),
    "object": (
        "ALLOWED elements: object, note.\n"
        "NEVER use: class, component, participant, node, queue, cloud.\n"
        'Object instances: object "instanceName : ClassName" as alias\n'
        "Field values: alias : field = value\n"
        "DECLARATION ORDER (critical — PlantUML errors if violated):\n"
        "  1. ALL  object \"Name\" as alias  declarations\n"
        "  2. ALL  alias : field = value  field assignments\n"
        "  3. ALL edges\n"
        "Never reference an alias in an edge before its object declaration.\n"
        "Aliases must be plain identifiers — NEVER quote an alias:\n"
        "  WRONG:  object \"config:ConfigParser\" as \"config\"\n"
        "  CORRECT: object \"config:ConfigParser\" as config\n"
        "Field value syntax:\n"
        "  alias : field = value           (bare value)\n"
        "  alias : name = \"John Doe\"      (quoted string value)\n"
        "  alias : count = 42\n"
        "\n"
        "Map objects (key/value display):\n"
        "  map \"Config\" as cfg {\n"
        "    host => localhost\n"
        "    port => 8080\n"
        "  }\n"
        "\n"
        "Links: o1 --> o2 : label  or  o1 *-- o2"
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

def preflight_vram_check(model: str, tp: int, gpu_memory_utilization: float) -> None:
    """
    Estimate whether the model's weights will fit in available VRAM before
    attempting to load.  Uses HuggingFace safetensors metadata (no weight
    download) + torch device queries.  Prints a warning but does NOT abort —
    quantized or sharded models may still work even when the estimate looks
    tight.
    """
    try:
        from huggingface_hub import model_info as hf_model_info
    except ImportError:
        print("[PREFLIGHT] huggingface_hub not available, skipping VRAM check.")
        return

    # --- query physical VRAM across the TP GPUs we'll actually use ----------
    n_gpus = torch.cuda.device_count()
    use_gpus = min(tp, n_gpus)
    try:
        gpu_vram_gb = [
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(use_gpus)
        ]
        total_vram_gb = sum(gpu_vram_gb)
        gpu_desc = f"{use_gpus}×{gpu_vram_gb[0]:.0f} GiB" if gpu_vram_gb else "unknown"
    except Exception:
        print("[PREFLIGHT] Could not query GPU VRAM, skipping check.")
        return

    # --- fetch param count from HF safetensors metadata (no weights) --------
    try:
        info = hf_model_info(model)
        st = getattr(info, "safetensors", None)
        total_params = getattr(st, "total", None) if st else None
    except Exception as e:
        print(f"[PREFLIGHT] Could not fetch model info from HuggingFace ({e}), skipping check.")
        return

    if total_params is None:
        print("[PREFLIGHT] No safetensors metadata found for this model, skipping check.")
        return

    # --- estimate weight footprint ------------------------------------------
    # Assume bfloat16 (2 bytes) as the conservative default.
    # FP8 / NVFP4 quantised models will be smaller; the check is intentionally
    # pessimistic so a ✓ here means "definitely fits", not "might fit".
    bytes_per_param = 2  # bfloat16
    weights_gb = (total_params * bytes_per_param) / (1024 ** 3)

    # vLLM allocates gpu_memory_utilization × total_vram for the engine;
    # weights must fit within that pool.
    usable_vram_gb = total_vram_gb * gpu_memory_utilization
    kv_headroom_gb  = usable_vram_gb - weights_gb

    print(f"[PREFLIGHT] {total_params / 1e9:.1f}B params  |  "
          f"~{weights_gb:.1f} GiB weights (bf16)  |  "
          f"{gpu_desc} = {total_vram_gb:.1f} GiB total  |  "
          f"{gpu_memory_utilization:.0%} usable = {usable_vram_gb:.1f} GiB")

    if kv_headroom_gb < 1.0:
        print(f"[PREFLIGHT] ⚠ WARNING: only ~{kv_headroom_gb:.1f} GiB left for KV cache after weights. "
              f"Model will likely OOM or have near-zero context. "
              f"Consider a smaller model or fewer GPUs' worth of context.")
    elif kv_headroom_gb < 8.0:
        print(f"[PREFLIGHT] ⚠ TIGHT: ~{kv_headroom_gb:.1f} GiB for KV cache — "
              f"max usable context will be limited. "
              f"If this is a quantised model (FP8/NVFP4) actual headroom will be larger.")
    else:
        print(f"[PREFLIGHT] ✓ ~{kv_headroom_gb:.1f} GiB estimated KV cache headroom "
              f"(assuming bf16; quantised models have more).")


def load_model(model: str, tp: int, max_model_len: int,
               gpu_memory_utilization: float, enforce_eager: bool,
               tokenizer_mode: str = "auto",
               config_format: str = "auto",
               load_format: str = "auto") -> LLM:
    """Load the vLLM model once. Pass the returned object to all generate calls."""
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        tokenizer_mode=tokenizer_mode,
        config_format=config_format,
        load_format=load_format,
    )


def make_sampling_params(max_tokens: int, temperature: float = 0.0) -> SamplingParams:
    if max_tokens < 1:
        raise ValueError(
            f"make_sampling_params called with max_tokens={max_tokens}. "
            "Pass --max-tokens (default 2048) or --registry-tokens / --extract-tokens "
            "with a value ≥ 1."
        )
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
    import warnings
    try:
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_msg},
        ]
        # tokenize=False is intentional — we need a string, not token ids.
        # Wrap in catch_warnings to suppress the MistralCommonTokenizer advisory.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def _count_tokens(tokenizer, text: str) -> int:
    """Return the token count for *text* using whatever interface the tokenizer exposes."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
        try:
            return len(tokenizer(text)["input_ids"])
        except Exception:
            pass
    # Last-resort estimate: ~3.5 chars / token (conservative for code)
    return max(1, int(len(text) / 3.5))


def _compress_extractions(extractions: List[Dict], level: int) -> List[Dict]:
    """
    Return a progressively slimmer copy of the extractions list so that the
    Pass 2 registry prompt fits within the model's context window.

    level 0 — original (full fidelity)
    level 1 — drop per-class 'attributes'
    level 2 — also drop 'imports' per file
    level 3 — also truncate 'methods' to first 5 per class
    level 4 — also drop per-class 'relationships'
    level 5 — also drop per-file 'relationships'
    level 6 — keep only file, classes (name+bases only), functions[:10], summary
    level 7 — keep only file, class names (strings), summary  (minimum viable)
    """
    import copy
    result = []
    for ex in extractions:
        e = copy.deepcopy(ex)
        if level >= 1:
            for cls in e.get("classes", []):
                cls.pop("attributes", None)
        if level >= 2:
            e.pop("imports", None)
        if level >= 3:
            for cls in e.get("classes", []):
                cls["methods"] = cls.get("methods", [])[:5]
        if level >= 4:
            for cls in e.get("classes", []):
                cls.pop("relationships", None)
        if level >= 5:
            e.pop("relationships", None)
        if level >= 6:
            slim_classes = [
                {"name": c.get("name", ""), "bases": c.get("bases", [])}
                for c in e.get("classes", [])
            ]
            e = {
                "file":      e.get("file", ""),
                "classes":   slim_classes,
                "functions": e.get("functions", [])[:10],
                "summary":   e.get("summary", ""),
            }
        if level >= 7:
            e = {
                "file":    e.get("file", ""),
                "classes": [c.get("name", "") for c in e.get("classes", [])],
                "summary": e.get("summary", ""),
            }
        result.append(e)
    return result


def _compress_registry(registry: Dict, level: int) -> Dict:
    """
    Return a progressively slimmer copy of the entity registry for use in
    Pass 3 generation prompts, so the prompt fits within the model's context.

    level 0 — original (full fidelity)
    level 1 — drop per-class 'relationships'
    level 2 — also truncate 'key_methods' to first 5 per class
    level 3 — also drop 'bases' per class; slim modules to name+file only
    level 4 — classes: name+file only; drop top_level_functions
    level 5 — classes: names only (list of strings); modules: names only
    level 6 — keep only class names, component names, entry_points, actors
    level 7 — class names only (minimum viable for the LLM to use canonical names)
    """
    import copy
    r = copy.deepcopy(registry)

    if level >= 1:
        for cls in r.get("classes", []):
            cls.pop("relationships", None)

    if level >= 2:
        for cls in r.get("classes", []):
            cls["key_methods"] = cls.get("key_methods", [])[:5]

    if level >= 3:
        for cls in r.get("classes", []):
            cls.pop("bases", None)
        r["modules"] = [
            {"canonical_name": m.get("canonical_name", ""), "file": m.get("file", "")}
            for m in r.get("modules", [])
        ]

    if level >= 4:
        r["classes"] = [
            {"canonical_name": c.get("canonical_name", ""), "defined_in": c.get("defined_in", "")}
            for c in r.get("classes", [])
        ]
        r.pop("top_level_functions", None)

    if level >= 5:
        r["classes"]  = [c.get("canonical_name", "") for c in r.get("classes", [])]
        r["modules"]  = [m.get("canonical_name", "") if isinstance(m, dict)
                         else m for m in r.get("modules", [])]

    if level >= 6:
        r = {
            "repo_name":    r.get("repo_name", ""),
            "classes":      r.get("classes", []),
            "components":   r.get("components", []),
            "actors":       r.get("actors", []),
            "entry_points": r.get("entry_points", []),
            "states":       r.get("states", []),
        }

    if level >= 7:
        r = {
            "repo_name": r.get("repo_name", ""),
            "classes":   r.get("classes", []),
        }

    return r


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
    "- State: transitions use 'StateName --> OtherState : label'. NEVER -->|label| (Mermaid).\n"
    "- State: transition targets MUST be plain identifiers. NEVER quoted strings: WRONG [*] --> \"Idle\"  CORRECT [*] --> Idle\n"
    "- Use case/component/deployment: ALL multi-word names must be double-quoted.\n"
    "- Class: <|-- inherit, *-- compose, o-- aggregate, --> associate.\n"
    "- Sequence: declare all participants before first arrow.\n\n"

    "ALIAS RULES (critical):\n"
    "- Aliases must be plain identifiers: letters, digits, underscores ONLY.\n"
    "- NEVER quote an alias: WRONG: as \"Mongo Cluster\"  CORRECT: as Mongo_Cluster\n"
    "- Every alias must be unique within the diagram — never assign the same alias twice.\n"
    "- After declaring  component \"Foo\" as foo  always refer to the element as  foo  not  \"Foo\".\n\n"

    "DECLARATION ORDER (critical):\n"
    "- Declare ALL elements first, then write ALL edges. Never interleave.\n"
    "- An element used in an edge must have an explicit declaration above that edge.\n"
    "- For object diagrams: declare all objects, then field assignments, then edges.\n\n"

    "BRACE DISCIPLINE:\n"
    "- Every  {  must have exactly one matching  }.\n"
    "- Never emit a bare  }  that has no matching opener.\n"
    "- Group blocks (node/package/rectangle) must be fully closed before writing edges.\n\n"

    "EDGE RULES:\n"
    "- Allowed arrows: -->, <-->, ..>, <.., ..|>, <|--, *--, o--.\n"
    "- NEVER use .> or ->> or =>.\n"
    "- Never declare new elements inline on an edge.\n"
    "- Edge labels MUST have text: WRONG: A --> B :   CORRECT: A --> B : calls  or just  A --> B\n\n"

    "ELEMENT TYPES PER DIAGRAM (do not mix):\n"
    "- sequence:   participant, actor, boundary, control, entity, database, collections\n"
    "- component:  component, interface, queue, database, cloud, artifact, node, package, rectangle\n"
    "- deployment: node, component, artifact, database, cloud, queue, package\n"
    "- class:      class, abstract class, interface, enum, package, namespace\n"
    "- object:     object, note\n"
    "- usecase:    actor, usecase, rectangle, package, note\n"
    "- NEVER use 'participant' outside sequence diagrams.\n"
    "- NEVER use 'node' or 'cloud' inside class or sequence diagrams.\n\n"

    "ALLOWED KEYWORDS (global):\n"
    "- Forbidden keywords: exchange, topic, fanout, pubsub, service, lambda.\n"
    "- Represent exchange/topic concepts as:  queue \"X\" <<exchange>>\n\n"

    "NAME SANITISATION:\n"
    "- File paths and names containing  /  must be replaced with  _  before use.\n"
    "- CORRECT: participant \"ui_server.py\" as server\n"
    "- WRONG:   participant \"ui/server.py\" as server\n\n"

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


def repair_duplicate_aliases(puml: str) -> str:
    """
    Detect and fix duplicate `as <alias>` declarations in any PlantUML diagram.

    The LLM commonly produces collisions when multiple element names share the
    same initials (e.g. MqttPublisher → mp, MessageProcessor → mp).

    Strategy
    --------
    1. Parse every declaration line that ends with `as <alias>`.
    2. On the first occurrence of an alias, keep it unchanged.
    3. On every subsequent collision, generate a new unique alias by appending
       an incrementing numeric suffix (mp2, mp3, …) and rewrite ALL references
       to the old alias in the remainder of the diagram.
    """
    if "@startuml" not in puml:
        return puml

    _AS_RE = re.compile(
        r"^(\s*(?:object|class|abstract\s+class|interface|enum|participant|actor"
        r"|component|node|database|cloud|queue|artifact|rectangle|frame|package"
        r"|usecase|state|boundary|control|entity|collections|(?:create\s+)?[\w]+)"
        r"\s+.+?\bas\s+)(\w+)(\s*(?:#\S+)?\s*)$",
        re.IGNORECASE,
    )

    lines = puml.splitlines()
    seen: Dict[str, int] = {}       # alias → count of times seen so far
    renames: Dict[str, str] = {}    # old_alias → new_alias for in-flight rewrites

    out: List[str] = []
    for ln in lines:
        m = _AS_RE.match(ln)
        if m:
            prefix, alias, suffix = m.group(1), m.group(2), m.group(3)
            if alias not in seen:
                seen[alias] = 1
                out.append(ln)
            else:
                seen[alias] += 1
                new_alias = f"{alias}{seen[alias]}"
                while new_alias in seen:
                    seen[alias] += 1
                    new_alias = f"{alias}{seen[alias]}"
                seen[new_alias] = 1
                renames[alias] = new_alias
                out.append(f"{prefix}{new_alias}{suffix}")
                print(f"      [REPAIR] Duplicate alias '{alias}' → renamed to '{new_alias}'")
        else:
            if renames:
                for old, new in renames.items():
                    ln = re.sub(rf"\b{re.escape(old)}\b", new, ln)
            out.append(ln)

    return "\n".join(out)


def repair_unquoted_multiword_edges(puml: str) -> str:
    """
    Fix edge lines where a multi-word element name is used bare (without quotes
    or an alias), which causes a PlantUML syntax error.

    Example of the bug:
        node "Local Machine" {        ← declared with quotes, no alias
            ...
        }
        Local Machine --> MQTT        ← ERROR: unquoted multi-word source

    Two-pass strategy
    -----------------
    Pass 1: Collect every element name that was declared with quotes but has NO
            `as <alias>` clause.  Also collect every declared alias so we know
            what short names already exist.

    Pass 2: On edge lines (lines containing -->, <--, <-->, ..>, <..) scan the
            LHS and RHS tokens.  If a contiguous run of bare words (no quotes,
            not an alias) matches a known multi-word name, wrap it in quotes.
            This makes `Local Machine --> MQTT` become `"Local Machine" --> MQTT`.
    """
    if "@startuml" not in puml:
        return puml

    # Matches any quoted-name declaration, optionally with `as alias`
    # Captures: (quoted_name, alias_or_empty)
    _DECL_RE = re.compile(
        r'^\s*(?:node|component|artifact|database|cloud|queue|rectangle|frame'
        r'|package|actor|participant|object|class|interface|usecase|state'
        r'|boundary|control|entity|storage|agent|card|collections)\s+'
        r'"([^"]+)"'                  # group 1: the quoted display name
        r'(?:\s+as\s+(\w+))?',       # group 2: optional alias
        re.IGNORECASE,
    )

    # Arrow pattern for edge lines
    _EDGE_RE = re.compile(
        r'(-->|<--|<-->|\.{2}>|<\.{2}|\*--|o--|<\|--|--\|>|-+>)',
        re.IGNORECASE,
    )

    lines = puml.splitlines()

    # Pass 1: build set of multi-word names that have NO alias
    unaliased_multiword: set = set()
    all_aliases: set = set()
    for ln in lines:
        m = _DECL_RE.match(ln)
        if m:
            name, alias = m.group(1), m.group(2)
            if alias:
                all_aliases.add(alias)
            elif " " in name:
                unaliased_multiword.add(name)

    if not unaliased_multiword:
        return puml  # nothing to fix

    # Sort longest-first so "Local Machine Alpha" is tried before "Local Machine"
    candidates = sorted(unaliased_multiword, key=len, reverse=True)

    out: List[str] = []
    for ln in lines:
        if _EDGE_RE.search(ln) and not ln.strip().startswith("'"):
            original = ln
            for name in candidates:
                # Only replace bare (unquoted) occurrences
                # Use a word-boundary-aware pattern that won't touch already-quoted names
                pat = r'(?<!")\b' + re.escape(name) + r'\b(?!")'
                if re.search(pat, ln):
                    ln = re.sub(pat, f'"{name}"', ln)
            if ln != original:
                print(f"      [REPAIR] Quoted multi-word name on edge: {original.strip()!r} → {ln.strip()!r}")
        out.append(ln)

    return "\n".join(out)


def repair_truncated_activity(puml: str) -> str:
    """
    Fix activity diagrams that were truncated mid-generation (hit max_tokens).

    Two symptoms:
    1. Last action line is missing its closing semicolon:
           :Initialize Dummy        ← should be  :Initialize Dummy;
    2. Diagram has no `stop` / `end` before @enduml — PlantUML requires one.

    We fix both unconditionally for activity diagrams.
    """
    if "@startuml" not in puml:
        return puml

    lines = puml.splitlines()

    # Find the last non-blank, non-@enduml line
    body_lines = []
    enduml_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "@enduml":
            enduml_idx = i
        else:
            body_lines.append((i, ln))

    if not body_lines:
        return puml

    last_idx, last_ln = body_lines[-1]
    stripped = last_ln.strip()

    # Fix dangling action (starts with : but no closing ;)
    if stripped.startswith(":") and not stripped.endswith(";"):
        lines[last_idx] = last_ln.rstrip() + ";"
        stripped = lines[last_idx].strip()
        print(f"      [REPAIR] Closed dangling activity action: {stripped!r}")

    # Ensure a stop/end exists before @enduml
    has_terminator = any(
        ln.strip() in ("stop", "end", "detach", "kill")
        for ln in lines
        if ln.strip() not in ("@startuml", "@enduml", "")
    )
    if not has_terminator:
        insert_at = enduml_idx if enduml_idx is not None else len(lines)
        lines.insert(insert_at, "stop")
        print("      [REPAIR] Inserted missing 'stop' before @enduml")

    return "\n".join(lines)


def repair_slash_names(puml: str) -> str:
    """
    Quote any bare element names that contain '/' (e.g. HTTP/HTTPS, TCP/IP).

    PlantUML misparses 'HTTP/HTTPS' as a comment or sequence token.
    This fix applies to declaration lines AND edge lines.

    Declaration:  HTTP/HTTPS                     →  "HTTP/HTTPS"
    Declaration:  component HTTP/HTTPS           →  component "HTTP/HTTPS"
    Declaration:  component HTTP/HTTPS as h      →  component "HTTP/HTTPS" as h
    Edge:         HTTP/HTTPS --> Server           →  "HTTP/HTTPS" --> Server
    """
    if "@startuml" not in puml:
        return puml

    # Match a bare word-with-slash token that is not already inside quotes.
    # The token must contain at least one '/' and consist only of word chars and '/'.
    _SLASH_TOKEN = re.compile(r'(?<!")\b([\w][\w/]*\/[\w/]*\w)\b(?!")')

    out = []
    for ln in puml.splitlines():
        s = ln.strip()
        if not s or s.startswith("'") or s.startswith("@") or s.startswith("'"):
            out.append(ln)
            continue
        new_ln = _SLASH_TOKEN.sub(lambda m: f'"{m.group(1)}"', ln)
        if new_ln != ln:
            print(f"      [REPAIR] Quoted slash-name: {ln.strip()!r} → {new_ln.strip()!r}")
        out.append(new_ln)

    return "\n".join(out)


def repair_unbalanced_class_braces(puml: str) -> str:
    """
    Fix unbalanced braces in class diagrams that cause PlantUML to throw
    java.lang.IllegalStateException (Assumed diagram type: class).

    Handles two failure modes:
    1. Extra `}` (orphaned closer) — the LLM emits a double `}` after a class body.
       These are dropped.
    2. Unclosed `{` — the last class declaration was never closed before @enduml
       (common when the model hits max_tokens mid-class-body or puts arrows after
       the last class without closing it first).  Missing `}` are inserted before
       the @enduml line.
    """
    if "@startuml" not in puml:
        return puml

    lines = puml.splitlines()

    # Split off @enduml so we can insert closers just before it
    enduml_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "@enduml":
            enduml_idx = i
            break

    body   = lines[:enduml_idx] if enduml_idx is not None else lines
    tail   = lines[enduml_idx:] if enduml_idx is not None else []

    depth = 0
    out: List[str] = []
    for ln in body:
        stripped = ln.strip()
        opens  = stripped.count("{")
        closes = stripped.count("}")
        net    = opens - closes

        if net < 0 and depth + net < 0:
            # More closes than current depth — drop orphaned `}`
            to_drop = -(depth + net)
            fixed = ln
            for _ in range(to_drop):
                idx = fixed.rfind("}")
                if idx != -1:
                    fixed = fixed[:idx] + fixed[idx+1:]
            fixed = fixed.rstrip()
            if fixed != ln.rstrip():
                print(f"      [REPAIR] Removed {to_drop} unbalanced '}}' from: {ln.strip()!r}")
            if fixed.strip():
                out.append(fixed)
            depth = max(0, depth + net + to_drop)
        else:
            out.append(ln)
            depth = max(0, depth + net)

    # Close any still-open blocks before @enduml
    if depth > 0:
        print(f"      [REPAIR] Inserting {depth} missing '}}' before @enduml to close unclosed class body")
        out.extend(["}"] * depth)

    return "\n".join(out + tail)


# Element keywords that are only valid in specific diagram families.
# Maps diagram_type → set of line-start keywords that should NOT appear there.
_WRONG_ELEMENT_RE: Dict[str, re.Pattern] = {
    "deployment": re.compile(
        r"^\s*(participant|actor\b|boundary|control|entity|collections)\b",
        re.IGNORECASE,
    ),
    "component": re.compile(
        r"^\s*(participant|boundary|control|entity|collections)\b",
        re.IGNORECASE,
    ),
    "class": re.compile(
        r"^\s*(participant|node|cloud|queue\b|artifact|rectangle\b|frame\b|actor\b|boundary|control|entity|collections)\b",
        re.IGNORECASE,
    ),
    "object": re.compile(
        r"^\s*(participant|node|cloud|queue\b|artifact|rectangle\b|frame\b|boundary|control|entity|collections)\b",
        re.IGNORECASE,
    ),
    "usecase": re.compile(
        r"^\s*(participant|node|cloud|queue\b|artifact|boundary|control|entity|collections)\b",
        re.IGNORECASE,
    ),
}


def repair_wrong_element_types(puml: str, diagram_type: str) -> str:
    """
    Strip lines that use element keywords illegal for this diagram type.

    Example: a deployment diagram containing `participant "MqttListener" ...`
    will fail with 'Syntax Error? (Assumed diagram type: component)'.
    Dropping the offending line is safer than a retry.
    """
    rx = _WRONG_ELEMENT_RE.get(diagram_type)
    if rx is None or "@startuml" not in puml:
        return puml

    out: List[str] = []
    for ln in puml.splitlines():
        if rx.match(ln):
            print(f"      [REPAIR] Removed wrong-type element from {diagram_type}: {ln.strip()!r}")
        else:
            out.append(ln)
    return "\n".join(out)


def repair_slash_in_quoted_name(puml: str) -> str:
    """
    Fix broken quoting caused by slashes inside participant/actor display names.

    The LLM (or repair_slash_names) sometimes produces:
        participant "imagery-"ui/server".py" as Server
    where the slash caused inner quoting that breaks the outer string.

    Two patterns fixed:
    1. Inner quotes around a slash-containing token inside a larger quoted string:
           "text-"word/word"suffix"  →  "text-word_word-suffix"
    2. A bare slash remaining inside an already-quoted name:
           "some/path.py"            →  "some_path.py"
    """
    if "@startuml" not in puml:
        return puml

    # Pattern 1: "outer-"inner/word"rest" — inner pair of quotes inside outer quotes
    _INNER_QUOTED_SLASH = re.compile(r'"([^"]*)"([\w./]+/[\w./]+)"([^"]*)"')

    # Pattern 2: slash remaining inside a quoted string
    _SLASH_INSIDE_QUOTE = re.compile(r'"([^"]*)/([^"]*)"')

    out: List[str] = []
    for ln in puml.splitlines():
        s = ln.strip()
        # Only process participant/actor declaration lines
        if not re.match(r'^\s*(participant|actor)\b', s, re.IGNORECASE):
            out.append(ln)
            continue

        original = ln
        # Fix pattern 1: collapse inner quotes + slash into clean token
        ln = _INNER_QUOTED_SLASH.sub(
            lambda m: f'"{m.group(1)}{m.group(2).replace("/", "_")}{m.group(3)}"', ln
        )
        # Fix pattern 2: replace slash inside quotes with underscore
        ln = _SLASH_INSIDE_QUOTE.sub(
            lambda m: f'"{m.group(1)}_{m.group(2)}"', ln
        )
        if ln != original:
            print(f"      [REPAIR] Fixed slash in quoted name: {original.strip()!r} → {ln.strip()!r}")
        out.append(ln)

    return "\n".join(out)


def repair_trailing_edge_colon(puml: str) -> str:
    """
    Remove a trailing bare ':' on edge lines that has no label text following it.

    Example:
        rekognition --> mqtt_listener :    ← syntax error
        rekognition --> mqtt_listener      ← fixed

    PlantUML treats ' :' as the start of a label; if nothing follows it gets
    confused and sometimes misidentifies the diagram type.
    """
    if "@startuml" not in puml:
        return puml

    _EDGE_BARE_COLON = re.compile(
        r'^(\s*.*(?:-->|<--|<-->|\.\.>|<\.\.|--|\.\.).*?)\s*:\s*$'
    )
    out = []
    for ln in puml.splitlines():
        m = _EDGE_BARE_COLON.match(ln)
        if m:
            fixed = m.group(1)
            print(f"      [REPAIR] Removed trailing bare colon from edge: {ln.strip()!r}")
            out.append(fixed)
        else:
            out.append(ln)
    return "\n".join(out)


def repair_quoted_aliases(puml: str) -> str:
    """
    Fix `as "Multi Word Alias"` → `as Multi_Word_Alias`.

    PlantUML requires aliases to be plain identifiers (no spaces, no quotes).
    The LLM frequently writes:
        node "MongoDB Cluster" as "Mongo Cluster" {
    which causes a syntax error.  We slugify the quoted alias and rewrite all
    downstream references to the old quoted form.
    """
    if "@startuml" not in puml:
        return puml

    # Match:  ... as "Some Name"  (optionally followed by { or end of line)
    _QUOTED_AS = re.compile(r'\bas\s+"([^"]+)"', re.IGNORECASE)

    lines = puml.splitlines()
    # First pass: collect all quoted→slug replacements
    replacements: Dict[str, str] = {}   # "Mongo Cluster" → Mongo_Cluster
    for ln in lines:
        for m in _QUOTED_AS.finditer(ln):
            quoted = m.group(1)
            if quoted not in replacements:
                slug = re.sub(r'\W+', '_', quoted).strip('_')
                replacements[quoted] = slug

    if not replacements:
        return puml

    # Second pass: rewrite declarations and all references
    out = []
    for ln in lines:
        original = ln
        # Replace  as "Quoted Name"  →  as Slug
        for quoted, slug in replacements.items():
            ln = re.sub(
                r'\bas\s+"' + re.escape(quoted) + r'"',
                f'as {slug}',
                ln, flags=re.IGNORECASE,
            )
            # Also replace bare "Quoted Name" used as a reference on edge lines
            # (only on non-declaration lines to avoid double-processing)
            ln = re.sub(r'"' + re.escape(quoted) + r'"', slug, ln)
        if ln != original:
            print(f"      [REPAIR] Slugified quoted alias: {original.strip()!r} → {ln.strip()!r}")
        out.append(ln)
    return "\n".join(out)


def repair_forward_referenced_objects(puml: str) -> str:
    """
    Fix 'Object already exists' errors in object diagrams.

    When edges reference an alias before its `object "..." as alias` declaration,
    PlantUML implicitly creates a bare element for that alias.  The explicit
    declaration later then collides.

    Fix: move all `object "..." as alias` declarations to the top of the diagram
    body (just after @startuml), before any edges or field assignments.
    """
    if "@startuml" not in puml:
        return puml

    _OBJ_DECL = re.compile(r'^\s*object\s+', re.IGNORECASE)
    _EDGE_OR_FIELD = re.compile(
        r'(-->|<--|<-->|\.\.>|<\.\.|--|\.\.|^\s*\w+\s*:\s*\w+\s*=)',
    )

    lines = puml.splitlines()
    header = []
    decls = []
    body = []

    for ln in lines:
        s = ln.strip()
        if s.startswith('@startuml') or s.startswith('@enduml') or not s:
            header.append(ln) if s.startswith('@startuml') or not s and not body else body.append(ln)
            if s.startswith('@enduml'):
                body.append(ln)
            elif s.startswith('@startuml'):
                pass  # already added
        elif _OBJ_DECL.match(ln):
            decls.append(ln)
        else:
            body.append(ln)

    if not decls:
        return puml

    # Rebuild: @startuml, object declarations, then rest
    startuml_line = next((ln for ln in lines if ln.strip().startswith('@startuml')), '@startuml')
    enduml_line   = next((ln for ln in lines if ln.strip() == '@enduml'), '@enduml')
    inner = [ln for ln in lines
             if not ln.strip().startswith('@startuml')
             and ln.strip() != '@enduml'
             and not _OBJ_DECL.match(ln)]

    result = [startuml_line] + decls + inner + [enduml_line]
    rebuilt = "\n".join(result)
    if rebuilt != puml:
        print(f"      [REPAIR] Moved {len(decls)} object declaration(s) before edges.")
    return rebuilt


def repair_quoted_state_targets(puml: str) -> str:
    """
    Fix state diagram transitions that use quoted strings as targets instead of
    plain identifiers or declared aliases.

    PlantUML state diagrams do not allow quoted strings as transition endpoints:
        WRONG:   [*] --> "Initializing"
        CORRECT: [*] --> Initializing       (if declared as plain state)
        CORRECT: [*] --> initializing       (if declared as  state "Initializing" as initializing)

    Strategy:
    1. Collect all declared alias identifiers (from  state "..." as ALIAS  lines).
    2. On transition lines (containing --> or <--), strip quotes from targets:
       - If a quoted name matches a declared alias's display name, replace with alias.
       - Otherwise, convert the quoted name to a plain snake_case identifier.
    """
    if "@startuml" not in puml:
        return puml

    lines = puml.splitlines()

    # Collect declared aliases:  state "Display Name" as alias
    alias_map: Dict[str, str] = {}  # display_name.lower() -> alias
    for ln in lines:
        m = re.match(r'\s*state\s+"([^"]+)"\s+as\s+(\w+)', ln)
        if m:
            alias_map[m.group(1).lower()] = m.group(2)

    # Transition line pattern: anything --> "Quoted" or "Quoted" --> anything
    TRANS_RE = re.compile(r'(-->|<--)')
    QUOTED_TARGET = re.compile(r'"([^"]+)"')

    out = []
    for ln in lines:
        if TRANS_RE.search(ln) and '"' in ln:
            def replace_quoted(m):
                name = m.group(1)
                # Check if it matches a known alias display name
                if name.lower() in alias_map:
                    return alias_map[name.lower()]
                # Otherwise convert to snake_case identifier
                ident = re.sub(r'[^\w]+', '_', name).strip('_')
                print(f"      [REPAIR] state transition: quoted \"{name}\" -> {ident}")
                return ident
            fixed = QUOTED_TARGET.sub(replace_quoted, ln)
            out.append(fixed)
        else:
            out.append(ln)

    return "\n".join(out)


def repair_stray_closing_braces(puml: str, diagram_type: str) -> str:
    """
    Remove `}` lines that have no matching opener in the diagram body.

    This catches the pattern seen in usecase diagrams where the LLM emits a
    stray `}` (likely meant to close a `rectangle` block it forgot to open),
    which breaks PlantUML's group context and causes a java.lang.IllegalStateException
    when subsequent edges try to add elements outside any valid group.

    Only applied to diagram types where bare group context errors occur:
    usecase, component, deployment.
    """
    if diagram_type not in ("usecase", "component", "deployment", "sequence"):
        return puml
    if "@startuml" not in puml:
        return puml

    depth = 0
    out = []
    for ln in puml.splitlines():
        s = ln.strip()
        # Track openers (lines ending with { or being pure {)
        opens  = s.count('{')
        closes = s.count('}')
        if closes > opens and depth + (opens - closes) < 0:
            to_drop = closes - opens - depth
            fixed = ln
            for _ in range(to_drop):
                idx = fixed.rfind('}')
                if idx != -1:
                    fixed = fixed[:idx] + fixed[idx+1:]
            fixed = fixed.rstrip()
            if fixed != ln.rstrip():
                print(f"      [REPAIR] Removed {to_drop} stray '}}' in {diagram_type}: {ln.strip()!r}")
            if fixed.strip():
                out.append(fixed)
            depth = max(0, depth + opens - closes + to_drop)
        else:
            out.append(ln)
            depth = max(0, depth + opens - closes)

    return "\n".join(out)


def repair_orphan_activity_keywords(puml: str) -> str:
    """
    Remove activity control keywords that have no matching opener.

    Patterns caught:
    - `endwhile` with no preceding `while`
    - `end fork` / `fork again` with no preceding `fork`
    - `endif` with no preceding `if`
    - `end` with no preceding `repeat` or `while`

    Strategy: single forward pass tracking open counts for each block type.
    Any closer that would push its counter below zero is dropped.
    """
    if "@startuml" not in puml:
        return puml

    # (opener_pattern, closer_pattern)
    _BLOCKS = [
        (re.compile(r'^\s*while\b',      re.IGNORECASE),
         re.compile(r'^\s*endwhile\b',   re.IGNORECASE)),
        (re.compile(r'^\s*fork\b',       re.IGNORECASE),
         re.compile(r'^\s*(end\s+fork|fork\s+again)\b', re.IGNORECASE)),
        (re.compile(r'^\s*if\b',         re.IGNORECASE),
         re.compile(r'^\s*endif\b',      re.IGNORECASE)),
        (re.compile(r'^\s*repeat\b',     re.IGNORECASE),
         re.compile(r'^\s*repeat\s+while\b', re.IGNORECASE)),
    ]

    counters = [0] * len(_BLOCKS)
    out = []
    for ln in puml.splitlines():
        drop = False
        for i, (opener, closer) in enumerate(_BLOCKS):
            if opener.match(ln):
                counters[i] += 1
            elif closer.match(ln):
                if counters[i] <= 0:
                    print(f"      [REPAIR] Removed orphan activity keyword: {ln.strip()!r}")
                    drop = True
                    break
                else:
                    counters[i] -= 1
        if not drop:
            out.append(ln)
    return "\n".join(out)


def repair_undeclared_alias_edges(puml: str, diagram_type: str) -> str:
    """
    Remove edge lines that reference an alias which was never declared.

    Root cause: the LLM declares  rectangle "Imagery Pipeline" { }  with no
    `as alias` clause, then later writes edges like:
        view_imagery --> imagery_pipeline : uses
    PlantUML implicitly creates `imagery_pipeline` as an unknown type, which
    causes diagram-type detection to misfire (often 'Assumed type: component').

    Strategy:
    1. Collect all declared aliases (from `as <alias>` clauses).
    2. Collect all bare identifiers used on edge lines.
    3. Drop edge lines where BOTH endpoints are unknown (fully undeclared edges).
       Edges where at least one end is a known alias are kept.
    """
    if "@startuml" not in puml:
        return puml

    _AS_DECL   = re.compile(r'\bas\s+(\w+)', re.IGNORECASE)
    _EDGE_LINE = re.compile(
        r'(-->|<--|<-->|\.\.>|<\.\.|--|\.\.|<\|--|--\|>|\*--|o--)',
        re.IGNORECASE,
    )
    # Split an edge line into LHS and RHS around the arrow
    _EDGE_SPLIT = re.compile(
        r'^(\s*)([\w"]+(?:\s+[\w"]+)*?)\s*'
        r'(-->|<--|<-->|\.\.>|<\.\.|--|\.\.)'
        r'\s*([\w"]+(?:\s+[\w"]+)*?)(\s*:.*)?$',
        re.IGNORECASE,
    )

    lines = puml.splitlines()

    # Pass 1: collect declared aliases and bare-word element names
    declared: set = set()
    for ln in lines:
        for m in _AS_DECL.finditer(ln):
            declared.add(m.group(1))

    if not declared:
        return puml  # nothing to cross-reference against

    # Pass 2: drop edges where both sides are undeclared bare identifiers
    out = []
    for ln in lines:
        if not _EDGE_LINE.search(ln) or ln.strip().startswith("'"):
            out.append(ln)
            continue
        m = _EDGE_SPLIT.match(ln)
        if not m:
            out.append(ln)
            continue
        lhs = m.group(2).strip().strip('"')
        rhs = m.group(4).strip().strip('"')
        # Keep if at least one side is a known declared alias
        if lhs in declared or rhs in declared:
            out.append(ln)
        else:
            print(f"      [REPAIR] Removed edge with undeclared aliases ({lhs!r}, {rhs!r}): {ln.strip()!r}")

    return "\n".join(out)


def extract_start_end_block(raw: str) -> str:
    """
    Extract the @startuml...@enduml block from raw model output.

    Handles:
    - Clean output (happy path)
    - Markdown fences (```plantuml or ``` wrapping the block)
    - Truncated output (model hit max_tokens before writing @enduml):
        * Appends @enduml after stripping any incomplete trailing line
        * Warns so the operator knows to increase --max-tokens
    """
    # Strip markdown code fences regardless of position.
    # Matches ```plantuml, ```uml, ``` on their own line, plus closing ```.
    clean = re.sub(r"^\s*```+(?:plantuml|uml)?\s*$", "", raw, flags=re.MULTILINE)
    clean = re.sub(r"^\s*```+\s*$", "", clean, flags=re.MULTILINE)

    # Happy path: complete block present
    m = re.search(r"@startuml.*?@enduml", clean, re.DOTALL)
    if m:
        return m.group(0).strip()

    # Truncated: @startuml present but @enduml missing
    start = re.search(r"@startuml", clean)
    if start:
        fragment = clean[start.start():].rstrip()
        lines = fragment.splitlines()
        # Drop any trailing incomplete line (e.g. 'object "Foo" as' with no alias,
        # or 'class Bar {' that was never closed at the token boundary).
        # A line is "incomplete" if it ends with common truncation patterns.
        _INCOMPLETE = re.compile(
            r"(?:"
            r"object\s+\"[^\"]*\"\s+as\s*$"      # object "X:Y" as   ← no alias
            r"|class\s+\w[\w.]*\s*\{?\s*$"        # class Foo {       ← open brace, no body
            r"|\bas\s*$"                           # dangling 'as'
            r")"
        )
        while lines and _INCOMPLETE.search(lines[-1].strip()):
            lines.pop()
        # Also strip a trailing open brace with no matching content
        fragment = "\n".join(lines)
        print("      [WARN] Output truncated (no @enduml found). "
              "Diagram may be incomplete — consider raising --max-tokens.")
        return fragment + "\n@enduml"

    # No @startuml at all — wrap entire cleaned content as a best-effort
    if clean.strip():
        return f"@startuml\n{clean.strip()}\n@enduml"
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
    max_model_len: int = 32768,
) -> Dict:
    # Guard: max_tokens must be at least 1 to avoid consuming the entire context.
    # Add CONTEXT_SAFETY_MARGIN so our check is always stricter than vLLM's hard limit.
    effective_output = max(max_tokens, 1)
    input_budget = max_model_len - effective_output - CONTEXT_SAFETY_MARGIN

    # Try progressively slimmer representations of the extractions until the
    # formatted prompt fits within the model's input token budget.
    MAX_COMPRESS_LEVEL = 7
    chosen_prompt = None
    for level in range(MAX_COMPRESS_LEVEL + 1):
        slim = _compress_extractions(extractions, level)
        user_msg = REGISTRY_TMPL.format(
            repo_name=repo_name,
            summaries_json=json.dumps(slim, indent=2),
        )
        prompt = format_prompt(tokenizer, REGISTRY_SYSTEM, user_msg)
        n_tokens = _count_tokens(tokenizer, prompt)
        if level == 0:
            print(f"      Registry prompt: {len(user_msg):,} chars  ({n_tokens:,} tokens)")
        if n_tokens <= input_budget:
            if level > 0:
                print(f"      [INFO] Compressed extractions to level {level} "
                      f"({n_tokens:,} tokens ≤ budget {input_budget:,})")
            chosen_prompt = prompt
            break
        else:
            print(f"      [WARN] Level {level}: {n_tokens:,} tokens > budget {input_budget:,} "
                  f"— compressing further…")

    if chosen_prompt is None:
        # Absolute last resort: send level-7 anyway and let vLLM hard-truncate
        print("      [ERROR] Could not fit registry prompt even at maximum compression. "
              "Sending anyway — expect a vLLM truncation error or degraded output.")
        chosen_prompt = prompt  # last attempted prompt from the loop

    raw = vllm_generate_one(llm, chosen_prompt, max_tokens=max_tokens)
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
    char_budget: int = GENERATION_CHUNK_BUDGET,
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
        if total + len(block) > char_budget:
            rem = char_budget - total
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
    max_model_len: int = 32768,
) -> Dict[str, str]:
    """
    Build all 8 diagram prompts, submit as one batch.
    If a diagram fails lightweight validation, retry that diagram once with a repair prompt.

    For models with tight context windows (e.g. Mistral-Large at 32k), the
    combined prompt (system + registry JSON + RAG examples + code chunks) can
    exceed the input budget even when GENERATION_CHUNK_BUDGET looks safe in
    characters.  We therefore measure the *token* overhead for each diagram
    type's fixed parts (registry + RAG + template boilerplate) and compute a
    per-diagram code-chunk character budget that guarantees the prompt fits.
    """
    # Guard: max_tokens must be at least 1 to avoid consuming the entire context.
    # Add CONTEXT_SAFETY_MARGIN so our check is always stricter than vLLM's hard limit.
    effective_output = max(max_tokens, 1)
    input_budget = max_model_len - effective_output - CONTEXT_SAFETY_MARGIN

    dtypes  = [dt for dt, _ in diagram_types]
    prompts: List[str] = []

    MAX_REG_LEVEL = 7

    for dtype, desc in diagram_types:
        rag_examples = rag_by_type.get(dtype, "")

        # ── Find the lowest registry compression level that fits ──
        # We try increasing compression until both the fixed overhead fits AND
        # there are tokens left over for at least some code chunks.
        chosen_prompt = None
        for reg_level in range(MAX_REG_LEVEL + 1):
            slim_registry = _compress_registry(registry, reg_level)
            registry_json = json.dumps(slim_registry, indent=2)

            # Step 1: measure fixed overhead (no code chunks yet)
            user_msg_empty = GENERATION_TMPL.format(
                dtype_upper=dtype.upper(),
                repo_name=repo_name,
                description=desc,
                syntax_hint=DIAGRAM_SYNTAX_HINTS.get(dtype, ""),
                registry_json=registry_json,
                code_chunks="",
                rag_examples=rag_examples,
            )
            overhead_tokens = _count_tokens(
                tokenizer, format_prompt(tokenizer, GENERATION_SYSTEM, user_msg_empty)
            )

            if overhead_tokens > input_budget:
                if reg_level == 0:
                    print(f"      [WARN] {dtype}: full registry overhead "
                          f"({overhead_tokens} tok) > budget ({input_budget} tok) "
                          f"— compressing registry…")
                continue  # try next compression level

            # Overhead fits — now compute code chunk budget
            remaining_tokens = input_budget - overhead_tokens
            char_budget = int(remaining_tokens * 3.0)
            char_budget = min(char_budget, GENERATION_CHUNK_BUDGET)
            code_chunks = select_chunks(dtype, files, registry, char_budget) if char_budget > 200 else ""

            # Verify full prompt and shrink code chunks if still over
            user_msg = GENERATION_TMPL.format(
                dtype_upper=dtype.upper(),
                repo_name=repo_name,
                description=desc,
                syntax_hint=DIAGRAM_SYNTAX_HINTS.get(dtype, ""),
                registry_json=registry_json,
                code_chunks=code_chunks,
                rag_examples=rag_examples,
            )
            prompt_candidate = format_prompt(tokenizer, GENERATION_SYSTEM, user_msg)
            n_tokens = _count_tokens(tokenizer, prompt_candidate)

            shrink = 0
            while n_tokens > input_budget and char_budget > 500:
                char_budget = int(char_budget * 0.85)
                shrink += 1
                code_chunks = select_chunks(dtype, files, registry, char_budget)
                user_msg = GENERATION_TMPL.format(
                    dtype_upper=dtype.upper(),
                    repo_name=repo_name,
                    description=desc,
                    syntax_hint=DIAGRAM_SYNTAX_HINTS.get(dtype, ""),
                    registry_json=registry_json,
                    code_chunks=code_chunks,
                    rag_examples=rag_examples,
                )
                prompt_candidate = format_prompt(tokenizer, GENERATION_SYSTEM, user_msg)
                n_tokens = _count_tokens(tokenizer, prompt_candidate)

            if reg_level > 0 or shrink:
                print(f"      [INFO] {dtype}: registry level {reg_level}, "
                      f"code_chunks shrunk {shrink}x → {n_tokens:,} tokens")

            chosen_prompt = prompt_candidate
            break

        if chosen_prompt is None:
            # Even level-7 registry overflows — send it anyway, vLLM will error
            # but this is a genuinely pathological repo for this context size.
            print(f"      [ERROR] {dtype}: cannot fit even minimal registry in "
                  f"{input_budget} token budget. Sending anyway.")
            chosen_prompt = format_prompt(tokenizer, GENERATION_SYSTEM,
                GENERATION_TMPL.format(
                    dtype_upper=dtype.upper(), repo_name=repo_name,
                    description=desc, syntax_hint=DIAGRAM_SYNTAX_HINTS.get(dtype, ""),
                    registry_json=json.dumps(_compress_registry(registry, 7), indent=2),
                    code_chunks="", rag_examples="",
                ))

        prompts.append(chosen_prompt)

    print(f"      Submitting {len(prompts)} diagram prompts as one batch...")
    raw_outputs = vllm_generate_batch(llm, prompts, max_tokens=max_tokens)

    results: Dict[str, str] = {}
    raw_by_type: Dict[str, str] = {}
    for dtype, raw in zip(dtypes, raw_outputs):
        raw_by_type[dtype] = raw
        puml = extract_start_end_block(raw)
        # Auto-repair common structural mistakes before validation
        puml = repair_duplicate_aliases(puml)
        puml = repair_quoted_aliases(puml)
        puml = repair_unquoted_multiword_edges(puml)
        puml = repair_slash_names(puml)
        puml = repair_slash_in_quoted_name(puml)
        puml = repair_wrong_element_types(puml, dtype)
        puml = repair_trailing_edge_colon(puml)
        puml = repair_stray_closing_braces(puml, dtype)
        puml = repair_undeclared_alias_edges(puml, dtype)
        if dtype == "activity":
            puml = repair_orphan_activity_keywords(puml)
            puml = repair_truncated_activity(puml)
        if dtype == "class":
            puml = repair_class_diagram_lines(puml)
            puml = repair_unbalanced_class_braces(puml)
        if dtype == "state":
            puml = repair_quoted_state_targets(puml)
        if dtype == "object":
            puml = repair_forward_referenced_objects(puml)
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
            repaired = repair_duplicate_aliases(repaired)
            repaired = repair_quoted_aliases(repaired)
            repaired = repair_unquoted_multiword_edges(repaired)
            repaired = repair_slash_names(repaired)
            repaired = repair_slash_in_quoted_name(repaired)
            repaired = repair_wrong_element_types(repaired, dtype)
            repaired = repair_trailing_edge_colon(repaired)
            repaired = repair_stray_closing_braces(repaired, dtype)
            repaired = repair_undeclared_alias_edges(repaired, dtype)
            if dtype == "activity":
                repaired = repair_orphan_activity_keywords(repaired)
                repaired = repair_truncated_activity(repaired)
            if dtype == "class":
                repaired = repair_class_diagram_lines(repaired)
                repaired = repair_unbalanced_class_braces(repaired)
            if dtype == "state":
                repaired = repair_quoted_state_targets(repaired)
            if dtype == "object":
                repaired = repair_forward_referenced_objects(repaired)
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
    _env_len = os.environ.get("VLLM_MAX_LEN")
    parser.add_argument("--max-model-len", type=int,
                        default=int(_env_len) if _env_len else None,
                        help="Override context length. Auto-detected from model name if omitted.")
    parser.add_argument("--max-tokens",    type=int,
                        default=int(os.environ.get("VLLM_MAX_TOKENS",  "8192")))
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
    parser.add_argument("--tokenizer-mode", default="auto",
                        help="vLLM tokenizer mode. Use 'mistral' for Mistral models (default: auto).")
    parser.add_argument("--config-format", default="auto",
                        help="vLLM config format. Use 'mistral' for Mistral models (default: auto).")
    parser.add_argument("--load-format", default="auto",
                        help="vLLM load format. Use 'mistral' for Mistral quantized weights (default: auto).")
    parser.add_argument("--rag-k",  type=int,
                        default=int(os.environ.get("RAG_TOP_K", "5")))
    parser.add_argument("--registry-file",
                        help="Load existing registry JSON — skips Pass 1 and Pass 2.")
    args = parser.parse_args()

    # Resolve effective context length (override > model table > fallback)
    _override = args.max_model_len  # None if not supplied
    args.max_model_len = resolve_max_model_len(args.model, _override)
    if _override is None:
        print(f"[INFO] Auto-detected max_model_len={args.max_model_len:,} for model '{args.model}'")
    else:
        print(f"[INFO] Using --max-model-len={args.max_model_len:,} (override)")

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
    preflight_vram_check(args.model, args.tp, args.gpu_memory_utilization)
    print(f"    {args.model}  |  tp={args.tp}  |  max_len={args.max_model_len}")
    if args.tokenizer_mode != "auto" or args.load_format != "auto":
        print(f"    tokenizer_mode={args.tokenizer_mode}  config_format={args.config_format}  load_format={args.load_format}")
    llm = load_model(
        model=args.model,
        tp=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tokenizer_mode=args.tokenizer_mode,
        config_format=args.config_format,
        load_format=args.load_format,
    )
    # Grab tokenizer for chat template formatting
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(
        args.model,
        trust_remote_code=True,
        tokenizer_mode=args.tokenizer_mode,
    )
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
        registry = pass2_build_registry(extractions, repo_name, llm, tokenizer, args.registry_tokens, args.max_model_len)
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
        max_model_len=args.max_model_len,
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
