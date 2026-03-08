# PlantUML-RAG: Automated UML Diagram Generation

**Generate complete, consistent UML diagrams from any code repository using RAG + LLMs.**

This toolkit parses a code repository, builds a canonical entity registry, retrieves
high-quality PlantUML examples via FAISS-based RAG, and generates **all 8 UML diagram
types in one LLM call** to guarantee naming consistency and correctness.

---

## ✨ Features

- **Three-pass generation pipeline** for 8 UML diagram types:
  - Class, Sequence, Activity, State
  - Component, Deployment, Use-case, Object diagrams
- **Consistent naming** across all diagrams via a canonical Entity Registry
- **FAISS RAG** for syntactically correct PlantUML examples
- **Automatic context-window management** — compresses inputs to fit any model's context
- **Automatic PlantUML repair** — fixes common LLM output errors before rendering
- **Two deployment options**: Chunked vLLM (recommended) or Ollama
- **Simple, standalone scripts** — no complex frameworks
- **Re-run acceleration** — skip extraction with `--registry-file`

---

## 🏗️ How the Pipeline Works

The vLLM script uses a **three-pass architecture** where the model is loaded once and
reused across all passes:

```
Pass 1 — Extraction (batched)
  For each .py file → LLM extracts classes, functions, relationships, imports
  All files submitted as a single vLLM batch across all GPUs

Pass 2 — Entity Registry (single call)
  All per-file extractions → LLM consolidates into one canonical registry
  Resolves name conflicts, picks authoritative class names
  Registry is saved to disk — use --registry-file to skip on re-runs

Pass 3 — Diagram Generation (batched)
  All 8 diagram types submitted as a single vLLM batch
  Each prompt contains: registry + RAG examples + relevant source chunks
  One retry pass for any diagrams that fail validation
```

The registry saved after Pass 2 lets you regenerate or tweak diagrams without
re-running the expensive extraction step:

```bash
python repo_to_diagrams_chunked_vllm.py \
  --registry-file uml_out/myrepo_entity_registry.json \
  --input /path/to/repo \
  --output uml_out \
  ...
```

---

## 🚀 Quick Start

Choose your deployment model:

### Option 1: Chunked vLLM (Recommended for DGX / GPU Workstation)

**Best if:** You have GPUs and want maximum quality, performance, and model choice

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install vllm  # Works with CUDA 11.8 and 12.x

# 2. Create RAG index (one-time setup)
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# 3. Generate diagrams
python repo_to_diagrams_chunked_vllm.py \
  --input ~/Code/YourProject \
  --output ./uml_out \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 4 \
  --max-model-len 48000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Option 2: Ollama (Easiest Setup)

**Best if:** You want the simplest installation, or need to access a remote server
from a laptop without local GPU

**Note:** Ollama can run locally (laptop, smaller models) or remotely (DGX, large models)

```bash
# 1. Install Ollama and pull a model
# Visit https://ollama.ai for installation
ollama pull llama3.1:8b           # local/laptop
# OR on a DGX: ollama pull llama4:maverick

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create RAG index (requires Ollama running for embeddings)
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# 4. Generate diagrams
# Local Ollama:
python repo_to_diagrams_ollama.py \
  --input ~/Code/YourProject \
  --llm-model llama3.1:8b \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# OR Remote Ollama (e.g., on DGX):
python repo_to_diagrams_ollama.py \
  --input ~/Code/YourProject \
  --llm-model llama4:maverick \
  --ollama-url http://192.168.100.100:11434 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 📦 What You Get

For each repository, generates 8 diagram files plus an entity registry:

```
uml_out/
├── myrepo_entity_registry.json  # Canonical entity registry (re-use with --registry-file)
├── myrepo_class.puml            # Class structure & relationships
├── myrepo_sequence.puml         # Interaction flows over time
├── myrepo_activity.puml         # Process workflows
├── myrepo_state.puml            # State machine transitions
├── myrepo_component.puml        # High-level architecture
├── myrepo_deployment.puml       # Deployment view
├── myrepo_usecase.puml          # User-facing use cases
└── myrepo_object.puml           # Runtime object instances
```

Render them:
```bash
plantuml uml_out/*.puml          # PNG (default)
plantuml -tsvg uml_out/*.puml    # SVG (vector)
plantuml -checkonly uml_out/*.puml  # Syntax check only
```

---

## 🛠️ Installation

### Core Dependencies

```bash
pip install -r requirements.txt
```

This installs: `faiss-cpu`, `sentence-transformers`, `numpy`, `requests`

### Backend-Specific

**For Chunked vLLM:**
```bash
pip install vllm  # Works with CUDA 11.8 and 12.x
```

**For Ollama:**
```bash
# Install from https://ollama.ai/
ollama pull llama3.1:8b        # Smaller model, runs on laptop
ollama pull llama4:maverick    # Larger model, needs GPU server
```

---

## ⚖️ Choosing Your Backend: vLLM vs Ollama

Understanding the tradeoffs will help you pick the right tool for each situation.

### Architecture Difference

**vLLM (`repo_to_diagrams_chunked_vllm.py`)**
- Loads the model directly into GPU memory at startup
- Runs the three-pass pipeline (extract → registry → generate) with the model in-process
- All GPU resources are dedicated to this one job for its duration
- Model is unloaded when the script exits

**Ollama (`repo_to_diagrams_ollama.py`)**
- Connects to a running Ollama server over HTTP
- Ollama manages model loading/unloading and keeps the model resident between calls
- Each generation is an HTTP request; results stream back over the network
- Multiple clients can share the same Ollama server

### Detailed Comparison

| Factor | vLLM (Chunked) | Ollama |
|--------|---------------|--------|
| **Setup complexity** | Moderate — vLLM install, CUDA deps | Easy — single binary + `ollama pull` |
| **GPU requirement** | Required (CUDA) | Optional — CPU fallback available |
| **Throughput** | Very high — native GPU, tensor parallel | Lower — HTTP overhead, serial requests |
| **Batching** | All 8 diagrams in one GPU call | One diagram at a time |
| **Model selection** | Any HuggingFace model (GGUF not needed) | Ollama model library only |
| **Context handling** | Automatic compression ladder | Fixed context, manual tuning |
| **Multi-user** | Single-user only (owns all GPUs) | Multi-user — Ollama server shared |
| **Remote access** | Script must run on the GPU machine | Script runs anywhere, server is remote |
| **Startup time** | Slow — model loads each run (~1-2 min) | Fast — model already resident |
| **Interruption** | Loses all progress | Can resume (registry + `--registry-file`) |
| **Large repos** | Handles gracefully with compression | May hit context limits silently |
| **Diagram quality** | Higher — larger models, more context | Variable — depends on model size |

### When to Use vLLM

- You're the sole user of a DGX or GPU workstation
- You want maximum diagram quality (largest models, most context)
- You're processing many repositories in a batch job
- You need models not available in Ollama (e.g., Mistral-Large, Llama-4-Scout)
- You want automatic context window management for large repos

### When to Use Ollama

- You're on a laptop or machine without a GPU
- You want to share a model server among several users or scripts
- You need to run the generation script from a different machine than the GPU
- You want quick iteration — model stays loaded between test runs
- You're prototyping with smaller models before committing to a full vLLM run

### The "Remote DGX" Pattern

A common hybrid approach: run Ollama on the DGX with a large model, and run the
Python script from your laptop. This gives you large-model quality without needing
to SSH into the DGX and without the vLLM startup penalty:

```
Laptop                          DGX
──────                          ───
repo_to_diagrams_ollama.py  →   Ollama server (llama4:maverick)
FAISS index (local copy)        GPU inference
```

```bash
# On DGX: ensure Ollama is serving and model is pulled
ollama serve &
ollama pull llama4:maverick

# On laptop: run script pointing at remote Ollama
python repo_to_diagrams_ollama.py \
  --input ~/Code/MyProject \
  --llm-model llama4:maverick \
  --ollama-url http://dgx-hostname:11434 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 📂 Project Structure

```
plantuml-rag/
├── repo_to_diagrams_chunked_vllm.py  # Main script: Chunked vLLM (3-pass pipeline)
├── repo_to_diagrams_ollama.py        # Main script: Ollama
├── requirements.txt                   # Python dependencies
│
├── util/                              # Utility scripts
│   ├── build_faiss_rag.py            # Build RAG index
│   ├── test_faiss_query.py           # Test RAG retrieval
│   └── test_plantuml_vllm.py         # Validate vLLM setup
│
└── rag/                               # RAG data (created by you)
    ├── plantuml_rag_corpus.jsonl     # Your training data
    ├── faiss.index                   # FAISS index (generated)
    └── faiss_meta.json               # Metadata (generated)
```

---

## 💡 Usage Examples

### Chunked vLLM — Llama-4-Scout (recommended default)

```bash
python repo_to_diagrams_chunked_vllm.py \
  --input /path/to/code/repo \
  --output /path/to/uml_out \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 4 \
  --max-model-len 48000 \
  --max-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --temperature 0.0 \
  --rag-k 10 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Chunked vLLM — Mistral-Large

Mistral-Large produces high-quality output but has a tighter context window and a
less efficient tokenizer than Llama. The script handles this automatically, but
these settings are recommended:

```bash
python repo_to_diagrams_chunked_vllm.py \
  --input /path/to/code/repo \
  --output /path/to/uml_out \
  --model mistralai/Mistral-Large-Instruct-2411 \
  --tp 4 \
  --max-model-len 48000 \
  --max-tokens 2048 \
  --gpu-memory-utilization 0.80 \
  --temperature 0.0 \
  --rag-k 10 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

> **Note:** Mistral's tokenizer is less efficient than Llama's — the same text
> tokenizes to ~10–15% more tokens. The script automatically compresses the entity
> registry and code chunks to fit. For large repos (200+ classes) you may see
> `[INFO] registry level N` messages in the output — this is expected and harmless.

### Re-running Pass 3 Only (skip extraction)

If you already have a registry from a previous run, skip Passes 1 and 2:

```bash
python repo_to_diagrams_chunked_vllm.py \
  --input /path/to/code/repo \
  --output /path/to/uml_out \
  --registry-file uml_out/myrepo_entity_registry.json \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 4 \
  --max-model-len 48000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

This reduces a multi-minute run to just the ~10 minute diagram generation step.

### Ollama — Local

```bash
ollama serve
python repo_to_diagrams_ollama.py \
  --input /path/to/code/repo \
  --output /path/to/uml_out \
  --llm-model llama3.1:8b \
  --rag-k 10 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Ollama — Remote DGX

```bash
python repo_to_diagrams_ollama.py \
  --input ~/Code/MyProject \
  --output ./diagrams \
  --llm-model llama4:maverick \
  --ollama-url http://192.168.100.100:11434 \
  --rag-k 20 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 🧠 Creating Your RAG Index

### Prerequisites

You need a PlantUML corpus in JSONL format (Alpaca-style):

```jsonl
{"instruction": "Create a class diagram for...", "input": "", "output": "@startuml\n...\n@enduml"}
{"instruction": "Generate a sequence diagram...", "input": "", "output": "@startuml\n...\n@enduml"}
```

Save the corpus as `rag/plantuml_rag_corpus.jsonl`.

### Build the Index

The RAG index uses `sentence-transformers` for embeddings — **no Ollama required**.

```bash
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --embed-model nomic-embed-text
```

If your `build_faiss_rag.py` still uses Ollama for embeddings:
```bash
# Ensure Ollama is running with the embedding model
ollama serve
ollama pull nomic-embed-text

python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --embed-model nomic-embed-text \
  --ollama-url http://localhost:11434
```

**Environment variables alternative:**
```bash
export RAG_CORPUS=rag/plantuml_rag_corpus.jsonl
export RAG_FAISS_INDEX=rag/faiss.index
export RAG_FAISS_META=rag/faiss_meta.json
export RAG_EMBED_MODEL=nomic-embed-text

python util/build_faiss_rag.py
```

### Test the Index

```bash
python util/test_faiss_query.py \
  "sequence diagram for async worker queues" \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --top-k 5
```

---

## 🔧 Configuration & Tuning

### Chunked vLLM Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Llama-4-Scout | HuggingFace model ID |
| `--tp` | `4` | Tensor parallel (number of GPUs) |
| `--max-model-len` | `32000` | Model context window size |
| `--max-tokens` | `2048` | Max output tokens per diagram |
| `--extract-tokens` | `1024` | Max output tokens for Pass 1 extraction |
| `--registry-tokens` | `2048` | Max output tokens for Pass 2 registry |
| `--gpu-memory-utilization` | `0.85` | Fraction of GPU VRAM to use |
| `--temperature` | `0.0` | Sampling temperature (0 = deterministic) |
| `--rag-k` | `5` | Number of RAG examples per diagram type |
| `--registry-file` | — | Skip Passes 1+2, load existing registry |
| `--enforce-eager` | off | Disable CUDA graph capture (debug mode) |

### DGX A100 Recommended Settings

**Llama-4-Scout (best quality/speed balance):**
```bash
--model meta-llama/Llama-4-Scout-17B-16E-Instruct
--tp 4 --max-model-len 48000 --max-tokens 2048
--gpu-memory-utilization 0.80 --temperature 0.0 --rag-k 10
```

**Mistral-Large (high quality, tighter context):**
```bash
--model mistralai/Mistral-Large-Instruct-2411
--tp 4 --max-model-len 48000 --max-tokens 2048
--gpu-memory-utilization 0.80 --temperature 0.0 --rag-k 10
```

**Llama-3.3-70B (large, high quality):**
```bash
--model meta-llama/Llama-3.3-70B-Instruct
--tp 4 --max-model-len 32000 --max-tokens 2048
--gpu-memory-utilization 0.85 --temperature 0.0 --rag-k 10
```

**Fast testing:**
```bash
--model meta-llama/Llama-3-8B-Instruct --tp 1 --rag-k 5 --temperature 0.1
```

### Quality vs Speed Tradeoffs

**High Quality:**
```bash
--rag-k 20 --max-tokens 4096 --temperature 0.0
```

**Balanced (Recommended):**
```bash
--rag-k 10 --max-tokens 2048 --temperature 0.0
```

**Fast (Testing):**
```bash
--rag-k 5 --max-tokens 1024 --temperature 0.1
```

---

## 🔄 Automatic Context Management

For large repositories, prompts can exceed the model's context window. The chunked
vLLM script handles this automatically and transparently — **no manual tuning needed**.

### Pass 2 (Registry): Extraction Compression Ladder

If the combined per-file extractions exceed the token budget, the script progressively
strips less-important fields until it fits:

| Level | What's kept |
|-------|-------------|
| 0 | Full fidelity — all fields |
| 1 | Drop per-class `attributes` |
| 2 | Also drop `imports` per file |
| 3 | Truncate `methods` to 5 per class |
| 4 | Drop per-class `relationships` |
| 5 | Drop per-file `relationships` |
| 6 | Class names + bases + 10 functions + summary |
| 7 | Class names + summary only (minimum viable) |

### Pass 3 (Generation): Registry Compression Ladder

If the entity registry itself is too large for a diagram prompt, it is compressed
independently per diagram type:

| Level | What's kept |
|-------|-------------|
| 0 | Full fidelity — all fields |
| 1 | Drop per-class `relationships` |
| 2 | Truncate `key_methods` to 5 |
| 3 | Drop `bases`; slim modules to name+file |
| 4 | Classes to name+file only |
| 5 | Classes and modules as flat name lists |
| 6 | Class names, components, actors, entry_points |
| 7 | Class names only |

Code chunks are also independently trimmed per diagram type to fill whatever token
budget remains after the registry. You will see log messages like:

```
[INFO] sequence: registry level 0, code_chunks shrunk 4x → 45,877 tokens
[INFO] component: registry level 1, code_chunks shrunk 3x → 45,553 tokens
```

This is normal for large repos on models with 32–48k context windows. Llama-4-Scout
and Llama-3.3 with `--max-model-len 128000` will avoid compression entirely for
most repositories.

---

## 🔨 Automatic PlantUML Repair

After generation, the script applies a chain of repair passes before validation.
These fixes are no-ops when the LLM produces clean output (as Llama typically does)
and only activate on actual errors.

| Repair | Trigger | Fix |
|--------|---------|-----|
| `repair_duplicate_aliases` | Two elements get the same `as X` alias | Renames collisions to `X2`, `X3`, etc.; rewrites all edge references |
| `repair_unquoted_multiword_edges` | `Local Machine --> MQTT` (spaces, no quotes) | Wraps bare multi-word names in quotes on edge lines |
| `repair_slash_names` | `HTTP/HTTPS` used bare | Quotes any `WORD/WORD` token not already quoted |
| `repair_truncated_activity` | Activity diagram cut off mid-action | Closes dangling `:action` with `;`; inserts missing `stop` |
| `repair_class_diagram_lines` | `class Foo <|-- Bar {` on one line | Splits into separate declaration and relationship lines |

---

## 🎨 Environment Variables

### All Backends
```bash
export RAG_FAISS_INDEX=rag/faiss.index
export RAG_FAISS_META=rag/faiss_meta.json
export RAG_TOP_K=10
```

### Ollama Specific
```bash
export OLLAMA_URL=http://localhost:11434          # local
export OLLAMA_URL=http://192.168.100.100:11434   # remote DGX
export RAG_LLM_MODEL=llama4:maverick
export RAG_EMBED_MODEL=nomic-embed-text
```

### vLLM Specific
```bash
export VLLM_MODEL=meta-llama/Llama-4-Scout-17B-16E-Instruct
export VLLM_TP=4
export VLLM_MAX_LEN=48000
export VLLM_MAX_TOKENS=2048
export VLLM_EXTRACT_TOKENS=1024
export VLLM_REGISTRY_TOKENS=2048
export VLLM_TEMPERATURE=0.0
```

---

## ✅ Validation & Testing

### Test Local vLLM Setup

```bash
python util/test_plantuml_vllm.py \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 4
```

Expected output:
```
==================================================================
Test Results Summary
==================================================================
basic                ✓ PASSED
plantuml             ✓ PASSED
json                 ✓ PASSED
==================================================================
✓ All tests passed! Ready to generate PlantUML diagrams.
```

### Validate Generated Diagrams

```bash
# Check syntax only (no rendering)
plantuml -checkonly uml_out/*.puml

# Render to PNG
plantuml uml_out/*.puml

# Render to SVG (vector, better for large diagrams)
plantuml -tsvg uml_out/*.puml
```

---

## 📊 Performance Benchmarks

### Chunked vLLM on DGX A100 (4× A100 80GB)

| Model | Repo Size | Files | Total Time |
|-------|-----------|-------|------------|
| Llama-4-Scout-17B | Small | 10–50 | 3–5 min |
| Llama-4-Scout-17B | Medium | 50–200 | 8–12 min |
| Llama-4-Scout-17B | Large | 200–500 | 14–18 min |
| Mistral-Large | Small | 10–50 | 4–6 min |
| Mistral-Large | Medium | 50–200 | 10–15 min |
| Mistral-Large | Large | 200–500 | 16–22 min |

*Times include all three passes. Pass 1 (extraction) dominates for large repos.*

### Ollama (varies by model and deployment)

**Local (llama3.1:8b on laptop):**

| Repo Size | Files | Time |
|-----------|-------|------|
| Small | 10–50 | 3–5 min |
| Medium | 50–200 | 8–12 min |
| Large | 200–500 | 15–20 min |

**Remote (llama4:maverick on DGX via HTTP):**

| Repo Size | Files | Time |
|-----------|-------|------|
| Small | 10–50 | 5–8 min |
| Medium | 50–200 | 12–18 min |
| Large | 200–500 | 20–30 min |

*Remote times include network round-trip overhead per request.*

---

## 🐛 Troubleshooting

### "CUDA out of memory" (vLLM)

```bash
# Reduce GPU memory fraction
--gpu-memory-utilization 0.75

# Reduce context window
--max-model-len 32000

# Use tensor parallelism across more GPUs
--tp 4   # instead of 2

# Use a smaller model
--model meta-llama/Llama-3-8B-Instruct
```

### "VLLMValidationError: input tokens exceed context length"

This should no longer occur — the script automatically compresses prompts to fit.
If you still see it, check that you are running the latest version of
`repo_to_diagrams_chunked_vllm.py`. You can also try increasing the context window:

```bash
--max-model-len 64000
```

### "Registry JSON parse failed — building fallback registry"

The model produced malformed JSON for Pass 2 (common at high compression levels,
e.g., level 5+). The fallback registry is built directly from the per-file
extractions and is usually sufficient. To improve this:

```bash
# Give the model more output budget for the registry response
--registry-tokens 4096

# Or increase the context window so less compression is needed
--max-model-len 64000
```

### Mistral tokenizer warnings

These are advisory-only and do not affect output:

```
MistralCommonTokenizer.apply_chat_template(..., tokenize=False) is unsafe...
`get_control_token` is deprecated. Use `get_special_token` instead.
```

To suppress them, add to the top of the script after imports:

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mistral_common")
```

### Activity diagram has no `stop` / truncated output

This is repaired automatically by `repair_truncated_activity`. If you see it
persisting after repair, the model hit `--max-tokens` before finishing. Increase:

```bash
--max-tokens 4096
```

### "Invalid PlantUML output" / Syntax errors after rendering

The repair chain handles the most common cases automatically. For persistent errors:

```bash
# More RAG examples give the model better syntax guidance
--rag-k 20

# Lower temperature ensures more deterministic output
--temperature 0.0

# Use a larger model
--model meta-llama/Llama-4-Scout-17B-16E-Instruct
```

### "Model not found" (vLLM)

```bash
# Download explicitly first
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct

# Or point to the local cache path directly
--model ~/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/...
```

### "Connection refused" (Ollama)

```bash
# Verify Ollama is running
ollama list

# Test the endpoint directly
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### "No embeddings field" (RAG building)

Make sure Ollama is running and the embedding model is available:

```bash
ollama serve
ollama pull nomic-embed-text
```

---

## 📌 Model Recommendations

| Model | Size | Context | Quality | Speed | Backend |
|-------|------|---------|---------|-------|---------|
| Llama-4-Scout-17B | 17B MoE | 128K | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | vLLM |
| Mistral-Large-2411 | 123B | 128K | ⭐⭐⭐⭐⭐ | ⭐⭐ | vLLM |
| Llama-3.3-70B | 70B | 128K | ⭐⭐⭐⭐⭐ | ⭐⭐ | vLLM |
| llama4:maverick | 17B MoE | 128K | ⭐⭐⭐⭐ | ⭐⭐⭐ | Ollama (DGX) |
| llama3.1:8b | 8B | 128K | ⭐⭐⭐ | ⭐⭐⭐⭐ | Ollama (laptop) |
| Llama-3-8B | 8B | 8K | ⭐⭐ | ⭐⭐⭐⭐ | vLLM |

**Notes:**
- Mistral-Large produces excellent diagrams but is slower and needs automatic context
  compression for large repos. The script handles this transparently.
- Llama-4-Scout is the best default: fast, large context, handles large repos
  without compression in most cases.
- For Ollama on a laptop, llama3.1:8b is the practical floor for usable output.

**Priority:** Correctness > Speed. Larger models generate more accurate PlantUML.

---

## 🔬 Advanced Usage

### GPU Selection

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python repo_to_diagrams_chunked_vllm.py --tp 4 ...

# Monitor GPU usage during generation
watch -n 1 nvidia-smi
```

### Batch Processing Multiple Repos

```bash
#!/bin/bash
for repo in ~/Code/*/; do
  name=$(basename "$repo")
  python repo_to_diagrams_chunked_vllm.py \
    --input "$repo" \
    --output "./diagrams/$name" \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --tp 4 \
    --max-model-len 48000 \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

### Regenerating Diagrams from an Existing Registry

After the first run, re-run only diagram generation (Pass 3) to try different
models, tweak RAG settings, or fix a specific diagram type without re-extracting:

```bash
python repo_to_diagrams_chunked_vllm.py \
  --input /path/to/repo \
  --output uml_out \
  --registry-file uml_out/myrepo_entity_registry.json \
  --model mistralai/Mistral-Large-Instruct-2411 \
  --tp 4 --max-model-len 48000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 📚 Documentation Files

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Command cheat sheet
- **[VLLM_GUIDE.md](VLLM_GUIDE.md)** — Complete vLLM deployment guide
- **[VLLM_LOCAL_COMPARISON.md](VLLM_LOCAL_COMPARISON.md)** — Detailed Ollama vs vLLM comparison

---

## 💬 FAQ

**Q: Which backend should I use?**
- **Chunked vLLM on DGX**: Best quality, handles any repo size, single-user GPU workstation
- **Ollama (remote DGX)**: Good quality, run scripts from laptop, share the DGX
- **Ollama (local laptop)**: Quick testing and prototyping with smaller models

**Q: My registry said "JSON parse failed — building fallback registry". Is that bad?**
The fallback registry is built from the raw extractions and works for most repos.
For better results, try `--registry-tokens 4096` or a higher `--max-model-len`.

**Q: Why does vLLM compress my prompts?**
For large repos (200+ classes), the entity registry alone can exceed 30k tokens on
Mistral. The compression ladder drops verbose fields while preserving canonical names,
which is what diagram generation actually needs. Llama-4-Scout on `--max-model-len 128000`
avoids compression entirely.

**Q: Does compression affect output quality?**
Minimally. The LLM primarily needs canonical class and component names to generate
valid PlantUML. Method lists and relationship details in the registry help with
richer diagrams but are not required for syntactic correctness.

**Q: Will Mistral-specific fixes affect Llama runs?**
No. Every repair function is a no-op if the LLM's output is already correct. The
compression loop starts at level 0 and only advances if the token count exceeds the
budget. Llama on a 128k context window will use level 0 for everything.

**Q: How do I create a corpus?**
Prepare PlantUML training data in JSONL format. See "Creating Your RAG Index" above.
Fine-tuning training data in Alpaca format with PlantUML examples works directly —
just point `--corpus` to it.

**Q: How much VRAM do I need?**
- Mistral-Large (123B): 4× A100 80GB
- Llama-3.3-70B: 4× A100 80GB (or 2× A100 80GB at lower utilization)
- Llama-4-Scout-17B MoE: 2× A100 80GB (or 1× A100 80GB at reduced utilization)
- Llama-3-8B: 1× RTX 3090/4090 (24GB)

**Q: Can I generate diagrams for non-Python code?**
Yes — extend the `collect_files()` function in the scripts to include your
language's file extensions. The extraction prompt may also need updating for
non-Python syntax.

---

## 📝 License

MIT License — use freely, no warranty provided.

---

## 🎉 Summary

**For best quality on DGX (single-user):**
1. Install dependencies (5 min)
2. Build RAG index from your corpus (5 min)
3. Generate diagrams with chunked vLLM (8–18 min depending on repo size)
4. On subsequent runs, use `--registry-file` to skip extraction (saves 5–10 min)

**For remote access to DGX models:**
1. Run Ollama on DGX (`ollama serve && ollama pull llama4:maverick`)
2. Install dependencies on laptop (2 min)
3. Build RAG index (5 min)
4. Generate diagrams via remote Ollama (12–20 min)

**For quick local testing:**
1. Install Ollama locally with a small model (2 min)
2. Build RAG index (5 min)
3. Generate diagrams (8–15 min for a typical project)

**Total time: 15–30 minutes to production-quality UML diagrams.**

---

**Questions?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands or
[VLLM_GUIDE.md](VLLM_GUIDE.md) for detailed setup.
