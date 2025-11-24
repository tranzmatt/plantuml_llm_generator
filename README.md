# PlantUML-RAG: Automated UML Diagram Generation

**Generate complete, consistent UML diagrams from any code repository using RAG + LLMs.**

This toolkit parses a code repository, retrieves high-quality PlantUML examples via FAISS-based RAG, and generates **all 8 UML diagram types in one LLM call** to guarantee naming consistency and correctness.

---

## ✨ Features

- **Single-pass generation** of 8 UML diagram types:
  - Class, Sequence, Activity, State
  - Component, Deployment, Use-case, Object diagrams
- **Consistent naming** across all diagrams
- **FAISS RAG** for syntactically correct PlantUML
- **Two deployment options**: Local vLLM or Ollama
- **Simple, standalone scripts** - no complex frameworks
- **Optional validation** using PlantUML CLI

---

## 🚀 Quick Start

Choose your deployment model:

### Option 1: Local vLLM (Recommended for DGX/Single-User)

**Best if:** You have GPUs and want maximum quality and performance

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install vllm  # Works with CUDA 11.8 and 12.x

# 2. Create RAG index (one-time, requires Ollama for embeddings)
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# 3. Generate diagrams
python repo_to_diagrams_vllm_local.py \
  --input ~/Code/YourProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Option 2: Ollama (Easiest Setup)

**Best if:** Want simplest installation and quick testing

**Note:** Ollama can run locally (laptop, smaller models) or remotely (DGX, large models like llama4:maverick)

```bash
# 1. Install Ollama and pull a model
# Visit https://ollama.ai for installation
# For local/laptop use (smaller models):
ollama pull llama3.1:8b

# OR for remote DGX use (larger models):
# On the DGX: ollama serve
# Then: ollama pull llama4:maverick

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create RAG index (one-time, requires Ollama running)
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

For each repository, generates 8 diagram files:

```
/path/to/plantum/diagrams/
├── project_class.puml       # Class structure & relationships
├── project_sequence.puml    # Interaction flows over time
├── project_activity.puml    # Process workflows
├── project_state.puml       # State machine transitions
├── project_component.puml   # High-level architecture
├── project_deployment.puml  # Deployment view
├── project_usecase.puml     # User-facing use cases
└── project_object.puml      # Runtime object instances
```

Render them:
```bash
cd /path/to/plantum/diagrams/
plantuml *.puml
```

---

## 🛠️ Installation

### Core Dependencies

```bash
pip install -r requirements.txt
```

This installs: `faiss-cpu`, `sentence-transformers`, `numpy`, `requests`

### Backend-Specific

**For Local vLLM:**
```bash
pip install vllm  # Works with CUDA 11.8 and 12.x
```

**For Ollama:**
```bash
# Install from https://ollama.ai/
# Then pull your desired model, for example:
ollama pull llama3.1:8b        # Smaller model, runs on laptop
ollama pull llama4:maverick    # Larger model, needs more resources
```

---

## 📂 Project Structure

```
plantuml-rag/
├── repo_to_diagrams_vllm_local.py    # Main script: Local vLLM
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

## 🎯 Choosing Your Backend

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Local vLLM** | Single-user DGX/workstation | Fastest, best quality, direct GPU access | Model loads each run, requires GPU |
| **Ollama** | Quick testing, flexible deployment | Easiest setup, can run locally or remotely, good model selection | Persistent server needed, HTTP overhead |

**Ollama Deployment Options:**
- **Local** (laptop/workstation): Smaller models (llama3.1:8b, llama3.2:3b)
- **Remote** (DGX): Larger models (llama4:maverick, qwen2.5:72b)

---

## 💡 Usage Examples

### Local vLLM (DGX A100 with 4 GPUs)

```bash
python repo_to_diagrams_vllm_local.py \
  --input /path/to/code/repo \
  --output /path/to/plantuml/diagrams \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --max-model-len 32000 \
  --max-tokens 12000 \
  --temperature 0.0 \
  --rag-k 20 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Ollama

**Local Ollama (laptop/workstation):**
```bash
# Start Ollama locally
ollama serve

# Generate diagrams with local model
python repo_to_diagrams_ollama.py \
  --input /path/to/code/repo \
  --output /path/to/plantuml/diagrams \
  --llm-model llama3.1:8b \
  --rag-k 10 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Remote Ollama (e.g., DGX):**
```bash
# Ollama running on remote server (e.g., DGX at 192.168.100.100)
# No local Ollama server needed

# Generate diagrams using remote model
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

The corpus should be saved as `rag/plantuml_rag_corpus.jsonl`.

### Build the Index

**Important:** RAG index building requires Ollama to be running (for embeddings), regardless of which backend you'll use for diagram generation.

```bash
# Start Ollama (local or remote)
ollama serve

# Pull the embedding model if not already available
ollama pull nomic-embed-text

# Build FAISS index
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --embed-model nomic-embed-text \
  --ollama-url http://localhost:11434
```

For remote Ollama:
```bash
python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --embed-model nomic-embed-text \
  --ollama-url http://192.168.100.100:11434
```

**Environment variables alternative:**
```bash
export RAG_CORPUS=rag/plantuml_rag_corpus.jsonl
export RAG_FAISS_INDEX=rag/faiss.index
export RAG_FAISS_META=rag/faiss_meta.json
export RAG_EMBED_MODEL=nomic-embed-text
export OLLAMA_URL=http://localhost:11434

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

### DGX A100 Optimization (Local vLLM)

```bash
--model openai/gpt-oss-120b      # Large model (120B)
--tp 4                            # Use all 4 A100 GPUs
--max-model-len 32000             # Large context
--gpu-memory-utilization 0.85     # Safe memory usage
--temperature 0.0                 # Deterministic output
--max-tokens 12000                # Ample output space
--rag-k 20                        # High quality RAG examples
```

### Quality vs Speed Tradeoffs

**High Quality (Slow):**
```bash
--model openai/gpt-oss-120b --tp 4 --rag-k 20 --temperature 0.0
```

**Balanced (Recommended):**
```bash
--model openai/gpt-oss-120b --tp 4 --rag-k 10 --temperature 0.0
```

**Fast (Testing):**
```bash
--model meta-llama/Llama-3-8B-Instruct --tp 1 --rag-k 5 --temperature 0.1
```

---

## 🎨 Environment Variables

### All Backends
```bash
export RAG_FAISS_INDEX=rag/faiss.index
export RAG_FAISS_META=rag/faiss_meta.json
export RAG_TOP_K=20
```

### Ollama Specific
```bash
# For local Ollama:
export OLLAMA_URL=http://localhost:11434
export RAG_LLM_MODEL=llama3.1:8b
export RAG_EMBED_MODEL=nomic-embed-text

# OR for remote Ollama (e.g., DGX):
export OLLAMA_URL=http://192.168.100.100:11434
export RAG_LLM_MODEL=llama4:maverick
export RAG_EMBED_MODEL=nomic-embed-text
```

### vLLM Specific
```bash
export VLLM_MODEL=openai/gpt-oss-120b
export VLLM_TP=4
export VLLM_MAX_LEN=32000
export VLLM_MAX_TOKENS=8000
export VLLM_TEMPERATURE=0.0
```

---

## ✅ Validation & Testing

### Test Local vLLM Setup

```bash
python util/test_plantuml_vllm.py \
  --model openai/gpt-oss-120b \
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
# Check syntax
plantuml -checkonly uml_out/*.puml

# Render to PNG
plantuml uml_out/*.puml

# Render to SVG (vector)
plantuml -tsvg uml_out/*.puml
```

---

## 📊 Performance Benchmarks

### Local vLLM on DGX A100 (4 GPUs, GPT-OSS-120B)

| Repository Size | Files | Generation Time |
|----------------|-------|-----------------|
| Small | 10-50 | 2-3 minutes |
| Medium | 50-200 | 5-8 minutes |
| Large | 200-500 | 10-15 minutes |
| Very Large | 500+ | 20-30 minutes |

### Ollama (varies by model and deployment)

**Local Ollama (llama3.1:8b on laptop):**
| Repository Size | Files | Generation Time |
|----------------|-------|-----------------|
| Small | 10-50 | 3-5 minutes |
| Medium | 50-200 | 8-12 minutes |
| Large | 200-500 | 15-20 minutes |
| Very Large | 500+ | 30-40 minutes |

**Remote Ollama (llama4:maverick on DGX):**
| Repository Size | Files | Generation Time |
|----------------|-------|-----------------|
| Small | 10-50 | 5-8 minutes |
| Medium | 50-200 | 12-18 minutes |
| Large | 200-500 | 20-30 minutes |
| Very Large | 500+ | 40-60 minutes |

*Note: Times include network overhead for remote Ollama*

---

## 🐛 Troubleshooting

### "CUDA out of memory" (vLLM)

**Solutions:**
```bash
# Reduce GPU memory usage
--gpu-memory-utilization 0.85

# Reduce context window
--max-model-len 16000

# Use more GPUs
--tp 4  # instead of 2

# Use smaller model
--model meta-llama/Llama-3-8B-Instruct
```

### "Model not found" (vLLM)

**Solutions:**
```bash
# Download explicitly
huggingface-cli download openai/gpt-oss-120b

# Or use cached path
--model ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/...
```

### "Invalid PlantUML output"

**Solutions:**
```bash
# More RAG examples
--rag-k 20

# Lower temperature
--temperature 0.0

# Use larger model
--model openai/gpt-oss-120b
```

### "Connection refused" (Ollama)

**Check:**
```bash
# Verify Ollama is running
ollama list

# Test endpoint
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### "No embeddings field" (RAG building)

**Solution:**
Make sure Ollama is running and the embedding model is pulled:
```bash
ollama serve
ollama pull nomic-embed-text
```

---

## 📚 Documentation Files

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet
- **[VLLM_GUIDE.md](VLLM_GUIDE.md)** - Complete vLLM deployment guide
- **[VLLM_LOCAL_COMPARISON.md](VLLM_LOCAL_COMPARISON.md)** - Ollama vs vLLM comparison
- **[README.txt](README.txt)** - Quick command examples

---

## 🔬 Advanced Usage

### GPU Selection

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
python repo_to_diagrams_vllm_local.py --tp 2 ...

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Batch Processing

```bash
#!/bin/bash
for repo in ~/Code/*/; do
  python repo_to_diagrams_vllm_local.py \
    --input "$repo" \
    --output "./diagrams/$(basename $repo)" \
    --model openai/gpt-oss-120b \
    --tp 4 \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

### Remote Ollama Server

Use Ollama running on a remote server (e.g., DGX) for larger models:

```bash
# Point to remote Ollama server
python repo_to_diagrams_ollama.py \
  --input ~/Code/Project \
  --ollama-url http://192.168.100.100:11434 \
  --llm-model llama4:maverick \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Common setup:**
- **DGX/Server**: Runs Ollama with large models (llama4:maverick, qwen2.5:72b)
- **Laptop/Workstation**: Runs the Python scripts, connects to remote Ollama

---

## 📌 Model Recommendations

| Model | Size | Context | Quality | Speed | Best For |
|-------|------|---------|---------|-------|----------|
| GPT-OSS-120B | 120B | 2M | ⭐⭐⭐⭐⭐ | ⭐⭐ | Production, large repos (vLLM only) |
| Llama-4-Maverick-17B | 17B | 128K | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced quality/speed (vLLM only) |
| llama4:maverick | N/A | 128K | ⭐⭐⭐⭐ | ⭐⭐⭐ | Large repos (Ollama, needs DGX/server) |
| llama3.1:8b | 8B | 128K | ⭐⭐⭐ | ⭐⭐⭐⭐ | Quick testing (Ollama, runs locally) |
| Llama-3-8B | 8B | 8K | ⭐⭐ | ⭐⭐⭐⭐ | Quick prototypes (vLLM only) |

**Ollama Model Selection:**
- **Local (laptop)**: llama3.1:8b, llama3.2:3b, qwen2.5:7b
- **Remote (DGX/server)**: llama4:maverick, qwen2.5:72b, command-r-plus

**Priority:** Correctness > Speed. Larger models generate more accurate PlantUML.

---

## 💬 FAQ

**Q: Which backend should I use?**
- **Local vLLM**: Best quality and speed on DGX/GPU workstation for single-user use
- **Ollama (local)**: Quick testing with smaller models on laptop/workstation
- **Ollama (remote)**: Access large models on DGX from any machine, easiest setup

**Q: How do I create a corpus?**
You need to prepare your own PlantUML training data in JSONL format. See the "Creating Your RAG Index" section.

**Q: Can I use my fine-tuning training data?**
Yes! If you have training data in the Alpaca format with PlantUML examples, just point `--corpus` to it.

**Q: How much VRAM do I need?**
- 120B model: 4x A100 (80GB) or 8x A100 (40GB)
- 17B model: 2x A100 (40GB) or 1x A100 (80GB)
- 8B model: 1x RTX 3090/4090 (24GB)

**Q: Can I generate diagrams for non-Python code?**
Yes - extend the `walk_repo_collect_code()` function in the scripts to include your language's file extensions.

**Q: Why does vLLM use V1 engine?**
The V1 engine is more stable for large models (70B+) on multi-GPU setups. The script automatically configures this.

---

## 📝 License

MIT License – use freely, no warranty provided.

---

## 🎉 Summary

**For best quality on DGX (single-user):**
1. Install dependencies (5 min)
2. Build RAG index from your corpus (5 min)
3. Generate diagrams with local vLLM (5-15 min)

**For remote access to DGX models:**
1. Run Ollama on DGX (e.g., llama4:maverick)
2. Install dependencies on laptop (2 min)
3. Build RAG index (5 min)
4. Generate diagrams via remote Ollama (10-20 min)

**For quick local testing:**
1. Install Ollama locally with small model (2 min)
2. Build RAG index (5 min)
3. Generate diagrams (10-15 min)

**Total time: 15-30 minutes to production-quality UML diagrams!**

---

**Questions?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands or [VLLM_GUIDE.md](VLLM_GUIDE.md) for detailed setup.
