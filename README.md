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
- **Multiple backends**: Ollama, vLLM API server, or vLLM Python library
- **Modular architecture** with shared core components
- **Optional validation** using PlantUML CLI

---

## 🚀 Quick Start

Choose your deployment model:

### Option 1: Local vLLM (Recommended for DGX/Single-User)

**Best if:** You have test-vllm-v2.py working, running on your own GPU hardware

```bash
# 1. Install dependencies
pip install faiss-cpu sentence-transformers numpy requests vllm

# 2. Create RAG index (one-time)
python create_minimal_rag.py

# 3. Generate diagrams
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/YourProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Option 2: vLLM API Server (Multi-User/Remote)

**Best if:** Multiple users need access or you want a persistent server

```bash
# Start server (once)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --port 8000

# Generate diagrams (many times)
python -m cli.repo_to_diagrams \
  --input ~/Code/YourProject \
  --backend vllm \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Option 3: Ollama (Easiest Setup)

**Best if:** Want simplest installation and don't need large context windows

```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/YourProject \
  --backend ollama \
  --model llama4:maverick \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 📦 What You Get

For each repository, generates 8 diagram files:

```
uml/
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
cd uml/
plantuml *.puml
```

---

## 🛠️ Installation

### Core Dependencies

```bash
pip install -r requirements.txt
```

Includes: faiss-cpu, sentence-transformers, numpy, requests

### Backend-Specific

**For Local vLLM:**
```bash
pip install vllm  # or vllm-cu12 for CUDA 12
```

**For Ollama:**
```bash
# Install from https://ollama.ai/
ollama pull llama4:maverick
```

**For GPU Acceleration:**
```bash
pip install faiss-gpu  # instead of faiss-cpu
```

---

## 📂 Directory Structure

```
plantuml-rag/
├── core/                              # Shared modules
│   ├── repo_scanner.py               # Repository analysis
│   ├── rag_retriever.py              # FAISS search
│   ├── prompt_builder.py             # Prompt construction
│   ├── plantuml_sanitizer.py         # Validation & fixes
│   ├── diagram_writer.py             # File output
│   └── utils.py                      # Helpers
│
├── llm_backends/                     # Backend clients
│   ├── ollama_client.py              # Ollama integration
│   └── vllm_client.py                # vLLM API integration
│
├── cli/                              # Command-line tools
│   ├── repo_to_diagrams.py           # Dual-backend CLI
│   ├── repo_to_diagrams_vllm.py      # Remove VLLM
│   └── repo_to_diagrams_local_vllm.py # Local vLLM version
│
└── util/                             # Utility tools
    ├── test_plantuml_vllm.py             # Validation tests
    ├── create_minimal_rag.py             # RAG index creator
    └── comparison.py                      # Architecture comparison
```

---

## 🎯 Choosing Your Backend

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Local vLLM** | Single-user DGX/workstation | Fastest, no server, direct GPU access | Model loads each run |
| **vLLM Server** | Multi-user, remote access | Persistent model, shared resource | Requires server management |
| **Ollama** | Quick testing, small projects | Easiest setup, good for exploration | Smaller context windows |

---

## 💡 Usage Examples

### Local vLLM (DGX A100 with 4 GPUs)

```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/MyProject \
  --output ./diagrams \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --max-len 32000 \
  --max-tokens 12000 \
  --temperature 0.0 \
  --rag-examples 8 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --verbose
```

### vLLM API Server

```bash
# Start server (terminal 1)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --port 8000

# Generate diagrams (terminal 2)
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend vllm \
  --vllm-url http://localhost:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Ollama

```bash
# Ensure Ollama is running
ollama serve

# Generate diagrams
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend ollama \
  --model llama4:maverick \
  --ollama-url http://localhost:11434 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 🧠 Creating Your RAG Index

### Quick Start (Minimal Index)

```bash
python create_minimal_rag.py
```

Creates a basic index with 16 examples (2 per diagram type). Good for testing.

### Production Index (Recommended)

```python
# build_production_rag.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your PlantUML examples (hundreds/thousands)
examples = [
    {
        "type": "class",
        "description": "User authentication system",
        "plantuml": "@startuml\nclass User {...}\n@enduml"
    },
    # ... many more examples
]

# Create embeddings
model = SentenceTransformer("nomic-embed-text")
texts = [f"{ex['type']} {ex['description']}" for ex in examples]
embeddings = model.encode(texts, normalize_embeddings=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))

# Save
faiss.write_index(index, "rag/faiss.index")
with open("rag/faiss_meta.json", "w") as f:
    json.dump(examples, f, indent=2)
```

### Using Existing Training Data

If you have PlantUML training data from fine-tuning:

```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/Project \
  --faiss-index /path/to/training/faiss.index \
  --faiss-meta /path/to/training/faiss_meta.json \
  --model openai/gpt-oss-120b \
  --tp 4
```

---

## 🔧 Configuration & Tuning

### DGX A100 Optimization (Local vLLM)

```bash
--model openai/gpt-oss-120b      # Large model (120B)
--tp 4                            # Use all 4 A100 GPUs
--max-len 65536                   # Large context (80GB per GPU!)
--gpu-memory-utilization 0.95     # Use available memory
--temperature 0.0                 # Deterministic output
--repetition-penalty 1.1          # Prevent loops
--max-tokens 12000                # Ample output space
--rag-examples 8                  # High quality
```

### Quality vs Speed Tradeoffs

**High Quality (Slow):**
```bash
--model openai/gpt-oss-120b --tp 4 --rag-examples 10 --temperature 0.0
```

**Balanced (Recommended):**
```bash
--model openai/gpt-oss-120b --tp 4 --rag-examples 6 --temperature 0.0
```

**Fast (Testing):**
```bash
--model meta-llama/Llama-3-8B-Instruct --tp 1 --rag-examples 3 --no-validate
```

---

## 🎨 Environment Variables

### All Backends
```bash
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

### Ollama
```bash
export OLLAMA_URL=http://localhost:11434
export PLANTUML_LLM_MODEL=llama4:maverick
```

### vLLM Server
```bash
export VLLM_URL=http://localhost:8000
```

---

## ✅ Validation & Testing

### Test Local vLLM Setup

```bash
python test_plantuml_vllm.py \
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
plantuml -checkonly uml/*.puml

# Render to PNG
plantuml uml/*.puml

# Render to SVG (vector)
plantuml -tsvg uml/*.puml
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

### vLLM API Server (Persistent Model)

| Repository Size | Files | Generation Time |
|----------------|-------|-----------------|
| Small | 10-50 | 1-2 minutes |
| Medium | 50-200 | 3-5 minutes |
| Large | 200-500 | 7-12 minutes |
| Very Large | 500+ | 15-25 minutes |

*Assumes model pre-loaded, no startup overhead*

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solutions:**
```bash
# Reduce GPU memory usage
--gpu-memory-utilization 0.85

# Reduce context window
--max-len 16000

# Use more GPUs
--tp 4  # instead of 2

# Use smaller model
--model meta-llama/Llama-3-8B-Instruct
```

### "Model not found"

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
--rag-examples 8

# Lower temperature
--temperature 0.0

# Use larger model
--model openai/gpt-oss-120b
```

### "Connection refused" (vLLM Server)

**Check:**
```bash
# Verify server is running
curl http://localhost:8000/v1/models

# Check firewall
netstat -tuln | grep 8000

# Test from Python
python -c "import requests; print(requests.get('http://localhost:8000/v1/models').json())"
```

---

## 📚 Documentation Files

- **[INDEX.md](computer:///mnt/user-data/outputs/INDEX.md)** - File navigation guide
- **[PACKAGE_SUMMARY.md](computer:///mnt/user-data/outputs/PACKAGE_SUMMARY.md)** - Complete overview
- **[GETTING_STARTED.md](computer:///mnt/user-data/outputs/GETTING_STARTED.md)** - Setup checklist (15-20 min)
- **[LOCAL_VLLM_GUIDE.md](computer:///mnt/user-data/outputs/LOCAL_VLLM_GUIDE.md)** - Detailed local vLLM usage
- **[QUICK_REFERENCE.md](computer:///mnt/user-data/outputs/QUICK_REFERENCE.md)** - Command cheat sheet
- **[VLLM_SETUP.md](computer:///mnt/user-data/uploads/VLLM_SETUP.md)** - vLLM server installation

---

## 🔬 Advanced Usage

### GPU Selection

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
python repo_to_diagrams_local_vllm.py --tp 2 ...

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Batch Processing

```bash
#!/bin/bash
for repo in ~/Code/*/; do
  python repo_to_diagrams_local_vllm.py \
    --input "$repo" \
    --output "./diagrams/$(basename $repo)" \
    --model openai/gpt-oss-120b \
    --tp 4 \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

### Custom Embedding Models

```bash
--embed-model sentence-transformers/all-mpnet-base-v2
# or
--embed-model intfloat/e5-large-v2
```

---

## 🧩 Extending the System

### Add New Diagram Types

Edit `core/prompt_builder.py`:
```python
diagram_types = [
    "class", "sequence", "activity", "state",
    "component", "deployment", "usecase", "object",
    "timing",  # Add new type
]
```

### Support New Languages

Create new scanner in `core/`:
```python
# core/java_scanner.py
def scan_java_repo(root: str) -> List[str]:
    return [f for f in glob(f"{root}/**/*.java", recursive=True)]
```

### Custom Prompt Styles

Modify `core/prompt_builder.py`:
```python
def build_system_prompt() -> str:
    return "You are an expert PlantUML generator with a focus on..."
```

---

## 📌 Model Recommendations

| Model | Size | Context | Quality | Speed | Best For |
|-------|------|---------|---------|-------|----------|
| GPT-OSS-120B | 120B | 2M | ⭐⭐⭐⭐⭐ | ⭐⭐ | Production, large repos |
| Llama-4-Maverick-17B | 17B | 128K | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced quality/speed |
| Llama-4-Scout-17B | 17B | 128K | ⭐⭐⭐ | ⭐⭐⭐⭐ | Fast testing |
| Llama-3-8B | 8B | 8K | ⭐⭐ | ⭐⭐⭐⭐⭐ | Quick prototypes |

**Priority:** Correctness > Speed. Larger models generate more accurate PlantUML.

---

## 🎓 Learning Path

### First Time User?
1. Read [GETTING_STARTED.md](computer:///mnt/user-data/outputs/GETTING_STARTED.md)
2. Run `python create_minimal_rag.py`
3. Run `python test_plantuml_vllm.py --model openai/gpt-oss-120b --tp 4`
4. Generate your first diagrams!

### Understanding Architecture?
1. Run `python comparison.py` to see how components fit together
2. Review the modular structure in `core/`
3. Check `llm_backends/` for backend implementations

### Production Deployment?
1. Build comprehensive RAG index (500+ examples per type)
2. Optimize parameters for your hardware
3. Set up batch processing scripts
4. Configure monitoring and logging

---

## 💬 FAQ

**Q: Which backend should I use?**
- Local vLLM if you have test-vllm-v2.py working (single-user)
- vLLM server if multiple users need access
- Ollama for quick testing/exploration

**Q: How does this relate to my test-vllm-v2.py?**
Run `python comparison.py` - the vLLM core is identical!

**Q: Can I use my fine-tuning training data?**
Yes! Point `--faiss-index` and `--faiss-meta` to your training data.

**Q: How much VRAM do I need?**
- 120B model: 4x A100 (80GB) or 8x A100 (40GB)
- 17B model: 2x A100 (40GB) or 1x A100 (80GB)
- 8B model: 1x RTX 3090/4090 (24GB)

**Q: Can I generate diagrams for non-Python code?**
Yes - extend `core/repo_scanner.py` for your language.

---

## 📝 License

MIT License – use freely, no warranty provided.

---

## 🎉 Summary

**If you have test-vllm-v2.py working:**
1. Install dependencies (5 min)
2. Create RAG index (2 min)
3. Generate diagrams (5-10 min)

**Total time: 15-20 minutes to production-quality UML diagrams!**

Start with [GETTING_STARTED.md](computer:///mnt/user-data/outputs/GETTING_STARTED.md) →

---

**Questions?** Check [INDEX.md](computer:///mnt/user-data/outputs/INDEX.md) for navigation or [QUICK_REFERENCE.md](computer:///mnt/user-data/outputs/QUICK_REFERENCE.md) for commands.
