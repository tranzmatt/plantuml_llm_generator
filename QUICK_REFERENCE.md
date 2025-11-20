# PlantUML Generation - Quick Reference

**Fast command lookup for all deployment options**

---

## 🎯 Choose Your Approach

### Local vLLM (Python Library)
✅ Single-user, DGX/workstation, fastest
📍 Script: `repo_to_diagrams_local_vllm.py`

### vLLM API Server
✅ Multi-user, remote access, persistent model
📍 Script: `cli/repo_to_diagrams.py --backend vllm`

### Ollama
✅ Easiest setup, quick testing
📍 Script: `cli/repo_to_diagrams.py --backend ollama`

---

## 💡 Most Common Commands

### 1. Local vLLM - Basic Usage

```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/MyProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### 2. vLLM Server - Basic Usage

```bash
# Start server (once)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --port 8000

# Generate diagrams (many times)
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend vllm \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### 3. Ollama - Basic Usage

```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend ollama \
  --model llama4:maverick \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 🚀 Quick Starts by Scenario

### High Quality Output (Production)

**Local vLLM:**
```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/Project \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --max-len 65536 \
  --max-tokens 12000 \
  --temperature 0.0 \
  --rag-examples 8 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --verbose
```

**vLLM Server:**
```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/Project \
  --backend vllm \
  --model local \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Fast Mode (Testing)

**Local vLLM:**
```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/Project \
  --model meta-llama/Llama-3-8B-Instruct \
  --tp 1 \
  --max-tokens 6000 \
  --no-validate \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Ollama:**
```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/Project \
  --backend ollama \
  --model llama3:8b \
  --no-validate \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Remote Server

**Local vLLM:**
```bash
# Not applicable - local only
```

**vLLM Server:**
```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/Project \
  --backend vllm \
  --vllm-url http://172.32.1.250:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Ollama:**
```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/Project \
  --backend ollama \
  --ollama-url http://172.32.1.250:11434 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 🔧 Starting Servers

### vLLM Server - DGX A100 (4 GPUs)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
```

### vLLM Server - 2 GPUs

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 1000000 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

### Ollama Server

```bash
ollama serve
# In another terminal:
ollama pull llama4:maverick
```

---

## ⚙️ Environment Variables

### Local vLLM
```bash
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

### vLLM Server
```bash
export VLLM_URL=http://localhost:8000
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

### Ollama
```bash
export OLLAMA_URL=http://localhost:11434
export PLANTUML_LLM_MODEL=llama4:maverick
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

---

## 📊 Common Parameters

### Local vLLM Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `.` | Repository path |
| `--output` | `uml` | Output directory |
| `--model` | (required) | Model name |
| `--tp` | `2` | Number of GPUs |
| `--max-len` | `16000` | Max sequence length |
| `--max-tokens` | `8000` | Max output tokens |
| `--temperature` | `0.0` | Sampling temperature |
| `--rag-examples` | `4` | RAG examples per type |
| `--verbose` | off | Detailed output |
| `--no-validate` | off | Skip validation |

### Dual-Backend Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `.` | Repository path |
| `--output` | `uml` | Output directory |
| `--backend` | (required) | `ollama` or `vllm` |
| `--model` | (varies) | Model name |
| `--vllm-url` | `$VLLM_URL` | vLLM server URL |
| `--ollama-url` | `$OLLAMA_URL` | Ollama server URL |
| `--no-validate` | off | Skip validation |

---

## 🎚️ Quality Tuning

### By Repository Size

**Small (<100 files):**
```bash
--max-tokens 8000 --rag-examples 4 --temperature 0.1
```

**Medium (100-500 files):**
```bash
--max-tokens 12000 --rag-examples 6 --temperature 0.05
```

**Large (500-1000 files):**
```bash
--max-tokens 16000 --rag-examples 8 --temperature 0.05
```

**Very Large (>1000 files):**
```bash
--max-tokens 20000 --rag-examples 10 --temperature 0.0
```

### By Quality Goal

**Maximum Quality:**
```bash
--model openai/gpt-oss-120b --temperature 0.0 --rag-examples 10
```

**Balanced:**
```bash
--model openai/gpt-oss-120b --temperature 0.0 --rag-examples 6
```

**Fast:**
```bash
--model meta-llama/Llama-3-8B-Instruct --temperature 0.1 --rag-examples 3
```

---

## 🐛 Troubleshooting Commands

### Check vLLM Server

```bash
curl http://localhost:8000/v1/models
```

### Check Ollama Server

```bash
curl http://localhost:11434/api/tags
```

### Check GPUs

```bash
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Test Local vLLM

```bash
python test_plantuml_vllm.py --model openai/gpt-oss-120b --tp 4
```

### Validate Generated Diagrams

```bash
plantuml -checkonly uml/*.puml
```

---

## 🔥 Quick Fixes

### "CUDA out of memory"

**Local vLLM:**
```bash
--gpu-memory-utilization 0.85 --max-len 16000
```

**vLLM Server:**
```bash
# Restart server with:
--gpu-memory-utilization 0.85 --max-model-len 500000
```

### "Connection refused"

**vLLM:**
```bash
# Check server is running
ps aux | grep vllm
netstat -tuln | grep 8000
```

**Ollama:**
```bash
# Check server is running
ps aux | grep ollama
ollama list
```

### "Model not found"

```bash
# Download model
huggingface-cli download openai/gpt-oss-120b
# Or for Ollama:
ollama pull llama4:maverick
```

### "Invalid PlantUML"

```bash
# Add more RAG examples and lower temperature
--rag-examples 8 --temperature 0.0
```

### "Timeout"

```bash
# Increase timeout (local vLLM only)
--timeout 7200  # 2 hours
```

---

## 📦 Output Files

For repository `my_project`, you get:

```
uml/
├── my_project_class.puml       # Class structure
├── my_project_sequence.puml    # Interactions
├── my_project_activity.puml    # Workflows
├── my_project_state.puml       # State machines
├── my_project_component.puml   # Architecture
├── my_project_deployment.puml  # Deployment
├── my_project_usecase.puml     # Use cases
└── my_project_object.puml      # Runtime objects
```

---

## 🎨 Rendering Diagrams

### PNG (Default)
```bash
plantuml uml/*.puml
```

### SVG (Vector)
```bash
plantuml -tsvg uml/*.puml
```

### PDF
```bash
plantuml -tpdf uml/*.puml
```

### Multiple Formats
```bash
plantuml -tpng -tsvg uml/*.puml
```

---

## 🔄 Batch Processing

### Local vLLM

```bash
#!/bin/bash
for repo in ~/Code/*/; do
  python repo_to_diagrams_local_vllm.py \
    --input "$repo" \
    --output "./uml/$(basename $repo)" \
    --model openai/gpt-oss-120b \
    --tp 4 \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

### vLLM Server (Server Already Running)

```bash
#!/bin/bash
for repo in ~/Code/*/; do
  python -m cli.repo_to_diagrams \
    --input "$repo" \
    --output "./uml/$(basename $repo)" \
    --backend vllm \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

---

## 🧪 Testing & Validation

### Create RAG Index (One-Time Setup)

```bash
python create_minimal_rag.py
```

### Test Installation

```bash
# Local vLLM
python test_plantuml_vllm.py --model openai/gpt-oss-120b --tp 4

# Dual-backend
python -m cli.repo_to_diagrams --help
```

### Validate Output

```bash
# Check syntax
plantuml -checkonly uml/*.puml

# Generate and view
plantuml uml/*.puml
eog uml/*.png
```

---

## 📚 Get More Help

### Command Help

```bash
# Local vLLM
python repo_to_diagrams_local_vllm.py --help

# Dual-backend
python -m cli.repo_to_diagrams --help

# Test script
python test_plantuml_vllm.py --help
```

### Full Documentation

- **README_CONSOLIDATED.md** - Complete system docs
- **VLLM_COMPLETE_GUIDE.md** - All vLLM deployment options
- **GETTING_STARTED.md** - Step-by-step setup
- **INDEX.md** - File navigation

---

## 💾 Performance Reference

### Local vLLM (DGX A100, 4 GPUs, GPT-OSS-120B)

| Repo Size | Files | Time |
|-----------|-------|------|
| Small | 10-50 | 2-3 min |
| Medium | 50-200 | 5-8 min |
| Large | 200-500 | 10-15 min |
| Very Large | 500+ | 20-30 min |

### vLLM Server (Pre-loaded Model)

| Repo Size | Files | Time |
|-----------|-------|------|
| Small | 10-50 | 1-2 min |
| Medium | 50-200 | 3-5 min |
| Large | 200-500 | 7-12 min |
| Very Large | 500+ | 15-25 min |

---

## 🎯 Model Selection

| Model | Size | Best For | Approach |
|-------|------|----------|----------|
| GPT-OSS-120B | 120B | Production, best quality | Local or Server |
| Llama-4-Maverick-17B | 17B | Balanced quality/speed | Local or Server |
| Llama-4-Scout-17B | 17B | Fast testing | Local or Server |
| Llama-3-8B | 8B | Quick prototypes | Local |
| llama4:maverick | N/A | Easy setup | Ollama |

---

## ⚡ Quick Command Templates

### Copy-Paste Templates

**Local vLLM - Production:**
```bash
python repo_to_diagrams_local_vllm.py \
  --input YOUR_REPO_PATH \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**vLLM Server - Production:**
```bash
python -m cli.repo_to_diagrams \
  --input YOUR_REPO_PATH \
  --backend vllm \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Ollama - Quick Test:**
```bash
python -m cli.repo_to_diagrams \
  --input YOUR_REPO_PATH \
  --backend ollama \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

**Keep this handy for daily use! 🚀**
