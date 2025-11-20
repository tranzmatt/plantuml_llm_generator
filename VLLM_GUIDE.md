# Complete vLLM Guide: All Deployment Options

**Comprehensive guide for using vLLM with PlantUML generation - API Server, Local Library, and Everything In Between**

---

## 📋 Overview

This guide covers **all vLLM deployment options** for PlantUML diagram generation:

1. **Local vLLM (Python Library)** - Direct GPU access, no server
2. **vLLM API Server** - OpenAI-compatible server for multi-user access
3. **Original Dual-Backend** - Flexibility to switch between Ollama and vLLM

---

## 🎯 Quick Decision Guide

### Choose Local vLLM Library If:
- ✅ Running on your own DGX/workstation
- ✅ Single-user environment
- ✅ Development and testing
- ✅ Don't want to manage servers
- ✅ Your test-vllm-v2.py already works
- ✅ **Fastest option** (no HTTP overhead)

### Choose vLLM API Server If:
- ✅ Multiple users need access
- ✅ Remote inference required
- ✅ Want persistent model (24/7)
- ✅ Multiple clients/applications
- ✅ Production deployment
- ✅ **Most flexible** (any HTTP client)

### Choose Original Dual-Backend If:
- ✅ Need to compare Ollama vs vLLM
- ✅ Want backend flexibility
- ✅ Testing different models
- ✅ Migrating from Ollama to vLLM

---

## 🚀 Quick Start by Approach

### Approach 1: Local vLLM (Recommended for DGX)

```bash
# Install
pip install vllm faiss-cpu sentence-transformers

# Create RAG index (one-time)
python create_minimal_rag.py

# Generate diagrams
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/YourProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Time: 15-20 minutes to first diagrams**

### Approach 2: vLLM API Server

```bash
# Terminal 1: Start server (once)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --port 8000

# Terminal 2: Generate diagrams (many times)
python -m cli.repo_to_diagrams \
  --input ~/Code/YourProject \
  --backend vllm \
  --vllm-url http://localhost:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

**Time: 2-3 minutes per repo (after server start)**

### Approach 3: Original Dual-Backend

```bash
# With vLLM server
python -m cli.repo_to_diagrams \
  --input ~/Code/YourProject \
  --backend vllm \
  --vllm-url http://localhost:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# Or with Ollama
python -m cli.repo_to_diagrams \
  --input ~/Code/YourProject \
  --backend ollama \
  --model llama4:maverick \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## 📊 Comparison Matrix

| Feature | Local vLLM | vLLM Server | Dual-Backend |
|---------|------------|-------------|--------------|
| **Startup** | Model loads each run | One-time load | Depends on backend |
| **Speed** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Variable |
| **Multi-user** | ❌ No | ✅ Yes | ❌ No (Ollama), ✅ Yes (vLLM) |
| **Complexity** | ⭐⭐⭐⭐⭐ Simplest | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate |
| **Flexibility** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Highest |
| **Best For** | Single-user dev | Production/Multi-user | Testing/Comparison |

---

## Part 1: Local vLLM (Python Library)

### Installation

```bash
# Install vLLM
pip install vllm  # or vllm-cu12 for CUDA 12

# Install other dependencies
pip install faiss-cpu sentence-transformers numpy requests

# Verify
python -c "from vllm import LLM, SamplingParams; print('vLLM OK')"
```

### System Requirements

- **GPU**: NVIDIA with Compute Capability 7.0+ (A100, RTX 3090/4090, etc.)
- **VRAM**: 
  - 120B models: 4x A100 (80GB) or 8x A100 (40GB)
  - 17B models: 2x A100 (40GB) or 1x A100 (80GB)
  - 8B models: 1x RTX 3090/4090 (24GB)
- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Python**: 3.9-3.12
- **CUDA**: 11.8 or 12.x

### Basic Usage

```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/MyProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --max-len 32000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Command-Line Arguments

**Required:**
```bash
--input PATH              # Repository to analyze
--model MODEL_NAME        # HuggingFace model name
--faiss-index PATH        # FAISS index file
--faiss-meta PATH         # FAISS metadata JSON
```

**GPU Configuration:**
```bash
--tp N                    # Number of GPUs (default: 2)
--max-len N               # Max sequence length (default: 16000)
--gpu-memory-utilization F # GPU memory fraction (default: 0.95)
```

**Generation:**
```bash
--max-tokens N            # Max output tokens (default: 8000)
--temperature F           # Sampling temperature (default: 0.0)
--top-p F                 # Top-p sampling (default: 1.0)
--repetition-penalty F    # Prevent loops (default: 1.1)
```

**Other:**
```bash
--output DIR              # Output directory (default: ./uml)
--rag-examples N          # RAG examples per type (default: 4)
--no-validate             # Skip PlantUML validation
--verbose, -v             # Detailed output
```

### DGX A100 Optimization

```bash
python repo_to_diagrams_local_vllm.py \
  --input ~/Code/LargeProject \
  --model openai/gpt-oss-120b \
  --tp 4 \
  --max-len 65536 \
  --gpu-memory-utilization 0.95 \
  --max-tokens 12000 \
  --temperature 0.0 \
  --rag-examples 8 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --verbose
```

### GPU Selection

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
python repo_to_diagrams_local_vllm.py --tp 2 ...

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Performance (Local vLLM on DGX A100, 4 GPUs)

| Repository Size | Files | Time |
|----------------|-------|------|
| Small | 10-50 | 2-3 minutes |
| Medium | 50-200 | 5-8 minutes |
| Large | 200-500 | 10-15 minutes |
| Very Large | 500+ | 20-30 minutes |

---

## Part 2: vLLM API Server

### Installation

```bash
# Install vLLM
pip install vllm-cu12  # for CUDA 12
# or
pip install vllm       # for CUDA 11.8

# Verify
python -c "import vllm; print('vLLM installed OK')"
```

### Download Models

```bash
# Using HuggingFace CLI
huggingface-cli download openai/gpt-oss-120b --local-dir /models/gpt-oss-120b

# Or let vLLM download automatically
python -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b
```

### Start Server

#### Basic Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --port 8000 \
  --max-model-len 2000000
```

#### Production Server (DGX A100)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --served-model-name local \
  --disable-log-requests
```

#### Key Server Parameters

| Flag | Purpose | Recommended Value |
|------|---------|-------------------|
| `--tensor-parallel-size` | Number of GPUs | 4 (for 4 GPUs) |
| `--max-model-len` | Context window | 2000000 (2M tokens) |
| `--gpu-memory-utilization` | Prevent OOMs | 0.90 |
| `--host` | Server address | 0.0.0.0 (allow remote) |
| `--port` | Server port | 8000 (default) |
| `--trust-remote-code` | For some models | Add if needed |

### Multi-GPU Configuration

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start server
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 2000000
```

### Using the Server

#### From PlantUML Generator

```bash
# Using original dual-backend script
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend vllm \
  --vllm-url http://localhost:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

#### From Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Generate PlantUML..."}],
    max_tokens=8000,
    temperature=0.0
)

print(response.choices[0].message.content)
```

#### From curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1000,
    "temperature": 0.0
  }'
```

### Remote Server Setup

```bash
# On remote server
ssh gpu-server
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8000

# From workstation
export VLLM_URL=http://gpu-server:8000
python -m cli.repo_to_diagrams \
  --backend vllm \
  --vllm-url $VLLM_URL \
  --input ~/Code/Project \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Docker Deployment

```bash
# Pull image
docker pull vllm/vllm-openai:latest

# Run server
docker run --gpus all -p 8000:8000 \
  -v /models:/models \
  vllm/vllm-openai:latest \
  --model /models/gpt-oss-120b \
  --tensor-parallel-size 4
```

### Performance (API Server with Persistent Model)

| Repository Size | Files | Time |
|----------------|-------|------|
| Small | 10-50 | 1-2 minutes |
| Medium | 50-200 | 3-5 minutes |
| Large | 200-500 | 7-12 minutes |
| Very Large | 500+ | 15-25 minutes |

*Assumes model is already loaded (no startup time)*

---

## Part 3: Original Dual-Backend System

### When to Use

- Need flexibility to switch backends
- Comparing Ollama vs vLLM performance
- Migrating from Ollama to vLLM
- Testing different model sizes

### Usage with vLLM Server

```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend vllm \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --vllm-url http://localhost:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Usage with Ollama

```bash
python -m cli.repo_to_diagrams \
  --input ~/Code/MyProject \
  --backend ollama \
  --model llama4:maverick \
  --ollama-url http://localhost:11434 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Environment Variables

```bash
# For vLLM
export VLLM_URL=http://localhost:8000

# For Ollama
export OLLAMA_URL=http://localhost:11434
export PLANTUML_LLM_MODEL=llama4:maverick

# For both
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

---

## 🔧 Configuration & Tuning

### Quality Optimization

#### High Quality (Slow)

**Local vLLM:**
```bash
--model openai/gpt-oss-120b --tp 4 --temperature 0.0 \
--repetition-penalty 1.15 --max-tokens 16000 --rag-examples 10
```

**API Server:**
```bash
--backend vllm --model local --max-tokens 16000 \
--rag-examples-per-type 10 --temperature 0.0
```

#### Balanced (Recommended)

**Local vLLM:**
```bash
--model openai/gpt-oss-120b --tp 4 --temperature 0.0 \
--max-tokens 8000 --rag-examples 6
```

**API Server:**
```bash
--backend vllm --model local --max-tokens 8000 \
--rag-examples-per-type 6 --temperature 0.0
```

#### Fast (Testing)

**Local vLLM:**
```bash
--model meta-llama/Llama-3-8B-Instruct --tp 1 \
--temperature 0.1 --max-tokens 6000 --no-validate
```

### Model Recommendations

| Model | Size | Context | Best For | Approach |
|-------|------|---------|----------|----------|
| GPT-OSS-120B | 120B | 2M | Production | Local or Server |
| Llama-4-Maverick-17B | 17B | 128K | Balanced | Local or Server |
| Llama-4-Scout-17B | 17B | 128K | Fast testing | Local or Server |
| Llama-3-8B | 8B | 8K | Quick prototypes | Local |

---

## 🐛 Troubleshooting

### Problem: CUDA Out of Memory

**For Local vLLM:**
```bash
# Reduce GPU memory
--gpu-memory-utilization 0.85

# Reduce context
--max-len 16000

# Use more GPUs
--tp 4
```

**For API Server:**
```bash
# Restart server with lower memory
--gpu-memory-utilization 0.85
--max-model-len 500000
```

### Problem: Model Not Found

**Solution:**
```bash
# Download explicitly
huggingface-cli download openai/gpt-oss-120b

# Or use cached path
--model ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/...
```

### Problem: Connection Refused (API Server)

**Check server:**
```bash
# Test server
curl http://localhost:8000/v1/models

# Check if running
ps aux | grep vllm

# Check port
netstat -tuln | grep 8000
```

### Problem: Slow Generation

**For Local vLLM:**
```bash
# Verify tensor parallelism
--tp 4  # Must match GPU count

# Check GPU visibility
python -c "import torch; print(torch.cuda.device_count())"

# Monitor usage
nvidia-smi
```

**For API Server:**
```bash
# Check server logs for bottlenecks
# May need to increase --max-num-seqs
# Or reduce --gpu-memory-utilization
```

### Problem: Invalid PlantUML

**Solutions (All Approaches):**
```bash
# More RAG examples
--rag-examples 8

# Lower temperature
--temperature 0.0

# Use larger model
--model openai/gpt-oss-120b

# Enable validation (catches errors early)
# Remove --no-validate flag
```

---

## 📈 Advanced Usage

### Batch Processing

**Local vLLM:**
```bash
#!/bin/bash
MODEL="openai/gpt-oss-120b"
for repo in ~/Code/*/; do
  python repo_to_diagrams_local_vllm.py \
    --input "$repo" \
    --model "$MODEL" --tp 4 \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

**API Server:**
```bash
#!/bin/bash
# Server already running
for repo in ~/Code/*/; do
  python -m cli.repo_to_diagrams \
    --input "$repo" --backend vllm \
    --faiss-index rag/faiss.index \
    --faiss-meta rag/faiss_meta.json
done
```

### Custom Model Sizes

**Small Models (7B-13B):**
```bash
--model meta-llama/Llama-3-8B-Instruct --tp 1
```

**Medium Models (13B-30B):**
```bash
--model meta-llama/Llama-4-Maverick-17B-128E-Instruct --tp 2
```

**Large Models (70B-120B):**
```bash
--model openai/gpt-oss-120b --tp 4
```

### Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# vLLM metrics (if server has metrics enabled)
curl http://localhost:8000/metrics

# Process monitoring
htop
```

---

## 🎓 Migration Guides

### From Ollama to vLLM Server

1. **Start vLLM server:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 4 --port 8000
```

2. **Update commands:**
```bash
# Old (Ollama)
--backend ollama --model llama4:maverick

# New (vLLM)
--backend vllm --vllm-url http://localhost:8000
```

### From API Server to Local vLLM

```bash
# Old (API Server)
python -m cli.repo_to_diagrams \
  --backend vllm --vllm-url http://localhost:8000

# New (Local Library)
python repo_to_diagrams_local_vllm.py \
  --model openai/gpt-oss-120b --tp 4
```

**Benefits:**
- ✅ No server management
- ✅ Faster (no HTTP overhead)
- ✅ Simpler deployment

**Tradeoffs:**
- ❌ Model loads each run
- ❌ Single-user only

### From Dual-Backend to Local vLLM

If you're only using vLLM backend:

```bash
# Old
python -m cli.repo_to_diagrams --backend vllm

# New
python repo_to_diagrams_local_vllm.py
```

---

## 📚 Summary

### Use Local vLLM If:
- Single-user DGX/workstation
- Development environment
- Don't want server overhead
- Want absolute fastest performance
- ⭐ **Recommended for your setup!**

### Use vLLM API Server If:
- Multiple users/clients
- Remote access needed
- 24/7 availability required
- Multiple applications sharing model
- ⭐ **Recommended for production!**

### Use Dual-Backend If:
- Need Ollama flexibility
- Comparing different backends
- Migration in progress
- Testing scenarios

---

## 🔗 Additional Resources

- **[GETTING_STARTED.md](computer:///mnt/user-data/outputs/GETTING_STARTED.md)** - Quick setup guide
- **[QUICK_REFERENCE.md](computer:///mnt/user-data/outputs/QUICK_REFERENCE.md)** - Command cheat sheet
- **[README_CONSOLIDATED.md](computer:///mnt/user-data/outputs/README_CONSOLIDATED.md)** - Main documentation
- **vLLM Documentation**: https://docs.vllm.ai/
- **PlantUML Documentation**: https://plantuml.com/

---

## ✅ Quick Validation

### Test Local vLLM
```bash
python test_plantuml_vllm.py --model openai/gpt-oss-120b --tp 4
```

### Test API Server
```bash
curl http://localhost:8000/v1/models
```

### Test Dual-Backend
```bash
python -m cli.repo_to_diagrams --help
```

---

**Choose your approach based on your deployment needs. All paths lead to high-quality PlantUML diagrams!**
