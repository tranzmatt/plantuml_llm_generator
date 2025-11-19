# vLLM Version Quick Reference

## Most Common Usage Patterns

### 1. Basic Local Usage
```bash
python repo_to_diagrams_vllm.py \
  --input ./my_repo \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```
**Assumes**: vLLM running on localhost:8000 with default model

---

### 2. Remote vLLM Server
```bash
python repo_to_diagrams_vllm.py \
  --input ./my_repo \
  --vllm-url http://172.32.1.250:8000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

### 3. Custom Model Name
```bash
python repo_to_diagrams_vllm.py \
  --input ./my_repo \
  --model local \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

### 4. Large Repository (Verbose Mode)
```bash
python repo_to_diagrams_vllm.py \
  --input ./large_repo \
  --max-tokens 16000 \
  --timeout 3600 \
  --verbose \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

### 5. High Quality Output
```bash
python repo_to_diagrams_vllm.py \
  --input ./my_repo \
  --rag-examples-per-type 8 \
  --temperature 0.05 \
  --max-tokens 12000 \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

### 6. Skip Validation (Fast Mode)
```bash
python repo_to_diagrams_vllm.py \
  --input ./my_repo \
  --no-validate \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

---

## Environment Setup

```bash
# Set once, use everywhere
export VLLM_URL=http://172.32.1.250:8000
export PLANTUML_EMBED_MODEL=nomic-embed-text

# Then just run:
python repo_to_diagrams_vllm.py --input ./repo --faiss-index rag/faiss.index --faiss-meta rag/faiss_meta.json
```

---

## Starting vLLM Server

### DGX A100 (4 GPUs)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
```

### Lambda Scalar (RTX 6000 Ada)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 1000000 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

---

## Troubleshooting Quick Fixes

### Connection Refused
```bash
# Check vLLM is running
curl http://localhost:8000/v1/models
```

### Timeout
```bash
--timeout 7200  # Increase to 2 hours
```

### OOM on vLLM Server
```bash
# Reduce context in vLLM startup:
--max-model-len 500000
# Or add more GPUs:
--tensor-parallel-size 8
```

### Invalid PlantUML
```bash
# Enable verbose to see what's wrong
--verbose
```

---

## Full Help
```bash
python repo_to_diagrams_vllm.py --help
```

---

## Recommended Settings by Repo Size

| Repo Size | Max Tokens | Timeout | RAG Examples | Temperature |
|-----------|------------|---------|--------------|-------------|
| Small (<100 files) | 8000 | 1800 | 4 | 0.1 |
| Medium (100-500) | 12000 | 3600 | 6 | 0.08 |
| Large (500-1000) | 16000 | 5400 | 8 | 0.05 |
| Very Large (>1000) | 20000 | 7200 | 10 | 0.05 |

---

## Output Files

For repo named `my_app`, you get:
```
uml/
├── my_app_class.puml       # Overall class structure
├── my_app_sequence.puml    # Interaction flows
├── my_app_activity.puml    # Process workflows
├── my_app_state.puml       # State transitions
├── my_app_component.puml   # High-level components
├── my_app_deployment.puml  # Deployment architecture
├── my_app_usecase.puml     # User-facing use cases
└── my_app_object.puml      # Runtime object instances
```

---

## Rendering Diagrams

```bash
# PNG (default)
plantuml uml/*.puml

# SVG (vector, better quality)
plantuml -tsvg uml/*.puml

# PDF
plantuml -tpdf uml/*.puml
```
