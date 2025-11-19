# vLLM-Only Version: Comparison and Usage Guide

## Overview

This repository now contains two versions of the diagram generator:

1. **`cli/repo_to_diagrams.py`** - Original dual-backend version (Ollama + vLLM)
2. **`cli/repo_to_diagrams_vllm.py`** - New vLLM-only version (streamlined)

## Key Differences

### Original Version (`cli/repo_to_diagrams.py`)
- ✅ Supports both Ollama and vLLM backends
- ✅ Good for testing multiple backends
- ⚠️ Requires backend selection via `--backend` flag
- ⚠️ Less optimized defaults for vLLM
- ⚠️ Minimal error handling and feedback

### vLLM-Only Version (`repo_to_diagrams_vllm.py`)
- ✅ Streamlined for vLLM deployment
- ✅ Better defaults for vLLM parameters
- ✅ Enhanced error handling and diagnostics
- ✅ Verbose mode for debugging
- ✅ Progress indicators for long-running operations
- ✅ Better validation and error messages
- ✅ Cleaner CLI with examples in help text
- ⚠️ No Ollama support

## Installation

Both versions use the same dependencies:

```bash
pip install -r requirements.txt
```

For vLLM server setup, see `VLLM_SETUP.md`.

## Usage Comparison

### Original Version

```bash
# With Ollama
python -m cli.repo_to_diagrams \
  --input /path/to/repo \
  --output uml \
  --backend ollama \
  --model llama4:maverick \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# With vLLM
python -m cli.repo_to_diagrams \
  --input /path/to/repo \
  --output uml \
  --backend vllm \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --vllm-url http://localhost:8000
```

### vLLM-Only Version

```bash
# Basic usage (assumes vLLM on localhost:8000)
python repo_to_diagrams_vllm.py \
  --input /path/to/repo \
  --output uml \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# With custom vLLM server
python repo_to_diagrams_vllm.py \
  --input /path/to/repo \
  --vllm-url http://172.32.1.250:8000 \
  --model local \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# With verbose output
python repo_to_diagrams_vllm.py \
  --input /path/to/repo \
  --verbose \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

## Advanced Configuration

### vLLM-Only Version Parameters

```bash
python repo_to_diagrams_vllm.py \
  --input ./myrepo \
  --output ./uml_output \
  --vllm-url http://gpu-server:8000 \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --max-tokens 12000 \          # Increase for complex repos
  --temperature 0.05 \           # Lower = more deterministic
  --timeout 3600 \               # 1 hour for very large repos
  --rag-examples-per-type 6 \    # More examples = better quality
  --faiss-index ./rag/faiss.index \
  --faiss-meta ./rag/faiss_meta.json \
  --embed-model nomic-embed-text \
  --verbose \
  --no-validate                  # Skip PlantUML validation
```

## Environment Variables

### Original Version
```bash
export OLLAMA_URL=http://172.32.1.250:11434
export PLANTUML_LLM_MODEL=llama4:maverick
export VLLM_URL=http://localhost:8000
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

### vLLM-Only Version
```bash
export VLLM_URL=http://172.32.1.250:8000
export PLANTUML_EMBED_MODEL=nomic-embed-text
```

## Performance Considerations

### When to Use vLLM-Only Version

1. **Production Deployments**
   - Dedicated vLLM infrastructure
   - High-throughput requirements
   - Large context windows (500k+ tokens)
   - Multi-GPU tensor parallelism

2. **Large Repositories**
   - 1000+ Python files
   - Complex microservice architectures
   - Need for detailed diagrams

3. **Batch Processing**
   - Multiple repos to process
   - Automated CI/CD pipelines
   - Consistent results required

### When to Use Original Version

1. **Development/Testing**
   - Comparing Ollama vs vLLM outputs
   - Don't have dedicated vLLM server
   - Quick prototyping with Ollama

2. **Small Projects**
   - Single-file or small repos
   - Ollama is "good enough"
   - Local development on laptop

## vLLM Server Configuration for Large Models

### For Llama-4-Maverick-17B-128E (DGX A100)

```bash
# Start vLLM server with optimal settings
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 2000000 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

### For Llama-4-Scout-17Bx16E (Smaller/Faster)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17Bx16E-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 1000000 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

## Troubleshooting

### vLLM Connection Issues

```bash
# Test vLLM server is responding
curl http://localhost:8000/v1/models

# Run with verbose mode to see full request details
python repo_to_diagrams_vllm.py \
  --input ./test_repo \
  --verbose \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Timeout Errors

If processing very large repositories:

```bash
python repo_to_diagrams_vllm.py \
  --input ./large_repo \
  --timeout 7200 \              # 2 hours
  --max-tokens 16000 \          # Larger output buffer
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

### Out of Memory (OOM)

If vLLM server runs out of memory:

1. Reduce context window:
```bash
# In vLLM server startup
--max-model-len 500000
```

2. Increase tensor parallelism:
```bash
# Use more GPUs
--tensor-parallel-size 8
```

3. Split large repositories into smaller chunks

### Invalid PlantUML Output

```bash
# Run with validation enabled (default)
python repo_to_diagrams_vllm.py \
  --input ./repo \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# Validation errors will show exact PlantUML issues

# If you want to skip validation:
python repo_to_diagrams_vllm.py \
  --input ./repo \
  --no-validate \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json
```

## Migration Guide

### From Ollama to vLLM

1. **Setup vLLM server** (see `VLLM_SETUP.md`)

2. **Convert model names**:
   - Ollama: `llama4:maverick`
   - vLLM: `meta-llama/Llama-4-Maverick-17B-128E-Instruct`

3. **Update scripts**:
```bash
# Old (Ollama)
python -m cli.repo_to_diagrams \
  --backend ollama \
  --model llama4:maverick \
  --ollama-url http://172.32.1.250:11434

# New (vLLM)
python repo_to_diagrams_vllm.py \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --vllm-url http://172.32.1.250:8000
```

### From Original to vLLM-Only Version

Simply remove `--backend vllm` and use the new script:

```bash
# Original
python -m cli.repo_to_diagrams \
  --backend vllm \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --vllm-url http://localhost:8000

# New
python repo_to_diagrams_vllm.py \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --vllm-url http://localhost:8000
```

## Example Workflows

### Workflow 1: Local Development with vLLM

```bash
# Terminal 1: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 1000000 \
  --port 8000

# Terminal 2: Generate diagrams
python repo_to_diagrams_vllm.py \
  --input ~/Code/my_project \
  --output ~/Diagrams/my_project \
  --verbose \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

# Terminal 3: Render diagrams
cd ~/Diagrams/my_project
plantuml *.puml
open *.png
```

### Workflow 2: Remote vLLM Server

```bash
# On your workstation
export VLLM_URL=http://dgx-01.mylab.local:8000

python repo_to_diagrams_vllm.py \
  --input /mnt/projects/microservices \
  --output /mnt/docs/uml \
  --model local \
  --max-tokens 16000 \
  --rag-examples-per-type 8 \
  --faiss-index /mnt/rag/faiss.index \
  --faiss-meta /mnt/rag/faiss_meta.json \
  --verbose
```

### Workflow 3: Batch Processing Multiple Repos

```bash
#!/bin/bash
# process_repos.sh

REPOS=(
  "/home/user/project1"
  "/home/user/project2"
  "/home/user/project3"
)

for repo in "${REPOS[@]}"; do
  echo "Processing: $repo"
  python repo_to_diagrams_vllm.py \
    --input "$repo" \
    --output "./diagrams/$(basename $repo)" \
    --faiss-index ./rag/faiss.index \
    --faiss-meta ./rag/faiss_meta.json \
    --verbose
done

echo "All repositories processed"
```

## Quality Optimization Tips

### 1. Increase RAG Examples
```bash
--rag-examples-per-type 8  # More examples = better syntax
```

### 2. Use Larger Models
- **Best**: Llama-4-Maverick-17B-128E (balanced)
- **Good**: Llama-4-Scout-17Bx16E (faster)
- **Avoid**: Models < 13B parameters (poor PlantUML syntax)

### 3. Lower Temperature
```bash
--temperature 0.05  # More deterministic output
```

### 4. Increase Max Tokens for Complex Repos
```bash
--max-tokens 16000  # For repos with 500+ files
```

### 5. Enable Validation
```bash
# Default behavior - validates all diagrams
# Catches syntax errors immediately
```

## Output Examples

After running successfully:

```
==================================================================
SUCCESS: All diagrams generated
==================================================================
Output location: /home/user/Diagrams/my_project/
Files generated:
  - my_project_class.puml
  - my_project_sequence.puml
  - my_project_activity.puml
  - my_project_state.puml
  - my_project_component.puml
  - my_project_deployment.puml
  - my_project_usecase.puml
  - my_project_object.puml

To render diagrams:
  plantuml /home/user/Diagrams/my_project/*.puml
==================================================================
```

## Conclusion

The vLLM-only version provides:
- ✅ Cleaner, more focused codebase
- ✅ Better error handling and diagnostics
- ✅ Optimized for production vLLM deployments
- ✅ Enhanced user experience with progress indicators
- ✅ Ready for CI/CD integration

Use the original version if you need Ollama support or want backend flexibility.
Use the vLLM-only version for production deployments and better user experience.
