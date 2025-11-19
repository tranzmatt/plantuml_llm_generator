
# VLLM Setup Guide (Local or Remote Machine)

This guide explains how to install and run **vLLM** on a local workstation, a remote GPU server (DGX/HPC), or inside Docker.  
It covers environment setup, model downloading, GPU requirements, and running an API server compatible with OpenAI/Ollama-style clients.

---

## 1. System Requirements

### **Hardware**
- NVIDIA GPU with **Compute Capability 7.0+**
- Recommended:
  - 24GB+ VRAM (A100, H100, 3090, 4090, etc.)
  - Multi-GPU supported (Tensor Parallel)

### **Software**
- Linux (Ubuntu 20.04/22.04 recommended)
- Python 3.9–3.12
- CUDA 11.8 or 12.x
- `pip` or `conda`

---

## 2. Install vLLM (pip)

### **Option A — Install with CUDA 12**
```bash
pip install vllm-cu12
```

### **Option B — Install with CUDA 11.8**
```bash
pip install vllm
```

To verify:
```bash
python -c "import vllm; print('vLLM installed OK')"
```

---

## 3. Running vLLM with HF Models

### **Download a model**
Example:
```bash
huggingface-cli download meta-llama/Llama-3.1-70B --local-dir /models/llama70b
```

---

## 4. Launching vLLM as an OpenAI-Compatible Server

### **Basic server run**
```bash
python -m vllm.entrypoints.openai.api_server   --model /models/llama70b   --tensor-parallel-size 4   --port 8000   --max-model-len 2000000
```

### **Key Parameters**
| Flag | Meaning |
|------|---------|
| `--tensor-parallel-size` | Number of GPUs |
| `--max-model-len` | Context window |
| `--gpu-memory-utilization` | Prevent OOMs |

---

## 5. Using vLLM from Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(resp.choices[0].message["content"])
```

---

## 6. Using vLLM with curl

```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "local",
        "messages": [{"role": "user", "content": "Write a haiku"}]
      }'
```

---

## 7. Running vLLM on a Remote Server

SSH into server:
```bash
ssh mygpu-server
```

Launch server:
```bash
python -m vllm.entrypoints.openai.api_server   --model /models/llama70b   --tensor-parallel-size 8   --host 0.0.0.0   --port 8000
```

Enable firewall/security group for port 8000.

Call from your workstation:
```bash
export OPENAI_API_BASE="http://server-ip:8000/v1"
```

---

## 8. Running vLLM with Docker

### **Pull official image**
```bash
docker pull vllm/vllm-openai:latest
```

### **Run**
```bash
docker run --gpus all -p 8000:8000   -v /models:/models   vllm/vllm-openai:latest   --model /models/llama70b
```

---

## 9. Multi-GPU Configuration (DGX / A100)

Edit environment:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Run:
```bash
python -m vllm.entrypoints.openai.api_server   --model /models/llama70b   --tensor-parallel-size 4   --gpu-memory-utilization 0.92   --max-model-len 2000000
```

---

## 10. Troubleshooting

### **OOM Errors**
- Reduce context:
  ```bash
  --max-model-len 400000
  ```
- Increase GPU count:
  ```bash
  --tensor-parallel-size 8
  ```

### **Slow inference**
- Add:
  ```bash
  --served-model-name local
  --disable-log-requests
  ```

### **Model not loading**
Check VRAM:
```bash
nvidia-smi
```

---

## 11. Summary

vLLM gives you:
- Extremely fast inference
- Massive context windows
- Multi-GPU parallelism
- OpenAI-compatible API

Perfect for running:
- Large models locally  
- High-throughput servers  
- Enterprise or HPC workflows  

---

If you want:  
✅ a turnkey `docker-compose.yml`  
✅ a systemd service file  
✅ a GPU monitoring script  
I can generate them too.

