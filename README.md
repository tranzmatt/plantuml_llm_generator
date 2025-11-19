# PlantUML-RAG Modular System  
**Generate complete, consistent UML diagrams from any code repository using RAG + local LLMs (Ollama or vLLM).**

This toolkit parses a code repository, retrieves high-quality PlantUML examples via FAISS-based RAG, and generates **all 8 UML diagram types in one LLM call** to guarantee naming consistency and correctness.

It supports both **Ollama** (llama4:maverick recommended) and **vLLM** with HuggingFace models (e.g., Llama-4-Maverick-17B-128E).

---

## ✨ Features

- **Single-pass LLM generation** for:
  - Class diagram  
  - Sequence diagram  
  - Activity diagram  
  - State diagram  
  - Component diagram  
  - Deployment diagram  
  - Object diagram  
  - Use-case diagram  

- Ensures **consistent naming** across diagrams.
- Uses **FAISS RAG** to enforce *syntactically correct PlantUML*.
- Supports **Ollama** *and* **vLLM** backends.
- Modular architecture:
  - Shared scanner, prompt builder, diagram sanitizer, writers.
  - Backend-specific clients (Ollama / vLLM).
- Validates diagrams using the PlantUML CLI (optional).

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Requirements include:

- faiss-cpu
- sentence-transformers
- numpy
- requests

(You may upgrade to `faiss-gpu` on DGX hardware.)

---

## 📂 Directory Structure

```
plantuml_modular/
├── core/
│   ├── repo_scanner.py
│   ├── rag_retriever.py
│   ├── prompt_builder.py
│   ├── plantuml_sanitizer.py
│   ├── diagram_writer.py
│   └── utils.py
│
├── llm_backends/
│   ├── ollama_client.py
│   └── vllm_client.py
│
└── cli/
    └── repo_to_diagrams.py
```

---

## 🚀 Usage

### **1. Generate UML using Ollama**

```bash
python -m cli.repo_to_diagrams   --input /path/to/repo   --output uml   --backend ollama   --model llama4:maverick   --faiss-index rag/faiss.index   --faiss-meta rag/faiss_meta.json   --ollama-url http://172.32.1.250:11434
```

Environment variables:

```
export OLLAMA_URL=http://172.32.1.250:11434
export PLANTUML_LLM_MODEL=llama4:maverick
```

---

### **2. Generate UML using vLLM**

```bash
python -m cli.repo_to_diagrams   --input /path/to/repo   --output uml   --backend vllm   --model meta-llama/Llama-4-Maverick-17B-128E-Instruct   --faiss-index rag/faiss.index   --faiss-meta rag/faiss_meta.json   --vllm-url http://localhost:8000
```

Environment variables:

```
export VLLM_URL=http://localhost:8000
```

---

## 📤 Output

If your repo is located at:

```
/home/user/Code/my-app/
```

and `repo_name` is derived automatically (`my-app`), output files will be:

```
uml/my-app_class.puml
uml/my-app_sequence.puml
uml/my-app_activity.puml
uml/my-app_state.puml
uml/my-app_component.puml
uml/my-app_deployment.puml
uml/my-app_object.puml
uml/my-app_usecase.puml
```

---

## 🔍 Diagram Validation

To validate all diagrams:

```bash
plantuml uml/*.puml
```

To disable validation in the generator:

```bash
--no-validate
```

---

## 🔧 Backend Selection

Choose backend:

```
--backend ollama
--backend vllm
```

Both share the same:

- code parser  
- RAG retriever  
- prompt builder  
- sanitizer  
- writer  

Only inference calls differ.

---

## 📌 Model Recommendations

| Backend | Recommended Model | Notes |
|--------|-------------------|------|
| **Ollama** | llama4:maverick | Best syntax correctness + coherent diagrams |
| **vLLM** | Llama-4-Maverick-17B-128E-Instruct | Supports 1M–10M ctx, ideal for large repos |
| **Fallback** | Llama-4-Scout-17Bx16E-Instruct | Smaller, faster, lower quality |

Correctness is **priority #1**, so larger models are preferred.

---

## 🧠 RAG Details

The FAISS index is built from:

- Thousands of high-quality PlantUML examples  
- All diagram types  
- Clean syntax guaranteed  
- Adjustable with your own examples  

---

## 🧪 Test Query Script

You can test your index:

```bash
python scripts/test_faiss_query.py "sequence diagram for async worker queues"
```

---

## 🧩 Extending the System

You can easily add:

- Custom prompt styles  
- Additional diagram types  
- Alternative embedding models  
- Support for Java / Go / TypeScript scanners  

---

## 📝 License

MIT License — do anything, no warranty.
