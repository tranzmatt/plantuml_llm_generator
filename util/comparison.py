#!/usr/bin/env python3
"""
Side-by-side comparison: Your test-vllm-v2.py vs full PlantUML system.

This demonstrates how your simple test script maps to the full diagram generator.
"""


def show_comparison():
    print("=" * 80)
    print("YOUR TEST SCRIPT (test-vllm-v2.py) vs FULL PLANTUML SYSTEM")
    print("=" * 80)
    print()

    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ YOUR TEST SCRIPT: test-vllm-v2.py                                           │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print("""
from vllm import LLM, SamplingParams

# 1. Configure sampling
sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    max_tokens=2048,
    repetition_penalty=1.1,
)

# 2. Load model
llm = LLM(
    model="openai/gpt-oss-120b",
    tensor_parallel_size=4,
    max_model_len=16000,
    gpu_memory_utilization=0.95,
)

# 3. Generate with a simple prompt
prompt = "You are a PlantUML generator. Create a class diagram..."
out = llm.generate([prompt], sampling)

# 4. Print result
print(out[0].outputs[0].text)
""")

    print()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ FULL SYSTEM: repo_to_diagrams_local_vllm.py                                 │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print("""
from vllm import LLM, SamplingParams
from core.repo_scanner import scan_python_repo, load_repo_text
from core.rag_retriever import RagRetriever
from core.prompt_builder import build_system_prompt, build_user_prompt
from core.diagram_writer import write_diagrams

# STEP 1: Scan repository (NEW)
py_files = scan_python_repo("/path/to/repo")
repo_text = load_repo_text(py_files)

# STEP 2: Load RAG examples (NEW)
retriever = RagRetriever("rag/faiss.index", "rag/faiss_meta.json")
rag_examples = retriever.search("class diagram", top_k=4)

# STEP 3: Build sophisticated prompts (NEW)
system_msg = build_system_prompt()
user_msg = build_user_prompt(repo_name, repo_text, rag_examples)
full_prompt = f"{system_msg}\\n\\n{user_msg}"

# STEP 4: Configure sampling (SAME AS YOUR TEST)
sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    max_tokens=8000,
    repetition_penalty=1.1,
)

# STEP 5: Load model (SAME AS YOUR TEST)
llm = LLM(
    model="openai/gpt-oss-120b",
    tensor_parallel_size=4,
    max_model_len=16000,
    gpu_memory_utilization=0.95,
)

# STEP 6: Generate (SAME AS YOUR TEST)
outputs = llm.generate([full_prompt], sampling)
response = outputs[0].outputs[0].text

# STEP 7: Parse and validate (NEW)
diagrams = json.loads(response)  # Parse JSON with 8 diagram types
validate_plantuml(diagrams)       # Check syntax

# STEP 8: Write to files (NEW)
write_diagrams("./uml", repo_name, diagrams)
""")

    print()
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
The core vLLM usage (steps 4-6) is IDENTICAL to your test script!

Your test:  LLM() + SamplingParams() + llm.generate()
Full system: SAME + repository scanning + RAG + output processing

The full system just adds:
  • Repository analysis (before)
  • RAG retrieval (before)  
  • Prompt engineering (before)
  • JSON parsing (after)
  • File writing (after)

But the vLLM inference is exactly what you're already doing!
""")

    print()
    print("=" * 80)
    print("MIGRATION PATH")
    print("=" * 80)
    print("""
From your test to full system:

1. Your test works → vLLM is installed correctly ✓

2. Install additional dependencies:
   pip install faiss-cpu sentence-transformers

3. Create RAG index (one-time setup):
   # See README_LOCAL_VLLM.md for instructions

4. Run full system with SAME parameters:
   python repo_to_diagrams_local_vllm.py \\
     --model openai/gpt-oss-120b \\    # Same model
     --tp 4 \\                          # Same tensor parallel
     --max-len 16000 \\                 # Same max length
     --input /path/to/repo \\
     --faiss-index rag/faiss.index \\
     --faiss-meta rag/faiss_meta.json

5. Done! You now have 8 PlantUML diagrams.
""")

    print()
    print("=" * 80)
    print("PARAMETER MAPPING")
    print("=" * 80)
    print("""
Your test-vllm-v2.py          →  repo_to_diagrams_local_vllm.py
─────────────────────────────────────────────────────────────────
--model MODEL                 →  --model MODEL
--tp N                        →  --tp N
--max-len N                   →  --max-len N
--prompt "text"               →  (Generated from repo + RAG)
(gpu_memory_utilization=0.95) →  --gpu-memory-utilization 0.95
(temperature=0.0)             →  --temperature 0.0
(max_tokens=2048)             →  --max-tokens 8000
(repetition_penalty=1.1)      →  --repetition-penalty 1.1
(new)                         →  --input /path/to/repo
(new)                         →  --faiss-index rag/faiss.index
(new)                         →  --faiss-meta rag/faiss_meta.json
""")

    print()
    print("=" * 80)
    print("EXAMPLE: YOUR EXACT COMMAND TRANSLATED")
    print("=" * 80)
    print("""
Your test command:
─────────────────
python test-vllm-v2.py \\
  --model openai/gpt-oss-120b \\
  --tp 4 \\
  --max-len 4096 \\
  --prompt "Create a PlantUML diagram..."

Equivalent full system command:
───────────────────────────────
python repo_to_diagrams_local_vllm.py \\
  --model openai/gpt-oss-120b \\
  --tp 4 \\
  --max-len 16000 \\
  --input ~/Code/MyProject \\
  --faiss-index rag/faiss.index \\
  --faiss-meta rag/faiss_meta.json

Everything else is identical under the hood!
""")

    print()
    print("=" * 80)
    print("WHAT'S ADDED (The Value-Add)")
    print("=" * 80)
    print("""
1. Repository Analysis
   • Recursively scans Python files
   • Extracts classes, functions, dependencies
   • Builds complete codebase context

2. RAG (Retrieval-Augmented Generation)
   • Retrieves high-quality PlantUML examples
   • Ensures syntactically correct output
   • Provides diagram-type-specific templates

3. Prompt Engineering
   • Constructs sophisticated prompts
   • Includes STRICT syntax rules
   • Requests all 8 diagram types at once

4. Output Processing
   • Parses JSON response
   • Validates PlantUML syntax
   • Fixes common LLM mistakes
   • Writes separate .puml files

5. Error Handling
   • Validates all inputs
   • Catches generation errors
   • Provides helpful error messages
""")

    print()
    print("=" * 80)
    print("TL;DR")
    print("=" * 80)
    print("""
If test-vllm-v2.py works → Full system will work!

Just add:
  • FAISS RAG index (one-time setup)
  • Point to your repository
  • Same model, same GPUs, same settings

Get: 8 professionally generated PlantUML diagrams
""")
    print("=" * 80)


if __name__ == "__main__":
    show_comparison()
