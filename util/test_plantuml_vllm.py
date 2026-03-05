#!/usr/bin/env python3
"""
Minimal test script to validate vLLM installation and PlantUML generation.
Similar to test-vllm-v2.py but tests the full diagram generation pipeline.
"""

import argparse
import sys
from vllm import LLM, SamplingParams


def test_basic_inference(args):
    """Test basic vLLM inference with a simple prompt."""
    print("\n" + "=" * 70)
    print("Test 1: Basic vLLM Inference")
    print("=" * 70)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=512,
        repetition_penalty=1.1,
    )

    print(f"Loading model: {args.model}")
    print(f"Using {args.tp} GPUs")
    print("(This may take 1-2 minutes...)")

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_len,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    test_prompt = "You are a helpful assistant. Say hello and confirm you are working."

    print("\nRunning inference...")
    try:
        outputs = llm.generate([test_prompt], sampling)
        response = outputs[0].outputs[0].text
        print(f"✓ Inference successful!")
        print(f"\nResponse: {response[:200]}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

    return True


def test_plantuml_generation(args):
    """Test PlantUML diagram generation."""
    print("\n" + "=" * 70)
    print("Test 2: PlantUML Diagram Generation")
    print("=" * 70)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=1024,
        repetition_penalty=1.1,
    )

    print(f"Loading model: {args.model}")

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_len,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    plantuml_prompt = """You are a PlantUML generator. Create ONLY valid PlantUML syntax.

Generate a simple class diagram for this Python code:

class Database:
    def connect(self):
        pass
    
class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, id):
        return self.db.query(id)

Output ONLY the PlantUML code, starting with @startuml and ending with @enduml.
"""

    print("Generating PlantUML diagram...")
    try:
        outputs = llm.generate([plantuml_prompt], sampling)
        response = outputs[0].outputs[0].text
        print("✓ Generation successful!")

        if "@startuml" in response and "@enduml" in response:
            print("✓ Valid PlantUML structure detected")
        else:
            print("⚠ Warning: Missing @startuml/@enduml tags")

        print("\nGenerated PlantUML:")
        print("-" * 70)
        print(response)
        print("-" * 70)

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

    return True


def test_json_output(args):
    """Test JSON structured output."""
    print("\n" + "=" * 70)
    print("Test 3: JSON Structured Output")
    print("=" * 70)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=512,
        repetition_penalty=1.1,
    )

    print(f"Loading model: {args.model}")

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_len,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    json_prompt = """Return ONLY a valid JSON object with these keys:
{
  "class": "A simple class diagram",
  "sequence": "A simple sequence diagram"
}

Replace the values with actual PlantUML code snippets (just basic examples).
Output ONLY the JSON, no other text.
"""

    print("Generating JSON output...")
    try:
        outputs = llm.generate([json_prompt], sampling)
        response = outputs[0].outputs[0].text.strip()
        print("✓ Generation successful!")

        import json
        try:
            parsed = json.loads(response)
            print("✓ Valid JSON structure")
            print(f"  Keys found: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            print(f"⚠ JSON parsing failed: {e}")
            print(f"  Response: {response[:200]}")

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM installation and PlantUML generation capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with GPT-OSS-120B on 4 GPUs (your setup)
  %(prog)s --model openai/gpt-oss-120b --tp 4
  
  # Test with Llama model on 2 GPUs
  %(prog)s --model meta-llama/Llama-4-Maverick-17B-128E-Instruct --tp 2
  
  # Quick test with smaller model
  %(prog)s --model meta-llama/Llama-3-8B-Instruct --tp 1
        """
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name or path (e.g., openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallel size (number of GPUs, default: 2)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=8192,
        help="Max model sequence length (default: 8192)",
    )
    parser.add_argument(
        "--test",
        choices=["basic", "plantuml", "json", "all"],
        default="all",
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("vLLM + PlantUML Generation Test Suite")
    print("=" * 70)
    print(f"Model:            {args.model}")
    print(f"Tensor Parallel:  {args.tp} GPUs")
    print(f"Max Length:       {args.max_len}")
    print("=" * 70)

    results = {}

    if args.test in ["basic", "all"]:
        results["basic"] = test_basic_inference(args)

    if args.test in ["plantuml", "all"]:
        results["plantuml"] = test_plantuml_generation(args)

    if args.test in ["json", "all"]:
        results["json"] = test_json_output(args)

    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.ljust(20)} {status}")

    all_passed = all(results.values())

    print("=" * 70)
    if all_passed:
        print("✓ All tests passed! Ready to generate PlantUML diagrams.")
        print("\nNext steps:")
        print("  1. Prepare your FAISS RAG index")
        print("  2. Run: python repo_to_diagrams_local_vllm.py --help")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
