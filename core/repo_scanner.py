import os
from typing import List

def scan_python_repo(root: str) -> List[str]:
    """Recursively find all .py files under root."""
    py_files: List[str] = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                py_files.append(os.path.join(dirpath, name))
    return py_files

def load_repo_text(files: List[str]) -> str:
    """Concatenate the contents of all Python files into a single string."""
    chunks = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f"""# File: {path}\n""" + f.read())
        except Exception as e:
            print(f"[repo_scanner] Warning: could not read {path}: {e}")
    return "\n\n".join(chunks)
