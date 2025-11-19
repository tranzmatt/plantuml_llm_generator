import os
import json
from typing import Any


def get_repo_name(path: str) -> str:
    path = os.path.abspath(path)
    return os.path.basename(path)


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Failed to parse LLM JSON output. "
            f"Error: {e}. Raw text (truncated):\n{text[:1000]}"
        )
