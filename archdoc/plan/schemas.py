"""
JSON schemas for planner output.

Reference fields are enums of exactly the refs the payload exposed, so a
schema-constrained backend cannot reference anything it was not shown. All
properties are required (optional values are "" / []), which keeps the schemas
valid for OpenAI-style strict mode as well as Ollama.
"""

from __future__ import annotations

from typing import Any, Dict, List

CLIENT = "client"
PARTICIPANT_KINDS = ["participant", "actor", "boundary", "control", "entity", "database", "queue", "collections"]
GROUP_KINDS = ["alt", "opt", "loop", "group"]


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


REF_DESCRIPTION = "a ref exactly as it appears in the input"


def _ref(values: List[str], allow_empty: bool = False, also: str = "") -> Dict[str, Any]:
    vals = list(dict.fromkeys(values + ([""] if allow_empty else [])))
    desc = REF_DESCRIPTION + (f', or "{also}"' if also else "") + (', or ""' if allow_empty else "")
    if not vals:
        return {"type": "string", "maxLength": 0, "description": '""'}
    return {"type": "string", "enum": vals, "description": desc}


def outline(schema: Dict[str, Any], indent: int = 0) -> str:
    """Human-readable response format for the prompt.

    Servers that do not enforce `format`/`response_format` (Ollama cloud models,
    some OpenAI-compatible servers) only know the shape from the prompt. Ref
    enums are described, not listed: the refs are already in the input.
    """
    pad = "  " * indent
    t = schema.get("type")
    if t == "object":
        lines = ["{"]
        for k, sub in schema["properties"].items():
            lines.append(f'{pad}  "{k}": {outline(sub, indent + 1)},')
        lines[-1] = lines[-1].rstrip(",")
        return "\n".join(lines + [pad + "}"])
    if t == "array":
        note = " (must be empty)" if schema.get("maxItems") == 0 else ""
        return f"[ {outline(schema['items'], indent)} ]{note}"
    if "enum" in schema:
        if schema.get("description", "").startswith(REF_DESCRIPTION):
            return f"<{schema['description']}>"
        return " | ".join(f'"{v}"' for v in schema["enum"])
    if "description" in schema:
        return f"<{schema['description']}>"
    return {"string": "<string>", "integer": "<integer>", "boolean": "true | false"}.get(t, "<value>")


def sequence_schema(participants: List[str], functions: List[str]) -> Dict[str, Any]:
    parts = participants + [CLIENT]
    return _obj({
        "title": {"type": "string"},
        "participants": {"type": "array", "minItems": 1, "items": _obj({
            "ref": _ref(parts, also=CLIENT),
            "kind": {"type": "string", "enum": PARTICIPANT_KINDS},
        })},
        "messages": {"type": "array", "minItems": 1, "items": _obj({
            "from": _ref(parts, also=CLIENT),
            "to": _ref(parts, also=CLIENT),
            "call": _ref(functions, allow_empty=True),
            "label": {"type": "string"},
            "reply": {"type": "boolean"},
        })},
        "groups": {"type": "array", "items": _obj({
            "kind": {"type": "string", "enum": GROUP_KINDS},
            "label": {"type": "string"},
            "first": {"type": "integer", "minimum": 0},
            "last": {"type": "integer", "minimum": 0},
        })},
    })


def narrative_schema(operations: List[str], params: List[str]) -> Dict[str, Any]:
    param_items = _obj({"name": _ref(params), "description": {"type": "string"}})
    params_schema: Dict[str, Any] = {"type": "array", "items": param_items}
    if not params:
        params_schema["maxItems"] = 0
    return _obj({
        "purpose": {"type": "string", "minLength": 1},
        "operations": {"type": "array", "items": _obj({
            "ref": _ref(operations),
            "summary": {"type": "string"},
            "params": params_schema,
        })},
    })
