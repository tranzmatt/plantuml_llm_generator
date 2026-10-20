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


def _ref(values: List[str], allow_empty: bool = False) -> Dict[str, Any]:
    vals = list(dict.fromkeys(values + ([""] if allow_empty else [])))
    return {"type": "string", "enum": vals} if vals else {"type": "string", "maxLength": 0}


def sequence_schema(participants: List[str], functions: List[str]) -> Dict[str, Any]:
    parts = participants + [CLIENT]
    return _obj({
        "title": {"type": "string"},
        "participants": {"type": "array", "minItems": 1, "items": _obj({
            "ref": _ref(parts),
            "kind": {"type": "string", "enum": PARTICIPANT_KINDS},
        })},
        "messages": {"type": "array", "minItems": 1, "items": _obj({
            "from": _ref(parts),
            "to": _ref(parts),
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
