"""
The planner loop: payload -> backend -> validated, restored plan.

    1. send instructions + payload (the payload is the only repo-derived content)
    2. parse JSON, validate against the schema
    3. ingress: unknown references are errors; aliases restored to real ids
    4. artifact-specific semantic checks
    5. on any error, one repair round with the errors listed

Every request and response is recorded in the egress log.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jsonschema

from ..backends import BackendError, LLMBackend
from ..egress import EgressLog, Payload
from ..ingress import accept
from ..pseudonym import ALIAS_RE, AliasMap

Validator = Callable[[Any], List[str]]


@dataclass
class PlanResult:
    artifact: str
    plan: Optional[Any] = None          # restored (real ids), None on failure
    raw: Optional[Any] = None           # the accepted answer as the model wrote it (aliases intact)
    payload_aliased: bool = False       # the model saw pseudonymized identifiers
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None


def _schema_errors(obj: Any, schema: Dict[str, Any], limit: int = 8) -> List[str]:
    v = jsonschema.Draft202012Validator(schema)
    errs = sorted(v.iter_errors(obj), key=lambda e: list(e.path))
    out = []
    for e in errs[:limit]:
        where = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.path)
        out.append(f"{where}: {e.message[:200]}")
    return out


def run_plan(
    backend: LLMBackend,
    payload: Payload,
    schema: Dict[str, Any],
    system: str,
    instructions: str,
    aliases: AliasMap,
    validate: Optional[Validator] = None,
    log: Optional[EgressLog] = None,
    repairs: int = 1,
    max_tokens: int = 2048,
) -> PlanResult:
    result = PlanResult(payload.artifact, payload_aliased=bool(ALIAS_RE.search(payload.text)))
    base_user = instructions + payload.text
    user = base_user
    for attempt in range(1, repairs + 2):
        result.attempts = attempt
        errors: List[str] = []
        raw = ""
        try:
            completion = backend.complete_json(system, user, schema, max_tokens=max_tokens)
            raw = completion.text
        except BackendError as exc:
            result.errors = [f"backend: {exc}"]
            if log:
                log.record(payload, backend.name, backend.url, model=backend.model, attempt=attempt,
                           ok=False, errors=result.errors)
            return result

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            obj, errors = None, [f"response is not valid JSON: {exc}"]
        if obj is not None:
            errors = _schema_errors(obj, schema)
        restored = None
        if obj is not None and not errors:
            ingress = accept(obj, aliases)
            errors = ingress.errors
            restored = ingress.data
        if restored is not None and not errors and validate:
            errors = validate(restored)

        if log:
            log.record(payload, backend.name, backend.url, model=backend.model, attempt=attempt,
                       ok=not errors, errors=errors,
                       response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                       response=raw if log.include_payload else None,
                       prompt_tokens=getattr(completion, "prompt_tokens", None),
                       completion_tokens=getattr(completion, "completion_tokens", None),
                       seconds=getattr(completion, "seconds", None))
        if not errors:
            result.plan, result.raw, result.errors = restored, obj, []
            return result
        result.errors = errors
        user = (base_user + "\n\nYOUR PREVIOUS ANSWER:\n" + raw[:6000]
                + "\n\nIT WAS REJECTED FOR THESE REASONS:\n" + "\n".join(f"- {e}" for e in errors)
                + "\n\nReturn a corrected JSON answer.")
    return result
