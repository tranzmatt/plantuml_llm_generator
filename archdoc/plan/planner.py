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
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jsonschema

from ..backends import BackendError, LLMBackend
from ..egress import EgressLog, Payload
from ..ingress import accept
from ..pseudonym import ALIAS_RE, AliasMap
from .schemas import REF_DESCRIPTION, outline

Validator = Callable[[Any], List[str]]
# how OpenAI-compatible servers (vLLM) reject a prompt longer than their window
_OVERFLOW_RE = re.compile(r"maximum context length|context length|too long|exceeds the model|max_model_len", re.I)


@dataclass
class PlanResult:
    artifact: str
    plan: Optional[Any] = None          # restored (real ids), None on failure
    raw: Optional[Any] = None           # the accepted answer as the model wrote it (aliases intact)
    # last answer that parsed, matched the schema and passed ingress but failed
    # `validate`: artifacts may prune it instead of discarding the whole plan
    candidate: Optional[Any] = None
    candidate_raw: Optional[Any] = None
    report: Dict[str, Any] = field(default_factory=dict)   # artifact-specific facts for provenance
    payload_aliased: bool = False       # the model saw pseudonymized identifiers
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    warnings: List[str] = field(default_factory=list)
    prompt_tokens: int = 0              # summed over attempts, where the server reports them
    completion_tokens: int = 0
    # provenance (DESIGN.md 2.1): what the answer was based on
    payload_sha256: str = ""
    prompt_sha256: str = ""             # system prompt + instructions, i.e. the prompt version
    shown: Dict[str, List[str]] = field(default_factory=dict)   # role -> real ids in the payload
    backend: Dict[str, Optional[str]] = field(default_factory=dict)
    levels: Dict[str, str] = field(default_factory=dict)        # item id -> stub | outline | source
    fit: Dict[str, Any] = field(default_factory=dict)           # window, budget and size actually sent
    # context fitting: the first request's size, for calibration; whether it overflowed
    first_prompt_chars: int = 0
    first_prompt_tokens: Optional[int] = None
    overflow: bool = False

    @property
    def ok(self) -> bool:
        return self.plan is not None


def _schema_errors(obj: Any, schema: Dict[str, Any], limit: int = 8) -> List[str]:
    v = jsonschema.Draft202012Validator(schema)
    errs = sorted(v.iter_errors(obj), key=lambda e: list(e.path))
    out = []
    for e in errs[:limit]:
        where = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.path)
        msg = e.message[:200]
        if e.validator == "additionalProperties":
            msg += f" (allowed keys: {', '.join(e.schema.get('properties', {}))})"
        elif e.validator == "enum" and str(e.schema.get("description", "")).startswith(REF_DESCRIPTION):
            msg = f"{e.instance!r} is not a ref from the input"
        out.append(f"{where}: {msg}")
    return out


_REF_FIELDS = ("ref", "from", "to", "call")


def _expand_short(obj: Any, short: Dict[str, str], key: str = "") -> Any:
    """Short ids back to full ids, in reference fields only (labels keep their wording)."""
    if not short:
        return obj
    if isinstance(obj, dict):
        return {k: _expand_short(v, short, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_short(v, short, key) for v in obj]
    if key in _REF_FIELDS and isinstance(obj, str):
        return short.get(obj, obj)
    return obj


def payload_terms(payload: Payload, aliases: AliasMap) -> Callable[[str], str]:
    """Rewrite text in the terms the model was shown.

    Validators report problems after ingress, i.e. with real ids. Before such a
    message goes back to the model in a repair prompt, every real term that this
    payload showed as an alias is replaced by that alias, so a repair round
    never discloses a name the payload hid. Terms the payload showed for real
    (local tier) are left alone.
    """
    present = sorted(set(ALIAS_RE.findall(payload.text)))
    pairs = [(aliases.resolve_ref(a), a) for a in present]
    pairs += [(full, short) for short, full in payload.short.items()]   # local ids in their short form
    pairs = sorted(((k, a) for k, a in pairs if k), key=lambda kv: -len(kv[0]))
    if not pairs:
        return lambda text: text
    rx = re.compile("|".join(rf"(?<![\w.:/-]){re.escape(k)}(?![\w/-])" for k, _ in pairs))
    lookup = dict(pairs)
    return lambda text: rx.sub(lambda m: lookup[m.group(0)], text)


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
    to_payload_terms = payload_terms(payload, aliases)
    system = system + "\n\nRESPONSE FORMAT: one JSON object with exactly these keys, all required:\n" \
        + outline(schema)
    result.payload_sha256 = payload.sha256
    result.prompt_sha256 = hashlib.sha256((system + "\0" + instructions).encode()).hexdigest()
    result.shown = {role: [aliases.resolve_ref(r) or r if ALIAS_RE.fullmatch(r) else payload.short.get(r, r)
                           for r in refs]
                    for role, refs in payload.allowed.items()}
    result.backend = {"name": backend.name, "model": backend.model, "url": backend.url,
                      "digest": getattr(backend, "digest", None)}
    base_user = instructions + payload.text
    user = base_user
    for attempt in range(1, repairs + 2):
        result.attempts = attempt
        errors: List[str] = []
        raw = ""
        if backend.unavailable:
            result.errors = [f"skipped: backend unavailable earlier in this run ({backend.unavailable})"]
            return result
        try:
            completion = backend.complete_json(system, user, schema, max_tokens=max_tokens)
            raw = completion.text
            result.prompt_tokens += completion.prompt_tokens or 0
            result.completion_tokens += completion.completion_tokens or 0
            if attempt == 1:
                result.first_prompt_chars = len(system) + len(user)
                result.first_prompt_tokens = completion.prompt_tokens
            window = getattr(backend, "window", None)
            if window and completion.prompt_tokens and completion.prompt_tokens >= window - 8:
                # Ollama keeps going with a silently truncated prompt: never trust that answer
                result.overflow = True
                result.errors = [f"the prompt filled the {window}-token window ({completion.prompt_tokens} "
                                 f"tokens) and was truncated by the server"]
                return result
        except BackendError as exc:
            if exc.unavailable:
                backend.unavailable = str(exc)
            result.overflow = bool(_OVERFLOW_RE.search(str(exc)))
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
            restored = _expand_short(ingress.data, payload.short)
        if restored is not None and not errors and validate:
            errors = validate(restored)
            if errors:
                result.candidate, result.candidate_raw = restored, obj

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
                + "\n\nIT WAS REJECTED FOR THESE REASONS:\n"
                + "\n".join(f"- {to_payload_terms(e)}" for e in errors)
                + "\n\nReturn a corrected JSON answer.")
    return result
