"""
LLM backends. Every backend answers one question: given a system prompt, a
user prompt and a JSON schema, return the model's raw text (expected JSON).

Validation, repair, alias restoration and logging happen in archdoc.plan,
never here, so every transport gets identical treatment.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..trust import Backend


class BackendError(Exception):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    seconds: Optional[float] = None


class LLMBackend:
    """Interface: complete_json(system, user, schema) -> Completion."""

    name: str = "?"
    url: Optional[str] = None
    model: Optional[str] = None

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      max_tokens: int = 2048) -> Completion:
        raise NotImplementedError


def _post(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise BackendError(f"{url}: HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"{url}: {exc}") from None


class OllamaBackend(LLMBackend):
    """Ollama native /api/chat: schema-constrained `format` and per-request num_ctx."""

    def __init__(self, cfg: Backend, temperature: float = 0.0):
        self.name, self.url, self.model = cfg.name, (cfg.url or "http://localhost:11434").rstrip("/"), cfg.model
        self.num_ctx = cfg.num_ctx or 8192
        self.timeout = cfg.timeout
        self.temperature = temperature

    def complete_json(self, system, user, schema, max_tokens=2048) -> Completion:
        body = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        d = _post(f"{self.url}/api/chat", body, {}, self.timeout)
        try:
            text = d["message"]["content"]
        except (KeyError, TypeError):
            raise BackendError(f"unexpected Ollama response: {str(d)[:300]}") from None
        dur = d.get("total_duration")
        return Completion(text, d.get("prompt_eval_count"), d.get("eval_count"), dur / 1e9 if dur else None)


class OpenAICompatBackend(LLMBackend):
    """POST /chat/completions with response_format json_schema (vLLM, OpenAI-compatible servers)."""

    def __init__(self, cfg: Backend, temperature: float = 0.0):
        self.name, self.url, self.model = cfg.name, (cfg.url or "").rstrip("/"), cfg.model
        if not self.url:
            raise BackendError(f"backend {cfg.name!r} needs a url")
        self.timeout = cfg.timeout
        self.temperature = temperature
        self.headers: Dict[str, str] = {}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise BackendError(f"backend {cfg.name!r}: environment variable {cfg.api_key_env} is not set")
            self.headers["Authorization"] = f"Bearer {key}"

    def complete_json(self, system, user, schema, max_tokens=2048) -> Completion:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "plan", "strict": True, "schema": schema}},
        }
        d = _post(f"{self.url}/chat/completions", body, self.headers, self.timeout)
        try:
            text = d["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"unexpected response: {str(d)[:300]}") from None
        usage = d.get("usage") or {}
        return Completion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"))


@dataclass
class ScriptedBackend(LLMBackend):
    """Test double: `respond(system, user, schema)` returns the text to send back."""

    respond: Callable[[str, str, Dict[str, Any]], str]
    name: str = "scripted"
    url: Optional[str] = None
    model: Optional[str] = "scripted"
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete_json(self, system, user, schema, max_tokens=2048) -> Completion:
        self.calls.append({"system": system, "user": user, "schema": schema})
        return Completion(self.respond(system, user, schema))


def make_backend(cfg: Backend, temperature: float = 0.0) -> LLMBackend:
    if cfg.kind == "ollama":
        return OllamaBackend(cfg, temperature)
    return OpenAICompatBackend(cfg, temperature)
