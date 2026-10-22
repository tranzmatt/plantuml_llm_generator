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
    """`unavailable`: the server could not be reached or did not answer in time
    (as opposed to rejecting this particular request). Trips the circuit breaker.
    `pending`: a deferred backend (archdoc.harness) has no answer yet."""

    def __init__(self, message: str, unavailable: bool = False, pending: bool = False):
        super().__init__(message)
        self.unavailable = unavailable
        self.pending = pending


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
    # set when the backend became unreachable during a run: later requests are
    # skipped instead of each waiting out its own timeout (circuit breaker)
    unavailable: Optional[str] = None
    digest: Optional[str] = None       # model content digest, where the server reports one (Ollama)
    window: Optional[int] = None       # context window in tokens, when known (archdoc.packing)

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      max_tokens: int = 2048) -> Completion:
        raise NotImplementedError

    def health(self, timeout: int = 10) -> None:
        """Raise BackendError if the server does not answer a cheap request quickly."""

    def warm_up(self) -> Optional[float]:
        """Load the model before real requests; returns seconds taken, or None if not applicable."""
        return None

    def ensure_ready(self) -> None:
        """Warm up once, before the first request actually sent: a run answered
        entirely from the response cache never loads the model."""
        if not getattr(self, "_ready", False):
            self._ready = True
            self.warm_up()


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
        raise BackendError(f"{url}: HTTP {exc.code}: {detail}", unavailable=exc.code in (502, 503, 504)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"{url}: {exc}", unavailable=True) from None


def _get(url: str, headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise BackendError(f"{url}: HTTP {exc.code}", unavailable=exc.code in (502, 503, 504)) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise BackendError(f"{url}: {exc}", unavailable=True) from None


class OllamaBackend(LLMBackend):
    """Ollama native /api/chat: schema-constrained `format` and per-request num_ctx."""

    def __init__(self, cfg: Backend, temperature: float = 0.0):
        self.name, self.url, self.model = cfg.name, (cfg.url or "http://localhost:11434").rstrip("/"), cfg.model
        self.num_ctx = cfg.num_ctx or 8192
        self.window = self.num_ctx
        self.timeout = cfg.timeout
        self.temperature = temperature
        self.keep_alive = cfg.keep_alive

    def health(self, timeout: int = 10) -> None:
        """Reachable, and serving this model (a server without it is not a candidate)."""
        tags = _get(f"{self.url}/api/tags", {}, timeout)
        names = {n for m in tags.get("models", []) for n in (m.get("name"), m.get("model")) if n}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise BackendError(f"{self.url} does not have model {self.model!r}")
        self.digest = next((m.get("digest") for m in tags.get("models", [])
                            if self.model in (m.get("name"), m.get("model"))
                            or f"{self.model}:latest" in (m.get("name"), m.get("model"))), None)

    def warm_up(self) -> Optional[float]:
        """An empty generate loads the model (and pays first-use GPU setup) once,
        with the same num_ctx as the real requests so it is not reloaded."""
        body = {"model": self.model, "prompt": "", "stream": False, "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.num_ctx}}
        d = _post(f"{self.url}/api/generate", body, {}, self.timeout)
        dur = d.get("load_duration") or d.get("total_duration")
        return dur / 1e9 if dur else None

    def complete_json(self, system, user, schema, max_tokens=2048) -> Completion:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
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
        self.window = cfg.context
        self.headers: Dict[str, str] = {}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise BackendError(f"backend {cfg.name!r}: environment variable {cfg.api_key_env} is not set")
            self.headers["Authorization"] = f"Bearer {key}"

    def health(self, timeout: int = 10) -> None:
        """Reachable, and serving this model (vLLM serves exactly the ids in /models)."""
        models = _get(f"{self.url}/models", self.headers, timeout)
        ids = [m.get("id") for m in models.get("data", [])]
        if ids and self.model not in ids:
            raise BackendError(f"{self.url} serves {', '.join(map(str, ids))}, not {self.model!r}")
        served = next((m.get("max_model_len") for m in models.get("data", []) if m.get("id") == self.model), None)
        if self.window is None and isinstance(served, int):
            self.window = served                     # vLLM reports the window it was started with

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


def ollama_remote_host(url: str, model: str, timeout: int = 10) -> Optional[str]:
    """Where Ollama says `model` actually runs: e.g. "https://ollama.com" for cloud
    models proxied through the local daemon, None for a local model or if the
    daemon cannot be asked. A stronger signal than the model-name patterns."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=timeout) as resp:
            models = json.loads(resp.read()).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for m in models:
        if model in (m.get("name"), m.get("model")):
            return m.get("remote_host") or None
    return None


def ollama_model_location(url: str, model: str, timeout: int = 5) -> Optional[str]:
    """"local", the remote host Ollama proxies `model` to, or None if the daemon
    cannot be asked or does not list the model (then nothing is known)."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=timeout) as resp:
            models = json.loads(resp.read()).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None
    names = {model, model if ":" in model else model + ":latest"}
    for m in models:
        if names & {m.get("name"), m.get("model")}:
            return m.get("remote_host") or "local"
    return None


def make_backend(cfg: Backend, temperature: float = 0.0) -> LLMBackend:
    if cfg.kind == "ollama":
        return OllamaBackend(cfg, temperature)
    return OpenAICompatBackend(cfg, temperature)
