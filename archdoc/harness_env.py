"""
Where a coding agent's model runs (DESIGN.md 13.2).

archdoc never sees the agent's model: the harness talks to it. What it can see
is the environment the harness started the MCP server with. `ollama launch`
passes the model there (Claude Code: ANTHROPIC_BASE_URL and the model aliases;
opencode: the whole config in OPENCODE_CONFIG_CONTENT), and Ollama says
whether a model it serves runs on its cloud. Anything archdoc cannot find out
counts as cloud unless the user's config says local.

This is checked once, at startup: a model switched mid-session is not seen.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from .backends import ollama_model_location
from .policy import Config
from .trust import CLOUD, LOCAL, Backend, TrustError, classify

_CLAUDE_MODELS = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                  "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


@dataclass
class ModelTier:
    tier: str                            # what archdoc treats the agent's model as
    harness: Optional[str] = None        # claude | opencode, when recognized
    models: List[str] = field(default_factory=list)
    detected: Optional[str] = None       # local | cloud | None (not found out)
    reasons: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        what = f"{self.harness} with {', '.join(self.models)}" if self.models else (self.harness or "harness")
        how = "detected" if self.detected else "not detected, from config"
        return f"{what}: {self.tier} ({how})"


def _api_root(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def locate(url: str, model: str, cfg: Config) -> Tuple[Optional[str], str]:
    """(local | cloud | None, why) for one model behind one endpoint."""
    pattern = next((p for p in cfg.cloud_model_patterns if fnmatch.fnmatch(model, p)), None)
    if pattern:
        return CLOUD, f"{model!r} matches cloud pattern {pattern!r}"
    try:
        classify(Backend(name="harness", url=url, model=model, trust=LOCAL), cfg.local_hosts,
                 cfg.cloud_model_patterns)
    except TrustError:
        return CLOUD, f"{url} is not a local address"
    where = ollama_model_location(_api_root(url), model)
    if where == "local":
        return LOCAL, f"Ollama at {url} runs {model!r} locally"
    if where:
        return CLOUD, f"Ollama at {url} runs {model!r} at {where}"
    # a local gateway can still forward to a cloud API: nothing is known
    return None, f"{url} is local, but it is not an Ollama that lists {model!r}"


def _combine(found: List[Tuple[Optional[str], str]]) -> Optional[str]:
    tiers = [t for t, _ in found]
    if CLOUD in tiers:
        return CLOUD
    return LOCAL if tiers and all(t == LOCAL for t in tiers) else None


def _claude(env: Mapping[str, str], cfg: Config) -> Tuple[List[str], Optional[str], List[str]]:
    models = list(dict.fromkeys(env[k] for k in _CLAUDE_MODELS if env.get(k)))
    base = env.get("ANTHROPIC_BASE_URL")
    if not base:
        return models, CLOUD, ["no ANTHROPIC_BASE_URL: Claude Code uses Anthropic's API"]
    if not models:
        return models, None, [f"ANTHROPIC_BASE_URL is {base}, but no model is named in the environment"]
    found = [locate(base, m, cfg) for m in models]
    return models, _combine(found), [why for _, why in found]


def _opencode(env: Mapping[str, str], cfg: Config) -> Tuple[List[str], Optional[str], List[str]]:
    try:
        conf = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
    except ValueError:
        return [], None, ["OPENCODE_CONFIG_CONTENT is not JSON"]
    model = conf.get("model")
    if isinstance(model, dict):                      # opencode 2: {"providerID", "model"}
        provider, name = model.get("providerID", ""), model.get("model", "")
    else:
        provider, _, name = str(model or "").partition("/")
    if not name:
        return [], None, ["the opencode config in the environment names no model"]
    pconf = (conf.get("provider") or conf.get("providers") or {}).get(provider, {})
    base = (pconf.get("options") or pconf.get("settings") or {}).get("baseURL")
    if not base:
        return [name], CLOUD, [f"provider {provider!r} has no baseURL here: a hosted provider"]
    tier, why = locate(base, name, cfg)
    return [name], tier, [why]


def detect(env: Mapping[str, str], cfg: Config) -> ModelTier:
    """What the environment shows about the agent's model."""
    if "OPENCODE_CONFIG_CONTENT" in env:
        models, tier, reasons = _opencode(env, cfg)
        return ModelTier(tier or CLOUD, "opencode", models, tier, reasons)
    if "CLAUDE_PROJECT_DIR" in env:            # set for the MCP servers Claude Code starts
        models, tier, reasons = _claude(env, cfg)
        return ModelTier(tier or CLOUD, "claude", models, tier, reasons)
    return ModelTier(CLOUD, None, [], None, ["the harness's model is not in the environment"])


def resolve(env: Mapping[str, str], cfg: Config, claimed: str = "auto") -> ModelTier:
    """The tier to use: what was detected, or what the user claims (CLI, then
    [harness] model) when nothing was. A claim of local never beats a model
    found to be cloud; a claim of cloud always holds."""
    found = detect(env, cfg)
    if claimed == CLOUD:
        found.tier = CLOUD
        found.reasons.append("set to cloud")
    elif found.detected == CLOUD:
        if claimed == LOCAL:
            found.reasons.append("set to local, overridden: the model runs in the cloud")
    elif found.detected == LOCAL:
        found.tier = LOCAL
    elif claimed == LOCAL:
        found.tier = LOCAL
        found.reasons.append("set to local; archdoc cannot check this")
    return found


__all__ = ["ModelTier", "detect", "resolve", "locate"]
