"""
Egress policy: what may leave the machine, per trust tier.

Configuration is read from two TOML files, later overriding earlier:
  1. user config  ~/.config/archdoc/config.toml  (backends, trust settings)
  2. repo config  <repo>/.archdoc.toml           (policy, paths)

Example:

    [policy.cloud]
    docstrings = "first_line"

    [paths]
    exclude   = ["tests/*"]
    sensitive = ["src/pricing/*"]

    [trust]
    local_hosts = ["dgx.lan", "100.64.0.0/10"]

    [backends.dgx]
    url = "http://10.0.0.5:8000/v1"
    model = "mistral-medium"
    trust = "local"
"""

from __future__ import annotations

import copy
import fnmatch
import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

from .trust import CLOUD, DEFAULT_CLOUD_MODEL_PATTERNS, LOCAL, TIERS, Backend

USER_CONFIG = os.path.join(os.path.expanduser("~"), ".config", "archdoc", "config.toml")
REPO_CONFIG = ".archdoc.toml"

LEVELS = ("none", "interfaces", "callgraph", "control_flow", "source")


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class EgressPolicy:
    level: str                  # none | interfaces | callgraph | control_flow | source (a ceiling; see archdoc.packing)
    pseudonymize: str           # off | internal | all
    docstrings: str             # full | first_line | none
    interface_literals: str     # real | alias | none   (route paths, CLI flags, env names, constraints)
    defaults: str               # simple_literals | presence | none
    members: str                # all | public           (protected/private methods and fields)
    on_secret: str              # drop | abort

    def level_at_least(self, level: str) -> bool:
        return LEVELS.index(self.level) >= LEVELS.index(level)


_CHOICES = {
    "level": LEVELS,
    "pseudonymize": ("off", "internal", "all"),
    "docstrings": ("full", "first_line", "none"),
    "interface_literals": ("real", "alias", "none"),
    "defaults": ("simple_literals", "presence", "none"),
    "members": ("all", "public"),
    "on_secret": ("drop", "abort"),
}

TIER_DEFAULTS: Dict[str, EgressPolicy] = {
    LOCAL: EgressPolicy(
        level="source", pseudonymize="off", docstrings="first_line", interface_literals="real",
        defaults="simple_literals", members="public", on_secret="drop",
    ),
    CLOUD: EgressPolicy(
        level="callgraph", pseudonymize="internal", docstrings="none", interface_literals="alias",
        defaults="presence", members="public", on_secret="abort",
    ),
}


@dataclass
class Config:
    policies: Dict[str, EgressPolicy] = field(default_factory=lambda: dict(TIER_DEFAULTS))
    exclude: List[str] = field(default_factory=list)
    sensitive: List[str] = field(default_factory=list)
    secrets_scan: bool = True
    local_hosts: List[str] = field(default_factory=list)
    cloud_model_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_CLOUD_MODEL_PATTERNS))
    backends: Dict[str, Backend] = field(default_factory=dict)
    # [llm] prefer: backends to try in order; the first reachable one serving its model is used
    llm_prefer: List[str] = field(default_factory=list)
    # whether that search may pick a cloud backend; never implied
    llm_allow_cloud_fallback: bool = False
    # how payloads are written for the model: compact "text" (default) or "json"
    payload_encoding: str = "text"
    # [overlap]: verbatim-overlap guard on model-written text (DESIGN.md 2.5)
    overlap_mode: str = "reject"
    overlap_min_words: int = 8
    # [privacy] cloud: "on" pseudonymizes what a cloud model is sent (the cloud
    # policy); "off" sends a cloud model what a local one gets, for code that
    # needs no protecting. Only a repo's own config (or --config) can turn it off.
    privacy_cloud: str = "on"
    # [harness]: where a coding agent's model runs when archdoc cannot tell
    # (auto: find out, else cloud), and the session's output dir and window
    harness_model: str = "auto"
    harness_out: Optional[str] = None
    harness_window: Optional[int] = None
    sources: List[str] = field(default_factory=list)   # config files that were read

    def policy_for(self, tier: str) -> EgressPolicy:
        if tier not in TIERS:
            raise PolicyError(f"unknown tier {tier!r}")
        return self.policies[tier]

    def is_sensitive(self, rel_path: str) -> bool:
        return any(fnmatch.fnmatch(rel_path, pat) for pat in self.sensitive)

    def backend(self, name: str) -> Backend:
        if name not in self.backends:
            known = ", ".join(sorted(self.backends)) or "none configured"
            raise PolicyError(f"unknown backend {name!r} (known: {known})")
        return self.backends[name]


def stricter(a: EgressPolicy, b: EgressPolicy) -> EgressPolicy:
    """Field-wise most restrictive combination (used for sensitive paths)."""
    order = {
        "level": list(LEVELS),
        "pseudonymize": ["all", "internal", "off"],
        "docstrings": ["none", "first_line", "full"],
        "interface_literals": ["none", "alias", "real"],
        "defaults": ["none", "presence", "simple_literals"],
        "members": ["public", "all"],
        "on_secret": ["abort", "drop"],
    }
    vals = {}
    for f in fields(EgressPolicy):
        x, y = getattr(a, f.name), getattr(b, f.name)
        vals[f.name] = min(x, y, key=order[f.name].index)
    return EgressPolicy(**vals)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_TOP_KEYS = {"policy", "paths", "secrets", "trust", "backends", "llm", "overlap", "privacy", "harness"}
PRIVACY = ("on", "off")
HARNESS_MODELS = ("auto", "local", "cloud")


def _read_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise PolicyError(f"{path}: unknown section(s) {', '.join(sorted(unknown))}")
    return data


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _policy_from(tier: str, overrides: Dict[str, Any], src: str) -> EgressPolicy:
    pol = TIER_DEFAULTS[tier]
    for key, value in overrides.items():
        if key not in _CHOICES:
            raise PolicyError(f"{src}: [policy.{tier}] unknown key {key!r}")
        if value not in _CHOICES[key]:
            raise PolicyError(f"{src}: [policy.{tier}] {key} = {value!r}; expected one of {_CHOICES[key]}")
        pol = replace(pol, **{key: value})
    return pol


def load_config(repo_root: Optional[str] = None, user_config: Optional[str] = USER_CONFIG,
                extra: Optional[str] = None) -> Config:
    """Read user config, then repo config, then an explicit --config file."""
    paths = [p for p in (user_config,
                         os.path.join(repo_root, REPO_CONFIG) if repo_root else None,
                         extra) if p]
    data: Dict[str, Any] = {}
    sources: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            part = _read_toml(p)
            if p == user_config and part.get("privacy", {}).get("cloud") == "off":
                # one switch for every repo is too easy to forget about
                raise PolicyError(f"{p}: [privacy] cloud = \"off\" belongs in the repo's own {REPO_CONFIG}, "
                                  "not the user config")
            data = _merge(data, part)
            sources.append(p)
        elif p == extra:
            raise PolicyError(f"config file not found: {p}")
    src = ", ".join(sources) or "<defaults>"

    cfg = Config(sources=sources)
    policy = data.get("policy", {})
    unknown_tiers = set(policy) - set(TIERS)
    if unknown_tiers:
        raise PolicyError(f"{src}: unknown policy tier(s) {', '.join(sorted(unknown_tiers))}")
    cfg.policies = {t: _policy_from(t, policy.get(t, {}), src) for t in TIERS}

    paths_sec = data.get("paths", {})
    cfg.exclude = list(paths_sec.get("exclude", []))
    cfg.sensitive = list(paths_sec.get("sensitive", []))
    cfg.secrets_scan = bool(data.get("secrets", {}).get("scan", True))
    trust = data.get("trust", {})
    cfg.local_hosts = list(trust.get("local_hosts", []))
    cfg.cloud_model_patterns = list(trust.get("cloud_model_patterns", DEFAULT_CLOUD_MODEL_PATTERNS))

    for name, b in data.get("backends", {}).items():
        unknown = set(b) - {"url", "model", "trust", "api_key_env", "kind", "num_ctx", "timeout", "keep_alive",
                            "context"}
        if unknown:
            raise PolicyError(f"{src}: [backends.{name}] unknown key(s) {', '.join(sorted(unknown))}")
        if "api_key" in b:
            raise PolicyError(f"{src}: put the key in an env var and set api_key_env instead")
        if b.get("kind", "openai") not in ("openai", "ollama"):
            raise PolicyError(f"{src}: [backends.{name}] kind must be 'openai' or 'ollama'")
        cfg.backends[name] = Backend(name=name, **b)

    llm = data.get("llm", {})
    unknown = set(llm) - {"prefer", "allow_cloud_fallback", "encoding"}
    if unknown:
        raise PolicyError(f"{src}: [llm] unknown key(s) {', '.join(sorted(unknown))}")
    cfg.llm_prefer = list(llm.get("prefer", []))
    missing = [n for n in cfg.llm_prefer if n not in cfg.backends]
    if missing:
        raise PolicyError(f"{src}: [llm] prefer names undefined backend(s): {', '.join(missing)}")
    cfg.llm_allow_cloud_fallback = bool(llm.get("allow_cloud_fallback", False))
    cfg.payload_encoding = llm.get("encoding", "text")
    if cfg.payload_encoding not in ("text", "json"):
        raise PolicyError(f"{src}: [llm] encoding must be 'text' or 'json'")

    ov = data.get("overlap", {})
    unknown = set(ov) - {"mode", "min_words"}
    if unknown:
        raise PolicyError(f"{src}: [overlap] unknown key(s) {', '.join(sorted(unknown))}")
    cfg.overlap_mode = ov.get("mode", "reject")
    if cfg.overlap_mode not in ("reject", "flag", "off"):
        raise PolicyError(f"{src}: [overlap] mode must be 'reject', 'flag' or 'off'")
    cfg.overlap_min_words = int(ov.get("min_words", 8))
    if cfg.overlap_min_words < 4:
        raise PolicyError(f"{src}: [overlap] min_words below 4 would flag ordinary phrases")

    pv = data.get("privacy", {})
    unknown = set(pv) - {"cloud"}
    if unknown:
        raise PolicyError(f"{src}: [privacy] unknown key(s) {', '.join(sorted(unknown))}")
    cfg.privacy_cloud = pv.get("cloud", "on")
    if cfg.privacy_cloud not in PRIVACY:
        raise PolicyError(f"{src}: [privacy] cloud must be 'on' or 'off'")

    hs = data.get("harness", {})
    unknown = set(hs) - {"model", "out", "window"}
    if unknown:
        raise PolicyError(f"{src}: [harness] unknown key(s) {', '.join(sorted(unknown))}")
    cfg.harness_model = hs.get("model", "auto")
    if cfg.harness_model not in HARNESS_MODELS:
        raise PolicyError(f"{src}: [harness] model must be one of {', '.join(HARNESS_MODELS)}")
    cfg.harness_out = hs.get("out")
    cfg.harness_window = int(hs["window"]) if "window" in hs else None
    return cfg


def policy_tier(model_tier: str, privacy: str) -> str:
    """The policy a request is built with: a cloud model gets the cloud policy
    unless privacy is off for this repo."""
    return LOCAL if model_tier == LOCAL or privacy == "off" else CLOUD
