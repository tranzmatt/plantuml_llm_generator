"""Trust classification, policy loading and secret scanning."""

import pytest

from archdoc.policy import TIER_DEFAULTS, Config, PolicyError, load_config, stricter
from archdoc.secrets_scan import REDACTED, redact, scan
from archdoc.trust import Backend, TrustError, classify


def fake_dns(table):
    def resolve(host):
        if host not in table:
            raise OSError("no such host")
        return table[host]
    return resolve


DNS = fake_dns({
    "localhost": ["127.0.0.1", "::1"],
    "dgx.lan": ["10.0.0.5"],
    "api.anthropic.com": ["160.79.104.10"],
    "box.tailnet": ["100.101.102.103"],
    "mixed.example": ["10.0.0.9", "8.8.8.8"],
})


# -- trust ----------------------------------------------------------------------

@pytest.mark.parametrize("backend, tier", [
    (Backend("a", "http://localhost:11434", "llama3.1:8b", "local"), "local"),
    (Backend("b", "http://dgx.lan:8000/v1", "mistral-medium", "local"), "local"),
    (Backend("c", "http://10.0.0.5:8000/v1", "m", "local"), "local"),
    (Backend("d", None, "m", "local"), "local"),
    (Backend("e", "https://api.anthropic.com", "claude", "cloud"), "cloud"),
    (Backend("f", "http://localhost:11434", "llama3.1:8b", None), "cloud"),
])
def test_classify(backend, tier):
    assert classify(backend, resolve=DNS).tier == tier


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3-coder:cloud"])
def test_ollama_cloud_models_are_cloud_even_on_localhost(model):
    d = classify(Backend("o", "http://localhost:11434", model, "local"), resolve=DNS)
    assert d.tier == "cloud"
    assert "overridden" in d.reasons[1]


def test_declared_local_but_public_fails():
    with pytest.raises(TrustError, match="non-private"):
        classify(Backend("x", "https://api.anthropic.com", "m", "local"), resolve=DNS)


def test_declared_local_with_any_public_address_fails():
    with pytest.raises(TrustError, match="8.8.8.8"):
        classify(Backend("x", "http://mixed.example", "m", "local"), resolve=DNS)


def test_unresolvable_local_fails():
    with pytest.raises(TrustError, match="cannot resolve"):
        classify(Backend("x", "http://nowhere.lan", "m", "local"), resolve=DNS)


def test_cgnat_needs_allowlist():
    b = Backend("t", "http://box.tailnet:8000", "m", "local")
    with pytest.raises(TrustError):
        classify(b, resolve=DNS)
    assert classify(b, local_hosts=["100.64.0.0/10"], resolve=DNS).tier == "local"
    assert classify(b, local_hosts=["box.tailnet"], resolve=DNS).tier == "local"


def test_cidr_allowlist_does_not_match_other_hosts():
    # regression: an allowlisted CIDR must not vouch for an unrelated public host
    with pytest.raises(TrustError):
        classify(Backend("x", "https://api.anthropic.com", "m", "local"),
                 local_hosts=["100.64.0.0/10"], resolve=DNS)


# -- policy ---------------------------------------------------------------------

def test_defaults_without_files(tmp_path):
    cfg = load_config(str(tmp_path), user_config=None)
    assert cfg.policy_for("local") == TIER_DEFAULTS["local"]
    assert cfg.policy_for("cloud").pseudonymize == "internal"
    assert cfg.sources == []


def test_user_then_repo_config(tmp_path):
    user = tmp_path / "user.toml"
    user.write_text('[backends.dgx]\nurl = "http://10.0.0.5:8000/v1"\nmodel = "m"\ntrust = "local"\n'
                    '[trust]\nlocal_hosts = ["dgx.lan"]\n')
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".archdoc.toml").write_text('[policy.cloud]\ndocstrings = "first_line"\n'
                                        '[paths]\nsensitive = ["src/pricing/*"]\n')
    cfg = load_config(str(repo), user_config=str(user))
    assert cfg.backend("dgx").trust == "local"
    assert cfg.local_hosts == ["dgx.lan"]
    assert cfg.policy_for("cloud").docstrings == "first_line"
    assert cfg.policy_for("cloud").pseudonymize == "internal"     # untouched keys keep tier defaults
    assert cfg.is_sensitive("src/pricing/engine.py")
    assert len(cfg.sources) == 2


@pytest.mark.parametrize("toml, msg", [
    ('[policy.cloud]\ndocstring = "none"\n', "unknown key 'docstring'"),
    ('[policy.cloud]\npseudonymize = "maybe"\n', "expected one of"),
    ('[policy.edge]\nlevel = "none"\n', "unknown policy tier"),
    ('[polcy]\n', "unknown section"),
    ('[backends.x]\nurl = "u"\napi_key = "k"\n', "unknown key"),
])
def test_config_errors(tmp_path, toml, msg):
    (tmp_path / ".archdoc.toml").write_text(toml)
    with pytest.raises(PolicyError, match=msg):
        load_config(str(tmp_path), user_config=None)


def test_unknown_backend():
    with pytest.raises(PolicyError, match="unknown backend"):
        Config().backend("nope")


def test_stricter_takes_most_restrictive_field():
    s = stricter(TIER_DEFAULTS["local"], TIER_DEFAULTS["cloud"])
    assert s == TIER_DEFAULTS["cloud"].__class__(
        level="callgraph", pseudonymize="internal", docstrings="none", interface_literals="alias",
        defaults="presence", members="public", on_secret="abort",
    )


# -- secrets --------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("key=AKIAQWERTYUIOPASDFGH", "aws_access_key"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
    ("ghp_" + "a1B2" * 9, "github_token"),
    ("sk-ant-" + "x9Y8z7W6v5U4t3S2r1Q0", "api_key"),
    ("postgres://admin:hunter2@db.internal/x", "url_credentials"),
    ("hf_" + "Ab3" * 11, "huggingface_token"),
    ("token 8fJ2kL9xQ4mN7pR1sT6vW3yZ0bC5dE8g", "high_entropy"),
])
def test_scan_detects(text, kind):
    assert [f.kind for f in scan(text)] == [kind]


@pytest.mark.parametrize("text", [
    "calculate_patent_pending_yield_curve_v2",
    "shop.repo:ItemRepo.get",
    "(item_id: int, verbose: bool = False) -> Item",
    "https://example.com/api/v1/items",
    "@C17 calls @F42",
])
def test_scan_ignores_normal_text(text):
    assert scan(text) == []


def test_redact():
    clean, findings = redact("a AKIAQWERTYUIOPASDFGH b")
    assert clean == f"a {REDACTED} b" and len(findings) == 1
