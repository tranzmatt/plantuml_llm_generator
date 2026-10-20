"""
Trust tiers: is a backend `local` (hardware the user controls) or `cloud`?

Rules (docs/DESIGN.md section 3.2):
  - a model name matching a cloud pattern (Ollama "*-cloud" tags) is cloud,
    whatever the URL says: those are proxied through the local daemon
  - a backend with no declared tier is cloud
  - a backend declared local must resolve only to private, loopback or
    link-local addresses, or match the `local_hosts` allowlist; otherwise
    classification fails rather than silently sending real names out

`local_hosts` entries may be hostnames, IPs or CIDR ranges. Note that
CGNAT space (100.64.0.0/10, used by e.g. Tailscale) is *not* private by
Python's definition and must be allowlisted explicitly.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional
from urllib.parse import urlparse

LOCAL = "local"
CLOUD = "cloud"
TIERS = (LOCAL, CLOUD)

DEFAULT_CLOUD_MODEL_PATTERNS = ["*-cloud", "*:cloud"]


class TrustError(Exception):
    """A backend's declared tier could not be verified."""


@dataclass
class Backend:
    name: str
    url: Optional[str] = None          # None for in-process backends (vllm offline)
    model: Optional[str] = None
    trust: Optional[str] = None        # declared tier; None means cloud
    api_key_env: Optional[str] = None  # name of the env var holding the key, never the key


@dataclass
class TrustDecision:
    tier: str
    reasons: List[str] = field(default_factory=list)


Resolver = Callable[[str], List[str]]


def _system_resolve(host: str) -> List[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


def _address_is_local(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _host_listed(host: str, local_hosts: Iterable[str]) -> bool:
    return any(entry.lower() == host.lower() for entry in local_hosts)


def _addrs_in_networks(addrs: List[str], local_hosts: Iterable[str]) -> bool:
    """True if every address falls inside one allowlisted IP / CIDR entry."""
    if not addrs:
        return False
    for entry in local_hosts:
        try:
            net = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if all(ipaddress.ip_address(a.split("%")[0]) in net for a in addrs):
            return True
    return False


def classify(
    backend: Backend,
    local_hosts: Iterable[str] = (),
    cloud_model_patterns: Iterable[str] = DEFAULT_CLOUD_MODEL_PATTERNS,
    resolve: Resolver = _system_resolve,
) -> TrustDecision:
    if backend.trust is not None and backend.trust not in TIERS:
        raise TrustError(f"backend {backend.name!r}: unknown trust {backend.trust!r} (use local or cloud)")

    model = backend.model or ""
    pattern = next((p for p in cloud_model_patterns if fnmatch.fnmatch(model, p)), None)
    if pattern:
        reasons = [f"model {model!r} matches cloud pattern {pattern!r}"]
        if backend.trust == LOCAL:
            reasons.append("declared local, overridden: inference for this model is remote")
        return TrustDecision(CLOUD, reasons)

    if backend.trust is None:
        return TrustDecision(CLOUD, ["no trust declared; defaulting to cloud"])
    if backend.trust == CLOUD:
        return TrustDecision(CLOUD, ["declared cloud"])

    if backend.url is None:
        return TrustDecision(LOCAL, ["in-process backend"])
    host = urlparse(backend.url).hostname
    if not host:
        raise TrustError(f"backend {backend.name!r}: cannot parse host from {backend.url!r}")
    local_hosts = list(local_hosts)
    if _host_listed(host, local_hosts):
        return TrustDecision(LOCAL, [f"host {host!r} is in local_hosts"])
    try:
        addrs = [host] if _is_ip(host) else resolve(host)
    except OSError as exc:
        raise TrustError(f"backend {backend.name!r}: cannot resolve {host!r} to verify it is local ({exc})")
    if not addrs:
        raise TrustError(f"backend {backend.name!r}: {host!r} resolved to no addresses")
    if _addrs_in_networks(addrs, local_hosts):
        return TrustDecision(LOCAL, [f"{host} ({', '.join(addrs)}) is in local_hosts"])
    public = [a for a in addrs if not _address_is_local(a)]
    if public:
        raise TrustError(
            f"backend {backend.name!r} is declared local but {host!r} resolves to non-private "
            f"address(es) {', '.join(public)}. Add it to [trust] local_hosts if you control it, "
            f"or declare it cloud."
        )
    return TrustDecision(LOCAL, [f"{host} resolves to private address(es) {', '.join(addrs)}"])


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%")[0])
        return True
    except ValueError:
        return False
