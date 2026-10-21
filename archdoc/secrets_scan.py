"""
Secret scanning: a backstop over anything about to leave the machine.

The primary control is that bodies and arbitrary literals never enter an
egress payload at all; this catches what slips through via defaults,
decorator arguments, constraints and docstrings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

REDACTED = "<redacted>"

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # no trailing boundary: a key glued to more characters ("AKIA...J"[:20]) is still a key
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("url_credentials", re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]+@")),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]{24,}")
ENTROPY_THRESHOLD = 4.2


@dataclass
class Finding:
    kind: str
    start: int
    end: int


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_random(token: str) -> bool:
    if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
        return False
    if token.lower() == token and "_" in token:
        return False       # long snake_case identifiers with a version number, etc.
    return shannon_entropy(token) >= ENTROPY_THRESHOLD


def scan(text: str) -> List[Finding]:
    findings: List[Finding] = []
    for kind, rx in _PATTERNS:
        findings.extend(Finding(kind, m.start(), m.end()) for m in rx.finditer(text))
    covered = [(f.start, f.end) for f in findings]
    for m in _TOKEN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in covered):
            continue
        if _looks_random(m.group()):
            findings.append(Finding("high_entropy", m.start(), m.end()))
    return sorted(findings, key=lambda f: f.start)


def redact(text: str) -> Tuple[str, List[Finding]]:
    findings = scan(text)
    if not findings:
        return text, []
    out, pos = [], 0
    for f in findings:
        if f.start < pos:
            continue
        out.append(text[pos:f.start])
        out.append(REDACTED)
        pos = f.end
    out.append(text[pos:])
    return "".join(out), findings
