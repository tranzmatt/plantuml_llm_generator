"""Utilities to clean and optionally validate PlantUML diagrams.

Goals:
- Fix common LLM mistakes like: A --> "uses" B
- Fix quoted nodes used as endpoints: A --> "Image API"
- Optionally run `plantuml -checkonly` for final validation.
"""

import os
import re
import subprocess
import tempfile
from typing import Tuple


# Pattern: A --> "label" B   (invalid)
RELATION_LABEL_IN_MIDDLE = re.compile(
    r"^\s*(\w+)\s*([-.]+>)\s*\"([^\"]+)\"\s*(\w+)\s*$"
)

# Pattern: "Node" --> B
QUOTED_LEFT_NODE = re.compile(
    r"^\s*\"([^\"]+)\"\s*([-.]+>)\s*(\w+)\s*$"
)

# Pattern: A --> "Node"
QUOTED_RIGHT_NODE = re.compile(
    r"^\s*(\w+)\s*([-.]+>)\s*\"([^\"]+)\"\s*$"
)


def _alias_from_label(label: str) -> str:
    alias = re.sub(r"\W+", "", label)
    if not alias:
        alias = "Node"
    return alias[0].lower() + alias[1:]


def fix_relationship_label_position(text: str) -> str:
    """Fix A --> "label" B -> A --> B : "label"."""
    lines = text.splitlines()
    fixed = []
    for line in lines:
        m = RELATION_LABEL_IN_MIDDLE.match(line)
        if m:
            left, arrow, label, right = m.groups()
            new_line = f'{left} {arrow} {right} : "{label}"'
            fixed.append(new_line)
        else:
            fixed.append(line)
    return "\n".join(fixed)


def fix_quoted_endpoint_nodes(text: str) -> str:
    """Introduce aliases for quoted endpoints used as nodes.

    "Image API" --> B
    becomes:
        "Image API" as imageAPI
        imageAPI --> B

    A --> "RabbitMQ"
    becomes:
        "RabbitMQ" as rabbitmq
        A --> rabbitmq
    """
    alias_map = {}
    new_lines = []

    for line in text.splitlines():
        # Left quoted node
        m_left = QUOTED_LEFT_NODE.match(line)
        if m_left:
            label, arrow, right = m_left.groups()
            alias = alias_map.get(label)
            if not alias:
                alias = _alias_from_label(label)
                alias_map[label] = alias
                new_lines.append(f'"{label}" as {alias}')
            new_lines.append(f'{alias} {arrow} {right}')
            continue

        # Right quoted node
        m_right = QUOTED_RIGHT_NODE.match(line)
        if m_right:
            left, arrow, label = m_right.groups()
            alias = alias_map.get(label)
            if not alias:
                alias = _alias_from_label(label)
                alias_map[label] = alias
                new_lines.append(f'"{label}" as {alias}')
            new_lines.append(f'{left} {arrow} {alias}')
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


def validate_plantuml(text: str) -> Tuple[bool, str]:
    """Validate PlantUML using local 'plantuml -checkonly'.

    Returns (ok, message). If plantuml is not installed, ok=True with a warning.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".puml", delete=False) as tmp:
            tmp.write(text.encode("utf-8"))
            tmp.flush()
            tmp_name = tmp.name

        try:
            proc = subprocess.run(
                ["plantuml", "-checkonly", "-quiet", tmp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return True, "[plantuml_sanitizer] plantuml not found in PATH; skipping validation."

        ok = proc.returncode == 0
        msg = (proc.stdout or "") + (proc.stderr or "")
        return ok, msg.strip()
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass


def sanitize_and_validate_plantuml(text: str, do_validate: bool = True) -> str:
    """Apply common fixes and optionally validate with PlantUML."""
    cleaned = fix_relationship_label_position(text)
    cleaned = fix_quoted_endpoint_nodes(cleaned)

    if do_validate:
        ok, msg = validate_plantuml(cleaned)
        if not ok:
            raise ValueError(
                "PlantUML validation failed. Details:\n" + msg + "\n\nSanitized output:\n" + cleaned
            )
        if msg:
            print(msg)

    return cleaned
