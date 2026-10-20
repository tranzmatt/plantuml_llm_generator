"""
PlantUML syntax checking via the real PlantUML (`-syntax`, reads stdin).

Located, in order: $ARCHDOC_PLANTUML (a command, or a path to a .jar),
$PLANTUML_JAR, then `plantuml` on PATH. If none is found, checks report
`available = False` instead of failing, so rendering never depends on Java.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SyntaxResult:
    ok: bool
    available: bool = True
    diagram_type: Optional[str] = None
    line: Optional[int] = None       # 1-based line of the error, counting @startuml as 1
    message: Optional[str] = None


def plantuml_command() -> Optional[List[str]]:
    for var in ("ARCHDOC_PLANTUML", "PLANTUML_JAR"):
        value = os.environ.get(var)
        if not value:
            continue
        if value.endswith(".jar"):
            java = shutil.which("java")
            return [java, "-Djava.awt.headless=true", "-jar", value] if java and os.path.isfile(value) else None
        return shlex.split(value)
    exe = shutil.which("plantuml")
    return [exe] if exe else None


def check_syntax(text: str, command: Optional[List[str]] = None, timeout: int = 60) -> SyntaxResult:
    cmd = command or plantuml_command()
    if not cmd:
        return SyntaxResult(ok=True, available=False)
    proc = subprocess.run(cmd + ["-syntax"], input=text, capture_output=True, text=True, timeout=timeout)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if lines and lines[0] != "ERROR":
        return SyntaxResult(ok=True, diagram_type=lines[0])
    line_no = int(lines[1]) + 1 if len(lines) > 1 and lines[1].isdigit() else None
    message = " ".join(lines[2:]) if len(lines) > 2 else (proc.stderr.strip() or "unknown error")
    return SyntaxResult(ok=False, line=line_no, message=message)
