import os
from typing import Dict

from .plantuml_sanitizer import sanitize_and_validate_plantuml


def write_diagrams(
    output_dir: str,
    repo_name: str,
    diagrams: Dict[str, str],
    validate: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for dtype, puml in diagrams.items():
        cleaned = sanitize_and_validate_plantuml(puml, do_validate=validate)
        filename = f"{repo_name}_{dtype}.puml"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"[diagram_writer] Wrote {path}")
