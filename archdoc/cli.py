"""archdoc command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from .extract import extract_python


def cmd_extract(args: argparse.Namespace) -> int:
    ir = extract_python(args.repo, exclude=args.exclude)
    data = json.dumps(ir.to_dict(), indent=2, sort_keys=False)
    if args.stdout:
        print(data)
        return 0

    state_dir = os.path.join(args.output, ".archdoc")
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "ir.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)

    n_cls = sum(1 for _ in ir.iter_classes())
    n_fn = sum(1 for _ in ir.iter_functions())
    errors = [m.path for m in ir.modules if m.parse_error]
    kinds = Counter(i.kind for i in ir.interfaces)
    print(f"{ir.repo}: {len(ir.modules)} modules, {n_cls} classes, {n_fn} functions/methods")
    print(f"  interfaces: {len(ir.interfaces)} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items())) or 'none'}), "
          f"data types: {len(ir.data_types)}")
    if errors:
        print(f"  [WARN] {len(errors)} file(s) could not be parsed: {', '.join(errors[:5])}"
              f"{' ...' if len(errors) > 5 else ''}")
    print(f"  wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="archdoc", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="Extract IR and interface inventory (no LLM)")
    p.add_argument("repo", help="Path to the repository to analyse")
    p.add_argument("-o", "--output", default="archdoc_out", help="Output directory")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="Repo-relative glob to skip (repeatable), e.g. 'tests/*'")
    p.add_argument("--stdout", action="store_true", help="Print IR JSON instead of writing it")
    p.set_defaults(func=cmd_extract)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
