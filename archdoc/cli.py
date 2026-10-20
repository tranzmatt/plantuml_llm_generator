"""archdoc command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from .egress import EgressAborted, build_payload, estimate_tokens
from .extract import extract_python
from .extract.python_ast import collect_python_files
from .policy import PolicyError, load_config
from .pseudonym import AliasMap
from .trust import TIERS, TrustError, classify


def _state_dir(output: str) -> str:
    path = os.path.join(output, ".archdoc")
    os.makedirs(path, exist_ok=True)
    return path


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    data = json.dumps(ir.to_dict(), indent=2, sort_keys=False)
    if args.stdout:
        print(data)
        return 0

    path = os.path.join(_state_dir(args.output), "ir.json")
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


def cmd_egress(args: argparse.Namespace) -> int:
    """Show exactly what would be sent to a backend. Never calls a model."""
    cfg = load_config(args.repo, extra=args.config)
    if args.backend:
        backend = cfg.backend(args.backend)
        decision = classify(backend, cfg.local_hosts, cfg.cloud_model_patterns)
        tier, reasons = decision.tier, decision.reasons
        label = f"{backend.name} ({backend.model or '?'} @ {backend.url or 'in-process'})"
    else:
        tier, reasons, label = args.trust, ["set with --trust"], f"--trust {args.trust}"

    exclude = cfg.exclude + args.exclude
    ir = extract_python(args.repo, exclude=exclude)
    alias_path = os.path.join(_state_dir(args.output), "alias_map.json")
    aliases = AliasMap.load(alias_path)
    modules = None
    if args.scope:
        modules = [m.id for m in ir.modules if m.id == args.scope or m.id.startswith(args.scope + ".")]
    payload = build_payload(ir, cfg, tier, aliases, artifact="dry-run", modules=modules)
    aliases.save(alias_path)

    preview = os.path.join(_state_dir(args.output), f"egress_preview_{tier}.json")
    with open(preview, "w", encoding="utf-8") as fh:
        json.dump(payload.data, fh, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(payload.data, indent=2, ensure_ascii=False))
        return 0

    raw_chars = 0
    root = os.path.abspath(args.repo)
    for rel in collect_python_files(root, exclude):
        try:
            raw_chars += os.path.getsize(os.path.join(root, rel))
        except OSError:
            pass
    raw_tokens = estimate_tokens("x" * raw_chars) if raw_chars else 0
    pol = cfg.policy_for(tier)
    print(f"backend : {label}")
    print(f"tier    : {tier}  ({'; '.join(reasons)})")
    print(f"policy  : level={pol.level} pseudonymize={pol.pseudonymize} docstrings={pol.docstrings} "
          f"interface_literals={pol.interface_literals} defaults={pol.defaults} members={pol.members}")
    if cfg.sensitive:
        print(f"sensitive paths (cloud rules always): {', '.join(cfg.sensitive)}")
    print(f"config  : {', '.join(cfg.sources) or 'built-in defaults'}")
    print(f"payload : ~{payload.token_estimate:,} tokens "
          f"(raw source ~{raw_tokens:,} tokens; "
          f"{(payload.token_estimate / raw_tokens * 100 if raw_tokens else 0):.1f}%)")
    print(f"aliases : {len(aliases):,} in {alias_path}")
    if payload.redactions:
        print(f"secrets : {len(payload.redactions)} redacted "
              f"({', '.join(sorted({f'{k} in {w}' for w, k in payload.redactions}))})")
    print(f"preview : {preview}")
    print("nothing was sent (dry run)")
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
    p.add_argument("--config", help="Extra config TOML (overrides user and repo config)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("egress", help="Dry run: show exactly what a backend would receive")
    p.add_argument("repo", help="Path to the repository to analyse")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--backend", help="Backend name from config; its trust tier is verified")
    who.add_argument("--trust", choices=TIERS, help="Assume this tier without a backend")
    p.add_argument("-o", "--output", default="archdoc_out", help="Output directory (alias map, preview)")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    p.add_argument("--scope", help="Only modules under this package, e.g. 'shop.api'")
    p.add_argument("--json", action="store_true", help="Print the payload JSON")
    p.add_argument("--config", help="Extra config TOML (overrides user and repo config)")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Accepted for clarity; egress never sends anything in this version")
    p.set_defaults(func=cmd_egress)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (PolicyError, TrustError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EgressAborted as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
