#!/usr/bin/env python3
"""
build_architecture_summary.py

Reduce facts.jsonl into a compact architecture_summary.json that supports ALL UML diagram types.

It infers:
- components (module groupings)
- internal dependency edges (imports/calls)
- key entities/models (dataclasses/pydantic-ish, enums)
- entrypoints (routes/tasks/commands/mains)
- external systems / IO kinds
- candidate state machines (status/state fields + transition-like methods)
- runtime roles hints (server/worker/cli) from imports/calls/entrypoints
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def module_prefix(mod: str, depth: int = 2) -> str:
    parts = mod.split(".")
    return ".".join(parts[: min(depth, len(parts))])


def main():
    ap = argparse.ArgumentParser(description="Build architecture_summary.json from facts.jsonl")
    ap.add_argument("--facts", required=True, help="facts.jsonl path")
    ap.add_argument("--output", "-o", default="uml_out/architecture_summary.json")
    ap.add_argument("--component-depth", type=int, default=2, help="Module prefix depth for grouping")
    args = ap.parse_args()

    facts = read_jsonl(args.facts)
    if not facts:
        raise SystemExit("No facts loaded")

    # Group modules into components by prefix
    comp_modules: Dict[str, Set[str]] = defaultdict(set)
    file_by_mod: Dict[str, str] = {}

    # Aggregates
    entrypoints = {"routes": [], "tasks": [], "commands": [], "mains": []}
    external_imports: Set[str] = set()
    io_kinds: Set[str] = set()
    io_endpoints: Set[str] = set()
    io_queues: Set[str] = set()
    io_buckets: Set[str] = set()
    env_vars: Set[str] = set()

    classes = []
    entities = []
    enums = []
    state_candidates = []
    dep_edges: Set[Tuple[str, str, str]] = set()  # (src_component, dst_component, kind)

    # For call/import linking
    comp_of_mod: Dict[str, str] = {}

    for r in facts:
        mod = r.get("module", "")
        path = r.get("path", "")
        if mod:
            file_by_mod[mod] = path
            c = module_prefix(mod, depth=args.component_depth)
            comp_modules[c].add(mod)
            comp_of_mod[mod] = c

    # Helper to map an imported module string to a known component
    def find_component_for_import(imp: str) -> str:
        # try exact module
        if imp in comp_of_mod:
            return comp_of_mod[imp]
        # try prefix matches
        parts = imp.split(".")
        for d in range(len(parts), 0, -1):
            pref = ".".join(parts[:d])
            if pref in comp_of_mod:
                return comp_of_mod[pref]
        # fallback: group by prefix even if not in repo map
        return module_prefix(imp, depth=args.component_depth)

    # Process records
    for r in facts:
        mod = r.get("module", "")
        src_comp = comp_of_mod.get(mod, module_prefix(mod, depth=args.component_depth))

        # Entrypoints
        ep = r.get("entrypoints", {})
        for k in entrypoints.keys():
            for item in ep.get(k, []) or []:
                entrypoints[k].append({"module": mod, **item})

        # Imports -> component deps
        im = r.get("imports", {})
        for imp in (im.get("modules", []) or []) + (im.get("from", []) or []):
            dst_comp = find_component_for_import(imp)
            if dst_comp and dst_comp != src_comp:
                dep_edges.add((src_comp, dst_comp, "import"))

        # Calls -> component deps (very heuristic)
        calls = r.get("calls", {}).get("targets", []) or []
        for ct in calls:
            # if call looks like package.module.func, map to component
            if "." in ct:
                maybe_mod = ".".join(ct.split(".")[:-1])
                dst_comp = find_component_for_import(maybe_mod)
                if dst_comp and dst_comp != src_comp:
                    dep_edges.add((src_comp, dst_comp, "call"))

        # IO and externals
        external_imports.update(im.get("external", []) or [])
        io = r.get("io", {})
        io_kinds.update(io.get("kinds", []) or [])
        io_endpoints.update(io.get("endpoints", []) or [])
        io_queues.update(io.get("queues", []) or [])
        io_buckets.update(io.get("buckets", []) or [])

        env_vars.update(r.get("env", []) or [])

        # Classes/entities/enums/state
        for c in (r.get("defines", {}).get("classes", []) or []):
            c_rec = {
                "name": c.get("name"),
                "module": mod,
                "bases": c.get("bases", []),
                "decorators": c.get("decorators", []),
                "methods": c.get("methods", []),
            }
            classes.append(c_rec)

            if c.get("dataclass"):
                entities.append({"name": c.get("name"), "module": mod, "kind": "dataclass"})
            if c.get("enum"):
                enums.append({"name": c.get("name"), "module": mod, "kind": "enum"})

        st = r.get("state", {})
        if st.get("fields") or st.get("transitions") or st.get("enums"):
            state_candidates.append(
                {
                    "module": mod,
                    "fields": st.get("fields", []),
                    "transitions": st.get("transitions", []),
                    "enums": st.get("enums", []),
                }
            )

    # Role hints
    roles: Dict[str, Set[str]] = defaultdict(set)
    for comp, mods in comp_modules.items():
        # Heuristic: if any module in comp has routes => api
        has_routes = any(ep["module"].startswith(comp) for ep in entrypoints["routes"])
        has_tasks = any(ep["module"].startswith(comp) for ep in entrypoints["tasks"])
        has_cmds = any(ep["module"].startswith(comp) for ep in entrypoints["commands"])
        if has_routes:
            roles[comp].add("api")
        if has_tasks:
            roles[comp].add("worker")
        if has_cmds:
            roles[comp].add("cli")
        # Also infer server if it imports uvicorn/gunicorn via external imports? (coarse)
        # We'll just tag "service" if it has api/worker/cli.
        if roles[comp]:
            roles[comp].add("service")

    summary = {
        "components": [
            {
                "name": comp,
                "modules": sorted(list(mods)),
                "roles": sorted(list(roles.get(comp, set()))),
            }
            for comp, mods in sorted(comp_modules.items(), key=lambda kv: kv[0])
        ],
        "dependency_edges": [
            {"from": a, "to": b, "kind": k} for (a, b, k) in sorted(dep_edges)
        ],
        "entrypoints": entrypoints,
        "external_imports": sorted(list(external_imports)),
        "io": {
            "kinds": sorted(list(io_kinds)),
            "endpoints": sorted(list(io_endpoints))[:200],
            "queues": sorted(list(io_queues))[:200],
            "buckets": sorted(list(io_buckets))[:200],
        },
        "env_vars": sorted(list(env_vars)),
        "classes": classes,
        "entities": entities,
        "enums": enums,
        "state_candidates": state_candidates,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary: {os.path.abspath(args.output)}")
    print(f"Components: {len(summary['components'])}, deps: {len(summary['dependency_edges'])}, classes: {len(classes)}")


if __name__ == "__main__":
    main()
