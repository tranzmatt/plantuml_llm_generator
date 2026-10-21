"""archdoc command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import re
import time
from types import SimpleNamespace

from .backends import BackendError, make_backend, ollama_remote_host
from .coverage import coverage_lines, icd_coverage
from .egress import EgressAborted, EgressLog, build_payload, estimate_tokens
from .extract import extract_python
from .extract.python_ast import collect_python_files
from .policy import PolicyError, load_config
from .pseudonym import AliasMap
from .plan.artifacts import (
    PlanContext, check_sequence, plan_activity, plan_narrative, plan_sequence, plan_usecase,
)
from .activity import build_activity, has_control_flow
from .objects import build_objects
from .states import state_machines
from .usecases import build_usecases
from .render.puml.activity import render_activity_diagram
from .render.puml.object_ import render_object_diagram
from .render.puml.state import render_state_diagram
from .render.puml.usecase import render_usecase_diagram
from .plan.verify import verify_sequence
from .plan.entries import Entry, call_slice, discover_entries, select_entries
from .render.icd import anchor, apply_narrative, build_icd_context, inventory_snapshot, render_icd
from .render.puml.sequence import render_sequence_diagram
from . import provenance
from .overlap import SourceIndex
from .packing import Calibration
from .review import Review, apply_to_icd, icd_evidence, offer_icd, sequence_evidence
from .cache import ResponseCache
from .export import EMBED_MODES, EXPORT_FORMATS, export as export_document, figures, render_images
from .design import DesignOptions, build_design_context, coverage_lines as design_coverage_lines, design_items
from .render.design import render_design
from .testsuite import inventory_excludes
from .flows import flow_entries, trace
from .render.puml import (
    render_class_diagram, render_component_diagram, render_deployment_diagram, render_flow_diagram,
)
from .render.puml.class_ import partition_modules
from .validate import check_syntax, plantuml_command
from .trust import CLOUD, TIERS, Backend, TrustError, classify


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


DETERMINISTIC_DIAGRAMS = ("class", "component", "deployment", "flow", "state", "object", "usecase", "activity")
PLANNED_DIAGRAMS = ("sequence",)
LLM_DIAGRAMS = PLANNED_DIAGRAMS
LABELLED_DIAGRAMS = ("activity", "usecase")      # drawn from the code; an LLM, if any, only writes labels
DEFAULT_URLS = {"ollama": "http://localhost:11434"}


def _candidate_backends(args, cfg):
    """(backends to try in order, whether the user named exactly one)."""
    if getattr(args, "model", None):
        return [Backend(name="adhoc", url=args.url or DEFAULT_URLS.get(args.kind), model=args.model,
                        trust=args.trust, kind=args.kind, num_ctx=args.num_ctx)], True
    requested = getattr(args, "backend", None)
    if requested and requested != "auto":
        names = [n.strip() for n in requested.split(",") if n.strip()]
        return [cfg.backend(n) for n in names], len(names) == 1
    if requested == "auto" and not cfg.llm_prefer:
        raise PolicyError("--backend auto needs [llm] prefer = [...] in the config")
    return [cfg.backends[n] for n in cfg.llm_prefer], False


def _tier_of(backend, cfg, args):
    decision = classify(backend, cfg.local_hosts, cfg.cloud_model_patterns)
    tier, reasons = decision.tier, decision.reasons
    if backend.kind == "ollama" and tier != CLOUD and backend.url:
        remote = ollama_remote_host(backend.url, backend.model or "")
        if remote:
            tier, reasons = CLOUD, reasons + [f"overridden: Ollama reports this model runs at {remote}"]
    if backend.name != "adhoc" and args.trust == CLOUD:
        tier, reasons = CLOUD, reasons + ["forced with --trust cloud"]
    elif backend.name != "adhoc" and args.trust and args.trust != tier:
        raise TrustError(f"backend {backend.name!r} classifies as {tier}; --trust can only make it stricter")
    return tier, reasons


def _llm_context(args, cfg, ir):
    """PlanContext for the first usable backend, or None (LLM outputs are then skipped).

    Candidates come from --model, --backend NAME[,NAME...] / auto, or [llm] prefer.
    Each is trust-classified, health-checked (reachable and serving its model)
    and warmed up; the first that passes is used. A search never lands on a
    cloud backend unless [llm] allow_cloud_fallback = true or that backend was
    named on its own.
    """
    candidates, explicit = _candidate_backends(args, cfg)
    if not candidates:
        return None
    tried = []
    for backend in candidates:
        label = f"{backend.name} ({backend.model} via {backend.kind} @ {backend.url or 'in-process'})"
        try:
            tier, reasons = _tier_of(backend, cfg, args)
        except TrustError as exc:
            if explicit:
                raise
            tried.append(f"{backend.name}: {exc}")
            continue
        if tier == CLOUD and not explicit and not cfg.llm_allow_cloud_fallback:
            tried.append(f"{backend.name}: cloud, and [llm] allow_cloud_fallback is false")
            continue
        llm_backend = make_backend(backend)
        try:
            llm_backend.health()
        except BackendError as exc:
            tried.append(f"{backend.name}: {exc}")
            continue
        print(f"  LLM: {label} (tier {tier}: {'; '.join(reasons)})")
        # the model is loaded before the first request not answered from the
        # response cache (LLMBackend.ensure_ready), so a cached rerun never loads it
        for t in tried:
            print(f"    (skipped {t})")
        state = _state_dir(args.output)
        alias_path = os.path.join(state, "alias_map.json")
        return PlanContext(
            ir=ir, config=cfg, tier=tier, aliases=AliasMap.load(alias_path),
            backend=llm_backend,
            log=EgressLog(os.path.join(state, "egress.jsonl")),
            depth=args.depth, max_nodes=args.max_nodes, alias_path=alias_path,
            overlap=SourceIndex(args.repo, ir, cfg.overlap_min_words, cfg.overlap_mode),
            root=args.repo, calibration=Calibration.load(os.path.join(state, "calibration.json")),
            cache=ResponseCache(os.path.join(state, "cache"), enabled=not getattr(args, "no_cache", False)),
        )
    print("  [SKIP] LLM outputs: no usable backend; writing the rest without it")
    for t in tried:
        print(f"    {t}")
    return None


def _save_plan(args, artifact: str, plan) -> None:
    path = os.path.join(_state_dir(args.output), "plans", re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact) + ".json")
    _write(path, json.dumps(plan, indent=1, ensure_ascii=False))


def _reviewed_sequence(ir, args, e, review, decision):
    """(stem, title, text, note, provenance) for a reviewed plan, or None if its plan file is missing."""
    plan = review.load_plan(e.handler)
    if plan is None:
        print(f"  [WARN] sequence:{e.handler} is {decision['status']} but {review.plan_path(e.handler)} is missing")
        return None
    ids = call_slice(ir, e.handler, args.depth, args.max_nodes)
    stale = review.is_stale(f"sequence:{e.handler}", sequence_evidence(ir, ids))
    issues = check_sequence(plan) or [str(x) for x in verify_sequence(ir, plan, ids, e.handler)]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", e.handler).strip("_")
    stem = f"{ir.repo}_sequence_{slug}"
    title = f"{e.via[0].name}: {e.via[1].name}" if e.via else e.label
    if stale:
        plan = {**plan, "title": f"{plan.get('title') or title} [reviewed before a code change: re-review]"}
    client = "User" if e.via and e.via[0].kind == "cli" else "Client"
    text, warnings = render_sequence_diagram(ir, plan, stem, client_name=client, title=title)
    for w in warnings + [f"reviewed plan disagrees with the code: {x}" for x in issues]:
        print(f"  [WARN] {stem}: {w}")
    items = provenance.sequence_items(ir, SimpleNamespace(plan=plan), ids, e.handler)
    for it in items:
        it.update({"class": "reviewed", "stale": stale})
        it.pop("request", None)
    prov = {"producer": "review", "items": items,
            "extra": {"entry": e.handler, "review": {"status": decision["status"], "stale": stale,
                                                     "plan": os.path.relpath(review.plan_path(e.handler), args.output),
                                                     "disagreements": issues}, "slice": ids}}
    return stem, f"{ir.repo}: {plan.get('title') or e.label}", text, f"(reviewed{', STALE' if stale else ''})", prov


def _sequence_files(ir, args, llm, review):
    """(stem, title, text, note, provenance) for sequence diagrams: reviewed plans as they
    are, the rest planned by the LLM (when there is one) and offered for review."""
    out = []
    try:
        entries = select_entries(ir, args.entry, args.max_entries, args.depth, args.max_nodes) if llm else []
    except ValueError as exc:
        raise PolicyError(str(exc)) from None
    known = {e.handler: e for e in discover_entries(ir)}
    for item_id in sorted(review.items):
        h = item_id.split(":", 1)[1] if item_id.startswith("sequence:") else None
        if h and review.decided(item_id) and h not in {e.handler for e in entries}:
            entries.append(known.get(h) or Entry(h, h))
    if not entries:
        print("  [SKIP] sequence: no entry points with internal calls found (use --entry)")
    for e in entries:
        decision = review.decided(f"sequence:{e.handler}")
        if decision and decision["status"] == "rejected":
            print(f"  [SKIP] sequence for {e.label}: rejected in review")
            continue
        if decision:
            reviewed = _reviewed_sequence(ir, args, e, review, decision)
            if reviewed:
                out.append(reviewed)
                continue
        if llm is None:
            continue
        t0 = time.monotonic()
        result = plan_sequence(llm, e)
        took = time.monotonic() - t0
        if not result.ok:
            print(f"  [FAIL] sequence for {e.label}: {'; '.join(result.errors[:3])} "
                  f"({result.attempts} attempt(s), {took:.0f}s)")
            continue
        # module + function: different modules often both have a `main`
        slug = re.sub(r"[^A-Za-z0-9]+", "_", e.handler).strip("_")
        stem = f"{ir.repo}_sequence_{slug}"
        client = "User" if e.via and e.via[0].kind == "cli" else "Client"
        text, warnings = render_sequence_diagram(
            ir, result.plan, stem, client_name=client,
            title=f"{e.via[0].name}: {e.via[1].name}" if e.via else e.label)
        _save_plan(args, f"sequence_{slug}", result.plan)
        review.offer_plan(e.handler, result.plan,
                          sequence_evidence(ir, call_slice(ir, e.handler, llm.depth, llm.max_nodes)))
        for w in result.warnings + warnings:
            print(f"  [WARN] {stem}: {w}")
        ids = call_slice(ir, e.handler, llm.depth, llm.max_nodes)
        prov = {"producer": "llm", "items": provenance.sequence_items(ir, result, ids, e.handler),
                "requests": [provenance.request_record(result, llm)],
                "extra": {"llm": provenance.llm_header(llm), "entry": e.handler,
                          "verification": {k: v for k, v in result.report.items() if k != "slice"},
                          "slice": ids}}
        out.append((stem, f"{ir.repo}: {result.plan.get('title') or e.label}", text,
                    f"({result.attempts} attempt(s), {took:.0f}s)", prov))
    return out


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _fact(what: str, evidence) -> dict:
    return {"producer": "deterministic", "items": provenance.deterministic_items(what, list(evidence))}


def _diagram_files(ir, types, depth, max_classes, members):
    """[(filename stem, title, text, note, provenance)] for the requested deterministic diagrams."""
    out = []
    if "component" in types:
        out.append((f"{ir.repo}_component", f"{ir.repo}: components",
                    render_component_diagram(ir, f"{ir.repo}_component", depth=depth), "",
                    _fact("component diagram", [m.id for m in ir.modules] + [i.id for i in ir.interfaces])))
    if "class" in types:
        n_classes = sum(1 for _ in ir.iter_classes())
        if n_classes <= max_classes:
            out.append((f"{ir.repo}_class", f"{ir.repo}: classes",
                        render_class_diagram(ir, f"{ir.repo}_class", members=members), "",
                        _fact("class diagram", [c.id for c in ir.iter_classes()])))
        else:
            for g, mods in partition_modules(ir, max_classes):
                stem = f"{ir.repo}_class_{g.replace('.', '_')}"
                in_mods = [c.id for m in ir.modules if m.id in set(mods) for c in m.classes]
                out.append((stem, f"{ir.repo}: classes in {g}",
                            render_class_diagram(ir, stem, modules=mods, members=members), "",
                            _fact("class diagram", in_mods)))
    if "deployment" in types:
        for dep in ir.deployments:
            stem = f"{ir.repo}_deployment_{re.sub(r'[^A-Za-z0-9]+', '_', dep.file.rsplit('.', 1)[0]).strip('_')}"
            out.append((stem, f"{ir.repo}: deployment ({dep.file})",
                        render_deployment_diagram(ir, dep, stem, title=f"Deployment: {dep.file}"), "",
                        _fact("deployment diagram", [dep.file] + [m for s in dep.services for m in s.modules]
                              + [s.dockerfile for s in dep.services if s.dockerfile])))
    if "flow" in types:
        # one cross-service sequence per user-facing entry, per deployment file (or the repo)
        for dep in ir.deployments or [None]:
            where = dep.file if dep else "repository"
            dslug = re.sub(r"[^A-Za-z0-9]+", "_", dep.file.rsplit(".", 1)[0]).strip("_") if dep else "repo"
            for iface, op in flow_entries(ir, dep):
                eslug = re.sub(r"[^A-Za-z0-9]+", "_", f"{iface.owner}_{op.name}").strip("_")
                stem = f"{ir.repo}_flow_{dslug}_{eslug}"
                client = "User" if iface.kind in ("ui", "cli") else "Client"
                flow = trace(ir, iface, op, dep)
                out.append((stem, f"{ir.repo}: {op.name} across services ({where})",
                            render_flow_diagram(flow, stem, dep, title=f"{iface.name}: {op.name} ({where})",
                                                client=client), "",
                            {"producer": "deterministic", "items": provenance.flow_items(flow),
                             "extra": {"entry": op.id, "deployment": dep.file if dep else None}}))
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _code_behavior_files(ir, args, types, llm):
    """State, object, use-case and activity diagrams: drawn from the code. For
    activity and use-case diagrams an LLM, when there is one and policy allows,
    only replaces labels, keyed by node or use-case id."""
    out = []
    root = args.repo
    if "state" in types:
        for m in state_machines(root, ir):
            stem = f"{ir.repo}_state_{_slug(m.key)}"
            items = [{"id": f"initial[{i}]", "class": "fact", "what": f"[*] -> {s}", "evidence": [w]}
                     for i, (s, w) in enumerate(m.initial)]
            items += [{"id": f"transition[{i}]", "class": "fact", "what": f"{t.src or '*'} -> {t.dst}",
                       "evidence": [f"{t.function}:{t.line}"]} for i, t in enumerate(m.transitions)]
            out.append((stem, f"{ir.repo}: states of {m.name}", render_state_diagram(m, stem), "",
                        {"producer": "deterministic", "items": items,
                         "extra": {"machine": m.key, "unreached": m.unreached}}))
    if "object" in types:
        model = build_objects(root, ir)
        if model.objects:
            stem = f"{ir.repo}_object"
            items = [{"id": f"object:{o.key}", "class": "fact", "evidence": [f"{o.module}:{o.line}"]}
                     for o in model.objects]
            out.append((stem, f"{ir.repo}: objects created at import", render_object_diagram(
                model, stem, title=f"{ir.repo}: objects created when the modules are imported"), "",
                {"producer": "deterministic", "items": items}))
    if "usecase" in types:
        model = build_usecases(root, ir)
        if model is not None:
            stem = f"{ir.repo}_usecase"
            goals, names, requests, note = {}, {}, [], ""
            items = [{"id": f"usecase:{c.id}", "class": "fact", "evidence": [c.id] + c.via} for c in model.cases]
            if llm is not None:
                res = plan_usecase(llm, model)
                requests.append(provenance.request_record(res, llm))
                if res.ok:
                    goals = {c["ref"]: c["goal"] for c in res.plan["use_cases"] if c["goal"]}
                    names = {a["ref"]: a["name"] for a in res.plan["actors"] if a["name"]}
                    items += [{"id": f"goal:{k}", "class": "derived", "request": 0, "evidence": [k]} for k in goals]
                    note = f"(labels: {res.attempts} attempt(s))"
                else:
                    print(f"  [WARN] {stem}: labels: {'; '.join(res.errors[:2])}")
            out.append((stem, f"{ir.repo}: use cases", render_usecase_diagram(model, stem, goals=goals,
                                                                                actor_names=names), note,
                        {"producer": "deterministic + llm" if goals else "deterministic", "items": items,
                         "requests": requests}))
    if "activity" in types:
        try:
            entries = select_entries(ir, args.entry, args.max_entries, args.depth, args.max_nodes)
        except ValueError as exc:
            raise PolicyError(str(exc)) from None
        for e in entries:
            act = build_activity(root, ir, e.handler)
            if act is None or not has_control_flow(act):
                continue
            stem = f"{ir.repo}_activity_{_slug(e.handler)}"
            title = f"{e.via[0].name}: {e.via[1].name}" if e.via else e.label
            labels, requests, note = {}, [], ""
            items = [{"id": f"node:{x.id}", "class": "fact", "what": getattr(x, "kind", ""),
                      "evidence": [f"{e.handler}:{x.line}"]} for x in act.labelable()]
            res = None
            if llm is not None:
                try:
                    res = plan_activity(llm, act)
                except EgressAborted:
                    pass                     # policy does not allow control flow for this module: code labels
            if res is not None:
                requests.append(provenance.request_record(res, llm))
                if res.ok:
                    labels = {x["node"]: x["text"] for x in res.plan["labels"] if x["text"]}
                    items += [{"id": f"label:{k}", "class": "derived", "request": 0, "evidence": [e.handler]}
                              for k in labels]
                    title = res.plan["title"] or title
                    note = f"(labels: {res.attempts} attempt(s))"
                else:
                    print(f"  [WARN] {stem}: labels: {'; '.join(res.errors[:2])}")
            out.append((stem, f"{ir.repo}: activity of {e.label}",
                        render_activity_diagram(act, stem, title=title, labels=labels), note,
                        {"producer": "deterministic + llm" if labels else "deterministic", "items": items,
                         "requests": requests, "extra": {"function": e.handler}}))
    return out


def _prov_header(args, ir) -> dict:
    """Run header shared by every sidecar of one invocation."""
    if getattr(args, "_prov", None) is None:
        args._prov = provenance.run_header(args.repo, ir, dated=not getattr(args, "no_date", False))
    return args._prov


def _generate_diagrams(ir, args, llm=None) -> list:
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = [t for t in types if t not in DETERMINISTIC_DIAGRAMS + LLM_DIAGRAMS]
    if unknown:
        raise PolicyError(f"unknown diagram type(s): {', '.join(unknown)}")
    planned = [t for t in types if t in PLANNED_DIAGRAMS]
    review = Review(args.output)
    reviewed_seq = any(k.startswith("sequence:") and review.decided(k) for k in review.items)
    if planned and llm is None:
        print(f"  [SKIP] {', '.join(planned)}: needs an LLM (--backend NAME or --model MODEL)"
              + (" except reviewed sequence diagrams" if reviewed_seq else ""))
    files = _diagram_files(ir, types, args.component_depth, args.max_classes, args.members)
    files += _code_behavior_files(ir, args, types, llm)
    if "sequence" in planned and (llm is not None or reviewed_seq):
        files += _sequence_files(ir, args, llm, review)
        _save_review(review)
    written = []
    validate = not args.no_validate and plantuml_command() is not None
    failures = 0
    header = _prov_header(args, ir)
    for stem, title, text, note, prov in files:
        path = _write(os.path.join(args.output, "diagrams", f"{stem}.puml"), text)
        provenance.write(os.path.join(args.output, "diagrams", f"{stem}.provenance.json"),
                         provenance.sidecar(header, f"{stem}.puml", prov["producer"], prov["items"],
                                            prov.get("requests"), **prov.get("extra", {})))
        status = ""
        if validate:
            res = check_syntax(text)
            status = "  syntax OK" if res.ok else f"  SYNTAX ERROR line {res.line}: {res.message}"
            failures += not res.ok
        print(f"  wrote {path}{status}{'  ' + note if note else ''}")
        extra = prov.get("extra", {})
        about = [x for x in (extra.get("entry"), extra.get("function")) if x]
        written.append({"title": title, "path": path, "kind": stem[len(ir.repo) + 1:].split("_")[0],
                        "about": about})
    if not validate and not args.no_validate:
        print("  (PlantUML not found: set PLANTUML_JAR or put `plantuml` on PATH to syntax-check)")
    embed = getattr(args, "embed", "link")
    fmts = {embed} if embed in ("svg", "png") else set()
    if getattr(args, "export", None) and set(args.export.split(",")) & {"docx", "pdf"}:
        fmts.add("png")                              # Word and LaTeX take PNG, not SVG
    images: dict = {}
    for fmt in sorted(fmts):
        if plantuml_command() is None:
            print(f"  [SKIP] {fmt} images: PlantUML not found; documents embed the diagram source instead")
            break
        rendered = render_images([d["path"] for d in written], fmt)
        if fmt == embed:
            images = rendered
        print(f"  rendered {len(rendered)} {fmt} image(s)")
    for d in written:
        d["image_path"] = images.get(d["path"])
    if failures:
        raise SystemExit(f"{failures} diagram(s) failed PlantUML syntax check")
    return written


def _has_open_slot(ctx, iface_id) -> bool:
    """Whether narrative could still fill anything (reviewed and rejected slots are settled)."""
    i = next((x for x in ctx["interfaces"] if x["id"] == iface_id), None)
    if i is None:
        return False
    slots = [(i, "purpose")] + [(o, "summary") for o in i["operations"]] + \
        [(p, "description") for o in i["operations"] for p in o["params"]]
    return any(not s.get(f) and not s.get(f"{f}_rejected") for s, f in slots)


def _narrate(llm, ctx, ifaces) -> None:
    """Fill the ICD context; records each request in ctx["requests"] for the sidecar."""
    for iface in ifaces:
        if not _has_open_slot(ctx, iface.id):
            print(f"  narrative for {iface.name}: nothing open (reviewed or from code)")
            continue
        t0 = time.monotonic()
        result = plan_narrative(llm, iface)
        took = time.monotonic() - t0
        ctx.setdefault("requests", []).append(provenance.request_record(result, llm))
        ctx.setdefault("request_index", {})[iface.id] = len(ctx["requests"]) - 1
        if result.ok:
            n = apply_narrative(ctx, iface.id, result.plan)
            print(f"  narrative for {iface.name}: {n} item(s) filled "
                  f"({result.attempts} attempt(s), {took:.0f}s)")
        else:
            print(f"  [FAIL] narrative for {iface.name}: {'; '.join(result.errors[:3])}")


def _doc_figures(diagrams, doc_dir, args):
    images = {d["path"]: d.get("image_path") for d in diagrams or [] if d.get("image_path")}
    return figures(diagrams or [], doc_dir, getattr(args, "embed", "link"), images)


def _generate_icd(ir, args, diagrams=None, llm=None, overview=None) -> list:
    """Writes the ICD(s); returns their paths."""
    icd_dir = os.path.join(args.output, "icd")
    links = _doc_figures(diagrams, icd_dir, args)
    written = []
    generated = None if args.no_date else __import__("datetime").date.today().isoformat()
    if args.narrative and llm is None:
        print("  [SKIP] narrative: needs an LLM (--backend NAME or --model MODEL)")
    narrate = args.narrative and llm is not None
    review = Review(args.output)
    evidence = icd_evidence(ir)
    if args.scope == "system":
        ctx = build_icd_context(ir, diagrams=links, generated=generated)
        ctx["system_overview"] = overview
        apply_to_icd(ctx, review, evidence)
        if narrate:
            ctx["narrative_aliased"] = llm.config.policy_for(llm.tier).pseudonymize != "off"
            _narrate(llm, ctx, ir.interfaces)
        offer_icd(ctx, review, evidence)
        ctx["coverage"] = icd_coverage(ctx, ir)
        _write(os.path.join(args.output, "coverage.json"), json.dumps(ctx["coverage"], indent=1))
        path = _write(os.path.join(icd_dir, f"{ir.repo}_icd.md"), render_icd(ctx, args.template))
        written.append(path)
        _icd_sidecar(args, ir, ctx, path, llm if narrate else None)
        print(f"  wrote {path}  ({ctx['summary']['interfaces']} interfaces, "
              f"{ctx['summary']['operations']} operations, {len(ctx['gaps'])} open items)")
        for line in coverage_lines(ctx["coverage"]):
            print(f"  {line}")
    else:
        for iface in ir.interfaces:
            ctx = build_icd_context(ir, [iface], title=f"{iface.name} Interface Control Document",
                                    diagrams=links, generated=generated)
            ctx["system_overview"] = overview
            apply_to_icd(ctx, review, evidence)
            if narrate:
                ctx["narrative_aliased"] = llm.config.policy_for(llm.tier).pseudonymize != "off"
                _narrate(llm, ctx, [iface])
            offer_icd(ctx, review, evidence)
            ctx["coverage"] = icd_coverage(ctx, ir)
            path = _write(os.path.join(icd_dir, f"{anchor(iface.id)}.md"), render_icd(ctx, args.template))
            written.append(path)
            _icd_sidecar(args, ir, ctx, path, llm if narrate else None)
            print(f"  wrote {path}")
    _save_review(review)
    snap = os.path.join(_state_dir(args.output), "icd_snapshot.json")
    with open(snap, "w", encoding="utf-8") as fh:
        json.dump(inventory_snapshot(ir), fh, indent=1)
    return written


def _save_review(review) -> None:
    path = review.save()
    counts = review.counts()
    print(f"  review: {path} ({', '.join(f'{v} {k}' for k, v in sorted(counts.items())) or 'empty'})")
    for item_id in review.stale:
        print(f"  [STALE] {item_id}: reviewed before the code changed; re-review")


def _icd_sidecar(args, ir, ctx, path, llm) -> None:
    items = provenance.icd_items(ctx, ctx.get("request_index", {}))
    extra = {"llm": provenance.llm_header(llm)} if llm is not None else {}
    provenance.write(path[: -len(".md")] + ".provenance.json",
                     provenance.sidecar(_prov_header(args, ir), os.path.basename(path),
                                        "deterministic + llm" if llm is not None else "deterministic",
                                        items, ctx.get("requests"), **extra))


def _finish_llm(llm) -> None:
    if llm is not None:
        llm.aliases.save(llm.alias_path)
        c = llm.cache
        if c is not None and (c.hits or c.misses):
            print(f"  model requests: {c.misses} sent, {c.hits} answered from the cache")


def cmd_diagrams(args: argparse.Namespace) -> int:
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    llm = _llm_context(args, cfg, ir)
    try:
        _generate_diagrams(ir, args, llm)
    finally:
        _finish_llm(llm)
    return 0


def cmd_icd(args: argparse.Namespace) -> int:
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    if not ir.interfaces:
        print(f"{ir.repo}: no interfaces detected; nothing to document")
        return 0
    llm = _llm_context(args, cfg, ir)
    try:
        _export(_generate_icd(ir, args, llm=llm), args)
    finally:
        _finish_llm(llm)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    llm = _llm_context(args, cfg, ir)
    try:
        diagrams = _generate_diagrams(ir, args, llm)
        docs, overview = [], None
        if args.design:                      # first: its system summary is the IDD's system overview
            path, overview = _generate_design(ir, args, cfg, llm, diagrams)
            docs.append(path)
        if ir.interfaces:
            docs = _generate_icd(ir, args, diagrams, llm, overview) + docs
        _export(docs, args)
    finally:
        _finish_llm(llm)
    return 0


def _generate_design(ir, args, cfg, llm=None, diagrams=None):
    """Design description: summaries, behavior specifications, dependencies, tests.
    Returns (path, the system summary's text or None)."""
    ir_all = extract_python(args.repo, exclude=inventory_excludes(cfg.exclude + args.exclude))
    if llm is None:
        print("  [SKIP] summaries, behavior specifications, test criteria: need an LLM "
              "(--backend NAME or --model MODEL); writing the rest without them")
    opts = DesignOptions(behavior=not args.no_behavior, summaries=not args.no_summaries,
                         tests=not args.no_criteria, only=args.only, max_functions=args.max_functions)
    generated = None if args.no_date else __import__("datetime").date.today().isoformat()
    ctx = build_design_context(ir, ir_all, args.repo, llm, opts, generated, say=lambda s: print(f"  {s}"))
    ctx["diagrams"] = _doc_figures(diagrams, os.path.join(args.output, "design"), args)
    path = _write(os.path.join(args.output, "design", f"{ir.repo}_design.md"),
                  render_design(ctx, args.design_template))
    extra = {"llm": provenance.llm_header(llm)} if llm is not None else {}
    provenance.write(path[: -len(".md")] + ".provenance.json",
                     provenance.sidecar(_prov_header(args, ir), os.path.basename(path),
                                        "deterministic + llm" if llm is not None else "deterministic",
                                        design_items(ctx), ctx["requests"], **extra))
    _write(os.path.join(args.output, "design", "manifest.json"), json.dumps(ctx["manifest"], indent=1))
    _write(os.path.join(args.output, "design", "tests.json"), json.dumps(ctx["tests"], indent=1))
    _write(os.path.join(args.output, "design_coverage.json"), json.dumps(ctx["coverage"], indent=1))
    print(f"  wrote {path}")
    for line in design_coverage_lines(ctx["coverage"]):
        print(f"  {line}")
    return path, (ctx["system"] or {}).get("responsibility")


def _export(paths, args) -> None:
    fmts = [f.strip() for f in (getattr(args, "export", None) or "").split(",") if f.strip()]
    bad = [f for f in fmts if f not in EXPORT_FORMATS]
    if bad:
        raise PolicyError(f"unknown export format(s): {', '.join(bad)} (use {', '.join(EXPORT_FORMATS)})")
    for fmt in fmts:
        for p in paths:
            try:
                print(f"  exported {export_document(p, fmt, png=fmt in ('docx', 'pdf'))}")
            except RuntimeError as exc:
                print(f"  [SKIP] {fmt} export of {os.path.basename(p)}: {exc}")
                break


def cmd_design(args: argparse.Namespace) -> int:
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    llm = _llm_context(args, cfg, ir)
    try:
        path, _ = _generate_design(ir, args, cfg, llm)
        _export([path], args)
    finally:
        _finish_llm(llm)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export every ICD and design description already in an output directory."""
    docs = []
    for sub in ("icd", "design"):
        d = os.path.join(args.output, sub)
        if os.path.isdir(d):
            docs += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".md"))
    if not docs:
        print(f"{args.output}: no documents found (icd/*.md, design/*.md)")
        return 1
    if set(args.to.split(",")) & {"docx", "pdf"}:
        pumls = sorted(os.path.join(args.output, "diagrams", f) for f in os.listdir(
            os.path.join(args.output, "diagrams")) if f.endswith(".puml")) if os.path.isdir(
            os.path.join(args.output, "diagrams")) else []
        if pumls and plantuml_command() is not None:
            render_images(pumls, "png")
    _export(docs, SimpleNamespace(export=args.to))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Structure against the spec's reference, then the same requests on every backend."""
    from .evaluate import evaluate_model, evaluate_structure, load_spec, report_markdown

    spec = load_spec(args.spec)
    cfg = load_config(args.repo, extra=args.config)
    ir = extract_python(args.repo, exclude=cfg.exclude + args.exclude)
    os.makedirs(args.output, exist_ok=True)
    structure = None
    if spec.get("truth", {}).get("mermaid"):
        try:
            structure = evaluate_structure(ir, args.repo, spec)
        except (OSError, ValueError) as exc:
            print(f"  [SKIP] structure: {exc}")
        else:
            pct = lambda x: "n/a" if x is None else f"{round(100 * x)}%"   # noqa: E731
            print(f"  structure: recall {pct(structure['recall'])}, precision {pct(structure['precision'])} "
                  f"against {structure['reference']}")

    llm_spec = spec.get("llm", {})
    max_entries = args.max_entries or llm_spec.get("max_entries", 3)
    runs = []
    names = [n.strip() for n in (args.backend or "").split(",") if n.strip()]
    if names or args.model:
        try:
            entries = select_entries(ir, args.entry or llm_spec.get("entries", []), max_entries,
                                     args.depth, args.max_nodes)
        except ValueError as exc:
            raise PolicyError(str(exc)) from None
        narrate = llm_spec.get("narrative", True) and not args.no_narrative
        ifaces = list(ir.interfaces) if narrate else []
        print(f"  {len(entries)} sequence entries, {len(ifaces)} narratives per backend")
        for name in names or ["adhoc"]:
            sub = argparse.Namespace(**vars(args))
            sub.backend = None if name == "adhoc" else name
            sub.output = os.path.join(args.output, re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
            llm = _llm_context(sub, cfg, ir)
            if llm is None:
                runs.append({"backend": name, "model": "unavailable"})
                continue
            try:
                run = evaluate_model(llm, entries, ifaces, llm.tier)
            finally:
                _finish_llm(llm)
            summary = run.summary()
            runs.append(summary)
            _write(os.path.join(sub.output, "details.json"), json.dumps(run.details, indent=1))
            print("  " + ", ".join(f"{k}={v}" for k, v in summary.items()))

    _write(os.path.join(args.output, "eval.json"),
           json.dumps({"structure": structure, "models": runs}, indent=1, default=list))
    path = _write(os.path.join(args.output, "eval.md"), report_markdown(structure, runs, ir.repo))
    print(f"  wrote {path}")
    return 0


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    """MCP server on stdio for coding-agent harnesses (DESIGN.md 13)."""
    from .harness import HarnessSession
    from .mcp_server import serve_stdio

    env = os.environ
    args.repo = args.repo or env.get("ARCHDOC_REPO")
    args.out = args.out or env.get("ARCHDOC_OUT") or "archdoc_out"
    args.trust = args.trust or env.get("ARCHDOC_TRUST") or CLOUD
    args.config = args.config or env.get("ARCHDOC_CONFIG")
    args.window = args.window or int(env.get("ARCHDOC_WINDOW") or 32768)
    if args.trust not in TIERS:
        raise PolicyError(f"trust tier {args.trust!r}: use {' or '.join(TIERS)}")
    if not args.repo or not os.path.isdir(args.repo):
        raise PolicyError(f"repo {args.repo!r}: not a directory (--repo or ARCHDOC_REPO)")
    session = HarnessSession(args.repo, args.out, trust=args.trust, config=args.config, exclude=args.exclude,
                             window=args.window, max_entries=args.max_entries, template=args.template,
                             design_template=args.design_template, embed=args.embed)
    print(f"archdoc MCP server: {os.path.abspath(args.repo)} -> {os.path.abspath(args.out)} "
          f"(tier {args.trust})", file=sys.stderr)
    serve_stdio(session)
    return 0


def cmd_harness_init(args: argparse.Namespace) -> int:
    from .profiles import write_profile

    try:
        paths = write_profile(args.dir, args.harness, args.repo, args.out, args.trust, args.model,
                              url=args.url, config=args.config, window=args.window,
                              context=args.context)
    except ValueError as exc:
        raise PolicyError(str(exc)) from None
    for p in paths:
        print(f"  wrote {p}")
    agent = f"archdoc-{args.trust}"
    run = f"opencode --agent {agent}" if args.harness == "opencode" else f"vibe --agent {agent} --trust"
    print(f"  start it with: cd {os.path.abspath(args.dir)} && {run}")
    print(f"  or headless:   {os.path.join(os.path.abspath(args.dir), 'run-' + args.harness + '.sh')}")
    print("  switch profile, not model: a local profile only offers its local model")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("repo", help="Path to the repository to analyse")
    p.add_argument("-o", "--output", default="archdoc_out", help="Output directory")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="Repo-relative glob to skip (repeatable), e.g. 'tests/*'")
    p.add_argument("--config", help="Extra config TOML (overrides user and repo config)")


def _add_llm_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM (optional; needed for sequence diagrams and ICD narrative)")
    who = g.add_mutually_exclusive_group()
    who.add_argument("--backend", help="Backend name from config, a comma-separated preference list, "
                                         "or 'auto' for [llm] prefer (default when configured)")
    who.add_argument("--model", help="Ad-hoc backend model, e.g. mistral-nemo:12b")
    g.add_argument("--url", help="Ad-hoc backend URL (default for ollama: http://localhost:11434)")
    g.add_argument("--kind", choices=("ollama", "openai"), default="ollama", help="Ad-hoc backend API")
    g.add_argument("--num-ctx", type=int, help="Ollama context window (default 8192)")
    g.add_argument("--trust", choices=TIERS,
                   help="Ad-hoc: declared tier (undeclared = cloud). Named backend: 'cloud' forces aliasing")
    g.add_argument("--entry", action="append", default=[],
                   help="Sequence entry: handler id (pkg.mod:func) or operation id (repeatable)")
    g.add_argument("--max-entries", type=int, default=3, help="Sequence diagrams to plan when no --entry")
    g.add_argument("--depth", type=int, default=3, help="Call-graph depth for sequence slices")
    g.add_argument("--max-nodes", type=int, default=25, help="Max functions per sequence slice")
    g.add_argument("--no-cache", action="store_true",
                   help="Ask the model again even where the same request was answered before")


def _add_diagram_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--types", default="class,component,deployment,flow,state,object,usecase,activity",
                   help="Comma-separated: class, component, deployment, flow, state, object, usecase, "
                        "activity (from the code; an LLM only labels activity and use-case diagrams); "
                        "sequence (LLM)")
    p.add_argument("--component-depth", type=int, help="Package depth for components (default: auto)")
    p.add_argument("--max-classes", type=int, default=60,
                   help="Above this many classes, write one class diagram per component")
    p.add_argument("--members", choices=("public", "all"), default="public")
    p.add_argument("--no-validate", action="store_true", help="Skip the PlantUML syntax check")


def _add_icd_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scope", choices=("system", "interface"), default="system",
                   help="One ICD for the system, or one per interface")
    p.add_argument("--template", default="lightweight", help="Bundled template name or path to a .j2 file")
    p.add_argument("--no-date", action="store_true", help="Omit the generation date (reproducible output)")
    p.add_argument("--narrative", action="store_true", help="Fill TBD descriptions with the LLM")


def _add_doc_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("documents")
    g.add_argument("--embed", choices=EMBED_MODES, default="svg",
                   help="How documents include diagrams: rendered svg/png (needs PlantUML; falls back to "
                        "the source), the PlantUML source, or a link")
    g.add_argument("--export", metavar="FORMATS",
                   help=f"Also convert the documents with pandoc: comma-separated {', '.join(EXPORT_FORMATS)}")


def _add_design_opts(p: argparse.ArgumentParser, flag: bool = False) -> None:
    g = p.add_argument_group("design description")
    if flag:
        g.add_argument("--design", action="store_true",
                       help="Also write the design description (one model request per public function)")
    g.add_argument("--no-behavior", action="store_true", help="No behavior specifications")
    g.add_argument("--no-summaries", action="store_true", help="No module, package or system summaries")
    g.add_argument("--no-criteria", action="store_true", help="No acceptance criteria for tests")
    g.add_argument("--only", metavar="MODULE", help="Behavior specifications only under this module or package")
    g.add_argument("--max-functions", type=int, help="At most this many behavior specifications")
    g.add_argument("--design-template", default="lightweight", help="Bundled template name or a .j2 path")


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

    p = sub.add_parser("diagrams", help="Write PlantUML diagrams")
    _add_common(p)
    _add_diagram_opts(p)
    _add_llm_opts(p)
    p.set_defaults(func=cmd_diagrams)

    p = sub.add_parser("icd", help="Write Interface Control Document(s)")
    _add_common(p)
    _add_icd_opts(p)
    _add_llm_opts(p)
    _add_doc_opts(p)
    p.set_defaults(func=cmd_icd)

    p = sub.add_parser("all", help="Diagrams and ICDs (and with --design, the design description)")
    _add_common(p)
    _add_diagram_opts(p)
    _add_icd_opts(p)
    _add_llm_opts(p)
    _add_design_opts(p, flag=True)
    _add_doc_opts(p)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("design", help="Design description: summaries, behavior specs, dependencies, tests")
    _add_common(p)
    _add_llm_opts(p)
    _add_design_opts(p)
    _add_doc_opts(p)
    p.add_argument("--no-date", action="store_true", help="Omit the generation date (reproducible output)")
    p.set_defaults(func=cmd_design)

    p = sub.add_parser("export", help="Convert the documents in an output directory with pandoc")
    p.add_argument("output", nargs="?", default="archdoc_out", help="archdoc output directory")
    p.add_argument("--to", default="docx", help=f"Comma-separated: {', '.join(EXPORT_FORMATS)}")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("serve-mcp", help="MCP server (stdio) for opencode, Mistral Vibe and other harnesses")
    # each setting can also come from the environment (ARCHDOC_REPO, ...): harness
    # profiles use that, so the command line a harness may show the model names nothing
    p.add_argument("--repo", help="Repository to document, fixed for the session [ARCHDOC_REPO]")
    p.add_argument("--out", help="Output directory, fixed for the session [ARCHDOC_OUT; default archdoc_out]")
    p.add_argument("--trust", choices=TIERS,
                   help="Tier of the harness's model [ARCHDOC_TRUST]; default cloud (pseudonymized, call graph only)")
    p.add_argument("--config", help="Extra config TOML (overrides user and repo config) [ARCHDOC_CONFIG]")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    p.add_argument("--window", type=int, help="Context window to pack requests for, in tokens "
                                              "[ARCHDOC_WINDOW; default 32768]")
    p.add_argument("--max-entries", type=int, default=3, help="Sequence diagrams to plan")
    p.add_argument("--template", default="lightweight", help="ICD template")
    p.add_argument("--design-template", default="lightweight", help="Design description template")
    p.add_argument("--embed", choices=EMBED_MODES, default="svg", help="How documents include diagrams")
    p.set_defaults(func=cmd_serve_mcp)

    p = sub.add_parser("harness-init", help="Write an opencode or Vibe working directory locked to archdoc's tools")
    p.add_argument("dir", help="Directory to write (must not be inside the target repo)")
    p.add_argument("--harness", choices=("opencode", "vibe"), required=True)
    p.add_argument("--repo", required=True, help="Repository to document")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--trust", choices=TIERS, required=True,
                   help="local: the profile offers only its local model; cloud: any model, pseudonymized input")
    p.add_argument("--model", required=True,
                   help="local: the model name on the server (e.g. mistral-nemo:12b); "
                        "cloud: the harness's model id (opencode provider/model, Vibe model alias)")
    p.add_argument("--url", help=f"local: OpenAI-compatible endpoint (default {'http://localhost:11434/v1'})")
    p.add_argument("--config", help="Extra archdoc config TOML for the server")
    p.add_argument("--window", type=int,
                   help="Tokens each task's input is packed for (default: 8192 local, 32768 cloud)")
    p.add_argument("--context", type=int,
                   help="local: the model's context window in tokens, e.g. Ollama's num_ctx (default 16384)")
    p.set_defaults(func=cmd_harness_init)

    p = sub.add_parser("eval", help="Evaluate accuracy against a reference and compare backends")
    _add_common(p)
    _add_llm_opts(p)
    p.add_argument("--spec", required=True, help="Evaluation spec TOML (see eval/)")
    p.add_argument("--no-narrative", action="store_true", help="Only evaluate sequence diagrams")
    p.set_defaults(func=cmd_eval, max_entries=None)
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
