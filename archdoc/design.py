"""
Design description (DESIGN.md 2.3, 12.5): the rebuild content beside the ICD.

  overview      system summary, from package summaries, from module summaries
  modules       each module's summary and the behavior specification of each
                public function: purpose, inputs, returns, preconditions, side
                effects, errors, algorithm in prose
  dependencies  the dependency manifest (archdoc.manifest)
  tests         the test inventory as acceptance criteria (archdoc.testsuite)

Everything model-written is limited by schema to refs the code shows and
labelled with the evidence it was written from: `source`, `outline`, or
`signature` (only a stub fitted, or the tier allows no bodies), the label a
reader sees next to the text. Summaries are hierarchical so that system-level
text never needs the whole repo in one request; with the response cache, a
rerun asks only for what changed (a module's summary when its code changed,
its package's when that summary changed, and so on up).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from . import __version__, provenance
from .egress import module_collaborators, module_externals
from .extract.bodies import locate
from .ir import Function, RepoIR
from .manifest import build_manifest
from .plan.artifacts import (
    PlanContext, plan_behavior, plan_group_summary, plan_module_summary, plan_tests,
)
from .render.icd.model import anchor
from .secrets_scan import redact
from .testsuite import build_inventory, exercised_by, is_test_path

EVIDENCE = {"source": "source", "outline": "outline", "stub": "signature"}
TESTS_PER_REQUEST = 12


@dataclass
class DesignOptions:
    behavior: bool = True
    summaries: bool = True
    tests: bool = True
    only: Optional[str] = None          # module id prefix for behavior specs
    max_functions: Optional[int] = None


def documented_modules(ir: RepoIR):
    return [m for m in ir.modules if not m.parse_error and not is_test_path(m.path)]


def _public(fn: Function) -> bool:
    return fn.visibility == "public" and not (fn.name.startswith("__") and fn.name.endswith("__"))


def behavior_targets(ir: RepoIR) -> List[str]:
    """Public functions and methods of the documented (non-test) modules, in module order."""
    out = []
    for m in documented_modules(ir):
        out += [f.id for f in m.functions if _public(f)]
        for c in m.classes:
            if not c.name.startswith("_"):
                out += [f.id for f in c.methods if _public(f)]
    return out


def package_of(module_id: str, path: str) -> str:
    if path.endswith("__init__.py") or "." not in module_id:
        return module_id
    return module_id.rsplit(".", 1)[0]


def packages(ir: RepoIR) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for m in documented_modules(ir):
        out.setdefault(package_of(m.id, m.path), []).append(m.id)
    return out


def signature_text(fn: Function, is_method: bool) -> str:
    params = [p for p in fn.params if not (is_method and p.name in ("self", "cls"))]
    parts = []
    for p in params:
        s = {"var_positional": "*", "var_keyword": "**"}.get(p.kind, "") + p.name
        if p.annotation:
            s += f": {p.annotation}"
        if p.default is not None:
            s += f" = {redact(p.default)[0]}"
        parts.append(s)
    ret = f" -> {fn.returns}" if fn.returns else ""
    return f"{'async ' if fn.is_async else ''}{fn.name}({', '.join(parts)}){ret}"


def _first_line(text: Optional[str]) -> Optional[str]:
    return (text or "").strip().split("\n", 1)[0] or None


def _spec_row(ir: RepoIR, fid: str, tested: Dict[str, List[str]]) -> Dict[str, Any]:
    fn = ir.find_function(fid)
    mid, _, qual = fid.partition(":")
    owner = qual.rsplit(".", 1)[0] if "." in qual else ""
    types = {p.name: p.annotation for p in fn.params}
    return {"id": fid, "name": qual, "owner": owner, "anchor": anchor(fid), "doc": _first_line(fn.docstring),
            "signature": signature_text(fn, bool(owner)), "evidence": None, "generated": False,
            "purpose": None, "inputs": [], "returns": "", "preconditions": [], "side_effects": [], "errors": [],
            "algorithm": [], "tested_by": tested.get(fid, []), "types": types, "failed": None}


def _apply_spec(row: Dict[str, Any], plan: Dict[str, Any], level: str) -> None:
    row.update(generated=True, evidence=EVIDENCE.get(level, "signature"), purpose=plan["purpose"] or None,
               returns=plan["returns"], preconditions=[p for p in plan["preconditions"] if p],
               side_effects=[s for s in plan["side_effects"] if s["description"]],
               errors=[e for e in plan["errors"] if e["when"] or e["ref"]],
               algorithm=[a for a in plan["algorithm"] if a],
               inputs=[{**i, "type": row["types"].get(i["name"])} for i in plan["inputs"] if i["description"]])


def _body(node) -> List[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return [st for st in body if not (isinstance(st, ast.Pass) or (
        isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and st.value.value is Ellipsis))]


_EMPTY = {ast.Dict: "an empty dict", ast.List: "an empty list", ast.Tuple: "an empty tuple", ast.Set: "an empty set"}


def trivial_spec(root: str, ir: RepoIR, fid: str) -> Optional[Dict[str, Any]]:
    """The exact specification of a function too simple to need a model: an empty
    body, returning a parameter or a constant, or returning one call made with
    the parameters as they are. Small models describe such functions with
    behavior they do not have (an empty command group "seeds the database"), so
    these are written from the AST instead."""
    found = locate(root, ir, fid)
    fn = ir.find_function(fid)
    if found is None or fn is None:
        return None
    body = _body(found[0])
    params = {p.name for p in fn.params}
    spec = {"inputs": [], "returns": "", "preconditions": [], "side_effects": [], "errors": [], "algorithm": []}
    if not body:
        return {**spec, "purpose": "Has no behavior of its own: the body is empty"
                + (" (its decorators give it its role)." if fn.decorators else ".")}
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    v = body[0].value
    if v is None or (isinstance(v, ast.Constant) and v.value is None):
        return {**spec, "purpose": "Does nothing and returns None."}
    if isinstance(v, ast.Name) and v.id in params:
        return {**spec, "purpose": f"Returns its argument `{v.id}` unchanged.", "returns": f"`{v.id}`, unchanged."}
    if isinstance(v, ast.Constant) and not isinstance(v.value, (str, bytes)):
        return {**spec, "purpose": f"Always returns `{v.value!r}`.", "returns": f"`{v.value!r}`."}
    if isinstance(v, ast.Constant):
        return {**spec, "purpose": "Always returns the same fixed string.", "returns": "A fixed string."}
    if type(v) in _EMPTY and not getattr(v, "keys", None) and not getattr(v, "elts", None):
        return {**spec, "purpose": f"Always returns {_EMPTY[type(v)]}.", "returns": f"{_EMPTY[type(v)].capitalize()}."}
    targets = [c for c in fn.calls + fn.creates if not c.startswith("ext:")]
    plain = isinstance(v, ast.Call) and not v.keywords and all(
        isinstance(a, ast.Name) and a.id in params for a in v.args)
    if plain and len(targets) == 1 and len(fn.calls) + len(fn.creates) == 1:
        t = targets[0]
        if t in fn.creates:
            text = f"Creates and returns a new `{t.split(':')[1]}`" + (
                f" from {', '.join(f'`{a.id}`' for a in v.args)}." if v.args else ".")
            return {**spec, "purpose": text, "returns": f"A new `{t.split(':')[1]}`."}
        args = f" with {', '.join(f'`{a.id}`' for a in v.args)}" if v.args else ""
        return {**spec, "purpose": f"Calls `{t}`{args} and returns its result.",
                "returns": f"Whatever `{t}` returns.", "side_effects": []}
    return None


def _summary_text(result) -> str:
    """A summary as the model wrote it (aliased for an aliased tier), for the next level up."""
    return (result.raw or {}).get("responsibility", "") if result.ok else ""


def build_design_context(ir: RepoIR, ir_all: RepoIR, root: str, llm: Optional[PlanContext] = None,
                         opts: Optional[DesignOptions] = None, generated: Optional[str] = None,
                         say: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    opts = opts or DesignOptions()
    tests = build_inventory(root, ir_all, ir)
    tested = exercised_by(tests)
    targets = behavior_targets(ir)
    ctx: Dict[str, Any] = {
        "title": f"{ir.repo} Design Description", "repo": ir.repo, "version": __version__,
        "generated": generated, "system": None, "packages": [], "modules": [],
        "manifest": build_manifest(root, ir).to_dict(),
        "tests": [t.to_dict() for t in tests], "requests": [], "request_index": {}, "diagrams": [],
        "interfaces": [{"id": i.id, "name": i.name, "kind": i.kind, "direction": i.direction, "owner": i.owner,
                        "operations": len(i.operations)} for i in ir.interfaces],
        "aliased": bool(llm and llm.config.policy_for(llm.tier).pseudonymize != "off"),
    }

    def record(item_id: str, result) -> None:
        ctx["requests"].append(provenance.request_record(result, llm))
        ctx["request_index"][item_id] = len(ctx["requests"]) - 1

    rows = {fid: _spec_row(ir, fid, tested) for fid in targets}
    mods = {}
    for m in documented_modules(ir):
        mods[m.id] = {"id": m.id, "path": m.path, "anchor": anchor(m.id), "doc": _first_line(m.docstring),
                      "summary": None, "functions": [rows[f] for f in targets if f.split(":")[0] == m.id]}
    ctx["modules"] = list(mods.values())

    # -- summaries, bottom up -------------------------------------------------------
    raw: Dict[str, str] = {}
    if llm is not None and opts.summaries:
        for mid, mod in mods.items():
            res = plan_module_summary(llm, mid)
            record(f"summary:{mid}", res)
            if res.ok:
                mod["summary"], raw[mid] = res.plan, _summary_text(res)
                say(f"summary of {mid}: ok ({_how(res)})")
            else:
                say(f"[FAIL] summary of {mid}: {'; '.join(res.errors[:2])}")
        pkgs = packages(ir)
        for pkg, members in pkgs.items():
            entry = {"id": pkg, "anchor": anchor(f"pkg:{pkg}"), "modules": members, "summary": None}
            if len(members) == 1:
                entry["summary"] = mods[members[0]]["summary"] and {
                    "responsibility": mods[members[0]]["summary"]["responsibility"], "parts": [],
                    "collaborators": [], "external": mods[members[0]]["summary"]["external"], "same_as": members[0]}
                raw[f"pkg:{pkg}"] = raw.get(members[0], "")
            else:
                parts = [(mid, raw[mid]) for mid in members if raw.get(mid)]
                collab = sorted({package_of(c, ir.module(c).path) for mid in members
                                 for c in module_collaborators(ir, mid)} - {pkg})
                external = sorted({e for mid in members for e in module_externals(ir, mid)})
                if parts:
                    res = plan_group_summary(llm, pkg, parts, collab, external)
                    record(f"summary:{pkg}", res)
                    if res.ok:
                        entry["summary"], raw[f"pkg:{pkg}"] = res.plan, _summary_text(res)
                        say(f"summary of package {pkg}: ok ({_how(res)})")
                    else:
                        say(f"[FAIL] summary of package {pkg}: {'; '.join(res.errors[:2])}")
            ctx["packages"].append(entry)
        parts = [(pkg, raw[f"pkg:{pkg}"]) for pkg in pkgs if raw.get(f"pkg:{pkg}")]
        if len(parts) > 1:
            external = sorted({e for m in documented_modules(ir) for e in module_externals(ir, m.id)})
            res = plan_group_summary(llm, "", parts, [], external)
            record("summary:<system>", res)
            if res.ok:
                ctx["system"] = res.plan
                say(f"system summary: ok ({_how(res)})")
            else:
                say(f"[FAIL] system summary: {'; '.join(res.errors[:2])}")
        elif len(parts) == 1:
            only = next(p for p in ctx["packages"] if p["id"] == parts[0][0])
            ctx["system"] = only["summary"]

    # -- behavior specifications ----------------------------------------------------
    if opts.behavior:
        for fid in targets:
            det = trivial_spec(root, ir, fid)
            if det is not None:
                _apply_spec(rows[fid], det, "source")
                rows[fid]["generated"] = False
    if llm is not None and opts.behavior:
        chosen = [f for f in targets if not rows[f]["evidence"] and (
            not opts.only or f.split(":")[0] == opts.only
            or f.startswith(opts.only + ".") or f.startswith(opts.only + ":"))]
        if opts.max_functions is not None:
            chosen = chosen[:opts.max_functions]
        for i, fid in enumerate(chosen, 1):
            res = plan_behavior(llm, fid)
            record(f"behavior:{fid}", res)
            if res.ok:
                _apply_spec(rows[fid], res.plan, res.report.get("evidence", "stub"))
                say(f"[{i}/{len(chosen)}] behavior of {fid}: {rows[fid]['evidence']} ({_how(res)})")
            else:
                rows[fid]["failed"] = "; ".join(res.errors[:2])
                say(f"[{i}/{len(chosen)}] [FAIL] behavior of {fid}: {rows[fid]['failed']}")

    # -- acceptance criteria ----------------------------------------------------------
    if llm is not None and opts.tests and tests:
        by_mod: Dict[str, List[Dict[str, Any]]] = {}
        for t in ctx["tests"]:
            by_mod.setdefault(t["module"], []).append(t)
        for mid, all_items in by_mod.items():
            # in batches: one sentence per test, but 38 tests overran a 2k-token answer
            for start in range(0, len(all_items), TESTS_PER_REQUEST):
                items = all_items[start:start + TESTS_PER_REQUEST]
                part = f"{mid}[{start // TESTS_PER_REQUEST}]" if len(all_items) > TESTS_PER_REQUEST else mid
                exercised = list(dict.fromkeys(x for t in items for x in t["exercises"] if ":" in x))
                res = plan_tests(llm, mid, [t["id"] for t in items], exercised, ir_all)
                record(f"tests:{part}", res)
                for t in items:
                    t["request_key"] = f"tests:{part}"
                if not res.ok:
                    say(f"[FAIL] criteria for {part}: {'; '.join(res.errors[:2])}")
                    continue
                got = {x["ref"]: x["criterion"] for x in res.plan["tests"] if x["criterion"]}
                for t in items:
                    if got.get(t["id"]):
                        t["criterion"], t["criterion_generated"] = got[t["id"]], True
                say(f"criteria for {part}: {len(got)}/{len(items)} ({_how(res)})")

    ctx["coverage"] = design_coverage(ctx, len(targets))
    return ctx


def _how(res) -> str:
    if res.cached and res.cached == res.attempts:
        return "cached"
    return f"{res.attempts} attempt(s)"


def design_coverage(ctx: Dict[str, Any], total: int) -> Dict[str, Any]:
    rows = [f for m in ctx["modules"] for f in m["functions"]]
    by = {"source": 0, "outline": 0, "signature": 0}
    for r in rows:
        if r["evidence"]:
            by[r["evidence"]] += 1
    tests = ctx["tests"]
    return {
        "behavior": {"total": total, **by, "none": total - sum(by.values())},
        "summaries": {"modules": sum(1 for m in ctx["modules"] if m["summary"]), "of": len(ctx["modules"]),
                      "system": ctx["system"] is not None},
        "tests": {"total": len(tests), "with_criterion": sum(1 for t in tests if t["criterion"]),
                  "functions_exercised": sum(1 for r in rows if r["tested_by"]), "of": len(rows)},
        "dependencies": {"declared": len(ctx["manifest"]["requirements"]),
                         "issues": len(ctx["manifest"]["issues"])},
    }


def coverage_lines(cov: Dict[str, Any]) -> List[str]:
    b, t, s = cov["behavior"], cov["tests"], cov["summaries"]
    return [
        f"behavior: {b['total'] - b['none']}/{b['total']} public functions specified "
        f"(source {b['source']}, outline {b['outline']}, signature only {b['signature']})",
        f"summaries: {s['modules']}/{s['of']} modules{', system' if s['system'] else ''}; "
        f"tests: {t['total']} ({t['with_criterion']} with criteria), "
        f"{t['functions_exercised']}/{t['of']} public functions exercised directly",
        f"dependencies: {cov['dependencies']['declared']} declared, {cov['dependencies']['issues']} issue(s)",
    ]


def design_items(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Provenance items: every model-written statement with its evidence and request."""
    idx = ctx["request_index"]
    items: List[Dict[str, Any]] = []
    for m in ctx["modules"]:
        if m["summary"]:
            s = m["summary"]
            items.append({"id": f"summary:{m['id']}", "class": "derived", "request": idx.get(f"summary:{m['id']}"),
                          "evidence": [m["id"]] + [x["ref"] for x in s["key_operations"] + s["collaborators"]
                                                   + s["external"]]})
        for f in m["functions"]:
            if f["evidence"] and not f["generated"]:
                items.append({"id": f"behavior:{f['id']}", "class": "fact", "detail": "source",
                              "evidence": [f["id"]]})
            elif f["generated"]:
                items.append({"id": f"behavior:{f['id']}", "class": "derived", "detail": f["evidence"],
                              "request": idx.get(f"behavior:{f['id']}"),
                              "evidence": [f["id"]] + [x["ref"] for x in f["side_effects"] + f["errors"]]})
            else:
                items.append({"id": f"behavior:{f['id']}", "class": "open", "evidence": [f["id"]]})
    for p in ctx["packages"]:
        if p["summary"] and not p["summary"].get("same_as"):
            items.append({"id": f"summary:{p['id']}", "class": "derived", "request": idx.get(f"summary:{p['id']}"),
                          "evidence": p["modules"]})
    if ctx["system"] and "summary:<system>" in idx:
        items.append({"id": "summary:<system>", "class": "derived", "request": idx["summary:<system>"],
                      "evidence": [p["id"] for p in ctx["packages"]]})
    for t in ctx["tests"]:
        items.append({"id": f"test:{t['id']}", "class": "fact", "evidence": [t["id"]] + t["exercises"]})
        if t["criterion_generated"]:
            items.append({"id": f"criterion:{t['id']}", "class": "derived",
                          "request": idx.get(t.get("request_key", f"tests:{t['module']}")),
                          "evidence": [t["id"]]})
    items.append({"id": "manifest", "class": "fact", "evidence": ctx["manifest"]["files"]})
    return items


__all__ = ["DesignOptions", "behavior_targets", "build_design_context", "coverage_lines", "design_coverage",
           "design_items", "packages", "signature_text"]
