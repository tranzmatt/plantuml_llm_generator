"""
Harness mode (DESIGN.md 13): a coding agent is the model.

The pipeline runs exactly as it does with a model server, except that the
backend defers: every request the pipeline would send becomes a task, and the
agent answers it through the MCP server (archdoc.mcp_server). So the agent sees
what a model of the launch tier would be sent and nothing else (pseudonymized
under `cloud`), and its answers go through the same checks a model's do:
schema, ingress, verification against the code, the overlap guard.

A task's id is its request key (archdoc.cache.request_key), shortened. An
answer is stored under that key and the request that produced the task is run
again, so the verdict is the one a model's answer would get. Rejected answers
are not repaired in a second round: the agent resubmits to the same task with
the errors in hand. Requests that depend on other answers (package and system
summaries) appear once those are accepted, on the next pass.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .backends import BackendError, Completion, LLMBackend
from .cache import ResponseCache, request_key
from .egress import EgressLog, Payload, build_repo_map
from .extract import extract_python
from .overlap import SourceIndex
from .packing import Window, encode_payload
from .plan.artifacts import PlanContext
from .policy import load_config
from .pseudonym import AliasMap
from .trust import CLOUD

DEFAULT_WINDOW = 32768           # agents run models with large windows; packing sends more source
TASK_ID_LEN = 12
KINDS = {"sequence": "sequence diagram", "activity": "activity diagram labels",
         "usecase": "use-case diagram labels", "icd": "ICD narrative", "summary": "module/package summary",
         "behavior": "behavior specification", "tests": "test acceptance criteria"}


@dataclass
class Task:
    key: str
    payload: Payload
    system: str
    user: str
    schema: Dict[str, Any]
    retry: Callable[[], Any]
    terms: Callable[[str], str]
    status: str = "open"                 # open | accepted | rejected
    errors: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.key[:TASK_ID_LEN]

    @property
    def kind(self) -> str:
        return self.payload.artifact.split(":", 1)[0]


class HarnessBackend(LLMBackend):
    """Answers from the agent's submissions; a request with none is a pending task."""

    name = "harness"
    model = "harness"
    deferred = True

    def __init__(self, answers_dir: str, window: int = DEFAULT_WINDOW):
        self.answers = ResponseCache(answers_dir)
        self.window = window
        self.tasks: Dict[str, Task] = {}
        self._current = None
        self._last: Optional[Task] = None

    def begin(self, payload: Payload, retry: Callable[[], Any], terms: Callable[[str], str]) -> None:
        self._current = (payload, retry, terms)

    def complete_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 2048) -> Completion:
        key = request_key(self.model, None, system, user, schema, max_tokens)
        task = self.tasks.get(key)
        if task is None:
            payload, retry, terms = self._current
            task = self.tasks[key] = Task(key, payload, system, user, schema, retry, terms)
        self._last = task
        answer = self.answers.get(key)
        if answer is None:
            task.status, task.errors = "open", []
            raise BackendError("waiting for the harness agent to answer this task", pending=True)
        return Completion(answer["text"])

    def verdict(self, errors: List[str]) -> None:
        if self._last is not None:
            self._last.status = "rejected" if errors else "accepted"
            self._last.errors = list(errors)


class HarnessSession:
    """One repo, output directory and tier, fixed at launch."""

    def __init__(self, repo: str, output: str, trust: str = CLOUD, config: Optional[str] = None,
                 exclude: Optional[List[str]] = None, window: int = DEFAULT_WINDOW, max_entries: int = 3,
                 template: str = "lightweight", design_template: str = "lightweight", embed: str = "svg"):
        self.repo, self.output, self.tier = os.path.abspath(repo), os.path.abspath(output), trust
        self.config_path, self.exclude = config, list(exclude or [])
        self.max_entries, self.template, self.design_template, self.embed = \
            max_entries, template, design_template, embed
        state = os.path.join(self.output, ".archdoc")
        self.backend = HarnessBackend(os.path.join(state, "harness", "answers"), window)
        self.log = EgressLog(os.path.join(state, "egress.jsonl"))
        self.alias_path = os.path.join(state, "alias_map.json")
        self.run_log = os.path.join(state, "harness", "runs.log")
        # for a runner outside the agent: is anything left, and were the outputs written since?
        self.status_path = os.path.join(state, "harness", "status.json")
        self.dirty = True
        self.last: Dict[str, Any] = {}

    @property
    def aliased(self) -> bool:
        return self.tier == CLOUD

    # -- passes -----------------------------------------------------------------------

    def _args(self, final: bool, export: Optional[str]):
        from .cli import build_parser
        argv = ["all", self.repo, "-o", self.output, "--design", "--narrative",
                "--types", "class,component,deployment,flow,state,object,usecase,activity,sequence",
                "--max-entries", str(self.max_entries), "--template", self.template,
                "--design-template", self.design_template, "--embed", self.embed if final else "link"]
        for g in self.exclude:
            argv += ["--exclude", g]
        if self.config_path:
            argv += ["--config", self.config_path]
        if not final:
            argv.append("--no-validate")
        if export:
            argv += ["--export", export]
        return build_parser().parse_args(argv)

    def run(self, final: bool = False, export: Optional[str] = None) -> Dict[str, Any]:
        """One pass of the whole pipeline: writes every output with the answers so far
        and collects the requests still open. Its console output goes to a log file,
        never to the agent (it names real files and functions)."""
        from . import cli
        args = self._args(final, export)
        cfg = load_config(self.repo, extra=self.config_path)
        buf = io.StringIO()
        result: Dict[str, Any] = {"syntax_errors": None}
        with redirect_stdout(buf):
            ir = extract_python(self.repo, exclude=cfg.exclude + self.exclude)
            ctx = PlanContext(
                ir=ir, config=cfg, tier=self.tier, aliases=AliasMap.load(self.alias_path), backend=self.backend,
                log=None,                # a task is logged when the agent fetches it (get), not on every pass
                depth=args.depth, max_nodes=args.max_nodes, alias_path=self.alias_path,
                overlap=SourceIndex(self.repo, ir, cfg.overlap_min_words, cfg.overlap_mode),
                root=self.repo, window=Window(tokens=self.backend.window))
            self.backend.tasks = {}
            try:
                diagrams = cli._generate_diagrams(ir, args, ctx)
            except SystemExit as exc:               # PlantUML syntax failures: outputs are written anyway
                result["syntax_errors"] = str(exc)
                diagrams = []
            path, overview = cli._generate_design(ir, args, cfg, ctx, diagrams)
            docs = [path] + (cli._generate_icd(ir, args, diagrams, ctx, overview) if ir.interfaces else [])
            cli._export(docs, args)
            ctx.aliases.save(self.alias_path)
        os.makedirs(os.path.dirname(self.run_log), exist_ok=True)
        with open(self.run_log, "a", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
        self.ir, self.config, self.aliases, self.dirty = ir, cfg, ctx.aliases, False
        written = [ln.split("wrote ", 1)[1].split("  ")[0] for ln in buf.getvalue().splitlines() if "wrote " in ln]
        result.update({"diagrams": len(diagrams), "documents": len(docs), "files": written, "documents_paths": docs})
        self.last = result
        todo = sum(c["open"] + c["rejected"] for c in self.counts().values())
        with open(self.status_path, "w", encoding="utf-8") as fh:
            json.dump({"todo": todo, "outputs_written": final, "tasks": self.counts()}, fh)
        return result

    def ensure(self) -> None:
        if self.dirty:
            self.run()

    # -- what the agent sees ------------------------------------------------------------

    def counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for t in self.backend.tasks.values():
            c = out.setdefault(t.kind, {"open": 0, "rejected": 0, "accepted": 0})
            c[t.status] += 1
        return out

    def overview(self) -> Dict[str, Any]:
        self.ensure()
        cap = Window(tokens=self.backend.window).input_chars(0) // 4
        repo_map = build_repo_map(self.ir, self.config, self.tier, self.aliases, cap)
        self.aliases.save(self.alias_path)
        return {
            "tier": self.tier,
            "pseudonymized": self.aliased,
            "system_map": encode_payload({"system_map": repo_map}, self.config.payload_encoding),
            "size": {"modules": len(self.ir.modules), "interfaces": len(self.ir.interfaces),
                     "functions": sum(1 for _ in self.ir.iter_functions())},
            "task_kinds": KINDS,
            "tasks": self.counts(),
        }

    def list_tasks(self, kind: Optional[str] = None, status: str = "todo", limit: int = 25) -> Dict[str, Any]:
        self.ensure()
        want = {"todo": ("open", "rejected"), "all": ("open", "rejected", "accepted")}.get(status, (status,))
        tasks = [t for t in self.backend.tasks.values() if t.status in want and (not kind or t.kind == kind)]
        rows = []
        for t in tasks[:max(1, limit)]:
            row = {"id": t.id, "kind": t.kind, "status": t.status}
            if not self.aliased:
                row["about"] = t.payload.artifact
            if t.errors:
                row["errors"] = len(t.errors)
            rows.append(row)
        return {"tasks": rows, "shown": len(rows), "matching": len(tasks), "totals": self.counts()}

    def _task(self, task_id: str) -> Task:
        self.ensure()
        task_id = task_id.strip()
        found = [t for t in self.backend.tasks.values() if t.key.startswith(task_id)] if task_id else []
        if len(found) != 1:
            raise KeyError(f"no task {task_id!r} (it may be settled or outdated by a code change: list_tasks)")
        return found[0]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        t = self._task(task_id)
        self.log.record(t.payload, "harness", None, model="harness agent", request=t.key, event="shown")
        out = {"id": t.id, "kind": t.kind, "status": t.status, "instructions": t.system, "input": t.user,
               "answer_schema": t.schema}
        if t.errors:
            out["previous_answer_rejected_for"] = t.errors
        return out

    def submit(self, task_id: str, answer: Any) -> Dict[str, Any]:
        t = self._task(task_id)
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        self.backend.answers.put(t.key, t.payload.artifact, text, None, None)
        with redirect_stdout(io.StringIO()):
            t.retry()
        self.log.record(t.payload, "harness", None, model="harness agent", request=t.key, event="answered",
                        ok=t.status == "accepted", errors=t.errors, response=text if self.log.include_payload else None)
        self.dirty = True                     # an accepted part can open the summary above it
        out: Dict[str, Any] = {"id": t.id, "status": t.status}
        if t.errors:
            out["errors"] = t.errors
            out["next"] = "fix these and submit again to the same id"
        return out

    def write_outputs(self, export: Optional[str] = None) -> Dict[str, Any]:
        res = self.run(final=True, export=export)
        out = {"diagrams": res["diagrams"], "documents": res["documents"], "tasks": self.counts()}
        if res["syntax_errors"]:
            out["syntax_errors"] = res["syntax_errors"]
        if not self.aliased:                 # file names carry real names
            out["documents_paths"] = res["documents_paths"]
            out["output_dir"] = self.output
        return out


__all__ = ["HarnessBackend", "HarnessSession", "Task", "KINDS"]
