"""
Planner, backends, sequence rendering and ICD narrative.

Model-free by default (ScriptedBackend and a throwaway local HTTP server).
Live test against a real Ollama model, opt-in:
    ARCHDOC_LIVE_MODEL=mistral-nemo:12b python -m pytest tests/test_plan.py -k live
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import archdoc.cli as cli
from archdoc.backends import BackendError, OllamaBackend, OpenAICompatBackend, ScriptedBackend
from archdoc.egress import EgressLog
from archdoc.extract import extract_python
from archdoc.plan.artifacts import PlanContext, check_sequence, plan_narrative, plan_sequence
from archdoc.plan.entries import call_slice, discover_entries, select_entries
from archdoc.plan.schemas import sequence_schema
from archdoc.policy import Config
from archdoc.pseudonym import AliasMap
from archdoc.render.icd import apply_narrative, build_icd_context, render_icd
from archdoc.render.puml.sequence import nested_groups, render_sequence_diagram
from archdoc.trust import Backend
from archdoc.validate import check_syntax, plantuml_command

from tests.test_egress import BODY_ONLY, CLOUD_FORBIDDEN, SECRET

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")
READ_ITEM = "shop.api:read_item"


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _enum(schema, *path):
    node = schema
    for key in path:
        node = node["properties"][key] if key != "items" else node["items"]
    return node["enum"]


def payload_of(user):
    """The JSON payload inside a planner prompt (repair rounds append text after it)."""
    return json.loads(user.split("INPUT:\n", 1)[1].split("\n\nYOUR PREVIOUS ANSWER")[0])


def sequence_responder(system, user, schema):
    """A faithful model: walks the call graph in the payload, depth first, in call order."""
    payload = payload_of(user)
    funcs = {f["ref"]: f for f in payload["functions"]}
    handler = payload["entry"]["handler"]
    owner = lambda f: funcs[f]["owner"]
    msgs = [{"from": "client", "to": owner(handler), "call": handler, "label": f"invoke {handler}", "reply": False}]
    seen = {handler}

    def walk(f):
        for c in funcs[f].get("calls", []):
            if c in funcs and c not in seen:
                seen.add(c)
                msgs.append({"from": owner(f), "to": owner(c), "call": c, "label": f"then {c}", "reply": False})
                walk(c)

    walk(handler)
    msgs.append({"from": owner(handler), "to": "client", "call": "", "label": "response", "reply": True})
    parts = list(dict.fromkeys(m[k] for m in msgs for k in ("from", "to") if m[k] != "client"))
    return json.dumps({
        "title": "flow",
        "participants": [{"ref": "client", "kind": "actor"}] + [{"ref": p, "kind": "participant"} for p in parts],
        "messages": msgs,
        "groups": [],
    })


def narrative_responder(system, user, schema):
    ops = _enum(schema, "operations", "items", "ref")
    params_node = schema["properties"]["operations"]["items"]["properties"]["params"]
    params = params_node["items"]["properties"]["name"].get("enum", [])
    return json.dumps({
        "purpose": f"Serves {ops[0]} to callers.",
        "operations": [{"ref": o, "summary": f"Does {o}.",
                        "params": [{"name": p, "description": f"The {p}."} for p in params]} for o in ops],
    })


def _ctx(ir, tier, responder, tmp_path=None, config=None):
    log = EgressLog(str(tmp_path / "egress.jsonl")) if tmp_path else None
    return PlanContext(ir=ir, config=config or Config(), tier=tier, aliases=AliasMap(),
                       backend=ScriptedBackend(responder), log=log)


def _entry(ir, handler=READ_ITEM):
    return next(e for e in discover_entries(ir) if e.handler == handler)


# -- entries and slices ---------------------------------------------------------

def test_call_slice_follows_source_order(ir):
    assert call_slice(ir, "shop.cli:cmd_serve") == [
        "shop.cli:cmd_serve", "shop.repo:build_repo", "shop.repo:CachedRepo.get", "shop.repo:ItemRepo.get",
    ]
    assert call_slice(ir, "shop.cli:cmd_serve", depth=1) == ["shop.cli:cmd_serve", "shop.repo:build_repo"]


def test_select_entries(ir):
    picked = select_entries(ir, [], limit=2, depth=3, max_nodes=25)
    # largest call graphs first; read_item and main tie (3 functions) and sort by id
    assert [e.handler for e in picked] == ["shop.cli:cmd_serve", "shop.api:read_item"]
    by_op = select_entries(ir, ["if:http:shop.api:router:GET /{item_id}"], 3, 3, 25)
    assert by_op[0].handler == READ_ITEM
    with pytest.raises(ValueError):
        select_entries(ir, ["nope:nope"], 3, 3, 25)


# -- planner loop -----------------------------------------------------------------

@pytest.mark.parametrize("tier", ["local", "cloud"])
def test_sequence_plan_round_trip(ir, tier):
    ctx = _ctx(ir, tier, sequence_responder)
    result = plan_sequence(ctx, _entry(ir))
    assert result.ok and result.attempts == 1
    refs = {p["ref"] for p in result.plan["participants"]}
    assert refs == {"client", "shop.api", "shop.repo:ItemRepo", "shop.repo:Database"}   # real ids either way
    assert result.plan["messages"][1]["call"] == "shop.repo:ItemRepo.get"


def test_cloud_prompt_contains_no_real_names(ir, tmp_path):
    ctx = _ctx(ir, "cloud", sequence_responder, tmp_path)
    plan_sequence(ctx, _entry(ir, "shop.cli:cmd_serve"))
    ctx.backend.respond = narrative_responder
    for iface in ir.interfaces:
        plan_narrative(ctx, iface)
    sent = "\n".join(c["user"].split("INPUT:\n", 1)[1] + json.dumps(c["schema"]) for c in ctx.backend.calls)
    leaked = [w for w in CLOUD_FORBIDDEN + BODY_ONLY + [SECRET]
              if re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", sent)]
    assert leaked == []
    log = [json.loads(line) for line in (tmp_path / "egress.jsonl").read_text().splitlines()]
    assert len(log) == 1 + len(ir.interfaces) and all(e["ok"] and e["tier"] == "cloud" for e in log)


def test_repair_after_invalid_json(ir):
    answers = iter(["not json at all", None])

    def responder(system, user, schema):
        a = next(answers)
        if a is None:
            assert "IT WAS REJECTED" in user and "not valid JSON" in user
            return sequence_responder(system, user, schema)
        return a

    result = plan_sequence(_ctx(ir, "cloud", responder), _entry(ir))
    assert result.ok and result.attempts == 2


def test_hallucinated_reference_is_rejected(ir):
    # a server that ignores the schema enum: the model invents @C999
    def responder(system, user, schema):
        plan = json.loads(sequence_responder(system, user, schema))
        plan["participants"].append({"ref": "@C999", "kind": "participant"})
        return json.dumps(plan)

    result = plan_sequence(_ctx(ir, "cloud", responder), _entry(ir))
    assert not result.ok and result.attempts == 2
    assert any("@C999" in e for e in result.errors)


def test_backend_error_is_reported_not_raised(ir):
    def responder(system, user, schema):
        raise BackendError("connection refused")

    result = plan_sequence(_ctx(ir, "local", responder), _entry(ir))
    assert not result.ok and result.errors == ["backend: connection refused"]


def test_check_sequence_group_ranges():
    plan = {"participants": [{"ref": "a"}, {"ref": "a"}],
            "messages": [{"from": "a", "to": "b"}], "groups": [{"first": 0, "last": 3}]}
    errors = check_sequence(plan)
    assert plan["participants"] == [{"ref": "a"}]          # duplicates are merged, not an error
    assert len(errors) == 1 and "outside" in errors[0]


# -- sequence rendering -------------------------------------------------------------

PLAN = {
    "title": "Read: item",
    "participants": [{"ref": "client", "kind": "actor"}, {"ref": "shop.api", "kind": "boundary"},
                     {"ref": "shop.repo:ItemRepo", "kind": "control"}],
    "messages": [
        {"from": "client", "to": "shop.api", "call": "", "label": "GET item", "reply": False},
        {"from": "shop.api", "to": "shop.repo:ItemRepo", "call": "shop.repo:ItemRepo.get",
         "label": "fetch it", "reply": False},
        {"from": "shop.repo:ItemRepo", "to": "shop.repo:Database", "call": "shop.repo:Database.query",
         "label": "", "reply": False},
        {"from": "shop.api", "to": "client", "call": "", "label": "404: not found", "reply": True},
    ],
    "groups": [{"kind": "alt", "label": "missing", "first": 0, "last": 2},
               {"kind": "loop", "label": "crossing", "first": 1, "last": 3},    # crosses "missing"
               {"kind": "opt", "label": "inner", "first": 2, "last": 2}],       # nests inside it
}


def test_sequence_render(ir):
    text, warnings = render_sequence_diagram(ir, PLAN, "seq")
    assert 'actor "Client" as P_client' in text
    assert 'participant "Database" as P_shop_repo_Database' in text      # implicitly declared
    assert "P_shop_api -> P_shop_repo_ItemRepo : fetch it" in text
    assert "P_shop_repo_ItemRepo -> P_shop_repo_Database : query()" in text   # empty label -> call name
    assert "P_shop_api --> P_client : 404 - not found" in text               # ':' cannot end a label
    assert warnings == ["group 'crossing': overlaps another group without nesting"]
    assert text.count("\nend") == 2


def test_sequence_title_fallback(ir):
    text, _ = render_sequence_diagram(ir, {**PLAN, "title": ""}, "seq", title="router: GET /{item_id}")
    assert "title router: GET /{item_id}" in text
    text, _ = render_sequence_diagram(ir, PLAN, "seq", title="ignored")
    assert "title Read: item" in text


def test_reverse_mapping_keeps_alias_labels_and_clears_invented_ones(ir):
    def responder(system, user, schema):
        plan = json.loads(sequence_responder(system, user, schema))
        plan["title"] = "Order Processing Flow"                 # invented: no reference
        plan["messages"][0]["label"] = "Place Order"            # invented, no call
        return json.dumps(plan)                                 # other labels: "then @F.."

    result = plan_sequence(_ctx(ir, "cloud", responder), _entry(ir, "shop.cli:cmd_serve"))
    assert result.ok and result.payload_aliased
    labels = [m["label"] for m in result.plan["messages"]]
    assert labels[0] == "" and result.plan["title"] == ""
    assert labels[1] == "then build_repo"                       # reverse-mapped, kept
    assert labels[-1] == ""                                     # "response": no reference either
    # title, "Place Order" and "response"
    assert any("cleared 3 label(s)" in w for w in result.warnings)
    text, _ = render_sequence_diagram(ir, result.plan, "s", client_name="User", title="shopctl: serve")
    assert "title shopctl: serve" in text and "Order" not in text
    assert not re.search(r"@[MCFIOLp]\d+", text)


def test_local_tier_keeps_all_labels(ir):
    def responder(system, user, schema):
        plan = json.loads(sequence_responder(system, user, schema))
        plan["messages"][0]["label"] = "Place Order"
        return json.dumps(plan)

    result = plan_sequence(_ctx(ir, "local", responder), _entry(ir, "shop.cli:cmd_serve"))
    assert not result.payload_aliased and result.plan["messages"][0]["label"] == "Place Order"


def test_nested_groups():
    kept, dropped = nested_groups([{"first": 0, "last": 5, "label": "a"}, {"first": 1, "last": 2, "label": "b"},
                                   {"first": 2, "last": 7, "label": "c"}], 6)
    assert [g["label"] for g in kept] == ["a", "b"] and len(dropped) == 1


@pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not available")
def test_sequence_plantuml_syntax(ir):
    text, _ = render_sequence_diagram(ir, PLAN, "seq")
    res = check_syntax(text)
    assert res.ok, f"line {res.line}: {res.message}"


# -- ICD narrative ------------------------------------------------------------------

def test_narrative_fills_only_tbd_and_marks_generated(ir):
    ctx = _ctx(ir, "cloud", narrative_responder)
    icd = build_icd_context(ir)
    before = len(icd["gaps"])
    router = next(i for i in ir.interfaces if i.id == "if:http:shop.api:router")
    result = plan_narrative(ctx, router)
    assert result.ok
    filled = apply_narrative(icd, router.id, result.plan)
    view = next(i for i in icd["interfaces"] if i["id"] == router.id)
    get, post = view["operations"]
    assert get["summary"] == "Fetch one item." and not get["summary_generated"]   # docstring wins
    assert post["summary_generated"] and post["summary"].startswith("Does ")
    assert view["purpose_generated"] and "@" not in view["purpose"]              # restored
    assert all(p["description_generated"] for p in get["params"])
    assert len(icd["gaps"]) == before - filled and filled == 5
    text = render_icd(icd)
    assert "*(generated)*" in text and "drafted by a" in text


# -- backends over HTTP ---------------------------------------------------------------

class _Recorder(BaseHTTPRequestHandler):
    reply = {}
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, dict(self.headers), body))
        data = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    _Recorder.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_backend_request(http_server):
    _Recorder.reply = {"message": {"content": '{"a":1}'}, "prompt_eval_count": 10, "eval_count": 4,
                       "total_duration": 2_000_000_000}
    b = OllamaBackend(Backend("o", http_server, "m", "local", kind="ollama", num_ctx=4096))
    c = b.complete_json("sys", "user", {"type": "object"}, max_tokens=99)
    path, _, body = _Recorder.seen[0]
    assert path == "/api/chat" and body["format"] == {"type": "object"} and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 99}
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.seconds) == ('{"a":1}', 10, 4, 2.0)


def test_openai_backend_request(http_server, monkeypatch):
    monkeypatch.setenv("ARCHDOC_TEST_KEY", "k-123")
    _Recorder.reply = {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 7}}
    b = OpenAICompatBackend(Backend("v", http_server + "/v1", "m", "local", api_key_env="ARCHDOC_TEST_KEY"))
    b.complete_json("sys", "user", {"type": "object"})
    path, headers, body = _Recorder.seen[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer k-123"
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_openai_backend_missing_key():
    with pytest.raises(BackendError, match="NOT_SET_VAR"):
        OpenAICompatBackend(Backend("v", "http://x", "m", api_key_env="NOT_SET_VAR"))


# -- CLI end to end with a scripted model -------------------------------------------

def test_cli_all_with_llm(tmp_path, monkeypatch, capsys):
    def respond(system, user, schema):
        return sequence_responder(system, user, schema) if "messages" in schema["properties"] \
            else narrative_responder(system, user, schema)

    monkeypatch.setattr(cli, "make_backend", lambda cfg: ScriptedBackend(respond))
    out = tmp_path / "out"
    rc = cli.main(["all", FIXTURE, "-o", str(out), "--types", "sequence", "--narrative", "--model", "m",
                   "--trust", "cloud", "--max-entries", "2", "--no-validate", "--no-date"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "tier cloud" in printed
    assert sorted(os.listdir(out / "diagrams")) == ["sample_app_sequence_shop_api_read_item.puml",
                                                     "sample_app_sequence_shop_cli_cmd_serve.puml"]
    assert (out / ".archdoc" / "alias_map.json").exists() and (out / ".archdoc" / "egress.jsonl").exists()
    assert len(os.listdir(out / ".archdoc" / "plans")) == 2
    seq = (out / "diagrams" / "sample_app_sequence_shop_cli_cmd_serve.puml").read_text()
    assert "then build_repo" in seq and not re.search(r"@[MCFIOLp]\d+", seq)      # reverse-mapped labels
    assert "*(generated)*" in (out / "icd" / "sample_app_icd.md").read_text()


def test_cli_adhoc_backend_defaults_to_cloud(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_backend", lambda cfg: ScriptedBackend(sequence_responder))
    cli.main(["diagrams", FIXTURE, "-o", str(tmp_path), "--types", "sequence", "--model", "m",
              "--max-entries", "1", "--no-validate"])
    assert "tier cloud: no trust declared" in capsys.readouterr().out


# -- live (opt-in) --------------------------------------------------------------------

LIVE_MODEL = os.environ.get("ARCHDOC_LIVE_MODEL")


@pytest.mark.skipif(not LIVE_MODEL, reason="set ARCHDOC_LIVE_MODEL to run against local Ollama")
@pytest.mark.parametrize("tier", ["local", "cloud"])
def test_live_ollama(ir, tier):
    backend = OllamaBackend(Backend("live", os.environ.get("ARCHDOC_LIVE_URL", "http://localhost:11434"),
                                    LIVE_MODEL, "local", kind="ollama"))
    ctx = PlanContext(ir=ir, config=Config(), tier=tier, aliases=AliasMap(), backend=backend)
    seq = plan_sequence(ctx, _entry(ir, "shop.cli:cmd_serve"))
    assert seq.ok, seq.errors
    router = next(i for i in ir.interfaces if i.id == "if:http:shop.api:router")
    nar = plan_narrative(ctx, router)
    assert nar.ok, nar.errors


# -- servers that do not enforce the schema ------------------------------------------

def test_alias_examples_only_reach_models_that_see_aliases(ir):
    # llama4:scout copied the "@C3"/"@L4" examples into labels when shown them
    # alongside real names
    for tier, expect in (("local", False), ("cloud", True)):
        ctx = _ctx(ir, tier, sequence_responder)
        plan_sequence(ctx, _entry(ir))
        ctx.backend.respond = narrative_responder
        plan_narrative(ctx, ir.interfaces[0])
        for c in ctx.backend.calls:
            prose = c["system"].split("RESPONSE FORMAT", 1)[0] + c["user"].split("INPUT:\n", 1)[0]
            assert bool(re.search(r"@[A-Z]\d", prose)) is expect, tier


def test_response_format_is_in_the_prompt(ir):
    ctx = _ctx(ir, "cloud", sequence_responder)
    plan_sequence(ctx, _entry(ir, "shop.cli:cmd_serve"))
    system = ctx.backend.calls[0]["system"]
    assert "RESPONSE FORMAT" in system and '"kind": "participant" | "actor"' in system
    assert '<a ref exactly as it appears in the input, or "client">' in system
    fmt = system.split("RESPONSE FORMAT", 1)[1]
    assert "@C" not in fmt and "@F" not in fmt                 # ref lists stay out of the format


def test_schema_errors_name_the_allowed_keys(ir):
    def wrong_keys(system, user, schema):                     # what gpt-oss did before the format was shown
        plan = json.loads(sequence_responder(system, user, schema))
        for p in plan["participants"]:
            p["type"] = p.pop("kind")
        return json.dumps(plan)

    backend = ScriptedBackend(wrong_keys)
    ctx = PlanContext(ir=ir, config=Config(), tier="cloud", aliases=AliasMap(), backend=backend)
    plan_sequence(ctx, _entry(ir, "shop.cli:cmd_serve"))
    assert "(allowed keys: ref, kind)" in backend.calls[1]["user"]


def test_ollama_remote_host_detection(http_server, monkeypatch):
    from archdoc.backends import ollama_remote_host

    class Tags(_Recorder):
        def do_GET(self):
            data = json.dumps({"models": [
                {"name": "big:cloud-ish", "model": "big:cloud-ish", "remote_host": "https://ollama.com"},
                {"name": "mistral-nemo:12b", "model": "mistral-nemo:12b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Tags)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert ollama_remote_host(url, "big:cloud-ish") == "https://ollama.com"   # name matches no pattern
        assert ollama_remote_host(url, "mistral-nemo:12b") is None
        assert ollama_remote_host("http://127.0.0.1:1", "x") is None             # unreachable: no opinion
    finally:
        server.shutdown()


# -- unreachable or overloaded backends ------------------------------------------------

def test_circuit_breaker_skips_after_unavailable(ir):
    calls = []

    def down(system, user, schema):
        calls.append(1)
        raise BackendError("timed out", unavailable=True)

    ctx = _ctx(ir, "local", down)
    first = plan_sequence(ctx, _entry(ir, READ_ITEM))
    second = plan_narrative(ctx, ir.interfaces[0])
    assert not first.ok and not second.ok and len(calls) == 1          # second request never sent
    assert second.errors[0].startswith("skipped: backend unavailable earlier in this run")


def test_rejected_request_does_not_trip_breaker(ir):
    def reject(system, user, schema):
        raise BackendError("HTTP 400: bad request")                  # this prompt, not the server

    ctx = _ctx(ir, "local", reject)
    plan_sequence(ctx, _entry(ir, READ_ITEM))
    assert ctx.backend.unavailable is None
    assert len(ctx.backend.calls) == 1
    plan_sequence(ctx, _entry(ir, READ_ITEM))
    assert len(ctx.backend.calls) == 2                                # still tried


def test_cli_skips_llm_when_backend_does_not_answer(tmp_path, monkeypatch, capsys):
    class Dead(ScriptedBackend):
        def health(self, timeout=10):
            raise BackendError("http://spark:11434/api/version: timed out", unavailable=True)

    monkeypatch.setattr(cli, "make_backend", lambda cfg: Dead(sequence_responder))
    out = tmp_path / "out"
    rc = cli.main(["all", FIXTURE, "-o", str(out), "--types", "class,sequence", "--narrative", "--model", "m",
                   "--trust", "local", "--no-validate", "--no-date"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "[SKIP] LLM outputs: no usable backend" in printed and "api/version: timed out" in printed
    assert os.listdir(out / "diagrams") == ["sample_app_class.puml"]   # deterministic output still written
    assert (out / "icd" / "sample_app_icd.md").exists()


def test_ollama_warm_up_and_keep_alive(http_server):
    _Recorder.reply = {"message": {"content": "{}"}, "load_duration": 3_000_000_000}
    b = OllamaBackend(Backend("o", http_server, "m", "local", kind="ollama", num_ctx=4096, keep_alive="1h"))
    assert b.warm_up() == 3.0
    b.complete_json("s", "u", {})
    (warm_path, _, warm), (chat_path, _, chat) = _Recorder.seen
    assert warm_path == "/api/generate" and warm["prompt"] == "" and warm["options"]["num_ctx"] == 4096
    assert warm["keep_alive"] == "1h" and chat["keep_alive"] == "1h"   # same settings: no reload between


def test_unreachable_server_is_unavailable():
    b = OllamaBackend(Backend("o", "http://127.0.0.1:1", "m", "local", kind="ollama", timeout=2))
    with pytest.raises(BackendError) as exc:
        b.health(timeout=2)
    assert exc.value.unavailable


# -- choosing among several backends -------------------------------------------------

def _selection_config(tmp_path, allow_cloud=False):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f"""
[llm]
prefer = ["spark-vllm", "spark-ollama", "laptop", "ollama-cloud"]
allow_cloud_fallback = {str(allow_cloud).lower()}

[backends.spark-vllm]
url = "http://10.0.0.5:9999/v1"
model = "nemotron"
kind = "openai"
trust = "local"

[backends.spark-ollama]
url = "http://10.0.0.5:11434"
model = "gemma4:31b"
kind = "ollama"
trust = "local"

[backends.laptop]
url = "http://127.0.0.1:11434"
model = "mistral-nemo:12b"
kind = "ollama"
trust = "local"

[backends.ollama-cloud]
url = "http://127.0.0.1:11434"
model = "gpt-oss:120b-cloud"
kind = "ollama"
trust = "local"
""")
    return str(cfg)


def _fleet(monkeypatch, healthy):
    """make_backend double: only backends named in `healthy` answer their health check."""
    made = []

    class Fake(ScriptedBackend):
        def health(self, timeout=10):
            if self.name not in healthy:
                raise BackendError(f"{self.name}: connection refused", unavailable=True)

    def make(cfg):
        b = Fake(sequence_responder, name=cfg.name, model=cfg.model, url=cfg.url)
        made.append(cfg.name)
        return b

    monkeypatch.setattr(cli, "make_backend", make)
    monkeypatch.setattr(cli, "ollama_remote_host", lambda url, model: None)
    return made


def _run(tmp_path, config, *extra):
    return cli.main(["diagrams", FIXTURE, "-o", str(tmp_path / "out"), "--types", "sequence", "--max-entries", "1",
                     "--no-validate", "--config", config, *extra])


def test_prefer_list_uses_first_available(tmp_path, monkeypatch, capsys):
    made = _fleet(monkeypatch, healthy={"laptop"})
    _run(tmp_path, _selection_config(tmp_path))                  # no --backend: [llm] prefer applies
    out = capsys.readouterr().out
    assert "LLM: laptop (mistral-nemo:12b" in out
    assert "(skipped spark-vllm: spark-vllm: connection refused)" in out
    assert made == ["spark-vllm", "spark-ollama", "laptop"]       # stopped at the first that worked


def test_prefer_list_never_falls_back_to_cloud_by_default(tmp_path, monkeypatch, capsys):
    _fleet(monkeypatch, healthy={"ollama-cloud"})                # only the cloud model is up
    _run(tmp_path, _selection_config(tmp_path))
    out = capsys.readouterr().out
    assert "[SKIP] LLM outputs: no usable backend" in out
    assert "ollama-cloud: cloud, and [llm] allow_cloud_fallback is false" in out


def test_cloud_fallback_when_allowed(tmp_path, monkeypatch, capsys):
    _fleet(monkeypatch, healthy={"ollama-cloud"})
    _run(tmp_path, _selection_config(tmp_path, allow_cloud=True))
    assert "LLM: ollama-cloud (gpt-oss:120b-cloud" in capsys.readouterr().out


def test_explicit_list_and_single_backend(tmp_path, monkeypatch, capsys):
    cfg = _selection_config(tmp_path)
    made = _fleet(monkeypatch, healthy={"spark-ollama", "ollama-cloud"})
    _run(tmp_path, cfg, "--backend", "laptop,spark-ollama")
    assert "LLM: spark-ollama" in capsys.readouterr().out and made == ["laptop", "spark-ollama"]
    _run(tmp_path, cfg, "--backend", "ollama-cloud")             # named alone: cloud is a choice, not a fallback
    assert "LLM: ollama-cloud" in capsys.readouterr().out


def test_prefer_must_name_defined_backends(tmp_path):
    from archdoc.policy import PolicyError, load_config
    (tmp_path / ".archdoc.toml").write_text('[llm]\nprefer = ["nope"]\n')
    with pytest.raises(PolicyError, match="undefined backend"):
        load_config(str(tmp_path), user_config=None)


def test_health_requires_the_model(http_server):
    _Recorder.reply = {}

    class Tags(_Recorder):
        def do_GET(self):
            data = json.dumps({"data": [{"id": "nvidia/nemotron"}]} if self.path.endswith("/models")
                              else {"models": [{"name": "gemma4:31b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Tags)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        OllamaBackend(Backend("o", url, "gemma4:31b", kind="ollama")).health()
        with pytest.raises(BackendError, match="does not have model 'llama4:scout'"):
            OllamaBackend(Backend("o", url, "llama4:scout", kind="ollama")).health()
        OpenAICompatBackend(Backend("v", url + "/v1", "nvidia/nemotron")).health()
        with pytest.raises(BackendError, match="serves nvidia/nemotron, not 'other'"):
            OpenAICompatBackend(Backend("v", url + "/v1", "other")).health()
    finally:
        server.shutdown()


def test_ids_in_labels_render_as_short_names(ir):
    plan = json.loads(json.dumps(PLAN))
    plan["messages"][1]["label"] = "call @shop.repo:ItemRepo.get for shop.models:Item"
    text, _ = render_sequence_diagram(ir, plan, "s")
    assert "P_shop_api -> P_shop_repo_ItemRepo : call ItemRepo.get for Item" in text


def test_misspelled_names_in_labels_are_corrected():
    from archdoc.plan.artifacts import correct_misspelled_names
    ids = ["shop.cli:main", "shop.repo:Database.query", "shop.repo:ItemRepo.get"]
    plan = {"title": "", "groups": [{"label": "if ItemRepo.gte fails"}],
            "messages": [{"label": "Dtaabase.query"}, {"label": "maxn reply"}, {"label": "return items"}]}
    warnings = correct_misspelled_names(None, plan, ids)
    assert [m["label"] for m in plan["messages"]] == ["Database.query", "main reply", "return items"]
    assert plan["groups"][0]["label"] == "if ItemRepo.get fails"
    assert len(warnings) == 3
