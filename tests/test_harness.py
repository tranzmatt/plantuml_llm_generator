"""
Harness mode (phase 5): the MCP server hands an agent the requests a model
would get and checks its answers the same way. Model-free: answers are
synthesized from each task's schema.
"""

import io
import json
import os
import tomllib

import pytest

from archdoc.harness import HarnessSession
from archdoc.mcp_server import TOOLS, Server
from archdoc.profiles import opencode_config, server_settings, write_profile

from tests.test_design import auto_answer
from tests.test_egress import CLOUD_FORBIDDEN, _word_hits

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")


def _answer_all(session, seen=None, shown=None):
    """Answer every task once, round after round, as an agent would; returns kinds per round."""
    seen = set() if seen is None else seen
    rounds = []
    while True:
        todo = [t for t in session.list_tasks(limit=200)["tasks"] if t["id"] not in seen]
        if not todo:
            return rounds
        rounds.append(sorted({t["kind"] for t in todo}))
        for t in todo:
            task = session.get_task(t["id"])
            res = session.submit(t["id"], auto_answer(task["answer_schema"]))
            if shown is not None:
                shown += [json.dumps(t), task["input"], json.dumps(task["answer_schema"]), json.dumps(res)]
            seen.add(t["id"])


def test_round_trip_local(tmp_path):
    s = HarnessSession(FIXTURE, str(tmp_path / "out"), trust="local", embed="link")
    first = s.list_tasks(limit=200)
    assert first["totals"]["summary"]["open"] == 7 and all("about" in t for t in first["tasks"])
    rounds = _answer_all(s)
    # module summaries first; the package and system summaries only once those were accepted
    assert len(rounds) >= 2 and rounds[1] == ["summary"]
    totals = s.counts()
    assert totals["behavior"]["accepted"] == 9 and totals["summary"]["accepted"] >= 8
    out = s.write_outputs()
    assert out["documents"] == 2 and os.path.isfile(out["documents_paths"][0])
    status = json.load(open(tmp_path / "out" / ".archdoc" / "harness" / "status.json"))
    # what a runner checks: the synthesized sequence answers stay rejected
    assert status["todo"] == sum(c["open"] + c["rejected"] for c in totals.values()) and status["outputs_written"]
    design = open(out["documents_paths"][0]).read()
    assert "does the thing" in design
    # answers are kept: a new session (server restart) has nothing left that was accepted
    again = HarnessSession(FIXTURE, str(tmp_path / "out"), trust="local", embed="link")
    assert again.counts() == {} and again.list_tasks()["totals"] == totals


def test_rejected_answer_gets_reasons_and_can_be_resubmitted(tmp_path):
    s = HarnessSession(FIXTURE, str(tmp_path / "out"), trust="local")
    tid = s.list_tasks(kind="behavior", limit=1)["tasks"][0]["id"]
    task = s.get_task(tid)
    bad = s.submit(tid, {"purpose": "x"})
    assert bad["status"] == "rejected" and bad["errors"] and "next" in bad
    assert s.list_tasks(status="rejected")["tasks"][0]["id"] == tid
    assert s.get_task(tid)["previous_answer_rejected_for"] == bad["errors"]
    good = s.submit(tid, auto_answer(task["answer_schema"]))
    assert good == {"id": tid, "status": "accepted"}
    with pytest.raises(KeyError):
        s.get_task("nope")


def test_cloud_agent_never_sees_real_names(tmp_path):
    s = HarnessSession(FIXTURE, str(tmp_path / "out"), trust="cloud", embed="link")
    shown = [json.dumps(s.overview())]
    _answer_all(s, shown=shown)
    s.submit(s.list_tasks(status="all")["tasks"][0]["id"], {"bogus": 1})   # rejection reasons too
    shown.append(json.dumps(s.list_tasks(status="all", limit=200)))
    shown.append(json.dumps(s.write_outputs()))
    assert _word_hits({"t": "\n".join(shown)}, CLOUD_FORBIDDEN) == []
    assert "about" not in shown[-2] and "documents_paths" not in shown[-1]
    # what the agent was shown is in the egress log
    log = [json.loads(x) for x in open(tmp_path / "out" / ".archdoc" / "egress.jsonl")]
    assert {e["event"] for e in log} == {"shown", "answered"} and all(e["tier"] == "cloud" for e in log)


def _rpc(server, *msgs):
    out = io.StringIO()
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in msgs)), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_mcp_protocol(tmp_path):
    server = Server(HarnessSession(FIXTURE, str(tmp_path / "out"), trust="cloud"))
    init, tools, call, bad, missing = _rpc(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_tasks", "arguments": {"limit": 1}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_task", "arguments": {"id": "zz"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"})
    assert init["result"]["protocolVersion"] == "2025-03-26" and "tools" in init["result"]["capabilities"]
    assert [t["name"] for t in tools["result"]["tools"]] == [t["name"] for t in TOOLS]
    listed = json.loads(call["result"]["content"][0]["text"])
    assert call["result"]["isError"] is False and listed["shown"] == 1
    assert bad["result"]["isError"] is True and "no task" in bad["result"]["content"][0]["text"]
    assert missing["error"]["code"] == -32601


def test_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTUML_JAR", "/opt/plantuml.jar")
    work = tmp_path / "work"
    paths = write_profile(str(work / "oc"), "opencode", FIXTURE, str(tmp_path / "o"), "local", "nemo")
    cfg = json.load(open(paths[1]))
    assert paths[2].endswith("run-opencode.sh") and os.access(paths[2], os.X_OK)
    assert os.path.join(str(tmp_path / "o"), ".archdoc", "harness", "status.json") in open(paths[2]).read()
    assert cfg["policies"][-1] == {"action": "provider.use", "resource": "archdoc-local", "effect": "allow"}
    assert cfg["permissions"][0]["effect"] == "deny" and cfg["permissions"][-1]["action"] == "archdoc_*"
    assert cfg["agents"]["archdoc-local"]["system"].startswith("# archdoc agent")
    server = cfg["mcp"]["servers"]["archdoc"]
    # settings in the environment: a harness that echoes the command line shows no path
    assert server["command"][-1] == "serve-mcp" and not any("sample_app" in c for c in server["command"])
    assert server["environment"]["ARCHDOC_TRUST"] == "local" and server["environment"]["ARCHDOC_WINDOW"] == "8192"
    assert server["codemode"] is False and cfg["share"] == "disabled"
    assert server["environment"]["PLANTUML_JAR"] == "/opt/plantuml.jar"    # diagrams are checked and rendered

    cloud = opencode_config(server_settings(FIXTURE, "o", "cloud"), "cloud", "mistral/medium", None, "x")
    assert "policies" not in cloud and cloud["model"] == {"providerID": "mistral", "model": "medium"}
    with pytest.raises(ValueError):
        opencode_config([], "cloud", "medium", None, "x")

    vpaths = write_profile(str(work / "v"), "vibe", FIXTURE, "o", "local", "nemo")
    (agent,) = [p for p in vpaths if p.endswith(".toml")]
    assert vpaths[-1].endswith("run-vibe.sh") and os.access(vpaths[-1], os.X_OK)
    assert "vibe --agent archdoc-local --trust" in open(vpaths[-1]).read()
    v = tomllib.load(open(agent, "rb"))
    assert v["enabled_tools"] == ["archdoc_*"] and v["allowed_models"] == ["archdoc-local"]
    assert v["enable_telemetry"] is False and v["mcp_servers"][0]["transport"] == "stdio"
    assert v["mcp_servers"][0]["env"]["ARCHDOC_REPO"] == os.path.abspath(FIXTURE)
    assert v["system_prompt_id"] == "minimal"
    assert v["tools"]["archdoc_submit_task"]["permission"] == "always"

    with pytest.raises(ValueError, match="inside the target repo"):
        write_profile(os.path.join(FIXTURE, "h"), "vibe", FIXTURE, "o", "local", "nemo")
