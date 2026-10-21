"""
Harness profiles (DESIGN.md 13.4): `archdoc harness-init` writes a working
directory for opencode or Mistral Vibe in which the agent's only tools are the
archdoc MCP server's.

The directory is separate from the target repo, so a permission that slips
exposes an empty directory, not the source. The server is launched with the
profile's tier: `local` profiles may use only the one local provider they
define (switch profile, not model), `cloud` ones any provider.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from typing import Dict, List, Optional

from ..trust import CLOUD, LOCAL

HARNESSES = ("opencode", "vibe")
AGENTS_MD = os.path.join(os.path.dirname(__file__), "AGENTS.md")
TOOL_NAMES = ("overview", "list_tasks", "get_task", "submit_task", "write_outputs")
LOCAL_PROVIDER = "archdoc-local"
DEFAULT_LOCAL_URL = "http://localhost:11434/v1"      # Ollama's OpenAI-compatible endpoint
MCP_TIMEOUT_S = 900                                  # write_outputs renders every diagram


# what a task's input is packed for: it shares the agent's window with the
# harness's own prompt and the conversation so far
DEFAULT_WINDOWS = {LOCAL: 8192, CLOUD: 32768}
DEFAULT_LOCAL_CONTEXT = 16384      # the local model's own window (Ollama: its num_ctx)


def server_command() -> List[str]:
    """The serve-mcp command line. Its settings travel in the environment
    (server_env), never as arguments: Vibe echoes the command line into every
    tool result, and the repo path would tell a cloud model the repo's name."""
    return [sys.executable, "-m", "archdoc", "serve-mcp"]


def server_settings(repo: str, out: str, trust: str, config: Optional[str] = None,
                    window: Optional[int] = None) -> Dict[str, str]:
    env = {"ARCHDOC_REPO": os.path.abspath(repo), "ARCHDOC_OUT": os.path.abspath(out), "ARCHDOC_TRUST": trust,
           "ARCHDOC_WINDOW": str(window or DEFAULT_WINDOWS[trust])}
    if config:
        env["ARCHDOC_CONFIG"] = os.path.abspath(config)
    if os.environ.get("PLANTUML_JAR"):         # harnesses may start the server with a bare environment
        env["PLANTUML_JAR"] = os.path.abspath(os.environ["PLANTUML_JAR"])
    return {**env, **server_env()}


def server_env() -> Dict[str, str]:
    """PYTHONPATH for a source checkout that is not installed, so `-m archdoc` resolves."""
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "site-packages" in pkg_parent or "dist-packages" in pkg_parent:
        return {}
    return {"PYTHONPATH": pkg_parent}


def _limits(context: int) -> Dict[str, int]:
    """The local model's window, and room to answer: a thinking model spends
    thousands of tokens before the answer, and opencode stops a run whose
    reply is still truncated after two continuations."""
    return {"context": context, "output": min(8192, context // 2)}


def opencode_config(env: Dict[str, str], trust: str, model: str, url: Optional[str], instructions: str,
                    context: int = DEFAULT_LOCAL_CONTEXT) -> Dict:
    """opencode 2.x configuration (`permissions` rulesets, `policies`, `agents`,
    `mcp.servers`, `providers`; the 1.x keys still load but are deprecated)."""
    agent_name = f"archdoc-{trust}"
    # last matching rule wins: everything denied, then archdoc's MCP tools allowed
    permissions = [{"action": "*", "resource": "*", "effect": "deny"},
                   {"action": "archdoc_*", "resource": "*", "effect": "allow"}]
    server: Dict = {"type": "local", "command": server_command(), "environment": env,
                    "codemode": False,         # the tools themselves, not a JavaScript runtime that can fetch
                    "timeout": {"startup": 60_000, "catalog": 60_000, "execution": MCP_TIMEOUT_S * 1000}}
    if trust == LOCAL:
        ref = {"providerID": LOCAL_PROVIDER, "model": model}
    else:
        provider_id, _, name = model.partition("/")
        if not name:
            raise ValueError("--model for an opencode cloud profile is provider/model, e.g. mistral/mistral-medium-latest")
        ref = {"providerID": provider_id, "model": name}
    cfg: Dict = {
        "$schema": "https://opencode.ai/config.json",
        "model": ref,
        "default_agent": agent_name,
        "share": "disabled",                       # sessions are never uploaded
        "update": "disable",
        "permissions": permissions,
        "mcp": {"servers": {"archdoc": server}},
        "agents": {
            agent_name: {
                "description": f"Architecture documentation through archdoc only ({trust} tier)",
                "mode": "primary", "model": ref, "system": instructions, "permissions": permissions,
                "request": {"body": {"temperature": 0.2}},
            },
            # the built-in agents get the same lock, should someone switch to them
            "build": {"permissions": permissions},
            "plan": {"permissions": permissions},
        },
    }
    if trust == LOCAL:
        cfg["providers"] = {LOCAL_PROVIDER: {
            "name": "archdoc local model", "package": "aisdk:@ai-sdk/openai-compatible",
            # Ollama's OpenAI endpoint can end a stream without one, which otherwise aborts the run
            "compatibility": {"requireFinishReason": False},
            "settings": {"baseURL": url or DEFAULT_LOCAL_URL},
            "models": {model: {"name": model, "limit": _limits(context),
                               "capabilities": {"tools": True, "input": ["text"], "output": ["text"]}}}}}
        # no other provider can be used, even if someone picks another model mid-session
        cfg["policies"] = [{"action": "provider.use", "resource": "*", "effect": "deny"},
                           {"action": "provider.use", "resource": LOCAL_PROVIDER, "effect": "allow"}]
    return cfg


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)          # a JSON string is a valid TOML basic string


def vibe_agent(env: Dict[str, str], trust: str, model: str, url: Optional[str], instructions: str,
               context: int = DEFAULT_LOCAL_CONTEXT) -> str:
    lines = [
        f"display_name = {_toml_str(f'archdoc ({trust})')}",
        f"description = {_toml_str(f'Architecture documentation through archdoc only ({trust} tier)')}",
        f"instructions = {_toml_str(instructions)}",
        'enabled_tools = ["archdoc_*"]',               # every other tool is off
        "enable_connectors = false",
        'system_prompt_id = "minimal"',                # the coding-agent prompt is ~10k tokens of no use here
    ]
    if trust == LOCAL:
        lines += [
            "enable_telemetry = false",
            f'active_model = "{LOCAL_PROVIDER}"',
            f'allowed_models = ["{LOCAL_PROVIDER}"]',  # nothing else can be picked mid-session
            "",
            "[[providers]]",
            f'name = "{LOCAL_PROVIDER}"',
            f"api_base = {_toml_str(url or DEFAULT_LOCAL_URL)}",
            'api_style = "openai"',
            "emits_finish_reason = false",             # Ollama's OpenAI endpoint does not always send one
            "",
            "[[models]]",
            f"name = {_toml_str(model)}",
            f'provider = "{LOCAL_PROVIDER}"',
            f'alias = "{LOCAL_PROVIDER}"',
            "temperature = 0.2",
            f"auto_compact_threshold = {int(context * 0.75)}",   # compact before the window overflows
        ]
    else:
        lines.append(f"active_model = {_toml_str(model)}")
    lines += ["", "[[mcp_servers]]", 'name = "archdoc"', 'transport = "stdio"',
              f"command = {json.dumps(server_command())}", f"tool_timeout_sec = {MCP_TIMEOUT_S}",
              "startup_timeout_sec = 60", "sampling_enabled = false",
              "env = {" + ", ".join(f"{k} = {_toml_str(v)}" for k, v in env.items()) + "}"]
    for t in TOOL_NAMES:                               # archdoc's tools only read the IR and write the output dir
        lines += ["", f"[tools.archdoc_{t}]", 'permission = "always"']
    return "\n".join(lines) + "\n"


OPENCODE_RUNNER = """\
#!/bin/sh
# Headless archdoc run with opencode. opencode attaches MCP tools to a session
# from its second step on, so: start a private server for this directory, open
# a session with a first message, then continue that session with the task.
# The run is over when the server's status says nothing is left and the outputs
# were written. A session that ends before that (an error, a full window, or a
# model that ends its turn without the next call, which opencode counts as done)
# is followed by a fresh one: answers are kept, so it picks up where that one stopped.
cd "$(dirname "$0")" || exit 1
TASK=${1:-"Document the system: work through every archdoc task until none are left, then write the outputs."}
SESSIONS=${SESSIONS:-10}
STATUS={status}
done_() {
    python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); sys.exit(not (s["outputs_written"] and not s["todo"]))' \
        "$STATUS" 2>/dev/null
}
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
OPENCODE_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')   # this run only
export OPENCODE_PASSWORD
opencode serve --hostname 127.0.0.1 --port "$PORT" > .opencode-serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
python3 -c 'import socket, sys, time
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1).close(); break
    except OSError:
        time.sleep(0.5)' "$PORT"
URL="http://127.0.0.1:$PORT"
# a step that never returns (a model that stops answering) counts as an error
LIMIT=$(command -v timeout >/dev/null && echo timeout)
i=1
while :; do
    $LIMIT ${LIMIT:+${WARMUP_TIMEOUT:-300}} opencode run --server "$URL" --agent {agent} \
            "Reply with the single word: ready." > /dev/null &&
        $LIMIT ${LIMIT:+${SESSION_TIMEOUT:-3600}} opencode run --server "$URL" --agent {agent} \
            --continue "$TASK"
    done_ && exit 0
    [ "$i" -ge "$SESSIONS" ] && { echo "archdoc: work left after $SESSIONS sessions (see $STATUS)" >&2; exit 1; }
    echo "archdoc: session $i stopped with work left; starting session $((i + 1)) of $SESSIONS" >&2
    i=$((i + 1))
    TASK="Continue documenting the system: work through every archdoc task still to do, then write the outputs."
done
"""


VIBE_RUNNER = """\
#!/bin/sh
# Headless archdoc run with Mistral Vibe. A small model's window fills after a
# task or two; Vibe then compacts the conversation, and a programmatic session
# often ends there. Answers are kept, so a fresh session picks up where that
# one stopped; the run is over when the server's status says nothing is left
# and the outputs were written.
cd "$(dirname "$0")" || exit 1
TASK=${1:-"Document the system: work through every archdoc task until none are left, then write the outputs."}
SESSIONS=${SESSIONS:-40}
STATUS={status}
done_() {
    python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); sys.exit(not (s["outputs_written"] and not s["todo"]))' \\
        "$STATUS" 2>/dev/null
}
LIMIT=$(command -v timeout >/dev/null && echo timeout)
i=1
while :; do
    $LIMIT ${LIMIT:+${SESSION_TIMEOUT:-3600}} vibe --agent {agent} --trust --max-turns 200 -p "$TASK"
    done_ && exit 0
    [ "$i" -ge "$SESSIONS" ] && { echo "archdoc: work left after $SESSIONS sessions (see $STATUS)" >&2; exit 1; }
    echo "archdoc: session $i stopped with work left; starting session $((i + 1)) of $SESSIONS" >&2
    i=$((i + 1))
    TASK="Continue documenting the system: work through every archdoc task still to do, then write the outputs."
done
"""


def write_profile(workdir: str, harness: str, repo: str, out: str, trust: str, model: str,
                  url: Optional[str] = None, config: Optional[str] = None,
                  window: Optional[int] = None, context: Optional[int] = None) -> List[str]:
    """Writes the harness files into `workdir`; returns their paths."""
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r} (use {', '.join(HARNESSES)})")
    if trust not in (LOCAL, CLOUD):
        raise ValueError("--trust must be local or cloud")
    workdir, repo = os.path.abspath(workdir), os.path.abspath(repo)
    if workdir == repo or workdir.startswith(repo + os.sep):
        raise ValueError("the harness directory must not be inside the target repo: "
                         "a tool permission that slips would then expose the source")
    os.makedirs(workdir, exist_ok=True)
    env = server_settings(repo, out, trust, config, window)
    with open(AGENTS_MD, encoding="utf-8") as fh:
        instructions = fh.read()
    written = []
    agents_md = os.path.join(workdir, "AGENTS.md")
    shutil.copyfile(AGENTS_MD, agents_md)          # for reading; the profiles carry it inline
    written.append(agents_md)
    if harness == "opencode":
        path = os.path.join(workdir, "opencode.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(opencode_config(env, trust, model, url, instructions, context or DEFAULT_LOCAL_CONTEXT), fh, indent=2)
            fh.write("\n")
        written.append(path)
        runner = OPENCODE_RUNNER
    else:
        path = os.path.join(workdir, ".vibe", "agents", f"archdoc-{trust}.toml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(vibe_agent(env, trust, model, url, instructions, context or DEFAULT_LOCAL_CONTEXT))
        written.append(path)
        runner = VIBE_RUNNER
    # headless runs: sessions until the server's status says the work is done
    path = os.path.join(workdir, f"run-{harness}.sh")
    status = shlex.quote(os.path.join(os.path.abspath(out), ".archdoc", "harness", "status.json"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(runner.replace("{agent}", f"archdoc-{trust}").replace("{status}", status))
    os.chmod(path, 0o755)
    written.append(path)
    return written


__all__ = ["HARNESSES", "write_profile", "opencode_config", "vibe_agent", "server_command", "server_settings"]
