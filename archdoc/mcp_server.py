"""
MCP server for coding-agent harnesses (DESIGN.md 13): `archdoc serve-mcp`.

stdio transport, newline-delimited JSON-RPC 2.0, standard library only. The
repo, output directory, config and trust tier are fixed at launch and never
taken from tool arguments. Every tool answers from archdoc.harness, which
shows the agent only what the pipeline would send a model of that tier.

stdout carries the protocol and nothing else: anything printed while a tool
runs goes to stderr (or the session's run log).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, IO, Optional

from . import __version__
from .harness import KINDS, HarnessSession

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = """\
archdoc writes architecture documentation from code. You never read the code:
each task gives you an input extracted from it and the JSON answer to write.
1. overview: what the system is and how many tasks there are.
2. list_tasks, then for each: get_task, write the answer from its input
   only, submit_task. A rejected answer comes back with reasons: fix it and
   submit again to the same id.
3. list_tasks again: summaries of packages and of the system appear once the
   summaries they build on are accepted. Repeat until nothing is left to do.
4. write_outputs: renders the diagrams and documents."""

_ID = {"type": "string", "description": "Task id from list_tasks"}

TOOLS = [
    {"name": "overview",
     "description": "The system map (modules and the interfaces they provide), its size, the trust tier "
                    "and how many tasks of each kind are open, rejected or accepted. Start here.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_tasks",
     "description": "Requests waiting for your answer. status: todo (open or rejected; default), open, "
                    "rejected, accepted or all.",
     "inputSchema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": sorted(KINDS), "description": "Only tasks of this kind"},
         "status": {"type": "string", "enum": ["todo", "open", "rejected", "accepted", "all"]},
         "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Default 25"}},
         "additionalProperties": False}},
    {"name": "get_task",
     "description": "One task: its instructions, its input (the only facts you may use) and the JSON "
                    "schema the answer must match.",
     "inputSchema": {"type": "object", "properties": {"id": _ID}, "required": ["id"],
                     "additionalProperties": False}},
    {"name": "submit_task",
     "description": "Submit the answer to a task: a JSON object matching its answer_schema. It is "
                    "checked against the code; the result is accepted, or rejected with the reasons.",
     "inputSchema": {"type": "object", "properties": {
         "id": _ID,
         # models often send the object as a JSON string; with "type": "object"
         # alone, Ollama's OpenAI endpoint drops such a call entirely
         "answer": {"anyOf": [{"type": "object"}, {"type": "string"}],
                    "description": "The answer: a JSON object matching the task's answer_schema"}},
         "required": ["id", "answer"], "additionalProperties": False}},
    {"name": "write_outputs",
     "description": "Render every diagram and document with the answers accepted so far. Run it when "
                    "no tasks are left (it can be run any time).",
     "inputSchema": {"type": "object", "properties": {
         "export": {"type": "string", "description": "Also convert the documents: comma-separated docx, html, pdf"}},
         "additionalProperties": False}},
]


class Server:
    def __init__(self, session: HarnessSession):
        self.session = session
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "overview": lambda a: session.overview(),
            "list_tasks": lambda a: session.list_tasks(a.get("kind"), a.get("status") or "todo",
                                                       int(a.get("limit") or 25)),
            "get_task": lambda a: session.get_task(str(a["id"])),
            "submit_task": lambda a: session.submit(str(a["id"]), a["answer"]),
            "write_outputs": lambda a: session.write_outputs(a.get("export")),
        }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.handlers.get(name)
        if handler is None:
            return _text({"error": f"unknown tool {name!r}"}, error=True)
        try:
            return _text(handler(arguments or {}))
        except KeyError as exc:
            return _text({"error": exc.args[0] if exc.args else "missing argument"}, error=True)
        except Exception as exc:                # noqa: BLE001 - reported to the agent, details to stderr
            traceback.print_exc(file=sys.stderr)
            # the type only: a message could carry real names under the cloud tier
            msg = str(exc) if not self.session.aliased else type(exc).__name__
            return _text({"error": f"archdoc failed: {msg}"}, error=True)

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method is None:                      # a response to something we never send
            return None
        if mid is None:                         # notification (initialized, cancelled, ...)
            return None
        if method == "initialize":
            asked = params.get("protocolVersion")
            return _ok(mid, {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                             "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": "archdoc", "version": __version__},
                             "instructions": INSTRUCTIONS})
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            return _ok(mid, self.call_tool(params.get("name", ""), params.get("arguments") or {}))
        if method in ("resources/list", "prompts/list", "resources/templates/list"):
            key = {"resources/list": "resources", "prompts/list": "prompts",
                   "resources/templates/list": "resourceTemplates"}[method]
            return _ok(mid, {key: []})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}

    def serve(self, inp: IO[str], out: IO[str]) -> None:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                batch = msg if isinstance(msg, list) else [msg]
                replies = [r for r in (self.handle(m) for m in batch if isinstance(m, dict)) if r]
                reply = replies if isinstance(msg, list) else (replies[0] if replies else None)
            if reply:
                try:
                    out.write(json.dumps(reply, ensure_ascii=False) + "\n")
                    out.flush()
                except BrokenPipeError:             # the harness has gone: nothing left to answer
                    return


def _ok(mid: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _text(obj: Any, error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=1, ensure_ascii=False)}],
            "isError": error}


def serve_stdio(session: HarnessSession) -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr                     # stray prints must never corrupt the protocol stream
    try:
        Server(session).serve(sys.stdin, protocol)
    finally:
        sys.stdout = protocol
        try:
            protocol.flush()
        except BrokenPipeError:
            # stop the interpreter flushing the closed pipe again on exit
            os.dup2(os.open(os.devnull, os.O_WRONLY), protocol.fileno())


__all__ = ["Server", "TOOLS", "serve_stdio"]
