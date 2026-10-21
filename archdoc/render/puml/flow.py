"""
Cross-service sequence diagram from a traced flow (archdoc.flows).

Participants are services (or top-level packages without a deployment file)
and the broker; messages are invocations, publishes and deliveries, each one
a call or pika operation found in the code. Fanout to several consumers is a
`par` block; a publish nobody consumes ends in a note on the broker.
"""

from __future__ import annotations

from typing import List, Optional

from ...flows import Flow
from ...ir import Deployment
from .common import Aliases, Diagram, label

MAX_FIELDS = 4
LEGEND = ("Traced from the code: calls in source order, pika publish/consume. "
          "Branches, retries and loops are not shown.")


def _text(s: str) -> str:
    return s.replace("\r", " ").replace("\n", " ")


def _fields(fields: List[str]) -> str:
    if not fields:
        return ""
    shown = ", ".join(fields[:MAX_FIELDS]) + (", ..." if len(fields) > MAX_FIELDS else "")
    return f"\\n{{{shown}}}"


def render_flow_diagram(flow: Flow, name: str, dep: Optional[Deployment] = None,
                        title: Optional[str] = None, client: str = "User") -> str:
    aliases = Aliases()
    d = Diagram(name, title, settings=["hide footbox", "skinparam sequenceMessageAlign center"])
    images = {s.name: s.image for s in dep.services} if dep else {}
    d.add(f"actor {label(client)} as {aliases('@client', 'A')}")
    for p in flow.participants:
        d.add(f"participant {label(p)} as {aliases('svc:' + p, 'S')}")
    if flow.broker:
        img = images.get(flow.broker)
        shown = flow.broker + (f" ({img})" if img else "")
        d.add(f"queue {label(shown)} as {aliases('@broker', 'B')}")

    def who(p: Optional[str]) -> str:
        if p in ("@client", "@broker"):
            return aliases(p)
        return aliases("svc:" + (p or "?"), "S")

    d.add("")
    for st in flow.steps:
        if st.kind == "invoke":
            d.add(f"{who(st.src)} -> {who(st.dst)} : {_text(st.text)}")
        elif st.kind == "publish":
            d.add(f"{who(st.src)} -> {who(st.dst)} : {_text(st.text)}{_fields(st.fields)}")
        elif st.kind == "deliver":
            d.add(f"{who(st.src)} -> {who(st.dst)} : {_text(st.text)}")
        elif st.kind == "unconsumed":
            d.add(f"note over {who('@broker')} #FFDDDD : {_text(st.text)}")
        elif st.kind == "par":
            d.add(f"par {_text(st.text)}")
        elif st.kind == "else":
            d.add("else")
        elif st.kind == "end":
            d.add("end")
        elif st.kind == "truncated":
            d.add(f"note across : {_text(st.text)}")
    d.add("")
    d.add(f"legend bottom left\n{LEGEND}\nendlegend")
    return d.text()
