"""Prompts for the planner. The model only ever sees an egress payload."""

COMMON_RULES = """\
- You see signatures and structure only, never implementation code.
- Identifiers may be opaque references such as @C3, @F12 or @L4. Copy them
  exactly as given; never invent, rename or expand them. Inside free text you
  may mention a reference (e.g. "@F12 validates the order") and it will be
  replaced with the real name.
- Do not invent behavior, protocols, security mechanisms, performance figures
  or components that the input does not show.
- Return only JSON matching the required schema."""

SEQUENCE_SYSTEM = f"""\
You are a software architect drawing a UML sequence diagram from a static
call-graph slice of a code base.

{COMMON_RULES}"""

SEQUENCE_INSTRUCTIONS = """\
The input describes one entry point:
- "entry": the handler function and, if any, the interface operation that reaches it
- "participants": classes or modules that own the functions in the slice
- "functions": each function's owner, signature, and the functions it calls,
  listed in source order ("ext:" calls are third-party libraries)

Produce a sequence diagram plan:
- participants: "client" (the external caller) first, then the participants
  involved, using their "ref". Pick a kind: actor for client, boundary for the
  entry handler's owner, control for coordinating classes, entity for data
  classes, database / queue only if the signatures clearly indicate one.
- messages, in order: start with client -> the handler's owner. Follow the call
  order. A message for a call has "call" set to the called function's ref and a
  short label naming the action, and reply=false. For a third-party call, send
  the message to the caller itself (a self-call) with a label naming the library.
  Use reply=true only for a few returns worth showing (e.g. the final response
  to client); a reply has call "".
  Keep it readable: at most about 15 messages; skip trivial helpers.
- groups: only when names clearly suggest a condition or loop (e.g. a raised
  error -> alt "not found"). "first" and "last" are 0-based message indexes,
  inclusive. Use [] if unsure.
- title: a short title for the flow.

INPUT:
"""

NARRATIVE_SYSTEM = f"""\
You are a technical writer filling in an Interface Control Document from an
interface's structure.

{COMMON_RULES}"""

NARRATIVE_INSTRUCTIONS = """\
The input describes one interface: its operations (with parameters), the
handler functions' signatures and docstrings, and the data types exchanged.

Write:
- purpose: 1-3 sentences on what the interface is for and who would use it.
- operations: for every operation "ref", a one-sentence summary of what it does,
  and a short description for each parameter you can describe from its name and
  type (leave out parameters you cannot describe).

Describe what the names, types and docstrings indicate. Where something is
uncertain, say what the operation "appears to" do rather than guessing details.
Describe only the operations listed in the input: do not claim the interface
offers anything else (no extra endpoints, commands or CRUD actions). Do not
quote internal identifiers such as "if:http:..." or module paths in the prose;
refer to the interface by what it does.

INPUT:
"""
