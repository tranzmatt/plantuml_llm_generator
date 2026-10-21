"""
Prompts for the planner. The model only ever sees an egress payload.

The alias rules are only included when the payload is pseudonymized: shown to a
model reading real names, their example references get copied verbatim.
"""

from typing import Tuple

_BASE_RULES = """\
- You see signatures and structure. Some functions may also show a
  control-flow outline (nesting, calls, raises, no expressions) or their full
  source; use those to get order, conditions and errors right.
- Copy refs exactly as they appear in the input; never invent, rename or expand them.
- Do not invent behavior, protocols, security mechanisms, performance figures
  or components that the input does not show.
- Return only JSON matching the required schema."""

_ALIAS_RULES = """
- Identifiers are opaque references such as @C3, @F12 or @L4. Write free text
  in terms of them (e.g. "@F12 loads @C3 from @C7"): each reference is replaced
  with the real name afterwards. Do not guess what the system is about or
  invent domain words; text that mentions no reference cannot be mapped back
  and is discarded."""


def common_rules(aliased: bool) -> str:
    return _BASE_RULES + (_ALIAS_RULES if aliased else "")


_SEQUENCE_ROLE = """\
You are a software architect drawing a UML sequence diagram from a static
call-graph slice of a code base.
"""

_SEQUENCE_INSTRUCTIONS = """\
The input describes one entry point:
- "system_map" (if present): the whole system's main modules, for orientation only
- "entry": the handler function and, if any, the interface operation that reaches it
- "participants": classes or modules that own the functions in the slice
- "functions": each function as a Python-style stub: signature, docstring, and
  after "#" its owner and the functions it calls in source order ("ext:" calls
  are third-party libraries); some include an outline or their source

Produce a sequence diagram plan:
- participants: "client" (the external caller) first, then the participants
  involved, using their "ref". Pick a kind: actor for client, boundary for the
  entry handler's owner, control for coordinating classes, entity for data
  classes, database / queue only if the signatures clearly indicate one.
- messages, in order: start with client -> the handler's owner. Follow the call
  order. A message for a call has "call" set to the called function's ref and a
  short label naming the action, and reply=false. For a third-party call, send
  the message to the caller itself (a self-call) with a label naming the library.
  {label_hint}
  Use reply=true only for a few returns worth showing (e.g. the final response
  to client); a reply has call "".
  Keep it readable: at most about 15 messages; skip trivial helpers.
- groups: only when names clearly suggest a condition or loop (e.g. a raised
  error -> alt "not found"). "first" and "last" are 0-based message indexes,
  inclusive. Use [] if unsure.
- title: a short title for the flow.

INPUT:
"""

_NARRATIVE_ROLE = """\
You are a technical writer filling in an Interface Control Document from an
interface's structure.
"""

_NARRATIVE_INSTRUCTIONS = """\
The input describes one interface: its operations (with parameters), the
handler functions' signatures and docstrings (some with an outline or their
source), and the data types exchanged. "system_map", if present, outlines the
whole system for orientation only.

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


def sequence_prompts(aliased: bool) -> Tuple[str, str]:
    hint = ('Build labels from the refs involved (e.g. "create @C2, then @F3").' if aliased
            else "Name labels after the functions and classes involved.")
    return _SEQUENCE_ROLE + "\n" + common_rules(aliased), _SEQUENCE_INSTRUCTIONS.replace("{label_hint}", hint)


def narrative_prompts(aliased: bool) -> Tuple[str, str]:
    return _NARRATIVE_ROLE + "\n" + common_rules(aliased), _NARRATIVE_INSTRUCTIONS
