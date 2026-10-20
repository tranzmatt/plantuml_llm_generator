# Design: Interface-Only Architecture Documentation Suite (v2)

Status: **Draft r2** · Branch: `feature/interface-pipeline-redesign`
Working name: `archdoc` (placeholder, see Open Questions)

## 1. Goals

1. **One suite, two products.** From a single extraction of a code repository,
   produce (a) PlantUML architecture diagrams and (b) Interface Control Documents
   (ICDs). Both share the same names, the same facts and the same trust controls,
   so they never disagree with each other.
2. **Minimize tokens.** The LLM never needs to know *how* a function is implemented,
   only that it exists, what it accepts and returns, and what it talks to.
3. **Protect IP, scaled to where the model runs.** Implementation code never
   leaves the machine. When the model is local or on the LAN, names and interface
   details flow freely. When the model is in the cloud, identifiers are
   pseudonymized on the way out and restored on the way back in.
4. **Harness-ready.** The same core runs as a batch CLI or as an MCP server that
   coding-agent harnesses (opencode, Mistral Vibe, and with care, Codex or Claude)
   drive, without the agent being able to read source files directly.

### Non-goals

- Perfect behavioral diagrams. Without function bodies, activity and sequence
  diagrams are approximations built from signatures and the static call graph.
- Writing ICD *requirements*. The suite documents interfaces as built; it does not
  invent requirements, rationale or performance figures. Gaps are marked **TBD**.

## 2. Current state (problems to fix)

| # | Problem | Where |
|---|---------|-------|
| P1 | Pass 1 sends the first 6,000 chars of every file (bodies included) to the LLM to extract structure that the AST provides exactly and for free. | `repo_to_diagrams_chunked_vllm.py` `pass1_extract_batch` |
| P2 | Pass 3 sends up to 24k chars of raw source in *each* of the 8 diagram prompts, plus retries. | `select_chunks`, `pass3_generate_all` |
| P3 | The single-pass Ollama script sends the entire repo in one prompt. | `repo_to_diagrams_ollama.py` `walk_repo_collect_code` |
| P4 | The AST extractor (`util/extract_repo_facts.py`) and reducer (`util/build_architecture_summary.py`) exist but are not wired into either pipeline. | `util/` |
| P5 | The LLM writes PlantUML text freehand, so ~1,000 lines of regex repair functions exist to fix its syntax. Validation is regex-based and never runs PlantUML itself. | `repair_*`, `validate_puml` |
| P6 | Even AST facts can leak to an untrusted model: default values (secrets), string literals (internal URLs, bucket and queue names), docstrings, and descriptive identifiers. | `extract_repo_facts.py` `finalize` |
| P7 | Four near-duplicate entry scripts (vLLM/Ollama × chunked/unchunked) with copied prompts and file walking. | top level |
| P8 | Python only. ICDs exist only as ad-hoc harness output, with no shared facts or consistency with the diagrams. | — |

## 3. Trust model

### 3.1 Trust tiers

Every LLM backend is assigned exactly one tier. The tier picks the default
egress policy (Section 6).

| Tier | Meaning | Examples |
|------|---------|----------|
| `local` | Inference runs on hardware the user controls: this machine or a host on the local network. | vLLM on the DGX, Ollama with a local model, llama.cpp, LM Studio |
| `cloud` | Inference runs on infrastructure the user does not control. | OpenAI/Codex, Anthropic/Claude, Mistral API, **Ollama cloud models** |

**Bodies never leave in either tier.** In `local` that is purely a token-budget
choice. In `cloud` it is also an IP control.

### 3.2 Classifying a backend

Backends are declared in config with an explicit tier:

```toml
[backends.dgx]
url   = "http://10.0.0.5:8000/v1"
model = "mistral-medium"
trust = "local"

[backends.claude]
url   = "https://api.anthropic.com"
model = "claude-sonnet-5-5"
trust = "cloud"
```

Safety checks that run before any request:

- **Undeclared means cloud.** A backend without `trust` is treated as `cloud`.
- **Declared local is verified.** If a `local` backend's host resolves to a
  non-private address (not loopback, RFC 1918, link-local, or an explicit
  `[trust] local_hosts` allowlist), the run aborts.
- **Model name overrides URL.** Ollama can serve *cloud* models through the local
  daemon (model tags ending in `-cloud`). The URL is `localhost`, but inference is
  remote. Any model matching `[trust] cloud_model_patterns`
  (default: `["*-cloud", "*:cloud"]`) is forced to `cloud` regardless of URL.

### 3.3 What crosses the boundary

| Category | `local` | `cloud` |
|----------|---------|---------|
| Function and method bodies, comments | never | never |
| Secrets (scanner hits) | never | never (run aborts) |
| Arbitrary string literals | never | never |
| Module, class, function, param names | real | **pseudonymized** |
| Type annotations | real | pseudonymized (internal types) |
| Docstrings | first line (opt-in: full) | none (opt-in: first line, pseudonymized) |
| Interface literals: route paths, CLI flags, topic and queue names, field constraints | real | pseudonymized |
| Default values | simple literals | presence only |
| Control-flow skeletons | allowed | off (opt-in) |
| Third-party and stdlib names | real | real |

**Residual leakage in `cloud` (accepted, documented):** the *shape* of the system:
counts, graph topology, which third-party libraries are used, HTTP methods.
Pseudonymization hides vocabulary, not architecture.

## 4. Architecture overview

```
                    LOCAL (always trusted)                       │  BACKEND (tier-dependent)
                                                                 │
 repo ──► extract ──► IR ──┬──► interface inventory              │
         (AST, .proto,     │          │                          │
          OpenAPI, ...)    │          ▼                          │
                           │   policy(tier) ──► sanitize ──► alias* ──► payload ──► LLM planner
                           │                                     │                   (JSON out)
                           │                                     │                       │
                           │                  restore* ◄── validate refs ◄───────────────┘
                           │                     │               │
                           ▼                     ▼               │
                deterministic facts ──────►  renderers           │
                (class, component,          ├─► PlantUML ──► plantuml -syntax
                 ICD tables)                └─► ICD (Markdown/AsciiDoc ──► docx/pdf)
                                                                 │
                     * alias/restore only when tier = cloud      │
```

Key properties:

- **Code, not the LLM, produces facts.** Every class, field, route, flag and type
  in a diagram or ICD comes from the IR. The LLM only adds *judgment*: which
  flow to show, how to label it, what an interface is for.
- **The LLM outputs JSON, never PlantUML or final documents.** Renderers merge the
  LLM's JSON with local facts. This removes the regex repair layer and means
  sensitive literals in `cloud` mode are only ever written locally.
- **One egress function and one ingress function.** Every byte sent goes through
  `egress.build_payload()`, and every byte received goes through
  `ingress.accept()`. Both are logged and dry-runnable.

## 5. Intermediate representation (IR)

`archdoc/ir.py`, plain dataclasses, JSON-serializable. The IR is always the
*full, real* local view. Sanitizing and aliasing produce derived views of it.

### 5.1 Code structure

```python
@dataclass
class Param:
    name: str
    annotation: str | None
    default: str | None         # literal text if simple, "<expr>" otherwise; policy decides egress
    kind: str                   # positional | keyword | var_positional | var_keyword

@dataclass
class Function:
    id: str                     # stable: "pkg.mod:Class.method"
    name: str
    params: list[Param]
    returns: str | None
    is_async: bool
    decorators: list[Decorator] # dotted name + literal args (route paths, etc.)
    visibility: str             # public | protected | private
    docstring: str | None
    calls: list[str]            # resolved ids, or "ext:requests.get"
    creates: list[str]          # class ids instantiated (drives "creates" edges)
    raises: list[str]

@dataclass
class Class:
    id: str
    name: str
    bases: list[str]
    kind: str                   # class | dataclass | enum | protocol | abc | pydantic | orm
    fields: list[Field]         # name, annotation, required, constraints, docstring
    methods: list[Function]
    enum_members: list[str]
    docstring: str | None

@dataclass
class Module:
    id: str
    path: str
    imports: list[str]
    classes: list[Class]
    functions: list[Function]
    docstring: str | None
    has_main: bool              # contains `if __name__ == "__main__":`
    parse_error: str | None     # reported, never fatal
```

### 5.2 Interfaces (shared by diagrams and ICDs)

```python
@dataclass
class Interface:
    id: str                     # "if:http:invoices"
    name: str
    kind: str                   # http | cli | message | rpc | library | file | env | db
    direction: str              # provided | required
    owner: str                  # component id that provides or consumes it
    protocol: str | None        # "HTTP/1.1 JSON", "AMQP", "gRPC", "argv"
    address: str | None         # base path, topic, exchange, service name
    operations: list[Operation]
    source_refs: list[str]      # IR ids it was derived from

@dataclass
class Operation:
    id: str
    name: str                   # "POST /invoices/{id}", "serve", "invoice.created"
    params: list[Field]         # path/query params, CLI options (constraints["in"] / ["flags"])
    request: TypeRef | None
    response: TypeRef | None
    errors: list[str]           # status codes, raised exceptions, exit codes
    handler: str | None         # Function id
    meta: dict[str, str]        # e.g. {"status_code": "201"}

@dataclass
class DataType:
    id: str
    name: str
    fields: list[Field]         # name, type, required, default, constraints, units?, description
    source: str                 # pydantic | dataclass | proto | openapi | jsonschema | orm
```

### 5.3 Extractors

Each extractor implements `extract(root) -> RepoIR` and adds to the interface inventory.

| Source | Provides | Phase |
|--------|----------|-------|
| Python AST | structure, calls, raises, docstrings | 0 |
| Python framework detectors | FastAPI/Flask/Django routes, argparse/Click/Typer CLIs, Celery tasks, Kafka/pika/kombu topics, SQLAlchemy models, pydantic/dataclass types, `os.getenv` names, outbound `requests`/`httpx`/`boto3` clients (*required* interfaces) | 0–2 |
| `.proto` files | gRPC services, messages | 6 |
| OpenAPI / AsyncAPI / JSON Schema files | HTTP and message interfaces, data types | 6 |
| tree-sitter (Java, TS, Go, C/C++) | structure | 6 |

Notes:

- **Call resolution:** `self.m()` → enclosing class; `x: Foo` / `Foo().m()` →
  `Foo.m`; `mod.f()` → via the import table. Unresolvable calls are dropped, not
  guessed.
- **Decorator literals** (route paths, CLI flag names) are captured because they
  *are* interface definitions. They count as interface literals in the policy.
- Bodies are read locally for calls, raises, field constraints and interface
  detection, but are never copied into the IR.

## 6. Policy

Configuration is read from up to three TOML files, each overriding the last:

1. user config `~/.config/archdoc/config.toml`: backends and trust settings
2. repo config `<repo>/.archdoc.toml`: policy and paths
3. `--config path`

**Each tier has a built-in default profile.** Files only state overrides, and
unknown keys or values are errors (a typo must not silently loosen policy).

| Key | Values | `local` default | `cloud` default |
|-----|--------|-----------------|-----------------|
| `level` | `none` · `interfaces` · `callgraph` · `control_flow` | `control_flow` | `callgraph` |
| `pseudonymize` | `off` · `internal` · `all` | `off` | `internal` |
| `docstrings` | `full` · `first_line` · `none` | `first_line` | `none` |
| `interface_literals` | `real` · `alias` · `none` | `real` | `alias` |
| `defaults` | `simple_literals` · `presence` · `none` | `simple_literals` | `presence` |
| `members` | `all` · `public` | `public` | `public` |
| `on_secret` | `drop` · `abort` | `drop` | `abort` |

`interface_literals` covers route paths, CLI program/command/flag names, env var
names, `Literal[...]` values, decorator arguments and non-numeric field constraints.
`docstrings` defaults to `first_line` even locally: full docstrings were the
largest single cost in measurements (about 25% of a local payload on pip).

```toml
# ~/.config/archdoc/config.toml
[trust]
local_hosts = ["dgx.lan", "100.64.0.0/10"]   # hostnames, IPs or CIDRs (CGNAT needs listing)

[backends.dgx]
url = "http://10.0.0.5:8000/v1"
model = "mistral-medium"
trust = "local"

[backends.claude]
url = "https://api.anthropic.com"
model = "claude-sonnet-5-5"
trust = "cloud"
api_key_env = "ANTHROPIC_API_KEY"            # name of the env var; keys are never stored in config

# <repo>/.archdoc.toml
[policy.cloud]
docstrings = "first_line"

[paths]
exclude   = ["tests/*", "migrations/*"]
sensitive = ["src/pricing/*"]   # field-wise strictest of tier and cloud policy, even locally

[secrets]
scan = true
```

A symbol in a sensitive module is aliased *everywhere it is referenced*, including
annotations, docstrings and decorator arguments in non-sensitive modules.

### 6.1 Egress levels

Independent of tier, the level controls *how much structure* is sent:

| Level | Sends | Enables |
|-------|-------|---------|
| `none` | nothing (no LLM) | class, component, ICD tables (all deterministic) |
| `interfaces` | IR minus call edges | + deployment, object, state, use-case, ICD narrative |
| `callgraph` | + resolved internal call edges | + sequence, ICD sequencing |
| `control_flow` | + per-function nesting of `if/for/while/try/return` with call ids, no expressions or literals | + activity (higher fidelity) |

### 6.2 Secret scanning

A backstop, not the primary control. High-entropy tokens and known patterns
(AWS keys, `sk-` keys, JWTs, private-key headers) found in anything about to egress
are dropped (`local`) or abort the run (`cloud`).

## 7. Pseudonymization (cloud tier)

### 7.1 Round trip

Aliasing happens **only around the LLM step**:

1. `egress`: the sanitized view is rewritten with aliases. The alias map stays local.
2. The LLM plans and writes using aliases only.
3. `ingress`: every alias the LLM references is checked against the map. Unknown
   references are rejected (this is the hallucination check). Known ones are
   restored to real names.
4. Renderers work only with real names and local facts.

Final diagrams and ICDs always contain real names. Only the LLM's view, and in
harness mode the agent's transcript, contains aliases.

### 7.2 Alias format

Aliases are sigil-prefixed so they can be restored safely inside free text such as
ICD prose, not just in JSON reference fields:

| Kind | Alias | Example |
|------|-------|---------|
| Module | `@M<n>` | `billing.invoices` → `@M3` |
| Class / type | `@C<n>` | `InvoiceReconciler` → `@C17` |
| Function / method | `@F<n>` | `reconcile_ledger` → `@F42` |
| Field / param | `@p<n>` | `ledger_id` → `@p8` |
| Interface / operation | `@I<n>` / `@O<n>` | ref `@O5`, displayed as `POST @L9` (the path is a literal) |
| Interface literal | `@L<n>` | `invoice.created` (topic) → `@L2` |

Restoration replaces only exact, known alias tokens. Anything that looks like an
alias but is not in the map is flagged.

- `internal` aliases everything repo-defined: modules, classes, functions,
  parameter and field names, enum members, module-level variables (type aliases)
  and internal dotted paths in annotations. Stdlib, builtin and third-party names
  stay real, because they are public and help the model infer roles.
- `all` additionally aliases third-party symbols and unknown names in annotations
  (most private, worst output).
- Docstring aliasing is best effort: class names and distinctive function names are
  rewritten, but arbitrary prose is not understood. That is why cloud docstrings
  default to `none`.
- Numbering is sequential, never a hash of the name. A hash of a guessable name
  could be confirmed by the receiver.
- The map is stable across runs, stored at `<out>/.archdoc/alias_map.json`, and
  gitignored.

### 7.3 Role hints

Names carry meaning, and aliasing removes it. An optional local `roles.toml`
(`"billing.invoices:InvoiceReconciler" = "reconciliation service"`) lets the user
supply a neutral description that is sent instead of the real name.

## 8. Diagrams

| Diagram | Producer | Input |
|---------|----------|-------|
| class | deterministic | classes, bases, fields, methods; composition from field types |
| component | deterministic | module import graph by package; **provided/required interfaces** from the inventory as lollipop/socket notation |
| deployment | LLM planner | entrypoints, IO categories, env var names, server/worker hints |
| object | LLM planner | data types with fields, constructor signatures |
| state | LLM planner | enums, `status`/`state` fields, transition-named methods |
| use-case | LLM planner | provided interfaces (routes, CLI commands, tasks), required interfaces as actors |
| sequence | LLM planner | call graph from a chosen entrypoint or operation, depth-limited |
| activity | LLM planner | call graph, plus control-flow skeletons if level allows |

Planner output is validated JSON (Section 10), rendered by
`archdoc/render/puml/<type>.py` and checked with `plantuml -syntax` when PlantUML is
available. The existing `repair_*` functions and the PlantUML-syntax FAISS RAG are
not used on the v2 path.

## 9. Interface Control Documents

### 9.1 What is deterministic vs. LLM-written

| ICD content | Source |
|-------------|--------|
| Interface inventory (id, kind, direction, owner, protocol, address) | deterministic |
| Operation tables (method + path, flags, topics; request/response types; errors) | deterministic |
| Data element dictionary (field, type, required, default, constraints) | deterministic |
| Interface diagram (component view filtered to one interface) | deterministic |
| Sequencing / protocol behavior | sequence diagram from the planner |
| Purpose, operation semantics, field descriptions where no docstring exists | LLM narrative |
| Unknowns (units, rates, latency, security, versioning) | **TBD** markers, listed in a gaps appendix |

The LLM returns narrative as JSON keyed by ids
(`{"@O5": {"summary": "...", "notes": "..."}}`). Renderers merge it into tables the
LLM never wrote. An LLM cannot add or remove a row, field or operation. In `cloud`
mode, real route paths, topic names and constraints are filled in locally and never
sent.

### 9.2 Output

- **Formats:** Markdown (default) and AsciiDoc, with PlantUML embedded as source or
  as rendered SVG. docx and PDF via pandoc (optional dependency).
- **Templates:** Jinja2, user-overridable (`--icd-template path`). Ship two:
  - `lightweight`: one document per interface or per component pair.
  - `idd`: section layout modeled on the MIL-STD-498 Interface Design Description
    (DI-IPSC-81436). The exact section mapping is to be confirmed against the DID
    during Phase 4.
- **Granularity:** `--icd-scope` `interface | component | system`.

### 9.3 Drift check

`archdoc check` compares the current interface inventory with the snapshot saved
when the ICD was last generated (`<out>/.archdoc/icd_snapshot.json`) and reports
added/removed operations, field type changes and removed required fields, flagging
breaking changes. It needs no LLM, so it can run in CI.

## 10. Planner (LLM step)

- Each LLM-produced artifact has a JSON schema in `archdoc/plan/schemas/` (eight
  diagram plans plus `icd_narrative`).
- Schema enforcement, in order of preference: backend structured output
  (vLLM / OpenAI `json_schema`, Ollama `format`), then parse, `jsonschema`
  validation and one repair round fed with the validation errors.
- Payloads are small enough that context management reduces to one check: if a
  payload exceeds budget, split by package and plan per package.

Example, sequence plan:

```json
{
  "title": "string",
  "participants": [{"ref": "@C17", "kind": "participant|actor|database|queue", "label": "string?"}],
  "messages": [{"from": "@C17", "to": "@C9", "call": "@F42", "label": "string", "async": false}],
  "groups":   [{"kind": "alt|opt|loop", "label": "string", "message_range": [0, 3]}]
}
```

(`ref` values are real ids in `local` tier and aliases in `cloud` tier. The schema is
the same for both.)

## 11. Harness integration

### 11.1 Core idea

A coding agent has its own `read`, `grep`, `bash`, `write` and web tools, and will
happily open source files, bypassing every control above. The integration has three
parts:

1. **An MCP server** that is the agent's only window into the repo, and its only
   way to write outputs.
2. **A restricted agent profile** with file, shell, edit and web tools disabled.
3. **A trust tier fixed at server launch** to match the profile's model.

### 11.2 Trust in harness mode

The MCP server cannot see which model the harness is using, so the tier is a launch
argument: `archdoc serve-mcp --repo ... --out ... --trust local|cloud`.
**If omitted, `cloud` is assumed.**

Ship paired profiles, each launching the server with the matching tier:

| Profile | Model | Server launched with |
|---------|-------|----------------------|
| `archdoc-local` | local provider only | `--trust local` |
| `archdoc-cloud` | any provider | `--trust cloud` |

**Pitfall:** harnesses let you switch models mid-session. If you switch an
`archdoc-local` session to a cloud model, real names start flowing to the cloud.
Mitigations: in the local profile, restrict the harness's enabled providers to local
ones (both harnesses have provider enable/disable settings; confirm keys in Phase 5),
and document the rule "switch profile, not model".

### 11.3 MCP tools

Repo root, policy, output directory and tier are fixed at launch, never passed as
tool arguments.

| Tool | Returns |
|------|---------|
| `repo_overview()` | package tree, counts, entrypoints |
| `list_interfaces(kind?, direction?)` | interface inventory |
| `get_interfaces(module_id)` | sanitized IR for one module |
| `get_interface(interface_id)` | operations + data types for one interface |
| `get_call_graph(entry_id, depth=3)` | call edges (level ≥ `callgraph`) |
| `get_control_flow(function_id)` | skeleton (level = `control_flow` only) |
| `render_deterministic(kind)` | writes class/component diagrams and ICD tables; returns status |
| `submit_diagram_plan(type, plan_json)` | validates, restores, renders, syntax-checks, writes; returns errors or OK |
| `submit_icd_narrative(scope_id, narrative_json)` | validates, restores, merges, writes; returns errors or OK |
| `check_drift()` | drift report |

All responses pass through `egress`, and all submissions through `ingress`.
In `cloud` mode the agent never sees real names: `submit_*` tools return validation
results only, not the rendered files.

### 11.4 Harness configuration sketches

> Sketches only. Config keys for both harnesses change between releases and must be
> checked against current docs in Phase 5.

**opencode** (`opencode.json`, local profile):

```jsonc
{
  "provider": { /* OpenAI-compatible entry for local vLLM / Ollama */ },
  "mcp": {
    "archdoc": {
      "type": "local",
      "command": ["archdoc", "serve-mcp", "--repo", "/path/to/target",
                  "--out", "docs_out", "--trust", "local"]
    }
  },
  "agent": {
    "archdoc-local": {
      "model": "local/<model>",
      "prompt": "{file:./harness/AGENTS.md}",
      "tools": { "read": false, "grep": false, "glob": false, "list": false,
                 "bash": false, "edit": false, "write": false,
                 "webfetch": false, "archdoc_*": true }
    }
  }
}
```

**Mistral Vibe** (`config.toml`): register the same stdio MCP server, define the two
profiles with built-in file, shell and web tools disabled and the `archdoc` MCP tools
enabled.

**Codex / Claude:** use the `archdoc-cloud` profile only. The provider's data-use
terms are a user decision; the suite guarantees only what the egress log shows.

Ship `harness/opencode/`, `harness/vibe/` and a shared `harness/AGENTS.md`
(workflow: overview → deterministic outputs → plan diagrams → write ICD narrative →
submit → fix errors).

**Run the harness from a directory that is not the target repo.** The MCP server is
given the path. A misconfigured tool permission then exposes an empty directory, not
the source.

## 12. Egress auditing

- `archdoc egress --dry-run --backend <name>` prints exactly what would be sent for
  each artifact under that backend's tier, with token counts, without calling a model.
- Every real run appends to `<out>/.archdoc/egress.jsonl`: timestamp, backend, tier,
  artifact, payload (or hash only, configurable), token count.
- Ingress rejections (unknown aliases, schema failures) are logged alongside.

## 13. Backends

`archdoc/backends/`:

- `openai_compat.py`, **primary**: `vllm serve`, Ollama `/v1`, llama.cpp,
  LM Studio, and OpenAI-compatible cloud endpoints.
- `anthropic.py`: optional, for direct Claude API use in batch mode.
- `vllm_offline.py`: keeps the current in-process batched `LLM.generate()` for large
  batch runs on the DGX (tier `local` by definition).

## 14. CLI and package layout

```
archdoc extract     <repo>                # write IR + interface inventory
archdoc egress      <repo> --backend X --dry-run
archdoc diagrams    <repo> --backend X [--types class,sequence]
archdoc icd         <repo> --backend X [--scope interface] [--template idd]
archdoc all         <repo> --backend X    # diagrams + ICDs, one extraction
archdoc check       <repo>                # ICD drift, no LLM
archdoc serve-mcp   --repo R --out O --trust local|cloud
```

```
archdoc/
  cli.py
  ir.py  interfaces.py
  extract/python_ast.py  extract/frameworks/*.py  extract/proto.py  extract/openapi.py
  trust.py               # tier classification + checks
  policy.py  sanitize.py  secrets_scan.py
  pseudonym.py           # alias map, restore
  egress.py  ingress.py
  plan/prompts.py  plan/schemas/*.json  plan/planner.py
  render/puml/*.py
  render/icd/{tables,narrative,export}.py  render/icd/templates/{lightweight,idd}/
  validate.py            # plantuml -syntax wrapper
  drift.py
  backends/{openai_compat,anthropic,vllm_offline}.py
  mcp_server.py
harness/  AGENTS.md  opencode/  vibe/
tests/    fixtures/canary_repo/  ...
```

Legacy scripts stay at the top level, untouched, until v2 reaches parity, then move
to `legacy/`.

## 15. Testing

- **Canary leak test (most important):** a fixture repo with unique canary strings in
  bodies, comments, docstrings, defaults, string literals, route paths, topic names
  and a fake AWS key. Assertions per tier:
  - `cloud`: no canary and no real internal identifier appears in any payload or
    MCP response.
  - `local`: no body, comment, literal or secret canary appears; names may.
- **Trust classification:** private vs. public hosts, undeclared backends,
  `-cloud` Ollama tags on localhost.
- **Pseudonymization round trip:** alias → restore is the identity, including inside
  prose; unknown aliases are rejected.
- **Extractor and framework detector tests:** golden IR and interface inventory for
  FastAPI, Flask, argparse, Click, Celery, pydantic, dataclass and SQLAlchemy
  fixtures.
- **Renderer tests:** golden `.puml` and ICD Markdown, plus a PlantUML syntax check
  (skipped if PlantUML is not installed).
- **Planner tests:** schema validation and alias rejection with canned LLM output
  (no model needed in CI).
- **Drift tests:** golden reports for added/removed/changed operations and fields.
- **Token benchmark:** v1 vs. v2 prompt tokens on this repo and one larger repo.

## 16. Phased plan

Status: Phases 0 to 3 are complete on the feature branch.

Notes from implementation:

- **Class diagrams on large repos** are partitioned automatically
  (`--max-classes`, default 60): a package over the limit is split one level
  deeper and small sibling groups are packed back together. pip (818 classes)
  gives 18 diagrams with a median of 53 classes, each class drawn in full exactly once.
  Related classes outside a diagram appear as member-less stubs.
- **Component diagrams** pick the package depth that gives the most components
  within 30. Provided interfaces are lollipops and required ones are sockets. Repos made
  of many standalone scripts produce wide diagrams (one lollipop per CLI); layout
  tuning is open.
- **ICD tables** come entirely from code. Anything code cannot answer is
  rendered as **TBD** and listed in an open-items appendix, which is also the
  worklist for the Phase 3 narrative step. Data types are included when
  referenced by an interface, transitively through fields.
- `archdoc icd` saves `.archdoc/icd_snapshot.json` for the Phase 6 drift check.
- **Phase 3 backends:** Ollama uses its native `/api/chat` (schema-constrained
  `format` and a per-request `num_ctx`, which the OpenAI-compatible endpoint can't
  set). vLLM and cloud APIs use OpenAI-compatible `response_format: json_schema`.
  Both use only the standard library.
- **Schemas constrain references:** every ref field is an enum of exactly the refs
  the payload exposed. A schema-enforcing backend cannot name anything it was not
  shown, and ingress still rejects unknown aliases for servers that ignore enums.
- **Calls are recorded in source order** (first occurrence). This adds order
  only, no new content, and it is what makes sequence diagrams follow the code.
- **Sequence plans are normalized after validation** to remove what a static
  call graph cannot support. A call is never drawn as a return arrow. `alt`/`opt`
  groups need a raising function in the slice, and `loop` needs control-flow data
  (Phase 4). Unused participants are dropped. A 12B model reliably invents all
  three otherwise.
- **Pseudonymized runs use code labels:** when the model saw aliases, call
  messages are labelled with the real function name and titles come from the
  entry point. Measured on the fixture: with names hidden, mistral-nemo invented
  an "Order Processing Flow: Place Order → Check Inventory" for what is really
  build_repo → get → get. The model still decides flow, order and kinds.
- **ICD narrative** fills only empty slots (code docstrings always win), is marked
  *(generated)*, closes the matching open items, and the banner states when
  the model only saw pseudonymized input. The prompt forbids claiming
  operations the input does not list (a 12B model otherwise "adds" CRUD).
- **Measured** (mistral-nemo:12b, RTX 3060, Ollama): this repo's 2 sequence
  diagrams + narrative for 14 interfaces took 16 requests, all valid on the first
  attempt, largest prompt 3,138 tokens, ~4 minutes. Open items dropped from 110 to 36.

| Phase | Deliverable | Exit criteria |
|-------|-------------|---------------|
| 0 | package skeleton, IR, Python AST extractor with call resolution, framework detectors for FastAPI, argparse, Click, pydantic, dataclass | golden IR + inventory tests pass |
| 1 | trust tiers, policy, sanitizer, secret scan, pseudonymizer round trip, egress/ingress, `egress --dry-run` | canary test passes for both tiers |
| 2 | deterministic class + component diagrams; **deterministic ICD tables** (inventory, operations, data dictionary) in Markdown | useful ICD and two diagrams with zero LLM calls |
| 3 | OpenAI-compatible backend, planner, sequence diagram + ICD narrative end to end | both tiers work against a local model (cloud simulated by `--trust cloud`) |
| 4 | remaining six diagrams; ICD templates (`lightweight`, `idd`), embedded diagrams, pandoc export | all outputs produced; token benchmark recorded |
| 5 | MCP server; opencode and Vibe local/cloud profiles; `AGENTS.md` | full output set from each harness with file tools disabled |
| 6 | drift check; Flask/Django/Celery/Kafka/SQLAlchemy detectors; `.proto` and OpenAPI extractors; tree-sitter languages; retire legacy scripts | — |

## 17. Open questions

1. **Name.** `archdoc` is a placeholder for the suite.
2. **ICD template.** What shape were the harness-generated ICDs? Is a
   MIL-STD-498 IDD layout wanted, a program-specific template, or something lighter?
3. **ICD granularity default:** per interface, per component pair, or one system ICD?
4. **Activity fidelity:** are control-flow skeletons acceptable for `local`
   (default on) and `cloud` (default off)?
5. **Glossary script:** fold into the ICD as an acronyms/terms appendix, or retire?
6. **RAG corpus:** with JSON plans there is no need for PlantUML syntax examples.
   Retire, or repurpose (e.g. example ICD narratives)?
7. **Target languages and interface formats after Python**, in priority order
   (`.proto`, OpenAPI, Java, C++, ...)?
8. **Very large repos:** per-package outputs automatically, or user-selected scopes?
