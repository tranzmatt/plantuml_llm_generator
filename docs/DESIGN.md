# Design: Interface-Only Architecture Documentation Suite (v2)

Status: **Draft r3** · Branch: `feature/interface-pipeline-redesign`
Working name: `archdoc` (placeholder, see Open Questions)

## 1. Goals

1. **Accurate enough to rebuild from.** The documentation must be accurate and
   complete enough that the system could be reimplemented from it alone, without
   the original code. Facts are extracted, model output is grounded and verified,
   and gaps are explicit (Section 2).
2. **One suite, two products.** From a single extraction of a code repository,
   produce (a) PlantUML architecture diagrams and (b) Interface Control Documents
   (ICDs), plus the behavior specifications a rebuild needs. All share the same
   names, the same facts and the same trust controls, so they never disagree.
3. **As much code as fits, never more than allowed.** With a local model the
   limit is the context window: each request carries as much real code as fits,
   packed by relevance (Section 12). With a cloud model, only the structure policy
   allows ever leaves the machine.
4. **Protect IP, scaled to where the model runs.** With a local or LAN model, names,
   interface details and source flow freely (the code stays on hardware you
   control). With a cloud model, implementation code never leaves the machine, and
   identifiers are pseudonymized on the way out and restored on the way back in.
5. **Harness-ready.** The same core runs as a batch CLI or as an MCP server that
   coding-agent harnesses (opencode, Mistral Vibe, and with care, Codex or Claude)
   drive, without the agent being able to read source files directly.

### Non-goals

- Precise behavior in cloud mode. Without function bodies, cloud-mode activity
  and sequence diagrams are approximations built from signatures and the static
  call graph, and are labelled as such. Locally, behavior comes from source.
- Writing ICD *requirements*. The suite documents interfaces as built; it does not
  invent requirements, rationale or performance figures. Gaps are marked **TBD**.
- Legal determinations. The suite provides traceability and an overlap guard
  (Section 2.5); whether a rebuild meets an IP requirement is not its call.

## 2. Accuracy and rebuildability

The documentation set must be accurate and complete enough that the system
could be **rebuilt from the documentation alone**: for example a clean-room
reimplementation, where the original code cannot be carried over. That
standard governs every other choice in this design, and it rests on three rules:

1. **Facts come from code.** Anything that can be extracted deterministically is
   extracted, never generated: names, signatures, types, routes, flags, fields,
   constraints, inheritance, imports, call edges, raised exceptions.
2. **Model output is grounded, verified and labelled.** A model may only describe
   what it was shown, must cite what it relied on, and is checked against the
   extracted facts. What fails the check is repaired once, then removed and
   reported. It is never silently kept.
3. **Gaps are explicit.** What cannot be determined is marked **TBD**, listed,
   and counted. Nothing is filled in by plausible guessing.

### 2.1 Provenance

Every item in every output belongs to exactly one class:

| Class | Source | Rendered as |
|-------|--------|-------------|
| **fact** | extracted from code | plain text |
| **derived** | written by a model, grounded in specific code it was shown | marked *(generated)*, with evidence in the sidecar |
| **open** | cannot be determined from the code | **TBD**, listed in open items |

Each output gets a machine-readable sidecar (`<artifact>.provenance.json`). For
every derived statement it records:
- the model and tier
- the evidence ids, and the detail level each was shown at (source / outline / stub / alias)
- the prompt version and payload hash
- the IR hash and git commit

A reviewer can therefore tell, for any sentence, whether the model read the
actual code or only a signature.

*Implemented (4b):* sidecars beside every diagram and ICD, schema
`archdoc.provenance/1`. The header has the tool version, the analysed repo's git
commit and dirty flag, the IR hash, and the creation time (omitted with
`--no-date`, so deterministic sidecars are byte-identical across runs). Items:
- deterministic diagrams: one `fact` item listing the IR ids drawn
- flow diagrams: one `fact` per step, citing its operation id
- sequence diagrams: one `derived`, `verified` item per message and group;
  the evidence is the called function and its callers in the slice, and the
  label is `fact` when it falls back to the function name
- ICDs: every purpose, summary, parameter and field slot as fact, derived or open

Each model request is recorded once: backend, model, Ollama digest, tier, policy
detail level, aliased or not, payload and prompt hashes, the real ids shown
(aliases resolved), attempts, tokens, warnings. Derived items point at their
request. *Not yet:* evidence the model cites per statement (it is derived from
verification and from what was shown), and per-item detail levels, which
need the Phase 4a packer.

### 2.2 Grounding and verification

- **Evidence is mandatory.** Every model-written statement carries an `evidence`
  list of ids it was shown, constrained by the schema like every other reference
  field. A statement with no evidence is rejected. This generalizes the rule already
  used in pseudonymized mode, where text that mentions no reference is discarded.
- **Deterministic verification against the IR**, per artifact:
  - *Sequence:* a message that is a call to `F` must go to the participant that
    owns `F`, from a participant that owns a function calling `F` (or from the
    client, for the entry call). Messages from one caller must follow its source
    call order. *(Not yet enforced; Phase 4b.)*
  - *Activity:* every action maps to a call or control-flow node of the function.
  - *State:* states must be enum members or values assigned to a state field.
    Every transition must be backed by a function that assigns the target state.
  - *ICD narrative:* referenced operations and parameters must exist (enforced).
    Claims about errors must match recorded raises and status codes.
- **The evidence level limits the claim.** Behavior descriptions written from full
  source are labelled as such. When only a stub fitted the window, the statement
  is labelled "inferred from signature" in the output, not just in the sidecar.

### 2.3 What a rebuildable documentation set contains

| To rebuild… | Artifact | Produced by | Status |
|-------------|----------|-------------|--------|
| every external interface, exactly | ICD tables (routes, CLI, env, messages) | extraction | FastAPI, argparse/Click/Typer, env: done; Flask/Django, messaging, gRPC, OpenAPI: 6 |
| data structures | data dictionary, class diagrams | extraction | done |
| module structure and dependencies | component diagram; dependency manifest (third-party packages and versions from requirements/pyproject) | extraction | done (4c) |
| interaction behavior | sequence diagrams per entry point | model + verification | diagrams: done; verification: 4b |
| algorithmic behavior | **behavior specifications** per public function: purpose, inputs, outputs, preconditions, side effects, errors, algorithm in prose | model, from source (local); trivial functions from the AST | done (4c) |
| state behavior | state diagrams | model + verification | 4d |
| expected behavior | **test inventory**: tests as acceptance criteria (what each test exercises and asserts) | extraction + model | done (4c) |
| configuration and runtime | env vars, ports, entry points, deployment diagram | extraction + model | partly done |

Behavior specifications go into a design-description document alongside the ICD,
comparable in role to a MIL-STD-498 Software Design Description (DI-IPSC-81435).
4c writes it with a lightweight template (`<out>/design/<repo>_design.md`, see
12.7); the SDD section mapping is to be confirmed when that template is built (4d).

Tests are read as evidence only: they don't appear as components. That way a
user can exclude `tests/` from the diagrams and still get the test inventory.

### 2.4 Coverage and rebuild readiness

Each run reports, per artifact and for the whole system:
- **interface coverage:** share of operations, parameters and fields with no TBD
- **behavior coverage:** share of public functions with a behavior specification,
  split by evidence level (source / outline / signature only)
- **open items**, by kind
- **known unknowns:** modules using a library that implies an interface (a broker,
  a web framework, an HTTP client) where no interface of that kind was extracted
- **cross-checks** where two statements of the architecture disagree, each a row
  in an ICD appendix: message channels with a publisher but no consumer (or the
  reverse), consumers reading fields no publisher sends (A.3); deployment files
  that leave code out of every image, require variables they do not set, set
  variables nothing reads, or write credentials literally (A.4). The tool reports
  the disagreement; which side is right is the rebuilder's decision.

**Rebuild-readiness check** (deterministic, Phase 6): regenerate interface stubs
(signatures, data types, routes, CLI definitions) from the documentation alone and
diff them against the IR. Every difference is a documentation defect: something a
rebuilder would get wrong. An optional stronger check hands the documentation to a
model to implement stubs, then compares those.

### 2.5 Describing, not copying

For a rebuild that must not carry over the original code, the documentation has to
describe behavior rather than reproduce it:
- **Overlap guard:** generated text is checked against the source it was given
  for verbatim reuse (long runs of identical tokens from function bodies), and
  flagged or rejected. Identifiers and signatures are allowed: they are the
  interface.
- Every document records what it was derived from (git commit, IR hash).

*Implemented (4b):* `archdoc.overlap`. The protected text is everything in the repo's
Python files except interface: function and class bodies, comments and string
literals, with def/class headers, decorators, docstrings and imports removed
(docstrings are already quoted as facts). It is indexed as runs of `min_words`
(default 8) lowercased words, from the whole file, not only what the model was
shown, since a model can reproduce public code from training. Every
model-written string (sequence labels, group labels, titles, ICD purposes,
summaries and parameter descriptions) is checked inside validation, so an
overlap is sent back in the repair round. The repair message says where
(`purpose repeats 14 consecutive words...`), never what, so no source text
leaves in a repair. What still overlaps is cleared (`[overlap] mode = "reject"`,
default) or kept and reported (`"flag"`); warnings and sidecars give the
length and `path:line`. `archdoc eval` counts overlaps per backend. Building the
index for pip (398 modules) takes about 6 s.

The suite provides traceability and the overlap guard. Whether a given rebuild
process meets a particular IP requirement is a legal question the tool does not
answer.

### 2.6 Reproducibility and review

- Temperature 0, with model name and digest, prompt version, payload hash, IR hash
  and git commit all recorded. Model output is cached on those keys, so a rerun
  with no code or model change makes no model calls and produces identical documents.

  *Implemented (4c):* `archdoc.cache`. Each attempt is cached on its exact
  request (model, digest, system prompt, user prompt including any repair text,
  schema, answer budget) in `<out>/.archdoc/cache/`. The payload already holds
  everything an answer depends on, so there is nothing else to invalidate.
  Answers are replayed, not verdicts: a cached answer goes through every check
  again, so a check added later applies to old answers too. The model is loaded
  lazily, before the first request not answered from the cache, so a fully
  cached rerun never loads it. `--no-cache` asks again.
- **Review overlay:** human edits and approvals live in `<out>/review/`, keyed by
  item id and the hash of the evidence they were based on. Regeneration never
  overwrites a reviewed item. If the underlying code changes, the item is
  flagged *stale* for re-review instead.

  *Implemented (4b):* `<out>/review/review.yaml`, keyed by the sidecar item ids
  (`purpose:`, `op:`, `param:`, `type:`, `field:`, `sequence:`). Every run offers
  generated and open slots as `pending` with their current text; the reviewer
  sets `approved`, `edited` (their `text`, which also fills a TBD by hand) or
  `rejected` (stays TBD). Decided entries are never rewritten and are applied
  before narrative, so the model is not asked again; an interface with nothing
  left open gets no request. Evidence hashes cover the facts an item describes
  (operation + handler signature, field, data type, a sequence slice's
  signatures and calls). On mismatch the reviewed text is kept, marked *re-review*
  in the document, listed as an open item, and `current_evidence` is written
  next to it, so re-approving means copying one value. Approved sequence plans live in
  `review/plans/` (editable), are redrawn without a model or backend, and are
  re-verified against the code, with disagreements reported, never pruned.
  Provenance gains a fourth class, `reviewed`. Not yet: reviewing the
  standard per-interface questions (authentication, rate limits), which are open
  items without a slot.
- **Accuracy evaluation:** a hand-checked reference set for the fixture repo and at
  least one real repo. Measured as precision and recall of sequence edges against
  the call graph, error claims against raises, and a reviewer score for behavior
  specifications. It is used to compare models and prompt versions before changing
  defaults.

## 3. Current state (problems to fix)

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

## 4. Trust model

### 4.1 Trust tiers

Every LLM backend is assigned exactly one tier. The tier picks the default
egress policy (Section 7).

| Tier | Meaning | Examples |
|------|---------|----------|
| `local` | Inference runs on hardware the user controls: this machine or a host on the local network. | vLLM on the DGX, Ollama with a local model, llama.cpp, LM Studio |
| `cloud` | Inference runs on infrastructure the user does not control. | OpenAI/Codex, Anthropic/Claude, Mistral API, **Ollama cloud models** |

**Source code never leaves the machine.** With a `local` model, source is sent to
the model (which runs on hardware you control) as far as the context window allows.
With a `cloud` model, function bodies are never sent.

### 4.2 Classifying a backend

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

### 4.3 What crosses the boundary

| Category | `local` | `cloud` |
|----------|---------|---------|
| Function and method bodies | as much as fits the window (level `source`) | never |
| Comments | inside sent source only (strippable to save tokens) | never |
| Secrets (scanner hits, including inside sent source) | never (redacted) | never (run aborts) |
| Arbitrary string literals | inside sent source only | never |
| Module, class, function, param names | real | **pseudonymized** |
| Type annotations | real | pseudonymized (internal types) |
| Docstrings | first line (opt-in: full) | none (opt-in: first line, pseudonymized) |
| Interface literals: route paths, CLI flags, topic and queue names, field constraints | real | pseudonymized |
| Default values | simple literals | presence only |
| Control-flow outlines | allowed | off (opt-in) |
| Third-party and stdlib names | real | real |

**Residual leakage in `cloud` (accepted, documented):** the *shape* of the system:
counts, graph topology, which third-party libraries are used, HTTP methods.
Pseudonymization hides vocabulary, not architecture.

## 5. Architecture overview

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

## 6. Intermediate representation (IR)

`archdoc/ir.py`, plain dataclasses, JSON-serializable. The IR is always the
*full, real* local view. Sanitizing and aliasing produce derived views of it.

### 6.1 Code structure

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

### 6.2 Interfaces (shared by diagrams and ICDs)

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

### 6.3 Extractors

Each extractor implements `extract(root) -> RepoIR` and adds to the interface inventory.

| Source | Provides | Phase |
|--------|----------|-------|
| Python AST | structure, calls, raises, docstrings | 0 |
| Python framework detectors | FastAPI/Flask/Django routes, argparse/Click/Typer CLIs, Celery tasks, Kafka/pika/kombu topics, SQLAlchemy models, pydantic/dataclass types, `os.getenv` names, outbound `requests`/`httpx`/`boto3` clients (*required* interfaces) | 0–2 |
| implemented so far | FastAPI, argparse, Click, Typer, pydantic/dataclass/enum, environment (direct and through wrapper functions, required vs. defaulted per module), **pika** (queues, exchanges, routing keys, message field names from `json.dumps({...})` / `json.loads(...)[...]`), **Gradio** (UI events, inputs, outputs, port) | 0–4b |
| docker-compose + Dockerfile | services, what each image contains and runs (ADD/COPY, WORKDIR, CMD, ENTRYPOINT), ports, depends_on, placement, where each env value comes from (never the value) | 4b |
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

## 7. Policy

Configuration is read from up to three TOML files, each overriding the last:

1. user config `~/.config/archdoc/config.toml`: backends and trust settings
2. repo config `<repo>/.archdoc.toml`: policy and paths
3. `--config path`

**Each tier has a built-in default profile.** Files only state overrides, and
unknown keys or values are errors (a typo must not silently loosen policy).

| Key | Values | `local` default | `cloud` default |
|-----|--------|-----------------|-----------------|
| `level` | `none` · `interfaces` · `callgraph` · `control_flow` · `source` | `source` | `callgraph` |
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

### 7.1 Egress levels

Independent of tier, the level controls *how much structure* is sent:

| Level | Sends | Enables |
|-------|-------|---------|
| `none` | nothing (no LLM) | class, component, ICD tables (all deterministic) |
| `interfaces` | IR minus call edges | + deployment, object, state, use-case, ICD narrative |
| `callgraph` | + resolved internal call edges | + sequence, ICD sequencing |
| `control_flow` | + per-function nesting of `if/for/while/try/return` with call ids, no expressions or literals | + activity (higher fidelity) |
| `source` | + full source of the most relevant functions, packed to fit the window (local only by default) | + behavior specifications; accurate activity, state and sequence groups |

A level is a ceiling, not a quota: the packer (Section 12) decides how much of
the allowed detail fits each request. `source` is implemented in Phase 4a; until
then `control_flow` is the effective local maximum.

### 7.2 Secret scanning

A backstop, not the primary control. High-entropy tokens and known patterns
(AWS keys, `sk-` keys, JWTs, private-key headers) found in anything about to egress
are dropped (`local`) or abort the run (`cloud`).

## 8. Pseudonymization (cloud tier)

### 8.1 Round trip

Aliasing happens **only around the LLM step**:

1. `egress`: the sanitized view is rewritten with aliases. The alias map stays local.
2. The LLM plans and writes using aliases only.
3. `ingress`: every alias the LLM references is checked against the map. Unknown
   references are rejected (this is the hallucination check). Known ones are
   restored to real names.
4. Renderers work only with real names and local facts.

Final diagrams and ICDs always contain real names. Only the LLM's view, and in
harness mode the agent's transcript, contains aliases.

### 8.2 Alias format

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

### 8.3 Role hints

Names carry meaning, and aliasing removes it. An optional local `roles.toml`
(`"billing.invoices:InvoiceReconciler" = "reconciliation service"`) lets the user
supply a neutral description that is sent instead of the real name.

## 9. Diagrams

| Diagram | Producer | Input |
|---------|----------|-------|
| class | deterministic | classes, bases, fields, methods; composition from field types |
| component | deterministic | module import graph by package; **provided/required interfaces** from the inventory as lollipop/socket notation; message interfaces drawn as the broker's queues/exchanges with publisher → channel → consumer arrows |
| deployment | deterministic (compose files) | services and the modules each image contains; message flows through the broker's service. Repos without deployment files: LLM planner later (entrypoints, IO categories, env var names) |
| flow (cross-service sequence) | deterministic | from each UI/HTTP/CLI operation: the handler's internal calls in source order to each publish, through the broker to every deployed consumer, on from each consumer's callback; fanout as `par`; a publish with no deployed consumer ends in a note naming any consumer the deployment leaves out. One per entry per deployment file. Source order, not run-time order: branches, retries and loops are not shown, and the legend says so |
| object | LLM planner | data types with fields, constructor signatures |
| state | LLM planner | enums, `status`/`state` fields, transition-named methods |
| use-case | LLM planner | provided interfaces (routes, CLI commands, tasks), required interfaces as actors |
| sequence | LLM planner | call graph from a chosen entrypoint or operation, depth-limited |
| activity | LLM planner | call graph, plus control-flow skeletons if level allows |

Planner output is validated JSON (Section 11), rendered by
`archdoc/render/puml/<type>.py` and checked with `plantuml -syntax` when PlantUML is
available. The existing `repair_*` functions and the PlantUML-syntax FAISS RAG are
not used on the v2 path.

## 10. Interface Control Documents

### 10.1 What is deterministic vs. LLM-written

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

### 10.2 Output

- **Formats:** Markdown (default) and AsciiDoc, with PlantUML embedded as source or
  as rendered SVG. docx and PDF via pandoc (optional dependency).
- **Templates:** Jinja2, user-overridable (`--icd-template path`). Ship two:
  - `lightweight`: one document per interface or per component pair.
  - `idd`: section layout modeled on the MIL-STD-498 Interface Design Description
    (DI-IPSC-81436). The exact section mapping is to be confirmed against the DID
    during Phase 4.
- **Granularity:** `--icd-scope` `interface | component | system`.

### 10.3 Drift check

`archdoc check` compares the current interface inventory with the snapshot saved
when the ICD was last generated (`<out>/.archdoc/icd_snapshot.json`) and reports
added/removed operations, field type changes and removed required fields, flagging
breaking changes. It needs no LLM, so it can run in CI.

## 11. Planner (LLM step)

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

## 12. Fitting the context window

With a local model the limiting factor is the context window, not cost. The
practical window is whatever `--max-model-len` (vLLM) or `num_ctx` (Ollama) fits
in VRAM alongside the KV cache, usually well below a model's advertised limit:
the legacy script uses 48k for Mistral Large on 4×A100. The goal is to give the
model **as much real code as fits** for each artifact, without truncating anything
mid-item, and without losing consistency across requests.

### 12.1 Consistency does not depend on chunking

The legacy pipeline had to chunk large repos, and its diagrams then disagreed on
names. The cause was that the model produced the names: Pass 2 built the entity
registry with an LLM, and Pass 3 wrote PlantUML freehand, so every chunk was a new
chance to rename things. Here the model never produces a name:

- names and ids come from the IR, identical in every request, diagram and ICD
- every reference field is an enum of the ids that request exposed, and
  ingress rejects anything else
- renderers take display names from the IR, never from model text

Work can therefore be split as finely as the window requires with no
drift between artifacts.

### 12.2 Window-aware packing

- **Window size** comes from backend config (`num_ctx`, or `context` for
  OpenAI-compatible servers). Reserved: instructions, schema, output
  (`max_tokens`) and a safety margin. The rest is the packing budget for that request.
- **Token counts are calibrated per model.** Backends report real prompt token
  counts (Ollama `prompt_eval_count`, OpenAI-compatible `usage.prompt_tokens`). The
  measured characters-per-token ratio is stored in the state directory. The first
  request uses a conservative 3.0. If a request still overflows, it is re-packed
  smaller and retried.
- **Detail levels per item** (a function, class or module):

  | Level | Content | Allowed |
  |-------|---------|---------|
  | name | id only | always |
  | stub | signature, types, first docstring line, calls | always |
  | outline | stub + nesting of if/for/while/try/return with call ids, no expressions | local; cloud opt-in |
  | source | the full function or class source | local only |

- **Ranking per artifact** decides who gets upgraded first:
  - *sequence:* call-graph distance from the entry, then how often a function is called
  - *ICD narrative:* operation handlers, then exchanged types
  - *behavior spec:* the target function, then its callees
  - *module summaries:* public, most-connected members first
- **Packing:** every item in the slice starts at the minimum level the artifact
  needs (normally stub). Items are then upgraded one level at a time in rank order
  while they fit. If even the minimum doesn't fit, the work is split (shallower
  slice, or per package) instead of truncating. **Nothing is ever cut
  mid-function.**
- The level each item received is recorded and feeds the provenance evidence
  level (Section 2.1).

### 12.3 Compact encoding

The model sees Python-stub text rather than JSON. JSON remains the internal,
audited form, and the egress log records both the JSON and the exact text sent.
Measured with mistral-nemo's own tokenizer on three 25-function call-graph slices:

| Slice | Raw source | JSON payload | Stub text |
|-------|-----------:|-------------:|----------:|
| archdoc `cmd_all` | 6,341 | 2,392 (38%) | 1,390 (22%) |
| archdoc `cmd_egress` | 4,952 | 2,277 (46%) | 1,360 (27%) |
| legacy vLLM `main` | 11,187 | 2,656 (24%) | 2,051 (18%) |

The same facts cost about 40% fewer tokens as stubs, and the space saved goes
into higher detail levels. An example of the encoding:

```python
class ItemRepo:  # shop.repo
    db: Database
    def get(self, item_id: int) -> Item:  # calls Database.query; raises KeyError
```

Local ids are long (`archdoc.extract.python_ast:Resolver.resolve_name`). Within
one request, the shortest unambiguous form (`Resolver.resolve_name`) is used and
mapped back at ingress, the same mechanism as aliases.

### 12.4 Shared repo map

Every request includes a compact, ranked outline of the whole system: packages,
components, the most connected classes, and interfaces. It is capped at a fraction
of the window (default 15%). Each chunk is therefore planned with the whole system
in view. The map is identical across all requests in a run and is placed directly
after the instructions, so vLLM's automatic prefix caching (and Ollama's reuse of
the cached prompt between consecutive requests) avoids recomputing it.

### 12.5 Hierarchical summaries

System-level artifacts (use-case and deployment diagrams, the system ICD's scope
section, the design-description overview) need the whole repo, which does not fit
in one request for large codebases. This repo is ~13k tokens as stubs; pip is
~160k. The approach:

1. **Module summaries.** For each module, the model writes a structured summary
   (responsibility, key operations, collaborators, external systems) from a pack
   of that module (locally, as much source as fits).
2. **Package summaries** from module summaries, then a **system summary**.
3. System-level artifacts are planned from the summaries plus the repo map.

Summaries reference ids, not free-text names, and are verified (a collaborator
must be a real import or call edge). They are cached by content hash (module
source hash + model + prompt version), so a rerun only re-summarizes modules
that changed.

### 12.6 Implementation status (4a)

Implemented in `archdoc.packing`, `archdoc.extract.bodies` and the planners:

- **Encoding:** `[llm] encoding = "text"` (default) or `"json"`. Functions are
  Python stubs; everything else is `key: value` lines with flat dicts and
  scalar lists inline. The JSON (`Payload.data`) stays the audited form.
- **Short ids:** in each planner request, every internal id in a reference field
  becomes its shortest unique suffix (`archdoc.cli:cmd_eval` → `cmd_eval`),
  including the schema enums. Ingress maps them back in `ref`/`from`/`to`/`call`
  fields, and repair messages are rewritten into the short forms. Aliases and `ext:`
  ids are left alone. The whole-repo dry-run preview keeps full ids.
- **Levels:** stub, outline (control flow, calls as written, raised types, no
  expressions; builtins dropped) and source (decorators included). Bodies
  come only from local, unaliased modules whose policy reaches `control_flow` or
  `source`, and every body passes the secret scanner. The local default level is
  now `source`, as designed.
- **Budget:** window from `num_ctx` (Ollama), `context` or vLLM's reported
  `max_model_len`, else 8k. Reserved: prompt, answer, the answer echoed in a
  repair, and a 10% margin. Characters per token start at 3.0 and are lowered
  to the lowest ratio measured per model (`.archdoc/calibration.json`).
- **Packing:** stubs first, then outline for every item in rank order (call-graph
  order), then source, each kept only if it fits. If not even the stubs fit, the
  repo map is dropped, then the slice is shrunk (25 → 16 → 10 ...), and finally
  the 2-function minimum is sent with a warning. An overflow (Ollama: prompt
  tokens reach `num_ctx`, which it would silently truncate; vLLM: a context-length
  error) recalibrates, repacks at ≤60% and retries once. Levels and the fit are
  recorded per request in the provenance sidecar, and per message as `detail`.
- **Repo map:** modules ranked by importers + 2 × interfaces owned, each with its
  docstring line and interface kinds, capped at 15% of the payload budget. Placed
  first in every payload, so it is identical across a run, logged, and covered by
  the leak tests.

Measured (mistral-nemo tokenizer, this repo's four largest slices and four interfaces,
local and cloud, stub level, same facts): local sequence payloads are **27–36%
smaller** than JSON, cloud ones about 10% (aliases are already short), and
narratives 11–16% locally. All 15 sequence entries and 14
narratives of this repo fit an 8k window with no truncation, carrying 59 functions
as source and 29 as outlines; at 32k every function is sent as source.

Found on the way: the secret scanner's AWS-key pattern required a word boundary,
so a key glued to more characters (`"AKIA...J"[:20]`) passed. It no longer does.

### 12.7 Implementation status (4c)

`archdoc design REPO` (or `archdoc all --design`) writes
`<out>/design/<repo>_design.md` with its provenance sidecar, plus
`manifest.json`, `tests.json` and `design_coverage.json`:

- **Behavior specifications** (`plan_behavior`): one request per public
  function. The payload is the target first in rank (source where policy
  allows), its callees (upgraded as room allows) and, for a method, its class's
  fields. Inputs, side effects and errors are enums of the target's parameters,
  its calls and the exception types it or a callee raises, so a claim about an
  error the code cannot raise cannot be expressed. Every text field goes
  through the overlap guard. The specification is labelled with its evidence
  (`source`, `outline`, `signature`), shown next to it in the document.
- **Trivial functions are specified from the AST**, without a model: an empty
  body, returning a parameter, a constant or an empty container, or returning
  one call made with the parameters as they are. mistral-nemo described an
  empty Click group as "seeding the database" and `return item` as validating
  and storing it; these now read "Has no behavior of its own" and "Returns its
  argument `item` unchanged". (9 of the fixture's 18 functions; 2 of this repo's 364.)
- **Hierarchical summaries** (12.5): module summaries from a pack of the module
  (members most called from elsewhere first), then package summaries from their
  modules' summaries, then the system summary from the packages'. Collaborators
  are enums of the module's import and call edges; externals are its
  third-party imports (the standard library is not an external system). The
  text passed upward is the model's own wording as it was written (aliased for
  an aliased tier), never the restored text. A package with one module reuses
  that module's summary; with one package, its summary is the system's. The
  content-hash cache is the response cache: a package's payload is made of
  its modules' summaries, so it is re-asked exactly when one of them changed.
- **Dependency manifest** (`archdoc.manifest`, deterministic): requirements
  files, `pyproject.toml` (PEP 621 and Poetry), `setup.cfg` and Dockerfile `pip
  install` lines, each covering the code under its directory; the third-party
  imports of every module, matched to distribution names (a table covers
  `yaml`→PyYAML, `cv2`→opencv-python and the like). Disagreements are listed:
  imported but not declared by any covering file, declared but not imported
  (tools and servers are noted as such).
- **Test inventory** (`archdoc.testsuite`): tests are extracted with the code
  even when the diagrams exclude them (only excludes that name tests are
  ignored), and never drawn. Per test: location, the documented functions and
  classes it calls, the exceptions it expects (`pytest.raises`,
  `assertRaises`), assertions, parametrized cases, fixtures and markers, all
  from the AST; one request per test module adds an acceptance criterion per
  test. Each function's specification lists the tests that exercise it.
- **Literal guard** (2.5): a distinctive string literal from a function body
  quoted verbatim is an overlap, however short. Distinctive: at least 8
  characters and structure (SQL, a regex, a format string) or at least 5 words.
  Names (identifier-shaped literals), URLs and interface strings are allowed.
  mistral-nemo quoted error details and magic strings into its first
  specifications; the guard sends them back without quoting them.
- **Coverage**: behavior specifications per evidence level, module summaries,
  tests with criteria, public functions exercised directly by a test, and
  dependency issues, printed and in `design_coverage.json`.

Not yet: review-overlay entries for specifications and summaries (they are
regenerated, or replayed from the cache), the SDD layout (4d), and checking an
algorithm's steps against the outline (a step naming a call the function
does not make).

## 13. Harness integration

### 13.1 Core idea

A coding agent has its own `read`, `grep`, `bash`, `write` and web tools, and will
happily open source files, bypassing every control above. The integration has three
parts:

1. **An MCP server** that is the agent's only window into the repo, and its only
   way to write outputs.
2. **A restricted agent profile** with file, shell, edit and web tools disabled.
3. **A trust tier fixed at server launch** to match the profile's model.

### 13.2 Trust in harness mode

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

### 13.3 MCP tools

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

### 13.4 Harness configuration sketches

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

## 14. Egress auditing

- `archdoc egress --dry-run --backend <name>` prints exactly what would be sent for
  each artifact under that backend's tier, with token counts, without calling a model.
- Every real run appends to `<out>/.archdoc/egress.jsonl`: timestamp, backend, tier,
  artifact, payload (or hash only, configurable), token count.
- Ingress rejections (unknown aliases, schema failures) are logged alongside.

## 15. Backends

Two transports, both stdlib-only (`archdoc/backends/`), selected per backend by `kind`:

| `kind` | Endpoint | Schema enforcement | Typical use |
|--------|----------|--------------------|-------------|
| `ollama` | native `/api/chat` | `format` (local models only; **not** enforced for Ollama cloud models) | laptops, the Spark's Ollama |
| `openai` | `/chat/completions` | `response_format: json_schema` | vLLM (enforced), OpenAI-compatible servers, cloud APIs |

The exact response format is also written into the prompt, because enforcement
can't be assumed. Schema validation and ingress remain the real guards.

### 15.1 Choosing a backend at run time

Several backends can be configured, and the run uses whichever is available:

```toml
[llm]
prefer = ["spark-vllm", "spark-ollama", "laptop"]   # tried in order
allow_cloud_fallback = false                         # default

[backends.spark-vllm]
url = "http://spark-c432-host-203-52.netbird.cloud:9999/v1"
model = "<id from /v1/models>"
kind = "openai"
trust = "local"
```

`--backend auto` (the default when `prefer` is set and an LLM artifact is requested),
`--backend a,b` (an explicit list) or `--backend NAME`. Each candidate is checked in order:

1. trust classification (a host that fails verification is skipped)
2. **no silent move to cloud:** a cloud candidate is skipped unless
   `allow_cloud_fallback = true` or it was named on its own
3. health: reachable within 10 s **and serving the configured model**
   (Ollama `/api/tags`, OpenAI-compatible `/models`)
4. warm-up: Ollama loads the model once, with the run's `num_ctx` and `keep_alive`

The first candidate that passes is used, and every skipped candidate is printed with
its reason. During the run, the first connection failure or timeout trips a circuit
breaker: remaining LLM requests are skipped and reported, and deterministic outputs
are still written. Provenance records which backend and model produced each item,
because results differ between models.

Measured on the DGX Spark: two large-model servers sharing its unified memory (a
120B model resident in vLLM while Ollama loads a 108B model) drove the host into
thrashing, freezing even `sshd`. Run one large-model server at a time there.
Ollama's first prompt after a load also paid a large one-time cost (19.8 s cold vs
50 ms warm on an 8B model), which the warm-up absorbs.

Planned: `vllm_offline` (in-process batched `LLM.generate()` for very large batch
runs) and a direct Anthropic client.

## 16. CLI and package layout

```
archdoc extract     <repo>                # write IR + interface inventory
archdoc egress      <repo> --backend X --dry-run
archdoc diagrams    <repo> --backend X [--types class,sequence]
archdoc icd         <repo> --backend X [--scope interface] [--template idd]
archdoc all         <repo> --backend X    # diagrams + ICDs, one extraction (--design: + design description)
archdoc design      <repo> --backend X    # summaries, behavior specs, dependency manifest, test inventory
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

## 17. Testing

- **Canary leak test (most important):** a fixture repo with unique canary strings in
  bodies, comments, docstrings, defaults, string literals, route paths, topic names
  and a fake AWS key. Assertions per tier:
  - `cloud`: no canary and no real internal identifier appears in any payload or
    MCP response.
  - `local`: below level `source`, no body, comment, literal or secret canary
    appears. At level `source`, body canaries appear only for functions the packer
    selected, and only whole. The secret canary never appears, inside sent source
    included.
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
- **Packing tests:** every request fits its configured window. No item is cut
  mid-function. Upgrades follow rank order. Overflow leads to re-packing, never
  truncation.
- **Verification tests:** injected faults (a message to the wrong owner, an
  out-of-order call, a state no code assigns, an error claim with no matching raise)
  are each caught and removed with a warning.
- **Accuracy evaluation (Section 2.6, 17.1):** recorded, not gated in CI.

### 17.1 Accuracy evaluation

`archdoc eval REPO --spec SPEC.toml --backend a,b,c` writes `eval.md` and
`eval.json`. A spec names a reference diagram (Mermaid flowchart) in the
evaluated repo, the deployment file to compare with, node-name overrides, and
flows the reference implies but that cannot be read from it mechanically.

- **Structure** (no model): reference nodes are matched to services by name.
  Edges through a broker node are paired into service-to-service flows by shared
  label words, and the pairing is printed so it can be checked. Each reference
  flow is *matched* (a deployed channel connects the services), *in code, not
  deployed*, or *missing*, with evidence: the channel, or what each side
  publishes and consumes. Deployed flows the reference lacks are listed. Recall
  and precision summarize, but the rows are the result: the reference can be
  wrong too.
- **Models**, measured against the code since no reference states sequence
  details: same entries and narratives on every backend; answers valid first
  time / after repair / failed; share of proposed messages verification kept;
  share of the call graph's edges each diagram shows; names corrected; labels
  cleared; narrative slots filled; tokens; time.

The first spec, `eval/celebrity-rec-rabbitmq.toml`, uses the author's public
repo and Mermaid diagram.

## 18. Phased plan

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
- **Pseudonymized runs keep only reversible text.** The prompt asks the model to
  write labels in terms of the references it was given ("Create @C2, then call
  @F3"), which reverse mapping turns into real names ("Create CachedRepo, then
  call get"). Sequence labels and titles with no reference in them were invented
  by a model that never saw real names, so there is nothing to map back. They are
  cleared (calls fall back to the function name, titles to the entry point) and
  reported. Measured on the fixture: without this instruction, mistral-nemo wrote
  "Place Order → Check Inventory" for build_repo → get → get. With it, nearly
  every label referenced aliases and came back accurate ("Invoke cmd_serve",
  "Create CachedRepo", "Call get").
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
| 4a | **Context fitting:** stub-text encoding, window config, calibrated token counts, tiered packer (name → stub → outline → source), shared repo map | same artifacts use about 40% fewer tokens at equal detail; the largest slices fit an 8k window with no truncation |
| 4b | **Accuracy:** evidence fields, provenance sidecars, IR verification for sequence diagrams, coverage report, review overlay, verbatim-overlap guard, reference set for accuracy evaluation | every generated statement cites evidence; an injected wrong call edge is caught; coverage reported |
| 4c | **Rebuild content:** behavior specifications from source (local), hierarchical summaries with content-hash cache, dependency manifest, test inventory | behavior coverage reported per evidence level; a rerun with no code change makes zero model calls |
| 4d | remaining diagrams (activity, state, use-case, deployment, object), each with its own verification; `idd` template; design-description document; embedded diagrams; pandoc export | all outputs produced and verified against the IR |
| 5 | MCP server; opencode and Vibe local/cloud profiles; `AGENTS.md` | full output set from each harness with file tools disabled |
| 6 | drift check; **rebuild-readiness check** (stubs regenerated from the documentation, diffed against the IR); Flask/Django/Celery/Kafka/SQLAlchemy detectors; `.proto` and OpenAPI extractors; tree-sitter languages; retire legacy scripts | rebuild-readiness diff is empty on the fixture |

## 19. Open questions

1. **Name.** `archdoc` is a placeholder for the suite.
2. **ICD template.** What shape were the harness-generated ICDs? Is a
   MIL-STD-498 IDD layout wanted, a program-specific template, or something lighter?
3. **ICD granularity default:** per interface, per component pair, or one system ICD?
4. **Behavior specification format:** a MIL-STD-498 SDD-style document, per-module
   Markdown, or something program-specific? How much detail per function
   (prose only, or prose plus pseudocode)? *4c: one Markdown document, modules
   in order, prose steps (no pseudocode); the SDD layout is 4d.*
5. **Unverifiable model claims:** remove them (current default) or keep them,
   clearly marked as unverified?
6. ~~**Tests as evidence**~~: read for the test inventory even when excluded from
   the diagrams (implemented in 4c).
7. ~~**Accuracy reference set**~~: github.com/tranzmatt/celebrity-rec-rabbitmq
   (public; author's Mermaid diagram as reference), spec in `eval/`.
8. **Glossary script:** fold into the ICD as an acronyms/terms appendix, or retire?
9. **RAG corpus:** with JSON plans there is no need for PlantUML syntax examples.
   Retire, or repurpose (e.g. example ICD narratives)?
10. **Target languages and interface formats after Python**, in priority order
    (`.proto`, OpenAPI, Java, C++, ...)?

Resolved: pseudonymization is per tier (off locally, on for cloud), with reverse
mapping of model text. Control-flow outlines and source are allowed locally; in
cloud mode outlines are opt-in and source is never sent.
