# Entelecheia architecture

Entelecheia separates model decisions from Host-owned execution, authorization and durable state. The promoted workflow is a bounded L1 document task in an isolated Session attached to a Project. The Python package is still named `personagraph` for internal compatibility; there is no persona subsystem or role-card injection path.

## Execution ownership

```text
Electron / Vue → local API → Runtime Entry → L1 controller
                                              ├─ model-call owner → provider adapter
                                              ├─ tool-call owner → tool/domain adapter
                                              └─ validated delivery → Session persistence
```

- `api/`: validates HTTP inputs, enforces local API identity, projects public responses and delegates application work.
- `runtime/entry/`: accepts/finalizes Turns, freezes routing and execution identity, coordinates recovery and selects the authorized lane.
- `runtime/l1/`: controls bounded plan/action/review iteration and model-context projection. It does not own database schema or provider transport.
- `runtime/model_calls/`, `runtime/tool_calls/`: logical/physical call authority, settlement, retry/recovery and observability.
- `model_io/`: provider/dialect adapters, structured output decoding, budgets and model-profile contracts.
- `tools/`: reviewed definitions, catalogs, resource binding, authorization and domain adapters. `tools/model_interface/` owns the model-facing projection, not a second tool execution implementation.
- `session/`, `workspace/`: durable Session and Project ownership. Domain storage receives explicit connections/resources; it does not invent independent GUI/eval state.

L2 task/auxiliary graphs remain referenced by routing, storage and regression tests. They are outside the promoted L1 feature/evaluation scope, but are not dead code simply because the default routing policy disables them. The retired top-level persona/legacy graph packages are absent. Local UI appearance and scoped Session context are separate active features.

## Documents and retrieval

Project files have stable file/version identities. The ingestion owner coordinates parsing, chunks, coverage checks and publication of retrieval representations. Unsupported or missing content should be represented as a coverage gap, not silently turned into evidence.

`input_processing/documents/` owns format parsing; `workspace/ingestion/` owns durable processing jobs; `retrieval/` owns representation generations, query execution and ranking. Production retrieval uses fixed BGE-M3 revisions for dense and learned-sparse encodings plus BM25, with a fixed reranker. Source/version/coverage and actual device identity constrain which generation may serve a request. A configuration change is not automatically a valid index migration.

Visual operations use separate source authorization and provider-call boundaries. Successful transport is not proof of complete visual content or answer correctness. Local model readiness, actual query-method use, gold-evidence recall and downstream answer quality are different checks.

## State boundaries

```text
private state root/
  project_catalog.sqlite
  projects/<project-id>/documents.sqlite
  sessions/<session-id>/session.sqlite

user Project root/                 private config root/
  user files and generated files     settings, model profiles, credentials
```

The catalog locates Projects/Sessions. A Project database owns file versions, document blocks and retrieval representations. A Session database owns history, runtime/task state, permissions and tool/model ledgers. Sharing a Project does not share Session history or grant another Session's permissions.

Schema definitions and catalog factories are required source code. Generated databases and catalog contents are private runtime data and are not shipped. The internal `persona_id` compatibility column remains fixed to product identity at public Session creation; public views do not expose role selection.

State/config/token overrides must be absolute and outside the source checkout. Model tools are denied access to Host private configuration and authority. User-bound workspaces are separate from these roots. A default new-file capability does not authorize overwriting arbitrary user files.

## Model context and recovery

Model-facing context carries current plan/notes and a bounded view of recent tool content. Full tool observations remain in the durable result owner and can be read using the tool-history interface. A history projection is not a new original evidence source.

Local Turn/attempt identity, leases and settlement records support resumption and replay. An uncertain remote request may still have run at its provider; local idempotence is not a promise of remote exactly-once execution. Preserve these error and recovery distinctions when changing adapters.

## Evaluation and public artifacts

The [DocBench runner](evals/docbench/reproduce_or_run_script/README.md) invokes the same L1 product path. It creates isolated case-attempt processes and state, never uses GUI Session databases, and never mounts the QA/reference source directory as a Project. See [state isolation](evals/docbench/docs/state_isolation.md).

Checked-in configs/selections/schema preserve experimental identity. Original benchmark materials, full prompts/replies, private attempts and databases remain outside the checkout. Only reviewed sanitized results are public. Frozen historical source/config and current source are distinct provenance identities; a historical benchmark result does not validate later code.

See [CONTRIBUTING.md](CONTRIBUTING.md) for change/test/privacy requirements and [README.md](README.md) for the supported installation and feature limitations.
