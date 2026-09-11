# Contributing to Entelecheia

Entelecheia is a local single-user document-task workbench under active development. Read [AGENTS.md](AGENTS.md), [the architecture](ARCHITECTURE.md) and the code/tests for the affected domain before changing it. Historical development reports and private evaluation workspaces are intentionally not part of this checkout.

## Scope and ownership

- Keep one canonical implementation and one active execution path per responsibility. Change its callers and tests together; do not add a second generation as a permanent fallback.
- Follow the dependency direction: API/frontend/tool adapter → application orchestration → domain contract/policy → explicit backend/store/provider port. Avoid back-importing API or orchestration from a lower layer.
- Separate pure validation/projection from I/O. Own mutable configuration, database access, tool registration and state transitions in one place.
- Introduce abstractions only for concrete boundaries. A long cohesive schema/contract table is not dead code; a test import alone does not make every internal symbol a public API.
- Do not mechanically rename the internal `personagraph` package or stored identities. L0/L1/L2 name processing levels, not generations. The retired persona subsystem must not be reintroduced.

## Contracts and safety

Tool schemas, API envelopes, persistence schemas, status enums and safety limits each have one source of truth. Model outputs are proposals: validate them before durable state changes or external effects. Unknown states/providers/backends fail closed.

Preserve path authorization, private-state isolation, output limits, credential redaction, replay/lease checks and error observability. Do not weaken an assertion, swallow an exception, broaden a fallback or hardcode success to make a test pass. Keep commands as structured arguments and use explicit scoped filesystem paths.

Storage and wire schema versions are allowed at their boundary; they do not justify duplicate business runtimes. A schema change must specify supported starting versions, integrity/fingerprint behavior and preservation of existing data. In particular, removing a compatibility column is not a packaging cleanup.

## Tests and local setup

Use the supported locked setup from [README.md](README.md). Python dependencies are locked for the declared platform; do not document ad hoc installs as equivalent reproducible environments. Do not copy another checkout's `.venv` or interpreter into this one.

```bash
.venv/bin/python -m ruff check src tests scripts evals
.venv/bin/python -m pytest -q
./scripts/run-project-pnpm.sh --dir frontend run check
```

Run targeted domain tests while iterating, then the full relevant suites. Tests mirror source responsibilities. Preserve `tests/conftest.py` isolation, synthetic fixtures and subprocess/cold-import checks. Ordinary tests must not inherit saved user credentials, real provider behavior, user databases or background jobs from a previous test. Real device/network/paid API checks require explicit opt-in and must be reported separately from mock results.

For a change, report the behavior affected, tests actually run and remaining limitations. A passing unit suite does not prove GPU inference, provider availability, document coverage or benchmark accuracy.

## Documentation and evaluation

Keep the public README, architecture and supported runbook aligned with current code. Prefer a durable explanation of interfaces and limitations over a historical change diary. Do not add links to omitted local reports, author-specific desktop paths or private run artifacts.

Root `doc/` and `docs/` are ignored local development-record areas and must not be published. Public `README.md`, `ARCHITECTURE.md` and `CONTRIBUTING.md` are maintained separately as usage, architecture and contribution guidance; do not restore old private reports into the public copy.

DocBench has one [repository-owned execution package](evals/docbench/reproduce_or_run_script/README.md). Keep experimental config/selection identities immutable when comparing historical runs. New current-code results must not be presented as exact reproduction of a different frozen source/config. Wrong answers, execution failure, coverage and official comparability are distinct measures.

## Privacy before publication

Never commit user documents, original benchmark PDF/QA, raw prompts/replies, trajectories, credentials, runtime databases, model weights or caches. Use synthetic fixtures and reviewed content-free selections/summaries. A `.summary.json` suffix alone is not approval: the schema and fields must pass the repository privacy policy.

```bash
.venv/bin/python scripts/check_repository_privacy.py --worktree
.venv/bin/python scripts/check_repository_privacy.py --staged
```

Review the exact files and intended commit tree before publication. `.gitignore` does not remove already-tracked secrets; the checker is a guard against known patterns, not proof that arbitrary prose or code contains no private information. Do not push an unsanitized history in order to test a remote check.

Preserve unrelated work in a dirty checkout. Keep changes coherent and reviewable; do not delete or reset another contributor's edits. Publishing, pushing and creating external resources require their own authorization.
