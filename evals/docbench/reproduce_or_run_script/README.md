# DocBench repository entry

This package is Entelecheia's sole DocBench implementation. From the repository root:

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

Replace the absolute example path and set it before Python starts. No separate desktop wrapper is required. See the [project setup](../../../README.md), [evaluation overview](../../README.md), [runbook](../docs/formal_l1_eval_runbook.md) and [state isolation](../docs/state_isolation.md).

## Commands

| Command | Effect |
| --- | --- |
| `prepare-data` | Explicit network download of the QA catalog or balanced125 PDFs; requires its selection flag. |
| `build-selection --balanced` | Deterministic balanced selection construction and strict reload; writes a new chosen output file. |
| `validate` | Validate the checked-in L1 configs/schema; no provider call. |
| `list` | List config IDs and canonical hashes; no provider call. |
| `readiness` | Offline, read-only check of configs, source hashes, credentials' presence and local model assets. |
| `run` | Create private attempt processes and execute real L1; requires `--allow-live`. |
| `retry-failed` | Explicitly retry matching failed cases while retaining old attempts; requires `--allow-live`. |
| `score` | Post-hoc judge with the pinned prompt; requires `--allow-live`. |

`run` does not automatically score. `--allow-live` authorizes potentially billable provider requests and transmission of needed document content. Do not use real calls to validate a documentation or packaging change.

## Canonical owners

- `cli.py`: argument dispatch and process exit status, not an alternative runner.
- `config.py`: schema, external path resolution and canonical config fingerprint.
- `download_selection.py`, `selection.py`, `derived_selection.py`: data acquisition and strictly bound selections.
- `readiness.py`: read-only prerequisites.
- `runner.py`: case worker, product Turn entry, generation gate, resume/retry ownership.
- `post_commit.py`: wait for runtime-owned post-commit settlement and project the report.
- `scorer.py`: judge protocol, checkpoints and aggregates.
- `provenance.py`, `retrieval_observability.py`: source/environment identity and retrieval observations.

`../configs/`, `../selections/` and `../config.schema.json` remain outside the implementation package intentionally. Config path spelling is included in its identity; moving or editing those artifacts may change the hash. Do not silently rewrite a frozen config while calling the result an exact reproduction.

`PERSONAGRAPH_BENCH_EVAL_DIR` points to a parent whose `docbench/` child contains private source, runs and workspaces. `bench://source/data`, `bench://source/evaluation_prompt.txt` and `bench://runs` resolve under that child. The directory is required to remain outside the product checkout. Explicit environment configuration is the portable public workflow; the retained historical fallback path is not a prerequisite or supported author-machine dependency.

The supported project dependency lock includes the `gdown` acquisition dependency. Fixed local BGE encoder/reranker assets are prepared through `scripts/prepare-local-models.py`; the eval never silently downloads missing weights. `prepare-data` does not prepare the scoring prompt, upstream checkout or every regression PDF. Consult each selection and the upstream manifest, then run `readiness`.

The default question preamble asks for an answer using the attached document without prescribing tool choice. Reference answers are judge-only inputs, never model-context or Project content. Closed-world configuration removes web tools. Reports distinguish engineering success, generation-gate eligibility and judge correctness; none alone proves the other measures.
