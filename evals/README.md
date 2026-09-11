# Entelecheia evaluations

DocBench is the supported benchmark integration, evaluating the L1 product path. The canonical entry is the repository module, not an author-specific launcher. Run from the checkout root after the [locked project setup](../README.md):

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
```

Replace the example with an absolute directory outside the checkout. Set it **before Python starts**; data-preparation defaults are resolved at import time. No external wrapper script is required. The module is run from source; it is not a separate installed console command.

## Ownership and prerequisites

| Location | Purpose |
| --- | --- |
| [docbench/reproduce_or_run_script/](docbench/reproduce_or_run_script/README.md) | One implementation of preparation, selection, validation, run/retry and scoring. |
| `docbench/configs/`, `docbench/selections/`, `docbench/config.schema.json` | Portable config spelling, frozen content-free selections and schema. Preserve their paths/hashes. |
| `docbench/results/` | Reviewed sanitized summaries, not original answers or trajectories. |
| [runbook](docbench/docs/formal_l1_eval_runbook.md), [state isolation](docbench/docs/state_isolation.md) | Public operator and process-boundary guidance. |
| `<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/` | Private `source/`, `runs/`, `workspaces/` and optional evaluation config. Never publish this tree. |

The runner uses the real Session/Project ingestion, retrieval and L1 execution path. It does not inject reference answers or dictate a retrieval/vision tool order. `runtime_closed_world.yaml` removes external web capabilities at the tool boundary; L2 is not part of this benchmark.

Install dependencies and fixed model assets using the project setup. The supported dependency lock includes `gdown` for explicit data download; do not install an unpinned standalone copy as a reproduction shortcut. Native document processing is the baseline; optional layout dependencies alone do not provide ready layout models.

Benchmark files are not distributed here. The [pinned upstream manifest](docbench/upstream.manifest.json) records the upstream revision, expected hashes and the prior license review; it is not a grant to redistribute upstream code or third-party PDF/QA. Obtain authorized materials separately, keep them outside the checkout, and provide the pinned scoring prompt at `docbench/source/evaluation_prompt.txt`.

## Configuration and preparation scope

`l1_bge_m3_live_1.yaml` is a minimal single-case run; `l1_storage_smoke_3.yaml` is an engineering smoke; `l1_stage_20.yaml` and the small `*_regression_*` configs are focused regressions. `l1_balanced_125.yaml` selects 125 questions. Small regression sets are not population accuracy estimates. Run `list` for the full current catalog and canonical hashes.

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

The first two commands require network access. They prepare the QA catalog and balanced125 PDFs only, not every regression config, upstream checkout, scoring prompt or model weights. The single-case doc 104 is included; other selections may require additional PDFs. There is no `prepare-data --config` capability. `readiness` checks the checked-in catalog offline and may fail because another config's materials are missing; it is not a paid Provider smoke.

## Running and interpreting results

Follow the [runbook](docbench/docs/formal_l1_eval_runbook.md) for live commands, resumption, explicit retry selection and scoring. `run`, `retry-failed` and `score` require `--allow-live`; they can incur cost and send necessary prompts/document excerpts/page images to configured providers. Generation and scoring are separate operations.

Execution completion, actual L1 routing, outstanding user interaction, source stability and answer score are separate facts. The mechanical generation gate is not a correctness score. The configured scorer is `docbench_prompt_compatible` with the configured local judge identity, not an official-comparable result (`official_comparable=false`).

The historical 125-question first pass and 3+5 retry batches belong to frozen source `l1-gpu-deadline25-20260910-a`. Eight retries do not create eight new independent questions. Keep first-pass results, retry lineage and replacement policy visible rather than reporting only selected later answers. Current code/config changes require a new run identity; historical results do not establish current-version performance. Exact reproduction requires the original frozen source/config/environment and authorized data, not just matching question IDs.

Only publish an explicit allowlist projection with stable IDs, hashes, numeric/enum results and methodology. Do not copy raw manifests unchecked: they can include local paths and private settings. Original PDFs/QA, full answers, prompts, judge output, logs, Session IDs/DBs and trajectories stay private. Run the repository privacy guard on any proposed summary; filename suffix alone does not make a file publishable.
