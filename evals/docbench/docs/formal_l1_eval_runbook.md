# Entelecheia DocBench L1 runbook

Run every command from the repository root after the [supported installation](../../../README.md). These are instructions for an operator, not evidence that a live evaluation or model download has already been performed.

## 1. Freeze an external evaluation root

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

Replace the example with your own absolute path outside the checkout and set it before Python starts. No author-specific wrapper is required. Source bytes, private run state and Session Projects have separate owners:

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/
  source/       PDF/QA, Drive mapping, authorized upstream files and judge prompt
  runs/         private manifests, case attempts, state and scoring
  workspaces/   allocation root for case Session Projects
  local_config/ optional evaluation-only private Provider configuration
```

Never bind `source/` or `runs/` as a Session Project: they contain reference material and private authority. See [state isolation](state_isolation.md).

## 2. Prepare prerequisites deliberately

The [upstream manifest](../upstream.manifest.json) pins the expected revision and hashes. Obtain authorized upstream materials outside Git and put the pinned `evaluation_prompt.txt` at `docbench/source/evaluation_prompt.txt`. The repository does not redistribute the upstream checkout, prompt, original PDFs or QA. The manifest's recorded license review is not permission to redistribute them.

Use the project-locked dependencies, including `gdown`, and the model preparation/check commands in the root README. Configure your own main/vision Provider as appropriate; do not copy a developer's credentials. If isolating providers from the GUI, set `PERSONAGRAPH_LOCAL_CONFIG_DIR` to an absolute external private directory before starting the parent process. Keep custom Hugging Face cache settings consistent across setup and evaluation.

To obtain the QA catalog and the balanced125 PDF subset explicitly:

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
```

These network commands do not fetch every config's PDFs, the judge prompt, upstream checkout or model weights. The single-case configuration uses doc 104, included in balanced125; storage-smoke3 and other selections can require additional PDFs. There is no `prepare-data --config` capability.

To independently reconstruct the balanced selection, use a new external output path; existing files are not overwritten:

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script build-selection \
  --balanced \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data" \
  --output "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/rebuilt-balanced-125.json"
```

Compare the result to the checked-in selection and its hashes; do not replace the authoritative manifest merely to accept drift. Selections contain case identities and source hashes, not question/answer/evidence bodies.

## 3. Validate and select a config

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

These commands do not call Providers. `validate` checks schema/config contracts; `readiness` additionally checks source hashes, scoring prompt, credential presence and local retrieval assets across the checked-in config catalog. Missing materials for any listed regression can prevent an all-catalog ready result. It does not prove real provider/device inference or answer quality.

The smallest live check is `l1_bge_m3_live_1.yaml` (CPU). `l1_storage_smoke_3.yaml` is a small engineering check, `l1_stage_20.yaml` and other regressions are focused subsets, and `l1_balanced_125.yaml` is the 125-case configuration (currently explicitly MPS). Check the chosen YAML's device and settings. Editing it for another device creates a new experimental identity; it is not an exact reproduction of the old config.

## 4. Execute, resume or retry

The following commands require a real configured Provider and can incur charges/send document excerpts or page images. Run only after choosing that action intentionally.

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_bge_m3_live_1.yaml \
  --run-id my-l1-smoke --allow-live
```

For a formal candidate, use a new run ID and a clean worktree:

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run-id my-balanced-125 --require-clean --allow-live
```

Use `run --resume` with the same config/run ID only to resume the matching unfinished run. Existing results, including failures, are not silently overwritten. Source/config/selection/environment identity drift is rejected.

To retry only a known class of failure while preserving old attempts:

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script retry-failed \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-balanced-125" \
  --max-workers 1 --error-code MODEL_TRANSPORT_FAILURE --allow-live
```

`--error-code` may be repeated. Omitting it chooses all failed cases, so use explicit filters when reporting a defined retry policy. Each retry receives fresh case state and a fresh Host-created Project. Previous results remain under attempts and the retry report; do not discard unfavorable earlier attempts.

## 5. Score and report

Generation and judging are separate:

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-balanced-125" \
  --resume --allow-live
```

The judge uses the pinned DocBench prompt with the configured judge identity: report `scoring_protocol=docbench_prompt_compatible`, `official_comparable=false`. An answer score is not execution success. Report mechanical gate fields separately: attempt completeness, execution, L1 lane, unresolved user interaction and source stability. A failed generation gate is not simply a low accuracy score; `baseline_eligible` additionally concerns the required provenance/clean-tree conditions.

Preserve run/config/selection/source/model identities and retry lineage. The historical 125 first-pass + 3 transport-retry + 5 execution-retry results belong to frozen source `l1-gpu-deadline25-20260910-a`; 8 retries are not 8 new questions. Current source/config reruns need a new identity and cannot inherit those scores or exact-reproduction claims.

Only export reviewed, allowlisted summaries with IDs, hashes and numeric/enum measurements. Keep original PDF/QA, reference answers, full responses, judge output, prompts, logs, raw manifests/trajectories and databases outside Git. Raw manifests can contain private paths/settings; copying them is not sanitization. A public summary must pass the repository's supported schema/key policy as well as privacy review.
