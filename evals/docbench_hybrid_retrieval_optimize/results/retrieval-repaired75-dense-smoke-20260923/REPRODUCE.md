# Reproduce the bounded CPU semantic check

Run from the repository with its .venv interpreter:

```bash
.venv/bin/python evals/docbench_hybrid_retrieval_optimize/results/retrieval-repaired75-dense-smoke-20260923/reproduce_smoke.py \
  --dataset evals/docbench_hybrid_retrieval_optimize/dataset/repaired75-20260923/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/A_NEW_SMOKE_DIRECTORY \
  --max-length 1446
```

Script SHA-256: `996effd4f56188cc61921a2f58b488076c791e45a9eb8b7f288ceb4d601d43f1`. The selected pool intentionally includes known relevant evidence; this is a semantic integration check, not a full-corpus retrieval benchmark.

This public script uses the new retrieval package and repository-relative artifact paths. The tokenizer identity is `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`; its files must already be locally available. Offline flags and `local_files_only=True` are preserved. No model was run while preparing this public projection.

For the full frozen-corpus BM25 baseline, use the package entry point from the repository root:

```bash
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --dataset evals/docbench_hybrid_retrieval_optimize/dataset/repaired75-20260923/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/A_NEW_BM25_DIRECTORY
```
