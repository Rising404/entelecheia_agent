# DocBench production hybrid retrieval

One original question per query; document-scoped candidates; no LLM or answer generation.
All corpus units are indexed. Text questions search chunks; multimodal questions search all visual descriptions.
Metrics score reviewed primary labels. Flat qrels measure evidence-unit coverage, not answer completeness.

| Stage | Cases | MRR | Hit@5 | Recall@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| dense | 75 | 0.8325 | 0.9467 | 0.9149 | 0.8323 |
| learned_sparse | 75 | 0.8508 | 0.9467 | 0.9149 | 0.8446 |
| bm25 | 75 | 0.8161 | 0.9067 | 0.8778 | 0.8101 |
| rrf | 75 | 0.8648 | 0.9600 | 0.9204 | 0.8585 |
| reranker | 75 | 0.9296 | 0.9867 | 0.9604 | 0.9180 |
| packed | 75 | 0.9296 | 0.9867 | 0.9604 | 0.9180 |

Per-type results and all cutoffs (1, 3, 5, 10) are in report.json; per-query ranks are in rankings.jsonl.
Production BM25 uses BGE token-ID shadow terms; it differs from the Unicode61 baseline.
Candidate limits: 128 per method/query, RRF top 64, final at most 96 items and 96,000 estimated tokens.
Reranker input is limited to 1,024 tokens per pair; overlong pairs are counted in diagnostics.
The published index remains immutable; production initialization runs against a temporary exact copy.
