# DocBench retrieval baseline

Route: `sqlite_fts5_bm25`. SQLite FTS5 BM25; not production BGE hybrid retrieval.

Search is scoped to the question document. Text queries search chunks; multimodal queries search table and figure descriptions together.
Only unit content is indexed. Answer, evidence and annotation fields never enter retrieval.
No minimum score threshold or artificial fallback hit is applied.
BM25 uses global IDF and length statistics from all snapshot units; document and modality restrictions filter candidates without recomputing those statistics.
Binary relevance over a flat primary-gold set: Hit means at least one gold unit; Recall measures gold-unit coverage. Equivalent alternatives and jointly necessary evidence are not grouped, so these metrics do not measure answer completeness.

Scored 66 / 75 queries. Pending: 9; reviewed without primary labels: 0.
Unscored queries are excluded from metric denominators, not treated as successes or failures.

| Group | Scored | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 66 | 0.7908 | 0.5871 | 0.7816 | 0.8561 | 0.9247 | 0.7953 |
| text-only | 49 | 0.9267 | 0.7296 | 0.8486 | 0.9286 | 0.9558 | 0.8979 |
| multimodal-t | 8 | 0.4365 | 0.2500 | 0.6250 | 0.6250 | 0.7750 | 0.5106 |
| multimodal-f | 9 | 0.3656 | 0.1111 | 0.5556 | 0.6667 | 0.8889 | 0.4901 |

Full per-case Hit, Recall and nDCG at 1, 3, 5 and 10 are in report.json.

Public migration note: this is the original experiment with source-path metadata projected for publication. Metrics, rankings and saved vectors were not recomputed. `report.json` preserves `origin_dataset_sha256`, `origin_manifest_sha256` and `origin_report_sha256`; its current dataset/manifest hashes identify the public metadata projection.
