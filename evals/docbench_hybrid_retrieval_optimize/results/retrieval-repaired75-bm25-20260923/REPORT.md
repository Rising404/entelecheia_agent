# DocBench retrieval baseline

Route: `sqlite_fts5_bm25`. SQLite FTS5 BM25; not production BGE hybrid retrieval.

Search is scoped to the question document. Text queries search chunks; multimodal queries search all supported visual descriptions together (figure, table, vector_graphics).
Only unit content is indexed. Answer, evidence and annotation fields never enter retrieval.
No minimum score threshold or artificial fallback hit is applied.
BM25 uses global IDF and length statistics from all snapshot units; document and modality restrictions filter candidates without recomputing those statistics.
Binary relevance over a flat primary-gold set: Hit means at least one gold unit; Recall measures gold-unit coverage. Equivalent alternatives and jointly necessary evidence are not grouped, so these metrics do not measure answer completeness.

Scored 75 / 75 queries. Pending: 0; reviewed without primary labels: 0.
Unscored queries are excluded from metric denominators, not treated as successes or failures.

| Group | Scored | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 75 | 0.8731 | 0.6900 | 0.8344 | 0.9000 | 0.9378 | 0.8590 |
| text-only | 50 | 0.9282 | 0.7350 | 0.8517 | 0.9300 | 0.9567 | 0.8980 |
| multimodal-t | 11 | 0.7583 | 0.5909 | 0.7727 | 0.8636 | 0.8636 | 0.7551 |
| multimodal-f | 14 | 0.7667 | 0.6071 | 0.8214 | 0.8214 | 0.9286 | 0.8017 |

Full per-case Hit, Recall and nDCG at 1, 3, 5 and 10 are in report.json.

Public migration note: this is the original experiment with source-path metadata projected for publication. Metrics, rankings and saved vectors were not recomputed. `report.json` preserves `origin_dataset_sha256`, `origin_manifest_sha256` and `origin_report_sha256`; its current dataset/manifest hashes identify the public metadata projection.
