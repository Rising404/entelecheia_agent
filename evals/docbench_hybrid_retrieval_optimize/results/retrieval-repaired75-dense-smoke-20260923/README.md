# CPU semantic smoke

This small pool deliberately includes all selected gold evidence and a few deterministic distractors. It checks whether repaired text representations produce usable dense vectors and query rankings. It is not a full-corpus Recall score or production hybrid result. No answer, evidence, annotation rationale, or old question-conditioned embedding is passed to the encoder.

The unchanged canonical encoder may compute unused learned-sparse outputs; only normalized 1024-dimensional dense vectors are stored and scored. The max_length was set from exact local tokenization including special tokens to retain every selected input without truncation. Per-input token counts and zero-truncation flags are recorded.

Public migration note: this is the original experiment with source-path metadata projected for publication. Metrics, rankings and saved vectors were not recomputed. `report.json` preserves `origin_dataset_sha256`, `origin_manifest_sha256` and `origin_report_sha256`; its current dataset/manifest hashes identify the public metadata projection.
