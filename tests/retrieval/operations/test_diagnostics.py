from __future__ import annotations

from personagraph.retrieval.operations.diagnostics import retrieval_health_snapshot


def test_health_snapshot_is_content_free_and_reports_missing_active_version(tmp_path):
    snapshot = retrieval_health_snapshot(db_path=tmp_path / "retrieval.sqlite")

    assert snapshot["runtime_activation"] == "consumer_composed"
    assert snapshot["reconciliation"]["healthy"] is False
    assert snapshot["reconciliation"]["issue_counts"] == {"no_active_data_version": 1}
    assert "content" not in repr(snapshot).lower()
