from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.docbench.reproduce_or_run_script import retrieval_observability
from evals.docbench.reproduce_or_run_script.retrieval_observability import (
    _summarize_authority_outbox,
    retrieval_preflight_snapshot,
    retrieval_runtime_snapshot,
)
from personagraph.retrieval.contracts import RetrievalMethod


def test_indexing_outbox_summary_keeps_safe_batch_and_failure_evidence() -> None:
    snapshot = _summarize_authority_outbox(
        {
            "available": True,
            "attempt_audit_available": True,
            "attempt_audit_count": 2,
            "attempt_audit_truncated": False,
            "status_counts": {
                "pending": 0,
                "applied": 1,
                "terminal_failed": 1,
            },
            "terminal_failures": [{"event_id": "evt-b", "reason_code": "failed"}],
            "attempt_audits": [
                {
                    "event_id": "evt-a",
                    "attempt": 1,
                    "worker_kind": "background",
                    "worker_instance_hash": "worker-hash-a",
                    "batch_id": "batch-a",
                    "batch_limit": 20,
                    "batch_size": 2,
                    "batch_ordinal": 1,
                    "outcome": "applied",
                    "failure_stage": None,
                    "safe_error_code": None,
                    "occurred_at": "2026-08-31T00:00:00+00:00",
                },
                {
                    "event_id": "evt-b",
                    "attempt": 1,
                    "worker_kind": "synchronous",
                    "worker_instance_hash": "worker-hash-b",
                    "batch_id": "batch-a",
                    "batch_limit": 20,
                    "batch_size": 2,
                    "batch_ordinal": 2,
                    "outcome": "terminal_failed",
                    "failure_stage": "encode",
                    "safe_error_code": "bge_m3_encode_failed:RuntimeError",
                    "occurred_at": "2026-08-31T00:00:01+00:00",
                },
            ],
        }
    )

    assert snapshot["attempt_count"] == 2
    assert snapshot["outcome_counts"] == {"applied": 1, "terminal_failed": 1}
    assert snapshot["batch_count"] == 1
    assert snapshot["worker_instance_count"] == 2
    assert snapshot["failure_counts"] == {
        "encode:bge_m3_encode_failed:RuntimeError": 1
    }
    assert snapshot["attempt_audits"][1]["safe_error_code"] == (
        "bge_m3_encode_failed:RuntimeError"
    )
    assert "authority_db_path" not in repr(snapshot)


def test_preflight_records_effective_profile_and_fingerprints_without_paths() -> None:
    snapshot = retrieval_preflight_snapshot({
        "PERSONAGRAPH_RETRIEVAL_PROFILE": "lexical",
        "PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY": "strict",
        "PERSONAGRAPH_RETRIEVAL_RERANKER": "off",
        "PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY": "true",
        "PERSONAGRAPH_RETRIEVAL_DEVICE": "cpu",
        "PERSONAGRAPH_RETRIEVAL_USE_FP16": "false",
    })

    assert snapshot["status"] == "ready"
    assert snapshot["requested_profile"]["required_methods"] == ["bm25"]
    assert snapshot["effective_profile"]["mode"] == "lexical"
    assert snapshot["encoder_fingerprint"].startswith("deterministic_lexical:")
    assert snapshot["reranker_fingerprint"] is None
    assert "/Users/" not in repr(snapshot)
    assert "/private/" not in repr(snapshot)


def test_runtime_records_generation_and_each_method_count(monkeypatch) -> None:
    from personagraph.retrieval.operations import document_maintenance as maintenance
    from personagraph.retrieval.lifecycle import rollout as generation_rollout
    from personagraph.retrieval.profile import DocumentRetrievalProfile

    profile = DocumentRetrievalProfile.production()
    active = SimpleNamespace(
        id="rdv-active",
        fingerprint="generation-fingerprint",
        role=SimpleNamespace(value="active"),
        state=SimpleNamespace(value="ready"),
    )
    catalog = SimpleNamespace(
        capability=lambda name: (True, None) if name == "sqlite_vec" else None,
        active_data_version=lambda: active,
    )
    capability = SimpleNamespace(
        diagnostic_snapshot=lambda: {"ready": True, "reason_codes": []}
    )
    composition = SimpleNamespace(
        requested_profile=profile,
        effective_profile=profile,
        capability=capability,
        degraded_reason=None,
        encoder=SimpleNamespace(fingerprint=lambda: "encoder-fingerprint"),
        reranker=SimpleNamespace(fingerprint=lambda: "reranker-fingerprint"),
        generation_spec=SimpleNamespace(
            index_recipe="rag-r1:dense1024+learned_sparse+bm25@1",
            fingerprint="generation-fingerprint",
        ),
        foundation=SimpleNamespace(catalog=catalog, method_store=object()),
    )
    monkeypatch.setattr(
        maintenance,
        "build_document_retrieval_composition",
        lambda: composition,
    )

    def readiness(**_kwargs):
        return tuple(
            SimpleNamespace(
                method=method,
                required=True,
                ready=True,
                unit_count=2,
                manifest_ready_count=2,
                representation_present_count=2,
                manifest_absent_count=0,
                invalid_unit_count=0,
            )
            for method in (
                RetrievalMethod.DENSE,
                RetrievalMethod.LEARNED_SPARSE,
                RetrievalMethod.BM25,
            )
        )

    monkeypatch.setattr(generation_rollout, "generation_readiness_snapshot", readiness)

    snapshot = retrieval_runtime_snapshot(
        preflight={"degradation_reasons": [], "status": "ready"}
    )

    assert snapshot["status"] == "ready"
    assert snapshot["active_generation"]["id"] == "rdv-active"
    assert snapshot["active_generation"]["matches_runtime"] is True
    assert [method["method"] for method in snapshot["methods"]] == [
        "dense",
        "learned_sparse",
        "bm25",
    ]
    assert all(method["required"] and method["ready"] for method in snapshot["methods"])
    assert snapshot["encoder_fingerprint"] == "encoder-fingerprint"
    assert snapshot["reranker_fingerprint"] == "reranker-fingerprint"


@pytest.mark.parametrize("preflight", [None, {}])
def test_runtime_snapshot_reuses_composition_but_rereads_current_index_state(
    monkeypatch,
    preflight,
) -> None:
    from personagraph.retrieval.operations import document_maintenance as maintenance
    from personagraph.retrieval.lifecycle import rollout as generation_rollout
    from personagraph.retrieval.profile import DocumentRetrievalProfile

    profile = DocumentRetrievalProfile.production()
    state: dict[str, object | None] = {"active": None}
    catalog = SimpleNamespace(
        capability=lambda name: (True, None) if name == "sqlite_vec" else None,
        active_data_version=lambda: state["active"],
    )
    composition = SimpleNamespace(
        requested_profile=profile,
        effective_profile=profile,
        capability=SimpleNamespace(
            diagnostic_snapshot=lambda: {"ready": True, "reason_codes": []}
        ),
        degraded_reason=None,
        encoder=SimpleNamespace(fingerprint=lambda: "encoder-fingerprint"),
        reranker=SimpleNamespace(fingerprint=lambda: "reranker-fingerprint"),
        generation_spec=SimpleNamespace(
            index_recipe="rag-r1:dense1024+learned_sparse+bm25@1",
            fingerprint="generation-fingerprint",
        ),
        foundation=SimpleNamespace(catalog=catalog, method_store=object()),
    )
    monkeypatch.setattr(
        maintenance,
        "build_document_retrieval_composition",
        lambda: (_ for _ in ()).throw(
            AssertionError("an established composition must not be rebuilt")
        ),
    )
    monkeypatch.setattr(
        retrieval_observability,
        "retrieval_preflight_snapshot",
        lambda: (_ for _ in ()).throw(
            AssertionError("an established composition must not rerun preflight")
        ),
    )
    readiness_calls: list[str] = []

    def readiness(**kwargs):
        readiness_calls.append(kwargs["generation_id"])
        return (
            SimpleNamespace(
                method=RetrievalMethod.DENSE,
                required=True,
                ready=True,
                unit_count=7,
                manifest_ready_count=7,
                representation_present_count=7,
                manifest_absent_count=0,
                invalid_unit_count=0,
            ),
        )

    monkeypatch.setattr(generation_rollout, "generation_readiness_snapshot", readiness)
    before = retrieval_runtime_snapshot(
        preflight=preflight,
        composition=composition,
    )
    state["active"] = SimpleNamespace(
        id="rdv-live",
        fingerprint="generation-fingerprint",
        role=SimpleNamespace(value="active"),
        state=SimpleNamespace(value="ready"),
    )
    after = retrieval_runtime_snapshot(
        preflight=preflight,
        composition=composition,
    )

    assert before["active_generation"] is None
    assert after["active_generation"]["id"] == "rdv-live"
    assert after["methods"][0]["unit_count"] == 7
    assert readiness_calls == ["rdv-live"]
    assert before["encoder_fingerprint"] == after["encoder_fingerprint"]


def test_runtime_without_active_generation_reports_all_methods(
    monkeypatch,
    tmp_path,
) -> None:
    from personagraph.workspace.storage import DocumentDatabase
    from personagraph.workspace.storage.context import bind

    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY", "strict")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_RERANKER", "off")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_DEVICE", "cpu")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_USE_FP16", "false")
    project_root = tmp_path / "project"
    project_root.mkdir()
    database = DocumentDatabase(
        "project-observability",
        project_root,
        tmp_path / "var/projects/project-observability/documents.sqlite",
    )

    with bind(database):
        snapshot = retrieval_runtime_snapshot()

    assert snapshot["status"] == "degraded"
    assert snapshot["active_generation"] is None
    assert snapshot["degradation_reasons"] == ["active_generation_unavailable"]
    assert [(item["method"], item["required"]) for item in snapshot["methods"]] == [
        ("dense", False),
        ("learned_sparse", False),
        ("bm25", True),
    ]
    assert all(item["ready"] is False for item in snapshot["methods"])


def test_runtime_failure_keeps_preflight_without_leaking_model_path(monkeypatch) -> None:
    from personagraph.retrieval.operations import document_maintenance as maintenance

    monkeypatch.setattr(
        maintenance,
        "build_document_retrieval_composition",
        lambda: (_ for _ in ()).throw(RuntimeError("/private/models/bge-m3")),
    )
    preflight = {
        "status": "ready",
        "requested_profile": {"mode": "bge_m3"},
        "degradation_reasons": [],
    }

    snapshot = retrieval_runtime_snapshot(preflight=preflight)

    assert snapshot["requested_profile"] == {"mode": "bge_m3"}
    assert snapshot["status"] == "unavailable"
    assert snapshot["degradation_reasons"] == [
        "retrieval_runtime_snapshot_unavailable:RuntimeError"
    ]
    assert "/private/models" not in repr(snapshot)
