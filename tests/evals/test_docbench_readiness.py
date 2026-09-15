"""DocBench readiness 的无网络、无运行状态写入合同。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.docbench.reproduce_or_run_script import readiness


def _ready_config(tmp_path: Path) -> SimpleNamespace:
    feature_path = tmp_path / "runtime.yaml"
    feature_path.write_text("features: {}\n", encoding="utf-8")
    prompt_path = tmp_path / "evaluation_prompt.txt"
    prompt_path.write_text("judge prompt\n", encoding="utf-8")
    prompt_sha256 = sha256(prompt_path.read_bytes()).hexdigest()
    return SimpleNamespace(
        sha256="a" * 64,
        raw={
            "schema_version": "docbench-l1-eval-v1",
            "lane": "L1",
            "scoring": {"prompt_sha256": prompt_sha256},
        },
        resolved={
            "dataset": {
                "selection": tmp_path / "selection.json",
                "data_root": tmp_path / "data",
            },
            "run": {
                "runtime_features": feature_path,
                "output_root": tmp_path / "runs",
            },
            "scoring": {
                "prompt": prompt_path,
                "judge": {"source": "main"},
            },
            "providers": {
                "main": {"source": "installation"},
                "vision": {"source": "installation"},
            },
        },
    )


def _ready_selection(tmp_path: Path) -> SimpleNamespace:
    case = SimpleNamespace(
        doc_id=104,
        pdf_path=tmp_path / "data/104/document.pdf",
        qa_path=tmp_path / "data/104/104_qa.jsonl",
        question="secret benchmark question",
        answer="secret benchmark answer",
        evidence="secret benchmark evidence",
    )
    return SimpleNamespace(sha256="b" * 64, cases=(case,))


def _ready_retrieval() -> dict[str, object]:
    return {
        "status": "ready",
        "requested_profile": {
            "fingerprint": "profile-fingerprint",
            "required_methods": ["dense", "learned_sparse", "bm25"],
            "local_files_only": True,
        },
        "encoder_fingerprint": "encoder-fingerprint",
        "reranker_fingerprint": "reranker-fingerprint",
        "preflight": {"ready": True},
        "sqlite_vec": {"ready": True},
        "degradation_reasons": [],
    }


def _patch_ready_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    config = _ready_config(tmp_path)
    selection = _ready_selection(tmp_path)
    monkeypatch.setattr(readiness, "load_docbench_config", lambda *_a, **_k: config)
    monkeypatch.setattr(
        readiness,
        "load_selection_manifest",
        lambda *_a, **_k: selection,
    )
    monkeypatch.setattr(
        readiness,
        "load_features",
        lambda _path: dict(readiness._REQUIRED_RUNTIME_FEATURES)
        | {"l1_semantic_verification_mode": "always"},
    )
    monkeypatch.setattr(
        readiness,
        "_resolve_provider_context",
        lambda _config: (
            {"PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY": "true"},
            {
                "main": {
                    "provider": "openai-compatible",
                    "base_url": "https://models.example.test/v1",
                    "model": "model",
                    "request_dialect": "deepseek-openai",
                    "credential_present": True,
                },
                "vision": {
                    "provider": "vision-compatible",
                    "base_url": "https://vision.example.test/v1",
                    "model": "vision-model",
                    "request_dialect": "auto",
                    "credential_present": True,
                },
                "judge": {
                    "provider": "openai-compatible",
                    "base_url": "https://models.example.test/v1",
                    "model": "model",
                    "request_dialect": "deepseek-openai",
                    "credential_present": True,
                    "source": "main",
                },
            },
        ),
    )
    monkeypatch.setattr(
        readiness,
        "retrieval_preflight_snapshot",
        lambda _environment: _ready_retrieval(),
    )
    return config


def test_readiness_reports_only_metadata_and_does_not_create_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _patch_ready_dependencies(monkeypatch, tmp_path)
    run_root = Path(config.resolved["run"]["output_root"])

    report = readiness.build_readiness_report((tmp_path / "l1_smoke.yaml",))

    assert report["status"] == "ready"
    assert report["network_calls_performed"] == 0
    assert report["run_directories_created"] == 0
    assert not run_root.exists()
    serialized = json.dumps(report, ensure_ascii=False)
    assert "secret benchmark question" not in serialized
    assert "secret benchmark answer" not in serialized
    assert "secret benchmark evidence" not in serialized
    assert "api_key" not in serialized.casefold()
    checks = report["configs"][0]["checks"]
    assert checks["dataset"]["source_hashes_verified"] is True
    assert checks["providers"]["main"]["credential_present"] is True
    assert checks["retrieval"]["sqlite_vec_ready"] is True


def test_readiness_aggregates_a_dataset_failure_without_running_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "load_selection_manifest",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("PDF hash drift")),
    )

    report = readiness.build_readiness_report((tmp_path / "l1_smoke.yaml",))

    assert report["status"] == "failed"
    config_report = report["configs"][0]
    assert config_report["checks"]["dataset"] == {
        "status": "failed",
        "error_code": "ValueError",
        "error_type": "ValueError",
        "message": "PDF hash drift",
    }
    assert config_report["checks"]["retrieval"]["status"] == "ready"


def test_readiness_fails_closed_when_local_retrieval_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "retrieval_preflight_snapshot",
        lambda _environment: {
            "status": "unavailable",
            "preflight": {"ready": False},
            "degradation_reasons": ["encoder_asset_missing"],
        },
    )

    report = readiness.build_readiness_report((tmp_path / "l1_smoke.yaml",))

    retrieval = report["configs"][0]["checks"]["retrieval"]
    assert report["status"] == "failed"
    assert retrieval["error_code"] == "retrieval_not_ready"
    assert "encoder_asset_missing" in retrieval["message"]


def test_readiness_rejects_an_empty_config_catalog() -> None:
    report = readiness.build_readiness_report(())

    assert report["status"] == "failed"
    assert report["config_count"] == 0
    assert report["catalog_error"]["error_code"] == "config_catalog_empty"


@pytest.mark.parametrize("mode", ["always", "conditional", "off"])
def test_runtime_readiness_accepts_and_reports_the_selected_semantic_gate_mode(
    tmp_path: Path, mode: str,
) -> None:
    config = _ready_config(tmp_path)
    features = dict(readiness._REQUIRED_RUNTIME_FEATURES)
    features["l1_semantic_verification_mode"] = mode
    Path(config.resolved["run"]["runtime_features"]).write_text(
        json.dumps({"features": features}), encoding="utf-8",
    )

    report = readiness._runtime_features_check(config)

    assert report["status"] == "ready"
    assert report["effective_policy"]["l1_semantic_verification_mode"] == mode
    assert report["effective_policy"]["l1_external_web_tools_enabled"] is False


@pytest.mark.parametrize("mode", ["disabled", False])
def test_runtime_readiness_rejects_invalid_semantic_gate_modes(
    tmp_path: Path, mode: object,
) -> None:
    config = _ready_config(tmp_path)
    features = dict(readiness._REQUIRED_RUNTIME_FEATURES)
    features["l1_semantic_verification_mode"] = mode
    Path(config.resolved["run"]["runtime_features"]).write_text(
        json.dumps({"features": features}), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic verification mode"):
        readiness._runtime_features_check(config)


def test_semantic_gate_off_does_not_relax_closed_world_readiness(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    features = dict(readiness._REQUIRED_RUNTIME_FEATURES)
    features.update(
        l1_semantic_verification_mode="off", l1_external_web_tools_enabled=True,
    )
    Path(config.resolved["run"]["runtime_features"]).write_text(
        json.dumps({"features": features}), encoding="utf-8",
    )

    with pytest.raises(readiness.DocBenchReadinessError, match="l1_external_web_tools"):
        readiness._runtime_features_check(config)


def test_public_provider_includes_only_non_secret_quota_configuration() -> None:
    public = readiness._public_provider(
        {
            "provider": "openai-compatible",
            "base_url": "https://models.example.test/v1",
            "model": "model",
            "request_dialect": "deepseek-openai",
            "api_key": "secret-key",
            "quota_scope_hash": "private-derived-identity",
            "quota": {
                "requests_per_minute": 10,
                "tokens_per_minute": 100_000,
                "tokens_per_week": 1_000_000_000,
                "max_in_flight": 2,
                "quota_group": "shared-account",
                "unexpected": "not-public",
            },
        },
        credential_present=True,
    )

    assert public == {
        "provider": "openai-compatible",
        "base_url": "https://models.example.test/v1",
        "model": "model",
        "request_dialect": "deepseek-openai",
        "credential_present": True,
        "quota": {
            "requests_per_minute": 10,
            "tokens_per_minute": 100_000,
            "tokens_per_week": 1_000_000_000,
            "max_in_flight": 2,
            "quota_group": "shared-account",
        },
        "quota_enabled": True,
    }
