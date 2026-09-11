from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.docbench.reproduce_or_run_script import (
    config as docbench_config,
    runner,
    scorer,
)


FUTURE_RUN_SCHEMA_VERSION = "personagraph-docbench-l1-run-v999"


def _case() -> dict[str, object]:
    return {
        "case_id": "docbench:1:0",
        "doc_id": 1,
        "question_index": 0,
        "domain": "academia",
        "question_type": "text-only",
        "pdf_path": "/private/tmp/source.pdf",
        "qa_path": "/private/tmp/1_qa.jsonl",
        "question": "Question?",
        "answer": "Reference",
        "evidence": "Evidence",
        "pdf_sha256": "a" * 64,
        "qa_sha256": "b" * 64,
    }


def _providers() -> dict[str, dict[str, str]]:
    return {
        "main": {
            "provider": "openai-compatible",
            "request_dialect": "deepseek",
            "base_url": "https://main.example.test/v1",
            "model": "main-model",
            "credential_source": "test",
        },
        "vision": {
            "provider": "openai-compatible",
            "request_dialect": "deepseek",
            "base_url": "https://vision.example.test/v1",
            "model": "vision-model",
            "credential_source": "test",
        },
    }


def _provenance() -> dict[str, object]:
    return {
        "code_revision": "a" * 40,
        "worktree_dirty": False,
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "initial_run_provenance_complete": True,
        "finished_code_revision": "a" * 40,
        "finished_worktree_dirty": False,
        "finished_source_sha256": "b" * 64,
        "source_fingerprint_complete": True,
        "source_changed_during_run": False,
    }


def _contracts(tmp_path: Path):
    retrieval = {
        "profile": "bge_m3",
        "failure_policy": "strict",
        "required_methods": ["dense", "learned_sparse", "bm25"],
        "encoder": {"model_id": "BAAI/bge-m3", "revision": "encoder-revision"},
        "reranker": {
            "mode": "bge_v2_m3",
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "reranker-revision",
        },
        "local_files_only": True,
        "device": "cpu",
        "use_fp16": False,
    }
    resolved = {
        "providers": {
            "main": {"source": "installation"},
            "vision": {"source": "installation"},
        },
        "retrieval": retrieval,
        "dataset": {
            "data_root": tmp_path / "data",
            "selection": tmp_path / "selection.json",
        },
        "run": {
            "runtime_features": tmp_path / "runtime.yaml",
            "output_root": tmp_path / "runs",
            "per_case_timeout_s": 30,
            "max_workers": 1,
        },
        "prompt": {"preamble": "Preamble\n"},
        "scoring": {
            "prompt": tmp_path / "evaluation_prompt.txt",
            "prompt_sha256": "d" * 64,
            "judge": {"source": "main"},
        },
    }
    case = _case()
    loaded_config = SimpleNamespace(
        resolved=resolved,
        source_path=tmp_path / "config.yaml",
        sha256="config-sha",
        canonical_snapshot={
            "schema_version": "docbench-l1-eval-v1",
            "retrieval": retrieval,
        },
    )
    loaded_selection = SimpleNamespace(
        raw={"schema_version": "docbench-formal-selection-v1"},
        sha256="selection-sha",
        cases=(case,),
    )
    return loaded_config, loaded_selection, [case]


def _result() -> dict[str, object]:
    return {
        "case_id": "docbench:1:0",
        "status": "completed",
        "processing_level": "L1",
        "reply": "Answer",
        "execution_ok": True,
        "lane_ok": True,
        "interaction_ok": True,
        "telemetry": {},
    }


def _write_run(
    root: Path,
    *,
    cases: list[dict[str, object]],
    manifest_schema: str = runner.RUN_SCHEMA_VERSION,
    report_schema: str = runner.RUN_SCHEMA_VERSION,
) -> None:
    result = _result()
    runner._atomic_write_json(
        root / "run_manifest.json",
        {
            "schema_version": manifest_schema,
            "benchmark_id": "docbench",
            "lane": "L1",
            "run_id": "fixed-run",
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "frozen_cases_sha256": runner._canonical_sha256(cases),
            "frozen_cases": cases,
            "providers": _providers(),
            "provenance": _provenance(),
        },
    )
    runner._atomic_write_json(
        root / "cases/docbench:1:0/result.json",
        result,
    )
    runner._atomic_write_json(
        root / "generation_report.json",
        {
            "schema_version": report_schema,
            "benchmark_id": "docbench",
            "lane": "L1",
            "run_id": "fixed-run",
            "status": "complete",
            "gate_passed": True,
            "baseline_eligible": True,
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "case_count": 1,
            "attempted_case_count": 1,
            "execution_ok_case_count": 1,
            "cases": [result],
            "provenance": _provenance(),
        },
    )


def _patch_offline_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[object, object, list[dict[str, object]]]:
    contracts = _contracts(tmp_path)
    monkeypatch.setattr(
        docbench_config,
        "docbench_root",
        lambda environment=None: tmp_path.resolve(),
    )
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, _providers()),
    )
    monkeypatch.setattr(
        runner.provenance,
        "read_git_provenance",
        lambda _root: ("a" * 40, False),
    )
    monkeypatch.setattr(
        runner.provenance,
        "compute_source_tree_sha256",
        lambda _root: "b" * 64,
    )
    monkeypatch.setattr(
        runner.provenance,
        "compute_environment_sha256",
        lambda _environment: "c" * 64,
    )
    monkeypatch.setattr(runner, "_retrieval_preflight_snapshot", lambda _env: {})
    return contracts


def test_resume_rejects_an_existing_run_without_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, cases = _patch_offline_runtime(monkeypatch, tmp_path)
    run_root = tmp_path / "runs/orphan-run"
    runner._atomic_write_json(
        run_root / "cases/docbench:1:0/result.json",
        _result(),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="manifest"):
        runner.run_from_config(
            tmp_path / "config.yaml",
            run_id="orphan-run",
            resume=True,
            allow_live=True,
        )

    assert cases[0]["case_id"] == "docbench:1:0"


@pytest.mark.parametrize(
    ("action", "future_artifact"),
    [
        ("resume", "manifest"),
        ("resume", "report"),
        ("retry", "manifest"),
        ("retry", "report"),
        ("score", "manifest"),
        ("score", "report"),
    ],
)
def test_future_run_artifact_schemas_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    future_artifact: str,
) -> None:
    _, _, cases = _patch_offline_runtime(monkeypatch, tmp_path)
    run_root = tmp_path / "runs/fixed-run"
    _write_run(
        run_root,
        cases=cases,
        manifest_schema=(
            FUTURE_RUN_SCHEMA_VERSION
            if future_artifact == "manifest"
            else runner.RUN_SCHEMA_VERSION
        ),
        report_schema=(
            FUTURE_RUN_SCHEMA_VERSION
            if future_artifact == "report"
            else runner.RUN_SCHEMA_VERSION
        ),
    )
    monkeypatch.setattr(
        runner,
        "_frozen_judge_provider",
        lambda _manifest, _config: {
            "provider": "openai-compatible",
            "base_url": "https://judge.example.test/v1",
            "model": "judge-model",
            "api_key": "test-key",
            "request_dialect": "deepseek",
        },
    )
    monkeypatch.setattr(
        scorer,
        "score_run",
        lambda **_kwargs: {"status": "complete", "case_count": 1},
    )

    with pytest.raises(runner.DocBenchRunnerError, match="schema"):
        if action == "resume":
            runner.run_from_config(
                tmp_path / "config.yaml",
                run_id="fixed-run",
                resume=True,
                allow_live=True,
            )
        elif action == "retry":
            runner.retry_failed_from_config(
                tmp_path / "config.yaml",
                run_dir=run_root,
                allow_live=True,
            )
        else:
            runner.score_from_config(
                tmp_path / "config.yaml",
                run_dir=run_root,
                allow_live=True,
            )
