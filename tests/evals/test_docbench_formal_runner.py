from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.docbench.reproduce_or_run_script import (
    cli,
    config as docbench_config,
    runner,
)
from personagraph.model_io.api_quota_controller import (
    API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
    MODEL_API_QUOTA_ENVIRONMENT_VARIABLE,
    model_profile_quota_from_environment,
)


@pytest.fixture(autouse=True)
def _isolated_docbench_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep synthetic run/workspace layouts inside each test's temp root."""

    monkeypatch.setattr(
        docbench_config,
        "docbench_root",
        lambda environment=None: tmp_path.resolve(),
        raising=False,
    )


def _case(case_id: str = "docbench:1:0") -> dict[str, object]:
    return {
        "case_id": case_id,
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


def _frozen_main_provider() -> dict[str, object]:
    return {
        "provider": "openai-compatible",
        "request_dialect": "deepseek",
        "base_url": "https://frozen.example.test/v1",
        "model": "frozen-deepseek",
        "credential_source": "installation_profile",
        "quota": {
            "requests_per_minute": 10,
            "tokens_per_minute": 300_000,
            "tokens_per_week": 1_000_000_000,
            "max_in_flight": 2,
            "quota_group": "docbench-shared-account",
        },
    }


def _contracts(tmp_path: Path):
    retrieval = {
        "profile": "bge_m3",
        "failure_policy": "strict",
        "required_methods": ["dense", "learned_sparse", "bm25"],
        "encoder": {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        },
        "reranker": {
            "mode": "bge_v2_m3",
            "model_id": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        },
        "local_files_only": True,
        "device": "cpu",
        "use_fp16": False,
    }
    config = {
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
            "prompt_sha256": "c" * 64,
            "judge": {"source": "main"},
        },
    }
    loaded_config = SimpleNamespace(
        resolved=config,
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
        cases=(_case(),),
    )
    return loaded_config, loaded_selection, [_case()]


def _contracts_with_cases(tmp_path: Path, cases: list[dict[str, object]]):
    loaded_config, loaded_selection, _ = _contracts(tmp_path)
    loaded_selection = SimpleNamespace(
        raw={"schema_version": "docbench-formal-selection-v1"},
        sha256="selection-sha",
        cases=tuple(cases),
    )
    return loaded_config, loaded_selection, cases


def _trajectory_snapshot(*steps: dict[str, object]) -> dict[str, object]:
    normalized_steps = [{"parts": [], **step} for step in steps]
    return {
        "format": "personagraph.trajectory",
        "steps": normalized_steps,
        "blobs": {},
        "integrity": {
            "step_count": len(steps),
            "part_count": 0,
            "blob_count": 0,
            "truncated_blob_count": 0,
            "recording_failure_count": sum(
                step.get("kind") == "recording_failure" for step in steps
            ),
        },
    }


def test_trajectory_summary_does_not_count_diagnostics_as_model_calls() -> None:
    metrics = runner._summarize_trajectory(
        _trajectory_snapshot(
            {
                "kind": "model_call",
                "purpose": "openai-compatible:model:runtime_l1_attempt",
                "outcome": "ok",
                "metrics": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_tokens": 3,
                },
            },
            {
                "kind": "model_call",
                "purpose": "model_output_validation:runtime_l1_attempt",
                "outcome": "rejected",
                "metrics": {},
            },
            {
                "kind": "model_call",
                "purpose": "runtime_l1_attempt",
                "outcome": "failed",
                "metrics": {"attempts": 4},
            },
        )
    )

    assert metrics["provider_response_count"] == 1
    assert metrics["rejected_output_count"] == 1
    assert metrics["terminal_model_request_failure_count"] == 1
    assert metrics["trajectory_recording_failure_count"] == 0
    assert metrics["telemetry_complete"] is True
    assert metrics["input_tokens_total"] == 10
    assert metrics["output_tokens_total"] == 2
    assert metrics["cache_read_tokens_total"] == 3
    assert metrics["provider_response_purposes"] == {
        "openai-compatible:model:runtime_l1_attempt": 1
    }


def test_trajectory_summary_requires_a_provider_response_and_valid_integrity() -> None:
    tool_only = _trajectory_snapshot(
        {
            "kind": "tool_call",
            "purpose": "search_files",
            "outcome": "ok",
            "metrics": {},
        }
    )
    mismatched = _trajectory_snapshot(
        {
            "kind": "model_call",
            "purpose": "runtime_l1_attempt",
            "outcome": "ok",
            "metrics": {},
        }
    )
    mismatched["integrity"]["step_count"] = 2
    recorder_failed = _trajectory_snapshot(
        {
            "kind": "model_call",
            "purpose": "runtime_l1_attempt",
            "outcome": "ok",
            "metrics": {},
        },
        {
            "kind": "recording_failure",
            "purpose": "record_tool_call",
            "outcome": "failed",
            "metrics": {},
        },
    )

    assert runner._summarize_trajectory(tool_only)["telemetry_complete"] is False
    assert runner._summarize_trajectory(mismatched)["telemetry_complete"] is False
    failed = runner._summarize_trajectory(recorder_failed)
    assert failed["trajectory_recording_failure_count"] == 1
    assert failed["telemetry_complete"] is False


def test_materialize_case_trajectory_reports_an_unreadable_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = tmp_path / "sessions" / "session-1" / "session.sqlite"
    database.parent.mkdir(parents=True)
    database.touch()
    from personagraph import trajectory as trajectory_api

    monkeypatch.setattr(
        trajectory_api,
        "export_trajectory",
        lambda _database, _destination: (_ for _ in ()).throw(
            RuntimeError("unreadable")
        ),
        raising=False,
    )

    reference, metrics = runner._materialize_case_trajectory(
        state_dir=tmp_path,
        run_root=tmp_path,
    )

    assert reference["status"] == "failed"
    assert reference["error_type"] == "RuntimeError"
    assert metrics["trajectory_database_count"] == 1
    assert metrics["trajectory_read_failure_count"] == 1
    assert metrics["telemetry_complete"] is False


@pytest.mark.parametrize(
    "state_locator",
    (
        "cases/docbench:1:0/state",
        "cases/docbench:1:0/attempts/attempt-002-state",
    ),
)
def test_materialize_case_trajectory_writes_the_attempt_local_artifact(
    tmp_path: Path,
    state_locator: str,
) -> None:
    from personagraph.trajectory import Step, StepKind, TrajectoryStore

    state_dir = tmp_path / state_locator
    database = state_dir / "sessions/session-2/session.sqlite"
    TrajectoryStore(database).record(
        Step(
            step_id="model-step",
            kind=StepKind.MODEL_CALL,
            occurred_at="2026-09-06T00:00:00+00:00",
            purpose="runtime_l1_attempt",
            metrics={"input_tokens": 12, "output_tokens": 3},
        )
    )
    TrajectoryStore(database).record(
        Step(
            step_id="tool-step",
            kind=StepKind.TOOL_CALL,
            occurred_at="2026-09-06T00:00:01+00:00",
            purpose="search_files",
        )
    )

    reference, metrics = runner._materialize_case_trajectory(
        state_dir=state_dir,
        run_root=tmp_path,
    )

    artifact = state_dir / "artifacts/trajectory.json"
    assert artifact.is_file()
    payload = runner._read_json(artifact)
    assert [step["step_id"] for step in payload["steps"]] == [
        "model-step",
        "tool-step",
    ]
    assert reference["status"] == "complete"
    assert reference["path"] == f"{state_locator}/artifacts/trajectory.json"
    assert len(reference["sha256"]) == 64
    assert reference["byte_count"] == artifact.stat().st_size
    assert reference["database_count"] == 1
    assert reference["step_count"] == 2
    assert reference["part_count"] == 0
    assert reference["blob_count"] == 0
    assert reference["truncated_blob_count"] == 0
    assert metrics["provider_response_count"] == 1
    assert metrics["tool_call_count"] == 1
    assert metrics["telemetry_complete"] is True


def _write_existing_run(
    root: Path,
    *,
    cases: list[dict[str, object]],
    results: list[dict[str, object]],
    config_sha256: str = "config-sha",
    selection_sha256: str = "selection-sha",
) -> bytes:
    runner._atomic_write_json(
        root / "run_manifest.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "benchmark_id": "docbench",
            "lane": "L1",
            "run_id": "fixed-run",
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
            "frozen_cases_sha256": runner._canonical_sha256(cases),
            "frozen_cases": cases,
        },
    )
    for case, result in zip(cases, results, strict=True):
        runner._atomic_write_json(
            root / "cases" / str(case["case_id"]) / "result.json",
            result,
        )
    report = {
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "benchmark_id": "docbench",
        "lane": "L1",
        "run_id": "fixed-run",
        "status": "complete",
        "config_sha256": config_sha256,
        "selection_sha256": selection_sha256,
        "case_count": len(cases),
        "attempted_case_count": len(cases),
        "execution_ok_case_count": sum(
            result.get("execution_ok") is True for result in results
        ),
        "cases": results,
    }
    runner._atomic_write_json(root / "generation_report.json", report)
    return (root / "generation_report.json").read_bytes()


def test_report_gate_rejects_an_execution_with_incomplete_trajectory(
    tmp_path: Path,
) -> None:
    case = _case()
    root = tmp_path / "runs/fixed-run"
    runner._atomic_write_json(
        root / "cases/docbench:1:0/result.json",
        {
            "case_id": case["case_id"],
            "execution_ok": True,
            "lane_ok": True,
            "interaction_ok": True,
            "telemetry": {"telemetry_complete": False},
        },
    )

    report = runner._report(
        root,
        {
            "run_id": "fixed-run",
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "frozen_cases": [case],
        },
    )

    assert report["generation_gate"]["telemetry_complete"] is False
    assert report["gate_passed"] is False


def test_installation_provider_prefers_the_active_model_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from personagraph.model_io import endpoint_profiles

    stale_config = tmp_path / "app_config.json"
    stale_config.write_text(
        json.dumps({
            "provider": "openai-compatible",
            "request_dialect": "auto",
            "base_url": "https://stale.example.test/v1",
            "model": "stale-model",
            "api_key": "stale-secret",
        }),
        encoding="utf-8",
    )
    profiles = tmp_path / "model_profiles.json"
    profiles.write_text(
        json.dumps({
            "profiles": [{
                "id": "active-main",
                "kind": "model",
                "name": "Active main",
                "provider": "openai-compatible",
                "request_dialect": "deepseek",
                "base_url": "https://active.example.test/v1",
            "model": "active-model",
            "api_key": "active-secret",
            "quota": {
                "requests_per_minute": 10,
                "tokens_per_minute": 100_000,
                "tokens_per_week": 1_000_000_000,
                "max_in_flight": 2,
                "quota_group": "active-main-account",
            },
            }],
            "active": {"model": "active-main"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "INSTALL_CONFIG", stale_config)
    monkeypatch.setattr(endpoint_profiles, "CONFIG_PATH", profiles)

    resolved = runner._resolve_provider(
        {"source": "installation"},
        kind="main",
        require_secret=True,
    )

    assert resolved == {
        "provider": "openai-compatible",
        "request_dialect": "deepseek",
        "base_url": "https://active.example.test/v1",
        "model": "active-model",
        "api_key": "active-secret",
        "credential_source": "installation_profile",
        "quota": {
            "requests_per_minute": 10,
            "tokens_per_minute": 100_000,
            "tokens_per_week": 1_000_000_000,
            "max_in_flight": 2,
            "quota_group": "active-main-account",
        },
    }


def test_provider_environment_replaces_inherited_retrieval_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_config, _, _ = _contracts(tmp_path)
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_BGE_MODEL", "/private/model")
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_BGE_IDENTITY", "stale")
    monkeypatch.setattr(
        runner,
        "_resolve_provider",
        lambda _specification, *, kind, require_secret: {
            "provider": kind,
            "request_dialect": "auto",
            "base_url": "https://example.test/v1",
            "model": f"{kind}-model",
            "api_key": "secret" if require_secret else "",
            "credential_source": "test",
        },
    )

    environment, _ = runner._provider_environment(loaded_config.resolved)

    assert environment["PERSONAGRAPH_RETRIEVAL_PROFILE"] == "bge_m3"
    assert environment["PERSONAGRAPH_RETRIEVAL_FAILURE_POLICY"] == "strict"
    assert environment["PERSONAGRAPH_RETRIEVAL_BGE_MODEL"] == "BAAI/bge-m3"
    assert environment["PERSONAGRAPH_RETRIEVAL_BGE_REVISION"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert environment["PERSONAGRAPH_RETRIEVAL_RERANKER"] == "bge_v2_m3"
    assert environment["PERSONAGRAPH_RETRIEVAL_METHODS"] == (
        "dense,learned_sparse,bm25"
    )
    assert environment["PERSONAGRAPH_RETRIEVAL_LOCAL_FILES_ONLY"] == "true"
    assert "PERSONAGRAPH_RETRIEVAL_BGE_IDENTITY" not in environment
    assert model_profile_quota_from_environment(environment) is not None
    assert environment[MODEL_API_QUOTA_ENVIRONMENT_VARIABLE]


def test_provider_environment_projects_dense_sparse_eval_method_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_config, _, _ = _contracts(tmp_path)
    loaded_config.resolved["retrieval"]["required_methods"] = [
        "dense",
        "learned_sparse",
    ]
    monkeypatch.setattr(
        runner,
        "_resolve_provider",
        lambda _specification, *, kind, require_secret: {
            "provider": kind,
            "request_dialect": "auto",
            "base_url": "https://example.test/v1",
            "model": f"{kind}-model",
            "api_key": "secret" if require_secret else "",
            "credential_source": "test",
        },
    )

    environment, _ = runner._provider_environment(loaded_config.resolved)

    assert environment["PERSONAGRAPH_RETRIEVAL_METHODS"] == (
        "dense,learned_sparse"
    )


def test_provider_environment_discards_inherited_project_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_config, _, _ = _contracts(tmp_path)
    monkeypatch.setenv(
        "PERSONAGRAPH_EVAL_WORKSPACE_DIR",
        str(tmp_path / "legacy-eval-workspaces"),
    )
    monkeypatch.setenv(
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
        str(tmp_path / "gui-projects"),
    )
    monkeypatch.setattr(
        runner,
        "_resolve_provider",
        lambda _specification, *, kind, require_secret: {
            "provider": kind,
            "request_dialect": "auto",
            "base_url": "https://example.test/v1",
            "model": f"{kind}-model",
            "api_key": "secret" if require_secret else "",
            "credential_source": "test",
        },
    )

    environment, _ = runner._provider_environment(loaded_config.resolved)

    assert "PERSONAGRAPH_EVAL_WORKSPACE_DIR" not in environment
    assert "PERSONAGRAPH_DEFAULT_PROJECTS_DIR" not in environment


def test_run_workspace_binding_uses_one_exact_allocation_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"

    environment = runner._bind_run_workspace(
        {"PATH": "/usr/bin"},
        run_root=run_root,
    )

    assert environment["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] == str(
        (tmp_path / "workspaces/fixed-run").resolve()
    )
    assert environment[API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE] == str(
        (run_root / "model_api_quota.sqlite3").resolve()
    )


def test_run_creates_isolated_artifacts_and_resume_skips_finished_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _contracts(tmp_path)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": {"model": "m"}, "vision": {"model": "v"}}),
    )
    preflight = {
        "schema_version": "docbench-retrieval-observability-v1",
        "status": "ready",
        "encoder_fingerprint": "bge-m3-test",
    }
    monkeypatch.setattr(
        runner,
        "_retrieval_preflight_snapshot",
        lambda _environment: preflight,
    )
    hashed_environments: list[dict[str, str]] = []

    def environment_sha256(environment):
        hashed_environments.append(dict(environment))
        return "environment-sha"

    monkeypatch.setattr(
        runner.provenance,
        "compute_environment_sha256",
        environment_sha256,
    )
    calls: list[str] = []

    def execute_case(**kwargs):
        case = kwargs["case"]
        path = (
            kwargs["run_root"]
            / "cases"
            / str(case["case_id"])
            / "result.json"
        )
        if kwargs["resume"] and path.is_file():
            return runner._read_json(path)
        calls.append(case["case_id"])
        result = {
            "case_id": case["case_id"],
            "doc_id": case["doc_id"],
            "question_index": case["question_index"],
            "domain": case["domain"],
            "question_type": case["question_type"],
            "status": "completed",
            "processing_level": "L1",
            "reply": "answer",
            "execution_ok": True,
            "lane_ok": True,
            "interaction_ok": True,
            "telemetry": {},
            "retrieval": preflight,
        }
        runner._atomic_write_json(path, result)
        return result

    monkeypatch.setattr(runner, "_execute_case", execute_case)
    first = runner.run_from_config(
        tmp_path / "config.yaml",
        run_id="fixed-run",
        allow_live=True,
    )
    assert first["status"] == "complete"
    assert first["execution_ok_case_count"] == 1
    assert calls == ["docbench:1:0"]
    assert hashed_environments[0]["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] == str(
        (tmp_path / "workspaces/fixed-run").resolve()
    )
    root = tmp_path / "runs/fixed-run"
    assert (root / "run_manifest.json").is_file()
    assert (root / "config.snapshot.yaml").is_file()
    assert (root / "selection.snapshot.json").is_file()
    assert (root / "generation_report.json").is_file()
    manifest = runner._read_json(root / "run_manifest.json")
    assert manifest["retrieval"] == {
        "configured": contracts[0].canonical_snapshot["retrieval"],
        "preflight": preflight,
        "runtime_observation": "generation_report.json#/retrieval/case_snapshots",
    }
    generation = runner._read_json(root / "generation_report.json")
    assert generation["retrieval"]["case_snapshots"] == [
        {"case_id": "docbench:1:0", "snapshot": preflight}
    ]

    second = runner.run_from_config(
        tmp_path / "config.yaml",
        run_id="fixed-run",
        resume=True,
        allow_live=True,
    )
    assert second["status"] == "complete"
    assert calls == ["docbench:1:0"]


def test_existing_run_refuses_implicit_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _contracts(tmp_path)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": {}, "vision": {}}),
    )
    (tmp_path / "runs/fixed-run").mkdir(parents=True)
    with pytest.raises(runner.DocBenchRunnerError, match="already exists"):
        runner.run_from_config(
            tmp_path / "config.yaml",
            run_id="fixed-run",
            allow_live=True,
        )


def test_case_worker_receives_its_state_namespace_before_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    state_dir = run_root / "cases/docbench:1:0/state"
    run_workspace = tmp_path / "workspaces/fixed-run"
    materialized: list[tuple[Path, Path]] = []

    def materialize(*, state_dir, run_root):
        materialized.append((state_dir, run_root))
        return (
            {
                "status": "complete",
                "path": "cases/docbench:1:0/state/artifacts/trajectory.json",
                "sha256": "c" * 64,
            },
            {"telemetry_complete": True},
        )

    def fake_run(command, **kwargs):
        assert "--allow-live" in command
        assert kwargs["env"]["PERSONAGRAPH_STATE_DIR"] == str(state_dir.resolve())
        assert kwargs["env"]["PERSONAGRAPH_LOCAL_CONFIG_DIR"] == str(
            (state_dir / "local_config").resolve()
        )
        assert kwargs["env"]["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] == str(
            run_workspace.resolve()
        )
        assert kwargs["env"][API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE] == str(
            (run_root / "model_api_quota.sqlite3").resolve()
        )
        output = Path(command[command.index("--output") + 1])
        runner._atomic_write_json(
            output,
            {
                "case_id": "docbench:1:0",
                "status": "completed",
                "processing_level": "L1",
                "reply": "answer",
                "execution_ok": True,
            },
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_materialize_case_trajectory", materialize)

    result = runner._execute_case(
        case=_case(),
        run_root=run_root,
        runtime_features=tmp_path / "runtime.yaml",
        preamble="Preamble\n",
        environment={
            "PATH": "/usr/bin",
            "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(run_workspace.resolve()),
            API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE: str(
                (run_root / "model_api_quota.sqlite3").resolve()
            ),
        },
        timeout_s=30,
        resume=False,
        allow_live=True,
    )

    assert result["execution_ok"] is True
    assert materialized == [(state_dir.resolve(), run_root.resolve())]
    assert result["trajectory"]["status"] == "complete"
    assert result["telemetry"] == {"telemetry_complete": True}


def test_case_timeout_still_materializes_the_partial_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    state_dir = run_root / "cases/docbench:1:0/state"
    run_workspace = tmp_path / "workspaces/fixed-run"
    calls: list[Path] = []

    def timeout(*_args, **_kwargs):
        raise runner.subprocess.TimeoutExpired(cmd=["worker"], timeout=30)

    def materialize(*, state_dir, run_root):
        calls.append(state_dir)
        return (
            {
                "status": "complete",
                "path": "cases/docbench:1:0/state/artifacts/trajectory.json",
                "sha256": "d" * 64,
            },
            {"telemetry_complete": True},
        )

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    monkeypatch.setattr(runner, "_materialize_case_trajectory", materialize)

    result = runner._execute_case(
        case=_case(),
        run_root=run_root,
        runtime_features=tmp_path / "runtime.yaml",
        preamble="Preamble\n",
        environment={
            "PATH": "/usr/bin",
            "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(run_workspace.resolve()),
        },
        timeout_s=30,
        resume=False,
        allow_live=True,
    )

    assert result["exception_code"] == "worker_timeout"
    assert result["trajectory"]["status"] == "complete"
    assert calls == [state_dir.resolve()]


def test_case_timeout_preserves_answer_committed_before_post_commit_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    output = run_root / "cases/docbench:1:0/result.json"

    def timeout(_command, **kwargs):
        assert kwargs["timeout"] == 30
        runner._atomic_write_json(output, {
            "case_id": "docbench:1:0", "status": "completed",
            "processing_level": "L1", "reply": "committed answer",
            "execution_ok": True, "answer_elapsed_s": 4.0,
            "post_commit_complete": False, "chain_complete": False,
            "post_commit": {"status": "pending", "jobs": [], "timed_out": False},
        })
        raise runner.subprocess.TimeoutExpired(cmd=["worker"], timeout=30)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    monkeypatch.setattr(runner, "_materialize_case_trajectory", lambda **_: ({}, {}))
    result = runner._execute_case(
        case=_case(), run_root=run_root, runtime_features=tmp_path / "runtime.yaml",
        preamble="", environment={"PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(
            tmp_path / "workspaces/fixed-run",
        )}, timeout_s=30, resume=False, allow_live=True,
    )

    assert result["reply"] == "committed answer"
    assert result["execution_ok"] is True
    assert result["answer_elapsed_s"] == 4.0
    assert result["post_commit_complete"] is False
    assert result["post_commit"]["timed_out"] is True
    assert result["chain_complete"] is False
    assert result["exception_code"] == "worker_timeout"


def test_case_abnormal_exit_preserves_settled_answer_but_not_chain_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    output = run_root / "cases/docbench:1:0/result.json"

    def crash(_command, **_kwargs):
        runner._atomic_write_json(output, {
            "case_id": "docbench:1:0", "reply": "committed answer",
            "execution_ok": True, "post_commit_complete": True, "chain_complete": True,
        })
        return SimpleNamespace(returncode=-11, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", crash)
    monkeypatch.setattr(runner, "_materialize_case_trajectory", lambda **_: ({}, {}))
    result = runner._execute_case(
        case=_case(), run_root=run_root, runtime_features=tmp_path / "runtime.yaml",
        preamble="", environment={"PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(
            tmp_path / "workspaces/fixed-run",
        )}, timeout_s=30, resume=False, allow_live=True,
    )

    assert result["reply"] == "committed answer"
    assert result["execution_ok"] is True
    assert result["post_commit_complete"] is True
    assert result["worker_exit_code"] == -11
    assert result["chain_complete"] is False


def test_run_workspace_is_shared_by_initial_and_retry_case_state(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    initial_state = run_root / "cases/docbench:1:0/state"
    retry_state = run_root / "cases/docbench:1:0/attempts/attempt-002-state"
    environment = {
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(
            tmp_path / "workspaces/fixed-run"
        )
    }

    assert runner._require_worker_project_root(
        state_dir=initial_state,
        environment=environment,
    ) == (tmp_path / "workspaces/fixed-run").resolve()
    assert runner._require_worker_project_root(
        state_dir=retry_state,
        environment=environment,
    ) == (tmp_path / "workspaces/fixed-run").resolve()


def test_run_workspace_rejects_run_outside_docbench_runs(tmp_path: Path) -> None:
    with pytest.raises(runner.DocBenchRunnerError, match="runs directory"):
        runner._run_workspace_root(tmp_path / "other/fixed-run", environment={})


@pytest.mark.parametrize("action", ("retry", "score"))
def test_existing_run_actions_reject_a_run_outside_docbench_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
) -> None:
    contracts = _contracts(tmp_path)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)

    with pytest.raises(runner.DocBenchRunnerError, match="runs directory"):
        if action == "retry":
            runner.retry_failed_from_config(
                tmp_path / "config.yaml",
                run_dir=tmp_path / "other/fixed-run",
                allow_live=True,
            )
        else:
            runner.score_from_config(
                tmp_path / "config.yaml",
                run_dir=tmp_path / "other/fixed-run",
                allow_live=True,
            )


def test_worker_project_root_requires_exact_run_workspace(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "runs/fixed-run/cases/case/state"

    with pytest.raises(runner.DocBenchRunnerError, match="parent runner"):
        runner._require_worker_project_root(state_dir=state_dir, environment={})

    with pytest.raises(runner.DocBenchRunnerError, match="exact run workspace"):
        runner._require_worker_project_root(
            state_dir=state_dir,
            environment={
                "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(
                    tmp_path / "workspaces/another-run"
                ),
            },
        )


def test_session_working_dir_locator_rejects_a_sibling_workspace(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.DocBenchRunnerError, match="outside"):
        runner._session_working_dir_locator(
            {
                "working_dir": str(
                    tmp_path / "workspaces/another-run/2026-09-05_session"
                )
            },
            run_workspace=tmp_path / "workspaces/fixed-run",
            environment={},
        )


@pytest.mark.parametrize(
    "stale_name",
    (
        "STATE_DIR",
        "LOCAL_CONFIG_DIR",
        "DEFAULT_SESSION_PROJECTS_DIR",
    ),
)
def test_worker_namespace_rejects_import_time_path_drift_before_state_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stale_name: str,
) -> None:
    from personagraph.configuration import paths

    state_dir = tmp_path / "runs/run/cases/case/state"
    project_root = tmp_path / "workspaces/run"
    expected = {
        "STATE_DIR": state_dir.resolve(),
        "LOCAL_CONFIG_DIR": (state_dir / "local_config").resolve(),
        "DEFAULT_SESSION_PROJECTS_DIR": project_root.resolve(),
    }
    for name, value in expected.items():
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(
        paths,
        stale_name,
        (tmp_path / f"stale-{stale_name}").resolve(),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="not frozen"):
        runner._require_worker_namespace(
            state_dir=state_dir,
            environment={
                "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(project_root),
            },
        )

    assert not state_dir.exists()


def test_worker_namespace_accepts_the_exact_parent_owned_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from personagraph.configuration import paths

    state_dir = tmp_path / "runs/run/cases/case/state"
    project_root = tmp_path / "workspaces/run"
    monkeypatch.setattr(paths, "STATE_DIR", state_dir.resolve())
    monkeypatch.setattr(
        paths,
        "LOCAL_CONFIG_DIR",
        (state_dir / "local_config").resolve(),
    )
    monkeypatch.setattr(
        paths,
        "DEFAULT_SESSION_PROJECTS_DIR",
        project_root.resolve(),
    )

    assert runner._require_worker_namespace(
        state_dir=state_dir,
        environment={
            "PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(project_root),
            API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE: str(
                (tmp_path / "runs/run/model_api_quota.sqlite3").resolve()
            ),
        },
    ) == project_root.resolve()


def test_worker_resolves_canonical_lifecycle_import_and_uses_managed_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from personagraph.api.service import sessions
    from personagraph.configuration import app_settings, features, paths
    from personagraph.session import store as session_store

    case_input = tmp_path / "case.json"
    output = tmp_path / "result.json"
    source = tmp_path / "source.pdf"
    payload = _case()
    payload["pdf_path"] = str(source)
    payload["preamble"] = "Preamble\n"
    runner._atomic_write_json(case_input, payload)
    created_payloads: list[dict[str, object]] = []

    worker_state_dir = tmp_path / "runs/run/cases/case/state"
    worker_project_root = tmp_path / "workspaces/run"
    session_working_dir = worker_project_root / "2026-09-05_docbench"
    monkeypatch.setenv("PERSONAGRAPH_STATE_DIR", str(worker_state_dir))
    monkeypatch.setenv(
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
        str(worker_project_root),
    )
    monkeypatch.setenv(
        API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
        str((tmp_path / "runs/run/model_api_quota.sqlite3").resolve()),
    )
    monkeypatch.setattr(paths, "STATE_DIR", worker_state_dir.resolve())
    monkeypatch.setattr(
        paths,
        "LOCAL_CONFIG_DIR",
        (worker_state_dir / "local_config").resolve(),
    )
    monkeypatch.setattr(
        paths,
        "DEFAULT_SESSION_PROJECTS_DIR",
        worker_project_root.resolve(),
    )
    monkeypatch.setattr(runner, "_retrieval_preflight_snapshot", lambda _env: {})
    monkeypatch.setattr(runner, "_sha256_file", lambda _path: "different-sha")
    monkeypatch.setattr(paths, "load_dotenv", lambda: None)
    monkeypatch.setattr(features, "load_features", lambda _path: {})
    monkeypatch.setattr(app_settings, "active_provider", lambda: "openai-compatible")
    monkeypatch.setattr(
        sessions,
        "create_session",
        lambda request: (
            created_payloads.append(dict(request))
            or {"session": {"id": "eval-session"}}
        ),
    )
    monkeypatch.setattr(
        session_store,
        "get_session",
        lambda _session_id: {
            "project_id": "eval-project",
            "working_dir": str(session_working_dir),
        },
    )
    monkeypatch.setattr(
        session_store,
        "session_database_scope",
        lambda _session_id: nullcontext(tmp_path / "session.sqlite"),
    )

    exit_code = runner._run_worker(
        case_input=case_input,
        state_dir=worker_state_dir,
        output=output,
        runtime_features=tmp_path / "runtime.yaml",
        allow_live=True,
    )

    assert exit_code == 1
    assert created_payloads == [
        {"title": "docbench-formal-l1-docbench:1:0"}
    ]
    result = runner._read_json(output)
    assert result["session_working_dir_locator"] == (
        "workspaces/run/2026-09-05_docbench"
    )
    assert result["exception_type"] == "DocBenchRunnerError"
    assert "PDF drifted before execution" in result["exception_message"]


@pytest.mark.parametrize(
    ("turn_result_changes", "expected_execution_ok"),
    [
        pytest.param({}, True, id="answer"),
        pytest.param(
            {
                "reply": "模型未能返回可用结果。",
                "error_code": "MODEL_OUTPUT_INVALID",
                "end_reason": "l1_terminal_notification",
            },
            False,
            id="terminal-failure-notification",
        ),
        pytest.param({"error_code": "INTERNAL_FAILURE"}, False, id="runtime-error"),
        pytest.param(
            {"end_reason": "l1_terminal_notification"}, False, id="terminal-reason",
        ),
        pytest.param({"status": "incomplete"}, False, id="incomplete"),
        pytest.param({"reply": " \n"}, False, id="empty-reply"),
        pytest.param(
            {"reply": "The document does not contain this information."},
            True,
            id="unanswerable-is-for-judge",
        ),
        pytest.param({"processing_level": "L0"}, True, id="lane-is-separate"),
    ],
)
@pytest.mark.parametrize("settlement_status", ["settled", "pending", "failed", "raises"])
def test_worker_reuses_retrieval_composition_and_classifies_turn_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    turn_result_changes: dict[str, object],
    expected_execution_ok: bool,
    settlement_status: str,
) -> None:
    from personagraph.api.service import attachments, sessions
    from personagraph.configuration import app_settings, features, paths
    from personagraph.retrieval.operations import document_maintenance
    from personagraph.runtime.post_commit import scheduler
    from personagraph.session import store as session_store
    from personagraph.tools.visual import publication_recovery
    from personagraph.workspace.ingestion import composition as ingestion_composition
    from personagraph.workspace.storage import context as project_context

    case_input = tmp_path / "case.json"
    output = tmp_path / "result.json"
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    payload = _case()
    payload.update({"pdf_path": str(source), "preamble": "Preamble\n"})
    runner._atomic_write_json(case_input, payload)

    state_dir = tmp_path / "runs/run/cases/case/state"
    project_root = tmp_path / "workspaces/run"
    working_dir = project_root / "session-workspace"
    monkeypatch.setenv("PERSONAGRAPH_STATE_DIR", str(state_dir))
    monkeypatch.setenv("PERSONAGRAPH_DEFAULT_PROJECTS_DIR", str(project_root))
    monkeypatch.setenv(
        API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
        str((tmp_path / "runs/run/model_api_quota.sqlite3").resolve()),
    )
    monkeypatch.setattr(paths, "STATE_DIR", state_dir.resolve())
    monkeypatch.setattr(paths, "LOCAL_CONFIG_DIR", (state_dir / "local_config").resolve())
    monkeypatch.setattr(paths, "DEFAULT_SESSION_PROJECTS_DIR", project_root.resolve())
    monkeypatch.setattr(paths, "load_dotenv", lambda: None)
    monkeypatch.setattr(features, "load_features", lambda _path: {})
    monkeypatch.setattr(app_settings, "active_provider", lambda: "openai-compatible")
    monkeypatch.setattr(app_settings, "redacted_view", lambda: {"provider": "test"})
    monkeypatch.setattr(runner, "_sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        runner,
        "_retrieval_preflight_snapshot",
        lambda _environment: {"status": "ready", "degradation_reasons": []},
    )
    monkeypatch.setattr(
        sessions,
        "create_session",
        lambda _request: {"session": {"id": "eval-session"}},
    )
    monkeypatch.setattr(
        session_store,
        "get_session",
        lambda _session_id: {
            "project_id": "eval-project",
            "working_dir": str(working_dir),
        },
    )
    monkeypatch.setattr(
        session_store,
        "session_database_scope",
        lambda _session_id: nullcontext(tmp_path / "session.sqlite"),
    )
    monkeypatch.setattr(
        session_store,
        "list_pending_user_questions",
        lambda *, session_id: [] if session_id == "eval-session" else None,
    )
    monkeypatch.setattr(
        attachments,
        "upload_attachment",
        lambda *_args, **_kwargs: {"attachment": {"attachment_id": "attachment-1"}},
    )
    monkeypatch.setattr(
        sessions,
        "chat_turn",
        lambda _request: {
            "result": {
                "turn_id": "turn-1",
                "status": "completed",
                "processing_level": "L1",
                "reply": "answer",
                **turn_result_changes,
            }
        },
    )

    established = SimpleNamespace(
        requested_profile=object(),
        encoder=object(),
        reranker=object(),
    )
    monkeypatch.setattr(
        document_maintenance,
        "build_document_retrieval_composition",
        lambda: established,
    )
    lifecycle_arguments: list[dict[str, object]] = []
    lifecycle_events: list[str] = []
    project_database = object()

    def recovery_callback():
        return 0

    recovery_bindings = []
    monkeypatch.setattr(project_context, "current", lambda: project_database)
    monkeypatch.setattr(
        publication_recovery,
        "build_visual_publication_recovery",
        lambda database: recovery_bindings.append(database) or recovery_callback,
    )

    class Lifecycle:
        def start(self) -> None:
            lifecycle_events.append("start")

        def stop(self, *, timeout_seconds: float) -> bool:
            lifecycle_events.append(f"stop:{timeout_seconds}")
            return True

    monkeypatch.setattr(
        ingestion_composition,
        "build_document_maintenance_lifecycle",
        lambda **kwargs: lifecycle_arguments.append(dict(kwargs)) or Lifecycle(),
    )
    observed_compositions: list[object] = []

    def observe(_preflight, *, composition=None):
        observed_compositions.append(composition)
        return {"observation": len(observed_compositions)}

    monkeypatch.setattr(runner, "_retrieval_runtime_snapshot", observe)

    def settle(*, session_id, turn_id, store, timeout_seconds):
        # Answer must be durable before waiting; the project services are still alive.
        checkpoint = runner._read_json(output)
        assert checkpoint["reply"] == turn_result_changes.get("reply", "answer")
        assert checkpoint["execution_ok"] is expected_execution_ok
        assert checkpoint["post_commit_complete"] is False
        assert lifecycle_events == ["start"]
        assert (session_id, turn_id, store, timeout_seconds) == (
            "eval-session", "turn-1", session_store, 120.0,
        )
        lifecycle_events.append("settle")
        if settlement_status == "raises":
            raise RuntimeError("settlement service unavailable")
        return SimpleNamespace(to_dict=lambda: {
            "status": settlement_status,
            "turn_id": turn_id,
            "window_released": settlement_status == "settled",
            "jobs": [{"job_kind": "session_history_index",
                      "status": "applied" if settlement_status == "settled" else settlement_status,
                      "reason_code": None}],
            "timed_out": settlement_status == "pending",
        })

    monkeypatch.setattr(scheduler, "wait_for_turn_post_commit_jobs", settle, raising=False)

    exit_code = runner._run_worker(
        case_input=case_input,
        state_dir=state_dir,
        output=output,
        runtime_features=tmp_path / "runtime.yaml",
        allow_live=True,
    )

    assert exit_code == (0 if expected_execution_ok and settlement_status == "settled" else 1)
    result = runner._read_json(output)
    assert result["execution_ok"] is expected_execution_ok
    assert result["lane_ok"] is (result["processing_level"] == "L1")
    assert result["error_code"] == turn_result_changes.get("error_code")
    assert result["end_reason"] == turn_result_changes.get("end_reason")
    assert observed_compositions == [established, established]
    assert lifecycle_arguments == [
        {
            "profile": established.requested_profile,
            "encoder": established.encoder,
            "reranker": established.reranker,
            "after_pass": recovery_callback,
        }
    ]
    assert recovery_bindings == [project_database]
    assert lifecycle_events == ["start", "settle", "stop:60.0"]
    assert result["post_commit_complete"] is (settlement_status == "settled")
    assert result["chain_complete"] is (expected_execution_ok and settlement_status == "settled")
    assert result["post_commit_elapsed_s"] >= 0
    if settlement_status == "raises":
        assert result["post_commit"]["status"] == "unavailable"
    assert runner._read_json(output)["retrieval"] == {"observation": 2}

    run_root = tmp_path / "runs/run"
    _write_existing_run(run_root, cases=[_case()], results=[result])
    report = runner._report(run_root, runner._read_json(run_root / "run_manifest.json"))
    assert report["execution_ok_case_count"] == int(expected_execution_ok)
    assert report["generation_gate"]["execution_ok"] is expected_execution_ok


def test_case_worker_rejects_state_outside_run_namespace(tmp_path: Path) -> None:
    with pytest.raises(runner.DocBenchRunnerError, match="own case directory"):
        runner._execute_case(
            case=_case(),
            run_root=tmp_path / "runs/fixed-run",
            runtime_features=tmp_path / "runtime.yaml",
            preamble="Preamble\n",
            environment={},
            timeout_s=30,
            resume=False,
            allow_live=True,
            state_dir=tmp_path / "gui-state",
        )


def test_case_worker_rejects_sibling_case_state_and_external_output(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"
    environment = runner._bind_run_workspace({}, run_root=run_root)

    with pytest.raises(runner.DocBenchRunnerError, match="own case directory"):
        runner._execute_case(
            case=_case(),
            run_root=run_root,
            runtime_features=tmp_path / "runtime.yaml",
            preamble="Preamble\n",
            environment=environment,
            timeout_s=30,
            resume=False,
            allow_live=True,
            state_dir=run_root / "cases/docbench:2:0/state",
        )

    with pytest.raises(runner.DocBenchRunnerError, match="case output"):
        runner._execute_case(
            case=_case(),
            run_root=run_root,
            runtime_features=tmp_path / "runtime.yaml",
            preamble="Preamble\n",
            environment=environment,
            timeout_s=30,
            resume=False,
            allow_live=True,
            output_path=tmp_path / "outside-result.json",
        )


def test_worker_requires_live_authorization_before_reading_case(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.DocBenchRunnerError, match="allow_live=True"):
        runner._run_worker(
            case_input=tmp_path / "missing-case.json",
            state_dir=tmp_path / "state",
            output=tmp_path / "result.json",
            runtime_features=tmp_path / "runtime.yaml",
        )


def test_case_execution_requires_live_authorization_before_writing_input(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/fixed-run"

    with pytest.raises(runner.DocBenchRunnerError, match="allow_live=True"):
        runner._execute_case(
            case=_case(),
            run_root=run_root,
            runtime_features=tmp_path / "runtime.yaml",
            preamble="Preamble\n",
            environment={},
            timeout_s=30,
            resume=False,
        )

    assert not run_root.exists()


@pytest.mark.parametrize(
    "failure_fields",
    [
        pytest.param(
            {
                "status": "incomplete",
                "error_code": "MODEL_TRANSPORT_FAILURE",
                "reply": "",
            },
            id="incomplete-turn",
        ),
        pytest.param(
            {
                "status": "completed",
                "error_code": "MODEL_OUTPUT_INVALID",
                "end_reason": "l1_terminal_notification",
                "reply": "模型未能返回可用结果。",
            },
            id="committed-terminal-notification",
        ),
    ],
)
def test_retry_failed_archives_only_failure_and_preserves_initial_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_fields: dict[str, object],
) -> None:
    cases = [_case("docbench:1:0"), _case("docbench:2:0")]
    cases[1]["doc_id"] = 2
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": {"model": "m"}, "vision": {"model": "v"}}),
    )
    success = {
        "case_id": "docbench:1:0",
        "execution_ok": True,
        "status": "completed",
        "reply": "already good",
    }
    failed = {
        "case_id": "docbench:2:0",
        "execution_ok": False,
        **failure_fields,
    }
    root = tmp_path / "runs/fixed-run"
    initial_report_bytes = _write_existing_run(
        root,
        cases=cases,
        results=[success, failed],
    )
    failed_result_bytes = (root / "cases/docbench:2:0/result.json").read_bytes()
    calls: list[str] = []

    def execute_case(**kwargs):
        case = kwargs["case"]
        calls.append(str(case["case_id"]))
        assert kwargs["resume"] is False
        assert kwargs["environment"]["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] == str(
            (tmp_path / "workspaces/fixed-run").resolve()
        )
        assert kwargs["state_dir"] == (
            root / "cases/docbench:2:0/attempts/attempt-002-state"
        )
        assert kwargs["output_path"].name.startswith(".retry-")
        assert (
            root / "cases/docbench:2:0/attempts/attempt-001.json"
        ).read_bytes() == failed_result_bytes
        return {
            "case_id": case["case_id"],
            "execution_ok": True,
            "status": "completed",
            "processing_level": "L1",
            "reply": "recovered",
            "lane_ok": True,
            "interaction_ok": True,
            "telemetry": {},
        }

    monkeypatch.setattr(runner, "_execute_case", execute_case)

    outcome = runner.retry_failed_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        max_workers=1,
        allow_live=True,
    )

    assert calls == ["docbench:2:0"]
    assert (root / "cases/docbench:1:0/result.json").read_text(
        encoding="utf-8"
    ) == json.dumps(success, ensure_ascii=False, indent=2) + "\n"
    assert (
        root / "cases/docbench:2:0/attempts/attempt-001.json"
    ).read_bytes() == failed_result_bytes
    assert (root / "generation_report.initial.json").read_bytes() == initial_report_bytes
    assert runner._read_json(root / "cases/docbench:2:0/result.json")[
        "execution_ok"
    ] is True
    assert runner._read_json(root / "generation_report.json")[
        "execution_ok_case_count"
    ] == 2
    retry_report = runner._read_json(root / "retry_report.json")
    assert retry_report["initial_failed_case_count"] == 1
    assert retry_report["invocations"][0]["selected_case_count"] == 1
    assert retry_report["invocations"][0]["error_code_filter"] is None
    assert retry_report["invocations"][0]["recovered_case_count"] == 1
    assert retry_report["invocations"][0]["attempts"][0][
        "archived_attempt_number"
    ] == 1
    assert outcome["status"] == "complete"
    assert outcome["recovered_case_count"] == 1
    assert outcome["remaining_failed_case_count"] == 0


def test_retry_failed_can_filter_error_codes_without_rerolling_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [
        _case("docbench:1:0"),
        _case("docbench:2:0"),
        _case("docbench:3:0"),
    ]
    for doc_id, case in enumerate(cases, start=1):
        case["doc_id"] = doc_id
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": {}, "vision": {}}),
    )
    results = [
        {"case_id": "docbench:1:0", "execution_ok": True},
        {
            "case_id": "docbench:2:0",
            "execution_ok": False,
            "error_code": "MODEL_TRANSPORT_FAILURE",
        },
        {
            "case_id": "docbench:3:0",
            "execution_ok": False,
            "error_code": "VERIFICATION_FAILED",
        },
    ]
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(root, cases=cases, results=results)
    verification_result = root / "cases/docbench:3:0/result.json"
    verification_result_bytes = verification_result.read_bytes()
    calls: list[str] = []

    def execute_case(**kwargs):
        case_id = str(kwargs["case"]["case_id"])
        calls.append(case_id)
        return {
            "case_id": case_id,
            "execution_ok": True,
            "status": "completed",
            "processing_level": "L1",
            "reply": "recovered",
            "lane_ok": True,
            "interaction_ok": True,
            "telemetry": {},
        }

    monkeypatch.setattr(runner, "_execute_case", execute_case)

    outcome = runner.retry_failed_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        error_codes=("MODEL_TRANSPORT_FAILURE",),
        allow_live=True,
    )

    assert calls == ["docbench:2:0"]
    assert verification_result.read_bytes() == verification_result_bytes
    assert not (root / "cases/docbench:3:0/attempts").exists()
    invocation = runner._read_json(root / "retry_report.json")["invocations"][0]
    assert invocation["error_code_filter"] == ["MODEL_TRANSPORT_FAILURE"]
    assert invocation["selected_case_ids"] == ["docbench:2:0"]
    assert outcome["selected_case_count"] == 1
    assert outcome["recovered_case_count"] == 1
    assert outcome["remaining_failed_case_count"] == 1


@pytest.mark.parametrize("error_codes", [(), ("",), ("  \t",)])
def test_retry_failed_rejects_empty_error_code_filters_before_loading_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_codes: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_contracts",
        lambda _path: pytest.fail("invalid filter must fail before loading contracts"),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="error code filter"):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=tmp_path / "run",
            error_codes=error_codes,
            allow_live=True,
        )


@pytest.mark.parametrize(
    ("config_sha256", "selection_sha256", "match"),
    [
        ("different", "selection-sha", "config differs"),
        ("config-sha", "different", "selection differs"),
    ],
)
def test_retry_failed_refuses_frozen_contract_drift_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_sha256: str,
    selection_sha256: str,
    match: str,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(
        root,
        cases=cases,
        results=[{"case_id": cases[0]["case_id"], "execution_ok": False}],
        config_sha256=config_sha256,
        selection_sha256=selection_sha256,
    )
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: pytest.fail("provider must not be resolved after contract drift"),
    )

    with pytest.raises(runner.DocBenchRunnerError, match=match):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )

    assert not (root / "generation_report.initial.json").exists()
    assert not (root / "retry_report.json").exists()


def test_retry_failed_refuses_source_drift_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(
        root,
        cases=cases,
        results=[{"case_id": cases[0]["case_id"], "execution_ok": False}],
    )
    manifest = runner._read_json(root / "run_manifest.json")
    manifest["provenance"] = {
        "code_revision": "a" * 40,
        "worktree_dirty": False,
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "initial_run_provenance_complete": True,
    }
    runner._atomic_write_json(root / "run_manifest.json", manifest)
    monkeypatch.setattr(
        runner.provenance,
        "read_git_provenance",
        lambda _root: ("d" * 40, False),
    )
    monkeypatch.setattr(
        runner.provenance,
        "compute_source_tree_sha256",
        lambda _root: "e" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: pytest.fail("source drift must fail before provider access"),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="source_sha256"):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )

    assert not (root / "generation_report.initial.json").exists()
    assert not (root / "retry_report.json").exists()


def test_retry_failed_refuses_environment_drift_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(
        root,
        cases=cases,
        results=[{"case_id": cases[0]["case_id"], "execution_ok": False}],
    )
    manifest = runner._read_json(root / "run_manifest.json")
    manifest["provenance"] = {
        "code_revision": "a" * 40,
        "worktree_dirty": False,
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "initial_run_provenance_complete": True,
    }
    runner._atomic_write_json(root / "run_manifest.json", manifest)
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
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": _frozen_main_provider()}),
    )

    def environment_sha256(environment):
        assert environment["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"] == str(
            (tmp_path / "workspaces/fixed-run").resolve()
        )
        return "d" * 64

    monkeypatch.setattr(
        runner.provenance,
        "compute_environment_sha256",
        environment_sha256,
    )

    with pytest.raises(runner.DocBenchRunnerError, match="environment_sha256"):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )

    assert not (root / "generation_report.initial.json").exists()
    assert not (root / "retry_report.json").exists()


def test_retry_failed_refuses_provider_identity_drift_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(
        root,
        cases=cases,
        results=[{"case_id": cases[0]["case_id"], "execution_ok": False}],
    )
    manifest = runner._read_json(root / "run_manifest.json")
    manifest["providers"] = {"main": _frozen_main_provider()}
    manifest["provenance"] = {
        "code_revision": "a" * 40,
        "worktree_dirty": False,
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "initial_run_provenance_complete": True,
    }
    runner._atomic_write_json(root / "run_manifest.json", manifest)
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
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: (
            {},
            {"main": {**_frozen_main_provider(), "model": "different-model"}},
        ),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="provider identities"):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )

    assert not (root / "generation_report.initial.json").exists()
    assert not (root / "retry_report.json").exists()


def test_retry_failed_repeated_invocation_increments_attempt_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: ({}, {"main": {}, "vision": {}}),
    )
    first_failure = {
        "case_id": cases[0]["case_id"],
        "execution_ok": False,
        "error_code": "MODEL_TRANSPORT_FAILURE",
        "reply": "",
    }
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(root, cases=cases, results=[first_failure])
    returned = iter(
        [
            {
                "case_id": cases[0]["case_id"],
                "execution_ok": False,
                "error_code": "MODEL_TRANSPORT_FAILURE",
                "reply": "",
            },
            {
                "case_id": cases[0]["case_id"],
                "execution_ok": True,
                "status": "completed",
                "processing_level": "L1",
                "reply": "recovered",
                "lane_ok": True,
                "interaction_ok": True,
                "telemetry": {},
            },
        ]
    )
    monkeypatch.setattr(runner, "_execute_case", lambda **_kwargs: next(returned))

    first = runner.retry_failed_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        allow_live=True,
    )
    second_failure_bytes = (root / "cases/docbench:1:0/result.json").read_bytes()
    second = runner.retry_failed_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        allow_live=True,
    )

    attempts = root / "cases/docbench:1:0/attempts"
    assert runner._read_json(attempts / "attempt-001.json") == first_failure
    assert (attempts / "attempt-002.json").read_bytes() == second_failure_bytes
    retry_report = runner._read_json(root / "retry_report.json")
    assert len(retry_report["invocations"]) == 2
    assert first["remaining_failed_case_count"] == 1
    assert second["remaining_failed_case_count"] == 0


def test_retry_failed_noops_without_resolving_provider_when_all_cases_passed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [_case()]
    contracts = _contracts_with_cases(tmp_path, cases)
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    root = tmp_path / "runs/fixed-run"
    _write_existing_run(
        root,
        cases=cases,
        results=[{"case_id": cases[0]["case_id"], "execution_ok": True}],
    )
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: pytest.fail("no failed cases means no provider access"),
    )

    outcome = runner.retry_failed_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        allow_live=True,
    )

    assert outcome["status"] == "complete"
    assert outcome["selected_case_count"] == 0
    assert outcome["remaining_failed_case_count"] == 0
    assert (root / "generation_report.initial.json").is_file()
    assert (root / "retry_report.json").is_file()


def test_score_propagates_partial_judge_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _contracts(tmp_path)
    _, _, cases = contracts
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_judge_provider",
        lambda _config: {
            **_frozen_main_provider(),
            "api_key": "secret",
        },
    )
    root = tmp_path / "runs/run"
    runner._atomic_write_json(
        root / "run_manifest.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "run_id": "run",
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "frozen_cases_sha256": runner._canonical_sha256(cases),
            "providers": {"main": _frozen_main_provider()},
        },
    )
    runner._atomic_write_json(
        root / "generation_report.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "status": "complete",
            "cases": [{"case_id": "docbench:1:0", "reply": "x"}],
        },
    )
    import evals.docbench.reproduce_or_run_script.scorer as scorer

    monkeypatch.setattr(
        scorer,
        "score_run",
        lambda **_kwargs: {
            "status": "partial",
            "score": None,
            "correct_count": 0,
            "case_count": 1,
        },
    )
    outcome = runner.score_from_config(
        tmp_path / "config.yaml",
        run_dir=root,
        allow_live=True,
    )
    assert outcome["status"] == "partial"
    assert outcome["official_comparable"] is False


def test_score_fails_closed_when_current_judge_identity_differs_from_frozen_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _contracts(tmp_path)
    _, _, cases = contracts
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_judge_provider",
        lambda _config: {
            "provider": "openai-compatible",
            "request_dialect": "deepseek",
            "base_url": "https://current.example.test/v1",
            "model": "different-deepseek",
            "api_key": "current-secret",
            "credential_source": "installation_profile",
        },
    )
    root = tmp_path / "runs/run"
    runner._atomic_write_json(
        root / "run_manifest.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "run_id": "run",
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "frozen_cases_sha256": runner._canonical_sha256(cases),
            "providers": {"main": _frozen_main_provider()},
        },
    )
    runner._atomic_write_json(
        root / "generation_report.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "status": "complete",
            "cases": [{"case_id": "docbench:1:0", "reply": "x"}],
        },
    )

    with pytest.raises(
        runner.DocBenchRunnerError,
        match="current judge provider identity differs from the frozen run provider",
    ):
        runner.score_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )


def test_score_fails_closed_when_current_quota_differs_from_frozen_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen = _frozen_main_provider()
    current = dict(frozen)
    current["quota"] = dict(frozen["quota"], tokens_per_minute=100_000)
    current["api_key"] = "current-secret"
    monkeypatch.setattr(runner, "_judge_provider", lambda _config: current)

    with pytest.raises(
        runner.DocBenchRunnerError,
        match="current judge quota policy differs",
    ):
        runner._frozen_judge_provider(
            {"providers": {"main": frozen}},
            {"scoring": {"judge": {"source": "main"}}},
        )


def test_score_rejects_legacy_run_without_frozen_main_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _contracts(tmp_path)
    _, _, cases = contracts
    monkeypatch.setattr(runner, "_load_contracts", lambda _path: contracts)
    monkeypatch.setattr(
        runner,
        "_judge_provider",
        lambda _config: pytest.fail(
            "a missing frozen identity must fail before credential resolution"
        ),
    )
    root = tmp_path / "runs/run"
    runner._atomic_write_json(
        root / "run_manifest.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "run_id": "run",
            "config_sha256": "config-sha",
            "selection_sha256": "selection-sha",
            "frozen_cases_sha256": runner._canonical_sha256(cases),
        },
    )
    runner._atomic_write_json(
        root / "generation_report.json",
        {
            "schema_version": runner.RUN_SCHEMA_VERSION,
            "status": "complete",
            "cases": [{"case_id": "docbench:1:0", "reply": "x"}],
        },
    )

    with pytest.raises(
        runner.DocBenchRunnerError,
        match="run manifest is missing frozen main provider identity",
    ):
        runner.score_from_config(
            tmp_path / "config.yaml",
            run_dir=root,
            allow_live=True,
        )


def test_safe_run_id_rejects_paths() -> None:
    with pytest.raises(runner.DocBenchRunnerError):
        runner._safe_run_id("../escape")


def test_formal_actions_require_live_authorization_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_contracts",
        lambda _path: pytest.fail("authorization must fail before config loading"),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="allow_live=True"):
        runner.run_from_config(tmp_path / "config.yaml")
    with pytest.raises(runner.DocBenchRunnerError, match="allow_live=True"):
        runner.retry_failed_from_config(
            tmp_path / "config.yaml",
            run_dir=tmp_path / "run",
        )
    with pytest.raises(runner.DocBenchRunnerError, match="allow_live=True"):
        runner.score_from_config(
            tmp_path / "config.yaml",
            run_dir=tmp_path / "run",
        )


def test_operator_requires_explicit_live_authority(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["run", "--config", "config.yaml"]) == 2
    assert "allow_live=True" in capsys.readouterr().out


def test_operator_retry_failed_requires_live_and_dispatches_worker_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli.main(
            [
                "retry-failed",
                "--config",
                "config.yaml",
                "--run",
                "run",
            ]
        )
        == 2
    )
    assert "allow_live=True" in capsys.readouterr().out
    received: dict[str, object] = {}

    def retry_failed(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return {
            "status": "complete",
            "remaining_failed_case_count": 0,
            "gate_passed": True,
        }

    monkeypatch.setattr(runner, "retry_failed_from_config", retry_failed)
    assert (
        cli.main(
            [
                "retry-failed",
                "--config",
                "config.yaml",
                "--run",
                "run",
                "--max-workers",
                "3",
                "--error-code",
                "MODEL_TRANSPORT_FAILURE",
                "--error-code",
                "PROVIDER_UNAVAILABLE",
                "--allow-live",
            ]
        )
        == 0
    )
    assert received["args"] == (Path("config.yaml"),)
    assert received["kwargs"] == {
        "run_dir": Path("run"),
        "max_workers": 3,
        "error_codes": ["MODEL_TRANSPORT_FAILURE", "PROVIDER_UNAVAILABLE"],
        "allow_live": True,
    }


def test_operator_returns_nonzero_for_partial_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "score_from_config",
        lambda *_args, **_kwargs: {"status": "partial", "score": None},
    )
    assert (
        cli.main(
            [
                "score",
                "--config",
                "config.yaml",
                "--run",
                "run",
                "--allow-live",
            ]
        )
        == 1
    )
