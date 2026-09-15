"""统一 DocBench 仓库入口的无网络合同测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.docbench.reproduce_or_run_script import cli
from evals.docbench.reproduce_or_run_script.config import BENCH_EVAL_DIR_ENV


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_CONFIG_IDS = (
    "l1_balanced_125",
    "l1_balanced_125_gate_off",
    "l1_balanced_125_gate_off_highland_235b",
    "l1_bge_m3_live_1",
    "l1_context_regression_5",
    "l1_gpu_context_regression_5",
    "l1_gpu_findings_regression_10",
    "l1_gpu_preparation_regression_1",
    "l1_gpu_preparation_smoke_3",
    "l1_gpu_retrieval_timeout_5",
    "l1_gpu_stage_remaining_15",
    "l1_gpu_vision_diagnostics_20",
    "l1_gpu_vision_publication_regression_5",
    "l1_stage_20",
    "l1_storage_smoke_3",
)


@pytest.mark.parametrize("root_value", (None, "", "relative-directory"))
@pytest.mark.parametrize("command", ((), ("prepare-data",), ("build-selection",)))
def test_help_does_not_resolve_benchmark_root(
    root_value: str | None, command: tuple[str, ...],
) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop(BENCH_EVAL_DIR_ENV, None)
    if root_value is not None:
        environment[BENCH_EVAL_DIR_ENV] = root_value

    result = subprocess.run(
        [sys.executable, "-B", "-m", "evals.docbench.reproduce_or_run_script", *command, "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Traceback" not in result.stderr


def test_checked_in_config_catalog_is_strict_and_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(tmp_path / "bench_eval"))
    expected_paths = tuple(sorted(cli.CONFIGS_DIR.glob("l1_*.yaml")))

    configs = cli.load_checked_in_configs()

    assert expected_paths
    assert tuple(config.source_path for config in configs) == tuple(
        path.resolve() for path in expected_paths
    )
    assert tuple(config.source_path.stem for config in configs) == CURRENT_CONFIG_IDS
    assert all(len(config.sha256) == 64 for config in configs)


def test_validate_and_list_load_configs_without_running_live_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench_eval_dir = tmp_path / "nonexistent-bench-eval"
    monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(bench_eval_dir))
    def unexpected_live_action(*_args, **_kwargs):
        pytest.fail("config discovery must not dispatch a live action")

    monkeypatch.setattr(cli.runner, "run_from_config", unexpected_live_action)
    monkeypatch.setattr(
        cli.runner,
        "retry_failed_from_config",
        unexpected_live_action,
    )
    monkeypatch.setattr(cli.runner, "score_from_config", unexpected_live_action)
    configs = cli.load_checked_in_configs()

    assert cli.main(["validate"]) == 0
    assert capsys.readouterr().out.strip() == (
        f"valid: {len(configs)} DocBench L1 configs"
    )

    assert cli.main(["list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{config.source_path.stem}\t{config.sha256}" for config in configs
    ]
    assert not bench_eval_dir.exists()


@pytest.mark.parametrize("command", ("validate", "list", "prepare-data"))
def test_commands_requiring_benchmark_root_fail_clearly_when_it_is_unset(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(BENCH_EVAL_DIR_ENV, raising=False)

    def unexpected_download(**_kwargs):
        pytest.fail("missing benchmark root must fail before download")

    monkeypatch.setattr(cli.download_selection, "download_qa_catalog", unexpected_download)
    arguments = [command]
    if command == "prepare-data":
        arguments.append("--qa-catalog-only")

    assert cli.main(arguments) == 2
    output = capsys.readouterr().out
    assert f"{BENCH_EVAL_DIR_ENV} is required" in output
    assert "absolute directory outside the source checkout" in output
    assert "Traceback" not in output


def test_readiness_reports_missing_benchmark_root_before_checking_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(BENCH_EVAL_DIR_ENV, raising=False)

    def unexpected_dependency_check(*_args, **_kwargs):
        pytest.fail("missing root must fail before data, provider, or retrieval checks")

    monkeypatch.setattr(cli.readiness, "_dataset_check", unexpected_dependency_check)
    monkeypatch.setattr(cli.readiness, "_resolve_provider_context", unexpected_dependency_check)
    monkeypatch.setattr(cli.readiness, "_retrieval_check", unexpected_dependency_check)

    assert cli.main(["readiness"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["network_calls_performed"] == 0
    assert report["run_directories_created"] == 0
    assert report["config_count"] == len(CURRENT_CONFIG_IDS)
    for config in report["configs"]:
        assert set(config["checks"]) == {"config"}
        error = config["checks"]["config"]
        assert error["error_code"] == "DocBenchConfigError"
        assert f"{BENCH_EVAL_DIR_ENV} is required" in error["message"]


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    (("ready", 0), ("failed", 2)),
)
def test_readiness_dispatches_to_read_only_owner_and_returns_json_status(
    status: str,
    expected_exit: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: list[tuple[Path, ...]] = []

    def build_report(paths):
        received.append(tuple(paths))
        return {
            "schema_version": "personagraph-docbench-readiness-v1",
            "status": status,
        }

    monkeypatch.setattr(cli.readiness, "build_readiness_report", build_report)

    assert cli.main(["readiness"]) == expected_exit
    assert received == [tuple(sorted(cli.CONFIGS_DIR.glob("l1_*.yaml")))]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_unified_entry_dispatches_data_preparation_and_selection_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BENCH_EVAL_DIR_ENV, raising=False)
    received: list[tuple[str, object]] = []
    monkeypatch.setattr(
        cli.download_selection,
        "_run_cli",
        lambda args: received.append(("prepare-data", args)) or 0,
    )
    monkeypatch.setattr(
        cli.selection,
        "_run_cli",
        lambda args: received.append(("build-selection", args)) or 0,
    )

    assert cli.main(["prepare-data", "--qa-catalog-only"]) == 0
    assert cli.main(
        [
            "build-selection",
            "--balanced",
            "--data-root",
            "data",
            "--output",
            "selection.json",
        ]
    ) == 0

    prepare_args = received[0][1]
    selection_args = received[1][1]
    assert received[0][0] == "prepare-data"
    assert prepare_args.qa_catalog_only is True
    assert received[1][0] == "build-selection"
    assert selection_args.balanced is True
    assert selection_args.data_root == Path("data")
    assert selection_args.output == Path("selection.json")


def test_unified_entry_does_not_expose_legacy_implicit_data_or_selection_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["prepare-data"]) == 2
    assert "requires --qa-catalog-only" in capsys.readouterr().out

    assert (
        cli.main(
            [
                "build-selection",
                "--data-root",
                "data",
                "--output",
                "selection.json",
            ]
        )
        == 2
    )
    assert "legacy formal-100" in capsys.readouterr().out


def test_build_selection_help_omits_retired_per_domain_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["build-selection", "--help"])

    assert raised.value.code == 0
    assert "--per-domain" not in capsys.readouterr().out


def test_help_uses_the_portable_repository_module_entrypoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "python -m evals.docbench.reproduce_or_run_script" in output
    assert "PERSONAGRAPH_BENCH_EVAL_DIR" in output
    assert "outside the checkout" in output
    assert "--help does not require this variable" in output
    assert "before starting Python" not in output
    assert "Desktop" not in output
    assert "script/run.py" not in output


def test_config_discovery_uses_explicit_external_root_without_changing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first-eval-root"
    second_root = tmp_path / "second-eval-root"
    monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(first_root))
    first = cli.load_checked_in_configs()
    monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(second_root))
    second = cli.load_checked_in_configs()

    assert tuple(config.sha256 for config in first) == tuple(
        config.sha256 for config in second
    )
    for configs, root in ((first, first_root), (second, second_root)):
        assert all(
            config.resolved["run"]["output_root"] == root / "docbench" / "runs"
            for config in configs
        )
        assert all(
            config.resolved["dataset"]["data_root"] == root / "docbench" / "source" / "data"
            for config in configs
        )
        assert not root.exists()


def test_validate_rejects_an_invalid_checked_in_config_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "l1_invalid.yaml").write_text("lane: L1\n", encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIGS_DIR", tmp_path)

    assert cli.main(["validate"]) == 2
    output = capsys.readouterr().out
    assert "schema validation failed" in output
    assert "Traceback" not in output


def test_validate_rejects_an_empty_config_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "CONFIGS_DIR", tmp_path)

    assert cli.main(["validate"]) == 2
    assert "no checked-in DocBench L1 configs" in capsys.readouterr().out


def test_module_entrypoint_runs_the_non_live_validator(tmp_path: Path) -> None:
    bench_eval_dir = tmp_path / "nonexistent-bench-eval"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment[BENCH_EVAL_DIR_ENV] = str(bench_eval_dir)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "evals.docbench.reproduce_or_run_script",
            "validate",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DocBench L1 configs" in completed.stdout
    assert not bench_eval_dir.exists()


def test_run_dispatches_to_the_formal_runner_without_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def run_from_config(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return {"status": "complete", "gate_passed": True, "run_id": "smoke"}

    monkeypatch.setattr(cli.runner, "run_from_config", run_from_config)

    assert (
        cli.main(
            [
                "run",
                "--config",
                "evals/docbench/configs/l1_storage_smoke_3.yaml",
                "--run-id",
                "smoke",
                "--resume",
                "--require-clean",
                "--allow-live",
            ]
        )
        == 0
    )
    assert received == {
        "args": (Path("evals/docbench/configs/l1_storage_smoke_3.yaml"),),
        "kwargs": {
            "run_id": "smoke",
            "resume": True,
            "require_clean": True,
            "allow_live": True,
        },
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete",
        "gate_passed": True,
        "run_id": "smoke",
    }


def test_run_returns_nonzero_when_generation_gates_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli.runner,
        "run_from_config",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "gate_passed": False,
        },
    )

    assert (
        cli.main(
            [
                "run",
                "--config",
                "evals/docbench/configs/l1_storage_smoke_3.yaml",
                "--allow-live",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["gate_passed"] is False


def test_retry_returns_nonzero_when_generation_gates_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli.runner,
        "retry_failed_from_config",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "remaining_failed_case_count": 0,
            "gate_passed": False,
        },
    )

    assert (
        cli.main(
            [
                "retry-failed",
                "--config",
                "evals/docbench/configs/l1_storage_smoke_3.yaml",
                "--run",
                "run",
                "--allow-live",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["gate_passed"] is False


def test_missing_run_config_is_reported_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(tmp_path / "missing.yaml"),
                "--allow-live",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "cannot read DocBench config" in output
    assert "Traceback" not in output


def test_retired_probe_and_adapter_sources_are_absent() -> None:
    retired_paths = (
        PROJECT_ROOT / "evals/docbench/adapters/common.py",
        PROJECT_ROOT / "evals/docbench/adapters/l1_live.py",
        PROJECT_ROOT / "evals/docbench/adapters/l2_live.py",
        PROJECT_ROOT / "evals/docbench/experiments/catalog.py",
        PROJECT_ROOT / "evals/docbench/experiments/contracts.py",
        PROJECT_ROOT / "evals/docbench/experiments/probe_runner.py",
        PROJECT_ROOT / "evals/docbench/experiments/probes/docbench_l1_live_v1.yaml",
    )

    assert not any(path.exists() for path in retired_paths)


def test_docbench_python_code_is_owned_by_the_unified_script_package() -> None:
    root_modules = sorted(
        path.name for path in (PROJECT_ROOT / "evals/docbench").glob("*.py")
    )
    script_modules = sorted(
        path.name
        for path in (
            PROJECT_ROOT / "evals/docbench/reproduce_or_run_script"
        ).glob("*.py")
    )

    assert root_modules == ["__init__.py"]
    assert script_modules == [
        "__init__.py",
        "__main__.py",
        "cli.py",
        "config.py",
        "derived_selection.py",
        "download_selection.py",
        "export_results.py",
        "post_commit.py",
        "provenance.py",
        "readiness.py",
        "retrieval_observability.py",
        "runner.py",
        "scorer.py",
        "selection.py",
    ]


def test_formal_runner_is_the_only_docbench_product_turn_owner() -> None:
    owners = []
    for path in sorted((PROJECT_ROOT / "evals/docbench").rglob("*.py")):
        if "sessions.chat_turn(" in path.read_text(encoding="utf-8"):
            owners.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert owners == [
        "evals/docbench/reproduce_or_run_script/runner.py"
    ]
