from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.configuration import paths

from evals.docbench.reproduce_or_run_script import download_selection, runner
from evals.docbench.reproduce_or_run_script.config import (
    BENCH_EVAL_DIR_ENV,
    load_docbench_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "evals/docbench/configs"


def test_docbench_installation_config_uses_app_state_not_benchmark_assets() -> None:
    assert runner.INSTALL_CONFIG == paths.LOCAL_CONFIG_DIR / "app_config.json"


@pytest.mark.parametrize(
    ("mode", "owner", "mapping_name"),
    (
        ([], "download_selected", "selected_drive_map.json"),
        (["--qa-catalog-only"], "download_qa_catalog", "drive_catalog_map.json"),
        (["--balanced-pdfs-only"], "download_balanced_pdfs", "drive_catalog_map.json"),
    ),
)
def test_download_defaults_resolve_the_current_external_root_only_at_dispatch(
    mode: list[str],
    owner: str,
    mapping_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object]] = []

    def download(**kwargs):
        received.append(kwargs)
        return {"document_count": 0, "file_count": 0, "qa_file_count": 0}

    monkeypatch.setattr(download_selection, owner, download)
    for name in ("first", "second"):
        bench_eval_dir = tmp_path / name
        monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(bench_eval_dir))
        assert download_selection.main(mode) == 0
        assert received[-1]["data_root"] == bench_eval_dir / "docbench/source/data"
        assert received[-1]["mapping_path"] == bench_eval_dir / "docbench/source" / mapping_name
        assert not bench_eval_dir.exists()


@pytest.mark.parametrize("root_value", (None, "", "relative-directory"))
def test_explicit_download_paths_do_not_resolve_the_benchmark_root(
    root_value: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BENCH_EVAL_DIR_ENV, raising=False)
    if root_value is not None:
        monkeypatch.setenv(BENCH_EVAL_DIR_ENV, root_value)
    received: list[dict[str, object]] = []

    def download(**kwargs):
        received.append(kwargs)
        return {"document_count": 0, "file_count": 0, "qa_file_count": 0}

    def unexpected_root_resolution():
        pytest.fail("explicit download paths must not resolve a benchmark root")

    monkeypatch.setattr(download_selection, "docbench_root", unexpected_root_resolution)
    monkeypatch.setattr(download_selection, "download_qa_catalog", download)
    data_root = tmp_path / "source-data"
    mapping_path = tmp_path / "catalog.json"
    assert download_selection.main([
        "--qa-catalog-only", "--data-root", str(data_root), "--mapping", str(mapping_path),
    ]) == 0
    assert received[0]["data_root"] == data_root
    assert received[0]["mapping_path"] == mapping_path
    assert not data_root.exists()
    assert not mapping_path.exists()


@pytest.mark.parametrize("override", ("--data-root", "--mapping"))
def test_partial_download_override_preserves_the_explicit_path(
    override: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_eval_dir = tmp_path / "bench_eval"
    monkeypatch.setenv(BENCH_EVAL_DIR_ENV, str(bench_eval_dir))
    received: list[dict[str, object]] = []

    def download(**kwargs):
        received.append(kwargs)
        return {"document_count": 0, "file_count": 0, "qa_file_count": 0}

    monkeypatch.setattr(download_selection, "download_qa_catalog", download)
    explicit_path = tmp_path / "explicit-path"
    assert download_selection.main(["--qa-catalog-only", override, str(explicit_path)]) == 0
    assert received[0]["data_root"] == (
        explicit_path if override == "--data-root" else bench_eval_dir / "docbench/source/data"
    )
    assert received[0]["mapping_path"] == (
        explicit_path if override == "--mapping"
        else bench_eval_dir / "docbench/source/drive_catalog_map.json"
    )
    assert not bench_eval_dir.exists()


def test_shipped_docbench_configs_resolve_assets_into_external_docbench_root(
    tmp_path: Path,
) -> None:
    configs = sorted(CONFIG_ROOT.glob("l1_*.yaml"))
    assert configs
    expected_root = (tmp_path / "bench_eval/docbench").resolve()
    environment = {BENCH_EVAL_DIR_ENV: str(tmp_path / "bench_eval")}

    for config_path in configs:
        loaded = load_docbench_config(
            config_path,
            project_root=PROJECT_ROOT,
            environment=environment,
        )
        assert loaded.raw["dataset"]["data_root"] == "bench://source/data"
        assert loaded.raw["run"]["output_root"] == "bench://runs"
        assert loaded.raw["scoring"]["prompt"] == (
            "bench://source/evaluation_prompt.txt"
        )
        assert loaded.resolved["dataset"]["data_root"] == expected_root / "source/data"
        assert loaded.resolved["run"]["output_root"] == expected_root / "runs"
        assert loaded.resolved["scoring"]["prompt"] == (
            expected_root / "source/evaluation_prompt.txt"
        )
