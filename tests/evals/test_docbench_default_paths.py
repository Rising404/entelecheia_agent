from __future__ import annotations

from pathlib import Path

from personagraph.configuration import paths

from evals.docbench.reproduce_or_run_script import download_selection, runner
from evals.docbench.reproduce_or_run_script.config import (
    BENCH_EVAL_DIR_ENV,
    docbench_root,
    load_docbench_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "evals/docbench/configs"


def test_docbench_runtime_defaults_separate_benchmark_assets_from_app_state() -> None:
    external_docbench = docbench_root()

    assert runner.INSTALL_CONFIG == paths.LOCAL_CONFIG_DIR / "app_config.json"
    assert download_selection.DEFAULT_DATA_ROOT == (
        external_docbench / "source/data"
    )
    assert download_selection.DEFAULT_SELECTED_MAPPING_PATH == (
        external_docbench / "source/selected_drive_map.json"
    )
    assert download_selection.DEFAULT_QA_CATALOG_MAPPING_PATH == (
        external_docbench / "source/drive_catalog_map.json"
    )


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
