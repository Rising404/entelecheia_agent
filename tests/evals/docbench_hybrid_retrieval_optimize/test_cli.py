"""Independent retrieval CLI ownership and dispatch contracts, without live models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evals.docbench_hybrid_retrieval_optimize import cli
from evals.docbench_hybrid_retrieval_optimize.paths import PROJECT_ROOT


@pytest.mark.parametrize("command", ((), ("build-retrieval",), ("annotate-retrieval",), ("evaluate-retrieval",)))
def test_module_help_is_independent_of_agent_benchmark_configuration(command: tuple[str, ...]) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PERSONAGRAPH_BENCH_EVAL_DIR", None)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "evals.docbench_hybrid_retrieval_optimize", *command, "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "python -m evals.docbench_hybrid_retrieval_optimize" in result.stdout
    assert "Traceback" not in result.stderr
    if not command or command == ("evaluate-retrieval",):
        assert "--allow-live" in result.stdout
    else:
        assert "--allow-live" not in result.stdout
    if not command:
        assert "SQLite BM25 baseline" in result.stdout


def test_build_dispatch_preserves_explicit_inputs_and_refreshes_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    selection = tmp_path / "selection.json"
    data_root = tmp_path / "source"
    source_runs = [tmp_path / "run-a", tmp_path / "run-b"]
    output = tmp_path / "dataset"
    curation = tmp_path / "curation.json"
    actions = []

    def build_retrieval_dataset(**kwargs):
        assert kwargs == {
            "selection_path": selection, "data_root": data_root,
            "run_dirs": source_runs, "output_dir": output,
            "visual_doc_ids": [7, 9], "curation_path": curation,
        }
        actions.append("build")
        return {"query_count": 2, "sources": {"private": "source-detail"}}

    def write_retrieval_exports(source, destination):
        assert source == output / "dataset.sqlite"
        assert destination == output / "exports"
        assert actions == ["build"]
        actions.append("export")

    monkeypatch.setattr(cli.retrieval_dataset, "build_retrieval_dataset", build_retrieval_dataset)
    monkeypatch.setattr(cli.retrieval_dataset, "write_retrieval_exports", write_retrieval_exports)
    assert cli.main([
        "build-retrieval", "--selection", str(selection), "--data-root", str(data_root),
        "--source-run", str(source_runs[0]), "--source-run", str(source_runs[1]),
        "--output", str(output), "--visual-doc-id", "7", "--visual-doc-id", "9",
        "--curation", str(curation),
    ]) == 0
    assert actions == ["build", "export"]
    assert json.loads(capsys.readouterr().out) == {"query_count": 2}


@pytest.mark.parametrize("failed", (False, True))
def test_evaluate_dispatch_keeps_baseline_errors_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failed: bool,
) -> None:
    dataset = tmp_path / "dataset.sqlite"
    output = tmp_path / "report"

    def run_retrieval_evaluation(source, destination):
        assert source == dataset and destination == output
        if failed:
            raise cli.retrieval_eval.RetrievalEvaluationError("dataset hash mismatch")
        return {"coverage": {"reviewed": 2}}

    monkeypatch.setattr(cli.retrieval_eval, "run_retrieval_evaluation", run_retrieval_evaluation)
    result = cli.main(["evaluate-retrieval", "--dataset", str(dataset), "--output", str(output)])
    printed = capsys.readouterr().out
    if failed:
        assert result == 2
        assert "dataset hash mismatch" in printed
        assert "Traceback" not in printed
    else:
        assert result == 0
        assert json.loads(printed) == {"reviewed": 2}


@pytest.mark.parametrize("projection_fails", (False, True))
def test_annotation_refresh_is_separate_from_committed_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], projection_fails: bool,
) -> None:
    dataset = tmp_path / "dataset.sqlite"
    annotations = tmp_path / "annotations.json"
    actions = []

    def apply_annotations(**kwargs):
        assert kwargs == {"dataset_path": dataset, "annotation_path": annotations}
        actions.append("committed")
        return {"annotation_counts": {"reviewed": 1}}

    def write_review_packets(source, destination):
        assert source == dataset and destination == tmp_path / "review"
        assert actions == ["committed"]
        actions.append("projection")
        if projection_fails:
            raise OSError("synthetic projection failure")

    monkeypatch.setattr(cli.retrieval_dataset, "apply_annotations", apply_annotations)
    monkeypatch.setattr(cli.retrieval_dataset, "write_review_packets", write_review_packets)
    monkeypatch.setattr(cli.retrieval_dataset, "write_retrieval_exports", lambda *_args: None)
    result = cli.main([
        "annotate-retrieval", "--dataset", str(dataset), "--annotations", str(annotations),
    ])
    output = capsys.readouterr().out
    assert actions == ["committed", "projection"]
    assert result == (2 if projection_fails else 0)
    if projection_fails:
        assert "Annotations committed" in output
        assert "Do not reapply" in output
    else:
        assert json.loads(output) == {"reviewed": 1}


@pytest.mark.parametrize("use_cache", (False, True))
def test_hybrid_dispatch_preserves_explicit_device_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    use_cache: bool,
) -> None:
    dataset, index, output = (tmp_path / name for name in ("dataset.sqlite", "index", "result"))
    cache = tmp_path / "encoding-cache" if use_cache else None

    def run(source, indexed, destination, *, device, batch_size, allow_live, progress, encoding_cache):
        assert (source, indexed, destination) == (dataset, index, output)
        assert (device, batch_size, allow_live) == ("mps", 4, True)
        assert encoding_cache == cache
        progress({"stage": "indexed", "completed": 2})
        return {"coverage": {"scored_queries": 2}}

    monkeypatch.setattr(cli.retrieval_hybrid, "run_hybrid_retrieval_evaluation", run)
    assert cli.main([
        "evaluate-retrieval", "--dataset", str(dataset), "--output", str(output),
        "--backend", "hybrid", "--index", str(index), "--device", "mps",
        "--batch-size", "4", "--allow-live",
        *(["--encoding-cache", str(cache)] if use_cache else []),
    ]) == 0
    printed = capsys.readouterr().out
    assert '"stage": "indexed"' in printed
    assert '"scored_queries": 2' in printed


@pytest.mark.parametrize("options", [
    ("--backend", "hybrid"),
    ("--backend", "hybrid", "--index", "unused"),
    ("--backend", "hybrid", "--allow-live"),
    ("--backend", "bm25", "--allow-live"),
    ("--backend", "bm25", "--device", "cpu"),
    ("--backend", "bm25", "--encoding-cache", "unused"),
])
def test_invalid_backend_options_do_not_start_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], options: tuple[str, ...],
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("invalid options must be rejected before starting evaluation")

    monkeypatch.setattr(cli.retrieval_hybrid, "run_hybrid_retrieval_evaluation", unexpected)
    monkeypatch.setattr(cli.retrieval_eval, "run_retrieval_evaluation", unexpected)
    assert cli.main([
        "evaluate-retrieval", "--dataset", str(tmp_path / "dataset.sqlite"),
        "--output", str(tmp_path / "report"), *options,
    ]) == 2
    assert "require" in capsys.readouterr().out
