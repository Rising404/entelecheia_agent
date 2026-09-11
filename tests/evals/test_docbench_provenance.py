from __future__ import annotations

from pathlib import Path

import pytest

from evals.docbench.reproduce_or_run_script import provenance, runner


def _manifest() -> dict[str, object]:
    return {
        "run_id": "run",
        "config_sha256": "config",
        "selection_sha256": "selection",
        "frozen_cases": [
            {
                "case_id": "docbench:1:0",
                "doc_id": 1,
                "domain": "academia",
                "question_type": "text-only",
            }
        ],
    }


def _provenance(*, source_changed: bool = False) -> dict[str, object]:
    return {
        "code_revision": "a" * 40,
        "worktree_dirty": False,
        "source_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "initial_run_provenance_complete": True,
        "finished_code_revision": "a" * 40,
        "finished_worktree_dirty": False,
        "finished_source_sha256": ("d" if source_changed else "b") * 64,
        "source_fingerprint_complete": True,
        "source_changed_during_run": source_changed,
    }


def _write_case_result(root: Path, **overrides: object) -> None:
    result = {
        "case_id": "docbench:1:0",
        "execution_ok": True,
        "post_commit_complete": True,
        "chain_complete": True,
        "lane_ok": True,
        "interaction_ok": True,
        "telemetry": {"telemetry_complete": True},
    }
    result.update(overrides)
    runner._atomic_write_json(
        root / "cases/docbench:1:0/result.json",
        result,
    )


def test_generation_gate_requires_all_runtime_checks(tmp_path: Path) -> None:
    _write_case_result(tmp_path)

    report = runner._report(
        tmp_path,
        _manifest(),
        run_provenance=_provenance(),
    )

    assert report["status"] == "complete"
    assert report["gate_passed"] is True
    assert report["baseline_eligible"] is True
    assert report["generation_gate"] == {
        "attempts_complete": True,
        "execution_ok": True,
        "post_commit_complete": True,
        "chain_complete": True,
        "lane_ok": True,
        "interaction_ok": True,
        "telemetry_complete": True,
        "source_stable": True,
    }


def test_complete_artifacts_do_not_hide_a_failed_runtime_gate(tmp_path: Path) -> None:
    _write_case_result(tmp_path, interaction_ok=False)

    report = runner._report(
        tmp_path,
        _manifest(),
        run_provenance=_provenance(),
    )

    assert report["status"] == "complete"
    assert report["gate_passed"] is False
    assert report["generation_gate"]["interaction_ok"] is False
    assert report["baseline_eligible"] is False


@pytest.mark.parametrize("settled", [False, None])
def test_answer_success_is_not_full_chain_success_without_settlement(
    tmp_path: Path, settled: bool | None,
) -> None:
    _write_case_result(tmp_path, post_commit_complete=settled, chain_complete=False)
    report = runner._report(tmp_path, _manifest(), run_provenance=_provenance())

    assert report["execution_ok_case_count"] == 1
    assert report["post_commit_complete_case_count"] == 0
    assert report["chain_complete_case_count"] == 0
    assert report["generation_gate"]["post_commit_complete"] is False
    assert report["gate_passed"] is False


def test_source_change_invalidates_generation(tmp_path: Path) -> None:
    _write_case_result(tmp_path)

    report = runner._report(
        tmp_path,
        _manifest(),
        run_provenance=_provenance(source_changed=True),
    )

    assert report["status"] == "invalidated"
    assert report["gate_passed"] is False
    assert report["generation_gate"]["source_stable"] is False


def test_unavailable_git_fingerprint_only_disables_baseline_eligibility(
    tmp_path: Path,
) -> None:
    _write_case_result(tmp_path)
    unavailable = _provenance()
    unavailable.update(
        {
            "code_revision": None,
            "source_sha256": None,
            "finished_code_revision": None,
            "finished_source_sha256": None,
            "source_fingerprint_complete": False,
        }
    )

    report = runner._report(
        tmp_path,
        _manifest(),
        run_provenance=unavailable,
    )

    assert report["status"] == "complete"
    assert report["gate_passed"] is True
    assert report["baseline_eligible"] is False


def test_require_clean_rejects_before_provider_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_contracts",
        lambda _path: pytest.fail("dirty run must fail before loading data contracts"),
    )
    monkeypatch.setattr(
        provenance,
        "read_git_provenance",
        lambda _root: ("a" * 40, True),
    )
    monkeypatch.setattr(
        provenance,
        "compute_source_tree_sha256",
        lambda _root: "b" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _config: pytest.fail("dirty run must fail before provider resolution"),
    )

    with pytest.raises(runner.DocBenchRunnerError, match="clean Git worktree"):
        runner.run_from_config(
            tmp_path / "config.yaml",
            allow_live=True,
            require_clean=True,
        )


def test_environment_fingerprint_redacts_credentials() -> None:
    first = provenance.compute_environment_sha256(
        {
            "PERSONAGRAPH_MODEL": "model-a",
            "PERSONAGRAPH_API_KEY": "secret-one",
            "PERSONAGRAPH_VISION_API_KEY": "vision-secret-one",
        }
    )
    second = provenance.compute_environment_sha256(
        {
            "PERSONAGRAPH_MODEL": "model-a",
            "PERSONAGRAPH_API_KEY": "secret-two",
            "PERSONAGRAPH_VISION_API_KEY": "vision-secret-two",
        }
    )

    assert first == second
    assert first != provenance.compute_environment_sha256(
        {
            "PERSONAGRAPH_MODEL": "model-b",
            "PERSONAGRAPH_API_KEY": "secret-two",
            "PERSONAGRAPH_VISION_API_KEY": "vision-secret-two",
        }
    )
