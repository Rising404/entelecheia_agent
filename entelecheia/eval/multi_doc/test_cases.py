"""Offline checks of evaluation inputs and live-call isolation; no app imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from entelecheia.eval.multi_doc.cases import CASES, materialize
from entelecheia.eval.multi_doc.complex_cases import COMPLEX_CASES
from entelecheia.eval.multi_doc.run import ALL_CASES, ROOT, SUITES, chain_succeeded, isolated_environment, require_worker_namespace
from entelecheia.eval.multi_doc.text_cases import TEXT_CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_only_documents_are_materialized_without_answers(tmp_path, case):
    files = materialize(case, tmp_path / "inputs")
    assert len(files) == 4
    assert len({file.name for file in files}) == 4
    for file in files:
        assert file.parent == tmp_path / "inputs"
        if file.suffix != ".png":
            content = file.read_text(encoding="utf-8")
            assert case.question not in content
            assert case.reference not in content
    with pytest.raises(FileExistsError):
        materialize(case, tmp_path / "inputs")


def test_live_execution_requires_explicit_opt_in():
    completed = subprocess.run(
        [sys.executable, "-m", "entelecheia.eval.multi_doc.run"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "real API requests require --allow-live" in completed.stderr


def test_worker_state_does_not_inherit_user_storage(tmp_path, monkeypatch):
    for key in ("PERSONAGRAPH_STATE_DIR", "PERSONAGRAPH_API_TOKEN_FILE",
                "PERSONAGRAPH_DEFAULT_PROJECTS_DIR", "PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH"):
        monkeypatch.setenv(key, "/unrelated/user-state")
    environment = isolated_environment(tmp_path / "case", tmp_path)
    assert Path(environment["PERSONAGRAPH_STATE_DIR"]) == tmp_path / "case" / "state"
    for key in ("PERSONAGRAPH_API_TOKEN_FILE", "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
                "PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH"):
        assert Path(environment[key]).is_relative_to(tmp_path)
    assert os.environ["PERSONAGRAPH_STATE_DIR"] == "/unrelated/user-state"


def test_direct_worker_refuses_user_storage_before_app_import(tmp_path, monkeypatch):
    case_root = tmp_path / "case"
    for key, value in isolated_environment(case_root, tmp_path).items():
        monkeypatch.setenv(key, value)
    require_worker_namespace(case_root)
    monkeypatch.setenv("PERSONAGRAPH_STATE_DIR", "/unrelated/user-state")
    with pytest.raises(ValueError, match="non-isolated PERSONAGRAPH_STATE_DIR"):
        require_worker_namespace(case_root)
    with pytest.raises(ValueError, match="outside the checkout"):
        require_worker_namespace(ROOT / "would-be-state")


def test_complex_sources_have_exact_physical_page_evidence_and_no_gold_leak(tmp_path):
    from pypdf import PdfReader

    case = COMPLEX_CASES[0]
    paths = materialize(case, tmp_path / "inputs")
    assert len(paths) == 16
    pages = {}
    for path in paths:
        if path.suffix == ".pdf":
            reader = PdfReader(path)
            assert len(reader.pages) == 6
            pages[path.name] = [" ".join(page.extract_text().split()) for page in reader.pages]
            for index, text in enumerate(pages[path.name], 1):
                assert f"{index} / 6" in text
                assert len(text) > 2000
        elif path.suffix == ".txt":
            pages[path.name] = [path.read_text(encoding="utf-8")]
    assert sum(len(text) for document in pages.values() for text in document) > 200_000
    for case in COMPLEX_CASES:
        for file, page, fact in case.evidence:
            if file.endswith(".png"):
                continue
            assert fact in pages[file][page - 1 if page else 0]
        corpus = "\n".join(text for document in pages.values() for text in document)
        assert case.reference not in corpus
        assert case.question not in corpus


def test_complex_pdf_generation_is_repeatable(tmp_path):
    first = materialize(COMPLEX_CASES[0], tmp_path / "first")
    second = materialize(COMPLEX_CASES[1], tmp_path / "second")
    assert [(p.name, p.read_bytes()) for p in first] == [(p.name, p.read_bytes()) for p in second]


def test_completed_failure_notification_is_not_a_successful_chain():
    settlement = {"status": "settled", "window_released": True, "timed_out": False,
                  "jobs": [{"status": "applied"}]}
    response = {"status": "completed", "processing_level": "L1", "error_code": None,
                "end_reason": None}
    assert chain_succeeded(response, settlement)
    # Observed live: the turn settles successfully even when the Host publishes an error notice.
    response.update(error_code="MODEL_OUTPUT_INVALID", end_reason="l1_terminal_notification")
    assert not chain_succeeded(response, settlement)
    response["error_code"] = None
    assert not chain_succeeded(response, settlement)


@pytest.mark.parametrize("case", TEXT_CASES, ids=lambda case: case.id)
def test_short_text_evidence_is_in_attachments_without_gold_leak(tmp_path, case):
    paths = materialize(case, tmp_path / "inputs")
    sources = {path.name: path.read_text(encoding="utf-8") for path in paths}
    assert 4 <= len(paths) <= 6
    assert all(path.suffix in {".txt", ".md"} for path in paths)
    assert 250 <= sum(len(content) for content in sources.values()) <= 2000
    for filename, page, fact in case.evidence:
        assert page is None
        assert fact in sources[filename]
    assert case.evidence and case.review_rubric
    corpus = "\n".join(sources.values())
    assert case.question not in corpus
    assert case.reference not in corpus


def test_text_suite_keeps_original_text_cases_and_unique_worker_ids():
    assert SUITES["text"] == tuple(case for case in CASES if not case.needs_vision) + TEXT_CASES
    assert len({case.id for case in ALL_CASES}) == len(ALL_CASES)
    assert all(not case.needs_vision for case in SUITES["text"])
