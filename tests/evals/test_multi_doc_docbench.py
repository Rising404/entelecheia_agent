"""Offline tests use authored source fixtures, never the user's DocBench copy."""

from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import pytest

from evals.docbench.reproduce_or_run_script.selection import (
    DOMAIN_ORDER, SelectionValidationError, generate_selection_manifest, select_document_ids,
)
from evals.multi_doc.cases import materialize
from evals.multi_doc.docbench import HERE, SCHEMA, export, load_cases
from evals.multi_doc.fixture_documents import write_pdf
from evals.multi_doc.run import input_statistics, question_message
from scripts.check_repository_privacy import path_violations


@pytest.fixture
def sources(tmp_path):
    counts = {domain: 4 if domain == "academia" else 0 for domain in DOMAIN_ORDER}
    ids = select_document_ids(domain_counts=counts)["academia"]
    root = tmp_path / "source"
    for index, doc_id in enumerate(ids):
        directory = root / str(doc_id)
        directory.mkdir(parents=True)
        write_pdf(directory / f"source-{doc_id}.pdf", f"Source {doc_id}\nFact for document {doc_id}.")
        row = {"question": f"Original question {doc_id}?", "answer": f"Original answer {doc_id}",
               "type": "multimodal-t" if index == 0 else "text-only", "evidence": f"Source evidence {doc_id}"}
        (directory / f"{doc_id}_qa.jsonl").write_text(json.dumps(row) + "\n")
    base = tmp_path / "base.json"
    base.write_text(json.dumps(generate_selection_manifest(root, domain_counts=counts)))
    spec = {"schema_version": SCHEMA, "source_selection": base.name,
            "source_selection_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            "cases": [{"case_id": f"docbench:{ids[0]}:0", "attachment_doc_ids": list(ids), "evidence_pages": [1]}]}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(spec))
    return root, selection, spec


def test_original_question_and_four_complete_pdfs_reach_existing_runner(sources, tmp_path):
    root, selection, spec = sources
    case, = load_cases(root, selection=selection)
    doc_id = spec["cases"][0]["attachment_doc_ids"][0]
    assert case.question == f"Original question {doc_id}?"
    assert case.reference == f"Original answer {doc_id}"
    assert case.needs_vision
    files = materialize(case, tmp_path / "inputs")
    assert [path.name for path in files] == [f"document_{i:02d}.pdf" for i in range(1, 5)]
    for path, source in zip(files, case.source_pdfs, strict=True):
        assert path.read_bytes() == source.path.read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source.sha256
    stats = input_statistics(case)
    assert stats["pdf_count"] == stats["document_count"] == stats["pdf_pages"] == 4
    assert stats["image_count"] == 0  # Multimodal source PDFs must not generate a synthetic PNG.
    message = question_message(case)
    assert message.endswith("\n\n" + case.question)
    for secret in (case.reference, case.evidence[0][2], case.evidence[0][0], str(root)):
        assert secret not in message


@pytest.mark.parametrize("mutation", ["duplicate_pdf", "unknown_pdf", "missing_target", "duplicate_case", "page", "base_hash", "qa_drift"])
def test_docbench_binding_rejects_invalid_source_identity(sources, mutation):
    root, selection, spec = sources
    row = spec["cases"][0]
    if mutation == "duplicate_pdf":
        row["attachment_doc_ids"][1] = row["attachment_doc_ids"][0]
    elif mutation == "unknown_pdf":
        row["attachment_doc_ids"][1] = 9999
    elif mutation == "missing_target":
        row["attachment_doc_ids"] = row["attachment_doc_ids"][1:]
    elif mutation == "duplicate_case":
        spec["cases"].append(dict(row))
    elif mutation == "page":
        row["evidence_pages"] = [2]
    elif mutation == "base_hash":
        spec["source_selection_sha256"] = "0" * 64
    else:
        doc_id = row["attachment_doc_ids"][0]
        path = root / str(doc_id) / f"{doc_id}_qa.jsonl"
        record = json.loads(path.read_text())
        record["question"] += " changed"
        path.write_text(json.dumps(record) + "\n")
    selection.write_text(json.dumps(spec))
    with pytest.raises((ValueError, SelectionValidationError)):
        load_cases(root, selection=selection)


def test_copied_pdf_cannot_change_after_source_validation(sources, tmp_path):
    root, selection, _ = sources
    case, = load_cases(root, selection=selection)
    case.source_pdfs[0].path.write_bytes(b"%PDF changed")
    with pytest.raises(ValueError, match="hash drift"):
        materialize(case, tmp_path / "inputs")


def test_source_pdf_rejects_unsafe_names_and_generated_fixture_mix(sources, tmp_path):
    root, selection, _ = sources
    case, = load_cases(root, selection=selection)
    unsafe = replace(case.source_pdfs[0], name="../escape.pdf")
    with pytest.raises(ValueError, match="plain PDF filename"):
        materialize(replace(case, source_pdfs=(unsafe,)), tmp_path / "unsafe")
    with pytest.raises(ValueError, match="cannot be mixed"):
        materialize(replace(case, documents=(("text.txt", "text"),)), tmp_path / "mixed")


def test_private_export_separates_questions_gold_and_attachment_bytes(sources, tmp_path):
    root, selection, _ = sources
    output = tmp_path / "export"
    manifest = export(root, selection=selection, output=output)
    questions = json.loads((output / "questions.json").read_text())
    gold = json.loads((output / "gold.json").read_text())
    assert len(questions) == len(gold) == manifest["case_count"] == 1
    assert len(questions[0]["attachments"]) == 4
    assert set(questions[0]) == {"case_id", "question", "attachments"}
    assert set(path.suffix for path in (output / "attachments").rglob("*") if path.is_file()) == {".pdf"}
    for row in manifest["files"]:
        assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    with pytest.raises(FileExistsError):
        export(root, selection=selection, output=output)
    with pytest.raises(ValueError, match="ignored"):
        export(root, selection=selection, output=HERE / "dataset" / "docbench")
    for name in ("questions.json", "gold.json", "manifest.json", "example.pdf"):
        assert path_violations("evals/multi_doc/local/docbench/" + name)


def test_docbench_live_worker_also_requires_explicit_opt_in():
    process = subprocess.run([sys.executable, "-m", "evals.multi_doc.run", "--suite", "docbench"],
                             cwd=HERE.parents[1], capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert "real API requests require --allow-live" in process.stderr
