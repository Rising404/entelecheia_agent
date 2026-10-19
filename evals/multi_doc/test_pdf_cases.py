"""Offline checks of PDF-only inputs, unique evidence, physical pages and export isolation."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
import pytest

from evals.multi_doc.cases import materialize
from evals.multi_doc.export_pdf import export
from evals.multi_doc.pdf_cases import ACTIVITIES, PDF_CASES, PDF_DOCUMENTS
from evals.multi_doc.run import ROOT, SUITES


def compact(text: str) -> str:
    return "".join(text.split())


@pytest.fixture(scope="module")
def pdf_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("single-source-pdfs")
    paths = materialize(PDF_CASES[0], root / "attachments")
    pages = {path.name: [page.extract_text() for page in PdfReader(path).pages] for path in paths}
    return paths, pages


def test_corpus_has_four_neutral_pdf_files_and_balanced_targets(pdf_inputs):
    paths, pages = pdf_inputs
    assert [p.name for p in paths] == [f"document_{number:02d}.pdf" for number in range(1, 5)]
    assert len(PDF_CASES) == 12
    assert Counter(case.evidence[0][0] for case in PDF_CASES) == {p.name: 3 for p in paths}
    assert SUITES["pdf"] == PDF_CASES
    for path, activity in zip(paths, ACTIVITIES, strict=True):
        assert len(pages[path.name]) == 4
        for number, page in enumerate(pages[path.name], 1):
            assert activity.scope in compact(page)
            assert f"{number}/4" in compact(page)
            assert len(compact(page)) > 250
        assert path.read_bytes().startswith(b"%PDF-")
        fonts = PdfReader(path).pages[0]["/Resources"]["/Font"].get_object()
        assert any("/FontFile2" in font.get_object().get("/FontDescriptor", {})
                   for font in fonts.values())


@pytest.mark.parametrize("case", PDF_CASES, ids=lambda case: case.id)
def test_exactly_one_pdf_contains_all_gold_evidence(case, pdf_inputs):
    paths, pages = pdf_inputs
    assert case.documents == PDF_DOCUMENTS
    assert not case.images and not case.needs_vision
    target = case.evidence[0][0]
    assert {filename for filename, _, _ in case.evidence} == {target}
    for filename, page, fact in case.evidence:
        assert 1 <= page <= 4
        assert compact(fact) in compact(pages[filename][page - 1])
    sufficient = [name for name, texts in pages.items()
                  if all(compact(fact) in compact("\n".join(texts)) for _, _, fact in case.evidence)]
    assert sufficient == [target]
    for path in paths:
        text = "\n".join(pages[path.name])
        assert compact(case.question) not in compact(text)
        assert compact(case.reference) not in compact(text)
        assert path.name not in case.question


def test_pdf_bytes_are_repeatable_across_materializations(pdf_inputs, tmp_path):
    paths, _ = pdf_inputs
    second = materialize(PDF_CASES[-1], tmp_path / "attachments")
    assert [(p.name, p.read_bytes()) for p in paths] == [(p.name, p.read_bytes()) for p in second]


def test_export_separates_all_questions_and_gold_from_pdf_attachments(tmp_path):
    output = tmp_path / "benchmark"
    manifest = export(output)
    assert {path.name for path in (output / "attachments").iterdir()} == {name for name, _ in PDF_DOCUMENTS}
    questions = json.loads((output / "questions.json").read_text())
    gold = json.loads((output / "gold.json").read_text())
    assert len(questions) == len(gold) == manifest["case_count"] == 12
    for question, expected in zip(questions, gold, strict=True):
        assert set(question) == {"case_id", "question", "attachments"}
        assert question["case_id"] == expected["case_id"]
        assert len(question["attachments"]) == 4
        assert set(expected["distractor_pdfs"]) == set(question["attachments"]) - {expected["target_pdf"]}
    for record in manifest["files"]:
        path = output / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    with pytest.raises(FileExistsError):
        export(output)


@pytest.mark.parametrize("output", [Path("relative-output"), ROOT / "output" / "pdf"])
def test_export_rejects_relative_or_checkout_output(output):
    with pytest.raises(ValueError, match="outside the checkout"):
        export(output)
