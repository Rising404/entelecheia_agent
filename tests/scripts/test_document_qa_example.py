"""Exercise the public synthetic document without a provider or model weights."""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader
import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "examples/document_qa/generate_sample.py"


@pytest.fixture
def sample():
    spec = importlib.util.spec_from_file_location("document_qa_example", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_is_reproducible_and_has_three_text_pages(sample):
    payload = sample.build_pdf()
    assert payload == sample.build_pdf()
    document = PdfReader(BytesIO(payload))
    assert len(document.pages) == 3
    pages = [" ".join(page.extract_text().split()) for page in document.pages]
    assert "Friday" in pages[0] and "16:30" in pages[0] and "Maple Desk" in pages[0]
    assert "two colored cards" in pages[1] and "six spare colored cards" in pages[1]
    assert "18 registered participants" in pages[2]
    assert "two facilitators" in pages[2]
    assert "42" not in " ".join(pages)  # The derived answer is not supplied in the PDF.
    assert all(f"{number} / 3" in page for number, page in enumerate(pages, 1))
    assert all(len(page.images) == 0 for page in document.pages)


@pytest.mark.parametrize("target", ["relative.pdf", "relative.txt"])
def test_sample_rejects_relative_paths(sample, target):
    with pytest.raises(ValueError, match="absolute"):
        sample.validate_output(target)


def test_sample_rejects_non_pdf_and_checkout_targets(sample, tmp_path):
    with pytest.raises(ValueError, match=".pdf"):
        sample.validate_output(str(tmp_path / "handbook.txt"))
    with pytest.raises(ValueError, match="outside"):
        sample.validate_output(str(ROOT / "harbor_handbook.pdf"))


def test_sample_rejects_missing_parent_without_creating_it(sample, tmp_path):
    parent = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError):
        sample.validate_output(str(parent / "handbook.pdf"))
    assert not parent.exists()


def test_sample_rejects_symlink_targets_and_checkout_aliases(sample, tmp_path):
    target = tmp_path / "existing.pdf"
    target.write_bytes(b"existing user file")
    alias = tmp_path / "alias.pdf"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        sample.validate_output(str(alias))
    checkout_alias = tmp_path / "checkout"
    checkout_alias.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        sample.validate_output(str(checkout_alias / "handbook.pdf"))
    assert target.read_bytes() == b"existing user file"


def test_sample_cli_creates_only_the_requested_file_and_refuses_overwrite(tmp_path):
    target = tmp_path / "harbor_handbook.pdf"
    command = [sys.executable, str(GENERATOR), "--output", str(target)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    assert "Created " in first.stdout
    payload = target.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert second.returncode == 1
    assert "overwriting is not allowed" in second.stderr
    assert target.read_bytes() == payload
    assert len(PdfReader(BytesIO(payload)).pages) == 3


def test_native_reader_preserves_the_sample_evidence_and_page_inventory(sample, tmp_path):
    from personagraph.input_processing.documents.contracts import (
        DocumentPageInventoryStatus,
        ProcessingAdmissionStatus,
    )
    from personagraph.input_processing.documents.readers.pdf import read_pdf

    target = tmp_path / "harbor_handbook.pdf"
    target.write_bytes(sample.build_pdf())
    result = read_pdf(target)
    assert result.admission_status is not ProcessingAdmissionStatus.REJECTED
    assert result.page_manifest is not None
    assert result.page_manifest.physical_page_count == 3
    assert result.page_manifest.inventory_status is DocumentPageInventoryStatus.COMPLETE
    for page_number, evidence in (
        (1, "16:30"),
        (1, "Maple Desk"),
        (2, "six spare colored cards"),
        (3, "18 registered participants"),
    ):
        text = " ".join(
            " ".join((element.text or "").split())
            for element in result.text_elements()
            if page_number in element.source_pages
        )
        assert evidence in text, (page_number, evidence)
