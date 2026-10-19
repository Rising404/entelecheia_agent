"""Published synthetic fixtures match their source and the native PDF reader."""

import hashlib
import json
from pathlib import Path

from evals.multi_doc.export_pdf import export
from evals.multi_doc.pdf_cases import PDF_CASES
from personagraph.input_processing.documents.contracts import (
    DocumentPageInventoryStatus,
    ProcessingAdmissionStatus,
)
from personagraph.input_processing.documents.readers.pdf import read_pdf


DATASET = Path(__file__).resolve().parents[2] / "evals/multi_doc/dataset"


def test_published_dataset_matches_manifest_and_regenerated_source(tmp_path):
    manifest = json.loads((DATASET / "manifest.json").read_text())
    expected_paths = {row["path"] for row in manifest["files"]} | {"manifest.json"}
    assert {str(p.relative_to(DATASET)) for p in DATASET.rglob("*") if p.is_file()} == expected_paths
    for row in manifest["files"]:
        content = (DATASET / row["path"]).read_bytes()
        assert (hashlib.sha256(content).hexdigest(), len(content)) == (row["sha256"], row["bytes"])
    regenerated = tmp_path / "export"
    export(regenerated)
    # Metadata must match even when a developer supplies a different CJK font.
    for name in ("questions.json", "gold.json"):
        assert (DATASET / name).read_bytes() == (regenerated / name).read_bytes()
    generated_manifest = json.loads((regenerated / "manifest.json").read_text())
    if generated_manifest["font_sha256"] == manifest["font_sha256"]:
        for name in expected_paths:
            assert (DATASET / name).read_bytes() == (regenerated / name).read_bytes()


def test_native_reader_preserves_all_frozen_pdf_evidence_and_physical_pages(tmp_path):
    pages = {}
    for source in sorted((DATASET / "attachments").glob("*.pdf")):
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        result = read_pdf(target)
        assert result.admission_status is not ProcessingAdmissionStatus.REJECTED
        assert result.page_manifest is not None
        assert result.page_manifest.physical_page_count == 4
        assert result.page_manifest.inventory_status is DocumentPageInventoryStatus.COMPLETE
        pages[source.name] = {
            number: "".join("".join((element.text or "").split())
                            for element in result.text_elements() if number in element.source_pages)
            for number in range(1, 5)
        }
    for case in PDF_CASES:
        for filename, number, fact in case.evidence:
            # Native table text retains cell separators; require the whole row in order.
            expected = "|".join(fact.split()) if case.id.endswith("_devices") and number == 4 else fact
            assert "".join(expected.split()) in pages[filename][number], (case.id, filename, number, fact)
