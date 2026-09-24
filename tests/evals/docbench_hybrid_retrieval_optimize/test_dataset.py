"""Synthetic historical snapshots exercise the private retrieval-data boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas

from evals.docbench_hybrid_retrieval_optimize import retrieval_dataset as dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _source_database(run: Path, case: dict, *, content: str, populated: bool = True) -> Path:
    path = run / "cases" / case["case_id"] / "state/projects/synthetic/documents.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents(id TEXT PRIMARY KEY,path TEXT,current_version_id TEXT);
            CREATE TABLE document_versions(
                id TEXT PRIMARY KEY,source_sha256 TEXT,page_manifest_json TEXT,
                source_elements_json TEXT,processor_fingerprint TEXT,
                chunker_fingerprint TEXT,physical_page_count INTEGER,processing_status TEXT
            );
            CREATE TABLE doc_chunks(
                id TEXT PRIMARY KEY,source_version_id TEXT,seq INTEGER,
                producer_chunk_id TEXT,content TEXT,content_sha256 TEXT,
                source_pages_json TEXT,span_json TEXT,metadata_json TEXT
            );
            CREATE TABLE picture_observations(purpose TEXT,question TEXT,text TEXT);
            """
        )
        if populated:
            visual = {
                "unit_id": "figure-one", "kind": "figure", "element_id": "visual-element",
                "source_pages": [1], "text_element_ids": [],
                "locator": {"page": 1, "ordinal": 1, "bbox": [20, 20, 90, 100],
                            "section_path": ["Synthetic results"]},
            }
            manifest = {"inventory_status": "complete", "physical_page_count": 1,
                        "pages": [{"page_number": 1, "nontext_units": [visual]}]}
            elements = [{"element_id": "caption", "source_pages": [1], "locator": "p1#2",
                         "content": "Figure 1: Synthetic widgets by year."}]
            connection.execute("INSERT INTO documents VALUES (?,?,?)", (
                "historical-document", "unused-historical-path.pdf", "historical-version",
            ))
            connection.execute("INSERT INTO document_versions VALUES (?,?,?,?,?,?,?,?)", (
                "historical-version", case["pdf_sha256"], json.dumps(manifest),
                json.dumps(elements), "synthetic-parser", "synthetic-chunker", 1, "complete",
            ))
            connection.execute("INSERT INTO doc_chunks VALUES (?,?,?,?,?,?,?,?,?)", (
                "historical-chunk", "historical-version", 0, "stable-chunk", content,
                _text_sha(content), "[1]", '{"start":{"page":1}}', '{"kind":"paragraph"}',
            ))
        connection.execute("INSERT INTO picture_observations VALUES (?,?,?)", (
            "question", "PRIVATE_QUESTION_SENTINEL", "CONDITIONED_ANSWER_SENTINEL",
        ))
    return path


@pytest.fixture
def historical_inputs(tmp_path: Path) -> dict:
    data_root = tmp_path / "source"
    run = tmp_path / "runs/preferred"
    cases = []
    databases = []
    for doc_id, question_type in ((1, "text-only"), (2, "multimodal-f")):
        source = data_root / str(doc_id)
        source.mkdir(parents=True)
        pdf_path = source / "synthetic.pdf"
        pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=(120, 160))
        pdf_canvas.setFont("Helvetica", 4)
        pdf_canvas.drawString(20, 125, "Figure 1: Synthetic widgets by year.")
        pdf_canvas.save()
        qa = source / f"{doc_id}_qa.jsonl"
        _write_json(qa, {"type": question_type, "question": "PRIVATE_QUESTION_SENTINEL",
                         "answer": "PRIVATE_REFERENCE_SENTINEL", "evidence": "PRIVATE_EVIDENCE_SENTINEL"})
        case = {"case_id": f"docbench:{doc_id}:0", "doc_id": doc_id, "question_index": 0,
                "question_type": question_type, "pdf_filename": pdf_path.name,
                "pdf_sha256": _sha(pdf_path), "qa_sha256": _sha(qa)}
        cases.append(case)
        databases.append(_source_database(run, case, content=f"Synthetic original paragraph {doc_id}."))
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, {"cases": cases})
    return {"selection_path": selection_path, "data_root": data_root, "run_dirs": [run],
            "output_dir": tmp_path / "dataset", "cases": cases, "databases": databases}


def _build(inputs: dict) -> dict:
    kwargs = {key: inputs[key] for key in (
        "selection_path", "data_root", "run_dirs", "output_dir",
    )}
    if "visual_doc_ids" in inputs:
        kwargs["visual_doc_ids"] = inputs["visual_doc_ids"]
    return dataset.build_retrieval_dataset(**kwargs)


def _rows(inputs: dict, sql: str) -> list[tuple]:
    with sqlite3.connect(inputs["output_dir"] / "dataset.sqlite") as connection:
        return connection.execute(sql).fetchall()


def test_build_renders_assets_without_question_conditioned_corpus_or_automatic_gold(historical_inputs: dict) -> None:
    inputs = historical_inputs
    manifest = _build(inputs)
    assert manifest["case_count"] == 2
    assert manifest["unit_counts"] == {"chunk": 2, "figure": 2}
    assert manifest["query_conditioned_observations_included"] is False
    assert _rows(inputs, "SELECT DISTINCT annotation_status FROM queries") == [("pending",)]
    assert _rows(inputs, "SELECT * FROM qrels") == []
    corpus = "\n".join(row[0] for row in _rows(inputs, "SELECT content FROM units"))
    assert "Synthetic original paragraph" in corpus
    assert "Figure 1: Synthetic widgets by year." in corpus
    for sentinel in ("PRIVATE_QUESTION_SENTINEL", "PRIVATE_REFERENCE_SENTINEL",
                     "PRIVATE_EVIDENCE_SENTINEL", "CONDITIONED_ANSWER_SENTINEL"):
        assert sentinel not in corpus
    asset_path, metadata_json = _rows(inputs, "SELECT asset_path,metadata_json FROM units WHERE kind='figure'")[0]
    asset = inputs["output_dir"] / asset_path
    assert asset.is_file()
    assert _sha(asset) == json.loads(metadata_json)["asset_sha256"]
    review = json.loads((inputs["output_dir"] / "review/docbench-2-0.json").read_text())
    assert review["query"]["evidence"] == "PRIVATE_EVIDENCE_SENTINEL"
    assert review["proposals"]  # Annotation assistance remains separate from qrels.


def test_build_does_not_overwrite_existing_directory(historical_inputs: dict) -> None:
    inputs = historical_inputs
    inputs["output_dir"].mkdir()
    marker = inputs["output_dir"] / "keep.txt"
    marker.write_text("existing artifact")
    with pytest.raises(dataset.RetrievalDatasetError, match="already exists"):
        _build(inputs)
    assert marker.read_text() == "existing artifact"
    assert list(inputs["output_dir"].iterdir()) == [marker]


@pytest.mark.parametrize("changed_file", ["synthetic.pdf", "1_qa.jsonl"])
def test_build_rejects_source_hash_drift_without_partial_output(historical_inputs: dict, changed_file: str) -> None:
    inputs = historical_inputs
    path = inputs["data_root"] / "1" / changed_file
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(dataset.RetrievalDatasetError, match="Source hash mismatch"):
        _build(inputs)
    assert not inputs["output_dir"].exists()
    assert not list(inputs["output_dir"].parent.glob(".dataset-*"))


def test_build_rejects_historical_chunk_corruption(historical_inputs: dict) -> None:
    inputs = historical_inputs
    with sqlite3.connect(inputs["databases"][0]) as connection:
        connection.execute("UPDATE doc_chunks SET content='tampered'")
    with pytest.raises(dataset.RetrievalDatasetError, match="Chunk content hash mismatch"):
        _build(inputs)
    assert not inputs["output_dir"].exists()


def test_build_prefers_first_populated_matching_run_and_records_explicit_fallback(historical_inputs: dict) -> None:
    inputs = historical_inputs
    fallback = inputs["output_dir"].parent / "runs/fallback"
    for case in inputs["cases"]:
        _source_database(fallback, case, content="Fallback original content.")
    with sqlite3.connect(inputs["databases"][1]) as connection:
        connection.execute("DELETE FROM doc_chunks")
    inputs["run_dirs"].append(fallback)
    _build(inputs)
    assert _rows(inputs, "SELECT doc_id,source_run_id FROM documents ORDER BY doc_id") == [
        ("1", "preferred"), ("2", "fallback"),
    ]
    assert _rows(inputs, "SELECT doc_id,content FROM units WHERE kind='chunk' ORDER BY doc_id") == [
        ("1", "Synthetic original paragraph 1."), ("2", "Fallback original content."),
    ]


def test_build_preserves_uncheckpointed_wal_and_never_mutates_source(historical_inputs: dict) -> None:
    inputs = historical_inputs
    database = inputs["databases"][0]
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        content = "Latest committed content exists only in WAL."
        writer.execute("UPDATE doc_chunks SET content=?,content_sha256=?", (content, _text_sha(content)))
        writer.commit()
        wal = Path(str(database) + "-wal")
        assert wal.stat().st_size > 0
        before = {"db": _sha(database), "-wal": _sha(wal)}
        manifest = _build(inputs)
        assert _rows(inputs, "SELECT content FROM units WHERE doc_id='1' AND kind='chunk'") == [(content,)]
        assert {"db": _sha(database), "-wal": _sha(wal)} == before
        assert manifest["sources"][0]["source_db_hashes"] == before
    finally:
        writer.close()


def _annotation(case_id: str, primary: str) -> dict:
    return {"case_id": case_id, "primary_unit_ids": [primary], "reviewer": "synthetic-reviewer",
            "note": "Reviewed against synthetic source material."}


def _apply(inputs: dict, items: list[dict]) -> dict:
    path = inputs["output_dir"].parent / "annotations.json"
    _write_json(path, items)
    return dataset.apply_annotations(dataset_path=inputs["output_dir"] / "dataset.sqlite", annotation_path=path)


@pytest.mark.parametrize(("primary", "message"), [
    ("docbench:1:chunk:stable-chunk", "outside query document"),
    ("docbench:2:chunk:stable-chunk", "wrong modality"),
    ("nonexistent-unit", "outside query document"),
])
def test_annotations_reject_invalid_primary_and_roll_back_whole_batch(
    historical_inputs: dict, primary: str, message: str,
) -> None:
    inputs = historical_inputs
    _build(inputs)
    manifest_bytes = (inputs["output_dir"] / "manifest.json").read_bytes()
    with pytest.raises(dataset.RetrievalDatasetError, match=message):
        _apply(inputs, [_annotation("docbench:1:0", "docbench:1:chunk:stable-chunk"),
                        _annotation("docbench:2:0", primary)])
    assert _rows(inputs, "SELECT * FROM qrels") == []
    assert _rows(inputs, "SELECT DISTINCT annotation_status FROM queries") == [("pending",)]
    assert (inputs["output_dir"] / "manifest.json").read_bytes() == manifest_bytes


def test_annotations_require_explicit_review_and_only_mark_selected_cases(historical_inputs: dict) -> None:
    inputs = historical_inputs
    _build(inputs)
    valid = _annotation("docbench:2:0", "docbench:2:visual:figure-one")
    invalid = {**valid, "reviewer": ""}
    with pytest.raises(dataset.RetrievalDatasetError, match="reviewer and rationale"):
        _apply(inputs, [invalid])
    assert _rows(inputs, "SELECT * FROM qrels") == []
    manifest = _apply(inputs, [valid])
    assert manifest["status"] == "partially_reviewed"
    assert manifest["annotation_counts"] == {"pending": 1, "reviewed": 1}
    assert manifest["dataset_sha256"] == _sha(inputs["output_dir"] / "dataset.sqlite")
    assert _rows(inputs, "SELECT case_id,annotation_status FROM queries ORDER BY case_id") == [
        ("docbench:1:0", "pending"), ("docbench:2:0", "reviewed"),
    ]
    assert _rows(inputs, "SELECT case_id,unit_id,role FROM qrels") == [
        ("docbench:2:0", "docbench:2:visual:figure-one", "primary"),
    ]
    with pytest.raises(dataset.RetrievalDatasetError, match="Already reviewed"):
        _apply(inputs, [valid])


@pytest.mark.parametrize("defect", ["missing", "invalid_json", "schema", "hash", "database_drift"])
def test_annotations_preflight_manifest_before_changing_database(historical_inputs: dict, defect: str) -> None:
    inputs = historical_inputs
    _build(inputs)
    manifest_path = inputs["output_dir"] / "manifest.json"
    if defect == "missing":
        manifest_path.unlink()
    elif defect == "invalid_json":
        manifest_path.write_text("{broken")
    elif defect == "database_drift":
        with sqlite3.connect(inputs["output_dir"] / "dataset.sqlite") as connection:
            connection.execute("UPDATE queries SET annotation_note='untracked change'")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version" if defect == "schema" else "dataset_sha256"] = "invalid"
        _write_json(manifest_path, manifest)
    before = _sha(inputs["output_dir"] / "dataset.sqlite")
    with pytest.raises(dataset.RetrievalDatasetError, match="manifest"):
        _apply(inputs, [_annotation("docbench:1:0", "docbench:1:chunk:stable-chunk")])
    assert _sha(inputs["output_dir"] / "dataset.sqlite") == before
    assert _rows(inputs, "SELECT * FROM qrels") == []


def test_annotations_refuse_unmerged_dataset_wal(historical_inputs: dict) -> None:
    inputs = historical_inputs
    _build(inputs)
    writer = sqlite3.connect(inputs["output_dir"] / "dataset.sqlite")
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE queries SET annotation_note='live writer'")
        writer.commit()
        with pytest.raises(dataset.RetrievalDatasetError, match="unmerged WAL"):
            _apply(inputs, [_annotation("docbench:1:0", "docbench:1:chunk:stable-chunk")])
        assert writer.execute("SELECT count(*) FROM qrels").fetchone() == (0,)
    finally:
        writer.close()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_manifest_publication_failure_restores_database_or_keeps_recovery_copies(
    historical_inputs: dict, monkeypatch: pytest.MonkeyPatch, rollback_fails: bool,
) -> None:
    inputs = historical_inputs
    _build(inputs)
    original_db = (inputs["output_dir"] / "dataset.sqlite").read_bytes()
    original_manifest = (inputs["output_dir"] / "manifest.json").read_bytes()
    real_replace = Path.replace

    def failing_replace(path: Path, target: Path) -> Path:
        if path.name == "updated-manifest.json" or (rollback_fails and path.name == "rollback.sqlite"):
            raise OSError("synthetic publication failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    message = "recovery copies retained" if rollback_fails else "original dataset restored"
    with pytest.raises(dataset.RetrievalDatasetError, match=message):
        _apply(inputs, [_annotation("docbench:1:0", "docbench:1:chunk:stable-chunk")])
    stage = inputs["output_dir"] / ".dataset.sqlite-annotation-staging"
    if rollback_fails:
        assert (stage / "original.sqlite").read_bytes() == original_db
        assert (stage / "original-manifest.json").read_bytes() == original_manifest
        assert json.loads((stage / "recovery.json").read_text())["original_dataset_sha256"] == hashlib.sha256(original_db).hexdigest()
    else:
        assert not stage.exists()
        assert (inputs["output_dir"] / "dataset.sqlite").read_bytes() == original_db
        assert _rows(inputs, "SELECT * FROM qrels") == []
    assert (inputs["output_dir"] / "manifest.json").read_bytes() == original_manifest


def test_pending_review_records_reason_without_creating_gold_and_can_be_resolved(historical_inputs: dict) -> None:
    inputs = historical_inputs
    _build(inputs)
    note = "Figure detector omitted the needed panel; remains unresolved."
    manifest = _apply(inputs, [{"case_id": "docbench:2:0", "status": "pending",
                                "note": note, "reviewer": "synthetic-reviewer"}])
    assert manifest["status"] == "awaiting_annotation"
    assert _rows(inputs, "SELECT annotation_status,annotation_note FROM queries WHERE doc_id='2'") == [("pending", note)]
    assert _rows(inputs, "SELECT * FROM qrels") == []
    with pytest.raises(dataset.RetrievalDatasetError, match="Pending annotations cannot include gold"):
        _apply(inputs, [{**_annotation("docbench:2:0", "docbench:2:visual:figure-one"), "status": "pending"}])
    manifest = _apply(inputs, [_annotation("docbench:2:0", "docbench:2:visual:figure-one")])
    assert manifest["annotation_counts"] == {"pending": 1, "reviewed": 1}


def test_interrupted_publication_keeps_recovery_files_and_blocks_next_annotation(
    historical_inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = historical_inputs
    _build(inputs)
    original_db = (inputs["output_dir"] / "dataset.sqlite").read_bytes()
    original_manifest = (inputs["output_dir"] / "manifest.json").read_bytes()
    real_replace = Path.replace

    def interrupted_replace(path: Path, target: Path) -> Path:
        if path.name == "updated-manifest.json":
            raise KeyboardInterrupt("synthetic interruption between replacements")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        _apply(inputs, [_annotation("docbench:1:0", "docbench:1:chunk:stable-chunk")])
    stage = inputs["output_dir"] / ".dataset.sqlite-annotation-staging"
    assert (stage / "original.sqlite").read_bytes() == original_db
    assert (stage / "original-manifest.json").read_bytes() == original_manifest
    assert (stage / "recovery.json").is_file()
    with pytest.raises(dataset.RetrievalDatasetError, match="recovery required"):
        _apply(inputs, [_annotation("docbench:2:0", "docbench:2:visual:figure-one")])


@pytest.mark.parametrize(("field", "value"), [
    ("question_index", -1), ("question_index", True), ("question_index", "0"),
    ("doc_id", -1), ("doc_id", True), ("doc_id", "1"),
    ("case_id", "docbench:1:9"), ("pdf_filename", "../2/synthetic.pdf"),
    ("pdf_filename", "/tmp/synthetic.pdf"), ("pdf_filename", "..\\synthetic.pdf"),
])
def test_build_rejects_invalid_selection_identity_before_export(
    historical_inputs: dict, field: str, value: object,
) -> None:
    inputs = historical_inputs
    selection = json.loads(inputs["selection_path"].read_text())
    selection["cases"][0][field] = value
    _write_json(inputs["selection_path"], selection)
    with pytest.raises(dataset.RetrievalDatasetError, match="Invalid selection"):
        _build(inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("filename", ["synthetic.pdf", "1_qa.jsonl"])
def test_build_rejects_source_symlink_outside_data_root(historical_inputs: dict, filename: str) -> None:
    inputs = historical_inputs
    source = inputs["data_root"] / "1" / filename
    outside = inputs["output_dir"].parent / filename
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(dataset.RetrievalDatasetError, match="outside data root"):
        _build(inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("malicious_id", ["../../../../escaped", "..\\escaped", "/tmp/escaped"])
def test_build_rejects_visual_id_path_traversal_before_writing_assets(
    historical_inputs: dict, malicious_id: str,
) -> None:
    inputs = historical_inputs
    with sqlite3.connect(inputs["databases"][1]) as connection:
        manifest = json.loads(connection.execute("SELECT page_manifest_json FROM document_versions").fetchone()[0])
        manifest["pages"][0]["nontext_units"][0]["unit_id"] = malicious_id
        connection.execute("UPDATE document_versions SET page_manifest_json=?", (json.dumps(manifest),))
    with pytest.raises(dataset.RetrievalDatasetError, match="Unsafe visual unit id"):
        _build(inputs)
    assert not inputs["output_dir"].exists()
    assert not list(inputs["output_dir"].parent.rglob("escaped.jpg"))


def test_visual_asset_cannot_follow_output_symlink_outside_stage(historical_inputs: dict) -> None:
    inputs = historical_inputs
    stage = inputs["output_dir"].parent / "stage"
    outside = inputs["output_dir"].parent / "outside-assets"
    stage.mkdir()
    outside.mkdir()
    (stage / "assets").symlink_to(outside, target_is_directory=True)
    unit = {"unit_id": "valid-unit", "kind": "figure", "locator": {"page": 1, "bbox": [1, 1, 20, 20]}}
    pdf = pdfium.PdfDocument(str(inputs["data_root"] / "2/synthetic.pdf"))
    try:
        with pytest.raises(dataset.RetrievalDatasetError, match="outside output directory"):
            dataset._render_visual_page(pdf, 1, [unit], stage, "2")
    finally:
        pdf.close()
    assert list(outside.iterdir()) == []


def test_visual_scope_is_explicit_and_independent_of_question_type(historical_inputs: dict) -> None:
    inputs = historical_inputs
    inputs["visual_doc_ids"] = [1]  # Deliberately the text-only query's document.
    manifest = _build(inputs)
    assert manifest["visual_document_ids"] == ["1"]
    assert manifest["visual_document_scope"] == "explicit_document_ids"
    assert _rows(inputs, "SELECT DISTINCT doc_id FROM units WHERE kind<>'chunk'") == [("1",)]
    assert _rows(inputs, "SELECT DISTINCT doc_id FROM units WHERE kind='chunk' ORDER BY doc_id") == [("1",), ("2",)]


def test_visual_scope_cannot_silently_add_unselected_documents(historical_inputs: dict) -> None:
    inputs = historical_inputs
    inputs["visual_doc_ids"] = [999]
    with pytest.raises(dataset.RetrievalDatasetError, match="unselected document"):
        _build(inputs)
    assert not inputs["output_dir"].exists()


def test_vector_inventory_is_exported_and_reviewed_without_relabeling_as_figure(historical_inputs: dict) -> None:
    inputs = historical_inputs
    with sqlite3.connect(inputs["databases"][1]) as connection:
        manifest = json.loads(connection.execute("SELECT page_manifest_json FROM document_versions").fetchone()[0])
        manifest["pages"][0]["nontext_units"][0]["kind"] = "vector_graphics"
        manifest["pages"][0]["nontext_units"][0]["locator"]["bbox"] = [0, 0, 120, 160]
        connection.execute("UPDATE document_versions SET page_manifest_json=?", (json.dumps(manifest),))
    _build(inputs)
    content, locator, method = _rows(inputs, "SELECT content,locator_json,description_method FROM units WHERE kind='vector_graphics'")[0]
    assert "Synthetic widgets" in content
    assert json.loads(locator)["asset_granularity"] == "page"
    assert method == "pdf_text_layer_region"
    review = json.loads((inputs["output_dir"] / "review/docbench-2-0.json").read_text())
    assert [proposal["kind"] for proposal in review["proposals"]] == ["vector_graphics"]


def _spatial_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "spatial.pdf"
    pdf_canvas = canvas.Canvas(str(path), pagesize=(400, 500))
    pdf_canvas.setFont("Helvetica", 10)
    pdf_canvas.drawString(50, 405, "Year 2020 2021")
    pdf_canvas.drawString(50, 365, "Consumer revenue 33.3 34.0")
    pdf_canvas.drawString(50, 330, "Global Markets 18.8 19.0")
    pdf_canvas.drawString(50, 20, "OUTSIDE FOOTER UNRELATED ETHNICITY")
    pdf_canvas.save()
    return path


def test_vector_description_uses_visible_region_instead_of_last_ordinal_elements(tmp_path: Path) -> None:
    pdf = pdfium.PdfDocument(str(_spatial_pdf(tmp_path)))
    unit = {"unit_id": "vector-last", "kind": "vector_graphics", "element_id": "vector-element",
            "locator": {"page": 1, "ordinal": 999, "bbox": [45, 110, 320, 210]}}
    try:
        result = dataset._render_visual_page(pdf, 1, [unit], tmp_path / "images", "1")[0]
    finally:
        pdf.close()
    description = dataset._visual_description(result, {"footer": {"content": "OUTSIDE FOOTER UNRELATED ETHNICITY"}})
    assert "Year 2020 2021" in description
    assert "Consumer revenue 33.3 34.0" in description
    assert "Global Markets 18.8 19.0" in description
    assert "OUTSIDE FOOTER" not in description
    assert result["export_locator"]["asset_granularity"] == "region"


@pytest.mark.parametrize("bbox", [
    [0.0, 338.7636, 0.0, 456.3016],
    [68.0316, 718.8328, 138.8976, 718.8328],
])
def test_native_pdf_vector_lines_get_valid_context_without_losing_source_geometry(tmp_path: Path, bbox: list) -> None:
    path = tmp_path / "native-lines.pdf"
    pdf_canvas = canvas.Canvas(str(path), pagesize=(595.276, 793.701))
    pdf_canvas.line(bbox[0], 793.701 - bbox[1], bbox[2], 793.701 - bbox[3])
    pdf_canvas.save()
    unit = {"unit_id": "native-line", "kind": "vector_graphics", "locator": {"page": 1, "bbox": bbox}}
    pdf = pdfium.PdfDocument(str(path))
    try:
        result = dataset._render_visual_page(pdf, 1, [unit], tmp_path / "images", "51")[0]
    finally:
        pdf.close()
    locator = result["export_locator"]
    assert locator["bbox"] == bbox
    assert locator["asset_extent"] == "region_vector_line_context"
    assert locator["asset_granularity"] == "region"
    x0, y0, x1, y1 = locator["render_pixel_box"]
    assert x1 > x0 and y1 > y0
    assert (tmp_path / "images" / result["asset_path"]).is_file()


@pytest.mark.parametrize("kind,bbox", [
    ("vector_graphics", [20, 10, 10, 30]),
    ("vector_graphics", [10, 30, 20, 10]),
    ("vector_graphics", [0, 0, float("nan"), 30]),
    ("vector_graphics", [0, 0, 10, float("inf")]),
    ("table", [0, 0, 0, 30]),
    ("figure", [0, 0, 10, 0]),
])
def test_vector_line_support_still_rejects_invalid_visual_geometry(kind: str, bbox: list) -> None:
    unit = {"unit_id": "invalid", "kind": kind, "locator": {"page": 1, "bbox": bbox}}
    with pytest.raises(dataset.RetrievalDatasetError, match="Invalid visual geometry"):
        dataset._visual_context_bbox(unit, [unit], 595.276, 793.701)


def test_adjacent_table_strips_share_image_with_headers_and_keep_canonical_aliases(tmp_path: Path) -> None:
    pdf = pdfium.PdfDocument(str(_spatial_pdf(tmp_path)))
    units = [{"unit_id": name, "kind": "table", "locator": {"page": 1, "bbox": box}}
             for name, box in (("z-row", [45, 128, 330, 143]), ("a-row", [45, 163, 330, 178]))]
    try:
        rendered = dataset._render_visual_page(pdf, 1, units, tmp_path / "images", "1")
    finally:
        pdf.close()
    assert rendered[0]["asset_path"] == rendered[1]["asset_path"]
    assert len(list((tmp_path / "images").rglob("*.jpg"))) == 1
    canonical = dataset._canonical_visual_units(rendered, {})
    assert len(canonical) == 1
    assert canonical[0]["unit_id"] == "a-row"
    assert canonical[0]["source_unit_aliases"] == ["z-row"]
    assert "Year 2020 2021" in canonical[0]["description"]
    assert "Consumer revenue 33.3" in canonical[0]["description"]
    assert "Global Markets 18.8" in canonical[0]["description"]
    assert "OUTSIDE FOOTER" not in canonical[0]["description"]
    different_kind = {**rendered[1], "unit_id": "figure-source", "kind": "figure"}
    assert len(dataset._canonical_visual_units([*rendered, different_kind], {})) == 2


def test_exported_visual_aliases_preserve_source_ids_in_metadata(historical_inputs: dict) -> None:
    inputs = historical_inputs
    with sqlite3.connect(inputs["databases"][1]) as connection:
        manifest = json.loads(connection.execute("SELECT page_manifest_json FROM document_versions").fetchone()[0])
        original = manifest["pages"][0]["nontext_units"][0]
        manifest["pages"][0]["nontext_units"] = [{**original, "unit_id": "z-duplicate"}, {**original, "unit_id": "a-canonical"}]
        connection.execute("UPDATE document_versions SET page_manifest_json=?", (json.dumps(manifest),))
    _build(inputs)
    rows = _rows(inputs, "SELECT unit_id,metadata_json FROM units WHERE doc_id='2' AND kind='figure'")
    assert len(rows) == 1
    assert rows[0][0] == "docbench:2:visual:a-canonical"
    assert json.loads(rows[0][1])["source_unit_aliases"] == ["z-duplicate"]
