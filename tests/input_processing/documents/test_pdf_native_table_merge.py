from __future__ import annotations

from pathlib import Path

from personagraph.input_processing.documents.contracts import (
    DiagnosticCode,
    DocumentElement,
    DocumentLocator,
    DocumentNonTextKind,
    DocumentNonTextUnit,
    DocumentPageInventoryStatus,
    DocumentPageManifest,
    DocumentPageRecord,
    DocumentPageState,
    ElementKind,
    ProcessingAdmissionStatus,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
)
from personagraph.input_processing.documents.readers.pdf import (
    _is_table_grid_scaffolding,
    merge_pdf_native_table_inventory,
)


def _write_one_table_pdf(
    path: Path,
    *,
    with_visual_cell: bool = False,
    with_distant_visual: bool = False,
    with_adjacent_visual: bool = False,
    partial_text_only: bool = False,
) -> None:
    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    canvas = reportlab.Canvas(str(path), pagesize=(300, 220))
    left, bottom, width, height = 40, 80, (200 if with_adjacent_visual else 220), 80
    for x in (left, left + width / 2, left + width):
        canvas.line(x, bottom, x, bottom + height)
    for y in (bottom, bottom + height / 2, bottom + height):
        canvas.line(left, y, left + width, y)
    cells = (
        (("Only extracted header text", 50, 135),)
        if partial_text_only
        else (
            ("Metric", 50, 135),
            ("Value", 165, 135),
            ("Accuracy", 50, 95),
            ("92%", 165, 95),
        )
    )
    for text, x, y in cells:
        canvas.drawString(x, y, text)
    if with_visual_cell:
        canvas.circle(225, 102, 8)
        canvas.line(217, 102, 233, 102)
        canvas.line(225, 94, 225, 110)
    if with_distant_visual:
        canvas.circle(20, 20, 8)
    if with_adjacent_visual:
        canvas.circle(260, 102, 8)
    canvas.save()


def _unresolved_table_result(
    *,
    gap_count: int,
    gap_bboxes: tuple[tuple[float, float, float, float], ...] | None = None,
    page_level_diagnostic: bool = False,
) -> ProcessingResult:
    if gap_bboxes is not None and len(gap_bboxes) != gap_count:
        raise ValueError("gap_bboxes must describe every requested gap")
    elements: list[DocumentElement] = []
    units: list[DocumentNonTextUnit] = []
    diagnostics: list[ProcessingDiagnostic] = []
    for index in range(gap_count):
        locator = DocumentLocator(
            page=1,
            ordinal=index,
            bbox=(
                gap_bboxes[index]
                if gap_bboxes is not None
                else (40.0, 80.0 + index * 10, 260.0, 160.0 + index * 10)
            ),
        )
        element_id = f"visual_table_{index + 1}"
        elements.append(
            DocumentElement(
                element_id=element_id,
                kind=ElementKind.IMAGE,
                text=None,
                locator=locator,
                needs_vision=True,
                source_pages=(1,),
            )
        )
        units.append(
            DocumentNonTextUnit(
                unit_id=f"table_gap_{index + 1}",
                kind=DocumentNonTextKind.TABLE,
                source_pages=(1,),
                element_id=element_id,
                locator=locator,
                requires_visual_read=True,
            )
        )
        diagnostics.append(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_NEEDS_VISION,
                DocumentLocator(page=1) if page_level_diagnostic else locator,
                detail="table requires visual interpretation",
            )
        )

    return ProcessingResult(
        elements=tuple(elements),
        processor=ProcessorFingerprint("docling", "test"),
        diagnostics=tuple(diagnostics),
        page_manifest=DocumentPageManifest(
            physical_page_count=1,
            inventory_status=DocumentPageInventoryStatus.COMPLETE,
            detector_fingerprint="docling-test",
            detector_capabilities=("table_inventory",),
            pages=(
                DocumentPageRecord(
                    page_number=1,
                    state=DocumentPageState.VISUAL_ONLY,
                    nontext_units=tuple(units),
                    diagnostics=tuple(diagnostics),
                ),
            ),
        ),
    )


def test_native_table_fallback_resolves_matching_empty_visual_table(tmp_path):
    path = tmp_path / "table.pdf"
    _write_one_table_pdf(path)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=1),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.COMPLETE
    assert merged.needs_vision is False
    assert merged.diagnostics == ()
    assert not any(element.needs_vision for element in merged.elements)
    assert [
        element.text
        for element in merged.elements
        if element.kind is ElementKind.TABLE
    ] == ["Metric | Value\nAccuracy | 92%"]
    assert merged.page_manifest is not None
    assert [
        (unit.kind, unit.requires_visual_read)
        for unit in merged.page_manifest.pages[0].nontext_units
    ] == [(DocumentNonTextKind.TABLE, False)]
    assert merge_pdf_native_table_inventory(path, merged) == merged


def test_native_table_fallback_does_not_hide_unmatched_visual_tables(tmp_path):
    path = tmp_path / "table.pdf"
    _write_one_table_pdf(path)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=2),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len(
        [
            diagnostic
            for diagnostic in merged.diagnostics
            if diagnostic.code is DiagnosticCode.PAGE_NEEDS_VISION
        ]
    ) == 2
    assert len([element for element in merged.elements if element.needs_vision]) == 2


def test_native_table_fallback_does_not_match_by_page_count_alone(tmp_path):
    path = tmp_path / "table.pdf"
    _write_one_table_pdf(path)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(
            gap_count=1,
            # PDF 原生表格在左下原点坐标系中占据 y=80..160。这个独立框表示
            # 一个栅格表格，只是恰好与原生表格位于同一页。
            gap_bboxes=((40.0, 5.0, 260.0, 45.0),),
        ),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1
    assert len(
        [element for element in merged.elements if element.kind is ElementKind.TABLE]
    ) == 1


def test_native_table_fallback_keeps_gap_when_table_contains_visual_cell(tmp_path):
    path = tmp_path / "hybrid-table.pdf"
    _write_one_table_pdf(path, with_visual_cell=True)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=1),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1
    assert len(
        [element for element in merged.elements if element.kind is ElementKind.TABLE]
    ) == 1


def test_native_table_fallback_keeps_hybrid_gap_with_distant_vector_art(tmp_path):
    path = tmp_path / "hybrid-table-with-distant-art.pdf"
    _write_one_table_pdf(
        path,
        with_visual_cell=True,
        with_distant_visual=True,
    )

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=1),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1


def test_native_table_fallback_keeps_gap_for_sparse_partial_extraction(tmp_path):
    path = tmp_path / "partial-table.pdf"
    _write_one_table_pdf(path, partial_text_only=True)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=1),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1


def test_native_table_fallback_checks_the_whole_primary_visual_gap(tmp_path):
    path = tmp_path / "table-with-adjacent-visual-column.pdf"
    _write_one_table_pdf(path, with_adjacent_visual=True)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(
            gap_count=1,
            # 该 Docling 缺口比原生文本表格更宽，并包含相邻矢量列。两个表格框的
            # IoU 仍大于 0.7，因此不能只凭几何关系抹去整个缺口。
            gap_bboxes=((40.0, 80.0, 280.0, 160.0),),
        ),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1


def test_only_grid_aligned_filled_bands_are_table_scaffolding():
    table_bbox = (40.0, 60.0, 260.0, 140.0)
    grid_segments = (
        (40.0, 60.0, 260.0, 60.0),
        (40.0, 100.0, 260.0, 100.0),
        (40.0, 140.0, 260.0, 140.0),
        (40.0, 60.0, 40.0, 140.0),
        (150.0, 60.0, 150.0, 140.0),
        (260.0, 60.0, 260.0, 140.0),
    )
    base = {
        "object_type": "rect",
        "fill": True,
        "stroke": False,
        "x0": 45.0,
        "x1": 255.0,
    }

    aligned = {**base, "top": 60.0, "bottom": 100.0}
    semantic_band = {**base, "top": 75.0, "bottom": 90.0}

    assert _is_table_grid_scaffolding(
        aligned,
        table_bboxes=(table_bbox,),
        table_grid_segments=grid_segments,
        table_cell_bboxes=(),
    )
    assert not _is_table_grid_scaffolding(
        semantic_band,
        table_bboxes=(table_bbox,),
        table_grid_segments=grid_segments,
        table_cell_bboxes=(),
    )


def test_native_table_fallback_fails_closed_for_unknown_vector_geometry(
    tmp_path,
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    path = tmp_path / "table-with-unlocated-vector.pdf"
    _write_one_table_pdf(path)
    original_bbox_of = pdf_reader._bbox_of

    def hide_vector_geometry(item):
        if isinstance(item, dict) and item.get("object_type") in {
            "curve",
            "line",
            "rect",
        }:
            return None
        return original_bbox_of(item)

    monkeypatch.setattr(pdf_reader, "_bbox_of", hide_vector_geometry)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(gap_count=1),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True


def test_native_table_fallback_keeps_gap_without_unit_bound_diagnostic(tmp_path):
    path = tmp_path / "table.pdf"
    _write_one_table_pdf(path)

    merged = merge_pdf_native_table_inventory(
        path,
        _unresolved_table_result(
            gap_count=1,
            page_level_diagnostic=True,
        ),
    )

    assert merged.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert merged.needs_vision is True
    assert len([element for element in merged.elements if element.needs_vision]) == 1
