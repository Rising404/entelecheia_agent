from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DiagnosticCode,
    DocumentChunk,
    DocumentElement,
    DocumentLocator,
    DocumentNonTextKind,
    DocumentNonTextUnit,
    DocumentPageInventoryStatus,
    DocumentPageManifest,
    DocumentPageRecord,
    DocumentPageState,
    DocumentTextEvidenceOrigin,
    DocumentTextEvidence,
    ElementKind,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
    evaluate_paper_page_authority_eligibility,
)


CAPABILITIES = (
    "figure_inventory",
    "formula_inventory",
    "nontext_unit_inventory",
    "physical_page_inventory",
    "table_inventory",
    "text_element_source_pages",
    "typed_page_diagnostics",
)


def _page(
    number: int,
    *,
    state: DocumentPageState = DocumentPageState.TEXT,
    text_ids: tuple[str, ...] | None = None,
    nontext: tuple[DocumentNonTextUnit, ...] = (),
    diagnostics: tuple[ProcessingDiagnostic, ...] = (),
) -> DocumentPageRecord:
    return DocumentPageRecord(
        page_number=number,
        state=state,
        text_element_ids=text_ids if text_ids is not None else (f"e{number}",),
        nontext_units=nontext,
        diagnostics=diagnostics,
    )


def _manifest(page_count: int) -> DocumentPageManifest:
    return DocumentPageManifest(
        physical_page_count=page_count,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=tuple(_page(number) for number in range(1, page_count + 1)),
    )


def _result_and_chunks(page_count: int) -> tuple[ProcessingResult, tuple[DocumentChunk, ...]]:
    elements = tuple(
        DocumentElement(
            element_id=f"e{number}",
            kind=ElementKind.PARAGRAPH,
            text=f"page {number}",
            locator=DocumentLocator(page=number, ordinal=0),
            source_pages=(number,),
        )
        for number in range(1, page_count + 1)
    )
    result = ProcessingResult(
        elements=elements,
        processor=ProcessorFingerprint("test", "1"),
        page_manifest=_manifest(page_count),
    )
    chunks = tuple(
        DocumentChunk(
            chunk_id=f"ch_{number}",
            text=f"page {number}",
            span=ChunkSpan(element.locator, element.locator),
            section_path=(),
            element_ids=(element.element_id,),
            token_count=2,
            source_pages=(number,),
        )
        for number, element in enumerate(elements, start=1)
    )
    return result, chunks


def test_page_manifest_is_immutable_self_authenticating_and_round_trips():
    manifest = _manifest(10)

    assert len(manifest.manifest_sha256) == 64
    assert DocumentPageManifest.from_dict(manifest.to_dict()) == manifest
    with pytest.raises(FrozenInstanceError):
        manifest.physical_page_count = 11  # type: ignore[misc]
    with pytest.raises(ValueError, match="manifest_sha256"):
        replace(manifest, manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "pages",
    [
        (_page(1), _page(3)),
        (_page(2), _page(1)),
        (_page(1), _page(1)),
    ],
)
def test_page_manifest_rejects_gaps_reordering_and_duplicates(pages):
    with pytest.raises(ValueError, match="continuous physical pages"):
        DocumentPageManifest(
            physical_page_count=2,
            inventory_status=DocumentPageInventoryStatus.COMPLETE,
            detector_fingerprint="test-detector@1",
            detector_capabilities=CAPABILITIES,
            pages=pages,
        )


def test_processing_result_rejects_unlocated_or_uninventoried_text():
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=(_page(1, text_ids=("different",)),),
    )
    element = DocumentElement(
        element_id="e1",
        kind=ElementKind.PARAGRAPH,
        text="body",
        locator=DocumentLocator(page=1),
        source_pages=(1,),
    )

    with pytest.raises(ValueError, match="text element inventory"):
        ProcessingResult(
            elements=(element,),
            processor=ProcessorFingerprint("test", "1"),
            page_manifest=manifest,
        )


def test_partial_page_inventory_requires_a_typed_diagnostic():
    element = DocumentElement(
        element_id="e1",
        kind=ElementKind.PARAGRAPH,
        text="body",
        locator=DocumentLocator(page=1),
        source_pages=(1,),
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.PARTIAL,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=(_page(1),),
    )

    with pytest.raises(ValueError, match="partial page inventory"):
        ProcessingResult(
            elements=(element,),
            processor=ProcessorFingerprint("test", "1"),
            page_manifest=manifest,
        )


@pytest.mark.parametrize("include_gap", (False, True))
def test_visual_read_inventory_and_page_gap_must_agree(include_gap):
    text = DocumentElement(
        element_id="e1",
        kind=ElementKind.PARAGRAPH,
        text="caption",
        locator=DocumentLocator(page=1, ordinal=0),
        source_pages=(1,),
    )
    image = DocumentElement(
        element_id="image-1",
        kind=ElementKind.IMAGE,
        text=None,
        locator=DocumentLocator(page=1, ordinal=1),
        needs_vision=True,
        source_pages=(1,),
    )
    unit = DocumentNonTextUnit(
        unit_id="unit-1",
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(1,),
        element_id=image.element_id,
        locator=image.locator,
        requires_visual_read=True,
    )
    diagnostics = (
        (ProcessingDiagnostic(DiagnosticCode.PAGE_NEEDS_VISION, DocumentLocator(page=1)),)
        if include_gap
        else ()
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=(DocumentPageRecord(
            page_number=1,
            state=DocumentPageState.MIXED,
            text_element_ids=(text.element_id,),
            nontext_units=(unit,),
            diagnostics=diagnostics,
        ),),
    )

    if include_gap:
        ProcessingResult(
            elements=(text, image),
            processor=ProcessorFingerprint("test", "1"),
            diagnostics=diagnostics,
            page_manifest=manifest,
        )
    else:
        with pytest.raises(ValueError, match="visual-read inventory"):
            ProcessingResult(
                elements=(text, image),
                processor=ProcessorFingerprint("test", "1"),
                diagnostics=diagnostics,
                page_manifest=manifest,
            )


def test_page_manifest_requires_exact_text_evidence_source_unit_linkage():
    text = DocumentElement(
        element_id="ocr-1",
        kind=ElementKind.PARAGRAPH,
        text="mechanically inferred text",
        locator=DocumentLocator(page=1, ordinal=0),
        source_pages=(1,),
        text_evidence=DocumentTextEvidence(
            origin=DocumentTextEvidenceOrigin.OCR,
            confidence=0.8,
            source_unit_id="image-source",
        ),
    )
    image = DocumentElement(
        element_id="image-1",
        kind=ElementKind.IMAGE,
        text=None,
        locator=DocumentLocator(page=1, ordinal=1),
        needs_vision=True,
        source_pages=(1,),
    )
    wrong_unit = DocumentNonTextUnit(
        unit_id="different-source",
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(1,),
        text_element_ids=(text.element_id,),
        element_id=image.element_id,
        locator=image.locator,
        requires_visual_read=True,
    )
    vision_gap = ProcessingDiagnostic(
        DiagnosticCode.PAGE_NEEDS_VISION,
        DocumentLocator(page=1),
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=(DocumentPageRecord(
            page_number=1,
            state=DocumentPageState.MIXED,
            text_element_ids=(text.element_id,),
            nontext_units=(wrong_unit,),
            diagnostics=(vision_gap,),
        ),),
    )

    with pytest.raises(ValueError, match="text evidence source unit linkage"):
        ProcessingResult(
            elements=(text, image),
            processor=ProcessorFingerprint("test", "1"),
            diagnostics=(vision_gap,),
            page_manifest=manifest,
        )

    with pytest.raises(ValueError, match="source_pages"):
        DocumentElement(
            element_id="unlocated",
            kind=ElementKind.PARAGRAPH,
            text="body",
            locator=DocumentLocator(),
            source_pages=(1,),
        )


def test_chunk_source_pages_are_exact_not_the_span_min_max_range():
    element_1 = DocumentElement(
        "e1", ElementKind.PARAGRAPH, "first", DocumentLocator(page=1),
        source_pages=(1,),
    )
    element_30 = DocumentElement(
        "e30", ElementKind.PARAGRAPH, "last", DocumentLocator(page=30),
        source_pages=(30,),
    )
    result = ProcessingResult(
        elements=(element_1, element_30),
        processor=ProcessorFingerprint("test", "1"),
    )
    from personagraph.input_processing.documents import chunk_document

    chunk = chunk_document(result, source_key="paper")[0]

    assert chunk.span.pages == tuple(range(1, 31))  # 仅用于展示的旧版范围
    assert chunk.source_pages == (1, 30)


@pytest.mark.parametrize("page_count", [9, 31])
def test_page_authority_eligibility_rejects_out_of_range_papers(page_count):
    result, chunks = _result_and_chunks(page_count)

    decision = evaluate_paper_page_authority_eligibility(result, chunks)

    assert decision.eligible is False
    assert "physical_page_count_out_of_range" in decision.reason_codes


def test_page_authority_eligibility_requires_every_text_page():
    result, chunks = _result_and_chunks(30)
    only_endpoints = DocumentChunk(
        chunk_id="ch_endpoints",
        text="first and last",
        span=ChunkSpan(
            DocumentLocator(page=1, ordinal=0),
            DocumentLocator(page=30, ordinal=0),
        ),
        section_path=(),
        element_ids=("e1", "e30"),
        token_count=4,
        source_pages=(1, 30),
    )

    decision = evaluate_paper_page_authority_eligibility(
        result,
        (only_endpoints,),
    )

    assert decision.eligible is False
    assert "text_element_not_chunked" in decision.reason_codes


def test_page_authority_allows_empty_pages_and_surfaces_visual_gaps():
    result, chunks = _result_and_chunks(10)
    pages = list(result.page_manifest.pages)
    pages[4] = _page(
        5,
        state=DocumentPageState.NO_EXTRACTABLE_CONTENT,
        text_ids=(),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_EMPTY,
                DocumentLocator(page=5),
            ),
        ),
    )
    elements = tuple(element for element in result.elements if element.element_id != "e5")
    chunks = tuple(chunk for chunk in chunks if chunk.element_ids != ("e5",))
    empty_manifest = DocumentPageManifest(
        physical_page_count=10,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=tuple(pages),
    )
    empty_result = ProcessingResult(
        elements=elements,
        processor=result.processor,
        diagnostics=(
            ProcessingDiagnostic(DiagnosticCode.PAGE_EMPTY, DocumentLocator(page=5)),
        ),
        page_manifest=empty_manifest,
    )
    empty_decision = evaluate_paper_page_authority_eligibility(
        empty_result,
        chunks,
    )
    assert empty_decision.eligible is True
    assert empty_decision.coverage_gaps[0].gap_code == "no_extractable_content"
    assert empty_decision.coverage_gaps[0].source_pages == (5,)

    figure = DocumentNonTextUnit(
        unit_id="figure-5",
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(5,),
        requires_visual_read=True,
    )
    visual_pages = list(empty_manifest.pages)
    visual_pages[4] = _page(
        5,
        state=DocumentPageState.VISUAL_ONLY,
        text_ids=(),
        nontext=(figure,),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_NEEDS_VISION,
                DocumentLocator(page=5),
            ),
        ),
    )
    visual_result = ProcessingResult(
        elements=elements,
        processor=result.processor,
        diagnostics=(
            ProcessingDiagnostic(DiagnosticCode.PAGE_NEEDS_VISION, DocumentLocator(page=5)),
        ),
        page_manifest=DocumentPageManifest(
            physical_page_count=10,
            inventory_status=DocumentPageInventoryStatus.COMPLETE,
            detector_fingerprint="test-detector@1",
            detector_capabilities=CAPABILITIES,
            pages=tuple(visual_pages),
        ),
    )
    decision = evaluate_paper_page_authority_eligibility(visual_result, chunks)
    assert decision.eligible is True
    assert decision.reason_codes == ()
    assert len(decision.coverage_gaps) == 1
    assert decision.coverage_gaps[0].unit_id == "figure-5"
    assert decision.coverage_gaps[0].source_pages == (5,)
    assert decision.coverage_gaps[0].gap_code == "visual_interpretation_pending"


def test_processing_contract_rejects_vision_gap_without_a_located_visual_unit():
    result, _chunks = _result_and_chunks(10)
    pages = list(result.page_manifest.pages)
    pages[4] = _page(
        5,
        state=DocumentPageState.NO_EXTRACTABLE_CONTENT,
        text_ids=(),
        diagnostics=(
            ProcessingDiagnostic(
                DiagnosticCode.PAGE_NEEDS_VISION,
                DocumentLocator(page=5),
            ),
        ),
    )
    elements = tuple(element for element in result.elements if element.element_id != "e5")
    with pytest.raises(ValueError, match="visual-read inventory"):
        ProcessingResult(
            elements=elements,
            processor=result.processor,
            diagnostics=(
                ProcessingDiagnostic(
                    DiagnosticCode.PAGE_NEEDS_VISION,
                    DocumentLocator(page=5),
                ),
            ),
            page_manifest=DocumentPageManifest(
                physical_page_count=10,
                inventory_status=DocumentPageInventoryStatus.COMPLETE,
                detector_fingerprint="test-detector@1",
                detector_capabilities=CAPABILITIES,
                pages=tuple(pages),
            ),
        )


@pytest.mark.parametrize(
    "kind",
    [DocumentNonTextKind.FORMULA, DocumentNonTextKind.TABLE],
)
def test_page_authority_surfaces_text_extracted_structured_units_as_gaps(kind):
    result, chunks = _result_and_chunks(10)
    pages = list(result.page_manifest.pages)
    unit = DocumentNonTextUnit(
        unit_id=f"{kind.value}-5",
        kind=kind,
        source_pages=(5,),
        text_element_ids=("e5",),
        element_id="e5",
        locator=DocumentLocator(page=5),
        requires_visual_read=False,
    )
    pages[4] = _page(
        5,
        state=DocumentPageState.MIXED,
        text_ids=("e5",),
        nontext=(unit,),
    )
    structured = ProcessingResult(
        elements=result.elements,
        processor=result.processor,
        page_manifest=DocumentPageManifest(
            physical_page_count=10,
            inventory_status=DocumentPageInventoryStatus.COMPLETE,
            detector_fingerprint="test-detector@1",
            detector_capabilities=CAPABILITIES,
            pages=tuple(pages),
        ),
    )

    decision = evaluate_paper_page_authority_eligibility(structured, chunks)

    assert decision.eligible is True
    assert decision.coverage_gaps[0].kind is kind
    assert decision.coverage_gaps[0].gap_code == "nontext_semantics_pending"


@pytest.mark.parametrize(
    "kind",
    [
        DocumentNonTextKind.VECTOR_GRAPHICS,
        DocumentNonTextKind.FIGURE,
        DocumentNonTextKind.FORMULA,
        DocumentNonTextKind.TABLE,
    ],
)
def test_nontext_units_are_typed_and_bound_to_exact_source_pages(kind):
    unit = DocumentNonTextUnit(
        unit_id=f"unit-{kind.value}",
        kind=kind,
        source_pages=(2, 4),
        text_element_ids=("e2",),
        requires_visual_read=kind in {
            DocumentNonTextKind.VECTOR_GRAPHICS,
            DocumentNonTextKind.FIGURE,
        },
    )

    assert unit.to_dict()["kind"] == kind.value
    assert unit.to_dict()["source_pages"] == [2, 4]


def test_page_manifest_requires_every_visual_element_to_have_one_exact_unit_binding():
    text = DocumentElement(
        "text", ElementKind.PARAGRAPH, "body", DocumentLocator(page=1, ordinal=0)
    )
    visual = DocumentElement(
        "visual",
        ElementKind.IMAGE,
        None,
        DocumentLocator(page=1, ordinal=1),
        needs_vision=True,
    )
    diagnostic = ProcessingDiagnostic(
        DiagnosticCode.PAGE_NEEDS_VISION,
        DocumentLocator(page=1),
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint="test-detector@1",
        detector_capabilities=CAPABILITIES,
        pages=(DocumentPageRecord(
            page_number=1,
            state=DocumentPageState.TEXT,
            text_element_ids=(text.element_id,),
            diagnostics=(diagnostic,),
        ),),
    )

    with pytest.raises(ValueError, match="visual element inventory is not exact"):
        ProcessingResult(
            elements=(text, visual),
            processor=ProcessorFingerprint("test", "1"),
            diagnostics=(diagnostic,),
            page_manifest=manifest,
        )
