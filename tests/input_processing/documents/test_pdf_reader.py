from __future__ import annotations

from personagraph.input_processing.documents.contracts import (
    DiagnosticCode,
    DocumentNonTextKind,
    DocumentPageState,
    ElementKind,
    ProcessingAdmissionStatus,
    ProcessingResult,
    ProcessorFingerprint,
)
from personagraph.input_processing.documents.readers.pdf import (
    MIN_TEXT_CHARS_PER_PAGE,
    PageKind,
    classify_page,
    read_pdf,
)
from personagraph.input_processing.documents.readers import read_document
from personagraph.input_processing.files import MAX_DOCUMENT_FILE_BYTES


def _codes(result):
    return {diagnostic.code for diagnostic in result.diagnostics}


def test_a_text_page_yields_paragraphs_with_page_and_geometry(text_pdf):
    result = read_pdf(text_pdf)

    paragraphs = [e for e in result.elements if e.kind is ElementKind.PARAGRAPH]
    assert paragraphs, "a readable page must produce text elements"
    for element in paragraphs:
        assert element.locator.page == 1
        assert element.locator.bbox is not None
        assert element.locator.describe().startswith("p1")
    assert result.needs_vision is False
    assert result.page_manifest is not None
    assert result.page_manifest.physical_page_count == 1
    assert result.page_manifest.pages[0].state is DocumentPageState.TEXT
    assert set(result.page_manifest.pages[0].text_element_ids) == {
        element.element_id for element in paragraphs
    }


def test_a_scanned_page_reports_needing_vision_instead_of_returning_nothing(scanned_pdf):
    """整个设计正是为防止这种失败：空提取结果静默变成模型以为已收到的文档。"""
    result = read_pdf(scanned_pdf)

    assert result.text_elements() == ()
    assert result.needs_vision is True
    assert DiagnosticCode.PAGE_NEEDS_VISION in _codes(result)
    image = [e for e in result.elements if e.kind is ElementKind.IMAGE]
    assert len(image) == 1 and image[0].locator.page == 1
    page = result.page_manifest.pages[0]
    assert page.state is DocumentPageState.VISUAL_ONLY
    assert [unit.kind for unit in page.nontext_units] == [DocumentNonTextKind.FIGURE]


def test_a_mixed_document_keeps_readable_pages_and_flags_only_the_scan(mixed_pdf):
    result = read_pdf(mixed_pdf)

    readable_pages = {e.locator.page for e in result.text_elements()}
    vision_pages = {e.locator.page for e in result.elements if e.needs_vision}
    assert readable_pages == {1}
    assert vision_pages == {2}
    assert [page.page_number for page in result.page_manifest.pages] == [1, 2]


def test_an_empty_page_is_diagnosed_rather_than_dropped(empty_pdf):
    result = read_pdf(empty_pdf)

    assert result.elements == ()
    assert DiagnosticCode.PAGE_EMPTY in _codes(result)
    assert result.is_complete is False
    assert result.page_manifest.pages[0].state is DocumentPageState.NO_EXTRACTABLE_CONTENT


def test_a_corrupt_file_fails_with_a_diagnosis_not_an_exception(corrupt_pdf):
    result = read_pdf(corrupt_pdf)

    assert result.elements == ()
    assert _codes(result) & {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.EMPTY_SOURCE}


def test_native_pdf_rejects_an_oversized_source_before_parser_io(tmp_path, monkeypatch):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    path = tmp_path / "oversized.pdf"
    with path.open("wb") as stream:
        stream.truncate(MAX_DOCUMENT_FILE_BYTES + 1)

    def must_not_open(_path):
        raise AssertionError("pdf parser must not see an oversized source")

    monkeypatch.setattr("pdfplumber.open", must_not_open)

    result = pdf_reader.read_pdf(path)

    assert result.elements == ()
    assert _codes(result) == {DiagnosticCode.LIMIT_REACHED}
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_document_router_rejects_oversized_pdf_before_security_trailer_parse(
    tmp_path,
    monkeypatch,
):
    from personagraph.input_processing.documents import readers
    from personagraph.input_processing.documents.readers import registry

    path = tmp_path / "oversized.pdf"
    with path.open("wb") as stream:
        stream.truncate(MAX_DOCUMENT_FILE_BYTES + 1)

    def must_not_inspect(_path):
        raise AssertionError("security parser must not see an oversized source")

    monkeypatch.setattr(registry, "_pdf_requires_password", must_not_inspect)

    result = readers.read_document(path, engine="native")

    assert result.elements == ()
    assert _codes(result) == {DiagnosticCode.LIMIT_REACHED}
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_document_router_rejects_pdf_above_absolute_render_budget(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "more-than-1000-letter-pages-equivalent.pdf"
    writer = PdfWriter()
    # 一个巨型 MediaBox 可保持夹具精简，同时证明累计准入预算依据来源几何尺寸，
    # 而非页数。
    writer.add_blank_page(width=612, height=792 * 1001)
    with path.open("wb") as stream:
        writer.write(stream)

    result = read_document(path, engine="native")

    assert result.elements == ()
    assert _codes(result) == {DiagnosticCode.LIMIT_REACHED}
    assert result.diagnostics[0].detail == (
        "pdf_ingest_total_pixel_limit_reached:page=1"
    )
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_encrypted_pdf_is_password_required_in_native_and_default_routes(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "locked.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    with path.open("wb") as stream:
        writer.write(stream)

    native = read_document(path, engine="native")
    configured = read_document(path)

    assert _codes(native) == {DiagnosticCode.PASSWORD_REQUIRED}
    assert _codes(configured) == {DiagnosticCode.PASSWORD_REQUIRED}
    assert native.admission_status is ProcessingAdmissionStatus.REJECTED
    assert configured.admission_status is ProcessingAdmissionStatus.REJECTED


def test_empty_user_password_pdf_is_readable_instead_of_password_blocked(tmp_path):
    from pypdf import PdfReader, PdfWriter

    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    base = tmp_path / "empty-password-base.pdf"
    canvas = reportlab.Canvas(str(base), pagesize=(300, 220))
    for index in range(12):
        canvas.drawString(
            30,
            200 - index * 12,
            f"Readable empty-password body line {index + 1}",
        )
    canvas.save()

    path = tmp_path / "empty-password.pdf"
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(str(base)))
    writer.encrypt("", owner_password="owner-secret")
    with path.open("wb") as stream:
        writer.write(stream)

    encrypted = PdfReader(str(path), strict=False)
    assert encrypted.is_encrypted is True
    assert encrypted.decrypt("")

    result = read_document(path, engine="native")

    assert DiagnosticCode.PASSWORD_REQUIRED not in _codes(result)
    assert "Readable empty-password body line" in "\n".join(
        element.text or "" for element in result.text_elements()
    )
    assert result.admission_status is not ProcessingAdmissionStatus.REJECTED


def test_xfa_form_is_an_explicit_partial_gap_in_native_and_default_routes(tmp_path):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    base = tmp_path / "xfa-base.pdf"
    canvas = reportlab.Canvas(str(base), pagesize=(300, 220))
    for index in range(12):
        canvas.drawString(30, 200 - index * 12, f"Readable body line {index + 1}")
    canvas.save()

    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(str(base)))
    xfa = DecodedStreamObject()
    xfa.set_data(b"<field><value>XFA SECRET 73500</value></field>")
    writer._root_object.update({
        NameObject("/AcroForm"): DictionaryObject({NameObject("/XFA"): xfa})
    })
    path = tmp_path / "xfa.pdf"
    with path.open("wb") as stream:
        writer.write(stream)

    native = read_document(path, engine="native")
    configured = read_document(path)

    for result in (native, configured):
        assert DiagnosticCode.PARSER_PARTIAL in _codes(result)
        assert "XFA SECRET 73500" not in "\n".join(
            element.text or "" for element in result.elements
        )
        assert result.admission_status is not ProcessingAdmissionStatus.COMPLETE
    assert native.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_element_ids_are_stable_across_reads(text_pdf):
    """只有重新读取未变文件仍得到相同 ID 时，派生数据才能复用。"""
    first = [e.element_id for e in read_pdf(text_pdf).elements]
    second = [e.element_id for e in read_pdf(text_pdf).elements]
    assert first == second and first


def test_same_name_and_size_pdf_bytes_do_not_reuse_element_ids(text_pdf):
    original = text_pdf.read_bytes()
    first = [element.element_id for element in read_pdf(text_pdf).elements]
    marker = original.index(b"%PDF-1.7") + len(b"%PDF-1.7")
    changed = original[:marker] + b" " + original[marker + 1:]
    assert len(changed) == len(original)
    text_pdf.write_bytes(changed)

    second = [element.element_id for element in read_pdf(text_pdf).elements]

    assert first and second
    assert first != second


def test_table_text_and_borders_have_one_authoritative_representation(tmp_path):
    """表格单元格不得同时泄漏为正文或虚假的视觉缺口。"""

    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    path = tmp_path / "table.pdf"
    canvas = reportlab.Canvas(str(path), pagesize=(300, 220))
    left, bottom, width, height = 40, 80, 220, 80
    for x in (left, left + width / 2, left + width):
        canvas.line(x, bottom, x, bottom + height)
    for y in (bottom, bottom + height / 2, bottom + height):
        canvas.line(left, y, left + width, y)
    for text, x, y in (
        ("Metric", 50, 135),
        ("Value", 165, 135),
        ("Accuracy", 50, 95),
        ("92%", 165, 95),
    ):
        canvas.drawString(x, y, text)
    canvas.save()

    result = read_pdf(path)
    configured = read_document(path)

    tables = [element for element in result.elements if element.kind is ElementKind.TABLE]
    paragraphs = [
        element for element in result.elements if element.kind is ElementKind.PARAGRAPH
    ]
    assert [element.text for element in tables] == [
        "Metric | Value\nAccuracy | 92%"
    ]
    assert paragraphs == []
    assert all(
        unit.kind is not DocumentNonTextKind.VECTOR_GRAPHICS
        for unit in result.page_manifest.pages[0].nontext_units
    )
    assert DiagnosticCode.PAGE_NEEDS_VISION not in _codes(result)
    assert any(
        element.kind is ElementKind.TABLE
        and element.text == "Metric | Value\nAccuracy | 92%"
        for element in configured.elements
    )


def test_vector_art_inside_a_table_cell_is_not_discarded_as_grid_scaffolding(
    tmp_path,
):
    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    path = tmp_path / "table-with-icon.pdf"
    canvas = reportlab.Canvas(str(path), pagesize=(300, 220))
    left, bottom, width, height = 40, 80, 220, 80
    for x in (left, left + width / 2, left + width):
        canvas.line(x, bottom, x, bottom + height)
    for y in (bottom, bottom + height / 2, bottom + height):
        canvas.line(left, y, left + width, y)
    for text, x, y in (
        ("Metric", 50, 135),
        ("Value", 165, 135),
        ("Accuracy", 50, 95),
        ("92%", 165, 95),
    ):
        canvas.drawString(x, y, text)
    # 右下单元格中的圆形与十字是语义图形，而非表格边框的一部分。
    canvas.circle(225, 102, 8)
    canvas.line(217, 102, 233, 102)
    canvas.line(225, 94, 225, 110)
    canvas.save()

    result = read_pdf(path)

    assert any(element.kind is ElementKind.TABLE for element in result.elements)
    assert any(
        unit.kind is DocumentNonTextKind.VECTOR_GRAPHICS
        for unit in result.page_manifest.pages[0].nontext_units
    )
    assert DiagnosticCode.PAGE_NEEDS_VISION in _codes(result)
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_pdf_form_values_and_comments_are_not_silently_dropped(tmp_path):
    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    path = tmp_path / "annotated.pdf"
    canvas = reportlab.Canvas(str(path), pagesize=(300, 220))
    canvas.drawString(40, 190, "Readable body evidence")
    canvas.acroForm.textfield(
        name="budget",
        value="FORM VALUE 73500",
        x=40,
        y=130,
        width=180,
        height=24,
    )
    canvas.textAnnotation("COMMENT EVIDENCE", Rect=(40, 80, 80, 110))
    canvas.save()

    result = read_pdf(path)
    text = "\n".join(element.text or "" for element in result.elements)

    assert "budget: FORM VALUE 73500" in text
    assert "COMMENT EVIDENCE" in text
    assert result.admission_status is ProcessingAdmissionStatus.COMPLETE


def test_pdf_highlight_comment_is_extracted_without_subtype_whitelisting(tmp_path):
    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["Canvas"])
    source = tmp_path / "highlight-source.pdf"
    canvas = reportlab.Canvas(str(source), pagesize=(300, 220))
    for index in range(12):
        canvas.drawString(40, 205 - index * 14, f"Highlighted body evidence {index}")
    canvas.save()

    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import Highlight
    from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

    annotation = Highlight(
        rect=(35, 170, 220, 195),
        quad_points=ArrayObject([
            FloatObject(value)
            for value in (35, 195, 220, 195, 35, 170, 220, 170)
        ]),
    )
    annotation[NameObject("/Contents")] = TextStringObject(
        "HIGHLIGHT CRITICAL 73500"
    )
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(str(source)))
    writer.add_annotation(page_number=0, annotation=annotation)
    path = tmp_path / "highlight.pdf"
    with path.open("wb") as stream:
        writer.write(stream)

    result = read_pdf(path)
    configured = read_document(path)

    for candidate in (result, configured):
        assert "HIGHLIGHT CRITICAL 73500" in "\n".join(
            element.text or "" for element in candidate.elements
        )
        assert candidate.admission_status in {
            ProcessingAdmissionStatus.COMPLETE,
            ProcessingAdmissionStatus.PARTIAL,
        }
    if configured.processor.reader == "docling":
        assert "semantic_annotation_inventory" in (
            configured.page_manifest.detector_capabilities
        )
    assert result.admission_status is ProcessingAdmissionStatus.COMPLETE


def test_annotation_inventory_open_failure_keeps_primary_parse_as_partial(
    text_pdf,
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    primary = read_pdf(text_pdf)

    def fail_open(_path):
        raise RuntimeError("private annotation parser failure")

    monkeypatch.setattr("pdfplumber.open", fail_open)

    result = pdf_reader.merge_pdf_annotation_inventory(text_pdf, primary)

    assert result.elements == primary.elements
    assert result.page_manifest == primary.page_manifest
    assert DiagnosticCode.CORRUPT_SOURCE not in _codes(result)
    assert DiagnosticCode.PARSER_PARTIAL in _codes(result)
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_annotation_inventory_page_count_drift_keeps_primary_parse_as_partial(
    text_pdf,
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    primary = read_pdf(text_pdf)

    class _Pdf:
        pages = (object(), object())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr("pdfplumber.open", lambda _path: _Pdf())

    result = pdf_reader.merge_pdf_annotation_inventory(text_pdf, primary)

    assert result.elements == primary.elements
    assert result.page_manifest == primary.page_manifest
    assert DiagnosticCode.CORRUPT_SOURCE not in _codes(result)
    assert DiagnosticCode.PARSER_PARTIAL in _codes(result)
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_annotation_inventory_page_failure_keeps_primary_page_readable(
    text_pdf,
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    primary = read_pdf(text_pdf)

    class _Page:
        @property
        def annots(self):
            raise RuntimeError("private annotation object failure")

    class _Pdf:
        pages = (_Page(),)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr("pdfplumber.open", lambda _path: _Pdf())

    result = pdf_reader.merge_pdf_annotation_inventory(text_pdf, primary)

    assert result.elements == primary.elements
    assert result.page_manifest.pages[0].state is DocumentPageState.TEXT
    assert DiagnosticCode.CORRUPT_SOURCE not in _codes(result)
    assert DiagnosticCode.PARSER_PARTIAL in _codes(result)
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_docling_primary_read_remains_admissible_when_annotation_pass_fails(
    text_pdf,
    monkeypatch,
):
    from personagraph.input_processing.documents import readers
    from personagraph.input_processing.documents.readers import (
        registry,
    )

    native_primary = read_pdf(text_pdf)
    docling_primary = ProcessingResult(
        elements=native_primary.elements,
        processor=ProcessorFingerprint("docling", "test"),
        diagnostics=native_primary.diagnostics,
        page_manifest=native_primary.page_manifest,
    )

    def primary_reader(_path):
        return docling_primary

    def fail_open(_path):
        raise RuntimeError("private annotation parser failure")

    monkeypatch.setattr(registry, "read_with_docling", primary_reader)
    monkeypatch.setattr(
        registry,
        "_configured_pdf_reader",
        lambda _path, _engine: (primary_reader, None),
    )
    monkeypatch.setattr(
        registry,
        "_reader_processor_fingerprint",
        lambda _reader: ProcessorFingerprint("docling", "test"),
    )
    monkeypatch.setattr(registry, "_pdf_requires_password", lambda _path: False)
    monkeypatch.setattr(registry, "_pdf_has_xfa", lambda _path: False)
    monkeypatch.setattr(
        registry,
        "merge_pdf_native_table_inventory",
        lambda _path, result: result,
    )
    monkeypatch.setattr("pdfplumber.open", fail_open)

    result = readers.read_document(text_pdf)

    assert result.elements == docling_primary.elements
    assert result.page_manifest == docling_primary.page_manifest
    assert DiagnosticCode.CORRUPT_SOURCE not in _codes(result)
    assert DiagnosticCode.PARSER_PARTIAL in _codes(result)
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL


def test_native_page_annotation_failure_does_not_erase_readable_body_text():
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    class _Page:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

        @property
        def annots(self):
            raise RuntimeError("private annotation object failure")

        def extract_tables(self):
            return []

        def extract_text_lines(self):
            return [{"text": "Readable body evidence"}]

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _Page(),
        1,
        "source",
        elements=elements,
        diagnostics=diagnostics,
    )

    assert [element.text for element in elements] == ["Readable body evidence"]
    assert page.state is DocumentPageState.TEXT
    assert {diagnostic.code for diagnostic in diagnostics} == {
        DiagnosticCode.PARSER_PARTIAL
    }


def test_empty_native_page_keeps_annotation_failure_in_page_diagnostics():
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    class _Page:
        chars = []
        images = []
        curves = []
        rects = []
        lines = []

        @property
        def annots(self):
            raise RuntimeError("private annotation object failure")

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _Page(),
        1,
        "source",
        elements=elements,
        diagnostics=diagnostics,
    )

    expected = {DiagnosticCode.PAGE_EMPTY, DiagnosticCode.PARSER_PARTIAL}
    assert {diagnostic.code for diagnostic in diagnostics} == expected
    assert {diagnostic.code for diagnostic in page.diagnostics} == expected


def test_page_classification_uses_what_the_parser_found():
    assert classify_page(MIN_TEXT_CHARS_PER_PAGE, 0) is PageKind.TEXT
    assert classify_page(0, 1) is PageKind.SCANNED
    assert classify_page(MIN_TEXT_CHARS_PER_PAGE, 2) is PageKind.MIXED
    assert classify_page(0, 0) is PageKind.EMPTY
    # 扫描件上的少量零散字形不得被视为文本层。
    assert classify_page(MIN_TEXT_CHARS_PER_PAGE - 1, 1) is PageKind.SCANNED
    assert classify_page(0, 0, vector_count=1) is PageKind.SCANNED


def test_pdf_page_cap_rejects_before_enumerating_unbounded_page_records(
    tmp_path,
    monkeypatch,
):
    import pdfplumber
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    path = tmp_path / "too-many-pages.pdf"
    path.write_bytes(b"%PDF-1.7\ncontrolled-test")

    class _Pages:
        def __len__(self):
            return pdf_reader.MAX_PAGES + 1

        def __iter__(self):
            raise AssertionError("over-limit pages must not be enumerated")

    class _Pdf:
        pages = _Pages()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(pdfplumber, "open", lambda _path: _Pdf())

    result = read_pdf(path)

    assert result.elements == ()
    assert result.page_manifest is None
    assert _codes(result) == {DiagnosticCode.LIMIT_REACHED}
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_table_extraction_failure_and_text_truncation_are_never_silent(monkeypatch):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    class _BrokenTablePage:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

        def extract_tables(self):
            raise RuntimeError("private table text")

        def extract_text_lines(self):
            return [{"text": "x" * (pdf_reader.MAX_ELEMENT_CHARS + 1)}]

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _BrokenTablePage(),
        1,
        "source",
        elements=elements,
        diagnostics=diagnostics,
    )

    assert page.state is DocumentPageState.UNREADABLE
    assert {diagnostic.code for diagnostic in diagnostics} == {
        DiagnosticCode.CORRUPT_SOURCE,
        DiagnosticCode.LIMIT_REACHED,
    }


def test_vector_only_page_is_a_visible_unread_nontext_unit():
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    class _VectorPage:
        chars = []
        images = []
        curves = [{"x0": 0, "top": 0, "x1": 20, "bottom": 20}]
        rects = []
        lines = []
        width = 100
        height = 100

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _VectorPage(), 1, "source", elements=elements, diagnostics=diagnostics
    )

    assert page.state is DocumentPageState.VISUAL_ONLY
    assert page.nontext_units[0].kind is DocumentNonTextKind.VECTOR_GRAPHICS
    assert DiagnosticCode.PAGE_NEEDS_VISION in {d.code for d in diagnostics}


def _assert_budget_failure(elements, diagnostics, page, expected_cap):
    assert len(elements) == expected_cap
    assert page.state is DocumentPageState.UNREADABLE
    assert DiagnosticCode.LIMIT_REACHED in {d.code for d in diagnostics}
    result = ProcessingResult(
        elements=tuple(elements),
        processor=ProcessorFingerprint("test_pdf", "1"),
        diagnostics=tuple(diagnostics),
    )
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_native_document_element_budget_scales_with_page_count_but_stays_bounded():
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    assert pdf_reader._document_element_limit(1) == pdf_reader.MAX_ELEMENTS
    first_scaled_page_count = (
        pdf_reader.MAX_ELEMENTS // pdf_reader.MAX_ELEMENTS_PER_PAGE + 1
    )
    assert pdf_reader._document_element_limit(first_scaled_page_count) == (
        first_scaled_page_count * pdf_reader.MAX_ELEMENTS_PER_PAGE
    )
    assert (
        pdf_reader._document_element_limit(pdf_reader.MAX_PAGES)
        == pdf_reader.MAX_DOCUMENT_ELEMENTS
    )


def test_native_reader_closes_each_page_cache_even_after_element_budget_exhaustion(
    tmp_path,
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    class _Page:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

        def __init__(self):
            self.closed = 0

        def extract_tables(self):
            return []

        def extract_text_lines(self):
            return [{"text": "one admitted line"}]

        def close(self):
            self.closed += 1

    pages = [_Page(), _Page()]

    class _Pdf:
        def __init__(self):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    path = tmp_path / "cache-probe.pdf"
    path.write_bytes(b"bounded source bytes")
    monkeypatch.setattr("pdfplumber.open", lambda _path: _Pdf())
    monkeypatch.setattr(pdf_reader, "_document_element_limit", lambda _pages: 1)

    result = pdf_reader.read_pdf(path)

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED
    assert [page.closed for page in pages] == [1, 1]


def test_one_pdf_page_cannot_exceed_the_element_budget_while_emitting_tables(
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    monkeypatch.setattr(pdf_reader, "MAX_ELEMENTS", 2)

    class _ManyTablesPage:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

        def extract_tables(self):
            return [[["table one"]], [["table two"]], [["table three"]]]

        def extract_text_lines(self):
            return [{"text": "a text line that must not cross the cap"}]

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _ManyTablesPage(), 1, "source", elements=elements, diagnostics=diagnostics
    )

    _assert_budget_failure(elements, diagnostics, page, 2)


def test_one_pdf_page_cannot_exceed_the_element_budget_while_emitting_text_lines(
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    monkeypatch.setattr(pdf_reader, "MAX_ELEMENTS", 2)

    class _ManyLinesPage:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

        def extract_tables(self):
            return []

        def extract_text_lines(self):
            return [
                {"text": "line one"},
                {"text": "line two"},
                {"text": "line three"},
            ]

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _ManyLinesPage(), 1, "source", elements=elements, diagnostics=diagnostics
    )

    _assert_budget_failure(elements, diagnostics, page, 2)


def test_one_pdf_page_cannot_exceed_the_element_budget_while_emitting_images(
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    monkeypatch.setattr(pdf_reader, "MAX_ELEMENTS", 2)

    class _ManyImagesPage:
        chars = []
        images = [{}, {}, {}]
        curves = []
        rects = []
        lines = []
        width = 100
        height = 100

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _ManyImagesPage(), 1, "source", elements=elements, diagnostics=diagnostics
    )

    _assert_budget_failure(elements, diagnostics, page, 2)


def test_one_pdf_page_cannot_exceed_the_element_budget_on_vector_emission(
    monkeypatch,
):
    from personagraph.input_processing.documents.readers import pdf as pdf_reader

    monkeypatch.setattr(pdf_reader, "MAX_ELEMENTS", 1)

    class _TextAndVectorPage:
        chars = [{}] * MIN_TEXT_CHARS_PER_PAGE
        images = []
        curves = [{}]
        rects = []
        lines = []
        width = 100
        height = 100

        def extract_tables(self):
            return []

        def extract_text_lines(self):
            return [{"text": "line consumes the only element slot"}]

    elements = []
    diagnostics = []
    page = pdf_reader._read_page(
        _TextAndVectorPage(), 1, "source", elements=elements, diagnostics=diagnostics
    )

    _assert_budget_failure(elements, diagnostics, page, 1)
