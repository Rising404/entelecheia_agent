from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfWriter

from personagraph.input_processing.documents import (
    DiagnosticCode,
    DocumentNonTextKind,
    DocumentPageState,
    ProcessorFingerprint,
)
from personagraph.input_processing.documents.readers import docling_reader
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionDetail,
    VisionPurpose,
    VisionRegion,
    VisionRequest,
)
from personagraph.input_processing.vision.imaging import VisionPayload, prepare_payload


class _Document:
    def __init__(self, *, pages, items):
        self.pages = pages
        self._items = items

    def iterate_items(self):
        return iter((item, 0) for item in self._items)


def _item(label: str, text: str, page: int):
    provenance = SimpleNamespace(page_no=page, bbox=None, charspan=None)
    return SimpleNamespace(label=label, text=text, prov=[provenance], level=1)


def _write_pdf(path, *page_sizes):
    writer = PdfWriter()
    for width, height in page_sizes or ((612, 792),):
        writer.add_blank_page(width=width, height=height)
    with path.open("wb") as target:
        writer.write(target)


def _write_bbox_origin_probe_pdf(path):
    from reportlab.pdfgen import canvas

    width, height = 595.28, 841.89
    sheet = canvas.Canvas(str(path), pagesize=(width, height))
    for page_number in range(1, 8):
        if page_number == 7:
    # 红色靠近页面视觉顶部，蓝色靠近底部。因此，若误把左下原点边界框当作
    # 左上原点使用，就会选中蓝色带；正确转换后则会选中红色带。
            sheet.setFillColorRGB(1, 0, 0)
            sheet.rect(0, height - 190, width, 150, stroke=0, fill=1)
            sheet.setFillColorRGB(0, 0, 1)
            sheet.rect(0, height - 800, width, 150, stroke=0, fill=1)
        sheet.showPage()
    sheet.save()


def _fake_recipe():
    identity = {
        "resource_limits": {
            "max_file_size_bytes": docling_reader.MAX_DOCUMENT_FILE_BYTES,
            "max_num_pages": docling_reader.MAX_DOCUMENT_PAGES,
            "pdf_render_geometry": {
                "algorithm": docling_reader.PDF_RENDER_PREFLIGHT_ALGORITHM,
                "backend_supersampling_factor": (
                    docling_reader.PDF_BACKEND_SUPERSAMPLING_FACTOR
                ),
                "max_page_pixels": docling_reader.MAX_RENDER_PAGE_PIXELS,
                "max_side_pixels": docling_reader.MAX_RENDER_SIDE_PIXELS,
                "max_total_pixels": docling_reader.MAX_RENDER_TOTAL_PIXELS,
                "render_scale": 3.0,
            },
        }
    }
    return docling_reader.DoclingProcessorRecipe(json.dumps(identity))


def _install_fake(
    monkeypatch,
    document,
    *,
    status="success",
    errors=(),
    calls=None,
):
    def convert(
        _path,
        *,
        max_num_pages,
        max_file_size,
        raises_on_error,
    ):
        assert max_num_pages == docling_reader.MAX_DOCUMENT_PAGES
        assert max_file_size == docling_reader.MAX_DOCUMENT_FILE_BYTES
        assert raises_on_error is False
        if calls is not None:
            calls.append(_path)
        return SimpleNamespace(
            document=document,
            errors=list(errors),
            status=status,
        )

    converter = SimpleNamespace(convert=convert)
    monkeypatch.setattr(
        docling_reader,
        "configured_docling_recipe",
        _fake_recipe,
    )
    monkeypatch.setattr(docling_reader, "_converter", lambda *_args: converter)
    monkeypatch.setattr(
        docling_reader,
        "fingerprint",
        lambda *_args: ProcessorFingerprint("docling-test", "1"),
    )


def test_docling_records_formula_table_figure_and_empty_physical_pages(tmp_path, monkeypatch):
    document = _Document(
        pages={page: SimpleNamespace(page_no=page) for page in range(1, 5)},
        items=[
            _item("formula", "E = mc^2", 1),
            _item("table", "A | B", 2),
            _item("picture", "", 3),
        ],
    )
    _install_fake(monkeypatch, document)
    path = tmp_path / "paper.pdf"
    _write_pdf(path, *((612, 792),) * 4)

    result = docling_reader.read_with_docling(path)

    manifest = result.page_manifest
    assert manifest is not None
    assert manifest.physical_page_count == 4
    assert [page.state for page in manifest.pages] == [
        DocumentPageState.MIXED,
        DocumentPageState.MIXED,
        DocumentPageState.VISUAL_ONLY,
        DocumentPageState.NO_EXTRACTABLE_CONTENT,
    ]
    assert manifest.pages[0].nontext_units[0].kind is DocumentNonTextKind.FORMULA
    assert manifest.pages[1].nontext_units[0].kind is DocumentNonTextKind.TABLE
    assert manifest.pages[2].nontext_units[0].kind is DocumentNonTextKind.FIGURE
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        DiagnosticCode.PAGE_NEEDS_VISION,
        DiagnosticCode.PAGE_EMPTY,
    }


def test_bottom_left_docling_bbox_is_top_left_before_vision_crop(
    tmp_path,
    monkeypatch,
):
    """Docling 第 7 页的几何信息必须选中视觉顶部，而不是其镜像位置。"""

    page_width, page_height = 595.28, 841.89
    provenance = SimpleNamespace(
        page_no=7,
        bbox=SimpleNamespace(
            l=80.0,
            t=780.034562,
            r=300.0,
            b=672.173844,
            coord_origin=SimpleNamespace(value="BOTTOMLEFT"),
        ),
        charspan=None,
    )
    item = SimpleNamespace(
        label="picture",
        text="",
        prov=[provenance],
        level=1,
    )
    document = _Document(
        pages={
            page: SimpleNamespace(
                page_no=page,
                size=SimpleNamespace(width=page_width, height=page_height),
            )
            for page in range(1, 8)
        },
        items=[item],
    )
    _install_fake(monkeypatch, document)
    path = tmp_path / "bbox-origin-probe.pdf"
    _write_bbox_origin_probe_pdf(path)

    result = docling_reader.read_with_docling(path)

    unit = result.page_manifest.pages[6].nontext_units[0]
    assert unit.locator.bbox == pytest.approx(
        (80.0, 61.855438, 300.0, 169.716156)
    )
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = prepare_payload(VisionRequest(
        source_unit_id=unit.unit_id,
        source_sha256=source_sha256,
        image_sha256=source_sha256,
        locator=unit.locator,
        mime_type="image/png",
        pixel_size=PixelSize(220, 108),
        byte_count=path.stat().st_size,
        purpose=VisionPurpose.GENERAL,
        prompt_contract_version="docling-bbox-origin-v1",
        image_path=str(path),
        detail=VisionDetail.LOW,
        region=VisionRegion.DETECTED,
    ))
    assert isinstance(payload, VisionPayload)
    with Image.open(io.BytesIO(payload.data)) as sent:
        red, green, blue = sent.getpixel((sent.width // 2, sent.height // 2))
    assert red > 240 and green < 15 and blue < 15


def test_docling_bbox_origin_metadata_is_honoured_or_dropped_safely():
    page_height = 841.89
    top_left = SimpleNamespace(
        bbox=SimpleNamespace(
            l=80.0,
            t=61.855438,
            r=300.0,
            b=169.716156,
            coord_origin=SimpleNamespace(value="TOPLEFT"),
        )
    )
    bottom_left = SimpleNamespace(
        bbox=SimpleNamespace(
            l=80.0,
            t=780.034562,
            r=300.0,
            b=672.173844,
            coord_origin=SimpleNamespace(value="BOTTOMLEFT"),
        )
    )

    assert docling_reader._bbox_of(
        top_left,
        page_height=page_height,
    ) == pytest.approx((80.0, 61.855438, 300.0, 169.716156))
    assert docling_reader._bbox_of(bottom_left, page_height=None) is None
    assert docling_reader._bbox_of(
        SimpleNamespace(bbox=SimpleNamespace(
            l=80.0,
            t=61.855438,
            r=300.0,
            b=169.716156,
        )),
        page_height=page_height,
    ) is None


def test_docling_refuses_to_claim_complete_when_physical_page_inventory_is_unavailable(
    tmp_path, monkeypatch
):
    _install_fake(
        monkeypatch,
        SimpleNamespace(iterate_items=lambda: iter((_item("text", "body", 1), 0))),
    )
    path = tmp_path / "paper.pdf"
    _write_pdf(path, (612, 792))

    result = docling_reader.read_with_docling(path)

    assert result.page_manifest is None
    assert result.is_complete is False
    assert result.admission_status.value == "rejected"
    assert DiagnosticCode.CORRUPT_SOURCE in {
        diagnostic.code for diagnostic in result.diagnostics
    }


@pytest.mark.parametrize(
    ("page_sizes", "expected_detail"),
    [
        (((4_000, 100),), "pdf_render_side_limit_reached:page=1"),
        (((1_600, 1_600),), "pdf_render_page_pixel_limit_reached:page=1"),
        (((1_300, 1_300),) * 12, "pdf_render_total_pixel_limit_reached:page=12"),
    ],
)
def test_docling_rejects_oversized_pdf_render_geometry_before_converter_io(
    tmp_path,
    monkeypatch,
    page_sizes,
    expected_detail,
):
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "converter must not see this", 1)],
    )
    calls = []
    _install_fake(monkeypatch, document, calls=calls)
    path = tmp_path / "giant-media-box.pdf"
    _write_pdf(path, *page_sizes)

    result = docling_reader.read_with_docling(path)

    assert calls == []
    assert result.elements == ()
    assert result.admission_status.value == "rejected"
    assert result.diagnostics == (
        docling_reader.ProcessingDiagnostic(
            DiagnosticCode.LIMIT_REACHED,
            detail=expected_detail,
        ),
    )


def test_partial_success_preserves_safe_error_shape_without_document_text_leak(
    tmp_path,
    monkeypatch,
):
    secret = "SECRET_DOCUMENT_BODY_MUST_NOT_ESCAPE"
    error = SimpleNamespace(
        category=SimpleNamespace(value="inference_failure"),
        component_type=SimpleNamespace(value="model"),
        page_no=1,
        error_message=secret,
        module_name=secret,
    )
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "usable extracted text", 1)],
    )
    _install_fake(
        monkeypatch,
        document,
        status="partial_success",
        errors=(error,),
    )
    path = tmp_path / "partial.pdf"
    _write_pdf(path, (612, 792))

    result = docling_reader.read_with_docling(path)

    assert result.text_elements()
    assert result.admission_status.value == "partial"
    partials = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is DiagnosticCode.PARSER_PARTIAL
    ]
    assert len(partials) == 1
    assert partials[0].locator.page == 1
    assert partials[0].detail == (
        "docling_status=partial_success;category=inference_failure;component=model"
    )
    assert secret not in repr(result)


def test_partial_success_without_error_records_is_still_typed_partial(
    tmp_path,
    monkeypatch,
):
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "usable extracted text", 1)],
    )
    _install_fake(monkeypatch, document, status="partial_success")
    path = tmp_path / "partial-without-error-records.pdf"
    _write_pdf(path, (612, 792))

    result = docling_reader.read_with_docling(path)

    assert result.admission_status.value == "partial"
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        DiagnosticCode.PARSER_PARTIAL
    ]


def test_success_status_with_error_records_is_not_silently_complete(
    tmp_path,
    monkeypatch,
):
    error = SimpleNamespace(
        category=SimpleNamespace(value="timeout"),
        component_type=SimpleNamespace(value="pipeline"),
        page_no=None,
        error_message="source-derived free-form detail",
        module_name="source-derived free-form module",
    )
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "usable extracted text", 1)],
    )
    _install_fake(monkeypatch, document, status="success", errors=(error,))
    path = tmp_path / "success-with-errors.pdf"
    _write_pdf(path, (612, 792))

    result = docling_reader.read_with_docling(path)

    assert result.admission_status.value == "partial"
    assert result.diagnostics[0].code is DiagnosticCode.PARSER_PARTIAL
    assert result.diagnostics[0].detail == (
        "docling_status=success;category=timeout;component=pipeline"
    )


@pytest.mark.parametrize("suffix", [".xlsx", ".html"])
def test_non_pdf_docling_inputs_do_not_run_pdf_geometry_preflight(
    tmp_path,
    monkeypatch,
    suffix,
):
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "non-PDF content", 1)],
    )
    calls = []
    _install_fake(monkeypatch, document, calls=calls)
    path = tmp_path / f"source{suffix}"
    path.write_bytes(b"not a PDF container")

    result = docling_reader.read_with_docling(path)

    assert len(calls) == 1
    assert [element.text for element in result.text_elements()] == [
        "non-PDF content"
    ]
    assert result.admission_status.value == "complete"


@pytest.mark.parametrize("status", ["failure", "skipped"])
def test_non_successful_docling_terminal_status_is_rejected(
    tmp_path,
    monkeypatch,
    status,
):
    secret = "SECRET_FAILURE_BODY_MUST_NOT_ESCAPE"
    error = SimpleNamespace(
        category=SimpleNamespace(value="backend_failure"),
        component_type=SimpleNamespace(value="document_backend"),
        page_no=None,
        error_message=secret,
        module_name=secret,
    )
    document = _Document(
        pages={1: SimpleNamespace(page_no=1)},
        items=[_item("text", "must not be admitted", 1)],
    )
    _install_fake(monkeypatch, document, status=status, errors=(error,))
    path = tmp_path / f"{status}.pdf"
    _write_pdf(path, (612, 792))

    result = docling_reader.read_with_docling(path)

    assert result.elements == ()
    assert result.page_manifest is None
    assert result.admission_status.value == "rejected"
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        DiagnosticCode.CORRUPT_SOURCE
    }
    assert secret not in repr(result)
