"""为分析准备图片，并覆盖所有可能的失败方式。

这些测试强制执行一条规则：调用方请求无法发送的图片时会得到原因，绝不会收到
异常，也绝不会在不知情的情况下得到另一张图片。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from personagraph.input_processing.vision.imaging import payload as payload_module
from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionDetail,
    VisionPurpose,
    VisionRequest,
)
from personagraph.input_processing.vision.imaging import (
    PayloadFailure,
    PayloadRefusal,
    VisionPayload,
    prepare_payload,
)


def _request(path, detail: VisionDetail = VisionDetail.STANDARD) -> VisionRequest:
    return VisionRequest(
        source_unit_id="figure-1",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
        locator=DocumentLocator(page=2),
        mime_type="image/png",
        pixel_size=PixelSize(10, 10),
        byte_count=1,
        purpose=VisionPurpose.CHART,
        prompt_contract_version="vision-purpose-v1",
        image_path=str(path),
        detail=detail,
    )


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    target = tmp_path / "figure.png"
    Image.new("RGB", (2400, 1600), "white").save(target)
    return target


def test_an_empty_path_is_refused_by_the_contract_before_any_load():
    with pytest.raises(ValueError):
        _request("   ")


@pytest.mark.parametrize(
    "make_path, expected",
    [
        (lambda tmp: tmp / "absent.png", PayloadFailure.PATH_NOT_A_FILE),
        (lambda tmp: tmp, PayloadFailure.PATH_NOT_A_FILE),
    ],
    ids=["missing-file", "directory"],
)
def test_an_unusable_path_comes_back_as_a_reason(tmp_path, make_path, expected):
    result = prepare_payload(_request(make_path(tmp_path)))
    assert isinstance(result, PayloadRefusal)
    assert result.failure is expected


def test_a_file_that_is_not_an_image_is_refused_without_raising(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("this is not a picture", encoding="utf-8")
    result = prepare_payload(_request(target))
    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.UNSUPPORTED_FORMAT


def test_an_empty_file_is_refused(tmp_path: Path):
    target = tmp_path / "empty.png"
    target.write_bytes(b"")
    result = prepare_payload(_request(target))
    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.DECODE_FAILED


def test_a_small_image_is_sent_unchanged(tmp_path: Path):
    target = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(target)
    payload = prepare_payload(_request(target))
    assert isinstance(payload, VisionPayload)
    assert payload.resampled is False
    assert payload.pixel_size == PixelSize(400, 300)
    assert payload.descriptor == "400x300"


@pytest.mark.parametrize(
    "detail, ceiling",
    [
        (VisionDetail.LOW, 350_000),
        (VisionDetail.STANDARD, 1_000_000),
        (VisionDetail.HIGH, 2_000_000),
    ],
)
def test_each_detail_level_bounds_the_pixels_actually_sent(picture, detail, ceiling):
    payload = prepare_payload(_request(picture, detail))
    assert isinstance(payload, VisionPayload)
    assert payload.resampled is True
    sent = payload.pixel_size.width * payload.pixel_size.height
    assert sent <= ceiling
    # 重采样结果必须接近预算，而不能远低于预算；若某档位静默浪费大部分额度，
    # 只会降低可读性，却没有任何节省。
    assert sent > ceiling * 0.8


def test_resampling_makes_the_sent_image_a_different_object_than_the_source(picture):
    payload = prepare_payload(_request(picture))
    assert isinstance(payload, VisionPayload)
    assert payload.source_sha256 == hashlib.sha256(picture.read_bytes()).hexdigest()
    assert payload.sent_sha256 != payload.source_sha256
    assert "resampled" in payload.descriptor


def test_a_webp_source_is_converted_rather_than_refused(tmp_path: Path):
    target = tmp_path / "photo.webp"
    Image.new("RGB", (640, 480), "white").save(target, "WEBP")
    payload = prepare_payload(_request(target))
    assert isinstance(payload, VisionPayload)
    assert payload.mime_type == "image/png"


def test_an_oversized_source_is_refused_before_it_is_resampled(tmp_path, monkeypatch):
    """边界限制不仅适用于发送，也适用于解码。"""

    monkeypatch.setattr(
        "personagraph.input_processing.vision.imaging.payload.MAX_SOURCE_PIXELS",
        1000,
    )
    target = tmp_path / "huge.png"
    Image.new("RGB", (200, 200), "white").save(target)
    result = prepare_payload(_request(target))
    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.SOURCE_TOO_LARGE


def test_preparation_is_deterministic(picture):
    first = prepare_payload(_request(picture))
    second = prepare_payload(_request(picture))
    assert first.sent_sha256 == second.sent_sha256


# --- PDF 区域 -------------------------------------------------------------
#
# PDF 内的图形没有独立文件。它按需渲染，因此未读取的图形在摄取时没有成本；
# 也正因如此，实际渲染的区域必须准确。


IMAGE_RECT = (60.0, 230.0, 535.0, 560.0)  # 图片实际所在位置，单位为点


@pytest.fixture
def pdf_with_figure(tmp_path: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    picture = tmp_path / "figure.png"
    Image.new("RGB", (1600, 1000), "white").save(picture)

    target = tmp_path / "report.pdf"
    width, height = A4
    sheet = canvas.Canvas(str(target), pagesize=A4)
    sheet.setFont("Helvetica", 11)
    sheet.drawString(60, height - 80, "Regional report")
    sheet.drawImage(
        str(picture),
        IMAGE_RECT[0],
        height - IMAGE_RECT[3],
        width=IMAGE_RECT[2] - IMAGE_RECT[0],
        height=IMAGE_RECT[3] - IMAGE_RECT[1],
    )
    sheet.showPage()
    sheet.save()
    return target


def _pdf_request(path: Path, **overrides) -> VisionRequest:
    from personagraph.input_processing.vision.contracts import VisionRegion

    payload = {
        "source_unit_id": "figure-1",
        "source_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "locator": DocumentLocator(page=1, bbox=(76.0, 305.0, 453.0, 597.0)),
        "mime_type": "image/png",
        "pixel_size": PixelSize(475, 330),
        "byte_count": 1,
        "purpose": VisionPurpose.CHART,
        "prompt_contract_version": "vision-purpose-v1",
        "image_path": str(path),
        "detail": VisionDetail.STANDARD,
        "region": VisionRegion.DETECTED,
    }
    payload.update(overrides)
    return VisionRequest(**payload)


def test_a_pdf_page_region_is_rendered_on_demand(pdf_with_figure: Path):
    payload = prepare_payload(_pdf_request(pdf_with_figure))
    assert isinstance(payload, VisionPayload)
    assert payload.mime_type == "image/png"
    assert payload.pixel_size.width > 1 and payload.pixel_size.height > 1


def test_a_pdf_is_stream_fingerprinted_without_the_ordinary_image_byte_gate(
    pdf_with_figure: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """渲染其中一页之前，不得把大型 PDF 整体复制到内存。"""

    original_read_bytes = Path.read_bytes
    target = pdf_with_figure.resolve()

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == target:
            raise AssertionError("PDF source must not be loaded with Path.read_bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(payload_module, "MAX_SOURCE_BYTES", 1)

    payload = prepare_payload(_pdf_request(pdf_with_figure))

    assert isinstance(payload, VisionPayload)
    assert payload.source_sha256 == hashlib.sha256(
        original_read_bytes(pdf_with_figure)
    ).hexdigest()


def test_the_ordinary_image_byte_gate_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "oversized.png"
    target.write_bytes(b"not-an-image")
    monkeypatch.setattr(payload_module, "MAX_SOURCE_BYTES", 1)

    result = prepare_payload(_request(target))

    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.SOURCE_TOO_LARGE


def test_the_exact_image_rectangle_beats_the_detector_box(pdf_with_figure: Path):
    """检测器报告它认为内容所在的位置；文件则记录内容的实际位置。

    在图表上测量时，检测框比真实矩形低 75 点，并截掉了最高柱的标注值。
    """

    payload = prepare_payload(_pdf_request(pdf_with_figure))
    # 真实图片为 475×330 点；检测器声称是 377×292。渲染检测框会得到检测框的
    # 宽高比，而不是真实图片的宽高比。
    ratio = payload.pixel_size.width / payload.pixel_size.height
    expected = (IMAGE_RECT[2] - IMAGE_RECT[0]) / (IMAGE_RECT[3] - IMAGE_RECT[1])
    assert abs(ratio - expected) < 0.05


def test_each_region_rung_covers_more_than_the_last(pdf_with_figure: Path):
    from personagraph.input_processing.vision.contracts import VisionRegion

    def aspect(region):
        payload = prepare_payload(_pdf_request(pdf_with_figure, region=region))
        return payload.pixel_size.width / payload.pixel_size.height

    # 每一级都会向页面上下延伸得更远，因此渲染区域逐级变得相对更高。
    detected = aspect(VisionRegion.DETECTED)
    expanded = aspect(VisionRegion.EXPANDED)
    page = aspect(VisionRegion.PAGE)
    assert detected > expanded > page


def test_a_page_that_does_not_exist_is_refused_with_its_reason(pdf_with_figure: Path):
    result = prepare_payload(
        _pdf_request(pdf_with_figure, locator=DocumentLocator(page=99, bbox=(10, 10, 50, 50)))
    )
    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.PAGE_OUT_OF_RANGE


def test_a_locator_with_no_box_renders_the_whole_page(pdf_with_figure: Path):
    payload = prepare_payload(_pdf_request(pdf_with_figure, locator=DocumentLocator(page=1)))
    assert isinstance(payload, VisionPayload)
    # A4 高于宽；整页渲染必须反映这一点。
    assert payload.pixel_size.height > payload.pixel_size.width


def test_a_box_that_runs_off_the_page_is_clipped_rather_than_refused(pdf_with_figure: Path):
    payload = prepare_payload(
        _pdf_request(pdf_with_figure, locator=DocumentLocator(page=1, bbox=(500, 700, 900, 1200)))
    )
    assert isinstance(payload, VisionPayload)


def test_a_corrupt_pdf_is_refused_without_raising(tmp_path: Path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a pdf\n")
    result = prepare_payload(_pdf_request(broken))
    assert isinstance(result, PayloadRefusal)
    assert result.failure is PayloadFailure.DECODE_FAILED


def test_rendering_the_same_region_twice_is_deterministic(pdf_with_figure: Path):
    first = prepare_payload(_pdf_request(pdf_with_figure))
    second = prepare_payload(_pdf_request(pdf_with_figure))
    assert first.sent_sha256 == second.sent_sha256
