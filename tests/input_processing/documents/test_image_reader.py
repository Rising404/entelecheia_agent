from __future__ import annotations

import hashlib

import pytest
from PIL import Image, ImageOps

from personagraph.input_processing.documents.contracts import (
    DiagnosticCode,
    DocumentNonTextKind,
    DocumentPageManifest,
    DocumentPageState,
    DocumentTextEvidenceOrigin,
    ElementKind,
    ProcessingAdmissionStatus,
)
from personagraph.input_processing.documents.readers.image import (
    configured_image_processor_fingerprint,
    read_image,
)
from personagraph.input_processing.vision.contracts import (
    OcrFailureCode,
    OcrLine,
    OcrResult,
    OcrStatus,
)


class _OcrService:
    engine_fingerprint = "fixture-ocr@1"

    def __init__(self, status=OcrStatus.SUCCESS):
        self.status = status
        self.requests = []
        self.image_sizes = []

    def recognize(self, image, request):
        self.requests.append(request)
        self.image_sizes.append(image.size)
        if self.status is OcrStatus.SUCCESS:
            width, height = image.size
            line = OcrLine(
                text="OCR 可读取文字",
                confidence=0.95,
                bbox_norm=(0.1, 0.2, 0.9, 0.4),
                bbox_px=(0.1 * width, 0.2 * height, 0.9 * width, 0.4 * height),
            )
            return OcrResult.from_request(
                request,
                status=OcrStatus.SUCCESS,
                engine_fingerprint=self.engine_fingerprint,
                lines=(line,),
                elapsed_ms=3,
            )
        if self.status is OcrStatus.BLANK:
            return OcrResult.from_request(
                request,
                status=OcrStatus.BLANK,
                engine_fingerprint=self.engine_fingerprint,
                elapsed_ms=2,
            )
        return OcrResult.from_request(
            request,
            status=OcrStatus.FAILED,
            engine_fingerprint=self.engine_fingerprint,
            elapsed_ms=1,
            failure_code=OcrFailureCode.BACKEND_FAILED,
            warnings=("ocr_backend_failed",),
        )


def _write_image(path, *, format, size=(80, 40), exif=None):
    image = Image.new("RGB", size, "white")
    kwargs = {"exif": exif} if exif is not None else {}
    image.save(path, format=format, **kwargs)


@pytest.mark.parametrize(("suffix", "format"), [(".png", "PNG"), (".jpg", "JPEG")])
def test_image_reader_emits_ocr_text_and_keeps_visual_semantics_gap(
    tmp_path, suffix, format
):
    path = tmp_path / f"source{suffix}"
    _write_image(path, format=format)
    service = _OcrService()

    result = read_image(path, ocr_service=service)

    text = [element for element in result.elements if element.text]
    images = [element for element in result.elements if element.kind is ElementKind.IMAGE]
    assert [element.text for element in text] == ["OCR 可读取文字"]
    assert text[0].locator.page == 1
    assert text[0].locator.bbox == (0.1, 0.2, 0.9, 0.4)
    assert len(images) == 1 and images[0].needs_vision is True
    assert result.needs_vision is True
    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert {item.code for item in result.diagnostics} == {
        DiagnosticCode.PAGE_NEEDS_VISION
    }
    page = result.page_manifest.pages[0]
    assert DocumentPageManifest.from_json(
        result.page_manifest.canonical_json()
    ) == result.page_manifest
    assert page.state is DocumentPageState.MIXED
    assert page.nontext_units[0].kind is DocumentNonTextKind.FIGURE
    assert page.nontext_units[0].requires_visual_read is True
    assert page.nontext_units[0].text_element_ids == (text[0].element_id,)
    assert text[0].text_evidence is not None
    assert text[0].text_evidence.origin is DocumentTextEvidenceOrigin.OCR
    assert text[0].text_evidence.confidence == 0.95
    assert text[0].text_evidence.uncertainty == pytest.approx(0.05)
    assert text[0].text_evidence.source_unit_id == page.nontext_units[0].unit_id


def test_image_reader_applies_exif_orientation_before_ocr(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # 顺时针旋转 90 度以便显示
    _write_image(path, format="JPEG", size=(20, 10), exif=exif)
    service = _OcrService()

    result = read_image(path, ocr_service=service)

    assert result.admission_status is ProcessingAdmissionStatus.PARTIAL
    assert service.image_sizes == [(10, 20)]
    assert service.requests[0].pixel_size.width == 10
    assert service.requests[0].pixel_size.height == 20
    request = service.requests[0]
    assert request.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as source:
        canonical = ImageOps.exif_transpose(source).convert("RGBA")
        header = (
            f"personagraph-oriented-raster-v1\0RGBA\0{canonical.width}x"
            f"{canonical.height}\0"
        ).encode("ascii")
        expected_image_sha256 = hashlib.sha256(
            header + canonical.tobytes()
        ).hexdigest()
    assert request.image_sha256 == expected_image_sha256
    assert request.image_sha256 != request.source_sha256


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (OcrStatus.BLANK, DiagnosticCode.PAGE_EMPTY),
        (OcrStatus.FAILED, DiagnosticCode.OCR_ENGINE_FAILED),
    ],
)
def test_blank_or_failed_ocr_is_rejected_with_a_typed_page_diagnostic(
    tmp_path, status, expected_code
):
    path = tmp_path / "source.png"
    _write_image(path, format="PNG")

    result = read_image(path, ocr_service=_OcrService(status))

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED
    assert expected_code in {item.code for item in result.diagnostics}
    assert DiagnosticCode.PAGE_NEEDS_VISION in {
        item.code for item in result.diagnostics
    }
    assert result.page_manifest.pages[0].state in {
        DocumentPageState.VISUAL_ONLY,
        DocumentPageState.UNREADABLE,
    }


def test_reader_rejects_extension_magic_mismatch_before_ocr(tmp_path):
    path = tmp_path / "pretend.png"
    _write_image(path, format="JPEG")
    service = _OcrService()

    result = read_image(path, ocr_service=service)

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED
    assert {item.code for item in result.diagnostics} == {
        DiagnosticCode.CORRUPT_SOURCE
    }
    assert service.requests == []


def test_reader_rejects_byte_and_pixel_limits_without_decompressing_or_ocr(
    tmp_path, monkeypatch
):
    from personagraph.input_processing.documents.readers import image as image_reader

    path = tmp_path / "large.png"
    _write_image(path, format="PNG", size=(30, 30))
    service = _OcrService()

    monkeypatch.setattr(image_reader, "MAX_IMAGE_BYTES", 4)
    too_many_bytes = read_image(path, ocr_service=service)
    assert DiagnosticCode.LIMIT_REACHED in {
        item.code for item in too_many_bytes.diagnostics
    }
    assert service.requests == []

    monkeypatch.setattr(image_reader, "MAX_IMAGE_BYTES", 1_000_000)
    monkeypatch.setattr(image_reader, "MAX_IMAGE_PIXELS", 100)
    too_many_pixels = read_image(path, ocr_service=service)
    assert DiagnosticCode.LIMIT_REACHED in {
        item.code for item in too_many_pixels.diagnostics
    }
    assert service.requests == []


def test_reader_fails_closed_when_injected_service_raises(tmp_path):
    path = tmp_path / "source.png"
    _write_image(path, format="PNG")

    class ExplodingService:
        engine_fingerprint = "exploding@1"

        def recognize(self, _image, _request):
            raise RuntimeError("private OCR crash")

    result = read_image(path, ocr_service=ExplodingService())

    assert result.admission_status is ProcessingAdmissionStatus.REJECTED
    failures = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is DiagnosticCode.OCR_ENGINE_FAILED
    ]
    assert len(failures) == 1
    assert "private OCR crash" not in (failures[0].detail or "")


def test_zero_confidence_ocr_is_a_typed_gap_not_an_ordinary_text_fact(tmp_path):
    path = tmp_path / "zero-confidence.png"
    _write_image(path, format="PNG")

    class ZeroConfidenceService:
        engine_fingerprint = "zero-confidence@1"

        def recognize(self, image, request):
            width, height = image.size
            return OcrResult.from_request(
                request,
                status=OcrStatus.SUCCESS,
                engine_fingerprint=self.engine_fingerprint,
                lines=(OcrLine(
                    text="UNTRUSTED ZERO CONFIDENCE",
                    confidence=0.0,
                    bbox_norm=(0.1, 0.2, 0.9, 0.4),
                    bbox_px=(0.1 * width, 0.2 * height, 0.9 * width, 0.4 * height),
                ),),
                elapsed_ms=1,
            )

    result = read_image(path, ocr_service=ZeroConfidenceService())

    assert result.text_elements() == ()
    assert all(element.text != "UNTRUSTED ZERO CONFIDENCE" for element in result.elements)
    assert DiagnosticCode.OCR_TEXT_UNCERTAIN in {
        diagnostic.code for diagnostic in result.diagnostics
    }
    assert result.page_manifest.pages[0].state is DocumentPageState.VISUAL_ONLY
    assert result.admission_status is ProcessingAdmissionStatus.REJECTED


def test_image_reader_rejects_ocr_line_and_character_limit_overflow(tmp_path, monkeypatch):
    from personagraph.input_processing.documents.readers import image as image_reader

    path = tmp_path / "ocr-overflow.png"
    _write_image(path, format="PNG")

    class ManyLinesService:
        engine_fingerprint = "many-lines@1"

        def __init__(self, texts):
            self.texts = texts

        def recognize(self, image, request):
            width, height = image.size
            lines = tuple(
                OcrLine(
                    text=text,
                    confidence=0.9,
                    bbox_norm=(0.0, 0.0, 1.0, 1.0),
                    bbox_px=(0.0, 0.0, float(width), float(height)),
                )
                for text in self.texts
            )
            return OcrResult.from_request(
                request,
                status=OcrStatus.SUCCESS,
                engine_fingerprint=self.engine_fingerprint,
                lines=lines,
                elapsed_ms=1,
            )

    monkeypatch.setattr(image_reader, "MAX_ELEMENTS", 2)
    too_many_lines = read_image(
        path, ocr_service=ManyLinesService(("first", "second"))
    )
    assert too_many_lines.text_elements() == ()
    assert DiagnosticCode.LIMIT_REACHED in {
        diagnostic.code for diagnostic in too_many_lines.diagnostics
    }
    assert too_many_lines.page_manifest.pages[0].state is DocumentPageState.UNREADABLE
    assert too_many_lines.admission_status is ProcessingAdmissionStatus.REJECTED

    monkeypatch.setattr(image_reader, "MAX_ELEMENTS", 10)
    monkeypatch.setattr(image_reader, "MAX_OCR_OUTPUT_CHARS", 5)
    too_many_chars = read_image(
        path, ocr_service=ManyLinesService(("sixsix",))
    )
    assert too_many_chars.text_elements() == ()
    assert DiagnosticCode.LIMIT_REACHED in {
        diagnostic.code for diagnostic in too_many_chars.diagnostics
    }
    assert too_many_chars.admission_status is ProcessingAdmissionStatus.REJECTED


def test_default_image_processor_recipe_is_available_without_reading_a_file():
    fingerprint = configured_image_processor_fingerprint()

    assert fingerprint.reader == "pillow-image-ocr"
    assert "ocrmac-vision@" in fingerprint.version
