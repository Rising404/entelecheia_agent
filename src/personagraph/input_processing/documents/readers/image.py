"""带本地 OCR 且如实保留视觉 gap 的有界 PNG/JPEG reader。

OCR 文本会作为可寻址文档文本生成，但源图像仍是未解析非文本单元。这可避免把成功转录误认为
理解了图表、场景、公式或布局。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import warnings
from io import BytesIO
from pathlib import Path

from ...vision.contracts import (
    DEFAULT_OCR_LANGUAGE_HINTS,
    OcrFailureCode,
    OcrRequest,
    OcrResult,
    OcrStatus,
    PixelSize,
)
from ...vision.ocr import OcrService, create_default_ocr_service
from ..contracts import (
    DiagnosticCode,
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
    make_element_id,
)


READER_NAME = "pillow-image-ocr"
READER_VERSION = "3"

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_DIMENSION = 16_384
MAX_DECOMPRESSED_BYTES = 160 * 1024 * 1024
MAX_ELEMENTS = 20_000
MAX_OCR_OUTPUT_CHARS = 2_000_000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_FORMAT_BY_SUFFIX = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}
_FORMAT_BY_MIME = {value: key for key, value in _MIME_BY_FORMAT.items()}


def read_image(
    path: Path,
    *,
    ocr_service: OcrService | None = None,
) -> ProcessingResult:
    """将一个有界静态图像读取为单页类型化文档结果。"""

    service = ocr_service or create_default_ocr_service()
    processor = _processor(service)
    expected_format = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if expected_format is None:
        return _failure_result(
            processor,
            DiagnosticCode.CORRUPT_SOURCE,
            "unsupported_image_suffix",
        )
    try:
        source_size = path.stat().st_size
    except PermissionError:
        return _failure_result(
            processor,
            DiagnosticCode.PERMISSION_DENIED,
            "image_source_permission_denied",
        )
    except OSError:
        return _failure_result(
            processor,
            DiagnosticCode.CORRUPT_SOURCE,
            "image_source_unreadable",
        )
    if source_size == 0:
        return _failure_result(
            processor,
            DiagnosticCode.EMPTY_SOURCE,
            "empty_image_source",
        )
    if source_size > MAX_IMAGE_BYTES:
        return _failure_result(
            processor,
            DiagnosticCode.LIMIT_REACHED,
            "image_byte_limit_reached",
        )
    try:
        payload = path.read_bytes()
    except PermissionError:
        return _failure_result(
            processor,
            DiagnosticCode.PERMISSION_DENIED,
            "image_source_permission_denied",
        )
    except OSError:
        return _failure_result(
            processor,
            DiagnosticCode.CORRUPT_SOURCE,
            "image_source_unreadable",
        )
    if len(payload) != source_size or len(payload) > MAX_IMAGE_BYTES:
        return _failure_result(
            processor,
            DiagnosticCode.LIMIT_REACHED,
            "image_changed_or_byte_limit_reached",
        )
    if _sniff_format(payload) != expected_format:
        return _failure_result(
            processor,
            DiagnosticCode.CORRUPT_SOURCE,
            "image_extension_magic_mismatch",
        )

    decoded = _decode_image(payload, expected_format)
    if isinstance(decoded, tuple) and isinstance(decoded[0], DiagnosticCode):
        code, detail = decoded
        return _failure_result(processor, code, detail)
    image, dpi = decoded
    width, height = image.size
    source_sha256 = hashlib.sha256(payload).hexdigest()
    image_sha256 = _oriented_image_sha256(image)
    source_unit_id = f"image_{source_sha256[:24]}"
    request = OcrRequest(
        source_unit_id=source_unit_id,
        source_sha256=source_sha256,
        image_sha256=image_sha256,
        page=1,
        pixel_size=PixelSize(width, height),
        dpi=dpi,
        language_hints=DEFAULT_OCR_LANGUAGE_HINTS,
    )
    try:
        ocr_result = service.recognize(image, request)
    except Exception:
        ocr_result = OcrResult.from_request(
            request,
            status=OcrStatus.FAILED,
            engine_fingerprint=_engine_fingerprint(service),
            elapsed_ms=0,
            failure_code=OcrFailureCode.BACKEND_FAILED,
            warnings=("ocr_service_exception",),
        )
    if not _ocr_result_matches_request(
        ocr_result,
        request,
        expected_engine_fingerprint=_engine_fingerprint(service),
    ):
        ocr_result = OcrResult.from_request(
            request,
            status=OcrStatus.FAILED,
            engine_fingerprint=_engine_fingerprint(service),
            elapsed_ms=0,
            failure_code=OcrFailureCode.INVALID_OUTPUT,
            warnings=("ocr_result_binding_mismatch",),
        )
    return _project_result(
        processor=processor,
        source_key=source_sha256,
        source_unit_id=source_unit_id,
        ocr_result=ocr_result,
    )


def configured_image_processor_fingerprint() -> ProcessorFingerprint:
    """在不读取源文件的情况下解析默认图像 recipe。"""

    return _processor(create_default_ocr_service())


def validate_image_payload(
    payload: bytes,
    media_type: str,
) -> tuple[DiagnosticCode, str] | None:
    """验证一张直接发送给供应商的图像，不调用 OCR 或模型。"""

    expected_format = _FORMAT_BY_MIME.get(str(media_type).lower())
    if expected_format is None:
        return DiagnosticCode.CORRUPT_SOURCE, "unsupported_image_media_type"
    if not payload:
        return DiagnosticCode.EMPTY_SOURCE, "empty_image_source"
    if len(payload) > MAX_IMAGE_BYTES:
        return DiagnosticCode.LIMIT_REACHED, "image_byte_limit_reached"
    if _sniff_format(payload) != expected_format:
        return DiagnosticCode.CORRUPT_SOURCE, "image_media_magic_mismatch"
    decoded = _decode_image(payload, expected_format)
    if isinstance(decoded, tuple) and isinstance(decoded[0], DiagnosticCode):
        return decoded
    image, _dpi = decoded
    image.close()
    return None


def _decode_image(payload: bytes, expected_format: str):
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as opened:
                if opened.format != expected_format:
                    return DiagnosticCode.CORRUPT_SOURCE, "decoded_image_format_mismatch"
                width, height = opened.size
                if width < 1 or height < 1:
                    return DiagnosticCode.CORRUPT_SOURCE, "invalid_image_dimensions"
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return DiagnosticCode.LIMIT_REACHED, "image_dimension_limit_reached"
                if width * height > MAX_IMAGE_PIXELS:
                    return DiagnosticCode.LIMIT_REACHED, "image_pixel_limit_reached"
                if width * height * 4 > MAX_DECOMPRESSED_BYTES:
                    return DiagnosticCode.LIMIT_REACHED, "image_decode_limit_reached"
                if int(getattr(opened, "n_frames", 1)) != 1:
                    return DiagnosticCode.LIMIT_REACHED, "multi_frame_image_unsupported"
                orientation = _exif_orientation(opened)
                dpi = _read_dpi(opened.info.get("dpi"))
                opened.load()
                oriented = ImageOps.exif_transpose(opened)
                if orientation in {5, 6, 7, 8} and dpi is not None:
                    dpi = (dpi[1], dpi[0])
    # 冻结 OCR 所见的精确像素。规范 RGBA 包含调色板和透明度语义，而单独使用原始 P-mode
    # 索引会丢失这些信息。
                image = oriented.convert("RGBA")
    except Image.DecompressionBombWarning:
        return DiagnosticCode.LIMIT_REACHED, "image_decompression_limit_reached"
    except Image.DecompressionBombError:
        return DiagnosticCode.LIMIT_REACHED, "image_decompression_limit_reached"
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return DiagnosticCode.CORRUPT_SOURCE, "image_decode_failed"
    return image, dpi


def _project_result(
    *,
    processor: ProcessorFingerprint,
    source_key: str,
    source_unit_id: str,
    ocr_result: OcrResult,
) -> ProcessingResult:
    text_elements: list[DocumentElement] = []
    output_limit_detail = _ocr_output_limit_detail(ocr_result)
    zero_confidence_lines = ()
    if ocr_result.status is OcrStatus.SUCCESS and output_limit_detail is None:
        zero_confidence_lines = tuple(
            line for line in ocr_result.lines if line.confidence == 0.0
        )
        for ordinal, line in enumerate(ocr_result.lines):
            if line.confidence == 0.0:
                continue
            locator = DocumentLocator(
                page=1,
                ordinal=len(text_elements),
                bbox=line.bbox_norm,
            )
            text_elements.append(DocumentElement(
                element_id=make_element_id(source_key, locator, line.text),
                kind=ElementKind.PARAGRAPH,
                text=line.text,
                locator=locator,
                source_pages=(1,),
                text_evidence=DocumentTextEvidence(
                    origin=DocumentTextEvidenceOrigin.OCR,
                    confidence=line.confidence,
                    source_unit_id=source_unit_id,
                ),
            ))

    image_locator = DocumentLocator(
        page=1,
        ordinal=len(text_elements),
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    image_element = DocumentElement(
        element_id=make_element_id(source_key, image_locator, None),
        kind=ElementKind.IMAGE,
        text=None,
        locator=image_locator,
        needs_vision=True,
        source_pages=(1,),
    )
    text_ids = tuple(element.element_id for element in text_elements)
    nontext_unit = DocumentNonTextUnit(
        unit_id=source_unit_id,
        kind=DocumentNonTextKind.FIGURE,
        source_pages=(1,),
        text_element_ids=text_ids,
        element_id=image_element.element_id,
        locator=image_locator,
        requires_visual_read=True,
    )
    diagnostics: list[ProcessingDiagnostic] = [ProcessingDiagnostic(
        DiagnosticCode.PAGE_NEEDS_VISION,
        DocumentLocator(page=1),
        detail="visual_semantics_unresolved",
    )]
    if output_limit_detail is not None:
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.LIMIT_REACHED,
            DocumentLocator(page=1),
            detail=output_limit_detail,
        ))
        state = DocumentPageState.UNREADABLE
    elif ocr_result.status is OcrStatus.BLANK:
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.PAGE_EMPTY,
            DocumentLocator(page=1),
            detail="ocr_confirmed_blank",
        ))
        state = DocumentPageState.VISUAL_ONLY
    elif ocr_result.status is OcrStatus.FAILED:
        failure = (
            ocr_result.failure_code.value
            if ocr_result.failure_code is not None
            else "unknown"
        )
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.OCR_ENGINE_FAILED,
            DocumentLocator(page=1),
            detail=f"ocr_failed:{failure}",
        ))
        state = DocumentPageState.UNREADABLE
    else:
        if zero_confidence_lines:
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.OCR_TEXT_UNCERTAIN,
                DocumentLocator(page=1, bbox=zero_confidence_lines[0].bbox_norm),
                detail=(
                    f"source_unit={source_unit_id};"
                    f"zero_confidence_lines_excluded={len(zero_confidence_lines)}"
                ),
            ))
        state = (
            DocumentPageState.MIXED
            if text_elements
            else DocumentPageState.VISUAL_ONLY
        )

    page = DocumentPageRecord(
        page_number=1,
        state=state,
        text_element_ids=text_ids,
        nontext_units=(nontext_unit,),
        diagnostics=tuple(diagnostics),
    )
    manifest = DocumentPageManifest(
        physical_page_count=1,
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint=f"{processor}:physical-image-v1",
        detector_capabilities=tuple(sorted({
            "image_pixel_inventory",
            "nontext_unit_inventory",
            "ocr_text_geometry",
            "ocr_text_confidence",
            "ocr_text_source_linkage",
            "physical_page_inventory",
            "raster_image_inventory",
            "text_element_source_pages",
            "typed_page_diagnostics",
        })),
        pages=(page,),
    )
    return ProcessingResult(
        elements=(*text_elements, image_element),
        processor=processor,
        diagnostics=tuple(diagnostics),
        page_manifest=manifest,
    )


def _failure_result(
    processor: ProcessorFingerprint,
    code: DiagnosticCode,
    detail: str,
) -> ProcessingResult:
    return ProcessingResult(
        elements=(),
        processor=processor,
        diagnostics=(ProcessingDiagnostic(code, detail=detail),),
    )


def _ocr_output_limit_detail(result: OcrResult) -> str | None:
    """在任何 OCR 行变成 prompt 文本之前，返回安全的类型化超限原因。"""

    if result.status is not OcrStatus.SUCCESS:
        return None
    # 为源图像/非文本证据元素预留一个位置。
    if len(result.lines) + 1 > MAX_ELEMENTS:
        return "ocr_text_element_limit_reached"
    total_chars = sum(len(line.text) for line in result.lines)
    if total_chars > MAX_OCR_OUTPUT_CHARS:
        return "ocr_output_character_limit_reached"
    return None


def _processor(service: OcrService) -> ProcessorFingerprint:
    runtime = _image_decode_runtime_identity()
    runtime_digest = hashlib.sha256(
        json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return ProcessorFingerprint(
        READER_NAME,
        (
            f"{READER_VERSION}+decode-{runtime_digest}+"
            f"{_engine_fingerprint(service)}"
        ),
    )


def _image_decode_runtime_identity() -> dict[str, str]:
    try:
        from PIL import features

        libjpeg = str(features.version("jpg") or "unknown")
        zlib = str(features.version("zlib") or "unknown")
    except Exception:
        libjpeg = "unavailable"
        zlib = "unavailable"
    try:
        pillow = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        pillow = "not-installed"
    return {
        "pillow": pillow,
        "jpeg_runtime": libjpeg,
        "zlib_runtime": zlib,
        "orientation_policy": "exif-transpose-rgba-v1",
        "decode_limits": (
            f"{MAX_IMAGE_BYTES}:{MAX_IMAGE_PIXELS}:"
            f"{MAX_IMAGE_DIMENSION}:{MAX_DECOMPRESSED_BYTES}"
        ),
    }


def _engine_fingerprint(service: OcrService) -> str:
    try:
        value = service.engine_fingerprint
    except Exception:
        return "ocr-backend@unavailable"
    return value if isinstance(value, str) and value.strip() else "ocr-backend@invalid"


def _sniff_format(payload: bytes) -> str | None:
    if payload.startswith(_PNG_SIGNATURE):
        return "PNG"
    if payload.startswith(_JPEG_SIGNATURE):
        return "JPEG"
    return None


def _read_dpi(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        dpi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) or item <= 0 for item in dpi):
        return None
    return dpi


def _oriented_image_sha256(image) -> str:
    """为实际提供给 OCR、已按 EXIF 定向的规范栅格计算哈希。"""

    width, height = image.size
    header = (
        f"personagraph-oriented-raster-v1\0{image.mode}\0{width}x{height}\0"
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(image.tobytes())
    return digest.hexdigest()


def _exif_orientation(image) -> int:
    try:
        value = image.getexif().get(274, 1)
        return int(value) if value is not None else 1
    except (AttributeError, TypeError, ValueError):
        return 1


def _ocr_result_matches_request(
    result: OcrResult,
    request: OcrRequest,
    *,
    expected_engine_fingerprint: str,
) -> bool:
    return (
        isinstance(result, OcrResult)
        and result.source_unit_id == request.source_unit_id
        and result.source_sha256 == request.source_sha256
        and result.image_sha256 == request.image_sha256
        and result.page == request.page
        and result.pixel_size == request.pixel_size
        and result.dpi == request.dpi
        and result.language_hints == request.language_hints
        and result.engine_fingerprint == expected_engine_fingerprint
    )


__all__ = [
    "MAX_DECOMPRESSED_BYTES",
    "MAX_ELEMENTS",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_PIXELS",
    "MAX_OCR_OUTPUT_CHARS",
    "READER_NAME",
    "READER_VERSION",
    "configured_image_processor_fingerprint",
    "read_image",
]
