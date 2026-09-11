"""仅限本地的 OCR 端口，具有显式后端选择和类型化失败。"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from typing import Protocol

from ..contracts import (
    OcrBackendResult,
    OcrFailureCode,
    OcrLine,
    OcrRequest,
    OcrResult,
    OcrStatus,
)


class OcrBackend(Protocol):
    @property
    def engine_fingerprint(self) -> str: ...

    def recognize(
        self,
        image,
        *,
        language_hints: tuple[str, ...],
    ) -> OcrBackendResult: ...


class OcrService(Protocol):
    @property
    def engine_fingerprint(self) -> str: ...

    def recognize(self, image, request: OcrRequest) -> OcrResult: ...


class LocalOcrService:
    """为一个显式注入的本地后端附加计时和源身份。"""

    def __init__(
        self,
        backend: OcrBackend,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._backend = backend
        self._clock_ns = clock_ns

    @property
    def engine_fingerprint(self) -> str:
        return _backend_fingerprint(self._backend)

    def recognize(self, image, request: OcrRequest) -> OcrResult:
        if tuple(image.size) != (request.pixel_size.width, request.pixel_size.height):
            return OcrResult.from_request(
                request,
                status=OcrStatus.FAILED,
                engine_fingerprint=self.engine_fingerprint,
                elapsed_ms=0,
                failure_code=OcrFailureCode.INVALID_OUTPUT,
                warnings=("ocr_input_geometry_mismatch",),
            )
        start = self._clock_ns()
        try:
            output = self._backend.recognize(
                image,
                language_hints=request.language_hints,
            )
        except Exception:
            output = OcrBackendResult.failed(
                OcrFailureCode.BACKEND_FAILED,
                warnings=("ocr_backend_exception",),
            )
        if not isinstance(output, OcrBackendResult):
            output = OcrBackendResult.failed(
                OcrFailureCode.INVALID_OUTPUT,
                warnings=("ocr_backend_invalid_result",),
            )
        elapsed_ns = max(0, self._clock_ns() - start)
        return OcrResult.from_request(
            request,
            status=output.status,
            engine_fingerprint=self.engine_fingerprint,
            lines=output.lines,
            elapsed_ms=elapsed_ns // 1_000_000,
            warnings=output.warnings,
            failure_code=output.failure_code,
        )


class OcrMacBackend:
    """通过 ``ocrmac`` 使用 Apple Vision OCR；绝不下载或调用网络。

    模块加载支持注入，使可用性与原生失败在测试中保持确定。语言列表始终显式提供，因此调用方
    绝不会继承后端 locale 或自动选择策略。
    """

    def __init__(
        self,
        *,
        module_loader: Callable[[], tuple[object, str]] | None = None,
        runtime_identity_loader: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self._module_loader = module_loader or _load_ocrmac_module
        self._runtime_identity_loader = (
            runtime_identity_loader or _apple_vision_runtime_identity
        )
        self._module: object | None = None
        self._version: str | None = None
        self._load_failed = False

    @property
    def engine_fingerprint(self) -> str:
        self._load()
        version = self._version if self._module is not None else "unavailable"
        try:
            runtime_identity = self._runtime_identity_loader()
        except Exception:
            runtime_identity = {"status": "unavailable"}
        recipe = {
            "contract": "ocrmac-apple-vision-recipe-v1",
            "framework": "vision",
            "recognition_level": "accurate",
            "confidence_threshold": 0.0,
            "runtime": runtime_identity,
        }
        digest = hashlib.sha256(
            json.dumps(
                recipe,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"ocrmac-vision@{version}:accurate:runtime-{digest}"

    def recognize(
        self,
        image,
        *,
        language_hints: tuple[str, ...],
    ) -> OcrBackendResult:
        module = self._load()
        if module is None:
            return OcrBackendResult.failed(
                OcrFailureCode.BACKEND_UNAVAILABLE,
                warnings=("ocrmac_backend_unavailable",),
            )
        try:
            rows = _recognize_with_ocrmac(module, image, language_hints)
        except _NativeOcrExecutionFailed:
            return OcrBackendResult.failed(
                OcrFailureCode.BACKEND_FAILED,
                warnings=("ocrmac_native_request_failed",),
            )
        except Exception:
            return OcrBackendResult.failed(
                OcrFailureCode.BACKEND_FAILED,
                warnings=("ocrmac_execution_failed",),
            )
        if rows is None or not isinstance(rows, (list, tuple)):
            return OcrBackendResult.failed(
                OcrFailureCode.INVALID_OUTPUT,
                warnings=("ocrmac_invalid_output",),
            )
        if not rows:
            return OcrBackendResult.blank(warnings=("ocrmac_no_text_detected",))
        try:
            lines = tuple(_convert_ocrmac_row(row, image.size) for row in rows)
        except (TypeError, ValueError, IndexError):
            return OcrBackendResult.failed(
                OcrFailureCode.INVALID_OUTPUT,
                warnings=("ocrmac_invalid_output",),
            )
        return OcrBackendResult.success(lines)

    def _load(self) -> object | None:
        if self._module is not None or self._load_failed:
            return self._module
        try:
            module, version = self._module_loader()
            if not version or not hasattr(module, "OCR"):
                raise ImportError("invalid ocrmac module")
        except Exception:
            self._load_failed = True
            return None
        self._module = module
        self._version = str(version)
        return self._module


def create_default_ocr_service() -> LocalOcrService:
    """创建冻结本地 recipe；它不会立即执行原生调用。"""

    return LocalOcrService(OcrMacBackend())


def _load_ocrmac_module() -> tuple[object, str]:
    import ocrmac
    from ocrmac import ocrmac as ocrmac_module

    return ocrmac_module, str(getattr(ocrmac, "__version__", "unknown"))


def _apple_vision_runtime_identity() -> dict[str, object]:
    """在无原生 I/O 的情况下冻结 OS 所有 OCR 实现身份。"""

    return {
        "macos_version": platform.mac_ver()[0] or "unknown",
        "kernel_release": platform.release() or "unknown",
        "kernel_build": platform.version() or "unknown",
        "machine": platform.machine() or "unknown",
    }


class _NativeOcrExecutionFailed(RuntimeError):
    pass


def _recognize_with_ocrmac(
    module: object,
    image,
    language_hints: tuple[str, ...],
):
    """运行 Apple Vision，同时保留其成功位。

    ``ocrmac.OCR.recognize`` 会把 ``performRequests=False`` 和零观测的有效请求都映射为
    ``[]``。在此调用它公开的原生 primitive，是区分引擎失败与已确认空白图像的唯一方式。
    对于注入的测试替身和缺少这些 primitive 的旧包变体，公开包装器仍作为兼容路径保留。
    """

    if all(
        hasattr(module, attribute)
        for attribute in ("Vision", "objc", "pil2buf")
    ):
        return _recognize_with_apple_vision(module, image, language_hints)
    engine = module.OCR(  # type: ignore[attr-defined]
        image,
        framework="vision",
        recognition_level="accurate",
        language_preference=list(language_hints),
        confidence_threshold=0.0,
        detail=True,
    )
    return engine.recognize(px=False)


def _recognize_with_apple_vision(
    module: object,
    image,
    language_hints: tuple[str, ...],
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    objc = module.objc  # type: ignore[attr-defined]
    vision = module.Vision  # type: ignore[attr-defined]
    with objc.autorelease_pool():
        request = vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(0)
        supported = request.supportedRecognitionLanguagesAndReturnError_(None)[0]
        if not set(language_hints).issubset(set(supported)):
            raise ValueError("requested OCR languages are unavailable")
        request.setRecognitionLanguages_(list(language_hints))
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(
            module.pil2buf(image),  # type: ignore[attr-defined]
            None,
        )
        performed = handler.performRequests_error_([request], None)
        if isinstance(performed, tuple):
            ok, error = performed
        else:
            ok, error = bool(performed), None
        if not ok or error is not None:
            raise _NativeOcrExecutionFailed
        observations = request.results()
        if observations is None:
            observations = ()
        rows = []
        for observation in observations:
            bbox = observation.boundingBox()
            rows.append((
                observation.text(),
                observation.confidence(),
                (
                    bbox.origin.x,
                    bbox.origin.y,
                    bbox.size.width,
                    bbox.size.height,
                ),
            ))
        return rows


def _convert_ocrmac_row(row, image_size: tuple[int, int]) -> OcrLine:
    if not isinstance(row, (list, tuple)) or len(row) != 3:
        raise ValueError("invalid ocrmac row")
    text, confidence, raw_bbox = row
    if not isinstance(text, str) or isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        raise ValueError("invalid ocrmac text or confidence")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("invalid ocrmac bbox")
    x, y, width, height = (float(value) for value in raw_bbox)
    left = _clean_coordinate(x)
    top = _clean_coordinate(1.0 - y - height)
    right = _clean_coordinate(x + width)
    bottom = _clean_coordinate(1.0 - y)
    bbox_norm = (left, top, right, bottom)
    image_width, image_height = image_size
    bbox_px = (
        _clean_coordinate(left * image_width),
        _clean_coordinate(top * image_height),
        _clean_coordinate(right * image_width),
        _clean_coordinate(bottom * image_height),
    )
    return OcrLine(
        text=text,
        confidence=float(confidence),
        bbox_norm=bbox_norm,
        bbox_px=bbox_px,
    )


def _clean_coordinate(value: float) -> float:
    # 原生浮点坐标通常与精确十进制数相差一个 ulp。舍入可保持稳定身份，又不隐藏真实溢出。
    return round(value, 12)


def _backend_fingerprint(backend: OcrBackend) -> str:
    try:
        value = backend.engine_fingerprint
    except Exception:
        return "ocr-backend@unavailable"
    if not isinstance(value, str) or not value.strip():
        return "ocr-backend@invalid"
    return value


__all__ = [
    "LocalOcrService",
    "OcrBackend",
    "OcrMacBackend",
    "OcrService",
    "create_default_ocr_service",
]
