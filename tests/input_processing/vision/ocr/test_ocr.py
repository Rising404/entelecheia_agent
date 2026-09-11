from __future__ import annotations

from contextlib import nullcontext
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image

from personagraph.input_processing.vision.contracts import (
    OcrBackendResult,
    OcrFailureCode,
    OcrLine,
    OcrRequest,
    OcrStatus,
    PixelSize,
)
from personagraph.input_processing.vision.ocr import (
    LocalOcrService,
    OcrMacBackend,
)


def test_vision_package_imports_in_a_fresh_process_without_document_import_order():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from personagraph.input_processing.vision.ocr.service "
                "import _load_ocrmac_module; print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def _request() -> OcrRequest:
    return OcrRequest(
        source_unit_id="image-1",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
        page=2,
        pixel_size=PixelSize(100, 50),
        dpi=None,
        language_hints=("zh-Hans", "en-US"),
    )


class _Backend:
    engine_fingerprint = "injected-ocr@7"

    def __init__(self, output: OcrBackendResult | Exception) -> None:
        self.output = output
        self.calls: list[tuple[tuple[int, int], tuple[str, ...]]] = []

    def recognize(self, image, *, language_hints):
        self.calls.append((image.size, language_hints))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_local_ocr_service_uses_only_the_injected_backend_and_preserves_binding():
    line = OcrLine(
        text="Hello 世界",
        confidence=0.91,
        bbox_norm=(0.1, 0.2, 0.9, 0.8),
        bbox_px=(10.0, 10.0, 90.0, 40.0),
    )
    backend = _Backend(OcrBackendResult.success((line,)))

    result = LocalOcrService(backend, clock_ns=lambda: 5_000_000).recognize(
        Image.new("RGB", (100, 50)), _request()
    )

    assert backend.calls == [((100, 50), ("zh-Hans", "en-US"))]
    assert result.status is OcrStatus.SUCCESS
    assert result.engine_fingerprint == "injected-ocr@7"
    assert result.source_sha256 == "a" * 64
    assert result.image_sha256 == "b" * 64
    assert result.page == 2
    assert result.lines == (line,)


def test_confirmed_blank_and_backend_exception_are_different_typed_results():
    blank = LocalOcrService(
        _Backend(OcrBackendResult.blank(warnings=("no_text_detected",)))
    ).recognize(Image.new("RGB", (100, 50)), _request())
    failed = LocalOcrService(_Backend(RuntimeError("private details"))).recognize(
        Image.new("RGB", (100, 50)), _request()
    )

    assert blank.status is OcrStatus.BLANK
    assert blank.failure_code is None
    assert blank.warnings == ("no_text_detected",)
    assert failed.status is OcrStatus.FAILED
    assert failed.failure_code is OcrFailureCode.BACKEND_FAILED
    assert all("private details" not in warning for warning in failed.warnings)


def test_ocrmac_backend_passes_explicit_language_order_and_converts_coordinates():
    calls = []

    class FakeOCR:
        def __init__(self, image, **kwargs):
            calls.append((image.size, kwargs))

        def recognize(self, *, px=False):
            assert px is False
    # ocrmac 使用以左下角为原点的 x/y/宽度/高度坐标。
            return [("中文 English", 0.88, (0.1, 0.6, 0.4, 0.2))]

    class FakeModule:
        OCR = FakeOCR

    backend = OcrMacBackend(
        module_loader=lambda: (FakeModule, "1.0.1"),
    )
    output = backend.recognize(
        Image.new("RGB", (200, 100)),
        language_hints=("zh-Hans", "en-US"),
    )

    assert calls[0][1]["language_preference"] == ["zh-Hans", "en-US"]
    assert calls[0][1]["framework"] == "vision"
    assert output.status is OcrStatus.SUCCESS
    assert output.lines[0].bbox_norm == (0.1, 0.2, 0.5, 0.4)
    assert output.lines[0].bbox_px == (20.0, 20.0, 100.0, 40.0)


def test_ocrmac_backend_exposes_unavailable_and_runtime_failure():
    unavailable = OcrMacBackend(
        module_loader=lambda: (_ for _ in ()).throw(ImportError("missing"))
    )
    unavailable_result = unavailable.recognize(
        Image.new("RGB", (10, 10)), language_hints=("zh-Hans", "en-US")
    )

    class BrokenOCR:
        def __init__(self, *_args, **_kwargs):
            pass

        def recognize(self, *, px=False):
            raise RuntimeError("native engine failed")

    class BrokenModule:
        OCR = BrokenOCR

    broken = OcrMacBackend(module_loader=lambda: (BrokenModule, "1.0.1"))
    broken_result = broken.recognize(
        Image.new("RGB", (10, 10)), language_hints=("zh-Hans", "en-US")
    )

    assert unavailable_result.status is OcrStatus.FAILED
    assert unavailable_result.failure_code is OcrFailureCode.BACKEND_UNAVAILABLE
    assert broken_result.status is OcrStatus.FAILED
    assert broken_result.failure_code is OcrFailureCode.BACKEND_FAILED


def test_ocrmac_native_false_result_is_failure_not_confirmed_blank():
    class Request:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setRecognitionLevel_(self, _value):
            pass

        def supportedRecognitionLanguagesAndReturnError_(self, _error):
            return (["zh-Hans", "en-US"], None)

        def setRecognitionLanguages_(self, _languages):
            pass

        def results(self):
            raise AssertionError("failed native request has no observations")

    class Handler:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithData_options_(self, _data, _options):
            return self

        def performRequests_error_(self, _requests, _error):
            return False, None

    class Vision:
        VNRecognizeTextRequest = Request
        VNImageRequestHandler = Handler

    class ObjC:
        @staticmethod
        def autorelease_pool():
            return nullcontext()

    native_module = SimpleNamespace(
        OCR=object,
        Vision=Vision,
        objc=ObjC,
        pil2buf=lambda _image: b"pixels",
    )

    backend = OcrMacBackend(module_loader=lambda: (native_module, "1.0.1"))

    result = backend.recognize(
        Image.new("RGB", (10, 10)), language_hints=("zh-Hans", "en-US")
    )

    assert result.status is OcrStatus.FAILED
    assert result.failure_code is OcrFailureCode.BACKEND_FAILED
    assert result.warnings == ("ocrmac_native_request_failed",)
