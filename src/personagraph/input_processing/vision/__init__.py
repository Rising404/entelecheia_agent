"""类型化本地 OCR 与可选视觉语义边界。"""

from .providers import (
    UnavailableVisionModelAdapter,
    VisionModelAdapter,
    vision_adapter_transmits_externally,
)
from .contracts import (
    DEFAULT_OCR_LANGUAGE_HINTS,
    OcrBackendResult,
    OcrFailureCode,
    OcrLine,
    OcrRequest,
    OcrResult,
    OcrStatus,
    PixelSize,
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionDetail,
    VisionRegion,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from .ocr import (
    LocalOcrService,
    OcrBackend,
    OcrMacBackend,
    OcrService,
    create_default_ocr_service,
)

__all__ = [
    "DEFAULT_OCR_LANGUAGE_HINTS",
    "LocalOcrService",
    "OcrBackend",
    "OcrBackendResult",
    "OcrFailureCode",
    'OcrLine',
    "OcrMacBackend",
    'OcrRequest',
    'OcrResult',
    "OcrService",
    "OcrStatus",
    "PixelSize",
    "UnavailableVisionModelAdapter",
    'VisionCapabilitySnapshot',
    "VisionModelAdapter",
    "vision_adapter_transmits_externally",
    'VisionObservation',
    "VisionDetail",
    "VisionRegion",
    "VisionPurpose",
    'VisionRequest',
    'VisionResult',
    "VisionStatus",
    "create_default_ocr_service",
]
