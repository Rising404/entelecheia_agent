"""本地 OCR 服务与后端。"""

from .service import (
    LocalOcrService,
    OcrBackend,
    OcrMacBackend,
    OcrService,
    create_default_ocr_service,
)

__all__ = [
    "LocalOcrService",
    "OcrBackend",
    "OcrMacBackend",
    "OcrService",
    "create_default_ocr_service",
]
