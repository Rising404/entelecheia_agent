"""可选视觉模型 provider 的稳定适配边界。"""

from .contracts import (
    UnavailableVisionModelAdapter,
    VisionModelAdapter,
    vision_adapter_transmits_externally,
)
__all__ = [
    "UnavailableVisionModelAdapter",
    "VisionModelAdapter",
    "vision_adapter_transmits_externally",
]
