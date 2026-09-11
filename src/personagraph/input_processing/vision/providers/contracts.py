"""视觉语义适配器边界及其保守失败默认实现。"""

from __future__ import annotations

from typing import Protocol

from ..contracts import (
    VisionCapabilitySnapshot,
    VisionRequest,
    VisionResult,
    VisionStatus,
)


class VisionModelAdapter(Protocol):
    transmits_externally: bool

    def capabilities(self) -> VisionCapabilitySnapshot: ...

    def analyze(self, request: VisionRequest) -> VisionResult: ...


def vision_adapter_transmits_externally(adapter: VisionModelAdapter) -> bool:
    """Return the adapter's explicit, fail-closed egress classification."""

    try:
        value = adapter.transmits_externally
    except AttributeError as exc:
        raise TypeError(
            "vision adapter must declare transmits_externally as bool"
        ) from exc
    if not isinstance(value, bool):
        raise TypeError(
            "vision adapter must declare transmits_externally as bool"
        )
    return value


class UnavailableVisionModelAdapter:
    """供应商通过 gate 前使用的稳定零 I/O 实现。"""

    transmits_externally = False

    _FAILURE_CODE = "vision_provider_unavailable"
    _FINGERPRINT = "vision-unavailable@1"

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=False,
            provider="unavailable",
            model="none",
            endpoint_identity="none",
            processor_fingerprint=self._FINGERPRINT,
            reason_code=self._FAILURE_CODE,
        )

    def analyze(self, request: VisionRequest) -> VisionResult:
        return VisionResult(
            status=VisionStatus.UNAVAILABLE,
            provider="unavailable",
            model="none",
            endpoint_identity="none",
            processor_fingerprint=self._FINGERPRINT,
            input_sha256=request.image_sha256,
            unresolved_gap_refs=(request.source_unit_id,),
            failure_code=self._FAILURE_CODE,
        )


__all__ = [
    "UnavailableVisionModelAdapter",
    "VisionModelAdapter",
    "vision_adapter_transmits_externally",
]
