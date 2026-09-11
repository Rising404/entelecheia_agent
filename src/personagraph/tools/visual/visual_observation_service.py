"""视觉单元观察的唯一类型化执行服务。

Tool handler、workspace 文档观察、附件视觉读取和挂载资源读取共享这里的同一条
默认外发、payload 准备与 provider 调用路径。调用方只负责冻结自己的资源作用域，并将
类型化结果投影成各自的公开合同；它们不再通过构造另一个 ToolRegistration 来复用能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ...input_processing.vision.providers import (
    UnavailableVisionModelAdapter,
    VisionModelAdapter,
    vision_adapter_transmits_externally,
)
from ...input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionDetail,
    VisionRegion,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
    normalize_vision_question,
)
from ...input_processing.vision.providers.http import (
    PROMPT_CONTRACT_VERSION,
    HttpVisionModelAdapter,
    load_provider_config,
)
from ...runtime.model_calls.vision import (
    MountedVisualCallLedgerError,
    MountedVisualCallReceipt,
    MountedVisualProjectPublicationEnvelope,
)
from ...input_processing.vision.imaging import (
    PayloadFailure,
    PayloadRefusal,
    PreparedVisualArtifactBundle,
    PreparedVisualArtifactReceipt,
    prepare_payload_with_receipt,
)
from ..execution import ToolBusinessFailure
from .question_contract import visual_call_identity
from .egress_policy import auto_visual_egress_receipt
from .visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef


# 一次调用只解析少量单元，控制 payload 与模型成本，避免逐图往返。
MAX_UNITS_PER_CALL = 3

VisualPublicationEnvelopeFactory = Callable[
    [PreparedVisualArtifactReceipt],
    MountedVisualProjectPublicationEnvelope,
]


@dataclass(frozen=True, slots=True)
class VisualObservationRequest:
    """在冻结视觉边界内读取一个单元的类型化请求。"""

    unit_id: str
    purpose: VisionPurpose
    detail: VisionDetail = VisionDetail.STANDARD
    region: VisionRegion = VisionRegion.DETECTED
    question: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("unit_id must not be empty")
        if not isinstance(self.purpose, VisionPurpose):
            raise TypeError("purpose must be a VisionPurpose")
        if not isinstance(self.detail, VisionDetail):
            raise TypeError("detail must be a VisionDetail")
        if not isinstance(self.region, VisionRegion):
            raise TypeError("region must be a VisionRegion")
        object.__setattr__(self, "question", normalize_vision_question(self.purpose, self.question))


@dataclass(frozen=True, slots=True)
class VisualObservationResult:
    """provider 结果经过稳定、模型可见投影后的类型化观察。"""

    unit_id: str
    purpose: VisionPurpose
    region: VisionRegion
    status: VisionStatus
    at: str
    observation: str | None = None
    observation_id: str | None = None
    uncertainty: float | None = None
    failure_code: str | None = None
    failure_diagnostics: dict[str, object] | None = None
    resampled: bool = False
    question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "unit_id": self.unit_id,
            "purpose": self.purpose.value,
            "region": self.region.value,
            "status": self.status.value,
            "at": self.at,
        }
        if self.observation is not None:
            view["observation"] = self.observation
        if self.uncertainty is not None:
            view["uncertainty"] = self.uncertainty
        if self.observation_id is not None:
            view["observation_id"] = self.observation_id
        if self.failure_code is not None:
            view["failure_code"] = self.failure_code
        if self.failure_diagnostics is not None:
            view["failure_diagnostics"] = dict(self.failure_diagnostics)
        if self.resampled:
            view["resampled"] = True
        if self.question is not None:
            view["question"] = self.question
        return view


@dataclass(frozen=True, slots=True)
class VisualObservationBatch:
    """一次有界视觉观察的完整类型化结果。"""

    results: tuple[VisualObservationResult, ...]

    @property
    def requested(self) -> int:
        return len(self.results)

    @property
    def resolved(self) -> int:
        return sum(
            item.status is VisionStatus.COMPLETED for item in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [item.to_dict() for item in self.results],
            "requested": self.requested,
            "resolved": self.resolved,
        }


@dataclass(frozen=True, slots=True)
class HostVisualObservationExecution:
    """Host-private projection paired with its replayable physical-call receipt."""

    projection: VisualObservationResult
    durable_receipt: MountedVisualCallReceipt | None

    def __post_init__(self) -> None:
        if not isinstance(self.projection, VisualObservationResult):
            raise TypeError("projection must be VisualObservationResult")
        if self.durable_receipt is not None and not isinstance(
            self.durable_receipt,
            MountedVisualCallReceipt,
        ):
            raise TypeError("durable_receipt must be MountedVisualCallReceipt or None")


def default_vision_adapter() -> VisionModelAdapter:
    """解析已配置的视觉提供商，或返回稳定的不可用适配器。"""

    config = load_provider_config()
    if config is None:
        return UnavailableVisionModelAdapter()
    return HttpVisionModelAdapter(config)


class VisualObservationService:
    """在调用方冻结的视觉边界内执行唯一的视觉观察数据面。"""

    def __init__(
        self,
        *,
        adapter: VisionModelAdapter | None = None,
    ) -> None:
        self._adapter = adapter or default_vision_adapter()
        self._capabilities = self._adapter.capabilities()
        self._transmits_externally = vision_adapter_transmits_externally(
            self._adapter
        )

    @property
    def capabilities(self) -> VisionCapabilitySnapshot:
        return self._capabilities

    @property
    def transmits_externally(self) -> bool:
        return self._transmits_externally

    def observe(
        self,
        boundary: FrozenVisualToolBoundary,
        requests: Sequence[VisualObservationRequest],
    ) -> VisualObservationBatch:
        """验证完整批次后，按顺序读取每个冻结视觉单元。"""

        selected = self._select(boundary, requests)

        return VisualObservationBatch(
            tuple(
                self._observe_unit(
                    boundary.session_id,
                    unit=unit,
                    request=request,
                ).projection
                for unit, request in selected
            )
        )

    def observe_with_durable_receipt(
        self,
        boundary: FrozenVisualToolBoundary,
        request: VisualObservationRequest,
        *,
        publication_envelope_factory: VisualPublicationEnvelopeFactory,
    ) -> HostVisualObservationExecution:
        """Execute one frozen unit and bind publication after exact preparation."""

        if not callable(publication_envelope_factory):
            raise TypeError("publication_envelope_factory must be callable")
        ((unit, selected_request),) = self._select(boundary, (request,))
        return self._observe_unit(
            boundary.session_id,
            unit=unit,
            request=selected_request,
            publication_envelope_factory=publication_envelope_factory,
        )

    @staticmethod
    def _select(
        boundary: FrozenVisualToolBoundary,
        requests: Sequence[VisualObservationRequest],
    ) -> tuple[tuple[VisualUnitRef, VisualObservationRequest], ...]:
        if not isinstance(boundary, FrozenVisualToolBoundary):
            raise TypeError("boundary must be a FrozenVisualToolBoundary")
        if isinstance(requests, (str, bytes)):
            raise TypeError("requests must contain VisualObservationRequest values")
        frozen_requests = tuple(requests)
        if not frozen_requests:
            raise ToolBusinessFailure(
                "invalid_request", "units must be a non-empty list"
            )
        if len(frozen_requests) > MAX_UNITS_PER_CALL:
            raise ToolBusinessFailure(
                "invalid_request",
                f"at most {MAX_UNITS_PER_CALL} units may be read at once",
            )
        if any(
            not isinstance(request, VisualObservationRequest)
            for request in frozen_requests
        ):
            raise TypeError("requests must contain VisualObservationRequest values")

        selected: list[tuple[VisualUnitRef, VisualObservationRequest]] = []
        seen: set[str] = set()
        for request in frozen_requests:
            # 整批先验证身份，避免前一项已外发后才发现后续问答没有 Host 身份。
            visual_call_identity(request.purpose)
            unit = boundary.unit(request.unit_id)
            if unit is None:
                raise ToolBusinessFailure(
                    "unknown_unit",
                    f"no readable unit named {request.unit_id!r}",
                )
            if request.unit_id in seen:
                raise ToolBusinessFailure(
                    "invalid_request", "unit_id values must be unique"
                )
            seen.add(request.unit_id)
            if request.purpose not in unit.allowed_purposes:
                allowed = ", ".join(value.value for value in unit.allowed_purposes)
                raise ToolBusinessFailure(
                    "purpose_not_allowed",
                    f"{request.unit_id} is a {unit.kind.value}; "
                    f"allowed purposes are {allowed}",
                )
            selected.append((unit, request))
        return tuple(selected)

    def _observe_unit(
        self,
        session_id: str,
        *,
        unit: VisualUnitRef,
        request: VisualObservationRequest,
        publication_envelope_factory: (
            VisualPublicationEnvelopeFactory | None
        ) = None,
    ) -> HostVisualObservationExecution:
        logical_tool_call_id = visual_call_identity(request.purpose)
        disclosure_receipt_id = (
            auto_visual_egress_receipt(
                session_id=session_id,
                source_sha256=unit.source_sha256,
                endpoint_identity=self._capabilities.endpoint_identity,
                model=self._capabilities.model,
                purpose=request.purpose,
            ) if self._transmits_externally else None
        )
        provider_request = _vision_request(
            unit,
            request,
            disclosure_receipt_id=disclosure_receipt_id,
            logical_tool_call_id=logical_tool_call_id,
        )
        if self._transmits_externally:
            bundle = prepare_payload_with_receipt(provider_request)
            if isinstance(bundle, PayloadRefusal):
                return HostVisualObservationExecution(
                    projection=_project(
                        request,
                        _preparation_failure(
                            provider_request,
                            capabilities=self._capabilities,
                            failure_code=bundle.failure.value,
                        ),
                    ),
                    durable_receipt=None,
                )
            assert isinstance(bundle, PreparedVisualArtifactBundle)
            if bundle.payload.source_sha256 != provider_request.image_sha256:
                # 路径不再包含冻结边界指定的图像，不得改发新内容。
                return HostVisualObservationExecution(
                    projection=_project(
                        request,
                        _preparation_failure(
                            provider_request,
                            capabilities=self._capabilities,
                            failure_code=PayloadFailure.SOURCE_CHANGED.value,
                        ),
                    ),
                    durable_receipt=None,
                )
            provider_request = bundle.prepared_request

            if publication_envelope_factory is not None:
                try:
                    envelope = publication_envelope_factory(bundle.receipt)
                except (TypeError, ValueError) as exc:
                    raise MountedVisualCallLedgerError(
                        "visual publication envelope construction failed"
                    ) from exc
                if not isinstance(
                    envelope,
                    MountedVisualProjectPublicationEnvelope,
                ):
                    raise MountedVisualCallLedgerError(
                        "visual publication factory returned an invalid envelope"
                    )
                analyze_with_receipt = getattr(
                    self._adapter,
                    "analyze_with_receipt",
                    None,
                )
                if not callable(analyze_with_receipt):
                    raise MountedVisualCallLedgerError(
                        "external visual adapter has no durable receipt API"
                    )
                receipt = analyze_with_receipt(
                    provider_request,
                    publication_envelope=envelope,
                )
                if not isinstance(receipt, MountedVisualCallReceipt):
                    raise MountedVisualCallLedgerError(
                        "external visual adapter returned an invalid receipt"
                    )
                return HostVisualObservationExecution(
                    projection=_project(request, receipt.result),
                    durable_receipt=receipt,
                )
        elif publication_envelope_factory is not None:
            raise MountedVisualCallLedgerError(
                "durable Project publication requires an external visual adapter"
            )

        result = self._adapter.analyze(provider_request)
        return HostVisualObservationExecution(
            projection=_project(request, result),
            durable_receipt=None,
        )


def _preparation_failure(
    request: VisionRequest,
    *,
    capabilities: VisionCapabilitySnapshot,
    failure_code: str,
) -> VisionResult:
    return VisionResult(
        status=VisionStatus.FAILED,
        provider=capabilities.provider,
        model=capabilities.model,
        endpoint_identity=capabilities.endpoint_identity,
        processor_fingerprint=capabilities.processor_fingerprint,
        input_sha256=request.image_sha256,
        unresolved_gap_refs=(request.source_unit_id,),
        failure_code=failure_code,
    )


def _vision_request(
    unit: VisualUnitRef,
    request: VisualObservationRequest,
    *,
    disclosure_receipt_id: str | None,
    logical_tool_call_id: str | None = None,
) -> VisionRequest:
    return VisionRequest(
        source_unit_id=unit.unit_id,
        source_sha256=unit.source_sha256,
        image_sha256=unit.image_sha256,
        locator=unit.locator,
        mime_type=unit.mime_type,
        pixel_size=unit.pixel_size,
        byte_count=unit.byte_count,
        purpose=request.purpose,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        image_path=unit.image_path,
        detail=request.detail,
        region=request.region,
        disclosure_receipt_id=disclosure_receipt_id,
        question=request.question,
        logical_tool_call_id=logical_tool_call_id,
    )


def _project(
    request: VisualObservationRequest,
    result: VisionResult,
) -> VisualObservationResult:
    """将 provider 结果投影为稳定观察，并显式保留未解析状态。"""

    observation = (
        result.observations[0]
        if result.status in {VisionStatus.COMPLETED, VisionStatus.PARTIAL}
        and result.observations
        else None
    )
    return VisualObservationResult(
        unit_id=request.unit_id,
        purpose=request.purpose,
        region=request.region,
        status=result.status,
        at=observation.kind if observation is not None else request.purpose.value,
        observation=observation.text if observation is not None else None,
        observation_id=(
            observation.observation_id if observation is not None else None
        ),
        uncertainty=observation.uncertainty if observation is not None else None,
        failure_code=(
            None
            if observation is not None
            else result.failure_code or "vision_unavailable"
        ),
        resampled="image_resampled" in result.warnings,
        failure_diagnostics=(
            result.failure_diagnostics.to_dict()
            if result.failure_diagnostics is not None else None
        ),
        question=request.question,
    )


__all__ = [
    "MAX_UNITS_PER_CALL",
    "HostVisualObservationExecution",
    "VisualObservationBatch",
    "VisualObservationRequest",
    "VisualObservationResult",
    "VisualObservationService",
    "VisualPublicationEnvelopeFactory",
    "default_vision_adapter",
]
