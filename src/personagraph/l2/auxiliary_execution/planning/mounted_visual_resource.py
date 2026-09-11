"""AuxiliaryGraph：仅 Host 挂载的视觉资源用于辅助图规划。

Document 存储持久化一个自认证的页面清单，包括所有未解决的视觉单元，但故意不在面向规划者的状态中持久化裁剪或私有源路径。此模块将该清单与确切的挂载/当前源生成进行联结，验证源字节，并仅在不透明别名后面冻结可渲染的 PDF/JPG/PNG 单元。私有的``VisualUnitRef`` 从未进入 Architect 提示或持久化原始请求。

执行重用统一的 ``VisualObservationService`` 数据面，而不是在 Host 内构造并直调另一个
Tool handler。外发默认允许；提供者绑定的物理发送账本在适配器 I/O 之前预留调用，
精确重演已完成的结果，并从不自动重新发送待处理或不确定的调用。不可用的本地提供者变成
密封的失败缺口制品，从不成为成功的观察。

当前 OOXML 读取器持久化图片存在的事实，但不持久化恢复其像素所需的包部分标识符。这些单元故意在生产索引存在之前被排除。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningObservationStatus,
)
from personagraph.input_processing.documents.contracts import (
    DocumentNonTextUnit,
)
from personagraph.input_processing.vision.providers import (
    vision_adapter_transmits_externally,
)
from personagraph.input_processing.vision.contracts import (
    VisionStatus,
)
from personagraph.tools.visual.egress_policy import auto_visual_egress_receipt
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.visual.visual_tool_boundary import (
    FrozenVisualToolBoundary,
)
from personagraph.tools.visual.visual_observation_service import (
    VisualObservationRequest,
    VisualObservationService,
    default_vision_adapter,
)
from .host_primitive_controller import (
    AuxiliaryHostPrimitiveDispatchContext,
    AuxiliaryHostPrimitiveWaitingExternal,
)
from personagraph.runtime.model_calls.vision import (
    DurableMountedVisionAdapter,
    MountedVisualCallWaitingExternal,
    SqliteMountedVisualCallLedger,
)
from personagraph.tools.visual.mounted_visual_source_authority import (
    FrozenMountedVisualBinding,
    MAX_MOUNTED_VISUAL_PLANNING_RESOURCES,
    MountedVisualDocumentSource,
    MountedVisualPlanningAuthorityError,
    MountedVisualPlanningProjectionLimitExceeded,
    derive_page_visual_fallback_unit,
    freeze_mounted_visual_bindings,
    mounted_visual_binding_is_current,
    visual_unit_render_pages,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceEvidenceKind,
    PlanningResourceEvidenceUnit,
    PlanningResourceFormat,
    PlanningResourceGapReason,
    PlanningResourcePerceptionRequest,
    PlanningResourceReadOutcome,
    PlanningResourceReadPort,
    PlanningResourceReadRequest,
)


MOUNTED_VISUAL_READ_CAPABILITY = "host_mounted_visual_read"
_PENDING_AUTHORITY_SNAPSHOT_ID = "pending_authority_snapshot"


@dataclass(frozen=True, slots=True)
class FrozenMountedVisualPlanningBinding(FrozenMountedVisualBinding):
    """一个提示别名绑定到一个精确的私有 ``VisualUnitRef``。"""

    resource: FrozenPlanningResource
    authority_anchor: PlanningAuthorityAnchor
    source_card: PlanningAuthoritySourceCard

    def __post_init__(self) -> None:
        FrozenMountedVisualBinding.__post_init__(self)
        if self.resource.resource_alias != self.authority_anchor.projection_alias:
            raise ValueError("visual resource and authority aliases differ")
        if self.source_card.alias != self.resource.resource_alias:
            raise ValueError("visual resource and prompt aliases differ")
        if (
            self.authority_anchor.authority_class
            is not PlanningAuthorityClass.EVIDENCE
            or self.authority_anchor.origin_kind
            is not PlanningAuthorityOriginKind.VISUAL_UNIT
            or self.source_card.authority_class
            is not PlanningAuthorityClass.EVIDENCE
            or self.source_card.source_kind
            is not PlanningAuthoritySourceKind.VISUAL
        ):
            raise ValueError("mounted visuals may only contribute visual evidence")
        if (
            self.authority_anchor.projection_sha256
            != self.source_card.projection_sha256
        ):
            raise ValueError("visual authority projection hashes differ")
        if self.visual_unit.source_sha256 != self.resource.content_sha256:
            raise ValueError("visual unit and resource source hashes differ")
        if self.purpose not in self.visual_unit.allowed_purposes:
            raise ValueError("visual purpose is not allowed for this unit kind")
        if len(self.page_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.page_manifest_sha256
        ):
            raise ValueError("visual binding requires a page-manifest hash")


def freeze_mounted_visual_planning_bindings(
    *,
    session_id: str,
    sources: Sequence[MountedVisualDocumentSource],
    maximum_visual_units: int | None = None,
    source_pages: Sequence[int] | None = None,
    include_all_manifest_units: bool = False,
) -> tuple[FrozenMountedVisualPlanningBinding, ...]:
    """冻结 Host 视觉单元并投影 Auxiliary planning 元数据。"""

    host_bindings = freeze_mounted_visual_bindings(
        session_id=session_id,
        sources=sources,
        maximum_visual_units=maximum_visual_units,
        source_pages=source_pages,
        include_all_manifest_units=include_all_manifest_units,
    )
    return project_mounted_visual_planning_bindings(
        session_id=session_id,
        bindings=host_bindings,
    )


def project_mounted_visual_planning_bindings(
    *,
    session_id: str,
    bindings: Sequence[FrozenMountedVisualBinding],
) -> tuple[FrozenMountedVisualPlanningBinding, ...]:
    """Map already-frozen Host visual bindings into planning resources."""

    raw_bindings = tuple(bindings)
    if any(binding.session_id != session_id for binding in raw_bindings):
        raise MountedVisualPlanningAuthorityError(
            "mounted visual bindings belong to another Session"
        )
    projected_bindings: list[FrozenMountedVisualPlanningBinding] = []
    for host_binding in raw_bindings:
        unit = host_binding.source_unit
        alias = f"mounted_visual_{host_binding.ordinal:03d}"
        identity_sha256 = _sha256_value(
            {
                "schema_version": "mounted-visual-resource-identity-v1",
                "session_id": session_id,
                "document_id": host_binding.parent_document_id,
                "document_version": host_binding.parent_document_version,
                "page_manifest_sha256": host_binding.page_manifest_sha256,
                "unit": unit.to_dict(),
            }
        )
        resource_format = PlanningResourceFormat(
            host_binding.source_format.value
        )
        resource = FrozenPlanningResource(
            session_id=session_id,
            resource_alias=alias,
            resource_id="visual_resource_" + identity_sha256[:32],
            resource_version="visual_version_" + identity_sha256[32:],
            content_sha256=host_binding.visual_unit.source_sha256,
            coverage=PlanningResourceCoverage.COMPLETE,
            resource_format=resource_format,
            media_type=_resource_media_type(resource_format),
            file_extension=f".{resource_format.value}",
        )
        excerpt = _visual_card_excerpt(
            alias=alias,
            parent_alias=host_binding.parent_document_alias,
            unit=unit,
        )
        projection_sha256 = _sha256_text(excerpt)
        freshness_sha256 = _sha256_value(
            {
                "schema_version": "mounted-visual-freshness-binding-v1",
                "resource": _resource_payload(resource),
                "parent_document_alias": host_binding.parent_document_alias,
                "parent_document_id": host_binding.parent_document_id,
                "parent_document_version": host_binding.parent_document_version,
                "page_manifest_sha256": host_binding.page_manifest_sha256,
                "unit": unit.to_dict(),
                "purpose": host_binding.purpose.value,
                "source_byte_count": host_binding.visual_unit.byte_count,
            }
        )
        anchor = PlanningAuthorityAnchor(
            anchor_id="auxanchor_visual_" + freshness_sha256[:32],
            authority_snapshot_id=_PENDING_AUTHORITY_SNAPSHOT_ID,
            projection_alias=alias,
            authority_class=PlanningAuthorityClass.EVIDENCE,
            origin_kind=PlanningAuthorityOriginKind.VISUAL_UNIT,
            origin_id=(
                f"{host_binding.parent_document_id}:{unit.unit_id}"
            ),
            content_sha256=host_binding.visual_unit.source_sha256,
            item_ordinal=host_binding.ordinal,
            projection_sha256=projection_sha256,
            freshness_binding_sha256=freshness_sha256,
        )
        projected_bindings.append(
            FrozenMountedVisualPlanningBinding(
                ordinal=host_binding.ordinal,
                session_id=host_binding.session_id,
                parent_document_alias=host_binding.parent_document_alias,
                parent_document_id=host_binding.parent_document_id,
                parent_document_version=host_binding.parent_document_version,
                page_manifest_sha256=host_binding.page_manifest_sha256,
                source_format=host_binding.source_format,
                source_unit=host_binding.source_unit,
                visual_unit=host_binding.visual_unit,
                purpose=host_binding.purpose,
                resource=resource,
                authority_anchor=anchor,
                source_card=PlanningAuthoritySourceCard(
                    alias=alias,
                    authority_class=PlanningAuthorityClass.EVIDENCE,
                    source_kind=PlanningAuthoritySourceKind.VISUAL,
                    source_label=(
                        f"Mounted visual unit {host_binding.ordinal}"
                    ),
                    excerpt=excerpt,
                    projection_sha256=projection_sha256,
                ),
            )
        )
    return tuple(projected_bindings)


def build_mounted_visual_perception_request(
    *,
    context: AuxiliaryHostPrimitiveDispatchContext,
    input_resource_aliases: tuple[str, ...],
    visual_bindings: Sequence[FrozenMountedVisualPlanningBinding],
) -> PlanningResourcePerceptionRequest:
    """解析一个由 Architect 选定的视觉别名，而不暴露其路径。"""

    selected = tuple(
        binding
        for binding in visual_bindings
        if binding.resource.resource_alias in input_resource_aliases
    )
    if len(selected) != 1:
        raise MountedVisualPlanningAuthorityError(
            "a mounted visual perception node must select exactly one visual alias"
        )
    binding = selected[0]
    if binding.resource.session_id != context.session_id:
        raise MountedVisualPlanningAuthorityError(
            "Host primitive and visual authority belong to different Sessions"
        )
    suffix = f"{binding.ordinal:03d}"
    return PlanningResourcePerceptionRequest(
        binding=context.artifact_binding(
            scope_snapshot_sha256=(
                binding.authority_anchor.freshness_binding_sha256
            ),
            alias_prefix=f"visual_{suffix}",
            artifact_alias=f"visual_context_{suffix}",
        ),
        read_request=PlanningResourceReadRequest(
            resource=binding.resource,
            max_evidence_units=1,
        ),
    )


class MountedVisualPlanningResourceReadPort:
    """执行一个冻结的视觉，并披露信息，同时确保远程 I/O 操作的安全性。"""

    def __init__(
        self,
        visual_bindings: Sequence[FrozenMountedVisualPlanningBinding],
        *,
        adapter=None,
        call_ledger: SqliteMountedVisualCallLedger | None = None,
    ) -> None:
        aliases = tuple(item.resource.resource_alias for item in visual_bindings)
        if len(aliases) != len(set(aliases)):
            raise ValueError("visual read port aliases must be unique")
        self._bindings: Mapping[str, FrozenMountedVisualPlanningBinding] = (
            MappingProxyType(
                {item.resource.resource_alias: item for item in visual_bindings}
            )
        )
        self._adapter = adapter or default_vision_adapter()
        self._call_ledger = call_ledger or SqliteMountedVisualCallLedger()

    @property
    def resource_aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def read_frozen_resource(
        self,
        request: PlanningResourceReadRequest,
    ) -> PlanningResourceReadOutcome:
        if not isinstance(request, PlanningResourceReadRequest):
            raise TypeError("request must be a PlanningResourceReadRequest")
        resource = request.resource
        binding = self._bindings.get(resource.resource_alias)
        if binding is None:
            return _blocked_outcome()
        if binding.resource != resource:
            return _stale_outcome(binding.resource)
        if not mounted_visual_binding_is_current(binding):
            return _failed_outcome(resource)

        adapter = self._adapter
        egress_receipt = None
        if vision_adapter_transmits_externally(adapter):
            capabilities = adapter.capabilities()
            egress_receipt = auto_visual_egress_receipt(
                session_id=resource.session_id,
                source_sha256=binding.visual_unit.source_sha256,
                endpoint_identity=capabilities.endpoint_identity,
                model=capabilities.model,
                purpose=binding.purpose,
            )
            adapter = DurableMountedVisionAdapter(
                adapter,
                session_id=resource.session_id,
                ledger=self._call_ledger,
            )
        observation_service = VisualObservationService(
            adapter=adapter,
        )
        try:
            result = observation_service.observe(
                FrozenVisualToolBoundary(
                    session_id=resource.session_id,
                    units=(binding.visual_unit,),
                ),
                (
                    VisualObservationRequest(
                        unit_id=binding.visual_unit.unit_id,
                        purpose=binding.purpose,
                    ),
                ),
            ).results[0]
        except MountedVisualCallWaitingExternal as exc:
            raise AuxiliaryHostPrimitiveWaitingExternal(
                reason_code=exc.reason_code
            ) from exc
        except ToolBusinessFailure:
            return _failed_outcome(resource)
        status = result.status
        observation = result.observation
        if status is VisionStatus.UNAVAILABLE:
            return _vision_unavailable_outcome(resource)
        if status not in {
            VisionStatus.COMPLETED,
            VisionStatus.PARTIAL,
        } or not isinstance(observation, str) or not observation.strip():
            return _failed_outcome(resource)

        statement = observation.strip()
        evidence = PlanningResourceEvidenceUnit(
            source_unit_id=binding.visual_unit.unit_id,
            statement=statement,
            locator=f"resource:{resource.resource_alias}#visual=1",
            content_sha256=_sha256_text(statement),
            evidence_kind=PlanningResourceEvidenceKind.VISUAL_OBSERVATION,
            disclosure_receipt_id=egress_receipt,
        )
        if status is VisionStatus.PARTIAL:
            return PlanningResourceReadOutcome(
                status=PlanningObservationStatus.PARTIAL,
                observed_resource_version=resource.resource_version,
                observed_content_sha256=resource.content_sha256,
                observed_coverage=resource.coverage,
                evidence=(evidence,),
                gap_reasons=(
                    PlanningResourceGapReason.INCOMPLETE_COVERAGE,
                ),
            )
        return PlanningResourceReadOutcome(
            status=PlanningObservationStatus.SUCCESS,
            observed_resource_version=resource.resource_version,
            observed_content_sha256=resource.content_sha256,
            observed_coverage=resource.coverage,
            evidence=(evidence,),
        )


class MountedDocumentAndVisualPlanningResourceReadPort:
    """通过其精确的 Host 端口分发文档和视觉别名。"""

    def __init__(
        self,
        *,
        document_port: PlanningResourceReadPort,
        visual_port: MountedVisualPlanningResourceReadPort,
    ) -> None:
        self._document_port = document_port
        self._visual_port = visual_port

    def read_frozen_resource(
        self,
        request: PlanningResourceReadRequest,
    ) -> PlanningResourceReadOutcome:
        if request.resource.resource_alias in self._visual_port.resource_aliases:
            return self._visual_port.read_frozen_resource(request)
        return self._document_port.read_frozen_resource(request)


def visual_binding_private_payload(
    binding: FrozenMountedVisualPlanningBinding,
) -> dict[str, object]:
    """仅在 Host 哈希之下使用的标准私有作用域载荷。"""

    return {
        "ordinal": binding.ordinal,
        "parent_document_alias": binding.parent_document_alias,
        "parent_document_id": binding.parent_document_id,
        "parent_document_version": binding.parent_document_version,
        "page_manifest_sha256": binding.page_manifest_sha256,
        "resource": _resource_payload(binding.resource),
        "visual_unit": {
            "unit_id": binding.visual_unit.unit_id,
            "kind": binding.visual_unit.kind.value,
            "image_path": binding.visual_unit.image_path,
            "source_sha256": binding.visual_unit.source_sha256,
            "image_sha256": binding.visual_unit.image_sha256,
            "locator": _locator_payload(binding.visual_unit.locator),
            "mime_type": binding.visual_unit.mime_type,
            "pixel_size": {
                "width": binding.visual_unit.pixel_size.width,
                "height": binding.visual_unit.pixel_size.height,
            },
            "byte_count": binding.visual_unit.byte_count,
        },
        "purpose": binding.purpose.value,
        "authority_anchor": binding.authority_anchor.model_dump(mode="json"),
        "source_card": binding.source_card.model_dump(mode="json"),
    }


def _resource_media_type(resource_format: PlanningResourceFormat) -> str:
    return {
        PlanningResourceFormat.PDF: "application/pdf",
        PlanningResourceFormat.JPG: "image/jpeg",
        PlanningResourceFormat.JPEG: "image/jpeg",
        PlanningResourceFormat.PNG: "image/png",
    }[resource_format]


def _vision_media_type(resource_format: PlanningResourceFormat) -> str:
    if resource_format in {
        PlanningResourceFormat.JPG,
        PlanningResourceFormat.JPEG,
    }:
        return "image/jpeg"
    return "image/png"


def _visual_card_excerpt(
    *,
    alias: str,
    parent_alias: str,
    unit: DocumentNonTextUnit,
) -> str:
    pages = ",".join(str(page) for page in unit.source_pages)
    return (
        f"An unresolved {unit.kind.value} from {parent_alias} is available as "
        f"{alias} on source page(s) {pages}; the Host can perform one bounded "
        "visual semantics read without exposing its private path."
    )


def _blocked_outcome() -> PlanningResourceReadOutcome:
    return PlanningResourceReadOutcome(
        status=PlanningObservationStatus.BLOCKED,
        observed_resource_version=None,
        observed_content_sha256=None,
        observed_coverage=None,
        gap_reasons=(PlanningResourceGapReason.ACCESS_BLOCKED,),
    )


def _failed_outcome(
    resource: FrozenPlanningResource,
) -> PlanningResourceReadOutcome:
    return PlanningResourceReadOutcome(
        status=PlanningObservationStatus.FAILED,
        observed_resource_version=resource.resource_version,
        observed_content_sha256=resource.content_sha256,
        observed_coverage=resource.coverage,
        gap_reasons=(PlanningResourceGapReason.READ_FAILED,),
    )


def _stale_outcome(
    resource: FrozenPlanningResource,
) -> PlanningResourceReadOutcome:
    return PlanningResourceReadOutcome(
        status=PlanningObservationStatus.STALE,
        observed_resource_version=resource.resource_version,
        observed_content_sha256=resource.content_sha256,
        observed_coverage=resource.coverage,
        gap_reasons=(PlanningResourceGapReason.RESOURCE_STALE,),
    )


def _vision_unavailable_outcome(
    resource: FrozenPlanningResource,
) -> PlanningResourceReadOutcome:
    return PlanningResourceReadOutcome(
        status=PlanningObservationStatus.FAILED,
        observed_resource_version=resource.resource_version,
        observed_content_sha256=resource.content_sha256,
        observed_coverage=resource.coverage,
        gap_reasons=(
            PlanningResourceGapReason.READ_FAILED,
            PlanningResourceGapReason.VISION_PROVIDER_UNAVAILABLE,
        ),
    )


def _resource_payload(resource: FrozenPlanningResource) -> dict[str, object]:
    return {
        "session_id": resource.session_id,
        "resource_alias": resource.resource_alias,
        "resource_id": resource.resource_id,
        "resource_version": resource.resource_version,
        "content_sha256": resource.content_sha256,
        "coverage": resource.coverage.value,
        "resource_format": resource.resource_format.value,
        "media_type": resource.media_type,
        "file_extension": resource.file_extension,
    }


def _locator_payload(locator) -> dict[str, object]:
    return {
        "page": locator.page,
        "ordinal": locator.ordinal,
        "section_path": list(locator.section_path),
        "bbox": list(locator.bbox) if locator.bbox is not None else None,
        "char_range": (
            list(locator.char_range) if locator.char_range is not None else None
        ),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MAX_MOUNTED_VISUAL_PLANNING_RESOURCES",
    "MOUNTED_VISUAL_READ_CAPABILITY",
    "FrozenMountedVisualPlanningBinding",
    "MountedDocumentAndVisualPlanningResourceReadPort",
    'MountedVisualDocumentSource',
    "MountedVisualPlanningAuthorityError",
    "MountedVisualPlanningProjectionLimitExceeded",
    "MountedVisualPlanningResourceReadPort",
    "build_mounted_visual_perception_request",
    "derive_page_visual_fallback_unit",
    "freeze_mounted_visual_planning_bindings",
    "mounted_visual_binding_is_current",
    "project_mounted_visual_planning_bindings",
    "visual_unit_render_pages",
    "visual_binding_private_payload",
]
