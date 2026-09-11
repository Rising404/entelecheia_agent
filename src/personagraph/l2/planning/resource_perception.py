"""供最终 AuxiliaryGraph 规划器使用的有界文件资源感知。

此适配器不解析文件，也不是模型工具。Host 会冻结资源标识，恰好调用一次只读端口，
并将已物化观测封存到 RAG 和目录原语所使用的同一规划权威及上下文契约中。

只有有限的 PDF、JPG/JPEG、PNG、TXT、MD/MARKDOWN、DOC/DOCX 和 PPT/PPTX 格式跨越此边界。
私有资源标识、来源单元 ID 和定位器留在重放载荷及权威锚点中。``prompt_inputs`` 只暴露
不透明别名和有界不可信证据；文件内容绝不能创建授权权威。
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
import hashlib
import json
import re
from typing import Any, Protocol

from pydantic import BaseModel

from ..auxiliary_graph.contracts import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningContextArtifactProjection,
    PlanningContextArtifact,
    PlanningContextFactProjection,
    PlanningContextFact,
    PlanningContextGapProjection,
    PlanningContextGap,
    PlanningContextPromptInputs,
    PlanningEvidenceRef,
    PlanningObservationStatus,
    planning_document_group_alias,
)
from .invocation_contracts import (
    FrozenPlanningContextPrimitiveInvocation,
    FrozenPlanningContextArtifactBinding,
    PlanningContextPrimitiveKind,
    freeze_serialized_planning_context_primitive_invocation,
)


_DURABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_LOCAL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

MAX_RESOURCE_EVIDENCE_UNITS = 64
MAX_PORT_EVIDENCE_UNITS = 256
MAX_RAW_STATEMENT_CHARACTERS = 32_000
MAX_PROMPT_STATEMENT_CHARACTERS = 4_000


class PlanningResourceFormat(StrEnum):
    PDF = "pdf"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    TXT = "txt"
    MD = "md"
    MARKDOWN = "markdown"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"


_FORMAT_MEDIA_TYPE = {
    PlanningResourceFormat.PDF: "application/pdf",
    PlanningResourceFormat.JPG: "image/jpeg",
    PlanningResourceFormat.JPEG: "image/jpeg",
    PlanningResourceFormat.PNG: "image/png",
    PlanningResourceFormat.TXT: "text/plain",
    PlanningResourceFormat.MD: "text/markdown",
    PlanningResourceFormat.MARKDOWN: "text/markdown",
    PlanningResourceFormat.DOC: "application/msword",
    PlanningResourceFormat.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    PlanningResourceFormat.PPT: "application/vnd.ms-powerpoint",
    PlanningResourceFormat.PPTX: (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
}


class PlanningResourceCoverage(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    REJECTED = "rejected"


class PlanningResourceEvidenceKind(StrEnum):
    DOCUMENT_TEXT = "document_text"
    VISUAL_OBSERVATION = "visual_observation"


class PlanningResourceGapReason(StrEnum):
    ACCESS_BLOCKED = "access_blocked"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    NO_RELEVANT_CONTENT = "no_relevant_content"
    PROJECTION_BOUNDED = "projection_bounded"
    READ_FAILED = "read_failed"
    RESOURCE_STALE = "resource_stale"
    VISION_PROVIDER_UNAVAILABLE = "vision_provider_unavailable"


@dataclass(frozen=True, slots=True)
class FrozenPlanningResource:
    """Host 冻结的私有标识及其不透明规划器别名。"""

    session_id: str
    resource_alias: str
    resource_id: str
    resource_version: str
    content_sha256: str
    coverage: PlanningResourceCoverage
    resource_format: PlanningResourceFormat
    media_type: str
    file_extension: str

    def __post_init__(self) -> None:
        _require_durable_id("session_id", self.session_id)
        _require_local_key("resource_alias", self.resource_alias)
        _require_durable_id("resource_id", self.resource_id)
        _require_text(
            "resource_version",
            self.resource_version,
            maximum=200,
        )
        _require_sha256("content_sha256", self.content_sha256)
        if not isinstance(self.coverage, PlanningResourceCoverage):
            raise ValueError("coverage must be a PlanningResourceCoverage")
        if not isinstance(self.resource_format, PlanningResourceFormat):
            raise ValueError("resource_format must be a PlanningResourceFormat")
        expected_media_type = _FORMAT_MEDIA_TYPE[self.resource_format]
        if self.media_type != expected_media_type:
            raise ValueError(
                f"{self.resource_format.value} requires media_type {expected_media_type}"
            )
        expected_extension = f".{self.resource_format.value}"
        if self.file_extension != expected_extension:
            raise ValueError(
                f"{self.resource_format.value} requires file_extension "
                f"{expected_extension}"
            )


@dataclass(frozen=True, slots=True)
class PlanningResourceEvidenceUnit:
    """一个私有来源单元及其有界不可信证据文本。"""

    source_unit_id: str
    statement: str
    locator: str
    content_sha256: str
    evidence_kind: PlanningResourceEvidenceKind
    disclosure_receipt_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("source_unit_id", self.source_unit_id, maximum=200)
        _require_text(
            "statement",
            self.statement,
            maximum=MAX_RAW_STATEMENT_CHARACTERS,
        )
        _require_text("locator", self.locator, maximum=1_000)
        _require_sha256("content_sha256", self.content_sha256)
        if not isinstance(self.evidence_kind, PlanningResourceEvidenceKind):
            raise ValueError("evidence_kind must be a PlanningResourceEvidenceKind")
        if self.disclosure_receipt_id is not None:
            _require_durable_id("disclosure_receipt_id", self.disclosure_receipt_id)


@dataclass(frozen=True, slots=True)
class PlanningResourceReadRequest:
    """一次性传给 Host 持有只读端口的精确请求。"""

    resource: FrozenPlanningResource
    max_evidence_units: int = MAX_RESOURCE_EVIDENCE_UNITS

    def __post_init__(self) -> None:
        if not isinstance(self.resource, FrozenPlanningResource):
            raise ValueError("resource must be a FrozenPlanningResource")
        if (
            isinstance(self.max_evidence_units, bool)
            or not isinstance(self.max_evidence_units, int)
            or not 1 <= self.max_evidence_units <= MAX_RESOURCE_EVIDENCE_UNITS
        ):
            raise ValueError(
                "max_evidence_units must be within "
                f"1..{MAX_RESOURCE_EVIDENCE_UNITS}"
            )


@dataclass(frozen=True, slots=True)
class PlanningResourceReadOutcome:
    """单次只读调用返回的已物化结果。"""

    status: PlanningObservationStatus
    observed_resource_version: str | None
    observed_content_sha256: str | None
    observed_coverage: PlanningResourceCoverage | None
    evidence: tuple[PlanningResourceEvidenceUnit, ...] = ()
    gap_reasons: tuple[PlanningResourceGapReason, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanningObservationStatus):
            raise ValueError("status must be a PlanningObservationStatus")
        identity_fields = (
            self.observed_resource_version,
            self.observed_content_sha256,
            self.observed_coverage,
        )
        if any(value is None for value in identity_fields) and not all(
            value is None for value in identity_fields
        ):
            raise ValueError("observed resource identity must be complete or absent")
        identity_observed = self.observed_resource_version is not None
        if identity_observed:
            _require_text(
                "observed_resource_version",
                self.observed_resource_version,  # type: ignore[arg-type]
                maximum=200,
            )
            _require_sha256(
                "observed_content_sha256",
                self.observed_content_sha256,  # type: ignore[arg-type]
            )
            if not isinstance(self.observed_coverage, PlanningResourceCoverage):
                raise ValueError(
                    "observed_coverage must be a PlanningResourceCoverage"
                )
        if self.status in {
            PlanningObservationStatus.SUCCESS,
            PlanningObservationStatus.NO_MATCH,
            PlanningObservationStatus.PARTIAL,
            PlanningObservationStatus.STALE,
        } and not identity_observed:
            raise ValueError(f"{self.status.value} requires an observed resource identity")
        if len(self.evidence) > MAX_PORT_EVIDENCE_UNITS:
            raise ValueError(
                f"evidence cannot exceed {MAX_PORT_EVIDENCE_UNITS} units"
            )
        if any(
            not isinstance(item, PlanningResourceEvidenceUnit)
            for item in self.evidence
        ):
            raise ValueError(
                "evidence must contain PlanningResourceEvidenceUnit values"
            )
        unit_ids = tuple(item.source_unit_id for item in self.evidence)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("resource evidence source_unit_id values must be unique")
        if any(
            not isinstance(item, PlanningResourceGapReason)
            for item in self.gap_reasons
        ):
            raise ValueError(
                "gap_reasons must contain PlanningResourceGapReason values"
            )
        reason_values = tuple(item.value for item in self.gap_reasons)
        if reason_values != tuple(sorted(reason_values)) or len(reason_values) != len(
            set(reason_values)
        ):
            raise ValueError("gap_reasons must use unique ascending value order")
        if self.observed_coverage is PlanningResourceCoverage.REJECTED and self.evidence:
            raise ValueError("rejected source coverage cannot carry evidence")

        if self.status is PlanningObservationStatus.SUCCESS:
            if not self.evidence or self.gap_reasons:
                raise ValueError("success requires evidence and cannot carry a gap reason")
        elif self.status is PlanningObservationStatus.PARTIAL:
            if not self.evidence or not self.gap_reasons:
                raise ValueError("partial requires evidence and a gap reason")
        else:
            if self.evidence:
                raise ValueError(f"{self.status.value} cannot carry evidence")
            if not self.gap_reasons:
                raise ValueError(f"{self.status.value} requires a gap reason")
        required_reason = {
            PlanningObservationStatus.NO_MATCH: (
                PlanningResourceGapReason.NO_RELEVANT_CONTENT
            ),
            PlanningObservationStatus.BLOCKED: (
                PlanningResourceGapReason.ACCESS_BLOCKED
            ),
            PlanningObservationStatus.FAILED: PlanningResourceGapReason.READ_FAILED,
            PlanningObservationStatus.STALE: PlanningResourceGapReason.RESOURCE_STALE,
        }.get(self.status)
        if required_reason is not None and required_reason not in self.gap_reasons:
            raise ValueError(
                f"{self.status.value} requires gap reason {required_reason.value}"
            )


class PlanningResourceReadPort(Protocol):
    """无副作用的 Host 端口；Runtime 每次原语调用只调用它一次。"""

    def read_frozen_resource(
        self,
        request: PlanningResourceReadRequest,
    ) -> PlanningResourceReadOutcome: ...


class PlanningResourceReadPortError(RuntimeError):
    """类型化操作停止；其私有异常文本绝不会被投影。"""

    def __init__(
        self,
        *,
        status: PlanningObservationStatus,
        reason: PlanningResourceGapReason,
        private_detail: str = "",
    ) -> None:
        expected = {
            PlanningObservationStatus.BLOCKED: (
                PlanningResourceGapReason.ACCESS_BLOCKED
            ),
            PlanningObservationStatus.FAILED: PlanningResourceGapReason.READ_FAILED,
        }
        if status not in expected or reason is not expected[status]:
            raise ValueError("port errors may only represent blocked or failed reads")
        super().__init__(private_detail)
        self.status = status
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PlanningResourcePerceptionRequest:
    """绑定到 Auxiliary 节点且由 Host 持有的一次感知调用。"""

    binding: FrozenPlanningContextArtifactBinding
    read_request: PlanningResourceReadRequest

    def __post_init__(self) -> None:
        if not isinstance(self.binding, FrozenPlanningContextArtifactBinding):
            raise ValueError(
                "binding must be a FrozenPlanningContextArtifactBinding"
            )
        if not isinstance(self.read_request, PlanningResourceReadRequest):
            raise ValueError("read_request must be a PlanningResourceReadRequest")
        if self.read_request.resource.session_id != self.binding.session_id:
            raise ValueError("resource must belong to the bound Session")


@dataclass(frozen=True, slots=True)
class PlanningResourcePerceptionResult:
    """一次有限文件感知调用的可重放结算。"""

    observation_status: PlanningObservationStatus
    logical_request_json: str
    logical_request_sha256: str
    raw_observation_json: str
    raw_observation_sha256: str
    freshness_manifest_sha256: str
    authority_anchors: tuple[PlanningAuthorityAnchor, ...]
    artifact: PlanningContextArtifact
    prompt_inputs: PlanningContextPromptInputs
    verification_receipt_sha256: str
    settlement_sha256: str
    read_call_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.observation_status, PlanningObservationStatus):
            raise ValueError(
                "observation_status must be a PlanningObservationStatus"
            )
        _require_canonical_json_hash(
            "logical request",
            self.logical_request_json,
            self.logical_request_sha256,
        )
        _require_canonical_json_hash(
            "raw observation",
            self.raw_observation_json,
            self.raw_observation_sha256,
        )
        for name, value in (
            ("freshness_manifest_sha256", self.freshness_manifest_sha256),
            ("verification_receipt_sha256", self.verification_receipt_sha256),
            ("settlement_sha256", self.settlement_sha256),
        ):
            _require_sha256(name, value)
        if self.read_call_count != 1:
            raise ValueError("resource perception must contain exactly one read call")
        if any(
            not isinstance(item, PlanningAuthorityAnchor)
            for item in self.authority_anchors
        ):
            raise ValueError(
                "authority_anchors must contain PlanningAuthorityAnchor values"
            )
        aliases = tuple(item.projection_alias for item in self.authority_anchors)
        if aliases != tuple(sorted(aliases)) or len(aliases) != len(set(aliases)):
            raise ValueError("authority anchors must use unique ascending aliases")
        if any(
            item.authority_class is PlanningAuthorityClass.AUTHORIZATION
            for item in self.authority_anchors
        ):
            raise ValueError("file resource observations cannot create authorization")
        if any(
            item.authority_class is PlanningAuthorityClass.AUTHORIZATION
            for item in self.prompt_inputs.source_cards
        ):
            raise ValueError("file resource prompts cannot create authorization")
        if self.artifact.constraints:
            raise ValueError("file resource observations cannot create constraints")
        if self.artifact.freshness_manifest_sha256 != self.freshness_manifest_sha256:
            raise ValueError("artifact freshness does not match the settlement")
        if self.artifact.verification_receipt_sha256 != self.verification_receipt_sha256:
            raise ValueError("artifact verification does not match the settlement")
        projection = self.prompt_inputs.context_artifact
        if (
            projection.artifact_id != self.artifact.artifact_id
            or projection.artifact_sha256 != self.artifact.artifact_sha256
        ):
            raise ValueError("prompt projection is detached from its private artifact")
        if {item.projection_alias for item in self.authority_anchors} != {
            item.alias for item in self.prompt_inputs.source_cards
        }:
            raise ValueError("private anchors and prompt cards must share opaque aliases")
        if self.observation_status is PlanningObservationStatus.SUCCESS:
            if not self.artifact.facts or self.artifact.gaps:
                raise ValueError("successful resource perception requires facts only")
        elif self.observation_status is PlanningObservationStatus.PARTIAL:
            if not any(
                item.observation_status is PlanningObservationStatus.PARTIAL
                for item in self.artifact.gaps
            ):
                raise ValueError("partial resource perception requires a partial gap")
        else:
            if self.artifact.facts or self.artifact.evidence_refs:
                raise ValueError(
                    f"{self.observation_status.value} cannot retain positive evidence"
                )
            if not any(
                item.observation_status is self.observation_status
                for item in self.artifact.gaps
            ):
                raise ValueError(
                    "non-success resource perception requires a same-status gap"
                )
        expected = _settlement_sha256(
            observation_status=self.observation_status,
            logical_request_sha256=self.logical_request_sha256,
            raw_observation_sha256=self.raw_observation_sha256,
            freshness_manifest_sha256=self.freshness_manifest_sha256,
            authority_anchors=self.authority_anchors,
            artifact=self.artifact,
            prompt_inputs=self.prompt_inputs,
            verification_receipt_sha256=self.verification_receipt_sha256,
            read_call_count=self.read_call_count,
        )
        if self.settlement_sha256 != expected:
            raise ValueError("resource perception settlement hash is invalid")


def serialize_planning_resource_perception_request(
    request: PlanningResourcePerceptionRequest,
) -> str:
    """返回读取前后使用的唯一规范逻辑请求。"""

    if not isinstance(request, PlanningResourcePerceptionRequest):
        raise TypeError("request must be a PlanningResourcePerceptionRequest")
    return _canonical_json(
        {
            "schema_version": "planning-resource-perception-request-v1",
            "binding": _binding_payload(request.binding),
            "read_request": _jsonable(request.read_request),
        }
    )


def freeze_planning_resource_perception_invocation(
    request: PlanningResourcePerceptionRequest,
    *,
    invocation_turn_id: str,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_control_state_version: int,
    expected_goal_state_version: int,
    expected_revision_state_version: int,
    expected_budget_state_version: int,
    authority_snapshot_sha256: str,
    structure_sha256: str,
    budget_snapshot_sha256: str,
) -> FrozenPlanningContextPrimitiveInvocation:
    """将一次有限文件观测绑定到其当前图权威信息。"""

    if not isinstance(request, PlanningResourcePerceptionRequest):
        raise TypeError("request must be a PlanningResourcePerceptionRequest")
    return freeze_serialized_planning_context_primitive_invocation(
        primitive_kind=PlanningContextPrimitiveKind.RESOURCE_PERCEPTION,
        binding=request.binding,
        logical_request_json=serialize_planning_resource_perception_request(request),
        invocation_turn_id=invocation_turn_id,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_control_state_version=expected_control_state_version,
        expected_goal_state_version=expected_goal_state_version,
        expected_revision_state_version=expected_revision_state_version,
        expected_budget_state_version=expected_budget_state_version,
        authority_snapshot_sha256=authority_snapshot_sha256,
        structure_sha256=structure_sha256,
        budget_snapshot_sha256=budget_snapshot_sha256,
    )


def run_planning_resource_perception(
    request: PlanningResourcePerceptionRequest,
    *,
    read_port: PlanningResourceReadPort,
) -> PlanningResourcePerceptionResult:
    """恰好调用一次冻结资源端口，并封存其观测。"""

    if not isinstance(request, PlanningResourcePerceptionRequest):
        raise ValueError("request must be a PlanningResourcePerceptionRequest")
    logical_request_json = serialize_planning_resource_perception_request(request)
    logical_request_sha256 = _sha256_text(logical_request_json)

    try:
        outcome = read_port.read_frozen_resource(request.read_request)
    except PlanningResourceReadPortError as exc:
        outcome = PlanningResourceReadOutcome(
            status=exc.status,
            observed_resource_version=None,
            observed_content_sha256=None,
            observed_coverage=None,
            gap_reasons=(exc.reason,),
        )
    if not isinstance(outcome, PlanningResourceReadOutcome):
        raise ValueError("read port must return PlanningResourceReadOutcome")

    raw_observation_json = _canonical_json(
        {
            "schema_version": "planning-resource-raw-observation-v1",
            "resource_id": request.read_request.resource.resource_id,
            "resource_alias": request.read_request.resource.resource_alias,
            "outcome": _jsonable(outcome),
        }
    )
    raw_observation_sha256 = _sha256_text(raw_observation_json)
    status, evidence, gap_reasons = _normalize_observation(request, outcome)
    freshness_manifest_sha256 = _sha256_value(
        _freshness_payload(request.read_request.resource, outcome)
    )
    (
        authority_anchors,
        artifact,
        prompt_inputs,
        verification_receipt_sha256,
    ) = _materialize_resource_artifact(
        request=request,
        status=status,
        evidence=evidence,
        gap_reasons=gap_reasons,
        logical_request_sha256=logical_request_sha256,
        raw_observation_sha256=raw_observation_sha256,
        freshness_manifest_sha256=freshness_manifest_sha256,
    )
    settlement_sha256 = _settlement_sha256(
        observation_status=status,
        logical_request_sha256=logical_request_sha256,
        raw_observation_sha256=raw_observation_sha256,
        freshness_manifest_sha256=freshness_manifest_sha256,
        authority_anchors=authority_anchors,
        artifact=artifact,
        prompt_inputs=prompt_inputs,
        verification_receipt_sha256=verification_receipt_sha256,
        read_call_count=1,
    )
    return PlanningResourcePerceptionResult(
        observation_status=status,
        logical_request_json=logical_request_json,
        logical_request_sha256=logical_request_sha256,
        raw_observation_json=raw_observation_json,
        raw_observation_sha256=raw_observation_sha256,
        freshness_manifest_sha256=freshness_manifest_sha256,
        authority_anchors=authority_anchors,
        artifact=artifact,
        prompt_inputs=prompt_inputs,
        verification_receipt_sha256=verification_receipt_sha256,
        settlement_sha256=settlement_sha256,
    )


def _normalize_observation(
    request: PlanningResourcePerceptionRequest,
    outcome: PlanningResourceReadOutcome,
) -> tuple[
    PlanningObservationStatus,
    tuple[PlanningResourceEvidenceUnit, ...],
    tuple[PlanningResourceGapReason, ...],
]:
    resource = request.read_request.resource
    identity_mismatch = (
        outcome.observed_resource_version is not None
        and (
            outcome.observed_resource_version != resource.resource_version
            or outcome.observed_content_sha256 != resource.content_sha256
            or outcome.observed_coverage is not resource.coverage
        )
    )
    if identity_mismatch or outcome.status is PlanningObservationStatus.STALE:
        return (
            PlanningObservationStatus.STALE,
            (),
            (PlanningResourceGapReason.RESOURCE_STALE,),
        )

    status = outcome.status
    evidence = outcome.evidence
    reasons = set(outcome.gap_reasons)
    if resource.coverage is PlanningResourceCoverage.REJECTED and status not in {
        PlanningObservationStatus.BLOCKED,
        PlanningObservationStatus.FAILED,
    }:
        return (
            PlanningObservationStatus.BLOCKED,
            (),
            (PlanningResourceGapReason.ACCESS_BLOCKED,),
        )
    if resource.coverage is PlanningResourceCoverage.PARTIAL and status in {
        PlanningObservationStatus.SUCCESS,
        PlanningObservationStatus.NO_MATCH,
    }:
        status = PlanningObservationStatus.PARTIAL
        reasons.add(PlanningResourceGapReason.INCOMPLETE_COVERAGE)

    bounded = evidence[: request.read_request.max_evidence_units]
    projection_bounded = len(bounded) != len(evidence)
    bounded_items: list[PlanningResourceEvidenceUnit] = []
    for item in bounded:
        statement = item.statement
        if len(statement) > MAX_PROMPT_STATEMENT_CHARACTERS:
            statement = statement[:MAX_PROMPT_STATEMENT_CHARACTERS].rstrip()
            projection_bounded = True
        bounded_items.append(
            PlanningResourceEvidenceUnit(
                source_unit_id=item.source_unit_id,
                statement=statement,
                locator=item.locator,
                content_sha256=item.content_sha256,
                evidence_kind=item.evidence_kind,
                disclosure_receipt_id=item.disclosure_receipt_id,
            )
        )
    if projection_bounded:
        status = PlanningObservationStatus.PARTIAL
        reasons.add(PlanningResourceGapReason.PROJECTION_BOUNDED)
    if status not in {
        PlanningObservationStatus.SUCCESS,
        PlanningObservationStatus.PARTIAL,
    }:
        bounded_items = []
    return (
        status,
        tuple(bounded_items),
        tuple(sorted(reasons, key=lambda item: item.value)),
    )


def _materialize_resource_artifact(
    *,
    request: PlanningResourcePerceptionRequest,
    status: PlanningObservationStatus,
    evidence: tuple[PlanningResourceEvidenceUnit, ...],
    gap_reasons: tuple[PlanningResourceGapReason, ...],
    logical_request_sha256: str,
    raw_observation_sha256: str,
    freshness_manifest_sha256: str,
) -> tuple[
    tuple[PlanningAuthorityAnchor, ...],
    PlanningContextArtifact,
    PlanningContextPromptInputs,
    str,
]:
    binding = request.binding
    resource = request.read_request.resource
    anchors: list[PlanningAuthorityAnchor] = []
    cards: list[PlanningAuthoritySourceCard] = []
    facts: list[PlanningContextFact] = []
    fact_projections: list[PlanningContextFactProjection] = []
    evidence_refs: list[PlanningEvidenceRef] = []

    for ordinal, item in enumerate(evidence, start=1):
        alias = f"{binding.alias_prefix}_obs_{ordinal:03d}"
        fact_alias = f"{binding.alias_prefix}_fact_{ordinal:03d}"
        anchor_id = _derived_durable_id(
            "planning_resource_anchor",
            binding.primitive_call_id,
            "evidence",
            str(ordinal),
        )
        source_kind = (
            PlanningAuthoritySourceKind.VISUAL
            if item.evidence_kind
            is PlanningResourceEvidenceKind.VISUAL_OBSERVATION
            else PlanningAuthoritySourceKind.DOCUMENT
        )
        origin_kind = (
            PlanningAuthorityOriginKind.VISUAL_UNIT
            if item.evidence_kind
            is PlanningResourceEvidenceKind.VISUAL_OBSERVATION
            else PlanningAuthorityOriginKind.RETRIEVED_SOURCE_UNIT
        )
        source_label = (
            f"Untrusted {resource.resource_format.value.upper()} resource "
            f"{resource.resource_alias} evidence"
        )
        document_group_alias = (
            planning_document_group_alias(
                session_id=resource.session_id,
                document_id=resource.resource_id,
            )
            if source_kind is PlanningAuthoritySourceKind.DOCUMENT
            else None
        )
        projection_sha256 = _source_card_projection_sha256(
            alias=alias,
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=source_kind,
            document_group_alias=document_group_alias,
            source_label=source_label,
            excerpt=item.statement,
        )
        freshness_binding_sha256 = _sha256_value(
            {
                "resource_alias": resource.resource_alias,
                "resource_id": resource.resource_id,
                "resource_version": resource.resource_version,
                "resource_content_sha256": resource.content_sha256,
                "resource_coverage": resource.coverage.value,
                "source_unit_id": item.source_unit_id,
                "unit_content_sha256": item.content_sha256,
                "freshness_manifest_sha256": freshness_manifest_sha256,
            }
        )
        anchors.append(
            PlanningAuthorityAnchor(
                anchor_id=anchor_id,
                authority_snapshot_id=binding.authority_snapshot_id,
                projection_alias=alias,
                authority_class=PlanningAuthorityClass.EVIDENCE,
                origin_kind=origin_kind,
                origin_id=f"{resource.resource_id}:{item.source_unit_id}",
                content_sha256=item.content_sha256,
                item_ordinal=ordinal - 1,
                projection_sha256=projection_sha256,
                freshness_binding_sha256=freshness_binding_sha256,
                disclosure_receipt_id=item.disclosure_receipt_id,
            )
        )
        cards.append(
            PlanningAuthoritySourceCard(
                alias=alias,
                authority_class=PlanningAuthorityClass.EVIDENCE,
                source_kind=source_kind,
                document_group_alias=document_group_alias,
                source_label=source_label,
                excerpt=item.statement,
                projection_sha256=projection_sha256,
            )
        )
        facts.append(
            PlanningContextFact(
                fact_id=fact_alias,
                statement=item.statement,
                evidence_anchor_ids=(anchor_id,),
            )
        )
        fact_projections.append(
            PlanningContextFactProjection(
                fact_alias=fact_alias,
                statement=item.statement,
                evidence_aliases=(alias,),
            )
        )
        evidence_refs.append(
            PlanningEvidenceRef(
                evidence_anchor_id=anchor_id,
                source_alias=alias,
                locator=item.locator,
                content_sha256=item.content_sha256,
                freshness_binding_sha256=freshness_binding_sha256,
            )
        )

    typed_gaps: list[PlanningContextGap] = []
    gap_projections: list[PlanningContextGapProjection] = []
    for gap_ordinal, reason in enumerate(gap_reasons, start=1):
        alias = f"{binding.alias_prefix}_gap_{gap_ordinal:03d}"
        description, blocking, resolution_hint = _gap_projection(reason)
        payload_sha256 = _sha256_value(
            {
                "status": status.value,
                "reason": reason.value,
                "description": description,
                "blocking": blocking,
                "affected_obligations": list(binding.affected_obligations),
                "resolution_hint": resolution_hint,
            }
        )
        projection_sha256 = _source_card_projection_sha256(
            alias=alias,
            authority_class=PlanningAuthorityClass.GAP,
            source_kind=PlanningAuthoritySourceKind.GAP,
            source_label="Planning resource coverage gap",
            excerpt=description,
        )
        anchor_id = _derived_durable_id(
            "planning_resource_anchor",
            binding.primitive_call_id,
            "gap",
            str(gap_ordinal),
        )
        freshness_binding_sha256 = _sha256_value(
            {
                "scope_snapshot_sha256": binding.scope_snapshot_sha256,
                "freshness_manifest_sha256": freshness_manifest_sha256,
                "gap_payload_sha256": payload_sha256,
            }
        )
        anchors.append(
            PlanningAuthorityAnchor(
                anchor_id=anchor_id,
                authority_snapshot_id=binding.authority_snapshot_id,
                projection_alias=alias,
                authority_class=PlanningAuthorityClass.GAP,
                origin_kind=PlanningAuthorityOriginKind.GAP_OBSERVATION,
                origin_id=f"{binding.primitive_call_id}:gap:{gap_ordinal}",
                content_sha256=payload_sha256,
                item_ordinal=len(evidence) + gap_ordinal - 1,
                projection_sha256=projection_sha256,
                freshness_binding_sha256=freshness_binding_sha256,
            )
        )
        cards.append(
            PlanningAuthoritySourceCard(
                alias=alias,
                authority_class=PlanningAuthorityClass.GAP,
                source_kind=PlanningAuthoritySourceKind.GAP,
                source_label="Planning resource coverage gap",
                excerpt=description,
                projection_sha256=projection_sha256,
            )
        )
        typed_gaps.append(
            PlanningContextGap(
                gap_id=alias,
                observation_status=status,
                description=description,
                blocking=blocking,
                affected_obligations=binding.affected_obligations,
                resolution_hint=resolution_hint,
            )
        )
        gap_projections.append(
            PlanningContextGapProjection(
                gap_alias=alias,
                observation_status=status,
                description=description,
                blocking=blocking,
                affected_obligations=binding.affected_obligations,
                resolution_hint=resolution_hint,
            )
        )

    if not facts and not typed_gaps:
        raise ValueError("resource perception requires a fact or typed gap")
    anchors_tuple = tuple(sorted(anchors, key=lambda item: item.projection_alias))
    cards_tuple = tuple(sorted(cards, key=lambda item: item.alias))
    verification_receipt_sha256 = _sha256_value(
        {
            "schema_version": "planning-resource-verification-receipt-v1",
            "primitive_call_id": binding.primitive_call_id,
            "resource_alias": resource.resource_alias,
            "resource_format": resource.resource_format.value,
            "observation_status": status.value,
            "logical_request_sha256": logical_request_sha256,
            "raw_observation_sha256": raw_observation_sha256,
            "freshness_manifest_sha256": freshness_manifest_sha256,
            "authority_anchors": [
                item.model_dump(mode="json") for item in anchors_tuple
            ],
            "facts": [item.model_dump(mode="json") for item in facts],
            "gaps": [item.model_dump(mode="json") for item in typed_gaps],
        }
    )
    artifact = PlanningContextArtifact.create(
        artifact_id=binding.artifact_id,
        session_id=binding.session_id,
        task_id=binding.task_id,
        auxiliary_graph_id=binding.auxiliary_graph_id,
        goal_id=binding.goal_id,
        producer_auxiliary_node=binding.producer_auxiliary_node,
        producer_primitive_call_id=binding.primitive_call_id,
        scope_snapshot_sha256=binding.scope_snapshot_sha256,
        facts=tuple(facts),
        gaps=tuple(typed_gaps),
        evidence_refs=tuple(evidence_refs),
        freshness_manifest_sha256=freshness_manifest_sha256,
        verification_receipt_id=binding.verification_receipt_id,
        verification_receipt_sha256=verification_receipt_sha256,
    )
    projection = PlanningContextArtifactProjection.create(
        artifact_alias=binding.artifact_alias,
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.artifact_sha256,
        producer_node_alias=binding.producer_node_alias,
        facts=tuple(fact_projections),
        gaps=tuple(gap_projections),
    )
    prompt_inputs = PlanningContextPromptInputs(
        source_cards=cards_tuple,
        context_artifact=projection,
    )
    return anchors_tuple, artifact, prompt_inputs, verification_receipt_sha256


def _gap_projection(
    reason: PlanningResourceGapReason,
) -> tuple[str, bool, str]:
    return {
        PlanningResourceGapReason.ACCESS_BLOCKED: (
            "The frozen resource could not be read within the authorized access boundary.",
            True,
            "Obtain access or ask the user for an accessible resource.",
        ),
        PlanningResourceGapReason.INCOMPLETE_COVERAGE: (
            "The frozen resource has incomplete processing coverage; "
            "unread regions may contain relevant evidence.",
            False,
            "Use a narrower document or visual read before relying on complete coverage.",
        ),
        PlanningResourceGapReason.NO_RELEVANT_CONTENT: (
            "The bounded read found no relevant evidence in the frozen resource; "
            "this does not prove that the fact is absent.",
            False,
            "Use another authorized source or ask for clarification if the evidence is required.",
        ),
        PlanningResourceGapReason.PROJECTION_BOUNDED: (
            "Some resource evidence was omitted or shortened to satisfy the bounded "
            "planning projection.",
            False,
            "Use a narrower follow-up read for omitted detail.",
        ),
        PlanningResourceGapReason.READ_FAILED: (
            "The bounded resource read failed before reliable evidence could be produced.",
            True,
            "Retry only within the execution profile or ask the user for another source.",
        ),
        PlanningResourceGapReason.RESOURCE_STALE: (
            "The observed resource identity no longer matches the frozen version, "
            "content hash, or coverage.",
            True,
            "Freeze and read the current resource version before planning.",
        ),
        PlanningResourceGapReason.VISION_PROVIDER_UNAVAILABLE: (
            "No configured vision provider could resolve the frozen visual unit.",
            True,
            "Configure an authorized vision provider or use another evidence source.",
        ),
    }[reason]


def _freshness_payload(
    resource: FrozenPlanningResource,
    outcome: PlanningResourceReadOutcome,
) -> dict[str, object]:
    return {
        "schema_version": "planning-resource-freshness-manifest-v1",
        "expected": {
            "resource_alias": resource.resource_alias,
            "resource_id": resource.resource_id,
            "resource_version": resource.resource_version,
            "content_sha256": resource.content_sha256,
            "coverage": resource.coverage.value,
        },
        "observed": (
            None
            if outcome.observed_resource_version is None
            else {
                "resource_version": outcome.observed_resource_version,
                "content_sha256": outcome.observed_content_sha256,
                "coverage": outcome.observed_coverage.value,  # type: ignore[union-attr]
            }
        ),
    }


def _binding_payload(
    binding: FrozenPlanningContextArtifactBinding,
) -> dict[str, object]:
    return {
        "session_id": binding.session_id,
        "task_id": binding.task_id,
        "auxiliary_graph_id": binding.auxiliary_graph_id,
        "goal_id": binding.goal_id,
        "producer_auxiliary_node": _jsonable(binding.producer_auxiliary_node),
        "primitive_call_id": binding.primitive_call_id,
        "artifact_id": binding.artifact_id,
        "verification_receipt_id": binding.verification_receipt_id,
        "authority_snapshot_id": binding.authority_snapshot_id,
        "scope_snapshot_sha256": binding.scope_snapshot_sha256,
        "alias_prefix": binding.alias_prefix,
        "artifact_alias": binding.artifact_alias,
        "producer_node_alias": binding.producer_node_alias,
        "affected_obligations": list(binding.affected_obligations),
    }


def _source_card_projection_sha256(
    *,
    alias: str,
    authority_class: PlanningAuthorityClass,
    source_kind: PlanningAuthoritySourceKind,
    document_group_alias: str | None = None,
    source_label: str,
    excerpt: str,
) -> str:
    payload = {
        "alias": alias,
        "authority_class": authority_class.value,
        "source_kind": source_kind.value,
        "source_label": source_label,
        "excerpt": excerpt,
    }
    if document_group_alias is not None:
        payload["document_group_alias"] = document_group_alias
    return _sha256_value(payload)


def _settlement_sha256(
    *,
    observation_status: PlanningObservationStatus,
    logical_request_sha256: str,
    raw_observation_sha256: str,
    freshness_manifest_sha256: str,
    authority_anchors: tuple[PlanningAuthorityAnchor, ...],
    artifact: PlanningContextArtifact,
    prompt_inputs: PlanningContextPromptInputs,
    verification_receipt_sha256: str,
    read_call_count: int,
) -> str:
    return _sha256_value(
        {
            "schema_version": "planning-resource-perception-settlement-v1",
            "observation_status": observation_status.value,
            "logical_request_sha256": logical_request_sha256,
            "raw_observation_sha256": raw_observation_sha256,
            "freshness_manifest_sha256": freshness_manifest_sha256,
            "authority_anchors": [
                item.model_dump(mode="json") for item in authority_anchors
            ],
            "artifact": artifact.model_dump(mode="json"),
            "prompt_inputs": _jsonable(prompt_inputs),
            "verification_receipt_sha256": verification_receipt_sha256,
            "read_call_count": read_call_count,
        }
    )


def _jsonable(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not canonically serializable: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _derived_durable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{_sha256_value([prefix, *parts])[:40]}"


def _require_durable_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _DURABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a durable identifier")


def _require_local_key(name: str, value: str) -> None:
    if not isinstance(value, str) or _LOCAL_KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a local opaque alias")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_text(name: str, value: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be canonical text within {maximum} characters")


def _require_canonical_json_hash(name: str, payload: str, digest: str) -> None:
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    canonical = _canonical_json(decoded)
    if payload != canonical or digest != _sha256_text(canonical):
        raise ValueError(f"{name} JSON/hash binding is invalid")


__all__ = [
    'FrozenPlanningResource',
    "MAX_PORT_EVIDENCE_UNITS",
    "MAX_PROMPT_STATEMENT_CHARACTERS",
    "MAX_RAW_STATEMENT_CHARACTERS",
    "MAX_RESOURCE_EVIDENCE_UNITS",
    'PlanningResourceCoverage',
    'PlanningResourceEvidenceKind',
    'PlanningResourceEvidenceUnit',
    'PlanningResourceFormat',
    'PlanningResourceGapReason',
    'PlanningResourcePerceptionRequest',
    'PlanningResourcePerceptionResult',
    'PlanningResourceReadOutcome',
    "PlanningResourceReadPort",
    "PlanningResourceReadPortError",
    'PlanningResourceReadRequest',
    "freeze_planning_resource_perception_invocation",
    "run_planning_resource_perception",
    "serialize_planning_resource_perception_request",
]
