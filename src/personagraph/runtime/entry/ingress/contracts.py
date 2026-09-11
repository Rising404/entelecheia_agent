"""Entry ingress 的确定性数据契约与预检评估。

Ingress 有意不解释开放式用户语义，只消费可信 host envelope、权威 runtime 事实与
能力上限。表层 feature 仅作为不具授权效力的提示发出。
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IngressAnalysisFloor(StrEnum):
    """Ingress 机械预检要求的最低分析层级，不是最终路由决定。"""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class IngressDisposition(StrEnum):
    ACCEPT = "ACCEPT"
    HOLD_APPROVAL = "HOLD_APPROVAL"
    RECONCILE = "RECONCILE"
    REJECT = "REJECT"
    OVERFLOW = "OVERFLOW"


class IngressHandler(StrEnum):
    MODEL = "model"
    TYPED_CONTROL = "typed_control"
    NONE = "none"


class IngressGuard(StrEnum):
    OUTPUT_CLAIMS = "output_claims"
    PRIVACY = "privacy"
    TYPED_CONTROL = "typed_control"
    RECONCILIATION = "reconciliation"
    TOOL_POLICY = "tool_policy"
    RESOURCE_SCOPE = "resource_scope"
    SIDE_EFFECT_APPROVAL = "side_effect_approval"
    IDEMPOTENCY = "idempotency"
    ATTACHMENT_BOUNDARY = "attachment_boundary"


class IngressReason(StrEnum):
    TEXT_ACCEPTED = "text_accepted"
    TYPED_CONTROL_ACCEPTED = "typed_control_accepted"
    TYPED_CONTROL_STALE = "typed_control_stale"
    EMPTY_TEXT = "empty_text"
    INPUT_OVER_HARD_LIMIT = "input_over_hard_limit"
    PENDING_APPROVAL_EXISTS = "pending_approval_exists"
    ACTIVE_RUN_REQUIRES_RECONCILIATION = "active_run_requires_reconciliation"
    UNRESOLVED_OPERATION_EXISTS = "unresolved_operation_exists"
    RESPONSE_ONLY_CEILING = "response_only_ceiling"
    TOOL_CAPABILITY_AVAILABLE = "tool_capability_available"
    ATTACHMENT_PRESENT = "attachment_present"


class AttachmentRef(BaseModel):
    """可信附件边界元数据；绝不包含附件内容。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str = Field(min_length=1, max_length=160)
    media_type: str = Field(min_length=1, max_length=160)


class TypedControlEvent(BaseModel):
    """由 host 创建、无法从自由文本合成的控制事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["approve", "reject", "cancel", "resume"]
    target_id: str = Field(min_length=1, max_length=160)
    expected_revision: int | None = Field(default=None, ge=0)


class TrustedTurnEnvelope(BaseModel):
    """由可信 host 边界组装的版本化 ingress 对象。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    turn_id: str = Field(min_length=1, max_length=160)
    session_id: str | None = Field(default=None, min_length=1, max_length=160)
    received_at: datetime
    input_kind: Literal["user_text", "typed_control"]
    user_text: str | None = None
    control_event: TypedControlEvent | None = None
    attachments: tuple[AttachmentRef, ...] = ()

    @field_validator("received_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("received_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_input_union(self) -> "TrustedTurnEnvelope":
        if self.input_kind == "user_text":
            if self.user_text is None or self.control_event is not None:
                raise ValueError("user_text input requires text and forbids control_event")
        elif self.control_event is None or self.user_text is not None:
            raise ValueError("typed_control input requires control_event and forbids user_text")
        return self


class AuthoritativeRuntimeSnapshot(BaseModel):
    """从权威 runtime 存储读取的最小控制摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pending_decision_id: str | None = Field(default=None, min_length=1, max_length=160)
    active_run_id: str | None = Field(default=None, min_length=1, max_length=160)
    active_run_status: Literal["running", "paused", "unknown"] | None = None
    unresolved_operation_ids: tuple[str, ...] = ()
    revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_active_run(self) -> "AuthoritativeRuntimeSnapshot":
        if (self.active_run_id is None) != (self.active_run_status is None):
            raise ValueError("active_run_id and active_run_status must be provided together")
        return self


class CapabilityCeiling(BaseModel):
    """Host/policy 权威上限；用户文本绝不能扩大此对象。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_model: bool = True
    allow_tools: bool = False
    allow_protected_writes: bool = False
    allow_persistence: bool = False
    allow_external_network: bool = False
    allow_delegation: bool = False

    @model_validator(mode="after")
    def _validate_capability_implications(self) -> "CapabilityCeiling":
        if self.allow_protected_writes and not (self.allow_tools and self.allow_persistence):
            raise ValueError("protected writes require tool and persistence capability")
        if self.allow_delegation and not self.allow_tools:
            raise ValueError("delegation requires tool capability")
        return self

    @property
    def response_only(self) -> bool:
        return not (
            self.allow_tools
            or self.allow_protected_writes
            or self.allow_persistence
            or self.allow_external_network
            or self.allow_delegation
        )


class SurfaceHints(BaseModel):
    """当前 envelope 的非语义、非授权 feature。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    char_count: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    line_count: int = Field(ge=0)
    question_mark_count: int = Field(ge=0)
    list_item_count: int = Field(ge=0)
    quote_line_count: int = Field(ge=0)
    code_fence_count: int = Field(ge=0)
    attachment_count: int = Field(ge=0)


class IngressDecision(BaseModel):
    """确定性预检输出；绝不构成操作授权。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    disposition: IngressDisposition
    handler: IngressHandler
    analysis_floor: IngressAnalysisFloor
    capability_ceiling: CapabilityCeiling
    required_guards: tuple[IngressGuard, ...]
    reason_codes: tuple[IngressReason, ...]
    heuristic_hints: SurfaceHints


_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_QUOTE_LINE = re.compile(r"^\s*>")


def evaluate_ingress(
    envelope: TrustedTurnEnvelope,
    snapshot: AuthoritativeRuntimeSnapshot,
    capability_ceiling: CapabilityCeiling,
    *,
    estimated_input_tokens: int,
    context_hard_limit: int,
) -> IngressDecision:
    """评估可信控制事实，而不解释文本意图。"""
    if estimated_input_tokens < 0:
        raise ValueError("estimated_input_tokens must be non-negative")
    if context_hard_limit <= 0:
        raise ValueError("context_hard_limit must be positive")

    hints = _surface_hints(envelope, estimated_input_tokens)
    guards: list[IngressGuard] = [IngressGuard.OUTPUT_CLAIMS]
    reasons: list[IngressReason] = []

    if not capability_ceiling.allow_persistence:
        guards.append(IngressGuard.PRIVACY)
    if capability_ceiling.allow_tools:
        guards.extend((IngressGuard.TOOL_POLICY, IngressGuard.RESOURCE_SCOPE))
    if capability_ceiling.allow_protected_writes:
        guards.extend((IngressGuard.SIDE_EFFECT_APPROVAL, IngressGuard.IDEMPOTENCY))
    if envelope.attachments:
        guards.append(IngressGuard.ATTACHMENT_BOUNDARY)
        reasons.append(IngressReason.ATTACHMENT_PRESENT)

    # Capability 与附件事实约束执行，但不解释用户任务。route 模型在已接受 Turn policy
    # 范围内选择。
    floor = IngressAnalysisFloor.L0
    reasons.append(
        IngressReason.RESPONSE_ONLY_CEILING
        if capability_ceiling.response_only
        else IngressReason.TOOL_CAPABILITY_AVAILABLE
    )

    if envelope.input_kind == "typed_control":
        guards.append(IngressGuard.TYPED_CONTROL)
        target = envelope.control_event.target_id  # 已由信封对象校验
        expected = snapshot.pending_decision_id or snapshot.active_run_id
        if expected is None or target != expected:
            reasons.append(IngressReason.TYPED_CONTROL_STALE)
            return _decision(
                IngressDisposition.REJECT,
                IngressHandler.NONE,
                floor,
                capability_ceiling,
                guards,
                reasons,
                hints,
            )
        reasons.append(IngressReason.TYPED_CONTROL_ACCEPTED)
        return _decision(
            IngressDisposition.ACCEPT,
            IngressHandler.TYPED_CONTROL,
            IngressAnalysisFloor.L0,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )

    text = envelope.user_text or ""
    if not text.strip():
        reasons.append(IngressReason.EMPTY_TEXT)
        return _decision(
            IngressDisposition.REJECT,
            IngressHandler.NONE,
            floor,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )
    if estimated_input_tokens > context_hard_limit:
        reasons.append(IngressReason.INPUT_OVER_HARD_LIMIT)
        return _decision(
            IngressDisposition.OVERFLOW,
            IngressHandler.NONE,
            floor,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )
    if snapshot.pending_decision_id or snapshot.active_run_status == "paused":
        reasons.append(IngressReason.PENDING_APPROVAL_EXISTS)
        return _decision(
            IngressDisposition.HOLD_APPROVAL,
            IngressHandler.NONE,
            IngressAnalysisFloor.L2,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )
    if snapshot.unresolved_operation_ids:
        guards.append(IngressGuard.RECONCILIATION)
        reasons.append(IngressReason.UNRESOLVED_OPERATION_EXISTS)
        return _decision(
            IngressDisposition.RECONCILE,
            IngressHandler.NONE,
            IngressAnalysisFloor.L2,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )
    if snapshot.active_run_status in {"running", "unknown"}:
        guards.append(IngressGuard.RECONCILIATION)
        reasons.append(IngressReason.ACTIVE_RUN_REQUIRES_RECONCILIATION)
        return _decision(
            IngressDisposition.RECONCILE,
            IngressHandler.NONE,
            IngressAnalysisFloor.L2,
            capability_ceiling,
            guards,
            reasons,
            hints,
        )

    reasons.append(IngressReason.TEXT_ACCEPTED)
    return _decision(
        IngressDisposition.ACCEPT,
        IngressHandler.MODEL,
        floor,
        capability_ceiling,
        guards,
        reasons,
        hints,
    )


def _surface_hints(
    envelope: TrustedTurnEnvelope,
    estimated_input_tokens: int,
) -> SurfaceHints:
    text = envelope.user_text or ""
    lines = text.splitlines()
    return SurfaceHints(
        char_count=len(text),
        estimated_input_tokens=estimated_input_tokens,
        line_count=len(lines) if text else 0,
        question_mark_count=text.count("?") + text.count("？"),
        list_item_count=sum(1 for line in lines if _LIST_ITEM.match(line)),
        quote_line_count=sum(1 for line in lines if _QUOTE_LINE.match(line)),
        code_fence_count=text.count("```"),
        attachment_count=len(envelope.attachments),
    )


def _decision(
    disposition: IngressDisposition,
    handler: IngressHandler,
    floor: IngressAnalysisFloor,
    ceiling: CapabilityCeiling,
    guards: list[IngressGuard],
    reasons: list[IngressReason],
    hints: SurfaceHints,
) -> IngressDecision:
    return IngressDecision(
        disposition=disposition,
        handler=handler,
        analysis_floor=floor,
        capability_ceiling=ceiling,
        required_guards=tuple(dict.fromkeys(guards)),
        reason_codes=tuple(dict.fromkeys(reasons)),
        heuristic_hints=hints,
    )
