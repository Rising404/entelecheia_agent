"""证据支持的 SessionContext 状态带类型契约。

本模块拥有不可变跨模块数据形态和稳定枚举值，不拥有持久化、模型调用、图编排或 Prompt
组装。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class SessionDomain(StrEnum):
    USER = "user"
    TASK = "task"
    INTERACTION = "interaction"


class EvidenceKind(StrEnum):
    USER_TURN = "user_turn"
    ASSISTANT_TURN = "assistant_turn"
    TOOL_RESULT = "tool_result"
    ARTIFACT = "artifact"
    APPROVAL = "approval"
    CORRECTION = "correction"
    SYSTEM_EVENT = "system_event"


class SourceKind(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    TOOL = "tool"
    ASSISTANT = "assistant"


class Operation(StrEnum):
    SET = "set"
    APPEND = "append"
    RESOLVE = "resolve"
    RETRACT = "retract"
    TOUCH = "touch"


class StateStatus(StrEnum):
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    EXPIRED = "expired"
    RETRACTED = "retracted"


class UpdateMode(StrEnum):
    SINGLETON = "singleton"
    EVENT = "event"


class Lifetime(StrEnum):
    TURN_WINDOW = "turn_window"
    SESSION = "session"
    EXPLICIT = "explicit"


class TransitionDecision(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    CONFLICTED = "conflicted"
    EXPIRED = "expired"
    RETRACTED = "retracted"


class ReasonCode(StrEnum):
    APPLIED_NEW = "applied_new"
    APPLIED_UPDATE = "applied_update"
    APPLIED_APPEND = "applied_append"
    DUPLICATE_NOOP = "duplicate_noop"
    TOUCHED = "touched"
    RESOLVED = "resolved"
    RETRACTED = "retracted"
    EXPIRED = "expired"
    INVALID_SESSION = "invalid_session"
    INVALID_TYPE = "invalid_type"
    INVALID_OPERATION = "invalid_operation"
    INVALID_KEY = "invalid_key"
    INVALID_VALUE = "invalid_value"
    INVALID_CONFIDENCE = "invalid_confidence"
    MISSING_EVIDENCE = "missing_evidence"
    SCOPE_MISMATCH = "scope_mismatch"
    SOURCE_NOT_ALLOWED = "source_not_allowed"
    SOURCE_REQUIRES_VERIFICATION = "source_requires_verification"
    CONFIDENCE_BELOW_AUTO_APPLY = "confidence_below_auto_apply"
    LOWER_PRECEDENCE_CONFLICT = "lower_precedence_conflict"
    APPEND_KEY_CONFLICT = "append_key_conflict"
    TARGET_MISSING = "target_missing"


@dataclass(frozen=True)
class StateTypePolicy:
    """一个领域与类型组合的确定性更新策略。"""

    domain: SessionDomain
    state_type: str
    update_mode: UpdateMode
    allowed_operations: frozenset[Operation]
    allowed_sources: frozenset[SourceKind]
    auto_apply_sources: frozenset[SourceKind]
    lifetime: Lifetime
    default_turn_ttl: int | None = None
    min_inferred_confidence: float | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    """稳定证据引用；大型来源载荷仍由其所有者保管。"""

    id: str
    session_id: str
    kind: EvidenceKind
    source_ref: str
    content_excerpt: str
    content_hash: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            kind=EvidenceKind(str(data["kind"])),
            source_ref=str(data["source_ref"]),
            content_excerpt=str(data["content_excerpt"]),
            content_hash=str(data["content_hash"]),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ObservationCandidate:
    """使用前必须通过策略和 reducer 检查的抽取器提案。"""

    candidate_id: str
    session_id: str
    domain: SessionDomain
    state_type: str
    key: str
    proposed_value: Any
    operation: Operation
    source_kind: SourceKind
    derived_from: tuple[str, ...]
    extractor_version: str
    confidence_hint: float = 1.0
    valid_from: str | None = None
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationCandidate":
        """从持久化或 API 映射恢复一个候选。"""
        return cls(
            candidate_id=str(data["candidate_id"]),
            session_id=str(data["session_id"]),
            domain=SessionDomain(str(data["domain"])),
            state_type=str(data["state_type"]),
            key=str(data["key"]),
            proposed_value=data.get("proposed_value"),
            operation=Operation(str(data["operation"])),
            source_kind=SourceKind(str(data["source_kind"])),
            derived_from=tuple(data.get("derived_from") or ()),
            extractor_version=str(data["extractor_version"]),
            confidence_hint=float(data.get("confidence_hint", 1.0)),
            valid_from=str(data["valid_from"]) if data.get("valid_from") is not None else None,
            expires_at=str(data["expires_at"]) if data.get("expires_at") is not None else None,
        )


@dataclass(frozen=True)
class SessionStateItem:
    """带证据来源的一个物化 SessionContext 条目。"""

    id: str
    session_id: str
    domain: SessionDomain
    state_type: str
    key: str
    value_json: Any
    status: StateStatus
    source_kind: SourceKind
    derived_from: tuple[str, ...]
    extractor_version: str
    reducer_version: str
    valid_from: str
    expires_at: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionStateItem":
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            domain=SessionDomain(str(data["domain"])),
            state_type=str(data["state_type"]),
            key=str(data["key"]),
            value_json=data.get("value_json"),
            status=StateStatus(str(data["status"])),
            source_kind=SourceKind(str(data["source_kind"])),
            derived_from=tuple(data.get("derived_from") or ()),
            extractor_version=str(data["extractor_version"]),
            reducer_version=str(data["reducer_version"]),
            valid_from=str(data["valid_from"]),
            expires_at=str(data["expires_at"]) if data.get("expires_at") is not None else None,
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


@dataclass(frozen=True)
class TransitionAudit:
    """可审计的 reducer 决策；证据仍是权威源。"""

    transition_id: str
    session_id: str
    candidate_id: str
    old_state_ref: str | None
    new_state_ref: str | None
    decision: TransitionDecision
    reason_code: ReasonCode
    derived_from: tuple[str, ...]
    reducer_version: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionResult:
    """纯 reducer 输出；调用方决定是否以及如何持久化。"""

    state_item: SessionStateItem | None
    audit: TransitionAudit


@dataclass(frozen=True)
class CandidateRejection:
    """在到达 reducer 前被拒绝的一个抽取器条目。"""

    index: int
    reason_code: str


@dataclass(frozen=True)
class SessionExtractionResult:
    """无持久化副作用、可软失败的 Session 抽取输出。"""

    candidates: tuple[ObservationCandidate, ...]
    rejected: tuple[CandidateRejection, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [asdict(item) for item in self.rejected],
            "error": self.error,
        }
