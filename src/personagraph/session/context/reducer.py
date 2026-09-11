"""SessionContext 物化视图的纯确定性转换。

reducer 根据带版本目录和可选当前条目校验一个候选。它不执行持久化、模型调用、图访问
或时钟读取；调用方提供 UTC 时间戳并持久化结果。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from typing import Any

from .catalog import CATALOG_VERSION, policy_for
from .models import (
    ObservationCandidate,
    Operation,
    ReasonCode,
    SessionStateItem,
    SourceKind,
    StateStatus,
    TransitionAudit,
    TransitionDecision,
    TransitionResult,
    UpdateMode,
)


REDUCER_VERSION = f"{CATALOG_VERSION}:reducer-v1"
_SOURCE_PRECEDENCE = {
    SourceKind.INFERRED: 1,
    SourceKind.ASSISTANT: 2,
    SourceKind.TOOL: 3,
    SourceKind.EXPLICIT: 4,
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _require_utc_iso(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.isoformat()


def _is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _merge_refs(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for group in groups for ref in group if ref))


def _validation_reason(candidate: ObservationCandidate) -> ReasonCode | None:
    if not candidate.session_id.strip():
        return ReasonCode.INVALID_SESSION
    if not candidate.key.strip():
        return ReasonCode.INVALID_KEY
    if not candidate.derived_from or any(not ref.strip() for ref in candidate.derived_from):
        return ReasonCode.MISSING_EVIDENCE
    if not 0.0 <= candidate.confidence_hint <= 1.0:
        return ReasonCode.INVALID_CONFIDENCE
    policy = policy_for(candidate.domain, candidate.state_type)
    if policy is None:
        return ReasonCode.INVALID_TYPE
    if candidate.operation not in policy.allowed_operations:
        return ReasonCode.INVALID_OPERATION
    if candidate.source_kind not in policy.allowed_sources:
        return ReasonCode.SOURCE_NOT_ALLOWED
    if candidate.source_kind not in policy.auto_apply_sources:
        return ReasonCode.SOURCE_REQUIRES_VERIFICATION
    if (candidate.source_kind == SourceKind.INFERRED
            and policy.min_inferred_confidence is not None
            and candidate.confidence_hint < policy.min_inferred_confidence):
        return ReasonCode.CONFIDENCE_BELOW_AUTO_APPLY
    if candidate.operation in {Operation.SET, Operation.APPEND} and not _is_json_value(candidate.proposed_value):
        return ReasonCode.INVALID_VALUE
    return None


def _audit(
    candidate: ObservationCandidate,
    current: SessionStateItem | None,
    new_item: SessionStateItem | None,
    decision: TransitionDecision,
    reason: ReasonCode,
    now: str,
) -> TransitionAudit:
    old_ref = current.id if current else None
    new_ref = new_item.id if new_item else None
    transition_id = _stable_id(
        "transition",
        candidate.candidate_id,
        old_ref or "",
        new_ref or "",
        decision,
        reason,
        REDUCER_VERSION,
    )
    return TransitionAudit(
        transition_id=transition_id,
        session_id=candidate.session_id,
        candidate_id=candidate.candidate_id,
        old_state_ref=old_ref,
        new_state_ref=new_ref,
        decision=decision,
        reason_code=reason,
        derived_from=candidate.derived_from,
        reducer_version=REDUCER_VERSION,
        created_at=now,
    )


def _result(
    candidate: ObservationCandidate,
    current: SessionStateItem | None,
    item: SessionStateItem | None,
    decision: TransitionDecision,
    reason: ReasonCode,
    now: str,
) -> TransitionResult:
    return TransitionResult(item, _audit(candidate, current, item, decision, reason, now))


def _new_item(candidate: ObservationCandidate, now: str) -> SessionStateItem:
    state_id = _stable_id(
        "session-state",
        candidate.session_id,
        candidate.domain,
        candidate.state_type,
        candidate.key,
    )
    return SessionStateItem(
        id=state_id,
        session_id=candidate.session_id,
        domain=candidate.domain,
        state_type=candidate.state_type,
        key=candidate.key,
        value_json=candidate.proposed_value,
        status=StateStatus.ACTIVE,
        source_kind=candidate.source_kind,
        derived_from=_merge_refs(candidate.derived_from),
        extractor_version=candidate.extractor_version,
        reducer_version=REDUCER_VERSION,
        valid_from=candidate.valid_from or now,
        expires_at=candidate.expires_at,
        created_at=now,
        updated_at=now,
    )


def _same_slot(candidate: ObservationCandidate, current: SessionStateItem) -> bool:
    return (
        candidate.session_id == current.session_id
        and candidate.domain == current.domain
        and candidate.state_type == current.state_type
        and candidate.key == current.key
    )


def _lower_precedence(candidate: ObservationCandidate, current: SessionStateItem) -> bool:
    return _SOURCE_PRECEDENCE[candidate.source_kind] < _SOURCE_PRECEDENCE[current.source_kind]


def _apply_set(
    candidate: ObservationCandidate, current: SessionStateItem | None, now: str
) -> TransitionResult:
    if current is None:
        item = _new_item(candidate, now)
        return _result(candidate, None, item, TransitionDecision.APPLIED, ReasonCode.APPLIED_NEW, now)
    if current.value_json == candidate.proposed_value and current.status == StateStatus.ACTIVE:
        return _result(candidate, current, current, TransitionDecision.APPLIED, ReasonCode.DUPLICATE_NOOP, now)
    if _lower_precedence(candidate, current):
        return _result(
            candidate, current, current, TransitionDecision.CONFLICTED,
            ReasonCode.LOWER_PRECEDENCE_CONFLICT, now,
        )
    item = replace(
        current,
        value_json=candidate.proposed_value,
        status=StateStatus.ACTIVE,
        source_kind=candidate.source_kind,
        derived_from=_merge_refs(current.derived_from, candidate.derived_from),
        extractor_version=candidate.extractor_version,
        reducer_version=REDUCER_VERSION,
        valid_from=candidate.valid_from or now,
        expires_at=candidate.expires_at,
        updated_at=now,
    )
    return _result(candidate, current, item, TransitionDecision.APPLIED, ReasonCode.APPLIED_UPDATE, now)


def _apply_append(
    candidate: ObservationCandidate, current: SessionStateItem | None, now: str
) -> TransitionResult:
    if current is None:
        item = _new_item(candidate, now)
        return _result(candidate, None, item, TransitionDecision.APPLIED, ReasonCode.APPLIED_APPEND, now)
    if current.value_json == candidate.proposed_value:
        return _result(candidate, current, current, TransitionDecision.APPLIED, ReasonCode.DUPLICATE_NOOP, now)
    return _result(
        candidate, current, current, TransitionDecision.CONFLICTED,
        ReasonCode.APPEND_KEY_CONFLICT, now,
    )


def _apply_terminal(
    candidate: ObservationCandidate, current: SessionStateItem | None, now: str
) -> TransitionResult:
    if current is None:
        return _result(candidate, None, None, TransitionDecision.REJECTED, ReasonCode.TARGET_MISSING, now)
    if _lower_precedence(candidate, current):
        return _result(
            candidate, current, current, TransitionDecision.CONFLICTED,
            ReasonCode.LOWER_PRECEDENCE_CONFLICT, now,
        )
    reason = ReasonCode.RESOLVED if candidate.operation == Operation.RESOLVE else ReasonCode.RETRACTED
    item = replace(
        current,
        status=StateStatus.RETRACTED,
        derived_from=_merge_refs(current.derived_from, candidate.derived_from),
        reducer_version=REDUCER_VERSION,
        updated_at=now,
    )
    return _result(candidate, current, item, TransitionDecision.RETRACTED, reason, now)


def _apply_touch(
    candidate: ObservationCandidate, current: SessionStateItem | None, now: str
) -> TransitionResult:
    if current is None:
        return _result(candidate, None, None, TransitionDecision.REJECTED, ReasonCode.TARGET_MISSING, now)
    if _lower_precedence(candidate, current):
        return _result(
            candidate, current, current, TransitionDecision.CONFLICTED,
            ReasonCode.LOWER_PRECEDENCE_CONFLICT, now,
        )
    item = replace(
        current,
        derived_from=_merge_refs(current.derived_from, candidate.derived_from),
        reducer_version=REDUCER_VERSION,
        updated_at=now,
    )
    return _result(candidate, current, item, TransitionDecision.APPLIED, ReasonCode.TOUCHED, now)


def reduce_candidate(
    candidate: ObservationCandidate,
    current: SessionStateItem | None,
    *,
    now: str,
) -> TransitionResult:
    """校验并归约一个候选，不产生任何副作用。"""
    timestamp = _require_utc_iso(now)
    reason = _validation_reason(candidate)
    if reason is not None:
        return _result(candidate, current, current, TransitionDecision.REJECTED, reason, timestamp)
    if current is not None and not _same_slot(candidate, current):
        return _result(
            candidate, current, current, TransitionDecision.REJECTED,
            ReasonCode.SCOPE_MISMATCH, timestamp,
        )

    policy = policy_for(candidate.domain, candidate.state_type)
    if candidate.operation == Operation.SET:
        return _apply_set(candidate, current, timestamp)
    if candidate.operation == Operation.APPEND and policy and policy.update_mode == UpdateMode.EVENT:
        return _apply_append(candidate, current, timestamp)
    if candidate.operation in {Operation.RESOLVE, Operation.RETRACT}:
        return _apply_terminal(candidate, current, timestamp)
    if candidate.operation == Operation.TOUCH:
        return _apply_touch(candidate, current, timestamp)
    return _result(
        candidate, current, current, TransitionDecision.REJECTED,
        ReasonCode.INVALID_OPERATION, timestamp,
    )


def expire_state_item(item: SessionStateItem, *, now: str) -> SessionStateItem:
    """``expires_at`` 到期时返回过期副本，否则返回原条目。"""
    timestamp = _require_utc_iso(now)
    if not item.expires_at or item.status != StateStatus.ACTIVE:
        return item
    expires_at = _require_utc_iso(item.expires_at)
    if datetime.fromisoformat(expires_at) > datetime.fromisoformat(timestamp):
        return item
    return replace(item, status=StateStatus.EXPIRED, reducer_version=REDUCER_VERSION, updated_at=timestamp)
