"""对 SessionContext 状态项执行纯 Turn 窗口生命周期评估。

本模块根据目录策略、稳定 Turn 证据 ID 和最新持久化对话标记派生确定性到期转换。它不
查询 SQLite、不调用模型、不修改证据，也不访问 LangGraph 状态。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

from .catalog import policy_for
from .models import (
    Lifetime,
    ReasonCode,
    SessionStateItem,
    StateStatus,
    TransitionAudit,
    TransitionDecision,
    TransitionResult,
)
from .reducer import REDUCER_VERSION


@dataclass(frozen=True)
class TurnWindowSweep:
    """确定性到期结果及有界可观测计数器。"""

    transitions: tuple[TransitionResult, ...]
    checked_count: int
    not_due_count: int
    unevaluable_count: int


def sweep_turn_windows(
    items: Sequence[SessionStateItem],
    *,
    user_turn_markers: Sequence[tuple[int, str]],
) -> TurnWindowSweep:
    """返回已到期转换，不持久化任何变化。"""
    transitions = []
    checked = not_due = unevaluable = 0
    for item in items:
        policy = policy_for(item.domain, item.state_type)
        if policy is None or policy.lifetime != Lifetime.TURN_WINDOW:
            continue
        checked += 1
        origin_turn_idx = _latest_origin_turn_idx(item)
        if origin_turn_idx is None or policy.default_turn_ttl is None:
            unevaluable += 1
            continue
        subsequent_turns = [
            marker for marker in user_turn_markers if marker[0] > origin_turn_idx
        ]
        if len(subsequent_turns) < policy.default_turn_ttl:
            not_due += 1
            continue
        boundary_idx, boundary_time = subsequent_turns[policy.default_turn_ttl - 1]
        transitions.append(_expiry_transition(item, boundary_idx, boundary_time))
    return TurnWindowSweep(tuple(transitions), checked, not_due, unevaluable)


def _latest_origin_turn_idx(item: SessionStateItem) -> int | None:
    prefix = f"turn:{item.session_id}:"
    indices = []
    for evidence_id in item.derived_from:
        if not evidence_id.startswith(prefix):
            continue
        raw_index = evidence_id[len(prefix):]
        if raw_index.isdigit():
            indices.append(int(raw_index))
    return max(indices) if indices else None


def _expiry_transition(
    item: SessionStateItem,
    boundary_turn_idx: int,
    observed_at: str,
) -> TransitionResult:
    candidate_id = f"lifecycle-expiry:{item.id}:{boundary_turn_idx}"
    raw = "\x1f".join((candidate_id, item.id, REDUCER_VERSION))
    transition_id = "transition:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    expired = replace(
        item,
        status=StateStatus.EXPIRED,
        reducer_version=REDUCER_VERSION,
        updated_at=observed_at,
    )
    audit = TransitionAudit(
        transition_id=transition_id,
        session_id=item.session_id,
        candidate_id=candidate_id,
        old_state_ref=item.id,
        new_state_ref=item.id,
        decision=TransitionDecision.EXPIRED,
        reason_code=ReasonCode.EXPIRED,
        derived_from=item.derived_from,
        reducer_version=REDUCER_VERSION,
        created_at=observed_at,
    )
    return TransitionResult(expired, audit)
