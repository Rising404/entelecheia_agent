"""SessionContext 状态策略目录。

该目录定义允许的类型、来源、操作和生命周期，不包含抽取、持久化或运行时逻辑。
"""

from __future__ import annotations

from .models import (
    Lifetime,
    Operation,
    SessionDomain,
    SourceKind,
    StateTypePolicy,
    UpdateMode,
)


CATALOG_VERSION = "session-context-v1"

_SET = frozenset({Operation.SET, Operation.RETRACT, Operation.TOUCH})
_EVENT = frozenset({Operation.APPEND, Operation.RESOLVE, Operation.RETRACT, Operation.TOUCH})
_EXPLICIT = frozenset({SourceKind.EXPLICIT})
_EXPLICIT_INFERRED = frozenset({SourceKind.EXPLICIT, SourceKind.INFERRED})
_EXPLICIT_TOOL = frozenset({SourceKind.EXPLICIT, SourceKind.TOOL})
_TASK_SOURCES = frozenset({SourceKind.EXPLICIT, SourceKind.INFERRED, SourceKind.TOOL})
_ASSISTANT = frozenset({SourceKind.ASSISTANT})


def _policy(
    domain: SessionDomain,
    state_type: str,
    mode: UpdateMode,
    operations: frozenset[Operation],
    allowed: frozenset[SourceKind],
    auto_apply: frozenset[SourceKind],
    lifetime: Lifetime,
    *,
    turn_ttl: int | None = None,
    min_inferred_confidence: float | None = None,
) -> StateTypePolicy:
    return StateTypePolicy(
        domain=domain,
        state_type=state_type,
        update_mode=mode,
        allowed_operations=operations,
        allowed_sources=allowed,
        auto_apply_sources=auto_apply,
        lifetime=lifetime,
        default_turn_ttl=turn_ttl,
        min_inferred_confidence=min_inferred_confidence,
    )


_POLICIES = (
    _policy(SessionDomain.USER, "temporary_preference", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.USER, "affect_signal", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT_INFERRED, Lifetime.TURN_WINDOW,
            turn_ttl=6, min_inferred_confidence=0.85),
    _policy(SessionDomain.USER, "knowledge_state", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.USER, "interaction_feedback", UpdateMode.EVENT, _EVENT,
            _EXPLICIT, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.USER, "session_constraint", UpdateMode.SINGLETON, _SET,
            _EXPLICIT, _EXPLICIT, Lifetime.EXPLICIT),
    _policy(SessionDomain.USER, "situational_context", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.TURN_WINDOW, turn_ttl=6),
    _policy(SessionDomain.TASK, "task_focus", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT_INFERRED, Lifetime.TURN_WINDOW,
            turn_ttl=4, min_inferred_confidence=0.80),
    _policy(SessionDomain.TASK, "local_goal", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "current_step", UpdateMode.SINGLETON, _SET,
            _TASK_SOURCES, _EXPLICIT_TOOL, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "progress_delta", UpdateMode.EVENT, _EVENT,
            _TASK_SOURCES, _EXPLICIT_TOOL, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "blocker", UpdateMode.EVENT, _EVENT,
            _TASK_SOURCES, _EXPLICIT_TOOL, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "open_question", UpdateMode.EVENT, _EVENT,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "assumption", UpdateMode.EVENT, _EVENT,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "decision_candidate", UpdateMode.EVENT, _EVENT,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "deadline_candidate", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "artifact_ref", UpdateMode.EVENT, _EVENT,
            _EXPLICIT_TOOL, _EXPLICIT_TOOL, Lifetime.SESSION),
    _policy(SessionDomain.TASK, "tool_result_ref", UpdateMode.EVENT, _EVENT,
            frozenset({SourceKind.TOOL}), frozenset({SourceKind.TOOL}), Lifetime.SESSION),
    _policy(SessionDomain.INTERACTION, "referent", UpdateMode.SINGLETON, _SET,
            _EXPLICIT_INFERRED, _EXPLICIT_INFERRED, Lifetime.TURN_WINDOW,
            turn_ttl=4, min_inferred_confidence=0.80),
    _policy(SessionDomain.INTERACTION, "user_correction", UpdateMode.EVENT, _EVENT,
            _EXPLICIT, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.INTERACTION, "open_loop", UpdateMode.EVENT, _EVENT,
            _EXPLICIT_INFERRED, _EXPLICIT, Lifetime.SESSION),
    _policy(SessionDomain.INTERACTION, "assistant_commitment", UpdateMode.EVENT, _EVENT,
            _ASSISTANT, _ASSISTANT, Lifetime.SESSION),
)

TYPE_POLICIES: dict[tuple[SessionDomain, str], StateTypePolicy] = {
    (policy.domain, policy.state_type): policy for policy in _POLICIES
}


def policy_for(domain: SessionDomain, state_type: str) -> StateTypePolicy | None:
    """返回领域与类型组合对应的策略；不支持时返回 None。"""
    return TYPE_POLICIES.get((domain, state_type))


def policies_for_domain(domain: SessionDomain) -> tuple[StateTypePolicy, ...]:
    """按稳定声明顺序返回策略，供文档和测试使用。"""
    return tuple(policy for policy in _POLICIES if policy.domain == domain)


def all_policies() -> tuple[StateTypePolicy, ...]:
    """按稳定声明顺序返回完整目录。"""
    return _POLICIES
