"""L1 的收尾事实与运行通知选择；不执行模型调用或 Session 消息事务。"""

from collections.abc import Mapping

from ...context_budget import ContextBudgetExceeded
from ...model_io.gateway import ModelGatewayError
from ...persistent_turn_content.delivery import (
    L1TerminalNotification,
    build_l1_terminal_notification,
)
from ..turn_events import RuntimeErrorCode


def terminal_l1_notification(error: BaseException) -> L1TerminalNotification | None:
    """仅为已停止的 L1 运行选择安全通知，不把被拒候选转为通过。"""
    if isinstance(error, ModelGatewayError):
        failure_code = error.code
    elif isinstance(error, ContextBudgetExceeded):
        failure_code = RuntimeErrorCode.CONTEXT_BUDGET_EXCEEDED.value
    else:
        # 不向上依赖 controller 的异常类；只有其带类型的公开分类可进入此出口。
        code = getattr(error, "error_code", None)
        if not isinstance(code, RuntimeErrorCode):
            return None
        failure_code = code.value
    return build_l1_terminal_notification(failure_code)


def project_l1_review_stop_context(
    model_view: Mapping[str, object], execution: Mapping[str, object],
) -> dict[str, object]:
    """投影候选决策时冻结的额度；不读取当前时间，保证重放仍是同一审查输入。"""
    raw_limits = model_view.get("execution_limits")
    limits = raw_limits if isinstance(raw_limits, Mapping) else {}
    raw_state = execution.get("state")
    state = raw_state if isinstance(raw_state, Mapping) else {}
    remaining = limits.get("attempts_remaining")
    maximum = state.get("max_attempts")
    remaining = remaining if type(remaining) is int and remaining >= 0 else None
    maximum = maximum if type(maximum) is int and maximum > 0 else None
    must_finalize = limits.get("finalization_required")
    must_finalize = must_finalize if isinstance(must_finalize, bool) else None
    return {
        "source": "host_frozen_execution_state",
        "attempt_limit": maximum,
        "candidate_attempt_ordinal": (
            maximum - remaining + 1
            if maximum is not None and remaining is not None and 0 < remaining <= maximum
            else None
        ),
        "attempts_remaining_after_candidate": (
            max(0, remaining - 1) if remaining is not None else None
        ),
        "must_finalize": must_finalize,
        "stop_reason": "attempt_limit_after_candidate" if must_finalize else None,
        "milliseconds_until_deadline_at_decision": limits.get("milliseconds_until_deadline"),
    }


__all__ = ["project_l1_review_stop_context", "terminal_l1_notification"]
