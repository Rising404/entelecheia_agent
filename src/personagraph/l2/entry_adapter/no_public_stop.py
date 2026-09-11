"""纯验证一个由 Store 持有的权威非公开结算。

Entry 继续负责 Store reducer、重放/租约决策、事件发射和公开
``EntryTurnResult`` 构造。本模块只验证 reducer 选择非公开停止后返回的原始结算事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from personagraph.runtime.turn_events import RuntimeErrorCode, RuntimeStage


AuthoritativeNoPublicStopEndReason = Literal["host_stopped", "module_error"]
_AuthoritativeNoPublicStopKind = Literal[
    "waiting_external",
    "turn_limit_reached",
    "work_run_failed",
]
_ExpectedAuthoritativeNoPublicStop = tuple[
    RuntimeStage,
    AuthoritativeNoPublicStopEndReason,
    _AuthoritativeNoPublicStopKind,
]


@dataclass(frozen=True, slots=True)
class AuthoritativeNoPublicStopSettlementProjection:
    """供 Entry 现有 incomplete 投影使用的已验证不可变事实。"""

    window_revision: int
    end_reason: AuthoritativeNoPublicStopEndReason
    error_code: RuntimeErrorCode
    stage: RuntimeStage


_EXPECTED_AUTHORITATIVE_NO_PUBLIC_STOPS: dict[
    RuntimeErrorCode,
    _ExpectedAuthoritativeNoPublicStop,
] = {
    RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED: (
        RuntimeStage.TOOL,
        "host_stopped",
        "waiting_external",
    ),
    RuntimeErrorCode.TURN_DEADLINE_EXCEEDED: (
        RuntimeStage.RESPONSE,
        "host_stopped",
        "turn_limit_reached",
    ),
    RuntimeErrorCode.INTERNAL_FAILURE: (
        RuntimeStage.RESPONSE,
        "module_error",
        "work_run_failed",
    ),
}


def project_authoritative_no_public_stop_settlement(
    settled: object,
    *,
    expected_turn_id: str,
) -> AuthoritativeNoPublicStopSettlementProjection | None:
    """在不与 Store 交互的情况下验证现有原始 reducer 结果。

    精确保留原有 Entry 解析行为：只接受具体 ``dict`` 值；enum/revision 解析期间
    只有 ``TypeError``/``ValueError`` 会转为 ``None``；不投影无法识别的错误标记。
    """

    if not isinstance(settled, dict):
        return None
    window = settled.get("window")
    if not isinstance(window, dict):
        return None
    try:
        stage = RuntimeStage(str(settled.get("stage") or ""))
        error_code = RuntimeErrorCode(str(settled.get("error_code") or ""))
        revision = int(window.get("state_version") or 0)
    except (TypeError, ValueError):
        return None
    expected_outcome = _EXPECTED_AUTHORITATIVE_NO_PUBLIC_STOPS.get(error_code)
    end_reason = str(settled.get("end_reason") or "")
    stop_kind = str(settled.get("stop_kind") or "")
    if (
        expected_outcome != (stage, end_reason, stop_kind)
        or str(window.get("turn_id") or "") != expected_turn_id
        or str(window.get("window_state") or "") != "interrupted"
        or str(window.get("stage") or "") != stage.value
        or str(window.get("interruption_reason") or "") != error_code.value
        or revision < 1
    ):
        return None
    return AuthoritativeNoPublicStopSettlementProjection(
        window_revision=revision,
        end_reason=expected_outcome[1],
        error_code=error_code,
        stage=stage,
    )


__all__ = [
    "AuthoritativeNoPublicStopSettlementProjection",
    "project_authoritative_no_public_stop_settlement",
]
