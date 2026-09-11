"""在接受下一输入前审计持久轮次执行窗口。

执行窗口刻意不是由事件派生的状态机。本模块在接受新轮次前读取其存储投影，
并且只执行当前 Slice A 能够证明安全的少量显式修复：

* 将已知或已放弃的中断轮次结算为 ``incomplete``；
* 释放所需提交后作业均已完成的轮次；
* 拒绝当前轮次或派生状态作业仍然活动的会话。

它不会重放模型调用、推断 Task/WorkRun 状态，也不会把自由文本转成恢复命令。
这些选择属于后续 L2 循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from ....session.turn_execution_contracts import (
    TurnExecutionLeaseConflict,
    TurnExecutionWindowRevisionConflict,
)
from ...turn.contracts import EntryRecoveryProjection
from ...turn.timing import ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S


# 恢复审计是修复路径，并非调度器。并发持有者可在本次读取与条件变更之间合法
# 推进窗口；因此只执行少量有界重读，随后保留槽位，而不是猜测哪个持有者应当
# 获胜。
TURN_WINDOW_AUDIT_MAX_CONFLICT_RETRIES = 2


class TurnWindowAuditStore(Protocol):
    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...
    def mark_turn_execution_interrupted(self, **kwargs: object) -> dict[str, object]: ...
    def settle_interrupted_turn_execution(self, **kwargs: object) -> dict[str, object]: ...
    def release_turn_execution_window(self, **kwargs: object) -> dict[str, object]: ...


class TurnWindowBlockedError(RuntimeError):
    """会话的上一轮次尚未到达安全交接点。"""

    code = "TURN_IN_PROGRESS"

    def __init__(self, window: dict[str, object], *, reason: str) -> None:
        self.details = {
            "session_id": str(window.get("session_id") or ""),
            "turn_id": str(window.get("turn_id") or ""),
            "window_state": str(window.get("window_state") or ""),
            "window_revision": int(window.get("state_version") or 0),
            "reason": reason,
        }
        super().__init__(self.code)


@dataclass(frozen=True)
class TurnWindowAuditResult:
    """可补充到新入口模型中的唯一安全前序轮次事实。"""

    recovery_projection: EntryRecoveryProjection | None = None


def audit_turn_window_before_accept(
    *,
    session_id: str,
    store: TurnWindowAuditStore,
    has_local_execution: bool,
) -> TurnWindowAuditResult:
    """在新轮次声明旧槽位前对其执行审计。

    进程本地防护只能证明本地存活性。另一个运行时进程可能持有同一共享 SQLite
    数据库，因此新鲜的持久心跳/租约同样具有权威，会阻止竞争输入。只有过时的
    活动租约才会被标记并结算为 ``process_lost``。
    """

    for _ in range(TURN_WINDOW_AUDIT_MAX_CONFLICT_RETRIES):
        inspection = store.inspect_turn_execution(session_id)
        window = inspection.get("window")
        if not isinstance(window, dict) or window.get("turn_id") is None:
            return TurnWindowAuditResult()

        state = str(window.get("window_state") or "")
        if state == "empty":
            return TurnWindowAuditResult()
        if state == "active":
            if has_local_execution:
                raise TurnWindowBlockedError(window, reason="active_local_execution")
            if _active_window_lease_is_fresh(
                window,
                observed_at=datetime.now(timezone.utc),
            ):
                raise TurnWindowBlockedError(window, reason="active_remote_execution")
            try:
                marked = store.mark_turn_execution_interrupted(
                    session_id=session_id,
                    turn_id=str(window["turn_id"]),
                    expected_window_revision=int(window["state_version"]),
                    stage=str(window.get("stage") or "RESPONSE"),
                    interruption_reason="process_lost",
                    expected_heartbeat_at=_optional_text(window.get("heartbeat_at")),
                    require_heartbeat_match=True,
                )
                return _settle_interrupted_window(
                    session_id=session_id,
                    store=store,
                    window=marked,
                )
            except (TurnExecutionLeaseConflict, TurnExecutionWindowRevisionConflict):
                # 活动持有者可能在本次读取后续订了心跳或推进了阶段。应根据持久
                # 状态重新分类，而不是将存储冲突泄漏为 500 或宣告丢失。
                continue
        if state == "interrupted":
            try:
                return _settle_interrupted_window(
                    session_id=session_id,
                    store=store,
                    window=window,
                )
            except TurnExecutionWindowRevisionConflict:
                continue
        if state == "post_commit_pending":
            jobs = inspection.get("post_commit_jobs")
            if not isinstance(jobs, list) or any(
                not isinstance(job, dict) or job.get("status") not in {"applied", "waived"}
                for job in jobs
            ):
                raise TurnWindowBlockedError(window, reason="post_commit_pending")
            try:
                store.release_turn_execution_window(
                    session_id=session_id,
                    turn_id=str(window["turn_id"]),
                    expected_window_revision=int(window["state_version"]),
                )
                return TurnWindowAuditResult()
            except TurnExecutionWindowRevisionConflict:
                continue
        raise TurnWindowBlockedError(window, reason="unknown_window_state")

    # 每次尝试结算时窗口都发生了移动。重新读取可在活动/本地错误变得明确时保留
    # 这一有用信息；否则以封闭方式失败，而不执行无界修复循环。
    inspection = store.inspect_turn_execution(session_id)
    window = inspection.get("window")
    if not isinstance(window, dict) or window.get("turn_id") is None:
        return TurnWindowAuditResult()
    if str(window.get("window_state") or "") == "empty":
        return TurnWindowAuditResult()
    if str(window.get("window_state") or "") == "active":
        if has_local_execution:
            raise TurnWindowBlockedError(window, reason="active_local_execution")
        if _active_window_lease_is_fresh(
            window,
            observed_at=datetime.now(timezone.utc),
        ):
            raise TurnWindowBlockedError(window, reason="active_remote_execution")
    raise TurnWindowBlockedError(window, reason="window_recovery_race")


def _settle_interrupted_window(
    *,
    session_id: str,
    store: TurnWindowAuditStore,
    window: dict[str, object],
) -> TurnWindowAuditResult:
    reason, error_code = _public_interruption_outcome(
        str(window.get("interruption_reason") or "")
    )
    settled = store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=str(window["turn_id"]),
        expected_window_revision=int(window["state_version"]),
        end_reason=reason,
        error_code=error_code,
    )
    turn = settled.get("turn")
    input_message = settled.get("input_message")
    if not isinstance(turn, dict) or not isinstance(input_message, dict):
        raise RuntimeError("interrupted Turn settlement returned an invalid projection")
    return TurnWindowAuditResult(
        recovery_projection=EntryRecoveryProjection(
            turn_id=str(turn["turn_id"]),
            end_reason=str(turn.get("end_reason") or reason),
            error_code=_optional_text(turn.get("error_code")),
            input_message_id=str(input_message["message_id"]),
        )
    )


def _public_interruption_outcome(marker: str) -> tuple[str, str | None]:
    """将宿主标记转换为精简的公开异常轮次词汇。"""

    if marker == "TOOL_COMPLETION_UNCONFIRMED":
        return "host_stopped", marker
    if marker == "MODEL_COMPLETION_UNCONFIRMED":
        return "host_stopped", marker
    if marker == "TURN_DEADLINE_EXCEEDED":
        return "host_stopped", marker
    if marker == "CONTEXT_BUDGET_EXCEEDED":
        return "context_budget_exceeded", marker
    if marker in {
        "MODEL_TIMEOUT",
        "MODEL_TRANSPORT_FAILURE",
        "MODEL_OUTPUT_INVALID",
        "MODEL_CONFIGURATION_FAILURE",
        "VERIFICATION_FAILED",
    }:
        return "provider_unavailable", marker
    if marker == "PERSIST_FAILED":
        return "persistence_error", marker
    if marker == "INTERNAL_FAILURE":
        return "module_error", marker
    if marker == "user_paused":
        return "user_paused", None
    if marker == "host_stopped":
        return "host_stopped", None
    if marker == "process_lost":
        return "process_lost", None
    return "unknown", marker or None


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _active_window_lease_is_fresh(
    window: dict[str, object],
    *,
    observed_at: datetime,
) -> bool:
    """返回有效持久持有者最近是否续订过窗口。

    未来时间戳会被保守视为仍然活动：时钟偏差不得导致仍被持有的轮次被重新
    分类为进程丢失。
    """

    owner = window.get("lease_owner")
    heartbeat_at = _optional_text(window.get("heartbeat_at"))
    if not isinstance(owner, str) or not owner.strip() or heartbeat_at is None:
        return False
    try:
        heartbeat = datetime.fromisoformat(heartbeat_at)
    except ValueError:
        return False
    if heartbeat.tzinfo is None:
        return False
    return observed_at - heartbeat <= timedelta(
        seconds=ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S
    )
