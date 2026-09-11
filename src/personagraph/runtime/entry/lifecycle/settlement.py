"""针对已持久化 Entry 结算的只读恢复投影。

Entry 负责 Window 转换、租约检查、事件发射及所有持久化 finalizer。本模块只会在
finalizer 响应丢失时重读已提交的通用 Entry 事实；绝不启动事务或更改执行 Window。
"""

from __future__ import annotations

from typing import Any, Protocol

from ...turn.contracts import (
    AcceptedEntryTurn,
    EntryTurnResult,
)
from ...turn.persisted_projection import (
    require_entry_window_state as _require_entry_window_state,
)
from ..routing.policy import ProcessingLevel


class EntrySettlementRecoveryStorePort(Protocol):
    """结算恢复所需的 Entry 通用只读持久化事实。"""

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None: ...

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...

    def list_turn_linked_work_run_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]: ...


def recover_completed_entry_turn_from_commit(
    *,
    accepted: AcceptedEntryTurn,
    processing_level: ProcessingLevel,
    related_insession_task_ids: tuple[str, ...],
    store: EntrySettlementRecoveryStorePort,
    work_run_ids: tuple[str, ...] = (),
    error_code: str | None = None,
    end_reason: str | None = None,
) -> EntryTurnResult | None:
    """finalizer 抛错后，按 commit_<turn_id> 重读正式对话对，识别“已提交但回执丢失”。

    只有匹配原 Turn 的持久 assistant 正文和可解释的当前 Window 才能返回 completed；
    无法证明则返回 None，由 Entry 继续中断处理。此处不重新运行模型、不创建消息，
    也不把 Store 抛错一概解释成“事务一定回滚”。
    """

    try:
        committed = store.get_committed_turn_pair(
            accepted.session_id,
            f"commit_{accepted.turn_id}",
        )
        if committed is None or str(committed.get("turn_id") or "") != accepted.turn_id:
            return None
        reply = committed.get("assistant_content")
        if not isinstance(reply, str):
            return None
        inspected = store.inspect_turn_execution(accepted.session_id)
        window = inspected.get("window")
        if not isinstance(window, dict):
            return None
        window_state = _require_entry_window_state(window.get("window_state"))
        window_revision = int(window.get("state_version") or 0)
        if processing_level == "L1":
            related_insession_task_ids = ()
            work_run_ids = ()
        elif not work_run_ids:
            work_run_ids = store.list_turn_linked_work_run_ids(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
            )
    except Exception:
        return None
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="completed",
        processing_level=processing_level,
        reply=reply,
        error_code=error_code,
        end_reason=end_reason,
        related_insession_task_ids=related_insession_task_ids,
        work_run_ids=work_run_ids,
        window_state=window_state,
        window_revision=window_revision,
    )


__all__ = [
    "EntrySettlementRecoveryStorePort",
    "recover_completed_entry_turn_from_commit",
]
