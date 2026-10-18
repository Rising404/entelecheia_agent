"""Project accepted Turn replays without rerunning models or tools.

L0/L1 recovery remains owned by Entry.
"""

from __future__ import annotations

from typing import Any, Protocol

from ...turn.contracts import AcceptedEntryTurn, EntryTurnResult
from ...turn_events import RuntimeStage


class EntryReplayReadPort(Protocol):
    """投影或对账一次 request-id 重放所需的持久化事实。"""

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None: ...

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...


def reconcile_replayed_entry_turn(
    *,
    accepted: AcceptedEntryTurn,
    store: EntryReplayReadPort,
) -> EntryTurnResult:
    """返回已接受 request-id 重放的权威结果。

    重放只读取正式交付与持久中断状态，不启动 Provider 或工具。
    """

    # 已确定的处理层级优先；尚未分类时才用该 Turn 冻结的策略识别 L1 模式。
    l1_replay = accepted.processing_level == "L1" or (
        accepted.processing_level is None and accepted.routing_policy.allows("L1")
    )
    if accepted.turn_status == "completed":
        committed = store.get_committed_turn_pair(
            accepted.session_id,
            f"commit_{accepted.turn_id}",
        )
        if committed is None:
            raise RuntimeError("completed Turn is missing its formal delivery")
        return EntryTurnResult(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
            status="completed",
            processing_level=accepted.processing_level,
            reply=str(committed["assistant_content"]),
            error_code=accepted.error_code,
            end_reason=accepted.end_reason,
            window_state=accepted.window_state,
            window_revision=accepted.window_revision,
        )

    if accepted.turn_status == "running" and accepted.window_state == "interrupted":
        try:
            inspected = store.inspect_turn_execution(accepted.session_id)
            interrupted_window = inspected.get("window")
        except Exception:
            interrupted_window = None
        if (
            isinstance(interrupted_window, dict)
            and str(interrupted_window.get("turn_id") or "") == accepted.turn_id
            and str(interrupted_window.get("window_state") or "") == "interrupted"
            and str(interrupted_window.get("interruption_reason") or "")
            == "CONTEXT_BUDGET_EXCEEDED"
        ):
            return EntryTurnResult(
                session_id=accepted.session_id,
                turn_id=accepted.turn_id,
                status="incomplete",
                processing_level=accepted.processing_level,
                end_reason="context_budget_exceeded",
                error_code="CONTEXT_BUDGET_EXCEEDED",
                window_state="interrupted",
                window_revision=int(interrupted_window.get("state_version") or 0),
            )

    if accepted.turn_status == "running" and l1_replay:
        try:
            inspected = store.inspect_turn_execution(accepted.session_id)
            l1_window = inspected.get("window")
        except Exception:
            l1_window = None
        if (
            isinstance(l1_window, dict)
            and str(l1_window.get("turn_id") or "") == accepted.turn_id
            and str(l1_window.get("window_state") or "") == "interrupted"
            and l1_window.get("current_l1_turn_run_id") is not None
        ):
            reason = str(l1_window.get("interruption_reason") or "")
            end_reason = None
            if reason == "TOOL_COMPLETION_UNCONFIRMED":
                # 未确认副作用只能投影持久中断；request-id 重放不能重新派发工具。
                end_reason = "tool_completion_unconfirmed"
            elif (
                reason == "L1_RUNTIME_NOT_READY"
                and str(l1_window.get("stage") or "") == RuntimeStage.L1_BOOTSTRAP.value
            ):
                end_reason = "l1_runtime_not_ready"
            if end_reason is not None:
                return EntryTurnResult(
                    session_id=accepted.session_id,
                    turn_id=accepted.turn_id,
                    status="incomplete",
                    processing_level="L1",
                    end_reason=end_reason,
                    error_code=reason,
                    window_state="interrupted",
                    window_revision=int(l1_window.get("state_version") or 0),
                )

    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status=accepted.turn_status,
        processing_level=accepted.processing_level,
        end_reason=accepted.end_reason,
        error_code=accepted.error_code,
        # Store 在 request-id 重放时返回当前 Session Window。此 schema 中不存在历史
        # Window 快照，因此保留该契约，而不伪造旧的中断投影。
        window_state=accepted.window_state,
        window_revision=accepted.window_revision,
    )


__all__ = [
    "EntryReplayReadPort",
    "reconcile_replayed_entry_turn",
]
