"""恢复已验证、尚未发布的 L2 Delivery。

该投影只读取 L2 Delivery 与 Turn Window 的持久化事实。它不持有 Entry 生命周期
权限，也不会写入 Window、事件或最终回复。
"""

from __future__ import annotations

from typing import Protocol

from ...runtime.turn.contracts import AcceptedEntryTurn, EntryTurnResult
from ...runtime.turn_events import RuntimeStage


class L2PendingPublicationWindowStorePort(Protocol):
    """恢复投影读取活动 Turn Window 所需的最小接口。"""

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...


class L2CompletedVerifiedDeliveryStorePort(Protocol):
    """恢复投影读取已验证 Delivery 所需的最小接口。"""

    def list_turn_completed_verified_delivery_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]: ...


class _SessionCompletedVerifiedDeliveryStore:
    """按需解析 L2 Session store，保持适配模块冷启动。"""

    def list_turn_completed_verified_delivery_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]:
        from ...session.l2_store import work_run

        return work_run.list_turn_completed_verified_delivery_ids(
            session_id=session_id,
            turn_id=turn_id,
        )


_SESSION_COMPLETED_VERIFIED_DELIVERY_STORE = (
    _SessionCompletedVerifiedDeliveryStore()
)


def project_pending_verified_publication_result(
    *,
    accepted: AcceptedEntryTurn,
    delivery_id: str,
    related_insession_task_ids: tuple[str, ...],
    store: L2PendingPublicationWindowStorePort,
    verified_delivery_store: L2CompletedVerifiedDeliveryStorePort | None = None,
) -> EntryTurnResult | None:
    """在发布响应丢失后，投影精确且已验证的待发布 L2 结果。"""

    if verified_delivery_store is None:
        verified_delivery_store = _SESSION_COMPLETED_VERIFIED_DELIVERY_STORE
    try:
        if verified_delivery_store.list_turn_completed_verified_delivery_ids(
            session_id=accepted.session_id,
            turn_id=accepted.turn_id,
        ) != (delivery_id,):
            return None
        inspected = store.inspect_turn_execution(accepted.session_id)
        window = inspected.get("window")
        if (
            not isinstance(window, dict)
            or str(window.get("turn_id") or "") != accepted.turn_id
            or str(window.get("window_state") or "") != "active"
            or str(window.get("stage") or "") != RuntimeStage.PERSIST.value
            or any(
                window.get(field) is not None
                for field in (
                    "current_work_run_id",
                    "current_attempt_id",
                    "current_l1_turn_run_id",
                    "current_l1_attempt_id",
                    "latest_checkpoint_id",
                )
            )
        ):
            return None
        window_revision = int(window.get("state_version") or 0)
        if window_revision < 1:
            return None
    except Exception:
        return None
    return EntryTurnResult(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
        status="running",
        processing_level="L2",
        related_insession_task_ids=related_insession_task_ids,
        window_state="active",
        window_revision=window_revision,
    )


__all__ = [
    "L2CompletedVerifiedDeliveryStorePort",
    "L2PendingPublicationWindowStorePort",
    "project_pending_verified_publication_result",
]
