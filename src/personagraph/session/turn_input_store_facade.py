"""受信 Turn 输入附属持久化的 Store 组合门面。

``session.store`` 保留既定公开导入路径。本模块只负责动态依赖组合：为每项持久 Turn
信封附属操作解析新的 ``StoreDeps``，再把未变更的附件、runtime-turn、结转和事件参数
直接转发给附件和 runtime-turn 持久化记录。它刻意不拥有 SQLite 事务、执行 Window 状态机、
Runtime/API 策略或 schema 行为。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .persistence.deps import StoreDeps
from .persistence.turns import attachments, runtime_turns


# 保留既定 Store 契约身份；校验和事务仍归底层记录所有。
MAX_RUNTIME_TURN_EVENT_PAGE_SIZE = runtime_turns.MAX_RUNTIME_TURN_EVENT_PAGE_SIZE
UnknownRuntimeTurnEventCursor = runtime_turns.UnknownRuntimeTurnEventCursor
AttachmentBindingError = attachments.AttachmentBindingError


class TurnInputStoreFacade:
    """保留 Store 的 Turn 输入接口，但不拥有持久化。"""

    def __init__(self, *, deps_factory: Callable[[], StoreDeps]) -> None:
        self._deps_factory = deps_factory

    def create_runtime_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        source: str,
        user_text: str,
    ) -> dict[str, object]:
        return runtime_turns.create_runtime_turn(
            self._deps_factory(),
            turn_id=turn_id,
            session_id=session_id,
            source=source,
            user_text=user_text,
        )

    def complete_runtime_turn(
        self,
        *,
        turn_id: str,
        status: str,
        processing_level: str | None,
        error_code: str | None = None,
    ) -> dict[str, object]:
        return runtime_turns.complete_runtime_turn(
            self._deps_factory(),
            turn_id=turn_id,
            status=status,  # type: ignore[arg-type]
            processing_level=processing_level,  # type: ignore[arg-type]
            error_code=error_code,
        )

    def create_attachment(self, **kwargs: Any) -> dict[str, Any]:
        """记录一个已存储上传，它尚未附加到任何 Turn。"""

        return attachments.create_attachment(self._deps_factory(), **kwargs)

    def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        return attachments.get_attachment(self._deps_factory(), attachment_id)

    def list_turn_attachments(
        self,
        session_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]:
        """按上传顺序列出绑定到一个 Turn 的附件。"""

        return attachments.list_turn_attachments(
            self._deps_factory(),
            session_id,
            turn_id,
        )

    def list_unbound_attachments(self, session_id: str) -> list[dict[str, Any]]:
        """列出用户尚未发送的上传。"""

        return attachments.list_unbound_attachments(
            self._deps_factory(),
            session_id,
        )

    def bind_attachments_to_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        attachment_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """把上传文件附加到一个 Turn，操作全部成功或全部失败。"""

        return attachments.bind_attachments_to_turn(
            self._deps_factory(),
            session_id=session_id,
            turn_id=turn_id,
            attachment_ids=attachment_ids,
        )

    def delete_unbound_attachment(
        self,
        *,
        session_id: str,
        attachment_id: str,
    ) -> dict[str, Any] | None:
        """丢弃暂存上传；已绑定附件绝不可移除。"""

        return attachments.delete_unbound_attachment(
            self._deps_factory(),
            session_id=session_id,
            attachment_id=attachment_id,
        )

    def session_attachment_total_bytes(self, session_id: str) -> int:
        return attachments.session_attachment_total_bytes(
            self._deps_factory(),
            session_id,
        )

    def list_pending_carry_over_turns(
        self,
        session_id: str,
        *,
        error_codes: tuple[str, ...],
        limit: int = 16,
    ) -> list[dict[str, object]]:
        """列出仍待合并、已接受但未回答的 Turn 输入。"""

        return runtime_turns.list_pending_carry_over_turns(
            self._deps_factory(),
            session_id,
            error_codes=error_codes,
            limit=limit,
        )

    def list_turn_input_segments(self, turn_id: str) -> list[dict[str, object]]:
        """重建组成合并 Turn 的各条独立用户消息。"""

        return runtime_turns.list_turn_input_segments(self._deps_factory(), turn_id)

    def attach_runtime_turn_carry_over(
        self,
        *,
        turn_id: str,
        merged_user_text: str,
        carried_turn_ids: tuple[str, ...],
    ) -> int:
        """原子采用合并输入，并把其来源 Turn 标记为已消费。"""

        return runtime_turns.attach_runtime_turn_carry_over(
            self._deps_factory(),
            turn_id=turn_id,
            merged_user_text=merged_user_text,
            carried_turn_ids=carried_turn_ids,
        )

    def append_runtime_turn_event(
        self,
        event: Any,
        *,
        active_window_lease_owner: str | None = None,
    ) -> int:
        """追加事件；若提供了自有活动 Window，则原子续租。"""

        return runtime_turns.append_runtime_turn_event(
            self._deps_factory(),
            event,
            active_window_lease_owner=active_window_lease_owner,
        )

    def list_runtime_turn_events(
        self,
        session_id: str,
        *,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        return runtime_turns.list_runtime_turn_events(
            self._deps_factory(),
            session_id,
            after=after,
            limit=limit,
        )


def build_turn_input_store_facade(
    *,
    deps_factory: Callable[[], StoreDeps],
) -> TurnInputStoreFacade:
    """构建 Store Turn 输入门面，暂不解析依赖。"""

    return TurnInputStoreFacade(deps_factory=deps_factory)
