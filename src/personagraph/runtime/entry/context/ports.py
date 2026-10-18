"""Entry context 组装所需的只读持久化边界。"""

from __future__ import annotations

from typing import Any, Protocol

from ....session.session_summary import SessionSummaryState


class EntryContextAssemblyStorePort(Protocol):
    """组装一份模型输入上下文所需的三项只读事实。"""

    def list_turn_attachments(
        self,
        session_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]: ...

    def get_session_summary_state(
        self,
        session_id: str,
    ) -> SessionSummaryState | None: ...

    def list_committed_turn_pairs(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


__all__ = ["EntryContextAssemblyStorePort"]
