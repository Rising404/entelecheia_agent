"""Session 元数据与对话历史的 Store 组合门面。

``session.store`` 保留既定公开导入路径。本模块只负责动态依赖组合：为每次元数据或
对话调用解析新的 ``StoreDeps``，并把未变更参数直接转发给对应持久化记录。它刻意排除
Session 创建、工作目录权威、工作记忆摘要、Runtime 策略、事务和 schema 所有权。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .persistence.deps import StoreDeps
from .persistence.history import turns
from .persistence.metadata import sessions


# 这些仍是原始持久化记录对象。门面只拥有动态 Store 依赖组合，不拥有对话或元数据权威。
TranscriptDeliveryReferenceError = turns.TranscriptDeliveryReferenceError
CommittedTurnPairIndexState = turns.CommittedTurnPairIndexState
CommittedTurnPairIndexBinding = turns.CommittedTurnPairIndexBinding
CommittedTurnPairIndexBindingSnapshot = turns.CommittedTurnPairIndexBindingSnapshot


class HistoryStoreFacade:
    """保留 Store 的历史公开接口，但不拥有持久化。"""

    def __init__(self, *, deps_factory: Callable[[], StoreDeps]) -> None:
        self._deps_factory = deps_factory

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return sessions.get_session(self._deps_factory(), session_id)

    def list_sessions(
        self,
        include_archived: bool = False,
        include_trashed: bool = False,
        *,
        status: str = "active",
        folder_id: str | None = None,
        persona_id: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return sessions.list_sessions(
            self._deps_factory(),
            include_archived,
            include_trashed,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            query=query,
            limit=limit,
        )

    def search_sessions(
        self,
        query: str,
        status: str = "active",
        folder_id: str | None = None,
        persona_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """搜索 Session 标题和 Turn；默认排除回收站中的 Session。"""

        return sessions.search_sessions(
            self._deps_factory(),
            query,
            status,
            folder_id,
            persona_id,
            limit,
        )

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        return turns.get_turns(self._deps_factory(), session_id)

    def get_turn(self, session_id: str, turn_idx: int) -> dict[str, Any] | None:
        """按稳定组合键返回一个权威对话行。"""

        return turns.get_turn(self._deps_factory(), session_id, turn_idx)

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None:
        """按稳定运行 ID 返回一个完整 Turn 对，供检索或回填使用。"""

        return turns.get_committed_turn_pair(
            self._deps_factory(),
            session_id,
            run_id,
        )

    def list_committed_turn_pairs(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """列出完整权威 Turn 对，排除不完整旧版片段。

        ``limit`` 只返回最近 N 对，顺序仍为最旧项在前。
        """

        return turns.list_committed_turn_pairs(
            self._deps_factory(),
            session_id,
            limit=limit,
        )

    def get_committed_turn_pair_index_state(
        self,
        session_id: str,
    ) -> CommittedTurnPairIndexState:
        """返回供检索使用、不含内容的已提交 Turn 对 revision 事实。"""

        return turns.get_committed_turn_pair_index_state(
            self._deps_factory(),
            session_id,
        )

    def get_committed_turn_pair_index_binding_snapshot(
        self,
        session_id: str,
        *,
        maximum_bindings: int,
    ) -> CommittedTurnPairIndexBindingSnapshot:
        """返回有界已提交 Turn 对指针，不加载对话文本。"""

        return turns.get_committed_turn_pair_index_binding_snapshot(
            self._deps_factory(),
            session_id,
            maximum_bindings=maximum_bindings,
        )

    def get_user_turn_markers(self, session_id: str) -> list[tuple[int, str]]:
        """返回有序用户 Turn 索引和时间戳，不加载 Turn 内容。"""

        return turns.get_user_turn_markers(self._deps_factory(), session_id)

    def get_history_messages(self, session_id: str) -> list[dict[str, str]]:
        """从权威历史表返回可用于 Prompt 的对话消息。"""

        return turns.get_history_messages(self._deps_factory(), session_id)


def build_history_store_facade(
    *,
    deps_factory: Callable[[], StoreDeps],
) -> HistoryStoreFacade:
    """构建 Store 历史门面，暂不解析依赖。"""

    return HistoryStoreFacade(deps_factory=deps_factory)
