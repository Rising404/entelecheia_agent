"""读取 L2 Task 执行 lane 的持久化 replay 权威。

公共 Entry 决定是否恢复 Turn 及如何投影最终结果；本模块只负责 L2 专属的
receipt gate 与 lane manifest 读取。默认 Session 适配器按需解析持久化 owner，
因此不检查 L2 replay 的 Entry 分支不会加载 L2 store。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast


class L2ReplayTaskExecutionLaneStorePort(Protocol):
    """读取 L2 lane receipt 与 manifest 所需的最小接口。"""

    def has_turn_task_execution_lane_receipt(
        self,
        session_id: str,
        turn_id: str,
    ) -> bool: ...

    def get_insession_task_execution_lane_manifest(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> object: ...


class _SessionTaskExecutionLaneStore:
    """先读中性 receipt gate，再按需解析 L2 TaskGraph owner。"""

    def __init__(self, deps_factory: Callable[[], object]) -> None:
        self._deps_factory = deps_factory

    def has_turn_task_execution_lane_receipt(
        self,
        session_id: str,
        turn_id: str,
    ) -> bool:
        from ...session.persistence.turns.entry_replay import (
            has_turn_task_execution_lane_receipt,
        )

        return has_turn_task_execution_lane_receipt(
            self._deps_factory(),
            session_id,
            turn_id,
        )

    def get_insession_task_execution_lane_manifest(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> object:
        from ...session.l2_store import task_graph

        return task_graph.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )


def require_task_execution_lane_manifest(
    *,
    replay_store: object,
    task_execution_lane_store: L2ReplayTaskExecutionLaneStorePort | None,
    session_id: str,
    turn_id: str,
) -> object:
    """返回 receipt 证明存在的 L2 lane manifest，否则拒绝推测。"""

    lane_store = task_execution_lane_store
    if lane_store is None:
        deps_factory = getattr(replay_store, "current_store_deps", None)
        if not callable(deps_factory):
            raise RuntimeError("Entry replay store cannot resolve Session persistence")
        lane_store = _SessionTaskExecutionLaneStore(
            cast(Callable[[], object], deps_factory)
        )
    if not lane_store.has_turn_task_execution_lane_receipt(
        session_id,
        turn_id,
    ):
        raise LookupError("Turn has no Task execution lane receipt")
    return lane_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=turn_id,
    )


def inspect_task_execution_lane_manifest(
    *,
    replay_store: object,
    task_execution_lane_store: L2ReplayTaskExecutionLaneStorePort | None,
    session_id: str,
    turn_id: str,
) -> tuple[bool, object | None]:
    """Fail closed，并保留“manifest 存在但值为空”的读取语义。"""

    try:
        manifest = require_task_execution_lane_manifest(
            replay_store=replay_store,
            task_execution_lane_store=task_execution_lane_store,
            session_id=session_id,
            turn_id=turn_id,
        )
    except Exception:
        return False, None
    return True, manifest


__all__ = [
    "L2ReplayTaskExecutionLaneStorePort",
    "inspect_task_execution_lane_manifest",
    "require_task_execution_lane_manifest",
]
