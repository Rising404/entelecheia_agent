"""Entry 路由阶段消费的 lane-neutral 持久化边界。"""

from __future__ import annotations

from typing import Protocol

from .contracts import EntryTaskMatchApplyResult


class EntryTaskAdmissionStorePort(Protocol):
    """持久应用一次已校验 task-match 提案所需的唯一写入。"""

    def apply_insession_task_matches(
        self,
        *,
        session_id: str,
        source_turn_id: str,
        apply_id: str,
        proposal: object,
        exposed_catalog_ids: tuple[str, ...],
        expected_window_revision: int,
    ) -> EntryTaskMatchApplyResult: ...


class EntryTaskRoutingAuthorityStorePort(Protocol):
    """认证所选持久任务执行 lane 所需的三项只读事实。"""

    def get_insession_task_details(
        self,
        session_id: str,
        insession_task_id: str,
    ) -> object | None: ...

    def get_insession_task_execution_lane_manifest(
        self,
        **kwargs: object,
    ) -> object: ...

    def list_pending_user_questions(
        self,
        **kwargs: object,
    ) -> tuple[object, ...]: ...


class EntryRoutingStorePort(
    EntryTaskAdmissionStorePort,
    EntryTaskRoutingAuthorityStorePort,
    Protocol,
):
    """Entry composition root 提供的完整路由持久化表面。"""


__all__ = [
    "EntryRoutingStorePort",
    "EntryTaskAdmissionStorePort",
    "EntryTaskRoutingAuthorityStorePort",
]
