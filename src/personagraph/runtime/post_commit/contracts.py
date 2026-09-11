"""Contracts for durable Turn post-commit scheduling and execution."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from ...retrieval.sources.session.post_commit import SessionRetrievalPostCommitStore
from ...session.session_summary_jobs import SessionSummaryJobStore


class TurnPostCommitJobStore(
    SessionRetrievalPostCommitStore,
    SessionSummaryJobStore,
    Protocol,
):
    """提交后工作线程所需的最小持久边界。"""

    def claim_due_turn_post_commit_jobs(
        self,
        *,
        session_id: str,
        worker_id: str,
        lease_seconds: int,
        limit: int,
    ) -> list[dict[str, object]]: ...

    def release_turn_execution_window(
        self,
        *,
        session_id: str,
        turn_id: str,
        expected_window_revision: int,
    ) -> dict[str, object]: ...

    def list_turn_post_commit_jobs(self, turn_id: str) -> list[dict[str, object]]: ...


class ScheduledTurnPostCommitJobStore(TurnPostCommitJobStore, Protocol):
    """后台线程除 job 操作外所需的精确 Session 路由能力。"""

    def session_database_scope(
        self,
        session_id: str,
    ) -> AbstractContextManager[object]: ...


class PostCommitDiscoveryStore(ScheduledTurnPostCommitJobStore, Protocol):
    """仅宿主发现器需要枚举会话，单会话执行不依赖目录枚举。"""

    def list_session_ids_for_post_commit_recovery(self) -> tuple[str, ...]: ...

    def get_session(self, session_id: str) -> dict[str, object] | None: ...


@dataclass(frozen=True)
class TurnPostCommitJobRunResult:
    """一次有界工作线程调用的安全运行事实。"""

    claimed_job_count: int
    applied_job_count: int
    failed_job_count: int
    released_window: bool


@dataclass(frozen=True)
class TurnPostCommitSettlement:
    """回答之外的派生工作结算；超时或读取失败不能被投影为完成。"""

    status: Literal["settled", "failed", "pending", "unavailable"]
    turn_id: str
    window_released: bool
    jobs: tuple[dict[str, object], ...] = ()
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "turn_id": self.turn_id,
            "window_released": self.window_released,
            "jobs": [dict(job) for job in self.jobs],
            "timed_out": self.timed_out,
        }
