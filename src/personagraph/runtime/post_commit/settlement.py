"""只读投影指定 Turn 的派生任务结算，不调度、不补写、不推断答案质量。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import ScheduledTurnPostCommitJobStore, TurnPostCommitSettlement


def read_turn_post_commit_settlement(
    *, session_id: str, turn_id: str, store: ScheduledTurnPostCommitJobStore,
) -> TurnPostCommitSettlement:
    from .contracts import TurnPostCommitSettlement

    with store.session_database_scope(session_id):
        raw_jobs = store.list_turn_post_commit_jobs(turn_id)
        inspection = store.inspect_turn_execution(session_id)
    jobs = tuple(
        {key: job.get(key) for key in ("job_kind", "status", "reason_code")}
        for job in raw_jobs if isinstance(job, Mapping)
    )
    window = inspection.get("window")
    if not jobs or len(jobs) != len(raw_jobs) or not isinstance(window, Mapping):
        return TurnPostCommitSettlement("unavailable", turn_id, False, jobs)
    statuses = {job["status"] for job in jobs}
    if not statuses <= {"pending", "processing", "retryable_failed", "terminal_failed", "applied", "waived"}:
        return TurnPostCommitSettlement("unavailable", turn_id, False, jobs)
    # 下一轮已取得窗口，也证明旧窗口已释放；不能因此把旧轮未完成的 jobs 算作完成。
    released = window.get("window_state") == "empty" or (
        isinstance(window.get("turn_id"), str) and window["turn_id"] != turn_id
    )
    if statuses <= {"applied", "waived"} and released:
        status = "settled"
    elif "terminal_failed" in statuses and not statuses & {"pending", "processing", "retryable_failed"}:
        status = "failed"
    else:
        status = "pending"
    return TurnPostCommitSettlement(status, turn_id, released, jobs)
