"""评测的收尾等待与报告投影；实际调度和索引始终由 Runtime 唯一服务负责。"""

from __future__ import annotations

import time
from typing import Any


POST_COMMIT_TIMEOUT_SECONDS = 120.0


def pending_post_commit(turn_id: str | None) -> dict[str, Any]:
    """回答先落盘，不能将尚未观察的后台状态记成成功。"""

    return {
        "post_commit": {
            "status": "pending" if turn_id else "unavailable",
            "turn_id": turn_id,
            "window_released": False,
            "jobs": [],
            "timed_out": False,
        },
        "post_commit_complete": False,
        "post_commit_elapsed_s": 0.0,
    }


def settle_post_commit(*, session_id: str, turn_id: str | None, store: Any) -> dict[str, Any]:
    """保持答案与收尾失败相互独立，不因后台异常丢弃已提交答复。"""

    result = pending_post_commit(turn_id)
    if not turn_id:
        return result
    started = time.monotonic()
    try:
        from personagraph.runtime.post_commit.scheduler import wait_for_turn_post_commit_jobs

        settlement = wait_for_turn_post_commit_jobs(
            session_id=session_id,
            turn_id=turn_id,
            store=store,
            timeout_seconds=POST_COMMIT_TIMEOUT_SECONDS,
        ).to_dict()
        result["post_commit"] = settlement
        jobs = settlement.get("jobs")
        result["post_commit_complete"] = (
            settlement["status"] == "settled"
            and settlement["window_released"] is True
            and settlement["timed_out"] is False
            and isinstance(jobs, list)
            and bool(jobs)
            and all(isinstance(job, dict) and job.get("status") == "applied" for job in jobs)
        )
    except Exception as exc:
        # 不输出异常正文，可能含路径、Provider 请求或凭据。
        result["post_commit"].update({
            "status": "unavailable",
            "reason_code": "post_commit_wait_failed",
            "exception_type": type(exc).__name__,
        })
    result["post_commit_elapsed_s"] = round(time.monotonic() - started, 3)
    return result
