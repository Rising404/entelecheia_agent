"""正式答复提交后，认领并结算 durable 派生任务（post-commit jobs）。

job handler 更新摘要/历史索引；runner 持有分派、计数和窗口释放检查。
“答复已完成”与“下一 Turn 可进入”分属正式提交与派生状态结算两个边界。
"""

from __future__ import annotations

from uuid import uuid4

from ...retrieval.sources.session.post_commit import (
    SESSION_RETRIEVAL_INDEX_JOB_KIND,
    SessionRetrievalIndexer,
    process_session_retrieval_index_job,
)
from ...session.session_summary import SESSION_SUMMARY_JOB_KIND
from ...session.session_summary_jobs import (
    SessionSummaryJobGenerator,
    process_session_summary_job,
)
from ...trajectory.scope import turn_linkage_scope
from .contracts import TurnPostCommitJobRunResult, TurnPostCommitJobStore


# 租约必须长于最慢 handler 的六次 60 秒提供方调用，再加上有界重试抖动。
POST_COMMIT_WORKER_LEASE_SECONDS = 420


def process_due_turn_post_commit_jobs(
    *,
    session_id: str,
    store: TurnPostCommitJobStore,
    worker_id: str | None = None,
    max_jobs: int = 4,
    summary_generator: SessionSummaryJobGenerator | None = None,
    session_retrieval_indexer: SessionRetrievalIndexer | None = None,
) -> TurnPostCommitJobRunResult:
    """认领一页有租约的 Session jobs，按 kind 分派并尝试释放执行窗口。

    summary 模型只在实际处理对应 job 时加载，并显式绑定原 Turn 的 trajectory scope；
    它可能在用户已收到正式回复后继续产生记录。handler 的成功/失败计数不等于
    Window 已释放，后者必须由所有持久 job 的结算状态共同决定。
    """

    if not session_id.strip():
        raise ValueError("session_id must not be blank")
    if not 1 <= max_jobs <= 16:
        raise ValueError("max_jobs must be within 1..16")
    owner = worker_id or f"summary-worker_{uuid4().hex}"
    if not owner.strip():
        raise ValueError("worker_id must not be blank")

    claimed_job_count = 0
    applied = 0
    failed = 0
    generate = summary_generator

    for _ in range(max_jobs):
        # 只租用马上要执行的一项；后面的工作不消耗前一项 I/O 等待期间的租约，
        # 也不会在进程中断时被误记成已经开始处理。
        claimed = store.claim_due_turn_post_commit_jobs(
            session_id=session_id,
            worker_id=owner,
            lease_seconds=POST_COMMIT_WORKER_LEASE_SECONDS,
            limit=1,
        )
        if not claimed:
            break
        job = claimed[0]
        claimed_job_count += 1
        job_kind = str(job.get("job_kind") or "")
        if job_kind == SESSION_RETRIEVAL_INDEX_JOB_KIND:
            if process_session_retrieval_index_job(
                session_id=session_id,
                job=job,
                worker_id=owner,
                store=store,
                index_pair=session_retrieval_indexer,
            ):
                applied += 1
            else:
                failed += 1
            continue
        if job_kind != SESSION_SUMMARY_JOB_KIND:
            store.mark_turn_post_commit_job_failed(
                job_id=_required_job_id(job),
                worker_id=owner,
                reason_code="POST_COMMIT_HANDLER_UNAVAILABLE",
                retry_after_seconds=None,
            )
            failed += 1
            continue
        job_session_id = _required_text(job, "session_id")
        turn_id = _required_text(job, "turn_id")
        if generate is None:
            # 默认模型栈只在实际声明 summary job 时加载；普通 Retrieval job 与模块
            # 冷导入不需要初始化 provider/configuration。
            from ..model_calls.session_summary import generate_session_summary

            generate = generate_session_summary
        with turn_linkage_scope(session_id=job_session_id, turn_id=turn_id):
            if process_session_summary_job(
                session_id=session_id,
                job_id=_required_job_id(job),
                turn_id=turn_id,
                worker_id=owner,
                store=store,
                generate=generate,
            ):
                applied += 1
            else:
                failed += 1

    return TurnPostCommitJobRunResult(
        claimed_job_count=claimed_job_count,
        applied_job_count=applied,
        failed_job_count=failed,
        released_window=release_turn_window_if_post_commit_settled(
            session_id=session_id,
            store=store,
        ),
    )


def release_turn_window_if_post_commit_settled(
    *,
    session_id: str,
    store: TurnPostCommitJobStore,
) -> bool:
    """仅在 post_commit_pending 且全部 jobs 为 applied / waived 时释放 Turn Window。

    failed / pending 或缺失 job 投影都不视为完成；用当前 state_version 释放，
    避免检查与写入之间的竞态把别的执行窗口清掉。
    """

    inspection = store.inspect_turn_execution(session_id)
    window = inspection.get("window")
    jobs = inspection.get("post_commit_jobs")
    if not isinstance(window, dict) or window.get("window_state") != "post_commit_pending":
        return False
    if not isinstance(jobs, list) or not jobs:
        return False
    if any(
        not isinstance(job, dict) or job.get("status") not in {"applied", "waived"}
        for job in jobs
    ):
        return False
    turn_id = window.get("turn_id")
    revision = window.get("state_version")
    if not isinstance(turn_id, str) or not turn_id or not isinstance(revision, int):
        return False
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=revision,
    )
    return True


def _required_job_id(job: dict[str, object]) -> str:
    return _required_text(job, "job_id")


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"post-commit job is missing {key}")
    return value
