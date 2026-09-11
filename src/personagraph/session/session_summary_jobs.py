"""Session summary post-commit job 的应用服务。

本模块读取持久摘要状态、构造下一份可信更新，并通过窄 Store port 原子结算 job。
Runtime 负责 job kind 路由、Turn 轨迹和实际模型生成器；本模块不导入 Runtime、Provider
或 SQLite 实现。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..context_budget import ContextBudgetExceeded
from .session_summary import (
    SESSION_SUMMARY_CANDIDATE_PAIR_LIMIT,
    SessionSummaryGenerationError,
    SessionSummaryState,
    SessionSummaryStatus,
    SessionSummaryTurnPair,
    SessionSummaryUpdate,
    build_session_summary_update,
)


SessionSummaryJobGenerator = Callable[
    [
        SessionSummaryState | None,
        tuple[SessionSummaryTurnPair, ...],
        str,
        str,
    ],
    str,
]


class SessionSummaryJobStore(Protocol):
    """摘要 job 应用服务所需的最小持久边界。"""

    def get_session_summary_state(self, session_id: str) -> SessionSummaryState | None: ...

    def list_committed_turn_pairs_for_summary(
        self,
        session_id: str,
        *,
        after_turn_id: str | None,
        limit: int = ...,
    ) -> tuple[SessionSummaryTurnPair, ...]: ...

    def commit_session_summary_post_commit_job(
        self,
        *,
        session_id: str,
        job_id: str,
        worker_id: str,
        expected_state_version: int,
        expected_summarized_through_turn_id: str | None,
        update: SessionSummaryUpdate,
    ) -> SessionSummaryState: ...

    def fail_session_summary_post_commit_job(
        self,
        *,
        session_id: str,
        job_id: str,
        worker_id: str,
        expected_state_version: int,
        expected_summarized_through_turn_id: str | None,
        status: SessionSummaryStatus,
        reason_code: str,
        retry_after_seconds: int | None,
    ) -> SessionSummaryState: ...


def process_session_summary_job(
    *,
    session_id: str,
    job_id: str,
    turn_id: str,
    worker_id: str,
    store: SessionSummaryJobStore,
    generate: SessionSummaryJobGenerator,
) -> bool:
    """处理已认领的 summary job：读旧摘要边界，生成增量更新，再原子结算。

    候选只来自已提交完整对话对；build_session_summary_update 再选择可摘要部分，
    无新增可摘要内容时可不调用 generate。提交同时检查旧 state_version / boundary
    与 worker 归属，避免重复 worker 把基于旧摘要生成的结果覆盖到新状态。
    失败记录稳定 reason 并返回 False，不把派生摘要错误改写成正式聊天内容。
    """

    state = store.get_session_summary_state(session_id)
    state_version = state.state_version if state is not None else 0
    boundary = state.summarized_through_turn_id if state is not None else None
    try:
        pairs = store.list_committed_turn_pairs_for_summary(
            session_id,
            after_turn_id=boundary,
            limit=SESSION_SUMMARY_CANDIDATE_PAIR_LIMIT,
        )
        update = build_session_summary_update(
            state=state,
            pairs=pairs,
            generate=lambda current_state, bounded_pairs: generate(
                current_state,
                bounded_pairs,
                turn_id,
                job_id,
            ),
        )
        store.commit_session_summary_post_commit_job(
            session_id=session_id,
            job_id=job_id,
            worker_id=worker_id,
            expected_state_version=state_version,
            expected_summarized_through_turn_id=boundary,
            update=update,
        )
        return True
    except ContextBudgetExceeded:
        _record_summary_failure(
            store=store,
            session_id=session_id,
            job_id=job_id,
            worker_id=worker_id,
            state=state,
            reason_code="SUMMARY_CONTEXT_BUDGET_EXCEEDED",
        )
        return False
    except SessionSummaryGenerationError as exc:
        _record_summary_failure(
            store=store,
            session_id=session_id,
            job_id=job_id,
            worker_id=worker_id,
            state=state,
            reason_code=summary_generation_failure_code(exc),
        )
        return False
    except Exception:
        # 只把稳定类别写入持久状态；调用方日志可保留完整的本地异常。
        _record_summary_failure(
            store=store,
            session_id=session_id,
            job_id=job_id,
            worker_id=worker_id,
            state=state,
            reason_code="SUMMARY_INTERNAL_FAILURE",
        )
        return False


def _record_summary_failure(
    *,
    store: SessionSummaryJobStore,
    session_id: str,
    job_id: str,
    worker_id: str,
    state: SessionSummaryState | None,
    reason_code: str,
) -> None:
    try:
        store.fail_session_summary_post_commit_job(
            session_id=session_id,
            job_id=job_id,
            worker_id=worker_id,
            expected_state_version=state.state_version if state is not None else 0,
            expected_summarized_through_turn_id=(
                state.summarized_through_turn_id if state is not None else None
            ),
            status=(
                SessionSummaryStatus.STALE
                if state is not None and state.running_summary.strip()
                else SessionSummaryStatus.UNAVAILABLE
            ),
            reason_code=reason_code,
            # 一个逻辑摘要请求已经使用共享的六次有界尝试；持久 job 不再叠加重试。
            retry_after_seconds=None,
        )
    except Exception:
        # 持久化失败时保留原 worker lease，供后续 Host 调用安全恢复。
        return


def summary_generation_failure_code(error: SessionSummaryGenerationError) -> str:
    """把纯摘要生成失败投影为稳定持久 reason。"""

    if error.code in {
        "SUMMARY_BOUNDARY_UNAVAILABLE",
        "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE",
        "SUMMARY_INPUT_OVER_BUDGET",
    }:
        return error.code
    if error.code == "SUMMARY_EMPTY_OUTPUT":
        return "SUMMARY_MODEL_OUTPUT_INVALID"
    if error.code in {
        "SUMMARY_MODEL_TIMEOUT",
        "SUMMARY_MODEL_OUTPUT_INVALID",
        "SUMMARY_MODEL_CONFIGURATION_FAILURE",
        "SUMMARY_MODEL_TRANSPORT_FAILURE",
    }:
        return error.code
    return "SUMMARY_INTERNAL_FAILURE"


__all__ = [
    "SessionSummaryJobGenerator",
    "SessionSummaryJobStore",
    "process_session_summary_job",
    "summary_generation_failure_code",
]
