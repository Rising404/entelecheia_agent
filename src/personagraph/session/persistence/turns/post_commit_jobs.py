"""已最终确定 Turn 的派生状态任务持久化门面。

``session.store`` 仍是稳定公开兼容接口。本模块汇集正式 Turn 提交后的持久化调用：通用
任务租约与控制、当前启用的 Session 摘要投影，以及最终 Window 释放。它刻意不拥有已
接受 Turn 状态机或附件绑定事务；二者分别仍是 :mod:`turn_execution` 和
:mod:`attachments` 内的原子记录。

此处只允许显式 :class:`StoreDeps` 和相邻持久化记录。Runtime 决定运行哪个处理器及运行
时机；本模块不导入 Runtime，也不调用模型 provider。
"""

from __future__ import annotations

from ...session_summary import (
    SessionSummaryState,
    SessionSummaryStatus,
    SessionSummaryTurnPair,
    SessionSummaryUpdate,
)
from ..history import session_summaries, turns
from . import turn_execution
from ..deps import StoreDeps


SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS = (
    session_summaries.SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS
)


def list_turn_post_commit_jobs(
    deps: StoreDeps,
    turn_id: str,
) -> list[dict[str, object]]:
    """列出一个已接受 Turn 的持久派生状态任务。"""

    return turn_execution.list_turn_post_commit_jobs(deps, turn_id)


def claim_due_turn_post_commit_jobs(
    deps: StoreDeps,
    *,
    session_id: str,
    worker_id: str,
    lease_seconds: int,
    limit: int = 16,
) -> list[dict[str, object]]:
    """为一个 Session 租用一页有界提交后任务。"""

    return turn_execution.claim_due_turn_post_commit_jobs(
        deps,
        session_id=session_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        limit=limit,
    )


def mark_turn_post_commit_job_applied(
    deps: StoreDeps,
    *,
    job_id: str,
    worker_id: str,
) -> dict[str, object]:
    """把一个工作器所有通用任务结算为已应用。"""

    return turn_execution.mark_turn_post_commit_job_applied(
        deps,
        job_id=job_id,
        worker_id=worker_id,
    )


def reconcile_turn_post_commit_job_applied(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    job_id: str,
    expected_job_kind: str,
) -> dict[str, object]:
    """Settle a replay-safe job whose exact derived effect was re-proven."""

    return turn_execution.reconcile_turn_post_commit_job_applied(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        job_id=job_id,
        expected_job_kind=expected_job_kind,
    )


def mark_turn_post_commit_job_failed(
    deps: StoreDeps,
    *,
    job_id: str,
    worker_id: str,
    reason_code: str,
    retry_after_seconds: int | None,
) -> dict[str, object]:
    """把一个工作器所有通用任务结算为可重试或终态失败。"""

    return turn_execution.mark_turn_post_commit_job_failed(
        deps,
        job_id=job_id,
        worker_id=worker_id,
        reason_code=reason_code,
        retry_after_seconds=retry_after_seconds,
    )


def get_committed_turn_pair_for_post_commit(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object] | None:
    """读取一个任务的精确正式 Turn 对和冻结 History generation。"""

    return turns.get_committed_turn_pair_by_turn_id(
        deps,
        session_id,
        turn_id,
    )


def get_session_summary_state(
    deps: StoreDeps,
    session_id: str,
) -> SessionSummaryState | None:
    """读取任务处理器使用的带类型 Session 摘要投影。"""

    return session_summaries.get_session_summary_state(deps, session_id)


def list_committed_turn_pairs_for_summary(
    deps: StoreDeps,
    session_id: str,
    *,
    after_turn_id: str | None,
    retain_recent_pairs: int = SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS,
    limit: int = 32,
) -> tuple[SessionSummaryTurnPair, ...]:
    """读取受保护热历史之外的一页有界时间顺序记录。"""

    return session_summaries.list_committed_turn_pairs_for_summary(
        deps,
        session_id,
        after_turn_id=after_turn_id,
        retain_recent_pairs=retain_recent_pairs,
        limit=limit,
    )


def commit_session_summary_post_commit_job(
    deps: StoreDeps,
    *,
    session_id: str,
    job_id: str,
    worker_id: str,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
    update: SessionSummaryUpdate,
) -> SessionSummaryState:
    """原子持久化一个已验证摘要更新，并结算其任务。"""

    return session_summaries.commit_session_summary_post_commit_job(
        deps,
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        update=update,
    )


def fail_session_summary_post_commit_job(
    deps: StoreDeps,
    *,
    session_id: str,
    job_id: str,
    worker_id: str,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
    status: SessionSummaryStatus,
    reason_code: str,
    retry_after_seconds: int | None,
) -> SessionSummaryState:
    """记录摘要降级，并结算工作器所有任务。"""

    return session_summaries.fail_session_summary_post_commit_job(
        deps,
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        status=status,
        reason_code=reason_code,
        retry_after_seconds=retry_after_seconds,
    )


def apply_turn_post_commit_job_control(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    request_id: str,
    action: str,
    job_ids: tuple[str, ...],
    expected_failed_job_digest: str,
    actor: str,
) -> dict[str, object]:
    """对终态失败任务应用已认证重试或豁免。"""

    return turn_execution.apply_turn_post_commit_job_control(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        request_id=request_id,
        action=action,  # type: ignore[arg-type]
        job_ids=job_ids,
        expected_failed_job_digest=expected_failed_job_digest,
        actor=actor,
    )


def release_turn_execution_window(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> dict[str, object]:
    """所有必需任务结算后释放已完成 Turn 的槽位。"""

    return turn_execution.release_turn_execution_window(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
    )
