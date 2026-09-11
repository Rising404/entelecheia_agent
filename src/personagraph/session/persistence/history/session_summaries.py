"""有界提交后 Session 摘要投影的 SQLite 持久化。

本模块只拥有持久摘要事实，以及 ``session_summary`` 提交后任务与其摘要状态之间的原子
转换。它既不调用模型，也不决定是否可以豁免失败摘要任务；这些属于 Runtime 和控制平面。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

from pydantic import ValidationError

from ...session_summary import (
    SESSION_SUMMARY_JOB_KIND,
    SessionSummaryProgressError,
    SessionSummaryStateConflict,
    SessionSummaryStateError,
    SessionSummaryState,
    SessionSummaryStatus,
    SessionSummaryTurnPair,
    SessionSummaryUpdate,
)
from . import turns as transcript_turns
from ..deps import StoreDeps
from ..turns.turn_execution import TurnExecutionPersistenceError, TurnPostCommitJobLeaseError


SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS = 5
MAX_SESSION_SUMMARY_RETAIN_RECENT_PAIRS = 64
MAX_SESSION_SUMMARY_CANDIDATE_PAIRS = 64
MAX_SESSION_SUMMARY_IDENTIFIER_LENGTH = 200


def get_session_summary_state(
    deps: StoreDeps,
    session_id: str,
) -> SessionSummaryState | None:
    """读取当前摘要状态，不创建或修复它。"""

    _require_identifier("session_id", session_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        return _load_summary_state(conn, session_id)


def list_committed_turn_pairs_for_summary(
    deps: StoreDeps,
    session_id: str,
    *,
    after_turn_id: str | None,
    retain_recent_pairs: int = SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS,
    limit: int = 32,
) -> tuple[SessionSummaryTurnPair, ...]:
    """返回热历史之外的一页有界时间顺序摘要。

    每次摘要请求都至少排除最近五个完整 Turn 对。缺少 Runtime ``turn_id`` 的旧版 Turn 对
    对初始重建仍是有用输入，但自身不能成为持久推进边界。
    """

    _require_identifier("session_id", session_id)
    _require_optional_identifier("after_turn_id", after_turn_id)
    _require_summary_page_limits(retain_recent_pairs=retain_recent_pairs, limit=limit)

    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        after_turn_idx = _resolve_summary_boundary_turn_idx(
            conn,
            session_id=session_id,
            after_turn_id=after_turn_id,
        )
        rows = conn.execute(
            """
            WITH hot_pair_indices AS (
                SELECT c.user_turn_idx
                FROM session_turn_commits c
                JOIN session_turns u
                    ON u.session_id=c.session_id
                    AND u.turn_idx=c.user_turn_idx
                    AND u.role='user'
                JOIN session_turns a
                    ON a.session_id=c.session_id
                    AND a.turn_idx=c.assistant_turn_idx
                    AND a.role='assistant'
                WHERE c.session_id=?
                ORDER BY c.user_turn_idx DESC, c.run_id DESC
                LIMIT ?
            )
            SELECT c.run_id, c.turn_id, c.session_id, c.user_turn_idx,
                   c.assistant_turn_idx, c.created_at,
                   u.content AS user_content, a.content AS assistant_content
            FROM session_turn_commits c
            JOIN session_turns u
                ON u.session_id=c.session_id
                AND u.turn_idx=c.user_turn_idx
                AND u.role='user'
            JOIN session_turns a
                ON a.session_id=c.session_id
                AND a.turn_idx=c.assistant_turn_idx
                AND a.role='assistant'
            LEFT JOIN runtime_turns rt
                ON rt.turn_id=c.turn_id AND rt.session_id=c.session_id
            WHERE c.session_id=?
                AND c.user_turn_idx > ?
                AND NOT EXISTS (
                    SELECT 1
                    FROM hot_pair_indices hot
                    WHERE hot.user_turn_idx=c.user_turn_idx
                )
                AND (c.turn_id IS NULL OR rt.status='completed')
            ORDER BY c.user_turn_idx, c.run_id
            LIMIT ?
            """,
            (session_id, retain_recent_pairs, session_id, after_turn_idx, limit),
        ).fetchall()
        return tuple(
            _summary_pair_from_row(
                row,
                assistant_content=(
                    transcript_turns._resolve_commit_assistant_content_in_transaction(
                        conn,
                        session_id=session_id,
                        turn_id=(
                            str(row["turn_id"])
                            if row["turn_id"] is not None
                            else None
                        ),
                        commit_run_id=str(row["run_id"]),
                        inline_content=str(row["assistant_content"]),
                    )
                ),
            )
            for row in rows
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
    """原子应用一个已验证摘要更新，并结算其任务。

    工作器必须携带模型 I/O 前读取的状态版本和边界。存储层会重新检查二者、拒绝受保护的
    热历史边界，并在把工作器所有任务标记为已应用的同一事务中将摘要记录为 ``ok``。
    """

    _validate_summary_job_args(
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
    )
    if not isinstance(update, SessionSummaryUpdate):
        raise TypeError("update must be SessionSummaryUpdate")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session(conn, session_id)
        job = _require_summary_job(conn, session_id=session_id, job_id=job_id)
        current = _load_summary_state(conn, session_id)
        if str(job["status"]) == "applied":
            if current is None:
                raise SessionSummaryStateError(
                    "an applied session summary job has no durable summary state"
                )
            return current
        _require_worker_owned_processing_job(job, worker_id=worker_id)
        _require_expected_summary_state(
            current,
            expected_state_version=expected_state_version,
            expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        )

        legacy_summarized_upto = _validate_summary_update_progression(
            conn,
            session_id=session_id,
            job_turn_id=str(job["turn_id"]),
            current=current,
            update=update,
        )
        state = _write_summary_state(
            conn,
            session_id=session_id,
            running_summary=update.running_summary,
            summarized_upto=legacy_summarized_upto,
            summarized_through_turn_id=update.summarized_through_turn_id,
            status=SessionSummaryStatus.OK,
            state_version=(current.state_version if current is not None else 0) + 1,
            updated_at=now,
            last_error_code=None,
        )
        updated = conn.execute(
            "UPDATE turn_post_commit_jobs "
            "SET status='applied', next_retry_at=NULL, lease_owner=NULL, lease_until=NULL, "
            "reason_code=NULL, updated_at=?, completed_at=? "
            "WHERE job_id=? AND status='processing' AND lease_owner=?",
            (now, now, job_id, worker_id),
        ).rowcount
        if not updated:
            raise TurnPostCommitJobLeaseError("post-commit job is not owned by this worker")
        return state


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
    """原子记录真实摘要降级，并结算其任务失败。"""

    _validate_summary_job_args(
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
    )
    _require_identifier("reason_code", reason_code)
    if not isinstance(status, SessionSummaryStatus):
        raise TypeError("status must be SessionSummaryStatus")
    if status is SessionSummaryStatus.OK:
        raise ValueError("a failed summary job must set stale or unavailable status")
    if retry_after_seconds is not None and retry_after_seconds <= 0:
        raise ValueError("retry_after_seconds must be greater than zero when provided")

    job_status = "retryable_failed" if retry_after_seconds is not None else "terminal_failed"
    deps.init_db()
    now = deps.now()
    next_retry_at = _add_seconds(now, retry_after_seconds) if retry_after_seconds else None
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session(conn, session_id)
        job = _require_summary_job(conn, session_id=session_id, job_id=job_id)
        current = _load_summary_state(conn, session_id)
        _require_worker_owned_processing_job(job, worker_id=worker_id)
        _require_expected_summary_state(
            current,
            expected_state_version=expected_state_version,
            expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        )

        state = _write_summary_state(
            conn,
            session_id=session_id,
            running_summary=current.running_summary if current is not None else "",
            summarized_upto=_summary_progress_mirror(conn, session_id),
            summarized_through_turn_id=(
                current.summarized_through_turn_id if current is not None else None
            ),
            status=status,
            state_version=(current.state_version if current is not None else 0) + 1,
            updated_at=now,
            last_error_code=reason_code,
        )
        updated = conn.execute(
            "UPDATE turn_post_commit_jobs "
            "SET status=?, next_retry_at=?, lease_owner=NULL, lease_until=NULL, "
            "reason_code=?, updated_at=?, completed_at=? "
            "WHERE job_id=? AND status='processing' AND lease_owner=?",
            (
                job_status,
                next_retry_at,
                reason_code,
                now,
                now if job_status == "terminal_failed" else None,
                job_id,
                worker_id,
            ),
        ).rowcount
        if not updated:
            raise TurnPostCommitJobLeaseError("post-commit job is not owned by this worker")
        return state


def _validate_summary_job_args(
    *,
    session_id: str,
    job_id: str,
    worker_id: str,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
) -> None:
    _require_identifier("session_id", session_id)
    _require_identifier("job_id", job_id)
    _require_identifier("worker_id", worker_id)
    _require_optional_identifier(
        "expected_summarized_through_turn_id", expected_summarized_through_turn_id
    )
    if expected_state_version < 0:
        raise ValueError("expected_state_version must not be negative")


def _require_summary_page_limits(*, retain_recent_pairs: int, limit: int) -> None:
    if not (
        SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS
        <= retain_recent_pairs
        <= MAX_SESSION_SUMMARY_RETAIN_RECENT_PAIRS
    ):
        raise ValueError(
            "retain_recent_pairs must keep at least the protected recent summary history"
        )
    if not 1 <= limit <= MAX_SESSION_SUMMARY_CANDIDATE_PAIRS:
        raise ValueError(
            f"limit must be between 1 and {MAX_SESSION_SUMMARY_CANDIDATE_PAIRS}"
        )


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
        raise ValueError(f"unknown session: {session_id}")


def _resolve_summary_boundary_turn_idx(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    after_turn_id: str | None,
) -> int:
    if after_turn_id is None:
        return -1
    row = _load_complete_runtime_pair(conn, session_id=session_id, turn_id=after_turn_id)
    if row is None:
        raise SessionSummaryProgressError(
            "summary boundary does not identify a completed Runtime transcript pair"
        )
    return int(row["user_turn_idx"])


def _require_summary_job(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    job_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT job_id, session_id, turn_id, job_kind, status, lease_owner "
        "FROM turn_post_commit_jobs WHERE job_id=?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown post-commit job: {job_id}")
    if str(row["session_id"]) != session_id:
        raise TurnExecutionPersistenceError("post-commit job belongs to another Session")
    if str(row["job_kind"]) != SESSION_SUMMARY_JOB_KIND:
        raise TurnExecutionPersistenceError("post-commit job is not a session summary job")
    return row


def _require_worker_owned_processing_job(job: sqlite3.Row, *, worker_id: str) -> None:
    if str(job["status"]) != "processing" or str(job["lease_owner"] or "") != worker_id:
        raise TurnPostCommitJobLeaseError("post-commit job is not owned by this worker")


def _require_expected_summary_state(
    current: SessionSummaryState | None,
    *,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
) -> None:
    actual_state_version = current.state_version if current is not None else 0
    actual_boundary = current.summarized_through_turn_id if current is not None else None
    if (
        actual_state_version != expected_state_version
        or actual_boundary != expected_summarized_through_turn_id
    ):
        raise SessionSummaryStateConflict(
            expected_state_version=expected_state_version,
            actual_state_version=actual_state_version,
            expected_summarized_through_turn_id=expected_summarized_through_turn_id,
            actual_summarized_through_turn_id=actual_boundary,
        )


def _validate_summary_update_progression(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    job_turn_id: str,
    current: SessionSummaryState | None,
    update: SessionSummaryUpdate,
) -> int:
    """校验新边界后返回兼容整数水位。"""

    if update.summarized_through_turn_id is None:
        if current is not None and current.summarized_through_turn_id is not None:
            raise SessionSummaryProgressError("summary boundary cannot regress to none")
        if current is not None and current.running_summary.strip():
            raise SessionSummaryProgressError(
                "an unbound legacy summary requires a verified replacement boundary"
            )
        return _summary_progress_mirror(conn, session_id)

    target = _load_complete_runtime_pair(
        conn,
        session_id=session_id,
        turn_id=update.summarized_through_turn_id,
    )
    if target is None:
        raise SessionSummaryProgressError(
            "summary boundary does not identify a completed Runtime transcript pair"
        )
    job_pair = _load_complete_runtime_pair(conn, session_id=session_id, turn_id=job_turn_id)
    if job_pair is None:
        raise TurnExecutionPersistenceError(
            "session summary job does not belong to a completed Runtime transcript pair"
        )
    if int(target["user_turn_idx"]) >= int(job_pair["user_turn_idx"]):
        raise SessionSummaryProgressError(
            "a summary job cannot cover its own or a later completed Turn"
        )

    if current is not None and current.summarized_through_turn_id is not None:
        current_boundary = _load_complete_runtime_pair(
            conn,
            session_id=session_id,
            turn_id=current.summarized_through_turn_id,
        )
        if current_boundary is None:
            raise SessionSummaryStateError(
                "current summary boundary no longer identifies a completed Runtime pair"
            )
        if int(target["user_turn_idx"]) < int(current_boundary["user_turn_idx"]):
            raise SessionSummaryProgressError("summary boundary cannot move backward")

    newer_pairs = conn.execute(
        """
        SELECT c.run_id, c.turn_id, a.content AS assistant_content
        FROM session_turn_commits c
        JOIN session_turns u
            ON u.session_id=c.session_id
            AND u.turn_idx=c.user_turn_idx
            AND u.role='user'
        JOIN session_turns a
            ON a.session_id=c.session_id
            AND a.turn_idx=c.assistant_turn_idx
            AND a.role='assistant'
        WHERE c.session_id=? AND c.user_turn_idx > ?
        """,
        (session_id, int(target["user_turn_idx"])),
    ).fetchall()
    for pair in newer_pairs:
        transcript_turns._resolve_commit_assistant_content_in_transaction(
            conn,
            session_id=session_id,
            turn_id=(
                str(pair["turn_id"]) if pair["turn_id"] is not None else None
            ),
            commit_run_id=str(pair["run_id"]),
            inline_content=str(pair["assistant_content"]),
        )
    if len(newer_pairs) < SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS:
        raise SessionSummaryProgressError(
            "summary boundary would include protected recent transcript pairs"
        )
    return int(target["assistant_turn_idx"]) + 1


def _load_complete_runtime_pair(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT c.run_id, c.turn_id, c.session_id, c.user_turn_idx,
               c.assistant_turn_idx, a.content AS assistant_content
        FROM session_turn_commits c
        JOIN runtime_turns rt
            ON rt.turn_id=c.turn_id
            AND rt.session_id=c.session_id
            AND rt.status='completed'
        JOIN session_turns u
            ON u.session_id=c.session_id
            AND u.turn_idx=c.user_turn_idx
            AND u.role='user'
        JOIN session_turns a
            ON a.session_id=c.session_id
            AND a.turn_idx=c.assistant_turn_idx
            AND a.role='assistant'
        WHERE c.session_id=? AND c.turn_id=?
        """,
        (session_id, turn_id),
    ).fetchone()
    if row is None:
        return None
    transcript_turns._resolve_commit_assistant_content_in_transaction(
        conn,
        session_id=session_id,
        turn_id=str(row["turn_id"]),
        commit_run_id=str(row["run_id"]),
        inline_content=str(row["assistant_content"]),
    )
    return row


def _load_summary_state(
    conn: sqlite3.Connection,
    session_id: str,
) -> SessionSummaryState | None:
    row = conn.execute(
        "SELECT session_id, running_summary, summarized_through_turn_id, status, "
        "state_version, updated_at, last_error_code "
        "FROM session_working_memory WHERE session_id=?",
        (session_id,),
    ).fetchone()
    return _summary_state_from_row(row) if row is not None else None


def _summary_state_from_row(row: sqlite3.Row) -> SessionSummaryState:
    try:
        return SessionSummaryState(
            session_id=str(row["session_id"]),
            running_summary=str(row["running_summary"] or ""),
            summarized_through_turn_id=(
                str(row["summarized_through_turn_id"])
                if row["summarized_through_turn_id"] is not None
                else None
            ),
            status=str(row["status"]),
            state_version=int(row["state_version"]),
            updated_at=str(row["updated_at"] or ""),
            last_error_code=(
                str(row["last_error_code"]) if row["last_error_code"] is not None else None
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise SessionSummaryStateError("stored session summary state is invalid") from exc


def _summary_pair_from_row(
    row: sqlite3.Row,
    *,
    assistant_content: str | None = None,
) -> SessionSummaryTurnPair:
    try:
        return SessionSummaryTurnPair(
            turn_id=str(row["turn_id"]) if row["turn_id"] is not None else None,
            user_turn_idx=int(row["user_turn_idx"]),
            assistant_turn_idx=int(row["assistant_turn_idx"]),
            user_content=str(row["user_content"]),
            assistant_content=(
                assistant_content
                if assistant_content is not None
                else str(row["assistant_content"])
            ),
            created_at=str(row["created_at"]),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise SessionSummaryStateError("stored summary candidate pair is invalid") from exc


def _summary_progress_mirror(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT summarized_upto FROM session_working_memory WHERE session_id=?",
        (session_id,),
    ).fetchone()
    return int(row["summarized_upto"] or 0) if row is not None else 0


def _write_summary_state(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    running_summary: str,
    summarized_upto: int,
    summarized_through_turn_id: str | None,
    status: SessionSummaryStatus,
    state_version: int,
    updated_at: str,
    last_error_code: str | None,
) -> SessionSummaryState:
    conn.execute(
        """
        INSERT INTO session_working_memory (
            session_id, running_summary, summarized_upto, updated_at,
            summarized_through_turn_id, status, state_version, last_error_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            running_summary=excluded.running_summary,
            summarized_upto=excluded.summarized_upto,
            updated_at=excluded.updated_at,
            summarized_through_turn_id=excluded.summarized_through_turn_id,
            status=excluded.status,
            state_version=excluded.state_version,
            last_error_code=excluded.last_error_code
        """,
        (
            session_id,
            running_summary,
            summarized_upto,
            updated_at,
            summarized_through_turn_id,
            status.value,
            state_version,
            last_error_code,
        ),
    )
    state = _load_summary_state(conn, session_id)
    if state is None:
        raise SessionSummaryStateError("session summary state was not persisted")
    return state


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_SESSION_SUMMARY_IDENTIFIER_LENGTH
    ):
        raise ValueError(
            f"{name} must be a non-empty string of at most "
            f"{MAX_SESSION_SUMMARY_IDENTIFIER_LENGTH} characters"
        )


def _require_optional_identifier(name: str, value: str | None) -> None:
    if value is not None:
        _require_identifier(name, value)


def _add_seconds(value: str, seconds: int) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed + timedelta(seconds=seconds)).isoformat()
