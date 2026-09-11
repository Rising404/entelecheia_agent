"""一个已接受 Runtime Turn 及其执行窗口的持久化存储。

本模块有意位于 Runtime、API 和前端层之下，只负责 SQLite 事务边界与持久化事实：

* 接受不可变用户消息、其附件以及每个 Session 的执行槽位；
* 最终化正式 assistant 交付，而不重复该用户消息；
* 在提交后任务待处理或失败时继续占用槽位。

Task graph、WorkRun、模型调用、流式交付与面向用户的控制语义均位于此持久化边界之外。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from typing import Literal

from ...insession_task_contracts import InSessionTaskPersistenceError
from ....entry_execution_snapshot import authenticate_entry_execution_snapshot_payload
from ...turn_execution_contracts import (
    TurnExecutionBusyError,
    TurnExecutionLeaseConflict,
    TurnExecutionPersistenceError,
    TurnExecutionWindowRevisionConflict,
)
from .attachments import bind_attachments_to_turn_in_transaction
from ..deps import StoreDeps
from .pending_questions import require_pending_question_proof
from ..history import turns as transcript_turns
from ..l1 import turn_runs as l1_turn_runs
from ..l1.delivery import require_terminal_notification
from . import turn_routing_policies


FormalTurnReference = transcript_turns.FormalTurnReference


WindowState = Literal["empty", "active", "post_commit_pending", "interrupted"]
PostCommitJobStatus = Literal[
    "pending",
    "processing",
    "applied",
    "retryable_failed",
    "terminal_failed",
    "waived",
]
PostCommitControlAction = Literal["retry", "waive"]

_WINDOW_COLUMNS = (
    "session_id, turn_id, window_state, stage, input_message_id, "
    "attachment_binding_revision, turn_task_link_revision, turn_workrun_link_revision, "
    "current_work_run_id, current_attempt_id, latest_checkpoint_id, pending_operation_id, "
    "last_event_sequence, state_version, lease_owner, heartbeat_at, interruption_reason, "
    "claimed_at, updated_at, current_l1_turn_run_id, current_l1_attempt_id"
)
_TURN_CORE_COLUMNS = (
    "turn_id, session_id, client_request_id, source, user_text, input_message_id, status, "
    "processing_level, error_code, end_reason, received_at, completed_at"
)
_JOB_COLUMNS = (
    "job_id, session_id, turn_id, job_kind, status, attempts, next_retry_at, "
    "lease_owner, lease_until, reason_code, created_at, updated_at, completed_at"
)
MAX_TURN_EXECUTION_IDENTIFIER_LENGTH = 200


class TurnExecutionRequestIdCollision(TurnExecutionPersistenceError):
    """某个客户端 request id 被复用于不同的可信输入。"""


class TurnExecutionFinalizationConflict(TurnExecutionPersistenceError):
    """某个已完成 Turn 使用不同的正式交付事实进行了重放。"""


class TurnPostCommitJobsPending(TurnExecutionPersistenceError):
    """仍有必需的派生状态工作时，不能释放 Window。"""

    def __init__(self, jobs: list[dict[str, object]]) -> None:
        self.jobs = jobs
        self.details = {
            "pending_job_ids": [item["job_id"] for item in jobs],
            "statuses": {str(item["job_id"]): item["status"] for item in jobs},
        }
        super().__init__("turn post-commit jobs are not settled")


class TurnPostCommitJobLeaseError(TurnExecutionPersistenceError):
    """某 worker 试图结束一项已不归其所有的任务。"""


def accept_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    client_request_id: str,
    source: str,
    user_text: str,
    attachment_ids: tuple[str, ...] = (),
    turn_id: str | None = None,
    input_message_id: str | None = None,
    lease_owner: str | None = None,
    initial_stage: str = "INGRESS",
    routing_policy_source: str | None = None,
    routing_policy_snapshot_json: str | None = None,
    routing_policy_snapshot_hash: str | None = None,
    session_routing_policy_json: str | None = None,
    session_routing_policy_hash: str | None = None,
    execution_snapshot_json: str | None = None,
    execution_snapshot_sha256: str | None = None,
) -> dict[str, object]:
    """以原子方式接受一个新 Turn，或返回其幂等重放。

    此事务会一并写入可信 Runtime 记录、可见的不可变用户消息、精确附件绑定以及
    一个已认领 Window。后续正式响应有意*不*包含在此事务中。

    BEGIN IMMEDIATE 内先检查同 request-id 的精确内容重放，再检查 Window 是否空闲。
    新请求同时写 runtime_turns、用户 transcript、input linkage、附件/文件版本引用、
    routing/execution snapshot 与 Window；任何一步失败都不留下半接受状态。
    返回回执后上层才发送 accepted，HTTP/SSE 通知不是这个事务的一部分。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("client_request_id", client_request_id)
    _require_identifier("source", source)
    _require_identifier("initial_stage", initial_stage)
    if lease_owner is not None:
        _require_identifier("lease_owner", lease_owner)
    _validate_attachment_ids(attachment_ids)
    policy_snapshot = turn_routing_policies.validate_snapshot_payload(
        source=routing_policy_source,
        snapshot_json=routing_policy_snapshot_json,
        snapshot_hash=routing_policy_snapshot_hash,
    )
    session_policy = turn_routing_policies.validate_session_policy_payload(
        policy_json=session_routing_policy_json,
        policy_hash=session_routing_policy_hash,
    )
    if session_policy is not None and policy_snapshot is None:
        raise ValueError("Session routing policy requires a Turn policy snapshot")
    execution_snapshot = authenticate_entry_execution_snapshot_payload(
        snapshot_json=execution_snapshot_json,
        snapshot_sha256=execution_snapshot_sha256,
    )

    allocated_turn_id = turn_id or f"turn_{deps.new_id()}"
    allocated_input_id = input_message_id or f"input_{deps.new_id()}"
    _require_identifier("turn_id", allocated_turn_id)
    _require_identifier("input_message_id", allocated_input_id)

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session(conn, session_id)

        existing = conn.execute(
            f"SELECT {_turn_select_columns()} FROM runtime_turns "
            "WHERE session_id=? AND client_request_id=?",
            (session_id, client_request_id),
        ).fetchone()
        if existing is not None:
            turn = _turn_from_row(existing)
            _validate_request_replay(
                conn,
                turn=turn,
                source=source,
                user_text=user_text,
                attachment_ids=attachment_ids,
            )
            persisted_policy = (
                turn_routing_policies.load_turn_snapshot_in_transaction(
                    conn,
                    turn_id=str(turn["turn_id"]),
                )
            )
            if (
                policy_snapshot is not None
                and policy_snapshot[0] == "request_override"
                and (
                    persisted_policy is None
                    or persisted_policy["snapshot_hash"] != policy_snapshot[2]
                )
            ):
                raise TurnExecutionRequestIdCollision(
                    "client request id was reused with a different routing policy"
                )
            input_message = _require_turn_input(conn, str(turn["turn_id"]))
            window = _load_window(conn, session_id)
            return {
                "replayed": True,
                "turn": turn,
                "input_message": input_message,
                "attachments": _list_turn_attachment_bindings(conn, str(turn["turn_id"])),
                "routing_policy": persisted_policy,
                "window": window,
            }

        turn_id_collision = conn.execute(
            "SELECT session_id, client_request_id FROM runtime_turns WHERE turn_id=?",
            (allocated_turn_id,),
        ).fetchone()
        if turn_id_collision is not None:
            raise TurnExecutionRequestIdCollision("runtime turn id is already in use")
        input_id_collision = conn.execute(
            "SELECT 1 FROM runtime_turn_inputs WHERE message_id=?",
            (allocated_input_id,),
        ).fetchone()
        if input_id_collision is not None:
            raise TurnExecutionRequestIdCollision("runtime input message id is already in use")

        window = _load_window(conn, session_id)
        if window is not None and window["turn_id"] is not None:
            raise TurnExecutionBusyError(window)
        if window is not None and window["window_state"] != "empty":
            raise TurnExecutionPersistenceError("empty Turn window has non-empty state")

        # Accept transaction：新 Turn 的用户正文、引用与 Window 必须共同提交。
        next_turn_idx = _next_turn_idx(conn, session_id)
        conn.execute(
            "INSERT INTO runtime_turns "
            "(turn_id, session_id, source, user_text, client_request_id, "
            "input_message_id, status, received_at, execution_snapshot_json, "
            "execution_snapshot_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)",
            (
                allocated_turn_id,
                session_id,
                source,
                user_text,
                client_request_id,
                allocated_input_id,
                now,
                None if execution_snapshot is None else execution_snapshot[0],
                None if execution_snapshot is None else execution_snapshot[1],
            ),
        )
        if policy_snapshot is not None:
            turn_routing_policies.insert_turn_snapshot_in_transaction(
                conn,
                session_id=session_id,
                turn_id=allocated_turn_id,
                source=policy_snapshot[0],
                snapshot_json=policy_snapshot[1],
                snapshot_hash=policy_snapshot[2],
                created_at=now,
            )
        if session_policy is not None:
            turn_routing_policies.upsert_session_policy_in_transaction(
                conn,
                session_id=session_id,
                policy_json=session_policy[0],
                policy_hash=session_policy[1],
                updated_at=now,
            )
        conn.execute(
            "INSERT INTO session_turns (session_id, turn_idx, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (session_id, next_turn_idx, user_text, now),
        )
        conn.execute(
            "INSERT INTO runtime_turn_inputs (message_id, session_id, turn_id, turn_idx, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (allocated_input_id, session_id, allocated_turn_id, next_turn_idx, now),
        )
        bound_attachments = bind_attachments_to_turn_in_transaction(
            conn,
            session_id=session_id,
            turn_id=allocated_turn_id,
            attachment_ids=attachment_ids,
            bound_at=now,
        )
        bound_by_id = {
            str(item["attachment_id"]): item for item in bound_attachments
        }
        for ordinal, attachment_id in enumerate(attachment_ids):
            conn.execute(
                "INSERT INTO runtime_turn_attachment_bindings "
                "(turn_id, attachment_id, binding_ordinal, created_at) VALUES (?, ?, ?, ?)",
                (allocated_turn_id, attachment_id, ordinal, now),
            )
            attachment = bound_by_id[attachment_id]
            project_id = attachment.get("project_id")
            file_id = attachment.get("file_id")
            file_version_id = attachment.get("file_version_id")
            if project_id is not None:
                conn.execute(
                    "INSERT INTO runtime_turn_input_file_refs "
                    "(message_id, ordinal, project_id, file_id, file_version_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        allocated_input_id,
                        ordinal,
                        project_id,
                        file_id,
                        file_version_id,
                        now,
                    ),
                )

        if window is None:
            conn.execute(
                "INSERT INTO turn_execution_windows "
                "(session_id, turn_id, window_state, stage, input_message_id, "
                "attachment_binding_revision, state_version, lease_owner, heartbeat_at, "
                "interruption_reason, claimed_at, updated_at) "
                "VALUES (?, ?, 'active', ?, ?, ?, 1, ?, ?, NULL, ?, ?)",
                (
                    session_id,
                    allocated_turn_id,
                    initial_stage,
                    allocated_input_id,
                    int(bool(attachment_ids)),
                    lease_owner,
                    now,
                    now,
                    now,
                ),
            )
        else:
            next_revision = int(window["state_version"]) + 1
            updated = conn.execute(
                "UPDATE turn_execution_windows SET "
                "turn_id=?, window_state='active', stage=?, input_message_id=?, "
                "attachment_binding_revision=?, turn_task_link_revision=0, turn_workrun_link_revision=0, "
                "current_work_run_id=NULL, current_attempt_id=NULL, "
                f"{_l1_window_reset_clause(conn)}"
                "latest_checkpoint_id=NULL, "
                "pending_operation_id=NULL, last_event_sequence=NULL, state_version=?, lease_owner=?, "
                "heartbeat_at=?, interruption_reason=NULL, claimed_at=?, updated_at=? "
                "WHERE session_id=? AND turn_id IS NULL AND state_version=?",
                (
                    allocated_turn_id,
                    initial_stage,
                    allocated_input_id,
                    int(bool(attachment_ids)),
                    next_revision,
                    lease_owner,
                    now,
                    now,
                    now,
                    session_id,
                    int(window["state_version"]),
                ),
            ).rowcount
            if not updated:
                raise TurnExecutionPersistenceError("failed to claim an empty Turn window")

        session = conn.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise TurnExecutionPersistenceError("accepted Turn lost its Session")
        if not session["title"]:
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (user_text[:20], session_id))
        conn.execute("UPDATE sessions SET last_active_at=? WHERE id=?", (now, session_id))

        turn = _require_turn(conn, allocated_turn_id)
        input_message = _require_turn_input(conn, allocated_turn_id)
        claimed_window = _require_window(conn, session_id, allocated_turn_id)
        return {
            "replayed": False,
            "turn": turn,
            "input_message": input_message,
            "attachments": _list_turn_attachment_bindings(conn, allocated_turn_id),
            "routing_policy": turn_routing_policies.load_turn_snapshot_in_transaction(
                conn,
                turn_id=allocated_turn_id,
            ),
            "window": claimed_window,
        }


def get_turn_execution_window(
    deps: StoreDeps,
    session_id: str,
) -> dict[str, object] | None:
    """读取 Session 的当前执行槽位，而不推断恢复状态。"""

    _require_identifier("session_id", session_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        return _load_window(conn, session_id)


def get_turn_execution_input(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object]:
    """根据其所属 Runtime Turn 读取一份不可变的已接受用户输入。

    该输入在 Window 释放后仍可用，因此恢复与 UI 代码无需从已完成的对话记录对或
    易失 Runtime 投影中推断原始请求。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        turn = _require_turn(conn, turn_id)
        if turn["session_id"] != session_id:
            raise TurnExecutionPersistenceError("Turn belongs to another Session")
        return _require_turn_input(conn, turn_id)


def get_turn_execution_for_client_request(
    deps: StoreDeps,
    *,
    session_id: str,
    client_request_id: str,
) -> dict[str, object] | None:
    """通过公开幂等键读取一个已接受 Turn。

    这里有意只执行查找，并非第二套重放实现。调用方仍使用
    :func:`accept_turn_execution` 验证重试是否携带同一可信输入。此查找只让活动的
    本地租约能在创建新 Window 前区分真正重试与不同 Prompt。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("client_request_id", client_request_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        row = conn.execute(
            f"SELECT {_turn_select_columns()} FROM runtime_turns "
            "WHERE session_id=? AND client_request_id=?",
            (session_id, client_request_id),
        ).fetchone()
        if row is None:
            return None
        turn = _turn_from_row(row)
        turn_id = str(turn["turn_id"])
        return {
            "turn": turn,
            "input_message": _require_turn_input(conn, turn_id),
            "attachments": _list_turn_attachment_bindings(conn, turn_id),
            "routing_policy": turn_routing_policies.load_turn_snapshot_in_transaction(
                conn,
                turn_id=turn_id,
            ),
            "window": _load_window(conn, session_id),
        }


def inspect_turn_execution(
    deps: StoreDeps,
    session_id: str,
) -> dict[str, object]:
    """返回存储层持有的当前 Window 恢复投影。"""

    _require_identifier("session_id", session_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        window = _load_window(conn, session_id)
        if window is None or window["turn_id"] is None:
            return {
                "window": window,
                "turn": None,
                "input_message": None,
                "attachments": [],
                "post_commit_jobs": [],
                "failed_job_digest": None,
            }
        turn_id = str(window["turn_id"])
        jobs = _list_post_commit_jobs(conn, turn_id)
        return {
            "window": window,
            "turn": _require_turn(conn, turn_id),
            "input_message": _require_turn_input(conn, turn_id),
            "attachments": _list_turn_attachment_bindings(conn, turn_id),
            "routing_policy": turn_routing_policies.load_turn_snapshot_in_transaction(
                conn,
                turn_id=turn_id,
            ),
            "post_commit_jobs": jobs,
            "failed_job_digest": _terminal_failure_digest(jobs),
        }


def advance_turn_execution_window(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    stage: str,
    lease_owner: str | None = None,
    last_event_sequence: int | None = None,
) -> dict[str, object]:
    """在事务内推进 active Window 的阶段、revision、heartbeat 和可选事件游标。

    先核对 Session/Turn、expected revision 及已存在的 lease owner，再用相同 revision
    执行条件 UPDATE（CAS）。此函数不创建 Attempt 或提交答案；旧执行者或状态竞态
    必须失败，不能通过无条件更新覆盖当前窗口。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("stage", stage)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if lease_owner is not None:
        _require_identifier("lease_owner", lease_owner)
    if last_event_sequence is not None and last_event_sequence < 1:
        raise ValueError("last_event_sequence must be positive when provided")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] != "active":
            raise TurnExecutionPersistenceError("only an active Turn window can advance")
        previous_owner = window["lease_owner"]
        if (
            lease_owner is not None
            and previous_owner is not None
            and previous_owner != lease_owner
        ):
            raise TurnExecutionPersistenceError("Turn window is leased by another owner")
        next_revision = int(window["state_version"]) + 1
        updated = conn.execute(
            "UPDATE turn_execution_windows SET stage=?, state_version=?, "
            "lease_owner=COALESCE(?, lease_owner), heartbeat_at=?, "
            "last_event_sequence=COALESCE(?, last_event_sequence), updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (
                stage,
                next_revision,
                lease_owner,
                now,
                last_event_sequence,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount
        if updated != 1:
            raise TurnExecutionPersistenceError("Turn window changed during advance")
        return _require_window(conn, session_id, turn_id)


def mark_turn_execution_interrupted(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    stage: str,
    interruption_reason: str | None,
    expected_heartbeat_at: str | None = None,
    require_heartbeat_match: bool = False,
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """记录一个已知安全停止，但不提前判定 Turn 分类。

    A3 的下一输入审计负责将残留的 ``running`` Turn 转为 ``incomplete``。此标记
    只是跨越 host 关闭而保留，使审计无需猜测 host 在停止边界已知的信息。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("stage", stage)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if interruption_reason is not None:
        _require_identifier("interruption_reason", interruption_reason)
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)
    if not isinstance(require_heartbeat_match, bool):
        raise TypeError("require_heartbeat_match must be bool")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] != "active":
            raise TurnExecutionPersistenceError("only an active Turn window can be interrupted")
        _require_expected_lease_owner(window, expected_lease_owner)
        actual_heartbeat_at = _optional_str(window["heartbeat_at"])
        if require_heartbeat_match and actual_heartbeat_at != expected_heartbeat_at:
            raise TurnExecutionLeaseConflict(
                expected_heartbeat_at=expected_heartbeat_at,
                actual_heartbeat_at=actual_heartbeat_at,
            )
        next_revision = int(window["state_version"]) + 1
        updated = conn.execute(
            "UPDATE turn_execution_windows SET window_state='interrupted', stage=?, "
            "state_version=?, lease_owner=NULL, heartbeat_at=?, interruption_reason=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=? "
            "AND (? IS NULL OR lease_owner=?)",
            (
                stage,
                next_revision,
                now,
                interruption_reason,
                now,
                session_id,
                turn_id,
                expected_window_revision,
                expected_lease_owner,
                expected_lease_owner,
            ),
        ).rowcount
        if updated != 1:
            raise TurnExecutionPersistenceError(
                "Turn window changed during interruption"
            )
        return _require_window(conn, session_id, turn_id)


def settle_interrupted_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    end_reason: str,
    error_code: str | None,
) -> dict[str, object]:
    """在下一输入审计期间关闭一个已中断 Turn。

    原始中断只留下持久化标记。此操作是后续唯一的审计变更：记录 ``incomplete``，
    并释放旧 Window 以供新接受的 Turn 使用。它有意不创建 assistant 消息、Task
    事实、WorkRun 事实或合成检查点。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("end_reason", end_reason)
    if error_code is not None:
        _require_identifier("error_code", error_code)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        turn = _require_turn(conn, turn_id)
        if turn["session_id"] != session_id:
            raise TurnExecutionPersistenceError("Turn belongs to another Session")
        if turn["status"] == "incomplete":
            return {
                "replayed": True,
                "turn": turn,
                "input_message": _require_turn_input(conn, turn_id),
                "window": _load_window(conn, session_id),
            }
        if turn["status"] != "running":
            raise TurnExecutionPersistenceError(
                f"only a running Turn can settle as incomplete, got {turn['status']}"
            )

        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] != "interrupted":
            raise TurnExecutionPersistenceError("only an interrupted Window can settle")
        next_revision = int(window["state_version"]) + 1
        conn.execute(
            "UPDATE runtime_turns SET status='incomplete', error_code=?, end_reason=?, completed_at=? "
            "WHERE turn_id=? AND status='running'",
            (error_code, end_reason, now, turn_id),
        )
        conn.execute(
            "UPDATE turn_execution_windows SET turn_id=NULL, window_state='empty', stage=NULL, "
            "input_message_id=NULL, attachment_binding_revision=0, turn_task_link_revision=0, "
            "turn_workrun_link_revision=0, current_work_run_id=NULL, current_attempt_id=NULL, "
            f"{_l1_window_reset_clause(conn)}"
            "latest_checkpoint_id=NULL, pending_operation_id=NULL, last_event_sequence=NULL, "
            "state_version=?, lease_owner=NULL, heartbeat_at=NULL, interruption_reason=NULL, "
            "claimed_at=NULL, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (next_revision, now, session_id, turn_id, expected_window_revision),
        )
        return {
            "replayed": False,
            "turn": _require_turn(conn, turn_id),
            "input_message": _require_turn_input(conn, turn_id),
            "window": _load_window(conn, session_id),
        }


def finalize_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    processing_level: Literal["L0", "L1", "L2"],
    assistant_content: str,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
    l1_terminal_failure_code: str | None = None,
) -> dict[str, object]:
    """以原子方式提交正式交付、Turn 结果和持久化任务。

    即使客户端已能显示正式回答，Window 仍会以 ``post_commit_pending`` 状态保持
    占用。只有所有任务完成结算（或明确记录过期豁免）后才能释放。

    L1 分支先在同一事务验证并完成 L1 run，然后补齐正式 assistant 半边与 pair commit，
    插入 jobs，标记 runtime Turn completed，最后把 Window 改成 post_commit_pending。
    L1 运行通知只接受已失败 run 对应的固定正文，保留其失败状态和公开错误分类；
    不伪造完成报告，也不调用另一份消息发布器。
    已 completed 的重复调用走 _replay_finalization 校验相同正文/级别/job 集，
    不把重复提交解释成再追加一条回答。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if processing_level not in {"L0", "L1", "L2"}:
        raise ValueError("processing_level must be L0, L1, or L2")
    if l1_terminal_failure_code is not None:
        _require_identifier("l1_terminal_failure_code", l1_terminal_failure_code)
        if processing_level != "L1":
            raise ValueError("terminal L1 notification requires processing_level=L1")
    _validate_job_kinds(post_commit_job_kinds)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        turn = _require_turn(conn, turn_id)
        if turn["session_id"] != session_id:
            raise TurnExecutionFinalizationConflict("Turn belongs to another Session")
        owned_window = None
        if expected_lease_owner is not None:
            owned_window = _require_window(conn, session_id, turn_id)
            _require_expected_lease_owner(owned_window, expected_lease_owner)
        if turn["status"] == "completed":
            terminal_replay = turn.get("end_reason") == "l1_terminal_notification"
            if terminal_replay != (l1_terminal_failure_code is not None):
                raise TurnExecutionFinalizationConflict("completed Turn has another delivery kind")
            if l1_terminal_failure_code is not None:
                notification = require_terminal_notification(
                    conn, session_id=session_id, turn_id=turn_id,
                    failure_code=l1_terminal_failure_code, assistant_content=assistant_content,
                )
                if turn.get("error_code") != notification.error_code:
                    raise TurnExecutionFinalizationConflict("completed Turn has another terminal error")
            return _replay_finalization(
                conn,
                turn=turn,
                processing_level=processing_level,
                assistant_content=assistant_content,
                post_commit_job_kinds=post_commit_job_kinds,
                session_id=session_id,
            )
        if turn["status"] != "running":
            raise TurnExecutionFinalizationConflict(
                f"only a running Turn can finalize, got {turn['status']}"
            )

        window = owned_window or _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        _require_expected_lease_owner(window, expected_lease_owner)
        if window["window_state"] != "active":
            raise TurnExecutionFinalizationConflict("Turn window is not active")
        if processing_level == "L2" and _turn_owns_auxiliary_root_delivery(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        ):
            raise TurnExecutionFinalizationConflict(
                "Auxiliary root Delivery requires reference-only PASS publication"
            )
        notification = None
        if l1_terminal_failure_code is not None:
            bound_run_id = str(window.get("current_l1_turn_run_id") or "")
            if not bound_run_id:
                raise TurnExecutionFinalizationConflict("L1 notification has no Window-bound run")
            notification = require_terminal_notification(
                conn, session_id=session_id, turn_id=turn_id,
                failure_code=l1_terminal_failure_code, assistant_content=assistant_content,
                bound_run_id=bound_run_id,
            )
        elif processing_level == "L1":
            l1_turn_runs.complete_l1_run_in_transaction(
                conn,
                session_id=session_id,
                turn_id=turn_id,
                assistant_content=assistant_content,
                completed_at=now,
            )

        # Formal commit：沿用已接受的用户消息，只追加 assistant 与完整对话对关联。
        delivery = transcript_turns.append_assistant_for_accepted_input_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            assistant_content=assistant_content,
            occurred_at=now,
        )
        jobs = _insert_post_commit_jobs(
            conn,
            deps=deps,
            session_id=session_id,
            turn_id=turn_id,
            job_kinds=post_commit_job_kinds,
            now=now,
        )
        conn.execute(
            "UPDATE runtime_turns SET status='completed', processing_level=?, error_code=?, "
            "end_reason=?, completed_at=? WHERE turn_id=? AND status='running'",
            (
                processing_level,
                notification.error_code if notification is not None else None,
                "l1_terminal_notification" if notification is not None else None,
                now, turn_id,
            ),
        )
        next_revision = int(window["state_version"]) + 1
        updated = conn.execute(
            "UPDATE turn_execution_windows SET window_state='post_commit_pending', stage='PERSIST', "
            "state_version=?, lease_owner=NULL, heartbeat_at=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=? "
            "AND (? IS NULL OR lease_owner=?)",
            (
                next_revision,
                now,
                now,
                session_id,
                turn_id,
                expected_window_revision,
                expected_lease_owner,
                expected_lease_owner,
            ),
        ).rowcount
        if not updated:
            raise TurnExecutionPersistenceError("Turn window changed during finalization")
        return {
            "replayed": False,
            "turn": _require_turn(conn, turn_id),
            "delivery": delivery,
            "post_commit_jobs": jobs,
            "window": _require_window(conn, session_id, turn_id),
        }


def finalize_verified_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    delivery_id: str,
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """用于一个已验证根 Delivery 的兼容包装器。"""

    return finalize_verified_turn_deliveries(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        delivery_ids=(delivery_id,),
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_verified_turn_deliveries(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    delivery_ids: tuple[str, ...],
    pending_question_attempt_ids: tuple[str, ...] = (),
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """发布有序的已验证根 Delivery，而不复制其正文。

    每个冻结 OutputWindow 仍是其正文唯一的物理所有者。此事务会创建一个空的正式
    assistant 槽位及一个有序引用元组，再结算 Runtime Turn 和持久化提交后任务。
    """

    _validate_delivery_ids(delivery_ids)
    _validate_question_attempt_ids(pending_question_attempt_ids)
    return finalize_referenced_turn_execution(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        reference_items=(
            tuple(
                FormalTurnReference("node_delivery", delivery_id)
                for delivery_id in delivery_ids
            )
            + tuple(
                FormalTurnReference(
                    "pending_question_attempt",
                    question_attempt_id,
                )
                for question_attempt_id in pending_question_attempt_ids
            )
        ),
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_referenced_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    reference_items: tuple[FormalTurnReference, ...],
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """重建 lane 权威状态后，发布调用方携带的精确元组。"""

    return _finalize_referenced_turn_execution(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        reference_items=reference_items,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_authoritative_referenced_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """在一个事务中推导并发布精确有序的正式引用。"""

    return _finalize_referenced_turn_execution(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        reference_items=None,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def mark_authoritative_no_public_turn_stop(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> dict[str, object]:
    """以原子方式证明并标记一个不含公开引用的全 lane 停止。

    持久化 v54 lane manifest 与当前 WorkRun 聚合是唯一结果权威状态。若存在部分
    lane 前缀、任何公开 root/question 或任何未结算执行 lane，则会在事务写入前
    抛出异常。Turn 有意保持 ``running``，直到常规下一输入审计完成结算；在此期间，
    完全重复的请求仍可将此有类型标记重放为同一公开 ``incomplete`` 结果。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        turn = _require_turn(conn, turn_id)
        if turn["session_id"] != session_id:
            raise TurnExecutionFinalizationConflict(
                "Turn belongs to another Session"
            )
        if turn["status"] != "running":
            raise TurnExecutionFinalizationConflict(
                f"only a running Turn can record a no-public stop, got {turn['status']}"
            )

        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] not in {"active", "interrupted"}:
            raise TurnExecutionFinalizationConflict(
                "no-public stop requires an active or exactly replayed interrupted Window"
            )
        if any(
            window[field] is not None
            for field in (
                "current_work_run_id",
                "current_attempt_id",
                "latest_checkpoint_id",
            )
        ):
            raise TurnExecutionFinalizationConflict(
                "no-public stop requires a settled Turn work window"
            )

        stop = _derive_authoritative_no_public_stop(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
        if window["window_state"] == "interrupted":
            if (
                str(window["stage"] or "") != str(stop["stage"])
                or str(window["interruption_reason"] or "")
                != str(stop["error_code"])
            ):
                raise TurnExecutionFinalizationConflict(
                    "interrupted Window disagrees with authoritative no-public stop"
                )
            return {
                "replayed": True,
                "turn": turn,
                "window": window,
                **stop,
            }

        if window["stage"] not in {"L2_PLAN", "PERSIST"}:
            raise TurnExecutionFinalizationConflict(
                "no-public stop requires an orchestration-settled Turn window"
            )
        next_revision = int(window["state_version"]) + 1
        source_stage = str(window["stage"])
        updated = conn.execute(
            "UPDATE turn_execution_windows SET window_state='interrupted', stage=?, "
            "state_version=?, lease_owner=NULL, heartbeat_at=?, interruption_reason=?, "
            "updated_at=? WHERE session_id=? AND turn_id=? AND state_version=? "
            "AND window_state='active' AND stage=?",
            (
                stop["stage"],
                next_revision,
                now,
                stop["error_code"],
                now,
                session_id,
                turn_id,
                expected_window_revision,
                source_stage,
            ),
        ).rowcount
        if updated != 1:
            raise TurnExecutionPersistenceError(
                "Turn window changed during authoritative no-public stop"
            )
        return {
            "replayed": False,
            "turn": turn,
            "window": _require_window(conn, session_id, turn_id),
            **stop,
        }


def _finalize_referenced_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    reference_items: tuple[FormalTurnReference, ...] | None,
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """持有一个即时事务时，发布按来源排序的引用。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if reference_items is not None:
        _validate_formal_reference_items(reference_items)
        for item in reference_items:
            _require_identifier("formal_reference_id", item.reference_id)
    _validate_job_kinds(post_commit_job_kinds)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        turn = _require_turn(conn, turn_id)
        if turn["session_id"] != session_id:
            raise TurnExecutionFinalizationConflict(
                "Turn belongs to another Session"
            )
        owned_window = None
        if expected_lease_owner is not None:
            owned_window = _require_window(conn, session_id, turn_id)
            _require_expected_lease_owner(owned_window, expected_lease_owner)
        if turn["status"] == "completed":
            replay_items = reference_items
            if replay_items is None:
                commit = conn.execute(
                    "SELECT run_id FROM session_turn_commits "
                    "WHERE session_id=? AND turn_id=?",
                    (session_id, turn_id),
                ).fetchone()
                if commit is None:
                    raise TurnExecutionFinalizationConflict(
                        "completed Turn has no formal transcript commit"
                    )
                try:
                    replay_items = (
                        transcript_turns._load_commit_references_in_transaction(
                            conn,
                            run_id=str(commit["run_id"]),
                        )
                    )
                except (RuntimeError, ValueError) as exc:
                    raise TurnExecutionFinalizationConflict(
                        "completed Turn formal reference authority is invalid"
                    ) from exc
                if not replay_items:
                    raise TurnExecutionFinalizationConflict(
                        "completed Turn is not a reference-only L2 delivery"
                    )
            return _replay_verified_finalization(
                conn,
                turn=turn,
                session_id=session_id,
                reference_items=replay_items,
                post_commit_job_kinds=post_commit_job_kinds,
            )
        if turn["status"] != "running":
            raise TurnExecutionFinalizationConflict(
                f"only a running Turn can finalize, got {turn['status']}"
            )
        window = owned_window or _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        _require_expected_lease_owner(window, expected_lease_owner)
        authoritative_stage = (
            reference_items is None
            and window["stage"] in {"L2_PLAN", "PERSIST"}
        )
        if window["window_state"] != "active" or not (
            window["stage"] == "PERSIST" or authoritative_stage
        ):
            raise TurnExecutionFinalizationConflict(
                "verified delivery requires an active PERSIST Turn window"
            )
        if any(
            window[field] is not None
            for field in (
                "current_work_run_id",
                "current_attempt_id",
                "latest_checkpoint_id",
            )
        ):
            raise TurnExecutionFinalizationConflict(
                "verified delivery requires a settled Turn work window"
            )
        authoritative_items = _derive_authoritative_formal_references(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
        if reference_items is None:
            reference_items = authoritative_items
        elif authoritative_items != reference_items:
            raise TurnExecutionFinalizationConflict(
                "formal reference tuple disagrees with current Turn lane authority"
            )

        try:
            transcript_commit = (
                transcript_turns.append_referenced_assistant_for_accepted_input_in_transaction(
                    conn,
                    session_id=session_id,
                    turn_id=turn_id,
                    reference_items=reference_items,
                    occurred_at=now,
                )
            )
        except ValueError as exc:
            raise TurnExecutionFinalizationConflict(
                "verified transcript commit conflicts with durable authority"
            ) from exc
        if not bool(transcript_commit["created"]):
            raise TurnExecutionFinalizationConflict(
                "running Turn already has a formal transcript commit"
            )
        jobs = _insert_post_commit_jobs(
            conn,
            deps=deps,
            session_id=session_id,
            turn_id=turn_id,
            job_kinds=post_commit_job_kinds,
            now=now,
        )
        updated_turn = conn.execute(
            "UPDATE runtime_turns SET status='completed', processing_level='L2', "
            "error_code=NULL, end_reason=NULL, completed_at=? "
            "WHERE turn_id=? AND status='running'",
            (now, turn_id),
        ).rowcount
        if updated_turn != 1:
            raise TurnExecutionPersistenceError(
                "Turn changed during verified finalization"
            )
        next_revision = int(window["state_version"]) + 1
        source_stage = str(window["stage"])
        updated_window = conn.execute(
            "UPDATE turn_execution_windows SET window_state='post_commit_pending', "
            "stage='PERSIST', state_version=?, lease_owner=NULL, heartbeat_at=?, "
            "updated_at=? WHERE session_id=? AND turn_id=? AND state_version=? "
            "AND window_state='active' AND stage=? "
            "AND (? IS NULL OR lease_owner=?)",
            (
                next_revision,
                now,
                now,
                session_id,
                turn_id,
                expected_window_revision,
                source_stage,
                expected_lease_owner,
                expected_lease_owner,
            ),
        ).rowcount
        if updated_window != 1:
            raise TurnExecutionPersistenceError(
                "Turn window changed during verified finalization"
            )
        return {
            "replayed": False,
            "turn": _require_turn(conn, turn_id),
            "delivery": transcript_commit,
            "formal_reference_items": reference_items,
            "node_delivery_ids": tuple(
                item.reference_id
                for item in reference_items
                if item.reference_kind == "node_delivery"
            ),
            "pending_question_attempt_ids": tuple(
                item.reference_id
                for item in reference_items
                if item.reference_kind == "pending_question_attempt"
            ),
            "post_commit_jobs": jobs,
            "window": _require_window(conn, session_id, turn_id),
        }


def list_turn_post_commit_jobs(
    deps: StoreDeps,
    turn_id: str,
) -> list[dict[str, object]]:
    """列出某个已接受 Turn 的持久化派生状态任务。"""

    _require_identifier("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_turn(conn, turn_id)
        return _list_post_commit_jobs(conn, turn_id)


def claim_due_turn_post_commit_jobs(
    deps: StoreDeps,
    *,
    session_id: str,
    worker_id: str,
    lease_seconds: int,
    limit: int = 16,
) -> list[dict[str, object]]:
    """租用一页有界提交后任务，且不跨 I/O 持有租约。"""

    _require_identifier("session_id", session_id)
    _require_identifier("worker_id", worker_id)
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than zero")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    deps.init_db()
    now = deps.now()
    lease_until = _add_seconds(now, lease_seconds)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session(conn, session_id)
        candidates = conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM turn_post_commit_jobs WHERE session_id=? AND ("
            "(status IN ('pending', 'retryable_failed') AND "
            "(next_retry_at IS NULL OR next_retry_at <= ?)) OR "
            "(status='processing' AND lease_until IS NOT NULL AND lease_until <= ?)"
            ") ORDER BY created_at, job_id LIMIT ?",
            (session_id, now, now, limit),
        ).fetchall()
        claimed: list[dict[str, object]] = []
        for row in candidates:
            updated = conn.execute(
                "UPDATE turn_post_commit_jobs SET status='processing', attempts=attempts+1, "
                "lease_owner=?, lease_until=?, reason_code=NULL, updated_at=? "
                "WHERE job_id=?",
                (worker_id, lease_until, now, row["job_id"]),
            ).rowcount
            if updated:
                claimed_row = conn.execute(
                    f"SELECT {_JOB_COLUMNS} FROM turn_post_commit_jobs WHERE job_id=?",
                    (row["job_id"],),
                ).fetchone()
                if claimed_row is not None:
                    claimed.append(_job_from_row(claimed_row))
        return claimed


def mark_turn_post_commit_job_applied(
    deps: StoreDeps,
    *,
    job_id: str,
    worker_id: str,
) -> dict[str, object]:
    """将 worker 持有的派生状态任务标记为完成；重放具有幂等性。"""

    _require_identifier("job_id", job_id)
    _require_identifier("worker_id", worker_id)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _require_job(conn, job_id)
        if job["status"] == "applied":
            return job
        updated = conn.execute(
            "UPDATE turn_post_commit_jobs SET status='applied', lease_owner=NULL, lease_until=NULL, "
            "next_retry_at=NULL, reason_code=NULL, updated_at=?, completed_at=? "
            "WHERE job_id=? AND status='processing' AND lease_owner=?",
            (now, now, job_id, worker_id),
        ).rowcount
        if not updated:
            raise TurnPostCommitJobLeaseError("post-commit job is not owned by this worker")
        return _require_job(conn, job_id)


def reconcile_turn_post_commit_job_applied(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    job_id: str,
    expected_job_kind: str,
) -> dict[str, object]:
    """Settle a durable job after its idempotent external effect was re-proven.

    This recovery path intentionally does not require the old worker lease: a Host may
    crash after committing the derived index but before settling the Session database.
    Only a trusted handler that has independently proven the exact effect may call it.
    """

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("job_id", job_id),
        ("expected_job_kind", expected_job_kind),
    ):
        _require_identifier(name, value)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _require_job(conn, job_id)
        if job["session_id"] != session_id or job["turn_id"] != turn_id:
            raise TurnExecutionPersistenceError(
                "post-commit reconciliation crossed its Session or Turn boundary"
            )
        if job["job_kind"] != expected_job_kind:
            raise TurnExecutionPersistenceError(
                "post-commit reconciliation named the wrong job kind"
            )
        if job["status"] == "waived":
            return job
        if job["status"] == "applied":
            return job
        conn.execute(
            "UPDATE turn_post_commit_jobs SET status='applied', next_retry_at=NULL, "
            "lease_owner=NULL, lease_until=NULL, reason_code=NULL, updated_at=?, "
            "completed_at=? WHERE job_id=?",
            (now, now, job_id),
        )
        return _require_job(conn, job_id)


def mark_turn_post_commit_job_failed(
    deps: StoreDeps,
    *,
    job_id: str,
    worker_id: str,
    reason_code: str,
    retry_after_seconds: int | None,
) -> dict[str, object]:
    """将 worker 持有的任务结束为可重试失败或终态失败。"""

    _require_identifier("job_id", job_id)
    _require_identifier("worker_id", worker_id)
    _require_identifier("reason_code", reason_code)
    if retry_after_seconds is not None and retry_after_seconds <= 0:
        raise ValueError("retry_after_seconds must be greater than zero when provided")

    status: PostCommitJobStatus = (
        "retryable_failed" if retry_after_seconds is not None else "terminal_failed"
    )
    deps.init_db()
    now = deps.now()
    next_retry_at = _add_seconds(now, retry_after_seconds) if retry_after_seconds else None
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_job(conn, job_id)
        updated = conn.execute(
            "UPDATE turn_post_commit_jobs SET status=?, next_retry_at=?, lease_owner=NULL, lease_until=NULL, "
            "reason_code=?, updated_at=?, completed_at=? "
            "WHERE job_id=? AND status='processing' AND lease_owner=?",
            (
                status,
                next_retry_at,
                reason_code,
                now,
                now if status == "terminal_failed" else None,
                job_id,
                worker_id,
            ),
        ).rowcount
        if not updated:
            raise TurnPostCommitJobLeaseError("post-commit job is not owned by this worker")
        return _require_job(conn, job_id)


def apply_turn_post_commit_job_control(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    request_id: str,
    action: PostCommitControlAction,
    job_ids: tuple[str, ...],
    expected_failed_job_digest: str,
    actor: str,
) -> dict[str, object]:
    """为终态失败持久地应用显式重试或过期豁免。

    Authorization 由调用方/Guard 验证。存储层将命令绑定到精确失败集合与当前
    Window revision，因此旧确认无法静默影响后续 Turn 或已变化的任务集合。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("request_id", request_id)
    _require_identifier("expected_failed_job_digest", expected_failed_job_digest)
    _require_identifier("actor", actor)
    if action not in {"retry", "waive"}:
        raise ValueError("action must be retry or waive")
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if not job_ids or len(set(job_ids)) != len(job_ids):
        raise ValueError("job_ids must be non-empty and unique")
    for job_id in job_ids:
        _require_identifier("job_id", job_id)

    deps.init_db()
    now = deps.now()
    canonical_job_ids = tuple(sorted(job_ids))
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM turn_post_commit_job_controls WHERE session_id=? AND request_id=?",
            (session_id, request_id),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["turn_id"]) != turn_id
                or str(existing["action"]) != action
                or int(existing["expected_window_revision"]) != expected_window_revision
                or str(existing["failed_job_digest"]) != expected_failed_job_digest
                or tuple(json.loads(str(existing["job_ids_json"]))) != canonical_job_ids
                or str(existing["actor"]) != actor
            ):
                raise TurnExecutionRequestIdCollision(
                    "post-commit control request id was reused for different input"
                )
            return {
                "replayed": True,
                "control": _control_from_row(existing),
                "post_commit_jobs": _list_post_commit_jobs(conn, turn_id),
                # 原控制回执不依赖旧 Turn 继续占窗；此处只读当前窗口，不能修改
                # 已释放的窗口或后续 Turn，也不能伪造当初控制后的窗口快照。
                "window": _load_window(conn, session_id),
            }

        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] != "post_commit_pending":
            raise TurnExecutionPersistenceError("post-commit control requires a pending Window")
        jobs = _jobs_by_ids(conn, turn_id=turn_id, session_id=session_id, job_ids=canonical_job_ids)
        if any(job["status"] != "terminal_failed" for job in jobs):
            raise TurnExecutionPersistenceError("post-commit control requires terminal failed jobs")
        actual_digest = _failed_job_digest(jobs)
        if actual_digest != expected_failed_job_digest:
            raise TurnExecutionWindowRevisionConflict(
                expected=expected_window_revision,
                actual=int(window["state_version"]),
            )

        if action == "retry":
            conn.executemany(
                "UPDATE turn_post_commit_jobs SET status='pending', next_retry_at=?, lease_owner=NULL, "
                "lease_until=NULL, reason_code=NULL, updated_at=?, completed_at=NULL "
                "WHERE job_id=? AND status='terminal_failed'",
                ((now, now, job_id) for job_id in canonical_job_ids),
            )
        else:
            conn.executemany(
                "UPDATE turn_post_commit_jobs SET status='waived', next_retry_at=NULL, lease_owner=NULL, "
                "lease_until=NULL, reason_code='USER_AUTHORIZED_STALE', updated_at=?, completed_at=? "
                "WHERE job_id=? AND status='terminal_failed'",
                ((now, now, job_id) for job_id in canonical_job_ids),
            )

        control_id = f"postctl_{deps.new_id()}"
        conn.execute(
            "INSERT INTO turn_post_commit_job_controls "
            "(control_id, session_id, turn_id, request_id, action, expected_window_revision, "
            "failed_job_digest, job_ids_json, actor, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                control_id,
                session_id,
                turn_id,
                request_id,
                action,
                expected_window_revision,
                expected_failed_job_digest,
                json.dumps(canonical_job_ids, separators=(",", ":")),
                actor,
                now,
            ),
        )
        next_revision = int(window["state_version"]) + 1
        conn.execute(
            "UPDATE turn_execution_windows SET state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (next_revision, now, session_id, turn_id, expected_window_revision),
        )
        control = conn.execute(
            "SELECT * FROM turn_post_commit_job_controls WHERE control_id=?",
            (control_id,),
        ).fetchone()
        if control is None:
            raise TurnExecutionPersistenceError("post-commit control was not persisted")
        return {
            "replayed": False,
            "control": _control_from_row(control),
            "post_commit_jobs": _list_post_commit_jobs(conn, turn_id),
            "window": _require_window(conn, session_id, turn_id),
        }


def release_turn_execution_window(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> dict[str, object]:
    """仅在所有任务均已结算后，释放已完成 Turn 的槽位。

    runner 的检查只是调度提示，这里在事务内再次核对 completed Turn、
    post_commit_pending Window、revision 和所有 jobs 的 applied/waived 状态。
    释放清空的是当前执行引用并递增版本，不删除 Turn、正式历史或 trajectory。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        window = _require_window(conn, session_id, turn_id)
        _require_expected_window_revision(window, expected_window_revision)
        if window["window_state"] != "post_commit_pending":
            raise TurnExecutionPersistenceError("only post-commit-pending Window can release")
        turn = _require_turn(conn, turn_id)
        if turn["status"] != "completed":
            raise TurnExecutionPersistenceError("only completed Turn can release its Window")
        unsettled = [
            job
            for job in _list_post_commit_jobs(conn, turn_id)
            if job["status"] not in {"applied", "waived"}
        ]
        if unsettled:
            raise TurnPostCommitJobsPending(unsettled)

        next_revision = int(window["state_version"]) + 1
        conn.execute(
            "UPDATE turn_execution_windows SET turn_id=NULL, window_state='empty', stage=NULL, "
            "input_message_id=NULL, attachment_binding_revision=0, turn_task_link_revision=0, "
            "turn_workrun_link_revision=0, current_work_run_id=NULL, current_attempt_id=NULL, "
            f"{_l1_window_reset_clause(conn)}"
            "latest_checkpoint_id=NULL, pending_operation_id=NULL, last_event_sequence=NULL, "
            "state_version=?, lease_owner=NULL, heartbeat_at=NULL, interruption_reason=NULL, "
            "claimed_at=NULL, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (next_revision, now, session_id, turn_id, expected_window_revision),
        )
        released = _load_window(conn, session_id)
        if released is None:
            raise TurnExecutionPersistenceError("released Turn window disappeared")
        return released


def _replay_finalization(
    conn: sqlite3.Connection,
    *,
    turn: dict[str, object],
    processing_level: str,
    assistant_content: str,
    post_commit_job_kinds: tuple[str, ...],
    session_id: str,
) -> dict[str, object]:
    """校验已完成 Turn 的重复提交确实对应同一份正式交付，再返回原回执投影。

    processing level、assistant 正文和完整 job kind 集必须一致；任何差异均为
    finalization conflict。Window 读取当前 Session 状态，不能伪造旧窗口快照。
    """

    if turn["processing_level"] != processing_level:
        raise TurnExecutionFinalizationConflict("completed Turn has another processing level")
    try:
        delivery = transcript_turns.append_assistant_for_accepted_input_in_transaction(
            conn,
            session_id=session_id,
            turn_id=str(turn["turn_id"]),
            assistant_content=assistant_content,
            occurred_at=str(turn["completed_at"] or turn["received_at"]),
        )
    except ValueError as exc:
        raise TurnExecutionFinalizationConflict(
            "completed Turn has another transcript delivery"
        ) from exc
    jobs = _list_post_commit_jobs(conn, str(turn["turn_id"]))
    if tuple(str(item["job_kind"]) for item in jobs) != tuple(sorted(post_commit_job_kinds)):
        raise TurnExecutionFinalizationConflict("completed Turn has another post-commit job set")
    return {
        "replayed": True,
        "turn": turn,
        "delivery": delivery,
        "post_commit_jobs": jobs,
        "window": _load_window(conn, session_id),
    }


def _replay_verified_finalization(
    conn: sqlite3.Connection,
    *,
    turn: dict[str, object],
    session_id: str,
    reference_items: tuple[FormalTurnReference, ...],
    post_commit_job_kinds: tuple[str, ...],
) -> dict[str, object]:
    if turn["processing_level"] != "L2":
        raise TurnExecutionFinalizationConflict(
            "completed Turn is not a verified L2 delivery"
        )
    try:
        transcript_commit = (
            transcript_turns.append_referenced_assistant_for_accepted_input_in_transaction(
                conn,
                session_id=session_id,
                turn_id=str(turn["turn_id"]),
                reference_items=reference_items,
                occurred_at=str(turn["completed_at"] or turn["received_at"]),
            )
        )
    except ValueError as exc:
        raise TurnExecutionFinalizationConflict(
            "completed Turn has another transcript delivery"
        ) from exc
    jobs = _list_post_commit_jobs(conn, str(turn["turn_id"]))
    if tuple(str(item["job_kind"]) for item in jobs) != tuple(
        sorted(post_commit_job_kinds)
    ):
        raise TurnExecutionFinalizationConflict(
            "completed Turn has another post-commit job set"
        )
    return {
        "replayed": True,
        "turn": turn,
        "delivery": transcript_commit,
        "formal_reference_items": reference_items,
        "node_delivery_ids": tuple(
            item.reference_id
            for item in reference_items
            if item.reference_kind == "node_delivery"
        ),
        "pending_question_attempt_ids": tuple(
            item.reference_id
            for item in reference_items
            if item.reference_kind == "pending_question_attempt"
        ),
        "post_commit_jobs": jobs,
        "window": _load_window(conn, session_id),
    }


def _derive_authoritative_formal_references(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[FormalTurnReference, ...]:
    """从当前 lane 权威状态重建非空公开元组。"""

    references, _no_public_statuses = _derive_authoritative_turn_settlement(
        conn,
        session_id=session_id,
        turn_id=turn_id,
    )
    if not references:
        raise TurnExecutionFinalizationConflict(
            "Turn has no authoritative formal Delivery or pending question"
        )
    return references


def _derive_authoritative_turn_settlement(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[tuple[FormalTurnReference, ...], tuple[str, ...]]:
    """从持久化 lane manifest 重建公开引用和有类型的非公开状态。"""

    manifest = _load_turn_lane_manifest_in_transaction(
        conn,
        session_id=session_id,
        turn_id=turn_id,
    )
    lane_by_task = {
        lane.insession_task_id: lane
        for lane in manifest.lanes
    }

    root_rows = conn.execute(
        "SELECT delivery.delivery_id, delivery.insession_task_id "
        "FROM insession_task_node_deliveries AS delivery "
        "JOIN insession_tasks AS task "
        "ON task.session_id=delivery.session_id "
        "AND task.insession_task_id=delivery.insession_task_id "
        "AND task.current_graph_revision=delivery.graph_revision "
        "WHERE delivery.session_id=? AND delivery.created_turn_id=? "
        "AND delivery.insession_task_node_id=delivery.insession_task_id "
        "ORDER BY delivery.insession_task_id, delivery.delivery_id",
        (session_id, turn_id),
    ).fetchall()
    roots_by_task: dict[str, list[str]] = {}
    for row in root_rows:
        delivery_id = str(row["delivery_id"])
        task_id = _require_verified_root_delivery_for_turn(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=delivery_id,
        )
        roots_by_task.setdefault(task_id, []).append(delivery_id)

    question_rows = conn.execute(
        "SELECT attempt.attempt_id FROM insession_work_run_attempts AS attempt "
        "JOIN insession_work_runs AS run ON run.work_run_id=attempt.work_run_id "
        "WHERE run.session_id=? AND attempt.turn_id=? "
        "AND attempt.status='closed' AND attempt.action='request_user_input' "
        "AND run.status='waiting_user' AND run.reason='needs_input' "
        "AND run.updated_turn_id=? AND run.current_attempt_id IS NULL "
        "AND run.current_verification_request_id IS NULL "
        "AND attempt.ordinal=run.attempts_started "
        "ORDER BY attempt.attempt_id",
        (session_id, turn_id, turn_id),
    ).fetchall()
    questions_by_task: dict[str, list[tuple[int, int, str]]] = {}
    for row in question_rows:
        attempt_id = str(row["attempt_id"])
        task_id, node_ordinal, attempt_ordinal = (
            _require_current_pending_question_for_turn(
                conn,
                session_id=session_id,
                turn_id=turn_id,
                question_attempt_id=attempt_id,
            )
        )
        questions_by_task.setdefault(task_id, []).append(
            (node_ordinal, attempt_ordinal, attempt_id)
        )

    candidate_tasks = set(roots_by_task) | set(questions_by_task)
    for task_id in candidate_tasks:
        lane = lane_by_task.get(task_id)
        if lane is None or not lane.execution_requested:
            raise TurnExecutionFinalizationConflict(
                "formal reference candidate is outside an execution-requested Turn lane"
            )

    ordered: list[FormalTurnReference] = []
    no_public_statuses: list[str] = []
    for lane in manifest.lanes:
        root_ids = roots_by_task.get(lane.insession_task_id, [])
        questions = sorted(questions_by_task.get(lane.insession_task_id, []))
        if len(root_ids) > 1:
            raise TurnExecutionFinalizationConflict(
                "one Task lane has multiple current root Deliveries"
            )
        if root_ids and questions:
            raise TurnExecutionFinalizationConflict(
                "one Task lane cannot be completed and waiting for user input"
            )
        if not lane.execution_requested and (root_ids or questions):
            raise TurnExecutionFinalizationConflict(
                "non-execution Task lane produced a formal reference"
            )
        if lane.execution_requested and not root_ids and not questions:
            no_public_statuses.extend(
                _require_no_reference_lane_settlement(
                    conn,
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=lane.insession_task_id,
                )
            )
        if root_ids:
            ordered.append(FormalTurnReference("node_delivery", root_ids[0]))
        else:
            ordered.extend(
                FormalTurnReference("pending_question_attempt", attempt_id)
                for _node_ordinal, _attempt_ordinal, attempt_id in questions
            )
    return tuple(ordered), tuple(no_public_statuses)


def _derive_authoritative_no_public_stop(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, str]:
    """为全无正文 lane 集合投影唯一的公开异常结果。"""

    references, statuses = _derive_authoritative_turn_settlement(
        conn,
        session_id=session_id,
        turn_id=turn_id,
    )
    if references:
        raise TurnExecutionFinalizationConflict(
            "Turn has an authoritative public reference"
        )
    if not statuses:
        raise TurnExecutionFinalizationConflict(
            "Turn has no authoritative no-public execution stop"
        )
    if "waiting_external" in statuses:
        return {
            "stop_kind": "waiting_external",
            "end_reason": "host_stopped",
            "error_code": "TOOL_COMPLETION_UNCONFIRMED",
            "stage": "TOOL",
        }
    if "turn_limit_reached" in statuses:
        return {
            "stop_kind": "turn_limit_reached",
            "end_reason": "host_stopped",
            "error_code": "TURN_DEADLINE_EXCEEDED",
            "stage": "RESPONSE",
        }
    if set(statuses) == {"failed"}:
        return {
            "stop_kind": "work_run_failed",
            "end_reason": "module_error",
            "error_code": "INTERNAL_FAILURE",
            "stage": "RESPONSE",
        }
    raise TurnExecutionFinalizationConflict(
        "Turn has an unsupported no-public execution stop"
    )


def _require_no_reference_lane_settlement(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> tuple[str, ...]:
    """证明一个执行所请求的 lane 已安全停止且没有公开文本。"""

    rows = conn.execute(
        "SELECT run.work_run_id, run.status, run.reason, run.updated_turn_id, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "link.link_revision FROM insession_work_runs AS run "
        "JOIN insession_work_run_turn_links AS link "
        "ON link.session_id=run.session_id AND link.work_run_id=run.work_run_id "
        "AND link.turn_id=? "
        "WHERE run.session_id=? AND run.insession_task_id=? "
        "ORDER BY link.link_revision, run.work_run_id",
        (turn_id, session_id, task_id),
    ).fetchall()
    safe_statuses = {"waiting_external", "turn_limit_reached", "failed"}
    safe_rows = tuple(
        row
        for row in rows
        if str(row["updated_turn_id"]) == turn_id
        and (
            (
                str(row["status"]) == "waiting_external"
                and str(row["reason"] or "")
                == "operation_completion_unconfirmed"
            )
            or (
                str(row["status"]) == "turn_limit_reached"
                and str(row["reason"] or "") == "turn_limit_reached"
            )
            or str(row["status"]) == "failed"
        )
        and row["current_attempt_id"] is None
        and row["current_verification_request_id"] is None
    )
    unsafe_current = tuple(
        row
        for row in rows
        if str(row["updated_turn_id"]) == turn_id
        and str(row["status"])
        not in {"completed", "cancelled", *safe_statuses}
    )
    if not safe_rows or unsafe_current:
        raise TurnExecutionFinalizationConflict(
            "execution-requested Task is not completed and has no durable safe lane settlement"
        )
    # 在计入无正文 lane 前，完整聚合加载会发现过期 subject、损坏 budget 和不一致的
    # 当前 Attempt 投影。
    from ..l2.work_run.work_execution import _load_record

    for row in safe_rows:
        try:
            record = _load_record(
                conn,
                str(row["work_run_id"]),
                session_id=session_id,
            )
        except (RuntimeError, ValueError) as exc:
            raise TurnExecutionFinalizationConflict(
                "no-body Task lane settlement authority is invalid"
            ) from exc
        if record.work_run.status.value != str(row["status"]):
            raise TurnExecutionFinalizationConflict(
                "no-body Task lane settlement status is inconsistent"
            )
        if (
            record.current_attempt_id is not None
            or record.current_verification_request_id is not None
        ):
            raise TurnExecutionFinalizationConflict(
                "no-body Task lane settlement retains active execution authority"
            )
    return tuple(str(row["status"]) for row in safe_rows)


def _load_turn_lane_manifest_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
):
    from ..l2.task_graph.insession_tasks import (
        _decode_ids,
        _load_authoritative_user_input,
        _load_task_match_lane_manifest,
    )

    rows = conn.execute(
        "SELECT related_insession_task_ids_json, "
        "execution_lane_manifest_json, execution_lane_manifest_hash "
        "FROM insession_task_match_apply_receipts "
        "WHERE session_id=? AND source_turn_id=? ORDER BY created_at, apply_id",
        (session_id, turn_id),
    ).fetchall()
    if len(rows) != 1:
        raise TurnExecutionFinalizationConflict(
            "Turn has no unique durable Task execution lane manifest"
        )
    try:
        manifest = _load_task_match_lane_manifest(rows[0])
        related_task_ids = _decode_ids(rows[0]["related_insession_task_ids_json"])
        user_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
    except (RuntimeError, ValueError) as exc:
        raise TurnExecutionFinalizationConflict(
            "Turn Task execution lane manifest is invalid"
        ) from exc
    if tuple(lane.insession_task_id for lane in manifest.lanes) != related_task_ids:
        raise TurnExecutionFinalizationConflict(
            "Turn Task execution lane manifest disagrees with its match receipt"
        )
    for lane in manifest.lanes:
        if conn.execute(
            "SELECT 1 FROM insession_tasks WHERE session_id=? "
            "AND insession_task_id=?",
            (session_id, lane.insession_task_id),
        ).fetchone() is None:
            raise TurnExecutionFinalizationConflict(
                "Turn Task execution lane references a missing Task"
            )
        for match in lane.matches:
            span = match.source_span
            excerpt = user_text[span.start : span.end]
            if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != span.text_sha256:
                raise TurnExecutionFinalizationConflict(
                    "Turn Task execution lane source binding has drifted"
                )
    return manifest


def _require_current_pending_question_for_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    question_attempt_id: str,
) -> tuple[str, int, int]:
    try:
        proof = require_pending_question_proof(
            conn,
            session_id=session_id,
            question_attempt_id=question_attempt_id,
        )
    except (InSessionTaskPersistenceError, sqlite3.Error, ValueError) as exc:
        raise TurnExecutionFinalizationConflict(
            "pending-question WorkRun authority is invalid"
        ) from exc
    if proof.question_turn_id != turn_id:
        raise TurnExecutionFinalizationConflict(
            "pending-question Attempt belongs to another Turn"
        )
    return (
        proof.insession_task_id,
        proof.node_ordinal,
        proof.attempt_ordinal,
    )


def _require_verified_deliveries_for_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    delivery_ids: tuple[str, ...],
) -> None:
    task_ids: set[str] = set()
    for delivery_id in delivery_ids:
        task_id = _require_verified_root_delivery_for_turn(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=delivery_id,
        )
        if task_id in task_ids:
            raise TurnExecutionFinalizationConflict(
                "verified formal Deliveries must belong to distinct root Tasks"
            )
        task_ids.add(task_id)


def _turn_owns_auxiliary_root_delivery(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> bool:
    """返回此 Turn 是否生成了 规范根 Delivery。"""

    row = conn.execute(
        "SELECT 1 FROM insession_task_node_deliveries AS delivery "
        "JOIN insession_auxiliary_v2_task_graph_commit_receipts AS commit_receipt "
        "ON commit_receipt.session_id=delivery.session_id "
        "AND commit_receipt.insession_task_id=delivery.insession_task_id "
        "AND commit_receipt.committed_task_graph_revision=delivery.graph_revision "
        "WHERE delivery.session_id=? AND delivery.created_turn_id=? "
        "AND delivery.insession_task_node_id=delivery.insession_task_id "
        "LIMIT 1",
        (session_id, turn_id),
    ).fetchone()
    return row is not None


def _require_verified_root_delivery_for_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    delivery_id: str,
) -> str:
    from ..l2.work_run.work_verification import _load_task_node_delivery

    try:
        resolved = _load_task_node_delivery(
            conn,
            session_id=session_id,
            delivery_id=delivery_id,
        )
    except (RuntimeError, ValueError) as exc:
        raise TurnExecutionFinalizationConflict(
            "verified NodeDelivery authority is invalid"
        ) from exc
    delivery = resolved.delivery
    if delivery.created_turn_id != turn_id:
        raise TurnExecutionFinalizationConflict(
            "verified NodeDelivery belongs to another Turn"
        )
    subject = delivery.subject
    if subject.node_id != subject.task_id:
        raise TurnExecutionFinalizationConflict(
            "verified formal Delivery is not a canonical root TaskNode"
        )
    task = conn.execute(
        "SELECT task.current_graph_revision, task.current_status, "
        "task.state_version, "
        "node.node_kind, node.node_revision "
        "FROM insession_tasks AS task "
        "LEFT JOIN insession_task_graph_nodes AS node "
        "ON node.insession_task_id=task.insession_task_id "
        "AND node.graph_revision=task.current_graph_revision "
        "AND node.insession_task_node_id=task.insession_task_id "
        "WHERE task.session_id=? AND task.insession_task_id=?",
        (session_id, subject.task_id),
    ).fetchone()
    if (
        task is None
        or str(task["current_status"]) != "completed"
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"])
        != subject.graph_revision
        or str(task["node_kind"] or "") != "root"
        or task["node_revision"] is None
        or int(task["node_revision"]) != subject.node_revision
    ):
        raise TurnExecutionFinalizationConflict(
            "verified delivery Task is not completed at its exact graph revision"
        )
    try:
        from ..l2.delivery.task_delivery_validation import (
            require_pass_settlement_for_publication_in_transaction,
        )

        require_pass_settlement_for_publication_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            task_id=subject.task_id,
            graph_revision=subject.graph_revision,
            task_state_version=int(task["state_version"]),
            root_delivery_id=delivery_id,
        )
    except Exception as exc:
        raise TurnExecutionFinalizationConflict(
            "root Delivery lacks current whole-Task PASS authority"
        ) from exc
    return subject.task_id


def _validate_delivery_ids(delivery_ids: tuple[str, ...]) -> None:
    if not isinstance(delivery_ids, tuple):
        raise ValueError("delivery_ids must be a tuple")
    if len(set(delivery_ids)) != len(delivery_ids):
        raise ValueError("delivery_ids must not contain duplicates")


def _validate_question_attempt_ids(
    question_attempt_ids: tuple[str, ...],
) -> None:
    if not isinstance(question_attempt_ids, tuple):
        raise ValueError("pending_question_attempt_ids must be a tuple")
    if len(set(question_attempt_ids)) != len(question_attempt_ids):
        raise ValueError(
            "pending_question_attempt_ids must not contain duplicates"
        )


def _validate_formal_reference_items(
    reference_items: tuple[FormalTurnReference, ...],
) -> None:
    if not isinstance(reference_items, tuple) or not reference_items:
        raise ValueError("reference_items must be a non-empty tuple")
    if any(
        not isinstance(item, FormalTurnReference) for item in reference_items
    ):
        raise TypeError("reference_items must contain FormalTurnReference values")
    reference_ids = tuple(item.reference_id for item in reference_items)
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("formal reference IDs must not contain duplicates")


def _insert_post_commit_jobs(
    conn: sqlite3.Connection,
    *,
    deps: StoreDeps,
    session_id: str,
    turn_id: str,
    job_kinds: tuple[str, ...],
    now: str,
) -> list[dict[str, object]]:
    for job_kind in sorted(job_kinds):
        conn.execute(
            "INSERT INTO turn_post_commit_jobs "
            "(job_id, session_id, turn_id, job_kind, status, attempts, next_retry_at, "
            "lease_owner, lease_until, reason_code, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, NULL, ?, ?, NULL)",
            (
                _post_commit_job_id(turn_id, job_kind),
                session_id,
                turn_id,
                job_kind,
                now,
                now,
                now,
            ),
        )
    return _list_post_commit_jobs(conn, turn_id)


def _validate_request_replay(
    conn: sqlite3.Connection,
    *,
    turn: dict[str, object],
    source: str,
    user_text: str,
    attachment_ids: tuple[str, ...],
) -> None:
    if turn["source"] != source or turn["user_text"] != user_text:
        raise TurnExecutionRequestIdCollision(
            "client request id was reused for different user input"
        )
    persisted_ids = tuple(
        str(item["attachment_id"])
        for item in _list_turn_attachment_bindings(conn, str(turn["turn_id"]))
    )
    if persisted_ids != attachment_ids:
        raise TurnExecutionRequestIdCollision(
            "client request id was reused for different attachment bindings"
        )


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
        raise ValueError(f"unknown session: {session_id}")


def _require_turn(conn: sqlite3.Connection, turn_id: str) -> dict[str, object]:
    row = conn.execute(
        f"SELECT {_turn_select_columns()} FROM runtime_turns WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown runtime turn: {turn_id}")
    return _turn_from_row(row)


def _require_turn_input(conn: sqlite3.Connection, turn_id: str) -> dict[str, object]:
    row = conn.execute(
        "SELECT i.message_id, i.session_id, i.turn_id, i.turn_idx, t.content, i.created_at "
        "FROM runtime_turn_inputs i JOIN session_turns t "
        "ON t.session_id=i.session_id AND t.turn_idx=i.turn_idx "
        "WHERE i.turn_id=? AND t.role='user'",
        (turn_id,),
    ).fetchone()
    if row is None:
        raise TurnExecutionPersistenceError(f"accepted runtime turn input not found: {turn_id}")
    return {
        "message_id": str(row["message_id"]),
        "session_id": str(row["session_id"]),
        "turn_id": str(row["turn_id"]),
        "turn_idx": int(row["turn_idx"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
    }


def _load_window(conn: sqlite3.Connection, session_id: str) -> dict[str, object] | None:
    row = conn.execute(
        f"SELECT {_WINDOW_COLUMNS} FROM turn_execution_windows WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    window = _window_from_row(row)
    window.setdefault("current_l1_turn_run_id", None)
    window.setdefault("current_l1_attempt_id", None)
    return window


def _l1_window_reset_clause(conn: sqlite3.Connection) -> str:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(turn_execution_windows)").fetchall()
    }
    if {"current_l1_turn_run_id", "current_l1_attempt_id"} <= columns:
        return "current_l1_turn_run_id=NULL, current_l1_attempt_id=NULL, "
    return ""


def _require_window(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
) -> dict[str, object]:
    window = _load_window(conn, session_id)
    if window is None or window["turn_id"] != turn_id:
        raise TurnExecutionPersistenceError("Turn does not own the current execution window")
    return window


def _require_expected_window_revision(
    window: dict[str, object],
    expected_window_revision: int,
) -> None:
    actual = int(window["state_version"])
    if actual != expected_window_revision:
        raise TurnExecutionWindowRevisionConflict(
            expected=expected_window_revision,
            actual=actual,
        )


def _require_expected_lease_owner(
    window: dict[str, object],
    expected_lease_owner: str | None,
) -> None:
    if expected_lease_owner is None:
        return
    if str(window.get("lease_owner") or "") != expected_lease_owner:
        raise TurnExecutionPersistenceError(
            "Turn execution lease belongs to another Runtime"
        )


def _next_turn_idx(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(turn_idx), -1) + 1 AS next_idx FROM session_turns WHERE session_id=?",
        (session_id,),
    ).fetchone()
    return int(row["next_idx"])


def _list_turn_attachment_bindings(
    conn: sqlite3.Connection,
    turn_id: str,
) -> list[dict[str, object]]:
    rows = conn.execute(
        "SELECT attachment_id, binding_ordinal, created_at "
        "FROM runtime_turn_attachment_bindings WHERE turn_id=? "
        "ORDER BY binding_ordinal, attachment_id",
        (turn_id,),
    ).fetchall()
    return [
        {
            "attachment_id": str(row["attachment_id"]),
            "binding_ordinal": int(row["binding_ordinal"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def _list_post_commit_jobs(
    conn: sqlite3.Connection,
    turn_id: str,
) -> list[dict[str, object]]:
    rows = conn.execute(
        f"SELECT {_JOB_COLUMNS} FROM turn_post_commit_jobs WHERE turn_id=? "
        "ORDER BY job_kind, job_id",
        (turn_id,),
    ).fetchall()
    return [_job_from_row(row) for row in rows]


def _require_job(conn: sqlite3.Connection, job_id: str) -> dict[str, object]:
    row = conn.execute(
        f"SELECT {_JOB_COLUMNS} FROM turn_post_commit_jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown post-commit job: {job_id}")
    return _job_from_row(row)


def _jobs_by_ids(
    conn: sqlite3.Connection,
    *,
    turn_id: str,
    session_id: str,
    job_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"SELECT {_JOB_COLUMNS} FROM turn_post_commit_jobs "
        f"WHERE turn_id=? AND session_id=? AND job_id IN ({placeholders}) "
        "ORDER BY job_id",
        (turn_id, session_id, *job_ids),
    ).fetchall()
    if len(rows) != len(job_ids):
        raise TurnExecutionPersistenceError("post-commit control references an unknown job")
    return [_job_from_row(row) for row in rows]


def _turn_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "turn_id": str(row["turn_id"]),
        "session_id": str(row["session_id"]),
        "client_request_id": _optional_str(row["client_request_id"]),
        "source": str(row["source"]),
        "user_text": str(row["user_text"]),
        "input_message_id": _optional_str(row["input_message_id"]),
        "status": str(row["status"]),
        "processing_level": _optional_str(row["processing_level"]),
        "error_code": _optional_str(row["error_code"]),
        "end_reason": _optional_str(row["end_reason"]),
        "received_at": str(row["received_at"]),
        "completed_at": _optional_str(row["completed_at"]),
        "execution_snapshot_json": _optional_str(
            row["execution_snapshot_json"]
        ),
        "execution_snapshot_sha256": _optional_str(
            row["execution_snapshot_sha256"]
        ),
    }


def _turn_select_columns() -> str:
    return (
        f"{_TURN_CORE_COLUMNS}, execution_snapshot_json, "
        "execution_snapshot_sha256"
    )


def _window_from_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    for field in (
        "attachment_binding_revision",
        "turn_task_link_revision",
        "turn_workrun_link_revision",
        "last_event_sequence",
        "state_version",
    ):
        if result[field] is not None:
            result[field] = int(result[field])
    return result


def _job_from_row(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    result["attempts"] = int(result["attempts"])
    return result


def _control_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "control_id": str(row["control_id"]),
        "session_id": str(row["session_id"]),
        "turn_id": str(row["turn_id"]),
        "request_id": str(row["request_id"]),
        "action": str(row["action"]),
        "expected_window_revision": int(row["expected_window_revision"]),
        "failed_job_digest": str(row["failed_job_digest"]),
        "job_ids": tuple(json.loads(str(row["job_ids_json"]))),
        "actor": str(row["actor"]),
        "created_at": str(row["created_at"]),
    }


def _terminal_failure_digest(jobs: list[dict[str, object]]) -> str | None:
    failed = [job for job in jobs if job["status"] == "terminal_failed"]
    return _failed_job_digest(failed) if failed else None


def _failed_job_digest(jobs: list[dict[str, object]]) -> str:
    payload = [
        {
            "job_id": str(job["job_id"]),
            "job_kind": str(job["job_kind"]),
            "status": str(job["status"]),
            "reason_code": job["reason_code"],
        }
        for job in sorted(jobs, key=lambda item: str(item["job_id"]))
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _post_commit_job_id(turn_id: str, job_kind: str) -> str:
    digest = hashlib.sha256(f"{turn_id}\x1f{job_kind}".encode("utf-8")).hexdigest()
    return f"pcj_{digest[:28]}"


def _validate_attachment_ids(attachment_ids: tuple[str, ...]) -> None:
    if len(set(attachment_ids)) != len(attachment_ids):
        raise ValueError("attachment_ids must not contain duplicates")
    for attachment_id in attachment_ids:
        _require_identifier("attachment_id", attachment_id)


def _validate_job_kinds(job_kinds: tuple[str, ...]) -> None:
    if len(set(job_kinds)) != len(job_kinds):
        raise ValueError("post_commit_job_kinds must not contain duplicates")
    for job_kind in job_kinds:
        _require_identifier("post_commit_job_kind", job_kind)


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_TURN_EXECUTION_IDENTIFIER_LENGTH
    ):
        raise ValueError(
            f"{name} must be a non-empty string of at most "
            f"{MAX_TURN_EXECUTION_IDENTIFIER_LENGTH} characters"
        )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _add_seconds(value: str, seconds: int) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed + timedelta(seconds=seconds)).isoformat()
