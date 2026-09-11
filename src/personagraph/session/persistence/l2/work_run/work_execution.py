"""由 TaskNode 持有的 P2 WorkRun/Attempt 切片的 SQLite 权威状态。

本模块有意止于 Runtime 编排层之下。它只接受由 Host 物化的 Attempt 决策，不将
任何模型提案持久化为执行权威状态，不调用工具，也不执行节点验证或终态 GC。
每项变更都持有一个短 ``BEGIN IMMEDIATE`` 事务，以及一个精确且带操作限定的
apply receipt。
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.work_run import (
    AcceptanceProgressSnapshot,
    AttemptDecision,
    AttemptStatus,
    Attempt,
    AuxiliaryNodeSubject,
    ExecutionSubject,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedSubmitOutputWindowAction,
    HostMaterializedToolCall,
    HostMaterializedWriteOutputWindowAction,
    NodeVerificationResult,
    OutputWindow,
    PendingUserQuestion,
    ResolvedCurrentTaskNodeDelivery,
    RequestUserInputAction,
    SubmitOutputWindowAction,
    TaskNodeSubject,
    TaskNodeDeliveryCarryAuthority,
    CurrentTaskNodeDeliveryResolutionKind,
    ToolResultStatus,
    ToolResult,
    WorkRunBudget,
    WorkRunBudgetDisposition,
    WorkRunBudgetTransition,
    WorkExecutionMutationResult,
    WorkRunStatus,
    WorkRun,
    WriteOutputWindowAction,
    apply_output_window_action,
    charge_work_run_active_seconds,
    create_work_run,
    initialize_acceptance_progress,
    initialize_output_window,
    merge_acceptance_progress,
)
from personagraph.l2.work_run.contracts import (
    RequestTaskGraphRevisionAction,
    TaskGraphExecutionReplanApplication,
    TaskGraphExecutionReplanRequest,
)
from ....turn_execution_contracts import TurnExecutionWindowRevisionConflict
from ...deps import StoreDeps
from ...execution_findings import (
    ExecutionFindingsPersistenceError,
    create_execution_findings_owner_companion_in_transaction,
)


_STARTABLE_TASK_STATUSES = {"proposed", "active", "interrupted"}
_STARTABLE_NODE_STATUSES = {"proposed", "active", "interrupted"}
_MAX_ATTEMPTS = 32
_SOFT_ACTIVE_SECONDS = 720.0
_HARD_ACTIVE_SECONDS = 900.0
class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredMaterializedToolCall(_Record):
    attempt_id: str
    ordinal: int = Field(ge=1)
    call: HostMaterializedToolCall


class StoredAttempt(_Record):
    attempt: Attempt
    # ``turn_id`` 是当前执行所有者，在未决 Attempt 恢复期间可能变化。
    # ``input_turn_id`` 是创建此语义 Attempt 的用户输入之不可变来源。
    turn_id: str
    input_turn_id: str
    predecessor_question_attempt_id: str | None = None
    action: Literal[
        "call_tools",
        "write_output_window",
        "submit_output_window",
        "request_user_input",
        "request_task_graph_revision",
    ] | None = None
    decision: HostAcceptedAttemptDecision | None = None
    input_output_revision: int = Field(ge=1)
    input_output_hash: str = Field(min_length=64, max_length=64)
    committed_output_revision: int | None = Field(default=None, ge=1)
    input_checkpoint_id: str | None = None
    input_verification_request_id: str | None = None
    input_verification_result: NodeVerificationResult | None = None
    catalog_snapshot: dict[str, Any]
    budget_before: WorkRunBudget
    budget_after: WorkRunBudget | None = None
    budget_charge_id: str | None = None
    close_reason: str | None = None


class StoredWorkRunBudgetCharge(_Record):
    budget_charge_id: str = Field(min_length=1, max_length=200)
    operation: Literal[
        "charge_active_time",
        "commit_attempt_decision",
        "commit_output_action",
        "close_attempt",
        "commit_verification_result",
        "interrupt_verification",
    ]
    turn_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=200)
    work_run_revision_before: int = Field(ge=1)
    work_run_revision_after: int = Field(ge=2)
    window_state_version_before: int = Field(ge=1)
    window_state_version_after: int = Field(ge=2)
    work_run_status_after: WorkRunStatus
    work_run_reason_after: str | None = None
    transition: WorkRunBudgetTransition


class StoredWorkRun(_Record):
    session_id: str
    work_run: WorkRun
    acceptance_progress: AcceptanceProgressSnapshot
    output_window: OutputWindow
    current_attempt_id: str | None = None
    current_verification_request_id: str | None = None
    node_delivery_id: str | None = None
    auxiliary_node_completion_id: str | None = None
    pending_user_question: str | None = None
    attempts: tuple[StoredAttempt, ...] = ()
    tool_calls: tuple[StoredMaterializedToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    budget_charges: tuple[StoredWorkRunBudgetCharge, ...] = ()
    related_turn_ids: tuple[str, ...] = ()


class TurnLinkedNonterminalWorkRunCandidate(_Record):
    """仅由某 Turn 的 Task 链接授权的小型恢复候选项。

    此投影有意不承担选择职责。调用方必须处理完整稳定元组，包括有歧义的
    ``N > 1`` 情况。
    """

    session_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    subject: ExecutionSubject
    status: WorkRunStatus
    reason: str | None = Field(default=None, min_length=1, max_length=160)
    work_run_revision: int = Field(ge=1)
    current_attempt_id: str | None = Field(default=None, min_length=1, max_length=200)
    current_verification_request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    created_turn_id: str = Field(min_length=1, max_length=200)
    updated_turn_id: str = Field(min_length=1, max_length=200)


class ReadyTaskNodeExecutionCandidate(_Record):
    """一个新的当前节点候选项，仍受创建时 CAS 约束。"""

    subject: TaskNodeSubject
    ordinal: int = Field(ge=0)
    node_status: Literal["proposed", "active", "interrupted"]
    node_state_version: int = Field(ge=1)
    dependency_delivery_ids: tuple[str, ...] = ()


class RecoverableTaskNodeExecutionCandidate(_Record):
    """一个精确的当前节点非终态 WorkRun；绝不表示自动恢复授权。"""

    subject: TaskNodeSubject
    ordinal: int = Field(ge=0)
    node_status: Literal[
        "active", "awaiting_user", "waiting_external", "interrupted"
    ]
    node_state_version: int = Field(ge=1)
    work_run_id: str = Field(min_length=1, max_length=200)
    work_run_status: WorkRunStatus
    work_run_reason: str | None = Field(default=None, min_length=1, max_length=160)
    work_run_revision: int = Field(ge=1)
    current_attempt_id: str | None = Field(default=None, min_length=1, max_length=200)
    current_verification_request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )


class TaskNodeExecutionFrontier(_Record):
    """按稳定 graph 顺序排列的当前 Task 本地前沿，不执行选择。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    graph_revision: int = Field(ge=1)
    task_status: Literal[
        "proposed",
        "active",
        "awaiting_user",
        "waiting_external",
        "interrupted",
        "blocked",
    ]
    task_state_version: int = Field(ge=1)
    ready_fresh: tuple[ReadyTaskNodeExecutionCandidate, ...] = ()
    recoverable: tuple[RecoverableTaskNodeExecutionCandidate, ...] = ()


class _ExecutionReplanPreflight(_Record):
    subject: TaskNodeSubject
    source_node_alias: str = Field(pattern=r"^base_node_[0-9]{3}$")
    task_state_version: int = Field(ge=1)
    node_state_version: int = Field(ge=1)


class WorkRunBudgetChargeMutationResult(_Record):
    """一笔 Host 活跃时间 charge 的精确持久化投影。"""

    status: Literal["applied", "replayed"]
    budget_charge_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=200)
    work_run_revision: int = Field(ge=2)
    work_run_status: WorkRunStatus
    work_run_reason: str | None = None
    transition: WorkRunBudgetTransition
    window_state_version: int = Field(ge=1)


class WorkExecutionPersistenceError(RuntimeError):
    """某项 WorkRun 持久化不变量以失败关闭方式触发。"""


class WorkExecutionApplyIdCollision(WorkExecutionPersistenceError):
    """某个 apply id 被复用于不同操作或载荷。"""


class WorkExecutionRevisionConflict(WorkExecutionPersistenceError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"work-run revision conflict: expected {expected}, actual {actual}"
        )


class WorkExecutionProgressRevisionConflict(WorkExecutionPersistenceError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Acceptance progress revision conflict: expected {expected}, actual {actual}"
        )


class WorkExecutionOutputRevisionConflict(WorkExecutionPersistenceError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"OutputWindow revision conflict: expected {expected}, actual {actual}"
        )


def create_task_node_work_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_window_revision: int,
    apply_id: str,
    work_run_id: str | None = None,
) -> WorkExecutionMutationResult:
    """创建一个活动 TaskNode WorkRun，并将其绑定到活动 Turn。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("apply_id", apply_id)
    _require_positive("expected_task_state_version", expected_task_state_version)
    _require_positive("expected_node_state_version", expected_node_state_version)
    _require_positive("expected_window_revision", expected_window_revision)
    if not isinstance(subject, TaskNodeSubject):
        raise TypeError("subject must be a TaskNode execution subject")
    if work_run_id is not None:
        _require_identifier("work_run_id", work_run_id)

    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "subject": subject.model_dump(mode="json"),
        "expected_task_state_version": expected_task_state_version,
        "expected_node_state_version": expected_node_state_version,
        "expected_window_revision": expected_window_revision,
        "requested_work_run_id": work_run_id,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="create_work_run",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay

        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if window["current_work_run_id"] is not None:
            raise WorkExecutionPersistenceError(
                "the active Turn already points to a WorkRun"
            )
        task = conn.execute(
            "SELECT session_id, current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        if task is None or str(task["session_id"]) != session_id:
            raise WorkExecutionPersistenceError("Task is outside this Session")
        if task["current_graph_revision"] is None or int(task["current_graph_revision"]) != subject.graph_revision:
            raise WorkExecutionPersistenceError("WorkRun subject is not on the current TaskGraph revision")
        if str(task["current_status"]) not in _STARTABLE_TASK_STATUSES:
            raise WorkExecutionPersistenceError(
                "Task status does not permit starting a WorkRun"
            )
        actual_task_version = int(task["state_version"])
        if actual_task_version != expected_task_state_version:
            raise WorkExecutionRevisionConflict(
                expected=expected_task_state_version,
                actual=actual_task_version,
            )
        node = conn.execute(
            "SELECT node.node_revision, node.acceptance_criteria_json, "
            "state.status, state.state_version "
            "FROM insession_task_graph_nodes AS node "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "AND node.insession_task_node_id=?",
            (subject.task_id, subject.graph_revision, subject.node_id),
        ).fetchone()
        if node is None or int(node["node_revision"]) != subject.node_revision:
            raise WorkExecutionPersistenceError("unknown or stale TaskNode subject")
        if str(node["status"]) not in _STARTABLE_NODE_STATUSES:
            raise WorkExecutionPersistenceError(
                "TaskNode status does not permit starting a WorkRun"
            )
        unfinished_child = conn.execute(
            "SELECT 1 FROM insession_task_graph_edges AS edge "
            "JOIN insession_task_graph_nodes AS child "
            "ON child.insession_task_id=edge.insession_task_id "
            "AND child.graph_revision=edge.graph_revision "
            "AND child.insession_task_node_id=edge.child_insession_task_node_id "
            "LEFT JOIN insession_task_node_states AS child_state "
            "ON child_state.insession_task_id=child.insession_task_id "
            "AND child_state.insession_task_node_id=child.insession_task_node_id "
            "AND child_state.node_revision=child.node_revision "
            "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
            "AND edge.parent_insession_task_node_id=? "
            "AND (child_state.status IS NULL OR child_state.status!='completed') LIMIT 1",
            (subject.task_id, subject.graph_revision, subject.node_id),
        ).fetchone()
        if unfinished_child is not None:
            raise WorkExecutionPersistenceError(
                "TaskNode is not ready because a direct child is incomplete"
            )
        # 仅有 completed 状态不足以构成依赖权威。只有每个直接子节点在这个精确当前
        # graph 版本上都有一个可完整解析的 PASS/冻结 Delivery 后，父节点才能启动。
        _load_current_task_node_dependency_delivery_ids(
            conn,
            session_id=session_id,
            subject=subject,
        )
        actual_node_version = int(node["state_version"])
        if actual_node_version != expected_node_state_version:
            raise WorkExecutionRevisionConflict(
                expected=expected_node_state_version,
                actual=actual_node_version,
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links "
            "WHERE session_id=? AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "the active Turn is not authoritatively linked to this Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE session_id=? AND status='active' LIMIT 1",
            (session_id,),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "the Session already has an active WorkRun"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE subject_kind='task_node' "
            "AND insession_task_id=? AND insession_task_node_id=? AND node_revision=? "
            "AND status NOT IN ('completed', 'failed', 'cancelled') LIMIT 1",
            (
                subject.task_id,
                subject.node_id,
                subject.node_revision,
            ),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "the TaskNode version already has a nonterminal WorkRun"
            )

        acceptance_ids = _acceptance_ids(node["acceptance_criteria_json"])
        allocated_run_id = work_run_id or _allocate_id(deps, "workrun")
        run = create_work_run(work_run_id=allocated_run_id, subject=subject)
        progress = initialize_acceptance_progress(
            work_run_id=allocated_run_id,
            subject=subject,
            acceptance_ids=acceptance_ids,
        )
        output_window = initialize_output_window(
            work_run_id=allocated_run_id,
            updated_turn_id=turn_id,
        )
        execution_subject_id = _ensure_task_node_execution_subject(
            conn,
            session_id=session_id,
            subject=subject,
            created_at=now,
        )
        try:
            conn.execute(
                "INSERT INTO insession_work_runs "
                "(work_run_id, execution_subject_id, session_id, subject_kind, "
                "insession_task_id, graph_revision, "
                "insession_task_node_id, node_revision, status, reason, revision, "
                "max_attempts, soft_active_seconds, hard_active_seconds, attempts_started, "
                "active_seconds_consumed, current_attempt_id, created_turn_id, updated_turn_id, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, 'task_node', ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                (
                    allocated_run_id,
                    execution_subject_id,
                    session_id,
                    subject.task_id,
                    subject.graph_revision,
                    subject.node_id,
                    subject.node_revision,
                    run.revision,
                    run.budget.max_attempts,
                    run.budget.soft_active_seconds,
                    run.budget.hard_active_seconds,
                    run.budget.attempts_started,
                    run.budget.active_seconds_consumed,
                    turn_id,
                    turn_id,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkExecutionPersistenceError(
                "WorkRun creation conflicts with existing execution authority"
            ) from exc
        try:
            create_execution_findings_owner_companion_in_transaction(
                conn,
                session_id=session_id,
                owner_kind="work_run",
                execution_owner_id=allocated_run_id,
                now=now,
            )
        except ExecutionFindingsPersistenceError as exc:
            raise WorkExecutionPersistenceError(
                "WorkRun findings companion could not be created"
            ) from exc
        progress_json = _model_json(progress)
        conn.execute(
            "INSERT INTO insession_work_run_acceptance_progress "
            "(work_run_id, progress_revision, snapshot_hash, snapshot_json, "
            "updated_attempt_id, created_at, updated_at, evaluated_output_revision) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
            (
                allocated_run_id,
                progress.revision,
                _text_hash(progress_json),
                progress_json,
                now,
                now,
                progress.evaluated_output_revision,
            ),
        )
        output_json = _model_json(output_window)
        conn.execute(
            "INSERT INTO insession_work_run_output_windows "
            "(work_run_id, session_id, output_revision, snapshot_hash, snapshot_json, "
            "updated_turn_id, updated_attempt_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                allocated_run_id,
                session_id,
                output_window.output_revision,
                _text_hash(output_json),
                output_json,
                turn_id,
                now,
                now,
            ),
        )
        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'started', ?)",
            (session_id, turn_id, allocated_run_id, link_revision, now),
        )
        if conn.execute(
            "UPDATE insession_task_node_states SET status='active', state_version=state_version+1, "
            "updated_at=? WHERE insession_task_id=? AND insession_task_node_id=? "
            "AND node_revision=? AND state_version=?",
            (
                now,
                subject.task_id,
                subject.node_id,
                subject.node_revision,
                expected_node_state_version,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("TaskNode state changed during WorkRun creation")
        if conn.execute(
            "UPDATE insession_tasks SET current_status='active', state_version=state_version+1, "
            "updated_at=? WHERE insession_task_id=? AND state_version=?",
            (now, subject.task_id, expected_task_state_version),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Task state changed during WorkRun creation")
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, current_attempt_id=NULL, "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' AND state_version=? "
            "AND current_work_run_id IS NULL",
            (
                allocated_run_id,
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Turn Window changed during WorkRun creation")

        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=allocated_run_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="create_work_run",
            session_id=session_id,
            work_run_id=allocated_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def detach_safe_work_run_lane(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_window_revision: int,
    apply_id: str,
) -> WorkExecutionMutationResult:
    """一个持久化 lane 安全停止后释放标量 Turn 游标。

    WorkRun、待处理问题、验证历史、Task 与节点状态均保持不变。只有不存在活动
    Attempt、verification 或待处理 operation，且具有持久化 waiting/limited/failed
    disposition 的 run 才能解除绑定，使同一 Turn 可串行认领另一个 Task lane。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_positive("expected_window_revision", expected_window_revision)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "expected_window_revision": expected_window_revision,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="detach_safe_lane",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        window = _require_owned_work_run_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            expected_window_revision,
        )
        if (
            window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "safe lane detach requires no active Attempt or Operation"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        status = WorkRunStatus(str(run_row["status"]))
        reason = str(run_row["reason"] or "")
        allowed = (
            (status is WorkRunStatus.WAITING_USER and reason == "needs_input")
            or (
                status is WorkRunStatus.WAITING_EXTERNAL
                and reason == "operation_completion_unconfirmed"
            )
            or (
                status is WorkRunStatus.TURN_LIMIT_REACHED
                and reason == "turn_limit_reached"
            )
            or status is WorkRunStatus.FAILED
        )
        if (
            not allowed
            or run_row["current_attempt_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun is not at a detachable durable safe-stop"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=NULL, "
            "current_attempt_id=NULL, latest_checkpoint_id=NULL, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND state_version=? "
            "AND current_work_run_id=? AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL",
            (
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
                work_run_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during safe lane detach"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="detach_safe_lane",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def get_work_run(
    deps: StoreDeps,
    *,
    session_id: str,
    work_run_id: str,
) -> StoredWorkRun:
    _require_identifier("session_id", session_id)
    _require_identifier("work_run_id", work_run_id)
    deps.init_db()
    conn = deps.connect()
    try:
        # _load_record 有意跨越多个规范化表。显式延迟读取事务会在第一次 SELECT 时
        # 固定一个 SQLite 快照；否则自动提交 SELECT 可能与写入方交错，投影出彼此
        # 不可能同时存在的 generation。
        conn.execute("BEGIN")
        record = _load_record(conn, work_run_id, session_id=session_id)
        conn.commit()
        return record
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_active_task_graph_execution_replan_request(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> TaskGraphExecutionReplanRequest | None:
    """加载唯一活动的执行时 TaskGraph revision 权威状态。"""

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        request = _load_authenticated_execution_replan_request(
            conn,
            request_id=None,
            session_id=session_id,
            task_id=task_id,
            require_active=True,
        )
        conn.commit()
        return request
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_turn_linked_work_run_ids(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """按链接顺序返回与此精确 Turn 关联的每个 WorkRun。

    与恢复发现不同，此投影包含终态 WorkRun，且不按 TaskNode 或 AuxiliaryNode
    subject 过滤。它是只读响应重建原语；持久化 Turn--WorkRun 链接仍是成员资格与
    顺序的唯一权威状态。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        turn = conn.execute(
            "SELECT session_id FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise WorkExecutionPersistenceError(
                "Turn is unknown or outside this Session"
            )
        rows = conn.execute(
            "SELECT link.work_run_id FROM insession_work_run_turn_links AS link "
            "JOIN insession_work_runs AS run "
            "ON run.session_id=link.session_id "
            "AND run.work_run_id=link.work_run_id "
            "WHERE link.session_id=? AND link.turn_id=? "
            "ORDER BY link.link_revision, link.link_id",
            (session_id, turn_id),
        ).fetchall()
    return tuple(str(row["work_run_id"]) for row in rows)


def list_turn_linked_nonterminal_work_runs(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[TurnLinkedNonterminalWorkRunCandidate, ...]:
    """列出其 Task 与此 Turn 相连的每个非终态 WorkRun。

    此处唯一的发现权威是从 Turn 到根 Task 的链接。该读取不会绑定、恢复或选择
    WorkRun，也绝不会将搜索扩大到 Session 中的其他 Task。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        turn = conn.execute(
            "SELECT session_id FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise WorkExecutionPersistenceError(
                "Turn is unknown or outside this Session"
            )
        rows = conn.execute(
            "WITH linked_tasks AS ("
            "SELECT insession_task_id, MIN(link_id) AS first_link_id "
            "FROM insession_task_turn_links "
            "WHERE session_id=? AND turn_id=? GROUP BY insession_task_id"
            ") "
            "SELECT run.work_run_id, run.session_id, run.insession_task_id, "
            "run.subject_kind, run.graph_revision, run.insession_task_node_id, "
            "run.auxiliary_graph_id, run.auxiliary_graph_revision, "
            "run.auxiliary_node_id, run.node_revision, "
            "run.status, run.reason, run.revision, run.current_attempt_id, "
            "run.current_verification_request_id, run.created_turn_id, "
            "run.updated_turn_id "
            "FROM linked_tasks AS linked "
            "JOIN insession_work_runs AS run "
            "ON run.insession_task_id=linked.insession_task_id "
            "AND run.session_id=? "
            "WHERE run.status NOT IN ('completed', 'failed', 'cancelled') "
            "ORDER BY linked.first_link_id, run.created_at, run.work_run_id",
            (session_id, turn_id, session_id),
        ).fetchall()
    return tuple(
        TurnLinkedNonterminalWorkRunCandidate(
            session_id=str(row["session_id"]),
            work_run_id=str(row["work_run_id"]),
            subject=_subject_from_run_row(row),
            status=WorkRunStatus(str(row["status"])),
            reason=(str(row["reason"]) if row["reason"] is not None else None),
            work_run_revision=int(row["revision"]),
            current_attempt_id=(
                str(row["current_attempt_id"])
                if row["current_attempt_id"] is not None
                else None
            ),
            current_verification_request_id=(
                str(row["current_verification_request_id"])
                if row["current_verification_request_id"] is not None
                else None
            ),
            created_turn_id=str(row["created_turn_id"]),
            updated_turn_id=str(row["updated_turn_id"]),
        )
        for row in rows
    )


def project_task_node_execution_frontier(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> TaskNodeExecutionFrontier:
    """投影一个 Task 本地 ready/recovery 前沿，但不选择其中任何项。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("task_id", task_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        turn = conn.execute(
            "SELECT session_id FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise WorkExecutionPersistenceError(
                "Turn is unknown or outside this Session"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "Turn is not authoritatively linked to this Task"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if task is None or task["current_graph_revision"] is None:
            raise WorkExecutionPersistenceError(
                "TaskNode frontier requires a current TaskGraph"
            )
        graph_revision = int(task["current_graph_revision"])
        task_status = str(task["current_status"])
        if task_status in {"completed", "cancelled"}:
            raise WorkExecutionPersistenceError(
                "terminal Task has no executable TaskNode frontier"
            )
        nodes = conn.execute(
            "SELECT node.insession_task_node_id, node.node_revision, "
            "node.node_kind, node.ordinal, state.status, state.state_version "
            "FROM insession_task_graph_nodes AS node "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "ORDER BY node.ordinal, node.insession_task_node_id",
            (task_id, graph_revision),
        ).fetchall()
        roots = tuple(row for row in nodes if str(row["node_kind"]) == "root")
        if (
            not nodes
            or len(roots) != 1
            or str(roots[0]["insession_task_node_id"]) != task_id
        ):
            raise WorkExecutionPersistenceError(
                "TaskNode frontier requires one canonical root"
            )
        node_by_id = {
            str(row["insession_task_node_id"]): row for row in nodes
        }
        if len(node_by_id) != len(nodes):
            raise WorkExecutionPersistenceError(
                "TaskNode frontier contains duplicate node identities"
            )
        children_by_parent: dict[str, list[str]] = {
            node_id: [] for node_id in node_by_id
        }
        for edge in conn.execute(
            "SELECT parent_insession_task_node_id, child_insession_task_node_id "
            "FROM insession_task_graph_edges WHERE insession_task_id=? "
            "AND graph_revision=? ORDER BY ordinal, child_insession_task_node_id",
            (task_id, graph_revision),
        ).fetchall():
            parent_id = str(edge["parent_insession_task_node_id"])
            child_id = str(edge["child_insession_task_node_id"])
            if parent_id not in node_by_id or child_id not in node_by_id:
                raise WorkExecutionPersistenceError(
                    "TaskNode frontier graph edge is corrupt"
                )
            children_by_parent[parent_id].append(child_id)

        completed_delivery_by_node: dict[str, str] = {}
        for node_id, node in node_by_id.items():
            if str(node["status"]) == "completed":
                completed_delivery_by_node[node_id] = (
                    _load_current_task_node_delivery_id(
                        conn,
                        session_id=session_id,
                        task_id=task_id,
                        graph_revision=graph_revision,
                        node_id=node_id,
                        node_revision=int(node["node_revision"]),
                    )
                )

        run_rows = conn.execute(
            "SELECT * FROM insession_work_runs WHERE insession_task_id=? "
            "AND status NOT IN ('completed', 'failed', 'cancelled') "
            "ORDER BY created_at, work_run_id",
            (task_id,),
        ).fetchall()
        run_by_node: dict[str, sqlite3.Row] = {}
        for run in run_rows:
            if str(run["subject_kind"]) != "task_node":
                raise WorkExecutionPersistenceError(
                    "TaskNode frontier conflicts with a non-TaskNode WorkRun"
                )
            node_id = str(run["insession_task_node_id"])
            node = node_by_id.get(node_id)
            if (
                node is None
                or int(run["graph_revision"]) != graph_revision
                or int(run["node_revision"]) != int(node["node_revision"])
                or node_id in run_by_node
            ):
                raise WorkExecutionPersistenceError(
                    "TaskNode frontier WorkRun authority is stale or ambiguous"
                )
            _load_record(conn, str(run["work_run_id"]), session_id=session_id)
            run_by_node[node_id] = run

        ready: list[ReadyTaskNodeExecutionCandidate] = []
        recoverable: list[RecoverableTaskNodeExecutionCandidate] = []
        for node in nodes:
            node_id = str(node["insession_task_node_id"])
            node_status = str(node["status"])
            subject = TaskNodeSubject(
                task_id=task_id,
                graph_revision=graph_revision,
                node_id=node_id,
                node_revision=int(node["node_revision"]),
            )
            run = run_by_node.get(node_id)
            if run is not None:
                if node_status not in {
                    "active",
                    "awaiting_user",
                    "waiting_external",
                    "interrupted",
                }:
                    raise WorkExecutionPersistenceError(
                        "recoverable WorkRun disagrees with TaskNode state"
                    )
                recoverable.append(
                    RecoverableTaskNodeExecutionCandidate(
                        subject=subject,
                        ordinal=int(node["ordinal"]),
                        node_status=node_status,
                        node_state_version=int(node["state_version"]),
                        work_run_id=str(run["work_run_id"]),
                        work_run_status=WorkRunStatus(str(run["status"])),
                        work_run_reason=(
                            str(run["reason"])
                            if run["reason"] is not None
                            else None
                        ),
                        work_run_revision=int(run["revision"]),
                        current_attempt_id=(
                            str(run["current_attempt_id"])
                            if run["current_attempt_id"] is not None
                            else None
                        ),
                        current_verification_request_id=(
                            str(run["current_verification_request_id"])
                            if run["current_verification_request_id"] is not None
                            else None
                        ),
                    )
                )
                continue
            if node_status not in _STARTABLE_NODE_STATUSES:
                continue
            child_ids = children_by_parent[node_id]
            if not all(
                child_id in completed_delivery_by_node for child_id in child_ids
            ):
                continue
            dependencies = _load_current_task_node_dependency_delivery_ids(
                conn,
                session_id=session_id,
                subject=subject,
            )
            ready.append(
                ReadyTaskNodeExecutionCandidate(
                    subject=subject,
                    ordinal=int(node["ordinal"]),
                    node_status=node_status,
                    node_state_version=int(node["state_version"]),
                    dependency_delivery_ids=dependencies,
                )
            )

        projection = TaskNodeExecutionFrontier(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            graph_revision=graph_revision,
            task_status=task_status,
            task_state_version=int(task["state_version"]),
            ready_fresh=tuple(ready),
            recoverable=tuple(recoverable),
        )
        conn.commit()
        return projection
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_current_task_node_dependency_deliveries(
    deps: StoreDeps,
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> tuple[ResolvedCurrentTaskNodeDelivery, ...]:
    """使用直接/carry 当前权威状态解析直接子节点正文。"""

    _require_identifier("session_id", session_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task = conn.execute(
            "SELECT current_graph_revision, current_status FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, subject.task_id),
        ).fetchone()
        if (
            task is None
            or task["current_graph_revision"] is None
            or int(task["current_graph_revision"]) != subject.graph_revision
            or str(task["current_status"]) in {"completed", "cancelled"}
        ):
            raise WorkExecutionPersistenceError(
                "TaskNode dependency projection requires the current nonterminal TaskGraph"
            )
        node = conn.execute(
            "SELECT node.node_revision, state.status FROM insession_task_graph_nodes AS node "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "AND node.insession_task_node_id=?",
            (subject.task_id, subject.graph_revision, subject.node_id),
        ).fetchone()
        if node is None or int(node["node_revision"]) != subject.node_revision:
            raise WorkExecutionPersistenceError(
                "TaskNode dependency projection subject is stale"
            )
        dependencies = _load_current_task_node_dependency_deliveries(
            conn,
            session_id=session_id,
            subject=subject,
        )
        conn.commit()
        return dependencies
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_completed_task_final_delivery_id(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> str:
    """解析一个已完成当前 Task 的规范根 Delivery。"""

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task = conn.execute(
            "SELECT current_graph_revision, current_status FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if (
            task is None
            or task["current_graph_revision"] is None
            or str(task["current_status"]) != "completed"
        ):
            raise WorkExecutionPersistenceError(
                "formal Task Delivery requires a completed current TaskGraph"
            )
        graph_revision = int(task["current_graph_revision"])
        roots = conn.execute(
            "SELECT insession_task_node_id, node_revision "
            "FROM insession_task_graph_nodes WHERE insession_task_id=? "
            "AND graph_revision=? AND node_kind='root'",
            (task_id, graph_revision),
        ).fetchall()
        if len(roots) != 1 or str(roots[0]["insession_task_node_id"]) != task_id:
            raise WorkExecutionPersistenceError(
                "formal Task Delivery requires one canonical root TaskNode"
            )
        delivery_id = _load_current_task_node_delivery_id(
            conn,
            session_id=session_id,
            task_id=task_id,
            graph_revision=graph_revision,
            node_id=task_id,
            node_revision=int(roots[0]["node_revision"]),
        )
        conn.commit()
        return delivery_id
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_turn_completed_verified_delivery_ids(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """发现由某个精确已接受 Turn 创建的每个已验证 Delivery。

    这是只读恢复投影，而不是选择器。候选项发现会使用持久化权威边界的两端：声称
    属于此 Turn 的 Delivery，以及链接到 Turn 且声称已在此 Turn 完成验证的
    WorkRun。联合考察两端会使缺失链接或缺失 Delivery 以失败关闭方式暴露，而不是
    看起来像正常的空结果。

    每个候选项都通过同一 SQLite 读取快照中的现有完整 NodeDelivery 加载器解析。
    只有稳定引用 ID 会逸出；冻结 OutputWindow 正文仍由其唯一持久化权威持有。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        turn = conn.execute(
            "SELECT session_id FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise WorkExecutionPersistenceError(
                "Turn is unknown or outside this Session"
            )

        rows = conn.execute(
            "SELECT run.work_run_id, run.status, run.reason, "
            "run.current_attempt_id, run.current_verification_request_id, "
            "run.updated_turn_id, link.link_revision, "
            "delivery.delivery_id, delivery.created_turn_id "
            "FROM insession_work_runs AS run "
            "LEFT JOIN insession_work_run_turn_links AS link "
            "ON link.session_id=run.session_id "
            "AND link.work_run_id=run.work_run_id AND link.turn_id=? "
            "LEFT JOIN insession_task_node_deliveries AS delivery "
            "ON delivery.session_id=run.session_id "
            "AND delivery.work_run_id=run.work_run_id "
            "AND delivery.created_turn_id=? "
            "WHERE run.session_id=? AND run.subject_kind='task_node' AND ("
            "delivery.delivery_id IS NOT NULL OR ("
            "link.link_id IS NOT NULL AND run.updated_turn_id=? "
            "AND run.status='completed' "
            "AND run.reason='verification_passed')) "
            "ORDER BY CASE WHEN link.link_revision IS NULL THEN 1 ELSE 0 END, "
            "link.link_revision, run.created_at, run.work_run_id",
            (turn_id, turn_id, session_id, turn_id),
        ).fetchall()

        from .work_verification import _load_task_node_delivery

        delivery_ids: list[str] = []
        seen_ids: set[str] = set()
        for row in rows:
            delivery_id = (
                str(row["delivery_id"])
                if row["delivery_id"] is not None
                else None
            )
            if (
                row["link_revision"] is None
                or str(row["status"]) != WorkRunStatus.COMPLETED.value
                or str(row["reason"] or "") != "verification_passed"
                or row["current_attempt_id"] is not None
                or row["current_verification_request_id"] is not None
                or str(row["updated_turn_id"]) != turn_id
                or delivery_id is None
                or str(row["created_turn_id"] or "") != turn_id
            ):
                raise WorkExecutionPersistenceError(
                    "Turn verified Delivery discovery authority is corrupt"
                )
            resolved = _load_task_node_delivery(
                conn,
                session_id=session_id,
                delivery_id=delivery_id,
            )
            if (
                resolved.delivery.delivery_id != delivery_id
                or resolved.delivery.work_run_id != str(row["work_run_id"])
                or resolved.delivery.created_turn_id != turn_id
                or delivery_id in seen_ids
            ):
                raise WorkExecutionPersistenceError(
                    "Turn verified Delivery projection is corrupt"
                )
            seen_ids.add(delivery_id)
            delivery_ids.append(delivery_id)
        conn.commit()
        return tuple(delivery_ids)
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_pending_user_questions(
    deps: StoreDeps,
    *,
    session_id: str,
) -> tuple[PendingUserQuestion, ...]:
    """推导每个当前问题，而不创建第二份权威状态。

    问题正文仍由精确且已关闭的 ``request_user_input`` AttemptDecision 持有。此投影
    适用于持久化弹窗，并携带足够的 revision 事实，供后续 Task 范围 controller
    请求原子延续命令。
    """

    _require_identifier("session_id", session_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        if conn.execute(
            "SELECT 1 FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError("unknown Session")
        rows = conn.execute(
            "SELECT run.work_run_id FROM insession_work_runs AS run "
            "JOIN insession_execution_subjects AS registry "
            "ON registry.execution_subject_id=run.execution_subject_id "
            "AND registry.session_id=run.session_id "
            "AND registry.insession_task_id=run.insession_task_id "
            "WHERE run.session_id=? AND run.status='waiting_user' "
            "AND registry.subject_kind='task_node' "
            "AND registry.subject_contract_version='task_node_v1' "
            "ORDER BY run.updated_at, run.work_run_id",
            (session_id,),
        ).fetchall()
        pending: list[PendingUserQuestion] = []
        for row in rows:
            work_run_id = str(row["work_run_id"])
            record = _load_record(
                conn,
                work_run_id,
                session_id=session_id,
            )
            if (
                record.work_run.status is not WorkRunStatus.WAITING_USER
                or record.work_run.reason != "needs_input"
                or record.current_attempt_id is not None
                or record.current_verification_request_id is not None
                or not record.attempts
            ):
                raise WorkExecutionPersistenceError(
                    "waiting_user WorkRun projection is inconsistent"
                )
            question_attempt = record.attempts[-1]
            decision = question_attempt.decision
            if (
                question_attempt.attempt.status is not AttemptStatus.CLOSED
                or question_attempt.action != "request_user_input"
                or decision is None
                or not isinstance(decision.action, RequestUserInputAction)
                or question_attempt.attempt.ordinal
                != record.work_run.budget.attempts_started
            ):
                raise WorkExecutionPersistenceError(
                    "waiting_user WorkRun does not end at its question Attempt"
                )
            if conn.execute(
                "SELECT 1 FROM insession_work_run_attempts "
                "WHERE predecessor_question_attempt_id=? LIMIT 1",
                (question_attempt.attempt.attempt_id,),
            ).fetchone() is not None:
                raise WorkExecutionPersistenceError(
                    "waiting_user question was already consumed"
                )
            subject = record.work_run.subject
            if not isinstance(subject, TaskNodeSubject):
                raise WorkExecutionPersistenceError(
                    "Task continuation query returned a non-TaskNode WorkRun"
                )
            task = conn.execute(
                "SELECT current_graph_revision, current_status, state_version "
                "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
                (session_id, subject.task_id),
            ).fetchone()
            node = conn.execute(
                "SELECT state.status, state.state_version "
                "FROM insession_task_graph_nodes AS node "
                "JOIN insession_task_node_states AS state "
                "ON state.insession_task_id=node.insession_task_id "
                "AND state.insession_task_node_id=node.insession_task_node_id "
                "AND state.node_revision=node.node_revision "
                "WHERE node.insession_task_id=? AND node.graph_revision=? "
                "AND node.insession_task_node_id=? AND node.node_revision=?",
                (
                    subject.task_id,
                    subject.graph_revision,
                    subject.node_id,
                    subject.node_revision,
                ),
            ).fetchone()
            valid_authority = (
                task is not None
                and task["current_graph_revision"] is not None
                and int(task["current_graph_revision"])
                == subject.graph_revision
                and str(task["current_status"])
                in {"active", "awaiting_user"}
                and node is not None
                and str(node["status"]) == "awaiting_user"
            )
            if not valid_authority:
                raise WorkExecutionPersistenceError(
                    "pending user question is detached from current execution authority"
                )
            assert task is not None and node is not None
            pending.append(
                PendingUserQuestion(
                    session_id=session_id,
                    work_run_id=work_run_id,
                    work_run_revision=record.work_run.revision,
                    subject=subject,
                    question_attempt_id=question_attempt.attempt.attempt_id,
                    question_attempt_ordinal=question_attempt.attempt.ordinal,
                    question_turn_id=question_attempt.turn_id,
                    question=decision.action.question,
                    task_state_version=int(task["state_version"]),
                    node_state_version=int(node["state_version"]),
                    acceptance_progress_revision=(
                        record.acceptance_progress.revision
                    ),
                    output_window_revision=record.output_window.output_revision,
                )
            )
        conn.commit()
        return tuple(pending)
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def continue_waiting_user_work_run_and_start_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    subject: TaskNodeSubject,
    question_attempt_id: str,
    expected_work_run_revision: int,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    apply_id: str,
    catalog_snapshot: Mapping[str, Any],
    attempt_id: str | None = None,
) -> WorkExecutionMutationResult:
    """消费一个精确待处理问题，并以原子方式创建 Attempt N+1。"""

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("question_attempt_id", question_attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_task_state_version", expected_task_state_version)
    _require_positive("expected_node_state_version", expected_node_state_version)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if not isinstance(subject, TaskNodeSubject):
        raise TypeError("subject must be a TaskNode execution subject")
    if attempt_id is not None:
        _require_identifier("attempt_id", attempt_id)
    catalog_json = _canonical_json(dict(catalog_snapshot))
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "subject": subject.model_dump(mode="json"),
        "question_attempt_id": question_attempt_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_task_state_version": expected_task_state_version,
        "expected_node_state_version": expected_node_state_version,
        "expected_progress_revision": expected_progress_revision,
        "expected_window_revision": expected_window_revision,
        "catalog_snapshot": json.loads(catalog_json),
        "requested_attempt_id": attempt_id,
    }
    payload_hash = _payload_hash(payload)
    operation = "continue_waiting_user_and_start_attempt"
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation=operation,
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            if replay.attempt is None:
                raise WorkExecutionPersistenceError(
                    "waiting-user continuation receipt lost its new Attempt"
                )
            stored = conn.execute(
                "SELECT input_turn_id, predecessor_question_attempt_id "
                "FROM insession_work_run_attempts "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, replay.attempt.attempt_id),
            ).fetchone()
            link = conn.execute(
                "SELECT link_revision, relation FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id=?",
                (turn_id, work_run_id),
            ).fetchone()
            if (
                stored is None
                or str(stored["input_turn_id"]) != turn_id
                or str(stored["predecessor_question_attempt_id"] or "")
                != question_attempt_id
                or link is None
                or str(link["relation"]) != "continued"
                or replay.turn_work_run_link_revision
                != int(link["link_revision"])
            ):
                raise WorkExecutionPersistenceError(
                    "waiting-user continuation receipt lost its durable bindings"
                )
            return replay

        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if (
            window["current_work_run_id"] is not None
            or window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "waiting-user continuation requires an unbound active Turn Window"
            )

        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        stored_subject = _subject_from_run_row(run_row)
        if stored_subject != subject:
            raise WorkExecutionPersistenceError(
                "waiting-user continuation subject is stale"
            )
        if (
            str(run_row["status"]) != WorkRunStatus.WAITING_USER.value
            or str(run_row["reason"] or "") != "needs_input"
            or run_row["current_attempt_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun does not own one pending user question"
            )

        current_budget = _budget_from_row(run_row)
        attempts_started = _require_attempt_row_aggregate(
            conn,
            work_run_id=work_run_id,
            expected_attempts_started=current_budget.attempts_started,
        )
        if (
            attempts_started >= current_budget.max_attempts
            or attempts_started >= _MAX_ATTEMPTS
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun Attempt budget is exhausted"
            )
        if (
            current_budget.active_seconds_consumed
            >= current_budget.hard_active_seconds
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun hard active-time budget is exhausted"
            )
        if (
            current_budget.active_seconds_consumed
            >= current_budget.soft_active_seconds
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun soft active-time budget forbids a new Attempt"
            )

        question_attempt = conn.execute(
            "SELECT * FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=?",
            (work_run_id, question_attempt_id),
        ).fetchone()
        if (
            question_attempt is None
            or int(question_attempt["ordinal"]) != attempts_started
            or str(question_attempt["status"]) != AttemptStatus.CLOSED.value
            or str(question_attempt["action"] or "") != "request_user_input"
            or str(question_attempt["close_reason"] or "")
            != "request_user_input"
            or question_attempt["decision_json"] is None
            or str(run_row["updated_turn_id"])
            != str(question_attempt["turn_id"])
        ):
            raise WorkExecutionPersistenceError(
                "question Attempt is not the exact current pending interaction"
            )
        try:
            question_decision = HostAcceptedAttemptDecision.model_validate_json(
                str(question_attempt["decision_json"])
            )
            question_budget_after = WorkRunBudget.model_validate_json(
                str(question_attempt["budget_after_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "pending question Attempt payload is invalid"
            ) from exc
        if (
            not isinstance(
                question_decision.action,
                RequestUserInputAction,
            )
            or question_budget_after != current_budget
        ):
            raise WorkExecutionPersistenceError(
                "pending question Attempt disagrees with WorkRun authority"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_attempts "
            "WHERE predecessor_question_attempt_id=? LIMIT 1",
            (question_attempt_id,),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "pending user question was already consumed"
            )

        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        output_window, output_hash = _load_output_window(conn, work_run_id)
        if progress.evaluated_output_revision != output_window.output_revision:
            raise WorkExecutionPersistenceError(
                "Acceptance progress is not bound to the current OutputWindow"
            )

        _require_waiting_user_continuation_authority(
            conn,
            session_id=session_id,
            subject=subject,
            expected_task_state_version=expected_task_state_version,
            expected_node_state_version=expected_node_state_version,
        )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "continuation Turn is not authoritatively linked to this Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (turn_id, work_run_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "continuation Turn already has a WorkRun link without this receipt"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE session_id=? "
            "AND status='active' LIMIT 1",
            (session_id,),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "the Session already has an active WorkRun"
            )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'continued', ?)",
            (session_id, turn_id, work_run_id, link_revision, now),
        )
        allocated_attempt_id = attempt_id or _allocate_id(deps, "attempt")
        ordinal = attempts_started + 1
        attempt = Attempt(
            attempt_id=allocated_attempt_id,
            work_run_id=work_run_id,
            ordinal=ordinal,
        )
        try:
            conn.execute(
                "INSERT INTO insession_work_run_attempts "
                "(attempt_id, work_run_id, turn_id, input_turn_id, "
                "predecessor_question_attempt_id, ordinal, status, action, "
                "decision_json, progress_revision_before, progress_revision_after, "
                "input_output_revision, input_output_hash, "
                "committed_output_revision, submitted_output_revision, "
                "input_checkpoint_id, input_verification_request_id, "
                "catalog_snapshot_json, catalog_snapshot_hash, budget_before_json, "
                "budget_after_json, close_reason, created_at, closed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', NULL, NULL, ?, NULL, ?, ?, "
                "NULL, NULL, ?, NULL, ?, ?, ?, NULL, NULL, ?, NULL)",
                (
                    allocated_attempt_id,
                    work_run_id,
                    turn_id,
                    turn_id,
                    question_attempt_id,
                    ordinal,
                    progress.revision,
                    output_window.output_revision,
                    output_hash,
                    question_attempt_id,
                    catalog_json,
                    _text_hash(catalog_json),
                    _model_json(current_budget),
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkExecutionPersistenceError(
                "waiting-user continuation Attempt conflicts with stored authority"
            ) from exc

        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET status='active', reason=NULL, "
            "revision=?, attempts_started=attempts_started+1, "
            "current_attempt_id=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND session_id=? AND revision=? "
            "AND status='waiting_user' AND reason='needs_input' "
            "AND current_attempt_id IS NULL "
            "AND current_verification_request_id IS NULL "
            "AND attempts_started=? AND attempts_started < max_attempts",
            (
                next_run_revision,
                allocated_attempt_id,
                turn_id,
                now,
                work_run_id,
                session_id,
                expected_work_run_revision,
                attempts_started,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during waiting-user continuation"
            )
        next_task_state_version, next_node_state_version = (
            _activate_waiting_execution_subject(
                conn,
                session_id=session_id,
                subject=subject,
                expected_task_state_version=expected_task_state_version,
                expected_node_state_version=expected_node_state_version,
                now=now,
            )
        )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=?, stage='L2_PLAN', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                work_run_id,
                allocated_attempt_id,
                question_attempt_id,
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during waiting-user continuation"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=allocated_attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        ).model_copy(
            update={
                "task_state_version": next_task_state_version,
                "node_state_version": next_node_state_version,
            }
        )
        if result.attempt != attempt:
            raise WorkExecutionPersistenceError(
                "waiting-user continuation Attempt projection is inconsistent"
            )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation=operation,
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def resume_active_work_run_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    expected_work_run_revision: int,
    expected_window_revision: int,
    apply_id: str,
) -> WorkExecutionMutationResult:
    """在前一 Turn 结算后重新绑定一个未决的活动 Attempt。

    这一狭窄的恢复命令绝不会创建新的语义 Attempt，也不会重建部分执行的决策。
    前一 Turn 必须已持久化为 ``incomplete``；Attempt 仍不得有 decision、call 或
    result；新的 running Turn 必须已链接到同一 Task。旧进程中任何未提交的
    active-time 尾段有意不计费。此 receipt 提交后，Runtime 会启动新的 Host 计时器。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("attempt_id", attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_window_revision": expected_window_revision,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            link = conn.execute(
                "SELECT link_revision, relation FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id=?",
                (turn_id, work_run_id),
            ).fetchone()
            attempt_owner = conn.execute(
                "SELECT ordinal FROM insession_work_run_attempts "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, attempt_id),
            ).fetchone()
            if (
                replay.attempt is None
                or replay.attempt.attempt_id != attempt_id
                or replay.attempt.work_run_id != work_run_id
                or attempt_owner is None
                or replay.attempt.ordinal != int(attempt_owner["ordinal"])
                or link is None
                or str(link["relation"]) != "continued"
                or replay.turn_work_run_link_revision
                != int(link["link_revision"])
            ):
                raise WorkExecutionPersistenceError(
                    "Attempt resume receipt lost its durable Turn link"
                )
            return replay

        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if (
            window["current_work_run_id"] is not None
            or window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "Attempt resume requires an unbound active Turn Window"
            )

        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if (
            str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or run_row["reason"] is not None
            or str(run_row["current_attempt_id"] or "") != attempt_id
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun does not own one resumable active Attempt"
            )
        budget = _budget_from_row(run_row)
        if budget.active_seconds_consumed >= budget.soft_active_seconds:
            raise WorkExecutionPersistenceError(
                "WorkRun active-time budget forbids Attempt resume"
            )

        attempt = _require_active_attempt(conn, work_run_id, attempt_id)
        previous_turn_id = str(attempt["turn_id"])
        if previous_turn_id == turn_id:
            raise WorkExecutionPersistenceError(
                "Attempt resume requires a different Turn"
            )
        undecided_fields = (
            "action",
            "decision_json",
            "progress_revision_after",
            "committed_output_revision",
            "submitted_output_revision",
            "budget_after_json",
            "budget_charge_id",
            "close_reason",
            "closed_at",
        )
        if any(attempt[field] is not None for field in undecided_fields):
            raise WorkExecutionPersistenceError(
                "only an undecided active Attempt can resume on another Turn"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_tool_calls "
            "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
            (work_run_id, attempt_id),
        ).fetchone() is not None or conn.execute(
            "SELECT 1 FROM insession_work_run_tool_results "
            "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
            (work_run_id, attempt_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "Attempt with durable tool activity cannot use undecided resume"
            )

        previous_turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (previous_turn_id,),
        ).fetchone()
        if (
            previous_turn is None
            or str(previous_turn["session_id"]) != session_id
            or str(previous_turn["status"]) != "incomplete"
        ):
            raise WorkExecutionPersistenceError(
                "previous Attempt Turn has not durably settled as incomplete"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=? LIMIT 1",
            (previous_turn_id, work_run_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "Attempt lost its previous Turn ownership link"
            )

        subject = _subject_from_run_row(run_row)
        _require_active_attempt_resume_authority(
            conn,
            session_id=session_id,
            subject=subject,
            run_row=run_row,
        )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "resume Turn is not authoritatively linked to this Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (turn_id, work_run_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "resume Turn already has a WorkRun link without this receipt"
            )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'continued', ?)",
            (session_id, turn_id, work_run_id, link_revision, now),
        )
        if conn.execute(
            "UPDATE insession_work_run_attempts SET turn_id=? "
            "WHERE work_run_id=? AND attempt_id=? AND turn_id=? "
            "AND status='active' AND action IS NULL AND decision_json IS NULL "
            "AND progress_revision_after IS NULL",
            (turn_id, work_run_id, attempt_id, previous_turn_id),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Attempt changed during Turn resume"
            )
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND status='active' AND reason IS NULL "
            "AND current_attempt_id=? AND current_verification_request_id IS NULL",
            (
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                attempt_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during Attempt resume"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=?, stage='L2_PLAN', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                work_run_id,
                attempt_id,
                attempt["input_checkpoint_id"],
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during Attempt resume"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def resume_decided_readonly_tool_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    expected_work_run_revision: int,
    expected_window_revision: int,
    catalog_snapshot: Mapping[str, Any],
    apply_id: str,
    allow_protected_recovery: bool = False,
) -> WorkExecutionMutationResult:
    """将一个精确且经 bridge 授权的 ToolCall 批次重新绑定到新 Turn。

    此操作有意独立于未决 Attempt 恢复。模型决策和每个已物化 ToolCall 都已存在且
    保持不可变。现有最终 ToolResult 会保留；事务结束后 Runtime 只能执行缺失调用。
    除非所属 bridge 显式提供 ``allow_protected_recovery``，否则受保护调用仍会被
    拒绝；该 bridge 必须持有禁止自动重发的 Operation 账本。
    旧进程中未提交的 active-time 尾段会被丢弃，新 Turn 则启动新的计时器。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("attempt_id", attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if not isinstance(allow_protected_recovery, bool):
        raise WorkExecutionPersistenceError(
            "allow_protected_recovery must be boolean"
        )
    try:
        catalog_json = _canonical_json(dict(catalog_snapshot))
        normalized_catalog = json.loads(catalog_json)
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "decided tool recovery catalog snapshot is not canonical JSON"
        ) from exc
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_window_revision": expected_window_revision,
        "catalog_snapshot": normalized_catalog,
        "allow_protected_recovery": allow_protected_recovery,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            link = conn.execute(
                "SELECT link_revision, relation FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id=?",
                (turn_id, work_run_id),
            ).fetchone()
            attempt_owner = conn.execute(
                "SELECT turn_id, ordinal, status, action, decision_json "
                "FROM insession_work_run_attempts "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, attempt_id),
            ).fetchone()
            if (
                replay.attempt is None
                or replay.attempt.attempt_id != attempt_id
                or replay.attempt.work_run_id != work_run_id
                or attempt_owner is None
                or str(attempt_owner["turn_id"]) != turn_id
                or str(attempt_owner["action"] or "") != "call_tools"
                or attempt_owner["decision_json"] is None
                or replay.attempt.ordinal != int(attempt_owner["ordinal"])
                or link is None
                or str(link["relation"]) != "continued"
                or replay.turn_work_run_link_revision
                != int(link["link_revision"])
            ):
                raise WorkExecutionPersistenceError(
                    "decided tool resume receipt lost its durable Turn binding"
                )
            return replay

        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if (
            window["current_work_run_id"] is not None
            or window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "decided tool resume requires an unbound active Turn Window"
            )

        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if (
            str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or run_row["reason"] is not None
            or str(run_row["current_attempt_id"] or "") != attempt_id
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun does not own one recoverable decided tool Attempt"
            )
        budget = _budget_from_row(run_row)
        if budget.active_seconds_consumed >= budget.soft_active_seconds:
            raise WorkExecutionPersistenceError(
                "WorkRun active-time budget forbids decided tool resume"
            )

        attempt = _require_active_attempt(conn, work_run_id, attempt_id)
        previous_turn_id = str(attempt["turn_id"])
        if previous_turn_id == turn_id:
            raise WorkExecutionPersistenceError(
                "decided tool resume requires a different Turn"
            )
        if (
            str(attempt["action"] or "") != "call_tools"
            or attempt["decision_json"] is None
            or attempt["progress_revision_after"] is None
            or attempt["committed_output_revision"] is not None
            or attempt["submitted_output_revision"] is not None
            or attempt["budget_after_json"] is not None
            or attempt["budget_charge_id"] is not None
            or attempt["close_reason"] is not None
            or attempt["closed_at"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "Attempt is not one active decided call_tools batch"
            )
        stored_catalog_json = str(attempt["catalog_snapshot_json"])
        if (
            _text_hash(stored_catalog_json)
            != str(attempt["catalog_snapshot_hash"])
            or stored_catalog_json != catalog_json
        ):
            raise WorkExecutionPersistenceError(
                "decided tool resume catalog snapshot drifted"
            )
        try:
            decision = HostAcceptedAttemptDecision.model_validate_json(
                str(attempt["decision_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "decided tool recovery decision is corrupt"
            ) from exc
        if not isinstance(decision.action, HostMaterializedCallToolsAction):
            raise WorkExecutionPersistenceError(
                "decided tool recovery requires a materialized call_tools decision"
            )
        _require_readonly_recovery_batch(
            conn,
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            decision=decision,
            catalog_snapshot=normalized_catalog,
            allow_protected_recovery=allow_protected_recovery,
        )

        previous_turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (previous_turn_id,),
        ).fetchone()
        if (
            previous_turn is None
            or str(previous_turn["session_id"]) != session_id
            or str(previous_turn["status"]) != "incomplete"
        ):
            raise WorkExecutionPersistenceError(
                "previous decided-tool Turn has not durably settled as incomplete"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=? LIMIT 1",
            (previous_turn_id, work_run_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "decided tool Attempt lost its previous Turn ownership link"
            )

        subject = _subject_from_run_row(run_row)
        _require_active_attempt_resume_authority(
            conn,
            session_id=session_id,
            subject=subject,
            run_row=run_row,
        )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "decided tool resume Turn is not linked to this Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (turn_id, work_run_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "decided tool resume Turn already has a WorkRun link"
            )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'continued', ?)",
            (session_id, turn_id, work_run_id, link_revision, now),
        )
        if conn.execute(
            "UPDATE insession_work_run_attempts SET turn_id=? "
            "WHERE work_run_id=? AND attempt_id=? AND turn_id=? "
            "AND status='active' AND action='call_tools' "
            "AND decision_json IS NOT NULL AND progress_revision_after IS NOT NULL "
            "AND budget_after_json IS NULL AND budget_charge_id IS NULL "
            "AND close_reason IS NULL AND closed_at IS NULL",
            (turn_id, work_run_id, attempt_id, previous_turn_id),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "decided tool Attempt changed during Turn resume"
            )
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND status='active' AND reason IS NULL "
            "AND current_attempt_id=? AND current_verification_request_id IS NULL",
            (
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                attempt_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during decided tool resume"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=?, stage='TOOL', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                work_run_id,
                attempt_id,
                attempt["input_checkpoint_id"],
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during decided tool resume"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def resume_idle_readonly_work_run_and_start_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    closed_attempt_id: str,
    next_attempt_id: str,
    expected_work_run_revision: int,
    expected_window_revision: int,
    catalog_snapshot: Mapping[str, Any],
    apply_id: str,
    allow_protected_recovery: bool = False,
) -> WorkExecutionMutationResult:
    """在获授权工具关闭跨进程丢失后幸存时，以原子方式继续。"""

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("closed_attempt_id", closed_attempt_id)
    _require_identifier("next_attempt_id", next_attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if not isinstance(allow_protected_recovery, bool):
        raise WorkExecutionPersistenceError(
            "allow_protected_recovery must be boolean"
        )
    try:
        catalog_json = _canonical_json(dict(catalog_snapshot))
        normalized_catalog = json.loads(catalog_json)
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "idle readonly recovery catalog snapshot is not canonical JSON"
        ) from exc
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "closed_attempt_id": closed_attempt_id,
        "next_attempt_id": next_attempt_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_window_revision": expected_window_revision,
        "catalog_snapshot": normalized_catalog,
        "allow_protected_recovery": allow_protected_recovery,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            link = conn.execute(
                "SELECT link_revision, relation FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id=?",
                (turn_id, work_run_id),
            ).fetchone()
            attempt_owner = conn.execute(
                "SELECT turn_id, input_turn_id, ordinal FROM "
                "insession_work_run_attempts WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, next_attempt_id),
            ).fetchone()
            if (
                replay.attempt is None
                or replay.attempt.attempt_id != next_attempt_id
                or replay.current_attempt_id != next_attempt_id
                or attempt_owner is None
                or str(attempt_owner["turn_id"]) != turn_id
                or str(attempt_owner["input_turn_id"]) != turn_id
                or replay.attempt.ordinal != int(attempt_owner["ordinal"])
                or link is None
                or str(link["relation"]) != "continued"
                or replay.turn_work_run_link_revision
                != int(link["link_revision"])
            ):
                raise WorkExecutionPersistenceError(
                    "idle readonly resume receipt lost its next-Attempt binding"
                )
            _load_record(conn, work_run_id, session_id=session_id)
            return replay

        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if (
            window["current_work_run_id"] is not None
            or window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "idle readonly resume requires an unbound active Turn Window"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        validated = _load_record(conn, work_run_id, session_id=session_id)
        if (
            validated.work_run.revision != expected_work_run_revision
            or validated.current_attempt_id is not None
            or str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or run_row["reason"] is not None
            or run_row["current_attempt_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun is not one recoverable active-idle readonly checkpoint"
            )
        budget = _budget_from_row(run_row)
        attempts_started = int(run_row["attempts_started"])
        if (
            budget.active_seconds_consumed >= budget.soft_active_seconds
            or attempts_started >= int(run_row["max_attempts"])
            or attempts_started >= _MAX_ATTEMPTS
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun budget forbids idle readonly continuation"
            )
        _require_attempt_row_aggregate(
            conn,
            work_run_id=work_run_id,
            expected_attempts_started=attempts_started,
        )
        closed = conn.execute(
            "SELECT * FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=?",
            (work_run_id, closed_attempt_id),
        ).fetchone()
        if (
            closed is None
            or int(closed["ordinal"]) != attempts_started
            or str(closed["status"]) != "closed"
            or str(closed["action"] or "") != "call_tools"
            or closed["decision_json"] is None
            or closed["progress_revision_after"] is None
            or closed["budget_after_json"] is None
            or closed["budget_charge_id"] is None
            or str(closed["close_reason"] or "") != "tool_results_recorded"
            or closed["closed_at"] is None
        ):
            raise WorkExecutionPersistenceError(
                "idle readonly recovery requires the latest closed call_tools Attempt"
            )
        previous_turn_id = str(closed["turn_id"])
        if previous_turn_id == turn_id:
            raise WorkExecutionPersistenceError(
                "idle readonly resume requires a different Turn"
            )
        stored_catalog_json = str(closed["catalog_snapshot_json"])
        if (
            _text_hash(stored_catalog_json)
            != str(closed["catalog_snapshot_hash"])
            or stored_catalog_json != catalog_json
        ):
            raise WorkExecutionPersistenceError(
                "idle readonly recovery catalog snapshot drifted"
            )
        try:
            decision = HostAcceptedAttemptDecision.model_validate_json(
                str(closed["decision_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "idle readonly recovery decision is corrupt"
            ) from exc
        _require_readonly_recovery_batch(
            conn,
            work_run_id=work_run_id,
            attempt_id=closed_attempt_id,
            decision=decision,
            catalog_snapshot=normalized_catalog,
            allow_protected_recovery=allow_protected_recovery,
        )
        call_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_tool_calls "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, closed_attempt_id),
            ).fetchone()[0]
        )
        result_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_tool_results "
                "WHERE work_run_id=? AND attempt_id=?",
                (work_run_id, closed_attempt_id),
            ).fetchone()[0]
        )
        if call_count < 1 or result_count != call_count:
            raise WorkExecutionPersistenceError(
                "idle readonly recovery requires the fully closed result batch"
            )

        previous_turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (previous_turn_id,),
        ).fetchone()
        if (
            previous_turn is None
            or str(previous_turn["session_id"]) != session_id
            or str(previous_turn["status"]) != "incomplete"
        ):
            raise WorkExecutionPersistenceError(
                "previous idle-readonly Turn has not settled as incomplete"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=? LIMIT 1",
            (previous_turn_id, work_run_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "idle readonly WorkRun lost its previous Turn link"
            )
        subject = _subject_from_run_row(run_row)
        _require_active_attempt_resume_authority(
            conn,
            session_id=session_id,
            subject=subject,
            run_row=run_row,
        )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "idle readonly resume Turn is not linked to this Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (turn_id, work_run_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "idle readonly resume Turn already has a WorkRun link"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_attempts WHERE attempt_id=?",
            (next_attempt_id,),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "idle readonly next Attempt identity already exists"
            )

        progress = _load_progress(conn, work_run_id)
        output_window, output_hash = _load_output_window(conn, work_run_id)
        if progress.evaluated_output_revision != output_window.output_revision:
            raise WorkExecutionPersistenceError(
                "idle readonly progress is not bound to the OutputWindow"
            )
        next_ordinal = attempts_started + 1
        verification_feedback = _load_latest_nonpass_verification_result(
            conn,
            work_run_id=work_run_id,
            subject=progress.subject,
            output_revision=output_window.output_revision,
            acceptance_ids=tuple(
                item.acceptance_id for item in progress.items
            ),
            consuming_attempt_ordinal=next_ordinal,
        )
        input_verification_request_id: str | None = None
        effective_checkpoint_id = window["latest_checkpoint_id"]
        if verification_feedback is not None:
            input_verification_request_id, _ = verification_feedback
            if effective_checkpoint_id not in (
                None,
                input_verification_request_id,
            ):
                raise WorkExecutionPersistenceError(
                    "idle readonly checkpoint conflicts with verification feedback"
                )
            effective_checkpoint_id = input_verification_request_id
        next_attempt = Attempt(
            attempt_id=next_attempt_id,
            work_run_id=work_run_id,
            ordinal=next_ordinal,
        )
        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'continued', ?)",
            (session_id, turn_id, work_run_id, link_revision, now),
        )
        conn.execute(
            "INSERT INTO insession_work_run_attempts "
            "(attempt_id, work_run_id, turn_id, input_turn_id, "
            "predecessor_question_attempt_id, ordinal, status, action, decision_json, "
            "progress_revision_before, progress_revision_after, input_output_revision, "
            "input_output_hash, committed_output_revision, submitted_output_revision, "
            "input_checkpoint_id, input_verification_request_id, catalog_snapshot_json, "
            "catalog_snapshot_hash, budget_before_json, budget_after_json, close_reason, "
            "created_at, closed_at) VALUES (?, ?, ?, ?, NULL, ?, 'active', NULL, NULL, "
            "?, NULL, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)",
            (
                next_attempt_id,
                work_run_id,
                turn_id,
                turn_id,
                next_ordinal,
                progress.revision,
                output_window.output_revision,
                output_hash,
                effective_checkpoint_id,
                input_verification_request_id,
                catalog_json,
                _text_hash(catalog_json),
                _model_json(budget),
                now,
            ),
        )
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET current_attempt_id=?, "
            "attempts_started=attempts_started+1, revision=?, updated_turn_id=?, "
            "updated_at=? WHERE work_run_id=? AND revision=? AND status='active' "
            "AND reason IS NULL AND current_attempt_id IS NULL "
            "AND current_verification_request_id IS NULL "
            "AND attempts_started < max_attempts",
            (
                next_attempt_id,
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during idle readonly continuation"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=COALESCE(?, latest_checkpoint_id), "
            "stage='L2_PLAN', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                work_run_id,
                next_attempt_id,
                effective_checkpoint_id,
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during idle readonly continuation"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=next_attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        )
        if result.attempt != next_attempt:
            raise WorkExecutionPersistenceError(
                "idle readonly continuation returned an unexpected Attempt"
            )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="resume_active_attempt",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def charge_work_run_active_time(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    checkpoint_id: str,
    active_seconds_delta: float,
    expected_work_run_revision: int,
    expected_window_revision: int,
    apply_id: str,
) -> WorkRunBudgetChargeMutationResult:
    """在安全边界以原子方式计收一段由 Host 测量的活动时间。

    此处不读取时钟，且事务中不发生外部操作。这是供原本空闲的 WorkRun 检查点使用
    的重放安全独立原语。Attempt/Tool/Verifier 终态路径已将其测量区间与业务结算
    事务绑定。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("checkpoint_id", checkpoint_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    # 纯 reducer 负责有限/正数验证和精确边界分类。在打开 SQLite 前验证规范零账本，
    # 使 NaN/inf 永远无法进入载荷哈希或事务。
    probe = charge_work_run_active_seconds(
        WorkRunBudget(),
        active_seconds_delta=active_seconds_delta,
    )
    normalized_delta = probe.active_seconds_delta
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "checkpoint_id": checkpoint_id,
        "active_seconds_delta": normalized_delta,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_window_revision": expected_window_revision,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_budget_charge_replay(
            conn,
            apply_id=apply_id,
            operation="charge_active_time",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        window = _require_owned_work_run_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            expected_window_revision,
        )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        current_budget = _budget_from_row(run_row)
        _require_attempt_row_aggregate(
            conn,
            work_run_id=work_run_id,
            expected_attempts_started=current_budget.attempts_started,
        )
        current_status = WorkRunStatus(str(run_row["status"]))
        current_reason = (
            str(run_row["reason"]) if run_row["reason"] is not None else None
        )
        if (
            run_row["current_attempt_id"] is not None
            or window["current_attempt_id"] is not None
            or window["pending_operation_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "active time can be charged only at a safe WorkRun checkpoint"
            )
        if not (
            (
                current_status is WorkRunStatus.ACTIVE
                and current_reason is None
            )
            or (
                current_status is WorkRunStatus.TURN_LIMIT_REACHED
                and current_reason == "turn_limit_reached"
            )
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun state does not permit an active-time charge"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_budget_charges "
            "WHERE work_run_id=? AND checkpoint_id=?",
            (work_run_id, checkpoint_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "WorkRun checkpoint already has an active-time charge"
            )

        transition = charge_work_run_active_seconds(
            current_budget,
            active_seconds_delta=normalized_delta,
        )
        if transition.disposition is WorkRunBudgetDisposition.HARD_LIMIT_REACHED:
            next_status = WorkRunStatus.FAILED
            next_reason = "work_run_limit_reached"
            _project_execution_subject_budget_failure(
                conn, run_row=run_row, now=now
            )
        elif (
            transition.disposition
            is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
        ):
            next_status = WorkRunStatus.TURN_LIMIT_REACHED
            next_reason = "turn_limit_reached"
        else:
            next_status = WorkRunStatus.ACTIVE
            next_reason = None

        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET active_seconds_consumed=?, status=?, "
            "reason=?, revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND session_id=? AND revision=? AND status=? "
            "AND active_seconds_consumed=? AND current_attempt_id IS NULL "
            "AND current_verification_request_id IS NULL",
            (
                transition.budget_after.active_seconds_consumed,
                next_status.value,
                next_reason,
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                session_id,
                expected_work_run_revision,
                current_status.value,
                transition.budget_before.active_seconds_consumed,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun budget changed during active-time charge"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET latest_checkpoint_id=?, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id IS NULL AND pending_operation_id IS NULL "
            "AND state_version=?",
            (
                checkpoint_id,
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during active-time charge"
            )
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="charge_active_time",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=checkpoint_id,
            work_run_revision_before=expected_work_run_revision,
            work_run_revision_after=next_run_revision,
            window_state_version_before=expected_window_revision,
            window_state_version_after=next_window_revision,
            transition=transition,
            work_run_status_after=next_status,
            work_run_reason_after=next_reason,
            now=now,
        )
        result = WorkRunBudgetChargeMutationResult(
            status="applied",
            budget_charge_id=apply_id,
            work_run_id=work_run_id,
            checkpoint_id=checkpoint_id,
            work_run_revision=next_run_revision,
            work_run_status=next_status,
            work_run_reason=next_reason,
            transition=transition,
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="charge_active_time",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def _normalize_active_seconds_delta(
    active_seconds_delta: float | None,
) -> float | None:
    if active_seconds_delta is None:
        return None
    probe = charge_work_run_active_seconds(
        WorkRunBudget(),
        active_seconds_delta=active_seconds_delta,
    )
    return probe.active_seconds_delta


def _require_settlement_budget_transition(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    work_run_id: str,
    active_seconds_delta: float | None,
) -> WorkRunBudgetTransition:
    if active_seconds_delta is None:
        raise WorkExecutionPersistenceError(
            "new terminal settlement requires active_seconds_delta"
        )
    budget = _budget_from_row(run_row)
    _require_attempt_row_aggregate(
        conn,
        work_run_id=work_run_id,
        expected_attempts_started=budget.attempts_started,
    )
    return charge_work_run_active_seconds(
        budget,
        active_seconds_delta=active_seconds_delta,
    )


def _settlement_budget_checkpoint_id(operation: str, apply_id: str) -> str:
    return f"{operation}:{apply_id}"


def _insert_budget_charge(
    conn: sqlite3.Connection,
    *,
    budget_charge_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    turn_id: str,
    checkpoint_id: str,
    work_run_revision_before: int,
    work_run_revision_after: int,
    window_state_version_before: int,
    window_state_version_after: int,
    transition: WorkRunBudgetTransition,
    work_run_status_after: WorkRunStatus,
    work_run_reason_after: str | None,
    now: str,
) -> None:
    try:
        conn.execute(
            "INSERT INTO insession_work_run_budget_charges "
            "(budget_charge_id, operation, session_id, work_run_id, turn_id, "
            "checkpoint_id, work_run_revision_before, work_run_revision_after, "
            "window_state_version_before, window_state_version_after, "
            "active_seconds_delta, active_seconds_before, active_seconds_after, "
            "disposition, work_run_status_after, work_run_reason_after, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                budget_charge_id,
                operation,
                session_id,
                work_run_id,
                turn_id,
                checkpoint_id,
                work_run_revision_before,
                work_run_revision_after,
                window_state_version_before,
                window_state_version_after,
                transition.active_seconds_delta,
                transition.budget_before.active_seconds_consumed,
                transition.budget_after.active_seconds_consumed,
                transition.disposition.value,
                work_run_status_after.value,
                work_run_reason_after,
                now,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise WorkExecutionPersistenceError(
            "WorkRun budget charge conflicts with stored authority"
        ) from exc


def start_work_run_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    apply_id: str,
    catalog_snapshot: Mapping[str, Any],
    input_checkpoint_id: str | None = None,
    attempt_id: str | None = None,
) -> WorkExecutionMutationResult:
    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    if input_checkpoint_id is not None:
        _require_identifier("input_checkpoint_id", input_checkpoint_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if attempt_id is not None:
        _require_identifier("attempt_id", attempt_id)
    catalog_json = _canonical_json(dict(catalog_snapshot))
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_window_revision": expected_window_revision,
        "input_checkpoint_id": input_checkpoint_id,
        "catalog_snapshot": json.loads(catalog_json),
        "requested_attempt_id": attempt_id,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(conn, apply_id=apply_id, operation="start_attempt", session_id=session_id, payload_hash=payload_hash)
        if replay is not None:
            return replay
        window = _require_owned_work_run_window(conn, session_id, turn_id, work_run_id, expected_window_revision)
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if str(run_row["status"]) != WorkRunStatus.ACTIVE.value:
            raise WorkExecutionPersistenceError("only an active WorkRun can start an Attempt")
        if run_row["reason"] is not None:
            raise WorkExecutionPersistenceError(
                "WorkRun phase does not permit starting another Attempt"
            )
        if run_row["current_attempt_id"] is not None or window["current_attempt_id"] is not None:
            raise WorkExecutionPersistenceError("WorkRun already has an active Attempt")
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        output_window, output_hash = _load_output_window(conn, work_run_id)
        if progress.evaluated_output_revision != output_window.output_revision:
            raise WorkExecutionPersistenceError(
                "Acceptance progress is not bound to the current OutputWindow"
            )
        current_budget = _budget_from_row(run_row)
        if (
            current_budget.active_seconds_consumed
            >= current_budget.hard_active_seconds
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun hard active-time budget is exhausted"
            )
        if (
            current_budget.active_seconds_consumed
            >= current_budget.soft_active_seconds
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun soft active-time budget forbids a new Attempt"
            )
        attempts_started = int(run_row["attempts_started"])
        max_attempts = int(run_row["max_attempts"])
        if attempts_started >= max_attempts or attempts_started >= _MAX_ATTEMPTS:
            raise WorkExecutionPersistenceError("WorkRun Attempt budget is exhausted")
        _require_attempt_row_aggregate(
            conn,
            work_run_id=work_run_id,
            expected_attempts_started=attempts_started,
        )
        verification_feedback = _load_latest_nonpass_verification_result(
            conn,
            work_run_id=work_run_id,
            subject=progress.subject,
            output_revision=output_window.output_revision,
            acceptance_ids=tuple(
                item.acceptance_id for item in progress.items
            ),
            consuming_attempt_ordinal=attempts_started + 1,
        )
        input_verification_request_id: str | None = None
        effective_checkpoint_id = input_checkpoint_id
        if verification_feedback is not None:
            input_verification_request_id, _ = verification_feedback
            if input_checkpoint_id not in (None, input_verification_request_id):
                raise WorkExecutionPersistenceError(
                    "Attempt checkpoint conflicts with current verification feedback"
                )
            effective_checkpoint_id = input_verification_request_id
        allocated_attempt_id = attempt_id or _allocate_id(deps, "attempt")
        ordinal = attempts_started + 1
        Attempt(
            attempt_id=allocated_attempt_id,
            work_run_id=work_run_id,
            ordinal=ordinal,
        )
        budget_before = _budget_from_row(run_row)
        conn.execute(
            "INSERT INTO insession_work_run_attempts "
            "(attempt_id, work_run_id, turn_id, input_turn_id, "
            "predecessor_question_attempt_id, ordinal, status, action, decision_json, "
            "progress_revision_before, progress_revision_after, input_output_revision, "
            "input_output_hash, committed_output_revision, submitted_output_revision, input_checkpoint_id, "
            "input_verification_request_id, catalog_snapshot_json, catalog_snapshot_hash, "
            "budget_before_json, budget_after_json, "
            "close_reason, created_at, closed_at) "
            "VALUES (?, ?, ?, ?, NULL, ?, 'active', NULL, NULL, ?, NULL, ?, ?, "
            "NULL, NULL, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)",
            (
                allocated_attempt_id,
                work_run_id,
                turn_id,
                turn_id,
                ordinal,
                progress.revision,
                output_window.output_revision,
                output_hash,
                effective_checkpoint_id,
                input_verification_request_id,
                catalog_json,
                _text_hash(catalog_json),
                _model_json(budget_before),
                now,
            ),
        )
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET current_attempt_id=?, attempts_started=attempts_started+1, "
            "revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND current_attempt_id IS NULL "
            "AND status='active' AND attempts_started < max_attempts",
            (
                allocated_attempt_id,
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("WorkRun changed during Attempt start")
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_attempt_id=?, stage='L2_PLAN', "
            "latest_checkpoint_id=COALESCE(?, latest_checkpoint_id), "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id IS NULL AND state_version=?",
            (
                allocated_attempt_id,
                effective_checkpoint_id,
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Turn Window changed during Attempt start")
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=allocated_attempt_id,
            window_state_version=next_window_revision,
        )
        _insert_receipt(conn, apply_id=apply_id, operation="start_attempt", session_id=session_id, work_run_id=work_run_id, payload_hash=payload_hash, result=result, now=now)
        return result


def commit_work_run_attempt_decision(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    decision: HostAcceptedAttemptDecision,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    apply_id: str,
    active_seconds_delta: float | None = None,
) -> WorkExecutionMutationResult:
    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("attempt_id", attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if isinstance(
        decision.action,
        (HostMaterializedWriteOutputWindowAction, HostMaterializedSubmitOutputWindowAction),
    ):
        raise WorkExecutionPersistenceError(
            "OutputWindow actions require commit_work_run_output_action"
        )
    normalized_delta = _normalize_active_seconds_delta(active_seconds_delta)
    if (
        isinstance(decision.action, HostMaterializedCallToolsAction)
        and normalized_delta is not None
    ):
        raise WorkExecutionPersistenceError(
            "call_tools decision must not charge active time before ToolCall settlement"
        )
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "decision": decision.model_dump(mode="json"),
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_window_revision": expected_window_revision,
    }
    if normalized_delta is not None:
        payload["active_seconds_delta"] = normalized_delta
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(conn, apply_id=apply_id, operation="commit_attempt_decision", session_id=session_id, payload_hash=payload_hash)
        if replay is not None:
            if isinstance(
                decision.action,
                RequestTaskGraphRevisionAction,
            ):
                replay_run = _require_run_row(conn, work_run_id, session_id)
                request = _load_replayed_execution_replan_request(
                    conn,
                    create_apply_id=apply_id,
                    session_id=session_id,
                    task_id=str(replay_run["insession_task_id"]),
                    work_run_id=work_run_id,
                    attempt_id=attempt_id,
                )
                if (
                    request is None
                    or request.create_apply_id != apply_id
                    or request.work_run_id != work_run_id
                    or request.attempt_id != attempt_id
                ):
                    raise WorkExecutionPersistenceError(
                        "replayed TaskGraph revision request lost exact durable authority"
                    )
            return replay
        _require_owned_attempt_window(conn, session_id, turn_id, work_run_id, attempt_id, expected_window_revision)
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if str(run_row["status"]) != "active" or str(run_row["current_attempt_id"] or "") != attempt_id:
            raise WorkExecutionPersistenceError("Attempt is not current on an active WorkRun")
        attempt_row = _require_active_attempt(conn, work_run_id, attempt_id)
        if attempt_row["decision_json"] is not None:
            raise WorkExecutionPersistenceError("AttemptDecision is already committed")
        is_execution_replan = isinstance(
            decision.action,
            RequestTaskGraphRevisionAction,
        )
        budget_transition = (
            _require_settlement_budget_transition(
                conn,
                run_row=run_row,
                work_run_id=work_run_id,
                active_seconds_delta=normalized_delta,
            )
            if isinstance(
                decision.action,
                (RequestUserInputAction, RequestTaskGraphRevisionAction),
            )
            else None
        )
        preexisting_execution_rows = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM insession_work_run_tool_calls WHERE attempt_id=?) AS calls, "
            "(SELECT COUNT(*) FROM insession_work_run_tool_results WHERE attempt_id=?) AS results",
            (attempt_id, attempt_id),
        ).fetchone()
        if preexisting_execution_rows is None or (
            int(preexisting_execution_rows["calls"]) != 0
            or int(preexisting_execution_rows["results"]) != 0
        ):
            raise WorkExecutionPersistenceError(
                "undecided Attempt already contains execution authority"
            )
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        validated_result_ids = {
            item.tool_result_id
            for item in _load_tool_results(conn, work_run_id=work_run_id)
        }
        historical_result_ids = {
            str(row["tool_result_id"])
            for row in conn.execute(
                "SELECT result.tool_result_id FROM insession_work_run_tool_results AS result "
                "JOIN insession_work_run_attempts AS attempt "
                "ON attempt.work_run_id=result.work_run_id "
                "AND attempt.attempt_id=result.attempt_id "
                "WHERE result.work_run_id=? AND result.status='succeeded' "
                "AND attempt.status='closed' AND attempt.ordinal < ?",
                (work_run_id, int(attempt_row["ordinal"])),
            ).fetchall()
        }
        if not historical_result_ids.issubset(validated_result_ids):
            raise WorkExecutionPersistenceError(
                "historical ToolResult authority is incomplete"
            )
        replan_preflight = None
        if is_execution_replan:
            assert isinstance(
                decision.action,
                RequestTaskGraphRevisionAction,
            )
            if not set(decision.action.supporting_tool_result_ids).issubset(
                historical_result_ids
            ):
                raise WorkExecutionPersistenceError(
                    "TaskGraph revision request cites unavailable ToolResults"
                )
            replan_preflight = _require_execution_replan_preconditions(
                conn,
                session_id=session_id,
                run_row=run_row,
                work_run_id=work_run_id,
                attempt_id=attempt_id,
            )
        merged = merge_acceptance_progress(
            progress,
            decision.acceptance_updates,
            known_historical_tool_result_ids=historical_result_ids,
            expected_progress_revision=expected_progress_revision,
            current_work_run_revision=expected_work_run_revision,
            expected_work_run_revision=expected_work_run_revision,
        )
        if merged.status != "applied":
            codes = ",".join(issue.code.value for issue in merged.issues)
            raise WorkExecutionPersistenceError(
                f"Acceptance progress update was rejected: {codes}"
            )
        materialized_calls: tuple[HostMaterializedToolCall, ...] = ()
        if isinstance(decision.action, HostMaterializedCallToolsAction):
            materialized_calls = decision.action.calls
            for ordinal, call in enumerate(materialized_calls, start=1):
                arguments_json = _canonical_json(call.arguments)
                try:
                    conn.execute(
                        "INSERT INTO insession_work_run_tool_calls "
                        "(tool_call_id, work_run_id, attempt_id, ordinal, provider_call_id, "
                        "tool_id, tool_version, modifies_environment, arguments_hash, arguments_json, created_at) "
                        "VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                        (
                            call.tool_call_id,
                            work_run_id,
                            attempt_id,
                            ordinal,
                            call.tool_id,
                            call.tool_version,
                            int(call.modifies_environment),
                            _text_hash(arguments_json),
                            arguments_json,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise WorkExecutionPersistenceError(
                        "Host-materialized ToolCalls conflict with stored authority"
                    ) from exc

        action_kind = decision.action.kind
        closes_attempt = not isinstance(decision.action, HostMaterializedCallToolsAction)
        if is_execution_replan:
            next_status = WorkRunStatus.CANCELLED
            next_reason = "task_graph_revision_requested"
        elif isinstance(decision.action, RequestUserInputAction):
            assert budget_transition is not None
            if (
                budget_transition.disposition
                is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
            ):
                next_status = WorkRunStatus.FAILED
                next_reason = "work_run_limit_reached"
            else:
                next_status = WorkRunStatus.WAITING_USER
                next_reason = "needs_input"
        else:
            next_status = WorkRunStatus.ACTIVE
            next_reason = None
        budget_after = (
            budget_transition.budget_after
            if budget_transition is not None
            else _budget_from_row(run_row)
        )
        next_run_revision = expected_work_run_revision + 1
        next_window_revision = expected_window_revision + 1
        if budget_transition is not None:
            _insert_budget_charge(
                conn,
                budget_charge_id=apply_id,
                operation="commit_attempt_decision",
                session_id=session_id,
                work_run_id=work_run_id,
                turn_id=turn_id,
                checkpoint_id=_settlement_budget_checkpoint_id(
                    "commit_attempt_decision", apply_id
                ),
                work_run_revision_before=expected_work_run_revision,
                work_run_revision_after=next_run_revision,
                window_state_version_before=expected_window_revision,
                window_state_version_after=next_window_revision,
                transition=budget_transition,
                work_run_status_after=next_status,
                work_run_reason_after=next_reason,
                now=now,
            )
        decision_json = _model_json(decision)
        if conn.execute(
            "UPDATE insession_work_run_attempts SET action=?, decision_json=?, "
            "progress_revision_after=?, status=?, budget_after_json=?, "
            "budget_charge_id=?, close_reason=?, closed_at=? "
            "WHERE attempt_id=? AND work_run_id=? AND status='active' AND decision_json IS NULL",
            (
                action_kind,
                decision_json,
                merged.snapshot.revision,
                "closed" if closes_attempt else "active",
                _model_json(budget_after) if closes_attempt else None,
                apply_id if budget_transition is not None else None,
                action_kind if closes_attempt else None,
                now if closes_attempt else None,
                attempt_id,
                work_run_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Attempt changed during decision commit")
        if merged.changed:
            snapshot_json = _model_json(merged.snapshot)
            if conn.execute(
                "UPDATE insession_work_run_acceptance_progress SET progress_revision=?, "
                "snapshot_hash=?, snapshot_json=?, updated_attempt_id=?, updated_at=? "
                "WHERE work_run_id=? AND progress_revision=?",
                (
                    merged.snapshot.revision,
                    _text_hash(snapshot_json),
                    snapshot_json,
                    attempt_id,
                    now,
                    work_run_id,
                    expected_progress_revision,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError("Acceptance progress changed during decision commit")
        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, current_attempt_id=?, "
            "updated_turn_id=?, updated_at=? WHERE work_run_id=? AND revision=? "
            "AND current_attempt_id=? AND status='active'",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_after.active_seconds_consumed,
                None if closes_attempt else attempt_id,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                attempt_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("WorkRun changed during decision commit")
        next_task_state_version: int | None = None
        next_node_state_version: int | None = None
        if is_execution_replan:
            assert replan_preflight is not None
            assert isinstance(
                decision.action,
                RequestTaskGraphRevisionAction,
            )
            (
                replan_request,
                next_task_state_version,
                next_node_state_version,
            ) = _apply_execution_replan_projection(
                conn,
                session_id=session_id,
                turn_id=turn_id,
                work_run_id=work_run_id,
                attempt_id=attempt_id,
                apply_id=apply_id,
                action=decision.action,
                preflight=replan_preflight,
                now=now,
            )
            _insert_execution_replan_request(
                conn,
                request=replan_request,
                now=now,
            )
        elif isinstance(decision.action, RequestUserInputAction):
            if next_status is WorkRunStatus.FAILED:
                _project_execution_subject_budget_failure(
                    conn, run_row=run_row, now=now
                )
            else:
                _project_execution_subject_wait_state(
                    conn,
                    run_row=run_row,
                    target_status="awaiting_user",
                    now=now,
                )
        if is_execution_replan:
            window_update = conn.execute(
                "UPDATE turn_execution_windows SET current_work_run_id=NULL, "
                "current_attempt_id=NULL, stage='L2_PLAN', state_version=?, "
                "updated_at=? WHERE session_id=? AND turn_id=? "
                "AND window_state='active' AND current_work_run_id=? "
                "AND current_attempt_id=? AND state_version=?",
                (
                    next_window_revision,
                    now,
                    session_id,
                    turn_id,
                    work_run_id,
                    attempt_id,
                    expected_window_revision,
                ),
            )
        else:
            window_update = conn.execute(
                "UPDATE turn_execution_windows SET current_attempt_id=?, "
                "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
                "AND window_state='active' AND current_work_run_id=? "
                "AND current_attempt_id=? AND state_version=?",
                (
                    None if closes_attempt else attempt_id,
                    next_window_revision,
                    now,
                    session_id,
                    turn_id,
                    work_run_id,
                    attempt_id,
                    expected_window_revision,
                ),
            )
        if window_update.rowcount != 1:
            raise WorkExecutionPersistenceError("Turn Window changed during decision commit")
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            new_tool_call_ids=tuple(call.tool_call_id for call in materialized_calls),
            window_state_version=next_window_revision,
            budget_transition=budget_transition,
        )
        if is_execution_replan:
            result = result.model_copy(
                update={
                    "task_state_version": next_task_state_version,
                    "node_state_version": next_node_state_version,
                }
            )
        _insert_receipt(conn, apply_id=apply_id, operation="commit_attempt_decision", session_id=session_id, work_run_id=work_run_id, payload_hash=payload_hash, result=result, now=now)
        return result


def commit_work_run_output_action(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    decision: AttemptDecision,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_output_revision: int,
    expected_window_revision: int,
    apply_id: str,
    active_seconds_delta: float | None = None,
) -> WorkExecutionMutationResult:
    """整体替换一个 OutputWindow，并以原子方式关闭其 Attempt。

    模型决策只在此调用边界间携带正文。持久化 Attempt 决策会物化为小型 revision
    引用；正文仅存在于 ``insession_work_run_output_windows``。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("attempt_id", attempt_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_output_revision", expected_output_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if not isinstance(
        decision.action,
        (
            WriteOutputWindowAction,
            SubmitOutputWindowAction,
        ),
    ):
        raise WorkExecutionPersistenceError(
            "commit_work_run_output_action accepts only write/submit output actions"
        )
    normalized_delta = _normalize_active_seconds_delta(active_seconds_delta)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "decision": decision.model_dump(mode="json"),
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_output_revision": expected_output_revision,
        "expected_window_revision": expected_window_revision,
    }
    if normalized_delta is not None:
        payload["active_seconds_delta"] = normalized_delta
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="commit_output_action",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        _require_owned_attempt_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            attempt_id,
            expected_window_revision,
        )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if (
            str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or run_row["reason"] is not None
            or str(run_row["current_attempt_id"] or "") != attempt_id
        ):
            raise WorkExecutionPersistenceError(
                "OutputWindow action requires the current unblocked active Attempt"
            )
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=work_run_id,
            active_seconds_delta=normalized_delta,
        )
        attempt_row = _require_active_attempt(conn, work_run_id, attempt_id)
        if attempt_row["decision_json"] is not None:
            raise WorkExecutionPersistenceError("AttemptDecision is already committed")
        preexisting_execution_rows = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM insession_work_run_tool_calls WHERE attempt_id=?) AS calls, "
            "(SELECT COUNT(*) FROM insession_work_run_tool_results WHERE attempt_id=?) AS results",
            (attempt_id, attempt_id),
        ).fetchone()
        if preexisting_execution_rows is None or (
            int(preexisting_execution_rows["calls"]) != 0
            or int(preexisting_execution_rows["results"]) != 0
        ):
            raise WorkExecutionPersistenceError(
                "undecided Attempt already contains execution authority"
            )

        current_output, current_output_hash = _load_output_window(conn, work_run_id)
        if current_output.output_revision != expected_output_revision:
            raise WorkExecutionOutputRevisionConflict(
                expected=expected_output_revision,
                actual=current_output.output_revision,
            )
        if (
            int(attempt_row["input_output_revision"]) != current_output.output_revision
            or str(attempt_row["input_output_hash"]) != current_output_hash
        ):
            raise WorkExecutionPersistenceError(
                "Attempt input is not bound to the current OutputWindow"
            )
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)

        validated_result_ids = {
            item.tool_result_id
            for item in _load_tool_results(conn, work_run_id=work_run_id)
        }
        historical_result_ids = {
            str(row["tool_result_id"])
            for row in conn.execute(
                "SELECT result.tool_result_id "
                "FROM insession_work_run_tool_results AS result "
                "JOIN insession_work_run_attempts AS attempt "
                "ON attempt.work_run_id=result.work_run_id "
                "AND attempt.attempt_id=result.attempt_id "
                "WHERE result.work_run_id=? AND attempt.status='closed' "
                "AND attempt.ordinal < ? AND result.status='succeeded'",
                (work_run_id, int(attempt_row["ordinal"])),
            ).fetchall()
        }
        if not historical_result_ids.issubset(validated_result_ids):
            raise WorkExecutionPersistenceError(
                "historical ToolResult authority is incomplete"
            )
        applied = apply_output_window_action(
            current_output,
            progress,
            decision.action,
            acceptance_updates=decision.acceptance_updates,
            updated_turn_id=turn_id,
            updated_attempt_id=attempt_id,
            known_historical_tool_result_ids=historical_result_ids,
            expected_progress_revision=expected_progress_revision,
            current_work_run_revision=expected_work_run_revision,
            expected_work_run_revision=expected_work_run_revision,
        )
        if applied.status != "applied" or applied.materialized_action is None:
            codes = ",".join(code.value for code in applied.progress_merge.error_codes)
            raise WorkExecutionPersistenceError(
                f"OutputWindow action was rejected: {codes}"
            )

        output_window = applied.output_window
        materialized_decision = HostAcceptedAttemptDecision(
            acceptance_updates=decision.acceptance_updates,
            action=applied.materialized_action,
        )
        materialized_json = _model_json(materialized_decision)
        is_submit = isinstance(
            applied.materialized_action,
            HostMaterializedSubmitOutputWindowAction,
        )
        if (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        ):
            next_status = WorkRunStatus.FAILED
            next_reason = "work_run_limit_reached"
        elif is_submit:
            # 在软边界，已提交输出仍可进行语义验证；只有绝对硬边界会阻止它。
            next_status = WorkRunStatus.ACTIVE
            next_reason = "verification_pending"
        elif (
            budget_transition.disposition
            is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
        ):
            next_status = WorkRunStatus.TURN_LIMIT_REACHED
            next_reason = "turn_limit_reached"
        else:
            next_status = WorkRunStatus.ACTIVE
            next_reason = None
        next_run_revision = expected_work_run_revision + 1
        next_window_revision = expected_window_revision + 1
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="commit_output_action",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "commit_output_action", apply_id
            ),
            work_run_revision_before=expected_work_run_revision,
            work_run_revision_after=next_run_revision,
            window_state_version_before=expected_window_revision,
            window_state_version_after=next_window_revision,
            transition=budget_transition,
            work_run_status_after=next_status,
            work_run_reason_after=next_reason,
            now=now,
        )
        if "content" in type(applied.materialized_action).model_fields:
            raise WorkExecutionPersistenceError(
                "materialized OutputWindow decision must not persist body content"
            )
        if conn.execute(
            "UPDATE insession_work_run_attempts SET status='closed', action=?, "
            "decision_json=?, progress_revision_after=?, committed_output_revision=?, "
            "submitted_output_revision=?, budget_after_json=?, budget_charge_id=?, "
            "close_reason=?, closed_at=? "
            "WHERE attempt_id=? AND work_run_id=? AND status='active' "
            "AND decision_json IS NULL AND committed_output_revision IS NULL",
            (
                applied.materialized_action.kind,
                materialized_json,
                applied.progress_merge.snapshot.revision,
                output_window.output_revision,
                output_window.output_revision if is_submit else None,
                _model_json(budget_transition.budget_after),
                apply_id,
                applied.materialized_action.kind,
                now,
                attempt_id,
                work_run_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Attempt changed during OutputWindow commit"
            )

        if applied.output_changed:
            output_json = _model_json(output_window)
            if conn.execute(
                "UPDATE insession_work_run_output_windows "
                "SET output_revision=?, snapshot_hash=?, snapshot_json=?, "
                "updated_turn_id=?, updated_attempt_id=?, updated_at=? "
                "WHERE work_run_id=? AND output_revision=? AND snapshot_hash=? "
                "AND frozen_at IS NULL",
                (
                    output_window.output_revision,
                    _text_hash(output_json),
                    output_json,
                    turn_id,
                    attempt_id,
                    now,
                    work_run_id,
                    expected_output_revision,
                    current_output_hash,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "OutputWindow changed during whole replacement"
                )

        merged = applied.progress_merge
        if merged.changed:
            snapshot_json = _model_json(merged.snapshot)
            if conn.execute(
                "UPDATE insession_work_run_acceptance_progress "
                "SET progress_revision=?, evaluated_output_revision=?, snapshot_hash=?, "
                "snapshot_json=?, updated_attempt_id=?, updated_at=? "
                "WHERE work_run_id=? AND progress_revision=? "
                "AND evaluated_output_revision=?",
                (
                    merged.snapshot.revision,
                    merged.snapshot.evaluated_output_revision,
                    _text_hash(snapshot_json),
                    snapshot_json,
                    attempt_id,
                    now,
                    work_run_id,
                    expected_progress_revision,
                    expected_output_revision,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "Acceptance progress changed during OutputWindow commit"
                )

        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, current_attempt_id=NULL, "
            "updated_turn_id=?, updated_at=? WHERE work_run_id=? AND revision=? "
            "AND status='active' AND reason IS NULL AND current_attempt_id=?",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_transition.budget_after.active_seconds_consumed,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                attempt_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during OutputWindow commit"
            )
        if next_status is WorkRunStatus.FAILED:
            _project_execution_subject_budget_failure(
                conn, run_row=run_row, now=now
            )
        if conn.execute(
            "UPDATE turn_execution_windows SET current_attempt_id=NULL, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id=? AND state_version=?",
            (
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                attempt_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during OutputWindow commit"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            window_state_version=next_window_revision,
            budget_transition=budget_transition,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="commit_output_action",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def append_work_run_tool_result(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    result: ToolResult,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    apply_id: str,
) -> WorkExecutionMutationResult:
    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "result": result.model_dump(mode="json"),
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_window_revision": expected_window_revision,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(conn, apply_id=apply_id, operation="append_tool_result", session_id=session_id, payload_hash=payload_hash)
        if replay is not None:
            return replay
        _require_owned_attempt_window(conn, session_id, turn_id, work_run_id, result.attempt_id, expected_window_revision)
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if str(run_row["status"]) != "active" or str(run_row["current_attempt_id"] or "") != result.attempt_id:
            raise WorkExecutionPersistenceError("ToolResult Attempt is not current")
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        attempt_row = _require_active_attempt(conn, work_run_id, result.attempt_id)
        if str(attempt_row["action"] or "") != "call_tools" or attempt_row["decision_json"] is None:
            raise WorkExecutionPersistenceError("ToolResult requires an accepted call_tools decision")
        call = conn.execute(
            "SELECT ordinal, modifies_environment FROM insession_work_run_tool_calls "
            "WHERE tool_call_id=? AND work_run_id=? AND attempt_id=?",
            (result.tool_call_id, work_run_id, result.attempt_id),
        ).fetchone()
        if call is None:
            raise WorkExecutionPersistenceError("ToolResult references an unknown materialized call")
        if int(call["ordinal"]) != result.ordinal:
            raise WorkExecutionPersistenceError("ToolResult ordinal does not match its materialized call")
        if (
            result.status is ToolResultStatus.COMPLETION_UNCONFIRMED
            and not bool(call["modifies_environment"])
        ):
            raise WorkExecutionPersistenceError(
                "completion_unconfirmed requires an environment-modifying ToolCall"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_tool_results "
            "WHERE tool_result_id=? OR tool_call_id=?",
            (result.tool_result_id, result.tool_call_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError("ToolResult identity or ToolCall is already finalized")
        result_json = _model_json(result)
        try:
            conn.execute(
                "INSERT INTO insession_work_run_tool_results "
                "(tool_result_id, work_run_id, attempt_id, tool_call_id, ordinal, status, result_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.tool_result_id,
                    work_run_id,
                    result.attempt_id,
                    result.tool_call_id,
                    result.ordinal,
                    result.status.value,
                    result_json,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkExecutionPersistenceError(
                "ToolResult conflicts with stored call/result authority"
            ) from exc
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND status='active' AND current_attempt_id=?",
            (next_run_revision, turn_id, now, work_run_id, expected_work_run_revision, result.attempt_id),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("WorkRun changed during ToolResult append")
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id=? AND current_attempt_id=? AND state_version=?",
            (next_window_revision, now, session_id, turn_id, work_run_id, result.attempt_id, expected_window_revision),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Turn Window changed during ToolResult append")
        result_record = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=result.attempt_id,
            tool_result_id=result.tool_result_id,
            window_state_version=next_window_revision,
        )
        _insert_receipt(conn, apply_id=apply_id, operation="append_tool_result", session_id=session_id, work_run_id=work_run_id, payload_hash=payload_hash, result=result_record, now=now)
        return result_record


def close_work_run_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_window_revision: int,
    apply_id: str,
    close_reason: str = "tool_results_recorded",
    active_seconds_delta: float | None = None,
) -> WorkExecutionMutationResult:
    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("attempt_id", attempt_id)
    _require_identifier("close_reason", close_reason)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    normalized_delta = _normalize_active_seconds_delta(active_seconds_delta)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "attempt_id": attempt_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_window_revision": expected_window_revision,
        "close_reason": close_reason,
    }
    if normalized_delta is not None:
        payload["active_seconds_delta"] = normalized_delta
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(conn, apply_id=apply_id, operation="close_attempt", session_id=session_id, payload_hash=payload_hash)
        if replay is not None:
            return replay
        _require_owned_attempt_window(conn, session_id, turn_id, work_run_id, attempt_id, expected_window_revision)
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if str(run_row["status"]) != "active" or str(run_row["current_attempt_id"] or "") != attempt_id:
            raise WorkExecutionPersistenceError("Attempt is not current on an active WorkRun")
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=work_run_id,
            active_seconds_delta=normalized_delta,
        )
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        attempt_row = _require_active_attempt(conn, work_run_id, attempt_id)
        if str(attempt_row["action"] or "") != "call_tools" or attempt_row["decision_json"] is None:
            raise WorkExecutionPersistenceError("only a decided call_tools Attempt closes here")
        counts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM insession_work_run_tool_calls WHERE attempt_id=?) AS calls, "
            "(SELECT COUNT(*) FROM insession_work_run_tool_results WHERE attempt_id=?) AS results",
            (attempt_id, attempt_id),
        ).fetchone()
        if counts is None or int(counts["calls"]) < 1 or int(counts["calls"]) != int(counts["results"]):
            raise WorkExecutionPersistenceError("all materialized ToolCalls require a ToolResult before close")
        attempt_results = _load_tool_results(
            conn,
            work_run_id=work_run_id,
            attempt_id=attempt_id,
        )
        if len(attempt_results) != int(counts["results"]):
            raise WorkExecutionPersistenceError(
                "Attempt ToolResult authority is incomplete"
            )
        result_statuses = {item.status.value for item in attempt_results}
        completion_unconfirmed = "completion_unconfirmed" in result_statuses
        attempt_budget_exhausted = (
            budget_transition.budget_after.attempts_started
            >= budget_transition.budget_after.max_attempts
        )
        if completion_unconfirmed:
            # 对账权威状态高于两个时间阈值：可能已应用的环境变更不能被放弃。
            next_status = WorkRunStatus.WAITING_EXTERNAL
            next_reason = "operation_completion_unconfirmed"
        elif (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        ):
            next_status = WorkRunStatus.FAILED
            next_reason = "work_run_limit_reached"
        elif (
            budget_transition.disposition
            is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
            or attempt_budget_exhausted
        ):
            next_status = WorkRunStatus.TURN_LIMIT_REACHED
            next_reason = "turn_limit_reached"
        else:
            next_status = WorkRunStatus.ACTIVE
            next_reason = None
        effective_close_reason = next_reason or close_reason
        next_run_revision = expected_work_run_revision + 1
        next_window_revision = expected_window_revision + 1
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="close_attempt",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "close_attempt", apply_id
            ),
            work_run_revision_before=expected_work_run_revision,
            work_run_revision_after=next_run_revision,
            window_state_version_before=expected_window_revision,
            window_state_version_after=next_window_revision,
            transition=budget_transition,
            work_run_status_after=next_status,
            work_run_reason_after=next_reason,
            now=now,
        )
        if conn.execute(
            "UPDATE insession_work_run_attempts SET status='closed', budget_after_json=?, "
            "budget_charge_id=?, close_reason=?, closed_at=? "
            "WHERE attempt_id=? AND work_run_id=? AND status='active'",
            (
                _model_json(budget_transition.budget_after),
                apply_id,
                effective_close_reason,
                now,
                attempt_id,
                work_run_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Attempt changed during close")
        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, "
            "current_attempt_id=NULL, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND status='active' AND current_attempt_id=?",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_transition.budget_after.active_seconds_consumed,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                attempt_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("WorkRun changed during Attempt close")
        if completion_unconfirmed:
            _project_execution_subject_wait_state(
                conn,
                run_row=run_row,
                target_status="waiting_external",
                now=now,
            )
        elif next_status is WorkRunStatus.FAILED:
            _project_execution_subject_budget_failure(
                conn, run_row=run_row, now=now
            )
        if conn.execute(
            "UPDATE turn_execution_windows SET current_attempt_id=NULL, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id=? AND current_attempt_id=? AND state_version=?",
            (next_window_revision, now, session_id, turn_id, work_run_id, attempt_id, expected_window_revision),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError("Turn Window changed during Attempt close")
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            window_state_version=next_window_revision,
            budget_transition=budget_transition,
        )
        _insert_receipt(conn, apply_id=apply_id, operation="close_attempt", session_id=session_id, work_run_id=work_run_id, payload_hash=payload_hash, result=result, now=now)
        return result


def _load_record(
    conn: sqlite3.Connection,
    work_run_id: str,
    *,
    session_id: str,
) -> StoredWorkRun:
    row = conn.execute(
        "SELECT * FROM insession_work_runs WHERE work_run_id=? AND session_id=?",
        (work_run_id, session_id),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError("unknown WorkRun in this Session")
    subject = _subject_from_run_row(row)
    run = WorkRun(
        work_run_id=work_run_id,
        subject=subject,
        revision=int(row["revision"]),
        status=WorkRunStatus(str(row["status"])),
        reason=(str(row["reason"]) if row["reason"] is not None else None),
        budget=_budget_from_row(row),
    )
    progress = _load_progress(conn, work_run_id)
    output_window, _ = _load_output_window(conn, work_run_id)
    if progress.evaluated_output_revision != output_window.output_revision:
        raise WorkExecutionPersistenceError(
            "Acceptance progress is not bound to the current OutputWindow"
        )
    current_verification_request_id = _validate_current_verification_projection(
        conn,
        run_row=row,
        run=run,
    )
    if isinstance(subject, TaskNodeSubject):
        node_delivery_id = _validate_delivery_projection(
            conn,
            run_row=row,
            run=run,
            output_window=output_window,
        )
        auxiliary_node_completion_id = None
    else:
        node_delivery_id = None
        auxiliary_node_completion_id = _validate_auxiliary_completion_projection(
            conn,
            run_row=row,
            run=run,
            output_window=output_window,
        )
    attempt_rows = conn.execute(
        "SELECT attempt_id, turn_id, input_turn_id, "
        "predecessor_question_attempt_id, ordinal, status, action, decision_json, "
        "input_output_revision, input_output_hash, committed_output_revision, "
        "submitted_output_revision, input_checkpoint_id, input_verification_request_id, "
        "progress_revision_before, catalog_snapshot_json, catalog_snapshot_hash, "
        "budget_before_json, budget_after_json, budget_charge_id, close_reason "
        "FROM insession_work_run_attempts "
        "WHERE work_run_id=? ORDER BY ordinal",
        (work_run_id,),
    ).fetchall()
    attempts_list: list[StoredAttempt] = []
    for item in attempt_rows:
        ordinal = int(item["ordinal"])
        if ordinal != len(attempts_list) + 1:
            raise WorkExecutionPersistenceError(
                "WorkRun Attempt ordinals are not contiguous"
            )
        catalog_json = str(item["catalog_snapshot_json"])
        if _text_hash(catalog_json) != str(item["catalog_snapshot_hash"]):
            raise WorkExecutionPersistenceError("Attempt catalog snapshot is corrupt")
        decision = (
            HostAcceptedAttemptDecision.model_validate_json(
                str(item["decision_json"])
            )
            if item["decision_json"] is not None
            else None
        )
        action = str(item["action"]) if item["action"] is not None else None
        if decision is not None and decision.action.kind != action:
            raise WorkExecutionPersistenceError(
                "Attempt action column and materialized decision disagree"
            )
        committed_output_revision = (
            int(item["committed_output_revision"])
            if item["committed_output_revision"] is not None
            else None
        )
        submitted_output_revision = (
            int(item["submitted_output_revision"])
            if item["submitted_output_revision"] is not None
            else None
        )
        if decision is not None and isinstance(
            decision.action,
            (
                HostMaterializedWriteOutputWindowAction,
                HostMaterializedSubmitOutputWindowAction,
            ),
        ):
            if (
                decision.action.work_run_id != work_run_id
                or committed_output_revision != decision.action.output_revision
            ):
                raise WorkExecutionPersistenceError(
                    "materialized OutputWindow decision binding is corrupt"
                )
            if decision.action.output_revision == output_window.output_revision and (
                decision.action.format is not output_window.format
                or decision.action.size_bytes
                != len(output_window.content.encode("utf-8"))
            ):
                raise WorkExecutionPersistenceError(
                    "materialized OutputWindow decision metadata is corrupt"
                )
        if submitted_output_revision is not None and (
            action != "submit_output_window"
            or submitted_output_revision != committed_output_revision
        ):
            raise WorkExecutionPersistenceError(
                "Attempt submitted OutputWindow binding is corrupt"
            )
        input_verification_request_id = (
            str(item["input_verification_request_id"])
            if item["input_verification_request_id"] is not None
            else None
        )
        input_checkpoint_id = (
            str(item["input_checkpoint_id"])
            if item["input_checkpoint_id"] is not None
            else None
        )
        if (
            input_verification_request_id is not None
            and input_checkpoint_id != input_verification_request_id
        ):
            raise WorkExecutionPersistenceError(
                "Attempt verification feedback and checkpoint disagree"
            )
        try:
            budget_before = WorkRunBudget.model_validate_json(
                str(item["budget_before_json"])
            )
            budget_after = (
                WorkRunBudget.model_validate_json(
                    str(item["budget_after_json"])
                )
                if item["budget_after_json"] is not None
                else None
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "Attempt budget snapshot is invalid"
            ) from exc
        attempt_status = AttemptStatus(str(item["status"]))
        budget_charge_id = (
            str(item["budget_charge_id"])
            if item["budget_charge_id"] is not None
            else None
        )
        if (
            not _same_budget_envelope(budget_before, run.budget)
            or budget_before.attempts_started != ordinal - 1
            or budget_before.active_seconds_consumed
            > run.budget.active_seconds_consumed
        ):
            raise WorkExecutionPersistenceError(
                "Attempt budget-before snapshot disagrees with WorkRun authority"
            )
        if attempt_status is AttemptStatus.ACTIVE:
            if budget_after is not None:
                raise WorkExecutionPersistenceError(
                    "active Attempt cannot have a budget-after snapshot"
                )
        elif (
            budget_after is None
            or not _same_budget_envelope(budget_after, run.budget)
            or budget_after.attempts_started != ordinal
            or budget_after.active_seconds_consumed
            < budget_before.active_seconds_consumed
            or budget_after.active_seconds_consumed
            > run.budget.active_seconds_consumed
        ):
            raise WorkExecutionPersistenceError(
                "closed Attempt budget-after snapshot disagrees with WorkRun authority"
            )
        if (
            attempt_status is AttemptStatus.CLOSED
            and budget_charge_id is None
            and budget_after is not None
            and budget_after.active_seconds_consumed
            != budget_before.active_seconds_consumed
        ):
            raise WorkExecutionPersistenceError(
                "legacy Attempt budget snapshot contains an unledgered active-time charge"
            )
        if item["input_turn_id"] is None:
            raise WorkExecutionPersistenceError(
                "Attempt has no immutable input Turn provenance"
            )
        input_turn_id = str(item["input_turn_id"])
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (input_turn_id, work_run_id),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "Attempt input Turn lost its WorkRun ownership link"
            )
        predecessor_question_attempt_id = (
            str(item["predecessor_question_attempt_id"])
            if item["predecessor_question_attempt_id"] is not None
            else None
        )
        if predecessor_question_attempt_id is not None:
            predecessor = attempts_list[-1] if attempts_list else None
            predecessor_decision = (
                predecessor.decision if predecessor is not None else None
            )
            if (
                predecessor is None
                or predecessor.attempt.attempt_id
                != predecessor_question_attempt_id
                or predecessor.attempt.status is not AttemptStatus.CLOSED
                or predecessor.action != "request_user_input"
                or predecessor_decision is None
                or not isinstance(
                    predecessor_decision.action,
                    RequestUserInputAction,
                )
                or input_turn_id == predecessor.turn_id
            ):
                raise WorkExecutionPersistenceError(
                    "Attempt predecessor question binding is corrupt"
                )
        attempts_list.append(
            # request 记录持有完整反馈。Attempt 历史只保留其精确外键，本读取器从同一
            # SQLite 快照中解引用该外键。
            StoredAttempt(
                attempt=Attempt(
                    attempt_id=str(item["attempt_id"]),
                    work_run_id=work_run_id,
                    ordinal=ordinal,
                    status=attempt_status,
                    submitted_output_revision=submitted_output_revision,
                ),
                turn_id=str(item["turn_id"]),
                input_turn_id=input_turn_id,
                predecessor_question_attempt_id=(
                    predecessor_question_attempt_id
                ),
                action=action,
                decision=decision,
                input_output_revision=int(item["input_output_revision"]),
                input_output_hash=str(item["input_output_hash"]),
                committed_output_revision=committed_output_revision,
                input_checkpoint_id=input_checkpoint_id,
                input_verification_request_id=input_verification_request_id,
                input_verification_result=(
                    _load_nonpass_verification_result(
                        conn,
                        work_run_id=work_run_id,
                        verification_request_id=str(
                            input_verification_request_id
                        ),
                        subject=subject,
                        output_revision=int(item["input_output_revision"]),
                        acceptance_ids=tuple(
                            progress_item.acceptance_id
                            for progress_item in progress.items
                        ),
                        consuming_attempt_ordinal=int(item["ordinal"]),
                    )
                    if item["input_verification_request_id"] is not None
                    else None
                ),
                catalog_snapshot=json.loads(catalog_json),
                budget_before=budget_before,
                budget_after=budget_after,
                budget_charge_id=budget_charge_id,
                close_reason=(
                    str(item["close_reason"])
                    if item["close_reason"] is not None
                    else None
                ),
            )
        )
    attempts = tuple(attempts_list)
    if isinstance(
        subject, AuxiliaryNodeSubject
    ) and _execution_subject_contract_version(conn, row) == "auxiliary_node_v2":
        has_answer_authority = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='insession_auxiliary_v2_waiting_user_answer_bindings'"
            ).fetchone()
            is not None
        )
        if not has_answer_authority:
            if any(
                item.predecessor_question_attempt_id is not None
                for item in attempts
            ):
                raise WorkExecutionPersistenceError(
                    "AuxiliaryGraph answer authority is unavailable"
                )
        else:
            # 回答来源保存在当前 Auxiliary 专用关系中，并在每次 WorkRun 投影时认证；
            # 延续后若改变已接受用户消息，必须在另一次模型 Prompt 前失败。
            from ..auxiliary_graph.auxiliary_continuation import (
                validate_auxiliary_answer_bindings_for_work_run,
            )

            validate_auxiliary_answer_bindings_for_work_run(
                conn,
                run_row=row,
            )
    if run.budget.attempts_started != len(attempts):
        raise WorkExecutionPersistenceError(
            "WorkRun Attempt aggregate disagrees with stored Attempt rows"
        )
    active_attempts = tuple(
        item for item in attempts if item.attempt.status is AttemptStatus.ACTIVE
    )
    projected_current_attempt_id = (
        str(row["current_attempt_id"])
        if row["current_attempt_id"] is not None
        else None
    )
    if projected_current_attempt_id is None:
        if active_attempts:
            raise WorkExecutionPersistenceError(
                "WorkRun has an active Attempt without a current projection"
            )
    elif (
        len(active_attempts) != 1
        or active_attempts[0].attempt.attempt_id != projected_current_attempt_id
        or active_attempts[0] is not attempts[-1]
        or active_attempts[0].turn_id != str(row["updated_turn_id"])
    ):
        raise WorkExecutionPersistenceError(
            "WorkRun current Attempt projection disagrees with Attempt history"
        )
    if run.status is WorkRunStatus.ACTIVE:
        latest_attempt = attempts[-1] if attempts else None
        latest_is_current_submit = bool(
            latest_attempt is not None
            and latest_attempt.attempt.status is AttemptStatus.CLOSED
            and latest_attempt.action == "submit_output_window"
            and latest_attempt.attempt.submitted_output_revision
            == output_window.output_revision
            and latest_attempt.committed_output_revision
            == output_window.output_revision
        )
        verification_pending = run.reason == "verification_pending"
        resolved_nonpass = False
        if latest_is_current_submit and latest_attempt is not None:
            resolved_nonpass = conn.execute(
                "SELECT 1 FROM insession_work_run_verification_requests "
                "WHERE work_run_id=? AND submitted_attempt_id=? "
                "AND output_revision=? AND status='completed' AND all_pass=0",
                (
                    work_run_id,
                    latest_attempt.attempt.attempt_id,
                    output_window.output_revision,
                ),
            ).fetchone() is not None
        if (
            (verification_pending and not latest_is_current_submit)
            or (
                not verification_pending
                and latest_is_current_submit
                and not resolved_nonpass
            )
        ):
            raise WorkExecutionPersistenceError(
                "verification_pending WorkRun/Attempt/OutputWindow binding is corrupt"
            )
        if verification_pending and row["current_attempt_id"] is not None:
            raise WorkExecutionPersistenceError(
                "verification_pending WorkRun cannot retain a current Attempt"
            )
    call_rows = conn.execute(
        "SELECT call.attempt_id, call.ordinal, call.tool_call_id, call.tool_id, "
        "call.tool_version, call.arguments_json, call.arguments_hash, "
        "call.modifies_environment "
        "FROM insession_work_run_tool_calls AS call "
        "JOIN insession_work_run_attempts AS attempt "
        "ON attempt.work_run_id=call.work_run_id AND attempt.attempt_id=call.attempt_id "
        "WHERE call.work_run_id=? ORDER BY attempt.ordinal, call.ordinal",
        (work_run_id,),
    ).fetchall()
    tool_calls_list: list[StoredMaterializedToolCall] = []
    for item in call_rows:
        arguments_json = str(item["arguments_json"])
        if _text_hash(arguments_json) != str(item["arguments_hash"]):
            raise WorkExecutionPersistenceError("ToolCall arguments are corrupt")
        tool_calls_list.append(
            StoredMaterializedToolCall(
                attempt_id=str(item["attempt_id"]),
                ordinal=int(item["ordinal"]),
                call=HostMaterializedToolCall(
                    tool_call_id=str(item["tool_call_id"]),
                    tool_id=str(item["tool_id"]),
                    tool_version=str(item["tool_version"]),
                    arguments=json.loads(arguments_json),
                    modifies_environment=bool(item["modifies_environment"]),
                ),
            )
        )
    tool_calls = tuple(tool_calls_list)
    tool_results = _load_tool_results(conn, work_run_id=work_run_id)
    budget_charges = _load_budget_charges(
        conn,
        session_id=session_id,
        work_run_id=work_run_id,
        current_work_run_revision=run.revision,
        current_budget=run.budget,
        attempt_count=len(attempts),
    )
    _validate_attempt_budget_charge_links(
        attempts=attempts,
        budget_charges=budget_charges,
    )
    link_rows = conn.execute(
        "SELECT turn_id FROM insession_work_run_turn_links "
        "WHERE work_run_id=? ORDER BY link_id",
        (work_run_id,),
    ).fetchall()
    pending_user_question: str | None = None
    if run.status is WorkRunStatus.WAITING_USER:
        for stored_attempt in reversed(attempts):
            decision = stored_attempt.decision
            if decision is not None and isinstance(
                decision.action, RequestUserInputAction
            ):
                pending_user_question = decision.action.question
                break
        if pending_user_question is None:
            raise WorkExecutionPersistenceError(
                "waiting_user WorkRun has no recoverable pending question"
            )
    return StoredWorkRun(
        session_id=str(row["session_id"]),
        work_run=run,
        acceptance_progress=progress,
        output_window=output_window,
        current_attempt_id=(
            str(row["current_attempt_id"])
            if row["current_attempt_id"] is not None
            else None
        ),
        current_verification_request_id=current_verification_request_id,
        node_delivery_id=node_delivery_id,
        auxiliary_node_completion_id=auxiliary_node_completion_id,
        pending_user_question=pending_user_question,
        attempts=attempts,
        tool_calls=tool_calls,
        tool_results=tool_results,
        budget_charges=budget_charges,
        related_turn_ids=tuple(str(item["turn_id"]) for item in link_rows),
    )


def _load_budget_charges(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    work_run_id: str,
    current_work_run_revision: int,
    current_budget: WorkRunBudget,
    attempt_count: int,
) -> tuple[StoredWorkRunBudgetCharge, ...]:
    rows = conn.execute(
        "SELECT budget_charge_id, operation, session_id, turn_id, checkpoint_id, "
        "work_run_revision_before, work_run_revision_after, "
        "window_state_version_before, window_state_version_after, "
        "active_seconds_delta, active_seconds_before, active_seconds_after, "
        "disposition, work_run_status_after, work_run_reason_after, created_at "
        "FROM insession_work_run_budget_charges WHERE work_run_id=? "
        "ORDER BY work_run_revision_after, budget_charge_id",
        (work_run_id,),
    ).fetchall()
    charges: list[StoredWorkRunBudgetCharge] = []
    previous_active_seconds = 0.0
    previous_revision_after = 0
    previous_attempts_started = 0
    for row in rows:
        budget_charge_id = str(row["budget_charge_id"])
        operation = str(row["operation"])
        turn_id = str(row["turn_id"])
        checkpoint_id = str(row["checkpoint_id"])
        revision_before = int(row["work_run_revision_before"])
        revision_after = int(row["work_run_revision_after"])
        window_revision_before = int(row["window_state_version_before"])
        window_revision_after = int(row["window_state_version_after"])
        active_seconds_delta = float(row["active_seconds_delta"])
        active_seconds_before = float(row["active_seconds_before"])
        active_seconds_after = float(row["active_seconds_after"])
        if (
            not math.isfinite(active_seconds_delta)
            or active_seconds_delta <= 0
            or not math.isfinite(active_seconds_before)
            or active_seconds_before < 0
            or not math.isfinite(active_seconds_after)
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge payload is invalid"
            )
        try:
            stored_disposition = WorkRunBudgetDisposition(
                str(row["disposition"])
            )
            stored_work_run_status = WorkRunStatus(
                str(row["work_run_status_after"])
            )
        except (TypeError, ValueError) as exc:
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge payload is invalid"
            ) from exc
        receipt = conn.execute(
            "SELECT operation, session_id, work_run_id, payload_hash, "
            "result_json, created_at FROM insession_work_run_apply_receipts "
            "WHERE apply_id=?",
            (budget_charge_id,),
        ).fetchone()
        if receipt is None:
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge is missing its exact replay receipt"
            )
        try:
            if operation == "charge_active_time":
                stored_result: BaseModel = (
                    WorkRunBudgetChargeMutationResult.model_validate_json(
                        str(receipt["result_json"])
                    )
                )
                stored_transition = stored_result.transition
            elif operation in {
                "commit_attempt_decision",
                "commit_output_action",
                "close_attempt",
            }:
                stored_result = WorkExecutionMutationResult.model_validate_json(
                    str(receipt["result_json"])
                )
                stored_transition = stored_result.budget_transition
            elif operation in {
                "commit_verification_result",
                "interrupt_verification",
            }:
                from personagraph.l2.work_run import TaskNodeVerificationMutationResult

                stored_result = (
                    TaskNodeVerificationMutationResult.model_validate_json(
                        str(receipt["result_json"])
                    )
                )
                stored_transition = stored_result.budget_transition
            else:
                raise ValueError("unknown budget charge operation")
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge receipt is invalid"
            ) from exc
        if stored_transition is None:
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge receipt has no transition"
            )
        expected_reason = (
            str(row["work_run_reason_after"])
            if row["work_run_reason_after"] is not None
            else None
        )
        payload_hash_matches = True
        if operation == "charge_active_time":
            payload_hash_matches = str(receipt["payload_hash"]) == _payload_hash(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "work_run_id": work_run_id,
                    "checkpoint_id": checkpoint_id,
                    "active_seconds_delta": active_seconds_delta,
                    "expected_work_run_revision": revision_before,
                    "expected_window_revision": window_revision_before,
                }
            )
        if (
            str(row["session_id"]) != session_id
            or (
                operation != "charge_active_time"
                and checkpoint_id
                != _settlement_budget_checkpoint_id(operation, budget_charge_id)
            )
            or revision_before < previous_revision_after
            or revision_after != revision_before + 1
            or revision_after > current_work_run_revision
            or window_revision_after != window_revision_before + 1
            or active_seconds_before != previous_active_seconds
            or active_seconds_after
            != active_seconds_before + active_seconds_delta
            or str(receipt["operation"]) != operation
            or str(receipt["session_id"]) != session_id
            or str(receipt["work_run_id"]) != work_run_id
            or not payload_hash_matches
            or str(receipt["created_at"]) != str(row["created_at"])
            or stored_result.status != "applied"
            or stored_result.work_run_id != work_run_id
            or stored_result.work_run_revision != revision_after
            or stored_result.work_run_status is not stored_work_run_status
            or stored_result.work_run_reason != expected_reason
            or stored_result.window_state_version != window_revision_after
            or not _same_budget_envelope(
                stored_transition.budget_before,
                current_budget,
            )
            or not _same_budget_envelope(
                stored_transition.budget_after,
                current_budget,
            )
            or stored_transition.budget_before.attempts_started
            < previous_attempts_started
            or stored_transition.budget_before.attempts_started
            > attempt_count
            or stored_transition.active_seconds_delta
            != active_seconds_delta
            or stored_transition.budget_before.active_seconds_consumed
            != active_seconds_before
            or stored_transition.budget_after.active_seconds_consumed
            != active_seconds_after
            or stored_transition.disposition is not stored_disposition
            or (
                operation == "charge_active_time"
                and (
                    stored_result.budget_charge_id != budget_charge_id
                    or stored_result.checkpoint_id != checkpoint_id
                    or _budget_charge_status_projection(stored_disposition)
                    != (stored_work_run_status, expected_reason)
                )
            )
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun budget charge ledger is not contiguous"
            )
        charges.append(
            StoredWorkRunBudgetCharge(
                budget_charge_id=budget_charge_id,
                operation=operation,
                turn_id=turn_id,
                checkpoint_id=checkpoint_id,
                work_run_revision_before=revision_before,
                work_run_revision_after=revision_after,
                window_state_version_before=window_revision_before,
                window_state_version_after=window_revision_after,
                work_run_status_after=stored_work_run_status,
                work_run_reason_after=expected_reason,
                transition=stored_transition,
            )
        )
        previous_active_seconds = active_seconds_after
        previous_revision_after = revision_after
        previous_attempts_started = (
            stored_transition.budget_after.attempts_started
        )
    if previous_active_seconds != current_budget.active_seconds_consumed:
        raise WorkExecutionPersistenceError(
            "WorkRun aggregate active time disagrees with its charge ledger"
        )
    return tuple(charges)


def _budget_charge_status_projection(
    disposition: WorkRunBudgetDisposition,
) -> tuple[WorkRunStatus, str | None]:
    if disposition is WorkRunBudgetDisposition.HARD_LIMIT_REACHED:
        return WorkRunStatus.FAILED, "work_run_limit_reached"
    if disposition is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED:
        return WorkRunStatus.TURN_LIMIT_REACHED, "turn_limit_reached"
    return WorkRunStatus.ACTIVE, None


def _require_budgeted_receipt_charge(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    work_run_revision: int,
    work_run_status: WorkRunStatus,
    work_run_reason: str | None,
    window_state_version: int,
    budget_transition: WorkRunBudgetTransition,
) -> StoredWorkRunBudgetCharge:
    run_row = conn.execute(
        "SELECT * FROM insession_work_runs WHERE work_run_id=? AND session_id=?",
        (work_run_id, session_id),
    ).fetchone()
    if run_row is None:
        raise WorkExecutionPersistenceError(
            "budgeted receipt lost its WorkRun authority"
        )
    current_budget = _budget_from_row(run_row)
    attempt_count = _require_attempt_row_aggregate(
        conn,
        work_run_id=work_run_id,
        expected_attempts_started=current_budget.attempts_started,
    )
    charges = _load_budget_charges(
        conn,
        session_id=session_id,
        work_run_id=work_run_id,
        current_work_run_revision=int(run_row["revision"]),
        current_budget=current_budget,
        attempt_count=attempt_count,
    )
    charge = next(
        (item for item in charges if item.budget_charge_id == apply_id),
        None,
    )
    if (
        charge is None
        or charge.operation != operation
        or charge.work_run_revision_after != work_run_revision
        or charge.work_run_status_after is not work_run_status
        or charge.work_run_reason_after != work_run_reason
        or charge.window_state_version_after != window_state_version
        or charge.transition != budget_transition
    ):
        raise WorkExecutionPersistenceError(
            "budgeted receipt disagrees with its active-time ledger"
        )
    return charge


def _validate_attempt_budget_charge_links(
    *,
    attempts: tuple[StoredAttempt, ...],
    budget_charges: tuple[StoredWorkRunBudgetCharge, ...],
) -> None:
    charges_by_id = {item.budget_charge_id: item for item in budget_charges}
    linked_ids: set[str] = set()
    expected_actions = {
        "commit_attempt_decision": {
            "request_user_input",
            "request_task_graph_revision",
        },
        "commit_output_action": {
            "write_output_window",
            "submit_output_window",
        },
        "close_attempt": {"call_tools"},
    }
    for stored_attempt in attempts:
        charge_id = stored_attempt.budget_charge_id
        if charge_id is None:
            continue
        charge = charges_by_id.get(charge_id)
        budget_after = stored_attempt.budget_after
        if (
            charge is None
            or charge.operation not in expected_actions
            or stored_attempt.action not in expected_actions[charge.operation]
            or budget_after is None
            or charge.transition.budget_before.attempts_started
            != stored_attempt.attempt.ordinal
            or charge.transition.budget_after.attempts_started
            != stored_attempt.attempt.ordinal
            or charge.transition.budget_before.active_seconds_consumed
            != stored_attempt.budget_before.active_seconds_consumed
            or charge.transition.budget_after.active_seconds_consumed
            != budget_after.active_seconds_consumed
        ):
            raise WorkExecutionPersistenceError(
                "Attempt budget snapshot disagrees with its settlement charge"
            )
        linked_ids.add(charge_id)
    expected_linked_ids = {
        item.budget_charge_id
        for item in budget_charges
        if item.operation in expected_actions
    }
    if linked_ids != expected_linked_ids:
        raise WorkExecutionPersistenceError(
            "Attempt settlement budget charge linkage is incomplete"
        )


def _load_tool_results(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    attempt_id: str | None = None,
) -> tuple[ToolResult, ...]:
    attempt_filter = " AND result.attempt_id=?" if attempt_id is not None else ""
    params: tuple[object, ...] = (
        (work_run_id, attempt_id)
        if attempt_id is not None
        else (work_run_id,)
    )
    rows = conn.execute(
        "SELECT result.work_run_id, result.attempt_id, result.tool_result_id, "
        "result.tool_call_id, result.ordinal, result.status, result.result_json, "
        "call.modifies_environment "
        "FROM insession_work_run_tool_results AS result "
        "JOIN insession_work_run_attempts AS attempt "
        "ON attempt.work_run_id=result.work_run_id "
        "AND attempt.attempt_id=result.attempt_id "
        "JOIN insession_work_run_tool_calls AS call "
        "ON call.work_run_id=result.work_run_id "
        "AND call.attempt_id=result.attempt_id "
        "AND call.tool_call_id=result.tool_call_id "
        "WHERE result.work_run_id=?"
        + attempt_filter
        + " ORDER BY attempt.ordinal, result.ordinal",
        params,
    ).fetchall()
    parsed_results: list[ToolResult] = []
    for row in rows:
        parsed = ToolResult.model_validate_json(str(row["result_json"]))
        if (
            str(row["work_run_id"]) != work_run_id
            or parsed.tool_result_id != str(row["tool_result_id"])
            or parsed.tool_call_id != str(row["tool_call_id"])
            or parsed.attempt_id != str(row["attempt_id"])
            or parsed.ordinal != int(row["ordinal"])
            or parsed.status.value != str(row["status"])
        ):
            raise WorkExecutionPersistenceError(
                "ToolResult row and typed payload disagree"
            )
        if (
            parsed.status is ToolResultStatus.COMPLETION_UNCONFIRMED
            and not bool(row["modifies_environment"])
        ):
            raise WorkExecutionPersistenceError(
                "completion_unconfirmed requires an environment-modifying ToolCall"
            )
        parsed_results.append(parsed)
    return tuple(parsed_results)


def _validate_current_verification_projection(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    run: WorkRun,
) -> str | None:
    """仅解析当前可恢复的验证请求引用。"""

    pointer = (
        str(run_row["current_verification_request_id"])
        if run_row["current_verification_request_id"] is not None
        else None
    )
    is_pending = (
        run.status is WorkRunStatus.ACTIVE
        and run.reason == "verification_pending"
    )
    is_interrupted = (
        run.status is WorkRunStatus.INTERRUPTED
        and run.reason == "verification_technical_failure"
    )
    if pointer is None:
        if is_interrupted:
            raise WorkExecutionPersistenceError(
                "interrupted verification WorkRun has no resumable request"
            )
        return None
    if not (is_pending or is_interrupted):
        raise WorkExecutionPersistenceError(
            "WorkRun verification request pointer disagrees with its lifecycle"
        )
    expected_status = "pending" if is_pending else "interrupted"
    # 延迟导入以避免模块初始化循环。每个带标签 subject 都持有独立的规范化请求
    # 加载器与不可变 binding digest。
    if isinstance(run.subject, TaskNodeSubject):
        from .work_verification import (
            _load_request_row,
            _record_from_row,
            _revalidate_binding,
            _validate_request_lifecycle,
        )

        request_row = _load_request_row(
            conn,
            session_id=str(run_row["session_id"]),
            verification_request_id=pointer,
        )
        record = _record_from_row(request_row)
        _revalidate_binding(conn, request_row)
        _validate_request_lifecycle(conn, record)
    else:
        from ..auxiliary_graph.auxiliary_node_execution_bindings import (
            _auxiliary_record_from_row,
            _load_auxiliary_request_row,
            _revalidate_auxiliary_request_binding,
        )

        if _execution_subject_contract_version(conn, run_row) != "auxiliary_node_v2":
            raise WorkExecutionPersistenceError(
                "Auxiliary WorkRun execution-subject contract is unsupported"
            )
        request_row = _load_auxiliary_request_row(
            conn,
            session_id=str(run_row["session_id"]),
            verification_request_id=pointer,
        )
        _revalidate_auxiliary_request_binding(conn, request_row)
        record = _auxiliary_record_from_row(request_row)
    if (
        record.request.work_run_id != run.work_run_id
        or record.request.subject != run.subject
        or record.request.status.value != expected_status
    ):
        raise WorkExecutionPersistenceError(
            "WorkRun current verification request authority is corrupt"
        )
    return pointer


def _validate_delivery_projection(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    run: WorkRun,
    output_window: OutputWindow,
) -> str | None:
    row = conn.execute(
        "SELECT delivery_id FROM insession_task_node_deliveries WHERE work_run_id=?",
        (run.work_run_id,),
    ).fetchone()
    frozen_row = conn.execute(
        "SELECT frozen_at FROM insession_work_run_output_windows WHERE work_run_id=?",
        (run.work_run_id,),
    ).fetchone()
    if frozen_row is None:
        raise WorkExecutionPersistenceError("WorkRun OutputWindow is missing")
    expects_delivery = (
        run.status is WorkRunStatus.COMPLETED
        and run.reason == "verification_passed"
    )
    if (row is not None) != expects_delivery:
        raise WorkExecutionPersistenceError(
            "WorkRun delivery and terminal verification state disagree"
        )
    if row is None:
        if frozen_row["frozen_at"] is not None:
            raise WorkExecutionPersistenceError(
                "unverified WorkRun cannot own a frozen OutputWindow"
            )
        return None
    from .work_verification import _load_task_node_delivery

    delivery_id = str(row["delivery_id"])
    resolved = _load_task_node_delivery(
        conn,
        session_id=str(run_row["session_id"]),
        delivery_id=delivery_id,
    )
    if (
        resolved.delivery.work_run_id != run.work_run_id
        or resolved.delivery.subject != run.subject
        or resolved.delivery.output_revision != output_window.output_revision
        or resolved.output_window != output_window
        or run_row["current_verification_request_id"] is not None
        or run_row["current_attempt_id"] is not None
    ):
        raise WorkExecutionPersistenceError(
            "verified delivery authority is corrupt"
        )
    return delivery_id


def _validate_auxiliary_completion_projection(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    run: WorkRun,
    output_window: OutputWindow,
) -> str | None:
    if _execution_subject_contract_version(conn, run_row) != "auxiliary_node_v2":
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun execution-subject contract is unsupported"
        )
    row = conn.execute(
        "SELECT completion_id, output_revision, completion_json, completion_sha256 "
        "FROM insession_auxiliary_node_completions_v2 WHERE work_run_id=?",
        (run.work_run_id,),
    ).fetchone()
    frozen_row = conn.execute(
        "SELECT frozen_at FROM insession_work_run_output_windows WHERE work_run_id=?",
        (run.work_run_id,),
    ).fetchone()
    if frozen_row is None:
        raise WorkExecutionPersistenceError("WorkRun OutputWindow is missing")
    expects_completion = (
        run.status is WorkRunStatus.COMPLETED
        and run.reason == "verification_passed"
    )
    if (row is not None) != expects_completion:
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun completion and terminal state disagree"
        )
    if row is None:
        if frozen_row["frozen_at"] is not None:
            raise WorkExecutionPersistenceError(
                "unverified Auxiliary WorkRun cannot own a frozen OutputWindow"
            )
        return None
    if (
        frozen_row["frozen_at"] is None
        or int(row["output_revision"]) != output_window.output_revision
        or run_row["current_verification_request_id"] is not None
        or run_row["current_attempt_id"] is not None
    ):
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun completion binding is corrupt"
        )
    completion_json = str(row["completion_json"])
    if (
        _text_hash(completion_json) != str(row["completion_sha256"])
        or json.loads(completion_json).get("execution_subject_id")
        != str(run_row["execution_subject_id"])
    ):
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun completion hash is corrupt"
        )
    return str(row["completion_id"])


def _subject_from_run_row(row: sqlite3.Row) -> ExecutionSubject:
    kind = str(row["subject_kind"])
    if kind == "task_node":
        try:
            return TaskNodeSubject(
                task_id=str(row["insession_task_id"]),
                graph_revision=int(row["graph_revision"]),
                node_id=str(row["insession_task_node_id"]),
                node_revision=int(row["node_revision"]),
            )
        except (TypeError, ValueError) as exc:
            raise WorkExecutionPersistenceError(
                "stored TaskNode WorkRun subject is corrupt"
            ) from exc
    if kind == "auxiliary_node":
        try:
            return AuxiliaryNodeSubject(
                task_id=str(row["insession_task_id"]),
                auxiliary_graph_id=str(row["auxiliary_graph_id"]),
                auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
                node_id=str(row["auxiliary_node_id"]),
                node_revision=int(row["node_revision"]),
            )
        except (TypeError, ValueError) as exc:
            raise WorkExecutionPersistenceError(
                "stored AuxiliaryNode WorkRun subject is corrupt"
            ) from exc
    raise WorkExecutionPersistenceError("stored WorkRun subject kind is unsupported")


def _stable_execution_subject_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(
        [prefix, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _execution_subject_contract_version(
    conn: sqlite3.Connection,
    run_row: sqlite3.Row,
) -> str:
    row = conn.execute(
        "SELECT subject_contract_version FROM insession_execution_subjects "
        "WHERE execution_subject_id=? AND session_id=? "
        "AND insession_task_id=? AND subject_kind=?",
        (
            str(run_row["execution_subject_id"]),
            str(run_row["session_id"]),
            str(run_row["insession_task_id"]),
            str(run_row["subject_kind"]),
        ),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "WorkRun execution-subject registry binding is missing"
        )
    return str(row["subject_contract_version"])


def _ensure_task_node_execution_subject(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
    created_at: str,
) -> str:
    binding_id = _stable_execution_subject_id(
        "taskbind",
        session_id,
        subject.task_id,
        subject.graph_revision,
        subject.node_id,
        subject.node_revision,
    )
    execution_subject_id = _stable_execution_subject_id(
        "execsubject",
        "task_node_v1",
        session_id,
        subject.task_id,
        subject.graph_revision,
        subject.node_id,
        subject.node_revision,
    )
    conn.execute(
        "INSERT OR IGNORE INTO insession_task_node_execution_subject_bindings "
        "(binding_id, session_id, insession_task_id, graph_revision, "
        "insession_task_node_id, node_revision, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            binding_id,
            session_id,
            subject.task_id,
            subject.graph_revision,
            subject.node_id,
            subject.node_revision,
            created_at,
        ),
    )
    conn.execute(
        "INSERT OR IGNORE INTO insession_execution_subjects "
        "(execution_subject_id, session_id, insession_task_id, subject_kind, "
        "subject_contract_version, task_node_binding_id, "
        "auxiliary_v2_binding_id, created_at) "
        "VALUES (?, ?, ?, 'task_node', 'task_node_v1', ?, NULL, ?)",
        (
            execution_subject_id,
            session_id,
            subject.task_id,
            binding_id,
            created_at,
        ),
    )
    row = conn.execute(
        "SELECT subject_kind, subject_contract_version, task_node_binding_id "
        "FROM insession_execution_subjects WHERE execution_subject_id=? "
        "AND session_id=? AND insession_task_id=?",
        (execution_subject_id, session_id, subject.task_id),
    ).fetchone()
    if (
        row is None
        or str(row["subject_kind"]) != "task_node"
        or str(row["subject_contract_version"]) != "task_node_v1"
        or str(row["task_node_binding_id"]) != binding_id
    ):
        raise WorkExecutionPersistenceError(
            "TaskNode execution-subject authority collides with another binding"
        )
    return execution_subject_id


def _load_latest_nonpass_verification_result(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    subject: ExecutionSubject,
    output_revision: int,
    acceptance_ids: tuple[str, ...],
    consuming_attempt_ordinal: int,
) -> tuple[str, NodeVerificationResult] | None:
    row = conn.execute(
        "SELECT request.*, submitted.ordinal AS submitted_ordinal, "
        "submitted.status AS submitted_status, submitted.action AS submitted_action, "
        "submitted.submitted_output_revision AS actual_submitted_output_revision "
        "FROM insession_work_run_verification_requests AS request "
        "JOIN insession_work_run_attempts AS submitted "
        "ON submitted.work_run_id=request.work_run_id "
        "AND submitted.attempt_id=request.submitted_attempt_id "
        "WHERE request.work_run_id=? AND request.status='completed' "
        "AND request.all_pass=0 AND request.output_revision=? "
        "ORDER BY submitted.ordinal DESC LIMIT 1",
        (work_run_id, output_revision),
    ).fetchone()
    if row is None:
        return None
    request_id = str(row["verification_request_id"])
    return request_id, _parse_nonpass_verification_result(
        row,
        work_run_id=work_run_id,
        verification_request_id=request_id,
        subject=subject,
        output_revision=output_revision,
        acceptance_ids=acceptance_ids,
        consuming_attempt_ordinal=consuming_attempt_ordinal,
    )


def _load_nonpass_verification_result(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    verification_request_id: str,
    subject: ExecutionSubject,
    output_revision: int,
    acceptance_ids: tuple[str, ...],
    consuming_attempt_ordinal: int,
) -> NodeVerificationResult:
    row = conn.execute(
        "SELECT request.*, submitted.ordinal AS submitted_ordinal, "
        "submitted.status AS submitted_status, submitted.action AS submitted_action, "
        "submitted.submitted_output_revision AS actual_submitted_output_revision "
        "FROM insession_work_run_verification_requests AS request "
        "JOIN insession_work_run_attempts AS submitted "
        "ON submitted.work_run_id=request.work_run_id "
        "AND submitted.attempt_id=request.submitted_attempt_id "
        "WHERE request.work_run_id=? AND request.verification_request_id=?",
        (work_run_id, verification_request_id),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "Attempt verification feedback request is missing"
        )
    return _parse_nonpass_verification_result(
        row,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        subject=subject,
        output_revision=output_revision,
        acceptance_ids=acceptance_ids,
        consuming_attempt_ordinal=consuming_attempt_ordinal,
    )


def _parse_nonpass_verification_result(
    row: sqlite3.Row,
    *,
    work_run_id: str,
    verification_request_id: str,
    subject: ExecutionSubject,
    output_revision: int,
    acceptance_ids: tuple[str, ...],
    consuming_attempt_ordinal: int,
) -> NodeVerificationResult:
    try:
        result = NodeVerificationResult.model_validate_json(str(row["result_json"]))
        acceptance_ids_raw = json.loads(str(row["acceptance_ids_json"]))
        stored_acceptance_ids = tuple(acceptance_ids_raw)
        stored_all_pass = int(row["all_pass"])
        stored_subject = _subject_from_run_row(row)
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "Attempt verification feedback payload is corrupt"
        ) from exc
    if (
        str(row["status"]) != "completed"
        or stored_all_pass != 0
        or result.all_pass
        or str(row["work_run_id"]) != work_run_id
        or str(row["verification_request_id"]) != verification_request_id
        or int(row["output_revision"]) != output_revision
        or stored_subject != subject
        or stored_acceptance_ids != acceptance_ids
        or result.verification_request_id != verification_request_id
        or result.verification_request_revision != int(row["request_revision"]) - 1
        or result.work_run_id != work_run_id
        or result.locked_work_run_revision != int(row["locked_work_run_revision"])
        or result.subject != subject
        or result.submitted_attempt_id != str(row["submitted_attempt_id"])
        or result.output_revision != output_revision
        or result.acceptance_progress_revision
        != int(row["acceptance_progress_revision"])
        or str(row["submitted_status"]) != "closed"
        or str(row["submitted_action"]) != "submit_output_window"
        or int(row["actual_submitted_output_revision"]) != int(row["output_revision"])
        or int(row["submitted_ordinal"]) >= consuming_attempt_ordinal
        or tuple(item.acceptance_id for item in result.acceptance_results)
        != stored_acceptance_ids
    ):
        raise WorkExecutionPersistenceError(
            "Attempt verification feedback binding is corrupt"
        )
    return result


def _build_mutation_result(
    conn: sqlite3.Connection,
    *,
    status: Literal["applied", "replayed"],
    work_run_id: str,
    window_state_version: int,
    attempt_id: str | None = None,
    new_tool_call_ids: tuple[str, ...] = (),
    tool_result_id: str | None = None,
    turn_work_run_link_revision: int | None = None,
    budget_transition: WorkRunBudgetTransition | None = None,
) -> WorkExecutionMutationResult:
    run_row = conn.execute(
        "SELECT revision, status, reason, current_attempt_id "
        "FROM insession_work_runs WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    progress_row = conn.execute(
        "SELECT progress_revision FROM insession_work_run_acceptance_progress "
        "WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    output_row = conn.execute(
        "SELECT output_revision FROM insession_work_run_output_windows "
        "WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if run_row is None or progress_row is None or output_row is None:
        raise WorkExecutionPersistenceError("WorkRun result projection is incomplete")
    attempt: Attempt | None = None
    if attempt_id is not None:
        attempt_row = conn.execute(
            "SELECT ordinal, status, submitted_output_revision "
            "FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=?",
            (work_run_id, attempt_id),
        ).fetchone()
        if attempt_row is None:
            raise WorkExecutionPersistenceError("Attempt result projection is missing")
        attempt = Attempt(
            attempt_id=attempt_id,
            work_run_id=work_run_id,
            ordinal=int(attempt_row["ordinal"]),
            status=AttemptStatus(str(attempt_row["status"])),
            submitted_output_revision=(
                int(attempt_row["submitted_output_revision"])
                if attempt_row["submitted_output_revision"] is not None
                else None
            ),
        )
    return WorkExecutionMutationResult(
        status=status,
        work_run_id=work_run_id,
        work_run_revision=int(run_row["revision"]),
        work_run_status=WorkRunStatus(str(run_row["status"])),
        work_run_reason=(
            str(run_row["reason"]) if run_row["reason"] is not None else None
        ),
        current_attempt_id=(
            str(run_row["current_attempt_id"])
            if run_row["current_attempt_id"] is not None
            else None
        ),
        acceptance_progress_revision=int(progress_row["progress_revision"]),
        output_window_revision=int(output_row["output_revision"]),
        attempt=attempt,
        new_tool_call_ids=new_tool_call_ids,
        tool_result_id=tool_result_id,
        turn_work_run_link_revision=turn_work_run_link_revision,
        window_state_version=window_state_version,
        budget_transition=budget_transition,
    )


def _require_active_attempt_resume_authority(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: ExecutionSubject,
    run_row: sqlite3.Row,
) -> None:
    task = conn.execute(
        "SELECT current_graph_revision, current_status FROM insession_tasks "
        "WHERE insession_task_id=? AND session_id=?",
        (subject.task_id, session_id),
    ).fetchone()
    if task is None or str(task["current_status"]) != "active":
        raise WorkExecutionPersistenceError(
            "Task authority no longer permits Attempt resume"
        )
    if isinstance(subject, TaskNodeSubject):
        if (
            task["current_graph_revision"] is None
            or int(task["current_graph_revision"]) != subject.graph_revision
        ):
            raise WorkExecutionPersistenceError(
                "TaskGraph authority no longer permits Attempt resume"
            )
        node = conn.execute(
            "SELECT state.status FROM insession_task_graph_nodes AS node "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "AND node.insession_task_node_id=? AND node.node_revision=?",
            (
                subject.task_id,
                subject.graph_revision,
                subject.node_id,
                subject.node_revision,
            ),
        ).fetchone()
        if node is None or str(node["status"]) != "active":
            raise WorkExecutionPersistenceError(
                "TaskNode authority no longer permits Attempt resume"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_graph_edges AS edge "
            "JOIN insession_task_graph_nodes AS child "
            "ON child.insession_task_id=edge.insession_task_id "
            "AND child.graph_revision=edge.graph_revision "
            "AND child.insession_task_node_id=edge.child_insession_task_node_id "
            "LEFT JOIN insession_task_node_states AS child_state "
            "ON child_state.insession_task_id=child.insession_task_id "
            "AND child_state.insession_task_node_id=child.insession_task_node_id "
            "AND child_state.node_revision=child.node_revision "
            "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
            "AND edge.parent_insession_task_node_id=? "
            "AND (child_state.status IS NULL OR child_state.status!='completed') "
            "LIMIT 1",
            (subject.task_id, subject.graph_revision, subject.node_id),
        ).fetchone() is not None:
            raise WorkExecutionPersistenceError(
                "TaskNode is no longer ready for Attempt resume"
            )
        return
    from ..auxiliary_graph.auxiliary_continuation import (
        _load_exact_auxiliary_authority,
        _require_authority_statuses,
        _require_auxiliary_authority_shape,
    )

    authority = _load_exact_auxiliary_authority(
        conn,
        session_id=session_id,
        work_run_id=str(run_row["work_run_id"]),
    )
    _require_auxiliary_authority_shape(authority)
    if _subject_from_run_row(authority) != subject:
        raise WorkExecutionPersistenceError(
            "AuxiliaryNode authority no longer permits Attempt resume"
        )
    _require_authority_statuses(
        authority,
        task_status="active",
        node_status="active",
        goal_status="active",
        revision_status="active",
    )


def _require_readonly_recovery_batch(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    attempt_id: str,
    decision: HostAcceptedAttemptDecision,
    catalog_snapshot: Mapping[str, Any],
    allow_protected_recovery: bool = False,
) -> None:
    """证明已决批次对其所属 bridge 而言精确且安全。"""

    if not isinstance(decision.action, HostMaterializedCallToolsAction):
        raise WorkExecutionPersistenceError(
            "decided tool recovery requires call_tools authority"
        )
    materialized_calls = decision.action.calls
    call_rows = conn.execute(
        "SELECT ordinal, tool_call_id, tool_id, tool_version, "
        "modifies_environment, arguments_hash, arguments_json "
        "FROM insession_work_run_tool_calls "
        "WHERE work_run_id=? AND attempt_id=? ORDER BY ordinal",
        (work_run_id, attempt_id),
    ).fetchall()
    if len(call_rows) != len(materialized_calls) or not call_rows:
        raise WorkExecutionPersistenceError(
            "decided tool recovery requires the complete materialized call batch"
        )

    entries = catalog_snapshot.get("entries")
    revision = catalog_snapshot.get("revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(entries, list)
    ):
        raise WorkExecutionPersistenceError(
            "decided tool recovery catalog descriptor is invalid"
        )

    for ordinal, (row, call) in enumerate(
        zip(call_rows, materialized_calls, strict=True),
        start=1,
    ):
        arguments_json = str(row["arguments_json"])
        try:
            json.loads(arguments_json)
        except (TypeError, ValueError) as exc:
            raise WorkExecutionPersistenceError(
                "decided tool recovery arguments are corrupt"
            ) from exc
        if (
            int(row["ordinal"]) != ordinal
            or str(row["tool_call_id"]) != call.tool_call_id
            or str(row["tool_id"]) != call.tool_id
            or str(row["tool_version"]) != call.tool_version
            or bool(row["modifies_environment"]) != call.modifies_environment
            or _text_hash(arguments_json) != str(row["arguments_hash"])
            or arguments_json != _canonical_json(call.arguments)
        ):
            raise WorkExecutionPersistenceError(
                "decided tool recovery call rows disagree with the accepted decision"
            )
        if call.modifies_environment and not allow_protected_recovery:
            raise WorkExecutionPersistenceError(
                "decided tool recovery requires non-modifying ToolCalls"
            )

        matches: list[Mapping[str, Any]] = []
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                raise WorkExecutionPersistenceError(
                    "decided tool recovery catalog entry is invalid"
                )
            if str(raw_entry.get("status") or "") not in {"active", "deprecated"}:
                continue
            registration = raw_entry.get("registration")
            if not isinstance(registration, Mapping):
                raise WorkExecutionPersistenceError(
                    "decided tool recovery catalog registration is invalid"
                )
            spec = registration.get("spec")
            if not isinstance(spec, Mapping):
                raise WorkExecutionPersistenceError(
                    "decided tool recovery ToolSpec descriptor is invalid"
                )
            if str(spec.get("tool_id") or "") == call.tool_id:
                matches.append(registration)
        if len(matches) != 1:
            raise WorkExecutionPersistenceError(
                "decided tool recovery call has unknown or ambiguous catalog authority"
            )
        registration = matches[0]
        effects = registration.get("effects")
        effect_count = registration.get("effect_count")
        read_only_effects = (
            isinstance(effects, list)
            and bool(effects)
            and all(
                isinstance(effect, Mapping)
                and str(effect.get("action") or "") in {"read", "search"}
                for effect in effects
            )
        )
        if (
            str(registration.get("implementation_version") or "")
            != call.tool_version
            or isinstance(effect_count, bool)
            or not isinstance(effect_count, int)
            or not isinstance(effects, list)
            or effect_count != len(effects)
            or not effects
        ):
            raise WorkExecutionPersistenceError(
                "decided tool recovery requires exact catalog authority"
            )
        if call.modifies_environment:
            if read_only_effects:
                raise WorkExecutionPersistenceError(
                    "protected recovery call disagrees with its modifying catalog authority"
                )
        elif not read_only_effects:
            raise WorkExecutionPersistenceError(
                "decided tool recovery requires exact READ/SEARCH catalog authority"
            )

    results = _load_tool_results(
        conn,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
    )
    if len(results) > len(materialized_calls):
        raise WorkExecutionPersistenceError(
            "decided tool recovery has more results than materialized calls"
        )
    if tuple(item.ordinal for item in results) != tuple(
        range(1, len(results) + 1)
    ):
        raise WorkExecutionPersistenceError(
            "decided tool recovery requires a durable ToolResult prefix"
        )
    calls_by_id = {call.tool_call_id: call for call in materialized_calls}
    for result in results:
        call = calls_by_id.get(result.tool_call_id)
        if (
            call is None
            or result.attempt_id != attempt_id
            or result.ordinal < 1
            or result.ordinal > len(materialized_calls)
            or materialized_calls[result.ordinal - 1].tool_call_id
            != result.tool_call_id
        ):
            raise WorkExecutionPersistenceError(
                "decided tool recovery ToolResult is outside the accepted batch"
            )
        if (
            result.status is ToolResultStatus.COMPLETION_UNCONFIRMED
            and (not allow_protected_recovery or not call.modifies_environment)
        ):
            raise WorkExecutionPersistenceError(
                "decided tool recovery cannot resume completion-uncertain results"
            )

    decision_receipts = 0
    for receipt in conn.execute(
        "SELECT result_json FROM insession_work_run_apply_receipts "
        "WHERE work_run_id=? AND operation='commit_attempt_decision'",
        (work_run_id,),
    ).fetchall():
        try:
            projection = WorkExecutionMutationResult.model_validate_json(
                str(receipt["result_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "decided tool recovery decision receipt is corrupt"
            ) from exc
        if (
            projection.status == "applied"
            and projection.work_run_id == work_run_id
            and projection.attempt is not None
            and projection.attempt.attempt_id == attempt_id
            and projection.new_tool_call_ids
            == tuple(call.tool_call_id for call in materialized_calls)
        ):
            decision_receipts += 1
    if decision_receipts != 1:
        raise WorkExecutionPersistenceError(
            "decided tool recovery lost its exact decision receipt projection"
        )

    receipt_results: list[WorkExecutionMutationResult] = []
    for receipt in conn.execute(
        "SELECT result_json FROM insession_work_run_apply_receipts "
        "WHERE work_run_id=? AND operation='append_tool_result'",
        (work_run_id,),
    ).fetchall():
        try:
            projection = WorkExecutionMutationResult.model_validate_json(
                str(receipt["result_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "decided tool recovery result receipt is corrupt"
            ) from exc
        if (
            projection.status == "applied"
            and projection.work_run_id == work_run_id
            and projection.attempt is not None
            and projection.attempt.attempt_id == attempt_id
            and projection.tool_result_id is not None
        ):
            receipt_results.append(projection)
    if any(
        sum(
            projection.tool_result_id == item.tool_result_id
            for projection in receipt_results
        )
        != 1
        for item in results
    ):
        raise WorkExecutionPersistenceError(
            "decided tool recovery ToolResult lost its exact append receipt"
        )


def _require_waiting_user_continuation_authority(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
    expected_task_state_version: int,
    expected_node_state_version: int,
) -> None:
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (session_id, subject.task_id),
    ).fetchone()
    if (
        task is None
        or str(task["current_status"]) not in {"active", "awaiting_user"}
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != subject.graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "TaskGraph authority no longer permits waiting-user continuation"
        )
    actual_task_version = int(task["state_version"])
    if actual_task_version != expected_task_state_version:
        raise WorkExecutionRevisionConflict(
            expected=expected_task_state_version,
            actual=actual_task_version,
        )
    node = conn.execute(
        "SELECT state.status, state.state_version "
        "FROM insession_task_graph_nodes AS node "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=node.insession_task_id "
        "AND state.insession_task_node_id=node.insession_task_node_id "
        "AND state.node_revision=node.node_revision "
        "WHERE node.insession_task_id=? AND node.graph_revision=? "
        "AND node.insession_task_node_id=? AND node.node_revision=?",
        (
            subject.task_id,
            subject.graph_revision,
            subject.node_id,
            subject.node_revision,
        ),
    ).fetchone()
    if node is None or str(node["status"]) != "awaiting_user":
        raise WorkExecutionPersistenceError(
            "TaskNode authority no longer permits waiting-user continuation"
        )
    if conn.execute(
        "SELECT 1 FROM insession_task_graph_edges AS edge "
        "JOIN insession_task_graph_nodes AS child "
        "ON child.insession_task_id=edge.insession_task_id "
        "AND child.graph_revision=edge.graph_revision "
        "AND child.insession_task_node_id=edge.child_insession_task_node_id "
        "LEFT JOIN insession_task_node_states AS child_state "
        "ON child_state.insession_task_id=child.insession_task_id "
        "AND child_state.insession_task_node_id=child.insession_task_node_id "
        "AND child_state.node_revision=child.node_revision "
        "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
        "AND edge.parent_insession_task_node_id=? "
        "AND (child_state.status IS NULL OR child_state.status!='completed') "
        "LIMIT 1",
        (subject.task_id, subject.graph_revision, subject.node_id),
    ).fetchone() is not None:
        raise WorkExecutionPersistenceError(
            "TaskNode is no longer ready for waiting-user continuation"
        )
    actual_node_version = int(node["state_version"])
    if actual_node_version != expected_node_state_version:
        raise WorkExecutionRevisionConflict(
            expected=expected_node_state_version,
            actual=actual_node_version,
        )



def _activate_waiting_execution_subject(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
    expected_task_state_version: int,
    expected_node_state_version: int,
    now: str,
) -> tuple[int, int]:
    if conn.execute(
        "UPDATE insession_task_node_states SET status='active', "
        "state_version=state_version+1, updated_at=? "
        "WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=? AND status='awaiting_user' AND state_version=?",
        (
            now,
            subject.task_id,
            subject.node_id,
            subject.node_revision,
            expected_node_state_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "TaskNode changed during waiting-user continuation"
        )
    _next_status, next_task_state_version = _refresh_task_graph_aggregate_status(
        conn,
        task_id=subject.task_id,
        graph_revision=subject.graph_revision,
        expected_task_state_version=expected_task_state_version,
        now=now,
    )
    return next_task_state_version, expected_node_state_version + 1



def _load_current_task_node_delivery_id(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    node_id: str,
    node_revision: int,
) -> str:
    """解析某个当前已完成节点唯一的冻结 PASS Delivery。"""

    return _load_current_task_node_delivery_projection(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        node_id=node_id,
        node_revision=node_revision,
    ).delivery_id


def _load_current_task_node_delivery_projection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    node_id: str,
    node_revision: int,
) -> ResolvedCurrentTaskNodeDelivery:
    """认证一个直接或显式 carry 的当前节点 Delivery。

    只有当前目标恰好有一个规范、不可变的 carry receipt，且其重复列、receipt
    正文/哈希、commit binding 与 immediate-base 投影全部一致时，才接受历史来源
    Delivery。carry-of-carry 会递归认证到唯一原始冻结 Delivery；同一目标同时存在
    直接 Delivery 与 carry receipt 会产生歧义，因此予以拒绝。
    """

    rows = conn.execute(
        "SELECT delivery_id FROM insession_task_node_deliveries "
        "WHERE session_id=? AND insession_task_id=? AND graph_revision=? "
        "AND insession_task_node_id=? AND node_revision=?",
        (session_id, task_id, graph_revision, node_id, node_revision),
    ).fetchall()
    if len(rows) > 1:
        raise WorkExecutionPersistenceError(
            "completed current TaskNode has no unique Delivery"
        )
    # 延迟导入：work_verification 会从本模块导入投影辅助函数，而此完整性检查只在
    # 初始化完成后执行。
    from .work_verification import _load_request_row, _load_task_node_delivery, _record_from_row

    carry_rows = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_task_graph_node_carry_receipts "
        "WHERE session_id=? AND insession_task_id=? "
        "AND target_task_graph_revision=? AND insession_task_node_id=? "
        "AND node_revision=?",
        (session_id, task_id, graph_revision, node_id, node_revision),
    ).fetchall()
    if rows and carry_rows:
        raise WorkExecutionPersistenceError(
            "completed current TaskNode has ambiguous direct and carried Delivery"
        )

    expected_subject = TaskNodeSubject(
        task_id=task_id,
        graph_revision=graph_revision,
        node_id=node_id,
        node_revision=node_revision,
    )
    if rows:
        resolved = _load_task_node_delivery(
            conn,
            session_id=session_id,
            delivery_id=str(rows[0]["delivery_id"]),
        )
        if resolved.delivery.subject != expected_subject:
            raise WorkExecutionPersistenceError(
                "completed current TaskNode Delivery is stale"
            )
        return ResolvedCurrentTaskNodeDelivery.direct(resolved)

    if len(carry_rows) != 1:
        raise WorkExecutionPersistenceError(
            "completed current TaskNode has no unique Delivery or carry receipt"
        )
    carry_row = carry_rows[0]
    receipt_json = str(carry_row["receipt_json"])
    receipt_sha256 = str(carry_row["receipt_sha256"])
    try:
        from ..auxiliary_graph.auxiliary_task_graph_commit import AuxiliaryTaskGraphNodeCarryReceipt

        receipt = AuxiliaryTaskGraphNodeCarryReceipt.model_validate_json(
            receipt_json
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry receipt is not typed"
        ) from exc
    if (
        _model_json(receipt) != receipt_json
        or _text_hash(receipt_json) != receipt_sha256
        or receipt.carry_receipt_id != str(carry_row["carry_receipt_id"])
        or receipt.apply_id != str(carry_row["apply_id"])
        or receipt.session_id != str(carry_row["session_id"])
        or receipt.task_id != str(carry_row["insession_task_id"])
        or receipt.base_task_graph_revision
        != int(carry_row["base_task_graph_revision"])
        or receipt.target_task_graph_revision
        != int(carry_row["target_task_graph_revision"])
        or receipt.node_id != str(carry_row["insession_task_node_id"])
        or receipt.node_revision != int(carry_row["node_revision"])
        or receipt.source_delivery_id != str(carry_row["source_delivery_id"])
        or receipt.definition_sha256 != str(carry_row["definition_sha256"])
        or _canonical_json(list(receipt.dependency_delivery_ids))
        != str(carry_row["dependency_delivery_ids_json"])
        or receipt.dependency_closure_sha256
        != str(carry_row["dependency_closure_sha256"])
        or receipt.source_authority_sha256
        != str(carry_row["source_authority_sha256"])
        or receipt.capability_catalog_sha256
        != str(carry_row["capability_catalog_sha256"])
        or receipt.freshness_authority_sha256
        != str(carry_row["freshness_authority_sha256"])
        or receipt.session_id != session_id
        or receipt.task_id != task_id
        or receipt.target_task_graph_revision != graph_revision
        or receipt.base_task_graph_revision + 1 != graph_revision
        or receipt.node_id != node_id
        or receipt.node_revision != node_revision
    ):
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry receipt is corrupt or stale"
        )

    commit_row = conn.execute(
        "SELECT session_id, insession_task_id, base_task_graph_revision, "
        "committed_task_graph_revision FROM "
        "insession_auxiliary_v2_task_graph_commit_receipts WHERE apply_id=?",
        (receipt.apply_id,),
    ).fetchone()
    if (
        commit_row is None
        or str(commit_row["session_id"]) != session_id
        or str(commit_row["insession_task_id"]) != task_id
        or int(commit_row["base_task_graph_revision"])
        != receipt.base_task_graph_revision
        or int(commit_row["committed_task_graph_revision"])
        != receipt.target_task_graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry commit binding is corrupt"
        )

    immediate_base_subject = expected_subject.model_copy(
        update={"graph_revision": receipt.base_task_graph_revision}
    )
    source_projection = _load_current_task_node_delivery_projection(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=receipt.base_task_graph_revision,
        node_id=node_id,
        node_revision=node_revision,
    )
    if (
        source_projection.target_subject != immediate_base_subject
        or source_projection.delivery_id != receipt.source_delivery_id
    ):
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry source chain is stale"
        )
    resolved = source_projection.source_delivery
    historical_source_subject = resolved.delivery.subject
    if (
        historical_source_subject.task_id != task_id
        or historical_source_subject.node_id != node_id
        or historical_source_subject.node_revision != node_revision
        or historical_source_subject.graph_revision
        > receipt.base_task_graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry historical Delivery is stale"
        )
    source_request = _record_from_row(
        _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=(
                resolved.delivery.verification_request_id
            ),
        )
    ).request
    if (
        source_request.subject != historical_source_subject
        or source_request.dependency_delivery_ids
        != receipt.dependency_delivery_ids
    ):
        raise WorkExecutionPersistenceError(
            "completed current TaskNode carry source verification is stale"
        )
    return ResolvedCurrentTaskNodeDelivery(
        target_subject=expected_subject,
        source_delivery=resolved,
        resolution_kind=CurrentTaskNodeDeliveryResolutionKind.CARRIED,
        carry_authority=TaskNodeDeliveryCarryAuthority(
            carry_receipt_id=receipt.carry_receipt_id,
            carry_receipt_sha256=receipt_sha256,
            apply_id=receipt.apply_id,
            source_delivery_id=receipt.source_delivery_id,
            base_task_graph_revision=receipt.base_task_graph_revision,
            target_subject=expected_subject,
        ),
    )


def _load_current_task_node_dependency_delivery_ids(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> tuple[str, ...]:
    """按稳定 graph 顺序解析精确的直接子节点来源 Delivery ID。"""

    return tuple(
        item.delivery_id
        for item in _load_current_task_node_dependency_deliveries(
            conn,
            session_id=session_id,
            subject=subject,
        )
    )


def _load_current_task_node_dependency_deliveries(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> tuple[ResolvedCurrentTaskNodeDelivery, ...]:
    """解析精确的直接子节点正文与当前目标权威状态。"""

    children = conn.execute(
        "SELECT child.insession_task_node_id, child.node_revision, "
        "child.ordinal, state.status FROM insession_task_graph_edges AS edge "
        "JOIN insession_task_graph_nodes AS child "
        "ON child.insession_task_id=edge.insession_task_id "
        "AND child.graph_revision=edge.graph_revision "
        "AND child.insession_task_node_id=edge.child_insession_task_node_id "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=child.insession_task_id "
        "AND state.insession_task_node_id=child.insession_task_node_id "
        "AND state.node_revision=child.node_revision "
        "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
        "AND edge.parent_insession_task_node_id=? "
        "ORDER BY child.ordinal, child.insession_task_node_id",
        (subject.task_id, subject.graph_revision, subject.node_id),
    ).fetchall()
    deliveries: list[ResolvedCurrentTaskNodeDelivery] = []
    for child in children:
        if str(child["status"]) != "completed":
            raise WorkExecutionPersistenceError(
                "TaskNode direct dependency is incomplete"
            )
        deliveries.append(
            _load_current_task_node_delivery_projection(
                conn,
                session_id=session_id,
                task_id=subject.task_id,
                graph_revision=subject.graph_revision,
                node_id=str(child["insession_task_node_id"]),
                node_revision=int(child["node_revision"]),
            )
        )
    delivery_ids = tuple(item.delivery_id for item in deliveries)
    if len(delivery_ids) != len(set(delivery_ids)):
        raise WorkExecutionPersistenceError(
            "TaskNode direct dependency Deliveries are duplicated"
        )
    return tuple(deliveries)


def _refresh_task_graph_aggregate_status(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    graph_revision: int,
    expected_task_state_version: int,
    now: str,
    terminalizing_work_run_id: str | None = None,
) -> tuple[str, int]:
    """从根 Task 的整个当前 TaskGraph 推导其状态。

    节点状态仍是本地执行事实。尤其是，一个 waiting 分支不会隐藏可运行的同级节点。
    此 reducer 不持有调度器游标，也不会完成 Task；全部节点 completed 时仍保持
    ``active``，直到事务性 FinishGate 证明完整 Delivery 链。
    """

    task = conn.execute(
        "SELECT session_id, current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE insession_task_id=?",
        (task_id,),
    ).fetchone()
    if (
        task is None
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "TaskGraph authority changed during aggregate projection"
        )
    actual_task_version = int(task["state_version"])
    if actual_task_version != expected_task_state_version:
        raise WorkExecutionRevisionConflict(
            expected=expected_task_state_version,
            actual=actual_task_version,
        )
    current_status = str(task["current_status"])
    if current_status in {"completed", "cancelled"}:
        raise WorkExecutionPersistenceError(
            "terminal Task cannot accept a graph aggregate projection"
        )

    nodes = conn.execute(
        "SELECT node.insession_task_node_id, node.node_revision, node.node_kind, "
        "node.ordinal, state.status FROM insession_task_graph_nodes AS node "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=node.insession_task_id "
        "AND state.insession_task_node_id=node.insession_task_node_id "
        "AND state.node_revision=node.node_revision "
        "WHERE node.insession_task_id=? AND node.graph_revision=? "
        "ORDER BY node.ordinal, node.insession_task_node_id",
        (task_id, graph_revision),
    ).fetchall()
    if not nodes:
        raise WorkExecutionPersistenceError(
            "current TaskGraph has no aggregateable nodes"
        )
    node_ids = tuple(str(row["insession_task_node_id"]) for row in nodes)
    if len(node_ids) != len(set(node_ids)):
        raise WorkExecutionPersistenceError("current TaskGraph node identity is corrupt")
    roots = tuple(
        row for row in nodes if str(row["node_kind"]) == "root"
    )
    if len(roots) != 1:
        raise WorkExecutionPersistenceError("current TaskGraph root is not unique")

    child_rows = conn.execute(
        "SELECT edge.parent_insession_task_node_id, "
        "edge.child_insession_task_node_id FROM insession_task_graph_edges AS edge "
        "WHERE edge.insession_task_id=? AND edge.graph_revision=?",
        (task_id, graph_revision),
    ).fetchall()
    children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in child_rows:
        parent_id = str(edge["parent_insession_task_node_id"])
        child_id = str(edge["child_insession_task_node_id"])
        if parent_id not in children_by_parent or child_id not in children_by_parent:
            raise WorkExecutionPersistenceError(
                "current TaskGraph edge references an unknown node"
            )
        children_by_parent[parent_id].append(child_id)

    nonterminal_rows = conn.execute(
        "SELECT work_run_id, graph_revision, insession_task_node_id, "
        "node_revision, status, reason FROM insession_work_runs "
        "WHERE subject_kind='task_node' AND insession_task_id=? "
        "AND status NOT IN ('completed', 'failed', 'cancelled')",
        (task_id,),
    ).fetchall()
    current_node_by_id = {
        str(row["insession_task_node_id"]): row for row in nodes
    }
    run_by_node: dict[str, sqlite3.Row] = {}
    for run in nonterminal_rows:
        if (
            terminalizing_work_run_id is not None
            and str(run["work_run_id"]) == terminalizing_work_run_id
        ):
            continue
        node_id = str(run["insession_task_node_id"])
        node = current_node_by_id.get(node_id)
        if (
            node is None
            or int(run["graph_revision"]) != graph_revision
            or int(run["node_revision"]) != int(node["node_revision"])
            or node_id in run_by_node
        ):
            raise WorkExecutionPersistenceError(
                "nonterminal WorkRun is detached from current TaskGraph authority"
            )
        run_by_node[node_id] = run

    delivery_by_node: dict[str, str] = {}
    for node in nodes:
        node_id = str(node["insession_task_node_id"])
        if str(node["status"]) == "completed":
            delivery_by_node[node_id] = _load_current_task_node_delivery_id(
                conn,
                session_id=str(task["session_id"]),
                task_id=task_id,
                graph_revision=graph_revision,
                node_id=node_id,
                node_revision=int(node["node_revision"]),
            )

    ready = False
    active_or_recoverable = False
    awaiting_user = False
    waiting_external = False
    all_proposed = True
    for node in nodes:
        node_id = str(node["insession_task_node_id"])
        node_status = str(node["status"])
        all_proposed = all_proposed and node_status == "proposed"
        run = run_by_node.get(node_id)
        children_complete = all(
            child_id in delivery_by_node
            for child_id in children_by_parent[node_id]
        )
        if (
            node_status in _STARTABLE_NODE_STATUSES
            and run is None
            and children_complete
        ):
            ready = True
        if run is not None:
            run_status = str(run["status"])
            run_reason = str(run["reason"] or "")
            if run_status == "waiting_user":
                if node_status != "awaiting_user" or run_reason != "needs_input":
                    raise WorkExecutionPersistenceError(
                        "waiting-user WorkRun disagrees with TaskNode state"
                    )
                awaiting_user = True
            elif run_status == "waiting_external":
                if node_status != "waiting_external":
                    raise WorkExecutionPersistenceError(
                        "waiting-external WorkRun disagrees with TaskNode state"
                    )
                waiting_external = True
            else:
                active_or_recoverable = True
        elif node_status == "awaiting_user":
            raise WorkExecutionPersistenceError(
                "awaiting-user TaskNode has no nonterminal WorkRun"
            )
        elif node_status == "waiting_external":
            raise WorkExecutionPersistenceError(
                "waiting-external TaskNode has no nonterminal WorkRun"
            )

    all_completed = len(delivery_by_node) == len(nodes)
    has_any_history = conn.execute(
        "SELECT 1 FROM insession_work_runs WHERE insession_task_id=? LIMIT 1",
        (task_id,),
    ).fetchone() is not None
    if all_completed or ready or active_or_recoverable:
        next_status = "active"
    elif all_proposed and not has_any_history and current_status == "proposed":
        next_status = "proposed"
    elif awaiting_user:
        next_status = "awaiting_user"
    elif waiting_external:
        next_status = "waiting_external"
    else:
        next_status = "blocked"

    next_task_version = expected_task_state_version + 1
    if conn.execute(
        "UPDATE insession_tasks SET current_status=?, state_version=?, updated_at=? "
        "WHERE insession_task_id=? AND current_graph_revision=? "
        "AND state_version=? AND current_status NOT IN ('completed', 'cancelled')",
        (
            next_status,
            next_task_version,
            now,
            task_id,
            graph_revision,
            expected_task_state_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Task changed during graph aggregate projection"
        )
    return next_status, next_task_version


def _project_execution_subject_wait_state(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    target_status: Literal["awaiting_user", "waiting_external"],
    now: str,
) -> None:
    subject = _subject_from_run_row(run_row)
    if isinstance(subject, TaskNodeSubject):
        _project_task_node_wait_state(
            conn,
            run_row=run_row,
            target_status=target_status,
            now=now,
        )
        return
    if _execution_subject_contract_version(conn, run_row) != "auxiliary_node_v2":
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun execution-subject contract is unsupported"
        )
    _project_auxiliary_node_wait_state(
        conn,
        run_row=run_row,
        subject=subject,
        target_status=target_status,
        now=now,
    )


def _require_execution_replan_preconditions(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    run_row: sqlite3.Row,
    work_run_id: str,
    attempt_id: str,
) -> _ExecutionReplanPreflight:
    subject = _subject_from_run_row(run_row)
    if not isinstance(subject, TaskNodeSubject):
        raise WorkExecutionPersistenceError(
            "only an ordinary TaskNode may request TaskGraph revision"
        )
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (session_id, subject.task_id),
    ).fetchone()
    node = conn.execute(
        "SELECT node.ordinal, state.status, state.state_version "
        "FROM insession_task_graph_nodes AS node "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=node.insession_task_id "
        "AND state.insession_task_node_id=node.insession_task_node_id "
        "AND state.node_revision=node.node_revision "
        "WHERE node.insession_task_id=? AND node.graph_revision=? "
        "AND node.insession_task_node_id=? AND node.node_revision=?",
        (
            subject.task_id,
            subject.graph_revision,
            subject.node_id,
            subject.node_revision,
        ),
    ).fetchone()
    if (
        task is None
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != subject.graph_revision
        or str(task["current_status"]) != "active"
        or node is None
        or str(node["status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task/TaskNode authority changed before TaskGraph revision request"
        )
    competing = conn.execute(
        "SELECT work_run_id FROM insession_work_runs "
        "WHERE session_id=? AND insession_task_id=? AND subject_kind='task_node' "
        "AND work_run_id<>? "
        "AND status NOT IN ('completed','failed','cancelled') LIMIT 1",
        (session_id, subject.task_id, work_run_id),
    ).fetchone()
    if competing is not None:
        raise WorkExecutionPersistenceError(
            "TaskGraph revision request cannot race another TaskNode WorkRun"
        )
    legacy_active = conn.execute(
        "SELECT 1 FROM insession_active_task_graph_revision_triggers "
        "WHERE session_id=? AND insession_task_id=? LIMIT 1",
        (session_id, subject.task_id),
    ).fetchone()
    execution_active = conn.execute(
        "SELECT 1 FROM insession_active_task_graph_execution_replan_requests "
        "WHERE session_id=? AND insession_task_id=? LIMIT 1",
        (session_id, subject.task_id),
    ).fetchone()
    if legacy_active is not None or execution_active is not None:
        raise WorkExecutionPersistenceError(
            "Task already owns an active TaskGraph revision authority"
        )
    planning_authority = conn.execute(
        "SELECT goal.status AS goal_status, state.status AS revision_status, "
        "EXISTS(SELECT 1 FROM "
        "insession_auxiliary_graph_revision_apply_receipts_v2 AS receipt "
        "WHERE receipt.operation='supersede_goal' "
        "AND receipt.session_id=control.session_id "
        "AND receipt.insession_task_id=control.insession_task_id "
        "AND receipt.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND receipt.goal_id=control.current_goal_id "
        "AND receipt.committed_auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "AND receipt.committed_control_state_version=control.state_version) "
        "AS has_pending_supersede "
        "FROM insession_auxiliary_graph_v2_containers AS control "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.session_id=control.session_id "
        "AND goal.insession_task_id=control.insession_task_id "
        "AND goal.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND goal.goal_id=control.current_goal_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS state "
        "ON state.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "WHERE control.session_id=? AND control.insession_task_id=?",
        (session_id, subject.task_id),
    ).fetchone()
    if planning_authority is not None and (
        int(planning_authority["has_pending_supersede"]) == 1
        or str(planning_authority["goal_status"]) == "superseded"
        or str(planning_authority["revision_status"]) == "superseded"
    ):
        raise WorkExecutionPersistenceError(
            "Task has a pending Auxiliary goal supersede authority"
        )
    ordinal = int(node["ordinal"])
    if ordinal < 0 or ordinal > 999:
        raise WorkExecutionPersistenceError(
            "TaskNode ordinal cannot enter the frozen base alias namespace"
        )
    return _ExecutionReplanPreflight(
        subject=subject,
        source_node_alias=f"base_node_{ordinal:03d}",
        task_state_version=int(task["state_version"]),
        node_state_version=int(node["state_version"]),
    )


def _apply_execution_replan_projection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    apply_id: str,
    action: RequestTaskGraphRevisionAction,
    preflight: _ExecutionReplanPreflight,
    now: str,
) -> tuple[TaskGraphExecutionReplanRequest, int, int]:
    subject = preflight.subject
    next_node_version = preflight.node_state_version + 1
    if conn.execute(
        "UPDATE insession_task_node_states SET status='cancelled', "
        "state_version=?, updated_at=? WHERE insession_task_id=? "
        "AND insession_task_node_id=? AND node_revision=? "
        "AND status='active' AND state_version=?",
        (
            next_node_version,
            now,
            subject.task_id,
            subject.node_id,
            subject.node_revision,
            preflight.node_state_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "TaskNode changed during TaskGraph revision request"
        )
    next_task_version = preflight.task_state_version + 1
    if conn.execute(
        "UPDATE insession_tasks SET current_status='active', state_version=?, "
        "updated_at=? WHERE session_id=? AND insession_task_id=? "
        "AND current_graph_revision=? AND current_status='active' "
        "AND state_version=?",
        (
            next_task_version,
            now,
            session_id,
            subject.task_id,
            subject.graph_revision,
            preflight.task_state_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Task changed during TaskGraph revision request"
        )
    identity_hash = _payload_hash(
        {
            "contract": "task-graph-execution-replan-request-id-v1",
            "create_apply_id": apply_id,
            "work_run_id": work_run_id,
            "attempt_id": attempt_id,
        }
    )
    request = TaskGraphExecutionReplanRequest.create(
        request_id=f"tger_v1:{identity_hash[:40]}",
        create_apply_id=apply_id,
        session_id=session_id,
        task_id=subject.task_id,
        base_graph_revision=subject.graph_revision,
        target_graph_revision=subject.graph_revision + 1,
        requesting_subject=subject,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        source_node_alias=preflight.source_node_alias,
        reason=action.reason,
        diagnosis=action.diagnosis,
        revision_objective=action.revision_objective,
        supporting_tool_result_ids=action.supporting_tool_result_ids,
        task_state_version=next_task_version,
        created_turn_id=turn_id,
    )
    return request, next_task_version, next_node_version


def _insert_execution_replan_request(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphExecutionReplanRequest,
    now: str,
) -> None:
    support_json = _canonical_json(list(request.supporting_tool_result_ids))
    request_json = _model_json(request)
    try:
        conn.execute(
            "INSERT INTO insession_task_graph_execution_replan_requests "
            "(request_id, create_apply_id, session_id, insession_task_id, "
            "base_graph_revision, target_graph_revision, work_run_id, attempt_id, "
            "insession_task_node_id, node_revision, source_node_alias, reason, "
            "diagnosis, revision_objective, supporting_tool_result_ids_json, "
            "supporting_tool_result_ids_sha256, task_state_version, "
            "created_turn_id, request_sha256, request_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.request_id,
                request.create_apply_id,
                request.session_id,
                request.task_id,
                request.base_graph_revision,
                request.target_graph_revision,
                request.work_run_id,
                request.attempt_id,
                request.requesting_subject.node_id,
                request.requesting_subject.node_revision,
                request.source_node_alias,
                request.reason.value,
                request.diagnosis,
                request.revision_objective,
                support_json,
                _text_hash(support_json),
                request.task_state_version,
                request.created_turn_id,
                request.request_sha256,
                request_json,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_active_task_graph_execution_replan_requests "
            "(request_id, request_sha256, session_id, insession_task_id, "
            "base_graph_revision, target_graph_revision, activated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request.request_id,
                request.request_sha256,
                request.session_id,
                request.task_id,
                request.base_graph_revision,
                request.target_graph_revision,
                now,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise WorkExecutionPersistenceError(
            "TaskGraph execution replan request conflicts with stored authority"
        ) from exc


def _execution_replan_request_from_row(
    row: sqlite3.Row,
) -> TaskGraphExecutionReplanRequest:
    try:
        request = TaskGraphExecutionReplanRequest.model_validate_json(
            str(row["request_json"])
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "stored execution replan request is not typed"
        ) from exc
    support_json = _canonical_json(list(request.supporting_tool_result_ids))
    if (
        _model_json(request) != str(row["request_json"])
        or request.request_sha256 != str(row["request_sha256"])
        or request.request_id != str(row["request_id"])
        or request.create_apply_id != str(row["create_apply_id"])
        or request.session_id != str(row["session_id"])
        or request.task_id != str(row["insession_task_id"])
        or request.base_graph_revision != int(row["base_graph_revision"])
        or request.target_graph_revision != int(row["target_graph_revision"])
        or request.work_run_id != str(row["work_run_id"])
        or request.attempt_id != str(row["attempt_id"])
        or request.requesting_subject.node_id
        != str(row["insession_task_node_id"])
        or request.requesting_subject.node_revision != int(row["node_revision"])
        or request.source_node_alias != str(row["source_node_alias"])
        or request.reason.value != str(row["reason"])
        or request.diagnosis != str(row["diagnosis"])
        or request.revision_objective != str(row["revision_objective"])
        or support_json != str(row["supporting_tool_result_ids_json"])
        or _text_hash(support_json)
        != str(row["supporting_tool_result_ids_sha256"])
        or request.task_state_version != int(row["task_state_version"])
        or request.created_turn_id != str(row["created_turn_id"])
    ):
        raise WorkExecutionPersistenceError(
            "stored execution replan request authority is corrupt"
        )
    return request


def _load_authenticated_execution_replan_request(
    conn: sqlite3.Connection,
    *,
    request_id: str | None,
    session_id: str,
    task_id: str,
    require_active: bool,
) -> TaskGraphExecutionReplanRequest | None:
    if request_id is None:
        rows = conn.execute(
            "SELECT request.* FROM "
            "insession_active_task_graph_execution_replan_requests AS active "
            "JOIN insession_task_graph_execution_replan_requests AS request "
            "ON request.request_id=active.request_id "
            "WHERE active.session_id=? AND active.insession_task_id=?",
            (session_id, task_id),
        ).fetchall()
        if len(rows) > 1:
            raise WorkExecutionPersistenceError(
                "Task owns multiple active execution replan requests"
            )
        if not rows:
            return None
        row = rows[0]
    else:
        row = conn.execute(
            "SELECT * FROM insession_task_graph_execution_replan_requests "
            "WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
    request = _execution_replan_request_from_row(row)
    if request.session_id != session_id or request.task_id != task_id:
        raise WorkExecutionPersistenceError(
            "execution replan request crossed Session/Task authority"
        )
    active = conn.execute(
        "SELECT * FROM insession_active_task_graph_execution_replan_requests "
        "WHERE request_id=?",
        (request.request_id,),
    ).fetchone()
    if require_active and active is None:
        raise WorkExecutionPersistenceError(
            "execution replan request is no longer active"
        )
    if active is not None and (
        str(active["request_sha256"]) != request.request_sha256
        or str(active["session_id"]) != request.session_id
        or str(active["insession_task_id"]) != request.task_id
        or int(active["base_graph_revision"]) != request.base_graph_revision
        or int(active["target_graph_revision"]) != request.target_graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "active execution replan pointer is corrupt"
        )
    run = conn.execute(
        "SELECT status, reason, current_attempt_id FROM insession_work_runs "
        "WHERE work_run_id=? AND session_id=?",
        (request.work_run_id, request.session_id),
    ).fetchone()
    attempt = conn.execute(
        "SELECT status, action, decision_json FROM insession_work_run_attempts "
        "WHERE work_run_id=? AND attempt_id=?",
        (request.work_run_id, request.attempt_id),
    ).fetchone()
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (request.session_id, request.task_id),
    ).fetchone()
    node = conn.execute(
        "SELECT status FROM insession_task_node_states "
        "WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=?",
        (
            request.task_id,
            request.requesting_subject.node_id,
            request.requesting_subject.node_revision,
        ),
    ).fetchone()
    try:
        stored_decision = HostAcceptedAttemptDecision.model_validate_json(
            str(attempt["decision_json"] if attempt is not None else "")
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "execution replan Attempt decision is corrupt"
        ) from exc
    action = stored_decision.action
    if (
        run is None
        or str(run["status"]) != "cancelled"
        or str(run["reason"] or "") != "task_graph_revision_requested"
        or run["current_attempt_id"] is not None
        or attempt is None
        or str(attempt["status"]) != "closed"
        or str(attempt["action"] or "") != "request_task_graph_revision"
        or not isinstance(action, RequestTaskGraphRevisionAction)
        or action.reason != request.reason
        or action.diagnosis != request.diagnosis
        or action.revision_objective != request.revision_objective
        or action.supporting_tool_result_ids
        != request.supporting_tool_result_ids
        or task is None
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != request.base_graph_revision
        or str(task["current_status"]) != "active"
        or int(task["state_version"]) < request.task_state_version
        or node is None
        or str(node["status"]) != "cancelled"
    ):
        raise WorkExecutionPersistenceError(
            "execution replan request lost its settled execution authority"
        )
    return request


def _load_replayed_execution_replan_request(
    conn: sqlite3.Connection,
    *,
    create_apply_id: str,
    session_id: str,
    task_id: str,
    work_run_id: str,
    attempt_id: str,
) -> TaskGraphExecutionReplanRequest:
    row = conn.execute(
        "SELECT * FROM insession_task_graph_execution_replan_requests "
        "WHERE create_apply_id=? AND session_id=? AND insession_task_id=?",
        (create_apply_id, session_id, task_id),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "replayed TaskGraph revision request lost durable history"
        )
    request = _execution_replan_request_from_row(row)
    if (
        request.work_run_id != work_run_id
        or request.attempt_id != attempt_id
    ):
        raise WorkExecutionPersistenceError(
            "replayed TaskGraph revision request crossed Attempt authority"
        )
    active_rows = conn.execute(
        "SELECT * FROM insession_active_task_graph_execution_replan_requests "
        "WHERE request_id=?",
        (request.request_id,),
    ).fetchall()
    application_rows = conn.execute(
        "SELECT * FROM insession_task_graph_execution_replan_applications "
        "WHERE request_id=?",
        (request.request_id,),
    ).fetchall()
    if len(active_rows) > 1 or len(application_rows) > 1 or bool(active_rows) == bool(
        application_rows
    ):
        raise WorkExecutionPersistenceError(
            "replayed TaskGraph revision request has ambiguous lifecycle state"
        )
    if active_rows:
        loaded = _load_authenticated_execution_replan_request(
            conn,
            request_id=request.request_id,
            session_id=session_id,
            task_id=task_id,
            require_active=True,
        )
        if loaded != request:
            raise WorkExecutionPersistenceError(
                "replayed active TaskGraph revision request changed"
            )
        return request

    application_row = application_rows[0]
    try:
        application = TaskGraphExecutionReplanApplication.model_validate_json(
            str(application_row["receipt_json"])
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "execution replan application history is not typed"
        ) from exc
    expected_application_hash = _payload_hash(
        {
            "contract": (
                "atomic-task-graph-execution-replan-application-id-v1"
            ),
            "task_graph_commit_apply_id": (
                application.task_graph_commit_apply_id
            ),
            "request_id": request.request_id,
        }
    )
    commit = conn.execute(
        "SELECT session_id, insession_task_id, source_turn_id, "
        "base_task_graph_revision, committed_task_graph_revision "
        "FROM insession_auxiliary_v2_task_graph_commit_receipts "
        "WHERE apply_id=?",
        (application.task_graph_commit_apply_id,),
    ).fetchone()
    if (
        _model_json(application) != str(application_row["receipt_json"])
        or application.apply_id != str(application_row["apply_id"])
        or application.request_id != str(application_row["request_id"])
        or application.request_sha256 != str(application_row["request_sha256"])
        or application.session_id != str(application_row["session_id"])
        or application.task_id != str(application_row["insession_task_id"])
        or application.base_graph_revision
        != int(application_row["base_graph_revision"])
        or application.committed_graph_revision
        != int(application_row["committed_graph_revision"])
        or application.task_graph_commit_apply_id
        != str(application_row["task_graph_commit_apply_id"])
        or application.consumed_turn_id
        != str(application_row["consumed_turn_id"])
        or application.receipt_sha256
        != str(application_row["receipt_sha256"])
        or application.apply_id
        != f"tger_apply_{expected_application_hash[:40]}"
        or application.request_id != request.request_id
        or application.request_sha256 != request.request_sha256
        or application.session_id != request.session_id
        or application.task_id != request.task_id
        or application.base_graph_revision != request.base_graph_revision
        or application.committed_graph_revision != request.target_graph_revision
        or commit is None
        or str(commit["session_id"]) != request.session_id
        or str(commit["insession_task_id"]) != request.task_id
        or str(commit["source_turn_id"]) != application.consumed_turn_id
        or int(commit["base_task_graph_revision"] or 0)
        != request.base_graph_revision
        or int(commit["committed_task_graph_revision"])
        != request.target_graph_revision
    ):
        raise WorkExecutionPersistenceError(
            "execution replan application history is corrupt"
        )
    run = conn.execute(
        "SELECT status, reason, current_attempt_id FROM insession_work_runs "
        "WHERE work_run_id=? AND session_id=?",
        (request.work_run_id, request.session_id),
    ).fetchone()
    attempt = conn.execute(
        "SELECT status, action, decision_json FROM insession_work_run_attempts "
        "WHERE work_run_id=? AND attempt_id=?",
        (request.work_run_id, request.attempt_id),
    ).fetchone()
    node = conn.execute(
        "SELECT status FROM insession_task_node_states "
        "WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=?",
        (
            request.task_id,
            request.requesting_subject.node_id,
            request.requesting_subject.node_revision,
        ),
    ).fetchone()
    try:
        stored_decision = HostAcceptedAttemptDecision.model_validate_json(
            str(attempt["decision_json"] if attempt is not None else "")
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "execution replan replay Attempt decision is corrupt"
        ) from exc
    action = stored_decision.action
    if (
        run is None
        or str(run["status"]) != "cancelled"
        or str(run["reason"] or "") != "task_graph_revision_requested"
        or run["current_attempt_id"] is not None
        or attempt is None
        or str(attempt["status"]) != "closed"
        or str(attempt["action"] or "") != "request_task_graph_revision"
        or not isinstance(action, RequestTaskGraphRevisionAction)
        or action.reason != request.reason
        or action.diagnosis != request.diagnosis
        or action.revision_objective != request.revision_objective
        or action.supporting_tool_result_ids
        != request.supporting_tool_result_ids
        or node is None
        or str(node["status"]) != "cancelled"
    ):
        raise WorkExecutionPersistenceError(
            "execution replan application lost originating Attempt authority"
        )
    return request


def _project_task_node_wait_state(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    target_status: Literal["awaiting_user", "waiting_external"],
    now: str,
) -> None:
    """通过 CAS 将所属 Task 与 TaskNode 置为 WorkRun 的持久化等待状态。"""

    task_id = str(run_row["insession_task_id"])
    graph_revision = int(run_row["graph_revision"])
    node_id = str(run_row["insession_task_node_id"])
    node_revision = int(run_row["node_revision"])
    task_row = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE insession_task_id=?",
        (task_id,),
    ).fetchone()
    node_row = conn.execute(
        "SELECT status, state_version FROM insession_task_node_states "
        "WHERE insession_task_id=? AND insession_task_node_id=? AND node_revision=?",
        (task_id, node_id, node_revision),
    ).fetchone()
    if (
        task_row is None
        or task_row["current_graph_revision"] is None
        or int(task_row["current_graph_revision"]) != graph_revision
        or str(task_row["current_status"]) != "active"
        or node_row is None
        or str(node_row["status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task/TaskNode authority changed before wait-state projection"
        )
    task_version = int(task_row["state_version"])
    node_version = int(node_row["state_version"])
    if conn.execute(
        "UPDATE insession_task_node_states SET status=?, state_version=state_version+1, "
        "updated_at=? WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=? AND status='active' AND state_version=?",
        (
            target_status,
            now,
            task_id,
            node_id,
            node_revision,
            node_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "TaskNode changed during wait-state projection"
        )
    _refresh_task_graph_aggregate_status(
        conn,
        task_id=task_id,
        graph_revision=graph_revision,
        expected_task_state_version=task_version,
        now=now,
    )


def _project_auxiliary_node_wait_state(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    subject: AuxiliaryNodeSubject,
    target_status: Literal["awaiting_user", "waiting_external"],
    now: str,
) -> None:
    from ..auxiliary_graph.auxiliary_continuation import (
        _load_exact_auxiliary_authority,
        _require_authority_statuses,
        _require_auxiliary_authority_shape,
        _transition_auxiliary_wait_states,
    )

    authority = _load_exact_auxiliary_authority(
        conn,
        session_id=str(run_row["session_id"]),
        work_run_id=str(run_row["work_run_id"]),
    )
    _require_auxiliary_authority_shape(authority)
    if _subject_from_run_row(authority) != subject:
        raise WorkExecutionPersistenceError(
            "Task/AuxiliaryNode authority changed before wait-state projection"
        )
    _require_authority_statuses(
        authority,
        task_status="active",
        node_status="active",
        goal_status="active",
        revision_status="active",
    )
    _transition_auxiliary_wait_states(
        conn,
        authority=authority,
        from_status="active",
        to_status=(
            "waiting_user"
            if target_status == "awaiting_user"
            else "waiting_external"
        ),
        from_task_status="active",
        to_task_status=target_status,
        now=now,
    )



def _project_execution_subject_budget_failure(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    now: str,
) -> None:
    subject = _subject_from_run_row(run_row)
    if isinstance(subject, TaskNodeSubject):
        _project_task_node_budget_failure(conn, run_row=run_row, now=now)
        return
    contract_version = _execution_subject_contract_version(conn, run_row)
    if contract_version != "auxiliary_node_v2":
        raise WorkExecutionPersistenceError(
            "Auxiliary WorkRun execution-subject contract is unsupported"
        )
    _project_auxiliary_node_budget_failure(
        conn,
        run_row=run_row,
        subject=subject,
        now=now,
    )


def _project_auxiliary_node_budget_failure(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    subject: AuxiliaryNodeSubject,
    now: str,
) -> None:
    """Interrupt the exact current planning aggregate with one CAS chain."""

    authority = conn.execute(
        "SELECT binding.goal_id, binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "control.current_goal_id, control.current_auxiliary_graph_revision, "
        "goal.base_task_graph_revision, goal.status AS goal_status, "
        "goal.state_version AS goal_state_version, "
        "revision.structure_contract_version, "
        "revision_state.status AS revision_status, "
        "revision_state.state_version AS revision_state_version, "
        "definition.executor_kind, definition.definition_sha256, "
        "node_state.status AS node_status, "
        "node_state.state_version AS node_state_version, "
        "task.current_graph_revision AS task_graph_revision, "
        "task.current_status AS task_status, "
        "task.state_version AS task_state_version "
        "FROM insession_execution_subjects AS registry "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=registry.session_id "
        "AND binding.insession_task_id=registry.insession_task_id "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=binding.session_id "
        "AND control.insession_task_id=binding.insession_task_id "
        "AND control.auxiliary_graph_id=binding.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_snapshots AS revision "
        "ON revision.auxiliary_graph_id=binding.auxiliary_graph_id "
        "AND revision.auxiliary_graph_revision=binding.auxiliary_graph_revision "
        "AND revision.insession_task_id=binding.insession_task_id "
        "AND revision.goal_id=binding.goal_id "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.goal_id=revision.goal_id "
        "AND goal.session_id=binding.session_id "
        "AND goal.insession_task_id=binding.insession_task_id "
        "AND goal.auxiliary_graph_id=binding.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "ON membership.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND membership.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "AND membership.auxiliary_node_id=binding.auxiliary_node_id "
        "AND membership.node_revision=binding.node_revision "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision="
        "membership.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND node_state.node_revision=membership.node_revision "
        "JOIN insession_tasks AS task ON task.session_id=binding.session_id "
        "AND task.insession_task_id=binding.insession_task_id "
        "WHERE registry.execution_subject_id=? AND registry.session_id=? "
        "AND registry.insession_task_id=? "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "AND registry.task_node_binding_id IS NULL "
        "AND binding.auxiliary_graph_id=? "
        "AND binding.auxiliary_graph_revision=? "
        "AND binding.auxiliary_node_id=? AND binding.node_revision=?",
        (
            str(run_row["execution_subject_id"]),
            str(run_row["session_id"]),
            subject.task_id,
            subject.auxiliary_graph_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
        ),
    ).fetchone()
    task_revision = (
        int(authority["task_graph_revision"])
        if authority is not None and authority["task_graph_revision"] is not None
        else None
    )
    base_revision = (
        int(authority["base_task_graph_revision"])
        if authority is not None
        and authority["base_task_graph_revision"] is not None
        else None
    )
    if (
        authority is None
        or str(authority["structure_contract_version"])
        != "auxiliary-graph-revision-v2"
        or str(authority["current_goal_id"]) != str(authority["goal_id"])
        or int(authority["current_auxiliary_graph_revision"])
        != subject.auxiliary_graph_revision
        or str(authority["bound_executor_kind"])
        != str(authority["executor_kind"])
        or str(authority["bound_definition_sha256"])
        != str(authority["definition_sha256"])
        or task_revision != base_revision
        or str(authority["task_status"]) != "active"
        or str(authority["node_status"]) != "active"
        or str(authority["goal_status"]) != "active"
        or str(authority["revision_status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task/AuxiliaryNode authority changed before hard-budget projection"
        )

    node_version = int(authority["node_state_version"])
    goal_version = int(authority["goal_state_version"])
    revision_version = int(authority["revision_state_version"])
    task_version = int(authority["task_state_version"])
    if conn.execute(
        "UPDATE insession_auxiliary_node_states_v2 SET status='interrupted', "
        "state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND auxiliary_node_id=? AND node_revision=? "
        "AND status='active' AND state_version=?",
        (
            now,
            subject.auxiliary_graph_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
            node_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "AuxiliaryNode changed during hard-budget projection"
        )
    if conn.execute(
        "UPDATE insession_auxiliary_graph_goals SET status='interrupted', "
        "state_version=state_version+1, updated_at=? "
        "WHERE goal_id=? AND status='active' AND state_version=?",
        (now, str(authority["goal_id"]), goal_version),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Auxiliary goal changed during hard-budget projection"
        )
    if conn.execute(
        "UPDATE insession_auxiliary_graph_revision_states_v2 "
        "SET status='interrupted', state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND status='active' AND state_version=?",
        (
            now,
            subject.auxiliary_graph_id,
            subject.auxiliary_graph_revision,
            revision_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Auxiliary revision changed during hard-budget projection"
        )
    if conn.execute(
        "UPDATE insession_tasks SET current_status='interrupted', "
        "state_version=state_version+1, updated_at=? "
        "WHERE insession_task_id=? AND session_id=? "
        "AND current_graph_revision IS ? AND current_status='active' "
        "AND state_version=?",
        (
            now,
            subject.task_id,
            str(run_row["session_id"]),
            base_revision,
            task_version,
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Task changed during Auxiliary hard-budget projection"
        )


def _project_task_node_budget_failure(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
    now: str,
) -> None:
    """中断此节点，并从整个 graph 推导所属 Task。"""

    task_id = str(run_row["insession_task_id"])
    graph_revision = int(run_row["graph_revision"])
    node_id = str(run_row["insession_task_node_id"])
    node_revision = int(run_row["node_revision"])
    task_row = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE insession_task_id=?",
        (task_id,),
    ).fetchone()
    node_row = conn.execute(
        "SELECT status, state_version FROM insession_task_node_states "
        "WHERE insession_task_id=? AND insession_task_node_id=? AND node_revision=?",
        (task_id, node_id, node_revision),
    ).fetchone()
    if (
        task_row is None
        or task_row["current_graph_revision"] is None
        or int(task_row["current_graph_revision"]) != graph_revision
        or str(task_row["current_status"]) != "active"
        or node_row is None
        or str(node_row["status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task/TaskNode authority changed before hard-budget projection"
        )
    task_version = int(task_row["state_version"])
    node_version = int(node_row["state_version"])
    if conn.execute(
        "UPDATE insession_task_node_states SET status='interrupted', "
        "state_version=state_version+1, updated_at=? "
        "WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=? AND status='active' AND state_version=?",
        (now, task_id, node_id, node_revision, node_version),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "TaskNode changed during hard-budget projection"
        )
    _refresh_task_graph_aggregate_status(
        conn,
        task_id=task_id,
        graph_revision=graph_revision,
        expected_task_state_version=task_version,
        now=now,
        terminalizing_work_run_id=str(run_row["work_run_id"]),
    )


def _load_progress(conn: sqlite3.Connection, work_run_id: str) -> AcceptanceProgressSnapshot:
    row = conn.execute(
        "SELECT progress_revision, evaluated_output_revision, snapshot_hash, snapshot_json "
        "FROM insession_work_run_acceptance_progress WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError("WorkRun Acceptance progress is missing")
    snapshot_json = str(row["snapshot_json"])
    if _text_hash(snapshot_json) != str(row["snapshot_hash"]):
        raise WorkExecutionPersistenceError("WorkRun Acceptance progress is corrupt")
    try:
        snapshot = AcceptanceProgressSnapshot.model_validate_json(snapshot_json)
    except ValueError as exc:
        raise WorkExecutionPersistenceError(
            "WorkRun Acceptance progress payload is invalid"
        ) from exc
    if (
        snapshot.work_run_id != work_run_id
        or snapshot.revision != int(row["progress_revision"])
        or snapshot.evaluated_output_revision
        != int(row["evaluated_output_revision"])
    ):
        raise WorkExecutionPersistenceError("WorkRun Acceptance progress binding is corrupt")
    run_row = conn.execute(
        "SELECT * FROM insession_work_runs WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if run_row is None:
        raise WorkExecutionPersistenceError(
            "WorkRun Acceptance progress owner is missing"
        )
    expected_subject = _subject_from_run_row(run_row)
    if snapshot.subject != expected_subject:
        raise WorkExecutionPersistenceError(
            "WorkRun Acceptance progress subject binding is corrupt"
        )
    output_row = conn.execute(
        "SELECT output_revision FROM insession_work_run_output_windows "
        "WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if (
        output_row is None
        or snapshot.evaluated_output_revision != int(output_row["output_revision"])
    ):
        raise WorkExecutionPersistenceError(
            "Acceptance progress is not bound to the current OutputWindow"
        )
    return snapshot


def _load_output_window(
    conn: sqlite3.Connection,
    work_run_id: str,
) -> tuple[OutputWindow, str]:
    row = conn.execute(
        "SELECT output_revision, snapshot_hash, snapshot_json, updated_turn_id, "
        "updated_attempt_id FROM insession_work_run_output_windows WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError("WorkRun OutputWindow is missing")
    snapshot_json = str(row["snapshot_json"])
    snapshot_hash = str(row["snapshot_hash"])
    if _text_hash(snapshot_json) != snapshot_hash:
        raise WorkExecutionPersistenceError("WorkRun OutputWindow is corrupt")
    try:
        output_window = OutputWindow.model_validate_json(snapshot_json)
    except ValueError as exc:
        raise WorkExecutionPersistenceError(
            "WorkRun OutputWindow payload is invalid"
        ) from exc
    updated_attempt_id = (
        str(row["updated_attempt_id"])
        if row["updated_attempt_id"] is not None
        else None
    )
    if (
        output_window.work_run_id != work_run_id
        or output_window.output_revision != int(row["output_revision"])
        or output_window.updated_turn_id != str(row["updated_turn_id"])
        or output_window.updated_attempt_id != updated_attempt_id
    ):
        raise WorkExecutionPersistenceError(
            "WorkRun OutputWindow row and typed payload disagree"
        )
    return output_window, snapshot_hash


def _budget_from_row(row: sqlite3.Row) -> WorkRunBudget:
    try:
        budget = WorkRunBudget(
            max_attempts=int(row["max_attempts"]),
            soft_active_seconds=float(row["soft_active_seconds"]),
            hard_active_seconds=float(row["hard_active_seconds"]),
            attempts_started=int(row["attempts_started"]),
            active_seconds_consumed=float(row["active_seconds_consumed"]),
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "WorkRun budget aggregate is invalid"
        ) from exc
    if (
        budget.max_attempts != _MAX_ATTEMPTS
        or budget.soft_active_seconds != _SOFT_ACTIVE_SECONDS
        or budget.hard_active_seconds != _HARD_ACTIVE_SECONDS
    ):
        raise WorkExecutionPersistenceError(
            "WorkRun budget envelope disagrees with the frozen authority"
        )
    return budget


def _same_budget_envelope(
    left: WorkRunBudget,
    right: WorkRunBudget,
) -> bool:
    return (
        left.max_attempts == right.max_attempts
        and left.soft_active_seconds == right.soft_active_seconds
        and left.hard_active_seconds == right.hard_active_seconds
    )


def _load_replay(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    payload_hash: str,
) -> WorkExecutionMutationResult | None:
    row = conn.execute(
        "SELECT operation, session_id, work_run_id, payload_hash, result_json "
        "FROM insession_work_run_apply_receipts WHERE apply_id=?",
        (apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != operation
        or str(row["session_id"]) != session_id
        or str(row["payload_hash"]) != payload_hash
    ):
        raise WorkExecutionApplyIdCollision(
            "WorkRun apply id was reused for a different operation or payload"
        )
    stored = WorkExecutionMutationResult.model_validate_json(str(row["result_json"]))
    if stored.work_run_id != str(row["work_run_id"]):
        raise WorkExecutionPersistenceError(
            "stored WorkRun apply receipt owner is corrupt"
        )
    if stored.budget_transition is not None:
        _require_budgeted_receipt_charge(
            conn,
            apply_id=apply_id,
            operation=operation,
            session_id=session_id,
            work_run_id=stored.work_run_id,
            work_run_revision=stored.work_run_revision,
            work_run_status=stored.work_run_status,
            work_run_reason=stored.work_run_reason,
            window_state_version=stored.window_state_version,
            budget_transition=stored.budget_transition,
        )
    return stored.model_copy(update={"status": "replayed"})


def _load_budget_charge_replay(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    payload_hash: str,
) -> WorkRunBudgetChargeMutationResult | None:
    row = conn.execute(
        "SELECT operation, session_id, work_run_id, payload_hash, result_json "
        "FROM insession_work_run_apply_receipts WHERE apply_id=?",
        (apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != operation
        or str(row["session_id"]) != session_id
        or str(row["work_run_id"]) != work_run_id
        or str(row["payload_hash"]) != payload_hash
    ):
        raise WorkExecutionApplyIdCollision(
            "WorkRun apply id was reused for a different operation or payload"
        )
    run_row = conn.execute(
        "SELECT * FROM insession_work_runs WHERE work_run_id=? AND session_id=?",
        (work_run_id, session_id),
    ).fetchone()
    if run_row is None:
        raise WorkExecutionPersistenceError(
            "WorkRun budget charge receipt lost its WorkRun authority"
        )
    current_budget = _budget_from_row(run_row)
    attempt_count = _require_attempt_row_aggregate(
        conn,
        work_run_id=work_run_id,
        expected_attempts_started=current_budget.attempts_started,
    )
    charges = _load_budget_charges(
        conn,
        session_id=session_id,
        work_run_id=work_run_id,
        current_work_run_revision=int(run_row["revision"]),
        current_budget=current_budget,
        attempt_count=attempt_count,
    )
    charge = next(
        (item for item in charges if item.budget_charge_id == apply_id),
        None,
    )
    if charge is None:
        raise WorkExecutionPersistenceError(
            "WorkRun budget charge receipt lost its ledger entry"
        )
    expected_status, expected_reason = _budget_charge_status_projection(
        charge.transition.disposition
    )
    expected = WorkRunBudgetChargeMutationResult(
        status="applied",
        budget_charge_id=charge.budget_charge_id,
        work_run_id=work_run_id,
        checkpoint_id=charge.checkpoint_id,
        work_run_revision=charge.work_run_revision_after,
        work_run_status=expected_status,
        work_run_reason=expected_reason,
        transition=charge.transition,
        window_state_version=charge.window_state_version_after,
    )
    try:
        stored = WorkRunBudgetChargeMutationResult.model_validate_json(
            str(row["result_json"])
        )
    except ValueError as exc:
        raise WorkExecutionPersistenceError(
            "WorkRun budget charge receipt is invalid"
        ) from exc
    if stored != expected:
        raise WorkExecutionPersistenceError(
            "WorkRun budget charge receipt disagrees with its ledger"
        )
    return expected.model_copy(update={"status": "replayed"})


def _require_attempt_row_aggregate(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    expected_attempts_started: int,
) -> int:
    ordinals = tuple(
        int(row["ordinal"])
        for row in conn.execute(
            "SELECT ordinal FROM insession_work_run_attempts "
            "WHERE work_run_id=? ORDER BY ordinal",
            (work_run_id,),
        ).fetchall()
    )
    if (
        len(ordinals) != expected_attempts_started
        or ordinals != tuple(range(1, len(ordinals) + 1))
    ):
        raise WorkExecutionPersistenceError(
            "WorkRun Attempt aggregate disagrees with stored Attempt rows"
        )
    return len(ordinals)


def _insert_receipt(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    payload_hash: str,
    result: BaseModel,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_work_run_apply_receipts "
        "(apply_id, operation, session_id, work_run_id, payload_hash, result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (apply_id, operation, session_id, work_run_id, payload_hash, _model_json(result), now),
    )


def _require_run_row(
    conn: sqlite3.Connection,
    work_run_id: str,
    session_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM insession_work_runs WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if row is None or str(row["session_id"]) != session_id:
        raise WorkExecutionPersistenceError("unknown WorkRun in this Session")
    return row


def _require_run_revision(row: sqlite3.Row, expected: int) -> None:
    actual = int(row["revision"])
    if actual != expected:
        raise WorkExecutionRevisionConflict(expected=expected, actual=actual)


def _require_progress_revision(
    snapshot: AcceptanceProgressSnapshot,
    expected: int,
) -> None:
    if snapshot.revision != expected:
        raise WorkExecutionProgressRevisionConflict(
            expected=expected,
            actual=snapshot.revision,
        )


def _require_active_attempt(
    conn: sqlite3.Connection,
    work_run_id: str,
    attempt_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM insession_work_run_attempts "
        "WHERE attempt_id=? AND work_run_id=?",
        (attempt_id, work_run_id),
    ).fetchone()
    if row is None or str(row["status"]) != "active":
        raise WorkExecutionPersistenceError("unknown or closed current Attempt")
    return row


def _require_active_window(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT turn_id, window_state, stage, state_version, "
        "turn_workrun_link_revision, current_work_run_id, current_attempt_id, "
        "pending_operation_id, latest_checkpoint_id FROM turn_execution_windows "
        "WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if row is None or str(row["turn_id"] or "") != turn_id or str(row["window_state"]) != "active":
        raise WorkExecutionPersistenceError("Turn does not own the active Window")
    actual = int(row["state_version"])
    if actual != expected_window_revision:
        raise TurnExecutionWindowRevisionConflict(expected=expected_window_revision, actual=actual)
    owner = conn.execute(
        "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    if owner is None or str(owner["session_id"]) != session_id or str(owner["status"]) != "running":
        raise WorkExecutionPersistenceError("WorkRun mutation requires a running authoritative Turn")
    return row


def _require_owned_work_run_window(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_window_revision: int,
) -> sqlite3.Row:
    row = _require_active_window(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
    )
    if str(row["current_work_run_id"] or "") != work_run_id:
        raise WorkExecutionPersistenceError("Turn Window does not point to this WorkRun")
    return row


def _require_owned_attempt_window(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    expected_window_revision: int,
) -> sqlite3.Row:
    row = _require_owned_work_run_window(
        conn,
        session_id,
        turn_id,
        work_run_id,
        expected_window_revision,
    )
    if str(row["current_attempt_id"] or "") != attempt_id:
        raise WorkExecutionPersistenceError("Turn Window does not point to this Attempt")
    return row


def _acceptance_ids(raw: object) -> tuple[str, ...]:
    try:
        payload = json.loads(str(raw))
        ids = tuple(str(item["acceptance_id"]) for item in payload)
    except (TypeError, ValueError, KeyError) as exc:
        raise WorkExecutionPersistenceError("TaskNode Acceptance payload is corrupt") from exc
    if not ids or any(not item.strip() for item in ids) or len(ids) != len(set(ids)):
        raise WorkExecutionPersistenceError("TaskNode Acceptance identities are invalid")
    return ids


def _require_mutation_identifiers(
    session_id: str,
    turn_id: str,
    work_run_id: str,
    apply_id: str,
) -> None:
    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("work_run_id", work_run_id)
    _require_identifier("apply_id", apply_id)


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty identifier of at most 200 characters")


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _allocate_id(deps: StoreDeps, prefix: str) -> str:
    return f"{prefix}_{deps.new_id()}"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_json(value: BaseModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_hash(value: object) -> str:
    return _text_hash(_canonical_json(value))


__all__ = [
    'PendingUserQuestion',
    'ReadyTaskNodeExecutionCandidate',
    'RecoverableTaskNodeExecutionCandidate',
    'StoredAttempt',
    'StoredMaterializedToolCall',
    'StoredWorkRun',
    'TaskNodeExecutionFrontier',
    'TurnLinkedNonterminalWorkRunCandidate',
    "WorkExecutionApplyIdCollision",
    'WorkExecutionMutationResult',
    "WorkExecutionOutputRevisionConflict",
    "WorkExecutionPersistenceError",
    "WorkExecutionProgressRevisionConflict",
    "WorkExecutionRevisionConflict",
    "append_work_run_tool_result",
    "close_work_run_attempt",
    "commit_work_run_attempt_decision",
    "commit_work_run_output_action",
    "create_task_node_work_run",
    "detach_safe_work_run_lane",
    "get_completed_task_final_delivery_id",
    "get_active_task_graph_execution_replan_request",
    "get_current_task_node_dependency_deliveries",
    "get_work_run",
    "list_turn_linked_work_run_ids",
    "project_task_node_execution_frontier",
    "list_turn_linked_nonterminal_work_runs",
    "resume_active_work_run_attempt",
    "resume_decided_readonly_tool_attempt",
    "resume_idle_readonly_work_run_and_start_attempt",
    "start_work_run_attempt",
]
