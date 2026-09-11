"""可恢复 TaskNode 语义验证的 SQLite 权威状态。

持久化请求只存储不可变引用和规范 binding digest。OutputWindow 内容与 ToolResult
载荷仍保留在现有的单一权威记录中，仅为待处理 verifier 调用进行物化。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from .....tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from personagraph.l2.work_run import (
    AcceptanceProgressSnapshot,
    AttemptStatus,
    Attempt,
    NodeVerificationResult,
    PreparedTaskNodeVerification,
    ResolvedTaskNodeDelivery,
    SupportingToolResult,
    TaskNodeDelivery,
    TaskNodeSubject,
    TaskNodeVerificationMutationResult,
    TaskNodeVerificationRecord,
    TaskNodeVerificationRequestStatus,
    TaskNodeVerificationRequest,
    ToolResultStatus,
    ToolResult,
    WorkRunBudgetDisposition,
    WorkRunStatus,
    WorkRun,
)
from ...deps import StoreDeps
from ..delivery import task_delivery_validation as task_delivery_validation_records
from .work_execution import (
    WorkExecutionApplyIdCollision,
    WorkExecutionPersistenceError,
    _allocate_id,
    _budget_from_row,
    _canonical_json,
    _insert_budget_charge,
    _load_output_window,
    _load_progress,
    _model_json,
    _payload_hash,
    _project_task_node_budget_failure,
    _require_active_window,
    _require_identifier,
    _require_mutation_identifiers,
    _require_owned_work_run_window,
    _require_positive,
    _require_progress_revision,
    _require_budgeted_receipt_charge,
    _require_run_revision,
    _require_run_row,
    _require_settlement_budget_transition,
    _settlement_budget_checkpoint_id,
    _normalize_active_seconds_delta,
)


class TaskNodeVerificationRequestRevisionConflict(WorkExecutionPersistenceError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"verification request revision conflict: expected {expected}, actual {actual}"
        )


class VerificationSettlementReceiptNotFound(WorkExecutionPersistenceError):
    """所请求的验证结算 receipt 不存在。"""


def prepare_task_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_output_revision: int,
    expected_window_revision: int,
    apply_id: str,
    verification_request_id: str | None = None,
    recover_unprepared_submit: bool = False,
) -> TaskNodeVerificationMutationResult:
    """从 submit lock 物化一个不可变逻辑请求。"""

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive("expected_progress_revision", expected_progress_revision)
    _require_positive("expected_output_revision", expected_output_revision)
    _require_positive("expected_window_revision", expected_window_revision)
    if verification_request_id is not None:
        _require_identifier("verification_request_id", verification_request_id)
    if not isinstance(recover_unprepared_submit, bool):
        raise TypeError("recover_unprepared_submit must be a bool")
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_output_revision": expected_output_revision,
        "expected_window_revision": expected_window_revision,
        "requested_verification_request_id": verification_request_id,
    }
    if recover_unprepared_submit:
        payload["recover_unprepared_submit"] = True
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="prepare_verification",
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
        recovery_link_revision: int | None = None
        if recover_unprepared_submit:
            if (
                window["current_work_run_id"] is not None
                or window["current_attempt_id"] is not None
                or window["pending_operation_id"] is not None
                or window["latest_checkpoint_id"] is not None
            ):
                raise WorkExecutionPersistenceError(
                    "unprepared verification recovery requires an unbound active Window"
                )
        elif str(window["current_work_run_id"] or "") != work_run_id:
            raise WorkExecutionPersistenceError(
                "Turn Window does not point to this WorkRun"
            )
        if window["current_attempt_id"] is not None:
            raise WorkExecutionPersistenceError(
                "verification prepare requires no current Attempt"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        current_budget = _budget_from_row(run_row)
        if (
            current_budget.active_seconds_consumed
            >= current_budget.hard_active_seconds
        ):
            raise WorkExecutionPersistenceError(
                "verification provider cannot start after hard active-time exhaustion"
            )
        if (
            str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or str(run_row["reason"] or "") != "verification_pending"
            or run_row["current_attempt_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun is not an unprepared verification_pending submit"
            )
        if recover_unprepared_submit:
            previous_turn_id = str(run_row["updated_turn_id"] or "")
            if not previous_turn_id or previous_turn_id == turn_id:
                raise WorkExecutionPersistenceError(
                    "unprepared verification recovery requires a different prior Turn"
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
                    "previous submit Turn has not durably settled as incomplete"
                )
            if conn.execute(
                "SELECT 1 FROM insession_work_run_turn_links "
                "WHERE session_id=? AND turn_id=? AND work_run_id=? LIMIT 1",
                (session_id, previous_turn_id, work_run_id),
            ).fetchone() is None:
                raise WorkExecutionPersistenceError(
                    "unprepared submit lost its previous Turn ownership link"
                )
            if conn.execute(
                "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
                "AND turn_id=? AND insession_task_id=? LIMIT 1",
                (session_id, turn_id, str(run_row["insession_task_id"])),
            ).fetchone() is None:
                raise WorkExecutionPersistenceError(
                    "recovery Turn is not authoritatively linked to this Task"
                )
            recovery_link_revision = (
                int(window["turn_workrun_link_revision"] or 0) + 1
            )
            if conn.execute(
                "SELECT 1 FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id=?",
                (turn_id, work_run_id),
            ).fetchone() is not None:
                raise WorkExecutionPersistenceError(
                    "recovery Turn already owns a conflicting WorkRun link"
                )
            conn.execute(
                "INSERT INTO insession_work_run_turn_links "
                "(session_id, turn_id, work_run_id, link_revision, relation, "
                "created_at) VALUES (?, ?, ?, ?, 'continued', ?)",
                (
                    session_id,
                    turn_id,
                    work_run_id,
                    recovery_link_revision,
                    now,
                ),
            )

        progress, progress_hash = _load_progress_with_hash(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        output_window, output_hash = _load_output_window(conn, work_run_id)
        if output_window.output_revision != expected_output_revision:
            from .work_execution import WorkExecutionOutputRevisionConflict

            raise WorkExecutionOutputRevisionConflict(
                expected=expected_output_revision,
                actual=output_window.output_revision,
            )
        if not output_window.content.strip():
            raise WorkExecutionPersistenceError(
                "verification requires a non-empty OutputWindow"
            )
        output_row = conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows WHERE work_run_id=?",
            (work_run_id,),
        ).fetchone()
        if output_row is None or output_row["frozen_at"] is not None:
            raise WorkExecutionPersistenceError(
                "verification cannot prepare a missing or frozen OutputWindow"
            )
        if progress.evaluated_output_revision != output_window.output_revision:
            raise WorkExecutionPersistenceError(
                "AcceptanceProgress is not bound to the submitted OutputWindow"
            )
        if not all(item.model_claimed_satisfied for item in progress.items):
            raise WorkExecutionPersistenceError(
                "verification requires every Acceptance self-claim to be true"
            )

        subject, node_title, node_objective, acceptances = _load_current_node(
            conn, run_row
        )
        _require_node_status(conn, subject=subject, expected_status="active")
        acceptance_ids = tuple(item.acceptance_id for item in acceptances)
        if set(item.acceptance_id for item in progress.items) != set(acceptance_ids):
            raise WorkExecutionPersistenceError(
                "AcceptanceProgress does not cover the current node Acceptance set"
            )
        submitted_attempt = _load_current_submit_attempt(
            conn,
            work_run_id=work_run_id,
            output_revision=output_window.output_revision,
        )
        supporting_ids = tuple(
            sorted(
                {
                    result_id
                    for item in progress.items
                    for result_id in item.supporting_tool_result_ids
                }
            )
        )
        supporting_results = _load_supporting_tool_results(
            conn,
            work_run_id=work_run_id,
            result_ids=supporting_ids,
            before_attempt_ordinal=submitted_attempt.ordinal,
        )
        dependency_delivery_ids = _load_bound_dependency_delivery_ids(
            conn,
            session_id=session_id,
            subject=subject,
        )
        prepared_budget = _budget_from_row(run_row)
        allocated_request_id = verification_request_id or _allocate_id(
            deps, "verification"
        )
        binding_hash = _verification_binding_hash(
            verification_request_id=allocated_request_id,
            session_id=session_id,
            request_turn_id=turn_id,
            work_run_id=work_run_id,
            subject=subject,
            node_title=node_title,
            node_objective=node_objective,
            submitted_attempt=submitted_attempt,
            locked_work_run_revision=expected_work_run_revision,
            progress=progress,
            progress_hash=progress_hash,
            output_window=output_window,
            output_hash=output_hash,
            acceptances=acceptances,
            supporting_results=supporting_results,
            dependency_delivery_ids=dependency_delivery_ids,
            prepared_budget=prepared_budget,
        )
        next_run_revision = expected_work_run_revision + 1
        try:
            conn.execute(
                "INSERT INTO insession_work_run_verification_requests "
                "(verification_request_id, execution_subject_id, session_id, "
                "request_turn_id, work_run_id, "
                "subject_kind, insession_task_id, graph_revision, "
                "insession_task_node_id, node_revision, "
                "submitted_attempt_id, output_revision, acceptance_progress_revision, "
                "acceptance_ids_json, supporting_tool_result_ids_json, "
                "dependency_delivery_ids_json, "
                "locked_work_run_revision, request_binding_hash, prepared_budget_json, "
                "request_revision, status, technical_error_code, result_json, all_pass, "
                "created_at, updated_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, 'task_node', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, "
                "'pending', NULL, NULL, NULL, ?, ?, NULL)",
                (
                    allocated_request_id,
                    str(run_row["execution_subject_id"]),
                    session_id,
                    turn_id,
                    work_run_id,
                    subject.task_id,
                    subject.graph_revision,
                    subject.node_id,
                    subject.node_revision,
                    submitted_attempt.attempt_id,
                    output_window.output_revision,
                    progress.revision,
                    _canonical_json(acceptance_ids),
                    _canonical_json(supporting_ids),
                    _canonical_json(dependency_delivery_ids),
                    expected_work_run_revision,
                    binding_hash,
                    _model_json(prepared_budget),
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkExecutionPersistenceError(
                "verification request conflicts with existing submit authority"
            ) from exc
        if conn.execute(
            "UPDATE insession_work_runs SET current_verification_request_id=?, "
            "revision=?, updated_turn_id=?, updated_at=? WHERE work_run_id=? "
            "AND revision=? AND status='active' AND reason='verification_pending' "
            "AND current_attempt_id IS NULL AND current_verification_request_id IS NULL",
            (
                allocated_request_id,
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during verification prepare"
            )
        next_window_revision = expected_window_revision + 1
        if recover_unprepared_submit:
            assert recovery_link_revision is not None
            window_sql = (
                "UPDATE turn_execution_windows SET current_work_run_id=?, "
                "current_attempt_id=NULL, latest_checkpoint_id=?, "
                "stage='VERIFICATION', turn_workrun_link_revision=?, "
                "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
                "AND window_state='active' AND current_work_run_id IS NULL "
                "AND current_attempt_id IS NULL AND pending_operation_id IS NULL "
                "AND latest_checkpoint_id IS NULL AND state_version=?"
            )
            window_params = (
                work_run_id,
                allocated_request_id,
                recovery_link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            )
        else:
            window_sql = (
                "UPDATE turn_execution_windows SET latest_checkpoint_id=?, "
                "stage='VERIFICATION', state_version=?, updated_at=? "
                "WHERE session_id=? AND turn_id=? AND window_state='active' "
                "AND current_work_run_id=? AND current_attempt_id IS NULL "
                "AND state_version=?"
            )
            window_params = (
                allocated_request_id,
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            )
        if conn.execute(window_sql, window_params).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during verification prepare"
            )
        result = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=allocated_request_id,
            verification_request_revision=1,
            verification_request_status=(
                TaskNodeVerificationRequestStatus.PENDING
            ),
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=WorkRunStatus.ACTIVE,
            work_run_reason="verification_pending",
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="prepare_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def get_task_node_verification_record(
    deps: StoreDeps,
    *,
    session_id: str,
    verification_request_id: str,
) -> TaskNodeVerificationRecord:
    """从单个 SQLite 快照读取任意状态下的一个请求。"""

    _require_identifier("session_id", session_id)
    _require_identifier("verification_request_id", verification_request_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        row = _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        record = _record_from_row(row)
        if record.request.status in {
            TaskNodeVerificationRequestStatus.PENDING,
            TaskNodeVerificationRequestStatus.INTERRUPTED,
        }:
            _revalidate_binding(conn, row)
            _validate_request_lifecycle(conn, record)
        elif record.result is not None and record.result.all_pass:
            _revalidate_binding(conn, row, require_current_task=False)
            _validate_request_lifecycle(conn, record)
        else:
            # 后续 Attempt 推进仅表示当前值的 Output/Progress 记录后，已完成但未通过的
            # 请求仍作为历史保留。当这些精确引用仍为当前值时，验证完整 digest；
            # 推进后，只有其自包含的有类型结果仍可读取，且不能投影为可调用输入。
            current = conn.execute(
                "SELECT output.output_revision, progress.progress_revision "
                "FROM insession_work_run_output_windows AS output "
                "JOIN insession_work_run_acceptance_progress AS progress "
                "ON progress.work_run_id=output.work_run_id "
                "WHERE output.work_run_id=?",
                (record.request.work_run_id,),
            ).fetchone()
            if (
                current is not None
                and int(current["output_revision"]) == record.request.output_revision
                and int(current["progress_revision"])
                == record.request.acceptance_progress_revision
            ):
                _revalidate_binding(conn, row)
        conn.commit()
        return record
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_prepared_task_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    invocation_turn_id: str,
    verification_request_id: str,
) -> PreparedTaskNodeVerification:
    """在不创建持久化副本的情况下重建精确待处理 verifier 输入。"""

    _require_identifier("session_id", session_id)
    _require_identifier("invocation_turn_id", invocation_turn_id)
    _require_identifier("verification_request_id", verification_request_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        row = _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        prepared = _materialize_prepared(
            conn,
            row=row,
            invocation_turn_id=invocation_turn_id,
        )
        conn.commit()
        return prepared
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_task_node_delivery(
    deps: StoreDeps,
    *,
    session_id: str,
    delivery_id: str,
) -> ResolvedTaskNodeDelivery:
    """加载一个已验证且仅含引用的 delivery；发生漂移时失败关闭。"""

    _require_identifier("session_id", session_id)
    _require_identifier("delivery_id", delivery_id)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        delivery = _load_task_node_delivery(
            conn,
            session_id=session_id,
            delivery_id=delivery_id,
        )
        conn.commit()
        return delivery
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def replay_task_node_verification_settlement(
    deps: StoreDeps,
    *,
    session_id: str,
    work_run_id: str,
    verification_request_id: str,
    apply_id: str,
    operation: Literal[
        "commit_verification_result",
        "interrupt_verification",
    ],
) -> TaskNodeVerificationMutationResult:
    """读取并验证一个已结算 receipt，而无需临时 provider 数据。"""

    _require_identifier("session_id", session_id)
    _require_identifier("work_run_id", work_run_id)
    _require_identifier("verification_request_id", verification_request_id)
    _require_identifier("apply_id", apply_id)
    if operation not in {
        "commit_verification_result",
        "interrupt_verification",
    }:
        raise ValueError("unsupported verification settlement operation")
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT operation, session_id, work_run_id, result_json "
            "FROM insession_work_run_apply_receipts WHERE apply_id=?",
            (apply_id,),
        ).fetchone()
        if row is None:
            raise VerificationSettlementReceiptNotFound(
                "verification settlement receipt was not found"
            )
        if (
            str(row["operation"]) != operation
            or str(row["session_id"]) != session_id
            or str(row["work_run_id"]) != work_run_id
        ):
            raise WorkExecutionApplyIdCollision(
                "verification settlement apply id belongs to another authority"
            )
        try:
            stored = TaskNodeVerificationMutationResult.model_validate_json(
                str(row["result_json"])
            )
        except ValueError as exc:
            raise WorkExecutionPersistenceError(
                "stored verification settlement receipt is corrupt"
            ) from exc
        if (
            stored.work_run_id != work_run_id
            or stored.verification_request_id != verification_request_id
        ):
            raise WorkExecutionPersistenceError(
                "stored verification settlement receipt binding is corrupt"
            )
        if stored.budget_transition is not None:
            _require_budgeted_receipt_charge(
                conn,
                apply_id=apply_id,
                operation=operation,
                session_id=session_id,
                work_run_id=work_run_id,
                work_run_revision=stored.work_run_revision,
                work_run_status=stored.work_run_status,
                work_run_reason=stored.work_run_reason,
                window_state_version=stored.window_state_version,
                budget_transition=stored.budget_transition,
            )
        conn.commit()
        return stored.model_copy(update={"status": "replayed"})
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def commit_task_node_verification_result(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    verification_request_id: str,
    result: NodeVerificationResult,
    expected_work_run_revision: int,
    expected_verification_request_revision: int,
    expected_window_revision: int,
    apply_id: str,
    delivery_id: str | None = None,
    active_seconds_delta: float | None = None,
    task_delivery_candidate_settlement: (
        task_delivery_validation_records.TaskDeliveryCandidateSettlementIntent
        | None
    ) = None,
) -> TaskNodeVerificationMutationResult:
    """将一个当前请求结算为未通过，或结算为一个终态 NodeDelivery。"""

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("verification_request_id", verification_request_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive(
        "expected_verification_request_revision",
        expected_verification_request_revision,
    )
    _require_positive("expected_window_revision", expected_window_revision)
    if delivery_id is not None:
        _require_identifier("delivery_id", delivery_id)
    normalized_delta = _normalize_active_seconds_delta(active_seconds_delta)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "verification_request_id": verification_request_id,
        "result": result.model_dump(mode="json"),
        "expected_work_run_revision": expected_work_run_revision,
        "expected_verification_request_revision": (
            expected_verification_request_revision
        ),
        "expected_window_revision": expected_window_revision,
        "requested_delivery_id": delivery_id,
    }
    if normalized_delta is not None:
        payload["active_seconds_delta"] = normalized_delta
    if task_delivery_candidate_settlement is not None:
        if not isinstance(
            task_delivery_candidate_settlement,
            task_delivery_validation_records.TaskDeliveryCandidateSettlementIntent,
        ):
            raise TypeError(
                "task delivery candidate settlement must be its typed intent"
            )
        payload["task_delivery_candidate_settlement"] = (
            task_delivery_candidate_settlement.model_dump(mode="json")
        )
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            if task_delivery_candidate_settlement is not None:
                if replay.delivery_id is None:
                    raise WorkExecutionPersistenceError(
                        "candidate settlement replay lost its root Delivery"
                    )
                task_delivery_validation_records.replay_task_delivery_candidate_settlement_in_transaction(
                    conn,
                    intent=task_delivery_candidate_settlement,
                    node_verification_commit_apply_id=apply_id,
                    expected_root_delivery_id=replay.delivery_id,
                )
            return replay
        window = _require_owned_work_run_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            expected_window_revision,
        )
        if (
            str(window["latest_checkpoint_id"] or "") != verification_request_id
            or str(window["stage"] or "") != "VERIFICATION"
        ):
            raise WorkExecutionPersistenceError(
                "Turn Window does not own the current verification checkpoint"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if (
            str(run_row["status"]) != WorkRunStatus.ACTIVE.value
            or str(run_row["reason"] or "") != "verification_pending"
            or str(run_row["current_verification_request_id"] or "")
            != verification_request_id
            or run_row["current_attempt_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun does not own the pending verification request"
            )
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=work_run_id,
            active_seconds_delta=normalized_delta,
        )
        request_row = _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        _require_request_revision(
            request_row, expected_verification_request_revision
        )
        if str(request_row["status"]) != "pending":
            raise WorkExecutionPersistenceError(
                "only a pending verification request can accept a result"
            )
        record = _record_from_row(request_row)
        request = record.request
        if (
            result.verification_request_id != verification_request_id
            or result.verification_request_revision
            != expected_verification_request_revision
            or result.work_run_id != work_run_id
            or result.locked_work_run_revision
            != request.locked_work_run_revision
            or result.subject != request.subject
            or result.submitted_attempt_id != request.submitted_attempt_id
            or result.output_revision != request.output_revision
            or result.acceptance_progress_revision
            != request.acceptance_progress_revision
            or tuple(item.acceptance_id for item in result.acceptance_results)
            != request.acceptance_ids
        ):
            raise WorkExecutionPersistenceError(
                "semantic result does not match the current verification request"
            )
        _materialize_prepared(
            conn,
            row=request_row,
            invocation_turn_id=turn_id,
        )
        next_request_revision = expected_verification_request_revision + 1
        next_run_revision = expected_work_run_revision + 1
        next_window_revision = expected_window_revision + 1
        if (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        ):
            checkpoint_id = _settlement_budget_checkpoint_id(
                "commit_verification_result", apply_id
            )
            _insert_budget_charge(
                conn,
                budget_charge_id=apply_id,
                operation="commit_verification_result",
                session_id=session_id,
                work_run_id=work_run_id,
                turn_id=turn_id,
                checkpoint_id=checkpoint_id,
                work_run_revision_before=expected_work_run_revision,
                work_run_revision_after=next_run_revision,
                window_state_version_before=expected_window_revision,
                window_state_version_after=next_window_revision,
                transition=budget_transition,
                work_run_status_after=WorkRunStatus.FAILED,
                work_run_reason_after="work_run_limit_reached",
                now=now,
            )
            if conn.execute(
                "UPDATE insession_work_run_verification_requests SET "
                "status='interrupted', request_revision=?, "
                "technical_error_code='work_run_limit_reached', result_json=NULL, "
                "all_pass=NULL, updated_at=?, completed_at=NULL "
                "WHERE verification_request_id=? AND work_run_id=? "
                "AND request_revision=? AND status='pending'",
                (
                    next_request_revision,
                    now,
                    verification_request_id,
                    work_run_id,
                    expected_verification_request_revision,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "verification request changed during hard-budget settlement"
                )
            _project_task_node_budget_failure(conn, run_row=run_row, now=now)
            if conn.execute(
                "UPDATE insession_work_runs SET status='failed', "
                "reason='work_run_limit_reached', revision=?, "
                "active_seconds_consumed=?, current_verification_request_id=NULL, "
                "updated_turn_id=?, updated_at=? WHERE work_run_id=? AND revision=? "
                "AND status='active' AND reason='verification_pending' "
                "AND current_verification_request_id=?",
                (
                    next_run_revision,
                    budget_transition.budget_after.active_seconds_consumed,
                    turn_id,
                    now,
                    work_run_id,
                    expected_work_run_revision,
                    verification_request_id,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "WorkRun changed during hard-budget verification settlement"
                )
            if conn.execute(
                "UPDATE turn_execution_windows SET latest_checkpoint_id=?, "
                "stage='VERIFICATION', state_version=?, updated_at=? "
                "WHERE session_id=? AND turn_id=? AND window_state='active' "
                "AND current_work_run_id=? AND current_attempt_id IS NULL "
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
                    "Turn Window changed during hard-budget verification settlement"
                )
            mutation = TaskNodeVerificationMutationResult(
                status="applied",
                verification_request_id=verification_request_id,
                verification_request_revision=next_request_revision,
                verification_request_status=(
                    TaskNodeVerificationRequestStatus.INTERRUPTED
                ),
                work_run_id=work_run_id,
                work_run_revision=next_run_revision,
                work_run_status=WorkRunStatus.FAILED,
                work_run_reason="work_run_limit_reached",
                window_state_version=next_window_revision,
                budget_transition=budget_transition,
            )
            _insert_receipt(
                conn,
                apply_id=apply_id,
                operation="commit_verification_result",
                session_id=session_id,
                work_run_id=work_run_id,
                payload_hash=payload_hash,
                result=mutation,
                now=now,
            )
            return mutation

        result_json = _model_json(result)
        if conn.execute(
            "UPDATE insession_work_run_verification_requests SET status='completed', "
            "request_revision=?, technical_error_code=NULL, result_json=?, all_pass=?, "
            "updated_at=?, completed_at=? WHERE verification_request_id=? "
            "AND work_run_id=? AND request_revision=? AND status='pending'",
            (
                next_request_revision,
                result_json,
                int(result.all_pass),
                now,
                now,
                verification_request_id,
                work_run_id,
                expected_verification_request_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "verification request changed during result commit"
            )

        allocated_delivery_id: str | None = None
        if result.all_pass:
            allocated_delivery_id = delivery_id or _allocate_id(deps, "delivery")
            subject = request.subject
            try:
                conn.execute(
                    "INSERT INTO insession_task_node_deliveries "
                    "(delivery_id, session_id, work_run_id, insession_task_id, "
                    "graph_revision, insession_task_node_id, node_revision, "
                    "verification_request_id, submitted_attempt_id, output_revision, "
                    "created_turn_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        allocated_delivery_id,
                        session_id,
                        work_run_id,
                        subject.task_id,
                        subject.graph_revision,
                        subject.node_id,
                        subject.node_revision,
                        verification_request_id,
                        request.submitted_attempt_id,
                        request.output_revision,
                        turn_id,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkExecutionPersistenceError(
                    "NodeDelivery conflicts with existing delivery authority"
                ) from exc
            if conn.execute(
                "UPDATE insession_work_run_output_windows SET frozen_at=? "
                "WHERE work_run_id=? AND output_revision=? AND frozen_at IS NULL",
                (now, work_run_id, request.output_revision),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "verified OutputWindow could not be frozen"
                )
            if conn.execute(
                "UPDATE insession_task_node_states SET status='completed', "
                "state_version=state_version+1, updated_at=? "
                "WHERE insession_task_id=? AND insession_task_node_id=? "
                "AND node_revision=? AND status='active'",
                (
                    now,
                    subject.task_id,
                    subject.node_id,
                    subject.node_revision,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "TaskNode changed during verified delivery commit"
                )
            if conn.execute(
                "UPDATE insession_work_runs SET status='completed', "
                "reason='verification_passed', revision=?, "
                "active_seconds_consumed=?, "
                "current_verification_request_id=NULL, updated_turn_id=?, updated_at=? "
                "WHERE work_run_id=? AND revision=? AND status='active' "
                "AND reason='verification_pending' "
                "AND current_verification_request_id=?",
                (
                    next_run_revision,
                    budget_transition.budget_after.active_seconds_consumed,
                    turn_id,
                    now,
                    work_run_id,
                    expected_work_run_revision,
                    verification_request_id,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "WorkRun changed during verified delivery commit"
                )
            next_status = WorkRunStatus.COMPLETED
            next_reason = "verification_passed"
        else:
            if (
                budget_transition.disposition
                is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
            ):
                next_status = WorkRunStatus.TURN_LIMIT_REACHED
                next_reason = "turn_limit_reached"
            else:
                next_status = WorkRunStatus.ACTIVE
                next_reason = None
            if conn.execute(
                "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
                "active_seconds_consumed=?, "
                "current_verification_request_id=NULL, updated_turn_id=?, updated_at=? "
                "WHERE work_run_id=? AND revision=? AND status='active' "
                "AND reason='verification_pending' "
                "AND current_verification_request_id=?",
                (
                    next_status.value,
                    next_reason,
                    next_run_revision,
                    budget_transition.budget_after.active_seconds_consumed,
                    turn_id,
                    now,
                    work_run_id,
                    expected_work_run_revision,
                    verification_request_id,
                ),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "WorkRun changed during non-pass verification commit"
                )

        if result.all_pass:
            window_sql = (
                "UPDATE turn_execution_windows SET current_work_run_id=NULL, "
                "current_attempt_id=NULL, latest_checkpoint_id=NULL, "
                "stage='PERSIST', "
                "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
                "AND window_state='active' AND current_work_run_id=? "
                "AND current_attempt_id IS NULL AND state_version=?"
            )
        else:
            window_sql = (
                "UPDATE turn_execution_windows SET latest_checkpoint_id=NULL, "
                "stage='L2_PLAN', "
                "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
                "AND window_state='active' AND current_work_run_id=? "
                "AND current_attempt_id IS NULL AND state_version=?"
            )
        if conn.execute(
            window_sql,
            (
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during verification result commit"
            )
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "commit_verification_result", apply_id
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
        mutation = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=verification_request_id,
            verification_request_revision=next_request_revision,
            verification_request_status=(
                TaskNodeVerificationRequestStatus.COMPLETED
            ),
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=next_status,
            work_run_reason=next_reason,
            window_state_version=next_window_revision,
            all_pass=result.all_pass,
            delivery_id=allocated_delivery_id,
            budget_transition=budget_transition,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=mutation,
            now=now,
        )
        if result.all_pass:
            assert allocated_delivery_id is not None
            _apply_task_finish_gate(
                conn,
                subject=request.subject,
                work_run_id=work_run_id,
                verification_request_id=verification_request_id,
                delivery_id=allocated_delivery_id,
                now=now,
            )
            if task_delivery_candidate_settlement is not None:
                if allocated_delivery_id != (
                    task_delivery_candidate_settlement.candidate_delivery_id
                ):
                    raise WorkExecutionPersistenceError(
                        "candidate settlement crossed root Delivery allocation"
                    )
                task_delivery_validation_records.settle_task_delivery_candidate_route_in_transaction(
                    conn,
                    intent=task_delivery_candidate_settlement,
                    node_verification_commit_apply_id=apply_id,
                    now=now,
                )
        elif task_delivery_candidate_settlement is not None:
            raise WorkExecutionPersistenceError(
                "candidate route settlement requires node Acceptance PASS"
            )
        return mutation


def interrupt_task_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    verification_request_id: str,
    technical_error_code: str,
    expected_work_run_revision: int,
    expected_verification_request_revision: int,
    expected_window_revision: int,
    apply_id: str,
    active_seconds_delta: float | None = None,
) -> TaskNodeVerificationMutationResult:
    """持久化技术中断，而不伪造语义结果。"""

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("verification_request_id", verification_request_id)
    _require_identifier("technical_error_code", technical_error_code)
    if len(technical_error_code) > 160:
        raise ValueError("technical_error_code must be at most 160 characters")
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive(
        "expected_verification_request_revision",
        expected_verification_request_revision,
    )
    _require_positive("expected_window_revision", expected_window_revision)
    normalized_delta = _normalize_active_seconds_delta(active_seconds_delta)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "verification_request_id": verification_request_id,
        "technical_error_code": technical_error_code,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_verification_request_revision": (
            expected_verification_request_revision
        ),
        "expected_window_revision": expected_window_revision,
    }
    if normalized_delta is not None:
        payload["active_seconds_delta"] = normalized_delta
    return _transition_verification_interrupted(
        deps,
        payload=payload,
        payload_hash=_payload_hash(payload),
        apply_id=apply_id,
        now=deps.now(),
    )


def _transition_verification_interrupted(
    deps: StoreDeps,
    *,
    payload: dict[str, object],
    payload_hash: str,
    apply_id: str,
    now: str,
) -> TaskNodeVerificationMutationResult:
    session_id = str(payload["session_id"])
    turn_id = str(payload["turn_id"])
    work_run_id = str(payload["work_run_id"])
    verification_request_id = str(payload["verification_request_id"])
    technical_error_code = str(payload["technical_error_code"])
    expected_work_run_revision = int(payload["expected_work_run_revision"])
    expected_request_revision = int(
        payload["expected_verification_request_revision"]
    )
    expected_window_revision = int(payload["expected_window_revision"])
    active_seconds_delta = payload.get("active_seconds_delta")
    deps.init_db()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="interrupt_verification",
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
            str(window["latest_checkpoint_id"] or "") != verification_request_id
            or str(window["stage"] or "") != "VERIFICATION"
        ):
            raise WorkExecutionPersistenceError(
                "Turn Window does not own the current verification checkpoint"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        if (
            str(run_row["status"]) != "active"
            or str(run_row["reason"] or "") != "verification_pending"
            or str(run_row["current_verification_request_id"] or "")
            != verification_request_id
            or run_row["current_attempt_id"] is not None
        ):
            raise WorkExecutionPersistenceError(
                "WorkRun does not own the pending verification request"
            )
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=work_run_id,
            active_seconds_delta=(
                float(active_seconds_delta)
                if active_seconds_delta is not None
                else None
            ),
        )
        request_row = _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        _require_request_revision(request_row, expected_request_revision)
        if str(request_row["status"]) != "pending":
            raise WorkExecutionPersistenceError(
                "only a pending request can be technically interrupted"
            )
        # 在为未来 Turn 保留不可变语义绑定前重新计算它。技术错误不能认可已漂移输入。
        _materialize_prepared(
            conn,
            row=request_row,
            invocation_turn_id=turn_id,
        )
        next_request_revision = expected_request_revision + 1
        hard_limit_reached = (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        )
        stored_error_code = (
            "work_run_limit_reached"
            if hard_limit_reached
            else technical_error_code
        )
        if conn.execute(
            "UPDATE insession_work_run_verification_requests SET "
            "status='interrupted', request_revision=?, technical_error_code=?, "
            "updated_at=? WHERE verification_request_id=? AND work_run_id=? "
            "AND request_revision=? AND status='pending'",
            (
                next_request_revision,
                stored_error_code,
                now,
                verification_request_id,
                work_run_id,
                expected_request_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "verification request changed during interruption"
            )
        subject = _subject_from_run_row(run_row)
        if hard_limit_reached:
            _project_task_node_budget_failure(conn, run_row=run_row, now=now)
            next_status = WorkRunStatus.FAILED
            next_reason = "work_run_limit_reached"
        else:
            if conn.execute(
                "UPDATE insession_task_node_states SET status='interrupted', "
                "state_version=state_version+1, updated_at=? "
                "WHERE insession_task_id=? AND insession_task_node_id=? "
                "AND node_revision=? AND status='active'",
                (now, subject.task_id, subject.node_id, subject.node_revision),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "TaskNode changed during verification interruption"
                )
            _bump_active_task_state(conn, subject=subject, now=now)
            next_status = WorkRunStatus.INTERRUPTED
            next_reason = "verification_technical_failure"
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, current_verification_request_id=?, "
            "updated_turn_id=?, updated_at=? WHERE work_run_id=? AND revision=? "
            "AND status='active' AND reason='verification_pending' "
            "AND current_verification_request_id=?",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_transition.budget_after.active_seconds_consumed,
                None if hard_limit_reached else verification_request_id,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                verification_request_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during verification interruption"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET latest_checkpoint_id=?, "
            "stage='VERIFICATION', "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id IS NULL AND state_version=?",
            (
                verification_request_id,
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during verification interruption"
            )
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="interrupt_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "interrupt_verification", apply_id
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
        mutation = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=verification_request_id,
            verification_request_revision=next_request_revision,
            verification_request_status=(
                TaskNodeVerificationRequestStatus.INTERRUPTED
            ),
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=next_status,
            work_run_reason=next_reason,
            window_state_version=next_window_revision,
            budget_transition=budget_transition,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="interrupt_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=mutation,
            now=now,
        )
        return mutation


def resume_task_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    verification_request_id: str,
    expected_work_run_revision: int,
    expected_verification_request_revision: int,
    expected_window_revision: int,
    apply_id: str,
) -> TaskNodeVerificationMutationResult:
    """从新的活动 Turn 认领同一逻辑请求。

    若进程在 ``prepare`` 提交后、provider 结果结算前丢失，待处理请求可能比其调用
    Turn 存活更久。此时重新绑定同一请求，而不是伪造第二个请求；同时推进其
    revision，以隔离由已放弃调用生成的结果。原始 interrupted-request 恢复仍是
    另一条有效分支。
    """

    _require_mutation_identifiers(session_id, turn_id, work_run_id, apply_id)
    _require_identifier("verification_request_id", verification_request_id)
    _require_positive("expected_work_run_revision", expected_work_run_revision)
    _require_positive(
        "expected_verification_request_revision",
        expected_verification_request_revision,
    )
    _require_positive("expected_window_revision", expected_window_revision)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "verification_request_id": verification_request_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_verification_request_revision": (
            expected_verification_request_revision
        ),
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
            operation="resume_verification",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            _validate_verification_resume_replay(
                conn,
                replay=replay,
                session_id=session_id,
                turn_id=turn_id,
                work_run_id=work_run_id,
                verification_request_id=verification_request_id,
                expected_work_run_revision=expected_work_run_revision,
                expected_verification_request_revision=(
                    expected_verification_request_revision
                ),
                expected_window_revision=expected_window_revision,
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
                "resume requires an unbound active Turn Window"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        pending_rebind = (
            str(run_row["status"]) == "active"
            and str(run_row["reason"] or "") == "verification_pending"
            and str(run_row["current_verification_request_id"] or "")
            == verification_request_id
            and run_row["current_attempt_id"] is None
        )
        interrupted_resume = (
            str(run_row["status"]) == "interrupted"
            and str(run_row["reason"] or "") == "verification_technical_failure"
            and str(run_row["current_verification_request_id"] or "")
            == verification_request_id
            and run_row["current_attempt_id"] is None
        )
        if not pending_rebind and not interrupted_resume:
            raise WorkExecutionPersistenceError(
                "WorkRun is not resumable on this verification request"
            )
        request_row = _load_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        _require_request_revision(request_row, expected_verification_request_revision)
        expected_request_status = "pending" if pending_rebind else "interrupted"
        if str(request_row["status"]) != expected_request_status:
            raise WorkExecutionPersistenceError(
                "verification request and WorkRun resume states disagree"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=? LIMIT 1",
            (session_id, turn_id, str(run_row["insession_task_id"])),
        ).fetchone() is None:
            raise WorkExecutionPersistenceError(
                "resume Turn is not authoritatively linked to this Task"
            )
        subject = _subject_from_run_row(run_row)
        node_state = conn.execute(
            "SELECT status FROM insession_task_node_states WHERE insession_task_id=? "
            "AND insession_task_node_id=? AND node_revision=?",
            (subject.task_id, subject.node_id, subject.node_revision),
        ).fetchone()
        expected_node_status = "active" if pending_rebind else "interrupted"
        if node_state is None or str(node_state["status"]) != expected_node_status:
            raise WorkExecutionPersistenceError(
                "TaskNode is not in the expected verification resume state"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status FROM insession_tasks "
            "WHERE insession_task_id=? AND session_id=?",
            (subject.task_id, session_id),
        ).fetchone()
        if (
            task is None
            or task["current_graph_revision"] is None
            or int(task["current_graph_revision"]) != subject.graph_revision
            or str(task["current_status"]) != "active"
        ):
            raise WorkExecutionPersistenceError(
                "Task authority no longer permits verification resume"
            )

        if pending_rebind:
            previous_turn_id = str(run_row["updated_turn_id"] or "")
            if not previous_turn_id or previous_turn_id == turn_id:
                raise WorkExecutionPersistenceError(
                    "pending verification rebind requires a different prior Turn"
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
                    "previous verification Turn has not durably settled as incomplete"
                )
            if conn.execute(
                "SELECT 1 FROM insession_work_run_turn_links "
                "WHERE session_id=? AND turn_id=? AND work_run_id=? LIMIT 1",
                (session_id, previous_turn_id, work_run_id),
            ).fetchone() is None:
                raise WorkExecutionPersistenceError(
                    "pending verification lost its previous Turn ownership link"
                )

        # 认领前重新计算同一不可变语义绑定。
        _revalidate_binding(conn, request_row)
        previous_link_revision = int(window["turn_workrun_link_revision"] or 0)
        link_revision = previous_link_revision + 1
        existing_link = conn.execute(
            "SELECT link_revision, relation FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (turn_id, work_run_id),
        ).fetchone()
        if existing_link is None:
            conn.execute(
                "INSERT INTO insession_work_run_turn_links "
                "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
                "VALUES (?, ?, ?, ?, 'continued', ?)",
                (session_id, turn_id, work_run_id, link_revision, now),
            )
        elif (
            str(existing_link["relation"]) != "continued"
            or int(existing_link["link_revision"]) != link_revision
        ):
            raise WorkExecutionPersistenceError(
                "resume Turn already has a conflicting WorkRun link"
            )
        next_request_revision = expected_verification_request_revision + 1
        if conn.execute(
            "UPDATE insession_work_run_verification_requests SET status='pending', "
            "request_revision=?, technical_error_code=NULL, updated_at=? "
            "WHERE verification_request_id=? AND work_run_id=? "
            "AND request_revision=? AND status=?",
            (
                next_request_revision,
                now,
                verification_request_id,
                work_run_id,
                expected_verification_request_revision,
                expected_request_status,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "verification request changed during resume"
            )
        if interrupted_resume:
            if conn.execute(
                "UPDATE insession_task_node_states SET status='active', "
                "state_version=state_version+1, updated_at=? "
                "WHERE insession_task_id=? AND insession_task_node_id=? "
                "AND node_revision=? AND status='interrupted'",
                (now, subject.task_id, subject.node_id, subject.node_revision),
            ).rowcount != 1:
                raise WorkExecutionPersistenceError(
                    "TaskNode changed during verification resume"
                )
            _bump_active_task_state(conn, subject=subject, now=now)
        next_run_revision = expected_work_run_revision + 1
        run_status_before = "active" if pending_rebind else "interrupted"
        run_reason_before = (
            "verification_pending"
            if pending_rebind
            else "verification_technical_failure"
        )
        if conn.execute(
            "UPDATE insession_work_runs SET status='active', "
            "reason='verification_pending', revision=?, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND revision=? AND status=? AND reason=? "
            "AND current_attempt_id IS NULL "
            "AND current_verification_request_id=?",
            (
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                run_status_before,
                run_reason_before,
                verification_request_id,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "WorkRun changed during verification resume"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=NULL, latest_checkpoint_id=?, "
            "stage='VERIFICATION', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND state_version=?",
            (
                work_run_id,
                verification_request_id,
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise WorkExecutionPersistenceError(
                "Turn Window changed during verification resume"
            )
        mutation = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=verification_request_id,
            verification_request_revision=next_request_revision,
            verification_request_status=TaskNodeVerificationRequestStatus.PENDING,
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=WorkRunStatus.ACTIVE,
            work_run_reason="verification_pending",
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="resume_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=mutation,
            now=now,
        )
        return mutation


def _validate_verification_resume_replay(
    conn: sqlite3.Connection,
    *,
    replay: TaskNodeVerificationMutationResult,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    verification_request_id: str,
    expected_work_run_revision: int,
    expected_verification_request_revision: int,
    expected_window_revision: int,
) -> None:
    """将有类型 resume receipt 隔离在其已提交 generation 与 Turn 链接上。"""

    if (
        replay.verification_request_id != verification_request_id
        or replay.verification_request_revision
        != expected_verification_request_revision + 1
        or replay.verification_request_status
        is not TaskNodeVerificationRequestStatus.PENDING
        or replay.work_run_id != work_run_id
        or replay.work_run_revision != expected_work_run_revision + 1
        or replay.work_run_status is not WorkRunStatus.ACTIVE
        or replay.work_run_reason != "verification_pending"
        or replay.window_state_version != expected_window_revision + 1
    ):
        raise WorkExecutionPersistenceError(
            "verification resume receipt projection is corrupt"
        )
    link = conn.execute(
        "SELECT session_id, relation FROM insession_work_run_turn_links "
        "WHERE turn_id=? AND work_run_id=?",
        (turn_id, work_run_id),
    ).fetchone()
    if (
        link is None
        or str(link["session_id"]) != session_id
        or str(link["relation"]) != "continued"
    ):
        raise WorkExecutionPersistenceError(
            "verification resume receipt lost its durable Turn link"
        )
    request_row = _load_request_row(
        conn,
        session_id=session_id,
        verification_request_id=verification_request_id,
    )
    if (
        str(request_row["work_run_id"]) != work_run_id
        or int(request_row["request_revision"])
        < replay.verification_request_revision
    ):
        raise WorkExecutionPersistenceError(
            "verification resume receipt lost its request generation"
        )
    run_row = _require_run_row(conn, work_run_id, session_id)
    if int(run_row["revision"]) < replay.work_run_revision:
        raise WorkExecutionPersistenceError(
            "verification resume receipt lost its WorkRun generation"
        )


def _materialize_prepared(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    invocation_turn_id: str,
) -> PreparedTaskNodeVerification:
    record = _record_from_row(row)
    request = record.request
    if request.status is not TaskNodeVerificationRequestStatus.PENDING:
        raise WorkExecutionPersistenceError(
            "only a pending verification request is callable"
        )
    run_row = _require_run_row(conn, request.work_run_id, request.session_id)
    if (
        str(run_row["status"]) != "active"
        or str(run_row["reason"] or "") != "verification_pending"
        or str(run_row["current_verification_request_id"] or "")
        != request.verification_request_id
        or run_row["current_attempt_id"] is not None
    ):
        raise WorkExecutionPersistenceError(
            "pending request is not current on its WorkRun"
        )
    window = conn.execute(
        "SELECT turn_id, window_state, stage, state_version, current_work_run_id, "
        "current_attempt_id, latest_checkpoint_id FROM turn_execution_windows "
        "WHERE session_id=?",
        (request.session_id,),
    ).fetchone()
    if (
        window is None
        or str(window["turn_id"] or "") != invocation_turn_id
        or str(window["window_state"]) != "active"
        or str(window["current_work_run_id"] or "") != request.work_run_id
        or window["current_attempt_id"] is not None
        or str(window["latest_checkpoint_id"] or "")
        != request.verification_request_id
        or str(window["stage"] or "") != "VERIFICATION"
    ):
        raise WorkExecutionPersistenceError(
            "invocation Turn does not own the pending verification request"
        )
    owner = conn.execute(
        "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
        (invocation_turn_id,),
    ).fetchone()
    if (
        owner is None
        or str(owner["session_id"]) != request.session_id
        or str(owner["status"]) != "running"
    ):
        raise WorkExecutionPersistenceError(
            "verification invocation requires a running authoritative Turn"
        )
    subject, node_title, node_objective, acceptances = _load_current_node(
        conn, run_row
    )
    _require_node_status(conn, subject=subject, expected_status="active")
    progress, _ = _load_progress_with_hash(conn, request.work_run_id)
    output_window, _ = _load_output_window(conn, request.work_run_id)
    submitted_attempt = _load_attempt(
        conn,
        work_run_id=request.work_run_id,
        attempt_id=request.submitted_attempt_id,
    )
    supporting_results = _load_supporting_tool_results(
        conn,
        work_run_id=request.work_run_id,
        result_ids=request.supporting_tool_result_ids,
        before_attempt_ordinal=submitted_attempt.ordinal,
    )
    _revalidate_binding(
        conn,
        row,
        subject=subject,
        node_title=node_title,
        node_objective=node_objective,
        submitted_attempt=submitted_attempt,
        progress=progress,
        output_window=output_window,
        acceptances=acceptances,
        supporting_results=supporting_results,
    )
    return PreparedTaskNodeVerification(
        record=record,
        invocation_turn_id=invocation_turn_id,
        window_state_version=int(window["state_version"]),
        work_run=_work_run_from_row(run_row),
        submitted_attempt=submitted_attempt,
        acceptance_progress=progress,
        node_title=node_title,
        node_objective=node_objective,
        acceptances=acceptances,
        locked_output_window=output_window,
        supporting_tool_results=supporting_results,
    )


def _revalidate_binding(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    subject: TaskNodeSubject | None = None,
    node_title: str | None = None,
    node_objective: str | None = None,
    submitted_attempt: Attempt | None = None,
    progress: AcceptanceProgressSnapshot | None = None,
    output_window=None,
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...] | None = None,
    supporting_results: tuple[SupportingToolResult, ...] | None = None,
    require_current_task: bool = True,
) -> None:
    record = _record_from_row(row)
    request = record.request
    run_row = _require_run_row(conn, request.work_run_id, request.session_id)
    if require_current_task:
        loaded_node = _load_current_node(conn, run_row)
    else:
        loaded_node = _load_bound_node(conn, run_row)
    (
        loaded_subject,
        loaded_node_title,
        loaded_node_objective,
        loaded_acceptances,
    ) = loaded_node
    subject = subject or loaded_subject
    node_title = node_title or loaded_node_title
    node_objective = node_objective or loaded_node_objective
    acceptances = acceptances or loaded_acceptances
    dependency_delivery_ids = _load_bound_dependency_delivery_ids(
        conn,
        session_id=request.session_id,
        subject=subject,
    )
    if dependency_delivery_ids != request.dependency_delivery_ids:
        raise WorkExecutionPersistenceError(
            "verification dependency Delivery binding has drifted"
        )
    submitted_attempt = submitted_attempt or _load_attempt(
        conn,
        work_run_id=request.work_run_id,
        attempt_id=request.submitted_attempt_id,
    )
    if progress is None:
        progress, progress_hash = _load_progress_with_hash(
            conn, request.work_run_id
        )
    else:
        progress_row = conn.execute(
            "SELECT snapshot_hash FROM insession_work_run_acceptance_progress "
            "WHERE work_run_id=?",
            (request.work_run_id,),
        ).fetchone()
        if progress_row is None:
            raise WorkExecutionPersistenceError(
                "verification AcceptanceProgress is missing"
            )
        progress_hash = str(progress_row["snapshot_hash"])
    if output_window is None:
        output_window, output_hash = _load_output_window(
            conn, request.work_run_id
        )
    else:
        output_row = conn.execute(
            "SELECT snapshot_hash FROM insession_work_run_output_windows "
            "WHERE work_run_id=?",
            (request.work_run_id,),
        ).fetchone()
        if output_row is None:
            raise WorkExecutionPersistenceError("verification OutputWindow is missing")
        output_hash = str(output_row["snapshot_hash"])
    if supporting_results is None:
        supporting_results = _load_supporting_tool_results(
            conn,
            work_run_id=request.work_run_id,
            result_ids=request.supporting_tool_result_ids,
            before_attempt_ordinal=submitted_attempt.ordinal,
        )
    binding_hash = _verification_binding_hash(
        verification_request_id=request.verification_request_id,
        session_id=request.session_id,
        request_turn_id=request.request_turn_id,
        work_run_id=request.work_run_id,
        subject=subject,
        node_title=node_title,
        node_objective=node_objective,
        submitted_attempt=submitted_attempt,
        locked_work_run_revision=request.locked_work_run_revision,
        progress=progress,
        progress_hash=progress_hash,
        output_window=output_window,
        output_hash=output_hash,
        acceptances=acceptances,
        supporting_results=supporting_results,
        dependency_delivery_ids=dependency_delivery_ids,
        prepared_budget=request.prepared_budget,
    )
    if binding_hash != str(row["request_binding_hash"]):
        raise WorkExecutionPersistenceError(
            "verification request immutable binding has drifted"
        )


def _verification_binding_hash(
    *,
    verification_request_id: str,
    session_id: str,
    request_turn_id: str,
    work_run_id: str,
    subject: TaskNodeSubject,
    node_title: str,
    node_objective: str,
    submitted_attempt: Attempt,
    locked_work_run_revision: int,
    progress: AcceptanceProgressSnapshot,
    progress_hash: str,
    output_window,
    output_hash: str,
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...],
    supporting_results: tuple[SupportingToolResult, ...],
    dependency_delivery_ids: tuple[str, ...],
    prepared_budget,
) -> str:
    payload: dict[str, object] = {
        "verification_request_id": verification_request_id,
        "session_id": session_id,
        "request_turn_id": request_turn_id,
        "work_run_id": work_run_id,
        "subject": subject.model_dump(mode="json"),
        "node_title": node_title,
        "node_objective": node_objective,
        "submitted_attempt": submitted_attempt.model_dump(mode="json"),
        "locked_work_run_revision": locked_work_run_revision,
        "acceptance_progress": progress.model_dump(mode="json"),
        "acceptance_progress_snapshot_hash": progress_hash,
        "output_window": output_window.model_dump(mode="json"),
        "output_window_snapshot_hash": output_hash,
        "acceptances": [item.model_dump(mode="json") for item in acceptances],
        "supporting_tool_results": [
            item.model_dump(mode="json") for item in supporting_results
        ],
        "dependency_delivery_ids": list(dependency_delivery_ids),
        "prepared_budget": prepared_budget.model_dump(mode="json"),
    }
    return _payload_hash(payload)


def _load_request_row(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    verification_request_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM insession_work_run_verification_requests "
        "WHERE verification_request_id=? AND session_id=?",
        (verification_request_id, session_id),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "unknown verification request in this Session"
        )
    return row


def _load_task_node_delivery(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    delivery_id: str,
) -> ResolvedTaskNodeDelivery:
    row = conn.execute(
        "SELECT delivery.*, output.frozen_at, node_state.status AS node_status "
        "FROM insession_task_node_deliveries AS delivery "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=delivery.work_run_id "
        "AND output.output_revision=delivery.output_revision "
        "JOIN insession_task_node_states AS node_state "
        "ON node_state.insession_task_id=delivery.insession_task_id "
        "AND node_state.insession_task_node_id=delivery.insession_task_node_id "
        "AND node_state.node_revision=delivery.node_revision "
        "WHERE delivery.delivery_id=? AND delivery.session_id=?",
        (delivery_id, session_id),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "unknown TaskNode delivery in this Session"
        )
    run_row = _require_run_row(conn, str(row["work_run_id"]), session_id)
    subject = _subject_from_run_row(run_row)
    if (
        str(run_row["status"]) != "completed"
        or str(run_row["reason"] or "") != "verification_passed"
        or run_row["current_attempt_id"] is not None
        or run_row["current_verification_request_id"] is not None
        or row["frozen_at"] is None
        or str(row["node_status"]) != "completed"
        or str(row["insession_task_id"]) != subject.task_id
        or int(row["graph_revision"]) != subject.graph_revision
        or str(row["insession_task_node_id"]) != subject.node_id
        or int(row["node_revision"]) != subject.node_revision
        or str(row["created_turn_id"]) != str(run_row["updated_turn_id"])
    ):
        raise WorkExecutionPersistenceError(
            "TaskNode delivery owner projection is corrupt"
        )
    output_window, _ = _load_output_window(conn, str(row["work_run_id"]))
    if output_window.output_revision != int(row["output_revision"]):
        raise WorkExecutionPersistenceError(
            "TaskNode delivery OutputWindow binding is corrupt"
        )
    request_row = _load_request_row(
        conn,
        session_id=session_id,
        verification_request_id=str(row["verification_request_id"]),
    )
    record = _record_from_row(request_row)
    submitted_attempt = _load_attempt(
        conn,
        work_run_id=str(row["work_run_id"]),
        attempt_id=str(row["submitted_attempt_id"]),
    )
    turn_owners = conn.execute(
        "SELECT turn_id, session_id FROM runtime_turns WHERE turn_id IN (?, ?)",
        (record.request.request_turn_id, str(row["created_turn_id"])),
    ).fetchall()
    owner_by_turn = {
        str(owner["turn_id"]): str(owner["session_id"]) for owner in turn_owners
    }
    if (
        record.result is None
        or not record.result.all_pass
        or record.request.work_run_id != str(row["work_run_id"])
        or record.request.subject != subject
        or record.request.submitted_attempt_id != str(row["submitted_attempt_id"])
        or record.request.output_revision != int(row["output_revision"])
        or submitted_attempt.attempt_id != record.request.submitted_attempt_id
        or submitted_attempt.submitted_output_revision
        != record.request.output_revision
        or owner_by_turn.get(record.request.request_turn_id) != session_id
        or owner_by_turn.get(str(row["created_turn_id"])) != session_id
    ):
        raise WorkExecutionPersistenceError(
            "TaskNode delivery verification binding is corrupt"
        )
    _revalidate_binding(conn, request_row, require_current_task=False)
    return ResolvedTaskNodeDelivery(
        delivery=TaskNodeDelivery(
            delivery_id=delivery_id,
            session_id=session_id,
            work_run_id=str(row["work_run_id"]),
            subject=subject,
            verification_request_id=str(row["verification_request_id"]),
            submitted_attempt_id=str(row["submitted_attempt_id"]),
            output_revision=int(row["output_revision"]),
            created_turn_id=str(row["created_turn_id"]),
        ),
        output_window=output_window,
    )


def _validate_request_lifecycle(
    conn: sqlite3.Connection,
    record: TaskNodeVerificationRecord,
) -> None:
    request = record.request
    run_row = _require_run_row(conn, request.work_run_id, request.session_id)
    subject = _subject_from_run_row(run_row)
    if subject != request.subject:
        raise WorkExecutionPersistenceError(
            "verification request WorkRun subject has drifted"
        )
    if request.status is TaskNodeVerificationRequestStatus.PENDING:
        valid = (
            str(run_row["status"]) == "active"
            and str(run_row["reason"] or "") == "verification_pending"
            and str(run_row["current_verification_request_id"] or "")
            == request.verification_request_id
            and run_row["current_attempt_id"] is None
        )
        expected_node_status = "active"
    elif request.status is TaskNodeVerificationRequestStatus.INTERRUPTED:
        valid = (
            (
                (
                    str(run_row["status"]) == "interrupted"
                    and str(run_row["reason"] or "")
                    == "verification_technical_failure"
                    and str(run_row["current_verification_request_id"] or "")
                    == request.verification_request_id
                )
                or (
                    str(run_row["status"]) == "failed"
                    and str(run_row["reason"] or "")
                    == "work_run_limit_reached"
                    and run_row["current_verification_request_id"] is None
                    and request.technical_error_code
                    == "work_run_limit_reached"
                )
            )
            and run_row["current_attempt_id"] is None
        )
        expected_node_status = "interrupted"
    elif record.result is not None and record.result.all_pass:
        delivery_row = conn.execute(
            "SELECT delivery_id FROM insession_task_node_deliveries "
            "WHERE work_run_id=? AND verification_request_id=?",
            (request.work_run_id, request.verification_request_id),
        ).fetchone()
        if delivery_row is None:
            raise WorkExecutionPersistenceError(
                "passed verification request has no NodeDelivery"
            )
        _load_task_node_delivery(
            conn,
            session_id=request.session_id,
            delivery_id=str(delivery_row["delivery_id"]),
        )
        return
    else:
        return
    if not valid:
        raise WorkExecutionPersistenceError(
            "verification request lifecycle projection is corrupt"
        )
    _require_node_status(
        conn,
        subject=subject,
        expected_status=expected_node_status,
    )


def _record_from_row(row: sqlite3.Row) -> TaskNodeVerificationRecord:
    try:
        acceptance_ids = _decode_id_tuple(row["acceptance_ids_json"])
        supporting_ids = _decode_id_tuple(
            row["supporting_tool_result_ids_json"], allow_empty=True
        )
        dependency_delivery_ids = _decode_id_tuple(
            row["dependency_delivery_ids_json"], allow_empty=True
        )
        request = TaskNodeVerificationRequest(
            verification_request_id=str(row["verification_request_id"]),
            session_id=str(row["session_id"]),
            request_turn_id=str(row["request_turn_id"]),
            work_run_id=str(row["work_run_id"]),
            subject=TaskNodeSubject(
                task_id=str(row["insession_task_id"]),
                graph_revision=int(row["graph_revision"]),
                node_id=str(row["insession_task_node_id"]),
                node_revision=int(row["node_revision"]),
            ),
            submitted_attempt_id=str(row["submitted_attempt_id"]),
            output_revision=int(row["output_revision"]),
            acceptance_progress_revision=int(row["acceptance_progress_revision"]),
            acceptance_ids=acceptance_ids,
            supporting_tool_result_ids=supporting_ids,
            dependency_delivery_ids=dependency_delivery_ids,
            locked_work_run_revision=int(row["locked_work_run_revision"]),
            prepared_budget=_budget_from_json(row["prepared_budget_json"]),
            revision=int(row["request_revision"]),
            status=TaskNodeVerificationRequestStatus(str(row["status"])),
            technical_error_code=(
                str(row["technical_error_code"])
                if row["technical_error_code"] is not None
                else None
            ),
        )
        result = (
            NodeVerificationResult.model_validate_json(str(row["result_json"]))
            if row["result_json"] is not None
            else None
        )
        if (result is not None) != (row["all_pass"] is not None):
            raise ValueError("verification result/all_pass columns disagree")
        if result is not None and result.all_pass != bool(row["all_pass"]):
            raise ValueError("verification result aggregate column disagrees")
        return TaskNodeVerificationRecord(request=request, result=result)
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "stored verification request is corrupt"
        ) from exc


def _load_current_node(
    conn: sqlite3.Connection,
    run_row: sqlite3.Row,
) -> tuple[
    TaskNodeSubject,
    str,
    str,
    tuple[InSessionTaskAcceptanceProposal, ...],
]:
    bound = _load_bound_node(conn, run_row)
    subject = bound[0]
    task = conn.execute(
        "SELECT current_graph_revision, current_status FROM insession_tasks "
        "WHERE insession_task_id=? AND session_id=?",
        (subject.task_id, str(run_row["session_id"])),
    ).fetchone()
    if (
        task is None
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != subject.graph_revision
        or str(task["current_status"]) not in {"active", "interrupted"}
    ):
        raise WorkExecutionPersistenceError(
            "verification Task authority is stale"
        )
    return bound


def _load_bound_node(
    conn: sqlite3.Connection,
    run_row: sqlite3.Row,
) -> tuple[
    TaskNodeSubject,
    str,
    str,
    tuple[InSessionTaskAcceptanceProposal, ...],
]:
    """读取 WorkRun subject 指定的不可变 graph/node 版本。"""

    subject = _subject_from_run_row(run_row)
    node = conn.execute(
        "SELECT node.title, node.objective, node.acceptance_criteria_json, "
        "state.status FROM insession_task_graph_nodes AS node "
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
    node_status = str(node["status"]) if node is not None else ""
    if node is None or node_status not in {"active", "interrupted", "completed"}:
        raise WorkExecutionPersistenceError(
            "verification bound TaskNode authority is stale"
        )
    try:
        raw = json.loads(str(node["acceptance_criteria_json"]))
        acceptances = tuple(
            InSessionTaskAcceptanceProposal.model_validate(item) for item in raw
        )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "TaskNode Acceptance payload is corrupt"
        ) from exc
    ids = [item.acceptance_id for item in acceptances]
    if not ids or len(ids) != len(set(ids)):
        raise WorkExecutionPersistenceError(
            "TaskNode Acceptance identities are invalid"
        )
    return subject, str(node["title"]), str(node["objective"]), acceptances


def _require_node_status(
    conn: sqlite3.Connection,
    *,
    subject: TaskNodeSubject,
    expected_status: str,
) -> None:
    row = conn.execute(
        "SELECT status FROM insession_task_node_states "
        "WHERE insession_task_id=? AND insession_task_node_id=? "
        "AND node_revision=?",
        (subject.task_id, subject.node_id, subject.node_revision),
    ).fetchone()
    if row is None or str(row["status"]) != expected_status:
        raise WorkExecutionPersistenceError(
            f"verification TaskNode must be {expected_status}"
        )


def _load_bound_dependency_delivery_ids(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> tuple[str, ...]:
    """解析某节点直接子节点精确有序的 PASS Delivery。

    返回的 ID 仅作为引用。每份正文仍由其冻结 OutputWindow 持有，且 ID 被接受前会
    通过现有 Delivery 加载器解析。因此叶节点返回真实空元组；非叶节点若存在缺失、
    过期或重复依赖，则无法进入验证。
    """

    # 延迟导入，因为 work_execution 会为不可变来源正文导入验证加载器。执行与验证
    # 现在共享这一精确的直接或 carry 当前 subject 解析器。
    from .work_execution import _load_current_task_node_dependency_deliveries

    return tuple(
        item.delivery_id
        for item in _load_current_task_node_dependency_deliveries(
            conn,
            session_id=session_id,
            subject=subject,
        )
    )


def _apply_task_finish_gate(
    conn: sqlite3.Connection,
    *,
    subject: TaskNodeSubject,
    work_run_id: str,
    verification_request_id: str,
    delivery_id: str,
    now: str,
) -> bool:
    """根据已验证根 Delivery 完成一个规范的当前 TaskGraph。

    中间节点 PASS 仍是持久化内部 Delivery，只会提升活动 Task 权威状态。最终根
    PASS 会在同一事务中重新读取整个当前树：每个成员都必须已完成，且恰好持有一个
    可完整解析的 PASS/冻结 Delivery；每项非叶验证必须指定精确有序的直接子节点
    Delivery。不会创建复制的子节点正文或第二份 Task delivery；根 NodeDelivery
    是唯一正式输出候选项。
    """

    task = conn.execute(
        "SELECT session_id, current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE insession_task_id=?",
        (subject.task_id,),
    ).fetchone()
    if (
        task is None
        or task["current_graph_revision"] is None
        or int(task["current_graph_revision"]) != subject.graph_revision
        or str(task["current_status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task authority changed during FinishGate"
        )

    resolved = _load_task_node_delivery(
        conn,
        session_id=str(task["session_id"]),
        delivery_id=delivery_id,
    )
    if (
        resolved.delivery.subject != subject
        or resolved.delivery.work_run_id != work_run_id
        or resolved.delivery.verification_request_id != verification_request_id
    ):
        raise WorkExecutionPersistenceError(
            "FinishGate delivery binding is corrupt"
        )

    nodes = conn.execute(
        "SELECT node.insession_task_node_id, node.node_revision, node.node_kind, "
        "state.status FROM insession_task_graph_nodes AS node "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=node.insession_task_id "
        "AND state.insession_task_node_id=node.insession_task_node_id "
        "AND state.node_revision=node.node_revision "
        "WHERE node.insession_task_id=? AND node.graph_revision=? "
        "ORDER BY node.ordinal, node.insession_task_node_id",
        (subject.task_id, subject.graph_revision),
    ).fetchall()
    roots = tuple(
        node
        for node in nodes
        if str(node["node_kind"]) == "root"
    )
    if (
        len(roots) != 1
        or str(roots[0]["insession_task_node_id"]) != subject.task_id
    ):
        # 历史或有意非规范的 graph 仍可能保留已验证内部 NodeDelivery，但绝不能通过
        # 此 gate 完成根 Task。
        _bump_active_task_state(conn, subject=subject, now=now)
        return False
    if subject.node_id != subject.task_id:
        _bump_active_task_state(conn, subject=subject, now=now)
        return False
    if any(str(node["status"]) != "completed" for node in nodes):
        _bump_active_task_state(conn, subject=subject, now=now)
        return False
    root = roots[0]
    if int(root["node_revision"]) != subject.node_revision:
        raise WorkExecutionPersistenceError(
            "FinishGate root revision is corrupt"
        )

    delivery_by_node: dict[str, str] = {}
    request_by_node: dict[str, sqlite3.Row] = {}
    from .work_execution import _load_current_task_node_delivery_projection

    for node in nodes:
        node_id = str(node["insession_task_node_id"])
        node_revision = int(node["node_revision"])
        current_delivery = _load_current_task_node_delivery_projection(
            conn,
            session_id=str(task["session_id"]),
            task_id=subject.task_id,
            graph_revision=subject.graph_revision,
            node_id=node_id,
            node_revision=node_revision,
        )
        node_delivery = current_delivery.source_delivery
        node_delivery_id = current_delivery.delivery_id
        expected_subject = TaskNodeSubject(
            task_id=subject.task_id,
            graph_revision=subject.graph_revision,
            node_id=node_id,
            node_revision=node_revision,
        )
        if current_delivery.target_subject != expected_subject:
            raise WorkExecutionPersistenceError(
                "FinishGate current node target authority is stale"
            )
        request_row = _load_request_row(
            conn,
            session_id=str(task["session_id"]),
            verification_request_id=(
                node_delivery.delivery.verification_request_id
            ),
        )
        record = _record_from_row(request_row)
        if (
            record.request.subject != node_delivery.delivery.subject
            or record.request.status
            is not TaskNodeVerificationRequestStatus.COMPLETED
            or record.result is None
            or not record.result.all_pass
        ):
            raise WorkExecutionPersistenceError(
                "FinishGate current node verification is not a complete PASS"
            )
        delivery_by_node[node_id] = node_delivery_id
        request_by_node[node_id] = request_row

    if delivery_by_node.get(subject.task_id) != delivery_id:
        raise WorkExecutionPersistenceError(
            "FinishGate root Delivery is not the just-created Delivery"
        )

    children_by_parent: dict[str, list[tuple[int, str]]] = {
        str(node["insession_task_node_id"]): [] for node in nodes
    }
    edges = conn.execute(
        "SELECT edge.parent_insession_task_node_id, "
        "edge.child_insession_task_node_id, child.ordinal "
        "FROM insession_task_graph_edges AS edge "
        "JOIN insession_task_graph_nodes AS child "
        "ON child.insession_task_id=edge.insession_task_id "
        "AND child.graph_revision=edge.graph_revision "
        "AND child.insession_task_node_id=edge.child_insession_task_node_id "
        "WHERE edge.insession_task_id=? AND edge.graph_revision=?",
        (subject.task_id, subject.graph_revision),
    ).fetchall()
    for edge in edges:
        parent_id = str(edge["parent_insession_task_node_id"])
        child_id = str(edge["child_insession_task_node_id"])
        if parent_id not in children_by_parent or child_id not in delivery_by_node:
            raise WorkExecutionPersistenceError(
                "FinishGate graph edge references an unknown current node"
            )
        children_by_parent[parent_id].append((int(edge["ordinal"]), child_id))
    for node_id, children in children_by_parent.items():
        ordered_children = tuple(
            child_id for _ordinal, child_id in sorted(children)
        )
        expected_dependencies = tuple(
            delivery_by_node[child_id] for child_id in ordered_children
        )
        request = _record_from_row(request_by_node[node_id]).request
        if request.dependency_delivery_ids != expected_dependencies:
            raise WorkExecutionPersistenceError(
                "FinishGate verification dependency coverage is incomplete"
            )

    completion_unconfirmed = conn.execute(
        "SELECT 1 FROM insession_work_runs AS run "
        "JOIN insession_work_run_tool_results AS result "
        "ON result.work_run_id=run.work_run_id "
        "WHERE run.insession_task_id=? "
        "AND result.status='completion_unconfirmed' LIMIT 1",
        (subject.task_id,),
    ).fetchone()
    if completion_unconfirmed is not None:
        raise WorkExecutionPersistenceError(
            "FinishGate cannot cross a completion-unconfirmed result"
        )

    nonterminal_run = conn.execute(
        "SELECT 1 FROM insession_work_runs WHERE insession_task_id=? "
        "AND status NOT IN ('completed', 'failed', 'cancelled') "
        "LIMIT 1",
        (subject.task_id,),
    ).fetchone()
    if nonterminal_run is not None:
        raise WorkExecutionPersistenceError(
            "FinishGate found another nonterminal WorkRun"
        )

    if conn.execute(
        "UPDATE insession_tasks SET current_status='completed', "
        "state_version=state_version+1, updated_at=? "
        "WHERE insession_task_id=? AND current_graph_revision=? "
        "AND current_status='active' AND state_version=?",
        (
            now,
            subject.task_id,
            subject.graph_revision,
            int(task["state_version"]),
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Task changed during FinishGate commit"
        )
    return True


def _bump_active_task_state(
    conn: sqlite3.Connection,
    *,
    subject: TaskNodeSubject,
    now: str,
) -> None:
    row = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE insession_task_id=?",
        (subject.task_id,),
    ).fetchone()
    if (
        row is None
        or row["current_graph_revision"] is None
        or int(row["current_graph_revision"]) != subject.graph_revision
        or str(row["current_status"]) != "active"
    ):
        raise WorkExecutionPersistenceError(
            "Task authority changed during verification projection"
        )
    if conn.execute(
        "UPDATE insession_tasks SET state_version=state_version+1, updated_at=? "
        "WHERE insession_task_id=? AND current_graph_revision=? "
        "AND current_status='active' AND state_version=?",
        (
            now,
            subject.task_id,
            subject.graph_revision,
            int(row["state_version"]),
        ),
    ).rowcount != 1:
        raise WorkExecutionPersistenceError(
            "Task changed during verification projection"
        )


def _subject_from_run_row(row: sqlite3.Row) -> TaskNodeSubject:
    if str(row["subject_kind"]) != "task_node":
        raise WorkExecutionPersistenceError(
            "node verification requires a TaskNode WorkRun"
        )
    return TaskNodeSubject(
        task_id=str(row["insession_task_id"]),
        graph_revision=int(row["graph_revision"]),
        node_id=str(row["insession_task_node_id"]),
        node_revision=int(row["node_revision"]),
    )


def _work_run_from_row(row: sqlite3.Row) -> WorkRun:
    return WorkRun(
        work_run_id=str(row["work_run_id"]),
        subject=_subject_from_run_row(row),
        revision=int(row["revision"]),
        status=WorkRunStatus(str(row["status"])),
        reason=(str(row["reason"]) if row["reason"] is not None else None),
        budget=_budget_from_row(row),
    )


def _load_current_submit_attempt(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    output_revision: int,
) -> Attempt:
    row = conn.execute(
        "SELECT attempt_id, ordinal, status, action, submitted_output_revision "
        "FROM insession_work_run_attempts WHERE work_run_id=? "
        "ORDER BY ordinal DESC LIMIT 1",
        (work_run_id,),
    ).fetchone()
    if (
        row is None
        or str(row["status"]) != "closed"
        or str(row["action"] or "") != "submit_output_window"
        or row["submitted_output_revision"] is None
        or int(row["submitted_output_revision"]) != output_revision
    ):
        raise WorkExecutionPersistenceError(
            "verification requires the latest closed submit Attempt"
        )
    return Attempt(
        attempt_id=str(row["attempt_id"]),
        work_run_id=work_run_id,
        ordinal=int(row["ordinal"]),
        status=AttemptStatus.CLOSED,
        submitted_output_revision=output_revision,
    )


def _load_attempt(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    attempt_id: str,
) -> Attempt:
    row = conn.execute(
        "SELECT ordinal, status, action, submitted_output_revision "
        "FROM insession_work_run_attempts WHERE work_run_id=? AND attempt_id=?",
        (work_run_id, attempt_id),
    ).fetchone()
    if (
        row is None
        or str(row["status"]) != "closed"
        or str(row["action"] or "") != "submit_output_window"
        or row["submitted_output_revision"] is None
    ):
        raise WorkExecutionPersistenceError(
            "verification submit Attempt is missing or corrupt"
        )
    return Attempt(
        attempt_id=attempt_id,
        work_run_id=work_run_id,
        ordinal=int(row["ordinal"]),
        status=AttemptStatus.CLOSED,
        submitted_output_revision=int(row["submitted_output_revision"]),
    )


def _load_supporting_tool_results(
    conn: sqlite3.Connection,
    *,
    work_run_id: str,
    result_ids: tuple[str, ...],
    before_attempt_ordinal: int,
) -> tuple[SupportingToolResult, ...]:
    if not result_ids:
        return ()
    placeholders = ",".join("?" for _ in result_ids)
    rows = conn.execute(
        "SELECT result.tool_result_id, result.tool_call_id, result.attempt_id, "
        "result.ordinal, result.status, result.result_json, call.tool_id, "
        "call.tool_version, attempt.ordinal AS attempt_ordinal, attempt.status AS attempt_status "
        "FROM insession_work_run_tool_results AS result "
        "JOIN insession_work_run_tool_calls AS call "
        "ON call.work_run_id=result.work_run_id "
        "AND call.attempt_id=result.attempt_id "
        "AND call.tool_call_id=result.tool_call_id AND call.ordinal=result.ordinal "
        "JOIN insession_work_run_attempts AS attempt "
        "ON attempt.work_run_id=result.work_run_id "
        "AND attempt.attempt_id=result.attempt_id "
        f"WHERE result.work_run_id=? AND result.tool_result_id IN ({placeholders})",
        (work_run_id, *result_ids),
    ).fetchall()
    by_id: dict[str, SupportingToolResult] = {}
    try:
        for row in rows:
            parsed = ToolResult.model_validate_json(str(row["result_json"]))
            result_id = str(row["tool_result_id"])
            if (
                parsed.tool_result_id != result_id
                or parsed.tool_call_id != str(row["tool_call_id"])
                or parsed.attempt_id != str(row["attempt_id"])
                or parsed.ordinal != int(row["ordinal"])
                or parsed.status.value != str(row["status"])
                or parsed.status is not ToolResultStatus.SUCCEEDED
                or str(row["attempt_status"]) != "closed"
                or int(row["attempt_ordinal"]) >= before_attempt_ordinal
                or str(row["tool_id"]) in EXECUTION_FINDINGS_TOOL_IDS
            ):
                raise ValueError("supporting ToolResult binding mismatch")
            by_id[result_id] = SupportingToolResult(
                work_run_id=work_run_id,
                tool_id=str(row["tool_id"]),
                tool_version=str(row["tool_version"]),
                result=parsed,
            )
    except (TypeError, ValueError) as exc:
        raise WorkExecutionPersistenceError(
            "supporting ToolResult authority is corrupt"
        ) from exc
    if set(by_id) != set(result_ids) or len(by_id) != len(result_ids):
        raise WorkExecutionPersistenceError(
            "supporting ToolResult references are missing or duplicated"
        )
    return tuple(by_id[result_id] for result_id in result_ids)


def _load_progress_with_hash(
    conn: sqlite3.Connection,
    work_run_id: str,
) -> tuple[AcceptanceProgressSnapshot, str]:
    progress = _load_progress(conn, work_run_id)
    row = conn.execute(
        "SELECT snapshot_hash FROM insession_work_run_acceptance_progress "
        "WHERE work_run_id=?",
        (work_run_id,),
    ).fetchone()
    if row is None:
        raise WorkExecutionPersistenceError(
            "WorkRun AcceptanceProgress hash is missing"
        )
    return progress, str(row["snapshot_hash"])


def _budget_from_json(raw: object):
    from personagraph.l2.work_run import WorkRunBudget

    return WorkRunBudget.model_validate_json(str(raw))


def _decode_id_tuple(raw: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    decoded = json.loads(str(raw))
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) and item.strip() for item in decoded
    ):
        raise ValueError("stored verification IDs are invalid")
    values = tuple(decoded)
    if (not allow_empty and not values) or len(values) != len(set(values)):
        raise ValueError("stored verification IDs are incomplete or duplicated")
    return values


def _require_request_revision(row: sqlite3.Row, expected: int) -> None:
    actual = int(row["request_revision"])
    if actual != expected:
        raise TaskNodeVerificationRequestRevisionConflict(
            expected=expected,
            actual=actual,
        )


def _load_replay(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    payload_hash: str,
) -> TaskNodeVerificationMutationResult | None:
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
    try:
        stored = TaskNodeVerificationMutationResult.model_validate_json(
            str(row["result_json"])
        )
    except ValueError as exc:
        raise WorkExecutionPersistenceError(
            "stored verification apply receipt is corrupt"
        ) from exc
    if stored.work_run_id != str(row["work_run_id"]):
        raise WorkExecutionPersistenceError(
            "stored verification apply receipt owner is corrupt"
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


def _insert_receipt(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    payload_hash: str,
    result: TaskNodeVerificationMutationResult,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_work_run_apply_receipts "
        "(apply_id, operation, session_id, work_run_id, payload_hash, result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            apply_id,
            operation,
            session_id,
            work_run_id,
            payload_hash,
            _model_json(result),
            now,
        ),
    )


__all__ = [
    "TaskNodeVerificationRequestRevisionConflict",
    "VerificationSettlementReceiptNotFound",
    "commit_task_node_verification_result",
    "get_prepared_task_node_verification",
    "get_task_node_delivery",
    "get_task_node_verification_record",
    "interrupt_task_node_verification",
    "prepare_task_node_verification",
    "replay_task_node_verification_settlement",
    "resume_task_node_verification",
]
