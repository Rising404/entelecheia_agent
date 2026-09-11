"""当前 Auxiliary 的等待用户与跨 Turn WorkRun 延续权威状态。

每个入口在接触 WorkRun 状态前都会证明精确的 execution-subject 注册绑定以及当前
goal/revision。
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from personagraph.l2.task_graph import (
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationFaultDomain,
    TaskDeliveryValidationVerdict,
)
from personagraph.l2.work_run import (
    AttemptStatus,
    AuxiliaryNodeSubject,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    RequestUserInputAction,
    TaskNodeVerificationMutationResult,
    TaskNodeVerificationRequestStatus,
    WorkRunBudgetDisposition,
    WorkRunStatus,
    merge_acceptance_progress,
)
from .auxiliary_node_execution_bindings import (
    _auxiliary_record_from_row,
    _load_auxiliary_request_row,
    _revalidate_auxiliary_request_binding,
)
from ..delivery import task_delivery_validation as task_delivery_validation_records
from ...deps import StoreDeps
from ..work_run.work_verification import _require_request_revision
from ..work_run.work_execution import (
    WorkExecutionApplyIdCollision,
    WorkExecutionMutationResult,
    WorkExecutionPersistenceError,
    WorkExecutionRevisionConflict,
    _budget_from_row,
    _build_mutation_result,
    _canonical_json,
    _execution_subject_contract_version,
    _insert_budget_charge,
    _insert_receipt as _insert_work_execution_receipt,
    _load_output_window,
    _load_progress,
    _load_record,
    _load_tool_results,
    _model_json,
    _payload_hash,
    _require_active_attempt,
    _require_active_window,
    _require_attempt_row_aggregate,
    _require_owned_attempt_window,
    _require_progress_revision,
    _require_readonly_recovery_batch,
    _require_run_revision,
    _require_run_row,
    _require_settlement_budget_transition,
    _settlement_budget_checkpoint_id,
    _subject_from_run_row,
    _text_hash,
)


_EXECUTION_SUBJECT_CONTRACT_VERSION = "auxiliary_node_v2"
_NONTERMINAL_RUN_STATUSES = (
    "active",
    "paused",
    "waiting_user",
    "waiting_authorization",
    "waiting_external",
    "turn_limit_reached",
    "interrupted",
)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryContinuationPersistenceError(WorkExecutionPersistenceError):
    """某项 等待用户/跨 Turn 权威不变量以失败关闭方式触发。"""


class AuxiliaryContinuationApplyIdCollision(
    WorkExecutionApplyIdCollision,
    AuxiliaryContinuationPersistenceError,
):
    """某个 apply id 被复用于其精确 命令载荷之外。"""


class AuxiliaryAnswerSourceBinding(_Record):
    """仅通过引用绑定到一条不可变的已接受用户消息。"""

    answer_binding_id: str = Field(min_length=1, max_length=200)
    question_attempt_id: str = Field(min_length=1, max_length=200)
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_attempt_id: str = Field(min_length=1, max_length=200)
    answer_source_turn_id: str = Field(min_length=1, max_length=200)
    answer_source_message_id: str = Field(min_length=1, max_length=200)
    answer_source_turn_idx: int = Field(ge=0)
    answer_source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_source_utf8_bytes: int = Field(ge=1)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuxiliaryPendingUserQuestion(_Record):
    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    execution_subject_id: str = Field(min_length=1, max_length=200)
    execution_subject_contract_version: Literal["auxiliary_node_v2"] = (
        "auxiliary_node_v2"
    )
    subject: AuxiliaryNodeSubject
    goal_id: str = Field(min_length=1, max_length=200)
    question_attempt_id: str = Field(min_length=1, max_length=200)
    question_attempt_ordinal: int = Field(ge=1)
    question_turn_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2_000)
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    work_run_revision: int = Field(ge=1)
    task_state_version: int = Field(ge=1)
    node_state_version: int = Field(ge=1)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    acceptance_progress_revision: int = Field(ge=1)
    output_window_revision: int = Field(ge=1)


class CommitAuxiliaryWaitingUserAttemptCommand(_Record):
    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    decision: HostAcceptedAttemptDecision
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    active_seconds_delta: float = Field(gt=0)

    @field_validator("active_seconds_delta")
    @classmethod
    def _finite_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("active_seconds_delta must be finite")
        return value

    @model_validator(mode="after")
    def _require_question(self) -> "CommitAuxiliaryWaitingUserAttemptCommand":
        if not isinstance(self.decision.action, RequestUserInputAction):
            raise ValueError("waiting-user settlement requires request_user_input")
        return self


class ContinueAuxiliaryWaitingUserCommand(_Record):
    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    question_attempt_id: str = Field(min_length=1, max_length=200)
    expected_question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_answer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    catalog_snapshot: dict[str, Any]
    attempt_id: str = Field(min_length=1, max_length=200)


class ResumeAuxiliaryActiveAttemptCommand(_Record):
    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    expected_work_run_revision: int = Field(ge=1)
    expected_progress_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)
    catalog_snapshot: dict[str, Any] | None = None
    allow_protected_recovery: bool = False

    @model_validator(mode="after")
    def _validate_recovery_mode(self) -> "ResumeAuxiliaryActiveAttemptCommand":
        if self.allow_protected_recovery and self.catalog_snapshot is None:
            raise ValueError(
                "protected Attempt recovery requires a catalog snapshot"
            )
        return self


class ResumeAuxiliaryVerificationCommand(_Record):
    """将一个待处理 verifier 调用移至新 Turn 的 CAS 栅栏。"""

    session_id: str = Field(min_length=1, max_length=200)
    turn_id: str = Field(min_length=1, max_length=200)
    work_run_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    expected_structure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_work_run_revision: int = Field(ge=1)
    expected_verification_request_revision: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    apply_id: str = Field(min_length=1, max_length=200)


class AuxiliaryVerificationResumeResult(
    TaskNodeVerificationMutationResult
):
    """由验证恢复投影且保证 receipt 安全的精确 权威状态。"""

    operation: Literal["resume_verification"] = "resume_verification"
    execution_subject_id: str = Field(min_length=1, max_length=200)
    execution_subject_contract_version: Literal["auxiliary_node_v2"] = (
        "auxiliary_node_v2"
    )
    subject: AuxiliaryNodeSubject
    goal_id: str = Field(min_length=1, max_length=200)
    structure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_source_turn_id: str = Field(min_length=1, max_length=200)
    request_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    output_revision: int = Field(ge=1)
    active_seconds_consumed: float = Field(ge=0)
    turn_work_run_link_revision: int = Field(ge=0)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)

    @field_validator("active_seconds_consumed")
    @classmethod
    def _finite_active_seconds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("active_seconds_consumed must be finite")
        return value


class AuxiliaryContinuationMutationResult(WorkExecutionMutationResult):
    operation: Literal[
        "commit_waiting_user_attempt",
        "continue_waiting_user_and_start_attempt",
        "resume_active_attempt",
    ]
    execution_subject_id: str = Field(min_length=1, max_length=200)
    execution_subject_contract_version: Literal["auxiliary_node_v2"] = (
        "auxiliary_node_v2"
    )
    subject: AuxiliaryNodeSubject
    goal_id: str = Field(min_length=1, max_length=200)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    question_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    answer_source_binding: AuxiliaryAnswerSourceBinding | None = None

    @model_validator(mode="after")
    def _validate_operation_shape(self) -> "AuxiliaryContinuationMutationResult":
        has_answer = self.answer_source_binding is not None
        if has_answer != (
            self.operation == "continue_waiting_user_and_start_attempt"
        ):
            raise ValueError("only answer continuation carries an answer binding")
        if (self.question_sha256 is not None) != (
            self.operation == "commit_waiting_user_attempt"
        ):
            raise ValueError("only question settlement carries its question hash")
        if self.task_state_version is None or self.node_state_version is None:
            raise ValueError("continuation result requires Task and node CAS versions")
        return self


def get_auxiliary_pending_user_question(
    deps: StoreDeps,
    *,
    session_id: str,
    insession_task_id: str,
) -> AuxiliaryPendingUserQuestion | None:
    """投影精确的当前 问题；绝不在多个游标间自动选择。"""

    _require_identifier("session_id", session_id)
    _require_identifier("insession_task_id", insession_task_id)
    deps.init_db()
    with deps.connect() as conn:
        conn.execute("BEGIN")
        rows = _current_auxiliary_cursor_rows(
            conn,
            session_id=session_id,
            task_id=insession_task_id,
        )
        if len(rows) > 1:
            raise AuxiliaryContinuationPersistenceError(
                "AuxiliaryGraph has more than one durable cursor"
            )
        if not rows or str(rows[0]["status"]) != "waiting_user":
            return None
        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=session_id,
            work_run_id=str(rows[0]["work_run_id"]),
        )
        _require_auxiliary_authority_shape(authority)
        _require_authority_statuses(
            authority,
            task_status="awaiting_user",
            node_status="waiting_user",
            goal_status="waiting_user",
            revision_status="waiting_user",
        )
        record = _load_record(
            conn,
            str(authority["work_run_id"]),
            session_id=session_id,
        )
        if (
            record.work_run.status is not WorkRunStatus.WAITING_USER
            or record.work_run.reason != "needs_input"
            or record.current_attempt_id is not None
            or record.current_verification_request_id is not None
            or not record.attempts
        ):
            raise AuxiliaryContinuationPersistenceError(
                "waiting-user WorkRun projection is inconsistent"
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
            raise AuxiliaryContinuationPersistenceError(
                "waiting-user WorkRun does not end at its question Attempt"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_attempts "
            "WHERE predecessor_question_attempt_id=? LIMIT 1",
            (question_attempt.attempt.attempt_id,),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "pending user question was already consumed"
            )
        question_row = conn.execute(
            "SELECT attempt_id, turn_id, budget_charge_id "
            "FROM insession_work_run_attempts WHERE work_run_id=? "
            "AND attempt_id=?",
            (record.work_run.work_run_id, question_attempt.attempt.attempt_id),
        ).fetchone()
        if (
            question_row is None
            or _require_question_settlement_sha256(
                conn,
                authority=authority,
                question_row=question_row,
            )
            != _text_hash(decision.action.question)
        ):
            raise AuxiliaryContinuationPersistenceError(
                "pending question changed after settlement"
            )
        progress = record.acceptance_progress
        output = record.output_window
        result = AuxiliaryPendingUserQuestion(
            session_id=session_id,
            task_id=insession_task_id,
            work_run_id=record.work_run.work_run_id,
            execution_subject_id=str(authority["execution_subject_id"]),
            subject=record.work_run.subject,
            goal_id=str(authority["goal_id"]),
            question_attempt_id=question_attempt.attempt.attempt_id,
            question_attempt_ordinal=question_attempt.attempt.ordinal,
            question_turn_id=question_attempt.turn_id,
            question=decision.action.question,
            question_sha256=_text_hash(decision.action.question),
            work_run_revision=record.work_run.revision,
            task_state_version=int(authority["task_state_version"]),
            node_state_version=int(authority["node_state_version"]),
            control_state_version=int(authority["control_state_version"]),
            goal_state_version=int(authority["goal_state_version"]),
            revision_state_version=int(authority["revision_state_version"]),
            acceptance_progress_revision=progress.revision,
            output_window_revision=output.output_revision,
        )
        conn.commit()
        return result


def list_auxiliary_pending_user_questions(
    deps: StoreDeps,
    *,
    session_id: str,
) -> tuple[AuxiliaryPendingUserQuestion, ...]:
    """通过同一精确游标权威状态列出当前 问题。

    这只是基于 :func:`get_auxiliary_pending_user_question` 的 Session 范围
    读取索引。它不推断回答归属，也不创建第二条延续路径；返回的每一项都会通过
    现有 current-goal/current-revision 验证器重新加载。
    """

    _require_identifier("session_id", session_id)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT run.insession_task_id "
            "FROM insession_work_runs AS run "
            "JOIN insession_execution_subjects AS registry "
            "ON registry.execution_subject_id=run.execution_subject_id "
            "AND registry.session_id=run.session_id "
            "AND registry.insession_task_id=run.insession_task_id "
            "AND registry.subject_kind='auxiliary_node' "
            "AND registry.subject_contract_version='auxiliary_node_v2' "
            "JOIN insession_auxiliary_v2_node_execution_subject_bindings "
            "AS binding ON binding.binding_id=registry.auxiliary_v2_binding_id "
            "AND binding.session_id=run.session_id "
            "AND binding.insession_task_id=run.insession_task_id "
            "AND binding.auxiliary_graph_id=run.auxiliary_graph_id "
            "AND binding.auxiliary_graph_revision="
            "run.auxiliary_graph_revision "
            "AND binding.auxiliary_node_id=run.auxiliary_node_id "
            "AND binding.node_revision=run.node_revision "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=run.session_id "
            "AND control.insession_task_id=run.insession_task_id "
            "AND control.auxiliary_graph_id=run.auxiliary_graph_id "
            "WHERE run.session_id=? AND run.status='waiting_user' "
            "AND run.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "AND binding.goal_id=control.current_goal_id "
            "ORDER BY run.insession_task_id",
            (session_id,),
        ).fetchall()
    pending: list[AuxiliaryPendingUserQuestion] = []
    for row in rows:
        task_id = str(row["insession_task_id"])
        item = get_auxiliary_pending_user_question(
            deps,
            session_id=session_id,
            insession_task_id=task_id,
        )
        if item is not None:
            pending.append(item)
    return tuple(pending)


def commit_auxiliary_waiting_user_attempt(
    deps: StoreDeps,
    *,
    command: CommitAuxiliaryWaitingUserAttemptCommand,
) -> AuxiliaryContinuationMutationResult:
    """关闭 request_user_input 并投影完整的 等待聚合。"""

    if not isinstance(command, CommitAuxiliaryWaitingUserAttemptCommand):
        raise TypeError(
            "command must be CommitAuxiliaryWaitingUserAttemptCommand"
        )
    operation = "commit_waiting_user_attempt"
    payload_hash = _payload_hash(command.model_dump(mode="json"))
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            command=command,
            operation=operation,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        _require_apply_id_available(conn, command.apply_id)
        _require_owned_attempt_window(
            conn,
            command.session_id,
            command.turn_id,
            command.work_run_id,
            command.attempt_id,
            command.expected_window_revision,
        )
        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=command.session_id,
            work_run_id=command.work_run_id,
        )
        _require_command_authority(
            conn,
            authority=authority,
            command=command,
            task_status="active",
            node_status="active",
            goal_status="active",
            revision_status="active",
        )
        run_row = authority
        _require_run_revision(run_row, command.expected_work_run_revision)
        if (
            str(run_row["status"]) != "active"
            or run_row["reason"] is not None
            or str(run_row["current_attempt_id"] or "") != command.attempt_id
            or run_row["current_verification_request_id"] is not None
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question Attempt is not current on an active WorkRun"
            )
        _require_single_auxiliary_cursor(conn, authority=authority)
        attempt_row = _require_active_attempt(
            conn,
            command.work_run_id,
            command.attempt_id,
        )
        if attempt_row["decision_json"] is not None:
            raise AuxiliaryContinuationPersistenceError(
                "question AttemptDecision is already committed"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_tool_calls "
            "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
            (command.work_run_id, command.attempt_id),
        ).fetchone() is not None or conn.execute(
            "SELECT 1 FROM insession_work_run_tool_results "
            "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
            (command.work_run_id, command.attempt_id),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "undecided question Attempt already has execution authority"
            )

        progress = _load_progress(conn, command.work_run_id)
        _require_progress_revision(progress, command.expected_progress_revision)
        valid_result_ids = {
            item.tool_result_id
            for item in _load_tool_results(conn, work_run_id=command.work_run_id)
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
                (command.work_run_id, int(attempt_row["ordinal"])),
            ).fetchall()
        }
        if not historical_result_ids.issubset(valid_result_ids):
            raise AuxiliaryContinuationPersistenceError(
                "historical ToolResult authority is incomplete"
            )
        merged = merge_acceptance_progress(
            progress,
            command.decision.acceptance_updates,
            known_historical_tool_result_ids=historical_result_ids,
            expected_progress_revision=command.expected_progress_revision,
            current_work_run_revision=command.expected_work_run_revision,
            expected_work_run_revision=command.expected_work_run_revision,
        )
        if merged.status != "applied":
            codes = ",".join(issue.code.value for issue in merged.issues)
            raise AuxiliaryContinuationPersistenceError(
                f"Acceptance progress update was rejected: {codes}"
            )
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=command.work_run_id,
            active_seconds_delta=command.active_seconds_delta,
        )
        hard_limit_reached = (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        )
        next_status = (
            WorkRunStatus.FAILED
            if hard_limit_reached
            else WorkRunStatus.WAITING_USER
        )
        next_reason = (
            "work_run_limit_reached" if hard_limit_reached else "needs_input"
        )

        next_run_revision = command.expected_work_run_revision + 1
        next_window_revision = command.expected_window_revision + 1
        # ``attempt.budget_charge_id`` 是即时外键。先插入配套 charge；外围事务仍会
        # 保证完整 Attempt/WorkRun/Auxiliary 聚合变更的原子性。
        _insert_budget_charge(
            conn,
            budget_charge_id=command.apply_id,
            operation="commit_attempt_decision",
            session_id=command.session_id,
            work_run_id=command.work_run_id,
            turn_id=command.turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "commit_attempt_decision", command.apply_id
            ),
            work_run_revision_before=command.expected_work_run_revision,
            work_run_revision_after=next_run_revision,
            window_state_version_before=command.expected_window_revision,
            window_state_version_after=next_window_revision,
            transition=budget_transition,
            work_run_status_after=next_status,
            work_run_reason_after=next_reason,
            now=now,
        )
        if conn.execute(
            "UPDATE insession_work_run_attempts SET action='request_user_input', "
            "decision_json=?, progress_revision_after=?, status='closed', "
            "budget_after_json=?, budget_charge_id=?, "
            "close_reason='request_user_input', closed_at=? "
            "WHERE attempt_id=? AND work_run_id=? AND status='active' "
            "AND decision_json IS NULL",
            (
                _model_json(command.decision),
                merged.snapshot.revision,
                _model_json(budget_transition.budget_after),
                command.apply_id,
                now,
                command.attempt_id,
                command.work_run_id,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "question Attempt changed during settlement"
            )
        if merged.changed:
            snapshot_json = _model_json(merged.snapshot)
            if conn.execute(
                "UPDATE insession_work_run_acceptance_progress "
                "SET progress_revision=?, snapshot_hash=?, snapshot_json=?, "
                "updated_attempt_id=?, updated_at=? WHERE work_run_id=? "
                "AND progress_revision=?",
                (
                    merged.snapshot.revision,
                    _text_hash(snapshot_json),
                    snapshot_json,
                    command.attempt_id,
                    now,
                    command.work_run_id,
                    command.expected_progress_revision,
                ),
            ).rowcount != 1:
                raise AuxiliaryContinuationPersistenceError(
                    "Acceptance progress changed during question settlement"
                )
        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, "
            "current_attempt_id=NULL, updated_turn_id=?, updated_at=? "
            "WHERE work_run_id=? AND session_id=? AND revision=? "
            "AND status='active' AND reason IS NULL AND current_attempt_id=? "
            "AND current_verification_request_id IS NULL",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_transition.budget_after.active_seconds_consumed,
                command.turn_id,
                now,
                command.work_run_id,
                command.session_id,
                command.expected_work_run_revision,
                command.attempt_id,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun changed during waiting-user settlement"
            )
        next_versions = _transition_auxiliary_wait_states(
            conn,
            authority=authority,
            from_status="active",
            to_status="interrupted" if hard_limit_reached else "waiting_user",
            from_task_status="active",
            to_task_status=(
                "interrupted" if hard_limit_reached else "awaiting_user"
            ),
            now=now,
        )
        if conn.execute(
            "UPDATE turn_execution_windows SET current_attempt_id=NULL, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id=? AND pending_operation_id IS NULL "
            "AND state_version=?",
            (
                next_window_revision,
                now,
                command.session_id,
                command.turn_id,
                command.work_run_id,
                command.attempt_id,
                command.expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "Turn Window changed during waiting-user settlement"
            )
        base = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=command.work_run_id,
            attempt_id=command.attempt_id,
            window_state_version=next_window_revision,
            budget_transition=budget_transition,
        ).model_copy(
            update={
                "task_state_version": next_versions["task"],
                "node_state_version": next_versions["node"],
            }
        )
        result = _typed_result(
            base=base,
            operation=operation,
            authority=authority,
            versions=next_versions,
            question_sha256=_text_hash(command.decision.action.question),
        )
        # active-time 账本覆盖整个 WorkRun，其完整性加载器通过通用 WorkRun receipt
        # 关系认证每笔 charge。除下方仅用于 的精确权威 receipt 外，还需保留
        # 该基础 receipt（而非用前者替代）。此关系不分派 Auxiliary 变更。
        _insert_work_execution_receipt(
            conn,
            apply_id=command.apply_id,
            operation="commit_attempt_decision",
            session_id=command.session_id,
            work_run_id=command.work_run_id,
            payload_hash=payload_hash,
            result=base,
            now=now,
        )
        _insert_receipt(
            conn,
            command=command,
            operation=operation,
            authority=authority,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def continue_auxiliary_waiting_user_and_start_attempt(
    deps: StoreDeps,
    *,
    command: ContinueAuxiliaryWaitingUserCommand,
) -> AuxiliaryContinuationMutationResult:
    """消费一个精确 问题，并将 Attempt N+1 绑定到其回答。"""

    if not isinstance(command, ContinueAuxiliaryWaitingUserCommand):
        raise TypeError("command must be ContinueAuxiliaryWaitingUserCommand")
    operation = "continue_waiting_user_and_start_attempt"
    payload_hash = _payload_hash(command.model_dump(mode="json"))
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            command=command,
            operation=operation,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        _require_apply_id_available(conn, command.apply_id)
        window = _require_active_window(
            conn,
            session_id=command.session_id,
            turn_id=command.turn_id,
            expected_window_revision=command.expected_window_revision,
        )
        if any(
            window[name] is not None
            for name in (
                "current_work_run_id",
                "current_attempt_id",
                "pending_operation_id",
            )
        ):
            raise AuxiliaryContinuationPersistenceError(
                "answer continuation requires an unbound active Turn Window"
            )
        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=command.session_id,
            work_run_id=command.work_run_id,
        )
        _require_command_authority(
            conn,
            authority=authority,
            command=command,
            task_status="awaiting_user",
            node_status="waiting_user",
            goal_status="waiting_user",
            revision_status="waiting_user",
        )
        _require_run_revision(authority, command.expected_work_run_revision)
        if (
            str(authority["status"]) != "waiting_user"
            or str(authority["reason"] or "") != "needs_input"
            or authority["current_attempt_id"] is not None
            or authority["current_verification_request_id"] is not None
        ):
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun does not own one pending user question"
            )
        _require_single_auxiliary_cursor(conn, authority=authority)
        record = _load_record(
            conn,
            command.work_run_id,
            session_id=command.session_id,
        )
        if (
            record.work_run.status is not WorkRunStatus.WAITING_USER
            or record.pending_user_question is None
            or not record.attempts
            or record.attempts[-1].attempt.attempt_id
            != command.question_attempt_id
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question WorkRun aggregate is corrupt"
            )
        current_budget = _budget_from_row(authority)
        attempts_started = _require_attempt_row_aggregate(
            conn,
            work_run_id=command.work_run_id,
            expected_attempts_started=current_budget.attempts_started,
        )
        if (
            attempts_started >= current_budget.max_attempts
            or current_budget.active_seconds_consumed
            >= current_budget.soft_active_seconds
        ):
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun budget forbids answer continuation"
            )
        question_row = conn.execute(
            "SELECT * FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=?",
            (command.work_run_id, command.question_attempt_id),
        ).fetchone()
        if (
            question_row is None
            or int(question_row["ordinal"]) != attempts_started
            or str(question_row["status"]) != "closed"
            or str(question_row["action"] or "") != "request_user_input"
            or str(question_row["close_reason"] or "") != "request_user_input"
            or question_row["decision_json"] is None
            or str(authority["updated_turn_id"]) != str(question_row["turn_id"])
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question Attempt is not the exact current interaction"
            )
        try:
            question_decision = HostAcceptedAttemptDecision.model_validate_json(
                str(question_row["decision_json"])
            )
        except ValueError as exc:
            raise AuxiliaryContinuationPersistenceError(
                "question Attempt decision is corrupt"
            ) from exc
        if not isinstance(question_decision.action, RequestUserInputAction):
            raise AuxiliaryContinuationPersistenceError(
                "question Attempt action binding is corrupt"
            )
        question_sha256 = _text_hash(question_decision.action.question)
        settled_question_sha256 = _require_question_settlement_sha256(
            conn,
            authority=authority,
            question_row=question_row,
        )
        if (
            question_sha256 != command.expected_question_sha256
            or settled_question_sha256 != command.expected_question_sha256
        ):
            raise AuxiliaryContinuationPersistenceError(
                "pending question changed before answer continuation"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_attempts "
            "WHERE predecessor_question_attempt_id=? LIMIT 1",
            (command.question_attempt_id,),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "pending user question was already consumed"
            )
        progress = _load_progress(conn, command.work_run_id)
        _require_progress_revision(progress, command.expected_progress_revision)
        output, output_hash = _load_output_window(conn, command.work_run_id)
        if progress.evaluated_output_revision != output.output_revision:
            raise AuxiliaryContinuationPersistenceError(
                "Acceptance progress is detached from its OutputWindow"
            )
        answer_source = _load_answer_source(
            conn,
            session_id=command.session_id,
            turn_id=command.turn_id,
        )
        if (
            answer_source["content_sha256"]
            != command.expected_answer_source_sha256
        ):
            raise AuxiliaryContinuationPersistenceError(
                "answer source changed before continuation"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (command.session_id, command.turn_id, command.subject.task_id),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "answer Turn is not linked to the WorkRun Task"
            )
        previous_turn = conn.execute(
            "SELECT status, error_code, end_reason FROM runtime_turns "
            "WHERE session_id=? AND turn_id=?",
            (command.session_id, str(question_row["turn_id"])),
        ).fetchone()
        if not _question_turn_allows_answer_continuation(
            conn,
            authority=authority,
            previous_turn=previous_turn,
            question=question_decision.action.question,
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question Turn has no durable answer-continuation authority"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (command.turn_id, command.work_run_id),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "answer Turn already has a WorkRun link without this receipt"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE session_id=? "
            "AND status='active' LIMIT 1",
            (command.session_id,),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "Session already has an active WorkRun"
            )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, "
            "created_at) VALUES (?, ?, ?, ?, 'continued', ?)",
            (
                command.session_id,
                command.turn_id,
                command.work_run_id,
                link_revision,
                now,
            ),
        )
        ordinal = attempts_started + 1
        try:
            conn.execute(
                "INSERT INTO insession_work_run_attempts "
                "(attempt_id, work_run_id, turn_id, input_turn_id, "
                "predecessor_question_attempt_id, ordinal, status, action, "
                "decision_json, progress_revision_before, "
                "progress_revision_after, input_output_revision, "
                "input_output_hash, committed_output_revision, "
                "submitted_output_revision, input_checkpoint_id, "
                "input_verification_request_id, catalog_snapshot_json, "
                "catalog_snapshot_hash, budget_before_json, budget_after_json, "
                "close_reason, created_at, closed_at) VALUES "
                "(?, ?, ?, ?, ?, ?, 'active', NULL, NULL, ?, NULL, ?, ?, "
                "NULL, NULL, ?, NULL, ?, ?, ?, NULL, NULL, ?, NULL)",
                (
                    command.attempt_id,
                    command.work_run_id,
                    command.turn_id,
                    command.turn_id,
                    command.question_attempt_id,
                    ordinal,
                    progress.revision,
                    output.output_revision,
                    output_hash,
                    command.question_attempt_id,
                    _canonical_json(command.catalog_snapshot),
                    _text_hash(_canonical_json(command.catalog_snapshot)),
                    _model_json(current_budget),
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise AuxiliaryContinuationPersistenceError(
                "answer Attempt conflicts with stored authority"
            ) from exc
        next_run_revision = command.expected_work_run_revision + 1
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
                command.attempt_id,
                command.turn_id,
                now,
                command.work_run_id,
                command.session_id,
                command.expected_work_run_revision,
                attempts_started,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun changed during answer continuation"
            )
        next_versions = _transition_auxiliary_wait_states(
            conn,
            authority=authority,
            from_status="waiting_user",
            to_status="active",
            from_task_status="awaiting_user",
            to_task_status="active",
            now=now,
        )
        next_window_revision = command.expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=?, stage='L2_PLAN', "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                command.work_run_id,
                command.attempt_id,
                command.question_attempt_id,
                link_revision,
                next_window_revision,
                now,
                command.session_id,
                command.turn_id,
                command.expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "Turn Window changed during answer continuation"
            )
        answer_binding = _build_answer_binding(
            authority=authority,
            question_attempt_id=command.question_attempt_id,
            question=question_decision.action.question,
            answer_attempt_id=command.attempt_id,
            answer_source=answer_source,
        )
        base = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=command.work_run_id,
            attempt_id=command.attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        ).model_copy(
            update={
                "task_state_version": next_versions["task"],
                "node_state_version": next_versions["node"],
            }
        )
        result = _typed_result(
            base=base,
            operation=operation,
            authority=authority,
            versions=next_versions,
            answer_source_binding=answer_binding,
        )
        _insert_receipt(
            conn,
            command=command,
            operation=operation,
            authority=authority,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        _insert_answer_binding(
            conn,
            command=command,
            authority=authority,
            binding=answer_binding,
            now=now,
        )
        validate_auxiliary_answer_bindings_for_work_run(
            conn,
            run_row=authority,
        )
        return result


def resume_auxiliary_active_attempt(
    deps: StoreDeps,
    *,
    command: ResumeAuxiliaryActiveAttemptCommand,
) -> AuxiliaryContinuationMutationResult:
    """在旧 Turn 未完成后重新绑定一个精确 Attempt。

    未作决策的 Attempt 从 ``L2_PLAN`` 恢复。已决定 ``call_tools`` 的 Attempt
    只有在其不可变决策、完整物化调用记录、CatalogSnapshot 和现有 ToolResult
    前缀经通用 WorkRun 控制器所用的同一恢复验证器认证后，才会从 ``TOOL``
    恢复。受保护调用需要显式的禁止重发桥接授权。
    """

    if not isinstance(command, ResumeAuxiliaryActiveAttemptCommand):
        raise TypeError("command must be ResumeAuxiliaryActiveAttemptCommand")
    operation = "resume_active_attempt"
    payload_hash = _payload_hash(command.model_dump(mode="json"))
    normalized_catalog: dict[str, Any] | None = None
    catalog_json: str | None = None
    if command.catalog_snapshot is not None:
        try:
            catalog_json = _canonical_json(command.catalog_snapshot)
            normalized_catalog = json.loads(catalog_json)
        except (TypeError, ValueError) as exc:
            raise AuxiliaryContinuationPersistenceError(
                "decided tool recovery catalog snapshot is not canonical JSON"
            ) from exc
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            command=command,
            operation=operation,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        _require_apply_id_available(conn, command.apply_id)
        window = _require_active_window(
            conn,
            session_id=command.session_id,
            turn_id=command.turn_id,
            expected_window_revision=command.expected_window_revision,
        )
        if any(
            window[name] is not None
            for name in (
                "current_work_run_id",
                "current_attempt_id",
                "pending_operation_id",
            )
        ):
            raise AuxiliaryContinuationPersistenceError(
                "Attempt resume requires an unbound active Turn Window"
            )
        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=command.session_id,
            work_run_id=command.work_run_id,
        )
        _require_command_authority(
            conn,
            authority=authority,
            command=command,
            task_status="active",
            node_status="active",
            goal_status="active",
            revision_status="active",
        )
        _require_run_revision(authority, command.expected_work_run_revision)
        if (
            str(authority["status"]) != "active"
            or authority["reason"] is not None
            or str(authority["current_attempt_id"] or "") != command.attempt_id
            or authority["current_verification_request_id"] is not None
        ):
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun does not own one resumable active Attempt"
            )
        _require_single_auxiliary_cursor(conn, authority=authority)
        record = _load_record(
            conn,
            command.work_run_id,
            session_id=command.session_id,
        )
        if (
            record.work_run.status is not WorkRunStatus.ACTIVE
            or record.current_attempt_id != command.attempt_id
            or not record.attempts
            or record.attempts[-1].attempt.attempt_id != command.attempt_id
        ):
            raise AuxiliaryContinuationPersistenceError(
                "resumable WorkRun aggregate is corrupt"
            )
        progress = _load_progress(conn, command.work_run_id)
        _require_progress_revision(progress, command.expected_progress_revision)
        budget = _budget_from_row(authority)
        if budget.active_seconds_consumed >= budget.soft_active_seconds:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun active-time budget forbids Attempt resume"
            )
        attempt = _require_active_attempt(
            conn,
            command.work_run_id,
            command.attempt_id,
        )
        previous_turn_id = str(attempt["turn_id"])
        if (
            previous_turn_id != str(authority["updated_turn_id"])
            or int(attempt["ordinal"]) != int(authority["attempts_started"])
        ):
            raise AuxiliaryContinuationPersistenceError(
                "resumable Attempt is detached from its WorkRun aggregate"
            )
        if previous_turn_id == command.turn_id:
            raise AuxiliaryContinuationPersistenceError(
                "Attempt resume requires a different Turn"
            )
        decided_tool_recovery = normalized_catalog is not None
        if not decided_tool_recovery:
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
                raise AuxiliaryContinuationPersistenceError(
                    "only an undecided Attempt may use plan resume"
                )
            if conn.execute(
                "SELECT 1 FROM insession_work_run_tool_calls "
                "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
                (command.work_run_id, command.attempt_id),
            ).fetchone() is not None or conn.execute(
                "SELECT 1 FROM insession_work_run_tool_results "
                "WHERE work_run_id=? AND attempt_id=? LIMIT 1",
                (command.work_run_id, command.attempt_id),
            ).fetchone() is not None:
                raise AuxiliaryContinuationPersistenceError(
                    "Attempt with durable tool activity cannot use plan resume"
                )
        else:
            if (
                str(attempt["action"] or "") != "call_tools"
                or attempt["decision_json"] is None
                or attempt["progress_revision_after"] is None
                or int(attempt["progress_revision_after"])
                != command.expected_progress_revision
                or attempt["committed_output_revision"] is not None
                or attempt["submitted_output_revision"] is not None
                or attempt["budget_after_json"] is not None
                or attempt["budget_charge_id"] is not None
                or attempt["close_reason"] is not None
                or attempt["closed_at"] is not None
            ):
                raise AuxiliaryContinuationPersistenceError(
                    "Attempt is not one active decided call_tools batch"
                )
            stored_catalog_json = str(attempt["catalog_snapshot_json"])
            if (
                catalog_json is None
                or _text_hash(stored_catalog_json)
                != str(attempt["catalog_snapshot_hash"])
                or stored_catalog_json != catalog_json
            ):
                raise AuxiliaryContinuationPersistenceError(
                    "decided tool recovery catalog snapshot drifted"
                )
            try:
                decision = HostAcceptedAttemptDecision.model_validate_json(
                    str(attempt["decision_json"])
                )
            except ValueError as exc:
                raise AuxiliaryContinuationPersistenceError(
                    "decided tool recovery decision is corrupt"
                ) from exc
            if not isinstance(decision.action, HostMaterializedCallToolsAction):
                raise AuxiliaryContinuationPersistenceError(
                    "decided tool recovery requires a materialized decision"
                )
            try:
                _require_readonly_recovery_batch(
                    conn,
                    work_run_id=command.work_run_id,
                    attempt_id=command.attempt_id,
                    decision=decision,
                    catalog_snapshot=normalized_catalog,
                    allow_protected_recovery=command.allow_protected_recovery,
                )
            except WorkExecutionPersistenceError as exc:
                raise AuxiliaryContinuationPersistenceError(str(exc)) from exc
        previous_turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (previous_turn_id,),
        ).fetchone()
        if (
            previous_turn is None
            or str(previous_turn["session_id"]) != command.session_id
            or str(previous_turn["status"]) != "incomplete"
        ):
            raise AuxiliaryContinuationPersistenceError(
                "previous Attempt Turn is not durably incomplete"
            )
        if decided_tool_recovery and conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=? LIMIT 1",
            (previous_turn_id, command.work_run_id),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "decided tool Attempt lost its previous Turn ownership link"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (command.session_id, command.turn_id, command.subject.task_id),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "resume Turn is not linked to the WorkRun Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (command.turn_id, command.work_run_id),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "resume Turn already has a WorkRun link without this receipt"
            )
        validate_auxiliary_answer_bindings_for_work_run(
            conn,
            run_row=authority,
        )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, "
            "created_at) VALUES (?, ?, ?, ?, 'continued', ?)",
            (
                command.session_id,
                command.turn_id,
                command.work_run_id,
                link_revision,
                now,
            ),
        )
        if decided_tool_recovery:
            attempt_update_count = conn.execute(
                "UPDATE insession_work_run_attempts SET turn_id=? "
                "WHERE work_run_id=? AND attempt_id=? AND turn_id=? "
                "AND status='active' AND action='call_tools' "
                "AND decision_json IS NOT NULL "
                "AND progress_revision_after IS NOT NULL "
                "AND committed_output_revision IS NULL "
                "AND submitted_output_revision IS NULL "
                "AND budget_after_json IS NULL AND budget_charge_id IS NULL "
                "AND close_reason IS NULL AND closed_at IS NULL",
                (
                    command.turn_id,
                    command.work_run_id,
                    command.attempt_id,
                    previous_turn_id,
                ),
            ).rowcount
        else:
            attempt_update_count = conn.execute(
                "UPDATE insession_work_run_attempts SET turn_id=? "
                "WHERE work_run_id=? AND attempt_id=? AND turn_id=? "
                "AND status='active' AND action IS NULL AND decision_json IS NULL "
                "AND progress_revision_after IS NULL",
                (
                    command.turn_id,
                    command.work_run_id,
                    command.attempt_id,
                    previous_turn_id,
                ),
            ).rowcount
        if attempt_update_count != 1:
            raise AuxiliaryContinuationPersistenceError(
                "Attempt changed during Turn resume"
            )
        next_run_revision = command.expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET revision=?, updated_turn_id=?, "
            "updated_at=? WHERE work_run_id=? AND session_id=? AND revision=? "
            "AND status='active' AND reason IS NULL AND current_attempt_id=? "
            "AND current_verification_request_id IS NULL",
            (
                next_run_revision,
                command.turn_id,
                now,
                command.work_run_id,
                command.session_id,
                command.expected_work_run_revision,
                command.attempt_id,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun changed during Attempt resume"
            )
        next_window_revision = command.expected_window_revision + 1
        window_stage = "TOOL" if decided_tool_recovery else "L2_PLAN"
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=?, latest_checkpoint_id=?, stage=?, "
            "turn_workrun_link_revision=?, state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND pending_operation_id IS NULL AND state_version=?",
            (
                command.work_run_id,
                command.attempt_id,
                attempt["input_checkpoint_id"],
                window_stage,
                link_revision,
                next_window_revision,
                now,
                command.session_id,
                command.turn_id,
                command.expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "Turn Window changed during Attempt resume"
            )
        versions = _authority_versions(authority)
        base = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=command.work_run_id,
            attempt_id=command.attempt_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        ).model_copy(
            update={
                "task_state_version": versions["task"],
                "node_state_version": versions["node"],
            }
        )
        result = _typed_result(
            base=base,
            operation=operation,
            authority=authority,
            versions=versions,
        )
        _insert_receipt(
            conn,
            command=command,
            operation=operation,
            authority=authority,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def resume_auxiliary_verification(
    deps: StoreDeps,
    *,
    command: ResumeAuxiliaryVerificationCommand,
) -> AuxiliaryVerificationResumeResult:
    """移动一个待处理 verifier 租约，但不替换其请求。

    不可变请求仍归其原始来源 Turn、已提交 Attempt、OutputWindow revision、
    execution subject 和 binding digest 所有。提升 request revision 只会隔离来自
    已放弃调用的 provider 结果；verifier 活跃时间随后由结果结算计费。
    """

    if not isinstance(command, ResumeAuxiliaryVerificationCommand):
        raise TypeError("command must be ResumeAuxiliaryVerificationCommand")
    operation = "resume_verification"
    payload_hash = _payload_hash(command.model_dump(mode="json"))
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_verification_resume_replay(
            conn,
            command=command,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        _require_apply_id_available(conn, command.apply_id)
        window = _require_active_window(
            conn,
            session_id=command.session_id,
            turn_id=command.turn_id,
            expected_window_revision=command.expected_window_revision,
        )
        if any(
            window[name] is not None
            for name in (
                "current_work_run_id",
                "current_attempt_id",
                "pending_operation_id",
                "latest_checkpoint_id",
            )
        ):
            raise AuxiliaryContinuationPersistenceError(
                "verification resume requires an unbound active Turn Window"
            )

        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=command.session_id,
            work_run_id=command.work_run_id,
        )
        _require_command_authority(
            conn,
            authority=authority,
            command=command,
            task_status="active",
            node_status="active",
            goal_status="active",
            revision_status="active",
        )
        _require_run_revision(authority, command.expected_work_run_revision)
        if (
            str(authority["status"]) != "active"
            or str(authority["reason"] or "") != "verification_pending"
            or authority["current_attempt_id"] is not None
            or str(authority["current_verification_request_id"] or "")
            != command.verification_request_id
        ):
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun does not own this pending verification request"
            )
        _require_single_auxiliary_cursor(conn, authority=authority)

        request_row = _load_auxiliary_request_row(
            conn,
            session_id=command.session_id,
            verification_request_id=command.verification_request_id,
        )
        _require_request_revision(
            request_row,
            command.expected_verification_request_revision,
        )
        record = _auxiliary_record_from_row(request_row)
        request = record.request
        if (
            request.status is not TaskNodeVerificationRequestStatus.PENDING
            or request.work_run_id != command.work_run_id
            or request.subject != command.subject
            or str(request_row["execution_subject_id"])
            != str(authority["execution_subject_id"])
        ):
            raise AuxiliaryContinuationPersistenceError(
                "verification request is detached from exact WorkRun authority"
            )

        previous_turn_id = str(authority["updated_turn_id"] or "")
        if (
            not previous_turn_id
            or previous_turn_id == command.turn_id
        ):
            raise AuxiliaryContinuationPersistenceError(
                "verification resume requires a different predecessor Turn"
            )
        previous_turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (previous_turn_id,),
        ).fetchone()
        if (
            previous_turn is None
            or str(previous_turn["session_id"]) != command.session_id
            or str(previous_turn["status"]) != "incomplete"
        ):
            raise AuxiliaryContinuationPersistenceError(
                "previous verification Turn is not durably incomplete"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE session_id=? AND turn_id=? AND work_run_id=?",
            (command.session_id, previous_turn_id, command.work_run_id),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "pending verification lost its source Turn ownership link"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE session_id=? AND turn_id=? AND work_run_id=?",
            (
                command.session_id,
                request.request_turn_id,
                command.work_run_id,
            ),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "pending verification lost its immutable request-source link"
            )
        if previous_turn_id != request.request_turn_id:
            _require_verification_resume_predecessor_receipt(
                conn,
                command=command,
                authority=authority,
                request_row=request_row,
                previous_turn_id=previous_turn_id,
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (command.session_id, command.turn_id, command.subject.task_id),
        ).fetchone() is None:
            raise AuxiliaryContinuationPersistenceError(
                "verification resume Turn is not linked to the WorkRun Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_auxiliary_v2_waiting_user_answer_bindings "
            "WHERE session_id=? AND answer_source_turn_id=?",
            (command.session_id, command.turn_id),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "a answer-message Turn cannot resume verification"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (command.turn_id, command.work_run_id),
        ).fetchone() is not None:
            raise AuxiliaryContinuationPersistenceError(
                "verification resume Turn already has a WorkRun link without this receipt"
            )

        # 此精确重建会认证已提交 Attempt、当前 OutputWindow 快照、Acceptance
        # 进度、请求来源 Turn 和当前节点定义。
        try:
            _revalidate_auxiliary_request_binding(
                conn,
                request_row,
            )
        except WorkExecutionPersistenceError as exc:
            raise AuxiliaryContinuationPersistenceError(str(exc)) from exc
        budget = _budget_from_row(authority)
        if budget != request.prepared_budget:
            raise AuxiliaryContinuationPersistenceError(
                "verification active-time budget drifted after prepare"
            )
        if budget.active_seconds_consumed >= budget.soft_active_seconds:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun active-time budget forbids verification resume"
            )

        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, "
            "created_at) VALUES (?, ?, ?, ?, 'continued', ?)",
            (
                command.session_id,
                command.turn_id,
                command.work_run_id,
                link_revision,
                now,
            ),
        )
        next_request_revision = command.expected_verification_request_revision + 1
        if conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET request_revision=?, updated_at=? "
            "WHERE verification_request_id=? AND session_id=? "
            "AND work_run_id=? AND execution_subject_id=? "
            "AND request_turn_id=? AND submitted_attempt_id=? "
            "AND output_revision=? AND request_binding_hash=? "
            "AND request_revision=? AND status='pending'",
            (
                next_request_revision,
                now,
                command.verification_request_id,
                command.session_id,
                command.work_run_id,
                str(authority["execution_subject_id"]),
                request.request_turn_id,
                request.submitted_attempt_id,
                request.output_revision,
                str(request_row["request_binding_hash"]),
                command.expected_verification_request_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "verification request changed during Turn resume"
            )
        next_run_revision = command.expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET revision=?, updated_turn_id=?, "
            "updated_at=? WHERE work_run_id=? AND session_id=? AND revision=? "
            "AND status='active' AND reason='verification_pending' "
            "AND current_attempt_id IS NULL "
            "AND current_verification_request_id=? "
            "AND execution_subject_id=?",
            (
                next_run_revision,
                command.turn_id,
                now,
                command.work_run_id,
                command.session_id,
                command.expected_work_run_revision,
                command.verification_request_id,
                str(authority["execution_subject_id"]),
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "WorkRun changed during verification resume"
            )
        next_window_revision = command.expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=NULL, latest_checkpoint_id=?, "
            "stage='VERIFICATION', turn_workrun_link_revision=?, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id IS NULL "
            "AND current_attempt_id IS NULL AND pending_operation_id IS NULL "
            "AND latest_checkpoint_id IS NULL AND state_version=?",
            (
                command.work_run_id,
                command.verification_request_id,
                link_revision,
                next_window_revision,
                now,
                command.session_id,
                command.turn_id,
                command.expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryContinuationPersistenceError(
                "Turn Window changed during verification resume"
            )

        versions = _authority_versions(authority)
        result = AuxiliaryVerificationResumeResult(
            status="applied",
            verification_request_id=command.verification_request_id,
            verification_request_revision=next_request_revision,
            verification_request_status=TaskNodeVerificationRequestStatus.PENDING,
            work_run_id=command.work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=WorkRunStatus.ACTIVE,
            work_run_reason="verification_pending",
            window_state_version=next_window_revision,
            task_state_version=versions["task"],
            node_state_version=versions["node"],
            execution_subject_id=str(authority["execution_subject_id"]),
            subject=command.subject,
            goal_id=str(authority["goal_id"]),
            structure_sha256=str(authority["structure_sha256"]),
            request_source_turn_id=request.request_turn_id,
            request_binding_sha256=str(request_row["request_binding_hash"]),
            submitted_attempt_id=request.submitted_attempt_id,
            output_revision=request.output_revision,
            active_seconds_consumed=budget.active_seconds_consumed,
            turn_work_run_link_revision=link_revision,
            control_state_version=versions["control"],
            goal_state_version=versions["goal"],
            revision_state_version=versions["revision"],
        )
        _insert_receipt(
            conn,
            command=command,
            operation=operation,
            authority=authority,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def validate_auxiliary_answer_bindings_for_work_run(
    conn: sqlite3.Connection,
    *,
    run_row: sqlite3.Row,
) -> None:
    """根据来源认证每个 前置问题的回答。"""

    if _execution_subject_contract_version(conn, run_row) != _EXECUTION_SUBJECT_CONTRACT_VERSION:
        raise AuxiliaryContinuationPersistenceError(
            "answer binding validator requires AuxiliaryNode execution authority"
        )
    work_run_id = str(run_row["work_run_id"])
    # ``work_execution._load_record`` 提供通用 WorkRun 记录。此处加载其不可变
    # 加载不可变 Auxiliary 绑定，使绑定哈希依据 goal/definition 权威状态核验，而不是信任
    # 通用投影中缺失的字段。
    authority = _load_exact_auxiliary_authority(
        conn,
        session_id=str(run_row["session_id"]),
        work_run_id=work_run_id,
    )
    predecessor_rows = conn.execute(
        "SELECT attempt_id, predecessor_question_attempt_id "
        "FROM insession_work_run_attempts WHERE work_run_id=? "
        "AND predecessor_question_attempt_id IS NOT NULL ORDER BY ordinal",
        (work_run_id,),
    ).fetchall()
    bindings = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_waiting_user_answer_bindings "
        "WHERE work_run_id=? ORDER BY created_at, answer_binding_id",
        (work_run_id,),
    ).fetchall()
    continuation_receipt_count = int(
        conn.execute(
            "SELECT COUNT(*) "
            "FROM insession_auxiliary_v2_execution_apply_receipts "
            "WHERE work_run_id=? "
            "AND operation='continue_waiting_user_and_start_attempt'",
            (work_run_id,),
        ).fetchone()[0]
    )
    if not (
        len(bindings)
        == len(predecessor_rows)
        == continuation_receipt_count
    ):
        raise AuxiliaryContinuationPersistenceError(
            "answer bindings do not cover every consumed question receipt"
        )
    by_attempt = {str(row["answer_attempt_id"]): row for row in bindings}
    if len(by_attempt) != len(bindings):
        raise AuxiliaryContinuationPersistenceError(
            "answer binding identities are ambiguous"
        )
    for predecessor in predecessor_rows:
        answer_attempt_id = str(predecessor["attempt_id"])
        binding = by_attempt.get(answer_attempt_id)
        if binding is None:
            raise AuxiliaryContinuationPersistenceError(
                "consumed question lost its answer-source binding"
            )
        source = _load_answer_source(
            conn,
            session_id=str(binding["session_id"]),
            turn_id=str(binding["answer_source_turn_id"]),
            require_running=False,
        )
        question = conn.execute(
            "SELECT attempt_id, turn_id, decision_json, budget_charge_id "
            "FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=? AND status='closed' "
            "AND action='request_user_input'",
            (work_run_id, str(binding["question_attempt_id"])),
        ).fetchone()
        answer = conn.execute(
            "SELECT input_turn_id, predecessor_question_attempt_id "
            "FROM insession_work_run_attempts WHERE work_run_id=? "
            "AND attempt_id=?",
            (work_run_id, answer_attempt_id),
        ).fetchone()
        try:
            decision = HostAcceptedAttemptDecision.model_validate_json(
                str(question["decision_json"]) if question is not None else ""
            )
        except ValueError as exc:
            raise AuxiliaryContinuationPersistenceError(
                "answer binding question decision is corrupt"
            ) from exc
        if not isinstance(decision.action, RequestUserInputAction):
            raise AuxiliaryContinuationPersistenceError(
                "answer binding question action is corrupt"
            )
        if _require_question_settlement_sha256(
            conn,
            authority=authority,
            question_row=question,
        ) != _text_hash(decision.action.question):
            raise AuxiliaryContinuationPersistenceError(
                "answer binding question changed after settlement"
            )
        typed = _answer_binding_from_row(binding)
        _require_answer_binding_receipt(
            conn,
            authority=authority,
            binding_row=binding,
            binding=typed,
        )
        expected = _build_answer_binding(
            authority=authority,
            question_attempt_id=str(binding["question_attempt_id"]),
            question=decision.action.question,
            answer_attempt_id=answer_attempt_id,
            answer_source=source,
        )
        if (
            typed != expected
            or answer is None
            or str(answer["input_turn_id"]) != typed.answer_source_turn_id
            or str(answer["predecessor_question_attempt_id"] or "")
            != typed.question_attempt_id
            or str(predecessor["predecessor_question_attempt_id"])
            != typed.question_attempt_id
            or str(binding["execution_subject_id"])
            != str(authority["execution_subject_id"])
        ):
            raise AuxiliaryContinuationPersistenceError(
                "answer-source binding is corrupt or has drifted"
            )


def _question_turn_allows_answer_continuation(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
    previous_turn: sqlite3.Row | None,
    question: str,
) -> bool:
    """接受公开问题 Turn 或一个由精确 trigger 持有的内部 gate。

    普通 USER_GATE 问题属于正式交互，因此要求 Turn 已完成。候选项级别的缺失信息
    修复有所不同：其冻结的根 Delivery 被有意设为内部内容，因此 host 会在接受
    回答前以 ``incomplete/WAITING_USER`` 关闭问题 Turn。只有在重新加载完整、
    不可变的 候选项结算，并确认其活动 N->N+1 trigger 与当前 goal、USER_GATE
    定义及冻结的阻塞问题一致后，才允许这一例外。
    """

    if previous_turn is None:
        return False
    if str(previous_turn["status"]) == "completed":
        return True
    if (
        str(previous_turn["status"]) != "incomplete"
        or str(previous_turn["end_reason"] or "") != "host_stopped"
        or str(previous_turn["error_code"] or "") != "WAITING_USER"
        or str(authority["executor_kind"]) != "user_gate"
        or str(authority["node_objective"]) != question
        or authority["base_task_graph_revision"] is None
        or authority["target_task_graph_revision"] is None
    ):
        return False

    trigger_rows = conn.execute(
        "SELECT * FROM insession_active_task_graph_revision_triggers "
        "WHERE session_id=? AND insession_task_id=? "
        "AND base_graph_revision=? AND target_graph_revision=?",
        (
            str(authority["session_id"]),
            str(authority["insession_task_id"]),
            int(authority["base_task_graph_revision"]),
            int(authority["target_task_graph_revision"]),
        ),
    ).fetchall()
    if len(trigger_rows) != 1:
        return False
    try:
        trigger = (
            task_delivery_validation_records
            ._load_authenticated_trigger_authority(
                conn,
                str(trigger_rows[0]["trigger_id"]),
            )
        )
        settlement_rows = conn.execute(
            "SELECT * FROM insession_task_delivery_validation_settlements "
            "WHERE settlement_id=?",
            (trigger.settlement_id,),
        ).fetchall()
        if len(settlement_rows) != 1:
            return False
        stored = (
            task_delivery_validation_records
            ._stored_candidate_from_settlement_row(
                conn,
                settlement_rows[0],
            )
        )
    except task_delivery_validation_records.TaskDeliveryValidationPersistenceError:
        return False

    result = stored.intent.result
    non_pass = tuple(
        finding
        for finding in result.findings
        if finding.verdict is not TaskDeliveryValidationVerdict.PASS
    )
    return bool(
        stored.trigger == trigger
        and trigger.session_id == str(authority["session_id"])
        and trigger.task_id == str(authority["insession_task_id"])
        and trigger.base_graph_revision
        == int(authority["base_task_graph_revision"])
        and trigger.target_graph_revision
        == int(authority["target_task_graph_revision"])
        and result.disposition is TaskDeliveryValidationDisposition.BLOCKED
        and non_pass
        and all(
            finding.fault_domain
            is TaskDeliveryValidationFaultDomain.MISSING_INFORMATION
            for finding in non_pass
        )
        and question in result.blocking_questions
    )


def _load_exact_auxiliary_authority(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    work_run_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT run.*, registry.subject_contract_version, "
        "binding.goal_id AS bound_goal_id, "
        "binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "control.current_goal_id, control.current_auxiliary_graph_revision, "
        "control.state_version AS control_state_version, "
        "goal.goal_id, goal.status AS goal_status, "
        "goal.state_version AS goal_state_version, "
        "goal.base_task_graph_revision, goal.target_task_graph_revision, "
        "revision.structure_contract_version, revision.structure_sha256, "
        "revision_state.status AS revision_status, "
        "revision_state.state_version AS revision_state_version, "
        "node_state.status AS node_status, "
        "node_state.state_version AS node_state_version, "
        "definition.executor_kind, definition.objective AS node_objective, "
        "definition.definition_sha256, "
        "task.current_graph_revision AS task_graph_revision, "
        "task.current_status AS task_status, "
        "task.state_version AS task_state_version "
        "FROM insession_work_runs AS run "
        "JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=run.execution_subject_id "
        "AND registry.session_id=run.session_id "
        "AND registry.insession_task_id=run.insession_task_id "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=run.session_id "
        "AND binding.insession_task_id=run.insession_task_id "
        "AND binding.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND binding.auxiliary_graph_revision=run.auxiliary_graph_revision "
        "AND binding.auxiliary_node_id=run.auxiliary_node_id "
        "AND binding.node_revision=run.node_revision "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=run.session_id "
        "AND control.insession_task_id=run.insession_task_id "
        "AND control.auxiliary_graph_id=run.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_snapshots AS revision "
        "ON revision.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND revision.auxiliary_graph_revision=run.auxiliary_graph_revision "
        "AND revision.insession_task_id=run.insession_task_id "
        "AND revision.goal_id=binding.goal_id "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.goal_id=revision.goal_id "
        "AND goal.session_id=run.session_id "
        "AND goal.insession_task_id=run.insession_task_id "
        "AND goal.auxiliary_graph_id=run.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "ON membership.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND membership.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "AND membership.auxiliary_node_id=run.auxiliary_node_id "
        "AND membership.node_revision=run.node_revision "
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
        "JOIN insession_tasks AS task ON task.session_id=run.session_id "
        "AND task.insession_task_id=run.insession_task_id "
        "WHERE run.session_id=? AND run.work_run_id=?",
        (session_id, work_run_id),
    ).fetchone()
    if row is None:
        raise AuxiliaryContinuationPersistenceError(
            "unknown or detached AuxiliaryGraph WorkRun"
        )
    return row


def _require_auxiliary_authority_shape(authority: sqlite3.Row) -> None:
    subject = _subject_from_run_row(authority)
    task_revision = (
        int(authority["task_graph_revision"])
        if authority["task_graph_revision"] is not None
        else None
    )
    base_revision = (
        int(authority["base_task_graph_revision"])
        if authority["base_task_graph_revision"] is not None
        else None
    )
    if (
        not isinstance(subject, AuxiliaryNodeSubject)
        or str(authority["subject_contract_version"])
        != _EXECUTION_SUBJECT_CONTRACT_VERSION
        or str(authority["structure_contract_version"])
        != "auxiliary-graph-revision-v2"
        or str(authority["bound_goal_id"]) != str(authority["goal_id"])
        or str(authority["current_goal_id"]) != str(authority["goal_id"])
        or int(authority["current_auxiliary_graph_revision"])
        != subject.auxiliary_graph_revision
        or str(authority["bound_executor_kind"])
        != str(authority["executor_kind"])
        or str(authority["bound_definition_sha256"])
        != str(authority["definition_sha256"])
        or task_revision != base_revision
    ):
        raise AuxiliaryContinuationPersistenceError(
            "AuxiliaryGraph execution authority is stale or spliced"
        )


def _require_command_authority(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
    command: Any,
    task_status: str,
    node_status: str,
    goal_status: str,
    revision_status: str,
) -> None:
    _require_auxiliary_authority_shape(authority)
    subject = _subject_from_run_row(authority)
    if subject != command.subject:
        raise AuxiliaryContinuationPersistenceError(
            "continuation execution subject is stale"
        )
    expected_structure_sha256 = getattr(
        command,
        "expected_structure_sha256",
        None,
    )
    if (
        expected_structure_sha256 is not None
        and str(authority["structure_sha256"]) != expected_structure_sha256
    ):
        raise AuxiliaryContinuationPersistenceError(
            "continuation graph structure is stale"
        )
    if _execution_subject_contract_version(conn, authority) != _EXECUTION_SUBJECT_CONTRACT_VERSION:
        raise AuxiliaryContinuationPersistenceError(
            "continuation crossed its execution-subject contract"
        )
    _require_authority_statuses(
        authority,
        task_status=task_status,
        node_status=node_status,
        goal_status=goal_status,
        revision_status=revision_status,
    )
    expected = {
        "task": command.expected_task_state_version,
        "node": command.expected_node_state_version,
        "control": command.expected_control_state_version,
        "goal": command.expected_goal_state_version,
        "revision": command.expected_revision_state_version,
    }
    actual = _authority_versions(authority)
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise WorkExecutionRevisionConflict(
                expected=expected_value,
                actual=actual[name],
            )


def _require_authority_statuses(
    authority: sqlite3.Row,
    *,
    task_status: str,
    node_status: str,
    goal_status: str,
    revision_status: str,
) -> None:
    if (
        str(authority["task_status"]) != task_status
        or str(authority["node_status"]) != node_status
        or str(authority["goal_status"]) != goal_status
        or str(authority["revision_status"]) != revision_status
    ):
        raise AuxiliaryContinuationPersistenceError(
            "Auxiliary node/goal/revision aggregate is not at the required state"
        )


def _authority_versions(authority: sqlite3.Row) -> dict[str, int]:
    return {
        "task": int(authority["task_state_version"]),
        "node": int(authority["node_state_version"]),
        "control": int(authority["control_state_version"]),
        "goal": int(authority["goal_state_version"]),
        "revision": int(authority["revision_state_version"]),
    }


def _transition_auxiliary_wait_states(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
    from_status: str,
    to_status: str,
    from_task_status: str,
    to_task_status: str,
    now: str,
) -> dict[str, int]:
    versions = _authority_versions(authority)
    if conn.execute(
        "UPDATE insession_auxiliary_node_states_v2 SET status=?, "
        "state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND auxiliary_node_id=? AND node_revision=? AND status=? "
        "AND state_version=?",
        (
            to_status,
            now,
            str(authority["auxiliary_graph_id"]),
            int(authority["auxiliary_graph_revision"]),
            str(authority["auxiliary_node_id"]),
            int(authority["node_revision"]),
            from_status,
            versions["node"],
        ),
    ).rowcount != 1:
        raise AuxiliaryContinuationPersistenceError(
            "node changed during wait-state transition"
        )
    if conn.execute(
        "UPDATE insession_auxiliary_graph_goals SET status=?, "
        "state_version=state_version+1, updated_at=? WHERE goal_id=? "
        "AND status=? AND state_version=?",
        (
            to_status,
            now,
            str(authority["goal_id"]),
            from_status,
            versions["goal"],
        ),
    ).rowcount != 1:
        raise AuxiliaryContinuationPersistenceError(
            "goal changed during wait-state transition"
        )
    if conn.execute(
        "UPDATE insession_auxiliary_graph_revision_states_v2 SET status=?, "
        "state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND status=? AND state_version=?",
        (
            to_status,
            now,
            str(authority["auxiliary_graph_id"]),
            int(authority["auxiliary_graph_revision"]),
            from_status,
            versions["revision"],
        ),
    ).rowcount != 1:
        raise AuxiliaryContinuationPersistenceError(
            "revision changed during wait-state transition"
        )
    if conn.execute(
        "UPDATE insession_tasks SET current_status=?, "
        "state_version=state_version+1, updated_at=? "
        "WHERE session_id=? AND insession_task_id=? "
        "AND current_graph_revision IS ? AND current_status=? "
        "AND state_version=?",
        (
            to_task_status,
            now,
            str(authority["session_id"]),
            str(authority["insession_task_id"]),
            authority["base_task_graph_revision"],
            from_task_status,
            versions["task"],
        ),
    ).rowcount != 1:
        raise AuxiliaryContinuationPersistenceError(
            "Task changed during wait-state transition"
        )
    return {
        "task": versions["task"] + 1,
        "node": versions["node"] + 1,
        "control": versions["control"],
        "goal": versions["goal"] + 1,
        "revision": versions["revision"] + 1,
    }


def _current_auxiliary_cursor_rows(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in _NONTERMINAL_RUN_STATUSES)
    return conn.execute(
        "SELECT run.work_run_id, run.status FROM insession_work_runs AS run "
        "JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=run.execution_subject_id "
        "AND registry.session_id=run.session_id "
        "AND registry.insession_task_id=run.insession_task_id "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=run.session_id "
        "AND binding.insession_task_id=run.insession_task_id "
        "AND binding.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND binding.auxiliary_graph_revision=run.auxiliary_graph_revision "
        "AND binding.auxiliary_node_id=run.auxiliary_node_id "
        "AND binding.node_revision=run.node_revision "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=run.session_id "
        "AND control.insession_task_id=run.insession_task_id "
        "AND control.auxiliary_graph_id=run.auxiliary_graph_id "
        "WHERE run.session_id=? AND run.insession_task_id=? "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "AND run.auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "AND binding.goal_id=control.current_goal_id "
        f"AND run.status IN ({placeholders}) "
        "ORDER BY run.created_at, run.work_run_id",
        (session_id, task_id, *_NONTERMINAL_RUN_STATUSES),
    ).fetchall()


def _require_single_auxiliary_cursor(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
) -> None:
    rows = _current_auxiliary_cursor_rows(
        conn,
        session_id=str(authority["session_id"]),
        task_id=str(authority["insession_task_id"]),
    )
    if len(rows) != 1 or str(rows[0]["work_run_id"]) != str(
        authority["work_run_id"]
    ):
        raise AuxiliaryContinuationPersistenceError(
            "AuxiliaryGraph does not have this WorkRun as its single cursor"
        )


def _load_answer_source(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    require_running: bool = True,
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT runtime.status, input.message_id, input.turn_idx, "
        "turn_input.content FROM runtime_turns AS runtime "
        "JOIN runtime_turn_inputs AS input "
        "ON input.session_id=runtime.session_id "
        "AND input.turn_id=runtime.turn_id "
        "JOIN session_turns AS turn_input "
        "ON turn_input.session_id=input.session_id "
        "AND turn_input.turn_idx=input.turn_idx AND turn_input.role='user' "
        "WHERE runtime.session_id=? AND runtime.turn_id=?",
        (session_id, turn_id),
    ).fetchone()
    if row is None or (require_running and str(row["status"]) != "running"):
        raise AuxiliaryContinuationPersistenceError(
            "continuation has no authoritative accepted answer source"
        )
    content = str(row["content"])
    if not content:
        raise AuxiliaryContinuationPersistenceError(
            "continuation answer source is empty"
        )
    return {
        "turn_id": turn_id,
        "message_id": str(row["message_id"]),
        "turn_idx": int(row["turn_idx"]),
        "content_sha256": _text_hash(content),
        "utf8_bytes": len(content.encode("utf-8")),
    }


def _build_answer_binding(
    *,
    authority: sqlite3.Row,
    question_attempt_id: str,
    question: str,
    answer_attempt_id: str,
    answer_source: Mapping[str, Any],
) -> AuxiliaryAnswerSourceBinding:
    answer_binding_id = "auxv2answer-" + _payload_hash(
        {
            "session_id": str(authority["session_id"]),
            "work_run_id": str(authority["work_run_id"]),
            "question_attempt_id": question_attempt_id,
            "answer_attempt_id": answer_attempt_id,
            "answer_source_turn_id": str(answer_source["turn_id"]),
        }
    )[:32]
    payload = {
        "answer_binding_id": answer_binding_id,
        "session_id": str(authority["session_id"]),
        "task_id": str(authority["insession_task_id"]),
        "work_run_id": str(authority["work_run_id"]),
        "execution_subject_id": str(authority["execution_subject_id"]),
        "auxiliary_graph_id": str(authority["auxiliary_graph_id"]),
        "goal_id": str(authority["goal_id"]),
        "auxiliary_graph_revision": int(authority["auxiliary_graph_revision"]),
        "auxiliary_node_id": str(authority["auxiliary_node_id"]),
        "node_revision": int(authority["node_revision"]),
        "question_attempt_id": question_attempt_id,
        "question_sha256": _text_hash(question),
        "answer_attempt_id": answer_attempt_id,
        "answer_source_turn_id": str(answer_source["turn_id"]),
        "answer_source_message_id": str(answer_source["message_id"]),
        "answer_source_turn_idx": int(answer_source["turn_idx"]),
        "answer_source_content_sha256": str(answer_source["content_sha256"]),
        "answer_source_utf8_bytes": int(answer_source["utf8_bytes"]),
    }
    return AuxiliaryAnswerSourceBinding(
        answer_binding_id=answer_binding_id,
        question_attempt_id=question_attempt_id,
        question_sha256=payload["question_sha256"],
        answer_attempt_id=answer_attempt_id,
        answer_source_turn_id=payload["answer_source_turn_id"],
        answer_source_message_id=payload["answer_source_message_id"],
        answer_source_turn_idx=payload["answer_source_turn_idx"],
        answer_source_content_sha256=payload["answer_source_content_sha256"],
        answer_source_utf8_bytes=payload["answer_source_utf8_bytes"],
        binding_sha256=_payload_hash(payload),
    )


def _answer_binding_from_row(
    row: sqlite3.Row,
) -> AuxiliaryAnswerSourceBinding:
    return AuxiliaryAnswerSourceBinding(
        answer_binding_id=str(row["answer_binding_id"]),
        question_attempt_id=str(row["question_attempt_id"]),
        question_sha256=str(row["question_sha256"]),
        answer_attempt_id=str(row["answer_attempt_id"]),
        answer_source_turn_id=str(row["answer_source_turn_id"]),
        answer_source_message_id=str(row["answer_source_message_id"]),
        answer_source_turn_idx=int(row["answer_source_turn_idx"]),
        answer_source_content_sha256=str(row["answer_source_content_sha256"]),
        answer_source_utf8_bytes=int(row["answer_source_utf8_bytes"]),
        binding_sha256=str(row["binding_sha256"]),
    )


def _insert_answer_binding(
    conn: sqlite3.Connection,
    *,
    command: ContinueAuxiliaryWaitingUserCommand,
    authority: sqlite3.Row,
    binding: AuxiliaryAnswerSourceBinding,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_waiting_user_answer_bindings "
        "(answer_binding_id, apply_id, session_id, insession_task_id, "
        "work_run_id, execution_subject_id, auxiliary_graph_id, goal_id, "
        "auxiliary_graph_revision, auxiliary_node_id, node_revision, "
        "question_attempt_id, question_sha256, answer_attempt_id, "
        "answer_source_turn_id, answer_source_message_id, "
        "answer_source_turn_idx, answer_source_content_sha256, "
        "answer_source_utf8_bytes, binding_sha256, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            binding.answer_binding_id,
            command.apply_id,
            command.session_id,
            command.subject.task_id,
            command.work_run_id,
            str(authority["execution_subject_id"]),
            command.subject.auxiliary_graph_id,
            str(authority["goal_id"]),
            command.subject.auxiliary_graph_revision,
            command.subject.node_id,
            command.subject.node_revision,
            binding.question_attempt_id,
            binding.question_sha256,
            binding.answer_attempt_id,
            binding.answer_source_turn_id,
            binding.answer_source_message_id,
            binding.answer_source_turn_idx,
            binding.answer_source_content_sha256,
            binding.answer_source_utf8_bytes,
            binding.binding_sha256,
            now,
        ),
    )


def _require_question_settlement_sha256(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
    question_row: sqlite3.Row,
) -> str:
    apply_id = str(question_row["budget_charge_id"] or "")
    receipt = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE apply_id=?",
        (apply_id,),
    ).fetchone()
    if (
        not apply_id
        or receipt is None
        or str(receipt["operation"]) != "commit_waiting_user_attempt"
        or str(receipt["session_id"]) != str(authority["session_id"])
        or str(receipt["work_run_id"]) != str(authority["work_run_id"])
        or str(receipt["execution_subject_id"])
        != str(authority["execution_subject_id"])
        or str(receipt["invocation_turn_id"]) != str(question_row["turn_id"])
        or _text_hash(str(receipt["result_json"]))
        != str(receipt["result_sha256"])
    ):
        raise AuxiliaryContinuationPersistenceError(
            "question lost its exact settlement receipt"
        )
    try:
        result = AuxiliaryContinuationMutationResult.model_validate_json(
            str(receipt["result_json"])
        )
    except ValueError as exc:
        raise AuxiliaryContinuationPersistenceError(
            "question settlement receipt is corrupt"
        ) from exc
    if (
        result.operation != "commit_waiting_user_attempt"
        or result.attempt is None
        or result.attempt.attempt_id != str(question_row["attempt_id"])
        or result.question_sha256 is None
        or result.subject != _subject_from_run_row(authority)
        or result.goal_id != str(authority["goal_id"])
    ):
        raise AuxiliaryContinuationPersistenceError(
            "question settlement receipt owner is corrupt"
        )
    return result.question_sha256


def _require_answer_binding_receipt(
    conn: sqlite3.Connection,
    *,
    authority: sqlite3.Row,
    binding_row: sqlite3.Row,
    binding: AuxiliaryAnswerSourceBinding,
) -> None:
    receipt = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE apply_id=?",
        (str(binding_row["apply_id"]),),
    ).fetchone()
    if (
        receipt is None
        or str(receipt["operation"])
        != "continue_waiting_user_and_start_attempt"
        or str(receipt["session_id"]) != str(authority["session_id"])
        or str(receipt["insession_task_id"])
        != str(authority["insession_task_id"])
        or str(receipt["work_run_id"]) != str(authority["work_run_id"])
        or str(receipt["execution_subject_id"])
        != str(authority["execution_subject_id"])
        or str(receipt["invocation_turn_id"])
        != binding.answer_source_turn_id
        or _text_hash(str(receipt["result_json"]))
        != str(receipt["result_sha256"])
    ):
        raise AuxiliaryContinuationPersistenceError(
            "answer binding lost its exact continuation receipt"
        )
    try:
        result = AuxiliaryContinuationMutationResult.model_validate_json(
            str(receipt["result_json"])
        )
    except ValueError as exc:
        raise AuxiliaryContinuationPersistenceError(
            "answer continuation receipt is corrupt"
        ) from exc
    if (
        result.operation != "continue_waiting_user_and_start_attempt"
        or result.answer_source_binding != binding
        or result.subject != _subject_from_run_row(authority)
        or result.goal_id != str(authority["goal_id"])
    ):
        raise AuxiliaryContinuationPersistenceError(
            "answer continuation receipt owner is corrupt"
        )


def _typed_result(
    *,
    base: WorkExecutionMutationResult,
    operation: Literal[
        "commit_waiting_user_attempt",
        "continue_waiting_user_and_start_attempt",
        "resume_active_attempt",
    ],
    authority: sqlite3.Row,
    versions: Mapping[str, int],
    question_sha256: str | None = None,
    answer_source_binding: AuxiliaryAnswerSourceBinding | None = None,
) -> AuxiliaryContinuationMutationResult:
    subject = _subject_from_run_row(authority)
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise AuxiliaryContinuationPersistenceError(
            "result projection lost its AuxiliaryNode subject"
        )
    return AuxiliaryContinuationMutationResult.model_validate(
        base.model_dump(mode="python")
        | {
            "operation": operation,
            "execution_subject_id": str(authority["execution_subject_id"]),
            "execution_subject_contract_version": (
                _EXECUTION_SUBJECT_CONTRACT_VERSION
            ),
            "subject": subject,
            "goal_id": str(authority["goal_id"]),
            "control_state_version": versions["control"],
            "goal_state_version": versions["goal"],
            "revision_state_version": versions["revision"],
            "question_sha256": question_sha256,
            "answer_source_binding": answer_source_binding,
        }
    )


def _require_verification_resume_predecessor_receipt(
    conn: sqlite3.Connection,
    *,
    command: ResumeAuxiliaryVerificationCommand,
    authority: sqlite3.Row,
    request_row: sqlite3.Row,
    previous_turn_id: str,
) -> None:
    """认证重复恢复，同时保持请求来源不变。"""

    rows = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE operation='resume_verification' AND session_id=? "
        "AND work_run_id=? AND invocation_turn_id=?",
        (command.session_id, command.work_run_id, previous_turn_id),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryContinuationPersistenceError(
            "verification predecessor Turn lacks one exact resume receipt"
        )
    row = rows[0]
    if (
        str(row["insession_task_id"]) != command.subject.task_id
        or str(row["execution_subject_id"])
        != str(authority["execution_subject_id"])
        or str(row["auxiliary_graph_id"])
        != command.subject.auxiliary_graph_id
        or int(row["auxiliary_graph_revision"])
        != command.subject.auxiliary_graph_revision
        or str(row["auxiliary_node_id"]) != command.subject.node_id
        or int(row["node_revision"]) != command.subject.node_revision
        or str(row["goal_id"]) != str(authority["goal_id"])
        or _text_hash(str(row["result_json"])) != str(row["result_sha256"])
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification predecessor resume receipt is corrupt"
        )
    try:
        predecessor = AuxiliaryVerificationResumeResult.model_validate_json(
            str(row["result_json"])
        )
    except ValueError as exc:
        raise AuxiliaryContinuationPersistenceError(
            "verification predecessor resume receipt is invalid"
        ) from exc
    request = _auxiliary_record_from_row(
        request_row
    ).request
    link = conn.execute(
        "SELECT session_id, relation, link_revision FROM "
        "insession_work_run_turn_links WHERE turn_id=? AND work_run_id=?",
        (previous_turn_id, command.work_run_id),
    ).fetchone()
    if (
        predecessor.operation != "resume_verification"
        or predecessor.status != "applied"
        or predecessor.execution_subject_id
        != str(authority["execution_subject_id"])
        or predecessor.subject != command.subject
        or predecessor.goal_id != str(authority["goal_id"])
        or predecessor.structure_sha256 != str(authority["structure_sha256"])
        or predecessor.verification_request_id
        != command.verification_request_id
        or predecessor.verification_request_revision != request.revision
        or predecessor.verification_request_status
        is not TaskNodeVerificationRequestStatus.PENDING
        or predecessor.work_run_id != command.work_run_id
        or predecessor.work_run_revision != int(authority["revision"])
        or predecessor.work_run_status is not WorkRunStatus.ACTIVE
        or predecessor.work_run_reason != "verification_pending"
        or predecessor.request_source_turn_id != request.request_turn_id
        or predecessor.request_binding_sha256
        != str(request_row["request_binding_hash"])
        or predecessor.submitted_attempt_id != request.submitted_attempt_id
        or predecessor.output_revision != request.output_revision
        or link is None
        or str(link["session_id"]) != command.session_id
        or str(link["relation"]) != "continued"
        or int(link["link_revision"])
        != predecessor.turn_work_run_link_revision
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification predecessor resume receipt lost current authority"
        )


def _load_verification_resume_replay(
    conn: sqlite3.Connection,
    *,
    command: ResumeAuxiliaryVerificationCommand,
    payload_hash: str,
) -> AuxiliaryVerificationResumeResult | None:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE apply_id=?",
        (command.apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != "resume_verification"
        or str(row["session_id"]) != command.session_id
        or str(row["insession_task_id"]) != command.subject.task_id
        or str(row["work_run_id"]) != command.work_run_id
        or str(row["auxiliary_graph_id"])
        != command.subject.auxiliary_graph_id
        or int(row["auxiliary_graph_revision"])
        != command.subject.auxiliary_graph_revision
        or str(row["auxiliary_node_id"]) != command.subject.node_id
        or int(row["node_revision"]) != command.subject.node_revision
        or str(row["invocation_turn_id"]) != command.turn_id
        or str(row["payload_sha256"]) != payload_hash
        or _text_hash(str(row["result_json"])) != str(row["result_sha256"])
    ):
        raise AuxiliaryContinuationApplyIdCollision(
            "verification resume apply id was reused for another command"
        )
    try:
        stored = AuxiliaryVerificationResumeResult.model_validate_json(
            str(row["result_json"])
        )
    except ValueError as exc:
        raise AuxiliaryContinuationPersistenceError(
            "verification resume receipt is corrupt"
        ) from exc
    if (
        stored.operation != "resume_verification"
        or stored.verification_request_id != command.verification_request_id
        or stored.verification_request_revision
        != command.expected_verification_request_revision + 1
        or stored.verification_request_status
        is not TaskNodeVerificationRequestStatus.PENDING
        or stored.work_run_id != command.work_run_id
        or stored.work_run_revision != command.expected_work_run_revision + 1
        or stored.work_run_status is not WorkRunStatus.ACTIVE
        or stored.work_run_reason != "verification_pending"
        or stored.window_state_version != command.expected_window_revision + 1
        or stored.execution_subject_id != str(row["execution_subject_id"])
        or stored.subject != command.subject
        or stored.goal_id != str(row["goal_id"])
        or stored.structure_sha256 != command.expected_structure_sha256
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification resume receipt projection is corrupt"
        )

    authority = _load_exact_auxiliary_authority(
        conn,
        session_id=command.session_id,
        work_run_id=command.work_run_id,
    )
    _require_auxiliary_authority_shape(authority)
    if (
        _subject_from_run_row(authority) != command.subject
        or str(authority["execution_subject_id"])
        != stored.execution_subject_id
        or str(authority["goal_id"]) != stored.goal_id
        or str(authority["structure_sha256"]) != stored.structure_sha256
        or int(authority["revision"]) < stored.work_run_revision
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification resume receipt lost exact graph authority"
        )
    request_row = _load_auxiliary_request_row(
        conn,
        session_id=command.session_id,
        verification_request_id=command.verification_request_id,
    )
    request = _auxiliary_record_from_row(
        request_row
    ).request
    if (
        request.work_run_id != command.work_run_id
        or request.subject != command.subject
        or request.revision < stored.verification_request_revision
        or request.request_turn_id != stored.request_source_turn_id
        or request.submitted_attempt_id != stored.submitted_attempt_id
        or request.output_revision != stored.output_revision
        or str(request_row["request_binding_hash"])
        != stored.request_binding_sha256
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification resume receipt lost its immutable request"
        )
    try:
        _revalidate_auxiliary_request_binding(
            conn,
            request_row,
            expected_node_status=str(authority["node_status"]),
        )
    except WorkExecutionPersistenceError as exc:
        raise AuxiliaryContinuationPersistenceError(str(exc)) from exc
    link = conn.execute(
        "SELECT session_id, link_revision, relation "
        "FROM insession_work_run_turn_links WHERE turn_id=? AND work_run_id=?",
        (command.turn_id, command.work_run_id),
    ).fetchone()
    if (
        link is None
        or str(link["session_id"]) != command.session_id
        or str(link["relation"]) != "continued"
        or int(link["link_revision"]) != stored.turn_work_run_link_revision
    ):
        raise AuxiliaryContinuationPersistenceError(
            "verification resume receipt lost its Turn binding"
        )
    return stored.model_copy(update={"status": "replayed"})


def _load_replay(
    conn: sqlite3.Connection,
    *,
    command: Any,
    operation: str,
    payload_hash: str,
) -> AuxiliaryContinuationMutationResult | None:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE apply_id=?",
        (command.apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != operation
        or str(row["session_id"]) != command.session_id
        or str(row["insession_task_id"]) != command.subject.task_id
        or str(row["work_run_id"]) != command.work_run_id
        or str(row["auxiliary_graph_id"])
        != command.subject.auxiliary_graph_id
        or int(row["auxiliary_graph_revision"])
        != command.subject.auxiliary_graph_revision
        or str(row["auxiliary_node_id"]) != command.subject.node_id
        or int(row["node_revision"]) != command.subject.node_revision
        or str(row["invocation_turn_id"]) != command.turn_id
        or str(row["payload_sha256"]) != payload_hash
        or _text_hash(str(row["result_json"])) != str(row["result_sha256"])
    ):
        raise AuxiliaryContinuationApplyIdCollision(
            "continuation apply id was reused for another command"
        )
    try:
        stored = AuxiliaryContinuationMutationResult.model_validate_json(
            str(row["result_json"])
        )
    except ValueError as exc:
        raise AuxiliaryContinuationPersistenceError(
            "continuation apply receipt is corrupt"
        ) from exc
    if (
        stored.operation != operation
        or stored.work_run_id != command.work_run_id
        or stored.execution_subject_id != str(row["execution_subject_id"])
        or stored.goal_id != str(row["goal_id"])
        or stored.subject != command.subject
    ):
        raise AuxiliaryContinuationPersistenceError(
            "continuation receipt owner is corrupt"
        )
    if operation == "commit_waiting_user_attempt":
        if stored.budget_transition is None:
            raise AuxiliaryContinuationPersistenceError(
                "question receipt lost its budget transition"
            )
        hard_limit_reached = (
            stored.budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        )
        expected_status = "failed" if hard_limit_reached else "waiting_user"
        expected_reason = (
            "work_run_limit_reached" if hard_limit_reached else "needs_input"
        )
        if (
            stored.work_run_status.value != expected_status
            or stored.work_run_reason != expected_reason
            or stored.question_sha256
            != _text_hash(command.decision.action.question)
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question receipt disagrees with its budget disposition"
            )
        charge = conn.execute(
            "SELECT operation, session_id, work_run_id, turn_id, "
            "work_run_revision_after, window_state_version_after, "
            "work_run_status_after, work_run_reason_after "
            "FROM insession_work_run_budget_charges "
            "WHERE budget_charge_id=?",
            (command.apply_id,),
        ).fetchone()
        if (
            charge is None
            or str(charge["operation"]) != "commit_attempt_decision"
            or str(charge["session_id"]) != command.session_id
            or str(charge["work_run_id"]) != command.work_run_id
            or str(charge["turn_id"]) != command.turn_id
            or int(charge["work_run_revision_after"])
            != stored.work_run_revision
            or int(charge["window_state_version_after"])
            != stored.window_state_version
            or str(charge["work_run_status_after"]) != expected_status
            or str(charge["work_run_reason_after"] or "") != expected_reason
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question receipt lost its budget charge"
            )
        # 重新运行通用 WorkRun 聚合认证器。它会证明 receipt 所扩展的配套通用
        # receipt、active-time 转换、Attempt 链接和连续 charge 账本。
        _load_record(
            conn,
            command.work_run_id,
            session_id=command.session_id,
        )
        authority = _load_exact_auxiliary_authority(
            conn,
            session_id=command.session_id,
            work_run_id=command.work_run_id,
        )
        question_row = conn.execute(
            "SELECT attempt_id, turn_id, budget_charge_id "
            "FROM insession_work_run_attempts WHERE work_run_id=? "
            "AND attempt_id=? AND status='closed' "
            "AND action='request_user_input'",
            (command.work_run_id, command.attempt_id),
        ).fetchone()
        if (
            question_row is None
            or stored.question_sha256 is None
            or _require_question_settlement_sha256(
                conn,
                authority=authority,
                question_row=question_row,
            )
            != stored.question_sha256
        ):
            raise AuxiliaryContinuationPersistenceError(
                "question receipt lost its settled Attempt"
            )
    elif operation == "continue_waiting_user_and_start_attempt":
        run_row = _require_run_row(conn, command.work_run_id, command.session_id)
        validate_auxiliary_answer_bindings_for_work_run(
            conn,
            run_row=run_row,
        )
        binding = conn.execute(
            "SELECT * FROM insession_auxiliary_v2_waiting_user_answer_bindings "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
        if (
            binding is None
            or stored.answer_source_binding != _answer_binding_from_row(binding)
        ):
            raise AuxiliaryContinuationPersistenceError(
                "continuation receipt lost its answer binding"
            )
    else:
        link = conn.execute(
            "SELECT link_revision, relation FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id=?",
            (command.turn_id, command.work_run_id),
        ).fetchone()
        attempt_owner = conn.execute(
            "SELECT turn_id, action, decision_json, catalog_snapshot_json, "
            "catalog_snapshot_hash FROM insession_work_run_attempts "
            "WHERE work_run_id=? AND attempt_id=?",
            (command.work_run_id, command.attempt_id),
        ).fetchone()
        decided_recovery = command.catalog_snapshot is not None
        expected_catalog_json = (
            None
            if command.catalog_snapshot is None
            else _canonical_json(command.catalog_snapshot)
        )
        if (
            link is None
            or str(link["relation"]) != "continued"
            or stored.turn_work_run_link_revision != int(link["link_revision"])
            or attempt_owner is None
            or str(attempt_owner["turn_id"]) != command.turn_id
            or (
                decided_recovery
                and (
                    str(attempt_owner["action"] or "") != "call_tools"
                    or attempt_owner["decision_json"] is None
                    or str(attempt_owner["catalog_snapshot_json"])
                    != expected_catalog_json
                    or _text_hash(str(attempt_owner["catalog_snapshot_json"]))
                    != str(attempt_owner["catalog_snapshot_hash"])
                )
            )
        ):
            raise AuxiliaryContinuationPersistenceError(
                "Attempt resume receipt lost its Turn binding"
            )
        _load_record(
            conn,
            command.work_run_id,
            session_id=command.session_id,
        )
    return stored.model_copy(update={"status": "replayed"})


def _require_apply_id_available(conn: sqlite3.Connection, apply_id: str) -> None:
    tables = (
        "insession_work_run_apply_receipts",
        "insession_auxiliary_graph_revision_apply_receipts_v2",
    )
    for table in tables:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE apply_id=?",
            (apply_id,),
        ).fetchone() is not None:
            raise AuxiliaryContinuationApplyIdCollision(
                "continuation apply id is already owned by another ledger"
            )


def _insert_receipt(
    conn: sqlite3.Connection,
    *,
    command: Any,
    operation: str,
    authority: sqlite3.Row,
    payload_hash: str,
    result: (
        AuxiliaryContinuationMutationResult
        | AuxiliaryVerificationResumeResult
    ),
    now: str,
) -> None:
    result_json = _model_json(result)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_execution_apply_receipts "
        "(apply_id, operation, session_id, insession_task_id, work_run_id, "
        "execution_subject_id, auxiliary_graph_id, goal_id, "
        "auxiliary_graph_revision, auxiliary_node_id, node_revision, "
        "invocation_turn_id, payload_sha256, result_json, result_sha256, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.apply_id,
            operation,
            command.session_id,
            command.subject.task_id,
            command.work_run_id,
            str(authority["execution_subject_id"]),
            command.subject.auxiliary_graph_id,
            str(authority["goal_id"]),
            command.subject.auxiliary_graph_revision,
            command.subject.node_id,
            command.subject.node_revision,
            command.turn_id,
            payload_hash,
            result_json,
            _text_hash(result_json),
            now,
        ),
    )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(
            f"{name} must be a non-empty identifier of at most 200 characters"
        )


__all__ = [
    "AuxiliaryAnswerSourceBinding",
    "AuxiliaryContinuationApplyIdCollision",
    "AuxiliaryContinuationMutationResult",
    "AuxiliaryContinuationPersistenceError",
    "AuxiliaryPendingUserQuestion",
    "AuxiliaryVerificationResumeResult",
    "CommitAuxiliaryWaitingUserAttemptCommand",
    "ContinueAuxiliaryWaitingUserCommand",
    "ResumeAuxiliaryActiveAttemptCommand",
    "ResumeAuxiliaryVerificationCommand",
    "commit_auxiliary_waiting_user_attempt",
    "continue_auxiliary_waiting_user_and_start_attempt",
    "get_auxiliary_pending_user_question",
    "resume_auxiliary_active_attempt",
    "resume_auxiliary_verification",
    "validate_auxiliary_answer_bindings_for_work_run",
]
