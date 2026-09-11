"""L2 waiting-user and recovery continuation persistence facade.

Each operation resolves the current Session route and delegates one complete
transaction to its existing persistence owner.  Combined pending-question
projection preserves the established generic-first, Auxiliary-specific-second order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from personagraph.l2.work_run import (
    PendingUserQuestion,
    TaskNodeSubject,
    WorkExecutionMutationResult,
)

from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_continuation as auxiliary_records
from ..persistence.l2.work_run import work_execution as work_records


AuxiliaryContinuationPersistenceError = (
    auxiliary_records.AuxiliaryContinuationPersistenceError
)
AuxiliaryContinuationApplyIdCollision = (
    auxiliary_records.AuxiliaryContinuationApplyIdCollision
)
AuxiliaryPendingUserQuestion = auxiliary_records.AuxiliaryPendingUserQuestion
CommitAuxiliaryWaitingUserAttemptCommand = (
    auxiliary_records.CommitAuxiliaryWaitingUserAttemptCommand
)
ContinueAuxiliaryWaitingUserCommand = (
    auxiliary_records.ContinueAuxiliaryWaitingUserCommand
)
ResumeAuxiliaryActiveAttemptCommand = (
    auxiliary_records.ResumeAuxiliaryActiveAttemptCommand
)
ResumeAuxiliaryVerificationCommand = (
    auxiliary_records.ResumeAuxiliaryVerificationCommand
)
AuxiliaryContinuationMutationResult = (
    auxiliary_records.AuxiliaryContinuationMutationResult
)
AuxiliaryVerificationResumeResult = (
    auxiliary_records.AuxiliaryVerificationResumeResult
)


def continue_waiting_user_work_run_and_start_attempt(
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
    return work_records.continue_waiting_user_work_run_and_start_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        subject=subject,
        question_attempt_id=question_attempt_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        catalog_snapshot=catalog_snapshot,
        attempt_id=attempt_id,
    )


def get_auxiliary_pending_user_question(
    *,
    session_id: str,
    insession_task_id: str,
) -> AuxiliaryPendingUserQuestion | None:
    return auxiliary_records.get_auxiliary_pending_user_question(
        session_store.current_store_deps(),
        session_id=session_id,
        insession_task_id=insession_task_id,
    )


def commit_auxiliary_waiting_user_attempt(
    *,
    command: CommitAuxiliaryWaitingUserAttemptCommand,
) -> AuxiliaryContinuationMutationResult:
    return auxiliary_records.commit_auxiliary_waiting_user_attempt(
        session_store.current_store_deps(),
        command=command,
    )


def continue_auxiliary_waiting_user_and_start_attempt(
    *,
    command: ContinueAuxiliaryWaitingUserCommand,
) -> AuxiliaryContinuationMutationResult:
    return auxiliary_records.continue_auxiliary_waiting_user_and_start_attempt(
        session_store.current_store_deps(),
        command=command,
    )


def resume_auxiliary_active_attempt(
    *,
    command: ResumeAuxiliaryActiveAttemptCommand,
) -> AuxiliaryContinuationMutationResult:
    return auxiliary_records.resume_auxiliary_active_attempt(
        session_store.current_store_deps(),
        command=command,
    )


def resume_auxiliary_verification(
    *,
    command: ResumeAuxiliaryVerificationCommand,
) -> AuxiliaryVerificationResumeResult:
    return auxiliary_records.resume_auxiliary_verification(
        session_store.current_store_deps(),
        command=command,
    )


def list_pending_user_questions(
    *,
    session_id: str,
) -> tuple[PendingUserQuestion | AuxiliaryPendingUserQuestion, ...]:
    generic = work_records.list_pending_user_questions(
        session_store.current_store_deps(),
        session_id=session_id,
    )
    auxiliary_questions = auxiliary_records.list_auxiliary_pending_user_questions(
        session_store.current_store_deps(),
        session_id=session_id,
    )
    return (*generic, *auxiliary_questions)


__all__ = [
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
    "continue_waiting_user_work_run_and_start_attempt",
    "get_auxiliary_pending_user_question",
    "list_pending_user_questions",
    "resume_auxiliary_active_attempt",
    "resume_auxiliary_verification",
]
