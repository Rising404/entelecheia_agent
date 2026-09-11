"""L2 node-verification persistence facade.

Each operation resolves the current Session route and delegates one complete
transaction to its existing persistence owner.
"""

from __future__ import annotations

from typing import Literal

from personagraph.l2.work_run import (
    NodeVerificationResult,
    PreparedTaskNodeVerification,
    ResolvedTaskNodeDelivery,
    TaskNodeVerificationMutationResult,
    TaskNodeVerificationRecord,
)

from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_graphs as auxiliary_graph_records
from ..persistence.l2.work_run import work_verification as verification_records
from .task_delivery import TaskDeliveryCandidateSettlementIntent


TaskNodeVerificationRequestRevisionConflict = (
    verification_records.TaskNodeVerificationRequestRevisionConflict
)
VerificationSettlementReceiptNotFound = (
    verification_records.VerificationSettlementReceiptNotFound
)


def prepare_task_node_verification(
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
    return verification_records.prepare_task_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_output_revision=expected_output_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        verification_request_id=verification_request_id,
        recover_unprepared_submit=recover_unprepared_submit,
    )


def get_task_node_verification_record(
    *,
    session_id: str,
    verification_request_id: str,
) -> TaskNodeVerificationRecord:
    return verification_records.get_task_node_verification_record(
        session_store.current_store_deps(),
        session_id=session_id,
        verification_request_id=verification_request_id,
    )


def get_prepared_task_node_verification(
    *,
    session_id: str,
    invocation_turn_id: str,
    verification_request_id: str,
) -> PreparedTaskNodeVerification:
    return verification_records.get_prepared_task_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        verification_request_id=verification_request_id,
    )


def get_task_node_delivery(
    *,
    session_id: str,
    delivery_id: str,
) -> ResolvedTaskNodeDelivery:
    return verification_records.get_task_node_delivery(
        session_store.current_store_deps(),
        session_id=session_id,
        delivery_id=delivery_id,
    )


def replay_task_node_verification_settlement(
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
    return verification_records.replay_task_node_verification_settlement(
        session_store.current_store_deps(),
        session_id=session_id,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        apply_id=apply_id,
        operation=operation,
    )


def commit_task_node_verification_result(
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
        TaskDeliveryCandidateSettlementIntent | None
    ) = None,
) -> TaskNodeVerificationMutationResult:
    return verification_records.commit_task_node_verification_result(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        result=result,
        expected_work_run_revision=expected_work_run_revision,
        expected_verification_request_revision=(
            expected_verification_request_revision
        ),
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        delivery_id=delivery_id,
        active_seconds_delta=active_seconds_delta,
        task_delivery_candidate_settlement=task_delivery_candidate_settlement,
    )


def interrupt_task_node_verification(
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
    return verification_records.interrupt_task_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        technical_error_code=technical_error_code,
        expected_work_run_revision=expected_work_run_revision,
        expected_verification_request_revision=(
            expected_verification_request_revision
        ),
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        active_seconds_delta=active_seconds_delta,
    )


def resume_task_node_verification(
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
    return verification_records.resume_task_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_verification_request_revision=(
            expected_verification_request_revision
        ),
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
    )


def prepare_auxiliary_node_verification(
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
) -> TaskNodeVerificationMutationResult:
    return auxiliary_graph_records.prepare_auxiliary_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_output_revision=expected_output_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        verification_request_id=verification_request_id,
    )


def get_prepared_auxiliary_node_verification(
    *,
    session_id: str,
    invocation_turn_id: str,
    verification_request_id: str,
) -> PreparedTaskNodeVerification:
    return auxiliary_graph_records.get_prepared_auxiliary_node_verification(
        session_store.current_store_deps(),
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        verification_request_id=verification_request_id,
    )


def commit_auxiliary_node_verification_result(
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
    active_seconds_delta: float,
    completion_id: str | None = None,
) -> TaskNodeVerificationMutationResult:
    return auxiliary_graph_records.commit_auxiliary_node_verification_result(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        verification_request_id=verification_request_id,
        result=result,
        expected_work_run_revision=expected_work_run_revision,
        expected_verification_request_revision=(
            expected_verification_request_revision
        ),
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        active_seconds_delta=active_seconds_delta,
        completion_id=completion_id,
    )


__all__ = [
    "TaskNodeVerificationRequestRevisionConflict",
    "VerificationSettlementReceiptNotFound",
    "commit_auxiliary_node_verification_result",
    "commit_task_node_verification_result",
    "get_prepared_auxiliary_node_verification",
    "get_prepared_task_node_verification",
    "get_task_node_delivery",
    "get_task_node_verification_record",
    "interrupt_task_node_verification",
    "prepare_auxiliary_node_verification",
    "prepare_task_node_verification",
    "replay_task_node_verification_settlement",
    "resume_task_node_verification",
]
