"""L2 WorkRun persistence facade.

Each operation resolves the current Session route and delegates one complete
transaction to its existing persistence owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from personagraph.l2.work_run import (
    AttemptDecision,
    AuxiliaryNodeSubject,
    HostAcceptedAttemptDecision,
    ResolvedCurrentTaskNodeDelivery,
    TaskGraphExecutionReplanRequest,
    TaskNodeSubject,
    ToolResult,
    WorkExecutionMutationResult,
)

from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_graphs as graph_records
from ..persistence.l2.work_run import work_execution as work_records


WorkExecutionPersistenceError = work_records.WorkExecutionPersistenceError
WorkExecutionApplyIdCollision = work_records.WorkExecutionApplyIdCollision
WorkExecutionRevisionConflict = work_records.WorkExecutionRevisionConflict
WorkExecutionOutputRevisionConflict = (
    work_records.WorkExecutionOutputRevisionConflict
)
StoredAttempt = work_records.StoredAttempt
StoredWorkRun = work_records.StoredWorkRun
TurnLinkedNonterminalWorkRunCandidate = (
    work_records.TurnLinkedNonterminalWorkRunCandidate
)


def create_task_node_work_run(
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
    return work_records.create_task_node_work_run(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        work_run_id=work_run_id,
    )


def create_auxiliary_node_work_run(
    *,
    session_id: str,
    turn_id: str,
    subject: AuxiliaryNodeSubject,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_window_revision: int,
    apply_id: str,
    work_run_id: str | None = None,
) -> WorkExecutionMutationResult:
    return graph_records.create_auxiliary_node_work_run(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        work_run_id=work_run_id,
    )


def get_work_run(*, session_id: str, work_run_id: str) -> StoredWorkRun:
    return work_records.get_work_run(
        session_store.current_store_deps(),
        session_id=session_id,
        work_run_id=work_run_id,
    )


def get_active_task_graph_execution_replan_request(
    *,
    session_id: str,
    task_id: str,
) -> TaskGraphExecutionReplanRequest | None:
    return work_records.get_active_task_graph_execution_replan_request(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
    )


def list_turn_linked_work_run_ids(
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    return work_records.list_turn_linked_work_run_ids(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def list_turn_linked_nonterminal_work_runs(
    *,
    session_id: str,
    turn_id: str,
) -> tuple[TurnLinkedNonterminalWorkRunCandidate, ...]:
    return work_records.list_turn_linked_nonterminal_work_runs(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def project_task_node_execution_frontier(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> work_records.TaskNodeExecutionFrontier:
    return work_records.project_task_node_execution_frontier(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )


def get_current_task_node_dependency_deliveries(
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> tuple[ResolvedCurrentTaskNodeDelivery, ...]:
    return work_records.get_current_task_node_dependency_deliveries(
        session_store.current_store_deps(),
        session_id=session_id,
        subject=subject,
    )


def get_completed_task_final_delivery_id(
    *,
    session_id: str,
    task_id: str,
) -> str:
    return work_records.get_completed_task_final_delivery_id(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
    )


def list_turn_completed_verified_delivery_ids(
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    return work_records.list_turn_completed_verified_delivery_ids(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def detach_safe_work_run_lane(
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_window_revision: int,
    apply_id: str,
) -> WorkExecutionMutationResult:
    return work_records.detach_safe_work_run_lane(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
    )


def resume_active_work_run_attempt(
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    attempt_id: str,
    expected_work_run_revision: int,
    expected_window_revision: int,
    apply_id: str,
) -> WorkExecutionMutationResult:
    return work_records.resume_active_work_run_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
    )


def resume_decided_readonly_tool_attempt(
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
    return work_records.resume_decided_readonly_tool_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_window_revision=expected_window_revision,
        catalog_snapshot=catalog_snapshot,
        apply_id=apply_id,
        allow_protected_recovery=allow_protected_recovery,
    )


def resume_idle_readonly_work_run_and_start_attempt(
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
    return work_records.resume_idle_readonly_work_run_and_start_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        closed_attempt_id=closed_attempt_id,
        next_attempt_id=next_attempt_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_window_revision=expected_window_revision,
        catalog_snapshot=catalog_snapshot,
        apply_id=apply_id,
        allow_protected_recovery=allow_protected_recovery,
    )


def start_work_run_attempt(
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
    return work_records.start_work_run_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        input_checkpoint_id=input_checkpoint_id,
        catalog_snapshot=catalog_snapshot,
        attempt_id=attempt_id,
    )


def commit_work_run_attempt_decision(
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
    return work_records.commit_work_run_attempt_decision(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        decision=decision,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        active_seconds_delta=active_seconds_delta,
    )


def commit_work_run_output_action(
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
    return work_records.commit_work_run_output_action(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        decision=decision,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_output_revision=expected_output_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        active_seconds_delta=active_seconds_delta,
    )


def append_work_run_tool_result(
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
    return work_records.append_work_run_tool_result(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        result=result,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
    )


def close_work_run_attempt(
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
    return work_records.close_work_run_attempt(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=attempt_id,
        expected_work_run_revision=expected_work_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=expected_window_revision,
        apply_id=apply_id,
        close_reason=close_reason,
        active_seconds_delta=active_seconds_delta,
    )


__all__ = [
    'StoredAttempt',
    'StoredWorkRun',
    'TurnLinkedNonterminalWorkRunCandidate',
    "WorkExecutionApplyIdCollision",
    "WorkExecutionOutputRevisionConflict",
    "WorkExecutionPersistenceError",
    "WorkExecutionRevisionConflict",
    "append_work_run_tool_result",
    "close_work_run_attempt",
    "commit_work_run_attempt_decision",
    "commit_work_run_output_action",
    "create_auxiliary_node_work_run",
    "create_task_node_work_run",
    "detach_safe_work_run_lane",
    "get_active_task_graph_execution_replan_request",
    "get_completed_task_final_delivery_id",
    "get_current_task_node_dependency_deliveries",
    "get_work_run",
    "list_turn_completed_verified_delivery_ids",
    "list_turn_linked_nonterminal_work_runs",
    "list_turn_linked_work_run_ids",
    "project_task_node_execution_frontier",
    "resume_active_work_run_attempt",
    "resume_decided_readonly_tool_attempt",
    "resume_idle_readonly_work_run_and_start_attempt",
    "start_work_run_attempt",
]
