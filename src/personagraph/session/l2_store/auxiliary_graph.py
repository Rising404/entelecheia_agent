"""L2 AuxiliaryGraph persistence facade.

Graph commit, projection, and dependency resolution remain owned by their
existing persistence modules. This facade only resolves the current Session
database route for each complete operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from personagraph.l2 import auxiliary_graph as graph_contracts
from personagraph.l2.auxiliary_graph import dependency_projection
from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionValidationContext,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject

from . import planning as planning_store
from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_graphs as graph_records
from ..persistence.l2.auxiliary_graph import auxiliary_dependencies as dependency_records


AuxiliaryGraphPersistenceError = graph_records.AuxiliaryGraphPersistenceError
AuxiliaryGraphApplyIdCollision = graph_records.AuxiliaryGraphApplyIdCollision
AuxiliaryGraphNodeProposalRecord = graph_records.AuxiliaryGraphNodeProposalRecord
AuxiliaryGraphEdgeProposalRecord = graph_records.AuxiliaryGraphEdgeProposalRecord
AuxiliaryGraphRevisionProposalRecord = (
    graph_records.AuxiliaryGraphRevisionProposalRecord
)
AuxiliaryGraphRevisionCommitResult = (
    graph_records.AuxiliaryGraphRevisionCommitResult
)
StoredAuxiliaryGraphDetails = graph_records.StoredAuxiliaryGraphDetails
AuxiliaryDependencyPersistenceError = (
    dependency_records.AuxiliaryDependencyPersistenceError
)


def commit_auxiliary_graph_revision(
    *,
    session_id: str,
    turn_id: str,
    insession_task_id: str,
    expected_task_state_version: int,
    expected_base_task_graph_revision: int | None,
    expected_control_state_version: int | None,
    expected_current_auxiliary_graph_revision: int | None,
    apply_id: str,
    goal_objective: str,
    proposal: AuxiliaryGraphRevisionProposalRecord,
    authority_context: Mapping[str, Any],
    budget_profile: Mapping[str, Any],
    auxiliary_graph_id: str | None = None,
    goal_id: str | None = None,
    initial_planning_completion: (
        planning_store.AuxiliaryInitialPlanningCompletionBinding | None
    ) = None,
    positive_planning_completion: (
        planning_store.AuxiliaryPositivePlanningCompletionBinding | None
    ) = None,
    terminal_candidate_semantic_settlement_id: str | None = None,
) -> AuxiliaryGraphRevisionCommitResult:
    return graph_records.commit_auxiliary_graph_revision(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=insession_task_id,
        expected_task_state_version=expected_task_state_version,
        expected_base_task_graph_revision=expected_base_task_graph_revision,
        expected_control_state_version=expected_control_state_version,
        expected_current_auxiliary_graph_revision=(
            expected_current_auxiliary_graph_revision
        ),
        apply_id=apply_id,
        goal_objective=goal_objective,
        proposal=proposal,
        authority_context=authority_context,
        budget_profile=budget_profile,
        auxiliary_graph_id=auxiliary_graph_id,
        goal_id=goal_id,
        initial_planning_completion=initial_planning_completion,
        positive_planning_completion=positive_planning_completion,
        terminal_candidate_semantic_settlement_id=(
            terminal_candidate_semantic_settlement_id
        ),
    )


def get_auxiliary_graph_for_task(
    *,
    session_id: str,
    insession_task_id: str,
) -> StoredAuxiliaryGraphDetails | None:
    return graph_records.get_auxiliary_graph_for_task(
        session_store.current_store_deps(),
        session_id=session_id,
        insession_task_id=insession_task_id,
    )


def get_current_auxiliary_graph_revision(
    *,
    session_id: str,
    insession_task_id: str,
) -> graph_contracts.AuxiliaryGraphRevision | None:
    return graph_records.get_current_auxiliary_graph_revision(
        session_store.current_store_deps(),
        session_id=session_id,
        insession_task_id=insession_task_id,
    )


def project_auxiliary_graph_execution_frontier(
    *,
    session_id: str,
    turn_id: str,
    insession_task_id: str,
) -> graph_contracts.AuxiliaryGraphExecutionFrontier:
    return graph_records.project_auxiliary_graph_execution_frontier(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=insession_task_id,
    )


def resolve_auxiliary_dependencies(
    *,
    session_id: str,
    turn_id: str,
    consumer_subject: AuxiliaryNodeSubject,
) -> dependency_projection.AuxiliaryDependencyBundle:
    return dependency_records.resolve_auxiliary_dependencies(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=consumer_subject,
    )


def require_auxiliary_graph_commit_context_valid(
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    context: InSessionTaskGraphRevisionValidationContext,
) -> None:
    graph_records.require_auxiliary_graph_commit_context_valid(
        session_store.current_store_deps(),
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
        context=context,
    )


__all__ = [
    "AuxiliaryGraphApplyIdCollision",
    "AuxiliaryGraphPersistenceError",
    "AuxiliaryGraphEdgeProposalRecord",
    "AuxiliaryGraphNodeProposalRecord",
    "AuxiliaryGraphRevisionCommitResult",
    "AuxiliaryGraphRevisionProposalRecord",
    "AuxiliaryDependencyPersistenceError",
    "StoredAuxiliaryGraphDetails",
    "commit_auxiliary_graph_revision",
    "get_auxiliary_graph_for_task",
    "get_current_auxiliary_graph_revision",
    "project_auxiliary_graph_execution_frontier",
    "require_auxiliary_graph_commit_context_valid",
    "resolve_auxiliary_dependencies",
]
