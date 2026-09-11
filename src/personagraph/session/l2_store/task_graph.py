"""L2 TaskGraph persistence facade.

Task matching and rich TaskGraph projections remain owned by their existing
persistence modules.  This facade only resolves the current Session database
route for each complete operation.
"""

from __future__ import annotations

from personagraph.l2.task_graph.contracts import (
    InSessionTaskDetails,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchApplyResult,
    InSessionTaskMatchesProposal,
    InSessionTaskMatchingLimits,
)

from .. import store as session_store
from ..insession_task_contracts import (
    InSessionTaskApplyIdCollision,
    InSessionTaskPersistenceError,
)
from ..persistence.l2.task_graph import insession_tasks as task_records
from ..persistence.l2.task_graph import task_graph_semantic_base as semantic_base_records


TaskGraphSemanticBaseProjectionError = (
    semantic_base_records.TaskGraphSemanticBaseProjectionError
)
TaskGraphSemanticBaseProjection = (
    semantic_base_records.TaskGraphSemanticBaseProjection
)


def apply_insession_task_matches(
    *,
    session_id: str,
    source_turn_id: str,
    apply_id: str,
    proposal: InSessionTaskMatchesProposal,
    exposed_catalog_ids: tuple[str, ...],
    expected_window_revision: int,
    limits: InSessionTaskMatchingLimits | None = None,
) -> InSessionTaskMatchApplyResult:
    """Atomically validate and persist one L2 entry Task-match proposal."""

    return task_records.apply_insession_task_matches(
        session_store.current_store_deps(),
        session_id=session_id,
        source_turn_id=source_turn_id,
        apply_id=apply_id,
        proposal=proposal,
        exposed_catalog_ids=exposed_catalog_ids,
        expected_window_revision=expected_window_revision,
        limits=limits,
    )


def get_insession_task_execution_lane_manifest(
    *,
    session_id: str,
    turn_id: str,
) -> InSessionTaskExecutionLaneManifest:
    return task_records.get_insession_task_execution_lane_manifest(
        session_store.current_store_deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def get_insession_task_creation_source(
    *,
    session_id: str,
    insession_task_id: str,
) -> InSessionTaskSourceAnchor:
    return task_records.get_insession_task_creation_source(
        session_store.current_store_deps(),
        session_id=session_id,
        insession_task_id=insession_task_id,
    )


def get_insession_task_details(
    session_id: str,
    insession_task_id: str,
) -> InSessionTaskDetails | None:
    return task_records.get_insession_task_details(
        session_store.current_store_deps(),
        session_id,
        insession_task_id,
    )


def project_task_graph_semantic_base(
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> TaskGraphSemanticBaseProjection:
    return semantic_base_records.project_task_graph_semantic_base(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
    )


__all__ = [
    "InSessionTaskApplyIdCollision",
    'InSessionTaskDetails',
    'InSessionTaskExecutionLaneManifest',
    'InSessionTaskMatchApplyResult',
    'InSessionTaskMatchesProposal',
    'InSessionTaskMatchingLimits',
    "InSessionTaskPersistenceError",
    'InSessionTaskSourceAnchor',
    "TaskGraphSemanticBaseProjectionError",
    'TaskGraphSemanticBaseProjection',
    "apply_insession_task_matches",
    "get_insession_task_creation_source",
    "get_insession_task_details",
    "get_insession_task_execution_lane_manifest",
    "project_task_graph_semantic_base",
]
