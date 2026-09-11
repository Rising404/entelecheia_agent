"""L2 Task Delivery candidate-settlement persistence facade.

The current runtime settles delivery candidates inside NodeVerification.  This
facade only resolves the Session database route for candidate replay, child
delivery projection, and revision-trigger reads.
"""

from __future__ import annotations

from .. import store as session_store
from ..persistence.l2.delivery import task_delivery_validation_port as delivery_records


TaskDeliveryValidationPersistenceError = (
    delivery_records.TaskDeliveryValidationPersistenceError
)
TaskDeliveryValidationStoredAuthorityCorrupt = (
    delivery_records.TaskDeliveryValidationStoredAuthorityCorrupt
)
TaskDeliveryCandidateSettlementIntent = (
    delivery_records.TaskDeliveryCandidateSettlementIntent
)
StoredTaskDeliveryCandidateSettlement = (
    delivery_records.StoredTaskDeliveryCandidateSettlement
)


def project_task_delivery_validation_child_deliveries(
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
):
    return delivery_records.project_task_delivery_validation_child_deliveries(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
    )


def get_task_delivery_candidate_settlement(
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> StoredTaskDeliveryCandidateSettlement | None:
    return delivery_records.get_task_delivery_candidate_settlement(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
    )


def get_active_task_graph_revision_trigger(
    *,
    session_id: str,
    task_id: str,
):
    return delivery_records.get_active_task_graph_revision_trigger(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
    )


__all__ = [
    "StoredTaskDeliveryCandidateSettlement",
    "TaskDeliveryCandidateSettlementIntent",
    "TaskDeliveryValidationPersistenceError",
    "TaskDeliveryValidationStoredAuthorityCorrupt",
    "get_active_task_graph_revision_trigger",
    "get_task_delivery_candidate_settlement",
    "project_task_delivery_validation_child_deliveries",
]
