"""完整 Task 交付验证权威的窄持久化门面。

此端口向交付验证记录传递显式 :class:`StoreDeps`，只公开候选结算读取、子 Delivery
投影与活动图 revision 触发器生命周期。候选结算的事务内写入由 NodeVerification
持久化 owner 直接组合；旧的 FinishGate 后独立 validation settlement 已硬切。
"""

from __future__ import annotations

from . import task_delivery_validation
from ...deps import StoreDeps


TaskDeliveryValidationPersistenceError = (
    task_delivery_validation.TaskDeliveryValidationPersistenceError
)
TaskDeliveryValidationIdentityCollision = (
    task_delivery_validation.TaskDeliveryValidationIdentityCollision
)
TaskDeliveryValidationStaleAuthority = (
    task_delivery_validation.TaskDeliveryValidationStaleAuthority
)
TaskDeliveryValidationStoredAuthorityCorrupt = (
    task_delivery_validation.TaskDeliveryValidationStoredAuthorityCorrupt
)
ConsumeTaskGraphRevisionTriggerCommand = (
    task_delivery_validation.ConsumeTaskGraphRevisionTriggerCommand
)
TaskDeliveryCandidateSettlementIntent = (
    task_delivery_validation.TaskDeliveryCandidateSettlementIntent
)
TaskDeliveryCandidateSettlement = (
    task_delivery_validation.TaskDeliveryCandidateSettlement
)
TaskDeliveryCandidateSettlementMutationResult = (
    task_delivery_validation.TaskDeliveryCandidateSettlementMutationResult
)
StoredTaskDeliveryCandidateSettlement = (
    task_delivery_validation.StoredTaskDeliveryCandidateSettlement
)
TaskGraphRevisionTriggerApplicationMutationResult = (
    task_delivery_validation.TaskGraphRevisionTriggerApplicationMutationResult
)


def project_task_delivery_validation_child_deliveries(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
):
    return task_delivery_validation.project_task_delivery_validation_child_deliveries(
        deps,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
    )


def get_task_delivery_candidate_settlement(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> StoredTaskDeliveryCandidateSettlement | None:
    return task_delivery_validation.get_task_delivery_candidate_settlement(
        deps,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
    )


def get_active_task_graph_revision_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
):
    return task_delivery_validation.get_active_task_graph_revision_trigger(
        deps,
        session_id=session_id,
        task_id=task_id,
    )


def consume_task_graph_revision_trigger(
    deps: StoreDeps,
    *,
    command: ConsumeTaskGraphRevisionTriggerCommand,
) -> TaskGraphRevisionTriggerApplicationMutationResult:
    return task_delivery_validation.consume_task_graph_revision_trigger(
        deps,
        command=command,
    )
