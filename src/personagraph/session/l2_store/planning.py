"""L2 planning persistence facade.

This module owns the narrow goal, primitive, initial-planning, and replan-trigger
surfaces used by L2. Each operation resolves fresh Session store dependencies so
database routing remains dynamic while transactions stay with persistence owners.
"""

from __future__ import annotations

from personagraph.l2.auxiliary_graph import (
    AuxiliaryPlanningGoal,
    AuxiliaryReplanTriggerApplicationReceipt,
    AuxiliaryReplanTriggerReceipt,
)
from personagraph.l2.planning.invocation_contracts import (
    FrozenPlanningContextPrimitiveInvocation,
)

from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_graphs as graph_records
from ..persistence.l2.planning import auxiliary_goal_supersede_port as supersede_records
from ..persistence.l2.planning import auxiliary_planning_completion_port as initial_records
from ..persistence.l2.planning import auxiliary_replan_trigger_port as replan_records
from ..persistence.l2.planning import artifact_seals as artifact_records
from ..persistence.l2.planning import primitive_invocations as primitive_records


AuxiliaryGoalSupersedePersistenceError = (
    supersede_records.AuxiliaryGoalSupersedePersistenceError
)
AuxiliaryGoalSupersedeIdentityCollision = (
    supersede_records.AuxiliaryGoalSupersedeIdentityCollision
)
AuxiliaryGoalSupersedeStaleAuthority = (
    supersede_records.AuxiliaryGoalSupersedeStaleAuthority
)
AuxiliaryGoalSupersedeUnsafeWorkRun = (
    supersede_records.AuxiliaryGoalSupersedeUnsafeWorkRun
)
PlanningGoalSupersedeReason = supersede_records.PlanningGoalSupersedeReason
PlanningGoalSupersedeReceipt = supersede_records.PlanningGoalSupersedeReceipt
SupersedeAuxiliaryPlanningGoalCommand = (
    supersede_records.SupersedeAuxiliaryPlanningGoalCommand
)
SupersedeAuxiliaryPlanningGoalResult = (
    supersede_records.SupersedeAuxiliaryPlanningGoalResult
)

AuxiliaryInitialPlanningPersistenceError = (
    initial_records.AuxiliaryInitialPlanningPersistenceError
)
AuxiliaryInitialPlanningCompletionBinding = (
    initial_records.AuxiliaryInitialPlanningCompletionBinding
)
StoredAuxiliaryInitialPlanningCompletion = (
    initial_records.StoredAuxiliaryInitialPlanningCompletion
)
AuxiliaryPositivePlanningCompletionBinding = (
    initial_records.AuxiliaryPositivePlanningCompletionBinding
)

PlanningPrimitiveInvocationPersistenceError = (
    primitive_records.PlanningPrimitiveInvocationPersistenceError
)
PlanningPrimitiveInvocationIdentityCollision = (
    primitive_records.PlanningPrimitiveInvocationIdentityCollision
)
PlanningPrimitiveInvocationStateGuardRejected = (
    primitive_records.PlanningPrimitiveInvocationStateGuardRejected
)
PlanningPrimitiveInvocationMutationResult = (
    primitive_records.PlanningPrimitiveInvocationMutationResult
)
StoredPlanningPrimitiveInvocation = (
    primitive_records.StoredPlanningPrimitiveInvocation
)

PlanningArtifactSealPersistenceError = (
    artifact_records.PlanningArtifactSealPersistenceError
)
PlanningArtifactSealApplyIdCollision = (
    artifact_records.PlanningArtifactSealApplyIdCollision
)
SealAuxiliaryHostPrimitiveResultCommand = (
    artifact_records.SealAuxiliaryHostPrimitiveResultCommand
)
PlanningHostPrimitiveSealResult = (
    artifact_records.PlanningHostPrimitiveSealResult
)

AuxiliaryReplanTriggerPersistenceError = (
    replan_records.AuxiliaryReplanTriggerPersistenceError
)
AuxiliaryReplanTriggerIdentityCollision = (
    replan_records.AuxiliaryReplanTriggerIdentityCollision
)
AuxiliaryReplanTriggerStaleAuthority = (
    replan_records.AuxiliaryReplanTriggerStaleAuthority
)
AuxiliaryReplanTriggerStoredAuthorityCorrupt = (
    replan_records.AuxiliaryReplanTriggerStoredAuthorityCorrupt
)
CreateAuxiliaryReplanTriggerCommand = (
    replan_records.CreateAuxiliaryReplanTriggerCommand
)
ConsumeAuxiliaryReplanTriggerCommand = (
    replan_records.ConsumeAuxiliaryReplanTriggerCommand
)
AuxiliaryReplanTriggerMutationResult = (
    replan_records.AuxiliaryReplanTriggerMutationResult
)
AuxiliaryReplanTriggerApplicationMutationResult = (
    replan_records.AuxiliaryReplanTriggerApplicationMutationResult
)


def supersede_auxiliary_planning_goal(
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> SupersedeAuxiliaryPlanningGoalResult:
    return supersede_records.supersede_auxiliary_planning_goal(
        session_store.current_store_deps(),
        command=command,
    )


def require_authenticated_planning_goal_supersede_receipt(
    *,
    receipt: PlanningGoalSupersedeReceipt,
) -> PlanningGoalSupersedeReceipt:
    return supersede_records.require_authenticated_planning_goal_supersede_receipt(
        session_store.current_store_deps(),
        receipt=receipt,
    )


def get_pending_auxiliary_goal_supersede_receipt(
    *,
    session_id: str,
    task_id: str,
) -> PlanningGoalSupersedeReceipt | None:
    return supersede_records.get_pending_auxiliary_goal_supersede_receipt(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
    )


def get_authenticated_user_target_change_receipt(
    *,
    session_id: str,
    task_id: str,
    invocation_turn_id: str,
    replacement_objective: str,
    source_start: int,
    source_end: int,
    source_sha256: str,
) -> PlanningGoalSupersedeReceipt | None:
    return supersede_records.get_authenticated_user_target_change_receipt(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
        invocation_turn_id=invocation_turn_id,
        replacement_objective=replacement_objective,
        source_start=source_start,
        source_end=source_end,
        source_sha256=source_sha256,
    )


def get_auxiliary_initial_planning_completion(
    *,
    session_id: str,
    insession_task_id: str,
) -> StoredAuxiliaryInitialPlanningCompletion | None:
    return initial_records.get_auxiliary_initial_planning_completion(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=insession_task_id,
    )


def get_current_auxiliary_planning_goal(
    *,
    session_id: str,
    insession_task_id: str,
) -> AuxiliaryPlanningGoal | None:
    return graph_records.get_current_auxiliary_planning_goal(
        session_store.current_store_deps(),
        session_id=session_id,
        insession_task_id=insession_task_id,
    )


def reserve_planning_primitive_invocation(
    *,
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> PlanningPrimitiveInvocationMutationResult:
    return primitive_records.reserve_planning_primitive_invocation(
        session_store.current_store_deps(),
        invocation=invocation,
    )


def require_planning_primitive_invocation_current(
    *,
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> None:
    primitive_records.require_planning_primitive_invocation_current(
        session_store.current_store_deps(),
        invocation=invocation,
    )


def get_planning_primitive_invocation(
    *,
    session_id: str,
    primitive_call_id: str,
) -> StoredPlanningPrimitiveInvocation | None:
    return primitive_records.get_planning_primitive_invocation(
        session_store.current_store_deps(),
        session_id=session_id,
        primitive_call_id=primitive_call_id,
    )


def seal_auxiliary_host_primitive_result(
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    result: object,
) -> PlanningHostPrimitiveSealResult:
    return artifact_records.seal_auxiliary_host_primitive_result(
        session_store.current_store_deps(),
        command=command,
        result=result,  # type: ignore[arg-type]
    )


def create_auxiliary_replan_trigger(
    *,
    command: CreateAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerMutationResult:
    return replan_records.create_auxiliary_replan_trigger(
        session_store.current_store_deps(),
        command=command,
    )


def consume_auxiliary_replan_trigger(
    *,
    command: ConsumeAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerApplicationMutationResult:
    return replan_records.consume_auxiliary_replan_trigger(
        session_store.current_store_deps(),
        command=command,
    )


def get_auxiliary_replan_trigger(
    *,
    session_id: str,
    trigger_id: str,
) -> AuxiliaryReplanTriggerReceipt | None:
    return replan_records.get_auxiliary_replan_trigger(
        session_store.current_store_deps(),
        session_id=session_id,
        trigger_id=trigger_id,
    )


def get_active_auxiliary_replan_trigger(
    *,
    session_id: str,
    task_id: str,
) -> AuxiliaryReplanTriggerReceipt | None:
    return replan_records.get_active_auxiliary_replan_trigger(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
    )


def get_auxiliary_replan_trigger_application(
    *,
    session_id: str,
    trigger_id: str,
) -> AuxiliaryReplanTriggerApplicationReceipt | None:
    return replan_records.get_auxiliary_replan_trigger_application(
        session_store.current_store_deps(),
        session_id=session_id,
        trigger_id=trigger_id,
    )


def count_auxiliary_replan_triggers(
    *,
    session_id: str,
    task_id: str,
    goal_id: str,
) -> int:
    return replan_records.count_auxiliary_replan_triggers(
        session_store.current_store_deps(),
        session_id=session_id,
        task_id=task_id,
        goal_id=goal_id,
    )


__all__ = [
    "AuxiliaryPlanningGoal",
    "AuxiliaryReplanTriggerApplicationMutationResult",
    "AuxiliaryReplanTriggerApplicationReceipt",
    "AuxiliaryReplanTriggerIdentityCollision",
    "AuxiliaryReplanTriggerMutationResult",
    "AuxiliaryReplanTriggerPersistenceError",
    "AuxiliaryReplanTriggerReceipt",
    "AuxiliaryReplanTriggerStaleAuthority",
    "AuxiliaryReplanTriggerStoredAuthorityCorrupt",
    "AuxiliaryGoalSupersedeIdentityCollision",
    "AuxiliaryGoalSupersedePersistenceError",
    "AuxiliaryGoalSupersedeStaleAuthority",
    "AuxiliaryGoalSupersedeUnsafeWorkRun",
    "AuxiliaryInitialPlanningCompletionBinding",
    "AuxiliaryInitialPlanningPersistenceError",
    "AuxiliaryPositivePlanningCompletionBinding",
    "ConsumeAuxiliaryReplanTriggerCommand",
    "CreateAuxiliaryReplanTriggerCommand",
    'FrozenPlanningContextPrimitiveInvocation',
    "PlanningArtifactSealApplyIdCollision",
    "PlanningArtifactSealPersistenceError",
    "PlanningGoalSupersedeReason",
    "PlanningGoalSupersedeReceipt",
    'PlanningHostPrimitiveSealResult',
    "PlanningPrimitiveInvocationIdentityCollision",
    'PlanningPrimitiveInvocationMutationResult',
    "PlanningPrimitiveInvocationPersistenceError",
    "PlanningPrimitiveInvocationStateGuardRejected",
    'SealAuxiliaryHostPrimitiveResultCommand',
    "StoredAuxiliaryInitialPlanningCompletion",
    'StoredPlanningPrimitiveInvocation',
    "SupersedeAuxiliaryPlanningGoalCommand",
    "SupersedeAuxiliaryPlanningGoalResult",
    "consume_auxiliary_replan_trigger",
    "count_auxiliary_replan_triggers",
    "create_auxiliary_replan_trigger",
    "get_active_auxiliary_replan_trigger",
    "get_authenticated_user_target_change_receipt",
    "get_auxiliary_replan_trigger",
    "get_auxiliary_replan_trigger_application",
    "get_auxiliary_initial_planning_completion",
    "get_current_auxiliary_planning_goal",
    "get_pending_auxiliary_goal_supersede_receipt",
    "get_planning_primitive_invocation",
    "require_authenticated_planning_goal_supersede_receipt",
    "require_planning_primitive_invocation_current",
    "reserve_planning_primitive_invocation",
    "seal_auxiliary_host_primitive_result",
    "supersede_auxiliary_planning_goal",
]
