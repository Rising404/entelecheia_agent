"""Auxiliary 规划目标取代的窄持久化门面。

``session.store`` 仍是稳定公开兼容接口。此端口向相邻目标取代记录传递显式
:class:`StoreDeps`，用于原子取代、密封回执认证和后继安全回执读取。它刻意排除初始
规划、语义审查、重规划触发器、图提交、Runtime/API 策略和 schema 行为。
"""

from __future__ import annotations

from . import auxiliary_goal_supersede
from ...deps import StoreDeps


AuxiliaryGoalSupersedePersistenceError = (
    auxiliary_goal_supersede.AuxiliaryGoalSupersedePersistenceError
)
AuxiliaryGoalSupersedeIdentityCollision = (
    auxiliary_goal_supersede.AuxiliaryGoalSupersedeIdentityCollision
)
AuxiliaryGoalSupersedeStaleAuthority = (
    auxiliary_goal_supersede.AuxiliaryGoalSupersedeStaleAuthority
)
AuxiliaryGoalSupersedeStoredAuthorityCorrupt = (
    auxiliary_goal_supersede.AuxiliaryGoalSupersedeStoredAuthorityCorrupt
)
AuxiliaryGoalSupersedeUnsafeWorkRun = (
    auxiliary_goal_supersede.AuxiliaryGoalSupersedeUnsafeWorkRun
)
PlanningGoalSupersedeReason = auxiliary_goal_supersede.PlanningGoalSupersedeReason
PlanningGoalSupersedeReceipt = (
    auxiliary_goal_supersede.PlanningGoalSupersedeReceipt
)
PlanningGoalSupersedeSourceBinding = (
    auxiliary_goal_supersede.PlanningGoalSupersedeSourceBinding
)
SupersedeAuxiliaryPlanningGoalCommand = (
    auxiliary_goal_supersede.SupersedeAuxiliaryPlanningGoalCommand
)
SupersedeAuxiliaryPlanningGoalResult = (
    auxiliary_goal_supersede.SupersedeAuxiliaryPlanningGoalResult
)


def supersede_auxiliary_planning_goal(
    deps: StoreDeps,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> SupersedeAuxiliaryPlanningGoalResult:
    return auxiliary_goal_supersede.supersede_auxiliary_planning_goal(
        deps,
        command=command,
    )


def require_authenticated_planning_goal_supersede_receipt(
    deps: StoreDeps,
    *,
    receipt: PlanningGoalSupersedeReceipt,
) -> PlanningGoalSupersedeReceipt:
    return auxiliary_goal_supersede.require_authenticated_planning_goal_supersede_receipt(
        deps,
        receipt=receipt,
    )


def get_pending_auxiliary_goal_supersede_receipt(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> PlanningGoalSupersedeReceipt | None:
    return auxiliary_goal_supersede.get_pending_auxiliary_goal_supersede_receipt(
        deps,
        session_id=session_id,
        task_id=task_id,
    )


def get_authenticated_user_target_change_receipt(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    invocation_turn_id: str,
    replacement_objective: str,
    source_start: int,
    source_end: int,
    source_sha256: str,
) -> PlanningGoalSupersedeReceipt | None:
    return auxiliary_goal_supersede.get_authenticated_user_target_change_receipt(
        deps,
        session_id=session_id,
        task_id=task_id,
        invocation_turn_id=invocation_turn_id,
        replacement_objective=replacement_objective,
        source_start=source_start,
        source_end=source_end,
        source_sha256=source_sha256,
    )
