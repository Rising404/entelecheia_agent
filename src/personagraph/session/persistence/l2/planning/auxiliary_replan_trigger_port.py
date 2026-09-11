"""Auxiliary 重规划触发器权威的窄持久化门面。

``session.store`` 仍是稳定公开兼容接口。此端口向相邻重规划触发器记录传递显式
:class:`StoreDeps`，用于创建、消费、读取和计数由验证派生的触发器。它刻意排除语义
验证、通用 TaskGraph 执行重规划请求、Runtime/API 策略和 schema 所有权。
"""

from __future__ import annotations

from . import auxiliary_replan_triggers
from ...deps import StoreDeps


AuxiliaryReplanTriggerPersistenceError = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerPersistenceError
)
AuxiliaryReplanTriggerIdentityCollision = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerIdentityCollision
)
AuxiliaryReplanTriggerStaleAuthority = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerStaleAuthority
)
AuxiliaryReplanTriggerStoredAuthorityCorrupt = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerStoredAuthorityCorrupt
)
CreateAuxiliaryReplanTriggerCommand = (
    auxiliary_replan_triggers.CreateAuxiliaryReplanTriggerCommand
)
ConsumeAuxiliaryReplanTriggerCommand = (
    auxiliary_replan_triggers.ConsumeAuxiliaryReplanTriggerCommand
)
AuxiliaryReplanTriggerMutationResult = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerMutationResult
)
AuxiliaryReplanTriggerApplicationMutationResult = (
    auxiliary_replan_triggers.AuxiliaryReplanTriggerApplicationMutationResult
)


def create_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    command: CreateAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerMutationResult:
    return auxiliary_replan_triggers.create_auxiliary_replan_trigger(
        deps,
        command=command,
    )


def consume_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    command: ConsumeAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerApplicationMutationResult:
    return auxiliary_replan_triggers.consume_auxiliary_replan_trigger(
        deps,
        command=command,
    )


def get_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    trigger_id: str,
) -> auxiliary_replan_triggers.AuxiliaryReplanTriggerReceipt | None:
    return auxiliary_replan_triggers.get_auxiliary_replan_trigger(
        deps,
        session_id=session_id,
        trigger_id=trigger_id,
    )


def get_active_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> auxiliary_replan_triggers.AuxiliaryReplanTriggerReceipt | None:
    return auxiliary_replan_triggers.get_active_auxiliary_replan_trigger(
        deps,
        session_id=session_id,
        task_id=task_id,
    )


def get_auxiliary_replan_trigger_application(
    deps: StoreDeps,
    *,
    session_id: str,
    trigger_id: str,
) -> auxiliary_replan_triggers.AuxiliaryReplanTriggerApplicationReceipt | None:
    return auxiliary_replan_triggers.get_auxiliary_replan_trigger_application(
        deps,
        session_id=session_id,
        trigger_id=trigger_id,
    )


def count_auxiliary_replan_triggers(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    goal_id: str,
) -> int:
    return auxiliary_replan_triggers.count_auxiliary_replan_triggers(
        deps,
        session_id=session_id,
        task_id=task_id,
        goal_id=goal_id,
    )
