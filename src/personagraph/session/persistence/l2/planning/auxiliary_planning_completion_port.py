"""Auxiliary 规划完成读取的窄持久化门面。

``session.store`` 仍是稳定公开兼容接口。此端口向相邻初始规划记录传递显式
:class:`StoreDeps`，用于 Store 所有、认证不可变引导完成项的唯一读取。它刻意排除现有
图提交事务内使用的连接作用域密封、加载与要求辅助函数，以及 TaskGraph 提交、
Runtime/API 策略和 schema 所有权。
"""

from __future__ import annotations

from . import auxiliary_planning_completions
from ...deps import StoreDeps


AuxiliaryInitialPlanningPersistenceError = (
    auxiliary_planning_completions.AuxiliaryInitialPlanningPersistenceError
)
AuxiliaryInitialPlanningCompletionBinding = (
    auxiliary_planning_completions.AuxiliaryInitialPlanningCompletionBinding
)
StoredAuxiliaryInitialPlanningCompletion = (
    auxiliary_planning_completions.StoredAuxiliaryInitialPlanningCompletion
)
AuxiliaryPositivePlanningCompletionBinding = (
    auxiliary_planning_completions.AuxiliaryPositivePlanningCompletionBinding
)
StoredAuxiliaryPositivePlanningCompletion = (
    auxiliary_planning_completions.StoredAuxiliaryPositivePlanningCompletion
)


def get_auxiliary_initial_planning_completion(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryInitialPlanningCompletion | None:
    """通过 Store 所有读取加载精确当前引导完成项。"""

    return auxiliary_planning_completions.get_auxiliary_initial_planning_completion(
        deps,
        session_id=session_id,
        task_id=task_id,
    )
