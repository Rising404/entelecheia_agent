"""运行已经由公共 Entry 准入的 L2 Task lane。

本模块只把持久 Task 解析为 executor 输入并运行 L2 生产链。Turn Window、租约、
事件结算和正式回复仍由 ``runtime.entry`` 持有，因此这里既不接收 Entry Store
合同，也不决定用户可见结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.turn_events import EntryEventEmitter

if TYPE_CHECKING:
    from personagraph.l2.auxiliary_execution.production_chain import (
        AuxiliaryProductionChainResult,
    )


class L2TaskObjective(Protocol):
    """L2 executor 启动前唯一需要读取的持久 Task 字段。"""

    objective: str


class L2TaskObjectiveStore(Protocol):
    """读取已准入 Task 目标的最小持久化端口。"""

    def get_insession_task_details(
        self,
        session_id: str,
        insession_task_id: str,
    ) -> L2TaskObjective | None: ...


class L2TaskExecutor(Protocol):
    """L2 lane 可调用的精确 executor 合同。"""

    def __call__(
        self,
        *,
        session_id: str,
        turn_id: str,
        task_id: str,
        desired_output: str,
        emit: EntryEventEmitter,
        deadline: TurnDeadline,
        features: Mapping[str, Any] | None = None,
        file_retrieval_data_version: str | None = None,
    ) -> AuxiliaryProductionChainResult: ...


class L2TaskTargetUnavailableError(RuntimeError):
    """已准入的 L2 route 不再指向可执行的持久 Task。"""


def run_l2_task_lane(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    emit: EntryEventEmitter,
    deadline: TurnDeadline,
    features: Mapping[str, Any] | None = None,
    file_retrieval_data_version: str | None = None,
    task_store: L2TaskObjectiveStore | None = None,
    executor: L2TaskExecutor | None = None,
) -> AuxiliaryProductionChainResult:
    """读取精确 Task 目标并运行 L2 executor，不处理 Entry 生命周期。"""

    if task_store is None:
        from personagraph.session.l2_store import task_graph as task_store

    task = task_store.get_insession_task_details(session_id, task_id)
    if task is None:
        raise L2TaskTargetUnavailableError(
            "the admitted L2 route no longer has a durable Task target"
        )

    if executor is None:
        from .executor import run_auxiliary_task_executor as executor

    return executor(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        desired_output=task.objective,
        emit=emit,
        deadline=deadline,
        features=features,
        file_retrieval_data_version=file_retrieval_data_version,
    )


__all__ = [
    "L2TaskObjective",
    "L2TaskObjectiveStore",
    "L2TaskTargetUnavailableError",
    "run_l2_task_lane",
]
