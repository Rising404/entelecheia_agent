"""不可变的逐节点 Tool Runtime 捆绑契约。

这些值将已获授权的 TaskNode 绑定到其精确 Catalog、桥接器和论文提示词范围。它们不加载
Task 图、不读取 Store、不预检工厂、不调度 WorkRun，也不发出事件。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from personagraph.l2.task_execution.tool_bridge.contracts import AttemptToolBridge
from personagraph.l2.work_run import TaskNodeSubject
from ..paper_prompt_context import PaperAttemptContext
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.tools.catalog.binding import ToolBinding


@dataclass(frozen=True, slots=True)
class TaskNodeToolRuntime:
    """一个节点的精确 Catalog、桥接器与提示词安全论文范围。

    此对象不包含图决定；调用方只能解析已有权威的主体。
    """

    catalog_snapshot: CatalogSnapshot
    tool_bridge: AttemptToolBridge | None = None
    paper_resources: PaperAttemptContext | None = None
    contextual_bindings: tuple[ToolBinding, ...] = ()


TaskNodeToolRuntimeFactory = Callable[
    [TaskNodeSubject],
    TaskNodeToolRuntime,
]


@dataclass(frozen=True, slots=True)
class TaskNodeToolRuntimeBinding:
    subject: TaskNodeSubject
    runtime: TaskNodeToolRuntime


@dataclass(frozen=True, slots=True)
class TaskNodeToolRuntimePlan:
    """在任何节点可变更前冻结的全部当前节点运行时。"""

    session_id: str
    task_id: str
    graph_revision: int
    default_runtime: TaskNodeToolRuntime
    bindings: tuple[TaskNodeToolRuntimeBinding, ...] = ()

    def runtime_for(self, subject: TaskNodeSubject) -> TaskNodeToolRuntime:
        if (
            subject.task_id != self.task_id
            or subject.graph_revision != self.graph_revision
        ):
            raise ValueError("Task node runtime plan crossed TaskGraph authority")
        for binding in self.bindings:
            if binding.subject == subject:
                return binding.runtime
        if self.bindings:
            raise ValueError("Task node is absent from the frozen runtime plan")
        return self.default_runtime


__all__ = (
    'TaskNodeToolRuntimeBinding',
    "TaskNodeToolRuntimeFactory",
    'TaskNodeToolRuntimePlan',
    'TaskNodeToolRuntime',
)
