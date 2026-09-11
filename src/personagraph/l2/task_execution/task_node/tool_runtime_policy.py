"""TaskNode Tool 运行时的纯选择与跨对象守卫。

本模块不加载当前图、不依据 Store 解析工厂、不调度工作，也不变更 WorkRun。它只会依据不可变
请求与可选的精确节点主体，验证一个已组装的运行时捆绑。
"""

from __future__ import annotations

from personagraph.l2.work_run import TaskNodeSubject
from ..paper_prompt_context import PaperAttemptContext

from ..task_graph.contracts import TaskGraphWorkRunRequest
from .model_authority_contracts import TaskNodeModelCallAuthorityFactory
from .tool_runtime_contracts import TaskNodeToolRuntime


def ordinary_task_node_model_authority_factory(
    runtime: TaskNodeToolRuntime,
    *,
    factory: TaskNodeModelCallAuthorityFactory | None,
) -> TaskNodeModelCallAuthorityFactory | None:
    """为每个普通 TaskNode 返回通用权威工厂。"""

    return factory


def validate_task_node_tool_runtime(
    runtime: TaskNodeToolRuntime,
    *,
    request: TaskGraphWorkRunRequest,
    subject: TaskNodeSubject | None = None,
) -> None:
    """当已组装运行时跨越其不可变权威时执行失败关闭。"""

    if runtime.catalog_snapshot.exposed() and runtime.tool_bridge is None:
        raise ValueError("an exposed node Tool catalog requires an execution bridge")
    paper = runtime.paper_resources
    if paper is not None and (
        not isinstance(paper, PaperAttemptContext)
        or paper.session_id != request.session_id
        or paper.task_id != request.task_id
    ):
        raise ValueError("node paper resources crossed Session/Task authority")


__all__ = (
    "ordinary_task_node_model_authority_factory",
    "validate_task_node_tool_runtime",
)
