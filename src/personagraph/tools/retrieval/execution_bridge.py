"""把工具调用控制映射为 Retrieval 自有窄端口，不向模型参数添加控制字段。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from ...retrieval.execution import RetrievalExecution, execution_scope
from ...retrieval.ports import RetrievalCancelled
from ..execution_context import (
    ToolExecutionContext,
    ToolInvocationCancelled,
    current_tool_execution,
)


@dataclass(frozen=True, slots=True)
class _RetrievalCancellation:
    invocation: ToolExecutionContext

    def checkpoint(self) -> None:
        try:
            self.invocation.checkpoint()
        except ToolInvocationCancelled as exc:
            raise RetrievalCancelled(exc.reason) from exc

    def remaining_seconds(self) -> float | None:
        return self.invocation.remaining_seconds()

    def snapshot(self) -> Mapping[str, object]:
        return self.invocation.snapshot()

    def on_settled(self, observer: Callable[[str], None]) -> None:
        self.invocation.on_settled(observer)


@contextmanager
def file_retrieval_execution() -> Iterator[None]:
    invocation = current_tool_execution()
    if invocation is None:
        yield
        return
    execution = RetrievalExecution(_RetrievalCancellation(invocation))
    try:
        with execution_scope(execution):
            yield
            execution.checkpoint()
    except RetrievalCancelled as exc:
        raise ToolInvocationCancelled(str(exc)) from exc
