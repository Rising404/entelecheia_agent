"""普通 TaskNode 模型调用的狭窄注入权威契约。

这些类型描述控制器如何提供一个持久权威工厂与当前状态重推导回调。它们不选择端点、
不加载或写入账本状态、不协调继续执行，也不重放模型结果。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from personagraph.runtime.model_calls.contracts import DurableLogicalModelCallAuthority

from .model_binding_contracts import TaskNodeBoundModelCall


class TaskNodeModelCallAuthorityFactory(Protocol):
    def __call__(
        self,
        binding: TaskNodeBoundModelCall,
        *,
        rederive_state_guard_sha256: Callable[[], str],
    ) -> DurableLogicalModelCallAuthority: ...


@dataclass(frozen=True, slots=True)
class TaskNodeModelCallPlan:
    """控制器持有的稳定标识与当前状态重推导端口。"""

    logical_call_id: str
    request_turn_id: str
    authority_factory: TaskNodeModelCallAuthorityFactory = field(
        repr=False,
        compare=False,
    )
    rederive_state_guard_sha256: Callable[[], str] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("logical_call_id", self.logical_call_id),
            ("request_turn_id", self.request_turn_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not callable(self.authority_factory):
            raise TypeError("authority_factory must be callable")
        if not callable(self.rederive_state_guard_sha256):
            raise TypeError("state-guard rederivation must be callable")


__all__ = [
    "TaskNodeModelCallAuthorityFactory",
    'TaskNodeModelCallPlan',
]
