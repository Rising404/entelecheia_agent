"""Tool Runtime 组装与控制器共享的轻量导入 Protocol。

具体的 attempt 控制器与 SQLite 桥接器都依赖 Session 记录，而这些结构化端口刻意不依赖它们：
这使组装模块可以公开精确注解，而无需急切加载任一持久实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ....tools.catalog import CatalogSnapshot
    from ...work_run import HostAcceptedAttemptDecision, ToolResult
    from ...work_run.mutation_receipt_contracts import WorkExecutionMutationResult
    from ..attempts.active_time import AttemptActiveTimeMeter
    from .attempt_contracts import (
        AttemptToolBridgePreflightRequest,
        AttemptToolBridgeRequest,
    )


class AttemptToolBridge(Protocol):
    """一个 ``call_tools`` 提案所用的注入式两阶段端口。"""

    @property
    def supports_protected_recovery(self) -> bool:
        """持久受保护调用能否在不自动重发的情况下重新进入。"""
        ...

    def preflight(
        self,
        request: AttemptToolBridgePreflightRequest,
    ) -> HostAcceptedAttemptDecision: ...

    def dispatch(
        self,
        request: AttemptToolBridgeRequest,
        *,
        active_time_meter: AttemptActiveTimeMeter,
    ) -> WorkExecutionMutationResult: ...


class CatalogRebindableAttemptToolBridge(AttemptToolBridge, Protocol):
    """显式选择加入不可变 CatalogSnapshot 重绑定的桥接器。"""

    def with_catalog_snapshot(
        self,
        catalog_snapshot: CatalogSnapshot,
    ) -> AttemptToolBridge: ...


class DurableToolResultObserver(Protocol):
    """ToolResult 持久化或重放后调用的幂等观察器。"""

    def __call__(
        self,
        *,
        session_id: str,
        turn_id: str,
        work_run_id: str,
        attempt_id: str,
        result: ToolResult,
    ) -> None: ...


def bridge_supports_protected_recovery(
    tool_bridge: AttemptToolBridge | None,
) -> bool:
    """只接受显式受保护恢复声明。

    缺失桥接器、损坏的桥接器属性或真值为真的非 ``bool`` 值，都绝不能扩大恢复权威。
    """

    if tool_bridge is None:
        return False
    try:
        return tool_bridge.supports_protected_recovery is True
    except Exception:
        return False


def bridge_supports_catalog_rebind(
    tool_bridge: AttemptToolBridge | None,
) -> bool:
    """Runtime 更改桥接器 Catalog 前，要求存在一个显式可调用端口。"""

    return tool_bridge is not None and callable(
        getattr(tool_bridge, "with_catalog_snapshot", None)
    )


__all__ = [
    "AttemptToolBridge",
    "CatalogRebindableAttemptToolBridge",
    "DurableToolResultObserver",
    "bridge_supports_catalog_rebind",
    "bridge_supports_protected_recovery",
]
