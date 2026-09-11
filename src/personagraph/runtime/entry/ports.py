"""Entry application composition root 的显式持久化边界。"""

from __future__ import annotations

from typing import Protocol

from ..l1.ports import L1EntryBootstrapStorePort, L1StorePort
from .context.ports import EntryContextAssemblyStorePort
from .lifecycle.ports import EntryLifecycleStorePort
from .routing.ports import EntryRoutingStorePort


class EntryApplicationStorePort(
    EntryContextAssemblyStorePort,
    EntryLifecycleStorePort,
    EntryRoutingStorePort,
    L1EntryBootstrapStorePort,
    L1StorePort,
    Protocol,
):
    """API composition root 提供给跨阶段 Entry 流程的能力组合。

    本协议不声明任何方法；各方法只归属 context、routing、lifecycle 或 L1 的窄
    port。L2 adapter 继续在真正选中 L2 的分支内以自己的局部结构契约接收同一 Store，
    因而导入本模块不会初始化 L2。
    """


__all__ = ["EntryApplicationStorePort"]
