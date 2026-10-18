"""Entry application composition root 的显式持久化边界。"""

from __future__ import annotations

from typing import Protocol

from ..l1.ports import L1EntryBootstrapStorePort, L1StorePort
from .context.ports import EntryContextAssemblyStorePort
from .lifecycle.ports import EntryLifecycleStorePort


class EntryApplicationStorePort(
    EntryContextAssemblyStorePort,
    EntryLifecycleStorePort,
    L1EntryBootstrapStorePort,
    L1StorePort,
    Protocol,
):
    """API composition root 提供给跨阶段 Entry 流程的能力组合。

    本协议不声明任何方法；各方法只归属 context、routing、lifecycle 或 L1 的窄
    port。仅组合当前 L0/L1 的运行能力。
    """


__all__ = ["EntryApplicationStorePort"]
