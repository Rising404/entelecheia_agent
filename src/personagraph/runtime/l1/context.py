"""L1 执行真正消费的 Turn 上下文视图。

公共 Entry 可以用自己的具体数据结构实现这个结构化合同；L1 不需要知道 Entry
如何组装历史或附件，也不反向依赖 ``runtime.entry``。
"""

from __future__ import annotations

from typing import Any, Protocol


class L1TurnContext(Protocol):
    """L1 controller 可读取、不可修改的最小上下文。"""

    history_pairs: tuple[dict[str, str], ...]
    session_summary: str | None
    attachments: Any


__all__ = ["L1TurnContext"]
