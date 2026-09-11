"""Entelecheia Runtime 包。

权威产品入口是 :func:`personagraph.runtime.entry.run_entry_turn`。
它由 Host 编排，并通过 SQLite/CAS 持久化类型化的 Turn/Window 状态；
它不导入 LangGraph。

已退休的编排逻辑不会从该包边界重新导出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .entry import run_entry_turn
    from .turn import EntryTurnResult

__all__ = ["EntryTurnResult", "run_entry_turn"]


def __getattr__(name: str) -> Any:
    """延迟暴露公共入口，避免加载 Provider 栈。"""

    if name == "run_entry_turn":
        from .entry import run_entry_turn

        return run_entry_turn
    if name == "EntryTurnResult":
        from .turn import EntryTurnResult

        return EntryTurnResult
    raise AttributeError(name)
