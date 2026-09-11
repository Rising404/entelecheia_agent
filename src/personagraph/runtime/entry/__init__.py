"""权威 Runtime Entry 的最小公共入口。

共享的冷 Turn 合同位于 :mod:`personagraph.runtime.turn`。导入本 facade 不应初始化
模型、Provider、Session 执行栈，也不会把 Entry 的内部实现模块伪装成公共 API。
"""

from __future__ import annotations

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

if _TYPE_CHECKING:
    from .application import (
        accept_entry_turn,
        execute_accepted_entry_turn,
        resume_active_l1_entry_turn,
        run_entry_turn,
    )

__all__ = [
    "accept_entry_turn",
    "execute_accepted_entry_turn",
    "resume_active_l1_entry_turn",
    "run_entry_turn",
]

def _application() -> _ModuleType:
    return _import_module(".application", __name__)


def __getattr__(name: str) -> _Any:
    """只惰性解析明确承诺的公共用例。"""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_application(), name)
