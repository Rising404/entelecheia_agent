"""由 Turn 负责执行的有界 L1 Runtime。

包入口保持惰性：导入 L1 契约不应同时初始化模型、检索或 Session 适配器。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# 下面的 noqa: F401 是必要的：本模块是惰性导出 facade，TYPE_CHECKING 块只为类型检查
# 提供符号，真实导出由 _EXPORTS 与模块级 __getattr__ 完成，静态检查看不到这层关系。
if TYPE_CHECKING:
    from ...output_protocol import L1AttemptDecisionProposal  # noqa: F401
    from ...persistent_turn_content import (
        L1Acceptance,  # noqa: F401
        L1Plan,  # noqa: F401
    )
    from .controller import L1ControllerResult  # noqa: F401
    from .ports import L1StorePort  # noqa: F401


_EXPORTS = {
    'L1Acceptance': (
        "personagraph.persistent_turn_content.plan",
        'L1Acceptance',
    ),
    'L1AttemptDecisionProposal': (
        "personagraph.output_protocol.l1",
        'L1AttemptDecisionProposal',
    ),
    'L1Plan': ("personagraph.persistent_turn_content.plan", 'L1Plan'),
    "materialize_l1_plan": (
        "personagraph.output_protocol.l1",
        "materialize_l1_plan",
    ),
    'L1ControllerResult': (
        "personagraph.runtime.l1.controller",
        'L1ControllerResult',
    ),
    "run_l1_turn": ("personagraph.runtime.l1.controller", "run_l1_turn"),
    "L1StorePort": ("personagraph.runtime.l1.ports", "L1StorePort"),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = list(_EXPORTS)
