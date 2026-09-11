"""模型一次调用可以输出的结构化协议。

这里的值都只是提案；Runtime/Host 校验并接受后，才会改变持久内容或执行真实工具。
L1 最终答复与 L2 持久 OutputWindow 由不同子模块持有。包入口使用惰性导出，
因此导入 L1 协议不会加载 L2 OutputWindow 合同。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# 下面的 noqa: F401 是必要的：本模块是惰性导出 facade，TYPE_CHECKING 块只为类型检查
# 提供符号，真实导出由 _EXPORTS 与模块级 __getattr__ 完成，静态检查看不到这层关系。
if TYPE_CHECKING:
    from .actions import AcceptanceUpdate, CallToolsAction, ToolCallProposal  # noqa: F401
    from .l1 import (
        L1AcceptanceProposal,  # noqa: F401
        L1AttemptActionProposal,  # noqa: F401
        L1AttemptDecisionProposal,  # noqa: F401
        L1PlanProposal,  # noqa: F401
        SubmitFinalReplyAction,  # noqa: F401
        materialize_l1_plan,  # noqa: F401
    )
    from .output_window import (
        SubmitOutputWindowAction,  # noqa: F401
        WriteOutputWindowAction,  # noqa: F401
    )
    from .validation import require_completed_support_submission  # noqa: F401


_EXPORTS = {
    "AcceptanceUpdate": ("personagraph.output_protocol.actions", "AcceptanceUpdate"),
    "CallToolsAction": ("personagraph.output_protocol.actions", "CallToolsAction"),
    "ToolCallProposal": ("personagraph.output_protocol.actions", "ToolCallProposal"),
    "L1AcceptanceProposal": ("personagraph.output_protocol.l1", "L1AcceptanceProposal"),
    "L1AttemptActionProposal": (
        "personagraph.output_protocol.l1",
        "L1AttemptActionProposal",
    ),
    "L1AttemptDecisionProposal": (
        "personagraph.output_protocol.l1",
        "L1AttemptDecisionProposal",
    ),
    "L1PlanProposal": ("personagraph.output_protocol.l1", "L1PlanProposal"),
    "SubmitFinalReplyAction": (
        "personagraph.output_protocol.l1",
        "SubmitFinalReplyAction",
    ),
    "materialize_l1_plan": (
        "personagraph.output_protocol.l1",
        "materialize_l1_plan",
    ),
    "SubmitOutputWindowAction": (
        "personagraph.output_protocol.output_window",
        "SubmitOutputWindowAction",
    ),
    "WriteOutputWindowAction": (
        "personagraph.output_protocol.output_window",
        "WriteOutputWindowAction",
    ),
    "require_completed_support_submission": (
        "personagraph.output_protocol.validation",
        "require_completed_support_submission",
    ),
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
