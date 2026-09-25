"""当前 Turn 内可恢复内容的稳定类型。

这里的 ``persistent`` 只表示内容会跨 Attempt、验证重试和进程恢复继续存在；
它不是长期记忆，也不表示内容已经通过语义验证。本包只定义状态及其不变量，
不执行模型工具，也不读写数据库。

包入口使用惰性导出，使 L1 的 Plan 不会传递式加载
仅归 L2 WorkRun 所有的 OutputWindow。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# 下面的 noqa: F401 是必要的：本模块是惰性导出 facade，TYPE_CHECKING 块只为类型检查
# 提供符号，真实导出由 _EXPORTS 与模块级 __getattr__ 完成，静态检查看不到这层关系。
if TYPE_CHECKING:
    from .acceptance import AcceptanceProgressItem  # noqa: F401
    from .evidence import (
        EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION,  # noqa: F401
        EmptySupportJustification,  # noqa: F401
        EmptySupportReason,  # noqa: F401
        format_l1_call_ref,  # noqa: F401
        parse_l1_call_ref,  # noqa: F401
    )
    from .output_window import (
        OutputWindow,  # noqa: F401
        OutputWindowFormat,  # noqa: F401
        initialize_output_window,  # noqa: F401
    )
    from .plan import (
        L1Acceptance,  # noqa: F401
        L1Plan,  # noqa: F401
        L1MessageSource,  # noqa: F401
    )


_EXPORTS = {
    "AcceptanceProgressItem": (
        "personagraph.persistent_turn_content.acceptance",
        "AcceptanceProgressItem",
    ),
    "EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION": (
        "personagraph.persistent_turn_content.evidence",
        "EMPTY_SUPPORT_JUSTIFICATION_CONTRACT_VERSION",
    ),
    "EmptySupportJustification": (
        "personagraph.persistent_turn_content.evidence",
        "EmptySupportJustification",
    ),
    "EmptySupportReason": (
        "personagraph.persistent_turn_content.evidence",
        "EmptySupportReason",
    ),
    "format_l1_call_ref": (
        "personagraph.persistent_turn_content.evidence", "format_l1_call_ref",
    ),
    "parse_l1_call_ref": (
        "personagraph.persistent_turn_content.evidence", "parse_l1_call_ref",
    ),
    "OutputWindow": (
        "personagraph.persistent_turn_content.output_window",
        "OutputWindow",
    ),
    "OutputWindowFormat": (
        "personagraph.persistent_turn_content.output_window",
        "OutputWindowFormat",
    ),
    "initialize_output_window": (
        "personagraph.persistent_turn_content.output_window",
        "initialize_output_window",
    ),
    "L1Acceptance": ("personagraph.persistent_turn_content.plan", "L1Acceptance"),
    "L1Plan": ("personagraph.persistent_turn_content.plan", "L1Plan"),
    "L1MessageSource": (
        "personagraph.persistent_turn_content.plan",
        "L1MessageSource",
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
