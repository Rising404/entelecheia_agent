"""Entry 的可信、只读、有界上下文组装。

本包拥有附件元数据投影、任务目录投影与最终 ``EntryContext`` 组装。它不接受 Turn、
不选择执行链路、不调用模型，也不持有 Window、租约或最终答复提交权。子模块须由调用方
显式导入；包根保持冷导入且不提供兼容性 re-export。
"""

from __future__ import annotations

__all__: list[str] = []
