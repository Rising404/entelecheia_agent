"""供应商中立的模型 I/O 契约、网关与协议适配器。

包入口保持轻量，只公开线上无关的输出契约。调用执行位于 :mod:`.gateway`；端点配置、
调用 tier、请求方言、供应商适配与流式投影由相邻模块分别拥有。
"""

from .contracts import (
    AssistantText,
    ModelTurnOutput,
    ProtocolError,
    ToolCall,
    ToolCallBatch,
    model_output_from_dict,
    model_output_to_dict,
    output_tool_calls,
)

__all__ = [
    "AssistantText",
    "ModelTurnOutput",
    "ProtocolError",
    "ToolCall",
    "ToolCallBatch",
    "model_output_from_dict",
    "model_output_to_dict",
    "output_tool_calls",
]
