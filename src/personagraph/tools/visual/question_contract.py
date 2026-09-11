"""视觉工具的问题参数与 Host 身份适配；共享内容规则，不签发身份或调用 provider。"""

from __future__ import annotations

from typing import Any

from ...input_processing.vision.contracts import (
    MAX_VISION_QUESTION_CHARS, VisionPurpose, normalize_vision_question,
    validate_vision_call_identity,
)
from ..execution import ToolBusinessFailure
from ..execution_context import current_tool_execution


def parse_visual_question(purpose: VisionPurpose, question: object) -> str | None:
    """把共享合同错误投影成模型可修复的工具参数错误。"""
    try:
        return normalize_vision_question(purpose, question)
    except (TypeError, ValueError) as exc:
        raise ToolBusinessFailure(
            "invalid_request", str(exc), {"field": "question", "purpose": purpose.value},
        ) from exc


def visual_call_identity(purpose: VisionPurpose) -> str | None:
    """从 Host 执行上下文取身份；供整批预检及实际派发使用同一边界。"""
    execution = current_tool_execution()
    identity = getattr(execution, "logical_tool_call_id", None)
    try:
        validate_vision_call_identity(purpose, identity)
    except ValueError as exc:
        raise ToolBusinessFailure(
            "visual_question_call_identity_required" if purpose is VisionPurpose.QUESTION else "visual_call_identity_invalid",
            "视觉调用缺少有效的 Host 调用身份。",
        ) from exc
    return identity


def visual_question_schema() -> dict[str, Any]:
    return {
        "type": ["string", "null"],
        "maxLength": MAX_VISION_QUESTION_CHARS,
        "description": (
            "purpose=question 时填写希望视觉模型结合当前图像回答的具体自然语言问题；"
            "不能为空白。其它用途必须省略或设为 null。"
        ),
    }


def visual_question_constraint() -> dict[str, Any]:
    """每次返回新 schema；条件属于单个视觉请求，不扩散到整个批次。"""
    return {
        "if": {"properties": {"purpose": {"const": VisionPurpose.QUESTION.value}}, "required": ["purpose"]},
        "then": {"required": ["question"], "properties": {"question": {"type": "string", "pattern": r"\S"}}},
        "else": {"properties": {"question": {"type": "null"}}},
    }
