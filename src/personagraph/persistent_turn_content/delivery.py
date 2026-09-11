"""可持久交付的 L1 运行通知：固定终止事实，不含模型候选或任意诊断。

Runtime 选择是否进入此出口，Session 在唯一结算事务中复核同一内容合同。
此模块不依赖 Runtime、模型、时钟或数据库；通知不表示任务满足或语义验证通过。
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class L1TerminalNotification:
    failure_code: str
    error_code: str
    reply: str


_TERMINAL_NOTICES = {
    "VERIFICATION_FAILED": (
        "VERIFICATION_FAILED", "本轮未能形成通过交付审查的答复。"
    ),
    "MODEL_OUTPUT_INVALID": (
        "MODEL_OUTPUT_INVALID", "模型未能生成符合输出要求的有效答复。"
    ),
    "MODEL_BAD_RESPONSE": (
        "MODEL_OUTPUT_INVALID", "模型未能形成通过格式检查的有效答复。"
    ),
    "TURN_DEADLINE_EXCEEDED": (
        "TURN_DEADLINE_EXCEEDED", "本轮处理时间已用尽，尚未形成可交付的答复。"
    ),
    "MODEL_CALL_TIMEOUT": (
        "MODEL_TIMEOUT", "模型服务响应超时，尚未形成可交付的答复。"
    ),
    "MODEL_TIMEOUT": (
        "MODEL_TIMEOUT", "模型服务响应超时，尚未形成可交付的答复。"
    ),
    "MODEL_CALL_FAILED": (
        "MODEL_TRANSPORT_FAILURE", "模型服务调用未能完成，尚未形成可交付的答复。"
    ),
    "MODEL_TRANSPORT_FAILURE": (
        "MODEL_TRANSPORT_FAILURE", "模型服务调用未能完成，尚未形成可交付的答复。"
    ),
    "MODEL_CONFIGURATION_FAILURE": (
        "MODEL_CONFIGURATION_FAILURE", "当前模型配置无法完成调用，尚未形成可交付的答复。"
    ),
    "CONTEXT_BUDGET_EXCEEDED": (
        "CONTEXT_BUDGET_EXCEEDED", "所需输入超过当前上下文容量，尚未形成可交付的答复。"
    ),
}


def build_l1_terminal_notification(failure_code: str) -> L1TerminalNotification | None:
    """仅允许已明确支持的可靠终止原因；未知或系统持久化错误不能伪称已交付。"""
    spec = _TERMINAL_NOTICES.get(failure_code) if isinstance(failure_code, str) else None
    if spec is None:
        return None
    error_code, reason = spec
    return L1TerminalNotification(
        failure_code=failure_code,
        error_code=error_code,
        reply=f"本轮未完成：{reason}未通过审查的内容没有作为答案发布。",
    )


__all__ = ["L1TerminalNotification", "build_l1_terminal_notification"]
