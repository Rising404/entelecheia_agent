"""L1 模型调用的具名输出上限。"""

from __future__ import annotations


# L1 的在线结构化调用上限。Attempt 可能提交完整最终回复，因此使用完整
# lane 能力边界；账户 TPM 由 endpoint profile 独立管理，不得反向改写模型输出能力。
L1_MAX_OUTPUT_TOKENS = 65_536
L1_ATTEMPT_MAX_OUTPUT_TOKENS = L1_MAX_OUTPUT_TOKENS

# 语义校验的合法极端形状包含 24 条 finding 及相关结果身份，32K 已足够。
L1_SEMANTIC_VERIFIER_MAX_OUTPUT_TOKENS = 32_768


def l1_attempt_max_output_tokens(*, thinking_enabled: bool) -> int:
    """返回 L1 Attempt 的线上运行上限。

    隐藏推理与最终结构化输出共享 provider completion envelope，因此 thinking
    不会再叠加第二份输出预算。
    """

    if not isinstance(thinking_enabled, bool):
        raise TypeError("thinking_enabled must be boolean")
    return L1_ATTEMPT_MAX_OUTPUT_TOKENS


__all__ = [
    "L1_ATTEMPT_MAX_OUTPUT_TOKENS",
    "L1_MAX_OUTPUT_TOKENS",
    "L1_SEMANTIC_VERIFIER_MAX_OUTPUT_TOKENS",
    "l1_attempt_max_output_tokens",
]
