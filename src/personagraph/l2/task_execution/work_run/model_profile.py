"""一个 WorkRun 的不可变物理模型限制。

该配置刻意作为纯数据契约：应用组装可以选择这些边界，而无需导入已配置的模型网关或任何
Runtime 执行机制。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkRunStructuredModelProfile:
    """由应用组装根选择的物理提供商限制。"""

    attempt_max_output_tokens: int
    verification_max_output_tokens: int
    timeout_s: float

    def __post_init__(self) -> None:
        if self.attempt_max_output_tokens < 1:
            raise ValueError("attempt_max_output_tokens must be positive")
        if self.verification_max_output_tokens < 1:
            raise ValueError("verification_max_output_tokens must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


__all__ = ['WorkRunStructuredModelProfile']
