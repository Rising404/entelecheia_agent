"""TaskNode 模型边界共享的冻结 Host 输入限制契约。

这些 DTO 只描述确定性的 Host 侧输入限制。它们不选择模型输入、不调用提供商、不访问持久层，
也不推进运行时状态，因此配置组装可以导入它们，而无需引入语义 Attempt 或验证运行时。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskNodeDependencyInputLimits(BaseModel):
    """精确完整依赖载荷的 Host 所有限制。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=200)
    max_items: int = Field(ge=0, le=512)
    max_serialized_utf8_bytes: int = Field(ge=1)


class AttemptDecisionInputLimits(BaseModel):
    """限制完整 Attempt 模型输入的一项必需 Host 配置。

    先前 ToolResult 子限制使历史选择具有确定性。总字节限制会针对发送给提供商的精确紧凑 JSON
    字符串检查，因此其他所有输入组成部分也受到限制。此处刻意不提供生产默认值：所属 Runtime
    配置必须显式注入全部限制，且 ``profile_id`` 仅用于可观测性。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    profile_id: str = Field(min_length=1, max_length=200)
    max_prior_tool_result_items: int = Field(ge=0)
    max_prior_tool_results_serialized_utf8_bytes: int = Field(ge=1)
    dependency_delivery_limits: TaskNodeDependencyInputLimits
    max_serialized_utf8_bytes: int = Field(ge=1)

    @field_validator("profile_id")
    @classmethod
    def _reject_blank_profile_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt input-limit profile_id must not be blank")
        return value


class NodeVerificationInputLimits(BaseModel):
    """一次完整验证器投影所需的冻结 Host 配置。

    这些是确定性输入守卫，不是模型可见预算。此处刻意不提供生产默认值：所属 Runtime 配置必须
    命名并注入初始执行与恢复所用的精确限制。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    profile_id: str = Field(min_length=1, max_length=200)
    max_acceptance_items: int = Field(ge=1)
    max_supporting_tool_result_items: int = Field(ge=0)
    dependency_delivery_limits: TaskNodeDependencyInputLimits
    max_serialized_utf8_bytes: int = Field(ge=1)


__all__ = [
    'AttemptDecisionInputLimits',
    'NodeVerificationInputLimits',
    'TaskNodeDependencyInputLimits',
]
