"""一个 TaskGraph WorkRun 的不可变输入与提供商限制。

本模块只持有 TaskGraph 控制器和 Auxiliary 组装所消费的冻结配置。它不选择节点、
不读取 Session 状态、不组装工具、不调用模型，也不推进 WorkRun。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.task_execution.work_run.model_profile import (
    WorkRunStructuredModelProfile,
)
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS

from ..task_node.input_limits import (
    AttemptDecisionInputLimits,
    NodeVerificationInputLimits,
    TaskNodeDependencyInputLimits,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGraphWorkRunProfile(_Contract):
    """默认关闭的多节点演示组装所用显式限制。"""

    attempt_input_limits: AttemptDecisionInputLimits = Field(
        default_factory=lambda: AttemptDecisionInputLimits(
            profile_id="demo-task-graph-attempt-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="demo-task-graph-attempt-dependencies-v1",
                max_items=63,
                max_serialized_utf8_bytes=256_000,
            ),
            max_serialized_utf8_bytes=1_000_000,
        )
    )
    verification_input_limits: NodeVerificationInputLimits = Field(
        default_factory=lambda: NodeVerificationInputLimits(
            profile_id="demo-task-graph-verification-v1",
            max_acceptance_items=64,
            max_supporting_tool_result_items=256,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="demo-task-graph-verification-dependencies-v1",
                max_items=63,
                max_serialized_utf8_bytes=256_000,
            ),
            max_serialized_utf8_bytes=1_000_000,
        )
    )
    attempt_max_output_tokens: int = Field(default=L2_MAX_OUTPUT_TOKENS, ge=1)
    verification_max_output_tokens: int = Field(default=L2_MAX_OUTPUT_TOKENS, ge=1)
    model_timeout_s: float = Field(default=45.0, gt=0)
    auxiliary_attempt_max_output_tokens: int = Field(
        default=L2_MAX_OUTPUT_TOKENS,
        ge=1,
    )
    auxiliary_verification_max_output_tokens: int = Field(
        default=L2_MAX_OUTPUT_TOKENS,
        ge=1,
    )
    auxiliary_model_timeout_s: float = Field(default=180.0, gt=0)
    max_work_runs_per_turn: int = Field(default=32, ge=1, le=64)
    execution_findings_enabled: bool = True

    def structured_model_profile(self) -> WorkRunStructuredModelProfile:
        """返回一个 TaskNode WorkRun 的物理提供商信封。"""

        return WorkRunStructuredModelProfile(
            attempt_max_output_tokens=self.attempt_max_output_tokens,
            verification_max_output_tokens=self.verification_max_output_tokens,
            timeout_s=self.model_timeout_s,
        )

    def auxiliary_structured_model_profile(self) -> WorkRunStructuredModelProfile:
        """返回完整 TaskGraph 快照所需的较大物理信封。"""

        return WorkRunStructuredModelProfile(
            attempt_max_output_tokens=self.auxiliary_attempt_max_output_tokens,
            verification_max_output_tokens=(
                self.auxiliary_verification_max_output_tokens
            ),
            timeout_s=self.auxiliary_model_timeout_s,
        )


__all__ = ['TaskGraphWorkRunProfile']
