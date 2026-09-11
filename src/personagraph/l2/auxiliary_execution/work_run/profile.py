"""延迟的 Host 输入限制配置文件用于一个 AuxiliaryGraph WorkRun。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.auxiliary_graph.dependency_projection import (
    AuxiliaryDependencyInputLimits,
)
from personagraph.l2.task_execution.task_node.input_limits import (
    AttemptDecisionInputLimits,
    NodeVerificationInputLimits,
    TaskNodeDependencyInputLimits,
)


class _ProfileContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AuxiliaryWorkRunProfile(_ProfileContract):
    """Host 组件拥有的标准信封用于一个 模型节点控制器运行。"""

    attempt_input_limits: AttemptDecisionInputLimits = Field(
        default_factory=lambda: AttemptDecisionInputLimits(
            profile_id="auxiliary-v2-standard-attempt-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="auxiliary-v2-no-task-node-dependencies-v1",
                max_items=0,
                max_serialized_utf8_bytes=1_024,
            ),
            max_serialized_utf8_bytes=2_000_000,
        )
    )
    verification_input_limits: NodeVerificationInputLimits = Field(
        default_factory=lambda: NodeVerificationInputLimits(
            profile_id="auxiliary-v2-standard-verification-v1",
            max_acceptance_items=64,
            max_supporting_tool_result_items=256,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="auxiliary-v2-no-task-node-dependencies-v1",
                max_items=0,
                max_serialized_utf8_bytes=1_024,
            ),
            max_serialized_utf8_bytes=2_000_000,
        )
    )
    dependency_input_limits: AuxiliaryDependencyInputLimits = Field(
        default_factory=lambda: AuxiliaryDependencyInputLimits(
            profile_id="auxiliary-v2-standard-dependencies-v1",
            max_items=64,
            max_serialized_utf8_bytes=1_000_000,
        )
    )
    max_controller_steps: int = Field(default=32, ge=1, le=128)
    execution_findings_enabled: bool = True


__all__ = ["AuxiliaryWorkRunProfile"]
