"""TaskNode 主机输入限制 DTO 的边界与兼容性检查。"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from personagraph.l2.task_execution.task_node import input_limits as limits


def _assert_required_keyword_only_signature(
    contract: type[object],
    names: tuple[str, ...],
) -> None:
    signature = inspect.signature(contract)
    assert tuple(signature.parameters) == names
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )


def test_input_limit_public_signatures_and_model_configs_are_preserved() -> None:
    _assert_required_keyword_only_signature(
        limits.TaskNodeDependencyInputLimits,
        (
            "profile_id",
            "max_items",
            "max_serialized_utf8_bytes",
        ),
    )
    _assert_required_keyword_only_signature(
        limits.AttemptDecisionInputLimits,
        (
            "profile_id",
            "max_prior_tool_result_items",
            "max_prior_tool_results_serialized_utf8_bytes",
            "dependency_delivery_limits",
            "max_serialized_utf8_bytes",
        ),
    )
    _assert_required_keyword_only_signature(
        limits.NodeVerificationInputLimits,
        (
            "profile_id",
            "max_acceptance_items",
            "max_supporting_tool_result_items",
            "dependency_delivery_limits",
            "max_serialized_utf8_bytes",
        ),
    )

    assert limits.TaskNodeDependencyInputLimits.model_config == {
        "extra": "forbid",
        "frozen": True,
    }
    assert limits.AttemptDecisionInputLimits.model_config == {
        "extra": "forbid",
        "frozen": True,
        "arbitrary_types_allowed": True,
    }
    assert limits.NodeVerificationInputLimits.model_config == {
        "extra": "forbid",
        "frozen": True,
        "arbitrary_types_allowed": True,
    }


def test_input_limit_validation_and_frozen_nested_contract_are_preserved() -> None:
    dependency = limits.TaskNodeDependencyInputLimits(
        profile_id="dependency-limits-v1",
        max_items=1,
        max_serialized_utf8_bytes=2,
    )
    attempt_limits = limits.AttemptDecisionInputLimits(
        profile_id="attempt-limits-v1",
        max_prior_tool_result_items=0,
        max_prior_tool_results_serialized_utf8_bytes=1,
        dependency_delivery_limits=dependency.model_dump(),
        max_serialized_utf8_bytes=3,
    )
    verification_limits = limits.NodeVerificationInputLimits(
        profile_id="verification-limits-v1",
        max_acceptance_items=1,
        max_supporting_tool_result_items=0,
        dependency_delivery_limits=dependency,
        max_serialized_utf8_bytes=4,
    )

    assert isinstance(
        attempt_limits.dependency_delivery_limits,
        limits.TaskNodeDependencyInputLimits,
    )
    assert verification_limits.dependency_delivery_limits is dependency

    with pytest.raises(ValidationError, match="frozen"):
        dependency.max_items = 2
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        limits.TaskNodeDependencyInputLimits(
            profile_id="dependency-limits-v1",
            max_items=1,
            max_serialized_utf8_bytes=2,
            unexpected=True,
        )
    with pytest.raises(
        ValidationError,
        match="Attempt input-limit profile_id must not be blank",
    ):
        limits.AttemptDecisionInputLimits(
            profile_id="  ",
            max_prior_tool_result_items=0,
            max_prior_tool_results_serialized_utf8_bytes=1,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=1,
        )
    with pytest.raises(ValidationError, match="less than or equal to 512"):
        limits.TaskNodeDependencyInputLimits(
            profile_id="dependency-limits-v1",
            max_items=513,
            max_serialized_utf8_bytes=1,
        )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        limits.NodeVerificationInputLimits(
            profile_id="verification-limits-v1",
            max_acceptance_items=0,
            max_supporting_tool_result_items=0,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=1,
        )
