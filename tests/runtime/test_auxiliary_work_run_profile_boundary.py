"""冷态 AuxiliaryGraph WorkRun 配置的边界覆盖。"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph.dependency_projection import (
    AuxiliaryDependencyInputLimits,
)
from personagraph.l2.auxiliary_execution.work_run import profile as profiles
from personagraph.l2.task_execution.task_node import input_limits as limits


def _input_limits() -> tuple[
    limits.AttemptDecisionInputLimits,
    limits.NodeVerificationInputLimits,
    AuxiliaryDependencyInputLimits,
]:
    task_dependencies = limits.TaskNodeDependencyInputLimits(
        profile_id="auxiliary-v2-test-task-dependencies-v1",
        max_items=0,
        max_serialized_utf8_bytes=1024,
    )
    return (
        limits.AttemptDecisionInputLimits(
            profile_id="auxiliary-v2-test-attempt-v1",
            max_prior_tool_result_items=1,
            max_prior_tool_results_serialized_utf8_bytes=1024,
            dependency_delivery_limits=task_dependencies,
            max_serialized_utf8_bytes=2048,
        ),
        limits.NodeVerificationInputLimits(
            profile_id="auxiliary-v2-test-verification-v1",
            max_acceptance_items=1,
            max_supporting_tool_result_items=1,
            dependency_delivery_limits=task_dependencies,
            max_serialized_utf8_bytes=2048,
        ),
        AuxiliaryDependencyInputLimits(
            profile_id="auxiliary-v2-test-dependencies-v1",
            max_items=1,
            max_serialized_utf8_bytes=2048,
        ),
    )


def test_profile_preserves_default_limits_and_pydantic_contract() -> None:
    profile = profiles.AuxiliaryWorkRunProfile()

    assert profile.model_config == {
        "extra": "forbid",
        "frozen": True,
        "arbitrary_types_allowed": True,
    }
    assert get_type_hints(profiles.AuxiliaryWorkRunProfile) == {
        "attempt_input_limits": limits.AttemptDecisionInputLimits,
        "verification_input_limits": limits.NodeVerificationInputLimits,
        "dependency_input_limits": AuxiliaryDependencyInputLimits,
        "max_controller_steps": int,
        "execution_findings_enabled": bool,
    }
    assert profile.attempt_input_limits.model_dump() == {
        "profile_id": "auxiliary-v2-standard-attempt-v1",
        "max_prior_tool_result_items": 32,
        "max_prior_tool_results_serialized_utf8_bytes": 128_000,
        "dependency_delivery_limits": {
            "profile_id": "auxiliary-v2-no-task-node-dependencies-v1",
            "max_items": 0,
            "max_serialized_utf8_bytes": 1_024,
        },
        "max_serialized_utf8_bytes": 2_000_000,
    }
    assert profile.verification_input_limits.model_dump() == {
        "profile_id": "auxiliary-v2-standard-verification-v1",
        "max_acceptance_items": 64,
        "max_supporting_tool_result_items": 256,
        "dependency_delivery_limits": {
            "profile_id": "auxiliary-v2-no-task-node-dependencies-v1",
            "max_items": 0,
            "max_serialized_utf8_bytes": 1_024,
        },
        "max_serialized_utf8_bytes": 2_000_000,
    }
    assert profile.dependency_input_limits.model_dump() == {
        "profile_id": "auxiliary-v2-standard-dependencies-v1",
        "max_items": 64,
        "max_serialized_utf8_bytes": 1_000_000,
    }
    assert profile.max_controller_steps == 32
    assert profile.execution_findings_enabled is True
    signature = inspect.signature(profiles.AuxiliaryWorkRunProfile)
    assert tuple(signature.parameters) == (
        "attempt_input_limits",
        "verification_input_limits",
        "dependency_input_limits",
        "max_controller_steps",
        "execution_findings_enabled",
    )
    assert signature.parameters["max_controller_steps"].default == 32
    assert signature.parameters["execution_findings_enabled"].default is True


def test_profile_accepts_direct_limit_contracts_and_remains_fail_closed() -> None:
    attempt, verification, dependencies = _input_limits()
    profile = profiles.AuxiliaryWorkRunProfile(
        attempt_input_limits=attempt,
        verification_input_limits=verification,
        dependency_input_limits=dependencies,
        max_controller_steps=7,
        execution_findings_enabled=False,
    )
    assert profile.attempt_input_limits is attempt
    assert profile.verification_input_limits is verification
    assert profile.dependency_input_limits is dependencies
    assert profile.max_controller_steps == 7
    assert profile.execution_findings_enabled is False

    with pytest.raises(ValidationError, match="frozen"):
        profile.max_controller_steps = 8  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        profiles.AuxiliaryWorkRunProfile(unexpected=True)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        profiles.AuxiliaryWorkRunProfile(max_controller_steps=0)
    with pytest.raises(ValidationError, match="less than or equal to 128"):
        profiles.AuxiliaryWorkRunProfile(max_controller_steps=129)
