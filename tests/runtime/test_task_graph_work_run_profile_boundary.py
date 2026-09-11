"""不可变 TaskGraph WorkRun 配置契约的边界检查。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.task_execution.task_graph import profile as profiles
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS
from personagraph.l2.task_execution.work_run.model_profile import WorkRunStructuredModelProfile


def test_task_graph_profile_preserves_default_input_and_provider_limits() -> None:
    profile = profiles.TaskGraphWorkRunProfile()

    assert profile.attempt_input_limits.model_dump() == {
        "profile_id": "demo-task-graph-attempt-v1",
        "max_prior_tool_result_items": 32,
        "max_prior_tool_results_serialized_utf8_bytes": 128_000,
        "dependency_delivery_limits": {
            "profile_id": "demo-task-graph-attempt-dependencies-v1",
            "max_items": 63,
            "max_serialized_utf8_bytes": 256_000,
        },
        "max_serialized_utf8_bytes": 1_000_000,
    }
    assert profile.verification_input_limits.model_dump() == {
        "profile_id": "demo-task-graph-verification-v1",
        "max_acceptance_items": 64,
        "max_supporting_tool_result_items": 256,
        "dependency_delivery_limits": {
            "profile_id": "demo-task-graph-verification-dependencies-v1",
            "max_items": 63,
            "max_serialized_utf8_bytes": 256_000,
        },
        "max_serialized_utf8_bytes": 1_000_000,
    }
    assert profile.max_work_runs_per_turn == 32
    assert profile.execution_findings_enabled is True
    assert profile.structured_model_profile() == WorkRunStructuredModelProfile(
        attempt_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
        verification_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
        timeout_s=45.0,
    )
    assert profile.auxiliary_structured_model_profile() == WorkRunStructuredModelProfile(
        attempt_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
        verification_max_output_tokens=L2_MAX_OUTPUT_TOKENS,
        timeout_s=180.0,
    )

    customized = profiles.TaskGraphWorkRunProfile(
        attempt_max_output_tokens=17,
        verification_max_output_tokens=19,
        model_timeout_s=23.5,
        auxiliary_attempt_max_output_tokens=29,
        auxiliary_verification_max_output_tokens=31,
        auxiliary_model_timeout_s=37.5,
        execution_findings_enabled=False,
    )
    assert customized.structured_model_profile() == WorkRunStructuredModelProfile(
        attempt_max_output_tokens=17,
        verification_max_output_tokens=19,
        timeout_s=23.5,
    )
    assert customized.auxiliary_structured_model_profile() == (
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=29,
            verification_max_output_tokens=31,
            timeout_s=37.5,
        )
    )
    assert customized.execution_findings_enabled is False

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        profiles.TaskGraphWorkRunProfile(attempt_max_output_tokens=0)
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        profiles.TaskGraphWorkRunProfile(max_work_runs_per_turn=65)
