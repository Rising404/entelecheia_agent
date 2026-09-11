"""WorkRun 模型限制契约的边界与兼容性检查。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from personagraph.l2.task_execution.work_run import model_profile as profile_contract


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "attempt_max_output_tokens": 0,
                "verification_max_output_tokens": 1,
                "timeout_s": 1.0,
            },
            "attempt_max_output_tokens must be positive",
        ),
        (
            {
                "attempt_max_output_tokens": 1,
                "verification_max_output_tokens": 0,
                "timeout_s": 1.0,
            },
            "verification_max_output_tokens must be positive",
        ),
        (
            {
                "attempt_max_output_tokens": 1,
                "verification_max_output_tokens": 1,
                "timeout_s": 0.0,
            },
            "timeout_s must be positive",
        ),
    ],
)
def test_model_profile_preserves_validation_and_frozen_contract(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        profile_contract.WorkRunStructuredModelProfile(**kwargs)

    profile = profile_contract.WorkRunStructuredModelProfile(
        attempt_max_output_tokens=1,
        verification_max_output_tokens=2,
        timeout_s=3.0,
    )
    with pytest.raises(FrozenInstanceError):
        profile.timeout_s = 4.0
