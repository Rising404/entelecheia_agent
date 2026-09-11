from __future__ import annotations

from pathlib import Path

import personagraph.runtime as runtime_package
from personagraph.runtime.l1.model_output_budgets import (
    L1_ATTEMPT_MAX_OUTPUT_TOKENS,
    L1_MAX_OUTPUT_TOKENS,
    L1_SEMANTIC_VERIFIER_MAX_OUTPUT_TOKENS,
    l1_attempt_max_output_tokens,
)


def test_l1_operational_budgets_use_the_lane_ceiling_for_attempts() -> None:
    assert L1_ATTEMPT_MAX_OUTPUT_TOKENS == 65_536
    assert L1_SEMANTIC_VERIFIER_MAX_OUTPUT_TOKENS == 32_768
    assert L1_ATTEMPT_MAX_OUTPUT_TOKENS == L1_MAX_OUTPUT_TOKENS
    assert L1_SEMANTIC_VERIFIER_MAX_OUTPUT_TOKENS < L1_MAX_OUTPUT_TOKENS
    assert l1_attempt_max_output_tokens(thinking_enabled=False) == 65_536
    assert l1_attempt_max_output_tokens(thinking_enabled=True) == 65_536
    assert 44_347 + l1_attempt_max_output_tokens(thinking_enabled=True) < 300_000


def test_runtime_root_no_longer_owns_cross_lane_model_output_budgets() -> None:
    runtime_root = Path(runtime_package.__file__).resolve().parent

    assert not (runtime_root / "model_output_budgets.py").exists()
