"""精确 WorkRun 变更重放回执的契约覆盖。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.work_run import (
    WorkExecutionMutationResult,
    WorkRunStatus,
)


def _receipt(**overrides: object) -> WorkExecutionMutationResult:
    values: dict[str, object] = {
        "status": "replayed",
        "work_run_id": "work-run-1",
        "work_run_revision": 2,
        "work_run_status": WorkRunStatus.ACTIVE,
        "acceptance_progress_revision": 3,
        "output_window_revision": 5,
        "window_state_version": 4,
        "budget_transition": None,
    }
    values.update(overrides)
    return WorkExecutionMutationResult(**values)


def test_mutation_receipt_preserves_the_exact_replay_projection_shape() -> None:
    receipt = _receipt()

    assert tuple(WorkExecutionMutationResult.model_fields) == (
        "status",
        "work_run_id",
        "work_run_revision",
        "work_run_status",
        "work_run_reason",
        "current_attempt_id",
        "acceptance_progress_revision",
        "output_window_revision",
        "attempt",
        "new_tool_call_ids",
        "tool_result_id",
        "turn_work_run_link_revision",
        "window_state_version",
        "task_state_version",
        "node_state_version",
        "budget_transition",
    )
    assert receipt.output_window_revision == 5
    assert receipt.budget_transition is None
    assert receipt.model_dump(mode="json")["work_run_status"] == "active"


def test_mutation_receipt_remains_frozen_and_rejects_unknown_fields() -> None:
    receipt = _receipt()

    with pytest.raises(ValidationError):
        receipt.work_run_id = "work-run-2"
    with pytest.raises(ValidationError):
        _receipt(unknown="not-a-receipt-field")


@pytest.mark.parametrize(
    "required_field",
    ("output_window_revision", "budget_transition"),
)
def test_mutation_receipt_rejects_missing_current_fields(
    required_field: str,
) -> None:
    values = _receipt().model_dump()
    values.pop(required_field)

    with pytest.raises(ValidationError):
        WorkExecutionMutationResult.model_validate(values)
