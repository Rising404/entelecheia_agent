from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.work_run import (
    AcceptanceProgressErrorCode,
    AcceptanceUpdate,
    TaskNodeSubject,
    initialize_acceptance_progress,
    merge_acceptance_progress,
)


def _subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=3,
        node_id="node-1",
        node_revision=2,
    )


def _snapshot():
    return initialize_acceptance_progress(
        work_run_id="run-1",
        subject=_subject(),
        acceptance_ids=("acceptance-a", "acceptance-b"),
    )


def _merge(snapshot, updates, *, known=("result-1", "result-2"), progress_revision=None,
           current_work_run_revision=4, expected_work_run_revision=4):
    return merge_acceptance_progress(
        snapshot,
        updates,
        known_historical_tool_result_ids=known,
        expected_progress_revision=(
            snapshot.revision if progress_revision is None else progress_revision
        ),
        current_work_run_revision=current_work_run_revision,
        expected_work_run_revision=expected_work_run_revision,
    )


def _codes(result):
    return tuple(issue.code for issue in result.issues)


def test_initial_progress_is_revision_one_all_false_and_empty() -> None:
    snapshot = _snapshot()

    assert snapshot.revision == 1
    assert snapshot.evaluated_output_revision == 1
    assert [item.acceptance_id for item in snapshot.items] == [
        "acceptance-a",
        "acceptance-b",
    ]
    assert all(not item.model_claimed_satisfied for item in snapshot.items)
    assert all(item.supporting_tool_result_ids == () for item in snapshot.items)


def test_initialization_rejects_empty_or_duplicate_acceptance_ids() -> None:
    with pytest.raises(ValueError):
        initialize_acceptance_progress(
            work_run_id="run-1", subject=_subject(), acceptance_ids=()
        )
    with pytest.raises(ValueError):
        initialize_acceptance_progress(
            work_run_id="run-1",
            subject=_subject(),
            acceptance_ids=("acceptance-a", "acceptance-a"),
        )


def test_explicit_items_replace_completely_and_omitted_items_are_preserved() -> None:
    initial = _snapshot()
    first = _merge(
        initial,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-1",),
            ),
        ),
    )

    assert first.status == "applied"
    assert first.changed is True
    assert first.snapshot.revision == 2
    assert first.snapshot.items[0].model_claimed_satisfied is True
    assert first.snapshot.items[0].supporting_tool_result_ids == ("result-1",)
    assert first.snapshot.items[1] == initial.items[1]

    replacement = _merge(
        first.snapshot,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-2",),
            ),
        ),
    )
    assert replacement.snapshot.items[0].supporting_tool_result_ids == ("result-2",)
    assert "result-1" not in replacement.snapshot.items[0].supporting_tool_result_ids


def test_false_update_clears_supporting_results_and_true_may_be_empty() -> None:
    initial = _snapshot()
    satisfied = _merge(
        initial,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-1",),
            ),
        ),
    ).snapshot
    cleared = _merge(
        satisfied,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=False,
                supporting_tool_result_ids=(),
            ),
            AcceptanceUpdate(
                acceptance_id="acceptance-b",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=(),
            ),
        ),
    )

    assert cleared.status == "applied"
    assert cleared.snapshot.items[0].supporting_tool_result_ids == ()
    assert cleared.snapshot.items[0].model_claimed_satisfied is False
    assert cleared.snapshot.items[1].model_claimed_satisfied is True
    assert cleared.snapshot.items[1].supporting_tool_result_ids == ()


def test_empty_or_same_value_update_does_not_increment_revision() -> None:
    snapshot = _snapshot()

    empty = _merge(snapshot, ())
    same = _merge(
        snapshot,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=False,
                supporting_tool_result_ids=(),
            ),
        ),
    )

    assert empty.status == same.status == "applied"
    assert empty.changed is same.changed is False
    assert empty.snapshot is snapshot
    assert same.snapshot is snapshot
    assert empty.snapshot.revision == same.snapshot.revision == 1


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        (
            (
                AcceptanceUpdate(
                    acceptance_id="acceptance-a", model_claimed_satisfied=True
                ),
                AcceptanceUpdate(
                    acceptance_id="acceptance-a", model_claimed_satisfied=False
                ),
            ),
            AcceptanceProgressErrorCode.DUPLICATE_ACCEPTANCE_UPDATE,
        ),
        (
            (
                AcceptanceUpdate(
                    acceptance_id="acceptance-unknown", model_claimed_satisfied=True
                ),
            ),
            AcceptanceProgressErrorCode.UNKNOWN_ACCEPTANCE_ID,
        ),
        (
            (
                # Deliberately bypass the shared proposal guard to retain the
                # merge layer's defense against corrupted internal proposals.
                AcceptanceUpdate.model_construct(
                    acceptance_id="acceptance-a",
                    model_claimed_satisfied=False,
                    supporting_tool_result_ids=("result-1",),
                ),
            ),
            AcceptanceProgressErrorCode.FALSE_UPDATE_HAS_SUPPORTING_RESULTS,
        ),
        (
            (
                AcceptanceUpdate.model_construct(
                    acceptance_id="acceptance-a",
                    model_claimed_satisfied=True,
                    supporting_tool_result_ids=("result-1", "result-1"),
                ),
            ),
            AcceptanceProgressErrorCode.DUPLICATE_SUPPORTING_TOOL_RESULT_ID,
        ),
        (
            (
                AcceptanceUpdate(
                    acceptance_id="acceptance-a",
                    model_claimed_satisfied=True,
                    supporting_tool_result_ids=("not-historical",),
                ),
            ),
            AcceptanceProgressErrorCode.UNKNOWN_SUPPORTING_TOOL_RESULT_ID,
        ),
    ],
)
def test_invalid_batch_is_rejected_atomically(updates, expected_code) -> None:
    snapshot = _snapshot()
    result = _merge(snapshot, updates)

    assert result.status == "rejected"
    assert result.changed is False
    assert result.snapshot is snapshot
    assert result.snapshot.revision == 1
    assert expected_code in _codes(result)


def test_current_attempt_result_is_invalid_until_host_adds_it_to_history() -> None:
    snapshot = _snapshot()
    update = AcceptanceUpdate(
        acceptance_id="acceptance-a",
        model_claimed_satisfied=True,
        supporting_tool_result_ids=("current-attempt-result",),
    )

    rejected = _merge(snapshot, (update,), known=("result-1",))
    applied = _merge(snapshot, (update,), known=("current-attempt-result",))

    assert rejected.status == "rejected"
    assert AcceptanceProgressErrorCode.UNKNOWN_SUPPORTING_TOOL_RESULT_ID in _codes(
        rejected
    )
    assert applied.status == "applied"


def test_revision_conflicts_fail_before_any_merge() -> None:
    snapshot = _snapshot()
    updates = (
        AcceptanceUpdate(
            acceptance_id="acceptance-a",
            model_claimed_satisfied=True,
            supporting_tool_result_ids=("result-1",),
        ),
    )

    result = _merge(
        snapshot,
        updates,
        progress_revision=99,
        current_work_run_revision=5,
        expected_work_run_revision=4,
    )

    assert result.status == "rejected"
    assert _codes(result) == (
        AcceptanceProgressErrorCode.PROGRESS_REVISION_CONFLICT,
        AcceptanceProgressErrorCode.WORK_RUN_REVISION_CONFLICT,
    )
    assert result.error_codes == (
        AcceptanceProgressErrorCode.PROGRESS_REVISION_CONFLICT,
        AcceptanceProgressErrorCode.WORK_RUN_REVISION_CONFLICT,
    )
    assert result.snapshot is snapshot


def test_one_invalid_update_prevents_other_valid_updates_from_applying() -> None:
    snapshot = _snapshot()
    result = _merge(
        snapshot,
        (
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-1",),
            ),
            AcceptanceUpdate(
                acceptance_id="unknown",
                model_claimed_satisfied=True,
            ),
        ),
    )

    assert result.status == "rejected"
    assert result.snapshot is snapshot
    assert snapshot.items[0].model_claimed_satisfied is False


def test_merge_result_and_snapshot_are_frozen() -> None:
    snapshot = _snapshot()
    result = _merge(snapshot, ())

    with pytest.raises(ValidationError):
        result.changed = True
    with pytest.raises(ValidationError):
        snapshot.revision = 10
