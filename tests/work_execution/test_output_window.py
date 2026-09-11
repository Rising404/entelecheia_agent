from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.work_run import (
    AcceptanceProgressErrorCode,
    AcceptanceUpdate,
    HostMaterializedSubmitOutputWindowAction,
    HostMaterializedWriteOutputWindowAction,
    OutputWindowFormat,
    OutputWindow,
    SubmitOutputWindowAction,
    TaskNodeSubject,
    WriteOutputWindowAction,
    apply_output_window_action,
    initialize_acceptance_progress,
    initialize_output_window,
)


def _subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=1,
        node_id="node-1",
        node_revision=1,
    )


def _state():
    window = initialize_output_window(work_run_id="run-1", updated_turn_id="turn-1")
    progress = initialize_acceptance_progress(
        work_run_id="run-1",
        subject=_subject(),
        acceptance_ids=("acceptance-a", "acceptance-b"),
    )
    return window, progress


def _apply(window, progress, action, *, updates=(), known=("result-1",)):
    return apply_output_window_action(
        window,
        progress,
        action,
        acceptance_updates=updates,
        updated_turn_id="turn-1",
        updated_attempt_id="attempt-1",
        known_historical_tool_result_ids=known,
        expected_progress_revision=progress.revision,
        current_work_run_revision=4,
        expected_work_run_revision=4,
    )


def test_canonical_empty_window_and_progress_share_revision_one() -> None:
    window, progress = _state()

    assert window == OutputWindow(
        work_run_id="run-1",
        updated_turn_id="turn-1",
    )
    assert window.output_revision == progress.evaluated_output_revision == 1
    assert window.format is OutputWindowFormat.PLAIN_TEXT
    assert window.content == ""
    assert window.updated_attempt_id is None

    with pytest.raises(ValidationError):
        OutputWindow(
            work_run_id="run-1",
            updated_turn_id="turn-1",
            content="not canonical",
        )


def test_submit_requires_nonempty_non_whitespace_content() -> None:
    for content in ("", "  \n"):
        with pytest.raises(ValidationError):
            SubmitOutputWindowAction(
                content=content,
                format=OutputWindowFormat.PLAIN_TEXT,
            )


def test_real_window_change_resets_then_applies_same_decision_updates() -> None:
    window, initial = _state()
    seeded = _apply(
        window,
        initial,
        WriteOutputWindowAction(
            content="draft one",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
        updates=(
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-1",),
            ),
        ),
    )
    assert seeded.status == "applied"
    assert seeded.output_changed is True
    assert seeded.output_window.output_revision == 2
    assert seeded.progress_merge.snapshot.evaluated_output_revision == 2
    assert seeded.progress_merge.snapshot.revision == 2
    assert seeded.progress_merge.snapshot.items[0].model_claimed_satisfied is True

    changed = _apply(
        seeded.output_window,
        seeded.progress_merge.snapshot,
        WriteOutputWindowAction(
            content="draft two",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
        updates=(
            AcceptanceUpdate(
                acceptance_id="acceptance-b",
                model_claimed_satisfied=True,
            ),
        ),
    )

    assert changed.output_window.output_revision == 3
    assert changed.progress_merge.snapshot.revision == 3
    assert changed.progress_merge.snapshot.evaluated_output_revision == 3
    assert changed.progress_merge.snapshot.items[0].model_claimed_satisfied is False
    assert changed.progress_merge.snapshot.items[0].supporting_tool_result_ids == ()
    assert changed.progress_merge.snapshot.items[1].model_claimed_satisfied is True


def test_real_change_advances_progress_even_when_final_items_are_identical() -> None:
    window, progress = _state()
    result = _apply(
        window,
        progress,
        WriteOutputWindowAction(
            content="new body",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
    )

    assert result.output_changed is True
    assert result.progress_merge.changed is True
    assert result.progress_merge.snapshot.items == progress.items
    assert result.progress_merge.snapshot.revision == progress.revision + 1


def test_exact_format_and_content_match_is_a_window_noop() -> None:
    window, progress = _state()
    first = _apply(
        window,
        progress,
        WriteOutputWindowAction(
            content="same\n",
            format=OutputWindowFormat.MARKDOWN,
        ),
    )
    no_op = _apply(
        first.output_window,
        first.progress_merge.snapshot,
        WriteOutputWindowAction(
            content="same\n",
            format=OutputWindowFormat.MARKDOWN,
        ),
    )

    assert no_op.output_changed is False
    assert no_op.output_window is first.output_window
    assert no_op.progress_merge.changed is False
    assert no_op.progress_merge.snapshot is first.progress_merge.snapshot

    format_change = _apply(
        first.output_window,
        first.progress_merge.snapshot,
        WriteOutputWindowAction(
            content="same\n",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
    )
    assert format_change.output_changed is True
    assert format_change.output_window.output_revision == 3


def test_materialized_reference_has_utf8_size_and_never_copies_content() -> None:
    window, progress = _state()
    applied = _apply(
        window,
        progress,
        WriteOutputWindowAction(
            content="你好",
            format=OutputWindowFormat.MARKDOWN,
        ),
    )

    assert isinstance(
        applied.materialized_action, HostMaterializedWriteOutputWindowAction
    )
    assert applied.materialized_action.size_bytes == len("你好".encode("utf-8"))
    assert "content" not in type(applied.materialized_action).model_fields


def test_submit_is_atomic_and_locks_the_exact_materialized_revision() -> None:
    window, progress = _state()
    submit = SubmitOutputWindowAction(
        content="final",
        format=OutputWindowFormat.PLAIN_TEXT,
    )

    rejected = _apply(
        window,
        progress,
        submit,
        updates=(
            AcceptanceUpdate(
                acceptance_id="acceptance-a",
                model_claimed_satisfied=True,
            ),
        ),
    )
    assert rejected.status == "rejected"
    assert rejected.output_window is window
    assert rejected.progress_merge.snapshot is progress
    assert rejected.materialized_action is None
    assert rejected.progress_merge.error_codes == (
        AcceptanceProgressErrorCode.SUBMIT_REQUIRES_ALL_ACCEPTANCES_SATISFIED,
    )

    applied = _apply(
        window,
        progress,
        submit,
        updates=tuple(
            AcceptanceUpdate(
                acceptance_id=acceptance_id,
                model_claimed_satisfied=True,
            )
            for acceptance_id in ("acceptance-a", "acceptance-b")
        ),
    )
    assert applied.status == "applied"
    assert isinstance(
        applied.materialized_action, HostMaterializedSubmitOutputWindowAction
    )
    assert (
        applied.materialized_action.output_revision
        == applied.output_window.output_revision
        == applied.progress_merge.snapshot.evaluated_output_revision
        == 2
    )


def test_invalid_update_rejects_window_and_progress_together() -> None:
    window, progress = _state()
    result = _apply(
        window,
        progress,
        WriteOutputWindowAction(
            content="candidate",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
        updates=(
            AcceptanceUpdate(
                acceptance_id="unknown",
                model_claimed_satisfied=True,
            ),
        ),
    )

    assert result.status == "rejected"
    assert result.output_window is window
    assert result.output_changed is False
    assert result.progress_merge.snapshot is progress


def test_stale_progress_output_binding_fails_closed() -> None:
    window, progress = _state()
    stale = progress.model_copy(update={"evaluated_output_revision": 2})

    result = _apply(
        window,
        stale,
        WriteOutputWindowAction(
            content="candidate",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
    )

    assert result.status == "rejected"
    assert result.progress_merge.error_codes == (
        AcceptanceProgressErrorCode.EVALUATED_OUTPUT_REVISION_CONFLICT,
    )
