"""Lane-neutral Runtime Entry processing-route tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.entry.routing.selection import (
    EntryL1ProcessingRoute,
    EntryResponseProcessingRoute,
    classification_requires_l2_task_processing,
    select_entry_processing_route,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)


def _accepted_turn() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="Continue the current task",
        attachment_ids=(),
        window_revision=1,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


@pytest.mark.parametrize(
    ("match", "expected"),
    (
        (SimpleNamespace(match_type="new_root"), True),
        (SimpleNamespace(match_type="existing_root_branch"), True),
        (
            SimpleNamespace(
                match_type="existing_root",
                execute_current=True,
            ),
            True,
        ),
        (
            SimpleNamespace(
                match_type="existing_root",
                execute_current=False,
            ),
            False,
        ),
    ),
)
def test_l2_processing_floor_is_lane_neutral(
    match: object,
    expected: bool,
) -> None:
    classification = SimpleNamespace(task_matches=(match,))

    assert classification_requires_l2_task_processing(classification) is expected


def test_l0_and_l1_routes_do_not_require_a_level_specific_adapter() -> None:
    accepted = _accepted_turn()

    l0_route = select_entry_processing_route(
        accepted=accepted,
        classification=EntryClassification(processing_level="L0"),
        applied=None,
        processing_level="L0",
    )
    l1_route = select_entry_processing_route(
        accepted=accepted,
        classification=EntryClassification(processing_level="L1"),
        applied=None,
        processing_level="L1",
    )

    assert l0_route == EntryResponseProcessingRoute(processing_level="L0")
    assert l1_route == EntryL1ProcessingRoute()


def test_unknown_processing_level_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported processing level"):
        select_entry_processing_route(
            accepted=_accepted_turn(),
            classification=EntryClassification(processing_level="L0"),
            applied=None,
            processing_level="L3",  # type: ignore[arg-type]
        )
