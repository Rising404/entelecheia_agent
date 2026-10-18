"""Lane-neutral Runtime Entry processing-route tests."""

from __future__ import annotations


import pytest

from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.entry.routing.selection import (
    EntryL1ProcessingRoute,
    EntryResponseProcessingRoute,
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


def test_l0_and_l1_routes_do_not_require_a_level_specific_adapter() -> None:
    accepted = _accepted_turn()

    l0_route = select_entry_processing_route(
        accepted=accepted,
        classification=EntryClassification(processing_level="L0"),
        processing_level="L0",
    )
    l1_route = select_entry_processing_route(
        accepted=accepted,
        classification=EntryClassification(processing_level="L1"),
        processing_level="L1",
    )

    assert l0_route == EntryResponseProcessingRoute(processing_level="L0")
    assert l1_route == EntryL1ProcessingRoute()


def test_unknown_processing_level_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="processing route differs"):
        select_entry_processing_route(
            accepted=_accepted_turn(),
            classification=EntryClassification(processing_level="L0"),
                processing_level="L3",  # type: ignore[arg-type]
        )
