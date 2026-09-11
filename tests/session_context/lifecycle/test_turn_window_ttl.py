from dataclasses import replace

import pytest

from personagraph.session.context import lifecycle
from personagraph.session import store as session_store
from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    ReasonCode,
    SessionDomain,
    SourceKind,
    StateStatus,
)
from personagraph.session.context.reducer import reduce_candidate


NOW = "2026-07-12T10:00:00+00:00"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _item(state_type: str, *, evidence_id: str = "turn:session-a:0"):
    candidate = ObservationCandidate(
        candidate_id=f"candidate-{state_type}",
        session_id="session-a",
        domain=SessionDomain.USER,
        state_type=state_type,
        key="signal",
        proposed_value="value",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version="test-v1",
    )
    return reduce_candidate(candidate, None, now=NOW).state_item


def test_turn_window_expires_only_after_configured_completed_turns():
    item = _item("affect_signal")
    markers = [(index, NOW) for index in (3, 7, 20, 21, 50, 99)]
    before = lifecycle.sweep_turn_windows(
        [item], user_turn_markers=markers[:-1]
    )
    due = lifecycle.sweep_turn_windows(
        [item], user_turn_markers=markers
    )

    assert before.transitions == ()
    assert before.not_due_count == 1
    assert due.transitions[0].state_item.status == StateStatus.EXPIRED
    assert due.transitions[0].audit.reason_code == ReasonCode.EXPIRED


def test_expiry_transition_id_is_stable_and_new_evidence_refreshes_window():
    original = _item("affect_signal")
    refreshed = replace(original, derived_from=(*original.derived_from, "turn:session-a:10"))
    markers = [(index, NOW) for index in (2, 4, 6, 8, 10, 12)]
    first = lifecycle.sweep_turn_windows([original], user_turn_markers=markers)
    replay = lifecycle.sweep_turn_windows([original], user_turn_markers=markers)
    refreshed_result = lifecycle.sweep_turn_windows(
        [refreshed], user_turn_markers=markers
    )

    assert first.transitions[0].audit.transition_id == replay.transitions[0].audit.transition_id
    assert refreshed_result.transitions == ()


def test_session_lifetime_and_missing_turn_evidence_do_not_guess_expiry():
    session_item = _item("temporary_preference")
    unknown_origin = _item("affect_signal", evidence_id="event:unknown")
    sweep = lifecycle.sweep_turn_windows(
        [session_item, unknown_origin],
        user_turn_markers=[(index, NOW) for index in range(2, 102, 2)],
    )

    assert sweep.transitions == ()
    assert sweep.checked_count == 1
    assert sweep.unevaluable_count == 1
