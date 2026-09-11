import sqlite3

import pytest

from personagraph.session.context import evidence
from personagraph.session.context import reset as context_reset
from personagraph.session.context import store as context_store
from personagraph.session import store as session_store
from personagraph.session.context.models import (
    EvidenceKind,
    ObservationCandidate,
    Operation,
    SessionDomain,
    SourceKind,
)
from personagraph.session.context.reducer import reduce_candidate
from tests.helpers.session_records import append_test_turn


NOW = "2026-07-12T10:00:00+00:00"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _seed_candidate(session_id: str, suffix: str):
    turn_idx = append_test_turn(session_id, "user", f"preference {suffix}")
    evidence_id = evidence.turn_evidence_id(session_id, turn_idx)
    candidate = ObservationCandidate(
        candidate_id=f"reset-candidate-{suffix}",
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key=f"response_depth_{suffix}",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version="test-v1",
    )
    context_store.save_candidates((candidate,))
    transition = reduce_candidate(candidate, None, now=NOW)
    context_store.apply_transition(transition)
    return candidate


def test_clear_removes_only_derived_session_context_state():
    session_id = session_store.create_session("Entelecheia", "clear")
    _seed_candidate(session_id, "first")
    _seed_candidate(session_id, "second")
    evidence.append_event(
        session_id,
        EvidenceKind.TOOL_RESULT,
        "tool:clear-test",
        "retained tool evidence",
    )
    turns_before = session_store.get_turns(session_id)
    evidence_before = evidence.list_evidence(session_id)

    report = context_reset.clear_session_context(
        session_id,
        request_id="clear-request-1",
        actor="user",
        reason="remove derived state",
    )

    assert report["cleared_counts"] == {
        "session_state_items": 2,
        "session_state_transitions": 2,
        "session_observation_candidates": 2,
    }
    assert report["preserved"] == {
        "transcript": True,
        "evidence_events": True,
        "working_memory": True,
    }
    assert "cancelled_promotion_ids" not in report
    assert "retained_committed_promotions" not in report
    assert context_store.list_state_items(session_id) == []
    assert context_store.list_transition_audits(session_id) == []
    assert context_store.list_candidates(session_id) == []
    assert session_store.get_turns(session_id) == turns_before
    assert evidence.list_evidence(session_id) == evidence_before
    replay = context_reset.clear_session_context(
        session_id,
        request_id="clear-request-1",
        actor="user",
        reason="remove derived state",
    )
    assert replay == report
    assert len(context_reset.list_resets(session_id)) == 1


def test_clear_idempotency_collision_is_rejected():
    session_id = session_store.create_session("Entelecheia", "collision")
    context_reset.clear_session_context(
        session_id,
        request_id="same-key",
        actor="user",
        reason="first reason",
    )

    with pytest.raises(context_reset.ContextResetError) as exc:
        context_reset.clear_session_context(
            session_id,
            request_id="same-key",
            actor="user",
            reason="different reason",
        )
    assert exc.value.code == "request_id_collision"
    assert len(context_reset.list_resets(session_id)) == 1


def test_database_failure_rolls_back_deletes_and_reset():
    session_id = session_store.create_session("Entelecheia", "rollback clear")
    _seed_candidate(session_id, "rollback")
    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.execute(
            "CREATE TRIGGER fail_context_clear BEFORE DELETE ON session_state_items"
            " BEGIN SELECT RAISE(ABORT, 'injected clear failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected clear failure"):
        context_reset.clear_session_context(
            session_id,
            request_id="rollback-request",
            actor="user",
            reason="exercise transaction rollback",
        )

    assert len(context_store.list_state_items(session_id)) == 1
    assert len(context_store.list_transition_audits(session_id)) == 1
    assert len(context_store.list_candidates(session_id)) == 1
    assert context_reset.list_resets(session_id) == []
