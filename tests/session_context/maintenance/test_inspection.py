import sqlite3

import pytest

from personagraph.session.context import evidence, inspection
from personagraph.session.context import reset as context_reset
from personagraph.session.context import store as context_store
from personagraph.session import store as session_store
from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    SessionDomain,
    SourceKind,
)
from personagraph.session.context.reducer import reduce_candidate
from tests.helpers.session_records import append_test_turn


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _seed_state():
    session_id = session_store.create_session("Entelecheia", "inspection")
    turn_idx = append_test_turn(session_id, "user", "以后回答详细一点")
    evidence_id = evidence.turn_evidence_id(session_id, turn_idx)
    candidate = ObservationCandidate(
        candidate_id="inspection-candidate",
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version="test-v1",
    )
    context_store.save_candidates((candidate,))
    transition = reduce_candidate(candidate, None, now="2026-07-12T10:00:00+00:00")
    context_store.apply_transition(transition)
    return session_id, evidence_id


def test_view_and_explanation_are_read_only_and_excerpt_is_opt_in():
    session_id, evidence_id = _seed_state()
    before = {
        "candidates": len(context_store.list_candidates(session_id)),
        "states": len(context_store.list_state_items(session_id)),
        "transitions": len(context_store.list_transition_audits(session_id)),
    }

    view = inspection.view_session_context(session_id)
    redacted = inspection.explain_state(
        session_id,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
    )
    visible = inspection.explain_state(
        session_id,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
        include_evidence_excerpt=True,
    )

    assert view["counts"] == {
        "user_state": 1,
        "task_state": 0,
        "interaction_state": 0,
        "total": 1,
    }
    assert redacted["explanation"]["support_status"] == "complete"
    assert redacted["explanation"]["evidence"][0]["id"] == evidence_id
    assert "content_excerpt" not in redacted["explanation"]["evidence"][0]
    assert visible["explanation"]["evidence"][0]["content_excerpt"] == "以后回答详细一点"
    after = {
        "candidates": len(context_store.list_candidates(session_id)),
        "states": len(context_store.list_state_items(session_id)),
        "transitions": len(context_store.list_transition_audits(session_id)),
    }
    assert after == before


def test_export_is_versioned_and_omits_raw_transcript_and_excerpt_by_default():
    session_id, _evidence_id = _seed_state()

    exported = inspection.export_session_context(session_id)

    assert exported["schema_version"] == 3
    assert exported["content_policy"] == {
        "includes_transcript": False,
        "includes_evidence_excerpt": False,
        "evidence_metadata_included": True,
    }
    assert exported["counts"] == {
        "state_items": 1,
        "candidates": 1,
        "transitions": 1,
        "resets": 0,
        "evidence": 1,
    }
    assert "promotions" not in exported
    assert all("content_excerpt" not in item for item in exported["evidence"])
    assert exported["state_items"][0]["value_json"] == "detailed"


def test_explanation_reports_missing_exact_evidence_without_guessing():
    session_id, evidence_id = _seed_state()
    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.execute("DELETE FROM session_turns WHERE session_id=?", (session_id,))

    explained = inspection.explain_state(
        session_id,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
    )

    assert explained["explanation"]["support_status"] == "incomplete"
    assert explained["explanation"]["missing_evidence_ids"] == [evidence_id]
    assert explained["explanation"]["evidence"] == [{"id": evidence_id, "status": "missing"}]


def test_export_includes_reset_audit_while_retaining_redacted_evidence():
    session_id, evidence_id = _seed_state()
    report = context_reset.clear_session_context(
        session_id,
        request_id="inspection-clear",
        actor="user",
        reason="start fresh",
    )

    view = inspection.view_session_context(session_id)
    exported = inspection.export_session_context(session_id)

    assert view["counts"]["total"] == 0
    assert view["latest_reset"]["reset_id"] == report["reset_id"]
    assert exported["schema_version"] == 3
    assert exported["counts"]["resets"] == 1
    assert exported["resets"][0]["cutoff_turn_idx"] == 0
    assert exported["evidence"][0]["id"] == evidence_id
    assert "content_excerpt" not in exported["evidence"][0]
