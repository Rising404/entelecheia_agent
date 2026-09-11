import pytest

from personagraph.api import router
from personagraph.api.service import ApiError
from personagraph.session.context import evidence
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


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _seed_state():
    session_id = session_store.create_session("Entelecheia", "clear api")
    turn_idx = append_test_turn(session_id, "user", "answer in detail")
    candidate = ObservationCandidate(
        candidate_id="clear-api-candidate",
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence.turn_evidence_id(session_id, turn_idx),),
        extractor_version="test-v1",
    )
    context_store.save_candidates((candidate,))
    context_store.apply_transition(
        reduce_candidate(candidate, None, now="2026-07-12T10:00:00+00:00")
    )
    return session_id


def test_clear_api_requires_literal_confirmation_and_idempotency_fields():
    session_id = _seed_state()
    path = f"/api/sessions/{session_id}/session-context/clear"

    with pytest.raises(ApiError) as missing_confirm:
        _route_payload("POST", path, {"request_id": "r1", "reason": "clear"})
    assert missing_confirm.value.code == "SESSION_CONTEXT_CLEAR_CONFIRMATION_REQUIRED"
    assert missing_confirm.value.status == 409

    with pytest.raises(ApiError) as string_confirm:
        _route_payload(
            "POST", path, {"confirm": "true", "request_id": "r1", "reason": "clear"}
        )
    assert string_confirm.value.code == "SESSION_CONTEXT_CLEAR_CONFIRMATION_REQUIRED"

    with pytest.raises(ApiError) as missing_request:
        _route_payload("POST", path, {"confirm": True, "reason": "clear"})
    assert missing_request.value.code == "MISSING_FIELD"
    assert missing_request.value.details == {"field": "request_id"}


def test_clear_api_is_session_scoped_idempotent_and_returns_empty_view():
    session_id = _seed_state()
    other_session = session_store.create_session("Entelecheia", "other")
    path = f"/api/sessions/{session_id}/session-context/clear"
    payload = {"confirm": True, "request_id": "api-clear-1", "reason": "start fresh"}

    first = _route_payload("POST", path, payload)
    replay = _route_payload("POST", path, payload)

    assert replay == first
    assert first["session_context"]["counts"]["total"] == 0
    assert first["session_context_clear"]["preserved"]["transcript"] is True
    assert len(session_store.get_turns(session_id)) == 1
    assert context_store.list_state_items(other_session) == []

    with pytest.raises(ApiError) as collision:
        _route_payload(
            "POST",
            path,
            {"confirm": True, "request_id": "api-clear-1", "reason": "changed"},
        )
    assert collision.value.code == "SESSION_CONTEXT_CLEAR_IDEMPOTENCY_CONFLICT"
    assert collision.value.status == 409


def test_clear_api_rejects_unknown_session_before_writing():
    with pytest.raises(ApiError) as missing:
        _route_payload(
            "POST",
            "/api/sessions/missing/session-context/clear",
            {"confirm": True, "request_id": "r1", "reason": "clear"},
        )
    assert missing.value.code == "SESSION_NOT_FOUND"
    assert missing.value.status == 404
