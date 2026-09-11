import pytest

from personagraph.api import router
from personagraph.api.service import ApiError
from personagraph.session.context import store as context_store, evidence
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
    session_id = session_store.create_session("Entelecheia", "inspection api")
    turn_idx = append_test_turn(session_id, "user", "以后回答详细一点")
    candidate = ObservationCandidate(
        candidate_id="inspection-api-candidate",
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
    context_store.apply_transition(
        reduce_candidate(candidate, None, now="2026-07-12T10:00:00+00:00")
    )
    return session_id


def test_inspection_routes_are_session_scoped_and_redacted_by_default():
    session_id = _seed_state()
    view = _route_payload(
        "GET", f"/api/sessions/{session_id}/session-context", {}
    )
    explained = _route_payload(
        "GET",
        f"/api/sessions/{session_id}/session-context/explain"
        "?domain=user&state_type=temporary_preference&key=response_depth",
        {},
    )
    exported = _route_payload(
        "GET", f"/api/sessions/{session_id}/session-context/export", {}
    )

    assert view["session_context"]["counts"]["total"] == 1
    assert view["session_context"]["user_state"][0]["correction_operations"] == [
        "set", "retract", "touch",
    ]
    assert "content_excerpt" not in explained["explanation"]["evidence"][0]
    assert exported["session_context_export"]["content_policy"]["includes_transcript"] is False

    visible = _route_payload(
        "GET",
        f"/api/sessions/{session_id}/session-context/explain"
        "?domain=user&state_type=temporary_preference&key=response_depth"
        "&include_evidence_excerpt=true",
        {},
    )
    assert visible["explanation"]["evidence"][0]["content_excerpt"] == "以后回答详细一点"


def test_inspection_api_rejects_invalid_filters_and_cross_session_lookup():
    session_id = _seed_state()
    other_session = session_store.create_session("Entelecheia", "other")

    with pytest.raises(ApiError) as invalid_bool:
        _route_payload(
            "GET",
            f"/api/sessions/{session_id}/session-context?include_inactive=yes",
            {},
        )
    assert invalid_bool.value.code == "INVALID_BOOLEAN"

    with pytest.raises(ApiError) as invalid_domain:
        _route_payload(
            "GET",
            f"/api/sessions/{session_id}/session-context/explain"
            "?domain=persona&state_type=x&key=y",
            {},
        )
    assert invalid_domain.value.code == "INVALID_SESSION_CONTEXT_DOMAIN"

    with pytest.raises(ApiError) as cross_session:
        _route_payload(
            "GET",
            f"/api/sessions/{other_session}/session-context/explain"
            "?domain=user&state_type=temporary_preference&key=response_depth",
            {},
        )
    assert cross_session.value.code == "SESSION_CONTEXT_STATE_NOT_FOUND"
    assert cross_session.value.status == 404
