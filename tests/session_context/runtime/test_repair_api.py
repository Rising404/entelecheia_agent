import pytest

from personagraph.api import router
from personagraph.api.service import ApiError
from personagraph.api.service import session_context as session_context_service
from personagraph.session.context import store as context_store, evidence, extractor
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
    session_id = session_store.create_session("Entelecheia", "repair api")
    turn_idx = append_test_turn(session_id, "user", "answer in detail")
    evidence_id = evidence.turn_evidence_id(session_id, turn_idx)
    record = evidence.get_evidence(evidence_id)
    candidate = ObservationCandidate(
        candidate_id="repair-api-candidate",
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version=extractor.EXTRACTOR_VERSION,
        valid_from=record.created_at,
    )
    context_store.save_candidates((candidate,))
    context_store.apply_transition(
        reduce_candidate(candidate, None, now="2026-07-12T10:00:00+00:00")
    )
    return session_id


def _correction_path(session_id: str) -> str:
    return f"/api/sessions/{session_id}/session-context/corrections"


def _correction_payload(value="brief"):
    return {
        "confirm": True,
        "domain": "user",
        "state_type": "temporary_preference",
        "key": "response_depth",
        "operation": "set",
        "value": value,
    }


def test_correction_requires_literal_confirmation_and_catalog_validity():
    session_id = _seed_state()
    with pytest.raises(ApiError) as missing_confirm:
        _route_payload(
            "POST", _correction_path(session_id),
            {**_correction_payload(), "confirm": "true"},
        )
    assert missing_confirm.value.code == "INVALID_REQUEST_FIELD"
    assert missing_confirm.value.details == {"field": "confirm", "expected": "bool"}

    with pytest.raises(ApiError) as invalid_type:
        _route_payload(
            "POST", _correction_path(session_id),
            {**_correction_payload(), "state_type": "invented_state"},
        )
    assert invalid_type.value.code == "INVALID_SESSION_CONTEXT_TYPE"
    assert all(record.kind.value != "correction" for record in evidence.list_evidence(session_id))


def test_correction_records_evidence_and_returns_detailed_dry_run_only():
    session_id = _seed_state()
    before = context_store.get_state_item(
        session_id, SessionDomain.USER, "temporary_preference", "response_depth"
    )

    payload = _route_payload(
        "POST", _correction_path(session_id), _correction_payload()
    )

    assert payload["correction_evidence"]["kind"] == "correction"
    assert payload["correction_evidence"]["metadata"]["actor"] == "api-user"
    assert payload["repair"]["dry_run"] is True
    assert payload["repair"]["applied"] is False
    assert payload["repair"]["preview_token"].startswith("repair-preview:")
    assert payload["repair"]["changes"]["changed"][0]["before"]["value_json"] == "detailed"
    assert payload["repair"]["changes"]["changed"][0]["after"]["value_json"] == "brief"
    assert payload["apply_enabled"] is False
    after = context_store.get_state_item(
        session_id, SessionDomain.USER, "temporary_preference", "response_depth"
    )
    assert after == before


def test_exact_correction_retry_is_idempotent_and_keeps_preview_token():
    session_id = _seed_state()
    first = _route_payload("POST", _correction_path(session_id), _correction_payload())
    second = _route_payload("POST", _correction_path(session_id), _correction_payload())
    correction_ids = [
        record.id for record in evidence.list_evidence(session_id)
        if record.kind.value == "correction"
    ]
    assert correction_ids == [first["correction_evidence"]["id"]]
    assert second["correction_evidence"]["id"] == first["correction_evidence"]["id"]
    assert second["repair"]["preview_token"] == first["repair"]["preview_token"]


def test_apply_is_flag_gated_then_uses_preview_token_when_enabled(monkeypatch):
    session_id = _seed_state()
    correction = _route_payload("POST", _correction_path(session_id), _correction_payload())
    apply_path = f"/api/sessions/{session_id}/session-context/repair/apply"
    body = {
        "confirm": True,
        "preview_token": correction["repair"]["preview_token"],
        "domain": "user",
        "state_type": "temporary_preference",
        "key": "response_depth",
    }

    with pytest.raises(ApiError) as disabled:
        _route_payload("POST", apply_path, body)
    assert disabled.value.code == "SESSION_CONTEXT_REPAIR_APPLY_DISABLED"
    assert disabled.value.status == 403

    monkeypatch.setattr(
        session_context_service,
        "load_features",
        lambda _path: {"session_context_repair_apply_enabled": True},
    )
    applied = _route_payload("POST", apply_path, body)
    assert applied["repair"]["applied"] is True
    assert applied["repair"]["already_applied"] is False
    assert applied["repair"]["applied_revision"] > applied["repair"]["context_revision"]
    item = context_store.get_state_item(
        session_id, SessionDomain.USER, "temporary_preference", "response_depth"
    )
    assert item.value_json == "brief"

    monkeypatch.setattr(
        session_context_service,
        "load_features",
        lambda _path: {"session_context_repair_apply_enabled": False},
    )
    replay = _route_payload("POST", apply_path, body)
    assert replay["repair"]["already_applied"] is True
    assert replay["repair"]["applied_revision"] == applied["repair"]["applied_revision"]
    assert context_store.get_repair_apply(body["preview_token"])["session_id"] == session_id

    with pytest.raises(ApiError) as collision:
        _route_payload("POST", apply_path, {
            **body,
            "key": "different_slot",
        })
    assert collision.value.code == "SESSION_CONTEXT_REPAIR_INVALID"
    assert collision.value.details["reason"] == "preview_token_scope_collision"


def test_apply_rejects_stale_preview_without_replacing(monkeypatch):
    session_id = _seed_state()
    correction = _route_payload("POST", _correction_path(session_id), _correction_payload())
    append_test_turn(session_id, "user", "new evidence after preview")
    monkeypatch.setattr(
        session_context_service,
        "load_features",
        lambda _path: {"session_context_repair_apply_enabled": True},
    )

    with pytest.raises(ApiError) as stale:
        _route_payload(
            "POST",
            f"/api/sessions/{session_id}/session-context/repair/apply",
            {
                "confirm": True,
                "preview_token": correction["repair"]["preview_token"],
                "domain": "user",
                "state_type": "temporary_preference",
                "key": "response_depth",
            },
        )
    assert stale.value.code == "SESSION_CONTEXT_REPAIR_PREVIEW_STALE"
    assert stale.value.status == 409
    item = context_store.get_state_item(
        session_id, SessionDomain.USER, "temporary_preference", "response_depth"
    )
    assert item.value_json == "detailed"


def test_product_preview_refuses_model_reextract_and_incomplete_slot():
    session_id = session_store.create_session("Entelecheia", "empty repair")
    path = f"/api/sessions/{session_id}/session-context/repair/preview"
    with pytest.raises(ApiError) as force:
        _route_payload("POST", path, {"force_reextract": True})
    assert force.value.code == "SESSION_CONTEXT_REEXTRACT_NOT_ALLOWED"

    with pytest.raises(ApiError) as incomplete:
        _route_payload("POST", path, {"domain": "user"})
    assert incomplete.value.code == "INCOMPLETE_SESSION_CONTEXT_SLOT"
