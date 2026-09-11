from __future__ import annotations

from datetime import datetime, timezone

import pytest

from personagraph.configuration import features as config
from personagraph.api import router, service
from personagraph.runtime.turn_events import RuntimeStage, TurnEventStatus, TurnEvent
from personagraph.session import store as session_store


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    session_store._INITIALIZED_PATHS.clear()
    yield
    session_store._INITIALIZED_PATHS.clear()


def _new_runtime_event(event_id: str, session_id: str, *, turn_id: str = "turn-events") -> TurnEvent:
    session_store.create_runtime_turn(
        turn_id=turn_id,
        session_id=session_id,
        source="runtime_test",
        user_text="test event",
    )
    return TurnEvent(
        event_id=event_id,
        session_id=session_id,
        turn_id=turn_id,
        stage=RuntimeStage.TOOL,
        status=TurnEventStatus.STARTED,
        occurred_at=datetime.now(timezone.utc),
    )


def test_runtime_event_endpoint_reads_the_new_runtime_store():
    session_id = session_store.create_session("Entelecheia", "events")

    payload = _route_payload("GET", f"/api/sessions/{session_id}/runtime-events", {})

    assert payload == {
        "enabled": True,
        "events": [],
        "next_after": None,
        "has_more": False,
        "limit": 100,
    }


def test_status_exposes_new_runtime_event_store_health():
    payload = service.system_status()

    assert payload["runtime_events"]["enabled"] is True
    assert payload["runtime_events"]["database_present"] is True
    assert payload["runtime_events"]["degraded"] is False
    assert "error" not in payload["runtime_events"]


def test_real_chat_events_are_written_to_session_runtime_storage_only(
    tmp_path,
    bound_partitioned_session,
):
    session_id = bound_partitioned_session(
        working_dir=tmp_path,
        title="entry",
    )
    response = service.chat_turn({
        "session_id": session_id,
        "message": "你好",
        "client_request_id": "event-request",
    })

    assert response["result"]["status"] == "completed"
    tail = _route_payload("GET", f"/api/sessions/{session_id}/runtime-events", {})
    assert tail["events"]
    assert all(item["turn_id"] == response["result"]["turn_id"] for item in tail["events"])


def test_runtime_event_endpoint_returns_tail_and_cursor_catch_up():
    session_id = session_store.create_session("Entelecheia", "events")
    first = _new_runtime_event("evt-1", session_id)
    session_store.append_runtime_turn_event(first)
    second = _new_runtime_event("evt-2", session_id)
    session_store.append_runtime_turn_event(second)

    tail = _route_payload("GET", f"/api/sessions/{session_id}/runtime-events?limit=1", {})
    caught_up = _route_payload(
        "GET", f"/api/sessions/{session_id}/runtime-events?after={first.event_id}", {}
    )

    assert [item["event_id"] for item in tail["events"]] == ["evt-2"]
    assert [item["event_id"] for item in caught_up["events"]] == ["evt-2"]
    assert set(tail["events"][0]) == {
        "schema_version", "event_id", "sequence", "session_id", "turn_id", "stage", "status", "occurred_at",
        "error_code", "retryable", "insession_task_id", "work_run_id", "attempt_id", "operation_id", "prompt_replay",
    }
    assert tail["events"][0]["sequence"] > 0
    assert tail["events"][0]["prompt_replay"] is False


def test_runtime_event_endpoint_projects_safe_ids_and_append_sequence_order():
    session_id = session_store.create_session("Entelecheia", "events")
    first = _new_runtime_event("evt-first", session_id)
    second = TurnEvent(
        event_id="evt-second",
        session_id=session_id,
        turn_id="turn-events",
        insession_task_id="task-1",
        work_run_id="workrun-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        stage=RuntimeStage.TOOL,
        status=TurnEventStatus.FAILED,
        occurred_at=datetime.now(timezone.utc),
        error_code="TOOL_FAILED",
        retryable=True,
        model_call_id="private-model-call",
        diagnostic_ref="private-diagnostic",
    )
    first_sequence = session_store.append_runtime_turn_event(first)
    second_sequence = session_store.append_runtime_turn_event(second)
    assert session_store.append_runtime_turn_event(first) == first_sequence

    payload = _route_payload("GET", f"/api/sessions/{session_id}/runtime-events", {})
    assert [item["event_id"] for item in payload["events"]] == ["evt-first", "evt-second"]
    assert [item["sequence"] for item in payload["events"]] == [first_sequence, second_sequence]
    public = payload["events"][1]
    assert public["insession_task_id"] == "task-1"
    assert public["work_run_id"] == "workrun-1"
    assert public["attempt_id"] == "attempt-1"
    assert public["operation_id"] == "operation-1"
    assert "model_call_id" not in public
    assert "diagnostic_ref" not in public


def test_runtime_event_endpoint_rejects_cross_session_cursor():
    session_a = session_store.create_session("Entelecheia", "a")
    session_b = session_store.create_session("Entelecheia", "b")
    session_store.append_runtime_turn_event(_new_runtime_event("evt-b", session_b))

    with pytest.raises(service.ApiError) as exc:
        _route_payload("GET", f"/api/sessions/{session_a}/runtime-events?after=evt-b", {})

    assert exc.value.status == 409
    assert exc.value.code == "RUNTIME_EVENT_CURSOR_NOT_FOUND"


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/sessions/{session_id}/active-run", {}),
    ("GET", "/api/sessions/{session_id}/pending-review", {}),
    ("GET", "/api/sessions/{session_id}/work-run", {}),
    ("GET", "/api/sessions/{session_id}/work-run-delivery", {}),
    ("POST", "/api/sessions/{session_id}/active-run/resolve", {"action": "close_stale"}),
    ("POST", "/api/chat/{session_id}/resume", {"approved": True}),
])
def test_retired_runtime_control_routes_are_not_public(method, path, body):
    session_id = session_store.create_session("Entelecheia", "retired")
    with pytest.raises(service.ApiError) as exc:
        _route_payload(method, path.format(session_id=session_id), body)
    assert exc.value.status == 404
    assert exc.value.code == "NOT_FOUND"
