from __future__ import annotations

from io import BytesIO
import json
from types import MethodType

from personagraph.session import project_catalog as session_projects
from personagraph.api import server, service
from personagraph.workspace.storage import context as document_context
from personagraph.configuration import paths
from personagraph.session import catalog as catalog_module
from personagraph.session.catalog import SessionCatalog
from personagraph.session import store as session_store


def _read_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        event = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if data_lines:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


def _stream_handler(payload: dict, output: BytesIO | None = None):
    raw = json.dumps(payload).encode("utf-8")
    handler = object.__new__(server.ApiHandler)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = BytesIO(raw)
    handler.wfile = output or BytesIO()
    handler.status = None
    handler.headers_sent = {}
    handler.send_response = MethodType(lambda self, status: setattr(self, "status", status), handler)
    handler.send_header = MethodType(
        lambda self, key, value: self.headers_sent.__setitem__(key, value), handler
    )
    handler.end_headers = MethodType(lambda self: None, handler)
    return handler


def test_chat_stream_emits_durable_acceptance_then_status_and_final(monkeypatch):
    def fake_chat_turn(payload, *, on_stream_event=None, on_turn_accepted=None):
        assert payload["message"] == "hello"
        assert payload["client_request_id"] == "request-1"
        assert on_stream_event is not None
        assert on_turn_accepted is not None
        on_turn_accepted({
            "session_id": "s1", "turn_id": "turn-1", "client_request_id": "request-1",
            "window_revision": 1, "replayed": False,
        })
        on_stream_event({"event": "answer_start", "generation_id": "g1"})
        on_stream_event({
            "event": "runtime_event",
            "runtime_event": {
                "schema_version": 1,
                "event_id": "evt_test",
                "sequence": 1,
                "session_id": "s1",
                "turn_id": "turn-1",
                "stage": "L2_UNDERSTAND",
                "status": "started",
                "occurred_at": "2026-07-14T00:00:00Z",
                "error_code": None,
                "retryable": None,
                "insession_task_id": None,
                "work_run_id": None,
                "attempt_id": None,
                "operation_id": None,
                "prompt_replay": False,
            },
        })
        on_stream_event({"event": "delta", "generation_id": "g1", "text": "逐字"})
        on_stream_event({"event": "delta", "generation_id": "g1", "text": "回复"})
        on_stream_event({"event": "answer_complete", "generation_id": "g1"})
        return {"result": {"status": "completed", "reply": "ok"}, "session": {"id": "s1"}}

    monkeypatch.setattr(service, "chat_turn", fake_chat_turn)
    handler = _stream_handler({"session_id": "s1", "message": "hello", "client_request_id": "request-1"})
    handler._handle_chat_stream()
    body = handler.wfile.getvalue().decode("utf-8")

    assert handler.status == 200
    assert handler.headers_sent["Content-Type"].startswith("text/event-stream")
    assert "id: evt_test\n" in body
    events = _read_sse_events(body)
    assert [event for event, _ in events] == [
        "accepted", "running", "answer_start", "runtime_event", "delta", "delta", "answer_complete", "final"
    ]
    assert events[0][1]["turn_id"] == "turn-1"
    assert events[0][1]["client_request_id"] == "request-1"
    assert "".join(data["text"] for event, data in events if event == "delta") == "逐字回复"
    assert events[-1][1]["result"]["reply"] == "ok"


def test_chat_stream_preaccept_error_never_claims_a_turn(monkeypatch):
    def fake_chat_turn(payload, *, on_stream_event=None, on_turn_accepted=None):
        raise service.ApiError("MODEL_CALL_FAILED", "模型调用失败", status=503)

    monkeypatch.setattr(service, "chat_turn", fake_chat_turn)
    handler = _stream_handler({"session_id": "s1", "message": "hello", "client_request_id": "request-1"})
    handler._handle_chat_stream()
    events = _read_sse_events(handler.wfile.getvalue().decode("utf-8"))

    assert [event for event, _ in events] == ["error"]
    assert events[0][1]["status"] == 503
    assert events[0][1]["error"]["code"] == "MODEL_CALL_FAILED"


def test_chat_stream_binds_session_and_project_databases_for_entire_handler(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "partitioned-state"
    project_root = tmp_path / "project"
    state_dir.mkdir()
    project_root.mkdir()
    shared_catalog = state_dir / "project_catalog.sqlite"
    monkeypatch.setattr(session_store, "DB_PATH", session_store._DEFAULT_DB_PATH)
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        shared_catalog,
    )
    monkeypatch.setattr(paths, "PROJECTS_DIR", state_dir / "projects")
    monkeypatch.setattr(session_projects, "DB_PATH", shared_catalog)
    project = session_projects.remember(str(project_root))
    SessionCatalog().create_session(session_id="s1", project_id=project.project_id)

    observed = []

    def fake_chat_turn(payload, *, on_stream_event=None, on_turn_accepted=None):
        observed.append(
            (
                session_store._SESSION_DATABASE_BINDING.get(),
                document_context.current(),
            )
        )
        return {"result": {"status": "completed", "reply": "ok"}}

    monkeypatch.setattr(service, "chat_turn", fake_chat_turn)
    handler = _stream_handler(
        {"session_id": "s1", "message": "hello", "client_request_id": "request-1"}
    )

    handler._handle_chat_stream()

    assert observed and observed[0][0][0] == "s1"
    assert observed[0][1] is not None
    assert observed[0][1].project_id == project.project_id
    assert session_store._SESSION_DATABASE_BINDING.get() is None
    assert document_context.current() is None


def test_chat_stream_disconnect_does_not_abort_an_accepted_runtime(monkeypatch):
    completed = False

    def fake_chat_turn(payload, *, on_stream_event=None, on_turn_accepted=None):
        nonlocal completed
        assert on_stream_event is not None
        assert on_turn_accepted is not None
        on_turn_accepted({
            "session_id": "s1", "turn_id": "turn-1", "client_request_id": "request-1",
            "window_revision": 1, "replayed": False,
        })
        on_stream_event({"event": "delta", "generation_id": "g1", "text": "still running"})
        completed = True
        return {"result": {"status": "completed", "reply": "done"}}

    class DisconnectOnThirdEvent(BytesIO):
        def __init__(self):
            super().__init__()
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1
            if self.flush_count == 3:
                raise BrokenPipeError("client closed the stream")
            return super().flush()

    monkeypatch.setattr(service, "chat_turn", fake_chat_turn)
    output = DisconnectOnThirdEvent()
    handler = _stream_handler(
        {"session_id": "s1", "message": "hello", "client_request_id": "request-1"},
        output,
    )
    handler._handle_chat_stream()

    assert completed is True
    assert output.flush_count == 3
