from __future__ import annotations

import http.client
import json
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

from personagraph.api import server, service
from personagraph.runtime.turn_events import (
    RuntimeStage,
    TurnEventStatus,
    TurnEvent as RuntimeEvent,
    project_turn_event,
)
from personagraph.session import store as session_store


def _get_json(port: int, path: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


def _post_stream_final(port: int, payload: dict) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload).encode("utf-8")
        connection.request(
            "POST",
            "/api/chat/stream",
            body=body,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        stream = response.read().decode("utf-8")
    finally:
        connection.close()
    for block in stream.strip().split("\n\n"):
        event = None
        data = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if event == "final" and data:
            return json.loads("\n".join(data))
    raise AssertionError("chat stream did not contain a final event")


def _wait_for_released_turn_window(port: int, session_id: str) -> dict:
    """正式回复会在隐藏摘要任务稳定之前返回。"""

    deadline = time.monotonic() + 3
    latest: dict | None = None
    while time.monotonic() < deadline:
        latest = _get_json(port, f"/api/sessions/{session_id}")
        window = latest.get("turn_window") or {}
        if window.get("window_state") == "empty":
            return latest
        assert window.get("window_state") == "post_commit_pending"
        time.sleep(0.02)
    raise AssertionError(f"post-commit window was not released: {latest}")


def test_disconnect_keeps_runtime_alive_and_refresh_reads_new_public_events(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")
    session_id = session_store.create_session("Entelecheia", "loopback recovery")
    turn_id = "turn-loopback"
    session_store.create_runtime_turn(
        turn_id=turn_id,
        session_id=session_id,
        source="runtime_test",
        user_text="run it",
    )

    release_runtime = threading.Event()
    finish_runtime = threading.Event()
    running_delivered = threading.Event()
    runtime_completed = threading.Event()
    disconnect_observed = threading.Event()

    def fake_chat_turn(payload, *, on_stream_event=None, on_turn_accepted=None):
        assert payload["session_id"] == session_id
        assert on_stream_event is not None
        assert on_turn_accepted is not None
        on_turn_accepted({
            "session_id": session_id,
            "turn_id": turn_id,
            "client_request_id": payload["client_request_id"],
            "window_revision": 1,
            "replayed": False,
        })
        assert release_runtime.wait(timeout=3)
        running = RuntimeEvent(
            event_id="evt-loopback-running",
            session_id=session_id,
            turn_id=turn_id,
            stage=RuntimeStage.TOOL,
            status=TurnEventStatus.STARTED,
            occurred_at=datetime.now(timezone.utc),
        )
        running_sequence = session_store.append_runtime_turn_event(running)
        on_stream_event({
            "event": "runtime_event",
            "runtime_event": project_turn_event(
                running, sequence=running_sequence
            ).model_dump(mode="json"),
        })
        running_delivered.set()
        assert finish_runtime.wait(timeout=3)
        completed = RuntimeEvent(
            event_id="evt-loopback-completed",
            session_id=session_id,
            turn_id=turn_id,
            stage=RuntimeStage.RESPONSE,
            status=TurnEventStatus.COMPLETED,
            occurred_at=datetime.now(timezone.utc),
        )
        completed_sequence = session_store.append_runtime_turn_event(completed)
        on_stream_event({
            "event": "runtime_event",
            "runtime_event": project_turn_event(
                completed, sequence=completed_sequence
            ).model_dump(mode="json"),
        })
        runtime_completed.set()
        return {"result": {"reply": "done"}}

    monkeypatch.setattr(service, "chat_turn", fake_chat_turn)

    class LoopbackHandler(server.ApiHandler):
        def _require_allowed_origin(self):
            return None

        def _require_authorization(self):
            return None

        def _send_sse(self, event, payload):
            try:
                return super()._send_sse(event, payload)
            except OSError:
                disconnect_observed.set()
                raise

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    port = int(httpd.server_address[1])

    stream = socket.create_connection(("127.0.0.1", port), timeout=3)
    body = json.dumps({
        "session_id": session_id,
        "message": "run it",
        "client_request_id": "loopback-request",
    }).encode("utf-8")
    request = (
        f"POST /api/chat/stream HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\n"
        "Accept: text/event-stream\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body

    try:
        stream.sendall(request)
        received = b""
        while b"event: running\n" not in received:
            chunk = stream.recv(4096)
            assert chunk
            received += chunk

        # 强制立即 TCP reset，让下一次 SSE 写入走真实 socket 失败路径，
        # 而非仅填满本地缓冲区。
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        stream.close()
        release_runtime.set()

        assert running_delivered.wait(timeout=3)
        assert disconnect_observed.wait(timeout=3)

        tail = _get_json(
            port,
            f"/api/sessions/{session_id}/runtime-events",
        )
        assert [item["event_id"] for item in tail["events"]] == ["evt-loopback-running"]

        finish_runtime.set()
        assert runtime_completed.wait(timeout=3)

        final_tail = _get_json(
            port,
            f"/api/sessions/{session_id}/runtime-events",
        )
        assert [item["event_id"] for item in final_tail["events"]] == [
            "evt-loopback-running",
            "evt-loopback-completed",
        ]
    finally:
        release_runtime.set()
        finish_runtime.set()
        try:
            stream.close()
        except OSError:
            pass
        httpd.shutdown()
        httpd.server_close()
        http_thread.join(timeout=3)


def test_new_entry_finalizes_its_window_without_legacy_control_state_over_loopback(
    tmp_path,
    monkeypatch,
    partitioned_project_state,
):
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        "pause refresh resume",
        working_dir=str(workspace),
    )

    class LoopbackHandler(server.ApiHandler):
        def _require_allowed_origin(self):
            return None

        def _require_authorization(self):
            return None

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    port = int(httpd.server_address[1])

    try:
        initial = _post_stream_final(
            port,
            {
                "session_id": session_id,
                "message": "记住我在验证刷新后的审批恢复。",
                "client_request_id": "loopback-final-request",
            },
        )
        assert initial["result"]["status"] == "completed"
        assert initial["result"]["window_state"] == "post_commit_pending"
        assert "pending_review" not in initial["result"]

        detail = _wait_for_released_turn_window(port, session_id)
        assert detail["turn_window"]["window_state"] == "empty"
    finally:
        httpd.shutdown()
        httpd.server_close()
        http_thread.join(timeout=3)
