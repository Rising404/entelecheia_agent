from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import http.client
import io
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import pytest

from personagraph.api import security as api_security
from personagraph.api import server
from personagraph.workspace.storage.context import (
    connect_current,
    current_database_path,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.session import store as session_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from tests.helpers.prepared_model_provider import as_prepared_test_provider


_API_TOKEN = "product-loopback-test-token-00000000000000000000"
_MESSAGE = "请阅读附件 PDF，创建一个任务并输出其中的准确率结论。"
_L2_RUNTIME_POLICY = {"l1_enabled": False, "l2_enabled": True}


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def _pdf(objects: list[bytes]) -> bytes:
    output = io.BytesIO()
    output.write(b"%PDF-1.7\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n".encode())
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return output.getvalue()


def _text_pdf_bytes() -> bytes:
    text = "Cedar benchmark accuracy is 71 percent."
    stream = f"BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n({text}) Tj\nET".encode(
        "latin-1"
    )
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 4 0 R >>",
        ]
    )


def _request(
    *,
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    expected_status: int = 200,
) -> tuple[dict[str, Any], http.client.HTTPMessage]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    request_headers = {
        "Authorization": f"Bearer {_API_TOKEN}",
        "Origin": "null",
        "Connection": "close",
        **dict(headers or {}),
    }
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        assert response.status == expected_status, payload.decode("utf-8", "replace")
        return json.loads(payload), response.headers
    finally:
        connection.close()


def _request_json(
    *,
    port: int,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    result, _headers = _request(
        port=port,
        method=method,
        path=path,
        body=body,
        headers={"Content-Type": "application/json"},
    )
    return result


def _parse_sse(body: bytes) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.decode("utf-8").strip().split("\n\n"):
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


def _post_chat_stream(
    *,
    port: int,
    payload: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        connection.request(
            "POST",
            "/api/chat/stream",
            body=body,
            headers={
                "Authorization": f"Bearer {_API_TOKEN}",
                "Origin": "null",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        response_body = response.read()
        assert response.status == 200, response_body.decode("utf-8", "replace")
        assert response.headers.get_content_type() == "text/event-stream"
        return _parse_sse(response_body)
    finally:
        connection.close()


def _final_payload(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    finals = [payload for event, payload in events if event == "final"]
    assert len(finals) == 1, events
    return finals[0]


def _wait_for_empty_turn_window(*, port: int, session_id: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        detail = _request_json(
            port=port,
            method="GET",
            path=f"/api/sessions/{session_id}",
        )
        window = detail.get("turn_window") or {}
        if window.get("window_state") == "empty":
            return
        assert window.get("window_state") == "post_commit_pending", detail
        time.sleep(0.02)
    raise AssertionError("post-commit Turn window was not released")


def _sqlite_row_counts(
    path: Path,
    *,
    connect: Callable[[], sqlite3.Connection] | None = None,
) -> dict[str, int]:
    if not path.exists():
        return {}
    with (connect() if connect is not None else sqlite3.connect(path)) as connection:
        tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        return {
            table: int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            )
            for table in tables
        }


def _tree_fingerprints(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _install_new_root_classifier_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def complete_classifier(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
        **_kwargs: object,
    ) -> ModelResult:
        assert purpose == "runtime_entry_classify"
        request = json.loads(user_content)
        requests.append(request)
        assert request["current_user_text"] == _MESSAGE
        assert request["host_facts"]["attachment_count"] == 1
        assert len(request["new_files"]) == 1
        return ModelResult(
            reply=json.dumps(
                {
                    "processing_level": "L2",
                    "task_matches": [
                        {
                            "match_type": "new_root",
                            "local_key": "pdf_summary",
                            "title": "附件准确率分析",
                            "objective": "读取已授权附件并输出其中的准确率结论",
                            "source_excerpt": _MESSAGE,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=0,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    # 这里只替换模型边界。chat_turn、run_entry_turn、V2 规划、TaskGraph 执行、
    # 两层验证以及交付终结仍使用 HTTP 路由组合的生产实现。
    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(complete_classifier),
    )
    return requests


def test_http_pdf_chat_runs_l2_to_verified_delivery_and_replays_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_ENGINE", "native")
    monkeypatch.setattr(api_security, "TOKEN_PATH", tmp_path / "api_secret")
    monkeypatch.setattr(api_security, "_API_TOKEN", None)
    api_security.initialize_api_token(_API_TOKEN)
    classifier_requests = _install_new_root_classifier_provider(monkeypatch)

    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ApiHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = int(httpd.server_address[1])

    try:
        created = _request_json(
            port=port,
            method="POST",
            path="/api/sessions",
            payload={
                "title": "V2 product loopback",
            },
        )
        session_id = str(created["session"]["id"])
        workspace = Path(created["session"]["working_dir"])
        uploaded, _headers = _request(
            port=port,
            method="POST",
            path=f"/api/sessions/{session_id}/attachments",
            body=_text_pdf_bytes(),
            headers={
                "Content-Type": "application/pdf",
                "X-Attachment-Filename": "cedar.pdf",
            },
        )
        attachment = uploaded["attachment"]
        assert attachment["media_type"] == "application/pdf"
        assert attachment["readable"] is True

        client_request_id = "v2-product-loopback-pdf"
        command = {
            "session_id": session_id,
            "message": _MESSAGE,
            "attachment_ids": [attachment["attachment_id"]],
            "client_request_id": client_request_id,
            "runtime_policy": _L2_RUNTIME_POLICY,
        }
        first_events = _post_chat_stream(port=port, payload=command)
        assert [event for event, _payload in first_events[:2]] == [
            "accepted",
            "running",
        ]
        assert first_events[0][1]["replayed"] is False
        first = _final_payload(first_events)["result"]
        assert first["status"] == "completed"
        assert first["processing_level"] == "L2"
        assert first["reply"].strip()
        assert len(first["related_insession_task_ids"]) == 1
        assert first["work_run_ids"], json.dumps(first, ensure_ascii=False)

        task_id = first["related_insession_task_ids"][0]
        task_payload = _request_json(
            port=port,
            method="GET",
            path=f"/api/sessions/{session_id}/insession-tasks/{task_id}",
        )
        public_task = task_payload["task"]
        assert public_task["status"] == "completed"
        assert public_task["current_graph_revision"] == 1
        assert len(public_task["nodes"]) >= 1
        assert all(node["status"] == "completed" for node in public_task["nodes"])

        with session_store.session_database_scope(session_id):
            stored_task = task_graph_store.get_insession_task_details(
                session_id,
                task_id,
            )
            assert stored_task is not None
            assert stored_task.current_graph_revision == 1
            for work_run_id in first["work_run_ids"]:
                stored_run = work_run_store.get_work_run(
                    session_id=session_id,
                    work_run_id=work_run_id,
                )
                assert stored_run.work_run.status.value == "completed"
                assert stored_run.work_run.reason == "verification_passed"
                assert stored_run.attempts
                if stored_run.work_run.subject.kind == "task_node":
                    assert stored_run.node_delivery_id is not None
                else:
                    assert stored_run.auxiliary_node_completion_id is not None

            delivery_id = work_run_store.get_completed_task_final_delivery_id(
                session_id=session_id,
                task_id=task_id,
            )
            delivery = verification_store.get_task_node_delivery(
                session_id=session_id,
                delivery_id=delivery_id,
            )
            assert delivery.delivery.subject.graph_revision == 1
            assert delivery.output_window.content == first["reply"]
            assert delivery.output_window.content.strip()

            candidate = task_delivery_store.get_task_delivery_candidate_settlement(
                session_id=session_id,
                task_id=task_id,
                graph_revision=1,
            )
            assert candidate is not None
            assert candidate.intent.result.disposition.value == "pass"
            assert candidate.settlement.disposition.value == "pass"
            assert candidate.settlement.root_delivery_id == delivery_id
            assert candidate.trigger is None

            _wait_for_empty_turn_window(port=port, session_id=session_id)
            settled = session_store.get_turn_execution_for_client_request(
                session_id=session_id,
                client_request_id=client_request_id,
            )
            assert settled is not None
            settled_turn = settled["turn"]
            assert settled_turn["turn_id"] == first["turn_id"]
            assert settled_turn["status"] == "completed"
            assert settled_turn["processing_level"] == "L2"
            assert settled_turn["error_code"] is None
            assert settled_turn["end_reason"] is None
            assert settled_turn["completed_at"]
            assert settled["window"]["window_state"] == "empty"
            formal_pair = session_store.get_committed_turn_pair(
                session_id,
                f"commit_{first['turn_id']}",
            )
            assert formal_pair is not None
            assert formal_pair["turn_id"] == first["turn_id"]
            assert formal_pair["user_content"] == _MESSAGE
            assert formal_pair["assistant_content"] == first["reply"]
            session_database_path = session_store.current_session_database_path()
            documents_database_path = current_database_path()
            session_counts = _sqlite_row_counts(session_database_path)
            document_counts = _sqlite_row_counts(
                documents_database_path,
                connect=connect_current,
            )
        attachment_root = workspace / "附件"
        attachment_files = _tree_fingerprints(attachment_root)

        replay_events = _post_chat_stream(port=port, payload=command)
        assert [event for event, _payload in replay_events[:2]] == [
            "accepted",
            "running",
        ]
        assert replay_events[0][1]["replayed"] is True
        replayed = _final_payload(replay_events)["result"]
        assert replayed["status"] == first["status"]
        assert replayed["processing_level"] == first["processing_level"]
        assert replayed["reply"] == first["reply"]
        assert (
            replayed["related_insession_task_ids"]
            == first["related_insession_task_ids"]
        )
        assert replayed["work_run_ids"] == first["work_run_ids"]
        assert replayed["window_state"] == "empty"
        assert replayed["window_revision"] >= first["window_revision"]
        assert len(classifier_requests) == 1
        assert _sqlite_row_counts(session_database_path) == session_counts
        with session_store.session_database_scope(session_id):
            assert _sqlite_row_counts(
                documents_database_path,
                connect=connect_current,
            ) == document_counts
        assert _tree_fingerprints(attachment_root) == attachment_files
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
        monkeypatch.setattr(api_security, "_API_TOKEN", None)


def test_http_verified_publication_fails_closed_when_work_run_refs_are_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_ENGINE", "native")
    monkeypatch.setattr(api_security, "TOKEN_PATH", tmp_path / "api_secret")
    monkeypatch.setattr(api_security, "_API_TOKEN", None)
    api_security.initialize_api_token(_API_TOKEN)
    classifier_requests = _install_new_root_classifier_provider(monkeypatch)

    original_list_work_run_ids = work_run_store.list_turn_linked_work_run_ids
    failed_reads: list[tuple[str, str]] = []

    def unreadable_work_run_refs(
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]:
        failed_reads.append((session_id, turn_id))
        raise work_run_store.WorkExecutionPersistenceError(
            "injected unreadable Turn--WorkRun authority"
        )

    monkeypatch.setattr(
        session_store,
        "list_turn_linked_work_run_ids",
        unreadable_work_run_refs,
    )

    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.ApiHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = int(httpd.server_address[1])

    try:
        created = _request_json(
            port=port,
            method="POST",
            path="/api/sessions",
            payload={
                "title": "V2 fail-closed loopback",
            },
        )
        session_id = str(created["session"]["id"])
        uploaded, _headers = _request(
            port=port,
            method="POST",
            path=f"/api/sessions/{session_id}/attachments",
            body=_text_pdf_bytes(),
            headers={
                "Content-Type": "application/pdf",
                "X-Attachment-Filename": "cedar.pdf",
            },
        )
        attachment_id = str(uploaded["attachment"]["attachment_id"])

        events = _post_chat_stream(
            port=port,
            payload={
                "session_id": session_id,
                "message": _MESSAGE,
                "attachment_ids": [attachment_id],
                "client_request_id": "v2-product-loopback-unreadable-work-runs",
                "runtime_policy": _L2_RUNTIME_POLICY,
            },
        )
        assert [event for event, _payload in events[:2]] == [
            "accepted",
            "running",
        ]
        result = _final_payload(events)["result"]
        assert result["status"] == "incomplete"
        assert result["processing_level"] == "L2"
        assert result["end_reason"] == "persistence_error"
        assert result["error_code"] == "PERSIST_FAILED"
        assert result["reply"] is None
        assert result["work_run_ids"] == []
        assert result["window_state"] == "interrupted"
        assert failed_reads == [
            (session_id, result["turn_id"]),
            (session_id, result["turn_id"]),
        ]
        assert len(classifier_requests) == 1

        # V2 链路确实已经完成并冻结其已验证权威，但无法重建精确 WorkRun 引用时，
        # 公开轮次并未正式提交。
        with session_store.session_database_scope(session_id):
            task_ids = session_store.list_turn_insession_task_ids(
                session_id,
                result["turn_id"],
            )
            assert len(task_ids) == 1
            durable_work_run_ids = original_list_work_run_ids(
                session_id=session_id,
                turn_id=result["turn_id"],
            )
            assert durable_work_run_ids
            delivery_id = work_run_store.get_completed_task_final_delivery_id(
                session_id=session_id,
                task_id=task_ids[0],
            )
            assert verification_store.get_task_node_delivery(
                session_id=session_id,
                delivery_id=delivery_id,
            ).output_window.content
            assert session_store.get_committed_turn_pair(
                session_id,
                f"commit_{result['turn_id']}",
            ) is None
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
        monkeypatch.setattr(api_security, "_API_TOKEN", None)
