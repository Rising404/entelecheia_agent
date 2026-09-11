from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.api import service
from personagraph.api.service import sessions as session_service
from personagraph.api.service import views as service_views
from personagraph.context_budget import ContextBudgetExceeded
from personagraph.model_io.gateway import ModelGatewayError
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
    EntryTurnResult,
)
from personagraph.runtime.concurrency import SessionRunBusyError
from personagraph.runtime.post_commit.runner import process_due_turn_post_commit_jobs
from personagraph.session import store as ss
from personagraph.session.insession_task_contracts import (
    InSessionTaskPersistenceError,
)
from personagraph.session.l2_store import task_graph as task_graph_store  # noqa: F401
from tests.helpers.session_records import append_test_turn


@pytest.fixture(autouse=True)
def _tmp_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")
    ss._INITIALIZED_PATHS.clear()


def _chat_payload(session_id: str, message: str = "hi", request_id: str = "request-1") -> dict[str, str]:
    return {
        "session_id": session_id,
        "message": message,
        "client_request_id": request_id,
    }


def test_session_list_create_get_patch_contract():
    created = service.create_session({"title": "前端 API"})
    sid = created["session"]["id"]
    assert "persona_id" not in created["session"]
    append_test_turn(sid, "user", "hello")
    append_test_turn(sid, "assistant", "hi")

    listed = service.list_sessions({"status": "active"})
    assert [s["id"] for s in listed["sessions"]] == [sid]
    assert all("persona_id" not in session for session in listed["sessions"])

    unfiltered = service.list_sessions({
        "status": "active",
        "persona_id": "client-must-not-select-this",
    })
    assert [session["id"] for session in unfiltered["sessions"]] == [sid]

    detail = service.get_session(sid)
    assert detail["session"]["title"] == "前端 API"
    assert "persona_id" not in detail["session"]
    assert [t["role"] for t in detail["turns"]] == ["user", "assistant"]
    assert detail["pending_user_questions"] == []
    assert detail["turn_window"] is None


def test_session_creation_ignores_client_persona_selection():
    created = service.create_session({"persona_id": "client-selected-role"})

    assert "persona_id" not in created["session"]
    assert ss.get_session(created["session"]["id"])["persona_id"] == "Entelecheia"


def test_get_session_projects_pending_questions_without_execution_authority(monkeypatch):
    sid = ss.create_session("Entelecheia", title="pending question")
    monkeypatch.setattr(
        ss,
        "list_pending_user_questions",
        lambda *, session_id: (
            SimpleNamespace(
                insession_task_id="task-trip",
                question="你希望哪天出发？",
                work_run_id="private-work-run",
                question_attempt_id="private-attempt",
                work_run_revision=7,
            ),
        ),
    )

    detail = service.get_session(sid)

    assert detail["pending_user_questions"] == [{
        "insession_task_id": "task-trip",
        "question": "你希望哪天出发？",
    }]
    assert "private-work-run" not in str(detail)
    assert "private-attempt" not in str(detail)

    patched = service.patch_session(sid, {"title": "新标题", "status_action": "archive"})
    assert patched["session"]["title"] == "新标题"
    assert patched["session"]["status"] == "archived"


def test_get_session_fails_closed_when_neutral_pending_projection_is_corrupt(
    monkeypatch,
):
    sid = ss.create_session("Entelecheia", title="corrupt pending question")

    def fail_pending_read(*, session_id: str) -> tuple[object, ...]:
        assert session_id == sid
        raise InSessionTaskPersistenceError("corrupt pending authority")

    monkeypatch.setattr(ss, "list_pending_user_questions", fail_pending_read)

    with pytest.raises(session_service.ApiError) as captured:
        service.get_session(sid)

    assert captured.value.code == "PENDING_USER_QUESTIONS_UNAVAILABLE"
    assert captured.value.status == 409


def test_chat_turn_forwards_client_id_and_projects_only_entry_contract(monkeypatch):
    sid = ss.create_session("Entelecheia", title="chat")
    accepted_events: list[dict[str, object]] = []
    stream_events: list[dict[str, object]] = []

    def fake_run_entry(**kwargs):
        assert "persona_id" not in kwargs
        assert kwargs["user_input"] == "hi"
        assert kwargs["session_id"] == sid
        assert kwargs["client_request_id"] == "request-1"
        assert "ingress_source" not in kwargs
        assert kwargs["store"] is ss
        kwargs["on_turn_accepted"](AcceptedEntryTurn(
            session_id=sid,
            turn_id="turn-1",
            client_request_id="request-1",
            user_input="hi",
            attachment_ids=(),
            window_revision=1,
            replayed=False,
            execution_snapshot=EntryExecutionSnapshot.create(
                features={},
                post_commit_job_kinds=(),
            ),
        ))
        kwargs["on_stream_event"]({"event": "delta", "generation_id": "g1", "text": "hello"})
        return EntryTurnResult(
            session_id=sid,
            turn_id="turn-1",
            status="completed",
            processing_level="L0",
            reply="hello",
            window_state="empty",
            window_revision=3,
        )

    monkeypatch.setattr(session_service, "run_entry_turn", fake_run_entry)
    result = service.chat_turn(
        _chat_payload(sid),
        on_stream_event=stream_events.append,
        on_turn_accepted=accepted_events.append,
    )

    assert accepted_events == [{
        "session_id": sid,
        "turn_id": "turn-1",
        "client_request_id": "request-1",
        "window_revision": 1,
        "replayed": False,
    }]
    assert stream_events == [{"event": "delta", "generation_id": "g1", "text": "hello"}]
    assert result["result"] == {
        "session_id": sid,
        "turn_id": "turn-1",
        "status": "completed",
        "processing_level": "L0",
        "reply": "hello",
        "end_reason": None,
        "error_code": None,
        "related_insession_task_ids": [],
        "work_run_ids": [],
        "window_state": "empty",
        "window_revision": 3,
        "available_controls": [],
        "delivery": None,
        "pending_decision": None,
    }
    assert result["session"]["id"] == sid


def test_real_entry_returns_trusted_turn_fields_and_idempotent_replay(
    tmp_path,
    bound_partitioned_session,
):
    sid = bound_partitioned_session(working_dir=tmp_path, title="entry")

    first = service.chat_turn(_chat_payload(sid, "你好", "same-request"))
    replayed = service.chat_turn(_chat_payload(sid, "你好", "same-request"))

    payload = first["result"]
    assert payload["status"] == "completed"
    assert payload["processing_level"] == "L0"
    assert payload["turn_id"]
    assert payload["session_id"] == sid
    assert payload["related_insession_task_ids"] == []
    assert replayed["result"]["turn_id"] == payload["turn_id"]
    assert [item["role"] for item in ss.get_turns(sid)] == ["user", "assistant"]


def test_client_request_id_collision_is_not_a_second_execution(
    tmp_path,
    bound_partitioned_session,
):
    sid = bound_partitioned_session(
        working_dir=tmp_path,
        title="idempotency",
    )
    service.chat_turn(_chat_payload(sid, "原始输入", "same-request"))

    with pytest.raises(service.ApiError) as exc:
        service.chat_turn(_chat_payload(sid, "不同输入", "same-request"))

    assert exc.value.status == 409
    assert exc.value.code == "CLIENT_REQUEST_ID_REUSED"
    assert [item["content"] for item in ss.get_turns(sid)] == [
        "原始输入",
        "[mock-entry-l0] 原始输入",
    ]


def test_turn_result_view_never_leaks_legacy_trace_or_diagnostics():
    result = service_views.turn_result_view(EntryTurnResult(
        session_id="s1",
        turn_id="t1",
        status="incomplete",
        processing_level="L2",
        window_state="interrupted",
        window_revision=4,
        end_reason="provider_unavailable",
        error_code="MODEL_TIMEOUT",
        available_controls=("retry",),
    ))

    assert result == {
        "session_id": "s1",
        "turn_id": "t1",
        "status": "incomplete",
        "processing_level": "L2",
        "reply": None,
        "end_reason": "provider_unavailable",
        "error_code": "MODEL_TIMEOUT",
        "related_insession_task_ids": [],
        "work_run_ids": [],
        "window_state": "interrupted",
        "window_revision": 4,
        "available_controls": ["retry"],
        "delivery": None,
        "pending_decision": None,
    }
    assert "diagnostics" not in result
    assert "trace" not in result
    assert "outcome" not in result


def test_get_session_projects_only_safe_turn_window_fields():
    sid = ss.create_session("Entelecheia", title="window")
    accepted = ss.accept_turn_execution(
        session_id=sid,
        client_request_id="window-request",
        source="api",
        user_text="已接受但未完成",
        lease_owner="test",
    )
    turn_id = accepted["turn"]["turn_id"]  # type: ignore[index]
    ss.mark_turn_execution_interrupted(
        session_id=sid,
        turn_id=turn_id,
        expected_window_revision=1,
        stage="RESPONSE",
        interruption_reason="MODEL_TIMEOUT",
    )

    detail = service.get_session(sid)
    assert detail["turn_window"] == {
        "turn_id": turn_id,
        "window_state": "interrupted",
        "window_revision": 2,
        "stage": "RESPONSE",
        "interruption_reason": "MODEL_TIMEOUT",
    }


def test_get_session_projects_safe_post_commit_failure_without_job_internals():
    sid = ss.create_session("Entelecheia", title="summary failure")
    accepted = ss.accept_turn_execution(
        session_id=sid,
        client_request_id="summary-failure-request",
        source="api",
        user_text="需要保存的回复",
        lease_owner="test",
    )
    turn_id = str(accepted["turn"]["turn_id"])  # type: ignore[index]
    finalized = ss.finalize_turn_execution(
        session_id=sid,
        turn_id=turn_id,
        expected_window_revision=int(accepted["window"]["state_version"]),  # type: ignore[index]
        processing_level="L0",
        assistant_content="正式回复已交付。",
        post_commit_job_kinds=("session_summary",),
    )
    job = finalized["post_commit_jobs"][0]  # type: ignore[index]
    ss.claim_due_turn_post_commit_jobs(
        session_id=sid,
        worker_id="summary-test-worker",
        lease_seconds=60,
    )
    ss.fail_session_summary_post_commit_job(
        session_id=sid,
        job_id=str(job["job_id"]),
        worker_id="summary-test-worker",
        expected_state_version=0,
        expected_summarized_through_turn_id=None,
        status=ss.SessionSummaryStatus.UNAVAILABLE,
        reason_code="SUMMARY_MODEL_TIMEOUT",
        retry_after_seconds=None,
    )

    detail = service.get_session(sid)

    assert detail["turn_window"] == {
        "turn_id": turn_id,
        "window_state": "post_commit_pending",
        "window_revision": finalized["window"]["state_version"],  # type: ignore[index]
        "stage": "PERSIST",
        "interruption_reason": None,
        "post_commit_status": "failed",
        "post_commit_error_codes": ["SUMMARY_MODEL_TIMEOUT"],
    }
    assert "job_id" not in str(detail["turn_window"])


def test_pending_post_commit_window_wakes_the_runtime_worker(monkeypatch):
    sid = ss.create_session("Entelecheia", title="schedule summary")
    accepted = ss.accept_turn_execution(
        session_id=sid,
        client_request_id="schedule-request",
        source="api",
        user_text="需要后台整理",
        lease_owner="test",
    )
    turn_id = str(accepted["turn"]["turn_id"])  # type: ignore[index]
    ss.finalize_turn_execution(
        session_id=sid,
        turn_id=turn_id,
        expected_window_revision=int(accepted["window"]["state_version"]),  # type: ignore[index]
        processing_level="L0",
        assistant_content="正式回复",
        post_commit_job_kinds=("session_summary",),
    )
    scheduled: list[tuple[str, object]] = []
    monkeypatch.setattr(
        session_service,
        "schedule_turn_post_commit_jobs",
        lambda *, session_id, store: scheduled.append((session_id, store)),
    )

    session_service._schedule_pending_turn_post_commit_jobs(sid)

    assert scheduled == [(sid, ss)]


def test_next_chat_wakes_pending_summary_then_waits_for_safe_window_release(
    monkeypatch,
    tmp_path,
    bound_partitioned_session,
):
    """守护进程丢失可以恢复，但绝不意味着可以绕过其窗口。

    这里有意让摘要调度器保持静止。下一条聊天输入必须尽力唤醒调度器，在持久
    任务稳定之前保持被拒绝，并且只有工作器释放上一轮的执行窗口后才能被接受。
    """

    sid = bound_partitioned_session(
        working_dir=tmp_path,
        title="post-commit recovery",
    )
    scheduled: list[tuple[str, object]] = []
    monkeypatch.setattr(
        session_service,
        "schedule_turn_post_commit_jobs",
        lambda *, session_id, store: scheduled.append((session_id, store)),
    )
    monkeypatch.setattr(
        session_service,
        "load_features",
        lambda _config: {
            "history_retrieval_read_enabled": False,
            "history_retrieval_write_enabled": False,
        },
    )

    first = service.chat_turn(
        _chat_payload(sid, "请先完成这一轮", "first-formal-turn"),
    )
    assert first["result"]["status"] == "completed"
    assert first["result"]["window_state"] == "post_commit_pending"
    assert scheduled == [(sid, ss)]
    assert [item["role"] for item in ss.get_turns(sid)] == ["user", "assistant"]

    with pytest.raises(service.ApiError) as blocked:
        service.chat_turn(
            _chat_payload(sid, "这条应等待摘要", "blocked-while-summary-pending"),
        )

    assert blocked.value.code == "TURN_IN_PROGRESS"
    assert blocked.value.details["reason"] == "post_commit_pending"
    assert scheduled == [(sid, ss), (sid, ss)]
    # 被阻塞的请求绝不会变成部分接受的轮次。
    assert [item["role"] for item in ss.get_turns(sid)] == ["user", "assistant"]

    settled = process_due_turn_post_commit_jobs(
        session_id=sid,
        store=ss,
        worker_id="post-commit-recovery-test-worker",
        summary_generator=lambda *_args: "已验证的会话摘要。",
    )
    assert settled.applied_job_count == 1
    assert settled.released_window is True
    assert ss.get_turn_execution_window(sid)["window_state"] == "empty"  # type: ignore[index]

    accepted = service.chat_turn(
        _chat_payload(sid, "现在可以继续", "accepted-after-summary-settles"),
    )
    assert accepted["result"]["status"] == "completed"
    assert accepted["result"]["window_state"] == "post_commit_pending"
    assert scheduled == [(sid, ss), (sid, ss), (sid, ss)]
    assert [item["role"] for item in ss.get_turns(sid)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_chat_turn_rejects_missing_or_overlong_client_request_id():
    sid = ss.create_session("Entelecheia", title="validation")
    with pytest.raises(service.ApiError) as missing:
        service.chat_turn({"session_id": sid, "message": "hi"})
    assert missing.value.code == "MISSING_FIELD"

    with pytest.raises(service.ApiError) as oversized:
        service.chat_turn(_chat_payload(sid, request_id="x" * 201))
    assert oversized.value.status == 400
    assert oversized.value.code == "CLIENT_REQUEST_ID_TOO_LONG"


def test_chat_turn_maps_model_gateway_error_to_api_outcome(monkeypatch):
    sid = ss.create_session("Entelecheia", title="chat")

    def fake_run_entry(*args, **kwargs):
        raise ModelGatewayError(
            "MODEL_CALL_FAILED",
            "Model provider request timed out.",
            retryable=True,
            details={"provider": "anthropic-compatible", "exception_type": "TimeoutException"},
        )

    monkeypatch.setattr(session_service, "run_entry_turn", fake_run_entry)
    with pytest.raises(service.ApiError) as exc:
        service.chat_turn(_chat_payload(sid))

    payload = exc.value.to_payload()
    assert exc.value.status == 503
    assert payload["error"]["code"] == "MODEL_CALL_FAILED"
    assert payload["outcome"]["error"]["domain"] == "model"


def test_chat_turn_maps_context_budget_and_local_busy_errors(monkeypatch):
    sid = ss.create_session("Entelecheia", title="errors")

    monkeypatch.setattr(
        session_service,
        "run_entry_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ContextBudgetExceeded(limit=1000, estimated_tokens=1250, degraded=("summary_dropped",))
        ),
    )
    with pytest.raises(service.ApiError) as budget:
        service.chat_turn(_chat_payload(sid, request_id="budget-request"))
    assert budget.value.status == 422
    assert budget.value.code == "CONTEXT_BUDGET_EXCEEDED"

    monkeypatch.setattr(
        session_service,
        "run_entry_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(SessionRunBusyError(sid)),
    )
    with pytest.raises(service.ApiError) as busy:
        service.chat_turn(_chat_payload(sid, request_id="busy-request"))
    assert busy.value.status == 409
    assert busy.value.code == "SESSION_RUN_BUSY"
