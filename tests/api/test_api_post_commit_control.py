"""提交后失败控制的公开范围、确认与陈旧请求边界。"""

from contextlib import contextmanager
from email.message import Message

import pytest

from personagraph.api import router, server, service
from personagraph.session import store
from tests.helpers.session_records import complete_test_turn_execution


def _failed_session(*, working_dir=None):
    session_id = store.create_session("Entelecheia", working_dir=working_dir)
    with store.session_database_scope(session_id):
        complete_test_turn_execution(
            session_id, 1,
            post_commit_job_kinds=("session_retrieval_index", "session_summary"),
        )
        for job in store.claim_due_turn_post_commit_jobs(
            session_id=session_id, worker_id="test-worker", lease_seconds=60,
        ):
            store.mark_turn_post_commit_job_failed(
                job_id=job["job_id"], worker_id="test-worker",
                reason_code="TEST_FAILURE", retry_after_seconds=None,
            )
    return session_id


def _request(session_id, action="retry"):
    inspected = store.inspect_turn_execution(session_id)
    return {
        "turn_id": inspected["window"]["turn_id"],
        "expected_window_revision": inspected["window"]["state_version"],
        "expected_failed_job_digest": inspected["failed_job_digest"],
        "job_ids": [job["job_id"] for job in inspected["post_commit_jobs"]],
        "request_id": "control-request", "action": action,
    }


@pytest.fixture
def scheduled(monkeypatch):
    from personagraph.api.service import post_commit
    calls = []
    monkeypatch.setattr(post_commit, "schedule_turn_post_commit_jobs", lambda **kw: calls.append(kw))
    return calls


def test_session_read_exposes_exact_failed_control_without_scheduling(scheduled):
    session_id = _failed_session()
    projection = service.get_session(session_id)["post_commit"]
    expected = _request(session_id)
    assert projection["turn_id"] == expected["turn_id"]
    assert projection["window_revision"] == expected["expected_window_revision"]
    assert projection["failed_job_digest"] == expected["expected_failed_job_digest"]
    assert {job["job_id"] for job in projection["failed_jobs"]} == set(expected["job_ids"])
    assert {job["job_kind"] for job in projection["failed_jobs"]} == {
        "session_retrieval_index", "session_summary",
    }
    assert scheduled == []


@pytest.mark.parametrize("action", ["retry", "waive"])
def test_control_updates_exact_jobs_and_schedules_once(action, scheduled):
    session_id = _failed_session()
    payload = _request(session_id, action)
    if action == "waive":
        payload["confirm_stale"] = True
    before = store.get_turns(session_id)
    response = router.dispatch_response(
        "POST", f"/api/sessions/{session_id}/post-commit/control", payload,
    )
    assert response.payload["replayed"] is False
    statuses = {job["status"] for job in store.list_turn_post_commit_jobs(payload["turn_id"])}
    assert statuses == ({"pending"} if action == "retry" else {"waived"})
    assert len(scheduled) == 1 and scheduled[0]["session_id"] == session_id
    assert store.get_turns(session_id) == before


def test_waive_requires_explicit_confirmation(scheduled):
    session_id = _failed_session()
    payload = _request(session_id, "waive")
    with pytest.raises(service.ApiError) as caught:
        service.control_turn_post_commit_jobs(session_id, payload)
    assert caught.value.code == "POST_COMMIT_CONFIRMATION_REQUIRED"
    assert scheduled == []
    assert all(job["status"] == "terminal_failed" for job in store.list_turn_post_commit_jobs(payload["turn_id"]))


def test_same_retry_request_replays_without_resetting_window_again(scheduled):
    session_id = _failed_session()
    payload = _request(session_id)
    assert service.control_turn_post_commit_jobs(session_id, payload) == {"replayed": False}
    revision = store.inspect_turn_execution(session_id)["window"]["state_version"]
    assert service.control_turn_post_commit_jobs(session_id, payload) == {"replayed": True}
    assert store.inspect_turn_execution(session_id)["window"]["state_version"] == revision
    assert len(scheduled) == 2  # 两个显式命令各唤醒一次；后台调度器负责工作去重。


@pytest.mark.parametrize("start_next_turn", [False, True])
def test_settled_control_replay_returns_receipt_without_scheduling_a_new_turn(start_next_turn, scheduled):
    session_id = _failed_session()
    payload = {**_request(session_id, "waive"), "confirm_stale": True}
    service.control_turn_post_commit_jobs(session_id, payload)
    window = store.inspect_turn_execution(session_id)["window"]
    store.release_turn_execution_window(
        session_id=session_id, turn_id=payload["turn_id"],
        expected_window_revision=window["state_version"],
    )
    if start_next_turn:
        complete_test_turn_execution(session_id, 2, post_commit_job_kinds=("session_summary",))
    before = store.inspect_turn_execution(session_id)
    response = router.dispatch_response("POST", f"/api/sessions/{session_id}/post-commit/control", payload)
    assert response.payload == {"replayed": True}
    assert store.inspect_turn_execution(session_id) == before
    assert len(scheduled) == 1


@pytest.mark.parametrize("status", ["archived", "trashed"])
def test_read_only_session_cannot_be_recovered(status, scheduled):
    session_id = _failed_session()
    payload = _request(session_id)
    (store.archive_session if status == "archived" else store.trash_session)(session_id)
    with pytest.raises(service.ApiError) as caught:
        service.control_turn_post_commit_jobs(session_id, payload)
    assert caught.value.code == "SESSION_READ_ONLY"
    assert scheduled == []


def test_scheduler_failure_does_not_report_the_committed_control_as_rejected(monkeypatch):
    from personagraph.api.service import post_commit
    session_id = _failed_session()
    payload = _request(session_id)

    def unavailable(**_kwargs):
        raise RuntimeError("isolated scheduler unavailable")

    monkeypatch.setattr(post_commit, "schedule_turn_post_commit_jobs", unavailable)
    assert service.control_turn_post_commit_jobs(session_id, payload) == {"replayed": False}
    assert {job["status"] for job in store.list_turn_post_commit_jobs(payload["turn_id"])} == {"pending"}


def test_explicit_waive_reaches_canonical_scheduler_and_releases_window():
    from personagraph.runtime.post_commit.scheduler import wait_for_turn_post_commit_jobs
    session_id = _failed_session()
    payload = {**_request(session_id, "waive"), "confirm_stale": True}
    before = store.get_turns(session_id)
    router.dispatch_response("POST", f"/api/sessions/{session_id}/post-commit/control", payload)
    settled = wait_for_turn_post_commit_jobs(
        session_id=session_id, turn_id=payload["turn_id"], store=store, timeout_seconds=2,
    )
    assert settled.status == "settled"
    assert service.get_session(session_id)["turn_window"]["window_state"] == "empty"
    assert store.get_turns(session_id) == before


@pytest.mark.parametrize("change", [
    {"expected_window_revision": 1}, {"expected_failed_job_digest": "0" * 64},
    {"turn_id": "unknown-turn"}, {"job_ids": ["unknown-job"]},
])
def test_stale_or_crossed_control_is_rejected(change, scheduled):
    session_id = _failed_session()
    payload = _request(session_id)
    payload.update(change)
    with pytest.raises(service.ApiError) as caught:
        service.control_turn_post_commit_jobs(session_id, payload)
    assert caught.value.status == 409
    assert scheduled == []


@pytest.mark.parametrize("change", [
    {"action": "delete"}, {"action": []}, {"job_ids": []}, {"job_ids": [1]},
    {"expected_window_revision": True}, {"request_id": ""},
    {"expected_failed_job_digest": "invalid"}, {"actor": "admin"},
    {"confirm_stale": "true"},
])
def test_invalid_control_never_writes(change, scheduled):
    session_id = _failed_session()
    payload = _request(session_id)
    payload.update(change)
    with pytest.raises(service.ApiError) as caught:
        service.control_turn_post_commit_jobs(session_id, payload)
    assert caught.value.status == 400
    assert scheduled == []


def test_cross_session_control_cannot_mutate_another_turn(scheduled):
    first, second = _failed_session(), _failed_session()
    payload = _request(first)
    with pytest.raises(service.ApiError) as caught:
        service.control_turn_post_commit_jobs(second, payload)
    assert caught.value.status == 409
    assert all(job["status"] == "terminal_failed" for job in store.list_turn_post_commit_jobs(payload["turn_id"]))
    assert scheduled == []


def test_control_resolves_real_partitioned_session_scope(partitioned_project_state, tmp_path, scheduled):
    roots = [tmp_path / "project-a", tmp_path / "project-b"]
    for root in roots:
        root.mkdir()
    first, second = [_failed_session(working_dir=str(root)) for root in roots]
    with store.session_database_scope(first):
        payload = _request(first)
    response = router.dispatch_response("GET", f"/api/sessions/{first}", {}).payload
    assert response["post_commit"]["failed_job_digest"] == payload["expected_failed_job_digest"]
    with pytest.raises(service.ApiError) as caught:
        router.dispatch_response("POST", f"/api/sessions/{second}/post-commit/control", payload)
    assert caught.value.status == 409
    assert scheduled == []
    router.dispatch_response("POST", f"/api/sessions/{first}/post-commit/control", payload)
    with store.session_database_scope(first):
        assert {job["status"] for job in store.list_turn_post_commit_jobs(payload["turn_id"])} == {"pending"}
    with store.session_database_scope(second):
        assert {job["status"] for job in store.inspect_turn_execution(second)["post_commit_jobs"]} == {"terminal_failed"}


def test_router_enters_session_scope_and_rejects_conflicting_scope(monkeypatch, scheduled):
    session_id = _failed_session()
    payload = _request(session_id)
    scopes = []

    @contextmanager
    def scope(actual):
        scopes.append(actual)
        yield

    monkeypatch.setattr(store, "session_database_scope", scope)
    router.dispatch_response("POST", f"/api/sessions/{session_id}/post-commit/control", payload)
    assert scopes == [session_id]
    with pytest.raises(service.ApiError) as caught:
        router.dispatch_response(
            "POST", f"/api/sessions/{session_id}/post-commit/control?session_id=other", payload,
        )
    assert caught.value.code == "SESSION_ID_MISMATCH"


def test_post_commit_endpoint_uses_http_authentication(monkeypatch):
    from personagraph.api import security
    monkeypatch.setattr(security, "_API_TOKEN", "test-token")
    monkeypatch.delenv("PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS", raising=False)
    handler = object.__new__(server.ApiHandler)
    handler.path = "/api/sessions/session-a/post-commit/control"
    handler.headers = Message()
    with pytest.raises(service.ApiError) as caught:
        handler._require_authorization()
    assert caught.value.code == "API_AUTH_REQUIRED"
    handler.headers["Authorization"] = "Bearer test-token"
    handler._require_authorization()
