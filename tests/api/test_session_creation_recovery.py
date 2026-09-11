from concurrent.futures import ThreadPoolExecutor

import pytest

from personagraph.api import router, service
from personagraph.api.service import sessions
from personagraph.configuration import app_settings
from personagraph.session import catalog, store


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def request(**overrides):
    return {"client_request_id": "create-request-1", "title": "整理报告", **overrides}


def test_completed_creation_replays_after_settings_and_title_change(tmp_path):
    original = service.create_session(request())["session"]
    store.rename_session(original["id"], "人工标题")
    app_settings.update_config({"default_projects_dir": str(tmp_path / "changed")})

    replay = service.create_session(request())["session"]

    assert replay["id"] == original["id"]
    assert replay["working_dir"] == original["working_dir"]
    assert replay["title"] == "人工标题"
    assert len(store.list_sessions()) == 1
    assert not (tmp_path / "changed").exists()


def test_request_id_cannot_be_reused_for_another_creation_payload():
    service.create_session(request())

    with pytest.raises(service.ApiError) as exc:
        service.create_session(request(title="另一个标题"))

    assert exc.value.code == "SESSION_CREATION_REQUEST_CONFLICT"
    assert len(store.list_sessions()) == 1


def test_concurrent_request_does_not_create_a_second_directory(monkeypatch):
    create_directory = sessions._create_default_project_directory
    observed = []
    entered = False

    def overlapping(title):
        nonlocal entered
        if entered:
            return create_directory(title)
        entered = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(service.ApiError) as exc:
                pool.submit(service.create_session, request()).result()
        observed.append(exc.value.code)
        return create_directory(title)

    monkeypatch.setattr(sessions, "_create_default_project_directory", overlapping)
    service.create_session(request())

    assert observed == ["SESSION_CREATION_IN_PROGRESS"]
    assert len(store.list_sessions()) == 1


def test_rejected_directory_releases_request_for_retry(tmp_path):
    chosen = tmp_path / "not-created-yet"
    payload = request(working_dir=str(chosen))

    with pytest.raises(service.ApiError) as exc:
        service.create_session(payload)
    assert exc.value.details["creation_not_committed"] is True

    chosen.mkdir()
    assert service.create_session(payload)["session"]["working_dir"] == str(chosen)


def test_failed_response_projection_can_replay_published_session(monkeypatch):
    original = sessions.require_session
    calls = 0

    def unavailable_once(session_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("response unavailable after publication")
        return original(session_id)

    monkeypatch.setattr(sessions, "require_session", unavailable_once)
    with pytest.raises(RuntimeError):
        service.create_session(request())
    replay = service.create_session(request())["session"]

    assert len(store.list_sessions()) == 1
    assert replay["id"] == store.list_sessions()[0]["id"]


def test_purged_session_request_does_not_recreate_it():
    created = service.create_session(request())["session"]
    assert store.purge_session(created["id"])

    with pytest.raises(service.ApiError) as exc:
        service.create_session(request())

    assert exc.value.code == "SESSION_CREATION_RESULT_UNAVAILABLE"
    assert store.list_sessions() == []


def test_http_creation_body_preserves_idempotency_key():
    first = router.dispatch_response("POST", "/api/sessions", request()).payload["session"]
    second = router.dispatch_response("POST", "/api/sessions", request()).payload["session"]
    assert first == second


def test_abrupt_interruption_keeps_pending_request_closed(monkeypatch):
    class Interrupted(BaseException):
        pass

    def interrupted(_title):
        raise Interrupted()

    monkeypatch.setattr(sessions, "_create_default_project_directory", interrupted)
    with pytest.raises(Interrupted):
        service.create_session(request())
    with pytest.raises(service.ApiError) as exc:
        service.create_session(request())
    assert exc.value.code == "SESSION_CREATION_IN_PROGRESS"
    assert store.list_sessions() == []


@pytest.mark.parametrize("request_id", ["", " ", "../escape", "x" * 129, 42])
def test_invalid_creation_request_id_is_rejected(request_id):
    with pytest.raises(service.ApiError) as exc:
        service.create_session(request(client_request_id=request_id))
    assert exc.value.code == "INVALID_CLIENT_REQUEST_ID"
    assert catalog.SessionCatalog().list_sessions() == []
