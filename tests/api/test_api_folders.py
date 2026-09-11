"""FR-6 API：文件夹树 CRUD + 系统管理的会话 working_dir。"""
from pathlib import Path

import pytest

from personagraph.api import router, service
from personagraph.configuration import paths
from personagraph.session import store as ss


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _d(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


def test_folder_crud_and_tree():
    root = _d("POST", "/api/folders", {"name": "项目A"})["folder"]["id"]
    child = _d("POST", "/api/folders", {"name": "子", "parent_id": root})["folder"]["id"]
    tree = _d("GET", "/api/folders")["folders"]
    assert [f["name"] for f in tree] == ["项目A"]
    assert tree[0]["children"][0]["id"] == child
    # 重命名
    _d("PATCH", f"/api/folders/{root}", {"name": "项目A改"})
    assert _d("GET", "/api/folders")["folders"][0]["name"] == "项目A改"


def test_folder_move_cycle_rejected():
    a = _d("POST", "/api/folders", {"name": "A"})["folder"]["id"]
    b = _d("POST", "/api/folders", {"name": "B", "parent_id": a})["folder"]["id"]
    with pytest.raises(service.ApiError) as ei:
        _d("PATCH", f"/api/folders/{a}", {"parent_id": b})
    assert ei.value.code == "FOLDER_MOVE_FAILED"


def test_folder_archive_cascade_and_status_filter():
    root = _d("POST", "/api/folders", {"name": "R"})["folder"]["id"]
    _d("POST", "/api/sessions", {"folder_id": root})
    _d("PATCH", f"/api/folders/{root}", {"status": "archived"})
    assert _d("GET", "/api/folders?status=active")["folders"] == []
    assert len(_d("GET", "/api/folders?status=archived")["folders"]) == 1
    # 恢复
    _d("PATCH", f"/api/folders/{root}", {"status": "active"})
    assert len(_d("GET", "/api/folders?status=active")["folders"]) == 1


def test_delete_empty_only():
    root = _d("POST", "/api/folders", {"name": "R"})["folder"]["id"]
    _d("POST", "/api/sessions", {"folder_id": root})
    with pytest.raises(service.ApiError) as ei:
        _d("DELETE", f"/api/folders/{root}")
    assert ei.value.code == "FOLDER_NOT_EMPTY"


def test_session_create_accepts_selected_directory(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    default_root = Path(paths.DEFAULT_SESSION_PROJECTS_DIR)

    session = _d("POST", "/api/sessions", {"working_dir": str(selected)})["session"]
    assert session["working_dir"] == str(selected.resolve())

    assert list(selected.iterdir()) == []
    assert not default_root.exists()


def test_session_create_allocates_an_empty_directory_below_default_root():
    session = _d("POST", "/api/sessions", {"title": "系统目录"})["session"]
    working_dir = Path(session["working_dir"])

    assert working_dir.parent == Path(paths.DEFAULT_SESSION_PROJECTS_DIR).resolve()
    assert working_dir.is_dir()
    assert list(working_dir.iterdir()) == []


def test_session_working_dir_cannot_be_rebound_or_unbound(tmp_path):
    created = _d("POST", "/api/sessions", {"title": "保持原目录"})["session"]
    session_id = created["id"]
    original_working_dir = created["working_dir"]
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    for supplied in (str(replacement), "", None, original_working_dir):
        with pytest.raises(service.ApiError) as error:
            _d(
                "PATCH",
                f"/api/sessions/{session_id}",
                {"title": "不得部分更新", "working_dir": supplied},
            )

        assert error.value.code == "SESSION_WORKING_DIR_MANAGED"
        assert error.value.status == 409
        assert error.value.details == {
            "field": "working_dir",
            "session_id": session_id,
        }
        unchanged = _d("GET", f"/api/sessions/{session_id}")["session"]
        assert unchanged["title"] == "保持原目录"
        assert unchanged["working_dir"] == original_working_dir

    assert list(replacement.iterdir()) == []


def test_session_metadata_updates_do_not_change_system_working_dir():
    created = _d("POST", "/api/sessions", {"title": "before"})["session"]

    updated = _d(
        "PATCH",
        f"/api/sessions/{created['id']}",
        {"title": "after"},
    )["session"]

    assert updated["title"] == "after"
    assert updated["working_dir"] == created["working_dir"]
