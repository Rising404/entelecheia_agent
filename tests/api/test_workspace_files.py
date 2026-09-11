"""FR-8 工作区文件管理：禁闭在 working_dir、增删改、删除→回收站、逃逸拦截。"""
from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.api import router
from personagraph.session import store as ss


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _d(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


def _session_with_server_workspace():
    session = _d("POST", "/api/sessions", {"title": "workspace files"})[
        "session"
    ]
    return session["id"], Path(session["working_dir"])


def test_create_list_read_write():
    sid, wd = _session_with_server_workspace()
    # 建文件夹 + 文件
    _d("POST", f"/api/sessions/{sid}/files", {"path": "notes", "kind": "dir"})
    _d("POST", f"/api/sessions/{sid}/files", {"path": "notes/a.md", "kind": "file"})
    # 写内容
    _d("PUT", f"/api/sessions/{sid}/file", {"path": "notes/a.md", "content": "# 大纲\n要点"})
    # 读回
    got = _d("GET", f"/api/sessions/{sid}/file?path=notes/a.md")
    assert got["content"] == "# 大纲\n要点"
    # 列目录
    listed = _d("GET", f"/api/sessions/{sid}/files?path=notes")
    assert [e["name"] for e in listed["entries"]] == ["a.md"]
    # 真实磁盘确有
    assert (wd / "notes" / "a.md").read_text() == "# 大纲\n要点"


def test_delete_goes_to_trash(monkeypatch):
    sid, wd = _session_with_server_workspace()
    (wd / "junk.txt").write_text("x")
    trashed = {}
    monkeypatch.setattr("send2trash.send2trash", lambda p: trashed.setdefault("path", p))
    r = _d("DELETE", f"/api/sessions/{sid}/files", {"path": "junk.txt"})
    assert r["trashed"] is True
    assert trashed["path"].endswith("junk.txt")   # 走了回收站，非 os.remove


def test_escape_is_blocked():
    from personagraph.api.service import ApiError

    sid, _wd = _session_with_server_workspace()
    with pytest.raises(ApiError) as ei:
        _d("POST", f"/api/sessions/{sid}/files", {"path": "../evil.txt", "kind": "file"})
    assert ei.value.code == "PATH_ESCAPES_WORKSPACE"


def test_user_can_manage_own_sensitive_named_files():
    # FR-9 决策①：用户是自己目录的主人，可创建/编辑 .env 等（不再按敏感名拦截用户）
    sid, wd = _session_with_server_workspace()
    r = _d("POST", f"/api/sessions/{sid}/files", {"path": ".env", "kind": "file"})
    assert r["ok"] is True
    _d("PUT", f"/api/sessions/{sid}/file", {"path": ".env", "content": "KEY=123"})
    assert (wd / ".env").read_text() == "KEY=123"
    # 但列目录能看到它（不再隐藏）
    names = [e["name"] for e in _d("GET", f"/api/sessions/{sid}/files")["entries"]]
    assert ".env" in names


def test_agent_private_tree_is_hidden_and_rejected_by_generic_file_management():
    from personagraph.api.service import ApiError

    sid, wd = _session_with_server_workspace()
    private = wd / ".personagraph" / "output" / "session-1" / "draft.txt"
    private.parent.mkdir(parents=True)
    private.write_text("agent state", encoding="utf-8")

    root_names = [
        item["name"]
        for item in _d("GET", f"/api/sessions/{sid}/files")["entries"]
    ]
    assert ".personagraph" not in root_names

    for method, path, body in (
        ("GET", f"/api/sessions/{sid}/files?path=.personagraph", None),
        ("GET", f"/api/sessions/{sid}/file?path=.personagraph/output/session-1/draft.txt", None),
        ("POST", f"/api/sessions/{sid}/files", {"path": ".personagraph/new.txt", "kind": "file"}),
        ("PUT", f"/api/sessions/{sid}/file", {"path": ".personagraph/new.txt", "content": "no"}),
        ("DELETE", f"/api/sessions/{sid}/files", {"path": ".personagraph/output/session-1/draft.txt"}),
    ):
        with pytest.raises(ApiError) as raised:
            _d(method, path, body)
        assert raised.value.code == "AGENT_PRIVATE_PATH"

    assert private.read_text(encoding="utf-8") == "agent state"


def test_session_creation_returns_its_server_managed_workspace():
    created = _d("POST", "/api/sessions", {})["session"]
    working_dir = Path(created["working_dir"])

    assert working_dir.is_dir()
    assert _d("GET", f"/api/sessions/{created['id']}/files")["entries"] == []


def test_cannot_delete_root():
    from personagraph.api.service import ApiError

    sid, _wd = _session_with_server_workspace()
    with pytest.raises(ApiError) as ei:
        _d("DELETE", f"/api/sessions/{sid}/files", {"path": "."})
    assert ei.value.code == "CANNOT_DELETE_ROOT"
