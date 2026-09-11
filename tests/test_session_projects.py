"""项目按照会话所绑定的目录对会话进行分组。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personagraph.session import project_catalog as session_projects
from personagraph.configuration import paths
from personagraph.workspace.binding import ensure_or_open_layout


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(session_projects, "DB_PATH", tmp_path / "project_catalog.sqlite")
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")


def test_catalog_assigns_one_stable_identity_and_document_database(tmp_path):
    root = tmp_path / "report"
    root.mkdir()

    first = session_projects.remember(str(root))
    second = session_projects.remember(str(root / "."))

    assert second.project_id == first.project_id
    assert first.canonical_root == str(root.resolve())
    assert first.documents_db_path == str(
        tmp_path / "projects" / first.project_id / "documents.sqlite"
    )
    assert session_projects.DB_PATH.read_bytes().startswith(b"SQLite format 3\x00")

    with sqlite3.connect(session_projects.DB_PATH) as connection:
        stored = connection.execute(
            "SELECT project_id, canonical_root FROM projects"
        ).fetchone()
    assert stored == (first.project_id, str(root.resolve()))


def test_project_catalog_rejects_a_preplanted_database_symlink(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    outside = tmp_path / "outside"
    state_dir.mkdir(exist_ok=True)
    outside.mkdir()
    escaped = outside / "escaped.sqlite"
    planted = state_dir / "project_catalog.sqlite"
    planted.symlink_to(escaped)
    monkeypatch.setattr(session_projects, "DB_PATH", planted)

    with pytest.raises(ValueError, match="project catalog"):
        session_projects.list_registered_projects()
    assert not escaped.exists()


def test_managed_workspace_id_becomes_the_project_id(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    layout = ensure_or_open_layout(root, session_id="session-1")

    project = session_projects.remember(str(root))

    assert project.project_id == layout.workspace_id


def test_grouping_registers_a_stable_project_identity(tmp_path):
    root = tmp_path / "grouped"
    root.mkdir()
    sessions = [{"id": "a", "working_dir": str(root)}]

    first = session_projects.group_sessions(sessions)["projects"][0]
    second = session_projects.group_sessions(sessions)["projects"][0]

    assert second["project_id"] == first["project_id"]
    assert first["canonical_root"] == str(root.resolve())
    assert Path(first["documents_db_path"]) == (
        tmp_path / "projects" / first["project_id"] / "documents.sqlite"
    )


def test_catalog_allows_session_metadata_tables_in_the_same_database(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    session_projects.remember(str(first_root))

    with sqlite3.connect(session_projects.DB_PATH) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE session_folders (id TEXT PRIMARY KEY)")

    second = session_projects.remember(str(second_root))

    assert second.canonical_root == str(second_root.resolve())


def test_grouping_needs_no_registration():
    grouped = session_projects.group_sessions([
        {"id": "a", "working_dir": "/tmp/report"},
        {"id": "b", "working_dir": "/tmp/report"},
    ])
    assert [(item["name"], len(item["sessions"])) for item in grouped["projects"]] == [("report", 2)]
    assert grouped["unbound"] == []


def test_unbound_sessions_stay_reachable():
    grouped = session_projects.group_sessions([{"id": "a", "working_dir": None}])
    assert grouped["projects"] == []
    assert [item["id"] for item in grouped["unbound"]] == ["a"]


def test_a_project_with_no_sessions_left_stops_being_listed():
    """project 的存在完全由"有没有会话绑着它"决定。

    以前注册过的路径会一直占着一行，于是一个会话全部归档的项目会在活跃区留下一个
    空壳，还得专门做个"不再列出"的按钮去扫它。存在改成派生之后，两个问题一起没了。
    """

    session_projects.remember("/tmp/empty")
    grouped = session_projects.group_sessions([])
    assert grouped["projects"] == []


def test_remember_keeps_the_chosen_name_even_while_nothing_is_listed():
    """名字要记住，哪怕这个项目当前一个会话都没有。

    这是注册表现在唯一的职责：存派生推不出来的东西。下次再有会话绑到同一个目录，
    用户起的名字应当原样回来，而不是退回文件夹名。
    """

    session_projects.remember("/tmp/report", name="季度报告")
    session_projects.remember("/tmp/report")

    assert session_projects.group_sessions([])["projects"] == []

    grouped = session_projects.group_sessions([{"id": "a", "working_dir": "/tmp/report"}])
    assert [item["name"] for item in grouped["projects"]] == ["季度报告"]


def test_pinned_projects_sort_ahead_of_the_rest():
    session_projects.set_pinned("/tmp/zzz-pinned", True)
    grouped = session_projects.group_sessions([
        {"id": "a", "working_dir": "/tmp/aaa-plain"},
        {"id": "b", "working_dir": "/tmp/zzz-pinned"},
    ])
    assert [item["name"] for item in grouped["projects"]] == ["zzz-pinned", "aaa-plain"]


def test_pinning_is_a_toggle_and_survives_a_rename():
    session_projects.remember("/tmp/pin", name="旧名")
    session_projects.set_pinned("/tmp/pin", True)
    session_projects.rename("/tmp/pin", "新名")

    sessions = [{"id": "a", "working_dir": "/tmp/pin"}]
    project = session_projects.group_sessions(sessions)["projects"][0]
    assert (project["name"], project["pinned"]) == ("新名", True)

    session_projects.set_pinned("/tmp/pin", False)
    assert session_projects.group_sessions(sessions)["projects"][0]["pinned"] is False


def test_forgetting_never_removes_the_sessions_themselves():
    original = session_projects.remember("/tmp/report")
    assert session_projects.forget("/tmp/report") is True
    grouped = session_projects.group_sessions([{"id": "a", "working_dir": "/tmp/report"}])
    # 目录还是会话说了算：忘记只是不再单独记账，会话自己会把它带回来。
    assert [item["path"] for item in grouped["projects"]] == ["/tmp/report"]
    assert grouped["projects"][0]["project_id"] == original.project_id


def test_a_damaged_registry_does_not_hide_sessions(tmp_path):
    (tmp_path / "project_catalog.sqlite").write_text("{ not sqlite", encoding="utf-8")
    grouped = session_projects.group_sessions([{"id": "a", "working_dir": "/tmp/report"}])
    assert [item["path"] for item in grouped["projects"]] == ["/tmp/report"]


def test_empty_path_is_refused():
    with pytest.raises(ValueError):
        session_projects.remember("   ")


class _Result:
    def __init__(self, reply: str) -> None:
        self.reply = reply


def test_autoname_retries_once_before_falling_back(monkeypatch):
    from personagraph.api.service import session_titles as service

    calls: list[int] = []

    def flaky(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("撞上正在跑的运行时")
        return _Result('{"title":"排查季度销售数据"}')

    monkeypatch.setattr("personagraph.model_io.gateway.complete_structured", flaky)
    assert service._proposed_title("问题", "回答", "问题") == ""
    assert service._proposed_title("问题", "回答", "问题") == "排查季度销售数据"
    assert len(calls) == 2


def test_autoname_gives_up_quietly_after_two_failures(monkeypatch):
    from personagraph.api.service import session_titles as service

    def always_fails(*_args, **_kwargs):
        raise RuntimeError("模型不可用")

    monkeypatch.setattr(
        "personagraph.model_io.gateway.complete_structured",
        always_fails,
    )
    # 起不出名字不是错误，只是没名字：调用方拿到空串，自己退回原话。
    assert service._proposed_title("问题", "回答", "问题") == ""


def test_dragged_order_wins_over_the_alphabet():
    sessions = [{"id": i, "working_dir": p} for i, p in enumerate(["/w/a", "/w/b", "/w/c"])]
    assert [x["name"] for x in session_projects.group_sessions(sessions)["projects"]] == ["a", "b", "c"]

    session_projects.set_order(["/w/c", "/w/a", "/w/b"])
    assert [x["name"] for x in session_projects.group_sessions(sessions)["projects"]] == ["c", "a", "b"]


def test_a_project_that_has_never_been_ordered_lands_at_the_end():
    """新出现的 project 稳定落在末尾，而不是按名字插进用户排好的序列中间。

    插队更让人意外：用户刚把顺序拖好，下次打开发现中间多了一条。
    """

    session_projects.set_order(["/w/z", "/w/y"])
    sessions = [
        {"id": 1, "working_dir": "/w/z"},
        {"id": 2, "working_dir": "/w/y"},
        {"id": 3, "working_dir": "/w/aaa-new"},
    ]
    assert [x["name"] for x in session_projects.group_sessions(sessions)["projects"]] == ["z", "y", "aaa-new"]


def test_pinning_lifts_a_project_without_disturbing_the_rest_of_the_order():
    session_projects.set_order(["/w/a", "/w/b", "/w/c"])
    session_projects.set_pinned("/w/c", True)
    sessions = [{"id": i, "working_dir": p} for i, p in enumerate(["/w/a", "/w/b", "/w/c"])]
    assert [x["name"] for x in session_projects.group_sessions(sessions)["projects"]] == ["c", "a", "b"]


def test_reordering_rejects_a_duplicated_path():
    with pytest.raises(ValueError):
        session_projects.set_order(["/w/a", "/w/a"])
