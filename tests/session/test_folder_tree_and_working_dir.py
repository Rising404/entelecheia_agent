"""FR-6 后端地基：文件夹树 + move 防环 + status 级联 + 固定 working_dir。"""
import pytest

from personagraph.session import store as ss


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def test_folder_tree_nesting_and_counts():
    root = ss.create_folder("根")
    child = ss.create_folder("子", parent_id=root)
    ss.create_folder("孙", parent_id=child)
    ss.create_session("Entelecheia", title="s1", folder_id=child)
    tree = ss.folder_tree()
    assert [n["name"] for n in tree] == ["根"]
    r = tree[0]
    assert [c["name"] for c in r["children"]] == ["子"]
    sub = r["children"][0]
    assert sub["session_count"] == 1
    assert [g["name"] for g in sub["children"]] == ["孙"]


def test_move_folder_prevents_cycles():
    a = ss.create_folder("A")
    b = ss.create_folder("B", parent_id=a)
    # 不能把 A 移进它的后代 B
    ok, reason = ss.move_folder(a, b)
    assert ok is False and reason == "cannot_move_into_descendant"
    # 不能移进自身
    ok, reason = ss.move_folder(a, a)
    assert ok is False and reason == "cannot_move_into_self"
    # 合法移动：B 移到根
    ok, reason = ss.move_folder(b, None)
    assert ok is True
    assert ss.get_folder(b)["parent_id"] is None


def test_folder_status_cascade_archive_and_restore():
    root = ss.create_folder("项目")
    child = ss.create_folder("子任务", parent_id=root)
    s1 = ss.create_session("Entelecheia", title="s1", folder_id=root)
    s2 = ss.create_session("Entelecheia", title="s2", folder_id=child)
    # 归档根 → 级联子文件夹 + 两个会话
    ok, _ = ss.set_folder_status(root, "archived")
    assert ok
    assert ss.get_folder(root)["status"] == "archived"
    assert ss.get_folder(child)["status"] == "archived"
    assert ss.get_session(s1)["status"] == "archived"
    assert ss.get_session(s2)["status"] == "archived"
    # active 树里看不到已归档
    assert ss.folder_tree("active") == []
    assert len(ss.folder_tree("archived")) == 1
    # 恢复 → 级联回 active
    ok, _ = ss.set_folder_status(root, "active")
    assert ok
    assert ss.get_folder(child)["status"] == "active"
    assert ss.get_session(s1)["status"] == "active"
    assert ss.get_session(s2)["status"] == "active"


def test_session_working_dir_can_only_be_set_during_low_level_creation():
    sid = ss.create_session("Entelecheia", title="w", working_dir="/tmp/proj")
    assert ss.get_session(sid)["working_dir"] == "/tmp/proj"
    assert not hasattr(ss, "bind_session_workspace_layout")
    assert not hasattr(ss, "set_session_working_dir")
