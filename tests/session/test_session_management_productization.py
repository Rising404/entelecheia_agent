import json

import pytest

from personagraph.session import store as ss
from tests.helpers.session_records import append_test_turn


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _session(persona: str, title: str, folder_id: str | None = None) -> str:
    sid = ss.create_session(persona, title=title, folder_id=folder_id)
    append_test_turn(sid, "user", f"{title} 用户问题")
    append_test_turn(sid, "assistant", f"{title} 助手回答")
    return sid


def test_explicit_database_override_is_available_to_session_owned_stores(tmp_path):
    assert ss.current_session_database_path() == (
        tmp_path / "sessions.sqlite"
    ).resolve()


def test_list_sessions_filters_by_status_folder_persona_query_and_limit():
    f1 = ss.create_folder("项目A")
    f2 = ss.create_folder("项目B")
    a = _session("Entelecheia", "AI PM 面试准备", f1)
    b = _session("Entelecheia", "工具链整理", f2)
    c = _session("oberon", "AI PM 简历复盘", f1)
    assert ss.archive_session(b)
    assert ss.trash_session(c)

    assert [r["id"] for r in ss.list_sessions(folder_id=f1)] == [a]
    assert [r["id"] for r in ss.list_sessions(status="archived")] == [b]
    assert [r["id"] for r in ss.list_sessions(status="trashed")] == [c]
    assert {r["id"] for r in ss.list_sessions(status="all", persona_id="Entelecheia")} == {a, b}
    assert [r["id"] for r in ss.list_sessions(status="all", query="简历")] == [c]
    assert len(ss.list_sessions(status="all", limit=1)) == 1


def test_search_sessions_hits_title_and_turn_content_excludes_trash_by_default():
    active = _session("Entelecheia", "阶段计划讨论")
    trashed = _session("Entelecheia", "阶段计划废稿")
    ss.trash_session(trashed)
    turn_hit = _session("Entelecheia", "普通标题")
    append_test_turn(turn_hit, "user", "这里讨论 ComfyUI 工作流")

    default_hits = {r["id"] for r in ss.search_sessions("阶段计划")}
    assert default_hits == {active}

    all_hits = {r["id"] for r in ss.search_sessions("阶段计划", status="all")}
    assert all_hits == {active, trashed}

    content_hits = ss.search_sessions("ComfyUI")
    assert [r["id"] for r in content_hits] == [turn_hit]
    assert content_hits[0]["hit_count"] == 1
    assert "ComfyUI" in content_hits[0]["match_snippet"]


def test_export_session_markdown_and_json():
    sid = _session("Entelecheia", "导出测试")
    md = ss.export_session(sid, "md")
    assert "# 导出测试" in md
    assert "session_id" in md
    assert "导出测试 用户问题" in md

    data = json.loads(ss.export_session(sid, "json"))
    assert data["session"]["id"] == sid
    assert "persona_id" not in data["session"]
    assert "persona" not in md.lower()
    assert len(data["turns"]) == 2
    assert ss.export_session("missing", "md") is None
