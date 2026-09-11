from concurrent.futures import Future

import pytest

from personagraph.api import service
from personagraph.api.service import session_titles
from personagraph.session import store
from tests.helpers.session_records import complete_test_turn_execution


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def make_session(question="整理这份报告"):
    session_id = service.create_session({"title": question})["session"]["id"]
    with store.session_database_scope(session_id):
        complete_test_turn_execution(session_id, 0, user_content=question, assistant_content="报告内容已整理")
    return session_id


def test_title_uses_committed_first_pair_and_keeps_project(monkeypatch):
    session_id = make_session()
    original = store.get_session(session_id)
    seen = []

    def generate(question, answer, fallback):
        seen.append((question, answer))
        return "报告整理"

    monkeypatch.setattr(session_titles, "_proposed_title", generate)
    with store.session_database_scope(session_id):
        result = session_titles.generate_session_title(session_id)
    assert result["title"] == "报告整理"
    assert seen == [("整理这份报告", "报告内容已整理")]
    current = store.get_session(session_id)
    assert current["working_dir"] == original["working_dir"]
    assert current["project_id"] == original["project_id"]


def test_manual_title_wins_while_model_is_generating(monkeypatch):
    session_id = make_session()

    def generate(*_args):
        store.rename_session(session_id, "我自己的标题")
        return "模型建议的标题"

    monkeypatch.setattr(session_titles, "_proposed_title", generate)
    with store.session_database_scope(session_id):
        assert session_titles.generate_session_title(session_id)["title"] == "我自己的标题"


def test_model_failure_leaves_placeholder_without_raising(monkeypatch):
    session_id = make_session()
    monkeypatch.setattr(session_titles, "_proposed_title", lambda *_args: "")
    with store.session_database_scope(session_id):
        assert session_titles.generate_session_title(session_id)["title"] == "整理这份报告"


def test_background_worker_reestablishes_session_scope(monkeypatch):
    session_id = make_session()
    monkeypatch.setattr(session_titles, "_proposed_title", lambda *_args: "后台标题")
    future = Future()
    session_titles._run_title_job(("test", "test", session_id), future)
    assert future.result()["title"] == "后台标题"


def test_no_committed_turn_does_not_invoke_model(monkeypatch):
    session_id = service.create_session({"title": "新会话"})["session"]["id"]
    monkeypatch.setattr(session_titles, "_proposed_title", lambda *_args: pytest.fail("no committed input"))
    with store.session_database_scope(session_id):
        assert session_titles.generate_session_title(session_id)["title"] == "新会话"
