"""DocBench 使用正式 Session 创建服务，不能受 GUI 目录和自动标题影响。"""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest

from evals.docbench.reproduce_or_run_script import config, runner
from personagraph.api.service import session_titles, sessions
from personagraph.configuration import app_settings
from personagraph.session import store
from tests.helpers.session_records import complete_test_turn_execution


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def test_run_root_overrides_gui_config_in_real_session_creation(tmp_path, monkeypatch):
    bench = tmp_path / "docbench"
    gui = tmp_path / "gui-projects"
    app_settings.update_config({"default_projects_dir": str(gui)})
    monkeypatch.setattr(config, "docbench_root", lambda environment=None: bench)
    environment = runner._bind_run_workspace(
        {"PERSONAGRAPH_DEFAULT_PROJECTS_DIR": str(gui)}, run_root=bench / "runs/run-1",
    )
    monkeypatch.setenv(
        "PERSONAGRAPH_DEFAULT_PROJECTS_DIR", environment["PERSONAGRAPH_DEFAULT_PROJECTS_DIR"],
    )

    first = sessions.create_session({"title": "docbench-formal-l1-case-1"})["session"]
    second = sessions.create_session({"title": "docbench-formal-l1-case-2"})["session"]

    expected = bench / "workspaces/run-1"
    assert Path(first["working_dir"]).parent == expected
    assert Path(second["working_dir"]).parent == expected
    assert store.get_session(first["id"])["project_id"] != store.get_session(second["id"])["project_id"]
    assert first["id"] != second["id"]
    assert not gui.exists()


def test_fixed_eval_title_does_not_invoke_autoname_model(monkeypatch):
    title = "docbench-formal-l1-case-1"
    created = sessions.create_session({"title": title})["session"]
    with store.session_database_scope(created["id"]):
        complete_test_turn_execution(
            created["id"], 0, user_content="根据附件回答问题", assistant_content="回答结果",
        )
    model = Mock(side_effect=AssertionError("evaluation must not request a model title"))
    monkeypatch.setattr(session_titles, "_proposed_title", model)
    result = Future()
    session_titles._run_title_job(("test", "test", created["id"]), result)

    assert result.result()["title"] == title
    model.assert_not_called()
