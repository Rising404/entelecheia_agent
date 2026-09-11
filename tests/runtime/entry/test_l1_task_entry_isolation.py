"""L1 首次执行和收尾不依赖长期任务关联。"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from personagraph.runtime.entry import application as entry
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.l1 import entry_lane
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.session import store as session_store


def test_new_l1_turn_does_not_read_or_link_long_term_tasks(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = session_store.create_session("L1 task isolation", working_dir=str(workspace))
    task_reads: list[str] = []

    def unexpected_task_read(*_args, **_kwargs):
        task_reads.append("task-read")
        raise AssertionError("L1 read long-term Task state")

    monkeypatch.setattr(session_store, "list_insession_task_catalog", unexpected_task_read)
    monkeypatch.setattr(session_store, "list_pending_user_questions", unexpected_task_read)
    monkeypatch.setattr(session_store, "list_turn_insession_task_ids", unexpected_task_read)
    monkeypatch.setattr(session_store, "list_turn_linked_work_run_ids", unexpected_task_read)
    monkeypatch.setattr(
        entry, "classify_turn",
        lambda *_args, **_kwargs: EntryClassification(processing_level="L1"),
    )

    result = entry.run_entry_turn(
        user_input="请回答当前问题",
        features={"l1_max_attempts": 3, "l1_semantic_verification_mode": "off"},
        session_id=session_id,
        client_request_id="l1-no-task-context",
        store=session_store,
    )

    assert result.status == "completed"
    assert result.processing_level == "L1"
    assert result.related_insession_task_ids == ()
    assert task_reads == []
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id,
    )
    assert execution["run"]["status"] == "completed"
    assert execution["state"]["plan_json"]
    assert execution["attempts"]

    # 同一请求重复到达应只读重放正文，不重新执行或查询长期任务。
    replayed = entry.run_entry_turn(
        user_input="请回答当前问题",
        features={"l1_max_attempts": 3, "l1_semantic_verification_mode": "off"},
        session_id=session_id,
        client_request_id="l1-no-task-context",
        store=session_store,
    )
    assert replayed.status == "completed"
    assert replayed.turn_id == result.turn_id
    assert replayed.reply == result.reply
    assert replayed.related_insession_task_ids == ()
    assert replayed.work_run_ids == ()
    assert task_reads == []


def test_l1_finalizer_cannot_receive_or_publish_task_links(monkeypatch):
    assert "related_insession_task_ids" not in inspect.signature(
        entry._run_l1_controller_and_finalize
    ).parameters
    captured: dict[str, object] = {}

    def run_l1(**kwargs):
        assert "related_insession_task_ids" not in kwargs
        return SimpleNamespace(reply="当前答复", turn_window_revision=9)

    def finalize(**kwargs):
        captured.update(kwargs)
        return "finalized"

    monkeypatch.setattr(entry_lane, "run_l1_entry_lane", run_l1)
    monkeypatch.setattr(entry, "_finalize_formal_reply", finalize)
    result = entry._run_l1_controller_and_finalize(
        accepted=SimpleNamespace(),
        context=SimpleNamespace(),
        l1_turn_run_id="run-1",
        revision=7,
        deadline=TurnDeadline.starting_now(60),
        features={},
        emit=lambda _event: None,
        store=SimpleNamespace(),
        lease_owner="lease-1",
    )

    assert result == "finalized"
    assert captured["processing_level"] == "L1"
    assert captured["related_insession_task_ids"] == ()
    assert captured["reply"] == "当前答复"
