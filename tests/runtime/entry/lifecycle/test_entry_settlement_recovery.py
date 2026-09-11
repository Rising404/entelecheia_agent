"""持久入口稳定恢复的单元与导入边界覆盖。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.lifecycle import settlement as recovery_module
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.entry.lifecycle.settlement import (
    recover_completed_entry_turn_from_commit,
)
from personagraph.runtime.turn.persisted_projection import require_entry_window_state


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="recover the durable settlement",
        attachment_ids=(),
        window_revision=7,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _active_window(**overrides: object) -> dict[str, object]:
    window: dict[str, object] = {
        "turn_id": "turn-1",
        "window_state": "active",
        "stage": "PERSIST",
        "state_version": 12,
        "current_work_run_id": None,
        "current_attempt_id": None,
        "latest_checkpoint_id": None,
    }
    window.update(overrides)
    return window


class _RecoveryStore:
    def __init__(
        self,
        *,
        committed: dict[str, Any] | None = None,
        window: object | None = None,
        work_run_ids: tuple[str, ...] = ("run-1",),
        fail_on: str | None = None,
    ) -> None:
        self.committed = committed or {
            "turn_id": "turn-1",
            "assistant_content": "durable reply",
        }
        self.window = _active_window() if window is None else window
        self.work_run_ids = work_run_ids
        self.fail_on = fail_on
        self.calls: list[str] = []

    def _read(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_on == name:
            raise OSError(f"lost {name} read")

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None:
        self._read("committed")
        assert (session_id, run_id) == ("session-1", "commit_turn-1")
        return self.committed

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
        self._read("window")
        assert session_id == "session-1"
        return {"window": self.window}

    def list_turn_linked_work_run_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]:
        self._read("links")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.work_run_ids


def test_completed_recovery_reads_the_exact_durable_chain_and_preserves_empty_reply() -> (
    None
):
    store = _RecoveryStore(
        committed={"turn_id": "turn-1", "assistant_content": ""},
        window=_active_window(window_state="post_commit_pending", state_version=14),
    )

    result = recover_completed_entry_turn_from_commit(
        accepted=_accepted(),
        processing_level="L2",
        related_insession_task_ids=("task-1",),
        store=store,
    )

    assert result is not None
    assert result.status == "completed"
    assert result.reply == ""
    assert result.processing_level == "L2"
    assert result.related_insession_task_ids == ("task-1",)
    assert result.work_run_ids == ("run-1",)
    assert result.window_state == "post_commit_pending"
    assert result.window_revision == 14
    assert store.calls == ["committed", "window", "links"]


def test_completed_recovery_keeps_supplied_work_run_ids_without_another_read() -> None:
    store = _RecoveryStore()

    result = recover_completed_entry_turn_from_commit(
        accepted=_accepted(),
        processing_level="L0",
        related_insession_task_ids=(),
        store=store,
        work_run_ids=("supplied-run",),
    )

    assert result is not None
    assert result.work_run_ids == ("supplied-run",)
    assert store.calls == ["committed", "window"]


@pytest.mark.parametrize("supplied_work_run_ids", ((), ("old-work-run",)))
def test_l1_commit_recovery_preserves_reply_without_task_or_work_run_links(
    supplied_work_run_ids: tuple[str, ...],
) -> None:
    reply = "已正式提交的旧答复 task-1，保留原文。"
    store = _RecoveryStore(
        committed={"turn_id": "turn-1", "assistant_content": reply},
        window=_active_window(window_state="post_commit_pending", state_version=14),
        fail_on="links",
    )

    result = recover_completed_entry_turn_from_commit(
        accepted=_accepted(),
        processing_level="L1",
        related_insession_task_ids=("old-task",),
        store=store,
        work_run_ids=supplied_work_run_ids,
    )

    assert result is not None
    assert result.status == "completed"
    assert result.processing_level == "L1"
    assert result.reply == reply
    assert result.related_insession_task_ids == ()
    assert result.work_run_ids == ()
    assert result.window_state == "post_commit_pending"
    assert result.window_revision == 14
    assert store.calls == ["committed", "window"]


@pytest.mark.parametrize(
    ("committed", "window", "fail_on"),
    (
        (None, None, None),
        ({"turn_id": "other", "assistant_content": "reply"}, None, None),
        ({"turn_id": "turn-1", "assistant_content": object()}, None, None),
        ({"turn_id": "turn-1", "assistant_content": "reply"}, object(), None),
        (
            {"turn_id": "turn-1", "assistant_content": "reply"},
            _active_window(window_state="unknown"),
            None,
        ),
        (None, None, "committed"),
        (None, None, "window"),
        (None, None, "links"),
    ),
)
def test_completed_recovery_fails_closed_for_missing_malformed_or_unreadable_facts(
    committed: dict[str, Any] | None,
    window: object | None,
    fail_on: str | None,
) -> None:
    store = _RecoveryStore(committed=committed, window=window, fail_on=fail_on)
    if committed is None and fail_on is None:
        store.committed = None

    assert (
        recover_completed_entry_turn_from_commit(
            accepted=_accepted(),
            processing_level="L2",
            related_insession_task_ids=(),
            store=store,
        )
        is None
    )


def test_window_state_decoder_keeps_the_closed_public_vocabulary() -> None:
    assert require_entry_window_state("empty") == "empty"
    assert require_entry_window_state("active") == "active"
    assert require_entry_window_state("post_commit_pending") == "post_commit_pending"
    assert require_entry_window_state("interrupted") == "interrupted"
    with pytest.raises(RuntimeError, match="unexpected execution Window state"):
        require_entry_window_state("unknown")


def test_recovery_module_has_only_read_projection_authority() -> None:
    tree = ast.parse(Path(recovery_module.__file__).read_text(encoding="utf-8"))
    relative_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    store_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    }
    recovery_protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "EntrySettlementRecoveryStorePort"
    )
    recovery_protocol_methods = {
        node.name
        for node in recovery_protocol.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "entry" not in relative_imports
    assert store_calls == {
        "get_committed_turn_pair",
        "inspect_turn_execution",
        "list_turn_linked_work_run_ids",
    }
    assert recovery_protocol_methods == store_calls
    assert not {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "l2_store" in node.module
    }
    assert (
        not {
            "advance_turn_execution_window",
            "append_runtime_turn_event",
            "finalize_turn_execution",
            "finalize_verified_turn_execution",
            "mark_turn_execution_interrupted",
        }
        & store_calls
    )
    assert not hasattr(
        recovery_module,
        "project_pending_verified_publication_result",
    )
    assert not hasattr(
        recovery_module,
        "EntryCompletedVerifiedDeliveryStorePort",
    )
    assert not hasattr(recovery_module, "require_entry_window_state")

    entry_tree = ast.parse(Path(entry_application.__file__).read_text(encoding="utf-8"))
    assert not {
        node.name
        for node in entry_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_completed_entry_turn_from_commit",
            "_pending_verified_publication_result",
            "_window_state",
        }
    }
