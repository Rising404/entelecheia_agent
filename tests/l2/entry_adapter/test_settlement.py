"""L2 已验证 Delivery 待发布恢复的单元与归属边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from personagraph.l2.entry_adapter import settlement as settlement_module
from personagraph.l2.entry_adapter.settlement import (
    project_pending_verified_publication_result,
)
from personagraph.runtime.entry.lifecycle import settlement as entry_recovery_module
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="recover the verified delivery",
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
        "current_l1_turn_run_id": None,
        "current_l1_attempt_id": None,
        "latest_checkpoint_id": None,
    }
    window.update(overrides)
    return window


class _WindowStore:
    def __init__(
        self,
        *,
        window: object | None = None,
        fail: bool = False,
    ) -> None:
        self.window = _active_window() if window is None else window
        self.fail = fail
        self.calls: list[str] = []

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
        self.calls.append("window")
        if self.fail:
            raise OSError("lost window read")
        assert session_id == "session-1"
        return {"window": self.window}


class _VerifiedDeliveryStore:
    def __init__(
        self,
        completed_delivery_ids: tuple[str, ...] = ("delivery-1",),
        *,
        fail: bool = False,
    ) -> None:
        self.completed_delivery_ids = completed_delivery_ids
        self.fail = fail
        self.calls: list[str] = []

    def list_turn_completed_verified_delivery_ids(
        self,
        **kwargs: object,
    ) -> tuple[str, ...]:
        self.calls.append("deliveries")
        if self.fail:
            raise OSError("lost deliveries read")
        assert kwargs == {"session_id": "session-1", "turn_id": "turn-1"}
        return self.completed_delivery_ids


@pytest.mark.parametrize(
    ("completed_delivery_ids", "window"),
    (
        (("other-delivery",), _active_window()),
        (("delivery-1", "delivery-2"), _active_window()),
        (("delivery-1",), _active_window(turn_id="other-turn")),
        (("delivery-1",), _active_window(window_state="post_commit_pending")),
        (("delivery-1",), _active_window(stage="L2_PLAN")),
        (("delivery-1",), _active_window(current_work_run_id="run-1")),
        (("delivery-1",), _active_window(current_attempt_id="attempt-1")),
        (("delivery-1",), _active_window(current_l1_turn_run_id="l1-run-1")),
        (("delivery-1",), _active_window(current_l1_attempt_id="l1-attempt-1")),
        (("delivery-1",), _active_window(latest_checkpoint_id="checkpoint-1")),
        (("delivery-1",), _active_window(state_version=0)),
    ),
)
def test_pending_publication_requires_the_exact_active_safe_window(
    completed_delivery_ids: tuple[str, ...],
    window: dict[str, object],
) -> None:
    assert project_pending_verified_publication_result(
        accepted=_accepted(),
        delivery_id="delivery-1",
        related_insession_task_ids=("task-1",),
        store=_WindowStore(window=window),
        verified_delivery_store=_VerifiedDeliveryStore(completed_delivery_ids),
    ) is None


@pytest.mark.parametrize("fail_on", ("deliveries", "window"))
def test_pending_publication_fails_closed_when_a_read_is_lost(
    fail_on: str,
) -> None:
    assert project_pending_verified_publication_result(
        accepted=_accepted(),
        delivery_id="delivery-1",
        related_insession_task_ids=(),
        store=_WindowStore(fail=fail_on == "window"),
        verified_delivery_store=_VerifiedDeliveryStore(
            fail=fail_on == "deliveries"
        ),
    ) is None


def test_pending_publication_projects_only_the_exact_durable_fact() -> None:
    store = _WindowStore(window=_active_window(state_version=15))
    verified_delivery_store = _VerifiedDeliveryStore()

    result = project_pending_verified_publication_result(
        accepted=_accepted(),
        delivery_id="delivery-1",
        related_insession_task_ids=("task-1",),
        store=store,
        verified_delivery_store=verified_delivery_store,
    )

    assert result is not None
    assert result.status == "running"
    assert result.processing_level == "L2"
    assert result.reply is None
    assert result.work_run_ids == ()
    assert result.window_state == "active"
    assert result.window_revision == 15
    assert verified_delivery_store.calls == ["deliveries"]
    assert store.calls == ["window"]


def test_pending_publication_defaults_to_the_l2_work_run_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.session.l2_store import work_run

    calls: list[dict[str, object]] = []

    def list_deliveries(**kwargs: object) -> tuple[str, ...]:
        calls.append(kwargs)
        return ("delivery-1",)

    monkeypatch.setattr(
        work_run,
        "list_turn_completed_verified_delivery_ids",
        list_deliveries,
    )

    result = project_pending_verified_publication_result(
        accepted=_accepted(),
        delivery_id="delivery-1",
        related_insession_task_ids=(),
        store=_WindowStore(),
    )

    assert result is not None
    assert calls == [{"session_id": "session-1", "turn_id": "turn-1"}]


def test_settlement_has_one_l2_owner_and_no_entry_reverse_dependency() -> None:
    source = Path(settlement_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    window_protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "L2PendingPublicationWindowStorePort"
    )
    delivery_protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "L2CompletedVerifiedDeliveryStorePort"
    )

    assert "personagraph.runtime.entry" not in source
    assert {
        node.name
        for node in window_protocol.body
        if isinstance(node, ast.FunctionDef)
    } == {"inspect_turn_execution"}
    assert {
        node.name
        for node in delivery_protocol.body
        if isinstance(node, ast.FunctionDef)
    } == {"list_turn_completed_verified_delivery_ids"}
    assert not {
        "advance_turn_execution_window",
        "append_runtime_turn_event",
        "finalize_turn_execution",
        "finalize_verified_turn_execution",
        "mark_turn_execution_interrupted",
    } & called_attributes
    assert not hasattr(
        entry_recovery_module,
        "project_pending_verified_publication_result",
    )
    assert not hasattr(
        entry_recovery_module,
        "EntryCompletedVerifiedDeliveryStorePort",
    )
