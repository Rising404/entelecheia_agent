"""入口重放对账的单元与依赖边界覆盖。"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from personagraph.runtime.entry.lifecycle import replay as replay_module
from personagraph.runtime.entry.lifecycle.replay import (
    reconcile_replayed_entry_turn,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
    EntryTurnResult,
    TurnRoutingPolicySnapshot,
    TurnRoutingPolicy,
)
from personagraph.runtime.turn_events import RuntimeStage


def _accepted(
    *,
    turn_status: str = "running",
    processing_level: str | None = None,
    end_reason: str | None = None,
    error_code: str | None = None,
    l1_enabled: bool = False,
    window_state: str = "active",
) -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="resume the durable turn",
        attachment_ids=(),
        window_revision=7,
        replayed=True,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
        turn_status=turn_status,  # type: ignore[arg-type]
        processing_level=processing_level,  # type: ignore[arg-type]
        end_reason=end_reason,
        error_code=error_code,
        window_state=window_state,  # type: ignore[arg-type]
        routing_policy=(
            TurnRoutingPolicySnapshot(
                source="request_override",
                policy=TurnRoutingPolicy(
                    l1_enabled=True,
                    l2_enabled=False,
                ),
                allowed_processing_levels=("L0", "L1"),
            )
            if l1_enabled
            else TurnRoutingPolicySnapshot(
                source="request_override",
                policy=TurnRoutingPolicy(
                    l1_enabled=False,
                    l2_enabled=True,
                ),
                allowed_processing_levels=("L0", "L2"),
            )
        ),
    )


def _result(*, status: str = "completed") -> EntryTurnResult:
    if status == "completed":
        return EntryTurnResult(
            session_id="session-1",
            turn_id="turn-1",
            status="completed",
            processing_level="L2",
            reply="settled reply",
            window_state="post_commit_pending",
            window_revision=12,
        )
    return EntryTurnResult(
        session_id="session-1",
        turn_id="turn-1",
        status="running",
        processing_level="L2",
        window_state="active",
        window_revision=12,
    )


class _ReplayStore:
    def __init__(
        self,
        *,
        committed: dict[str, Any] | None = None,
        window: object | None = None,
        task_ids: tuple[str, ...] = ("task-1",),
        work_run_ids: tuple[str, ...] = ("work-run-1",),
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self.committed = committed
        self.window = (
            {
                "turn_id": "turn-1",
                "window_state": "active",
                "stage": RuntimeStage.L2_PLAN.value,
                "state_version": 7,
                "current_work_run_id": None,
                "current_attempt_id": None,
            }
            if window is None
            else window
        )
        self.task_ids = task_ids
        self.work_run_ids = work_run_ids
        self.fail_on = fail_on
        self.calls: list[str] = []

    def _read(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
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

    def list_turn_insession_task_ids(
        self,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]:
        self._read("task_ids")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.task_ids

    def list_turn_linked_work_run_ids(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> tuple[str, ...]:
        self._read("work_run_ids")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.work_run_ids


class _ReplayLaneStore:
    def __init__(
        self,
        *,
        has_receipt: bool = True,
        manifest: object = SimpleNamespace(lanes=()),
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self.has_receipt = has_receipt
        self.manifest = manifest
        self.fail_on = fail_on
        self.calls: list[str] = []

    def _read(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise OSError(f"lost {name} read")

    def has_turn_task_execution_lane_receipt(
        self,
        session_id: str,
        turn_id: str,
    ) -> bool:
        self._read("receipt")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.has_receipt

    def get_insession_task_execution_lane_manifest(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> object:
        self._read("manifest")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.manifest


def _unexpected_settlement(**_kwargs: object) -> EntryTurnResult | None:
    raise AssertionError("authoritative settlement must not run")


def test_completed_replay_projects_the_exact_committed_turn_without_callbacks() -> None:
    store = _ReplayStore(
        committed={"assistant_content": "durable formal reply"},
    )

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(turn_status="completed", processing_level="L0"),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
    )

    assert result.status == "completed"
    assert result.processing_level == "L0"
    assert result.reply == "durable formal reply"
    assert result.related_insession_task_ids == ("task-1",)
    assert result.work_run_ids == ("work-run-1",)
    assert store.calls == ["committed", "task_ids", "work_run_ids"]


def test_completed_replay_requires_its_existing_formal_delivery() -> None:
    with pytest.raises(RuntimeError, match="missing its formal delivery"):
        reconcile_replayed_entry_turn(
            accepted=_accepted(turn_status="completed"),
            store=_ReplayStore(),
            reconcile_authoritative_settlement=_unexpected_settlement,
        )


@pytest.mark.parametrize(
    ("turn_status", "processing_level", "l1_enabled"),
    (
        ("completed", "L1", True),
        ("completed", "L1", False),
        ("running", "L1", True),
        ("running", "L1", False),
        ("incomplete", "L1", True),
        ("incomplete", "L1", False),
        ("running", None, True),
        ("incomplete", None, True),
    ),
)
def test_l1_replay_never_reads_tasks_or_enters_the_l2_adapter(
    turn_status: str,
    processing_level: str | None,
    l1_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_l2_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("personagraph.l2") or name == "l2.entry_adapter.replay":
            raise AssertionError("L1 replay imported the L2 adapter")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_l2_import)
    reply = "旧回复中提到 task-1，正文保持原样。"
    store = _ReplayStore(
        committed={"assistant_content": reply},
        fail_on=frozenset({"task_ids", "work_run_ids"}),
    )
    lane_store = _ReplayLaneStore(fail_on=frozenset({"receipt", "manifest"}))

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(
            turn_status=turn_status,
            processing_level=processing_level,
            l1_enabled=l1_enabled,
        ),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result.status == turn_status
    assert result.processing_level == processing_level
    assert result.related_insession_task_ids == ()
    assert result.work_run_ids == ()
    assert lane_store.calls == []
    assert not {"task_ids", "work_run_ids"}.intersection(store.calls)
    if turn_status == "completed":
        assert result.reply == reply
        assert store.calls == ["committed"]
        assert store.committed == {"assistant_content": reply}


@pytest.mark.parametrize(
    ("processing_level", "l1_enabled"),
    ((None, False), ("L2", True)),
)
def test_manifest_and_window_select_authoritative_settlement_independent_of_features(
    processing_level: str | None,
    l1_enabled: bool,
) -> None:
    store = _ReplayStore(
        window={
            "turn_id": "turn-1",
            "window_state": "active",
            "stage": RuntimeStage.PERSIST.value,
            "state_version": 19,
            "current_work_run_id": None,
            "current_attempt_id": None,
        },
    )
    lane_store = _ReplayLaneStore(
        manifest=SimpleNamespace(lanes=(object(),)),
    )
    captured: dict[str, object] = {}

    def reconcile_authoritative_settlement(**kwargs: object) -> EntryTurnResult:
        captured.update(kwargs)
        return _result()

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(processing_level=processing_level, l1_enabled=l1_enabled),
        store=store,
        reconcile_authoritative_settlement=reconcile_authoritative_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result == _result()
    assert captured == {
        "revision": 19,
        "related_insession_task_ids": ("task-1",),
    }
    assert lane_store.calls == ["receipt", "manifest"]
    assert store.calls == ["window", "task_ids"]


@pytest.mark.parametrize(
    ("reason", "stage", "end_reason"),
    (
        ("L1_RUNTIME_NOT_READY", RuntimeStage.L1_BOOTSTRAP.value, "l1_runtime_not_ready"),
        ("TOOL_COMPLETION_UNCONFIRMED", RuntimeStage.L1_BOOTSTRAP.value, "tool_completion_unconfirmed"),
        ("TOOL_COMPLETION_UNCONFIRMED", RuntimeStage.TOOL.value, "tool_completion_unconfirmed"),
        ("TOOL_COMPLETION_UNCONFIRMED", RuntimeStage.RESPONSE.value, "tool_completion_unconfirmed"),
    ),
)
def test_l1_typed_stop_is_reconstructed_from_the_durable_window(
    reason: str, stage: str, end_reason: str,
) -> None:
    store = _ReplayStore(
        window={
            "turn_id": "turn-1",
            "window_state": "interrupted",
            "stage": stage,
            "state_version": 9,
            "interruption_reason": reason,
            "current_l1_turn_run_id": "l1run-1",
            "current_work_run_id": None,
            "current_attempt_id": None,
        },
    )

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(l1_enabled=True),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L1"
    assert result.end_reason == end_reason
    assert result.error_code == reason
    assert result.window_state == "interrupted"
    assert result.window_revision == 9
    assert store.calls == ["window"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"turn_id": "other-turn"},
        {"window_state": "active"},
        {"current_l1_turn_run_id": None},
        {"interruption_reason": "OTHER_INTERRUPTION"},
        {"interruption_reason": "L1_RUNTIME_NOT_READY", "stage": RuntimeStage.RESPONSE.value},
    ),
)
def test_unconfirmed_l1_replay_requires_the_exact_current_interrupted_window(
    overrides: dict[str, object],
) -> None:
    store = _ReplayStore(window={
        "turn_id": "turn-1", "window_state": "interrupted",
        "stage": RuntimeStage.RESPONSE.value, "state_version": 9,
        "interruption_reason": "TOOL_COMPLETION_UNCONFIRMED",
        "current_l1_turn_run_id": "l1run-1", **overrides,
    })
    result = reconcile_replayed_entry_turn(
        accepted=_accepted(l1_enabled=True),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
    )
    assert result.status == "running"
    assert result.processing_level is None
    assert result.error_code is None
    assert result.end_reason is None
    assert store.calls == ["window"]


def test_context_budget_stop_is_reconstructed_before_any_work_replay() -> None:
    store = _ReplayStore(
        window={
            "turn_id": "turn-1",
            "window_state": "interrupted",
            "stage": RuntimeStage.RESPONSE.value,
            "state_version": 11,
            "interruption_reason": "CONTEXT_BUDGET_EXCEEDED",
            "current_work_run_id": None,
            "current_attempt_id": None,
        },
    )

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(
            processing_level="L2",
            window_state="interrupted",
        ),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L2"
    assert result.end_reason == "context_budget_exceeded"
    assert result.error_code == "CONTEXT_BUDGET_EXCEEDED"
    assert result.window_state == "interrupted"
    assert result.window_revision == 11
    assert store.calls == ["window"]


@pytest.mark.parametrize(
    "window",
    (
        {
            "turn_id": "other-turn",
            "window_state": "active",
            "stage": RuntimeStage.PERSIST.value,
            "state_version": 19,
            "current_work_run_id": None,
            "current_attempt_id": None,
        },
        {
            "turn_id": "turn-1",
            "window_state": "active",
            "stage": RuntimeStage.PERSIST.value,
            "state_version": 19,
            "current_work_run_id": "work-run-live",
            "current_attempt_id": None,
        },
        {
            "turn_id": "turn-1",
            "window_state": "active",
            "stage": RuntimeStage.PERSIST.value,
            "state_version": 0,
            "current_work_run_id": None,
            "current_attempt_id": None,
        },
    ),
)
def test_ineligible_authoritative_replay_never_calls_the_settlement_callback(
    window: dict[str, object],
) -> None:
    lane_store = _ReplayLaneStore(
        manifest=SimpleNamespace(lanes=(object(),)),
    )
    result = reconcile_replayed_entry_turn(
        accepted=_accepted(),
        store=_ReplayStore(window=window),
        reconcile_authoritative_settlement=_unexpected_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result.status == "running"
    assert result.processing_level is None


@pytest.mark.parametrize(
    ("turn_status", "processing_level", "l1_enabled", "store_calls", "lane_calls"),
    (
        ("running", "L1", True, ["window"], []),
        ("incomplete", "L0", False, [], ["receipt"]),
    ),
)
def test_replay_without_a_lane_receipt_never_requests_the_l2_manifest(
    turn_status: str,
    processing_level: str,
    l1_enabled: bool,
    store_calls: list[str],
    lane_calls: list[str],
) -> None:
    store = _ReplayStore()
    lane_store = _ReplayLaneStore(has_receipt=False)

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(
            turn_status=turn_status,
            processing_level=processing_level,
            l1_enabled=l1_enabled,
        ),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result.status == turn_status
    assert result.processing_level == processing_level
    assert lane_store.calls == lane_calls
    assert store.calls == store_calls


@pytest.mark.parametrize(
    ("fail_on", "expected_calls"),
    (
        (frozenset({"receipt"}), ["receipt"]),
        (frozenset({"manifest"}), ["receipt", "manifest"]),
    ),
)
def test_unreadable_lane_authority_fails_closed(
    fail_on: frozenset[str],
    expected_calls: list[str],
) -> None:
    lane_store = _ReplayLaneStore(
        manifest=SimpleNamespace(lanes=(object(),)),
        fail_on=fail_on,
    )

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(processing_level="L2"),
        store=_ReplayStore(),
        reconcile_authoritative_settlement=_unexpected_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result.status == "running"
    assert result.processing_level == "L2"
    assert lane_store.calls == expected_calls


@pytest.mark.parametrize(
    ("processing_level", "l1_enabled"),
    ((None, False), ("L2", True)),
)
def test_incomplete_manifest_replay_derives_l2_and_durable_references(
    processing_level: str | None,
    l1_enabled: bool,
) -> None:
    store = _ReplayStore()
    lane_store = _ReplayLaneStore(manifest=object())

    result = reconcile_replayed_entry_turn(
        accepted=_accepted(
            turn_status="incomplete",
            processing_level=processing_level,
            l1_enabled=l1_enabled,
            end_reason="host_stopped",
            error_code="TRANSITION_DENIED",
        ),
        store=store,
        reconcile_authoritative_settlement=_unexpected_settlement,
        task_execution_lane_store=lane_store,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L2"
    assert result.related_insession_task_ids == ("task-1",)
    assert result.work_run_ids == ("work-run-1",)
    assert lane_store.calls == ["receipt", "manifest"]
    assert store.calls == ["task_ids", "work_run_ids"]


def test_replay_controller_has_only_narrow_reads_and_entry_retains_write_callbacks() -> (
    None
):
    source = Path(replay_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_relative_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    store_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    }
    read_port = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EntryReplayReadPort"
    )
    read_port_methods = {
        node.name for node in read_port.body if isinstance(node, ast.FunctionDef)
    }
    assert top_level_relative_imports == {"turn.contracts", "turn_events"}
    assert store_calls == {
        "get_committed_turn_pair",
        "inspect_turn_execution",
        "list_turn_insession_task_ids",
        "list_turn_linked_work_run_ids",
    }
    assert read_port_methods == store_calls
    assert not {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and ("session.persistence" in node.module or "session.l2_store" in node.module)
    }
    assert "EntryReplayTaskExecutionLaneStorePort" not in source
    assert "_SessionTaskExecutionLaneStore" not in source
    assert "def _require_task_execution_lane_manifest(" not in source
    l2_replay_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 4
        and node.module == "l2.entry_adapter.replay"
    ]
    assert len(l2_replay_imports) == 2
    assert not any(node in tree.body for node in l2_replay_imports)
    assert (
        not {
            "accept_turn_execution",
            "advance_turn_execution_window",
            "append_runtime_turn_event",
            "finalize_turn_execution",
            "finalize_verified_turn_execution",
            "mark_turn_execution_interrupted",
        }
        & store_calls
    )

    entry_source = (
        Path(replay_module.__file__).resolve().parents[1] / "application.py"
    ).read_text(encoding="utf-8")
    entry_tree = ast.parse(entry_source)
    wrapper = next(
        node
        for node in entry_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_replayed_turn_result"
    )
    wrapper_source = ast.get_source_segment(entry_source, wrapper)
    assert wrapper_source is not None
    assert "reconcile_replayed_entry_turn(" in wrapper_source
    assert "_try_replay_authoritative_turn_settlement(" in wrapper_source
