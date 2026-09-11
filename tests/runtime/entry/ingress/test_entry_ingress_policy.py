"""入口纯摄入短路策略的单元与边界检查。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.runtime.entry import application as entry_application
import personagraph.runtime.entry.ingress.policy as entry_ingress
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.entry.ingress.policy import (
    select_entry_ingress_short_circuit_outcome,
)
from personagraph.runtime.entry.ingress.contracts import IngressDisposition
from personagraph.runtime.turn_events import (
    RuntimeErrorCode,
    RuntimeStage,
    TurnEventStatus,
)


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="继续任务",
        attachment_ids=(),
        window_revision=17,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


@pytest.mark.parametrize(
    ("disposition", "error_code", "reply"),
    (
        (
            IngressDisposition.OVERFLOW,
            RuntimeErrorCode.INPUT_INVALID,
            "输入超过当前安全上限，请缩短后重试。",
        ),
        (
            IngressDisposition.REJECT,
            RuntimeErrorCode.INGRESS_REJECTED,
            "输入为空或格式无效，请补充后重试。",
        ),
        (
            IngressDisposition.HOLD_APPROVAL,
            RuntimeErrorCode.INGRESS_REJECTED,
            "当前版本还不支持该输入类型。",
        ),
        (
            "unknown_disposition",
            RuntimeErrorCode.INGRESS_REJECTED,
            "当前版本还不支持该输入类型。",
        ),
    ),
)
def test_short_circuit_policy_preserves_every_existing_disposition_mapping(
    disposition: object,
    error_code: RuntimeErrorCode,
    reply: str,
) -> None:
    outcome = select_entry_ingress_short_circuit_outcome(disposition)

    assert outcome.error_code is error_code
    assert outcome.reply == reply


def test_short_circuit_outcome_is_immutable() -> None:
    outcome = select_entry_ingress_short_circuit_outcome(IngressDisposition.REJECT)

    with pytest.raises(FrozenInstanceError):
        outcome.reply = "不同的消息"  # type: ignore[misc]


def test_entry_retains_ingress_event_and_formal_finalization_authority(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    events = []
    store = object()

    def finalize(**kwargs: object) -> str:
        calls.append(kwargs)
        return "formal"

    def emit(event):
        events.append(event)
        return 1

    monkeypatch.setattr(entry_application, "_finalize_formal_reply", finalize)

    result = entry_application._complete_nonroutable_turn(
        accepted=_accepted(),
        ingress=SimpleNamespace(disposition=IngressDisposition.OVERFLOW),
        revision=17,
        emit=emit,
        store=store,
    )

    assert result == "formal"
    assert calls == [
        {
            "accepted": _accepted(),
            "revision": 17,
            "processing_level": "L0",
            "reply": "输入超过当前安全上限，请缩短后重试。",
            "related_insession_task_ids": (),
            "error_code": RuntimeErrorCode.INPUT_INVALID.value,
            "emit": emit,
            "store": store,
        }
    ]
    assert len(events) == 1
    assert events[0].stage is RuntimeStage.INGRESS
    assert events[0].status is TurnEventStatus.FAILED
    assert events[0].error_code is RuntimeErrorCode.INPUT_INVALID


def test_policy_has_no_entry_store_controller_or_lifecycle_effects() -> None:
    source = Path(entry_ingress.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level >= 1
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert relative_imports == {
        "context_budget.token_counter",
        "contracts",
        "turn_events",
    }
    assert "EntryStorePort" not in source
    assert not {
        "_finalize_formal_reply",
        "_incomplete_turn",
        "new_turn_event",
    } & called_names
    assert not {
        "accept_turn_execution",
        "advance_turn_execution_window",
        "mark_turn_execution_interrupted",
        "append_runtime_turn_event",
        "finalize_turn_execution",
    } & called_attributes

    entry_module = ast.parse(
        Path(entry_application.__file__).read_text(encoding="utf-8")
    )
    completer = next(
        node
        for node in entry_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_complete_nonroutable_turn"
    )
    completer_calls = {
        node.func.id
        for node in ast.walk(completer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "select_entry_ingress_short_circuit_outcome" in completer_calls
    assert {"_finalize_formal_reply", "new_turn_event"} <= completer_calls
