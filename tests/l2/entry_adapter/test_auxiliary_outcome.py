"""私有 Auxiliary 入口结果解释的单元与边界测试。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.runtime.entry import (
    application as entry_application,
)
from personagraph.l2.entry_adapter import (
    auxiliary_outcome as policy_module,
    application as l2_entry_application,
)
from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainStatus,
)
from personagraph.l2.entry_adapter.auxiliary_outcome import (
    AuxiliaryEntryOutcomeDecisionKind,
    interpret_auxiliary_production_chain_outcome,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.runtime.turn_events import RuntimeErrorCode, RuntimeStage


@dataclass(frozen=True)
class _ProductionResult:
    status: object
    final_delivery_id: str | None = None
    requested_user_questions: tuple[str, ...] = ()
    reason_code: str | None = None


class _DeliveryIdMustNotBeRead:
    status = AuxiliaryProductionChainStatus.DELIVERY_READY

    @property
    def final_delivery_id(self) -> str | None:
        raise AssertionError("missing Window must short-circuit before delivery id")

    @property
    def requested_user_questions(self) -> tuple[str, ...]:
        return ()


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="继续执行任务",
        attachment_ids=(),
        window_revision=7,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _assert_incomplete(
    result: _ProductionResult | _DeliveryIdMustNotBeRead,
    *,
    authoritative_window_available: bool = True,
    end_reason: str,
    error_code: RuntimeErrorCode,
    stage: RuntimeStage,
) -> None:
    decision = interpret_auxiliary_production_chain_outcome(
        result,
        authoritative_window_available=authoritative_window_available,
    )

    assert decision.kind is AuxiliaryEntryOutcomeDecisionKind.INCOMPLETE
    assert decision.delivery_id is None
    assert decision.reply is None
    assert decision.end_reason == end_reason
    assert decision.error_code is error_code
    assert decision.stage is stage


def test_ready_delivery_selects_verified_finalization_without_a_body() -> None:
    decision = interpret_auxiliary_production_chain_outcome(
        _ProductionResult(
            status=AuxiliaryProductionChainStatus.DELIVERY_READY,
            final_delivery_id="delivery-1",
        ),
        authoritative_window_available=True,
    )

    assert (
        decision.kind
        is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_VERIFIED_DELIVERY
    )
    assert decision.delivery_id == "delivery-1"
    assert decision.reply is None
    assert decision.end_reason is None
    assert decision.error_code is None
    assert decision.stage is None


def test_ready_delivery_requires_an_active_window_before_reading_its_id() -> None:
    _assert_incomplete(
        _DeliveryIdMustNotBeRead(),
        authoritative_window_available=False,
        end_reason="persistence_error",
        error_code=RuntimeErrorCode.PERSIST_FAILED,
        stage=RuntimeStage.PERSIST,
    )
    _assert_incomplete(
        _ProductionResult(
            status=AuxiliaryProductionChainStatus.DELIVERY_READY,
        ),
        end_reason="persistence_error",
        error_code=RuntimeErrorCode.PERSIST_FAILED,
        stage=RuntimeStage.PERSIST,
    )


@pytest.mark.parametrize(
    "questions",
    (
        (),
        ("需要补充范围。", "需要补充来源。"),
        ("   ",),
    ),
)
def test_waiting_user_requires_exactly_one_nonblank_question(
    questions: tuple[str, ...],
) -> None:
    _assert_incomplete(
        _ProductionResult(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            requested_user_questions=questions,
        ),
        end_reason="host_stopped",
        error_code=RuntimeErrorCode.TRANSITION_DENIED,
        stage=RuntimeStage.L2_PLAN,
    )


def test_waiting_user_preserves_the_exact_one_valid_question() -> None:
    decision = interpret_auxiliary_production_chain_outcome(
        _ProductionResult(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            requested_user_questions=("请补充需要覆盖的年份。",),
        ),
        authoritative_window_available=True,
    )

    assert (
        decision.kind
        is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_FORMAL_QUESTION
    )
    assert decision.reply == "请补充需要覆盖的年份。"
    assert decision.delivery_id is None


def test_closed_world_waiting_user_never_finalizes_a_formal_question() -> None:
    decision = interpret_auxiliary_production_chain_outcome(
        _ProductionResult(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            requested_user_questions=("请提供表格截图。",),
        ),
        authoritative_window_available=True,
        allow_user_input=False,
    )

    assert decision.kind is AuxiliaryEntryOutcomeDecisionKind.INCOMPLETE
    assert decision.reply is None
    assert decision.end_reason == "host_stopped"
    assert decision.error_code is RuntimeErrorCode.TRANSITION_DENIED
    assert decision.stage is RuntimeStage.L2_PLAN


@pytest.mark.parametrize(
    ("status", "end_reason", "error_code", "stage"),
    (
        (
            AuxiliaryProductionChainStatus.WAITING_EXTERNAL,
            "host_stopped",
            RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
            RuntimeStage.TOOL,
        ),
        (
            AuxiliaryProductionChainStatus.WAITING_AUTHORIZATION,
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.TRANSITION_GUARD,
        ),
        (
            AuxiliaryProductionChainStatus.TURN_LIMIT_REACHED,
            "host_stopped",
            RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            RuntimeStage.RESPONSE,
        ),
        (
            AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED,
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.L2_PLAN,
        ),
        (
            AuxiliaryProductionChainStatus.REVISION_REQUIRED,
            "module_error",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.TRANSITION_GUARD,
        ),
        (
            AuxiliaryProductionChainStatus.BLOCKED,
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.L2_PLAN,
        ),
        (
            AuxiliaryProductionChainStatus.FAILED,
            "module_error",
            RuntimeErrorCode.INTERNAL_FAILURE,
            RuntimeStage.RESPONSE,
        ),
        (
            "unknown-private-status",
            "module_error",
            RuntimeErrorCode.INTERNAL_FAILURE,
            RuntimeStage.RESPONSE,
        ),
    ),
)
def test_non_public_stops_and_unknown_statuses_fail_closed(
    status: object,
    end_reason: str,
    error_code: RuntimeErrorCode,
    stage: RuntimeStage,
) -> None:
    _assert_incomplete(
        _ProductionResult(status=status),
        end_reason=end_reason,
        error_code=error_code,
        stage=stage,
    )


class _AdapterStore:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def get_insession_task_details(
        self,
        _session_id: str,
        _task_id: str,
    ) -> object:
        self.trace.append("task")
        return SimpleNamespace(objective="生成经过验证的交付")


def test_entry_reloads_the_window_before_selecting_its_existing_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    store = _AdapterStore(trace)
    accepted = _accepted()
    def emitted(_event):
        return None
    captured: dict[str, object] = {}

    def run_executor(**kwargs: object) -> _ProductionResult:
        trace.append("executor")
        assert kwargs["session_id"] == accepted.session_id
        assert kwargs["turn_id"] == accepted.turn_id
        assert kwargs["task_id"] == "task-1"
        assert kwargs["emit"] is emitted
        return _ProductionResult(
            status=AuxiliaryProductionChainStatus.DELIVERY_READY,
            final_delivery_id="delivery-1",
        )

    def load_window(**kwargs: object) -> dict[str, object]:
        trace.append("window")
        assert kwargs["accepted"] is accepted
        assert kwargs["store"] is store
        assert kwargs["expected_lease_owner"] == "lease-1"
        return {"state_version": 12}

    def finalize(**kwargs: object) -> str:
        trace.append("finalize")
        captured.update(kwargs)
        return "verified"

    monkeypatch.setattr(
        l2_entry_application,
        "run_l2_task_lane",
        run_executor,
    )
    monkeypatch.setattr(
        entry_application,
        "_authoritative_active_turn_window",
        load_window,
    )
    monkeypatch.setattr(
        entry_application,
        "_finalize_verified_work_run_reply",
        finalize,
    )

    result = entry_application._execute_auxiliary_production_chain(
        accepted=accepted,
        task_id="task-1",
        revision=7,
        related_insession_task_ids=("task-1",),
        deadline=TurnDeadline.starting_now(60.0),
        emit=emitted,
        store=store,
        expected_lease_owner="lease-1",
    )

    assert result == "verified"
    assert trace == ["executor", "window", "finalize"]
    assert captured == {
        "accepted": accepted,
        "delivery_id": "delivery-1",
        "revision": 12,
        "related_insession_task_ids": ("task-1",),
        "emit": emitted,
        "store": store,
        "expected_lease_owner": "lease-1",
    }


def test_entry_closed_world_waiting_user_uses_incomplete_not_formal_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    store = _AdapterStore(trace)
    accepted = _accepted()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        l2_entry_application,
        "run_l2_task_lane",
        lambda **_kwargs: _ProductionResult(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            requested_user_questions=("请提供表格截图。",),
        ),
    )
    monkeypatch.setattr(
        entry_application,
        "_authoritative_active_turn_window",
        lambda **_kwargs: {"state_version": 12},
    )
    monkeypatch.setattr(
        entry_application,
        "_finalize_formal_reply",
        lambda **_kwargs: pytest.fail(
            "closed-world WAITING_USER must not become a completed formal reply"
        ),
    )

    def incomplete(**kwargs: object) -> str:
        captured.update(kwargs)
        return "incomplete"

    monkeypatch.setattr(entry_application, "_incomplete_turn", incomplete)

    result = entry_application._execute_auxiliary_production_chain(
        accepted=accepted,
        task_id="task-1",
        revision=7,
        related_insession_task_ids=("task-1",),
        deadline=TurnDeadline.starting_now(60.0),
        emit=lambda _event: None,
        store=store,
        features={"user_interaction_mode": "closed_world"},
    )

    assert result == "incomplete"
    assert captured["end_reason"] == "host_stopped"
    assert captured["error_code"] is RuntimeErrorCode.TRANSITION_DENIED
    assert captured["stage"] is RuntimeStage.L2_PLAN
    assert captured["processing_level"] == "L2"


def test_entry_raises_resume_lease_loss_before_interpreting_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    store = _AdapterStore(trace)
    accepted = _accepted()

    monkeypatch.setattr(
        l2_entry_application,
        "run_l2_task_lane",
        lambda **_kwargs: _ProductionResult(
            status=AuxiliaryProductionChainStatus.DELIVERY_READY,
            final_delivery_id="delivery-1",
        ),
    )
    monkeypatch.setattr(
        entry_application,
        "_authoritative_active_turn_window",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "interpret_auxiliary_production_chain_outcome",
        lambda *_args, **_kwargs: pytest.fail("lease loss must precede policy"),
    )

    with pytest.raises(entry_application._AuxiliaryTurnLeaseLost):
        entry_application._execute_auxiliary_production_chain(
            accepted=accepted,
            task_id="task-1",
            revision=7,
            related_insession_task_ids=("task-1",),
            deadline=TurnDeadline.starting_now(60.0),
            emit=lambda _event: None,
            store=store,
            expected_lease_owner="lease-1",
        )

    assert trace == []


def test_entry_maps_a_missing_l2_task_target_to_transition_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = _accepted()
    captured: dict[str, object] = {}
    marker = object()

    def missing_target(**_kwargs: object) -> object:
        raise l2_entry_application.L2TaskTargetUnavailableError("missing task")

    def incomplete(**kwargs: object) -> object:
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(l2_entry_application, "run_l2_task_lane", missing_target)
    monkeypatch.setattr(entry_application, "_incomplete_turn", incomplete)

    result = entry_application._execute_auxiliary_production_chain(
        accepted=accepted,
        task_id="task-1",
        revision=7,
        related_insession_task_ids=("task-1",),
        deadline=TurnDeadline.starting_now(60.0),
        emit=lambda _event: None,
        store=object(),  # type: ignore[arg-type]
    )

    assert result is marker
    assert captured["end_reason"] == "module_error"
    assert captured["error_code"] is RuntimeErrorCode.TRANSITION_DENIED
    assert captured["stage"] is RuntimeStage.TRANSITION_GUARD
    assert captured["processing_level"] == "L2"


def test_policy_has_no_entry_store_executor_or_lifecycle_authority() -> None:
    source = Path(policy_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level == 1
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

    assert not {
        "entry",
        "session",
        "session.store",
        "l2_entry_executor",
        "work_run_turn_controller",
    } & relative_imports
    assert "EntryStorePort" not in source
    assert "publication_body" not in source
    assert not any(isinstance(node, ast.Try) for node in ast.walk(module))
    assert not {
        "_finalize_formal_reply",
        "_finalize_verified_work_run_reply",
        "_incomplete_turn",
        "new_turn_event",
        "project_turn_event",
    } & called_names
    assert not {
        "accept_turn_execution",
        "advance_turn_execution_window",
        "append_runtime_turn_event",
        "finalize_turn_execution",
        "finalize_verified_turn_execution",
        "mark_turn_execution_interrupted",
    } & called_attributes

    entry_source = Path(entry_application.__file__).read_text(encoding="utf-8")
    entry_module = ast.parse(entry_source)
    executor = next(
        node
        for node in entry_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_auxiliary_production_chain_scoped"
    )
    executor_source = ast.get_source_segment(entry_source, executor)
    assert executor_source is not None
    window_index = executor_source.index("authoritative_window = _authoritative_active_turn_window(")
    lease_index = executor_source.index(
        "if expected_lease_owner is not None and authoritative_window is None:",
        window_index,
    )
    outcome_index = executor_source.index(
        "outcome = interpret_auxiliary_production_chain_outcome(",
        lease_index,
    )
    assert window_index < lease_index < outcome_index
    assert "_finalize_verified_work_run_reply(" in executor_source[outcome_index:]
    assert "_finalize_formal_reply(" in executor_source[outcome_index:]
    assert "_incomplete_turn(" in executor_source[outcome_index:]
