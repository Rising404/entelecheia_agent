"""入口监督器前分类阶段的单元与边界检查。"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

from personagraph.model_io.gateway import ModelGatewayError
import personagraph.runtime.entry.ingress.classification as stage_module
from personagraph.runtime.entry.ingress.classification import (
    run_entry_classification_stage,
)
from personagraph.runtime.entry.context.contracts import EntryContext
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.entry.ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    IngressDecision,
    IngressDisposition,
    IngressHandler,
    TrustedTurnEnvelope,
    evaluate_ingress,
)
from personagraph.runtime.turn_deadline import TurnDeadline


def _context() -> EntryContext:
    return EntryContext(
        envelope=TrustedTurnEnvelope(
            turn_id="turn-1",
            session_id="session-1",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="classify this turn",
        ),
        snapshot=AuthoritativeRuntimeSnapshot(),
        ceiling=CapabilityCeiling(),
        estimated_input_tokens=3,
        history_pairs=(),
        session_summary=None,
    )


def _accepted_ingress(context: EntryContext, features: dict[str, object]) -> IngressDecision:
    return evaluate_ingress(
        context.envelope,
        context.snapshot,
        context.ceiling,
        estimated_input_tokens=context.estimated_input_tokens,
        context_hard_limit=int(features["context_guard_limit"]),
    )


def _rejected_ingress(context: EntryContext, features: dict[str, object]) -> IngressDecision:
    return _accepted_ingress(context, features).model_copy(update={
        "disposition": IngressDisposition.REJECT,
        "handler": IngressHandler.NONE,
    })


def test_stage_submits_ingress_and_classifier_concurrently_with_the_exact_inputs() -> None:
    context = _context()
    features: dict[str, object] = {"context_guard_limit": 24000}
    deadline = TurnDeadline.starting_now(60.0)
    classification = EntryClassification(processing_level="L0")
    classifier_started = Event()
    emitted: list[object] = []
    received: dict[str, object] = {}

    def emit(event: object) -> int:
        emitted.append(event)
        return 1

    def ingress_evaluator(
        supplied_context: EntryContext,
        supplied_features: dict[str, object],
    ) -> IngressDecision:
        assert classifier_started.wait(timeout=1.0)
        assert supplied_context is context
        assert supplied_features is features
        return _accepted_ingress(supplied_context, supplied_features)

    def classifier(
        supplied_context: EntryContext,
        supplied_emit,
        supplied_deadline: TurnDeadline,
    ) -> EntryClassification:
        received.update({
            "context": supplied_context,
            "emit": supplied_emit,
            "deadline": supplied_deadline,
        })
        classifier_started.set()
        return classification

    def unexpected_terminal(*_args: object) -> object:
        raise AssertionError("model ingress with a classification must not finalize here")

    result = run_entry_classification_stage(
        context=context,
        features=features,
        emit=emit,
        deadline=deadline,
        ingress_evaluator=ingress_evaluator,
        classifier=classifier,
        on_nonmodel_ingress=unexpected_terminal,
        on_classifier_error=unexpected_terminal,
    )

    assert result.ingress.disposition is IngressDisposition.ACCEPT
    assert result.classification is classification
    assert result.terminal_result is None
    assert received == {"context": context, "emit": emit, "deadline": deadline}
    assert emitted == []


def test_non_model_ingress_ignores_an_irrelevant_classifier_error() -> None:
    context = _context()
    classifier_started = Event()
    classifier_finished = Event()
    release_classifier = Event()
    terminal_result = object()

    def ingress_evaluator(
        supplied_context: EntryContext,
        supplied_features: dict[str, object],
    ) -> IngressDecision:
        assert classifier_started.wait(timeout=1.0)
        return _rejected_ingress(supplied_context, supplied_features)

    def classifier(*_args: object) -> EntryClassification:
        classifier_started.set()
        assert release_classifier.wait(timeout=1.0)
        classifier_finished.set()
        raise ModelGatewayError("MODEL_CALL_TIMEOUT", "irrelevant classifier failure", retryable=True)

    def on_nonmodel_ingress(ingress: IngressDecision) -> object:
        assert ingress.disposition is IngressDisposition.REJECT
    # 持久终态回调必须先运行，随后执行器才等待已经启动且无法取消的分类器。
        assert not classifier_finished.is_set()
        release_classifier.set()
        return terminal_result

    def unexpected_classifier_error(*_args: object) -> object:
        raise AssertionError("rejected ingress must not observe classifier errors")

    result = run_entry_classification_stage(
        context=context,
        features={"context_guard_limit": 24000},
        emit=lambda _event: None,
        deadline=TurnDeadline.starting_now(60.0),
        ingress_evaluator=ingress_evaluator,
        classifier=classifier,
        on_nonmodel_ingress=on_nonmodel_ingress,
        on_classifier_error=unexpected_classifier_error,
    )

    assert result.ingress.disposition is IngressDisposition.REJECT
    assert result.classification is None
    assert result.terminal_result is terminal_result
    assert classifier_finished.is_set()


def test_model_ingress_returns_the_classifier_gateway_error_for_entry_to_project() -> None:
    error = ModelGatewayError(
        "MODEL_BAD_RESPONSE",
        "invalid classifier reply",
        retryable=False,
    )

    def classifier(*_args: object) -> EntryClassification:
        raise error

    terminal_result = object()
    received_errors: list[ModelGatewayError] = []

    def unexpected_nonmodel(*_args: object) -> object:
        raise AssertionError("model ingress must not use the non-model callback")

    def on_classifier_error(actual_error: ModelGatewayError) -> object:
        received_errors.append(actual_error)
        return terminal_result

    result = run_entry_classification_stage(
        context=_context(),
        features={"context_guard_limit": 24000},
        emit=lambda _event: None,
        deadline=TurnDeadline.starting_now(60.0),
        ingress_evaluator=_accepted_ingress,
        classifier=classifier,
        on_nonmodel_ingress=unexpected_nonmodel,
        on_classifier_error=on_classifier_error,
    )

    assert result.ingress.handler is IngressHandler.MODEL
    assert result.classification is None
    assert result.terminal_result is terminal_result
    assert received_errors == [error]


def test_ingress_evaluator_gateway_error_is_not_reclassified_as_a_classifier_failure() -> None:
    error = ModelGatewayError("MODEL_CALL_TIMEOUT", "ingress dependency failed", retryable=True)

    def ingress_evaluator(*_args: object) -> IngressDecision:
        raise error

    with pytest.raises(ModelGatewayError) as raised:
        run_entry_classification_stage(
            context=_context(),
            features={"context_guard_limit": 24000},
            emit=lambda _event: None,
            deadline=TurnDeadline.starting_now(60.0),
            ingress_evaluator=ingress_evaluator,
            classifier=lambda *_args: EntryClassification(processing_level="L0"),
            on_nonmodel_ingress=lambda _ingress: object(),
            on_classifier_error=lambda _error: object(),
        )

    assert raised.value is error


def test_model_ingress_rejects_a_missing_classifier_result() -> None:
    with pytest.raises(ValueError, match="requires exactly one classification result"):
        run_entry_classification_stage(
            context=_context(),
            features={"context_guard_limit": 24000},
            emit=lambda _event: None,
            deadline=TurnDeadline.starting_now(60.0),
            ingress_evaluator=_accepted_ingress,
            classifier=lambda *_args: None,  # type: ignore[arg-type]
            on_nonmodel_ingress=lambda _ingress: object(),
            on_classifier_error=lambda _error: object(),
        )


def test_stage_has_no_entry_store_or_lifecycle_authority() -> None:
    source = Path(stage_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level >= 1
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "entry" not in relative_imports
    assert "EntryStorePort" not in source
    assert "new_turn_event" not in source
    assert not {
        "accept_turn_execution",
        "advance_turn_execution_window",
        "append_runtime_turn_event",
        "finalize_turn_execution",
        "mark_turn_execution_interrupted",
    } & called_attributes
