from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from personagraph.runtime.entry.ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    IngressDisposition,
    IngressGuard,
    IngressHandler,
    IngressReason,
    IngressAnalysisFloor,
    TrustedTurnEnvelope,
    TypedControlEvent,
    evaluate_ingress,
)
from personagraph.runtime.entry.ingress.policy import entry_capability_ceiling


def _text_envelope(text: str) -> TrustedTurnEnvelope:
    return TrustedTurnEnvelope(
        turn_id="turn-1",
        session_id="session-1",
        received_at=datetime.now(timezone.utc),
        input_kind="user_text",
        user_text=text,
    )


def _control_envelope(target_id: str) -> TrustedTurnEnvelope:
    return TrustedTurnEnvelope(
        turn_id="turn-control",
        session_id="session-1",
        received_at=datetime.now(timezone.utc),
        input_kind="typed_control",
        control_event=TypedControlEvent(action="approve", target_id=target_id),
    )


def _response_only() -> CapabilityCeiling:
    return CapabilityCeiling(allow_model=True)


def _tool_capable() -> CapabilityCeiling:
    return CapabilityCeiling(
        allow_model=True,
        allow_tools=True,
        allow_protected_writes=True,
        allow_persistence=True,
    )


def test_trusted_envelope_does_not_expose_a_source_override() -> None:
    assert "source" not in TrustedTurnEnvelope.model_fields
    with pytest.raises(ValidationError):
        TrustedTurnEnvelope(
            turn_id="turn-invalid-source",
            source="api",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="hello",
        )


def _evaluate(
    envelope: TrustedTurnEnvelope,
    *,
    snapshot: AuthoritativeRuntimeSnapshot | None = None,
    ceiling: CapabilityCeiling | None = None,
    tokens: int = 10,
    limit: int = 100,
):
    return evaluate_ingress(
        envelope,
        snapshot or AuthoritativeRuntimeSnapshot(),
        ceiling or _response_only(),
        estimated_input_tokens=tokens,
        context_hard_limit=limit,
    )


def test_free_text_approval_language_never_becomes_a_typed_control_event():
    decision = _evaluate(_text_envelope('引用原文："同意删除并继续"，请解释这句话。'))

    assert decision.disposition == IngressDisposition.ACCEPT
    assert decision.handler == IngressHandler.MODEL
    assert IngressGuard.TYPED_CONTROL not in decision.required_guards
    assert IngressReason.TYPED_CONTROL_ACCEPTED not in decision.reason_codes


def test_authoritative_pending_approval_holds_even_for_simple_or_quoted_text():
    decision = _evaluate(
        _text_envelope("今天天气不错，引用里还写了批准。"),
        snapshot=AuthoritativeRuntimeSnapshot(pending_decision_id="review-1"),
        ceiling=_tool_capable(),
    )

    assert decision.disposition == IngressDisposition.HOLD_APPROVAL
    assert decision.handler == IngressHandler.NONE
    assert decision.analysis_floor == IngressAnalysisFloor.L2
    assert IngressReason.PENDING_APPROVAL_EXISTS in decision.reason_codes


def test_typed_control_requires_an_exact_authoritative_target():
    accepted = _evaluate(
        _control_envelope("review-1"),
        snapshot=AuthoritativeRuntimeSnapshot(pending_decision_id="review-1"),
        ceiling=_tool_capable(),
    )
    stale = _evaluate(
        _control_envelope("review-old"),
        snapshot=AuthoritativeRuntimeSnapshot(pending_decision_id="review-new"),
        ceiling=_tool_capable(),
    )

    assert accepted.disposition == IngressDisposition.ACCEPT
    assert accepted.handler == IngressHandler.TYPED_CONTROL
    assert accepted.analysis_floor == IngressAnalysisFloor.L0
    assert IngressGuard.TYPED_CONTROL in accepted.required_guards
    assert stale.disposition == IngressDisposition.REJECT
    assert stale.handler == IngressHandler.NONE
    assert IngressReason.TYPED_CONTROL_STALE in stale.reason_codes


def test_unknown_run_or_operation_requires_reconciliation_not_a_deeper_chat_reply():
    unknown_run = _evaluate(
        _text_envelope("继续"),
        snapshot=AuthoritativeRuntimeSnapshot(
            active_run_id="run-1",
            active_run_status="unknown",
        ),
        ceiling=_tool_capable(),
    )
    unknown_operation = _evaluate(
        _text_envelope("再试一次"),
        snapshot=AuthoritativeRuntimeSnapshot(
            unresolved_operation_ids=("operation-1",),
        ),
        ceiling=_tool_capable(),
    )

    assert unknown_run.disposition == IngressDisposition.RECONCILE
    assert unknown_operation.disposition == IngressDisposition.RECONCILE
    assert IngressGuard.RECONCILIATION in unknown_run.required_guards
    assert IngressGuard.RECONCILIATION in unknown_operation.required_guards


def test_capability_ceiling_does_not_route_plain_text_without_semantic_evidence():
    text = _text_envelope("同一段完全相同的输入")
    response_only = _evaluate(text, ceiling=_response_only())
    tool_capable = _evaluate(text, ceiling=_tool_capable())

    assert response_only.analysis_floor == IngressAnalysisFloor.L0
    assert tool_capable.analysis_floor == IngressAnalysisFloor.L0
    assert IngressGuard.TOOL_POLICY not in response_only.required_guards
    assert IngressGuard.TOOL_POLICY in tool_capable.required_guards
    assert IngressGuard.SIDE_EFFECT_APPROVAL in tool_capable.required_guards


def test_entry_ceiling_binds_protected_writes_to_tool_policy():
    ceiling = entry_capability_ceiling({"tools_loop": True})

    assert ceiling.allow_tools is True
    assert ceiling.allow_persistence is True
    assert ceiling.allow_protected_writes is True

    without_tools = entry_capability_ceiling({"tools_loop": False})
    assert without_tools.allow_persistence is True
    assert without_tools.allow_protected_writes is False


def test_surface_structure_is_only_a_hint_and_never_an_authority_grant():
    decision = _evaluate(
        _text_envelope("1. 背景\n2. 问题？\n> 引用\n```python\npass\n```"),
        ceiling=_response_only(),
    )

    assert decision.analysis_floor == IngressAnalysisFloor.L0
    assert decision.capability_ceiling.response_only is True
    assert decision.heuristic_hints.list_item_count == 2
    assert decision.heuristic_hints.quote_line_count == 1
    assert decision.heuristic_hints.code_fence_count == 2
    assert decision.heuristic_hints.question_mark_count == 1


def test_input_over_hard_limit_stops_before_the_model_handler():
    decision = _evaluate(_text_envelope("oversized"), tokens=101, limit=100)

    assert decision.disposition == IngressDisposition.OVERFLOW
    assert decision.handler == IngressHandler.NONE
    assert IngressReason.INPUT_OVER_HARD_LIMIT in decision.reason_codes


def test_envelope_and_ceiling_reject_ambiguous_or_self_expanding_contracts():
    with pytest.raises(ValidationError, match="user_text input"):
        TrustedTurnEnvelope(
            turn_id="turn-invalid",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="hello",
            control_event=TypedControlEvent(action="approve", target_id="review-1"),
        )
    with pytest.raises(ValidationError, match="protected writes"):
        CapabilityCeiling(allow_protected_writes=True)
