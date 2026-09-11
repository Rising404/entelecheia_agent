from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
    runtime_model_dispatch_request_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _logical_request(**overrides: object) -> RuntimeModelLogicalRequest:
    values: dict[str, object] = {
        "logical_call_id": "logical_prompt_01",
        "session_id": "session_01",
        "task_id": None,
        "auxiliary_graph_id": None,
        "goal_id": None,
        "execution_subject_id": None,
        "invocation_turn_id": "turn_01",
        "call_kind": "structured_test",
        "purpose": "structured_test",
        "provider": "mock",
        "model": "mock-model",
        "endpoint_fingerprint": SHA_A,
        "request_contract": "structured-test-request-v1",
        "request_payload": {"frozen": True},
        "typed_result_contract": "structured-test-result-v1",
        "max_physical_attempts": 6,
        "state_guard_sha256": SHA_B,
    }
    values.update(overrides)
    return RuntimeModelLogicalRequest.create(**values)


def test_logical_request_freezes_protocol_and_exact_first_two_messages() -> None:
    prompt = RuntimeModelStructuredPrompt.create(
        system_prompt="原始 system",
        user_content='{"原始":"user"}',
    )
    logical = _logical_request(
        output_repair_protocol=(
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        ),
        structured_prompt=prompt,
    )

    assert logical.structured_prompt == prompt
    assert logical.output_repair_protocol.value == (
        "four-message-whole-response-regeneration-v1"
    )
    with pytest.raises(ValidationError, match="requires an exact structured prompt"):
        _logical_request(
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            )
        )

    tampered = logical.model_dump(mode="json")
    tampered["structured_prompt"]["user_content"] = '{"changed":true}'
    with pytest.raises(ValidationError, match="hash does not match"):
        RuntimeModelLogicalRequest.model_validate(tampered)


def _issue(
    *,
    code: str = "schema.missing",
    path: str = "/action/completion_report/obligation_reports",
) -> RuntimeModelOutputRepairIssue:
    return RuntimeModelOutputRepairIssue(
        category="schema",
        code=code,
        paths=(path,),
        safe_explanation="目标合同要求此位置必须存在。",
    )


def _feedback(
    *,
    current_issues: tuple[RuntimeModelOutputRepairIssue, ...] | None = None,
    issue_coverage: str = "complete",
    omitted_issue_count: int = 0,
) -> RuntimeModelOutputRepairFeedback:
    return RuntimeModelOutputRepairFeedback(
        message_contract="four-message-whole-response-regeneration-v1",
        target_contract="l1-decision-proposal-v1",
        rejected_physical_ordinal=1,
        rejected_response_sha256=SHA_B,
        repair_mode="regenerate_complete_response",
        issue_coverage=issue_coverage,
        omitted_issue_count=omitted_issue_count,
        current_issues=current_issues or (_issue(),),
    )


def test_feedback_keeps_current_wire_contract_without_embedding_body() -> None:
    feedback = _feedback()

    loaded = RuntimeModelOutputRepairFeedback.model_validate(
        feedback.model_dump(mode="json")
    )
    assert loaded == feedback
    assert loaded.schema_version == "runtime-model-output-repair-feedback-v2"
    assert "rejected_response" not in feedback.model_dump(mode="json")
    assert runtime_model_dispatch_request_sha256(
        logical_request_sha256=SHA_A,
        output_repair_feedback=feedback,
    ) != SHA_A

    with pytest.raises(ValidationError, match="rejected_response_text"):
        RuntimeModelOutputRepairFeedback.model_validate(
            feedback.model_dump(mode="json") | {"rejected_response_text": "private"}
        )


def test_legacy_repair_records_fail_closed() -> None:
    with pytest.raises(ValidationError):
        RuntimeModelOutputRepairFeedback.model_validate(
            {
                "schema_version": "runtime-model-output-repair-feedback-v1",
                "rejected_physical_ordinal": 1,
                "reason_code": "contract_invalid",
                "safe_reason": "The response violates the required contract.",
                "rejected_response_sha256": SHA_A,
            }
        )

    with pytest.raises(ValueError, match="legacy-feedback-only-v1"):
        _logical_request(
            output_repair_protocol="legacy-feedback-only-v1",
        )


def test_feedback_enforces_coverage_issue_order_and_json_pointer_canonicality() -> None:
    assert _feedback(issue_coverage="partial").issue_coverage == "partial"
    with pytest.raises(ValidationError, match="omitted_issue_count"):
        _feedback(issue_coverage="complete", omitted_issue_count=1)
    with pytest.raises(ValidationError, match="omitted_issue_count"):
        _feedback(issue_coverage="partial", omitted_issue_count=1)
    with pytest.raises(ValidationError, match="omitted_issue_count"):
        _feedback(issue_coverage="truncated", omitted_issue_count=0)

    later = _issue(code="schema.missing", path="/z")
    earlier = _issue(code="schema.missing", path="/a")
    with pytest.raises(ValidationError, match="stable order"):
        _feedback(current_issues=(later, earlier))

    with pytest.raises(ValidationError, match="JSON Pointer"):
        _issue(path="/unsafe~2escape")
    with pytest.raises(ValidationError, match="stable order"):
        RuntimeModelOutputRepairIssue(
            category="host_guard",
            code="host_guard.cross_field",
            paths=("/z", "/a"),
            safe_explanation="多个字段之间的关系不符合合同。",
        )

    syntax_issue = RuntimeModelOutputRepairIssue(
        category="json_syntax",
        code="json_syntax.invalid_json",
        paths=("",),
        json_line=2,
        json_column=9,
        safe_explanation="JSON 在该行列附近无法解析。",
    )
    assert syntax_issue.json_line == 2
    with pytest.raises(ValidationError, match="supplied together"):
        RuntimeModelOutputRepairIssue(
            category="json_syntax",
            code="json_syntax.invalid_json",
            paths=("",),
            json_line=2,
            safe_explanation="JSON 无法解析。",
        )


def test_feedback_is_bound_into_existing_physical_request_hash() -> None:
    feedback = _feedback()
    attempt = RuntimeModelPhysicalAttemptRequest.create(
        physical_attempt_id="physical_repair_02",
        physical_attempt_key="physical_repair_key_02",
        logical_call_id="logical_01",
        logical_request_binding_sha256=SHA_A,
        physical_ordinal=2,
        started_turn_id="turn_02",
        provider="mock",
        model="mock-model",
        endpoint_fingerprint=SHA_A,
        request_sha256=SHA_A,
        output_repair_enabled=True,
        output_repair_feedback=feedback,
        provider_idempotency_key=None,
        dispatch_authority_sha256=SHA_B,
    )

    assert isinstance(attempt.output_repair_feedback, RuntimeModelOutputRepairFeedback)
    tampered = attempt.model_dump(mode="json")
    tampered["output_repair_feedback"]["current_issues"][0][
        "safe_explanation"
    ] = "另一条说明。"
    with pytest.raises(ValidationError, match="dispatch request hash"):
        RuntimeModelPhysicalAttemptRequest.model_validate(tampered)

    settlement = RuntimeModelPhysicalAttemptSettlement.create(
        settlement_id="settlement_repair_01",
        settle_apply_id="settle_apply_repair_01",
        physical_attempt_id="physical_repair_01",
        physical_request_binding_sha256=SHA_A,
        logical_call_id="logical_01",
        physical_ordinal=1,
        settled_turn_id="turn_01",
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
        outcome_fingerprint=SHA_B,
    )
    loaded = RuntimeModelPhysicalAttemptSettlement.model_validate(
        settlement.model_dump(mode="json")
    )
    assert isinstance(
        loaded.next_output_repair_feedback,
        RuntimeModelOutputRepairFeedback,
    )


def test_output_repair_move_preserves_persisted_hash_identities() -> None:
    prompt = RuntimeModelStructuredPrompt.create(
        system_prompt="原始 system",
        user_content='{"原始":"user"}',
    )
    logical = _logical_request(
        output_repair_protocol=(
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        ),
        structured_prompt=prompt,
    )
    feedback = _feedback()
    first_attempt = RuntimeModelPhysicalAttemptRequest.create(
        physical_attempt_id="physical_repair_01",
        physical_attempt_key="physical_repair_key_01",
        logical_call_id=logical.logical_call_id,
        logical_request_binding_sha256=logical.binding_sha256,
        physical_ordinal=1,
        started_turn_id="turn_01",
        provider=logical.provider,
        model=logical.model,
        endpoint_fingerprint=logical.endpoint_fingerprint,
        request_sha256=logical.request_sha256,
        output_repair_enabled=False,
        output_repair_feedback=None,
        provider_idempotency_key=None,
        dispatch_authority_sha256=SHA_B,
    )
    first_settlement = RuntimeModelPhysicalAttemptSettlement.create(
        settlement_id="settlement_repair_01",
        settle_apply_id="settle_apply_repair_01",
        physical_attempt_id=first_attempt.physical_attempt_id,
        physical_request_binding_sha256=first_attempt.binding_sha256,
        logical_call_id=logical.logical_call_id,
        physical_ordinal=first_attempt.physical_ordinal,
        settled_turn_id="turn_01",
        outcome=RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE,
        error_code="MODEL_BAD_RESPONSE",
        next_output_repair_feedback=feedback,
        outcome_fingerprint=SHA_A,
    )
    repair_attempt = RuntimeModelPhysicalAttemptRequest.create(
        physical_attempt_id="physical_repair_02",
        physical_attempt_key="physical_repair_key_02",
        logical_call_id=logical.logical_call_id,
        logical_request_binding_sha256=logical.binding_sha256,
        physical_ordinal=2,
        started_turn_id="turn_02",
        provider=logical.provider,
        model=logical.model,
        endpoint_fingerprint=logical.endpoint_fingerprint,
        request_sha256=logical.request_sha256,
        output_repair_enabled=True,
        output_repair_feedback=feedback,
        provider_idempotency_key=None,
        dispatch_authority_sha256=SHA_B,
    )

    assert (
        logical.binding_sha256,
        first_attempt.dispatch_request_sha256,
        first_attempt.binding_sha256,
        first_settlement.receipt_sha256,
        repair_attempt.dispatch_request_sha256,
        repair_attempt.binding_sha256,
    ) == (
        "6ca197f2034b708177653be7602c354b2201eae2e281789623fe67f593d429f8",
        "2c874ac39f4e74c77cd9ab7fc00e02525ea79397756815dc2ff6ba907cc1dd30",
        "319b22536fa80b4586748ad5f33ec96eaabe737295ca57d5d4296b291177f7c3",
        "1d57de439b882d20f96921110fc9a5b798ad9fddc4ae451ab75b157b8e4efa69",
        "c306bb70897aa3c19888242dc6977b036bbe1e9eaf7ce423bee30d8f31e0ea6a",
        "098ed274244df694082cbaf760d7f5ae9457b1cd762e63b25226874190332f4a",
    )
