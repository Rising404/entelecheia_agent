from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
    RuntimeModelTypedResult,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _logical_request() -> RuntimeModelLogicalRequest:
    return RuntimeModelLogicalRequest.create(
        logical_call_id="logical_01",
        session_id="session_01",
        task_id="task_01",
        auxiliary_graph_id="auxiliary_graph_01",
        goal_id="goal_01",
        execution_subject_id="subject_01",
        invocation_turn_id="turn_01",
        call_kind="task_graph_semantic_verification",
        purpose="runtime_task_graph_semantic_verification",
        provider="mock",
        model="mock-model",
        endpoint_fingerprint=SHA_A,
        request_contract="task-graph-semantic-verification-prompt-v1",
        request_payload={"proposal": {"root": "root"}, "facts": ["bounded"]},
        typed_result_contract="task-graph-semantic-verification-result-v1",
        max_physical_attempts=6,
        state_guard_sha256=SHA_B,
    )


def _physical_request() -> RuntimeModelPhysicalAttemptRequest:
    logical = _logical_request()
    return RuntimeModelPhysicalAttemptRequest.create(
        physical_attempt_id="physical_01",
        physical_attempt_key="physical_key_01",
        logical_call_id=logical.logical_call_id,
        logical_request_binding_sha256=logical.binding_sha256,
        physical_ordinal=1,
        started_turn_id="turn_01",
        provider=logical.provider,
        model=logical.model,
        endpoint_fingerprint=logical.endpoint_fingerprint,
        request_sha256=logical.request_sha256,
        provider_idempotency_key=None,
        dispatch_authority_sha256=SHA_A,
    )


def test_logical_request_is_canonical_self_authenticating_and_bounded() -> None:
    request = _logical_request()
    assert request.output_repair_target_contract == request.typed_result_contract
    assert request.request_json == json.dumps(
        {"facts": ["bounded"], "proposal": {"root": "root"}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValidationError, match="request hash"):
        RuntimeModelLogicalRequest.model_validate(
            request.model_dump() | {"request_sha256": SHA_A}
        )


def test_current_nonrepair_request_binds_explicit_null_protocol_fields() -> None:
    request = _logical_request()
    dumped = request.model_dump(mode="json")

    assert dumped["output_repair_protocol"] is None
    assert dumped["structured_prompt"] is None
    assert RuntimeModelLogicalRequest.model_validate(dumped) == request

    for field in ("output_repair_protocol", "structured_prompt"):
        missing = dict(dumped)
        missing.pop(field)
        with pytest.raises(ValidationError, match="Field required"):
            RuntimeModelLogicalRequest.model_validate(missing)

    old_style_payload = dict(dumped)
    old_style_payload.pop("binding_sha256")
    old_style_payload.pop("output_repair_protocol")
    old_style_payload.pop("structured_prompt")
    old_style_hash = hashlib.sha256(
        json.dumps(
            old_style_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValidationError, match="binding hash"):
        RuntimeModelLogicalRequest.model_validate(
            dumped | {"binding_sha256": old_style_hash}
        )


def test_model_repair_contract_is_frozen_without_changing_stored_result_contract() -> None:
    original = _logical_request()
    values = original.model_dump(exclude={"request_json", "request_sha256", "binding_sha256"})
    request = RuntimeModelLogicalRequest.create(
        **values, request_payload={"model_output_contract": "l1-short-reference-decision"},
    )
    assert request.output_repair_target_contract == "l1-short-reference-decision"
    assert request.typed_result_contract == original.typed_result_contract
    assert "output_repair_target_contract" not in request.model_dump()
    restored = RuntimeModelLogicalRequest.model_validate_json(request.model_dump_json())
    assert restored.output_repair_target_contract == request.output_repair_target_contract
    assert restored.binding_sha256 == request.binding_sha256


@pytest.mark.parametrize("invalid", [None, "", "not a contract", 3, []])
def test_invalid_frozen_model_repair_contract_is_rejected(invalid) -> None:
    values = _logical_request().model_dump(exclude={"request_json", "request_sha256", "binding_sha256"})
    with pytest.raises(ValidationError, match="model output contract"):
        RuntimeModelLogicalRequest.create(**values, request_payload={"model_output_contract": invalid})


def test_four_message_request_requires_and_binds_its_prompt() -> None:
    prompt = RuntimeModelStructuredPrompt.create(
        system_prompt="system",
        user_content="user",
    )
    request = RuntimeModelLogicalRequest.create(
        **(
            _logical_request().model_dump(
                exclude={
                    "binding_sha256",
                    "request_json",
                    "request_sha256",
                    "output_repair_protocol",
                    "structured_prompt",
                }
            )
            | {
                "request_payload": {"facts": ["bounded"]},
                "output_repair_protocol": (
                    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
                ),
                "structured_prompt": prompt,
            }
        )
    )
    assert request.structured_prompt == prompt
    with pytest.raises(ValidationError, match="canonical JSON"):
        RuntimeModelLogicalRequest.model_validate(
            request.model_dump() | {"request_json": '{"proposal": {"root": "root"}}'}
        )
    with pytest.raises(ValidationError, match="supplied together"):
        RuntimeModelLogicalRequest.create(
            **(
                request.model_dump(
                    exclude={
                        "binding_sha256",
                        "request_json",
                        "request_sha256",
                        "auxiliary_graph_id",
                    }
                )
                | {"request_payload": {"proposal": {"root": "root"}}}
            )
        )


def test_physical_request_binds_logical_and_provider_facts() -> None:
    attempt = _physical_request()
    assert attempt.model_call_id == "physical_key_01"
    with pytest.raises(ValidationError, match="binding hash"):
        RuntimeModelPhysicalAttemptRequest.model_validate(
            attempt.model_dump() | {"provider": "other-provider"}
        )


def test_success_settlement_requires_replayable_typed_result() -> None:
    attempt = _physical_request()
    typed = RuntimeModelTypedResult.create(
        result_contract="task-graph-semantic-verification-result-v1",
        result_payload={"items": [{"dimension": "goal_coverage", "verdict": "pass"}]},
    )
    settlement = RuntimeModelPhysicalAttemptSettlement.create(
        settlement_id="settlement_01",
        settle_apply_id="settle_apply_01",
        physical_attempt_id=attempt.physical_attempt_id,
        physical_request_binding_sha256=attempt.binding_sha256,
        logical_call_id=attempt.logical_call_id,
        physical_ordinal=attempt.physical_ordinal,
        settled_turn_id="turn_01",
        outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
        provider_request_id="provider-request-01",
        finish_reason="end_turn",
        error_code=None,
        typed_result=typed,
        outcome_fingerprint=SHA_B,
    )
    assert settlement.typed_result is not None
    assert settlement.typed_result.parsed()["items"][0]["verdict"] == "pass"

    with pytest.raises(ValidationError, match="requires a typed result"):
        RuntimeModelPhysicalAttemptSettlement.create(
            **(
                settlement.model_dump(exclude={"receipt_sha256", "typed_result"})
            )
        )


def test_uncertain_settlement_has_no_result_and_is_not_success() -> None:
    attempt = _physical_request()
    settlement = RuntimeModelPhysicalAttemptSettlement.create(
        settlement_id="settlement_uncertain_01",
        settle_apply_id="settle_apply_uncertain_01",
        physical_attempt_id=attempt.physical_attempt_id,
        physical_request_binding_sha256=attempt.binding_sha256,
        logical_call_id=attempt.logical_call_id,
        physical_ordinal=attempt.physical_ordinal,
        settled_turn_id="turn_02",
        outcome=RuntimeModelPhysicalOutcome.UNCERTAIN,
        error_code="provider_response_unknown",
        typed_result=None,
        outcome_fingerprint=SHA_A,
    )
    assert settlement.outcome is RuntimeModelPhysicalOutcome.UNCERTAIN
    assert settlement.typed_result is None

    with pytest.raises(ValidationError, match="only a succeeded"):
        RuntimeModelPhysicalAttemptSettlement.create(
            **(
                settlement.model_dump(exclude={"receipt_sha256"})
                | {
                    "typed_result": RuntimeModelTypedResult.create(
                        result_contract="unexpected-result-v1",
                        result_payload={"unexpected": True},
                    )
                }
            )
        )


def test_current_physical_request_requires_every_dispatch_and_repair_field() -> None:
    attempt = _physical_request()
    values = attempt.model_dump(mode="json")
    assert RuntimeModelPhysicalAttemptRequest.model_validate(values) == attempt
    for field in (
        "dispatch_request_sha256",
        "output_repair_enabled",
        "output_repair_feedback",
    ):
        missing = dict(values)
        missing.pop(field)
        with pytest.raises(ValidationError, match="Field required"):
            RuntimeModelPhysicalAttemptRequest.model_validate(missing)


def test_current_settlement_requires_explicit_nullable_repair_feedback() -> None:
    attempt = _physical_request()
    settlement = RuntimeModelPhysicalAttemptSettlement.create(
        settlement_id="settlement_required_feedback_01",
        settle_apply_id="settle_apply_required_feedback_01",
        physical_attempt_id=attempt.physical_attempt_id,
        physical_request_binding_sha256=attempt.binding_sha256,
        logical_call_id=attempt.logical_call_id,
        physical_ordinal=attempt.physical_ordinal,
        settled_turn_id="turn_01",
        outcome=RuntimeModelPhysicalOutcome.TERMINAL_FAILURE,
        error_code="MODEL_CALL_FAILED",
        outcome_fingerprint=SHA_A,
    )
    values = settlement.model_dump(mode="json")
    assert values["next_output_repair_feedback"] is None
    values.pop("next_output_repair_feedback")
    with pytest.raises(ValidationError, match="Field required"):
        RuntimeModelPhysicalAttemptSettlement.model_validate(values)


def test_physical_dispatch_hash_binds_bounded_repair_feedback() -> None:
    logical = _logical_request()
    feedback = RuntimeModelOutputRepairFeedback(
        target_contract="task-graph-semantic-verification-result-v1",
        rejected_physical_ordinal=1,
        rejected_response_sha256=SHA_B,
        issue_coverage="first_only",
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category="host_guard",
                code="architect_guard_rejected",
                paths=("",),
                safe_explanation=(
                    "The proposal violates a deterministic frozen Guard."
                ),
            ),
        ),
    )
    attempt = RuntimeModelPhysicalAttemptRequest.create(
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
        dispatch_authority_sha256=SHA_A,
    )
    assert attempt.request_sha256 == logical.request_sha256
    assert attempt.dispatch_request_sha256 != _physical_request().dispatch_request_sha256

    tampered = attempt.model_dump(mode="json")
    tampered["output_repair_feedback"]["current_issues"][0][
        "safe_explanation"
    ] = "A different reason."
    with pytest.raises(ValidationError, match="dispatch request hash"):
        RuntimeModelPhysicalAttemptRequest.model_validate(tampered)
