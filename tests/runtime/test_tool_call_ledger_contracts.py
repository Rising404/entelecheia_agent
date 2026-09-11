from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.runtime.tool_calls import (
    RuntimeToolEffectClass,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
    RuntimeToolTypedResult,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _logical(
    *,
    effect: RuntimeToolEffectClass = RuntimeToolEffectClass.READ_ONLY,
    retry: RuntimeToolRetryAuthority = RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
) -> RuntimeToolLogicalRequest:
    return RuntimeToolLogicalRequest.create(
        logical_tool_call_id="tool_call_01",
        session_id="session_01",
        work_run_id="work_run_01",
        attempt_id="attempt_01",
        call_ordinal=1,
        invocation_turn_id="turn_01",
        catalog_snapshot_sha256=SHA_A,
        tool_id="document_read",
        contract_version="document-read-v2",
        implementation_version="document-read-impl-v3",
        provider_identity_sha256=SHA_A,
        effect_profile_sha256=SHA_B,
        effect_class=effect,
        retry_authority=retry,
        arguments={"document_alias": "doc_01", "page": 2},
        result_contract="document-read-result-v2",
        max_physical_attempts=4,
        state_guard_sha256=SHA_A,
    )


def _physical(
    logical: RuntimeToolLogicalRequest,
    *,
    idempotency_key: str | None = None,
) -> RuntimeToolPhysicalAttemptRequest:
    return RuntimeToolPhysicalAttemptRequest.create(
        physical_attempt_id="tool_physical_01",
        physical_attempt_key="tool_physical_key_01",
        logical_tool_call_id=logical.logical_tool_call_id,
        logical_request_binding_sha256=logical.binding_sha256,
        physical_ordinal=1,
        started_turn_id="turn_01",
        retry_authority=logical.retry_authority,
        provider_idempotency_key=idempotency_key,
        dispatch_authority_sha256=SHA_B,
    )


def test_logical_tool_request_is_canonical_and_effect_bound() -> None:
    logical = _logical()
    assert logical.arguments_json == '{"document_alias":"doc_01","page":2}'
    with pytest.raises(ValidationError, match="binding hash"):
        RuntimeToolLogicalRequest.model_validate(
            logical.model_dump() | {"tool_id": "workspace_write"}
        )
    with pytest.raises(ValidationError, match="read_only_replay"):
        _logical(
            effect=RuntimeToolEffectClass.PROTECTED_EFFECT,
            retry=RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
        )


def test_provider_idempotency_requires_an_exact_key() -> None:
    logical = _logical(
        effect=RuntimeToolEffectClass.PROTECTED_EFFECT,
        retry=RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY,
    )
    with pytest.raises(ValidationError, match="requires exactly one key"):
        _physical(logical)
    physical = _physical(logical, idempotency_key="operation-key-01")
    assert physical.provider_idempotency_key == "operation-key-01"


def test_success_receipt_persists_only_canonical_typed_result() -> None:
    physical = _physical(_logical())
    typed = RuntimeToolTypedResult.create(
        result_contract="document-read-result-v2",
        result={"page": 2, "text": "bounded evidence"},
    )
    settled = RuntimeToolPhysicalAttemptSettlement.create(
        settlement_id="tool_settlement_01",
        settle_apply_id="tool_settle_apply_01",
        physical_attempt_id=physical.physical_attempt_id,
        physical_request_binding_sha256=physical.binding_sha256,
        logical_tool_call_id=physical.logical_tool_call_id,
        physical_ordinal=physical.physical_ordinal,
        settled_turn_id="turn_01",
        outcome=RuntimeToolPhysicalOutcome.SUCCEEDED,
        typed_result=typed,
        outcome_fingerprint=SHA_A,
    )
    assert settled.typed_result is not None
    assert settled.typed_result.parsed()["page"] == 2
    with pytest.raises(ValidationError, match="has a typed result"):
        RuntimeToolPhysicalAttemptSettlement.create(
            **settled.model_dump(exclude={"receipt_sha256", "typed_result"})
        )


def test_uncertain_receipt_is_explicit_and_has_no_retry_grant() -> None:
    physical = _physical(_logical())
    settled = RuntimeToolPhysicalAttemptSettlement.create(
        settlement_id="tool_settlement_uncertain_01",
        settle_apply_id="tool_settle_uncertain_01",
        physical_attempt_id=physical.physical_attempt_id,
        physical_request_binding_sha256=physical.binding_sha256,
        logical_tool_call_id=physical.logical_tool_call_id,
        physical_ordinal=physical.physical_ordinal,
        settled_turn_id="turn_02",
        outcome=RuntimeToolPhysicalOutcome.UNCERTAIN,
        error_code="response_unknown_after_dispatch",
        outcome_fingerprint=SHA_B,
    )
    assert settled.outcome is RuntimeToolPhysicalOutcome.UNCERTAIN
    assert settled.typed_result is None
    assert not hasattr(settled, "retry_allowed")
