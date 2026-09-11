from __future__ import annotations

import hashlib

import pytest

from personagraph.model_io.contracts import ModelResult
from personagraph.model_io.output_repair_contracts import (
    RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.runtime.model_calls.output_repair import (
    ModelOutputRejectionDisposition,
    rejected_response_sha256_text,
    resolve_model_output_rejection,
)


def _result(reply: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id="logical-call",
    )


def _resolve(
    error: ModelOutputValidationError,
    *,
    reply: str | None = "rejected",
    repair_enabled: bool = True,
) -> ModelOutputRejectionDisposition:
    return resolve_model_output_rejection(
        validation_error=error,
        model_result=None if reply is None else _result(reply),
        repair_enabled=repair_enabled,
        repair_target_contract="test-output",
        rejected_physical_ordinal=3,
    )


def test_rejection_builds_bounded_fallback_feedback_without_copying_body() -> None:
    rejected = '{"secret":"must-not-enter-feedback"}'

    disposition = _resolve(
        ModelOutputValidationError(
            "unsafe parser detail",
            repair_code="test.required_field_missing",
            safe_repair_reason="The required field is missing.",
        ),
        reply=rejected,
    )

    assert disposition.retryable is True
    assert disposition.rejected_response_text == rejected
    feedback = disposition.next_repair_feedback
    assert feedback is not None
    assert feedback.target_contract == "test-output"
    assert feedback.rejected_physical_ordinal == 3
    assert feedback.rejected_response_sha256 == hashlib.sha256(
        rejected.encode("utf-8")
    ).hexdigest()
    assert feedback.current_issues == (
        RuntimeModelOutputRepairIssue(
            category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
            code="test.required_field_missing",
            paths=("",),
            safe_explanation="The required field is missing.",
        ),
    )
    assert "must-not-enter-feedback" not in feedback.model_dump_json()


def test_rejection_preserves_explicit_issue_metadata() -> None:
    issue = RuntimeModelOutputRepairIssue(
        category=RuntimeModelOutputRepairIssueCategory.SCHEMA,
        code="schema.wrong_type",
        paths=("/items/0",),
        safe_explanation="The first item has the wrong type.",
    )

    disposition = _resolve(
        ModelOutputValidationError(
            "bad typed output",
            repair_issues=(issue,),
            repair_issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        )
    )

    feedback = disposition.next_repair_feedback
    assert feedback is not None
    assert feedback.current_issues == (issue,)
    assert feedback.issue_coverage is RuntimeModelOutputRepairIssueCoverage.COMPLETE
    assert feedback.omitted_issue_count == 0


@pytest.mark.parametrize(
    ("repair_enabled", "validation_retryable", "expected_retryable"),
    ((False, True, True), (True, False, False), (False, False, False)),
)
def test_rejection_without_an_enabled_retryable_repair_has_no_material(
    repair_enabled: bool,
    validation_retryable: bool,
    expected_retryable: bool,
) -> None:
    disposition = _resolve(
        ModelOutputValidationError(
            "bad typed output",
            retryable=validation_retryable,
        ),
        repair_enabled=repair_enabled,
    )

    assert disposition.retryable is expected_retryable
    assert disposition.next_repair_feedback is None
    assert disposition.rejected_response_text is None


def test_missing_model_result_binds_repair_to_the_empty_response() -> None:
    disposition = _resolve(
        ModelOutputValidationError("dispatch raised validation error"),
        reply=None,
    )

    assert disposition.rejected_response_text == ""
    feedback = disposition.next_repair_feedback
    assert feedback is not None
    assert feedback.rejected_response_sha256 == hashlib.sha256(b"").hexdigest()


def test_isolated_surrogate_terminalizes_without_repair_material() -> None:
    disposition = _resolve(
        ModelOutputValidationError("non-UTF-8 response"),
        reply="\ud800",
    )

    assert disposition.retryable is False
    assert disposition.next_repair_feedback is None
    assert disposition.rejected_response_text is None


def test_utf8_limit_accepts_exact_bytes_and_rejects_one_more() -> None:
    exact = "界" * (RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES // 3)
    assert len(exact.encode("utf-8")) == RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES

    accepted = _resolve(ModelOutputValidationError("bad output"), reply=exact)
    rejected = _resolve(ModelOutputValidationError("bad output"), reply=exact + "x")

    assert accepted.retryable is True
    assert accepted.next_repair_feedback is not None
    assert accepted.rejected_response_text == exact
    assert rejected.retryable is False
    assert rejected.next_repair_feedback is None
    assert rejected.rejected_response_text is None


def test_durable_rejected_response_hash_uses_strict_utf8() -> None:
    assert rejected_response_sha256_text("精确正文") == hashlib.sha256(
        "精确正文".encode("utf-8")
    ).hexdigest()
    with pytest.raises(UnicodeEncodeError):
        rejected_response_sha256_text("\ud800")


def test_repair_contract_errors_are_not_swallowed() -> None:
    with pytest.raises(ValueError):
        resolve_model_output_rejection(
            validation_error=ModelOutputValidationError("bad output"),
            model_result=_result("rejected"),
            repair_enabled=True,
            repair_target_contract="not a canonical contract",
            rejected_physical_ordinal=1,
        )
