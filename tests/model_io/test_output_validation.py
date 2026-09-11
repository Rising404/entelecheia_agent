from __future__ import annotations

import pytest

from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
)
from personagraph.model_io.output_validation import (
    ModelOutputValidationError,
    build_model_output_repair_feedback,
)


def _issue(code: str) -> RuntimeModelOutputRepairIssue:
    return RuntimeModelOutputRepairIssue(
        category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
        code=code,
        paths=("",),
        safe_explanation=f"{code} failed validation.",
    )


def test_build_repair_feedback_sorts_and_deduplicates_issues() -> None:
    first = _issue("alpha")
    second = _issue("zeta")

    feedback = build_model_output_repair_feedback(
        target_contract="example-v1",
        rejected_physical_ordinal=1,
        rejected_response_sha256="a" * 64,
        issues=(second, first, second),
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
        omitted_issue_count=0,
    )

    assert feedback.current_issues == (first, second)
    assert feedback.issue_coverage is RuntimeModelOutputRepairIssueCoverage.COMPLETE
    assert feedback.omitted_issue_count == 0


def test_build_repair_feedback_rejects_empty_issue_set() -> None:
    with pytest.raises(ValueError, match="at least one issue"):
        build_model_output_repair_feedback(
            target_contract="example-v1",
            rejected_physical_ordinal=1,
            rejected_response_sha256="a" * 64,
            issues=(),
            issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
            omitted_issue_count=0,
        )


def test_output_validation_error_preserves_bounded_repair_metadata() -> None:
    issues = (_issue("alpha"), _issue("zeta"))

    error = ModelOutputValidationError(
        "invalid result",
        retryable=False,
        repair_code="contract.invalid",
        safe_repair_reason="The typed result is invalid.",
        repair_issues=issues,
        repair_issue_coverage=RuntimeModelOutputRepairIssueCoverage.COMPLETE,
    )

    assert str(error) == "invalid result"
    assert error.retryable is False
    assert error.repair_code == "contract.invalid"
    assert error.safe_repair_reason == "The typed result is invalid."
    assert error.repair_issues == issues
    assert (
        error.repair_issue_coverage
        is RuntimeModelOutputRepairIssueCoverage.COMPLETE
    )
    assert error.omitted_repair_issue_count == 0


def test_output_validation_error_rejects_noncanonical_repair_metadata() -> None:
    with pytest.raises(ValueError, match="repair_code"):
        ModelOutputValidationError("invalid result", repair_code="not valid")

    with pytest.raises(ValueError, match="single-line"):
        ModelOutputValidationError(
            "invalid result",
            safe_repair_reason="invalid\nreason",
        )

    with pytest.raises(ValueError, match="unique stable order"):
        ModelOutputValidationError(
            "invalid result",
            repair_issues=(_issue("zeta"), _issue("alpha")),
        )

    with pytest.raises(ValueError, match="positive omitted count"):
        ModelOutputValidationError(
            "invalid result",
            repair_issue_coverage=RuntimeModelOutputRepairIssueCoverage.TRUNCATED,
        )
