"""Provider-neutral typed-output rejection contract."""

from __future__ import annotations

import re

from .output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCoverage,
    runtime_model_output_repair_issue_sort_key,
)


class ModelOutputValidationError(ValueError):
    """传输成功，但所需有类型结果不可用。"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        repair_code: str = "invalid_typed_output",
        safe_repair_reason: str = (
            "The response did not satisfy the required typed output contract."
        ),
        repair_issues: tuple[RuntimeModelOutputRepairIssue, ...] | None = None,
        repair_issue_coverage: RuntimeModelOutputRepairIssueCoverage = (
            RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY
        ),
        omitted_repair_issue_count: int = 0,
    ) -> None:
        super().__init__(message)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", repair_code):
            raise ValueError("repair_code must be a canonical bounded identifier")
        if (
            not safe_repair_reason
            or safe_repair_reason != safe_repair_reason.strip()
            or "\x00" in safe_repair_reason
            or "\n" in safe_repair_reason
            or "\r" in safe_repair_reason
            or len(safe_repair_reason.encode("utf-8")) > 500
        ):
            raise ValueError(
                "safe_repair_reason must be canonical single-line text within "
                "500 UTF-8 bytes"
            )
        self.retryable = retryable
        self.repair_code = repair_code
        self.safe_repair_reason = safe_repair_reason
        if repair_issues is not None:
            if not repair_issues or any(
                not isinstance(issue, RuntimeModelOutputRepairIssue)
                for issue in repair_issues
            ):
                raise ValueError(
                    "repair_issues must contain RuntimeModelOutputRepairIssue"
                )
            ordered = tuple(
                sorted(repair_issues, key=runtime_model_output_repair_issue_sort_key)
            )
            if repair_issues != ordered or len(set(ordered)) != len(ordered):
                raise ValueError("repair_issues must use unique stable order")
        if not isinstance(
            repair_issue_coverage,
            RuntimeModelOutputRepairIssueCoverage,
        ):
            raise TypeError(
                "repair_issue_coverage must be RuntimeModelOutputRepairIssueCoverage"
            )
        if (
            isinstance(omitted_repair_issue_count, bool)
            or not isinstance(omitted_repair_issue_count, int)
            or omitted_repair_issue_count < 0
            or omitted_repair_issue_count > 1_000_000_000
        ):
            raise ValueError(
                "omitted_repair_issue_count must be within 0..1000000000"
            )
        if (
            repair_issue_coverage
            is RuntimeModelOutputRepairIssueCoverage.TRUNCATED
        ) != (omitted_repair_issue_count > 0):
            raise ValueError(
                "truncated repair issues require a positive omitted count; "
                "other coverage values require zero"
            )
        self.repair_issues = repair_issues
        self.repair_issue_coverage = repair_issue_coverage
        self.omitted_repair_issue_count = omitted_repair_issue_count


def build_model_output_repair_feedback(
    *,
    target_contract: str,
    rejected_physical_ordinal: int,
    rejected_response_sha256: str,
    issues: tuple[RuntimeModelOutputRepairIssue, ...],
    issue_coverage: RuntimeModelOutputRepairIssueCoverage,
    omitted_issue_count: int,
) -> RuntimeModelOutputRepairFeedback:
    """Fit stable validation issues into the durable feedback envelope."""

    ordered = tuple(
        sorted(
            set(issues),
            key=runtime_model_output_repair_issue_sort_key,
        )
    )
    if not ordered:
        raise ValueError("repair feedback requires at least one issue")
    max_candidate_count = min(len(ordered), 64)
    for accepted_count in range(max_candidate_count, 0, -1):
        additionally_omitted = len(ordered) - accepted_count
        total_omitted = omitted_issue_count + additionally_omitted
        effective_coverage = (
            RuntimeModelOutputRepairIssueCoverage.TRUNCATED
            if total_omitted > 0
            else issue_coverage
        )
        try:
            return RuntimeModelOutputRepairFeedback(
                target_contract=target_contract,
                rejected_physical_ordinal=rejected_physical_ordinal,
                rejected_response_sha256=rejected_response_sha256,
                issue_coverage=effective_coverage,
                omitted_issue_count=total_omitted,
                current_issues=ordered[:accepted_count],
            )
        except ValueError:
            # The issue contracts are already valid. Removing a stable tail only
            # helps when the aggregate feedback envelope exceeds its byte limit.
            continue
    raise ValueError("one repair issue cannot fit the feedback envelope")


__all__ = [
    "ModelOutputValidationError",
    "build_model_output_repair_feedback",
]
