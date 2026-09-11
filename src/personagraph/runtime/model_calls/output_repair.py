"""将模型合同拒绝转换为有界修复材料（output repair），不执行请求或写账本。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from personagraph.model_io.contracts import ModelResult
from personagraph.model_io.output_repair_contracts import (
    RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES,
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCategory,
)
from personagraph.model_io.output_validation import (
    ModelOutputValidationError,
    build_model_output_repair_feedback,
)


@dataclass(frozen=True, slots=True)
class ModelOutputRejectionDisposition:
    """The retry and durable repair material produced by one rejection."""

    retryable: bool
    next_repair_feedback: RuntimeModelOutputRepairFeedback | None
    rejected_response_text: str | None


def resolve_model_output_rejection(
    *,
    validation_error: ModelOutputValidationError,
    model_result: ModelResult | None,
    repair_enabled: bool,
    repair_target_contract: str | None,
    rejected_physical_ordinal: int,
) -> ModelOutputRejectionDisposition:
    """为一次类型/合同校验失败生成 repair feedback 和精确被拒正文。

    修复内容必须能以 strict UTF-8 完整重放，并受 durable body 上限约束；
    编码失败或超限就终止该拒绝，不能截断后假装仍对应原 response hash。
    返回值只描述是否可重试及修复材料，真正的持久结算和下一次请求由 requests 驱动。
    """

    if not repair_enabled or not validation_error.retryable:
        return ModelOutputRejectionDisposition(
            retryable=validation_error.retryable,
            next_repair_feedback=None,
            rejected_response_text=None,
        )

    rejected_response_text = "" if model_result is None else model_result.reply
    try:
        rejected_response_bytes = rejected_response_text.encode("utf-8")
    except UnicodeEncodeError:
        return ModelOutputRejectionDisposition(
            retryable=False,
            next_repair_feedback=None,
            rejected_response_text=None,
        )
    if len(rejected_response_bytes) > RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES:
        return ModelOutputRejectionDisposition(
            retryable=False,
            next_repair_feedback=None,
            rejected_response_text=None,
        )

    issues = validation_error.repair_issues or (
        RuntimeModelOutputRepairIssue(
            category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
            code=validation_error.repair_code,
            paths=("",),
            safe_explanation=validation_error.safe_repair_reason,
        ),
    )
    feedback = build_model_output_repair_feedback(
        target_contract=str(repair_target_contract),
        rejected_physical_ordinal=rejected_physical_ordinal,
        rejected_response_sha256=hashlib.sha256(
            rejected_response_bytes
        ).hexdigest(),
        issues=issues,
        issue_coverage=validation_error.repair_issue_coverage,
        omitted_issue_count=validation_error.omitted_repair_issue_count,
    )
    return ModelOutputRejectionDisposition(
        retryable=validation_error.retryable,
        next_repair_feedback=feedback,
        rejected_response_text=rejected_response_text,
    )


def rejected_response_sha256_text(reply: str) -> str:
    """Hash the exact strict-UTF-8 response used by a durable repair turn."""

    return hashlib.sha256(reply.encode("utf-8")).hexdigest()


__all__ = [
    "ModelOutputRejectionDisposition",
    "rejected_response_sha256_text",
    "resolve_model_output_rejection",
]
