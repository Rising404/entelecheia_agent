"""L1 聚合语义审查的纯契约与确定性策略。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


L1_SEMANTIC_VERIFICATION_CONTRACT_VERSION = "l1-semantic-verification-v2"
L1_SEMANTIC_VERIFICATION_RECEIPT_VERSION = "l1-semantic-verification-receipt-v2"
L1_SEMANTIC_VERIFICATION_FEATURE = "l1_semantic_verification_mode"
L1_SEMANTIC_RESULT_CONTRACT = "l1-semantic-verification-result-v3"

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class L1SemanticVerificationMode(StrEnum):
    OFF = "off"
    CONDITIONAL = "conditional"
    ALWAYS = "always"


class L1SemanticIssue(_Contract):
    """阻止当前候选交付的具体问题；可选定位到既有验收条件。"""

    message: str = Field(min_length=1, max_length=1000)
    acceptance_id: str | None = Field(default=None, pattern=_ID_PATTERN)

    @field_validator("message", mode="before")
    @classmethod
    def _trim_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class L1SemanticVerificationResult(_Contract):
    """唯一模型输出和持久结果；无问题只表示可交付，不证明全部完成。"""

    # 不提供默认空值：缺字段、截断或解析失败均不能变成通过。
    issues: tuple[L1SemanticIssue, ...] = Field(max_length=24)

    @property
    def verdict(self) -> Literal["pass", "revise"]:
        """Host 决策投影，不进入模型输出或持久 typed result。"""
        return "revise" if self.issues else "pass"

    def safe_feedback(self) -> str:
        return "L1 semantic verification requested repair: " + "; ".join(
            item.message for item in self.issues
        )


class L1SemanticVerificationTrigger(_Contract):
    mode: L1SemanticVerificationMode
    reasons: tuple[
        Literal[
            "policy_always",
            "multiple_acceptances",
            "tool_results_present",
        ],
        ...,
    ] = Field(default=(), max_length=2)

    @property
    def required(self) -> bool:
        return bool(self.reasons)

    @model_validator(mode="after")
    def _canonical_trigger(self) -> "L1SemanticVerificationTrigger":
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("semantic trigger reasons must be unique")
        if self.mode is L1SemanticVerificationMode.OFF and self.reasons:
            raise ValueError("off semantic mode cannot carry trigger reasons")
        if self.mode is L1SemanticVerificationMode.ALWAYS and self.reasons != (
            "policy_always",
        ):
            raise ValueError("always semantic mode requires policy_always")
        if (
            self.mode is L1SemanticVerificationMode.CONDITIONAL
            and "policy_always" in self.reasons
        ):
            raise ValueError("conditional semantic mode cannot be forced")
        return self


class L1SemanticVerificationReceipt(_Contract):
    """绑定确切候选的交付审查回执，不是任务全部完成的证明。"""

    contract_version: Literal["l1-semantic-verification-receipt-v2"] = (
        L1_SEMANTIC_VERIFICATION_RECEIPT_VERSION
    )
    disposition: Literal["not_required", "passed"]
    trigger: L1SemanticVerificationTrigger
    decision_hash: str = Field(pattern=_SHA256_PATTERN)
    plan_hash: str = Field(pattern=_SHA256_PATTERN)
    mechanical_verification_hash: str = Field(pattern=_SHA256_PATTERN)
    state_guard_hash: str = Field(pattern=_SHA256_PATTERN)
    checked_acceptances: int = Field(ge=1, le=24)
    checked_tool_results: int = Field(ge=0)
    reviewer_logical_call_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )
    reviewer_result_hash: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    reviewer_result: L1SemanticVerificationResult | None = None

    @model_validator(mode="after")
    def _consistent_disposition(self) -> "L1SemanticVerificationReceipt":
        review_fields = (
            self.reviewer_logical_call_id,
            self.reviewer_result_hash,
            self.reviewer_result,
        )
        if self.disposition == "not_required":
            if self.trigger.required or any(
                value is not None for value in review_fields
            ):
                raise ValueError("not_required receipt cannot carry a review")
        else:
            if not self.trigger.required or any(
                value is None for value in review_fields
            ):
                raise ValueError("passed receipt requires one triggered review")
            assert self.reviewer_result is not None
            if self.reviewer_result.verdict != "pass":
                raise ValueError("passed receipt requires a passing reviewer result")
        return self


def parse_l1_semantic_verification_mode(
    value: object,
) -> L1SemanticVerificationMode:
    if value is None:
        return L1SemanticVerificationMode.ALWAYS
    if not isinstance(value, str):
        raise ValueError("L1 semantic verification mode must be a string")
    try:
        return L1SemanticVerificationMode(value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            "L1 semantic verification mode must be off, conditional, or always"
        ) from exc


def derive_l1_semantic_verification_trigger(
    *,
    mode: L1SemanticVerificationMode,
    acceptance_count: int,
    tool_result_count: int,
) -> L1SemanticVerificationTrigger:
    if isinstance(acceptance_count, bool) or not 1 <= acceptance_count <= 24:
        raise ValueError("semantic trigger acceptance_count must be within 1..24")
    if isinstance(tool_result_count, bool) or tool_result_count < 0:
        raise ValueError("semantic trigger tool_result_count cannot be negative")
    if mode is L1SemanticVerificationMode.OFF:
        reasons: tuple[str, ...] = ()
    elif mode is L1SemanticVerificationMode.ALWAYS:
        reasons = ("policy_always",)
    else:
        selected: list[str] = []
        if acceptance_count > 1:
            selected.append("multiple_acceptances")
        if tool_result_count > 0:
            selected.append("tool_results_present")
        reasons = tuple(selected)
    return L1SemanticVerificationTrigger(mode=mode, reasons=reasons)


__all__ = [
    "L1_SEMANTIC_VERIFICATION_CONTRACT_VERSION",
    "L1_SEMANTIC_VERIFICATION_FEATURE",
    "L1_SEMANTIC_VERIFICATION_RECEIPT_VERSION",
    "L1_SEMANTIC_RESULT_CONTRACT",
    "L1SemanticIssue",
    "L1SemanticVerificationMode",
    "L1SemanticVerificationReceipt",
    "L1SemanticVerificationResult",
    "L1SemanticVerificationTrigger",
    "derive_l1_semantic_verification_trigger",
    "parse_l1_semantic_verification_mode",
]
