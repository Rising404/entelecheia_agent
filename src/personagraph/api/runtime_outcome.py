from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RunStatus = Literal["completed", "partial", "needs_review", "failed", "cancelled"]


@dataclass(frozen=True)
class RunIssue:
    code: str
    domain: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "domain": self.domain,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


@dataclass(frozen=True)
class RetryHint:
    action: str
    label: str
    payload_hint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "payload_hint": self.payload_hint,
        }


@dataclass(frozen=True)
class RunOutcome:
    ok: bool
    status: RunStatus
    warnings: list[RunIssue] = field(default_factory=list)
    error: RunIssue | None = None
    retry: RetryHint | None = None
    partial: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "error": self.error.to_dict() if self.error else None,
            "retry": self.retry.to_dict() if self.retry else None,
            "partial": self.partial,
        }


def build_error_outcome(
    *,
    code: str,
    domain: str,
    message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> RunOutcome:
    issue = RunIssue(
        code=code,
        domain=domain,
        message=message,
        retryable=retryable,
        details=details or {},
    )
    retry = _retry_hint_for_error(code, retryable)
    return RunOutcome(
        ok=False,
        status="failed",
        warnings=[],
        error=issue,
        retry=retry,
        partial={},
    )


def _retry_hint_for_error(code: str, retryable: bool) -> RetryHint:
    if code in {"CHECKPOINT_MISSING", "REVIEW_STATE_EXPIRED"}:
        return RetryHint("restart_turn", "重新发起本轮请求", {})
    if retryable:
        return RetryHint("retry_turn", "重试本轮", {})
    return RetryHint("ask_user", "检查配置或输入", {})
