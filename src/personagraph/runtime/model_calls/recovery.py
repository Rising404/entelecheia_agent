"""Read-only recovery projections for durable Runtime model calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal, Protocol

from .contracts import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


RuntimeModelCallRecoveryDisposition = Literal[
    "ready",
    "retry_authorized",
    "waiting_pending",
    "waiting_uncertain",
    "succeeded",
    "terminal_failure",
    "physical_limit_exhausted",
]


@dataclass(frozen=True, slots=True)
class RuntimeModelCallRecoverySnapshot:
    """Read-only durable state exposed to an external provider coordinator."""

    logical_call_id: str
    provider: str
    model: str
    endpoint_fingerprint: str
    request_sha256: str
    disposition: RuntimeModelCallRecoveryDisposition
    physical_attempt_id: str | None
    physical_attempt_key: str | None
    physical_request_binding_sha256: str | None
    physical_ordinal: int | None
    started_turn_id: str | None
    provider_idempotency_key: str | None
    provider_request_id: str | None

    def __post_init__(self) -> None:
        if not self.logical_call_id:
            raise ValueError("logical_call_id must be non-empty")
        if not self.provider or not self.model:
            raise ValueError("recovery snapshot Provider identity is incomplete")
        if not _SHA256.fullmatch(self.endpoint_fingerprint):
            raise ValueError("recovery snapshot endpoint fingerprint is invalid")
        if not _SHA256.fullmatch(self.request_sha256):
            raise ValueError("recovery snapshot request hash is invalid")
        has_physical = self.physical_attempt_id is not None
        physical_parts = (
            self.physical_attempt_key,
            self.physical_request_binding_sha256,
            self.physical_ordinal,
            self.started_turn_id,
        )
        if has_physical != all(value is not None for value in physical_parts):
            raise ValueError("recovery snapshot physical identity is incomplete")
        if (
            self.physical_request_binding_sha256 is not None
            and not _SHA256.fullmatch(self.physical_request_binding_sha256)
        ):
            raise ValueError("recovery snapshot physical binding is invalid")
        if not has_physical and (
            self.provider_idempotency_key is not None
            or self.provider_request_id is not None
        ):
            raise ValueError("recovery snapshot external identity has no dispatch")


class _StoredPhysicalAttempt(Protocol):
    request: RuntimeModelPhysicalAttemptRequest
    settlement: RuntimeModelPhysicalAttemptSettlement | None


def project_runtime_model_call_recovery(
    *,
    logical_request: RuntimeModelLogicalRequest,
    physical_attempts: Sequence[_StoredPhysicalAttempt],
) -> RuntimeModelCallRecoverySnapshot:
    """Project immutable ledger state without granting retry authority."""

    if not physical_attempts:
        return RuntimeModelCallRecoverySnapshot(
            logical_call_id=logical_request.logical_call_id,
            provider=logical_request.provider,
            model=logical_request.model,
            endpoint_fingerprint=logical_request.endpoint_fingerprint,
            request_sha256=logical_request.request_sha256,
            disposition="ready",
            physical_attempt_id=None,
            physical_attempt_key=None,
            physical_request_binding_sha256=None,
            physical_ordinal=None,
            started_turn_id=None,
            provider_idempotency_key=None,
            provider_request_id=None,
        )
    last = physical_attempts[-1]
    settlement = last.settlement
    if settlement is None:
        disposition: RuntimeModelCallRecoveryDisposition = "waiting_pending"
        provider_request_id = None
    elif settlement.outcome is RuntimeModelPhysicalOutcome.UNCERTAIN:
        disposition = "waiting_uncertain"
        provider_request_id = settlement.provider_request_id
    elif settlement.outcome is RuntimeModelPhysicalOutcome.SUCCEEDED:
        disposition = "succeeded"
        provider_request_id = settlement.provider_request_id
    elif settlement.outcome is RuntimeModelPhysicalOutcome.TERMINAL_FAILURE:
        disposition = "terminal_failure"
        provider_request_id = settlement.provider_request_id
    elif len(physical_attempts) >= logical_request.max_physical_attempts:
        disposition = "physical_limit_exhausted"
        provider_request_id = settlement.provider_request_id
    else:
        disposition = "retry_authorized"
        provider_request_id = settlement.provider_request_id
    return RuntimeModelCallRecoverySnapshot(
        logical_call_id=logical_request.logical_call_id,
        provider=logical_request.provider,
        model=logical_request.model,
        endpoint_fingerprint=logical_request.endpoint_fingerprint,
        request_sha256=logical_request.request_sha256,
        disposition=disposition,
        physical_attempt_id=last.request.physical_attempt_id,
        physical_attempt_key=last.request.physical_attempt_key,
        physical_request_binding_sha256=last.request.binding_sha256,
        physical_ordinal=last.request.physical_ordinal,
        started_turn_id=last.request.started_turn_id,
        provider_idempotency_key=last.request.provider_idempotency_key,
        provider_request_id=provider_request_id,
    )


__all__ = [
    "RuntimeModelCallRecoveryDisposition",
    "RuntimeModelCallRecoverySnapshot",
    "project_runtime_model_call_recovery",
]
