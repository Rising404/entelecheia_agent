"""Tools-owned emergency-revocation checkpoints for current execution paths.

The guard is deliberately independent of Runtime and lane orchestration.  It reads
the latest deny overlay at every checkpoint and delegates all selector semantics to
the catalog-owned matcher.  It only authorizes a transition that has not happened
yet; completed historical results are outside this API and remain replayable facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .persistence.records import (
    EmergencyRevocation,
    EmergencyRevocationTarget,
)


class EmergencyRevocationGuardStage(StrEnum):
    """Current, pre-transition checkpoints where a deny overlay must be re-read."""

    RESOLVE = "resolve"
    RESUME = "resume"
    EXECUTE = "execute"


class EmergencyRevocationGuardDisposition(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class EmergencyRevocationGuardReason(StrEnum):
    NOT_MATCHED = "emergency_revocation_not_matched"
    ACTIVE = "tool_emergency_revoked"
    STATE_UNAVAILABLE = "emergency_revocation_state_unavailable"
    TARGET_INVALID = "emergency_revocation_target_invalid"


@dataclass(frozen=True, slots=True)
class EmergencyRevocationGuardDecision:
    """Stable, non-throwing result for one current guard checkpoint."""

    stage: EmergencyRevocationGuardStage
    disposition: EmergencyRevocationGuardDisposition
    reason: EmergencyRevocationGuardReason
    matched_revocation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, EmergencyRevocationGuardStage):
            raise TypeError("stage must be an EmergencyRevocationGuardStage")
        if not isinstance(self.disposition, EmergencyRevocationGuardDisposition):
            raise TypeError(
                "disposition must be an EmergencyRevocationGuardDisposition"
            )
        if not isinstance(self.reason, EmergencyRevocationGuardReason):
            raise TypeError("reason must be an EmergencyRevocationGuardReason")
        identifiers = tuple(self.matched_revocation_ids)
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise ValueError("matched revocation IDs must be non-empty strings")
        canonical = tuple(sorted(set(identifiers)))
        object.__setattr__(self, "matched_revocation_ids", canonical)
        if self.disposition is EmergencyRevocationGuardDisposition.ALLOW:
            if self.reason is not EmergencyRevocationGuardReason.NOT_MATCHED:
                raise ValueError("an allow decision must use the not-matched reason")
            if canonical:
                raise ValueError("an allow decision cannot contain revocation IDs")
        elif self.reason is EmergencyRevocationGuardReason.NOT_MATCHED:
            raise ValueError("a deny decision cannot use the not-matched reason")
        if self.reason is EmergencyRevocationGuardReason.ACTIVE and not canonical:
            raise ValueError("an active-revocation decision requires a revocation ID")
        if self.reason is not EmergencyRevocationGuardReason.ACTIVE and canonical:
            raise ValueError(
                "only an active-revocation decision may contain revocation IDs"
            )

    @property
    def allowed(self) -> bool:
        return self.disposition is EmergencyRevocationGuardDisposition.ALLOW

    @property
    def code(self) -> str:
        return self.reason.value

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "disposition": self.disposition.value,
            "code": self.code,
            "matched_revocation_ids": list(self.matched_revocation_ids),
        }


class EmergencyRevocationDenied(RuntimeError):
    """Stable exception used when a current transition is denied."""

    def __init__(self, decision: EmergencyRevocationGuardDecision) -> None:
        if decision.allowed:
            raise ValueError("an allowed decision cannot raise a denial")
        self.decision = decision
        self.code = decision.code
        super().__init__(_denial_message(decision.reason))


class ActiveEmergencyRevocationReader(Protocol):
    """Narrow live-overlay port implemented by ``ToolCatalogRepository``."""

    def list_active_emergency_revocations(
        self,
    ) -> tuple[EmergencyRevocation, ...]: ...


class EmergencyRevocationGuard:
    """Fail-closed guard shared by future materializers and executors.

    ``check_current`` and ``enforce_current`` always perform a fresh reader call;
    decisions are intentionally not reusable authority tokens.  ``RESUME`` means
    granting a pending invocation a new chance to execute, not replaying a completed
    result.  There is consequently no historical-result or replay stage.
    """

    def __init__(self, reader: ActiveEmergencyRevocationReader) -> None:
        method = getattr(reader, "list_active_emergency_revocations", None)
        if not callable(method):
            raise TypeError(
                "reader must provide list_active_emergency_revocations()"
            )
        self._reader = reader

    def check_current(
        self,
        *,
        stage: EmergencyRevocationGuardStage,
        target: EmergencyRevocationTarget,
    ) -> EmergencyRevocationGuardDecision:
        """Read and apply the latest overlay for one not-yet-run transition."""

        if not isinstance(stage, EmergencyRevocationGuardStage):
            raise TypeError("stage must be an EmergencyRevocationGuardStage")
        if not isinstance(target, EmergencyRevocationTarget):
            return _denied(
                stage,
                EmergencyRevocationGuardReason.TARGET_INVALID,
            )
        try:
            revocations = self._reader.list_active_emergency_revocations()
            # The reader contract is an already-materialized immutable snapshot.
            # Mutable lists and lazy iterators are rejected rather than observed
            # while their contents can still change underneath the match pass.
            if not isinstance(revocations, tuple) or any(
                not isinstance(revocation, EmergencyRevocation)
                for revocation in revocations
            ):
                raise TypeError("active revocation reader returned an invalid snapshot")
            matches = tuple(
                revocation.revocation_id
                for revocation in revocations
                if revocation.selector.matches(target)
            )
        except Exception:
            # Storage, schema, decoding and matcher failures have the same security
            # meaning here: the latest deny state could not be proven safe.
            return _denied(
                stage,
                EmergencyRevocationGuardReason.STATE_UNAVAILABLE,
            )
        if matches:
            return _denied(
                stage,
                EmergencyRevocationGuardReason.ACTIVE,
                matched_revocation_ids=matches,
            )
        return EmergencyRevocationGuardDecision(
            stage=stage,
            disposition=EmergencyRevocationGuardDisposition.ALLOW,
            reason=EmergencyRevocationGuardReason.NOT_MATCHED,
        )

    def enforce_current(
        self,
        *,
        stage: EmergencyRevocationGuardStage,
        target: EmergencyRevocationTarget,
    ) -> EmergencyRevocationGuardDecision:
        """Return an allow decision or raise one stable fail-closed error."""

        decision = self.check_current(stage=stage, target=target)
        if not decision.allowed:
            raise EmergencyRevocationDenied(decision)
        return decision


def _denied(
    stage: EmergencyRevocationGuardStage,
    reason: EmergencyRevocationGuardReason,
    *,
    matched_revocation_ids: tuple[str, ...] = (),
) -> EmergencyRevocationGuardDecision:
    return EmergencyRevocationGuardDecision(
        stage=stage,
        disposition=EmergencyRevocationGuardDisposition.DENY,
        reason=reason,
        matched_revocation_ids=matched_revocation_ids,
    )


def _denial_message(reason: EmergencyRevocationGuardReason) -> str:
    if reason is EmergencyRevocationGuardReason.ACTIVE:
        return "Tool transition is blocked by an active emergency revocation."
    if reason is EmergencyRevocationGuardReason.TARGET_INVALID:
        return "Tool transition is blocked because its revocation target is invalid."
    return "Tool transition is blocked because emergency revocation state is unavailable."


__all__ = [
    "ActiveEmergencyRevocationReader",
    "EmergencyRevocationDenied",
    "EmergencyRevocationGuard",
    "EmergencyRevocationGuardDecision",
    "EmergencyRevocationGuardDisposition",
    "EmergencyRevocationGuardReason",
    "EmergencyRevocationGuardStage",
]
