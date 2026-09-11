"""Lifecycle transactions for the persistent Tool Catalog repository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .records import (
    CatalogCorruptionError,
    CatalogRevisionSnapshot,
    DefaultProfileItem,
    DefaultProfileSelection,
    DefaultProfileSnapshot,
    EmergencyRevocation,
    EmergencyRevocationSelector,
    _CATALOG_ID,
    _canonical_json,
    _identity_values,
    _nonempty,
    _normalize_profile_selections,
    _require_non_negative_integer,
    _revocation_from_row,
    _utc_datetime,
)
from ..model import (
    CatalogConflictError,
    CatalogError,
    CatalogStatus,
    validate_catalog_status_transition,
)
from ..binding import ToolIdentity


class _CatalogLifecycleOperations:
    """Private method partition; the concrete repository owns all state and I/O."""

    def set_status(
        self,
        identity: ToolIdentity,
        status: CatalogStatus,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> CatalogRevisionSnapshot:
        """Publish one validated lifecycle transition using catalog CAS."""

        if not isinstance(identity, ToolIdentity):
            raise TypeError("identity must be a ToolIdentity")
        if not isinstance(status, CatalogStatus):
            raise TypeError("status must be CatalogStatus")
        _require_non_negative_integer(expected_revision, "expected_revision")
        actor = _nonempty(actor, "actor")
        reason = _nonempty(reason, "reason")
        with self._write_transaction() as (connection, current, _, timestamp):
            self._require_catalog_revision(current, expected_revision)
            row = connection.execute(
                "SELECT status, definition_digest FROM tool_catalog_entries "
                "WHERE tool_id=? AND contract_version=? AND implementation_version=?",
                _identity_values(identity),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown ToolDefinition identity: {identity!r}")
            try:
                previous = CatalogStatus(str(row["status"]))
            except ValueError as exc:
                raise CatalogCorruptionError(
                    "persisted catalog status is invalid"
                ) from exc
            if previous is status:
                return current
            validate_catalog_status_transition(previous, status)
            if status is CatalogStatus.ACTIVE:
                conflicting = connection.execute(
                    "SELECT 1 FROM tool_catalog_entries WHERE tool_id=? AND "
                    "contract_version=? AND implementation_version<>? AND status='active'",
                    _identity_values(identity),
                ).fetchone()
                if conflicting is not None:
                    raise CatalogConflictError(
                        "another implementation of this tool contract is already active"
                    )
            next_revision = current.revision + 1
            connection.execute(
                "UPDATE tool_catalog_entries SET status=?, updated_revision=? "
                "WHERE tool_id=? AND contract_version=? AND implementation_version=?",
                (status.value, next_revision, *_identity_values(identity)),
            )
            snapshot = self._publish_catalog_revision(
                connection,
                revision=next_revision,
                parent_revision=current.revision,
                actor=actor,
                reason=reason,
                action="set_status",
                change={
                    "identity": identity.to_dict(),
                    "from": previous.value,
                    "to": status.value,
                },
                timestamp=timestamp,
            )
            if status is CatalogStatus.RETIRED:
                connection.execute(
                    "INSERT INTO tool_catalog_tombstones("
                    "tool_id, contract_version, implementation_version, "
                    "definition_digest, retired_revision, retired_by, reason, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *_identity_values(identity),
                        str(row["definition_digest"]),
                        next_revision,
                        actor,
                        reason,
                        timestamp,
                    ),
                )
            return snapshot

    def publish_default_profile(
        self,
        selections: Sequence[DefaultProfileSelection],
        *,
        expected_catalog_revision: int,
        expected_profile_revision: int,
        actor: str,
        reason: str,
    ) -> DefaultProfileSnapshot:
        """CAS-replace the ordered defaults without changing catalog revision."""

        normalized = _normalize_profile_selections(selections)
        _require_non_negative_integer(
            expected_catalog_revision,
            "expected_catalog_revision",
        )
        _require_non_negative_integer(
            expected_profile_revision,
            "expected_profile_revision",
        )
        actor = _nonempty(actor, "actor")
        reason = _nonempty(reason, "reason")
        with self._write_transaction() as (
            connection,
            current,
            current_profile,
            timestamp,
        ):
            self._require_catalog_revision(current, expected_catalog_revision)
            if current_profile.revision != expected_profile_revision:
                raise CatalogConflictError(
                    "default profile revision mismatch: expected "
                    f"{expected_profile_revision}, current {current_profile.revision}"
                )
            entries = {entry.identity: entry for entry in current.entries}
            items: list[DefaultProfileItem] = []
            for ordinal, selection in enumerate(normalized):
                entry = entries.get(selection.identity)
                if entry is None:
                    raise KeyError(
                        f"unknown ToolDefinition identity: {selection.identity!r}"
                    )
                if entry.status is not CatalogStatus.ACTIVE:
                    raise CatalogError(
                        "default profile may contain only active ToolDefinitions"
                    )
                items.append(
                    DefaultProfileItem(
                        ordinal,
                        entry.identity,
                        entry.definition_digest,
                        selection.availability,
                    )
                )
            frozen_items = tuple(items)
            if (
                current_profile.catalog_revision == current.revision
                and current_profile.items == frozen_items
            ):
                return current_profile
            profile = DefaultProfileSnapshot(
                current_profile.revision + 1,
                current.revision,
                frozen_items,
            )
            self._insert_profile_revision(
                connection,
                profile=profile,
                actor=actor,
                reason=reason,
                timestamp=timestamp,
            )
            connection.execute(
                "UPDATE tool_catalog_control SET current_default_profile_revision=?, "
                "updated_at=? WHERE catalog_id=?",
                (profile.revision, timestamp, _CATALOG_ID),
            )
            self._insert_audit_event(
                connection,
                event_id=f"default-profile-revision:{profile.revision}",
                catalog_revision=current.revision,
                action="publish_default_profile",
                actor=actor,
                reason=reason,
                change={
                    "profile_revision": profile.revision,
                    "items": [item.descriptor() for item in profile.items],
                },
                timestamp=timestamp,
            )
            return profile

    def issue_emergency_revocation(
        self,
        revocation_id: str,
        selector: EmergencyRevocationSelector,
        *,
        issued_by: str,
        reason: str,
        effective_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> EmergencyRevocation:
        """Persist an immediate deny overlay without publishing a catalog revision."""

        revocation_id = _nonempty(revocation_id, "revocation_id")
        if not isinstance(selector, EmergencyRevocationSelector):
            raise TypeError("selector must be EmergencyRevocationSelector")
        issued_by = _nonempty(issued_by, "issued_by")
        reason = _nonempty(reason, "reason")
        with self._write_transaction() as (connection, current, _, _):
            now = self._now()
            revocation = EmergencyRevocation(
                revocation_id=revocation_id,
                selector=selector,
                reason=reason,
                issued_by=issued_by,
                effective_at=effective_at or now,
                expires_at=expires_at,
                created_at=now,
            )
            if connection.execute(
                "SELECT 1 FROM tool_emergency_revocations WHERE revocation_id=?",
                (revocation_id,),
            ).fetchone() is not None:
                raise CatalogConflictError(
                    f"emergency revocation already exists: {revocation_id!r}"
                )
            selector_json = _canonical_json(selector.descriptor())
            connection.execute(
                "INSERT INTO tool_emergency_revocations("
                "revocation_id, selector_json, selector_digest, disposition, reason, "
                "issued_by, effective_at, expires_at, cleared_at, cleared_by, "
                "clear_reason, created_at"
                ") VALUES (?, ?, ?, 'deny_execution', ?, ?, ?, ?, NULL, NULL, NULL, ?)",
                (
                    revocation.revocation_id,
                    selector_json,
                    selector.digest,
                    revocation.reason,
                    revocation.issued_by,
                    revocation.effective_at.isoformat(),
                    (
                        revocation.expires_at.isoformat()
                        if revocation.expires_at is not None
                        else None
                    ),
                    revocation.created_at.isoformat(),
                ),
            )
            self._insert_audit_event(
                connection,
                event_id=f"emergency-revocation:{revocation_id}:issue",
                catalog_revision=current.revision,
                action="issue_emergency_revocation",
                actor=issued_by,
                reason=reason,
                change={
                    "revocation_id": revocation_id,
                    "selector": selector.descriptor(),
                    "effective_at": revocation.effective_at.isoformat(),
                    "expires_at": (
                        revocation.expires_at.isoformat()
                        if revocation.expires_at is not None
                        else None
                    ),
                },
                timestamp=revocation.created_at.isoformat(),
            )
            return revocation

    def list_active_emergency_revocations(
        self,
        *,
        at: datetime | None = None,
    ) -> tuple[EmergencyRevocation, ...]:
        """Return exact active overlays, failing closed on any malformed record."""

        instant = _utc_datetime(at, "at") if at is not None else None
        with self._read_transaction() as connection:
            self._read_current_state(connection)
            rows = connection.execute(
                "SELECT * FROM tool_emergency_revocations ORDER BY effective_at, "
                "revocation_id"
            ).fetchall()
            revocations = tuple(_revocation_from_row(row) for row in rows)
        if instant is None:
            instant = _utc_datetime(self._now(), "at")
        return tuple(
            revocation
            for revocation in revocations
            if revocation.active_at(instant)
        )

    def clear_emergency_revocation(
        self,
        revocation_id: str,
        *,
        cleared_by: str,
        reason: str,
        cleared_at: datetime | None = None,
    ) -> EmergencyRevocation:
        """Auditably clear an overlay without rewriting catalog snapshots."""

        revocation_id = _nonempty(revocation_id, "revocation_id")
        cleared_by = _nonempty(cleared_by, "cleared_by")
        reason = _nonempty(reason, "reason")
        with self._write_transaction() as (connection, current, _, _):
            row = connection.execute(
                "SELECT * FROM tool_emergency_revocations WHERE revocation_id=?",
                (revocation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"unknown emergency revocation: {revocation_id!r}"
                )
            previous = _revocation_from_row(row)
            if previous.cleared_at is not None:
                return previous
            instant = _utc_datetime(cleared_at or self._now(), "cleared_at")
            cleared = EmergencyRevocation(
                revocation_id=previous.revocation_id,
                selector=previous.selector,
                reason=previous.reason,
                issued_by=previous.issued_by,
                effective_at=previous.effective_at,
                expires_at=previous.expires_at,
                created_at=previous.created_at,
                cleared_at=instant,
                cleared_by=cleared_by,
                clear_reason=reason,
            )
            connection.execute(
                "UPDATE tool_emergency_revocations SET cleared_at=?, cleared_by=?, "
                "clear_reason=? WHERE revocation_id=?",
                (instant.isoformat(), cleared_by, reason, revocation_id),
            )
            self._insert_audit_event(
                connection,
                event_id=f"emergency-revocation:{revocation_id}:clear",
                catalog_revision=current.revision,
                action="clear_emergency_revocation",
                actor=cleared_by,
                reason=reason,
                change={
                    "revocation_id": revocation_id,
                    "cleared_at": instant.isoformat(),
                },
                timestamp=instant.isoformat(),
            )
            return cleared
