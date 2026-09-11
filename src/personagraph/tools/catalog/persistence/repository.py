"""Durable control plane for the global default Tool Catalog.

Only stable :class:`ToolDefinition` values live here.  Runtime handlers, providers,
credentials, Session authority, resource scopes, and binding assertions are resolved
later and must never cross this persistence boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .lifecycle import _CatalogLifecycleOperations
from .schema import (
    EXPECTED_SCHEMA_FINGERPRINT as _EXPECTED_SCHEMA_FINGERPRINT,
    create_schema as _create_schema,
    schema_descriptor as _schema_descriptor,
    schema_fingerprint as _schema_fingerprint,
)
from .records import (
    BOOTSTRAP_MANIFEST_SCHEMA,
    CATALOG_DATABASE_SCHEMA_VERSION,
    CATALOG_REVISION_SCHEMA,
    DEFAULT_PROFILE_SCHEMA,
    DEFAULT_TOOL_CATALOG_DATABASE_PATH,
    CatalogAuditEvent,
    CatalogBootstrapResult,
    CatalogCorruptionError,
    CatalogEntrySnapshot,
    CatalogPersistenceError,
    CatalogRevisionSnapshot,
    CatalogSchemaError,
    DefaultProfileAvailability,
    DefaultCatalogResolutionState,
    DefaultProfileItem,
    DefaultProfileSelection,
    DefaultProfileSnapshot,
    EmergencyRevocation,
    EmergencyRevocationSelector,
    EmergencyRevocationTarget,
    ToolCatalogSeed,
    _CATALOG_ID,
    _PROFILE_ID,
    _audit_event_from_row,
    _bootstrap_manifest,
    _canonical_json,
    _decode_catalog_revision,
    _decode_default_profile,
    _definition_from_row,
    _entry_snapshot_from_row,
    _identity_values,
    _load_canonical_json_object,
    _nonempty,
    _normalize_seeds,
    _profile_item_from_row,
    _require_non_negative_integer,
    _resolve_database_path,
    _sha256_text,
    _stored_non_negative_integer,
)
from ..model import CatalogConflictError, CatalogStatus
from ..binding import ToolDefinition, ToolIdentity


class ToolCatalogRepository(_CatalogLifecycleOperations):
    """SQLite authority for immutable definitions and published default revisions."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        selected = (
            DEFAULT_TOOL_CATALOG_DATABASE_PATH
            if database_path is None
            else Path(database_path).expanduser()
        )
        self._database_path = _resolve_database_path(selected)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def database_path(self) -> Path:
        return self._database_path

    def initialize(self) -> None:
        path = self._validated_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path = self._validated_path()
            with self._connection(path) as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version == 0:
                    self._create_empty_database(connection)
                    return
                if version > CATALOG_DATABASE_SCHEMA_VERSION:
                    raise CatalogSchemaError(
                        "Tool Catalog was created by a newer application version"
                    )
                if version != CATALOG_DATABASE_SCHEMA_VERSION:
                    raise CatalogSchemaError(
                        f"unsupported Tool Catalog schema version: {version}"
                    )
                self._validate_schema(connection)
        except CatalogPersistenceError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise CatalogPersistenceError(
                f"failed to initialize Tool Catalog at {path}"
            ) from exc

    def current_snapshot(self) -> CatalogRevisionSnapshot:
        with self._read_transaction() as connection:
            catalog, _ = self._read_current_state(connection)
            return catalog

    def current_default_profile(self) -> DefaultProfileSnapshot:
        with self._read_transaction() as connection:
            _, profile = self._read_current_state(connection)
            return profile

    def default_resolution_state(self) -> DefaultCatalogResolutionState:
        """Atomically read the facts needed to build a new default catalog."""

        with self._read_transaction() as connection:
            current, profile = self._read_current_state(connection)
            profile_catalog = (
                current
                if profile.catalog_revision == current.revision
                else self._read_catalog_revision(
                    connection,
                    profile.catalog_revision,
                )
            )
            definitions = tuple(self._validated_definitions(connection).values())
            return DefaultCatalogResolutionState(
                current_catalog=current,
                default_profile=profile,
                profile_catalog=profile_catalog,
                definitions=definitions,
            )

    def snapshot_at(self, revision: int) -> CatalogRevisionSnapshot:
        _require_non_negative_integer(revision, "revision")
        with self._read_transaction() as connection:
            return self._read_catalog_revision(connection, revision)

    def load_definition(self, identity: ToolIdentity) -> ToolDefinition:
        if not isinstance(identity, ToolIdentity):
            raise TypeError("identity must be a ToolIdentity")
        with self._read_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tool_definitions WHERE tool_id=? AND "
                "contract_version=? AND implementation_version=?",
                _identity_values(identity),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown ToolDefinition identity: {identity!r}")
            return _definition_from_row(row)

    def list_definitions(self) -> tuple[ToolDefinition, ...]:
        with self._read_transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_definitions ORDER BY tool_id, "
                "contract_version, implementation_version"
            ).fetchall()
            return tuple(_definition_from_row(row) for row in rows)

    def list_audit_events(self) -> tuple[CatalogAuditEvent, ...]:
        with self._read_transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_catalog_audit_events "
                "ORDER BY catalog_revision, occurred_at, event_id"
            ).fetchall()
        return tuple(_audit_event_from_row(row) for row in rows)

    def bootstrap(
        self,
        seeds: Sequence[ToolCatalogSeed],
        *,
        actor: str = "system.bootstrap",
        reason: str = "bootstrap default Tool Catalog",
    ) -> CatalogBootstrapResult:
        """Apply a trusted seed manifest without mutating established defaults.

        The first non-empty manifest publishes active revision/profile ``1``.  Any
        identity introduced after that point is recorded as ``draft`` and is not
        silently added to the established default profile.
        """

        normalized_seeds = _normalize_seeds(seeds)
        actor = _nonempty(actor, "actor")
        reason = _nonempty(reason, "reason")
        manifest = _bootstrap_manifest(normalized_seeds)
        manifest_json = _canonical_json(manifest)
        manifest_digest = _sha256_text(manifest_json)
        self.initialize()
        with self._connection(self._validated_path()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_schema(connection)
                current, profile = self._read_current_state(connection)
                definitions_by_identity = self._validated_definitions(connection)
                manifest_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM tool_catalog_bootstrap_manifests"
                    ).fetchone()[0]
                )
                pristine = (
                    current.revision == 0
                    and profile.revision == 0
                    and not definitions_by_identity
                )
                if manifest_count == 0 and not pristine:
                    raise CatalogConflictError(
                        "initial bootstrap requires a pristine catalog; publish "
                        "manually mutated state through explicit lifecycle APIs"
                    )
                if manifest_count > 0 and (
                    current.revision == 0 or profile.revision == 0
                ):
                    raise CatalogCorruptionError(
                        "bootstrap history exists without a published catalog and "
                        "default profile"
                    )
                stored_manifest = connection.execute(
                    "SELECT manifest_json FROM tool_catalog_bootstrap_manifests "
                    "WHERE manifest_digest=?",
                    (manifest_digest,),
                ).fetchone()
                if stored_manifest is not None:
                    if str(stored_manifest["manifest_json"]) != manifest_json:
                        raise CatalogCorruptionError(
                            "bootstrap manifest digest does not match its bytes"
                        )
                    connection.rollback()
                    return CatalogBootstrapResult(
                        False,
                        manifest_digest,
                        current,
                        profile,
                    )

                for seed in normalized_seeds:
                    previous = definitions_by_identity.get(seed.definition.identity)
                    if previous is not None and previous.digest != seed.definition.digest:
                        raise CatalogConflictError(
                            "ToolDefinition changed under an existing ToolIdentity"
                        )

                timestamp = self._timestamp()
                is_initial = current.revision == 0 and profile.revision == 0
                new_seeds = tuple(
                    seed
                    for seed in normalized_seeds
                    if seed.definition.identity not in definitions_by_identity
                )
                changed = bool(new_seeds)
                outcome = "no_change"
                if is_initial:
                    next_revision = 1
                    for seed in normalized_seeds:
                        self._insert_definition(
                            connection,
                            seed.definition,
                            introduced_revision=next_revision,
                            timestamp=timestamp,
                        )
                        self._insert_entry(
                            connection,
                            seed.definition,
                            status=CatalogStatus.ACTIVE,
                            revision=next_revision,
                        )
                    current = self._publish_catalog_revision(
                        connection,
                        revision=next_revision,
                        parent_revision=0,
                        actor=actor,
                        reason=reason,
                        action="bootstrap_initialize",
                        change={
                            "active": [
                                seed.definition.identity.to_dict()
                                for seed in normalized_seeds
                            ]
                        },
                        timestamp=timestamp,
                    )
                    profile = self._publish_initial_profile(
                        connection,
                        seeds=normalized_seeds,
                        catalog_revision=next_revision,
                        actor=actor,
                        reason=reason,
                        timestamp=timestamp,
                    )
                    changed = True
                    outcome = "initialized"
                elif new_seeds:
                    next_revision = current.revision + 1
                    for seed in new_seeds:
                        self._insert_definition(
                            connection,
                            seed.definition,
                            introduced_revision=next_revision,
                            timestamp=timestamp,
                        )
                        self._insert_entry(
                            connection,
                            seed.definition,
                            status=CatalogStatus.DRAFT,
                            revision=next_revision,
                        )
                    current = self._publish_catalog_revision(
                        connection,
                        revision=next_revision,
                        parent_revision=next_revision - 1,
                        actor=actor,
                        reason=reason,
                        action="bootstrap_add_drafts",
                        change={
                            "drafts": [
                                seed.definition.identity.to_dict()
                                for seed in new_seeds
                            ]
                        },
                        timestamp=timestamp,
                    )
                    outcome = "drafts_added"

                connection.execute(
                    "INSERT INTO tool_catalog_bootstrap_manifests("
                    "manifest_digest, manifest_json, applied_catalog_revision, "
                    "profile_id, applied_profile_revision, outcome, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        manifest_digest,
                        manifest_json,
                        current.revision,
                        _PROFILE_ID,
                        profile.revision,
                        outcome,
                        timestamp,
                    ),
                )
                connection.commit()
                return CatalogBootstrapResult(
                    changed,
                    manifest_digest,
                    current,
                    profile,
                )
            except BaseException:
                connection.rollback()
                raise

    def register_draft(
        self,
        definition: ToolDefinition,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> CatalogRevisionSnapshot:
        """CAS-register one immutable definition without exposing it by default."""

        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        _require_non_negative_integer(expected_revision, "expected_revision")
        actor = _nonempty(actor, "actor")
        reason = _nonempty(reason, "reason")
        self.initialize()
        with self._connection(self._validated_path()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_schema(connection)
                current, _ = self._read_current_state(connection)
                if current.revision != expected_revision:
                    raise CatalogConflictError(
                        "catalog revision mismatch: "
                        f"expected {expected_revision}, current {current.revision}"
                    )
                row = connection.execute(
                    "SELECT definition_digest FROM tool_definitions WHERE tool_id=? "
                    "AND contract_version=? AND implementation_version=?",
                    _identity_values(definition.identity),
                ).fetchone()
                if row is not None:
                    if str(row["definition_digest"]) != definition.digest:
                        raise CatalogConflictError(
                            "ToolDefinition changed under an existing ToolIdentity"
                        )
                    raise CatalogConflictError(
                        f"ToolDefinition is already registered: {definition.identity!r}"
                    )
                next_revision = current.revision + 1
                timestamp = self._timestamp()
                self._insert_definition(
                    connection,
                    definition,
                    introduced_revision=next_revision,
                    timestamp=timestamp,
                )
                self._insert_entry(
                    connection,
                    definition,
                    status=CatalogStatus.DRAFT,
                    revision=next_revision,
                )
                snapshot = self._publish_catalog_revision(
                    connection,
                    revision=next_revision,
                    parent_revision=current.revision,
                    actor=actor,
                    reason=reason,
                    action="register_draft",
                    change={"drafts": [definition.identity.to_dict()]},
                    timestamp=timestamp,
                )
                connection.commit()
                return snapshot
            except BaseException:
                connection.rollback()
                raise

    def _create_empty_database(self, connection: sqlite3.Connection) -> None:
        try:
            # Several hosts may observe version zero concurrently.  Only the state
            # re-read after this write lock is authoritative.
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == CATALOG_DATABASE_SCHEMA_VERSION:
                self._validate_schema(connection)
                connection.commit()
                return
            if version > CATALOG_DATABASE_SCHEMA_VERSION:
                raise CatalogSchemaError(
                    "Tool Catalog was created by a newer application version"
                )
            if version != 0:
                raise CatalogSchemaError(
                    f"unsupported Tool Catalog schema version: {version}"
                )
            if _schema_descriptor(connection):
                raise CatalogSchemaError(
                    "refusing to adopt a non-empty version-zero Tool Catalog"
                )
            _create_schema(connection)
            timestamp = self._timestamp()
            actual_fingerprint = _schema_fingerprint(connection)
            if actual_fingerprint != _EXPECTED_SCHEMA_FINGERPRINT:
                raise CatalogSchemaError("created Tool Catalog schema is not canonical")
            empty_catalog = CatalogRevisionSnapshot(0, ())
            empty_profile = DefaultProfileSnapshot(0, 0, ())
            connection.execute(
                "INSERT INTO tool_catalog_control("
                "catalog_id, schema_fingerprint, current_revision, "
                "current_default_profile_revision, created_at, updated_at"
                ") VALUES (?, ?, 0, 0, ?, ?)",
                (_CATALOG_ID, actual_fingerprint, timestamp, timestamp),
            )
            connection.execute(
                "INSERT INTO tool_catalog_revisions("
                "revision, parent_revision, snapshot_json, snapshot_digest, "
                "actor, reason, created_at"
                ") VALUES (0, NULL, ?, ?, ?, ?, ?)",
                (
                    _canonical_json(empty_catalog.descriptor()),
                    empty_catalog.digest,
                    "system.initialize",
                    "empty Tool Catalog",
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO tool_default_profile_revisions("
                "profile_id, revision, catalog_revision, profile_json, "
                "profile_digest, actor, reason, created_at"
                ") VALUES (?, 0, 0, ?, ?, ?, ?, ?)",
                (
                    _PROFILE_ID,
                    _canonical_json(empty_profile.descriptor()),
                    empty_profile.digest,
                    "system.initialize",
                    "empty default profile",
                    timestamp,
                ),
            )
            connection.execute(
                f"PRAGMA user_version = {CATALOG_DATABASE_SCHEMA_VERSION}"
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        actual = _schema_fingerprint(connection)
        if actual != _EXPECTED_SCHEMA_FINGERPRINT:
            raise CatalogSchemaError("Tool Catalog schema fingerprint has drifted")
        rows = connection.execute(
            "SELECT * FROM tool_catalog_control"
        ).fetchall()
        if len(rows) != 1 or str(rows[0]["catalog_id"]) != _CATALOG_ID:
            raise CatalogCorruptionError(
                "Tool Catalog must contain exactly one default control row"
            )
        if str(rows[0]["schema_fingerprint"]) != actual:
            raise CatalogSchemaError(
                "Tool Catalog stored schema fingerprint does not match"
            )

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold one WAL snapshot across all integrity reads in an API call."""

        self.initialize()
        with self._connection(self._validated_path()) as connection:
            try:
                connection.execute("BEGIN DEFERRED")
                self._validate_schema(connection)
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def _write_transaction(
        self,
    ) -> Iterator[
        tuple[
            sqlite3.Connection,
            CatalogRevisionSnapshot,
            DefaultProfileSnapshot,
            str,
        ]
    ]:
        self.initialize()
        with self._connection(self._validated_path()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_schema(connection)
                catalog, profile = self._read_current_state(connection)
                yield connection, catalog, profile, self._timestamp()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _require_catalog_revision(
        current: CatalogRevisionSnapshot,
        expected_revision: int,
    ) -> None:
        if current.revision != expected_revision:
            raise CatalogConflictError(
                "catalog revision mismatch: "
                f"expected {expected_revision}, current {current.revision}"
            )

    def _read_current_state(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[CatalogRevisionSnapshot, DefaultProfileSnapshot]:
        control = connection.execute(
            "SELECT current_revision, current_default_profile_revision "
            "FROM tool_catalog_control WHERE catalog_id=?",
            (_CATALOG_ID,),
        ).fetchone()
        if control is None:
            raise CatalogCorruptionError("Tool Catalog control row is missing")
        revision = _stored_non_negative_integer(
            control["current_revision"],
            "current catalog revision",
        )
        profile_revision = _stored_non_negative_integer(
            control["current_default_profile_revision"],
            "current default profile revision",
        )
        self._validated_definitions(connection)
        catalog = self._read_catalog_revision(connection, revision)
        live_entries = self._live_entry_snapshots(connection)
        if live_entries != catalog.entries:
            raise CatalogCorruptionError(
                "current catalog entries do not match the revision snapshot"
            )
        profile = self._read_default_profile(connection, profile_revision)
        if profile.catalog_revision > catalog.revision:
            raise CatalogCorruptionError(
                "default profile points to a future catalog revision"
            )
        profile_catalog = (
            catalog
            if profile.catalog_revision == catalog.revision
            else self._read_catalog_revision(connection, profile.catalog_revision)
        )
        profile_entries = {
            entry.identity: entry for entry in profile_catalog.entries
        }
        for item in profile.items:
            entry = profile_entries.get(item.identity)
            if (
                entry is None
                or entry.definition_digest != item.definition_digest
                or entry.status is not CatalogStatus.ACTIVE
            ):
                raise CatalogCorruptionError(
                    "default profile item is not active in its catalog revision"
                )
        catalog_revisions = tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT revision FROM tool_catalog_revisions ORDER BY revision"
            ).fetchall()
        )
        if catalog_revisions != tuple(range(revision + 1)):
            raise CatalogCorruptionError("Tool Catalog revision history is not contiguous")
        profile_revisions = tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT revision FROM tool_default_profile_revisions "
                "WHERE profile_id=? ORDER BY revision",
                (_PROFILE_ID,),
            ).fetchall()
        )
        if profile_revisions != tuple(range(profile_revision + 1)):
            raise CatalogCorruptionError(
                "default profile revision history is not contiguous"
            )
        return catalog, profile

    def _read_catalog_revision(
        self,
        connection: sqlite3.Connection,
        revision: int,
    ) -> CatalogRevisionSnapshot:
        row = connection.execute(
            "SELECT snapshot_json, snapshot_digest FROM tool_catalog_revisions "
            "WHERE revision=?",
            (revision,),
        ).fetchone()
        if row is None:
            raise CatalogCorruptionError(
                f"Tool Catalog revision {revision} is missing"
            )
        value = _load_canonical_json_object(
            row["snapshot_json"],
            "Tool Catalog revision snapshot",
        )
        snapshot = _decode_catalog_revision(value)
        if snapshot.revision != revision:
            raise CatalogCorruptionError(
                "Tool Catalog snapshot revision does not match its row"
            )
        if snapshot.digest != str(row["snapshot_digest"]):
            raise CatalogCorruptionError(
                "Tool Catalog snapshot digest does not match"
            )
        return snapshot

    def _read_default_profile(
        self,
        connection: sqlite3.Connection,
        revision: int,
    ) -> DefaultProfileSnapshot:
        row = connection.execute(
            "SELECT catalog_revision, profile_json, profile_digest FROM "
            "tool_default_profile_revisions WHERE profile_id=? AND revision=?",
            (_PROFILE_ID, revision),
        ).fetchone()
        if row is None:
            raise CatalogCorruptionError(
                f"default profile revision {revision} is missing"
            )
        value = _load_canonical_json_object(
            row["profile_json"],
            "default Tool profile",
        )
        profile = _decode_default_profile(value)
        if (
            profile.revision != revision
            or profile.catalog_revision != int(row["catalog_revision"])
        ):
            raise CatalogCorruptionError(
                "default profile identity does not match its row"
            )
        if profile.digest != str(row["profile_digest"]):
            raise CatalogCorruptionError("default profile digest does not match")
        item_rows = connection.execute(
            "SELECT * FROM tool_default_profile_items WHERE profile_id=? "
            "AND profile_revision=? ORDER BY ordinal",
            (_PROFILE_ID, revision),
        ).fetchall()
        projected = tuple(_profile_item_from_row(item) for item in item_rows)
        if projected != profile.items:
            raise CatalogCorruptionError(
                "default profile items do not match its revision descriptor"
            )
        return profile

    @staticmethod
    def _validated_definitions(
        connection: sqlite3.Connection,
    ) -> dict[ToolIdentity, ToolDefinition]:
        rows = connection.execute(
            "SELECT * FROM tool_definitions ORDER BY tool_id, contract_version, "
            "implementation_version"
        ).fetchall()
        definitions = tuple(_definition_from_row(row) for row in rows)
        return {definition.identity: definition for definition in definitions}

    @staticmethod
    def _live_entry_snapshots(
        connection: sqlite3.Connection,
    ) -> tuple[CatalogEntrySnapshot, ...]:
        rows = connection.execute(
            "SELECT tool_id, contract_version, implementation_version, "
            "definition_digest, status FROM tool_catalog_entries ORDER BY tool_id, "
            "contract_version, implementation_version"
        ).fetchall()
        return tuple(_entry_snapshot_from_row(row) for row in rows)

    @staticmethod
    def _insert_definition(
        connection: sqlite3.Connection,
        definition: ToolDefinition,
        *,
        introduced_revision: int,
        timestamp: str,
    ) -> None:
        descriptor_json = _canonical_json(definition.descriptor())
        connection.execute(
            "INSERT INTO tool_definitions("
            "tool_id, contract_version, implementation_version, definition_digest, "
            "descriptor_json, implementation_ref, implementation_digest, "
            "introduced_revision, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                definition.identity.tool_id,
                definition.identity.contract_version,
                definition.identity.implementation_version,
                definition.digest,
                descriptor_json,
                definition.implementation_ref,
                definition.implementation_digest,
                introduced_revision,
                timestamp,
            ),
        )

    @staticmethod
    def _insert_entry(
        connection: sqlite3.Connection,
        definition: ToolDefinition,
        *,
        status: CatalogStatus,
        revision: int,
    ) -> None:
        connection.execute(
            "INSERT INTO tool_catalog_entries("
            "tool_id, contract_version, implementation_version, definition_digest, "
            "status, introduced_revision, updated_revision"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                definition.identity.tool_id,
                definition.identity.contract_version,
                definition.identity.implementation_version,
                definition.digest,
                status.value,
                revision,
                revision,
            ),
        )

    def _publish_catalog_revision(
        self,
        connection: sqlite3.Connection,
        *,
        revision: int,
        parent_revision: int,
        actor: str,
        reason: str,
        action: str,
        change: Mapping[str, Any],
        timestamp: str,
    ) -> CatalogRevisionSnapshot:
        snapshot = CatalogRevisionSnapshot(
            revision,
            self._live_entry_snapshots(connection),
        )
        connection.execute(
            "INSERT INTO tool_catalog_revisions("
            "revision, parent_revision, snapshot_json, snapshot_digest, actor, "
            "reason, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                revision,
                parent_revision,
                _canonical_json(snapshot.descriptor()),
                snapshot.digest,
                actor,
                reason,
                timestamp,
            ),
        )
        self._insert_audit_event(
            connection,
            event_id=f"catalog-revision:{revision}",
            catalog_revision=revision,
            action=action,
            actor=actor,
            reason=reason,
            change=change,
            timestamp=timestamp,
        )
        connection.execute(
            "UPDATE tool_catalog_control SET current_revision=?, updated_at=? "
            "WHERE catalog_id=?",
            (revision, timestamp, _CATALOG_ID),
        )
        return snapshot

    def _publish_initial_profile(
        self,
        connection: sqlite3.Connection,
        *,
        seeds: tuple[ToolCatalogSeed, ...],
        catalog_revision: int,
        actor: str,
        reason: str,
        timestamp: str,
    ) -> DefaultProfileSnapshot:
        items = tuple(
            DefaultProfileItem(
                ordinal,
                seed.definition.identity,
                seed.definition.digest,
                seed.availability,
            )
            for ordinal, seed in enumerate(seeds)
        )
        profile = DefaultProfileSnapshot(1, catalog_revision, items)
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
        return profile

    @staticmethod
    def _insert_profile_revision(
        connection: sqlite3.Connection,
        *,
        profile: DefaultProfileSnapshot,
        actor: str,
        reason: str,
        timestamp: str,
    ) -> None:
        connection.execute(
            "INSERT INTO tool_default_profile_revisions("
            "profile_id, revision, catalog_revision, profile_json, profile_digest, "
            "actor, reason, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _PROFILE_ID,
                profile.revision,
                profile.catalog_revision,
                _canonical_json(profile.descriptor()),
                profile.digest,
                actor,
                reason,
                timestamp,
            ),
        )
        for item in profile.items:
            connection.execute(
                "INSERT INTO tool_default_profile_items("
                "profile_id, profile_revision, ordinal, tool_id, contract_version, "
                "implementation_version, definition_digest, availability"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _PROFILE_ID,
                    profile.revision,
                    item.ordinal,
                    item.identity.tool_id,
                    item.identity.contract_version,
                    item.identity.implementation_version,
                    item.definition_digest,
                    item.availability.value,
                ),
            )

    @staticmethod
    def _insert_audit_event(
        connection: sqlite3.Connection,
        *,
        event_id: str,
        catalog_revision: int,
        action: str,
        actor: str,
        reason: str,
        change: Mapping[str, Any],
        timestamp: str,
    ) -> None:
        change_json = _canonical_json(dict(change))
        connection.execute(
            "INSERT INTO tool_catalog_audit_events("
            "event_id, catalog_revision, action, actor, reason, change_json, "
            "change_digest, occurred_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                catalog_revision,
                action,
                actor,
                reason,
                change_json,
                _sha256_text(change_json),
                timestamp,
            ),
        )

    def _timestamp(self) -> str:
        return self._now().isoformat()

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("catalog clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _validated_path(self) -> Path:
        return _resolve_database_path(self._database_path)

    @staticmethod
    @contextmanager
    def _connection(path: Path) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()


__all__ = [
    "BOOTSTRAP_MANIFEST_SCHEMA",
    "CATALOG_DATABASE_SCHEMA_VERSION",
    "CATALOG_REVISION_SCHEMA",
    "DEFAULT_PROFILE_SCHEMA",
    "DEFAULT_TOOL_CATALOG_DATABASE_PATH",
    "CatalogAuditEvent",
    "CatalogBootstrapResult",
    "CatalogCorruptionError",
    "CatalogEntrySnapshot",
    "CatalogPersistenceError",
    "CatalogRevisionSnapshot",
    "CatalogSchemaError",
    "DefaultProfileAvailability",
    "DefaultCatalogResolutionState",
    "DefaultProfileItem",
    "DefaultProfileSelection",
    "DefaultProfileSnapshot",
    "EmergencyRevocation",
    "EmergencyRevocationSelector",
    "EmergencyRevocationTarget",
    "ToolCatalogRepository",
    "ToolCatalogSeed",
]
