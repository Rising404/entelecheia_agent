"""Frozen SQLite schema and structural fingerprint for the Tool Catalog."""

from __future__ import annotations

import hashlib
import json
import sqlite3


SCHEMA_SQL = """
CREATE TABLE tool_catalog_control (
    catalog_id TEXT PRIMARY KEY CHECK(catalog_id = 'default'),
    schema_fingerprint TEXT NOT NULL CHECK(length(schema_fingerprint) = 64),
    current_revision INTEGER NOT NULL CHECK(current_revision >= 0),
    current_default_profile_revision INTEGER NOT NULL CHECK(current_default_profile_revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE tool_catalog_revisions (
    revision INTEGER PRIMARY KEY CHECK(revision >= 0),
    parent_revision INTEGER REFERENCES tool_catalog_revisions(revision),
    snapshot_json TEXT NOT NULL,
    snapshot_digest TEXT NOT NULL CHECK(length(snapshot_digest) = 64),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK((revision = 0 AND parent_revision IS NULL) OR (revision > 0 AND parent_revision = revision - 1))
);

CREATE TABLE tool_definitions (
    tool_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    definition_digest TEXT NOT NULL CHECK(length(definition_digest) = 64),
    descriptor_json TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    implementation_digest TEXT NOT NULL CHECK(length(implementation_digest) = 64),
    introduced_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision)
        DEFERRABLE INITIALLY DEFERRED,
    created_at TEXT NOT NULL,
    PRIMARY KEY(tool_id, contract_version, implementation_version),
    UNIQUE(tool_id, contract_version, implementation_version, definition_digest)
) WITHOUT ROWID;

CREATE TABLE tool_catalog_entries (
    tool_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    definition_digest TEXT NOT NULL CHECK(length(definition_digest) = 64),
    status TEXT NOT NULL CHECK(status IN ('draft', 'active', 'deprecated', 'disabled', 'retired')),
    introduced_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision)
        DEFERRABLE INITIALLY DEFERRED,
    updated_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision)
        DEFERRABLE INITIALLY DEFERRED,
    PRIMARY KEY(tool_id, contract_version, implementation_version),
    FOREIGN KEY(tool_id, contract_version, implementation_version, definition_digest)
        REFERENCES tool_definitions(tool_id, contract_version, implementation_version, definition_digest)
) WITHOUT ROWID;

CREATE TABLE tool_default_profile_revisions (
    profile_id TEXT NOT NULL CHECK(profile_id = 'default'),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    catalog_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision)
        DEFERRABLE INITIALLY DEFERRED,
    profile_json TEXT NOT NULL,
    profile_digest TEXT NOT NULL CHECK(length(profile_digest) = 64),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(profile_id, revision)
) WITHOUT ROWID;

CREATE TABLE tool_default_profile_items (
    profile_id TEXT NOT NULL,
    profile_revision INTEGER NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    tool_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    definition_digest TEXT NOT NULL CHECK(length(definition_digest) = 64),
    availability TEXT NOT NULL CHECK(availability IN ('required', 'if_available')),
    PRIMARY KEY(profile_id, profile_revision, ordinal),
    UNIQUE(profile_id, profile_revision, tool_id, contract_version, implementation_version),
    FOREIGN KEY(profile_id, profile_revision)
        REFERENCES tool_default_profile_revisions(profile_id, revision),
    FOREIGN KEY(tool_id, contract_version, implementation_version, definition_digest)
        REFERENCES tool_definitions(tool_id, contract_version, implementation_version, definition_digest)
) WITHOUT ROWID;

CREATE TABLE tool_catalog_tombstones (
    tool_id TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    definition_digest TEXT NOT NULL CHECK(length(definition_digest) = 64),
    retired_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision),
    retired_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(tool_id, contract_version, implementation_version),
    FOREIGN KEY(tool_id, contract_version, implementation_version, definition_digest)
        REFERENCES tool_definitions(tool_id, contract_version, implementation_version, definition_digest)
) WITHOUT ROWID;

CREATE TABLE tool_emergency_revocations (
    revocation_id TEXT PRIMARY KEY,
    selector_json TEXT NOT NULL,
    selector_digest TEXT NOT NULL CHECK(length(selector_digest) = 64),
    disposition TEXT NOT NULL CHECK(disposition = 'deny_execution'),
    reason TEXT NOT NULL,
    issued_by TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    expires_at TEXT,
    cleared_at TEXT,
    cleared_by TEXT,
    clear_reason TEXT,
    created_at TEXT NOT NULL,
    CHECK((cleared_at IS NULL AND cleared_by IS NULL AND clear_reason IS NULL) OR
          (cleared_at IS NOT NULL AND cleared_by IS NOT NULL AND clear_reason IS NOT NULL))
);

CREATE TABLE tool_catalog_audit_events (
    event_id TEXT PRIMARY KEY,
    catalog_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision),
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    change_json TEXT NOT NULL,
    change_digest TEXT NOT NULL CHECK(length(change_digest) = 64),
    occurred_at TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE tool_catalog_bootstrap_manifests (
    manifest_digest TEXT PRIMARY KEY CHECK(length(manifest_digest) = 64),
    manifest_json TEXT NOT NULL,
    applied_catalog_revision INTEGER NOT NULL REFERENCES tool_catalog_revisions(revision),
    profile_id TEXT NOT NULL CHECK(profile_id = 'default'),
    applied_profile_revision INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('initialized', 'drafts_added', 'no_change')),
    created_at TEXT NOT NULL,
    FOREIGN KEY(profile_id, applied_profile_revision)
        REFERENCES tool_default_profile_revisions(profile_id, revision)
) WITHOUT ROWID;

CREATE INDEX tool_catalog_entries_status_idx
    ON tool_catalog_entries(status, tool_id, contract_version, implementation_version);
CREATE INDEX tool_catalog_audit_revision_idx
    ON tool_catalog_audit_events(catalog_revision, occurred_at);
CREATE INDEX tool_emergency_revocations_active_idx
    ON tool_emergency_revocations(effective_at, expires_at, cleared_at);
"""


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the frozen schema without implicitly ending the caller's transaction."""

    pending = ""
    for line in SCHEMA_SQL.splitlines(keepends=True):
        pending += line
        if not sqlite3.complete_statement(pending):
            continue
        statement = pending.strip()
        if statement:
            connection.execute(statement)
        pending = ""
    if pending.strip():
        raise RuntimeError("Tool Catalog schema contains an incomplete SQL statement")


def schema_descriptor(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str], ...]:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    return tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        for row in rows
    )


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    encoded = json.dumps(
        schema_descriptor(connection),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_schema_fingerprint() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        create_schema(connection)
        return schema_fingerprint(connection)
    finally:
        connection.close()


EXPECTED_SCHEMA_FINGERPRINT = _expected_schema_fingerprint()


__all__ = [
    "EXPECTED_SCHEMA_FINGERPRINT",
    "SCHEMA_SQL",
    "create_schema",
    "schema_descriptor",
    "schema_fingerprint",
]
