"""创建并校验唯一受支持的 ``sessions.sqlite`` schema。"""

from __future__ import annotations

import hashlib
import re
import sqlite3

from .current_schema import CURRENT_SCHEMA_SQL


SCHEMA_VERSION = 104
CURRENT_SCHEMA_FINGERPRINT = (
    "388e6653e0392dbf3cc73eb94d33b0673b6b147ea02c0b61f290713fbd059df9"
)
_CANDIDATE_SCHEMA_FINGERPRINT = (
    "a26fd03c5cbba96c47da6dd95c93eb082204ba54aa127f54b85b568c881a0445"
)

_QUOTED_IDENTIFIER = re.compile(r'["`]([A-Za-z_][A-Za-z0-9_]*)["`]')


class UnsupportedSchemaVersion(RuntimeError):
    """数据库不是当前构建精确支持的 schema。"""


class SchemaDriftError(RuntimeError):
    """名义上为当前版本的数据库与当前 schema 不匹配。"""


def initialize_schema(conn: sqlite3.Connection) -> None:
    """创建空数据库，或校验现有当前数据库。

    仅支持已确认的 103→104 candidate 台账退役；其他历史版本不受支持。
    """

    version = schema_version(conn)
    if version == SCHEMA_VERSION:
        _assert_baseline_schema(conn)
        return
    if version == 103:
        _retire_candidate_ledger(conn)
        return

    if version != 0 or _has_user_schema_objects(conn):
        rendered = "unversioned" if version == 0 else f"version {version}"
        raise UnsupportedSchemaVersion(
            f"Unsupported sessions.sqlite schema {rendered}; "
            f"this build accepts only an empty database or version {SCHEMA_VERSION}"
        )

    _create_current_schema(conn)


def _retire_candidate_ledger(conn: sqlite3.Connection) -> None:
    """Remove only the obsolete selector ledger, preserving Session authority."""

    if _schema_fingerprint(conn) != _CANDIDATE_SCHEMA_FINGERPRINT:
        raise SchemaDriftError("sessions.sqlite version 103 schema differs from its baseline")
    _assert_foreign_key_integrity(conn)
    savepoint = "personagraph_retire_file_candidates"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        conn.execute("DROP TABLE runtime_turn_file_candidate_bindings")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        _assert_baseline_schema(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except BaseException:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _create_current_schema(conn: sqlite3.Connection) -> None:
    savepoint = "personagraph_current_schema"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for statement in _sql_statements(CURRENT_SCHEMA_SQL):
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        _assert_baseline_schema(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except BaseException:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def _assert_baseline_schema(conn: sqlite3.Connection) -> None:
    actual = _schema_fingerprint(conn)
    if actual != CURRENT_SCHEMA_FINGERPRINT:
        raise SchemaDriftError(
            "sessions.sqlite declares the current version but its schema differs "
            f"from this build (expected {CURRENT_SCHEMA_FINGERPRINT}, got {actual})"
        )
    _assert_foreign_key_integrity(conn)


def _assert_foreign_key_integrity(conn: sqlite3.Connection) -> None:
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise SchemaDriftError(
            f"sessions.sqlite foreign-key integrity check failed: {violations[:5]!r}"
        )


def _schema_fingerprint(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    payload = "\n".join(
        "\x1f".join(
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                _normalize_ddl(row[3]),
            )
        )
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_ddl(sql: object) -> str:
    normalized = " ".join(str(sql or "").split())
    return _QUOTED_IDENTIFIER.sub(lambda match: match.group(1), normalized)


def _has_user_schema_objects(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        is not None
    )


def _sql_statements(script: str) -> tuple[str, ...]:
    """拆分受信基线 DDL，同时保留保存点所有权。"""

    statements: list[str] = []
    pending: list[str] = []
    for line in script.splitlines():
        pending.append(line)
        candidate = "\n".join(pending).strip()
        if candidate and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            pending = []
    if any(line.strip() for line in pending):
        raise RuntimeError("Incomplete SQL statement in current schema baseline")
    return tuple(statements)
