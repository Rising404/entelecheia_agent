"""项目图片身份与观察记录的唯一 DDL 所有者。

本模块只在调用方提供的 ``documents.sqlite`` 连接上安装/验证图片子模式。它不
打开数据库、不开始或提交事务，也不依赖文档摄取、工具、Runtime、Session 或检索层。
"""

from __future__ import annotations

import sqlite3


class PictureSchemaError(RuntimeError):
    """数据库不符合图片子模式契约。"""


class PictureSchemaTransactionRequired(PictureSchemaError):
    """图片 DDL 安装没有处于调用方所有的事务中。"""


_TABLE_COLUMNS = {
    "pictures": {
        "picture_id",
        "file_id",
        "file_version_id",
        "source_kind",
        "source_locator_json",
        "source_locator_sha256",
        "source_content_sha256",
        "source_media_type",
        "created_at",
    },
    "picture_units": {
        "picture_unit_id",
        "picture_id",
        "unit_kind",
        "unit_locator_json",
        "unit_locator_sha256",
        "producer_fingerprint",
        "parent_picture_unit_id",
        "pixel_sha256",
        "media_type",
        "width",
        "height",
        "created_at",
    },
    "picture_observations": {
        "sequence",
        "observation_id",
        "picture_id",
        "picture_unit_id",
        "logical_invocation_id",
        "request_ordinal",
        "modality",
        "purpose",
        "kind",
        "text",
        "structured_payload_json",
        "uncertainty",
        "processor_fingerprint",
        "prompt_fingerprint",
        "request_sha256",
        "output_sha256",
        "payload_sha256",
        "created_at",
        "question",
    },
}

PICTURE_REQUIRED_TABLES = frozenset(_TABLE_COLUMNS)

_REQUIRED_INDEXES = {
    "idx_pictures_exact_source",
    "uq_picture_units_root_natural",
    "uq_picture_units_child_natural",
    "idx_picture_observations_fifo",
}

_REQUIRED_TRIGGERS = {
    "trg_pictures_locator_kind_insert",
    "trg_pictures_file_source_hash_insert",
    "trg_pictures_immutable_update",
    "trg_pictures_immutable_delete",
    "trg_picture_units_locator_kind_insert",
    "trg_picture_units_immutable_update",
    "trg_picture_units_immutable_delete",
    "trg_picture_observations_immutable_update",
    "trg_picture_observations_immutable_delete",
}

PICTURE_REQUIRED_OBJECTS = (
    PICTURE_REQUIRED_TABLES | _REQUIRED_INDEXES | _REQUIRED_TRIGGERS
)

_RETIRED_OBJECTS = frozenset({"picture_observation_units"})

_QUESTION_COLUMN_SQL = """
question TEXT CHECK(
    (purpose = 'question'
        AND modality = 'vlm'
        AND question IS NOT NULL
        AND length(trim(question)) > 0
        AND length(question) <= 4000)
    OR (purpose <> 'question' AND question IS NULL)
)
""".strip()


_DDL = (
    """
    CREATE TABLE pictures (
        picture_id TEXT PRIMARY KEY CHECK(length(trim(picture_id)) > 0),
        file_id TEXT NOT NULL CHECK(length(trim(file_id)) > 0),
        file_version_id TEXT NOT NULL CHECK(length(trim(file_version_id)) > 0),
        source_kind TEXT NOT NULL CHECK(source_kind IN (
            'whole_file', 'embedded_asset', 'document_surface'
        )),
        source_locator_json TEXT NOT NULL CHECK(json_valid(source_locator_json)),
        source_locator_sha256 TEXT NOT NULL CHECK(
            length(source_locator_sha256) = 64
            AND source_locator_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        source_content_sha256 TEXT NOT NULL CHECK(
            length(source_content_sha256) = 64
            AND source_content_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        source_media_type TEXT NOT NULL CHECK(
            source_media_type = lower(trim(source_media_type))
            AND instr(source_media_type, '/') > 1
            AND instr(substr(
                source_media_type,
                instr(source_media_type, '/') + 1
            ), '/') = 0
            AND substr(source_media_type, -1) <> '/'
            AND instr(source_media_type, ';') = 0
        ),
        created_at TEXT NOT NULL,
        UNIQUE(file_version_id, source_locator_json),
        FOREIGN KEY(file_version_id, file_id)
            REFERENCES file_versions(id, file_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE picture_units (
        picture_unit_id TEXT PRIMARY KEY CHECK(length(trim(picture_unit_id)) > 0),
        picture_id TEXT NOT NULL CHECK(length(trim(picture_id)) > 0),
        unit_kind TEXT NOT NULL CHECK(unit_kind IN ('full', 'render', 'crop', 'tile')),
        unit_locator_json TEXT NOT NULL CHECK(json_valid(unit_locator_json)),
        unit_locator_sha256 TEXT NOT NULL CHECK(
            length(unit_locator_sha256) = 64
            AND unit_locator_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        producer_fingerprint TEXT NOT NULL
            CHECK(
                length(trim(producer_fingerprint)) > 0
                AND length(producer_fingerprint) <= 512
            ),
        parent_picture_unit_id TEXT,
        pixel_sha256 TEXT NOT NULL CHECK(
            length(pixel_sha256) = 64
            AND pixel_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        media_type TEXT NOT NULL CHECK(
            media_type = lower(trim(media_type))
            AND media_type LIKE 'image/%'
            AND instr(media_type, ';') = 0
        ),
        width INTEGER NOT NULL CHECK(width > 0),
        height INTEGER NOT NULL CHECK(height > 0),
        created_at TEXT NOT NULL,
        CHECK(
            (unit_kind IN ('full', 'render') AND parent_picture_unit_id IS NULL)
            OR (unit_kind IN ('crop', 'tile') AND parent_picture_unit_id IS NOT NULL)
        ),
        UNIQUE(picture_unit_id, picture_id),
        FOREIGN KEY(picture_id) REFERENCES pictures(picture_id) ON DELETE RESTRICT,
        FOREIGN KEY(parent_picture_unit_id, picture_id)
            REFERENCES picture_units(picture_unit_id, picture_id) ON DELETE RESTRICT
    )
    """,
    f"""
    CREATE TABLE picture_observations (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_id TEXT NOT NULL UNIQUE CHECK(length(trim(observation_id)) > 0),
        picture_id TEXT NOT NULL CHECK(length(trim(picture_id)) > 0),
        picture_unit_id TEXT NOT NULL CHECK(length(trim(picture_unit_id)) > 0),
        logical_invocation_id TEXT NOT NULL
            CHECK(length(trim(logical_invocation_id)) > 0),
        request_ordinal INTEGER NOT NULL CHECK(request_ordinal >= 0),
        modality TEXT NOT NULL CHECK(modality IN ('ocr', 'vlm')),
        purpose TEXT NOT NULL CHECK(length(trim(purpose)) > 0),
        kind TEXT NOT NULL CHECK(length(trim(kind)) > 0),
        text TEXT NOT NULL,
        structured_payload_json TEXT CHECK(
            structured_payload_json IS NULL
            OR (
                length(CAST(structured_payload_json AS BLOB)) <= 65536
                AND json_valid(structured_payload_json)
                AND json_type(structured_payload_json) IS 'object'
                AND json_type(structured_payload_json, '$.contract') IS 'text'
                AND length(trim(
                    coalesce(
                        json_extract(structured_payload_json, '$.contract'),
                        ''
                    )
                )) > 0
                AND json_type(structured_payload_json, '$.payload') IS 'object'
            )
        ),
        uncertainty REAL CHECK(
            uncertainty IS NULL OR (uncertainty >= 0.0 AND uncertainty <= 1.0)
        ),
        processor_fingerprint TEXT NOT NULL
            CHECK(length(trim(processor_fingerprint)) > 0),
        prompt_fingerprint TEXT CHECK(
            prompt_fingerprint IS NULL OR length(trim(prompt_fingerprint)) > 0
        ),
        request_sha256 TEXT NOT NULL CHECK(
            length(request_sha256) = 64
            AND request_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        output_sha256 TEXT NOT NULL CHECK(
            length(output_sha256) = 64
            AND output_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        payload_sha256 TEXT NOT NULL CHECK(
            length(payload_sha256) = 64
            AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        created_at TEXT NOT NULL,
        {_QUESTION_COLUMN_SQL},
        UNIQUE(picture_id, logical_invocation_id, request_ordinal),
        FOREIGN KEY(picture_id) REFERENCES pictures(picture_id) ON DELETE RESTRICT,
        FOREIGN KEY(picture_unit_id, picture_id)
            REFERENCES picture_units(picture_unit_id, picture_id)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TRIGGER trg_pictures_locator_kind_insert
    BEFORE INSERT ON pictures
    WHEN json_extract(NEW.source_locator_json, '$.kind') IS NOT NEW.source_kind
      OR json_type(NEW.source_locator_json, '$.payload') IS NOT 'object'
    BEGIN
        SELECT RAISE(ABORT, 'picture source locator does not match source_kind');
    END
    """,
    """
    CREATE TRIGGER trg_pictures_file_source_hash_insert
    BEFORE INSERT ON pictures
    WHEN NEW.source_kind IN ('whole_file', 'document_surface')
      AND NOT EXISTS (
          SELECT 1
          FROM file_versions AS version
          WHERE version.id = NEW.file_version_id
            AND version.file_id = NEW.file_id
            AND version.content_sha256 = NEW.source_content_sha256
      )
    BEGIN
        SELECT RAISE(ABORT, 'file-backed picture hash does not match file version');
    END
    """,
    """
    CREATE TRIGGER trg_pictures_immutable_update
    BEFORE UPDATE ON pictures
    BEGIN
        SELECT RAISE(ABORT, 'picture records are immutable');
    END
    """,
    """
    CREATE TRIGGER trg_pictures_immutable_delete
    BEFORE DELETE ON pictures
    BEGIN
        SELECT RAISE(ABORT, 'picture records are immutable');
    END
    """,
    """
    CREATE TRIGGER trg_picture_units_locator_kind_insert
    BEFORE INSERT ON picture_units
    WHEN json_extract(NEW.unit_locator_json, '$.kind') IS NOT NEW.unit_kind
      OR json_type(NEW.unit_locator_json, '$.payload') IS NOT 'object'
    BEGIN
        SELECT RAISE(ABORT, 'picture unit locator does not match unit_kind');
    END
    """,
    """
    CREATE TRIGGER trg_picture_units_immutable_update
    BEFORE UPDATE ON picture_units
    BEGIN
        SELECT RAISE(ABORT, 'picture unit records are immutable');
    END
    """,
    """
    CREATE TRIGGER trg_picture_units_immutable_delete
    BEFORE DELETE ON picture_units
    BEGIN
        SELECT RAISE(ABORT, 'picture unit records are immutable');
    END
    """,
    """
    CREATE TRIGGER trg_picture_observations_immutable_update
    BEFORE UPDATE ON picture_observations
    BEGIN
        SELECT RAISE(ABORT, 'picture observations are immutable');
    END
    """,
    """
    CREATE TRIGGER trg_picture_observations_immutable_delete
    BEFORE DELETE ON picture_observations
    BEGIN
        SELECT RAISE(ABORT, 'picture observations are immutable');
    END
    """,
    """
    CREATE INDEX idx_pictures_exact_source
    ON pictures(file_version_id, source_locator_sha256, source_locator_json)
    """,
    """
    CREATE UNIQUE INDEX uq_picture_units_root_natural
    ON picture_units(picture_id, unit_locator_json, producer_fingerprint)
    WHERE parent_picture_unit_id IS NULL
    """,
    """
    CREATE UNIQUE INDEX uq_picture_units_child_natural
    ON picture_units(
        picture_id, parent_picture_unit_id, unit_locator_json, producer_fingerprint
    )
    WHERE parent_picture_unit_id IS NOT NULL
    """,
    """
    CREATE INDEX idx_picture_observations_fifo
    ON picture_observations(picture_id, sequence DESC)
    """,
)


def _normalize_owned_sql(value: str) -> str:
    return " ".join(value.split())


def _owned_object_name(statement: str) -> str:
    tokens = _normalize_owned_sql(statement).split(" ")
    name_index = 3 if tokens[1].upper() == "UNIQUE" else 2
    return tokens[name_index]


_OWNED_OBJECT_SQL = {
    _owned_object_name(statement): _normalize_owned_sql(statement)
    for statement in _DDL
}
if frozenset(_OWNED_OBJECT_SQL) != PICTURE_REQUIRED_OBJECTS:
    raise RuntimeError("picture DDL inventory does not match required objects")

_LEGACY_OBSERVATION_COLUMNS = _TABLE_COLUMNS["picture_observations"] - {"question"}
_LEGACY_OBSERVATION_SQL = _OWNED_OBJECT_SQL["picture_observations"].replace(
    ", " + _normalize_owned_sql(_QUESTION_COLUMN_SQL),
    "",
)


def initialize_picture_schema(conn: sqlite3.Connection) -> None:
    """在调用方事务内安装或验证图片子模式；绝不提交或回滚。"""

    _require_connection(conn)
    _require_foreign_keys(conn)
    if not conn.in_transaction:
        raise PictureSchemaTransactionRequired(
            "initialize_picture_schema requires a caller-owned transaction"
        )
    _validate_file_ledger_prerequisites(conn)

    existing = _application_objects(conn)
    owned_present = existing & PICTURE_REQUIRED_OBJECTS
    if not owned_present:
        for statement in _DDL:
            conn.execute(statement)
    elif owned_present != PICTURE_REQUIRED_OBJECTS:
        raise PictureSchemaError(
            "refusing to adopt a partial picture schema contribution"
        )
    validate_picture_schema(conn)


def migrate_picture_observation_question_schema(conn: sqlite3.Connection) -> None:
    """Add the optional question column without rewriting legacy observations."""

    _require_connection(conn)
    _require_foreign_keys(conn)
    if not conn.in_transaction:
        raise PictureSchemaTransactionRequired(
            "picture question migration requires a caller-owned transaction"
        )
    _validate_file_ledger_prerequisites(conn)
    existing = _application_objects(conn)
    if (existing & PICTURE_REQUIRED_OBJECTS) != PICTURE_REQUIRED_OBJECTS:
        raise PictureSchemaError(
            "picture question migration requires the complete picture contribution"
        )
    columns = _table_columns(conn, "picture_observations")
    if columns == _TABLE_COLUMNS["picture_observations"]:
        validate_picture_schema(conn)
        return
    if columns != _LEGACY_OBSERVATION_COLUMNS:
        raise PictureSchemaError(
            "picture question migration found unexpected observation columns"
        )
    for object_name, expected_sql in _OWNED_OBJECT_SQL.items():
        row = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE name=?",
            (object_name,),
        ).fetchone()
        actual_sql = None if row is None or row[0] is None else _normalize_owned_sql(
            str(row[0])
        )
        migration_expected = (
            _LEGACY_OBSERVATION_SQL
            if object_name == "picture_observations"
            else expected_sql
        )
        if actual_sql != migration_expected:
            raise PictureSchemaError(
                f"picture question migration found unexpected SQL for {object_name}"
            )
    conn.execute(
        "ALTER TABLE picture_observations ADD COLUMN " + _QUESTION_COLUMN_SQL
    )
    validate_picture_schema(conn)


def validate_picture_schema(conn: sqlite3.Connection) -> None:
    """只读校验图片子模式、基础文件账本与版本身份。"""

    _require_connection(conn)
    _require_foreign_keys(conn)
    _validate_file_ledger_prerequisites(conn)
    existing = _application_objects(conn)
    retired = sorted(_RETIRED_OBJECTS & existing)
    if retired:
        raise PictureSchemaError(
            "picture schema contains retired objects: " + ", ".join(retired)
        )
    missing = sorted(PICTURE_REQUIRED_OBJECTS - existing)
    if missing:
        raise PictureSchemaError(
            "picture schema is incomplete: " + ", ".join(missing)
        )

    for table, expected_columns in _TABLE_COLUMNS.items():
        columns = _table_columns(conn, table)
        if columns != expected_columns:
            raise PictureSchemaError(
                f"picture schema table {table} has unexpected columns"
            )

    for object_name, expected_sql in _OWNED_OBJECT_SQL.items():
        row = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE name=?",
            (object_name,),
        ).fetchone()
        if row is None or row[0] is None:
            raise PictureSchemaError(
                f"picture schema object {object_name} has no canonical SQL"
            )
        if _normalize_owned_sql(str(row[0])) != expected_sql:
            raise PictureSchemaError(
                f"picture schema object {object_name} has unexpected SQL"
            )


def _validate_file_ledger_prerequisites(conn: sqlite3.Connection) -> None:
    for table, required in {
        "files": {"id", "current_version_id"},
        "file_versions": {"id", "file_id", "content_sha256"},
    }.items():
        columns = _table_columns(conn, table)
        missing = required - columns
        if missing:
            raise PictureSchemaError(
                f"picture schema requires {table} columns: {', '.join(sorted(missing))}"
            )
    if not _has_exact_unique_index(conn, "file_versions", ("id", "file_id")):
        raise PictureSchemaError(
            "picture schema requires UNIQUE file_versions(id, file_id)"
        )


def _has_exact_unique_index(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    for row in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
        if not bool(row[2]) or bool(row[4]):
            continue
        index_name = str(row[1]).replace('"', '""')
        indexed = tuple(
            str(index_row[2])
            for index_row in conn.execute(
                f'PRAGMA index_info("{index_name}")'
            ).fetchall()
        )
        if indexed == columns:
            return True
    return False


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    quoted = table.replace('"', '""')
    return {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{quoted}")').fetchall()
    }


def _application_objects(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'index', 'trigger', 'view')"
        ).fetchall()
    )


def _require_connection(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("picture schema requires an explicit sqlite3.Connection")


def _require_foreign_keys(conn: sqlite3.Connection) -> None:
    enabled = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if enabled != 1:
        raise PictureSchemaError("picture schema requires PRAGMA foreign_keys=ON")
