"""图片身份与图片单元的 SQLite primitives/query。

本模块只执行显式连接上的持久化动作。它不生成身份或时间，不判断自然键重放是否与
业务载荷一致，也不决定父子准入；这些用例职责属于 ``workspace.pictures.admission``。
"""

from __future__ import annotations

import sqlite3

from ..contracts import (
    PictureRecord,
    PictureSourceKind,
    PictureSourceLocator,
    PictureUnitKind,
    PictureUnitLocator,
    PictureUnitRecord,
)


class PictureRepositoryError(RuntimeError):
    """图片 SQL 持久化无法满足请求。"""


class PicturePersistenceConflict(PictureRepositoryError):
    """图片 SQL 写入违反持久化约束。"""


class PictureTransactionRequired(PictureRepositoryError):
    """图片写入没有处于调用方所有的事务中。"""


class PictureForeignKeysRequired(PictureRepositoryError):
    """图片写入连接没有启用 SQLite 外键约束。"""


def require_picture_write_transaction(conn: sqlite3.Connection) -> None:
    """验证底层写入所需的显式连接、外键与调用方事务。"""

    _require_connection(conn)
    foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys != 1:
        raise PictureForeignKeysRequired(
            "picture mutation requires PRAGMA foreign_keys=ON"
        )
    if not conn.in_transaction:
        raise PictureTransactionRequired(
            "picture mutation requires a caller-owned transaction"
        )


def insert_picture_if_absent_in_transaction(
    conn: sqlite3.Connection,
    picture: PictureRecord,
) -> bool:
    """尝试插入完整记录；自然键已存在时返回 ``False``。"""

    require_picture_write_transaction(conn)
    if not isinstance(picture, PictureRecord):
        raise TypeError("picture must be PictureRecord")
    try:
        cursor = conn.execute(
            "INSERT INTO pictures "
            "(picture_id, file_id, file_version_id, source_kind, "
            "source_locator_json, source_locator_sha256, source_content_sha256, "
            "source_media_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(file_version_id, source_locator_json) DO NOTHING",
            (
                picture.picture_id,
                picture.file_id,
                picture.file_version_id,
                picture.source_locator.kind.value,
                picture.source_locator.canonical_json,
                picture.source_locator.sha256,
                picture.source_content_sha256,
                picture.source_media_type,
                picture.created_at,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise PicturePersistenceConflict(
            "picture insert violated an immutable storage constraint"
        ) from exc
    return cursor.rowcount == 1


def insert_picture_unit_if_absent_in_transaction(
    conn: sqlite3.Connection,
    unit: PictureUnitRecord,
) -> bool:
    """尝试插入完整图片单元；自然键已存在时返回 ``False``。"""

    require_picture_write_transaction(conn)
    if not isinstance(unit, PictureUnitRecord):
        raise TypeError("unit must be PictureUnitRecord")
    if unit.parent_picture_unit_id is None:
        conflict_target = (
            "ON CONFLICT(picture_id, unit_locator_json, producer_fingerprint) "
            "WHERE parent_picture_unit_id IS NULL DO NOTHING"
        )
    else:
        conflict_target = (
            "ON CONFLICT(picture_id, parent_picture_unit_id, unit_locator_json, "
            "producer_fingerprint) WHERE parent_picture_unit_id IS NOT NULL "
            "DO NOTHING"
        )
    try:
        cursor = conn.execute(
            "INSERT INTO picture_units "
            "(picture_unit_id, picture_id, unit_kind, unit_locator_json, "
            "unit_locator_sha256, producer_fingerprint, parent_picture_unit_id, "
            "pixel_sha256, media_type, width, height, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            + conflict_target,
            (
                unit.picture_unit_id,
                unit.picture_id,
                unit.locator.kind.value,
                unit.locator.canonical_json,
                unit.locator.sha256,
                unit.producer_fingerprint,
                unit.parent_picture_unit_id,
                unit.pixel_sha256,
                unit.media_type,
                unit.width,
                unit.height,
                unit.created_at,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise PicturePersistenceConflict(
            "picture unit insert violated an immutable storage constraint"
        ) from exc
    return cursor.rowcount == 1


def get_file_version_content_sha256(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    file_version_id: str,
) -> str | None:
    """读取精确 ``file/version`` 绑定的权威文件内容哈希。"""

    _require_connection(conn)
    row = conn.execute(
        "SELECT content_sha256 FROM file_versions WHERE id = ? AND file_id = ?",
        (file_version_id, file_id),
    ).fetchone()
    return None if row is None else str(row[0])


def get_picture(
    conn: sqlite3.Connection,
    picture_id: str,
) -> PictureRecord | None:
    """按统一图片身份读取不可变记录。"""

    _require_connection(conn)
    row = conn.execute(
        _PICTURE_SELECT + " WHERE picture_id = ?",
        (picture_id,),
    ).fetchone()
    return None if row is None else _picture_from_row(row)


def get_picture_for_source(
    conn: sqlite3.Connection,
    file_version_id: str,
    source_locator: PictureSourceLocator,
) -> PictureRecord | None:
    """按精确文件版本与规范来源定位器读取图片。"""

    _require_connection(conn)
    if not isinstance(source_locator, PictureSourceLocator):
        raise TypeError("source_locator must be PictureSourceLocator")
    row = conn.execute(
        _PICTURE_SELECT
        + " WHERE file_version_id = ? AND source_locator_sha256 = ? "
        "AND source_locator_json = ?",
        (
            file_version_id,
            source_locator.sha256,
            source_locator.canonical_json,
        ),
    ).fetchone()
    return None if row is None else _picture_from_row(row)


def get_current_picture_for_file(
    conn: sqlite3.Connection,
    file_id: str,
    source_locator: PictureSourceLocator,
) -> PictureRecord | None:
    """读取文件当前版本中精确来源对应的图片。"""

    _require_connection(conn)
    if not isinstance(source_locator, PictureSourceLocator):
        raise TypeError("source_locator must be PictureSourceLocator")
    row = conn.execute(
        _PICTURE_SELECT_ALIASED
        + " JOIN files AS file ON file.id = picture.file_id "
        "AND file.current_version_id = picture.file_version_id "
        "WHERE picture.file_id = ? "
        "AND picture.source_locator_sha256 = ? "
        "AND picture.source_locator_json = ?",
        (file_id, source_locator.sha256, source_locator.canonical_json),
    ).fetchone()
    return None if row is None else _picture_from_row(row)


def picture_is_current(conn: sqlite3.Connection, picture_id: str) -> bool:
    """图片是否仍绑定其文件的当前版本。"""

    _require_connection(conn)
    row = conn.execute(
        "SELECT EXISTS ("
        "SELECT 1 FROM pictures AS picture "
        "JOIN files AS file ON file.id = picture.file_id "
        "AND file.current_version_id = picture.file_version_id "
        "WHERE picture.picture_id = ?)",
        (picture_id,),
    ).fetchone()
    return bool(row[0])


def picture_binding_is_current(
    conn: sqlite3.Connection,
    *,
    picture_id: str,
    file_id: str,
    file_version_id: str,
    source_locator: PictureSourceLocator,
    source_content_sha256: str | None = None,
) -> bool:
    """校验调用方持有的完整图片绑定是否仍精确且当前。"""

    _require_connection(conn)
    if not isinstance(source_locator, PictureSourceLocator):
        raise TypeError("source_locator must be PictureSourceLocator")
    parameters: list[object] = [
        picture_id,
        file_id,
        file_version_id,
        source_locator.sha256,
        source_locator.canonical_json,
    ]
    asset_clause = ""
    if source_content_sha256 is not None:
        asset_clause = " AND picture.source_content_sha256 = ?"
        parameters.append(source_content_sha256)
    row = conn.execute(
        "SELECT EXISTS ("
        "SELECT 1 FROM pictures AS picture "
        "JOIN files AS file ON file.id = picture.file_id "
        "AND file.current_version_id = picture.file_version_id "
        "WHERE picture.picture_id = ? AND picture.file_id = ? "
        "AND picture.file_version_id = ? "
        "AND picture.source_locator_sha256 = ? "
        "AND picture.source_locator_json = ?"
        + asset_clause
        + ")",
        tuple(parameters),
    ).fetchone()
    return bool(row[0])


def get_picture_unit(
    conn: sqlite3.Connection,
    picture_unit_id: str,
) -> PictureUnitRecord | None:
    """按统一图片单元身份读取不可变记录。"""

    _require_connection(conn)
    row = conn.execute(
        _PICTURE_UNIT_SELECT + " WHERE picture_unit_id = ?",
        (picture_unit_id,),
    ).fetchone()
    return None if row is None else _picture_unit_from_row(row)


def get_picture_unit_for_locator(
    conn: sqlite3.Connection,
    picture_id: str,
    locator: PictureUnitLocator,
    *,
    parent_picture_unit_id: str | None,
    producer_fingerprint: str,
) -> PictureUnitRecord | None:
    """按图片内规范定位器与生产者身份读取精确处理单元。"""

    _require_connection(conn)
    if not isinstance(locator, PictureUnitLocator):
        raise TypeError("locator must be PictureUnitLocator")
    row = conn.execute(
        _PICTURE_UNIT_SELECT
        + " WHERE picture_id = ? AND unit_locator_sha256 = ? "
        "AND unit_locator_json = ? AND parent_picture_unit_id IS ? "
        "AND producer_fingerprint = ?",
        (
            picture_id,
            locator.sha256,
            locator.canonical_json,
            parent_picture_unit_id,
            producer_fingerprint,
        ),
    ).fetchone()
    return None if row is None else _picture_unit_from_row(row)


def list_picture_units(
    conn: sqlite3.Connection,
    picture_id: str,
) -> tuple[PictureUnitRecord, ...]:
    """稳定列出一张图片的全部处理单元。"""

    _require_connection(conn)
    rows = conn.execute(
        _PICTURE_UNIT_SELECT
        + " WHERE picture_id = ? ORDER BY created_at, picture_unit_id",
        (picture_id,),
    ).fetchall()
    return tuple(_picture_unit_from_row(row) for row in rows)


def picture_unit_is_current(
    conn: sqlite3.Connection,
    picture_unit_id: str,
) -> bool:
    """图片单元的父图片是否仍来自文件当前版本。"""

    _require_connection(conn)
    row = conn.execute(
        "SELECT EXISTS ("
        "SELECT 1 FROM picture_units AS unit "
        "JOIN pictures AS picture ON picture.picture_id = unit.picture_id "
        "JOIN files AS file ON file.id = picture.file_id "
        "AND file.current_version_id = picture.file_version_id "
        "WHERE unit.picture_unit_id = ?)",
        (picture_unit_id,),
    ).fetchone()
    return bool(row[0])


def picture_unit_binding_is_current(
    conn: sqlite3.Connection,
    *,
    picture_unit_id: str,
    picture_id: str,
    locator: PictureUnitLocator,
    parent_picture_unit_id: str | None,
    producer_fingerprint: str,
    pixel_sha256: str | None = None,
) -> bool:
    """校验调用方持有的图片单元绑定及其父图片当前性。"""

    _require_connection(conn)
    if not isinstance(locator, PictureUnitLocator):
        raise TypeError("locator must be PictureUnitLocator")
    parameters: list[object] = [
        picture_unit_id,
        picture_id,
        locator.sha256,
        locator.canonical_json,
        parent_picture_unit_id,
        producer_fingerprint,
    ]
    pixel_clause = ""
    if pixel_sha256 is not None:
        pixel_clause = " AND unit.pixel_sha256 = ?"
        parameters.append(pixel_sha256)
    row = conn.execute(
        "SELECT EXISTS ("
        "SELECT 1 FROM picture_units AS unit "
        "JOIN pictures AS picture ON picture.picture_id = unit.picture_id "
        "JOIN files AS file ON file.id = picture.file_id "
        "AND file.current_version_id = picture.file_version_id "
        "WHERE unit.picture_unit_id = ? AND unit.picture_id = ? "
        "AND unit.unit_locator_sha256 = ? AND unit.unit_locator_json = ? "
        "AND unit.parent_picture_unit_id IS ? AND unit.producer_fingerprint = ?"
        + pixel_clause
        + ")",
        tuple(parameters),
    ).fetchone()
    return bool(row[0])


_PICTURE_SELECT = (
    "SELECT picture_id, file_id, file_version_id, source_kind, "
    "source_locator_json, source_content_sha256, source_media_type, created_at "
    "FROM pictures"
)

_PICTURE_SELECT_ALIASED = (
    "SELECT picture.picture_id, picture.file_id, picture.file_version_id, "
    "picture.source_kind, picture.source_locator_json, picture.source_content_sha256, "
    "picture.source_media_type, picture.created_at FROM pictures AS picture"
)

_PICTURE_UNIT_SELECT = (
    "SELECT picture_unit_id, picture_id, unit_kind, unit_locator_json, "
    "producer_fingerprint, parent_picture_unit_id, pixel_sha256, media_type, "
    "width, height, created_at FROM picture_units"
)


def _picture_from_row(row: sqlite3.Row | tuple[object, ...]) -> PictureRecord:
    return PictureRecord(
        picture_id=str(row[0]),
        file_id=str(row[1]),
        file_version_id=str(row[2]),
        source_locator=PictureSourceLocator(
            kind=PictureSourceKind(str(row[3])),
            canonical_json=str(row[4]),
        ),
        source_content_sha256=str(row[5]),
        source_media_type=str(row[6]),
        created_at=str(row[7]),
    )


def _picture_unit_from_row(row: sqlite3.Row | tuple[object, ...]) -> PictureUnitRecord:
    return PictureUnitRecord(
        picture_unit_id=str(row[0]),
        picture_id=str(row[1]),
        locator=PictureUnitLocator(
            kind=PictureUnitKind(str(row[2])),
            canonical_json=str(row[3]),
        ),
        producer_fingerprint=str(row[4]),
        parent_picture_unit_id=None if row[5] is None else str(row[5]),
        pixel_sha256=str(row[6]),
        media_type=str(row[7]),
        width=int(row[8]),
        height=int(row[9]),
        created_at=str(row[10]),
    )


def _require_connection(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("picture repository requires an explicit sqlite3.Connection")


__all__ = [
    "PictureForeignKeysRequired",
    "PicturePersistenceConflict",
    "PictureRepositoryError",
    "PictureTransactionRequired",
    "get_current_picture_for_file",
    "get_file_version_content_sha256",
    "get_picture",
    "get_picture_for_source",
    "get_picture_unit",
    "get_picture_unit_for_locator",
    "insert_picture_if_absent_in_transaction",
    "insert_picture_unit_if_absent_in_transaction",
    "list_picture_units",
    "picture_binding_is_current",
    "picture_is_current",
    "picture_unit_binding_is_current",
    "picture_unit_is_current",
    "require_picture_write_transaction",
]
