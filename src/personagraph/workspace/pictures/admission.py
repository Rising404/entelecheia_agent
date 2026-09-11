"""统一图片身份与像素单元的业务准入用例。

准入层拥有候选身份/时间生成、精确文件版本校验、自然键幂等判定与 parent 业务约束。
SQL、事务和当前性查询仍由下层 ``storage.repository`` 提供。
"""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import uuid

from .contracts import (
    PictureRecord,
    PictureSourceKind,
    PictureSourceLocator,
    PictureUnitLocator,
    PictureUnitRecord,
    RegisteredPicture,
    RegisteredPictureUnit,
)
from .storage import repository


class PictureAdmissionError(RuntimeError):
    """图片身份或图片单元无法通过业务准入。"""


class PictureNotFound(PictureAdmissionError):
    """准入所需的权威文件、图片或图片单元不存在。"""


class PictureRegistrationConflict(PictureAdmissionError):
    """自然身份已存在，但不可变业务载荷与本次请求冲突。"""


class PictureAdmissionTransactionRequired(PictureAdmissionError):
    """图片准入没有处于调用方所有的事务中。"""


class PictureAdmissionForeignKeysRequired(PictureAdmissionError):
    """图片准入连接没有启用 SQLite 外键约束。"""


def ensure_picture_in_transaction(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    file_version_id: str,
    source_locator: PictureSourceLocator,
    source_content_sha256: str,
    source_media_type: str,
    picture_id: str | None = None,
    created_at: str | None = None,
) -> RegisteredPicture:
    """按精确 ``(file_version_id, source_locator)`` 幂等准入图片。"""

    _require_admission_write_context(conn)
    candidate = PictureRecord(
        picture_id=_new_picture_id() if picture_id is None else picture_id,
        file_id=file_id,
        file_version_id=file_version_id,
        source_locator=source_locator,
        source_content_sha256=source_content_sha256,
        source_media_type=source_media_type,
        created_at=_now() if created_at is None else created_at,
    )
    _require_exact_file_version(conn, candidate)

    existing = repository.get_picture_for_source(
        conn,
        candidate.file_version_id,
        candidate.source_locator,
    )
    if existing is not None:
        _require_same_picture_payload(existing, candidate)
        return RegisteredPicture(picture=existing, created=False)

    try:
        inserted = repository.insert_picture_if_absent_in_transaction(conn, candidate)
    except repository.PicturePersistenceConflict as exc:
        collision = repository.get_picture(conn, candidate.picture_id)
        if collision is not None:
            raise PictureRegistrationConflict(
                "picture_id is already bound to another immutable source"
            ) from exc
        raise PictureRegistrationConflict(
            "picture registration violates the file/version authority"
        ) from exc

    stored = repository.get_picture_for_source(
        conn,
        candidate.file_version_id,
        candidate.source_locator,
    )
    if stored is None:
        raise PictureAdmissionError("picture admission did not produce a record")
    _require_same_picture_payload(stored, candidate)
    return RegisteredPicture(picture=stored, created=inserted)


def ensure_picture_unit_in_transaction(
    conn: sqlite3.Connection,
    *,
    picture_id: str,
    locator: PictureUnitLocator,
    producer_fingerprint: str,
    parent_picture_unit_id: str | None,
    pixel_sha256: str,
    media_type: str,
    width: int,
    height: int,
    picture_unit_id: str | None = None,
    created_at: str | None = None,
) -> RegisteredPictureUnit:
    """按一张图片内的精确 locator 幂等准入完整像素、裁剪或切片。"""

    _require_admission_write_context(conn)
    if repository.get_picture(conn, picture_id) is None:
        raise PictureNotFound(f"picture does not exist: {picture_id}")
    candidate = PictureUnitRecord(
        picture_unit_id=(
            _new_picture_unit_id() if picture_unit_id is None else picture_unit_id
        ),
        picture_id=picture_id,
        locator=locator,
        producer_fingerprint=producer_fingerprint,
        parent_picture_unit_id=parent_picture_unit_id,
        pixel_sha256=pixel_sha256,
        media_type=media_type,
        width=width,
        height=height,
        created_at=_now() if created_at is None else created_at,
    )
    _require_valid_parent(conn, candidate)

    existing = repository.get_picture_unit_for_locator(
        conn,
        candidate.picture_id,
        candidate.locator,
        parent_picture_unit_id=candidate.parent_picture_unit_id,
        producer_fingerprint=candidate.producer_fingerprint,
    )
    if existing is not None:
        _require_same_unit_payload(existing, candidate)
        return RegisteredPictureUnit(unit=existing, created=False)

    try:
        inserted = repository.insert_picture_unit_if_absent_in_transaction(
            conn,
            candidate,
        )
    except repository.PicturePersistenceConflict as exc:
        collision = repository.get_picture_unit(conn, candidate.picture_unit_id)
        if collision is not None:
            raise PictureRegistrationConflict(
                "picture_unit_id is already bound to another immutable unit"
            ) from exc
        raise PictureRegistrationConflict(
            "picture unit registration violates picture authority"
        ) from exc

    stored = repository.get_picture_unit_for_locator(
        conn,
        candidate.picture_id,
        candidate.locator,
        parent_picture_unit_id=candidate.parent_picture_unit_id,
        producer_fingerprint=candidate.producer_fingerprint,
    )
    if stored is None:
        raise PictureAdmissionError("picture unit admission did not produce a record")
    _require_same_unit_payload(stored, candidate)
    return RegisteredPictureUnit(unit=stored, created=inserted)


def _require_exact_file_version(
    conn: sqlite3.Connection,
    candidate: PictureRecord,
) -> None:
    source_sha256 = repository.get_file_version_content_sha256(
        conn,
        file_id=candidate.file_id,
        file_version_id=candidate.file_version_id,
    )
    if source_sha256 is None:
        raise PictureNotFound(
            "picture source file_version does not belong to the requested file"
        )
    if (
        candidate.source_locator.kind
        in {PictureSourceKind.WHOLE_FILE, PictureSourceKind.DOCUMENT_SURFACE}
        and source_sha256 != candidate.source_content_sha256
    ):
        raise PictureRegistrationConflict(
            "file-backed picture source content hash must match its file version"
        )


def _require_admission_write_context(conn: sqlite3.Connection) -> None:
    try:
        repository.require_picture_write_transaction(conn)
    except repository.PictureTransactionRequired as exc:
        raise PictureAdmissionTransactionRequired(str(exc)) from exc
    except repository.PictureForeignKeysRequired as exc:
        raise PictureAdmissionForeignKeysRequired(str(exc)) from exc


def _require_valid_parent(
    conn: sqlite3.Connection,
    candidate: PictureUnitRecord,
) -> None:
    if candidate.parent_picture_unit_id == candidate.picture_unit_id:
        raise PictureRegistrationConflict("picture unit cannot be its own parent")
    if candidate.parent_picture_unit_id is None:
        return
    parent = repository.get_picture_unit(conn, candidate.parent_picture_unit_id)
    if parent is None or parent.picture_id != candidate.picture_id:
        raise PictureRegistrationConflict(
            "picture unit parent must exist in the same picture"
        )


def _require_same_picture_payload(
    stored: PictureRecord,
    candidate: PictureRecord,
) -> None:
    if (
        stored.file_id != candidate.file_id
        or stored.file_version_id != candidate.file_version_id
        or stored.source_locator != candidate.source_locator
        or stored.source_content_sha256 != candidate.source_content_sha256
        or stored.source_media_type != candidate.source_media_type
    ):
        raise PictureRegistrationConflict(
            "exact picture source is already registered with different immutable data"
        )


def _require_same_unit_payload(
    stored: PictureUnitRecord,
    candidate: PictureUnitRecord,
) -> None:
    if (
        stored.picture_id != candidate.picture_id
        or stored.locator != candidate.locator
        or stored.producer_fingerprint != candidate.producer_fingerprint
        or stored.parent_picture_unit_id != candidate.parent_picture_unit_id
        or stored.pixel_sha256 != candidate.pixel_sha256
        or stored.media_type != candidate.media_type
        or stored.width != candidate.width
        or stored.height != candidate.height
    ):
        raise PictureRegistrationConflict(
            "exact picture unit locator is already registered with different immutable data"
        )


def _new_picture_id() -> str:
    return f"pic_{uuid.uuid4().hex}"


def _new_picture_unit_id() -> str:
    return f"picunit_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PictureAdmissionError",
    "PictureAdmissionForeignKeysRequired",
    "PictureAdmissionTransactionRequired",
    "PictureNotFound",
    "PictureRegistrationConflict",
    "ensure_picture_in_transaction",
    "ensure_picture_unit_in_transaction",
]
