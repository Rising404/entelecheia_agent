"""Project 所有的聊天附件落盘与校验。

上传文件只在已绑定 Project 根目录下保留一份，并由该 Project 的
``documents.sqlite`` 持有文件与不可变版本身份。Session 只记录附件绑定，不拥有
``input``/``output`` 文件树，也不存在进程级全局存储回退。
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from personagraph.workspace.storage.context import current as current_project_document_database
from personagraph.workspace.storage.database import DocumentDatabase
from personagraph.workspace.files import (
    FileSource,
    ProjectFilePathError,
    WorkspaceFileAuthority,
    ProjectFileVersionRecord,
    normalize_project_relative_path,
)
from personagraph.workspace.files import ProjectUploadService
from personagraph.input_processing.files import DetectedType, detect_type


# 供报告普通 magic-prefix 预算的调用方使用。上传 payload 已经有界且驻留内存，
# 类型检测会收到证明 OOXML 中央目录所需的完整字节。
TYPE_SNIFF_BYTES = 8192

MAX_ATTACHMENT_BYTES = 64 * 1024 * 1024
MAX_SESSION_ATTACHMENT_BYTES = 512 * 1024 * 1024
MAX_ATTACHMENTS_PER_TURN = 16


class AttachmentTooLarge(ValueError):
    """payload 超过硬资源边界，因此未被存储。"""

    def __init__(self, limit_bytes: int, actual_bytes: int) -> None:
        super().__init__(f"attachment exceeds {limit_bytes} bytes")
        self.limit_bytes = limit_bytes
        self.actual_bytes = actual_bytes


class AttachmentContentMismatch(RuntimeError):
    """项目文件或版本记录不再匹配附件的不可变收据。"""


class AttachmentProjectContextRequired(RuntimeError):
    """附件操作缺少所属 Project 的文档数据库绑定。"""


@dataclass(frozen=True, slots=True)
class StoredAttachment:
    stored_rel_path: str
    absolute_path: Path
    original_name: str
    size_bytes: int
    content_hash: str
    detected: DetectedType
    project_id: str
    file_id: str
    file_version_id: str


class SessionStorageArea(StrEnum):
    """兼容现有读取授权语义的作者分类，而非磁盘目录名称。

    ``INPUT`` 表示用户上传或项目已有材料，``OUTPUT`` 表示 Agent 产物。实际文件
    来源仍由 ``files.origin`` 的封闭 ``FileSource`` 词汇记录。
    """

    INPUT = "input"
    OUTPUT = "output"


def session_root(session_id: str) -> Path:
    """返回当前 Session 已绑定 Project 的根目录。

    ``session_id`` 仅用于调用方身份校验；Project 绑定由 request/task 局部的
    ``session_database_scope`` 建立。未绑定时明确失败，不猜测全局路径。
    """

    _validate_session_id(session_id)
    return _require_project_database().project_root


def classify_session_path(
    session_id: str,
    candidate: Path | str,
    *,
    allowed_sources: frozenset[FileSource] | None = None,
) -> SessionStorageArea | None:
    """根据 Project 文件记录判断路径是用户材料还是 Agent 产物。

    只有 Project 根目录内、无符号链接改写且已登记在 ``files`` 表中的常规路径
    才会获得分类。Session 目录名不再参与授权。
    """

    _validate_session_id(session_id)
    database = current_project_document_database()
    if database is None:
        return None
    root = database.project_root.expanduser().resolve()
    try:
        lexical = Path(os.path.abspath(Path(candidate).expanduser()))
        resolved = Path(candidate).expanduser().resolve(strict=True)
        lexical_relative = lexical.relative_to(root)
        resolved_relative = resolved.relative_to(root)
        if (
            not resolved_relative.parts
            or lexical_relative != resolved_relative
            or not resolved.is_file()
        ):
            return None
        relative_path = resolved_relative.as_posix()
        record = WorkspaceFileAuthority(database).get_file_by_relative_path(
            relative_path
        )
    except (OSError, ProjectFilePathError, ValueError):
        return None
    if record is None or (
        allowed_sources is not None and record.source not in allowed_sources
    ):
        return None
    return (
        SessionStorageArea.OUTPUT
        if record.source is FileSource.AGENT_OUTPUT
        else SessionStorageArea.INPUT
    )


def store_attachment(
    *,
    session_id: str,
    attachment_id: str,
    raw_name: str,
    payload: bytes,
    max_bytes: int = MAX_ATTACHMENT_BYTES,
) -> StoredAttachment:
    """把上传原子落入已绑定 Project，并登记文件及不可变版本。"""

    if len(payload) > max_bytes:
        raise AttachmentTooLarge(max_bytes, len(payload))
    _validate_session_id(session_id)
    if not attachment_id or "/" in attachment_id or "\\" in attachment_id:
        raise ValueError(f"invalid attachment id: {attachment_id!r}")

    database = _require_project_database()
    upload = ProjectUploadService(database).store_upload(
        original_name=raw_name,
        payload=payload,
        file_id=attachment_id,
    )
    return StoredAttachment(
        stored_rel_path=upload.relative_path,
        absolute_path=upload.absolute_path,
        original_name=upload.sanitized_name,
        size_bytes=upload.size_bytes,
        content_hash=upload.content_sha256,
        detected=detect_type(payload, upload.sanitized_name),
        project_id=database.project_id,
        file_id=upload.file_id,
        file_version_id=upload.file_version_id,
    )


def read_attachment(session_id: str, stored_rel_path: str) -> bytes:
    """读取已登记 Project 文件，并核对当前不可变版本收据。"""

    _validate_session_id(session_id)
    target, version = _registered_attachment_target(stored_rel_path)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise AttachmentContentMismatch("registered attachment is unreadable") from exc
    actual_hash = hashlib.sha256(payload).hexdigest()
    if (
        len(payload) != version.size_bytes
        or not hmac.compare_digest(actual_hash, version.content_sha256)
    ):
        raise AttachmentContentMismatch(
            "project file bytes no longer match the registered file version"
        )
    return payload


def read_verified_attachment(
    session_id: str,
    stored_rel_path: str,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> bytes:
    """同时核对 Project 版本收据与 Session 附件收据。"""

    payload = read_attachment(session_id, stored_rel_path)
    actual_hash = hashlib.sha256(payload).hexdigest()
    if (
        len(payload) != expected_size_bytes
        or not hmac.compare_digest(actual_hash, expected_sha256.lower())
    ):
        raise AttachmentContentMismatch(
            "project file bytes no longer match the session attachment receipt"
        )
    return payload


def _registered_attachment_target(
    stored_rel_path: str,
) -> tuple[Path, ProjectFileVersionRecord]:
    database = _require_project_database()
    try:
        relative_path = normalize_project_relative_path(stored_rel_path)
    except ProjectFilePathError as exc:
        raise AttachmentContentMismatch("attachment path is not a project path") from exc

    file_authority = WorkspaceFileAuthority(database)
    record = file_authority.get_file_by_relative_path(relative_path)
    if record is None or record.current_version_id is None:
        raise AttachmentContentMismatch(
            "attachment path has no current project file record"
        )
    version = file_authority.get_version(record.current_version_id)
    if version is None or version.file_id != record.file_id:
        raise AttachmentContentMismatch(
            "attachment project file version is missing or inconsistent"
        )

    root = database.project_root.expanduser().resolve()
    target = root.joinpath(*relative_path.split("/"))
    try:
        resolved = target.resolve(strict=True)
        if resolved.relative_to(root).as_posix() != relative_path or not resolved.is_file():
            raise AttachmentContentMismatch(
                "attachment path is no longer a direct regular project file"
            )
    except AttachmentContentMismatch:
        raise
    except (OSError, ValueError) as exc:
        raise AttachmentContentMismatch(
            "attachment path is missing or escapes the project"
        ) from exc
    return target, version


def _require_project_database() -> DocumentDatabase:
    database = current_project_document_database()
    if database is None:
        raise AttachmentProjectContextRequired(
            "attachment storage requires a bound Project DocumentDatabase"
        )
    return database


def _validate_session_id(session_id: str) -> None:
    if (
        not session_id
        or "/" in session_id
        or "\\" in session_id
        or session_id in {".", ".."}
    ):
        raise ValueError(f"invalid session id for attachment storage: {session_id!r}")
