"""直接落入项目文件系统的原子上传。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import stat
from uuid import uuid4

from .admission import (
    WorkspaceFileAuthority,
    _new_id,
    _normalize_timestamp,
    _validate_storage_id,
)
from .contracts import (
    FileSource,
    ProjectFilePathError,
    RegisteredProjectFileVersion,
    WorkspaceDatabasePort,
)
from .observation import (
    _open_or_create_directory,
    _open_project_root,
    _validate_path_component,
)


class ProjectUploadError(RuntimeError):
    """无法安全落盘并注册上传内容。"""


class ProjectUploadPathError(ProjectUploadError):
    """项目上传目标包含不安全的路径组件。"""


class ProjectUploadConflict(ProjectUploadError):
    """上传的最终目标已经存在。"""


@dataclass(frozen=True, slots=True)
class StoredProjectUpload:
    registration: RegisteredProjectFileVersion
    relative_path: str
    absolute_path: Path
    sanitized_name: str
    media_type: str
    size_bytes: int
    content_sha256: str

    @property
    def file_id(self) -> str:
        return self.registration.file.file_id

    @property
    def file_version_id(self) -> str:
        return self.registration.version.file_version_id


class ProjectUploadService:
    """只落盘一次上传内容，不在会话侧保留持久字节副本。"""

    def __init__(
        self,
        database: WorkspaceDatabasePort,
        authority: WorkspaceFileAuthority | None = None,
    ) -> None:
        if authority is not None and not _same_database(
            authority.database,
            database,
        ):
            raise ValueError("authority must use the same Workspace database")
        self._database = database
        self._authority = authority or WorkspaceFileAuthority(database)

    def store_upload(
        self,
        *,
        original_name: str,
        payload: bytes,
        file_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> StoredProjectUpload:
        """在 ``project_root/附件`` 下原子发布字节并完成登记。"""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        resolved_file_id = (
            _new_id("file")
            if file_id is None
            else _validate_storage_id(file_id, label="file_id")
        )
        timestamp = _normalize_timestamp(created_at)
        safe_name, detected = _sanitize_and_detect(original_name, payload)
        if detected.extension and not safe_name.casefold().endswith(
            detected.extension.casefold()
        ):
            safe_name = f"{safe_name}{detected.extension}"
        try:
            _validate_path_component(safe_name, label="sanitized_name")
            directory_name = (
                f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{resolved_file_id}"
            )
            _validate_path_component(directory_name, label="upload directory")
        except ProjectFilePathError as exc:
            raise ProjectUploadPathError(str(exc)) from exc

        relative_path = f"附件/{directory_name}/{safe_name}"
        root_fd: int | None = None
        attachments_fd: int | None = None
        destination_fd: int | None = None
        temporary_fd: int | None = None
        temporary_name = f".upload-{uuid4().hex}.part"
        temporary_exists = False
        published_identity: tuple[int, int] | None = None
        registered = False
        try:
            root_fd = _open_project_root(self._database.project_root)
            attachments_fd = _open_or_create_directory(root_fd, "附件")
            destination_fd = _open_or_create_directory(
                attachments_fd,
                directory_name,
            )
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            )
            temporary_fd = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=destination_fd,
            )
            temporary_exists = True
            _write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
            temporary_stat = os.fstat(temporary_fd)
            if (
                not stat.S_ISREG(temporary_stat.st_mode)
                or int(temporary_stat.st_size) != len(payload)
            ):
                raise ProjectUploadError("temporary upload is not the expected file")
            os.close(temporary_fd)
            temporary_fd = None

            try:
                os.link(
                    temporary_name,
                    safe_name,
                    src_dir_fd=destination_fd,
                    dst_dir_fd=destination_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ProjectUploadConflict(
                    "project upload destination already exists"
                ) from exc
            published = os.stat(
                safe_name,
                dir_fd=destination_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(published.st_mode):
                raise ProjectUploadPathError(
                    "published upload destination is not a regular file"
                )
            published_identity = (int(published.st_dev), int(published.st_ino))
            os.unlink(temporary_name, dir_fd=destination_fd)
            temporary_exists = False
            os.fsync(destination_fd)

            registration = self._authority.register_path(
                relative_path,
                source=FileSource.USER_UPLOAD,
                file_id=resolved_file_id,
                media_type=detected.media_type,
                occurred_at=timestamp,
            )
            registered = True
        except (ProjectUploadError, TypeError, ValueError):
            raise
        except ProjectFilePathError as exc:
            raise ProjectUploadPathError(str(exc)) from exc
        except OSError as exc:
            raise ProjectUploadError(
                "project upload could not be written atomically"
            ) from exc
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            if temporary_exists and destination_fd is not None:
                try:
                    os.unlink(temporary_name, dir_fd=destination_fd)
                except OSError:
                    pass
            if (
                not registered
                and published_identity is not None
                and destination_fd is not None
            ):
                _unlink_published_if_unchanged(
                    destination_fd,
                    safe_name,
                    published_identity,
                )
            for descriptor in (destination_fd, attachments_fd, root_fd):
                if descriptor is not None:
                    os.close(descriptor)

        return StoredProjectUpload(
            registration=registration,
            relative_path=relative_path,
            absolute_path=self._database.project_root / relative_path,
            sanitized_name=safe_name,
            media_type=detected.media_type,
            size_bytes=registration.version.size_bytes,
            content_sha256=registration.version.content_sha256,
        )


def _same_database(
    left: WorkspaceDatabasePort,
    right: WorkspaceDatabasePort,
) -> bool:
    return (
        left.project_id == right.project_id
        and left.project_root == right.project_root
        and left.db_path == right.db_path
    )


def _sanitize_and_detect(original_name: str, payload: bytes):
    # 仅在实际接受上传时导入，避免 storage 包初始化时加载文件检测器。
    from ...input_processing.files import (
        detect_type,
        sanitize_original_name,
    )

    safe_name = sanitize_original_name(original_name)
    return safe_name, detect_type(payload, safe_name)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("failed to write upload bytes")
        remaining = remaining[written:]


def _unlink_published_if_unchanged(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_ISREG(current.st_mode)
            and (int(current.st_dev), int(current.st_ino)) == expected_identity
        ):
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    except OSError:
        pass
