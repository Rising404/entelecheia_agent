"""解析已授权路径和项目 File 身份；只观察字节，不登记文件或派生内容。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
import sqlite3
import stat

from personagraph.input_processing.files import SourceFingerprint
from personagraph.input_processing.files.snapshot import MAX_DOCUMENT_FILE_BYTES
from personagraph.workspace.storage.database import DocumentDatabase
from .contracts import FileSource, ProjectFileError, ProjectFileRecord, ProjectFileVersionRecord
from .observation import normalize_project_relative_path, observe_project_file
from .storage import repository


class FileAccessError(ValueError):
    """可公开投影的失败码；不在错误正文中暴露 Host 路径。"""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class AuthorizedFileSource:
    project_id: str
    file_id: str | None
    file_version_id: str | None
    canonical_path: str = field(repr=False)
    relative_path: str
    file_name: str
    origin: FileSource
    media_type: str | None
    fingerprint: SourceFingerprint


class FileAccess:
    """File ID 不授予权限：每次读取均调用当前访问策略并重新观察内容。"""

    def __init__(
        self, *, database: DocumentDatabase, validate_path: Callable[[str], bool],
    ) -> None:
        self.database = database
        self._validate_path = validate_path

    def resolve_path(self, path: str) -> AuthorizedFileSource:
        try:
            relative = normalize_project_relative_path(path)
            with self.database.connect_readonly() as conn:
                record = repository.get_file_by_relative_path(
                    conn, project_id=self.database.project_id, relative_path=relative,
                ) if conn is not None else None
                version = self._version(conn, record) if record is not None else None
            return self._observe(relative, record, version)
        except FileAccessError:
            raise
        except (OSError, ValueError, sqlite3.Error, ProjectFileError) as exc:
            raise FileAccessError("file_source_unavailable") from exc

    def resolve_file(
        self, *, file_id: str, file_version_id: str | None = None,
        allow_changed: bool = False,
    ) -> AuthorizedFileSource:
        """精读要求版本匹配；显式准备可观察变化并生成新版本。"""

        try:
            with self.database.connect_readonly() as conn:
                record = repository.get_file(
                    conn, project_id=self.database.project_id, file_id=file_id,
                ) if conn is not None else None
                if record is None:
                    raise FileAccessError("file_not_found")
                version = self._version(conn, record)
            if file_version_id is not None and record.current_version_id != file_version_id:
                raise FileAccessError("file_version_changed")
            source = self._observe(record.relative_path, record, version)
            if source.file_version_id is None and (not allow_changed or file_version_id is not None):
                raise FileAccessError("file_content_changed")
            return source
        except FileAccessError:
            raise
        except (OSError, ValueError, sqlite3.Error, ProjectFileError) as exc:
            raise FileAccessError("file_source_unavailable") from exc

    def revalidate(self, source: AuthorizedFileSource) -> bool:
        if source.project_id != self.database.project_id:
            return False
        try:
            current = self.resolve_path(source.relative_path)
        except FileAccessError:
            return False
        # mtime 是本次观察的并发保护，不参与内容版本身份。
        return (
            current.canonical_path == source.canonical_path
            and current.fingerprint == source.fingerprint
            and current.file_id == source.file_id
            and current.file_version_id == source.file_version_id
        )

    def authorized_versions(self) -> tuple[tuple[str, str], ...]:
        """已有 File 的授权清单；精读仍须 resolve_file 重验实际字节。"""
        with self.database.connect_readonly() as conn:
            records = repository.list_current_files(
                conn, project_id=self.database.project_id,
            ) if conn is not None else ()
        return tuple(
            (record.file_id, record.current_version_id)
            for record in records
            if self._validate_path(str(self.database.project_root / record.relative_path)) is True
        )

    def _version(self, conn, record: ProjectFileRecord) -> ProjectFileVersionRecord | None:
        return repository.get_version(
            conn, project_id=self.database.project_id,
            file_version_id=record.current_version_id,
        ) if record.current_version_id is not None else None

    def _observe(
        self, relative: str, record: ProjectFileRecord | None,
        version: ProjectFileVersionRecord | None,
    ) -> AuthorizedFileSource:
        path = self.database.project_root / relative
        canonical = str(path)
        if self._validate_path(canonical) is not True:
            raise FileAccessError("file_authority_denied")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise FileAccessError("file_source_unavailable")
        if metadata.st_size > MAX_DOCUMENT_FILE_BYTES:
            raise FileAccessError("file_too_large")
        observation = observe_project_file(
            self.database.project_root, relative, max_bytes=MAX_DOCUMENT_FILE_BYTES,
        )
        if observation.size_bytes > MAX_DOCUMENT_FILE_BYTES:
            raise FileAccessError("file_too_large")
        if self._validate_path(canonical) is not True:
            raise FileAccessError("file_authority_denied")
        matching = version is not None and (
            version.content_sha256, version.size_bytes,
        ) == (observation.content_sha256, observation.size_bytes)
        return AuthorizedFileSource(
            project_id=self.database.project_id,
            file_id=record.file_id if record is not None else None,
            file_version_id=version.file_version_id if matching else None,
            canonical_path=canonical, relative_path=relative, file_name=Path(relative).name,
            origin=record.source if record is not None else FileSource.WORKSPACE_EXISTING,
            media_type=(record.media_type if record is not None else None) or mimetypes.guess_type(relative)[0],
            fingerprint=SourceFingerprint(
                sha256=observation.content_sha256, size_bytes=observation.size_bytes,
                mtime_ns=observation.source_mtime_ns,
            ),
        )
