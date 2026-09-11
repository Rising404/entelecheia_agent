"""Workspace 文件观察、身份冲突判断与事务准入。"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
import sqlite3
from uuid import uuid4

from .contracts import (
    FileRegistrationConflict,
    FileSource,
    ProjectFileObservation,
    ProjectFileRecord,
    ProjectFileVersionRecord,
    RegisteredProjectFileVersion,
    WorkspaceDatabasePort,
)
from .observation import normalize_project_relative_path, observe_project_file
from .storage import repository


_STORAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkspaceFileAuthority:
    """将安全文件观察准入项目权威，并查询其不可变版本。"""

    def __init__(self, database: WorkspaceDatabasePort) -> None:
        self._database = database

    @property
    def database(self) -> WorkspaceDatabasePort:
        return self._database

    def register_path(
        self,
        relative_path: str | os.PathLike[str],
        *,
        source: FileSource | str,
        file_id: str | None = None,
        media_type: str | None = None,
        occurred_at: datetime | str | None = None,
    ) -> RegisteredProjectFileVersion:
        """为路径当前字节追加一个不可变版本。"""

        normalized_path = normalize_project_relative_path(relative_path)
        normalized_source = _normalize_source(source)
        requested_file_id = (
            _validate_storage_id(file_id, label="file_id")
            if file_id is not None
            else None
        )
        timestamp = _normalize_timestamp(occurred_at).isoformat()
        observation = observe_project_file(
            self._database.project_root,
            normalized_path,
        )
        return self._register_observation(
            relative_path=normalized_path,
            source=normalized_source,
            requested_file_id=requested_file_id,
            media_type=media_type,
            occurred_at=timestamp,
            observation=observation,
        )

    def ensure_current_path(
        self,
        relative_path: str | os.PathLike[str],
        *,
        source: FileSource | str,
        file_id: str | None = None,
        media_type: str | None = None,
        occurred_at: datetime | str | None = None,
    ) -> RegisteredProjectFileVersion:
        """返回当前内容版本；字节未变时只记录新的文件时间观察。

        整个判断在一个 ``BEGIN IMMEDIATE`` 事务内完成；文件只观察一次，因此
        崩溃重放不会因为二次哈希看到不同字节而产生模糊结果。
        """

        normalized_path = normalize_project_relative_path(relative_path)
        normalized_source = _normalize_source(source)
        requested_file_id = (
            _validate_storage_id(file_id, label="file_id")
            if file_id is not None
            else None
        )
        observation = observe_project_file(
            self._database.project_root,
            normalized_path,
        )
        try:
            with self._database.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = repository.get_file_by_relative_path(
                    conn,
                    project_id=self._database.project_id,
                    relative_path=normalized_path,
                )
                if existing is not None:
                    if (
                        requested_file_id is not None
                        and requested_file_id != existing.file_id
                    ):
                        raise FileRegistrationConflict(
                            "relative_path is already bound to another file_id"
                        )
                    current_version = (
                        repository.get_version(
                            conn,
                            project_id=self._database.project_id,
                            file_version_id=existing.current_version_id,
                        )
                        if existing.current_version_id is not None
                        else None
                    )
                    if current_version is not None and _content_matches(
                        current_version,
                        observation,
                    ):
                        if existing.observed_mtime_ns != observation.source_mtime_ns:
                            return repository.record_current_observation(
                                conn,
                                file=existing,
                                version=current_version,
                                source=normalized_source,
                                observation=observation,
                                event_id=_new_id("fileevent"),
                                occurred_at=_normalize_timestamp(occurred_at).isoformat(),
                            )
                        return RegisteredProjectFileVersion(
                            file=existing,
                            version=current_version,
                            created_file=False,
                        )
                return self._register_observation_on_connection(
                    conn,
                    existing=existing,
                    relative_path=normalized_path,
                    source=normalized_source,
                    requested_file_id=requested_file_id,
                    media_type=media_type,
                    occurred_at=_normalize_timestamp(occurred_at).isoformat(),
                    observation=observation,
                )
        except sqlite3.IntegrityError as exc:
            raise FileRegistrationConflict(
                "the project file identity conflicts with an existing record"
            ) from exc

    def get_file(self, file_id: str) -> ProjectFileRecord | None:
        normalized_id = _validate_storage_id(file_id, label="file_id")
        with self._database.connect() as conn:
            return repository.get_file(
                conn,
                project_id=self._database.project_id,
                file_id=normalized_id,
            )

    def get_file_by_relative_path(
        self,
        relative_path: str | os.PathLike[str],
    ) -> ProjectFileRecord | None:
        normalized_path = normalize_project_relative_path(relative_path)
        with self._database.connect() as conn:
            return repository.get_file_by_relative_path(
                conn,
                project_id=self._database.project_id,
                relative_path=normalized_path,
            )

    def get_file_with_version(
        self,
        file_id: str,
        file_version_id: str,
    ) -> tuple[ProjectFileRecord, ProjectFileVersionRecord] | None:
        """返回一个精确、同项目且相互关联的文件/版本对。"""

        normalized_file_id = _validate_storage_id(file_id, label="file_id")
        normalized_version_id = _validate_storage_id(
            file_version_id,
            label="file_version_id",
        )
        with self._database.connect() as conn:
            return repository.get_file_with_version(
                conn,
                project_id=self._database.project_id,
                file_id=normalized_file_id,
                file_version_id=normalized_version_id,
            )

    def get_version(
        self,
        file_version_id: str,
    ) -> ProjectFileVersionRecord | None:
        normalized_id = _validate_storage_id(
            file_version_id,
            label="file_version_id",
        )
        with self._database.connect() as conn:
            return repository.get_version(
                conn,
                project_id=self._database.project_id,
                file_version_id=normalized_id,
            )

    def list_versions(
        self,
        file_id: str,
    ) -> tuple[ProjectFileVersionRecord, ...]:
        normalized_id = _validate_storage_id(file_id, label="file_id")
        with self._database.connect() as conn:
            return repository.list_versions(
                conn,
                project_id=self._database.project_id,
                file_id=normalized_id,
            )

    def _register_observation(
        self,
        *,
        relative_path: str,
        source: FileSource,
        requested_file_id: str | None,
        media_type: str | None,
        occurred_at: str,
        observation: ProjectFileObservation,
    ) -> RegisteredProjectFileVersion:
        try:
            with self._database.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = repository.get_file_by_relative_path(
                    conn,
                    project_id=self._database.project_id,
                    relative_path=relative_path,
                )
                return self._register_observation_on_connection(
                    conn,
                    existing=existing,
                    relative_path=relative_path,
                    source=source,
                    requested_file_id=requested_file_id,
                    media_type=media_type,
                    occurred_at=occurred_at,
                    observation=observation,
                )
        except sqlite3.IntegrityError as exc:
            raise FileRegistrationConflict(
                "the project file identity conflicts with an existing record"
            ) from exc

    def _register_observation_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        existing: ProjectFileRecord | None,
        relative_path: str,
        source: FileSource,
        requested_file_id: str | None,
        media_type: str | None,
        occurred_at: str,
        observation: ProjectFileObservation,
    ) -> RegisteredProjectFileVersion:
        if existing is not None:
            if (
                requested_file_id is not None
                and requested_file_id != existing.file_id
            ):
                raise FileRegistrationConflict(
                    "relative_path is already bound to another file_id"
                )
            resolved_file_id = existing.file_id
        else:
            resolved_file_id = requested_file_id or _new_id("file")
        return repository.register_observation(
            conn,
            project_id=self._database.project_id,
            relative_path=relative_path,
            source=source,
            file_id=resolved_file_id,
            created_file=existing is None,
            new_file_version_id=_new_id("filever"),
            new_event_id=_new_id("fileevent"),
            media_type=media_type,
            occurred_at=occurred_at,
            observation=observation,
        )


def get_file_with_version_in_transaction(
    conn: sqlite3.Connection,
    database: WorkspaceDatabasePort,
    *,
    file_id: str,
    file_version_id: str,
) -> tuple[ProjectFileRecord, ProjectFileVersionRecord] | None:
    """在调用方事务中读取精确、同项目且相互关联的文件版本。"""

    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
        raise ValueError("file/version lookup requires a caller-owned transaction")
    if not isinstance(database, WorkspaceDatabasePort):
        raise TypeError("database must satisfy WorkspaceDatabasePort")
    return repository.get_file_with_version(
        conn,
        project_id=database.project_id,
        file_id=_validate_storage_id(file_id, label="file_id"),
        file_version_id=_validate_storage_id(
            file_version_id,
            label="file_version_id",
        ),
    )


def find_current_file_in_connection(
    conn: sqlite3.Connection,
    database: WorkspaceDatabasePort,
    relative_path: str,
) -> tuple[ProjectFileRecord, ProjectFileVersionRecord] | None:
    """只读查找路径的当前文件身份；不观察磁盘、不登记版本、不打开连接。"""

    file = repository.get_file_by_relative_path(
        conn,
        project_id=database.project_id,
        relative_path=normalize_project_relative_path(relative_path),
    )
    if file is None or file.current_version_id is None:
        return None
    return repository.get_file_with_version(
        conn, project_id=database.project_id,
        file_id=file.file_id, file_version_id=file.current_version_id,
    )


def _content_matches(
    version: ProjectFileVersionRecord,
    observation: ProjectFileObservation,
) -> bool:
    return (
        version.content_sha256 == observation.content_sha256
        and version.size_bytes == observation.size_bytes
    )


def _validate_storage_id(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _STORAGE_ID_PATTERN.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} is not a safe storage identity")
    return value


def _normalize_source(value: FileSource | str) -> FileSource:
    try:
        return FileSource(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source must be a supported file provenance") from exc


def _normalize_timestamp(value: datetime | str | None) -> datetime:
    try:
        parsed = (
            datetime.now(timezone.utc)
            if value is None
            else value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("occurred_at must be a valid datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"
