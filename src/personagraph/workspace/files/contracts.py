"""Workspace 文件权威在应用层与存储层之间共享的稳定合同。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import sqlite3
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkspaceDatabasePort(Protocol):
    """File authority 所需的最小项目数据库能力。"""

    project_id: str
    project_root: Path
    db_path: Path

    def connect(self) -> AbstractContextManager[sqlite3.Connection]: ...


class FileSource(StrEnum):
    """文件摄入与版本共享的封闭来源词汇。"""

    USER_UPLOAD = "user_upload"
    WORKSPACE_EXISTING = "workspace_existing"
    AGENT_OUTPUT = "agent_output"


class ProjectFileError(RuntimeError):
    """项目文件注册错误的基类。"""


class ProjectFilePathError(ProjectFileError):
    """相对路径未解析为安全的项目本地常规文件。"""


class ProjectFileChangedError(ProjectFileError):
    """捕获不可变版本指纹期间文件发生变化。"""


class ProjectFileSizeLimitError(ProjectFileError):
    """文件超过调用方声明的安全读取上限。"""


class FileRegistrationConflict(ProjectFileError):
    """请求的文件身份与现有项目记录冲突。"""


@dataclass(frozen=True, slots=True)
class ProjectFileRecord:
    file_id: str
    project_id: str
    relative_path: str
    source: FileSource
    media_type: str | None
    current_version_id: str | None
    observed_mtime_ns: int | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ProjectFileVersionRecord:
    """一个不可变内容版本；mtime 仅记录此版本首次捕获时的历史观察。"""

    file_version_id: str
    file_id: str
    version_number: int
    source: FileSource
    content_sha256: str
    size_bytes: int
    source_mtime_ns: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RegisteredProjectFileVersion:
    file: ProjectFileRecord
    version: ProjectFileVersionRecord
    created_file: bool


@dataclass(frozen=True, slots=True)
class ProjectFileObservation:
    """一次稳定读取获得的文件内容与来源元数据。"""

    content_sha256: str
    size_bytes: int
    source_mtime_ns: int
