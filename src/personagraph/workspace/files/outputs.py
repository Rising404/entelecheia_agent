"""在项目 output/ 下原子新建产物，并登记到唯一 File/FileVersion 账本。"""

from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4

from ...configuration.paths import deny_reason
from ..binding import (
    is_reserved_workspace_path, normalize_workspace_relative_path, ReservedWorkspacePathError,
)
from .admission import WorkspaceFileAuthority
from .contracts import (
    FileSource,
    ProjectFileError,
    ProjectFilePathError,
    RegisteredProjectFileVersion,
    WorkspaceDatabasePort,
)
from .observation import (
    _open_or_create_directory,
    _open_project_root,
    normalize_project_relative_path,
)


OUTPUT_DIRECTORY = "output"
MAX_OUTPUT_PATH_CHARS = 512
MAX_OUTPUT_CONTENT_BYTES = 128 * 1024


class ProjectOutputError(ProjectFileError):
    """新产物无法安全发布或登记。"""


class ProjectOutputConflict(ProjectOutputError):
    """目标已存在，产物创建绝不覆盖。"""


class ProjectOutputService:
    """只拥有 output 子树的 CREATE；不修改或删除任何既存用户文件。"""

    def __init__(
        self,
        database: WorkspaceDatabasePort,
        *,
        root_device: int,
        root_inode: int,
    ) -> None:
        self._database = database
        self._root_identity = (root_device, root_inode)
        self._authority = WorkspaceFileAuthority(database)

    def create_text(self, *, path: str, content: str) -> RegisteredProjectFileVersion:
        """path 相对 output/；父目录按需建立，已存在的最终目标一律拒绝。"""

        relative_path = _output_path(self._database.project_root, path)
        payload = _text_payload(content)
        parts = relative_path.split("/")
        parent_fd = _open_project_root(self._database.project_root)
        temporary_name = f".output-{uuid4().hex}.part"
        temporary_exists = False
        published: os.stat_result | None = None
        registered = False
        try:
            root = os.fstat(parent_fd)
            if (int(root.st_dev), int(root.st_ino)) != self._root_identity:
                raise ProjectFilePathError("the bound project root changed")
            for part in parts[:-1]:
                child_fd = _open_or_create_directory(parent_fd, part)
                os.close(parent_fd)
                parent_fd = child_fd
            flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
            temporary_exists = True
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("failed to write output bytes")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            # link 是原子 create-if-absent；replace/rename 会覆盖，不能用于此能力。
            try:
                os.link(
                    temporary_name, parts[-1], src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd, follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ProjectOutputConflict("output destination already exists") from exc
            published = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            os.unlink(temporary_name, dir_fd=parent_fd)
            temporary_exists = False
            os.fsync(parent_fd)
            self._require_current_root()
            registration = self._authority.register_path(
                relative_path, source=FileSource.AGENT_OUTPUT,
            )
            registered = True
            return registration
        except ProjectFileError:
            raise
        except OSError as exc:
            raise ProjectOutputError("output could not be created atomically") from exc
        finally:
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if published is not None and not registered:
                _remove_unregistered_output(parent_fd, parts[-1], published)
            os.close(parent_fd)

    def _require_current_root(self) -> None:
        current = self._database.project_root.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or (int(current.st_dev), int(current.st_ino)) != self._root_identity
        ):
            raise ProjectFilePathError("the bound project root changed")


def _output_path(root: Path, path: str) -> str:
    if not isinstance(path, str) or len(path) > MAX_OUTPUT_PATH_CHARS:
        raise ProjectFilePathError("output path must be bounded relative text")
    normalized = normalize_project_relative_path(path)
    relative_path = f"{OUTPUT_DIRECTORY}/{normalized}"
    target = root / relative_path
    try:
        normalize_workspace_relative_path(normalized)
        if is_reserved_workspace_path(root, target) or deny_reason(target):
            raise ProjectFilePathError("output path is private or blocked")
    except ReservedWorkspacePathError as exc:
        raise ProjectFilePathError("output path must not contain symbolic links") from exc
    return relative_path


def _text_payload(content: str) -> bytes:
    if not isinstance(content, str) or "\x00" in content:
        raise ValueError("output content must be UTF-8 text")
    payload = content.encode("utf-8")
    if len(payload) > MAX_OUTPUT_CONTENT_BYTES:
        raise ValueError("output content exceeds the 128 KiB byte limit")
    return payload


def _remove_unregistered_output(
    parent_fd: int, name: str, published: os.stat_result,
) -> None:
    """登记失败只清理本次发布且未被用户改动的 inode，不删除替换者的文件。"""

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISREG(current.st_mode) and identity(current) == identity(published):
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    except OSError:
        pass
