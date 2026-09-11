"""相对 Workspace 根目录安全观察一个常规文件。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .contracts import (
    ProjectFileChangedError,
    ProjectFileError,
    ProjectFileObservation,
    ProjectFilePathError,
    ProjectFileSizeLimitError,
)


_READ_CHUNK_BYTES = 1024 * 1024


def normalize_project_relative_path(value: str | os.PathLike[str]) -> str:
    """返回规范 POSIX 相对路径，否则在任何 I/O 前拒绝。"""

    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ProjectFilePathError("relative_path must be path-like") from exc
    if not isinstance(raw, str):
        raise ProjectFilePathError("relative_path must be text")
    if (
        not raw
        or raw.startswith("/")
        or "\\" in raw
        or "\x00" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise ProjectFilePathError("relative_path must be a canonical project path")
    return raw


def observe_project_file(
    root: Path, relative_path: str, *, max_bytes: int | None = None,
) -> ProjectFileObservation:
    """通过 no-follow 文件描述符读取并证明观察期间文件未变化。"""

    parts = relative_path.split("/")
    current_fd = _open_project_root(root)
    file_fd: int | None = None
    try:
        for component in parts[:-1]:
            next_fd = _open_existing_directory(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        flags = (os.O_RDONLY | _no_follow_flag() | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        try:
            file_fd = os.open(parts[-1], flags, dir_fd=current_fd)
        except OSError as exc:
            raise ProjectFilePathError(
                "project file is missing, inaccessible, or a symbolic link"
            ) from exc
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ProjectFilePathError("project path is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ProjectFileSizeLimitError(
                "project file exceeds the read size limit"
            )

        digest = hashlib.sha256()
        size_bytes = 0
        while True:
            chunk = os.read(file_fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            if max_bytes is not None and size_bytes > max_bytes:
                raise ProjectFileSizeLimitError(
                    "project file exceeds the read size limit"
                )
        after = os.fstat(file_fd)
        visible = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        identity_before = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        identity_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        identity_visible = (
            int(visible.st_dev),
            int(visible.st_ino),
            int(visible.st_size),
            int(visible.st_mtime_ns),
        )
        if (
            identity_before != identity_after
            or identity_after != identity_visible
            or size_bytes != int(after.st_size)
        ):
            raise ProjectFileChangedError(
                "project file changed while its version was registered"
            )
        return ProjectFileObservation(
            content_sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            source_mtime_ns=int(after.st_mtime_ns),
        )
    except ProjectFileError:
        raise
    except OSError as exc:
        raise ProjectFilePathError(
            "project file could not be inspected safely"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(current_fd)


def _open_project_root(root: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | _no_follow_flag()
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(str(root), flags)
        opened = os.fstat(descriptor)
        visible = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(visible.st_mode)
            or int(opened.st_dev) != int(visible.st_dev)
            or int(opened.st_ino) != int(visible.st_ino)
        ):
            raise ProjectFilePathError("project_root changed while opening it")
        return descriptor
    except ProjectFileError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except OSError as exc:
        raise ProjectFilePathError(
            "project_root is unavailable or is a symbolic link"
        ) from exc


def _open_existing_directory(parent_fd: int, component: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | _no_follow_flag()
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(component, flags, dir_fd=parent_fd)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ProjectFilePathError("project path component is not a directory")
        return descriptor
    except ProjectFileError:
        raise
    except OSError as exc:
        raise ProjectFilePathError(
            "project path contains a missing or symbolic-link directory"
        ) from exc


def _open_or_create_directory(parent_fd: int, component: str) -> int:
    _validate_path_component(component, label="directory")
    try:
        return _open_existing_directory(parent_fd, component)
    except ProjectFilePathError:
        try:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ProjectFilePathError(
                "project upload directory could not be created safely"
            ) from exc
        return _open_existing_directory(parent_fd, component)


def _validate_path_component(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 255
    ):
        raise ProjectFilePathError(f"{label} is not a safe path component")
    return value


def _no_follow_flag() -> int:
    try:
        return os.O_NOFOLLOW
    except AttributeError as exc:
        raise ProjectFilePathError(
            "safe project file access is unavailable on this host"
        ) from exc
