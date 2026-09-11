"""智能体拥有的 ``.personagraph`` 目录树的路径策略。

绑定工作区根目录本身仍对用户可见。只有该精确根目录下的 ``.personagraph``
子目录及其后代保留给主机管理状态。此区分有意依据相对根目录的路径，而非文件名
子串：名为 ``.personagraph-notes.md`` 的普通文件并非私有状态。

该模块是分类边界，而不是 I/O 原语。它会在分类候选前拒绝有歧义的路径和符号
链接组件；但随后打开文件的调用方仍必须自行执行禁止跟随链接的操作，并在该
I/O 边界重新验证绑定根目录身份。
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat


RESERVED_DIRECTORY_NAME = ".personagraph"


class ReservedWorkspacePathError(ValueError):
    """候选路径无法在绑定工作区根目录下安全分类。

    ``code`` 有意保持稳定，使工具或 API 适配器无需匹配错误文本，就能把不安全
    输入转换为自身的带类型拒绝。
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def normalize_workspace_relative_path(
    candidate: str | os.PathLike[str],
) -> PurePosixPath:
    """返回规范、安全且相对工作区的路径。

    返回路径使用 POSIX 分隔符，使其在所有主机平台上都能安全成为稳定的模型可见
    相对路径。``.`` 表示工作区根目录；绝对路径、路径穿越、盘符限定形式与反斜杠
    分隔的模型输入均会被拒绝，而不是由不同平台作不同解释。
    """

    raw = _path_text(candidate)
    native = Path(raw)
    if native.is_absolute() or PureWindowsPath(raw).drive:
        raise ReservedWorkspacePathError("absolute_path")
    if "\\" in raw:
        raise ReservedWorkspacePathError("invalid_path")
    return _normalized_posix_relative(raw)


def workspace_relative_path(
    root: str | os.PathLike[str],
    candidate: str | os.PathLike[str],
) -> PurePosixPath:
    """安全地将 ``candidate`` 表示为相对于现有绑定 ``root`` 的路径。

    相对候选路径解释为位于 ``root`` 下。绝对候选只有在词法上位于精确绑定根目录
    下时才接受；接受同一文件系统位置的另一种写法会让策略依赖符号链接解析。
    根目录下的现有符号链接组件即使当前指回根内也会被拒绝。允许不存在的后代，
    以便调用方在创建前分类计划输出路径。
    """

    bound_root = _bound_root(root)
    raw = _path_text(candidate)
    native = Path(raw)
    if native.is_absolute():
        _reject_foreign_windows_path(raw)
        _reject_parent_segments(native)
        try:
            relative = native.relative_to(bound_root)
        except ValueError as exc:
            raise ReservedWorkspacePathError("outside_bound_root") from exc
        normalized = _normalized_posix_relative(relative.as_posix())
    else:
        normalized = normalize_workspace_relative_path(raw)

    _reject_symlink_components(bound_root, normalized)
    return normalized


def is_reserved_workspace_path(
    root: str | os.PathLike[str],
    candidate: str | os.PathLike[str],
) -> bool:
    """判断 ``candidate`` 是否为 ``root/.personagraph`` 或其子路径。

    绑定根目录本身返回 ``False``。无效、根外、穿越或含符号链接的路径会抛出
    :class:`ReservedWorkspacePathError`，防止调用方误把安全检查失败当作普通的
    非保留路径。
    """

    relative = workspace_relative_path(root, candidate)
    # 主机常用的文件系统（尤其 macOS 默认卷）可能不区分大小写。将所有大小写形式
    # 都视为保留路径，避免请求 `.PERSONAGRAPH` 绕过磁盘上的精确目录树。
    return (
        bool(relative.parts)
        and relative.parts[0].casefold() == RESERVED_DIRECTORY_NAME
    )


def _bound_root(root: str | os.PathLike[str]) -> Path:
    raw = _path_text(root)
    native = Path(raw)
    if not native.is_absolute():
        raise ReservedWorkspacePathError("invalid_bound_root")
    _reject_foreign_windows_path(raw)
    _reject_parent_segments(native)
    try:
        metadata = native.lstat()
    except OSError as exc:
        raise ReservedWorkspacePathError("invalid_bound_root") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ReservedWorkspacePathError("symlink_path")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReservedWorkspacePathError("invalid_bound_root")
    return native


def _path_text(value: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ReservedWorkspacePathError("invalid_path") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ReservedWorkspacePathError("invalid_path")
    return raw


def _normalized_posix_relative(raw: str) -> PurePosixPath:
    if any(part == ".." for part in raw.split("/")):
        raise ReservedWorkspacePathError("path_traversal")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ReservedWorkspacePathError("absolute_path")
    return path


def _reject_parent_segments(path: Path) -> None:
    if ".." in path.parts:
        raise ReservedWorkspacePathError("path_traversal")


def _reject_foreign_windows_path(raw: str) -> None:
    """在 POSIX 上拒绝 Windows 路径写法，而不是将其视为文件名。"""

    if os.name != "nt" and ("\\" in raw or PureWindowsPath(raw).drive):
        raise ReservedWorkspacePathError("invalid_path")


def _reject_symlink_components(root: Path, relative: PurePosixPath) -> None:
    """若现有路径组件需要跟随链接，则故障关闭。"""

    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # 缺失组件尚不可能包含链接；没有它，后代也无法存在，因此其余计划
            # 路径可以安全分类。
            return
        except OSError as exc:
            raise ReservedWorkspacePathError("unreadable_path") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ReservedWorkspacePathError("symlink_path")


__all__ = [
    "RESERVED_DIRECTORY_NAME",
    "ReservedWorkspacePathError",
    "is_reserved_workspace_path",
    "normalize_workspace_relative_path",
    "workspace_relative_path",
]
