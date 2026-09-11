"""在一个冻结工作区内执行有界文本文件写入。

本模块特意作为 ``workspace_tools`` 的小型有副作用配套组件。它不读取全局工作区配置、
不复用已退役的 ``path_policy``，也不依赖 Runtime 批准状态。Host 会绑定
:class:`FrozenWorkspaceToolBoundary`；随后策略会在调用已注册处理器前，
授权声明的工作区效果。

处理器只接受工作区相对 UTF-8 文本路径。绝不会隐式创建不存在的父目录；覆盖写入会先在
目标目录中暂存，再原子替换。后者可避免普通写入中途失败时暴露被部分截断的文件。
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from ...configuration.paths import deny_reason
from ...workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..execution import ToolBusinessFailure
from ..registration import ToolExecutionProfile, ToolRegistration
from .workspace_tools import FrozenWorkspaceToolBoundary


WORKSPACE_WRITE_TOOL_CONTRACT_VERSION = "workspace-write-v1"
WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION = "1"
WORKSPACE_WRITE_TOOL_ID = "write_workspace_file"
WORKSPACE_WRITE_TOOL_IDS = (WORKSPACE_WRITE_TOOL_ID,)
WORKSPACE_WRITE_SOURCE_ID = "personagraph.workspace.write"
WORKSPACE_WRITE_SOURCE_DISPLAY_NAME = "Frozen workspace text writer"

MAX_PATH_CHARS = 512
MAX_CONTENT_BYTES = 128 * 1024
MAX_CONTENT_CHARS = MAX_CONTENT_BYTES

_WRITE_MODES = frozenset({"overwrite", "append"})
_TEMPORARY_PREFIX = ".personagraph-write-"


def build_workspace_write_tool_registrations(
    boundary: FrozenWorkspaceToolBoundary,
) -> tuple[ToolRegistration, ...]:
    """为冻结工作区构建唯一的显式文本写入能力。

    此配置声明先读取元数据，再执行受保护的 ``UPDATE``。可以创建缺失目标，
    但当前静态效果分类无法让操作取决于磁盘存在性检查。因此使用 ``UPDATE`` 作为
    保守的受保护写入类别；结果始终记录此次调用是否创建了新文件。
    """

    if not isinstance(boundary, FrozenWorkspaceToolBoundary):
        raise TypeError("boundary must be a FrozenWorkspaceToolBoundary")

    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id=WORKSPACE_WRITE_SOURCE_ID,
        display_name=WORKSPACE_WRITE_SOURCE_DISPLAY_NAME,
    )
    return (
        ToolRegistration(
            spec=build_workspace_write_tool_spec(),
            implementation_version=WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded_write_handler(boundary),
            effect_profile=build_workspace_write_effect_profile(
                default_scope=str(boundary.root),
            ),
            execution_profile=build_workspace_write_execution_profile(),
        ),
    )


def build_workspace_write_tool_spec() -> ToolSpec:
    """Return the workspace-independent model contract for the writer."""

    return ToolSpec(
        tool_id=WORKSPACE_WRITE_TOOL_ID,
        contract_version=WORKSPACE_WRITE_TOOL_CONTRACT_VERSION,
        name="Write a workspace text file",
        description=(
            "Create or modify one UTF-8 text file below the frozen working "
            "directory. Paths must be relative and cannot escape the workspace. "
            "Use overwrite to replace the whole file or append to add a text "
            "chunk. Missing parent directories are rejected unless "
            "create_parents is explicitly true. This tool never merges content "
            "or reads the existing file body; inspect it first when that matters."
        ),
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        catalog_tags=("file", "write"),
    )


def build_workspace_write_execution_profile() -> ToolExecutionProfile:
    """Return the stable execution limits for the non-retryable writer."""

    return ToolExecutionProfile(
        default_timeout_s=10.0,
        hard_timeout_s=20.0,
        max_output_bytes=16_000,
        # Append can duplicate content, while overwrite may commit immediately
        # before a process-level interruption.
        max_transparent_retries=0,
        concurrency_class="workspace_write",
    )


def build_workspace_write_effect_profile(
    *,
    default_scope: str,
) -> ToolEffectProfile:
    """Return the exact read-before-write effect shape for one scope."""

    if not isinstance(default_scope, str) or not default_scope:
        raise ValueError("default_scope must not be empty")
    common = {
        "resource": EffectResource.FILESYSTEM,
        "scope_kind": EffectScopeKind.WORKSPACE,
        "default_scope": default_scope,
        "resource_argument": "path",
    }
    return ToolEffectProfile(
        (
            EffectDescriptor(
                action=EffectAction.READ,
                data_egress=DataEgress.METADATA,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
                **common,
            ),
            EffectDescriptor(
                action=EffectAction.UPDATE,
                data_egress=DataEgress.NONE,
                idempotency=Idempotency.NOT_IDEMPOTENT,
                reversibility=Reversibility.IRREVERSIBLE,
                **common,
            ),
        )
    )


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "content"],
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_PATH_CHARS,
                "description": "File path relative to the frozen workspace root.",
            },
            "content": {
                "type": "string",
                "maxLength": MAX_CONTENT_CHARS,
                "description": (
                    "UTF-8 text to write. The encoded content is limited to "
                    f"{MAX_CONTENT_BYTES} bytes per call."
                ),
            },
            "mode": {
                "enum": ["overwrite", "append"],
                "default": "overwrite",
                "description": "overwrite replaces content; append adds a text chunk.",
            },
            "create_parents": {
                "type": "boolean",
                "default": False,
                "description": "Create missing parent directories only when explicitly true.",
            },
        },
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "path",
            "mode",
            "created",
            "overwrote",
            "previous_size",
            "new_size",
            "written_bytes",
            "content_sha256",
            "created_parent_count",
        ],
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": MAX_PATH_CHARS},
            "mode": {"enum": ["overwrite", "append"]},
            "created": {"type": "boolean"},
            "overwrote": {"type": "boolean"},
            "previous_size": {"type": ["integer", "null"], "minimum": 0},
            "new_size": {"type": "integer", "minimum": 0},
            "written_bytes": {"type": "integer", "minimum": 0, "maximum": MAX_CONTENT_BYTES},
            "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "created_parent_count": {"type": "integer", "minimum": 0},
        },
    }


def _guarded_write_handler(
    boundary: FrozenWorkspaceToolBoundary,
):
    """转换稳定本地失败，同时保留权威失败。"""

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            boundary.require_current_root()
            result = _write_text_file(boundary, payload)
            boundary.require_current_root()
            return result
        except ToolBusinessFailure:
            raise
        except UnicodeError as exc:
            raise ToolBusinessFailure(
                "invalid_request", "content must be valid UTF-8 text."
            ) from exc
        except PermissionError as exc:
            raise ToolBusinessFailure(
                "permission_denied", "The workspace path is not writable."
            ) from exc
        except OSError as exc:
            raise _business_failure_for_os_error(exc) from exc

    return run


def _write_text_file(
    boundary: FrozenWorkspaceToolBoundary,
    payload: dict[str, Any],
) -> dict[str, Any]:
    relative_path, parts = _relative_path(payload.get("path"))
    content = _content_bytes(payload.get("content"))
    mode = _write_mode(payload.get("mode"))
    create_parents = _create_parents(payload.get("create_parents"))

    root = boundary.require_current_root()
    target = root.joinpath(*parts)
    try:
        reserved = is_reserved_workspace_path(root, target)
    except ReservedWorkspacePathError as exc:
        raise ToolBusinessFailure(
            "workspace_path_blocked", "path must remain inside the user workspace."
        ) from exc
    if reserved:
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "The agent-managed private workspace area is not writable by this tool.",
        )
    if deny_reason(target) is not None:
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "Workspace policy does not permit writing that path.",
        )

    parent_fd, created_parent_count = _open_parent_directory(
        boundary,
        parts,
        create_parents=create_parents,
    )
    try:
        previous = _regular_target_status(parent_fd, parts[-1])
        previous_size = previous.st_size if previous is not None else None
        if mode == "overwrite":
            _atomic_overwrite(parent_fd, parts[-1], content, previous)
        else:
            _append_text(parent_fd, parts[-1], content, previous)
        final = _regular_target_status(parent_fd, parts[-1])
        if final is None:
            raise ToolBusinessFailure(
                "workspace_write_failed", "The written file is no longer available."
            )
    finally:
        os.close(parent_fd)

    _register_agent_output(root, relative_path)

    return {
        "path": relative_path,
        "mode": mode,
        "created": previous is None,
        "overwrote": mode == "overwrite" and previous is not None,
        "previous_size": previous_size,
        "new_size": final.st_size,
        "written_bytes": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "created_parent_count": created_parent_count,
    }


def _register_agent_output(root: Path, relative_path: str) -> None:
    """将写入字节追加到绑定项目的文件版本账本。"""

    from ...workspace.storage.context import current as current_project_documents
    from ...workspace.files import FileSource, ProjectFileError, WorkspaceFileAuthority

    database = current_project_documents()
    if database is None:
        return
    if root.resolve() != database.project_root:
        raise ToolBusinessFailure(
            "workspace_authority_changed",
            "The frozen workspace no longer matches the bound project.",
        )
    try:
        WorkspaceFileAuthority(database).register_path(
            relative_path,
            source=FileSource.AGENT_OUTPUT,
        )
    except ProjectFileError as exc:
        raise ToolBusinessFailure(
            "workspace_write_failed",
            "The written file could not be registered in the project.",
        ) from exc


def _relative_path(raw: Any) -> tuple[str, tuple[str, ...]]:
    if not isinstance(raw, str):
        raise ToolBusinessFailure("invalid_request", "path must be a string.")
    text = raw.strip()
    if not text:
        raise ToolBusinessFailure("invalid_request", "path must not be empty.")
    if len(text) > MAX_PATH_CHARS or "\x00" in text:
        raise ToolBusinessFailure("invalid_request", "path is invalid or too long.")
    candidate = Path(text)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "path must remain relative to the frozen workspace.",
        )
    parts = tuple(part for part in candidate.parts if part not in {".", ""})
    if not parts:
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "path must name a file below the frozen workspace.",
        )
    return Path(*parts).as_posix(), parts


def _content_bytes(raw: Any) -> bytes:
    if not isinstance(raw, str):
        raise ToolBusinessFailure("invalid_request", "content must be a string.")
    if "\x00" in raw:
        raise ToolBusinessFailure("invalid_request", "content must be plain text.")
    content = raw.encode("utf-8", "strict")
    if len(content) > MAX_CONTENT_BYTES:
        raise ToolBusinessFailure(
            "content_too_large",
            f"content exceeds the {MAX_CONTENT_BYTES}-byte write limit.",
        )
    return content


def _write_mode(raw: Any) -> str:
    if raw is None:
        return "overwrite"
    if not isinstance(raw, str) or raw not in _WRITE_MODES:
        raise ToolBusinessFailure(
            "invalid_request", "mode must be overwrite or append."
        )
    return raw


def _create_parents(raw: Any) -> bool:
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ToolBusinessFailure(
            "invalid_request", "create_parents must be a boolean."
        )
    return raw


def _open_parent_directory(
    boundary: FrozenWorkspaceToolBoundary,
    parts: tuple[str, ...],
    *,
    create_parents: bool,
) -> tuple[int, int]:
    """使用相对描述符且不跟随链接的遍历方式打开目标父目录。"""

    fd = _open_frozen_root_directory(boundary)
    created = 0
    try:
        for part in parts[:-1]:
            try:
                next_fd = _open_directory_at(fd, part)
            except FileNotFoundError:
                if not create_parents:
                    raise ToolBusinessFailure(
                        "parent_directory_missing",
                        "A parent directory is missing; set create_parents to true to create it.",
                    ) from None
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
            # 并发创建者赢得了竞争。下方不跟随链接的打开操作仍会拒绝符号链接或非目录。
                    pass
                else:
                    created += 1
                next_fd = _open_directory_at(fd, part)
            os.close(fd)
            fd = next_fd
        return fd, created
    except Exception:
        os.close(fd)
        raise


def _open_directory_path(path: Path) -> int:
    return os.open(str(path), _directory_open_flags())


def _open_frozen_root_directory(boundary: FrozenWorkspaceToolBoundary) -> int:
    """获取实际冻结目录，而不只是其当前路径名。"""

    fd = _open_directory_path(boundary.root)
    try:
        identity = os.fstat(fd)
        if (
            not stat.S_ISDIR(identity.st_mode)
            or int(identity.st_dev) != boundary.root_device
            or int(identity.st_ino) != boundary.root_inode
        ):
            raise ToolBusinessFailure(
                "workspace_authority_changed",
                "The frozen workspace directory is no longer the authorized directory.",
            )
        return fd
    except Exception:
        os.close(fd)
        raise


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(name, _directory_open_flags(), dir_fd=parent_fd)


def _directory_open_flags() -> int:
    try:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _close_on_exec_flag()
    except AttributeError as exc:
        raise ToolBusinessFailure(
            "safe_write_unavailable",
            "This host does not support safe workspace write traversal.",
        ) from exc


def _regular_target_status(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        status = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(status.st_mode):
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "Writing through symbolic links is not permitted.",
        )
    if not stat.S_ISREG(status.st_mode):
        raise ToolBusinessFailure(
            "target_not_regular_file",
            "path must name a regular workspace file.",
        )
    return status


def _atomic_overwrite(
    parent_fd: int,
    name: str,
    content: bytes,
    previous: os.stat_result | None,
) -> None:
    """写入同目录临时文件，并原子替换目标。"""

    temporary_name = f"{_TEMPORARY_PREFIX}{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | _close_on_exec_flag()
    )
    temporary_created = False
    try:
        fd = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        temporary_created = True
        try:
            if previous is not None:
            # 否则，原子替换会把现有脚本或源文件原本的普通权限改为 0600。
                os.fchmod(fd, stat.S_IMODE(previous.st_mode) & 0o777)
            _write_all(fd, content)
        finally:
            os.close(fd)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_created = False
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
        # 不要让尽力清理掩盖原始写入失败；该名称是随机的，且仍位于边界内。
                pass


def _append_text(
    parent_fd: int,
    name: str,
    content: bytes,
    previous: os.stat_result | None,
) -> None:
    if previous is not None and previous.st_nlink > 1:
    # 追加操作会编辑 inode，而非替换目录条目。拒绝可能通过硬链接逃逸到工作区外部数据的情况。
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "Appending to a multiply-linked file is not permitted.",
        )
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | _close_on_exec_flag()
    )
    fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink > 1:
            raise ToolBusinessFailure(
                "workspace_path_blocked",
                "path must remain an unlinked regular workspace file.",
            )
        _write_all(fd, content)
    finally:
        os.close(fd)


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "failed to write workspace file")
        view = view[written:]


def _close_on_exec_flag() -> int:
    return getattr(os, "O_CLOEXEC", 0)


def _business_failure_for_os_error(exc: OSError) -> ToolBusinessFailure:
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return ToolBusinessFailure(
            "permission_denied", "The workspace path is not writable."
        )
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return ToolBusinessFailure(
            "workspace_path_blocked",
            "path must remain inside the frozen workspace.",
        )
    return ToolBusinessFailure(
        "workspace_write_failed", "The workspace write could not be completed."
    )


__all__ = [
    "MAX_CONTENT_BYTES",
    "MAX_CONTENT_CHARS",
    "MAX_PATH_CHARS",
    "WORKSPACE_WRITE_SOURCE_DISPLAY_NAME",
    "WORKSPACE_WRITE_SOURCE_ID",
    "WORKSPACE_WRITE_TOOL_CONTRACT_VERSION",
    "WORKSPACE_WRITE_TOOL_ID",
    "WORKSPACE_WRITE_TOOL_IDS",
    "WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION",
    "build_workspace_write_effect_profile",
    "build_workspace_write_execution_profile",
    "build_workspace_write_tool_spec",
    "build_workspace_write_tool_registrations",
]
