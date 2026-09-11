"""单个绑定工作区拥有的私有文件系统布局。

``.personagraph`` 有意属于文件系统职责，而非会话或运行时职责。该模块只在已选
根目录下建立并识别小型私有目录树；它不在存储中绑定会话、不检查用户文件，
也不发布输出。将这些效果排除在模块外，使绑定工作流可以恢复中断的文件系统
步骤，而不假装它与 SQLite 构成同一事务。

根目录下每个名称都相对目录描述符以 ``O_NOFOLLOW`` 打开。该布局是私有控制平面，
若接受符号链接，后续输出写入器就可能越过工作区边界。
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from .contracts import WorkspaceLayout, WorkspaceLayoutError, WorkspaceRootIdentity


LAYOUT_DIRECTORY_NAME = ".personagraph"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
OUTPUT_DIRECTORY_NAME = "output"
STAGING_DIRECTORY_NAME = "staging"

# 该值已经写入 manifest，属于持久格式身份；包迁移不能重写现有 Workspace 的来源。
_CREATION_SOURCE = "personagraph.workspace.layout"
_MAX_MANIFEST_BYTES = 64 * 1024
_MANIFEST_TEMP_PREFIX = ".manifest-"
_MANIFEST_TEMP_SUFFIX = ".tmp"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class _Manifest:
    workspace_id: str
    root_identity: WorkspaceRootIdentity
    created_by: str

    def to_json(self) -> dict[str, object]:
        return {
            "created_by": self.created_by,
            "root_identity": self.root_identity.to_manifest(),
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
        }


def ensure_or_open_layout(
    root: str | Path,
    *,
    session_id: str,
    workspace_id: str | None = None,
) -> WorkspaceLayout:
    """为一个会话配置或识别 ``<root>/.personagraph``。

    ``root`` 必须指向现有目录。构建返回值前会将其规范化，并拒绝所给路径中的
    直接或祖先符号链接。可选的 ``workspace_id`` 让后续绑定层能够断言现有清单
    属于预期的精确工作区；省略时，会生成新的不透明 ID 或识别有效现有 ID。

    此处不打开任何用户文件；唯一写入是私有清单与缺失的私有目录。
    """

    canonical_root = _canonical_root(root)
    safe_session_id = _validate_identifier(session_id, field="session_id")
    expected_workspace_id = (
        _validate_identifier(workspace_id, field="workspace_id")
        if workspace_id is not None
        else None
    )

    root_fd = _open_root(canonical_root)
    try:
        root_identity = WorkspaceRootIdentity.from_stat(os.fstat(root_fd))
        layout_fd = _open_or_create_directory(
            root_fd,
            LAYOUT_DIRECTORY_NAME,
            error_code="unsafe_layout",
        )
        try:
            manifest, manifest_created = _open_or_create_manifest(
                layout_fd,
                root_identity=root_identity,
                expected_workspace_id=expected_workspace_id,
            )
            _reject_unexpected_layout_entries(layout_fd)

            output_fd = _open_or_create_directory(
                layout_fd,
                OUTPUT_DIRECTORY_NAME,
                error_code="unsafe_output_root",
            )
            try:
                _open_or_create_directory(
                    output_fd,
                    safe_session_id,
                    error_code="unsafe_session_output",
                )
            finally:
                os.close(output_fd)

            staging_fd = _open_or_create_directory(
                layout_fd,
                STAGING_DIRECTORY_NAME,
                error_code="unsafe_staging_root",
            )
            try:
                _open_or_create_directory(
                    staging_fd,
                    safe_session_id,
                    error_code="unsafe_session_staging",
                )
            finally:
                os.close(staging_fd)
        finally:
            os.close(layout_fd)
    finally:
        os.close(root_fd)

    layout_root = canonical_root / LAYOUT_DIRECTORY_NAME
    output_root = layout_root / OUTPUT_DIRECTORY_NAME
    staging_root = layout_root / STAGING_DIRECTORY_NAME
    return WorkspaceLayout(
        root=canonical_root,
        root_identity=root_identity,
        workspace_id=manifest.workspace_id,
        layout_root=layout_root,
        manifest_path=layout_root / MANIFEST_FILE_NAME,
        output_root=output_root,
        staging_root=staging_root,
        output_dir=output_root / safe_session_id,
        staging_dir=staging_root / safe_session_id,
        manifest_created=manifest_created,
    )


def _canonical_root(root: str | Path) -> Path:
    if isinstance(root, str) and not root.strip():
        raise WorkspaceLayoutError("invalid_root", "workspace root must not be empty")
    try:
        supplied = Path(root).expanduser()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise WorkspaceLayoutError("invalid_root", "workspace root is invalid") from exc
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    _reject_symlink_components(supplied)
    try:
        canonical = supplied.resolve(strict=True)
        facts = os.lstat(canonical)
    except OSError as exc:
        raise WorkspaceLayoutError("invalid_root", "workspace root does not exist") from exc
    if stat.S_ISLNK(facts.st_mode):
        raise WorkspaceLayoutError("unsafe_root", "workspace root must not be a symlink")
    if not stat.S_ISDIR(facts.st_mode):
        raise WorkspaceLayoutError("invalid_root", "workspace root must be a directory")
    return canonical


def _reject_symlink_components(path: Path) -> None:
    """在规范化前拒绝穿越符号链接的所给根路径。

    若先规范化符号链接再继续，调用方会以为绑定了自己提供的名称，而未来替换
    该链接可能把相同请求指向另一目录树。下方描述符操作也保护受管子树；此检查
    保护最初的权威获取。
    """

    cursor = Path(path.anchor)
    parts = path.parts[1:]
    for part in parts:
        if part in ("", "."):
            continue
        cursor /= part
        try:
            facts = os.lstat(cursor)
        except FileNotFoundError:
    # `_canonical_root` 会把缺失的最终路径转换为公开的无效根目录错误；此时尚不能
    # 信任任何后续组件。
            return
        except OSError as exc:
            raise WorkspaceLayoutError("invalid_root", "workspace root is unreadable") from exc
        if stat.S_ISLNK(facts.st_mode):
            raise WorkspaceLayoutError(
                "unsafe_root",
                "workspace root path must not traverse a symlink",
            )


def _open_root(root: Path) -> int:
    try:
        descriptor = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise WorkspaceLayoutError("invalid_root", "workspace root cannot be opened safely") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise WorkspaceLayoutError("invalid_root", "workspace root must be a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise WorkspaceLayoutError(
            "unsupported_platform",
            "workspace layout requires no-follow directory support",
        )
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow


def _file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise WorkspaceLayoutError(
            "unsupported_platform",
            "workspace layout requires no-follow file support",
        )
    return os.O_RDONLY | no_follow


def _open_or_create_directory(parent_fd: int, name: str, *, error_code: str) -> int:
    """打开直接子目录，且绝不跟随链接。"""

    try:
        return _open_existing_directory(parent_fd, name, error_code=error_code)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise WorkspaceLayoutError(error_code, f"managed directory {name!r} is unsafe") from exc

    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        # 另一绑定器可能在首次禁止跟随链接的打开后创建了它；应重新打开并验证，
        # 而不是信任其留下的名称。
        pass
    except OSError as exc:
        raise WorkspaceLayoutError(error_code, f"managed directory {name!r} cannot be created") from exc

    try:
        return _open_existing_directory(parent_fd, name, error_code=error_code)
    except OSError as exc:
        raise WorkspaceLayoutError(error_code, f"managed directory {name!r} is unsafe") from exc


def _open_existing_directory(parent_fd: int, name: str, *, error_code: str) -> int:
    descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise WorkspaceLayoutError(error_code, f"managed path {name!r} is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_manifest(
    layout_fd: int,
    *,
    root_identity: WorkspaceRootIdentity,
    expected_workspace_id: str | None,
) -> tuple[_Manifest, bool]:
    existing = _read_manifest(
        layout_fd,
        root_identity=root_identity,
        expected_workspace_id=expected_workspace_id,
    )
    if existing is not None:
        _discard_interrupted_manifest_temps(
            layout_fd,
            allowed_entries={
                MANIFEST_FILE_NAME,
                OUTPUT_DIRECTORY_NAME,
                STAGING_DIRECTORY_NAME,
            },
        )
        return existing, False

    _discard_interrupted_manifest_temps(layout_fd, allowed_entries=set())
    _reject_unknown_unmanaged_entries(layout_fd)
    new_manifest = _Manifest(
        workspace_id=expected_workspace_id or f"workspace-{uuid.uuid4().hex}",
        root_identity=root_identity,
        created_by=_CREATION_SOURCE,
    )
    published = _publish_new_manifest(layout_fd, new_manifest)
    if published:
        return new_manifest, True

        # 并发配置器赢得了禁止覆盖的创建竞争；其清单只有经过相同严格验证后才具权威。
    winner = _read_manifest(
        layout_fd,
        root_identity=root_identity,
        expected_workspace_id=expected_workspace_id,
    )
    if winner is None:
        raise WorkspaceLayoutError("manifest_race", "manifest disappeared during layout provisioning")
    return winner, False


def _read_manifest(
    layout_fd: int,
    *,
    root_identity: WorkspaceRootIdentity,
    expected_workspace_id: str | None,
) -> _Manifest | None:
    try:
        descriptor = os.open(MANIFEST_FILE_NAME, _file_open_flags(), dir_fd=layout_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise WorkspaceLayoutError("unsafe_manifest", "layout manifest is unsafe") from exc
    try:
        facts = os.fstat(descriptor)
        if not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1:
            raise WorkspaceLayoutError("unsafe_manifest", "layout manifest is not a private regular file")
        if facts.st_size > _MAX_MANIFEST_BYTES:
            raise WorkspaceLayoutError("malformed_manifest", "layout manifest is too large")
        raw = _read_all(descriptor, limit=_MAX_MANIFEST_BYTES)
    finally:
        os.close(descriptor)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceLayoutError("malformed_manifest", "layout manifest is not valid JSON") from exc
    return _validate_manifest(
        decoded,
        root_identity=root_identity,
        expected_workspace_id=expected_workspace_id,
    )


def _validate_manifest(
    decoded: object,
    *,
    root_identity: WorkspaceRootIdentity,
    expected_workspace_id: str | None,
) -> _Manifest:
    if not isinstance(decoded, dict):
        raise WorkspaceLayoutError("malformed_manifest", "layout manifest must be an object")
    expected_keys = {"created_by", "root_identity", "schema_version", "workspace_id"}
    if set(decoded) != expected_keys:
        raise WorkspaceLayoutError("malformed_manifest", "layout manifest has an unsupported shape")
    if decoded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise WorkspaceLayoutError("unsupported_manifest", "layout manifest schema is unsupported")
    try:
        workspace_id = _validate_identifier_value(decoded.get("workspace_id"), field="workspace_id")
        created_by = _validate_identifier_value(decoded.get("created_by"), field="created_by")
    except WorkspaceLayoutError as exc:
        raise WorkspaceLayoutError(
            "malformed_manifest",
            "layout manifest has an invalid identifier",
        ) from exc
    root = decoded.get("root_identity")
    if not isinstance(root, dict) or set(root) != {"device", "inode"}:
        raise WorkspaceLayoutError("malformed_manifest", "layout manifest root identity is malformed")
    device = _validate_identity_integer(root.get("device"), field="device")
    inode = _validate_identity_integer(root.get("inode"), field="inode")
    manifest_identity = WorkspaceRootIdentity(device=device, inode=inode)
    if manifest_identity != root_identity:
        raise WorkspaceLayoutError("foreign_manifest", "layout manifest belongs to another workspace root")
    if expected_workspace_id is not None and workspace_id != expected_workspace_id:
        raise WorkspaceLayoutError("foreign_manifest", "layout manifest belongs to another workspace")
    return _Manifest(
        workspace_id=workspace_id,
        root_identity=manifest_identity,
        created_by=created_by,
    )


def _validate_identity_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkspaceLayoutError("malformed_manifest", f"layout manifest {field} is invalid")
    return value


def _validate_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise WorkspaceLayoutError("invalid_identifier", f"{field} must be a string")
    return _validate_identifier_value(value, field=field)


def _validate_identifier_value(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise WorkspaceLayoutError("invalid_identifier", f"{field} is not a safe opaque identifier")
    return value


def _discard_interrupted_manifest_temps(
    layout_fd: int,
    *,
    allowed_entries: set[str],
) -> None:
    """清理自身中断的清单临时文件，但绝不触碰未知文件。

    崩溃可能发生在原子硬链接发布后、源临时名称解除链接前。不能让这种情况把
    有效布局变成永远无法重新打开的布局。反之，未知同级文件表示该恢复不归我们
    安全所有，因此保持不动，交由下方故障关闭处理。
    """

    names = _listdir(layout_fd)
    temp_names = [name for name in names if _is_manifest_temp(name)]
    if not temp_names:
        return
    if set(names) - set(temp_names) - allowed_entries:
        return
    for name in temp_names:
        try:
            facts = os.stat(name, dir_fd=layout_fd, follow_symlinks=False)
            if not stat.S_ISREG(facts.st_mode):
                raise WorkspaceLayoutError("unsafe_layout", "interrupted manifest entry is unsafe")
            os.unlink(name, dir_fd=layout_fd)
        except WorkspaceLayoutError:
            raise
        except OSError as exc:
            raise WorkspaceLayoutError("unsafe_layout", "interrupted manifest entry is unsafe") from exc


def _reject_unknown_unmanaged_entries(layout_fd: int) -> None:
    names = _listdir(layout_fd)
    if names:
        raise WorkspaceLayoutError(
            "unmanaged_layout",
            "existing .personagraph directory has no valid manifest",
        )


def _reject_unexpected_layout_entries(layout_fd: int) -> None:
    allowed = {MANIFEST_FILE_NAME, OUTPUT_DIRECTORY_NAME, STAGING_DIRECTORY_NAME}
    unexpected = set(_listdir(layout_fd)) - allowed
    if unexpected:
        raise WorkspaceLayoutError("unsafe_layout", "layout contains unexpected top-level entries")


def _publish_new_manifest(layout_fd: int, manifest: _Manifest) -> bool:
    payload = (
        json.dumps(manifest.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary_name = f"{_MANIFEST_TEMP_PREFIX}{uuid.uuid4().hex}{_MANIFEST_TEMP_SUFFIX}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=layout_fd,
        )
        _write_all(descriptor, payload)
        _best_effort_fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
    # link(2) 发布完整写入的 inode 且不替换现有清单；os.replace 则可能接管并发
    # 或外部布局。
            os.link(
                temporary_name,
                MANIFEST_FILE_NAME,
                src_dir_fd=layout_fd,
                dst_dir_fd=layout_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        _best_effort_fsync(layout_fd)
        return True
    except OSError as exc:
        raise WorkspaceLayoutError("manifest_write_failed", "layout manifest could not be published") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=layout_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise WorkspaceLayoutError("manifest_write_failed", "layout manifest cleanup failed") from exc


def _read_all(descriptor: int, *, limit: int) -> bytes:
    parts: list[bytes] = []
    remaining = limit + 1
    while remaining:
        block = os.read(descriptor, min(8192, remaining))
        if not block:
            break
        parts.append(block)
        remaining -= len(block)
    payload = b"".join(parts)
    if len(payload) > limit:
        raise WorkspaceLayoutError("malformed_manifest", "layout manifest is too large")
    return payload


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing workspace manifest")
        view = view[written:]


def _best_effort_fsync(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
    # 某些网络文件系统未实现目录 fsync；禁止覆盖的发布在那里仍能保持进程内原子性。
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def _listdir(descriptor: int) -> list[str]:
    try:
        return list(os.listdir(descriptor))
    except OSError as exc:
        raise WorkspaceLayoutError("unsafe_layout", "managed layout cannot be listed safely") from exc


def _is_manifest_temp(name: str) -> bool:
    return (
        name.startswith(_MANIFEST_TEMP_PREFIX)
        and name.endswith(_MANIFEST_TEMP_SUFFIX)
        and len(name) == len(_MANIFEST_TEMP_PREFIX) + 32 + len(_MANIFEST_TEMP_SUFFIX)
        and all(
            character in "0123456789abcdef"
            for character in name[len(_MANIFEST_TEMP_PREFIX):-len(_MANIFEST_TEMP_SUFFIX)]
        )
    )


__all__ = [
    "LAYOUT_DIRECTORY_NAME",
    "MANIFEST_FILE_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "OUTPUT_DIRECTORY_NAME",
    "STAGING_DIRECTORY_NAME",
    "WorkspaceLayout",
    "WorkspaceLayoutError",
    "WorkspaceRootIdentity",
    "ensure_or_open_layout",
]
