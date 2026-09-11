"""格式观测工具的冻结来源文件权威信息。

本模块持有本地格式读取器与可选外部视觉分析共享的安全链：模型路径在一个冻结工作区之下解析，
选定来源受到边界约束并生成指纹，且在观测后会再次检查其标识。它特意不了解任何文档如何读取、
渲染或分析。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...configuration.paths import deny_reason
from ...workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)
from ..execution import ToolBusinessFailure
from ..workspace.workspace_tools import FrozenWorkspaceToolBoundary


MAX_PATH_CHARS = 512
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _resolve_file(
    boundary: FrozenWorkspaceToolBoundary,
    payload: Mapping[str, Any],
    allowed_suffixes: frozenset[str],
) -> tuple[Path, str]:
    """解析一个允许的工作区相对来源文件。"""

    root = boundary.require_current_root()
    raw = payload.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ToolBusinessFailure("invalid_request", "path must be a non-empty string.")
    text = raw.strip()
    if len(text) > MAX_PATH_CHARS or "\x00" in text:
        raise ToolBusinessFailure("invalid_request", "path is invalid or too long.")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ToolBusinessFailure(
            "workspace_path_blocked", "path must remain inside the frozen workspace."
        )
    try:
        target = (root / candidate).resolve(strict=True)
        relative = target.relative_to(root)
    except FileNotFoundError as exc:
        raise ToolBusinessFailure(
            "file_unavailable", "No file exists at that workspace-relative path."
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolBusinessFailure(
            "workspace_path_blocked", "path must remain inside the frozen workspace."
        ) from exc
    if not target.is_file():
        raise ToolBusinessFailure(
            "file_unavailable", "The workspace-relative path is not a file."
        )
    try:
        reserved = is_reserved_workspace_path(root, target)
    except ReservedWorkspacePathError as exc:
        raise ToolBusinessFailure(
            "workspace_path_blocked", "path must remain inside the user workspace."
        ) from exc
    if reserved:
        raise ToolBusinessFailure(
            "workspace_path_blocked",
            "The agent-managed private workspace area is not a source file.",
        )
    if deny_reason(target) is not None:
        raise ToolBusinessFailure(
            "workspace_path_blocked", "Workspace policy does not permit reading that path."
        )
    boundary.require_current_root()
    suffix = target.suffix.lower()
    if suffix not in allowed_suffixes:
        raise ToolBusinessFailure(
            "format_mismatch",
            "The selected tool does not handle this file format.",
            {
                "actual_suffix": suffix,
                "allowed_suffixes": sorted(allowed_suffixes),
            },
        )
    return target, relative.as_posix()


def _guard_workspace_handler(
    boundary: FrozenWorkspaceToolBoundary,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """如果冻结目录在派发期间发生变化，则采用失败关闭策略。"""

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        boundary.require_current_root()
        result = handler(payload)
        boundary.require_current_root()
        return result

    return run


def _source_identity(target: Path) -> dict[str, Any]:
    """返回有界来源指纹，并拒绝不稳定输入。"""

    try:
        size = target.stat().st_size
    except PermissionError as exc:
        raise ToolBusinessFailure(
            "permission_denied", "The source file cannot be read."
        ) from exc
    except OSError as exc:
        raise ToolBusinessFailure("file_unavailable", "The source file is unavailable.") from exc
    if size < 1:
        raise ToolBusinessFailure("empty_source", "The source file is empty.")
    if size > MAX_SOURCE_BYTES:
        raise ToolBusinessFailure(
            "source_too_large",
            "The source exceeds the observation byte limit.",
            {"byte_count": size, "max_byte_count": MAX_SOURCE_BYTES},
        )
    digest = hashlib.sha256()
    read = 0
    try:
        with target.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                read += len(chunk)
                if read > MAX_SOURCE_BYTES:
                    raise ToolBusinessFailure(
                        "source_too_large",
                        "The source changed beyond the observation byte limit.",
                    )
                digest.update(chunk)
    except ToolBusinessFailure:
        raise
    except PermissionError as exc:
        raise ToolBusinessFailure(
            "permission_denied", "The source file cannot be read."
        ) from exc
    except OSError as exc:
        raise ToolBusinessFailure("file_unavailable", "The source file is unavailable.") from exc
    if read != size:
        raise ToolBusinessFailure(
            "source_changed_during_read", "The source changed while it was being observed."
        )
    return {
        "suffix": target.suffix.lower(),
        "byte_count": size,
        "sha256": digest.hexdigest(),
    }


def _assert_unchanged(target: Path, expected: Mapping[str, Any]) -> None:
    """拒绝根据生成指纹后又发生变化的来源所构建的输出。"""

    actual = _source_identity(target)
    if actual != dict(expected):
        raise ToolBusinessFailure(
            "source_changed_during_read",
            "The source changed while it was being observed; no result was accepted.",
        )
