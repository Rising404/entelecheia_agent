"""Workspace 根目录绑定、私有布局与保留路径策略。"""

from .contracts import WorkspaceLayout, WorkspaceLayoutError, WorkspaceRootIdentity
from .layout import (
    LAYOUT_DIRECTORY_NAME,
    MANIFEST_FILE_NAME,
    MANIFEST_SCHEMA_VERSION,
    OUTPUT_DIRECTORY_NAME,
    STAGING_DIRECTORY_NAME,
    ensure_or_open_layout,
)
from .reserved_paths import (
    RESERVED_DIRECTORY_NAME,
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
    normalize_workspace_relative_path,
    workspace_relative_path,
)

__all__ = [
    "LAYOUT_DIRECTORY_NAME",
    "MANIFEST_FILE_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "OUTPUT_DIRECTORY_NAME",
    "RESERVED_DIRECTORY_NAME",
    "STAGING_DIRECTORY_NAME",
    "ReservedWorkspacePathError",
    "WorkspaceLayout",
    "WorkspaceLayoutError",
    "WorkspaceRootIdentity",
    "ensure_or_open_layout",
    "is_reserved_workspace_path",
    "normalize_workspace_relative_path",
    "workspace_relative_path",
]
