"""Workspace 根目录绑定与私有布局的稳定合同。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class WorkspaceLayoutError(ValueError):
    """无法安全识别或配置 Workspace 私有布局。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WorkspaceRootIdentity:
    """根目录打开时的文件系统身份。"""

    device: int
    inode: int

    @classmethod
    def from_stat(cls, facts: os.stat_result) -> "WorkspaceRootIdentity":
        return cls(device=facts.st_dev, inode=facts.st_ino)

    def to_manifest(self) -> dict[str, int]:
        return {"device": self.device, "inode": self.inode}


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """一个安全打开的 Workspace 布局及其私有目录。"""

    root: Path
    root_identity: WorkspaceRootIdentity
    workspace_id: str
    layout_root: Path
    manifest_path: Path
    output_root: Path
    staging_root: Path
    output_dir: Path
    staging_dir: Path
    manifest_created: bool


__all__ = ["WorkspaceLayout", "WorkspaceLayoutError", "WorkspaceRootIdentity"]
