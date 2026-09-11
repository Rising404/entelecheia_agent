"""将现行轨迹存储导出为确定性的私有 JSON artifact。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .presentation import render_trajectory
from .store import TrajectoryStore


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    """一次完整导出的内容和可核验文件身份。"""

    path: Path
    sha256: str
    byte_count: int
    snapshot: dict[str, Any]


def export_trajectory(
    database: Path | str,
    destination: Path | str,
    *,
    format: Literal["json", "markdown"] = "json",
) -> TrajectoryArtifact:
    """只读获取同一完整快照，原子导出机器 JSON 或人类 Markdown 阅读投影。"""

    if format not in {"json", "markdown"}:
        raise ValueError("trajectory export format must be json or markdown")
    snapshot = TrajectoryStore(database).read_all()
    payload = (
        _canonical_bytes(snapshot) if format == "json"
        else render_trajectory(snapshot).encode("utf-8")
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        _fsync_directory(target.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return TrajectoryArtifact(
        path=target,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        snapshot=snapshot,
    )


def _canonical_bytes(snapshot: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["TrajectoryArtifact", "export_trajectory"]
