"""输入文件分类的稳定合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FileKind(StrEnum):
    """用于选择输入处理策略的粗粒度文件类型族。"""

    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedType:
    """由文件字节证明的媒体类型与处理类型族。"""

    media_type: str
    kind: FileKind
    extension: str


__all__ = ["DetectedType", "FileKind"]
