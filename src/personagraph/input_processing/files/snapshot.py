"""输入文件在有界读取前后的稳定字节快照。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


MAX_DOCUMENT_FILE_BYTES = 64 * 1024 * 1024


class SourceChangedDuringReadError(RuntimeError):
    """文件在计算字节指纹期间发生了变化。"""


class SourceSizeLimitError(RuntimeError):
    """源超过有界文档摄取字节策略。"""


@dataclass(frozen=True)
class SourceFingerprint:
    """强源内容观测及低成本诊断元数据。"""

    sha256: str
    size_bytes: int
    mtime_ns: int

    def record(self) -> dict[str, int | str]:
        return asdict(self)


def fingerprint_file(
    path: Path,
    *,
    max_bytes: int = MAX_DOCUMENT_FILE_BYTES,
) -> SourceFingerprint:
    """为一个普通文件计算哈希，并拒绝在读取途中变化的源。

    ``size`` 与 ``mtime_ns`` 让诊断更有用，但 SHA-256 摘要相等才是当前内容的真实证明。
    """

    before = path.stat()
    if before.st_size > max_bytes:
        raise SourceSizeLimitError(str(path))
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as source:
        while block := source.read(1_048_576):
            observed_bytes += len(block)
            if observed_bytes > max_bytes:
                raise SourceSizeLimitError(str(path))
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SourceChangedDuringReadError(str(path))
    return SourceFingerprint(
        sha256=digest.hexdigest(),
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
    )
