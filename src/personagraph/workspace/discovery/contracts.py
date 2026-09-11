"""目录认知的结果契约。

这些类型让调用方无需解析展示字符串，也无需猜测列表是否完整。有两项性质比
形态本身更重要，二者都来自 `24/001 D5`：

*截断是显式的。* 静默限制输出的工具会让模型误以为已经看到全部内容，进而对
从未完整扫描的目录断言“没有这个文件”。

*失败可与空结果区分。* “没有匹配项”与“三个文件无法读取”是不同事实，因此
不可读输入会按原因计数，而不是丢进同一个空列表。

此处有意只导入标准库；边界测试会强制这一点，使这些契约可供任意层使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SkipReason(StrEnum):
    """路径可见但不可用的封闭原因集合。

    之所以封闭，是因为自由文本最终会进入提示，而模型无法可靠处理从未见过的
    原因表述。
    """

    PERMISSION_DENIED = "permission_denied"
    TOO_LARGE = "too_large"
    DECODE_FAILED = "decode_failed"
    IO_ERROR = "io_error"
    DENIED_BY_POLICY = "denied_by_policy"


class MatchKind(StrEnum):
    """查找请求的哪一部分产生了命中。"""

    NAME = "name"
    CONTENT = "content"


@dataclass(frozen=True)
class DirEntry:
    """按所含内容汇总的单个目录。

    此处采用聚合而非列举：目录行只消耗少量词元，文件列表却可能消耗数千；
    它要回答的是“这里有什么”，而不是“列出每个文件”。
    """

    rel_path: str
    files: int
    bytes_total: int
    kinds: tuple[tuple[str, int], ...] = ()
    newest_mtime_ns: int | None = None

    def __post_init__(self) -> None:
        if self.files < 0 or self.bytes_total < 0:
            raise ValueError("counts must not be negative")


@dataclass(frozen=True)
class DirOverview:
    """有意设置边界的单个目录树结构。"""

    root: str
    depth: int
    entries: tuple[DirEntry, ...]
    total_files: int
    total_dirs: int
    truncated: bool = False
    omitted_dirs: int = 0
    scan_truncated: bool = False
    skipped: tuple[tuple[SkipReason, int], ...] = ()

    def __post_init__(self) -> None:
        if self.truncated and self.omitted_dirs <= 0:
            raise ValueError("a truncated overview must report omitted_dirs")
        if not self.truncated and self.omitted_dirs:
            raise ValueError("omitted_dirs requires truncated=True")


@dataclass(frozen=True)
class FindHit:
    """单个定位到的文件；若为文本匹配，则包含命中的行。"""

    rel_path: str
    match: MatchKind
    size_bytes: int
    mtime_ns: int
    line: int | None = None
    snippet: str | None = None

    def __post_init__(self) -> None:
        if self.match is MatchKind.NAME and (self.line is not None or self.snippet):
            raise ValueError("a filename match has no line or snippet")
        if self.match is MatchKind.CONTENT and self.line is None:
            raise ValueError("a content match must carry its line number")


@dataclass(frozen=True)
class FindResult:
    """一页命中结果，以及理解未展示内容所需的全部信息。"""

    hits: tuple[FindHit, ...]
    total_matched: int
    offset: int
    truncated: bool
    scanned_files: int
    elapsed_ms: int
    scan_truncated: bool = False
    skipped: tuple[tuple[SkipReason, int], ...] = ()
    ignore_rules_applied: bool = True

    @property
    def returned(self) -> int:
        return len(self.hits)

    def __post_init__(self) -> None:
        if self.hits and self.total_matched < len(self.hits) + self.offset:
            raise ValueError("total_matched cannot be smaller than what was returned")
        expected = self.offset + len(self.hits) < self.total_matched
        if self.truncated != expected:
            raise ValueError("truncated must state whether hits remain beyond this page")


def tally(reasons: dict[SkipReason, int]) -> tuple[tuple[SkipReason, int], ...]:
    """以稳定顺序呈现跳过计数。

    必须排序，因为对未变目录树运行两次应产生字节级相同结果；字典迭代顺序会把
    扫描顺序泄漏到输出。
    """

    return tuple(sorted(((r, n) for r, n in reasons.items() if n > 0), key=lambda x: x[0].value))


__all__ = [
    "DirEntry",
    "DirOverview",
    "FindHit",
    "FindResult",
    "MatchKind",
    "SkipReason",
    "tally",
]
