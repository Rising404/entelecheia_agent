"""有界、可分页且显式报告不完整性的 Workspace 目录认知。"""

from .contracts import (
    DirEntry,
    DirOverview,
    FindHit,
    FindResult,
    MatchKind,
    SkipReason,
    tally,
)
from .overview import DEFAULT_DEPTH, DEFAULT_MAX_ENTRIES, build_overview
from .ripgrep import (
    DEFAULT_MAX_COUNT_PER_FILE,
    DEFAULT_MAX_FILESIZE,
    DEFAULT_TIMEOUT_S,
    ContentMatch,
    RipgrepFailed,
    RipgrepTimeout,
    RipgrepUnavailable,
    ScanReport,
    is_available,
    list_files,
    search_content,
)
from .search import DEFAULT_LIMIT, MAX_LIMIT, find

__all__ = [
    "DEFAULT_DEPTH",
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_COUNT_PER_FILE",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILESIZE",
    "DEFAULT_TIMEOUT_S",
    "ContentMatch",
    "DirEntry",
    "DirOverview",
    "FindHit",
    "FindResult",
    "MAX_LIMIT",
    "MatchKind",
    "RipgrepFailed",
    "RipgrepTimeout",
    "RipgrepUnavailable",
    "ScanReport",
    "SkipReason",
    "build_overview",
    "find",
    "is_available",
    "list_files",
    "search_content",
    "tally",
]
