"""按名称或字面内容定位文件。

该模块替换 `tools/file_tools.py::file_search` 的行为；`24/001 §1.2` 记录了旧实现
的三个缺陷：结果硬性限制为 20 条却不告知调用方；文件名与内容匹配混入同一份
无差别列表；“没有匹配项”与“没有内容可读”返回相同形态。这三者都是诚实性而非
能力问题，因此这里的结果类型会分别报告当前页、总数与排除项。

对已摄取文档执行语义检索不属于该模块职责；PDF 文本位于 `doc_chunks`，通过
`22_retrieval_system` 访问。ripgrep 搜索磁盘字节，回答的是另一个问题：文件在哪里。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import time

from ...configuration.paths import deny_reason
from .contracts import FindHit, FindResult, MatchKind, SkipReason, tally
from .ripgrep import (
    DEFAULT_MAX_COUNT_PER_FILE,
    DEFAULT_TIMEOUT_S,
    ScanReport,
    RipgrepTimeout,
    list_files,
    search_content,
)


DEFAULT_LIMIT = 20
MAX_LIMIT = 200
MAX_SNIPPET_CHARS = 160

_SORTS = ("path", "mtime", "size")


def find(
    root: Path,
    *,
    name: str | None = None,
    content: str | None = None,
    subpath: str | None = None,
    sort: str = "path",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    hidden: bool = False,
    respect_ignore: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_scan_paths: int | None = None,
    path_allowed: Callable[[Path], bool] | None = None,
) -> FindResult:
    """在 ``root`` 下按文件名 glob、字面内容或二者共同查找文件。

    ``total_matched`` 统计 ripgrep 产生的全部命中，而不只当前页，使调用方能区分
    完整答案与第一页。匹配项始终先排序再切片，从而保证分页稳定。
    """

    if not name and not content:
        raise ValueError("find requires a name pattern, a content pattern, or both")
    if sort not in _SORTS:
        raise ValueError(f"sort must be one of {_SORTS}")
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be within 1..{MAX_LIMIT}")
    if offset < 0:
        raise ValueError("offset must not be negative")

    search_root = root if not subpath else (root / subpath).resolve()
    if not _within(search_root, root):
        raise ValueError("subpath escapes the search root")

    report = ScanReport()
    hits: list[FindHit] = []
    # 留出足够余量，以便在排序与拒绝过滤后填满请求页，同时不物化无界结果集。
    keep = offset + limit + MAX_LIMIT
    total = 0
    scan_truncated = False
    deadline = time.monotonic() + timeout_s

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RipgrepTimeout(f"workspace search exceeded {timeout_s:.0f}s")
        return remaining

    if name:
        paths, name_report = list_files(
            search_root, globs=[name], hidden=hidden,
            respect_ignore=respect_ignore, timeout_s=remaining_timeout(),
            max_paths=max_scan_paths,
        )
        scan_truncated = scan_truncated or name_report.truncated
        _merge(report, name_report)
        for rel_path in paths:
            if not _caller_allows_path(search_root, rel_path, report, path_allowed):
                continue
            hit = _name_hit(search_root, rel_path, report)
            if hit is None:
                continue
            total += 1
            if len(hits) < keep:
                hits.append(hit)

    if content:
        candidate_paths, candidate_report = list_files(
            search_root,
            hidden=hidden,
            respect_ignore=respect_ignore,
            timeout_s=remaining_timeout(),
            max_paths=max_scan_paths,
        )
        scan_truncated = scan_truncated or candidate_report.truncated
        # 列举只观察名称。在允许 rg 打开任何内容前先过滤精确允许列表，避免策略
        # 拒绝的匹配通过总数或扫描统计泄漏。
        safe_paths: list[str] = []
        for rel_path in candidate_paths:
            if (
                _content_candidate_is_safe(search_root, rel_path, report)
                and _caller_allows_path(search_root, rel_path, report, path_allowed)
            ):
                safe_paths.append(rel_path)
        report.elapsed_ms += candidate_report.elapsed_ms
        for reason, count in candidate_report.skipped.items():
            report.skipped[reason] = report.skipped.get(reason, 0) + count
        matches, content_total, content_report = search_content(
            search_root, content, hidden=hidden, respect_ignore=respect_ignore,
            max_count_per_file=DEFAULT_MAX_COUNT_PER_FILE, keep=keep,
            timeout_s=remaining_timeout(), paths=safe_paths,
        )
        _merge(report, content_report)
        total += content_total
        for match in matches:
            hit = _content_hit(search_root, match, report)
            if hit is None:
                continue
            if len(hits) < keep:
                hits.append(hit)

    ordered = _order(hits, sort)
    page = tuple(ordered[offset : offset + limit])
    return FindResult(
        hits=page,
        total_matched=total,
        offset=offset,
        truncated=offset + len(page) < total,
        scanned_files=report.scanned_files,
        elapsed_ms=report.elapsed_ms,
        scan_truncated=scan_truncated,
        skipped=tally(report.skipped),
        ignore_rules_applied=respect_ignore,
    )


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _stat_or_skip(absolute: Path, report: ScanReport) -> tuple[int, int] | None:
    """解析大小与修改时间，并将拒绝和 I/O 失败归入报告。"""

    if deny_reason(absolute):
        report.skip(SkipReason.DENIED_BY_POLICY)
        return None
    try:
        stat = absolute.stat()
    except OSError:
        report.skip(SkipReason.IO_ERROR)
        return None
    return stat.st_size, stat.st_mtime_ns


def _name_hit(root: Path, rel_path: str, report: ScanReport) -> FindHit | None:
    facts = _stat_or_skip(root / rel_path, report)
    if facts is None:
        return None
    size, mtime = facts
    return FindHit(rel_path=rel_path, match=MatchKind.NAME, size_bytes=size, mtime_ns=mtime)


def _content_hit(root: Path, match, report: ScanReport) -> FindHit | None:
    if not match.rel_path:
        return None
    facts = _stat_or_skip(root / match.rel_path, report)
    if facts is None:
        return None
    size, mtime = facts
    return FindHit(
        rel_path=match.rel_path,
        match=MatchKind.CONTENT,
        size_bytes=size,
        mtime_ns=mtime,
        line=match.line,
        snippet=match.text.strip()[:MAX_SNIPPET_CHARS] or None,
    )


def _content_candidate_is_safe(
    root: Path,
    rel_path: str,
    report: ScanReport,
) -> bool:
    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        report.skip(SkipReason.DENIED_BY_POLICY)
        return False
    target = root / candidate
    cursor = target
    while True:
        if deny_reason(cursor) is not None:
            report.skip(SkipReason.DENIED_BY_POLICY)
            return False
        if cursor == root:
            break
        if cursor.parent == cursor:
            report.skip(SkipReason.DENIED_BY_POLICY)
            return False
        cursor = cursor.parent
    try:
        cursor = root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                report.skip(SkipReason.DENIED_BY_POLICY)
                return False
        resolved = target.resolve(strict=True)
        resolved.relative_to(root.resolve())
        if not resolved.is_file():
            report.skip(SkipReason.IO_ERROR)
            return False
    except (OSError, RuntimeError, ValueError):
        report.skip(SkipReason.IO_ERROR)
        return False
    return True


def _caller_allows_path(
    root: Path,
    rel_path: str,
    report: ScanReport,
    path_allowed: Callable[[Path], bool] | None,
) -> bool:
    """在列出路径成为命中项前应用调用方拥有的策略。

    目录认知可在绑定会话之外复用，因此该包不硬编码任何特定私有子树。拥有绑定
    工作区的调用方可以提供策略（例如隐藏自己的控制平面）。返回假表示按策略
    跳过，绝不是普通的无匹配。
    """

    if path_allowed is None:
        return True
    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        report.skip(SkipReason.DENIED_BY_POLICY)
        return False
    try:
        allowed = path_allowed(root / candidate)
    except (OSError, RuntimeError, ValueError):
        allowed = False
    if not allowed:
        report.skip(SkipReason.DENIED_BY_POLICY)
        return False
    return True


def _order(hits: Sequence[FindHit], sort: str) -> list[FindHit]:
    """使用全序排序。

    每个键最终都回退到路径与行号，使并列项绝不依赖 ripgrep 恰好遍历目录树的
    顺序；`24/001 D5` 要求相同请求两次产生相同页面。
    """

    if sort == "mtime":
        key = lambda h: (-h.mtime_ns, h.rel_path, h.line or 0)  # noqa: E731
    elif sort == "size":
        key = lambda h: (-h.size_bytes, h.rel_path, h.line or 0)  # noqa: E731
    else:
        key = lambda h: (h.rel_path, h.line or 0, h.match.value)  # noqa: E731
    return sorted(hits, key=key)


def _merge(target: ScanReport, source: ScanReport) -> None:
    target.scanned_files += source.scanned_files
    target.elapsed_ms += source.elapsed_ms
    for reason, count in source.skipped.items():
        target.skipped[reason] = target.skipped.get(reason, 0) + count


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "find"]
