"""经过聚合并设有边界的目录结构。

这里回答的是“该目录树中有什么”，诚实答案并不是文件列表。以本仓库测量，
完整路径列表约需 17,600 个词元，而目录层级只需约 1,600 个；后者正是读取者
决定去哪里查看所需的信息，预算仅为前者十分之一。

因此聚合在此处而非调用方进行：若消费者先收到每条路径再汇总，就已经付出了
汇总本应避免的成本。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from ...configuration.paths import deny_reason
from .contracts import DirEntry, DirOverview, SkipReason, tally
from .ripgrep import DEFAULT_TIMEOUT_S, ScanReport, list_files


DEFAULT_DEPTH = 2
DEFAULT_MAX_ENTRIES = 60

# 只有最常见的后缀值得列出；大量仅出现一次的扩展名会消耗预算，却不会改变
# 读取者接下来查看的位置。
MAX_KINDS_PER_ENTRY = 5


def _group_key(rel_path: str, depth: int) -> str:
    parts = Path(rel_path).parts
    if len(parts) <= 1:
        return "."
    return str(Path(*parts[: min(depth, len(parts) - 1)]))


def build_overview(
    root: Path,
    *,
    depth: int = DEFAULT_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    hidden: bool = False,
    respect_ignore: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_scan_paths: int | None = None,
    path_allowed: Callable[[Path], bool] | None = None,
) -> DirOverview:
    """按目录汇总一棵目录树。

    即使 ripgrep 已跳过忽略路径，列表仍会应用安全拒绝：`.gitignore` 是仓库作者
    选择的降噪规则，不是我们选择的安全边界（`24/002 §7`）。否则，没有
    `.gitignore` 的目录会暴露 `.env` 等文件。
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")

    paths, report = list_files(
        root,
        hidden=hidden,
        respect_ignore=respect_ignore,
        timeout_s=timeout_s,
        max_paths=max_scan_paths,
    )
    buckets: dict[str, dict[str, object]] = {}
    total_files = 0

    for rel_path in paths:
        absolute = root / rel_path
        if path_allowed is not None and not path_allowed(absolute):
            report.skip(SkipReason.DENIED_BY_POLICY)
            continue
        if deny_reason(absolute):
            report.skip(SkipReason.DENIED_BY_POLICY)
            continue
        try:
            stat = absolute.stat()
        except OSError:
            report.skip(SkipReason.IO_ERROR)
            continue
        total_files += 1
        key = _group_key(rel_path, depth)
        bucket = buckets.setdefault(key, {"files": 0, "bytes": 0, "kinds": {}, "mtime": None})
        bucket["files"] = int(bucket["files"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + stat.st_size
        suffix = Path(rel_path).suffix.lower() or "(none)"
        kinds = bucket["kinds"]
        assert isinstance(kinds, dict)
        kinds[suffix] = kinds.get(suffix, 0) + 1
        current = bucket["mtime"]
        if current is None or stat.st_mtime_ns > int(current):
            bucket["mtime"] = stat.st_mtime_ns

    # 按规模选择、按路径输出。预算收紧时，按文件数选择可保留最繁忙的目录；
    # 按路径输出则保证对未变目录树运行两次得到字节级相同结果。
    ranked = sorted(buckets.items(), key=lambda item: (-int(item[1]["files"]), item[0]))
    kept = ranked[:max_entries]
    omitted = len(ranked) - len(kept)

    entries = tuple(
        DirEntry(
            rel_path=key,
            files=int(bucket["files"]),
            bytes_total=int(bucket["bytes"]),
            kinds=_top_kinds(bucket["kinds"]),
            newest_mtime_ns=bucket["mtime"],  # type: ignore[arg-type]
        )
        for key, bucket in sorted(kept, key=lambda item: item[0])
    )
    return DirOverview(
        root=str(root),
        depth=depth,
        entries=entries,
        total_files=total_files,
        total_dirs=len(buckets),
        truncated=omitted > 0,
        omitted_dirs=omitted,
        scan_truncated=report.truncated,
        skipped=tally(report.skipped),
    )


def _top_kinds(kinds: object) -> tuple[tuple[str, int], ...]:
    assert isinstance(kinds, dict)
    ranked = sorted(kinds.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ranked[:MAX_KINDS_PER_ENTRY])


__all__ = ["DEFAULT_DEPTH", "DEFAULT_MAX_ENTRIES", "build_overview"]
