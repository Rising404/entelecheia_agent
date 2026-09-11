"""围绕 ripgrep 二进制程序的轻量无 Shell 封装。

此处不重新实现遍历、忽略规则处理与匹配。`24/002 E2` 选择 ripgrep，是因为它在
这三方面都优于手写遍历器：支持包括嵌套文件在内的
`.gitignore`/`.ignore`/`.rgignore`，跳过二进制文件，处理编码，并能在 26 毫秒内
搜索本仓库的 1,528 个候选文件。

该模块只拥有封装器必须承担的部分——定位二进制文件、构建 argv 向量、在边界内
流式处理 `--json` 事件，以及把进程失败转换为带类型事实。它不了解会话、文档
或工具，因此仅凭临时目录即可测试。

输出采用流式而非缓冲方式。匹配事件会完整计数，使 `total_matched` 诚实；但只
物化请求页，因此异常目录树只消耗有界内存，而不会捕获数兆字节的数据块。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import SkipReason


BINARY_NAME = "rg"

# 调用方未覆盖时由封装器应用的边界，防止单次工具调用挂起轮次或逐行读取数 GB 日志。
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_FILESIZE = "8M"
DEFAULT_MAX_COUNT_PER_FILE = 5
MAX_EXACT_PATHS_PER_PROCESS = 256
MAX_EXACT_PATH_ARGUMENT_BYTES = 64_000


class RipgrepUnavailable(RuntimeError):
    """当前解释器环境未安装 ripgrep 二进制程序。"""


class RipgrepTimeout(RuntimeError):
    """搜索超过墙钟时间边界并被终止。"""


class RipgrepFailed(RuntimeError):
    """ripgrep 以错误状态退出，而不是返回匹配判定。"""


@dataclass(frozen=True)
class ContentMatch:
    """由 `match` 事件报告的一条匹配行。"""

    rel_path: str
    line: int
    text: str


@dataclass
class ScanReport:
    """单次调用实际执行的工作，与发现内容相区分。"""

    scanned_files: int = 0
    elapsed_ms: int = 0
    truncated: bool = False
    skipped: dict[SkipReason, int] = field(default_factory=dict)

    def skip(self, reason: SkipReason) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def binary_path() -> str:
    """定位 ripgrep，并优先使用当前环境中安装的副本。

    `ripgrep` wheel 会把 `rg` 放入解释器的 `bin/`，因此优先使用 `sys.prefix`
    可以固定项目声明的版本，而不是主机 PATH 上恰好存在的版本。Windows 没有该
    固定包的 wheel，但在回退系统 PATH 前仍接受受管或手动安装的
    ``Scripts\\rg.exe``。
    """

    candidates = (
        Path(sys.prefix) / "bin" / BINARY_NAME,
        Path(sys.prefix) / "Scripts" / f"{BINARY_NAME}.exe",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(BINARY_NAME)
    if found:
        return found
    raise RipgrepUnavailable(
        "ripgrep is not installed; add the 'ripgrep' dependency or install it on PATH"
    )


def is_available() -> bool:
    try:
        binary_path()
    except RipgrepUnavailable:
        return False
    return True


def _base_args(*, hidden: bool, respect_ignore: bool, globs: Sequence[str]) -> list[str]:
    """构建每次调用共享的标志。

    ``--no-require-git`` 并非细枝末节。ripgrep 默认只在 Git 仓库中遵循
    ``.gitignore``，而本包服务的是用户文档目录，通常不是仓库。若没有该标志，
    同一 `.gitignore` 会根据是否恰好存在 `.git` 目录而静默改变含义，模型会因
    无法观察的原因看到不同结果。
    """

    args: list[str] = []
    if hidden:
        args.append("--hidden")
    if respect_ignore:
        args.append("--no-require-git")
    else:
        args.append("--no-ignore")
    for pattern in globs:
        args.extend(["--glob", pattern])
    return args


def _classify_stderr(line: str) -> SkipReason:
    """将一条 ripgrep 诊断映射为封闭原因。

    ripgrep 会在 stderr 报告逐路径 I/O 问题并继续运行。对其计数，调用方才能
    区分“没有匹配项”和“四个目录不可读”。
    """

    lowered = line.lower()
    if "permission denied" in lowered:
        return SkipReason.PERMISSION_DENIED
    return SkipReason.IO_ERROR


def _run(
    args: list[str],
    *,
    timeout_s: float,
    report: ScanReport,
    max_stdout_lines: int | None = None,
) -> Iterator[str]:
    """流式读取 stdout 行，并将 stderr 诊断归入 ``report``。

    stderr 在独立线程中排空，而不是事后读取。ripgrep 会为每个不可读路径发出一条
    诊断，因此包含大量权限错误的目录树可能填满 stderr 管道缓冲区，并与仍在读取
    stdout 的进程形成死锁。

    退出码 1 表示“无匹配”，属于结果而非错误；只有 2 及以上表示 ripgrep 本身失败。
    """

    process = subprocess.Popen(  # noqa: S603 - 参数采用向量形式，绝不使用 shell
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    diagnostics: list[str] = []

    def _drain() -> None:
        assert process.stderr is not None
        try:
            for raw in process.stderr:
                line = raw.strip()
                if line:
                    diagnostics.append(line)
        finally:
            process.stderr.close()

    pump = threading.Thread(target=_drain, daemon=True)
    pump.start()

    deadline = time.monotonic() + timeout_s
    timed_out = False
    intentionally_stopped = False
    stdout_lines = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if time.monotonic() > deadline:
                timed_out = True
                process.kill()
                break
            stdout_lines += 1
            if max_stdout_lines is not None and stdout_lines > max_stdout_lines:
                report.truncated = True
                intentionally_stopped = True
                process.kill()
                break
            yield line
    finally:
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        pump.join(timeout=2.0)
        for line in diagnostics:
            report.skip(_classify_stderr(line))

    if timed_out:
        raise RipgrepTimeout(f"ripgrep exceeded {timeout_s:.0f}s")
    if process.returncode not in (0, 1) and not intentionally_stopped:
        detail = "; ".join(diagnostics[:3]) or f"exit {process.returncode}"
        raise RipgrepFailed(detail)


def list_files(
    root: Path,
    *,
    globs: Sequence[str] = (),
    hidden: bool = False,
    respect_ignore: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_paths: int | None = None,
) -> tuple[list[str], ScanReport]:
    """列出 ripgrep 将搜索的文件，路径相对于 ``root``。

    ``max_paths`` 是实际进程边界，而不是完整遍历后的切片。达到该值时会停止
    ripgrep，``report.truncated`` 会告知调用方，返回的按路径排序前缀并不完整。
    """

    if max_paths is not None and max_paths < 1:
        raise ValueError("max_paths must be positive when configured")

    args = [binary_path(), "--files"]
    if max_paths is not None:
    # 只有 ripgrep 按路径顺序输出时，提前停止才具有确定性。
        args.extend(["--sort", "path"])
    args.extend([*_base_args(
        hidden=hidden, respect_ignore=respect_ignore, globs=globs
    ), "--", str(root)])
    report = ScanReport()
    started = time.perf_counter()
    paths: list[str] = []
    root_str = str(root)
    for line in _run(
        args,
        timeout_s=timeout_s,
        report=report,
        max_stdout_lines=max_paths,
    ):
        value = line.rstrip("\n")
        if not value:
            continue
        paths.append(os.path.relpath(value, root_str))
    report.elapsed_ms = int((time.perf_counter() - started) * 1000)
    report.scanned_files = len(paths)
    paths.sort()
    return paths, report


def search_content(
    root: Path,
    pattern: str,
    *,
    fixed_string: bool = True,
    globs: Sequence[str] = (),
    hidden: bool = False,
    respect_ignore: bool = True,
    max_count_per_file: int = DEFAULT_MAX_COUNT_PER_FILE,
    max_filesize: str = DEFAULT_MAX_FILESIZE,
    keep: int = 100,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    paths: Sequence[str] | None = None,
) -> tuple[list[ContentMatch], int, ScanReport]:
    """搜索文件内容，返回 ``(page, total_matched, report)``。

    ``keep`` 限制物化的匹配数量；每条匹配仍会计数，因此调用方只持有一页也能报告
    准确总数。

    ``scanned_files`` 来自 ripgrep 自身摘要，回答“实际打开了多少文件”。当前版本
    不会逐项列出因 ``max_filesize`` 或二进制检测而排除的文件；但会报告边界本身，
    使排除至少可见。
    """

    if paths is None:
        targets = (str(root),)
        batches = (targets,)
    else:
        batches = tuple(_exact_path_batches(root, paths))
        if not batches:
            return [], 0, ScanReport()

    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    matches: list[ContentMatch] = []
    total = 0
    combined = ScanReport()
    for targets in batches:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise RipgrepTimeout(f"ripgrep exceeded {timeout_s:.0f}s")
        batch_matches, batch_total, batch_report = _search_content_once(
            root,
            pattern,
            targets=targets,
            fixed_string=fixed_string,
            globs=globs,
            hidden=hidden,
            respect_ignore=respect_ignore,
            max_count_per_file=max_count_per_file,
            max_filesize=max_filesize,
            keep=max(0, keep - len(matches)),
            timeout_s=remaining_s,
        )
        matches.extend(batch_matches)
        total += batch_total
        combined.scanned_files += batch_report.scanned_files
        for reason, count in batch_report.skipped.items():
            combined.skipped[reason] = combined.skipped.get(reason, 0) + count
    combined.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return matches, total, combined


def _search_content_once(
    root: Path,
    pattern: str,
    *,
    targets: Sequence[str],
    fixed_string: bool,
    globs: Sequence[str],
    hidden: bool,
    respect_ignore: bool,
    max_count_per_file: int,
    max_filesize: str,
    keep: int,
    timeout_s: float,
) -> tuple[list[ContentMatch], int, ScanReport]:
    args = [
        binary_path(), "--json",
        "--max-count", str(max_count_per_file),
        "--max-filesize", max_filesize,
        *_base_args(hidden=hidden, respect_ignore=respect_ignore, globs=globs),
    ]
    if fixed_string:
        args.append("--fixed-strings")
    args.extend(["-e", pattern, "--", *targets])

    report = ScanReport()
    matches: list[ContentMatch] = []
    total = 0
    root_str = str(root)
    started = time.perf_counter()
    for line in _run(args, timeout_s=timeout_s, report=report):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "match":
            total += 1
            if len(matches) < keep:
                data = event.get("data") or {}
                path = ((data.get("path") or {}).get("text")) or ""
                text = ((data.get("lines") or {}).get("text")) or ""
                matches.append(ContentMatch(
                    rel_path=os.path.relpath(path, root_str) if path else "",
                    line=int(data.get("line_number") or 0),
                    text=text.rstrip("\n"),
                ))
        elif kind == "summary":
            stats = (event.get("data") or {}).get("stats") or {}
            report.scanned_files = int(stats.get("searches") or 0)
    report.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return matches, total, report


def _exact_path_batches(root: Path, paths: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """为已经授权的相对路径允许列表产出有界 argv 批次。"""

    batch: list[str] = []
    batch_bytes = 0
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("content-search paths must stay relative to root")
        absolute = str(root / candidate)
        encoded_bytes = len(absolute.encode("utf-8")) + 1
        if encoded_bytes > MAX_EXACT_PATH_ARGUMENT_BYTES:
            raise ValueError("one content-search path exceeds the argv budget")
        if batch and (
            len(batch) >= MAX_EXACT_PATHS_PER_PROCESS
            or batch_bytes + encoded_bytes > MAX_EXACT_PATH_ARGUMENT_BYTES
        ):
            yield tuple(batch)
            batch = []
            batch_bytes = 0
        batch.append(absolute)
        batch_bytes += encoded_bytes
    if batch:
        yield tuple(batch)


__all__ = [
    "BINARY_NAME",
    "ContentMatch",
    "RipgrepFailed",
    "RipgrepTimeout",
    "RipgrepUnavailable",
    "ScanReport",
    "binary_path",
    "is_available",
    "list_files",
    "search_content",
]
