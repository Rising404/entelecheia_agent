"""有界工作区发现所使用的模型可见注册。

Host 会在模型收到任何问题前冻结搜索根目录，因此提案只能在其中缩小范围。之所以存在该
规则，是因为模型提供的路径可能出错，也可能受到刚读取文档内容的诱导。

本模块特意保持轻量。遍历、匹配和安全筛选位于 ``personagraph.workspace``；此处只包含边界、
JSON 契约及二者之间的投影。
"""

from __future__ import annotations

import codecs
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...configuration.paths import deny_reason
from ...workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)
from ...workspace.discovery import (
    DEFAULT_DEPTH,
    DEFAULT_LIMIT,
    DEFAULT_MAX_COUNT_PER_FILE,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_MAX_FILESIZE,
    DEFAULT_TIMEOUT_S,
    FindResult,
    RipgrepFailed,
    RipgrepTimeout,
    RipgrepUnavailable,
    SkipReason,
    build_overview,
    find,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..execution import ToolBusinessFailure
from ..registration import ToolExecutionProfile, ToolRegistration
from .workspace_directory_listing import (
    DEFAULT_DIRECTORY_PAGE_SIZE,
    MAX_DIRECTORY_PAGE_SIZE,
    list_workspace_directory_page,
)
from .workspace_tool_contracts import (
    DEFAULT_INSPECT_BYTES,
    DISCOVERY_MAX_ENTRIES,
    DISCOVERY_MAX_RESULTS,
    MAX_DEPTH,
    MAX_INSPECT_BYTES,
    WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
    _discovery_overview_output_schema,
    _find_files_input_schema,
    _find_files_output_schema,
    _inspect_file_input_schema,
    _inspect_file_output_schema,
    _list_workspace_directory_input_schema,
    _list_workspace_directory_output_schema,
    _overview_input_schema,
    _search_text_files_input_schema,
    _search_text_files_output_schema,
)


WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION = "2"
WORKSPACE_DISCOVERY_TOOL_IDS = (
    "workspace_overview",
    "list_workspace_directory",
    "find_files",
    "search_text_files",
    "inspect_file",
)

DISCOVERY_MAX_SCAN_PATHS = 4_096

_BINARY_INSPECT_SUFFIXES = frozenset(
    {
        ".bmp",
        ".doc",
        ".docx",
        ".gif",
        ".gz",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".tar",
        ".tif",
        ".tiff",
        ".webp",
        ".xls",
        ".xlsx",
        ".zip",
    }
)


@dataclass(frozen=True)
class FrozenWorkspaceToolBoundary:
    """此注册能够查看的唯一目录。

    ``root`` 由组合根根据 Session 绑定的工作目录解析。它并非模型可以覆盖的默认值：
    每个提案都相对于它解释，且逃逸的 ``subpath`` 会在任何进程启动前被拒绝。
    """

    session_id: str
    root: Path
    respect_ignore: bool = True
    hidden: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    freshness_check: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    root_device: int = field(init=False)
    root_inode: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.freshness_check is not None and not callable(
            self.freshness_check
        ):
            raise TypeError("freshness_check must be callable")
        resolved = Path(self.root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace root is not a directory: {resolved}")
        identity = resolved.stat()
        object.__setattr__(self, "root", resolved)
        object.__setattr__(self, "root_device", int(identity.st_dev))
        object.__setattr__(self, "root_inode", int(identity.st_ino))

    def require_current_root(self) -> Path:
        """重新验证冻结到此注册中的精确目录。"""

        try:
            current = self.root.resolve(strict=True)
            identity = current.stat()
            unchanged = (
                current == self.root
                and current.is_dir()
                and int(identity.st_dev) == self.root_device
                and int(identity.st_ino) == self.root_inode
            )
        except (OSError, RuntimeError):
            unchanged = False
        if not unchanged:
            raise ToolBusinessFailure(
                "workspace_authority_changed",
                "The frozen workspace directory is no longer the authorized directory.",
            )
        if self.freshness_check is not None:
            self.freshness_check()
        return current


def build_workspace_discovery_tool_registrations(
    boundary: FrozenWorkspaceToolBoundary,
) -> tuple[
    ToolRegistration,
    ToolRegistration,
    ToolRegistration,
    ToolRegistration,
    ToolRegistration,
]:
    """暴露五个互不重叠的工作区发现工具。

    """

    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id="personagraph.workspace.discovery",
    )
    execution = build_workspace_discovery_execution_profile(
        timeout_s=boundary.timeout_s,
    )
    specs = {
        spec.tool_id: spec for spec in build_workspace_discovery_tool_specs()
    }
    return (
        ToolRegistration(
            spec=specs["workspace_overview"],
            implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded(
                boundary,
                lambda payload: _run_discovery_overview(boundary, payload)
            ),
            effect_profile=_workspace_effect(EffectAction.READ, boundary.root),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs["list_workspace_directory"],
            implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded(
                boundary,
                lambda payload: _run_list_workspace_directory(
                    boundary,
                    payload,
                ),
            ),
            effect_profile=_workspace_effect(EffectAction.READ, boundary.root),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs["find_files"],
            implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded(
                boundary,
                lambda payload: _run_find_files(
                    boundary,
                    payload,
                ),
            ),
            effect_profile=_workspace_effect(EffectAction.SEARCH, boundary.root),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs["search_text_files"],
            implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded(
                boundary,
                lambda payload: _run_search_text_files(
                    boundary,
                    payload,
                ),
            ),
            effect_profile=_workspace_effect(
                EffectAction.SEARCH,
                boundary.root,
                data_egress=DataEgress.CONTENT,
            ),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs["inspect_file"],
            implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guarded(
                boundary,
                lambda payload: _run_inspect_file(
                    boundary,
                    payload,
                ),
            ),
            effect_profile=_workspace_effect(
                EffectAction.READ,
                boundary.root,
                data_egress=DataEgress.CONTENT,
            ),
            execution_profile=execution,
        ),
    )


def build_workspace_discovery_tool_specs() -> tuple[
    ToolSpec,
    ToolSpec,
    ToolSpec,
    ToolSpec,
    ToolSpec,
]:
    """Return the context-free model contracts in their canonical order."""

    return (
        ToolSpec(
            tool_id="workspace_overview",
            contract_version=WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
            name="Survey the workspace",
            description=(
                "Summarise the frozen session workspace by directory. The "
                "result is bounded, explicitly reports truncation and omitted "
                "paths, and never lists every file. Use find_files to discover "
                "individual paths."
            ),
            input_schema=_overview_input_schema(
                max_entries_ceiling=DISCOVERY_MAX_ENTRIES
            ),
            output_schema=_discovery_overview_output_schema(),
            catalog_tags=("file", "read"),
        ),
        ToolSpec(
            tool_id="list_workspace_directory",
            contract_version=WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
            name="List one workspace directory",
            description=(
                "List one bounded page of direct discoverable children beneath "
                "a workspace-relative folder. Results are stable, directories "
                "precede files, and continuation uses an opaque cursor bound to "
                "the current listing snapshot. Empty or wholly ignored folders "
                "are not discoverable. Use returned paths with check_files_state "
                "or prepare_files to inspect or prepare selected files. "
                "This tool never reads file contents, parses documents, or "
                "changes the retrieval corpus."
            ),
            input_schema=_list_workspace_directory_input_schema(),
            output_schema=_list_workspace_directory_output_schema(),
            catalog_tags=("file", "list", "read"),
        ),
        ToolSpec(
            tool_id="find_files",
            contract_version=WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
            name="Find files by name",
            description=(
                "Find files by filename glob inside the frozen workspace. "
                "Returned paths are always relative to the workspace root, even "
                "when subpath narrows the search. This tool never searches file "
                "contents."
            ),
            input_schema=_find_files_input_schema(),
            output_schema=_find_files_output_schema(),
            catalog_tags=("file", "search", "read"),
        ),
        ToolSpec(
            tool_id="search_text_files",
            contract_version=WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
            name="Search text-file contents",
            description=(
                "Search for literal text inside text files in the frozen "
                "workspace. It reports matching lines and does not match file "
                "names. PDF and Office document contents belong to the document "
                "ingestion and retrieval path, not this byte search."
            ),
            input_schema=_search_text_files_input_schema(),
            output_schema=_search_text_files_output_schema(),
            catalog_tags=("file", "search", "read"),
        ),
        ToolSpec(
            tool_id="inspect_file",
            contract_version=WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
            name="Inspect one workspace file",
            description=(
                "Inspect metadata and a bounded UTF-8 text preview for one "
                "workspace-relative path. Binary or rich documents return "
                "metadata only; use the document pipeline to understand their "
                "contents. Absolute paths and paths that escape the frozen "
                "workspace are rejected."
            ),
            input_schema=_inspect_file_input_schema(),
            output_schema=_inspect_file_output_schema(),
            catalog_tags=("file", "read"),
        ),
    )


def build_workspace_discovery_execution_profile(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ToolExecutionProfile:
    """Return the shared execution contract for one discovery binding."""

    return ToolExecutionProfile(
        default_timeout_s=timeout_s,
        hard_timeout_s=timeout_s * 2,
        # Later Attempts retain complete ToolResults below 128 KiB. Keep every
        # discovery result materially below that aggregate boundary.
        max_output_bytes=90_000,
        max_transparent_retries=0,
        concurrency_class="workspace_discovery",
    )


def _workspace_effect(
    action: EffectAction,
    workspace_root: Path,
    *,
    data_egress: DataEgress = DataEgress.METADATA,
) -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=action,
                scope_kind=EffectScopeKind.WORKSPACE,
                default_scope=str(workspace_root),
                # 发现界面仅限本地。内容外传对策略仍然重要，因为 inspect_file 可以返回文本前缀。
                data_egress=data_egress,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def _guarded(
    boundary: FrozenWorkspaceToolBoundary,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """避免让依赖失败文本进入模型可见结果。

    二进制文件缺失是终态：重试不会安装任何内容，因此它会变成业务失败，
    而非让执行器重复三次的异常。超时可能是瞬态，仍保留为普通异常，故有界重试策略继续适用。
    """

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            boundary.require_current_root()
            result = handler(payload)
            boundary.require_current_root()
            return result
        except ValueError as exc:
            raise ToolBusinessFailure("invalid_request", str(exc)) from exc
        except RipgrepUnavailable as exc:
            raise ToolBusinessFailure(
                "search_backend_unavailable",
                "The directory search backend is not installed on this host.",
            ) from exc
        except RipgrepTimeout as exc:
            raise RuntimeError("The directory search exceeded its time budget.") from exc
        except RipgrepFailed as exc:
            raise RuntimeError("The directory search backend failed.") from exc

    return run


def _run_discovery_overview(
    boundary: FrozenWorkspaceToolBoundary, payload: dict[str, Any]
) -> dict[str, Any]:
    depth = _bounded_int(payload.get("depth"), DEFAULT_DEPTH, 1, MAX_DEPTH, "depth")
    max_entries = _bounded_int(
        payload.get("max_entries"),
        DEFAULT_MAX_ENTRIES,
        1,
        DISCOVERY_MAX_ENTRIES,
        "max_entries",
    )
    root = _resolve_subpath(boundary, payload.get("subpath"))
    scope_path = _workspace_relative_path(boundary, root)
    overview = build_overview(
        root,
        depth=depth,
        max_entries=max_entries,
        hidden=boundary.hidden,
        respect_ignore=boundary.respect_ignore,
        timeout_s=boundary.timeout_s,
        max_scan_paths=DISCOVERY_MAX_SCAN_PATHS,
        path_allowed=lambda candidate: _is_user_workspace_path(
            boundary, candidate
        ),
    )
    items = [
        {
            "path": _path_in_scope(scope_path, entry.rel_path),
            "files": entry.files,
            "bytes": entry.bytes_total,
            "kinds": [
                {"suffix": suffix, "files": count}
                for suffix, count in entry.kinds
            ],
        }
        for entry in overview.entries
    ]
    return {
        "scope": _scope_view(boundary, scope_path),
        "items": items,
        "returned": len(items),
        "total": overview.total_dirs,
        "truncated": overview.truncated or overview.scan_truncated,
        "scan_truncated": overview.scan_truncated,
        "total_relation": "lower_bound" if overview.scan_truncated else "exact",
        "skipped": _skipped_view(overview.skipped),
        "depth": overview.depth,
        "total_files": overview.total_files,
        "omitted": overview.omitted_dirs,
    }


def _run_list_workspace_directory(
    boundary: FrozenWorkspaceToolBoundary,
    payload: dict[str, Any],
) -> dict[str, Any]:
    root = _resolve_subpath(boundary, payload.get("subpath"))
    scope_path = _workspace_relative_path(boundary, root)
    return list_workspace_directory_page(
        root,
        session_id=boundary.session_id,
        scope_path=scope_path,
        cursor=payload.get("cursor"),
        limit=_bounded_int(
            payload.get("limit"),
            DEFAULT_DIRECTORY_PAGE_SIZE,
            1,
            MAX_DIRECTORY_PAGE_SIZE,
            "limit",
        ),
        hidden=boundary.hidden,
        respect_ignore=boundary.respect_ignore,
        timeout_s=boundary.timeout_s,
        max_scan_paths=DISCOVERY_MAX_SCAN_PATHS,
        path_allowed=lambda candidate: _is_user_workspace_path(
            boundary, candidate
        ),
    )


def _run_find_files(
    boundary: FrozenWorkspaceToolBoundary,
    payload: dict[str, Any],
) -> dict[str, Any]:
    name = _required_text(payload.get("name"), "name", max_length=256)
    root, scope_path = _discovery_search_scope(boundary, payload)
    result = find(
        root,
        name=name,
        sort=str(payload.get("sort") or "path"),
        limit=_bounded_int(
            payload.get("limit"),
            DEFAULT_LIMIT,
            1,
            DISCOVERY_MAX_RESULTS,
            "limit",
        ),
        offset=_bounded_int(payload.get("offset"), 0, 0, 100_000, "offset"),
        hidden=boundary.hidden,
        respect_ignore=boundary.respect_ignore,
        timeout_s=boundary.timeout_s,
        max_scan_paths=DISCOVERY_MAX_SCAN_PATHS,
        path_allowed=lambda candidate: _is_user_workspace_path(
            boundary, candidate
        ),
    )
    items = [
        {
            "path": _path_in_scope(scope_path, hit.rel_path),
            "size": hit.size_bytes,
            "modified_ns": hit.mtime_ns,
        }
        for hit in result.hits
    ]
    return _discovery_search_view(
        boundary,
        scope_path=scope_path,
        result=result,
        items=items,
    )


def _run_search_text_files(
    boundary: FrozenWorkspaceToolBoundary,
    payload: dict[str, Any],
) -> dict[str, Any]:
    content = _required_text(payload.get("content"), "content", max_length=512)
    root, scope_path = _discovery_search_scope(boundary, payload)
    result = find(
        root,
        content=content,
        sort=str(payload.get("sort") or "path"),
        limit=_bounded_int(
            payload.get("limit"),
            DEFAULT_LIMIT,
            1,
            DISCOVERY_MAX_RESULTS,
            "limit",
        ),
        offset=_bounded_int(payload.get("offset"), 0, 0, 100_000, "offset"),
        hidden=boundary.hidden,
        respect_ignore=boundary.respect_ignore,
        timeout_s=boundary.timeout_s,
        max_scan_paths=DISCOVERY_MAX_SCAN_PATHS,
        path_allowed=lambda candidate: _is_user_workspace_path(
            boundary, candidate
        ),
    )
    items = [
        {
            "path": _path_in_scope(scope_path, hit.rel_path),
            "line": hit.line,
            "snippet": hit.snippet or "",
            "size": hit.size_bytes,
        }
        for hit in result.hits
    ]
    view = _discovery_search_view(
        boundary,
        scope_path=scope_path,
        result=result,
        items=items,
    )
    view["max_matches_per_file"] = DEFAULT_MAX_COUNT_PER_FILE
    view["max_file_size"] = DEFAULT_MAX_FILESIZE
    return view


def _run_inspect_file(
    boundary: FrozenWorkspaceToolBoundary,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = _required_text(payload.get("path"), "path", max_length=512)
    max_bytes = _bounded_int(
        payload.get("max_bytes"),
        DEFAULT_INSPECT_BYTES,
        1,
        MAX_INSPECT_BYTES,
        "max_bytes",
    )
    target = _resolve_relative_file(boundary, path)
    scope_path = _workspace_relative_path(boundary, target)
    base = {
        "scope": _scope_view(boundary, scope_path),
        "items": [],
        "returned": 0,
        "total": 1,
        "truncated": False,
        "skipped": [],
    }

    if deny_reason(target):
        base["skipped"] = _skipped_view(((SkipReason.DENIED_BY_POLICY, 1),))
        return base

    try:
        stat = target.stat()
        with target.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except PermissionError:
        base["skipped"] = _skipped_view(((SkipReason.PERMISSION_DENIED, 1),))
        return base
    except OSError:
        base["skipped"] = _skipped_view(((SkipReason.IO_ERROR, 1),))
        return base

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    item: dict[str, Any] = {
        "path": scope_path,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "media_type": media_type,
    }
    prefix = data[:max_bytes]
    if not _is_probably_text(target, media_type, prefix):
        item["content_kind"] = "binary"
    else:
        content_truncated = len(data) > max_bytes or stat.st_size > len(prefix)
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            preview = decoder.decode(prefix, final=not content_truncated)
        except UnicodeDecodeError:
            item["content_kind"] = "undecodable_text"
            base["skipped"] = _skipped_view(((SkipReason.DECODE_FAILED, 1),))
        else:
            item["content_kind"] = "text"
            item["text"] = preview
            base["truncated"] = content_truncated

    base["items"] = [item]
    base["returned"] = 1
    return base


def _discovery_search_scope(
    boundary: FrozenWorkspaceToolBoundary, payload: dict[str, Any]
) -> tuple[Path, str]:
    root = _resolve_subpath(boundary, payload.get("subpath"))
    return root, _workspace_relative_path(boundary, root)


def _discovery_search_view(
    boundary: FrozenWorkspaceToolBoundary,
    *,
    scope_path: str,
    result: FindResult,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": _scope_view(boundary, scope_path),
        "items": items,
        "returned": result.returned,
        "total": result.total_matched,
        "truncated": result.truncated or result.scan_truncated,
        "scan_truncated": result.scan_truncated,
        "total_relation": "lower_bound" if result.scan_truncated else "exact",
        "skipped": _skipped_view(result.skipped),
        "offset": result.offset,
        "scanned_files": result.scanned_files,
        "ignore_rules_applied": result.ignore_rules_applied,
    }


def _scope_view(
    boundary: FrozenWorkspaceToolBoundary, relative_path: str
) -> dict[str, str]:
    return {"session_id": boundary.session_id, "path": relative_path}


def _is_user_workspace_path(
    boundary: FrozenWorkspaceToolBoundary,
    candidate: Path,
) -> bool:
    """让 Host 持有的控制树远离通用发现。

    即使 ripgrep 默认隐藏点目录，也会特意调用分类器。Host 可能启用隐藏文件发现，
    而私有树在该模式下仍须保持不可用。
    """

    try:
        return not is_reserved_workspace_path(boundary.root, candidate)
    except ReservedWorkspacePathError:
        return False


def _require_user_workspace_path(
    boundary: FrozenWorkspaceToolBoundary,
    candidate: Path,
) -> None:
    if not _is_user_workspace_path(boundary, candidate):
        raise ValueError("path selects the agent-managed private workspace area")


def _workspace_relative_path(
    boundary: FrozenWorkspaceToolBoundary, target: Path
) -> str:
    relative = target.relative_to(boundary.root)
    return relative.as_posix() if relative.parts else "."


def _path_in_scope(scope_path: str, relative_path: str) -> str:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("The directory search backend returned an unsafe path.")
    relative = candidate.as_posix()
    if relative in ("", "."):
        return scope_path
    return relative if scope_path == "." else f"{scope_path}/{relative}"


def _skipped_view(skipped) -> list[dict[str, Any]]:
    return [{"reason": reason.value, "count": count} for reason, count in skipped]


def _is_probably_text(target: Path, media_type: str, prefix: bytes) -> bool:
    if b"\x00" in prefix:
        return False
    if target.suffix.lower() in _BINARY_INSPECT_SUFFIXES:
        return False
    if media_type.startswith(("audio/", "font/", "image/", "video/")):
        return False
        # 允许未知扩展名、源代码格式和无扩展名文件到达下方严格 UTF-8 解码器。
        # 这样无需维护不安全的允许列表，即可识别 TOML、INI、Dockerfile 及未来文本格式；
        # 无效字节仍会变成显式 decode_failed 跳过项。
    return True


def _resolve_subpath(
    boundary: FrozenWorkspaceToolBoundary, raw: Any
) -> Path:
    """严格在冻结根目录内解释可选的范围缩小提示。"""

    root = boundary.require_current_root()
    text = _optional_text(raw)
    if not text:
        return root
    candidate = Path(text)
    if candidate.is_absolute():
        raise ValueError("subpath must be relative to the working directory")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("subpath escapes the working directory") from exc
    if not resolved.is_dir():
        raise ValueError("subpath is not a directory inside the working directory")
    _require_user_workspace_path(boundary, resolved)
    boundary.require_current_root()
    return resolved


def _resolve_relative_file(
    boundary: FrozenWorkspaceToolBoundary, raw: str
) -> Path:
    """精确解析一个模型提供的相对文件，但不扩大范围。"""

    root = boundary.require_current_root()
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("path must be relative to the working directory")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("path escapes the working directory") from exc
    if not resolved.is_file():
        raise ValueError("path is not a file inside the working directory")
    _require_user_workspace_path(boundary, resolved)
    boundary.require_current_root()
    return resolved


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if len(text) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    if "\x00" in text:
        raise ValueError(f"{field} must not contain NUL")
    return text


def _bounded_int(value: Any, default: int, low: int, high: int, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{field} must be within {low}..{high}")
    return value


__all__ = [
    "DEFAULT_INSPECT_BYTES",
    'FrozenWorkspaceToolBoundary',
    "MAX_INSPECT_BYTES",
    "WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION",
    "WORKSPACE_DISCOVERY_TOOL_IDS",
    "WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION",
    "build_workspace_discovery_execution_profile",
    "build_workspace_discovery_tool_registrations",
    "build_workspace_discovery_tool_specs",
]
