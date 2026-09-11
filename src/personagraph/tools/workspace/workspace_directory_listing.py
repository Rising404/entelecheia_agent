"""Bounded, cursor-paged cognition for one workspace directory level.

The listing is derived from the same ripgrep-visible file set used by the other
workspace discovery tools.  Consequently ignored, hidden, denied and
agent-private paths keep the existing discovery semantics.  Directories are
projected from the first component of those visible file paths; empty or wholly
ignored directories are intentionally absent.

This module never opens file content and never parses, indexes or mounts a
document. It returns only public metadata and workspace-relative paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any

from ...configuration.paths import deny_reason
from ...workspace.discovery import DEFAULT_TIMEOUT_S, SkipReason, list_files, tally


DEFAULT_DIRECTORY_PAGE_SIZE = 50
MAX_DIRECTORY_PAGE_SIZE = 100
WORKSPACE_DIRECTORY_CURSOR_PATTERN = r"workspacedircursor_[0-9a-f]{40}"

_CURSOR = re.compile(WORKSPACE_DIRECTORY_CURSOR_PATTERN + r"\Z")


def list_workspace_directory_page(
    root: Path,
    *,
    session_id: str,
    scope_path: str,
    cursor: object,
    limit: int,
    hidden: bool = False,
    respect_ignore: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_scan_paths: int | None = None,
    path_allowed: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    """Return one stable page of direct discoverable children beneath ``root``.

    The cursor is bound to the selected scope and the complete public listing
    snapshot.  A changed directory therefore rejects an old cursor instead of
    silently skipping or duplicating entries.
    """

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must not be empty")
    if not isinstance(scope_path, str) or not scope_path:
        raise ValueError("scope_path must not be empty")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_DIRECTORY_PAGE_SIZE
    ):
        raise ValueError(
            f"limit must be within 1..{MAX_DIRECTORY_PAGE_SIZE}"
        )

    paths, report = list_files(
        root,
        hidden=hidden,
        respect_ignore=respect_ignore,
        timeout_s=timeout_s,
        max_paths=max_scan_paths,
    )
    items_by_name: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(
                "The directory search backend returned an unsafe path."
            )
        source = root / relative
        if not _path_is_allowed(source, path_allowed):
            report.skip(SkipReason.DENIED_BY_POLICY)
            continue

        name = relative.parts[0]
        if name in items_by_name:
            continue
        direct = root / name
        if not _path_is_allowed(direct, path_allowed):
            report.skip(SkipReason.DENIED_BY_POLICY)
            continue

        public_path = _path_in_scope(scope_path, name)
        if len(relative.parts) > 1:
            try:
                if not direct.is_dir():
                    raise OSError("direct child is not a directory")
            except OSError:
                report.skip(SkipReason.IO_ERROR)
                continue
            items_by_name[name] = {
                "path": public_path,
                "name": name,
                "kind": "directory",
            }
            continue

        try:
            stat = direct.stat()
            if not direct.is_file():
                raise OSError("direct child is not a regular file")
        except PermissionError:
            report.skip(SkipReason.PERMISSION_DENIED)
            continue
        except OSError:
            report.skip(SkipReason.IO_ERROR)
            continue
        item: dict[str, object] = {
            "path": public_path,
            "name": name,
            "kind": "file",
            "size": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }
        items_by_name[name] = item

    ordered = tuple(
        sorted(
            items_by_name.values(),
            key=lambda item: (
                0 if item["kind"] == "directory" else 1,
                str(item["name"]).casefold(),
                str(item["name"]),
            ),
        )
    )
    snapshot_sha256 = _listing_snapshot_sha256(
        session_id=session_id,
        scope_path=scope_path,
        items=ordered,
        scan_truncated=report.truncated,
    )
    offset = _cursor_offset(
        cursor,
        session_id=session_id,
        scope_path=scope_path,
        snapshot_sha256=snapshot_sha256,
        item_count=len(ordered),
    )
    page = ordered[offset : offset + limit]
    next_offset = offset + len(page)
    next_cursor = (
        _cursor_for_offset(
            session_id=session_id,
            scope_path=scope_path,
            snapshot_sha256=snapshot_sha256,
            offset=next_offset,
        )
        if next_offset < len(ordered)
        else None
    )
    return {
        "scope": {"session_id": session_id, "path": scope_path},
        "items": [dict(item) for item in page],
        "returned": len(page),
        "total": len(ordered),
        "truncated": bool(next_cursor is not None or report.truncated),
        "next_cursor": next_cursor,
        "scan_truncated": report.truncated,
        "total_relation": "lower_bound" if report.truncated else "exact",
        "scanned_files": report.scanned_files,
        "ignore_rules_applied": respect_ignore,
        "skipped": [
            {"reason": reason.value, "count": count}
            for reason, count in tally(report.skipped)
        ],
    }


def _path_is_allowed(
    candidate: Path,
    path_allowed: Callable[[Path], bool] | None,
) -> bool:
    if deny_reason(candidate) is not None:
        return False
    if path_allowed is None:
        return True
    try:
        return bool(path_allowed(candidate))
    except (OSError, RuntimeError, ValueError):
        return False


def _path_in_scope(scope_path: str, name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or ".." in candidate.parts:
        raise RuntimeError("The directory search backend returned an unsafe path.")
    return name if scope_path == "." else f"{scope_path}/{name}"


def _listing_snapshot_sha256(
    *,
    session_id: str,
    scope_path: str,
    items: tuple[Mapping[str, object], ...],
    scan_truncated: bool,
) -> str:
    return _sha256_json(
        {
            "schema_version": "workspace-directory-listing-snapshot-v1",
            "session_id": session_id,
            "scope_path": scope_path,
            "items": [dict(item) for item in items],
            "scan_truncated": scan_truncated,
        }
    )


def _cursor_for_offset(
    *,
    session_id: str,
    scope_path: str,
    snapshot_sha256: str,
    offset: int,
) -> str:
    digest = _sha256_json(
        {
            "schema_version": "workspace-directory-cursor-v1",
            "session_id": session_id,
            "scope_path": scope_path,
            "snapshot_sha256": snapshot_sha256,
            "offset": offset,
        }
    )
    return "workspacedircursor_" + digest[:40]


def _cursor_offset(
    cursor: object,
    *,
    session_id: str,
    scope_path: str,
    snapshot_sha256: str,
    item_count: int,
) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or _CURSOR.fullmatch(cursor) is None:
        raise ValueError("cursor is invalid for this directory listing")
    for offset in range(1, item_count):
        expected = _cursor_for_offset(
            session_id=session_id,
            scope_path=scope_path,
            snapshot_sha256=snapshot_sha256,
            offset=offset,
        )
        if hmac.compare_digest(cursor, expected):
            return offset
    raise ValueError("cursor is invalid for this directory listing")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_DIRECTORY_PAGE_SIZE",
    "MAX_DIRECTORY_PAGE_SIZE",
    "WORKSPACE_DIRECTORY_CURSOR_PATTERN",
    "list_workspace_directory_page",
]
