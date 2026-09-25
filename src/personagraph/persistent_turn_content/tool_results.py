"""历史结果的只读合同与业务字段分页；不拥有数据库或执行工具。

历史读取返回的 source 始终指向原调用。分页工具自身的执行回执不是原始证据身份。
原始 outcome 的完整性由读取端验证；分页值不冒充原始 outcome 字节。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .evidence import L1_CALL_REF_PATTERN

DEFAULT_HISTORY_LIST_LIMIT = 20
MAX_HISTORY_LIST_LIMIT = 100
DEFAULT_HISTORY_READ_LIMIT = 16_000
MAX_HISTORY_READ_LIMIT = 64_000
MAX_ARGUMENT_SUMMARY_CHARACTERS = 512
MAX_HISTORY_OFFSET = 2_147_483_647
MAX_HISTORY_VALUE_BYTES = 128_000


class ToolHistoryError(ValueError):
    """不含数据库路径或原始内容的公开历史读取错误码。"""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolResultSource(_Contract):
    tool_call_id: str = Field(min_length=1, max_length=200)
    tool_id: str = Field(min_length=1, max_length=200)
    status: str = Field(min_length=1, max_length=64)
    call_ref: str = Field(pattern=L1_CALL_REF_PATTERN)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ToolResultHistoryItem(_Contract):
    source: ToolResultSource
    attempt_ordinal: int = Field(ge=1)
    call_ordinal: int = Field(ge=1)
    arguments_summary: str = Field(max_length=MAX_ARGUMENT_SUMMARY_CHARACTERS)
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arguments_truncated: bool
    arguments_unavailable: bool = False
    arguments_unavailable_reason: Literal["unsuccessful_findings_call"] | None = None
    total_characters: int = Field(ge=0)


class ToolResultHistoryPage(_Contract):
    contract_version: Literal["tool-result-history-page-v2"] = (
        "tool-result-history-page-v2"
    )
    results: tuple[ToolResultHistoryItem, ...] = Field(
        max_length=MAX_HISTORY_LIST_LIMIT
    )
    offset: int = Field(ge=0)
    total_results: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    partial: bool


class ToolResultRecord(_Contract):
    """Host 已验证的原始记录；工具 adapter 再投影业务内容。"""

    source: ToolResultSource
    outcome: dict[str, Any]


class ToolResultContentPage(_Contract):
    contract_version: Literal["tool-result-content-page-v2"] = "tool-result-content-page-v2"
    source: ToolResultSource
    path: str
    kind: Literal["object", "array", "string", "scalar"]
    value: Any
    offset: int = Field(ge=0)
    total_items: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    expand_paths: tuple[str, ...] = ()
    partial: bool
    source_content_compacted: bool = False


class ToolHistoryReadPort(Protocol):
    @property
    def authority_sha256(self) -> str: ...

    def list_results(
        self,
        *,
        offset: int = 0,
        limit: int = DEFAULT_HISTORY_LIST_LIMIT,
    ) -> ToolResultHistoryPage: ...

    def read_result(
        self,
        *,
        call_ref: str,
    ) -> ToolResultRecord: ...


def validate_history_window(*, offset: int, limit: int, maximum: int) -> None:
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= MAX_HISTORY_OFFSET
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= maximum
    ):
        raise ToolHistoryError("invalid_history_request")


def validate_canonical_result(text: str, expected_sha256: str) -> dict[str, Any]:
    """校验完整持久原文，禁止用重新序列化的不同字节冒充原结果。"""
    try:
        value = json.loads(text)
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (ValueError, TypeError):
        raise ToolHistoryError("tool_result_integrity_invalid") from None
    if (
        not isinstance(value, dict)
        or canonical != text
        or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256
    ):
        raise ToolHistoryError("tool_result_integrity_invalid")
    return value


def select_tool_result_chunk(result: object, chunk_id: str) -> dict[str, object]:
    """仅按结果正文中的真实 chunk_id 取块；保留祖先元数据，不跟随邻接指针。"""
    matches: list[dict[str, object]] = []

    def visit(value: object, context: list[dict[str, object]]) -> None:
        if isinstance(value, Mapping):
            if value.get("chunk_id") == chunk_id:
                matches.append({"chunk": deepcopy(dict(value)), "source_context": context})
                return
            metadata = {key: deepcopy(item) for key, item in value.items()
                        if not isinstance(item, (Mapping, list, tuple))}
            ancestors = context + [metadata] if metadata else context
            for item in value.values():
                visit(item, ancestors)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, context)

    visit(result, [])
    if not matches:
        raise ToolHistoryError("chunk_not_in_referenced_result")
    canonical = [json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True)
                 for item in matches]
    if any(item != canonical[0] for item in canonical[1:]):
        raise ToolHistoryError("chunk_identity_ambiguous_in_result")
    return matches[0]


def project_tool_result_content(
    *,
    source: ToolResultSource,
    content: dict[str, Any],
    path: str = "",
    offset: int,
    limit: int,
) -> ToolResultContentPage:
    """JSON Pointer 选业务字段；对象按字段、数组按项、正文按 Unicode 字符分页。

    大容器成员不序列化为字符串或截碎；给出可继续展开的精确路径。分页的
    单次 value 预算独立于原工具输出上限，原始完整记录始终不受修改。
    """
    validate_history_window(offset=offset, limit=limit, maximum=MAX_HISTORY_READ_LIMIT)
    selected = _select_content(content, path)
    kind = ("object" if isinstance(selected, dict) else "array" if isinstance(selected, list)
            else "string" if isinstance(selected, str) else "scalar")
    total = len(selected) if kind != "scalar" else 1
    if offset > total:
        raise ToolHistoryError("invalid_history_request")
    end = min(offset + limit, total)
    expand_paths = []
    if kind in {"object", "array"}:
        entries = list(selected.items()) if kind == "object" else list(enumerate(selected))
        value = {} if kind == "object" else []
        end = offset
        for key, item in entries[offset:offset + limit]:
            candidate = value | {key: item} if kind == "object" else [*value, item]
            if _json_size([candidate, expand_paths]) > MAX_HISTORY_VALUE_BYTES:
                if value:
                    break
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                child_path = f"{path}/{escaped}"
                if len(child_path) > 2000:
                    raise ToolHistoryError("history_member_path_too_long")
                if _json_size([value, [*expand_paths, child_path]]) > MAX_HISTORY_VALUE_BYTES:
                    break
                expand_paths.append(child_path)
            else:
                value = candidate
            end += 1
    elif kind == "string":
        value = selected[offset:end]
        while _json_size(value) > MAX_HISTORY_VALUE_BYTES:
            end = offset + max(1, (end - offset) // 2)
            value = selected[offset:end]
    else:
        value = selected if offset == 0 else None
    return ToolResultContentPage(
        source=source, path=path, kind=kind, value=value,
        offset=offset, total_items=total,
        next_offset=end if end < total else None,
        expand_paths=tuple(expand_paths), partial=offset > 0 or end < total or bool(expand_paths),
    )


def _select_content(content: dict, path: str) -> Any:
    if not isinstance(path, str) or len(path) > 2000 or (path and not path.startswith("/")):
        raise ToolHistoryError("invalid_history_path")
    current = content
    for raw in path.split("/")[1:]:
        if re.search(r"~(?![01])", raw):
            raise ToolHistoryError("invalid_history_path")
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", key) and int(key) < len(current):
            current = current[int(key)]
        else:
            raise ToolHistoryError("history_path_unavailable")
    return current


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
