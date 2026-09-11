"""面向模型的工作区认知与发现工具之线上契约。

本模块只持有 JSON Schema 形态的常量和构建器。它不冻结目录、不遍历文件、不调用 ripgrep、
不注册工具，也不依赖 Runtime。``workspace_tools`` 保留这些执行与组合职责，
同时重新导出本模块现有的辅助函数名。
"""

from __future__ import annotations

from typing import Any

from ...workspace.discovery import DEFAULT_DEPTH, DEFAULT_LIMIT, DEFAULT_MAX_ENTRIES
from .workspace_directory_listing import (
    DEFAULT_DIRECTORY_PAGE_SIZE,
    MAX_DIRECTORY_PAGE_SIZE,
    WORKSPACE_DIRECTORY_CURSOR_PATTERN,
)


WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION = "workspace-discovery-v1"

MAX_DEPTH = 4
DISCOVERY_MAX_ENTRIES = 60
DISCOVERY_MAX_RESULTS = 50
DEFAULT_INSPECT_BYTES = 8_000
# JSON 转义可以把有效控制字符扩展为六个字节（``\u00xx``）。因此，即使面对最坏情况下的
# 有效 UTF-8 前缀，而非只有普通文本，此上限也能让完整信封保持在 90KB ToolResult 限制内。
MAX_INSPECT_BYTES = 12_000


_SUBPATH_SCHEMA = {
    "type": "string",
    "maxLength": 512,
    "description": "Optional folder inside the working directory, to narrow the search.",
}

_SKIPPED_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason", "count"],
        "properties": {
            "reason": {
                "enum": [
                    "permission_denied", "too_large", "decode_failed",
                    "io_error", "denied_by_policy",
                ]
            },
            "count": {"type": "integer", "minimum": 1},
        },
    },
    "description": "Paths that were seen but not usable. Empty means nothing was excluded.",
}

# 只有 Host 已将匹配文件冻结到当前候选项范围时，选择器才会出现。它被特意设为可选：
# 通用发现仍会返回不符合文档准备条件的普通文件，且绝不暴露私有来源权威信息或指纹。


def _overview_input_schema(
    *, max_entries_ceiling: int = DISCOVERY_MAX_ENTRIES
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "depth": {"type": "integer", "minimum": 1, "maximum": MAX_DEPTH, "default": DEFAULT_DEPTH},
            "max_entries": {
                "type": "integer", "minimum": 1,
                "maximum": max_entries_ceiling, "default": DEFAULT_MAX_ENTRIES,
            },
            "subpath": _SUBPATH_SCHEMA,
        },
    }


def _find_files_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Filename glob, for example '*.pdf' or '*contract*'.",
            },
            "subpath": _SUBPATH_SCHEMA,
            "sort": {"enum": ["path", "mtime", "size"], "default": "path"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": DISCOVERY_MAX_RESULTS,
                "default": DEFAULT_LIMIT,
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100_000,
                "default": 0,
            },
        },
    }


def _list_workspace_directory_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subpath": _SUBPATH_SCHEMA,
            "cursor": {
                "type": "string",
                "pattern": WORKSPACE_DIRECTORY_CURSOR_PATTERN,
                "description": (
                    "Opaque continuation cursor returned by the preceding page."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_DIRECTORY_PAGE_SIZE,
                "default": DEFAULT_DIRECTORY_PAGE_SIZE,
            },
        },
    }


def _search_text_files_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content"],
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
                "description": "Literal UTF-8 text to find inside text files.",
            },
            "subpath": _SUBPATH_SCHEMA,
            "sort": {"enum": ["path", "mtime", "size"], "default": "path"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": DISCOVERY_MAX_RESULTS,
                "default": DEFAULT_LIMIT,
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100_000,
                "default": 0,
            },
        },
    }


def _inspect_file_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
                "description": "File path relative to the frozen workspace root.",
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_INSPECT_BYTES,
                "default": DEFAULT_INSPECT_BYTES,
                "description": "Maximum UTF-8 prefix bytes returned for a text file.",
            },
        },
    }


def _discovery_scope_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["session_id", "path"],
        "properties": {
            "session_id": {"type": "string", "minLength": 1},
            "path": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Resolved scope relative to the frozen workspace root; '.' is "
                    "the root itself."
                ),
            },
        },
    }


def _discovery_envelope_properties(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": _discovery_scope_schema(),
        "items": {"type": "array", "items": item_schema},
        "returned": {"type": "integer", "minimum": 0},
        "total": {"type": "integer", "minimum": 0},
        "truncated": {"type": "boolean"},
        "skipped": _SKIPPED_SCHEMA,
    }


def _discovery_overview_output_schema() -> dict[str, Any]:
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "files", "bytes", "kinds"],
        "properties": {
            "path": {"type": "string"},
            "files": {"type": "integer", "minimum": 0},
            "bytes": {"type": "integer", "minimum": 0},
            "kinds": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["suffix", "files"],
                    "properties": {
                        "suffix": {"type": "string"},
                        "files": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
    }
    properties = _discovery_envelope_properties(item_schema)
    properties.update(
        {
            "depth": {"type": "integer", "minimum": 1},
            "total_files": {"type": "integer", "minimum": 0},
            "omitted": {"type": "integer", "minimum": 0},
            "scan_truncated": {"type": "boolean"},
            "total_relation": {"enum": ["exact", "lower_bound"]},
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scope",
            "items",
            "returned",
            "total",
            "truncated",
            "skipped",
            "depth",
            "total_files",
            "omitted",
            "scan_truncated",
            "total_relation",
        ],
        "properties": properties,
    }


def _find_files_output_schema() -> dict[str, Any]:
    properties = _discovery_envelope_properties(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "size", "modified_ns"],
            "properties": {
                "path": {"type": "string"},
                "size": {"type": "integer", "minimum": 0},
                "modified_ns": {"type": "integer", "minimum": 0},
            },
        }
    )
    properties.update(_discovery_search_metadata_schema())
    return _discovery_search_output_schema(properties)


def _list_workspace_directory_output_schema() -> dict[str, Any]:
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "name", "kind"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "name": {"type": "string", "minLength": 1},
            "kind": {"enum": ["directory", "file"]},
            "size": {"type": "integer", "minimum": 0},
            "modified_ns": {"type": "integer", "minimum": 0},
        },
        "allOf": [
            {
                "if": {
                    "properties": {"kind": {"const": "file"}},
                    "required": ["kind"],
                },
                "then": {"required": ["size", "modified_ns"]},
            }
        ],
    }
    properties = _discovery_envelope_properties(item_schema)
    properties.update(
        {
            "next_cursor": {
                "type": ["string", "null"],
                "pattern": WORKSPACE_DIRECTORY_CURSOR_PATTERN,
            },
            "scanned_files": {"type": "integer", "minimum": 0},
            "ignore_rules_applied": {"type": "boolean"},
            "scan_truncated": {"type": "boolean"},
            "total_relation": {"enum": ["exact", "lower_bound"]},
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scope",
            "items",
            "returned",
            "total",
            "truncated",
            "next_cursor",
            "scanned_files",
            "ignore_rules_applied",
            "scan_truncated",
            "total_relation",
            "skipped",
        ],
        "properties": properties,
    }


def _search_text_files_output_schema() -> dict[str, Any]:
    properties = _discovery_envelope_properties(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "line", "snippet", "size"],
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
                "snippet": {"type": "string"},
                "size": {"type": "integer", "minimum": 0},
            },
        }
    )
    properties.update(_discovery_search_metadata_schema())
    properties.update(
        {
            "max_matches_per_file": {"type": "integer", "minimum": 1},
            "max_file_size": {"type": "string", "minLength": 1},
        }
    )
    output = _discovery_search_output_schema(properties)
    output["required"].extend(["max_matches_per_file", "max_file_size"])
    return output


def _discovery_search_metadata_schema() -> dict[str, Any]:
    return {
        "offset": {"type": "integer", "minimum": 0},
        "scanned_files": {"type": "integer", "minimum": 0},
        "ignore_rules_applied": {"type": "boolean"},
        "scan_truncated": {"type": "boolean"},
        "total_relation": {"enum": ["exact", "lower_bound"]},
    }


def _discovery_search_output_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scope",
            "items",
            "returned",
            "total",
            "truncated",
            "skipped",
            "offset",
            "scanned_files",
            "ignore_rules_applied",
            "scan_truncated",
            "total_relation",
        ],
        "properties": properties,
    }


def _inspect_file_output_schema() -> dict[str, Any]:
    properties = _discovery_envelope_properties(
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "path",
                "size",
                "modified_ns",
                "media_type",
                "content_kind",
            ],
            "properties": {
                "path": {"type": "string"},
                "size": {"type": "integer", "minimum": 0},
                "modified_ns": {"type": "integer", "minimum": 0},
                "media_type": {"type": "string", "minLength": 1},
                "content_kind": {
                    "enum": ["text", "binary", "undecodable_text"]
                },
                "text": {"type": "string"},
            },
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scope",
            "items",
            "returned",
            "total",
            "truncated",
            "skipped",
        ],
        "properties": properties,
    }


__all__ = [
    "DEFAULT_INSPECT_BYTES",
    "DISCOVERY_MAX_ENTRIES",
    "DISCOVERY_MAX_RESULTS",
    "MAX_DEPTH",
    "MAX_INSPECT_BYTES",
    "WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION",
]
