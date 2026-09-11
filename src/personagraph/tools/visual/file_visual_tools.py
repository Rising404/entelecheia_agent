"""Model-facing visual contracts addressed by shared File and FileVersion IDs.

Metadata listing does not parse, render or invoke a model. Reading retains the
truthful local/external effect split and exact source binding; external analysis
of readable files is allowed by default without a separate approval step.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from ...input_processing.documents.contracts import DocumentNonTextKind
from ...input_processing.vision.contracts import (
    VisionDetail,
    VisionPurpose,
    VisionRegion,
    VisionStatus,
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
from ..registration import ToolExecutionProfile, ToolRegistration
from .failure_reporting import failure_diagnostics_schema
from .question_contract import visual_question_constraint, visual_question_schema


LIST_FILE_VISUALS_TOOL_ID = "list_file_visuals"
LIST_FILE_VISUALS_CONTRACT_VERSION = "file-visual-list-v2"
READ_FILE_VISUALS_TOOL_ID = "read_file_visuals"
READ_FILE_VISUALS_CONTRACT_VERSION = "file-visual-read-v2"
FILE_VISUAL_IMPLEMENTATION_VERSION = "3"
FILE_VISUAL_TOOL_IDS = (
    LIST_FILE_VISUALS_TOOL_ID,
    READ_FILE_VISUALS_TOOL_ID,
)
FILE_VISUAL_SOURCE_ID = "personagraph.files.visuals"
FILE_VISUAL_SOURCE_DISPLAY_NAME = "Shared file-version visual observer"

DEFAULT_VISUAL_PAGE_SIZE = 20
MAX_LIST_PAGE_FILTERS = 8
MAX_VISUALS_PER_PAGE = 64
MAX_VISUALS_PER_READ = 3
MAX_PAGES_PER_VISUAL = 64
MAX_OBSERVATION_CHARS = 16_000

_SAFE_SCOPE = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_SAFE_CODE = r"^[a-z][a-z0-9_]{0,95}$"
_OPAQUE_REF = r"^[a-z][a-z0-9_-]{7,255}$"

FileVisualHandler = Callable[[dict[str, Any]], dict[str, Any]]


def build_list_file_visuals_registration(
    *,
    handler: FileVisualHandler,
    effect_scope: str,
) -> ToolRegistration:
    """针对已准备的视觉元数据注册纯分页发现。"""

    _require_effect_scope(effect_scope)
    return ToolRegistration(
        spec=build_file_visual_tool_specs()[0],
        implementation_version=FILE_VISUAL_IMPLEMENTATION_VERSION,
        source=_source(LIST_FILE_VISUALS_TOOL_ID),
        handler=handler,
        effect_profile=build_list_file_visuals_effect_profile(
            default_scope=effect_scope,
        ),
        execution_profile=build_list_file_visuals_execution_profile(),
    )


def build_read_file_visuals_registration(
    *,
    handler: FileVisualHandler,
    effect_scope: str,
    sends_externally: bool,
) -> ToolRegistration:
    """使用如实配置的数据外传配置注册有界视觉读取。"""

    _require_effect_scope(effect_scope)
    if not isinstance(sends_externally, bool):
        raise ValueError("sends_externally must be a boolean")
    return ToolRegistration(
        spec=build_file_visual_tool_specs()[1],
        implementation_version=FILE_VISUAL_IMPLEMENTATION_VERSION,
        source=_source(READ_FILE_VISUALS_TOOL_ID),
        handler=handler,
        effect_profile=build_read_file_visuals_effect_profile(
            default_scope=effect_scope,
            sends_externally=sends_externally,
        ),
        execution_profile=build_read_file_visuals_execution_profile(
            sends_externally=sends_externally,
        ),
    )


def build_file_visual_tool_specs() -> tuple[ToolSpec, ToolSpec]:
    """Return the two file-version model contracts in canonical order."""

    return (
        ToolSpec(
            tool_id=LIST_FILE_VISUALS_TOOL_ID,
            contract_version=LIST_FILE_VISUALS_CONTRACT_VERSION,
            name="列出文件视觉候选区域",
            description=(
                "列出指定文件版本已检测的视觉候选区域及其 ID、页码和粗粒度类型。"
                "独立 JPG/PNG 通常对应一个区域；PDF 可能按嵌入对象或版面区域拆分，"
                "同一张语义上的图可能对应多个区域。区域数量和 kind 标签不代表图片数量或已确认内容。"
                "PDF 需指定最多八个页码；未指定时返回空清单并要求选页。图片清单可用返回的 cursor 续取。"
                "本工具只读元信息，不渲染像素、不调用视觉模型；读内容请用 read_file_visuals，"
                "需要整页关系时可用 analyze_pdf_page。"
            ),
            input_schema=_list_input_schema(),
            output_schema=_list_output_schema(),
            catalog_tags=("document", "file", "list", "read"),
        ),
        ToolSpec(
            tool_id=READ_FILE_VISUALS_TOOL_ID,
            contract_version=READ_FILE_VISUALS_CONTRACT_VERSION,
            name="读取文件视觉区域内容",
            description=(
                "根据 list_file_visuals 返回的 ID 读取最多三个视觉候选区域的内容，"
                "为每项指定用途、清晰度和观察范围。候选区域可能只是完整图片的一部分；"
                "purpose=question 时在 question 中提供具体自然语言问题；"
                "其它用途省略 question 或设为 null。"
                "需要关联上下文时可扩大范围或读取整页。可读文件默认允许发送给配置的视觉服务，"
                "无需另外请求用户批准；执行前仍校验实际来源和文件版本。"
            ),
            input_schema=_read_input_schema(),
            output_schema=_read_output_schema(),
            catalog_tags=("document", "file", "read"),
        ),
    )


def build_list_file_visuals_effect_profile(
    *,
    default_scope: str,
) -> ToolEffectProfile:
    """Return the exact metadata-only list effect for one Session file scope."""

    _require_effect_scope(default_scope, allow_template=True)
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=EffectAction.READ,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=default_scope,
                data_egress=DataEgress.METADATA,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def build_read_file_visuals_effect_profile(
    *,
    default_scope: str,
    sends_externally: bool,
) -> ToolEffectProfile:
    """Return the truthful local or external visual-read effect shape."""

    _require_effect_scope(default_scope, allow_template=True)
    if not isinstance(sends_externally, bool):
        raise ValueError("sends_externally must be a boolean")
    effects = [
        EffectDescriptor(
            resource=EffectResource.FILESYSTEM,
            action=EffectAction.READ,
            scope_kind=EffectScopeKind.SESSION,
            default_scope=default_scope,
            data_egress=DataEgress.CONTENT,
            idempotency=Idempotency.IDEMPOTENT,
            reversibility=Reversibility.REVERSIBLE,
        )
    ]
    if sends_externally:
        effects.append(
            EffectDescriptor(
                resource=EffectResource.NETWORK,
                action=EffectAction.TRANSMIT,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=default_scope,
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.NOT_IDEMPOTENT,
                reversibility=Reversibility.IRREVERSIBLE,
            )
        )
        effects.append(
            EffectDescriptor(
                resource=EffectResource.RUNTIME_STATE,
                action=EffectAction.UPDATE,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=default_scope,
                data_egress=DataEgress.NONE,
                idempotency=Idempotency.DEDUPLICATED,
                reversibility=Reversibility.UNKNOWN,
            )
        )
    return ToolEffectProfile(tuple(effects))


def build_list_file_visuals_execution_profile() -> ToolExecutionProfile:
    """Return the stable execution envelope for metadata discovery."""

    return ToolExecutionProfile(
        default_timeout_s=15.0,
        hard_timeout_s=30.0,
        max_output_bytes=96_000,
        max_transparent_retries=1,
        concurrency_class="file_visual_list",
    )


def build_read_file_visuals_execution_profile(
    *,
    sends_externally: bool,
) -> ToolExecutionProfile:
    """Return the retry envelope matching the selected adapter effect."""

    if not isinstance(sends_externally, bool):
        raise ValueError("sends_externally must be a boolean")
    return ToolExecutionProfile(
        default_timeout_s=120.0,
        hard_timeout_s=240.0,
        max_output_bytes=128_000,
        max_transparent_retries=0 if sends_externally else 1,
        concurrency_class="file_visual_read",
    )


def _source(tool_id: str) -> ToolSourceDescriptor:
    source_fingerprint = hashlib.sha256(
        (
            "file-visual-tool-source-v2:"
            f"{FILE_VISUAL_SOURCE_ID}:{tool_id}:"
            f"{FILE_VISUAL_IMPLEMENTATION_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    return ToolSourceDescriptor(
        ToolSourceKind.LOCAL,
        FILE_VISUAL_SOURCE_ID,
        fingerprint=source_fingerprint,
        display_name=FILE_VISUAL_SOURCE_DISPLAY_NAME,
    )


def _list_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["file_id", "file_version_id"],
        "properties": {
            "file_id": _identity_schema(),
            "file_version_id": _identity_schema(),
            "pages": _list_pages_schema(),
            "cursor": _opaque_ref_schema(),
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_VISUALS_PER_PAGE,
                "default": DEFAULT_VISUAL_PAGE_SIZE,
            },
        },
    }


def _list_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "contract_version",
            "file_id",
            "file_version_id",
            "status",
            "visuals",
            "next_cursor",
            "reason_code",
        ],
        "properties": {
            "contract_version": {
                "type": "string",
                "const": LIST_FILE_VISUALS_CONTRACT_VERSION,
            },
            "file_id": _identity_schema(),
            "file_version_id": _identity_schema(),
            "status": {
                "type": "string",
                "enum": ["ready", "not_established", "stale", "blocked"],
            },
            "visuals": {
                "type": "array",
                "maxItems": MAX_VISUALS_PER_PAGE,
                "items": _listed_visual_schema(),
            },
            "next_cursor": _nullable_opaque_ref_schema(),
            "reason_code": _reason_code_schema(),
        },
    }


def _read_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["requests"],
        "properties": {
            "requests": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_VISUALS_PER_READ,
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "file_id",
                        "file_version_id",
                        "visual_unit_id",
                        "purpose",
                        "detail",
                        "region",
                    ],
                    "allOf": [visual_question_constraint()],
                    "properties": {
                        "file_id": _identity_schema(),
                        "file_version_id": _identity_schema(),
                        "visual_unit_id": _identity_schema(),
                        "purpose": _purpose_schema(),
                        "question": visual_question_schema(),
                        "detail": _detail_schema(),
                        "region": _region_schema(),
                    },
                },
            },
        },
    }


def _read_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract_version", "results"],
        "properties": {
            "contract_version": {
                "type": "string",
                "const": READ_FILE_VISUALS_CONTRACT_VERSION,
            },
            "results": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_VISUALS_PER_READ,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "file_id",
                        "file_version_id",
                        "visual_unit_id",
                        "pages",
                        "kind",
                        "purpose",
                        "detail",
                        "region",
                        "status",
                        "observation",
                        "reason_code",
                    ],
                    "properties": {
                        "file_id": _identity_schema(),
                        "file_version_id": _identity_schema(),
                        "visual_unit_id": _identity_schema(),
                        "pages": _pages_schema(),
                        "kind": _kind_schema(),
                        "purpose": _purpose_schema(),
                        "question": visual_question_schema(),
                        "detail": _detail_schema(),
                        "region": _region_schema(),
                        "status": {
                            "type": "string",
                            "enum": [
                                *[item.value for item in VisionStatus],
                                "blocked",
                            ],
                        },
                        "observation": {
                            "type": ["string", "null"],
                            "minLength": 1,
                            "maxLength": MAX_OBSERVATION_CHARS,
                        },
                        "picture_id": _identity_schema(),
                        "picture_unit_id": _identity_schema(),
                        "observation_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                        },
                        "uncertainty": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reason_code": _reason_code_schema(),
                        "failure_diagnostics": failure_diagnostics_schema(),
                        "background_wait_active": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _listed_visual_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "visual_unit_id",
            "pages",
            "kind",
            "allowed_purposes",
            "default_purpose",
        ],
        "properties": {
            "visual_unit_id": _identity_schema(),
            "pages": _pages_schema(),
            "kind": _kind_schema(),
            "allowed_purposes": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(VisionPurpose),
                "uniqueItems": True,
                "items": _purpose_schema(),
            },
            "default_purpose": _purpose_schema(),
        },
    }


def _identity_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": 256}


def _opaque_ref_schema() -> dict[str, Any]:
    return {"type": "string", "pattern": _OPAQUE_REF}


def _nullable_opaque_ref_schema() -> dict[str, Any]:
    return {
        "type": ["string", "null"],
        "pattern": _OPAQUE_REF,
    }


def _pages_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_PAGES_PER_VISUAL,
        "uniqueItems": True,
        "items": {"type": "integer", "minimum": 1},
    }


def _list_pages_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_LIST_PAGE_FILTERS,
        "uniqueItems": True,
        "items": {"type": "integer", "minimum": 1},
    }


def _kind_schema() -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in DocumentNonTextKind]}


def _purpose_schema() -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in VisionPurpose]}


def _detail_schema() -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in VisionDetail]}


def _region_schema() -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in VisionRegion]}


def _reason_code_schema() -> dict[str, Any]:
    return {"type": ["string", "null"], "pattern": _SAFE_CODE}


def _require_effect_scope(value: str, *, allow_template: bool = False) -> None:
    if allow_template and value == "*":
        return
    if not isinstance(value, str) or _SAFE_SCOPE.fullmatch(value) is None:
        raise ValueError("effect_scope must be a bounded Host-generated scope")


__all__ = [
    "DEFAULT_VISUAL_PAGE_SIZE",
    "FILE_VISUAL_IMPLEMENTATION_VERSION",
    "FILE_VISUAL_SOURCE_DISPLAY_NAME",
    "FILE_VISUAL_SOURCE_ID",
    "FILE_VISUAL_TOOL_IDS",
    "LIST_FILE_VISUALS_CONTRACT_VERSION",
    "LIST_FILE_VISUALS_TOOL_ID",
    "MAX_LIST_PAGE_FILTERS",
    "MAX_OBSERVATION_CHARS",
    "MAX_PAGES_PER_VISUAL",
    "MAX_VISUALS_PER_PAGE",
    "MAX_VISUALS_PER_READ",
    "READ_FILE_VISUALS_CONTRACT_VERSION",
    "READ_FILE_VISUALS_TOOL_ID",
    "build_file_visual_tool_specs",
    "build_list_file_visuals_effect_profile",
    "build_list_file_visuals_execution_profile",
    "build_list_file_visuals_registration",
    "build_read_file_visuals_effect_profile",
    "build_read_file_visuals_execution_profile",
    "build_read_file_visuals_registration",
]
