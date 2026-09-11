"""文档字面查找和分块目录的模型协议；不执行存储查询。"""

from __future__ import annotations

import hashlib

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress, EffectAction, EffectDescriptor, EffectResource, EffectScopeKind,
    Idempotency, Reversibility, ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration

FILE_INSPECTION_TOOL_IDS = ("inspect_file_chunks", "search_file_text")
MAX_TARGETS = 32
MAX_LOCATIONS = 128
MAX_MATCHES = 100


def build_file_inspection_registration(*, tool_id, handler, effect_scope):
    if tool_id not in FILE_INSPECTION_TOOL_IDS or not effect_scope:
        raise ValueError("document inspection requires a known tool and explicit scope")
    searching = tool_id == "search_file_text"
    description = (
        "在指定文件已经解析保存的原始文本中做字面查找，返回全部匹配总数及一页匹配位置。"
        "不按相关度筛选，不解析文件，不统计语义对象；检索块的重叠和补充标题不会重复计数。"
        "case_sensitive 默认 false（Unicode casefold）；whitespace 默认 exact，collapse "
        "可将连续空白/换行等同一个空格；匹配允许重叠。跨元素用换行连接。"
        "scan_complete 只表示已扫描全部保存的解析文本，processing_status 表示解析本身的覆盖。"
        "旧文档未保存原始文本时明确返回 source_text_unavailable。"
        if searching else
        "批量查看已解析文件的分块目录与位置，不返回正文。返回总块数、从 0 开始的序列范围、"
        "页数，以及每个目标块的 chunk_id、sequence、页码和位置。"
        "每个 target 可选 chunk_ids 或 chunk_sequences 定位已有块；不指定时按 start_sequence/limit "
        "分页列出目录。next_sequence 可续查下一页；需要正文时调用 read_file_chunks。"
    )
    return ToolRegistration(
        spec=ToolSpec(tool_id=tool_id, contract_version="file-inspection-v1",
                      name="Search parsed file text" if searching else "Inspect file chunk locations",
                      description=description, input_schema=_input_schema(searching),
                      output_schema=_output_schema(searching),
                      catalog_tags=("file", "document", "search" if searching else "read")),
        implementation_version="1",
        source=ToolSourceDescriptor(kind=ToolSourceKind.LOCAL,
                                   source_id="personagraph.tools.documents.file_inspection",
                                   fingerprint=hashlib.sha256(tool_id.encode()).hexdigest()),
        handler=handler,
        effect_profile=ToolEffectProfile((EffectDescriptor(
            resource=EffectResource.FILESYSTEM,
            action=EffectAction.SEARCH if searching else EffectAction.READ,
            scope_kind=EffectScopeKind.SESSION, default_scope=effect_scope,
            data_egress=DataEgress.CONTENT if searching else DataEgress.METADATA,
            idempotency=Idempotency.IDEMPOTENT, reversibility=Reversibility.REVERSIBLE,
        ),)),
        execution_profile=ToolExecutionProfile(default_timeout_s=30, hard_timeout_s=60,
                                               max_output_bytes=600_000, max_transparent_retries=1,
                                               concurrency_class="file_document_inspection"),
    )


def _object(properties, required=None):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties) if required is None else required}


def _id(nullable=False):
    return {"type": ["string", "null"] if nullable else "string", "minLength": 1, "maxLength": 256}


def _integer(maximum=None, nullable=False):
    return {"type": ["integer", "null"] if nullable else "integer", "minimum": 0,
            **({"maximum": maximum} if maximum is not None else {})}


def _input_schema(searching):
    target = _object({"file_id": _id(), "document_version_id": _id()})
    if not searching:
        target["properties"].update({
            "chunk_ids": {"type": "array", "minItems": 1, "maxItems": MAX_LOCATIONS,
                          "uniqueItems": True, "items": _id()},
            "chunk_sequences": {"type": "array", "minItems": 1, "maxItems": MAX_LOCATIONS,
                                "uniqueItems": True, "items": _integer()},
            "start_sequence": _integer(), "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LOCATIONS},
        })
        target["not"] = {"required": ["chunk_ids", "chunk_sequences"]}
    properties = {"targets": {"type": "array", "minItems": 1, "maxItems": MAX_TARGETS, "items": target}}
    required = ["targets"]
    if searching:
        properties.update({
            "text": {"type": "string", "minLength": 1, "maxLength": 2000},
            "case_sensitive": {"type": "boolean", "default": False},
            "whitespace": {"enum": ["exact", "collapse"], "default": "exact"},
            "match_offset": _integer(), "limit": {"type": "integer", "minimum": 1, "maximum": MAX_MATCHES},
        })
        required.append("text")
    return _object(properties, required)


def _output_schema(searching):
    pages = {"type": "array", "items": {"type": "integer", "minimum": 1}, "uniqueItems": True}
    fields = {"file_id": _id(), "document_version_id": _id(), "file_version_id": _id(True),
              "status": {"enum": ["ready", "partial", "unavailable"]}, "reason_code": _id(True),
              "processing_status": {"type": ["string", "null"]}}
    if searching:
        fields.update({
            "scan_complete": {"type": "boolean"}, "total_match_count": _integer(nullable=True),
            "scanned_element_count": _integer(), "next_match_offset": _integer(nullable=True),
            "matches_truncated": {"type": "boolean"},
            "matches": {"type": "array", "maxItems": MAX_MATCHES, "items": _object({
                "match_index": _integer(), "text": {"type": "string"},
                "start_element_index": _integer(), "end_element_index": _integer(),
                "start_character": _integer(), "end_character": _integer(),
                "element_ids": {"type": "array", "items": _id()}, "source_pages": pages,
                "locators": {"type": "array", "items": {"type": "string"}},
                "chunk_sequences": {"type": "array", "items": _integer()},
            })},
        })
    else:
        fields.update({
            "total_chunk_count": _integer(nullable=True), "first_sequence": _integer(nullable=True),
            "last_sequence": _integer(nullable=True), "physical_page_count": _integer(nullable=True),
            "next_sequence": _integer(nullable=True),
            "chunks": {"type": "array", "maxItems": MAX_LOCATIONS, "items": _object({
                "chunk_id": _id(), "sequence": _integer(), "locator": {"type": "string"}, "source_pages": pages,
            })},
            "unavailable_selectors": {"type": "array", "items": {"type": ["string", "integer"]}},
        })
    return _object({"contract_version": {"const": "file-inspection-v1"},
                    "results": {"type": "array", "maxItems": MAX_TARGETS, "items": _object(fields)}})
