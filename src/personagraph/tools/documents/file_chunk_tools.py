"""Model contract for exact reading of mounted current Document chunks."""

from __future__ import annotations

import hashlib

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress, EffectAction, EffectDescriptor, EffectResource, EffectScopeKind,
    Idempotency, Reversibility, ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration


READ_FILE_CHUNKS_TOOL_ID = "read_file_chunks"
READ_FILE_CHUNKS_CONTRACT_VERSION = "file-chunk-read-v3"
READ_FILE_CHUNKS_IMPLEMENTATION_VERSION = "4"
MAX_READ_TARGETS = 32
MAX_CHUNKS_PER_TARGET = 32
MAX_CHUNK_CONTENT_CHARS = 16_000
MAX_TOTAL_CONTENT_CHARS = 64_000


def build_read_file_chunks_registration(*, handler, effect_scope, filesystem_scope_kind=EffectScopeKind.WORKSPACE):
    if not effect_scope or filesystem_scope_kind not in {EffectScopeKind.WORKSPACE, EffectScopeKind.SESSION}:
        raise ValueError("chunk reading requires explicit filesystem authority")
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=READ_FILE_CHUNKS_TOOL_ID, contract_version=READ_FILE_CHUNKS_CONTRACT_VERSION,
            name="Read exact file chunks",
            description=("精读已入库且当前会话获准的文档块，返回正文 content，"
                         "同时返回序号、页码及位置；本工具不是分块目录查询。"
                         "每个 target 只需提供 chunk_ids；Host 解析原生 ID 的精确文件/版本。"
                         "按序号读取则提供 file_id、document_version_id 和 chunk_sequences。"
                         "可选的文件/版本字段会作为一致性约束校验，不分配额外引用。"
                         "需要总块数或其他块的位置时先用 inspect_file_chunks。"
                         "只读当前精确版本；不解析、不入库、不自动改用新版本。"),
            input_schema=_input_schema(), output_schema=_output_schema(),
            catalog_tags=("document", "file", "read"),
        ),
        implementation_version=READ_FILE_CHUNKS_IMPLEMENTATION_VERSION,
        source=ToolSourceDescriptor(kind=ToolSourceKind.LOCAL,
                                    source_id="personagraph.tools.documents.file_chunks",
                                    fingerprint=hashlib.sha256(b"file-chunk-read-v3:4").hexdigest()),
        handler=handler,
        effect_profile=ToolEffectProfile((EffectDescriptor(
            resource=EffectResource.FILESYSTEM, action=EffectAction.READ,
            scope_kind=filesystem_scope_kind, default_scope=effect_scope,
            data_egress=DataEgress.CONTENT, idempotency=Idempotency.IDEMPOTENT,
            reversibility=Reversibility.REVERSIBLE,
        ),)),
        execution_profile=ToolExecutionProfile(default_timeout_s=10, hard_timeout_s=20,
                                               max_output_bytes=400_000, max_transparent_retries=1,
                                               concurrency_class="file_chunk_read"),
    )


def _id(nullable=False):
    return {"type": ["string", "null"] if nullable else "string", "minLength": 1, "maxLength": 256}


def _object(properties, required=None):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties) if required is None else required}


def _input_schema():
    target = _object({
        "file_id": _id(), "document_version_id": _id(),
        "chunk_ids": {"type": "array", "minItems": 1, "maxItems": MAX_CHUNKS_PER_TARGET,
                      "uniqueItems": True, "items": _id()},
        "chunk_sequences": {"type": "array", "minItems": 1, "maxItems": MAX_CHUNKS_PER_TARGET,
                            "uniqueItems": True, "items": {"type": "integer", "minimum": 0}},
    }, [])
    target["oneOf"] = [{"required": ["chunk_ids"], "not": {"required": ["chunk_sequences"]}},
                       {"required": ["file_id", "document_version_id", "chunk_sequences"],
                        "not": {"required": ["chunk_ids"]}}]
    return _object({"targets": {"type": "array", "minItems": 1, "maxItems": MAX_READ_TARGETS, "items": target}})


def _output_schema():
    chunk = _object({
        "chunk_id": _id(), "sequence": {"type": "integer", "minimum": 0},
        "locator": {"type": ["string", "null"], "maxLength": 512},
        "source_pages": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 1}},
        "content": {"type": "string", "minLength": 1, "maxLength": MAX_CHUNK_CONTENT_CHARS},
        "content_truncated": {"type": "boolean"},
        "content_sha256": {"type": "string", "pattern": "[0-9a-f]{64}"},
    })
    result = _object({
        "file_id": _id(), "file_version_id": _id(True), "document_id": _id(True),
        "document_version_id": _id(), "status": {"enum": ["ready", "partial", "unavailable"]},
        "chunks": {"type": "array", "maxItems": MAX_CHUNKS_PER_TARGET, "items": chunk},
    })
    return _object({
        "contract_version": {"const": READ_FILE_CHUNKS_CONTRACT_VERSION},
        "results": {"type": "array", "maxItems": MAX_READ_TARGETS * MAX_CHUNKS_PER_TARGET, "items": result},
        "unavailable_targets": {"type": "array", "items": _object({
            "file_id": _id(True), "document_version_id": _id(True), "chunk_id": _id(True),
            "chunk_sequence": {"type": ["integer", "null"], "minimum": 0},
            "reason_code": {"type": "string"},
        })},
        "truncated": {"type": "boolean"},
    })


__all__ = ["READ_FILE_CHUNKS_TOOL_ID", "READ_FILE_CHUNKS_CONTRACT_VERSION", "build_read_file_chunks_registration"]
