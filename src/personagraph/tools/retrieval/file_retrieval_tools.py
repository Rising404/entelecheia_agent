"""Model contract for read-only retrieval of already prepared File evidence."""

from __future__ import annotations

import hashlib
from typing import Callable

from ...retrieval.contracts import FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY
from ...retrieval.tooling.contracts import (
    DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT, DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT,
    MAX_FILE_RETRIEVAL_ITEMS, MAX_FILE_RETRIEVAL_QUERIES,
    MAX_FILE_RETRIEVAL_TOKEN_LIMIT, MAX_RETRIEVAL_QUERY_CHARS,
    MAX_RETRIEVAL_QUERY_TOKENS,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress, EffectAction, EffectDescriptor, EffectResource, EffectScopeKind,
    Idempotency, Reversibility, ToolEffectProfile,
)
from ..registration import CancellationMode, ToolExecutionProfile, ToolRegistration


RETRIEVE_FILES_TOOL_ID = "retrieve_files"
RETRIEVE_FILES_CONTRACT_VERSION = "file-retrieval-v8"
RETRIEVE_FILES_IMPLEMENTATION_VERSION = "7"
MAX_FILES_PER_RETRIEVE = 64
MAX_QUERY_CHARS = MAX_RETRIEVAL_QUERY_CHARS
MAX_CHUNK_CONTENT_CHARS = 16_000
MAX_FILE_RETRIEVAL_OUTPUT_BYTES = 1_048_576


def build_retrieve_files_registration(
    *, handler: Callable, effect_scope: str,
    filesystem_scope_kind: EffectScopeKind = EffectScopeKind.WORKSPACE,
) -> ToolRegistration:
    if not effect_scope or filesystem_scope_kind not in {
        EffectScopeKind.WORKSPACE, EffectScopeKind.SESSION,
    }:
        raise ValueError("file retrieval requires a bounded filesystem authority")
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=RETRIEVE_FILES_TOOL_ID,
            contract_version=RETRIEVE_FILES_CONTRACT_VERSION,
            name="Retrieve file evidence",
            description=(
                "检索当前 Session 已授权且已准备的 Document chunks 与已经发布的 "
                "Picture observations；既往 agent outputs 不在本语料范围内。queries "
                "提供 1-4 条面向同一信息需求的自然语言查询，第一条为主查询；仅在有帮助时"
                "增加源语言表达、同义术语或互补子问题。每条 query 独立召回、融合，"
                "候选统一评分后按精确来源合并去重；query_matches "
                "记录每条 query 的独立匹配排名。"
                "可选 file_ids 只收窄到指定文件；省略时召回会话内全部已获准语料。"
                "不扫描目录、不解析或索引文件，也不自动调用视觉模型。返回稳定的文件、"
                "文档版本及 chunk_id，可直接交给 read_file_chunks 精读。"
                "result_limit 控制合并后的总块数，默认 96，可设 1-96，"
                "另受 context_token_limit 正文估算预算保护（默认/最大 96000）。"
                "旧 limit 与 per_query_limit 参数已退役。"
            ),
            input_schema=_input_schema(), output_schema=_output_schema(),
            catalog_tags=("document", "artifact", "file", "search", "read"),
        ),
        implementation_version=RETRIEVE_FILES_IMPLEMENTATION_VERSION,
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL, source_id="personagraph.tools.retrieval.files",
            fingerprint=hashlib.sha256(b"file-retrieval-v8:7").hexdigest(),
        ),
        handler=handler,
        effect_profile=ToolEffectProfile(tuple(
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM, action=action,
                scope_kind=filesystem_scope_kind, default_scope=effect_scope,
                data_egress=DataEgress.CONTENT, idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ) for action in (EffectAction.READ, EffectAction.SEARCH)
        )),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=90, hard_timeout_s=90, max_output_bytes=MAX_FILE_RETRIEVAL_OUTPUT_BYTES,
            max_transparent_retries=1, concurrency_class="file_retrieval_read",
            cancellation_mode=CancellationMode.COOPERATIVE,
        ),
    )


def _identifier(*, nullable=False):
    return {"type": ["string", "null"] if nullable else "string", "minLength": 1, "maxLength": 256}


def _object(properties, *, required=None):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties) if required is None else required, "properties": properties}


def _queries():
    return {"type": "array", "minItems": 1, "maxItems": MAX_FILE_RETRIEVAL_QUERIES,
            "uniqueItems": True, "items": {"type": "string", "minLength": 1,
            "maxLength": MAX_QUERY_CHARS, "description": f"最多 {MAX_RETRIEVAL_QUERY_TOKENS} retrieval tokens。"}}


def _input_schema():
    return _object({
        "queries": _queries(),
        "file_ids": {"type": "array", "minItems": 1, "maxItems": MAX_FILES_PER_RETRIEVE,
                     "uniqueItems": True, "items": _identifier()},
        "result_limit": {"type": "integer", "minimum": 1, "maximum": MAX_FILE_RETRIEVAL_ITEMS,
                         "default": DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT},
        "context_token_limit": {"type": "integer", "minimum": 1,
                                "maximum": MAX_FILE_RETRIEVAL_TOKEN_LIMIT,
                                "default": DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT},
    }, required=["queries"])


def _source_schema():
    return _object({
        "file_name": {"type": ["string", "null"], "maxLength": 512},
        "relative_path": {"type": ["string", "null"], "maxLength": 2000},
        "origin": {"type": ["string", "null"], "enum": ["workspace", "user_upload", "mounted_document", "agent_output", None]},
        "source_modified_at": {"type": ["string", "null"], "maxLength": 64},
        "corpus_recorded_at": {"type": ["string", "null"], "maxLength": 64},
    })


def _evidence_schema(kind):
    properties = {
        "evidence_type": {"const": kind}, "file_id": _identifier(),
        "file_version_id": _identifier(),
        "locator": {"type": ["string", "null"], "maxLength": 512},
        "snippet": {"type": "string", "minLength": 1, "maxLength": MAX_CHUNK_CONTENT_CHARS},
        "content_sha256": {"type": "string", "pattern": "[0-9a-f]{64}"},
        "rank": {"type": "integer", "minimum": 1},
        "query_index": {"type": "integer", "minimum": 0, "maximum": MAX_FILE_RETRIEVAL_QUERIES - 1},
        "query_matches": {"type": "array", "minItems": 1, "maxItems": MAX_FILE_RETRIEVAL_QUERIES,
                          "items": _object({
                              "query_index": {"type": "integer", "minimum": 0, "maximum": MAX_FILE_RETRIEVAL_QUERIES - 1},
                              "rank": {"type": "integer", "minimum": 1, "maximum": FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY},
                              "fused_rank": {"type": "integer", "minimum": 1},
                              "fusion_score": {"type": ["number", "null"]},
                              "reranker_score": {"type": ["number", "null"]},
                          })},
        "snippet_truncated": {"type": "boolean"},
        "content_character_count": {"type": "integer", "minimum": 1},
        "source": _source_schema(),
    }
    if kind == "document_chunk":
        properties.update({"document_id": _identifier(), "document_version_id": _identifier(),
                           "chunk_id": _identifier(), "chunk_sequence": {"type": "integer", "minimum": 0}})
    else:
        properties.update({name: _identifier() for name in ("picture_id", "picture_unit_id", "observation_id")})
    return _object(properties)


def _output_schema():
    return _object({
        "contract_version": {"const": RETRIEVE_FILES_CONTRACT_VERSION},
        "retrieval_scope": {"enum": ["session_corpus", "selected_files"]},
        "status": {"enum": ["complete", "partial", "blocked"]},
        "outcome": {"enum": ["matched", "no_match", "not_established"]},
        "queries": _queries(),
        "evidence": {"type": "array", "maxItems": MAX_FILE_RETRIEVAL_ITEMS,
                     "items": {"oneOf": [_evidence_schema("document_chunk"), _evidence_schema("picture_observation")]}},
        "gaps": {"type": "array", "maxItems": MAX_FILES_PER_RETRIEVE, "items": _object({
            "code": {"type": "string", "pattern": "[a-z][a-z0-9_]{0,95}"},
            "blocking": {"type": "boolean"}, "file_id": _identifier(nullable=True),
            "known_count": {"type": ["integer", "null"], "minimum": 0},
        })},
        "coverage": _object({key: {"type": "integer", "minimum": 0} for key in (
            "requested_file_count", "document_ready_file_count", "returned_evidence_count",
            "document_chunk_count", "picture_observation_count",
        )}),
        "truncated": {"type": "boolean"},
    })


__all__ = ["RETRIEVE_FILES_TOOL_ID", "RETRIEVE_FILES_CONTRACT_VERSION",
           "RETRIEVE_FILES_IMPLEMENTATION_VERSION", "build_retrieve_files_registration"]
