"""在冻结挂载文档上供模型调用、基于游标的读取。

Host 原语可以只返回一个有界预览；此能力提供独立的迭代补充：
模型看到不透明别名，可检查权威分块数量、搜索完整冻结 generation，并通过精确分块
窗口继续读取。私有路径与文档 ID 保留在注册闭包内。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
    FrozenMountedDocumentChunk,
    FrozenMountedDocumentReadError,
    FrozenMountedDocumentReadFailure,
    FrozenMountedDocumentReader,
    FrozenMountedDocumentWindow,
)
from personagraph.tools.policy import AuthorityFacts, ScopeGrant
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration


MOUNTED_DOCUMENT_COGNITION_CAPABILITY = "mounted_document_cognition"
MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION = "mounted-document-cognition-v1"
MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION = "1"
MOUNTED_DOCUMENT_COGNITION_TOOL_IDS = (
    "inspect_mounted_document",
    "search_mounted_document",
    "read_mounted_document_chunks",
)
MOUNTED_DOCUMENT_COGNITION_SOURCE_ID = (
    "personagraph.tools.documents.mounted-document-cognition"
)
MOUNTED_DOCUMENT_COGNITION_SOURCE_DISPLAY_NAME = (
    "Task-authorized mounted documents"
)
MOUNTED_DOCUMENT_COGNITION_SOURCE_FINGERPRINT_SCHEMA = (
    "mounted-document-cognition-source-v1"
)

_MAX_READ_CHUNKS = 64
_SEARCH_BATCH_CHUNKS = 256
_MAX_SEARCH_RESULTS = 20
_MAX_QUERY_CHARACTERS = 2_000
_MAX_SNIPPET_CHARACTERS = 700
_MAX_READ_RESULT_JSON_BYTES = 110_000
_WORD = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentToolScope:
    """组合工具所需的完整、lane-neutral 冻结输入。"""

    session_id: str
    scope_snapshot_sha256: str
    documents: tuple[FrozenMountedDocument, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or len(self.session_id) > 200
        ):
            raise ValueError("session_id must be a bounded durable identity")
        if (
            not isinstance(self.scope_snapshot_sha256, str)
            or len(self.scope_snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.scope_snapshot_sha256
            )
        ):
            raise ValueError(
                "scope_snapshot_sha256 must be a canonical SHA-256 digest"
            )
        if not isinstance(self.documents, tuple) or any(
            not isinstance(document, FrozenMountedDocument)
            for document in self.documents
        ):
            raise TypeError("documents must be a tuple of FrozenMountedDocument")
        aliases = tuple(document.resource_alias for document in self.documents)
        if len(aliases) != len(set(aliases)):
            raise ValueError("mounted document aliases must be unique")
        if any(
            document.session_id != self.session_id
            for document in self.documents
        ):
            raise ValueError("mounted documents must belong to their session")
        if any(
            not document.has_exact_snapshot_binding
            for document in self.documents
        ):
            raise ValueError(
                "mounted cognition requires exact document snapshot bindings"
            )


@dataclass(frozen=True, slots=True)
class MountedDocumentCognitionToolSource:
    """一个挂载文档作用域的不可变工具来源。"""

    session_id: str
    catalog_snapshot: CatalogSnapshot
    scope_snapshot_sha256: str
    tool_ids: tuple[str, ...]
    authority: AuthorityFacts


class _MountedDocumentCognitionRuntime:
    def __init__(
        self,
        scope: FrozenMountedDocumentToolScope,
    ) -> None:
        self.reader = FrozenMountedDocumentReader()
        self.documents_by_alias = {
            document.resource_alias: document for document in scope.documents
        }

    def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        document = self._document(payload)
        snapshot = self._snapshot(document, start_sequence=0, maximum_chunks=1)
        return {
            "document_alias": document.resource_alias,
            "resource_version": snapshot.document_version_id,
            "content_sha256": snapshot.source_sha256,
            "resource_format": document.resource_format,
            "processing_coverage": snapshot.processing_status,
            "processing_diagnostic_codes": list(
                snapshot.processing_diagnostic_codes
            ),
            "total_chunk_count": snapshot.total_chunk_count,
        }

    def read(self, payload: dict[str, Any]) -> dict[str, Any]:
        document = self._document(payload)
        start = int(payload.get("start_sequence", 0))
        limit = int(payload.get("limit", 16))
        snapshot = self._snapshot(
            document,
            start_sequence=start,
            maximum_chunks=limit,
        )
        if start > snapshot.total_chunk_count:
            raise ToolBusinessFailure(
                "mounted_document_range_invalid",
                "The requested start sequence is beyond the frozen document.",
            )

        selected: list[dict[str, Any]] = []
        for chunk in snapshot.chunks:
            selected.append(_public_chunk(document, chunk))
            candidate = _read_result(
                document=document,
                snapshot=snapshot,
                start=start,
                chunks=selected,
            )
            if _json_utf8_bytes(candidate) <= _MAX_READ_RESULT_JSON_BYTES:
                continue
            selected.pop()
            if not selected:
                raise ToolBusinessFailure(
                    "mounted_document_chunk_too_large",
                    "One frozen document chunk exceeds the safe model projection.",
                )
            break

        result = _read_result(
            document=document,
            snapshot=snapshot,
            start=start,
            chunks=selected,
        )
        if _json_utf8_bytes(result) > _MAX_READ_RESULT_JSON_BYTES:
            raise ToolBusinessFailure(
                "mounted_document_projection_too_large",
                "The frozen document metadata exceeds the safe model projection.",
            )
        return result

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        document = self._document(payload)
        query = str(payload.get("query") or "").strip()
        limit = int(payload.get("limit", 8))
        best_matches: list[tuple[int, int, dict[str, Any]]] = []
        total_match_count = 0
        scanned = 0
        try:
            windows = self.reader.iter_windows(
                document,
                maximum_chunks=_SEARCH_BATCH_CHUNKS,
            )
            for snapshot in windows:
                for chunk in snapshot.chunks:
                    score = _match_score(query, chunk.content)
                    if score <= 0:
                        continue
                    total_match_count += 1
                    candidate = {
                        "sequence": chunk.sequence,
                        "locator": _public_locator(document, chunk),
                        "source_pages": list(chunk.source_pages),
                        "snippet": _centered_snippet(chunk.content, query),
                        "content_sha256": chunk.content_sha256,
                        "score": score,
                    }
                    ranked_entry = (score, -chunk.sequence, candidate)
                    if len(best_matches) < limit:
                        heapq.heappush(best_matches, ranked_entry)
                    elif ranked_entry[:2] > best_matches[0][:2]:
                        heapq.heapreplace(best_matches, ranked_entry)
                scanned += len(snapshot.chunks)
        except FrozenMountedDocumentReadError as exc:
            raise _tool_read_failure(exc) from exc

        total = document.total_chunk_count or 0
        ranked = sorted(
            (entry[2] for entry in best_matches),
            key=lambda item: (-int(item["score"]), int(item["sequence"])),
        )
        return {
            "document_alias": document.resource_alias,
            "resource_version": document.document_version_id,
            "content_sha256": document.source_sha256,
            "processing_coverage": document.processing_status,
            "query": query,
            "scanned_chunk_count": scanned,
            "total_chunk_count": total,
            "search_complete": scanned == total,
            "total_match_count": total_match_count,
            "matches_truncated": total_match_count > limit,
            "matches": ranked,
        }

    def _document(
        self,
        payload: Mapping[str, Any],
    ) -> FrozenMountedDocument:
        alias = str(payload.get("document_alias") or "")
        document = self.documents_by_alias.get(alias)
        if document is None:
            raise ToolBusinessFailure(
                "mounted_document_alias_unavailable",
                "The requested mounted document alias is unavailable.",
            )
        return document

    def _snapshot(
        self,
        document: FrozenMountedDocument,
        *,
        start_sequence: int,
        maximum_chunks: int,
    ) -> FrozenMountedDocumentWindow:
        try:
            return self.reader.read_window(
                document,
                start_sequence=start_sequence,
                maximum_chunks=maximum_chunks,
            )
        except FrozenMountedDocumentReadError as exc:
            raise _tool_read_failure(exc) from exc


def build_mounted_document_cognition_tool_specs() -> tuple[
    ToolSpec,
    ToolSpec,
    ToolSpec,
]:
    """Build the three process-stable model contracts in exposure order."""

    return (
        ToolSpec(
            tool_id=MOUNTED_DOCUMENT_COGNITION_TOOL_IDS[0],
            contract_version=MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION,
            name="Inspect a mounted document",
            description=(
                "Return the authoritative total chunk count, frozen version, "
                "format and processing coverage for one opaque mounted alias."
            ),
            input_schema=_alias_input_schema(),
            output_schema=_inspect_output_schema(),
            catalog_tags=("document", "read"),
        ),
        ToolSpec(
            tool_id=MOUNTED_DOCUMENT_COGNITION_TOOL_IDS[1],
            contract_version=MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION,
            name="Search a complete mounted document",
            description=(
                "Search every text chunk in one frozen mounted document. "
                "The result reports exact scan coverage and bounded matches."
            ),
            input_schema=_search_input_schema(),
            output_schema=_search_output_schema(),
            catalog_tags=("document", "search", "read"),
        ),
        ToolSpec(
            tool_id=MOUNTED_DOCUMENT_COGNITION_TOOL_IDS[2],
            contract_version=MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION,
            name="Read mounted document chunks",
            description=(
                "Read one exact contiguous chunk window by start_sequence. "
                "Continue with next_start_sequence until complete is true."
            ),
            input_schema=_read_input_schema(),
            output_schema=_read_output_schema(),
            catalog_tags=("document", "read"),
        ),
    )


def build_mounted_document_cognition_execution_profile() -> ToolExecutionProfile:
    """Build the stable execution contract shared by all three readers."""

    return ToolExecutionProfile(
        default_timeout_s=30.0,
        hard_timeout_s=60.0,
        max_output_bytes=120_000,
        max_transparent_retries=0,
        concurrency_class="mounted_document_readonly",
    )


def build_mounted_document_cognition_effect_profile(
    *,
    action: EffectAction,
    default_scope: str,
) -> ToolEffectProfile:
    """Build the stable read/search effect shape for one effective scope."""

    if action not in {EffectAction.READ, EffectAction.SEARCH}:
        raise ValueError("mounted document action must be read or search")
    return _session_effect(action=action, session_id=default_scope)


def derive_mounted_document_cognition_source_fingerprint(
    scope_snapshot_sha256: str,
) -> str:
    """Bind the live handler closure to one exact mounted-document scope."""

    if (
        not isinstance(scope_snapshot_sha256, str)
        or len(scope_snapshot_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in scope_snapshot_sha256
        )
    ):
        raise ValueError(
            "scope_snapshot_sha256 must be a canonical SHA-256 digest"
        )
    return _sha256_value(
        {
            "schema_version": (
                MOUNTED_DOCUMENT_COGNITION_SOURCE_FINGERPRINT_SCHEMA
            ),
            "authority_scope_snapshot_sha256": scope_snapshot_sha256,
        }
    )


def build_mounted_document_cognition_tool_source(
    scope: FrozenMountedDocumentToolScope,
) -> MountedDocumentCognitionToolSource | None:
    """为一个精确挂载作用域组合三个迭代工具。"""

    if not isinstance(scope, FrozenMountedDocumentToolScope):
        raise TypeError("scope must be FrozenMountedDocumentToolScope")
    if not scope.documents:
        return None
    runtime = _MountedDocumentCognitionRuntime(scope)
    source_fingerprint = derive_mounted_document_cognition_source_fingerprint(
        scope.scope_snapshot_sha256
    )
    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id=MOUNTED_DOCUMENT_COGNITION_SOURCE_ID,
        fingerprint=source_fingerprint,
        display_name=MOUNTED_DOCUMENT_COGNITION_SOURCE_DISPLAY_NAME,
    )
    execution = build_mounted_document_cognition_execution_profile()
    specs = build_mounted_document_cognition_tool_specs()
    registrations = (
        ToolRegistration(
            spec=specs[0],
            implementation_version=MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
            source=source,
            handler=runtime.inspect,
            effect_profile=build_mounted_document_cognition_effect_profile(
                action=EffectAction.READ,
                default_scope=scope.session_id,
            ),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs[1],
            implementation_version=MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
            source=source,
            handler=runtime.search,
            effect_profile=build_mounted_document_cognition_effect_profile(
                action=EffectAction.SEARCH,
                default_scope=scope.session_id,
            ),
            execution_profile=execution,
        ),
        ToolRegistration(
            spec=specs[2],
            implementation_version=MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
            source=source,
            handler=runtime.read,
            effect_profile=build_mounted_document_cognition_effect_profile(
                action=EffectAction.READ,
                default_scope=scope.session_id,
            ),
            execution_profile=execution,
        ),
    )
    catalog = ToolCatalog()
    for registration in registrations:
        catalog.register(registration)
    snapshot = catalog.snapshot()
    scope_snapshot_sha256 = _sha256_value(
        {
            "schema_version": "mounted-document-cognition-runtime-v1",
            "session_id": scope.session_id,
            "authority_scope_snapshot_sha256": scope.scope_snapshot_sha256,
            "catalog": snapshot.to_descriptor(),
        }
    )
    authority_facts = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.SESSION,
                scope.session_id,
            ),
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.SEARCH,
                EffectScopeKind.SESSION,
                scope.session_id,
            ),
        )
    )
    return MountedDocumentCognitionToolSource(
        session_id=scope.session_id,
        catalog_snapshot=snapshot,
        scope_snapshot_sha256=scope_snapshot_sha256,
        tool_ids=MOUNTED_DOCUMENT_COGNITION_TOOL_IDS,
        authority=authority_facts,
    )


def _tool_read_failure(
    error: FrozenMountedDocumentReadError,
) -> ToolBusinessFailure:
    if error.failure is FrozenMountedDocumentReadFailure.UNAVAILABLE:
        return ToolBusinessFailure(
            "mounted_document_unavailable",
            "The frozen mounted document is no longer available.",
        )
    if error.failure is FrozenMountedDocumentReadFailure.STALE:
        return ToolBusinessFailure(
            "mounted_document_authority_drift",
            "The frozen mounted document generation has changed.",
        )
    return ToolBusinessFailure(
        "mounted_document_read_failed",
        "The frozen mounted document could not be read safely.",
    )


def _public_chunk(
    document: FrozenMountedDocument,
    chunk: FrozenMountedDocumentChunk,
) -> dict[str, Any]:
    return {
        "sequence": chunk.sequence,
        "locator": _public_locator(document, chunk),
        "source_pages": list(chunk.source_pages),
        "content": chunk.content,
        "content_sha256": chunk.content_sha256,
    }


def _read_result(
    *,
    document: FrozenMountedDocument,
    snapshot: FrozenMountedDocumentWindow,
    start: int,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    end = start + len(chunks)
    next_start = end if end < snapshot.total_chunk_count else None
    return {
        "document_alias": document.resource_alias,
        "resource_version": snapshot.document_version_id,
        "content_sha256": snapshot.source_sha256,
        "processing_coverage": snapshot.processing_status,
        "processing_diagnostic_codes": list(
            snapshot.processing_diagnostic_codes
        ),
        "total_chunk_count": snapshot.total_chunk_count,
        "returned_start_sequence": start,
        "returned_end_sequence_exclusive": end,
        "next_start_sequence": next_start,
        "complete": next_start is None,
        "chunks": list(chunks),
    }


def _public_locator(
    document: FrozenMountedDocument,
    chunk: FrozenMountedDocumentChunk,
) -> str:
    prefix = f"resource:{document.resource_alias}"
    if len(chunk.source_pages) == 1:
        return f"{prefix}#page={chunk.source_pages[0]}&chunk={chunk.sequence}"
    if chunk.source_pages:
        pages = ",".join(str(page) for page in chunk.source_pages)
        return f"{prefix}#pages={pages}&chunk={chunk.sequence}"
    return f"{prefix}#chunk={chunk.sequence}"


def _match_score(query: str, content: str) -> int:
    normalized_query = _WHITESPACE.sub(" ", query.casefold()).strip()
    normalized_content = _WHITESPACE.sub(" ", content.casefold()).strip()
    if not normalized_query or not normalized_content:
        return 0
    terms = _query_terms(normalized_query)
    phrase_hits = normalized_content.count(normalized_query)
    term_hits = sum(normalized_content.count(term) for term in terms)
    matched_terms = sum(term in normalized_content for term in terms)
    return phrase_hits * 100 + matched_terms * 10 + term_hits


def _query_terms(normalized_query: str) -> tuple[str, ...]:
    terms = set(_WORD.findall(normalized_query))
    for run in _CJK_RUN.findall(normalized_query):
        if len(run) == 1:
            terms.add(run)
            continue
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            terms.update(
                run[index : index + 3] for index in range(len(run) - 2)
            )
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


def _centered_snippet(content: str, query: str) -> str:
    normalized_content = _WHITESPACE.sub(" ", content).strip()
    lowered = normalized_content.casefold()
    normalized_query = _WHITESPACE.sub(" ", query.casefold()).strip()
    needles = (
        normalized_query,
        *_query_terms(normalized_query),
    )
    positions = [lowered.find(item) for item in needles if item]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    half = _MAX_SNIPPET_CHARACTERS // 2
    start = max(0, center - half)
    end = min(len(normalized_content), start + _MAX_SNIPPET_CHARACTERS)
    start = max(0, end - _MAX_SNIPPET_CHARACTERS)
    return normalized_content[start:end]


def _alias_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["document_alias"],
        "properties": {
            "document_alias": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
            }
        },
    }


def _search_input_schema() -> dict[str, Any]:
    schema = _alias_input_schema()
    schema["required"] = ["document_alias", "query"]
    schema["properties"].update(
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_QUERY_CHARACTERS,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_SEARCH_RESULTS,
                "default": 8,
            },
        }
    )
    return schema


def _read_input_schema() -> dict[str, Any]:
    schema = _alias_input_schema()
    schema["properties"].update(
        {
            "start_sequence": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_READ_CHUNKS,
                "default": 16,
            },
        }
    )
    return schema


def _identity_properties() -> dict[str, Any]:
    return {
        "document_alias": {"type": "string"},
        "resource_version": {"type": "string"},
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "processing_coverage": {"enum": ["complete", "partial"]},
    }


def _inspect_output_schema() -> dict[str, Any]:
    properties = _identity_properties()
    properties.update(
        {
            "resource_format": {"type": "string"},
            "processing_diagnostic_codes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "total_chunk_count": {"type": "integer", "minimum": 0},
        }
    )
    return _object_schema(properties)


def _chunk_output_schema(*, include_content: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "sequence": {"type": "integer", "minimum": 0},
        "locator": {"type": "string"},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    properties["content" if include_content else "snippet"] = {
        "type": "string"
    }
    if not include_content:
        properties["score"] = {"type": "integer", "minimum": 1}
    return _object_schema(properties)


def _read_output_schema() -> dict[str, Any]:
    properties = _identity_properties()
    properties.update(
        {
            "processing_diagnostic_codes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "total_chunk_count": {"type": "integer", "minimum": 0},
            "returned_start_sequence": {"type": "integer", "minimum": 0},
            "returned_end_sequence_exclusive": {
                "type": "integer",
                "minimum": 0,
            },
            "next_start_sequence": {
                "type": ["integer", "null"],
                "minimum": 0,
            },
            "complete": {"type": "boolean"},
            "chunks": {
                "type": "array",
                "items": _chunk_output_schema(include_content=True),
            },
        }
    )
    return _object_schema(properties)


def _search_output_schema() -> dict[str, Any]:
    properties = _identity_properties()
    properties.update(
        {
            "query": {"type": "string"},
            "scanned_chunk_count": {"type": "integer", "minimum": 0},
            "total_chunk_count": {"type": "integer", "minimum": 0},
            "search_complete": {"type": "boolean"},
            "total_match_count": {"type": "integer", "minimum": 0},
            "matches_truncated": {"type": "boolean"},
            "matches": {
                "type": "array",
                "items": _chunk_output_schema(include_content=False),
            },
        }
    )
    return _object_schema(properties)


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _session_effect(
    *,
    action: EffectAction,
    session_id: str,
) -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=action,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=session_id,
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_utf8_bytes(value: object) -> int:
    return len(_canonical_json_bytes(value))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "MOUNTED_DOCUMENT_COGNITION_CAPABILITY",
    "MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION",
    "MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION",
    "MOUNTED_DOCUMENT_COGNITION_SOURCE_DISPLAY_NAME",
    "MOUNTED_DOCUMENT_COGNITION_SOURCE_FINGERPRINT_SCHEMA",
    "MOUNTED_DOCUMENT_COGNITION_SOURCE_ID",
    "MOUNTED_DOCUMENT_COGNITION_TOOL_IDS",
    "FrozenMountedDocumentToolScope",
    "MountedDocumentCognitionToolSource",
    "build_mounted_document_cognition_effect_profile",
    "build_mounted_document_cognition_execution_profile",
    "build_mounted_document_cognition_tool_specs",
    "build_mounted_document_cognition_tool_source",
    "derive_mounted_document_cognition_source_fingerprint",
]
