"""Retrieval Tool service 的请求、证据、缺口与状态投影。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from ..contracts import FileRetrievalReadinessStatus
from ...contracts import (
    FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    ContextPackOmissionReason,
    ContextStatus,
    ContextVerificationDropReason,
    CorpusKey,
    QueryProposal,
    RetrievalBudget,
    RetrievedContext,
    RetrievedItem,
    SourceAvailability,
    SourceDependency,
    SourceFilter,
    SourceRetrievalStatus,
    SourceType,
    TrustedRetrievalBoundary,
    RetrievalRequest,
)
from ...ports import RetrievalCancelled, RetrievalDataVersionProvider
from ...execution import checkpoint, measure
from ...service import RetrievalService
from ...orchestration.file_query_batch import pack_query_items
from ...sources.document_policy import (
    DOCUMENT_EVIDENCE_SCOPE_KEY,
    DOCUMENT_USER_EVIDENCE_SCOPE,
)
from ..contracts import (
    RetrievalCorpus,
    FrozenFileRetrievalBinding,
    FrozenFileVersionBinding,
    HistoryScope,
    RetrievalStatus,
    RetrievalToolEvidence,
    RetrievalToolGap,
    RetrievalToolRequest,
    RetrievalToolResult,
    parse_current_session_cutoff,
)
from .ports import RetrievalFoundationReadPort


MAX_HISTORY_CANDIDATES_PER_SOURCE = 80
MAX_PORT_GAPS = 64

_SAFE_GAP_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_GENERATION_UNAVAILABLE = "retrieval_generation_unavailable"


@dataclass(frozen=True, slots=True)
class _BoundRetrievedItem:
    item: RetrievedItem
    authority_id: str | None
    binding_index: int


@dataclass(frozen=True, slots=True)
class _ReadyFileBinding:
    binding: FrozenFileRetrievalBinding
    document_id: str | None = None
    document_version_id: str | None = None


class _FingerprintMismatch:
    pass


_FINGERPRINT_MISMATCH = _FingerprintMismatch()


def _require_foundation(
    foundation: RetrievalFoundationReadPort,
    expected: CorpusKey,
) -> None:
    if getattr(foundation, "corpus_key", None) is not expected:
        raise ValueError(f"foundation must be bound to the {expected.value} corpus")
    if not callable(getattr(getattr(foundation, "service", None), "retrieve_context", None)):
        raise TypeError("foundation service must implement retrieve_context")
    if not callable(
        getattr(
            getattr(foundation, "data_version_provider", None),
            "active_retrieval_data_version_id",
            None,
        )
    ):
        raise TypeError("foundation must expose a RetrievalDataVersion provider")


def _file_retrieve_contexts(service: RetrievalService):
    """文件工具只接统一批量入口，不按文件数回退到逐文件重排。"""

    method_name = "retrieve_file_query_batch"
    method = getattr(service, method_name, None)
    if not callable(method):
        raise TypeError(f"retrieval service must implement {method_name}")
    return method


def _active_generation(provider: RetrievalDataVersionProvider) -> str | None:
    try:
        value = provider.active_retrieval_data_version_id()
    except RetrievalCancelled:
        raise
    except Exception:
        return None
    return value if isinstance(value, str) and value.strip() else None


def _file_request(
    request: RetrievalToolRequest,
    ready: _ReadyFileBinding,
) -> RetrievalRequest:
    binding = ready.binding
    filters: dict[SourceType, SourceFilter] = {}
    if ready.document_id is not None and ready.document_version_id is not None:
        filters[SourceType.DOCUMENT] = SourceFilter.from_mapping(
            SourceType.DOCUMENT,
            {
                "doc_id": ready.document_id,
                "session_id": request.session_id,
            },
        )
    if binding.file_id is not None and binding.file_version_id is not None:
        filters[SourceType.PICTURE] = SourceFilter.from_mapping(
            SourceType.PICTURE,
            {
                "session_id": request.session_id,
                "file_id": binding.file_id,
                "file_version_id": binding.file_version_id,
            },
        )
    if not filters:
        raise ValueError("file retrieval binding has no authorized source filter")
    return RetrievalRequest(
        request_id=f"{request.request_id}:file:{binding.authority_id}",
        model_call_purpose="MODEL_RETRIEVE_FILES_V1",
        query_proposal=QueryProposal(
            request.queries,
            source_hints=frozenset(filters),
        ),
        boundary=TrustedRetrievalBoundary(
            filters,
            {source_type: SourceDependency.OPTIONAL for source_type in filters},
        ),
    )


def _session_document_request(request: RetrievalToolRequest) -> RetrievalRequest:
    """Build the read-only boundary for Session-mounted user evidence."""

    return RetrievalRequest(
        request_id=f"{request.request_id}:file:session-documents",
        model_call_purpose="MODEL_RETRIEVE_FILES_V1",
        query_proposal=QueryProposal(
            request.queries,
            source_hints=frozenset({SourceType.DOCUMENT}),
        ),
        boundary=TrustedRetrievalBoundary(
            {
                SourceType.DOCUMENT: SourceFilter.from_mapping(
                    SourceType.DOCUMENT,
                    {
                        "session_id": request.session_id,
                        DOCUMENT_EVIDENCE_SCOPE_KEY: (
                            DOCUMENT_USER_EVIDENCE_SCOPE
                        ),
                    },
                )
            },
            {SourceType.DOCUMENT: SourceDependency.OPTIONAL},
        ),
    )


def _session_picture_request(
    request: RetrievalToolRequest,
    binding: FrozenFileVersionBinding,
    *,
    ordinal: int,
) -> RetrievalRequest:
    """Build one exact Picture boundary; global Picture search is forbidden."""

    return RetrievalRequest(
        request_id=f"{request.request_id}:file:session-picture:{ordinal}",
        model_call_purpose="MODEL_RETRIEVE_FILES_V1",
        query_proposal=QueryProposal(
            request.queries,
            source_hints=frozenset({SourceType.PICTURE}),
        ),
        boundary=TrustedRetrievalBoundary(
            {
                SourceType.PICTURE: SourceFilter.from_mapping(
                    SourceType.PICTURE,
                    {
                        "session_id": request.session_id,
                        "file_id": binding.file_id,
                        "file_version_id": binding.file_version_id,
                    },
                )
            },
            {SourceType.PICTURE: SourceDependency.OPTIONAL},
        ),
    )


def _history_request(request: RetrievalToolRequest) -> RetrievalRequest:
    filters: dict[SourceType, SourceFilter] = {}
    if HistoryScope.CURRENT_SESSION in request.history_scopes:
        cutoff = parse_current_session_cutoff(request.current_session_cutoff)
        filters[SourceType.CURRENT_SESSION] = SourceFilter.from_mapping(
            SourceType.CURRENT_SESSION,
            {
                "session_id": request.session_id,
                CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY: str(cutoff),
            },
        )
    if HistoryScope.LONG_TERM_USER in request.history_scopes:
        filters[SourceType.LONG_TERM_USER] = SourceFilter.from_mapping(
            SourceType.LONG_TERM_USER,
            {"memory_scope": "user"},
        )
    if HistoryScope.CURRENT_TASK in request.history_scopes:
        if request.long_term_task_id is None:
            raise ValueError("current_task request requires a trusted Task ID")
        filters[SourceType.LONG_TERM_TASK] = SourceFilter.from_mapping(
            SourceType.LONG_TERM_TASK,
            {"task_id": request.long_term_task_id},
        )
    snapshots_by_source = {
        item.source_type: item for item in request.history_source_snapshots
    }
    expected_source_snapshots = {
        source_type: snapshot.expected_access(filters[source_type])
        for source_type, snapshot in snapshots_by_source.items()
    }
    return RetrievalRequest(
        request_id=f"{request.request_id}:history",
        model_call_purpose="MODEL_RETRIEVE_HISTORY_V1",
        query_proposal=QueryProposal(
            (request.query,),
            source_hints=frozenset(filters),
        ),
        boundary=TrustedRetrievalBoundary(
            filters,
            {source_type: SourceDependency.OPTIONAL for source_type in filters},
            expected_source_snapshots,
        ),
    )


def _budget(request: RetrievalToolRequest) -> RetrievalBudget:
    return RetrievalBudget(
        candidate_limit_per_source=min(
            MAX_HISTORY_CANDIDATES_PER_SOURCE,
            max(request.limit * 4, request.limit),
        ) if request.corpus is RetrievalCorpus.HISTORY else FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
        context_token_limit=request.context_token_limit,
        max_items=request.limit,
    )


def _history_source_types(
    scopes: Sequence[HistoryScope],
) -> frozenset[SourceType]:
    mapping = {
        HistoryScope.CURRENT_SESSION: SourceType.CURRENT_SESSION,
        HistoryScope.LONG_TERM_USER: SourceType.LONG_TERM_USER,
        HistoryScope.CURRENT_TASK: SourceType.LONG_TERM_TASK,
    }
    return frozenset(mapping[scope] for scope in scopes)


def _history_scope_is_stale(
    request: RetrievalToolRequest,
    context: RetrievedContext,
) -> bool:
    frozen_sources = {
        item.source_type for item in request.history_source_snapshots
    }
    for source_type in frozen_sources:
        outcome = context.source_outcomes.get(source_type)
        if outcome is None:
            return True
        reason = outcome.reason_code or ""
        if (
            reason == "history_scope_snapshot_stale"
            or reason == f"{source_type.value}_source_changed_during_retrieval"
        ):
            return True
    return False


def _file_item_is_authorized(
    item: RetrievedItem,
    ready: _ReadyFileBinding,
) -> bool:
    binding = ready.binding
    citation = item.citation
    if item.ref.source_type is SourceType.DOCUMENT:
        if ready.document_id is None or ready.document_version_id is None:
            return False
        if citation.get("doc_id") != ready.document_id:
            return False
        citation_revision = citation.get("source_version_id")
        if citation_revision is not None and citation_revision != item.ref.source_revision:
            return False
        if item.ref.source_revision != ready.document_version_id:
            return False
        return not (
            binding.expected_source_revision is not None
            and ready.document_version_id != binding.expected_source_revision
        )
    if item.ref.source_type is SourceType.PICTURE:
        return bool(
            binding.file_id is not None
            and binding.file_version_id is not None
            and citation.get("file_id") == binding.file_id
            and citation.get("file_version_id") == binding.file_version_id
            and isinstance(citation.get("picture_id"), str)
            and isinstance(citation.get("picture_unit_id"), str)
            and isinstance(citation.get("observation_id"), str)
        )
    return False


def _session_document_item_is_authorized(item: RetrievedItem) -> bool:
    """Validate the identity facts projected by an authorized Document read."""

    if item.ref.source_type is not SourceType.DOCUMENT:
        return False
    doc_id = item.citation.get("doc_id")
    citation_revision = item.citation.get("source_version_id")
    origin = item.citation.get("origin")
    return (
        isinstance(doc_id, str)
        and bool(doc_id.strip())
        and isinstance(citation_revision, str)
        and citation_revision == item.ref.source_revision
        and origin in {"workspace", "user_upload"}
    )


def _session_picture_item_is_authorized(
    item: RetrievedItem,
    binding: FrozenFileVersionBinding,
) -> bool:
    """Validate exact Picture authority facts returned by the source adapter."""

    citation = item.citation
    return bool(
        item.ref.source_type is SourceType.PICTURE
        and citation.get("file_id") == binding.file_id
        and citation.get("file_version_id") == binding.file_version_id
        and isinstance(citation.get("picture_id"), str)
        and isinstance(citation.get("picture_unit_id"), str)
        and isinstance(citation.get("observation_id"), str)
    )


def _history_item_is_authorized(
    request: RetrievalToolRequest,
    item: RetrievedItem,
    *,
    cutoff: int | None,
) -> bool:
    citation = item.citation
    if item.ref.source_type is SourceType.CURRENT_SESSION:
        if cutoff is None or citation.get("session_id") != request.session_id:
            return False
        assistant_turn_idx = _nonnegative_int(citation.get("assistant_turn_idx"))
        return assistant_turn_idx is not None and assistant_turn_idx <= cutoff
    if item.ref.source_type is SourceType.LONG_TERM_TASK:
        return (
            request.long_term_task_id is not None
            and citation.get("task_id") == request.long_term_task_id
        )
    return item.ref.source_type is SourceType.LONG_TERM_USER


@measure("port_packing")
def _pack_file_candidates(
    request: RetrievalToolRequest,
    candidates: Sequence[_BoundRetrievedItem],
) -> tuple[tuple[RetrievalToolEvidence, ...], list[RetrievalToolGap], bool]:
    """复用 query 内选择器进行最终端口保护，不再持有全局旧 12/32 排名。"""

    checkpoint()
    if request.corpus is not RetrievalCorpus.FILES:
        raise ValueError("file candidate packing requires the files corpus")
    if any(
        value.item.query_index >= len(request.queries)
        or any(match.query_index >= len(request.queries) for match in value.item.query_matches)
        for value in candidates
    ):
        raise ValueError("retrieved query_index is outside the request query set")
    packed = pack_query_items(
        [value.item for value in candidates],
        result_limit=request.limit,
        token_limit=request.context_token_limit,
    )
    authority_by_ref = {value.item.ref: value.authority_id for value in candidates}
    selected = tuple(
        _port_evidence(item, rank=rank, authority_id=authority_by_ref[item.ref])
        for rank, item in enumerate(packed.items, 1)
    )
    gaps = [
        RetrievalToolGap(
            code=("port_token_limit_exceeded" if omission.reason is ContextPackOmissionReason.TOKEN_LIMIT
                  else "port_item_limit_exceeded"),
            blocking=False, source_type=omission.source_type, known_count=omission.count,
        )
        for omission in packed.omissions
    ]
    return selected, gaps, bool(packed.omissions)
def _port_evidence(
    item: RetrievedItem,
    *,
    rank: int,
    authority_id: str | None = None,
) -> RetrievalToolEvidence:
    return RetrievalToolEvidence(
        source_type=item.ref.source_type,
        source_unit_id=item.ref.source_unit_id,
        source_revision=item.ref.source_revision,
        indexed_content_hash=item.ref.indexed_content_hash,
        content=item.content,
        estimated_tokens=item.estimated_tokens,
        rank=rank,
        query_index=item.query_index,
        query_matches=item.query_matches,
        authority_id=authority_id,
        citation=_safe_citation(item.ref.source_type, item.citation),
    )




def _safe_citation(
    source_type: SourceType,
    citation: Mapping[str, str],
) -> dict[str, str]:
    keys = {
        SourceType.DOCUMENT: (
            "file_id",
            "file_version_id",
            "doc_id",
            "source_version_id",
            "location",
            "page_start",
            "page_end",
            "producer_chunk_id",
            "chunk_id",
            "title",
            "origin",
            "source_path",
            "source_mtime_ns",
            "source_added_at",
        ),
        SourceType.PICTURE: (
            "file_id",
            "file_version_id",
            "picture_id",
            "picture_unit_id",
            "observation_id",
            "source_kind",
            "unit_kind",
            "surface_kind",
            "surface_ordinal",
            "created_at",
        ),
        SourceType.CURRENT_SESSION: (
            "session_id",
            "user_turn_idx",
            "assistant_turn_idx",
        ),
        SourceType.LONG_TERM_USER: ("memory_type",),
        SourceType.LONG_TERM_TASK: ("task_id", "memory_type"),
    }[source_type]
    result: dict[str, str] = {}
    for key in keys:
        value = citation.get(key)
        if isinstance(value, str) and len(value) <= 4_096:
            result[key] = value
    return result


def _context_gaps(
    context: RetrievedContext,
    *,
    requested_sources: Sequence[SourceType],
    authority_id: str | None = None,
) -> list[RetrievalToolGap]:
    requested = frozenset(requested_sources)
    gaps: list[RetrievalToolGap] = []
    for limitation in context.coverage_limitations:
        if limitation.source_type not in requested:
            continue
        outcome = context.source_outcomes[limitation.source_type]
        if outcome.availability is SourceAvailability.EMPTY:
            continue
        gaps.append(
            RetrievalToolGap(
                code=_gap_code(limitation.reason_code, "retrieval_coverage_limited"),
                blocking=outcome.availability in {
                    SourceAvailability.BLOCKED,
                    SourceAvailability.UNAVAILABLE,
                },
                source_type=limitation.source_type,
                authority_id=authority_id,
            )
        )
    for source_type in requested:
        outcome = context.source_outcomes[source_type]
        if (
            outcome.availability is SourceAvailability.READY
            and outcome.retrieval
            in {SourceRetrievalStatus.MATCHED, SourceRetrievalStatus.NO_MATCH}
        ) or (
            outcome.availability is SourceAvailability.EMPTY
            and outcome.retrieval is SourceRetrievalStatus.NOT_RUN
        ):
            continue
        gaps.append(
            RetrievalToolGap(
                code=_gap_code(outcome.reason_code, "source_retrieval_incomplete"),
                blocking=outcome.availability in {
                    SourceAvailability.BLOCKED,
                    SourceAvailability.UNAVAILABLE,
                },
                source_type=source_type,
                authority_id=authority_id,
            )
        )
    for omission in context.pack_omissions:
        if omission.source_type in requested:
            gaps.append(
                RetrievalToolGap(
                    code={
                        ContextPackOmissionReason.MAX_ITEMS: "context_pack_max_items",
                        ContextPackOmissionReason.TOKEN_LIMIT: "context_pack_token_limit",
                        ContextPackOmissionReason.SOURCE_POOL_ITEM_QUOTA: (
                            "context_pack_source_pool_quota"
                        ),
                    }[omission.reason],
                    blocking=False,
                    source_type=omission.source_type,
                    authority_id=authority_id,
                    known_count=omission.count,
                )
            )
    for drop in context.verification_drops:
        if drop.source_type in requested:
            gaps.append(
                RetrievalToolGap(
                    code={
                        ContextVerificationDropReason.SOURCE_FETCH_FAILED: (
                            "verification_source_fetch_failed"
                        ),
                        ContextVerificationDropReason.SOURCE_UNIT_MISSING: (
                            "verification_source_unit_missing"
                        ),
                        ContextVerificationDropReason.SOURCE_UNIT_NOT_RETRIEVABLE: (
                            "verification_source_unit_not_retrievable"
                        ),
                        ContextVerificationDropReason.SOURCE_UNIT_REF_MISMATCH: (
                            "verification_source_unit_ref_mismatch"
                        ),
                    }[drop.reason],
                    blocking=False,
                    source_type=drop.source_type,
                    authority_id=authority_id,
                    known_count=drop.count,
                )
            )
    if context.diagnostic_codes:
        gaps.append(
            RetrievalToolGap(
                code="retrieval_diagnostics_present",
                blocking=False,
                known_count=len(context.diagnostic_codes),
            )
        )
    if context.status is not ContextStatus.COMPLETE and not gaps:
        gaps.append(
            RetrievalToolGap(
                code="retrieval_partial",
                blocking=context.status is ContextStatus.BLOCKED,
            )
        )
    return gaps


def _consistent_fingerprints(
    values: Sequence[tuple[str, str]],
) -> tuple[str | _FingerprintMismatch, str | _FingerprintMismatch]:
    return (
        _one_fingerprint(tuple(value[0] for value in values)),
        _one_fingerprint(tuple(value[1] for value in values)),
    )


def _one_fingerprint(
    values: Sequence[str],
) -> str | _FingerprintMismatch:
    nonempty = {value for value in values if isinstance(value, str) and value}
    if len(nonempty) > 1:
        return _FINGERPRINT_MISMATCH
    return next(iter(nonempty), "")


def _optional_fingerprint(value: str | _FingerprintMismatch) -> str | None:
    return value or None if isinstance(value, str) else None


def _gap_code(value: str | None, fallback: str) -> str:
    generic_status_codes = {
        status.value for status in FileRetrievalReadinessStatus
    }
    if (
        isinstance(value, str)
        and value not in generic_status_codes
        and _SAFE_GAP_CODE.fullmatch(value)
    ):
        return value
    return fallback


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _bounded_gaps(
    gaps: Sequence[RetrievalToolGap],
) -> list[RetrievalToolGap]:
    deduplicated: list[RetrievalToolGap] = []
    seen: set[tuple[object, ...]] = set()
    for gap in gaps:
        key = (
            gap.code,
            gap.blocking,
            gap.source_type,
            gap.authority_id,
            gap.known_count,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(gap)
    if len(deduplicated) <= MAX_PORT_GAPS:
        return deduplicated
    omitted = len(deduplicated) - (MAX_PORT_GAPS - 1)
    return [
        *deduplicated[: MAX_PORT_GAPS - 1],
        RetrievalToolGap(
            code="retrieval_gap_limit_reached",
            blocking=False,
            known_count=omitted,
        ),
    ]


def _result(
    request: RetrievalToolRequest,
    *,
    status: RetrievalStatus,
    evidence: Sequence[RetrievalToolEvidence] = (),
    gaps: Sequence[RetrievalToolGap] = (),
    truncated: bool = False,
    scope_is_current: bool = True,
    retrieval_data_version: str | None = None,
    encoder_fingerprint: str | None = None,
    reranker_fingerprint: str | None = None,
) -> RetrievalToolResult:
    normalized_evidence = tuple(evidence)
    if status is RetrievalStatus.BLOCKED:
        normalized_evidence = ()
    packed_tokens = min(
        request.context_token_limit,
        sum(item.estimated_tokens for item in normalized_evidence),
    )
    return RetrievalToolResult(
        status=status,
        scope_snapshot_id=request.scope_snapshot_id,
        retrieval_data_version=(
            retrieval_data_version
            if isinstance(retrieval_data_version, str) and retrieval_data_version
            else request.retrieval_data_version
        ),
        evidence=normalized_evidence,
        gaps=tuple(_bounded_gaps(gaps)),
        configured_token_limit=request.context_token_limit,
        packed_tokens=packed_tokens,
        truncated=truncated,
        scope_is_current=scope_is_current,
        encoder_fingerprint=encoder_fingerprint,
        reranker_fingerprint=reranker_fingerprint,
    )


def _blocked(
    request: RetrievalToolRequest,
    code: str,
) -> RetrievalToolResult:
    return _result(
        request,
        status=RetrievalStatus.BLOCKED,
        gaps=(RetrievalToolGap(code=code, blocking=True),),
    )


def _generation_stale(
    request: RetrievalToolRequest,
    observed: str | None,
) -> RetrievalToolResult:
    return _result(
        request,
        status=RetrievalStatus.BLOCKED,
        retrieval_data_version=(observed or _GENERATION_UNAVAILABLE),
        gaps=(
            RetrievalToolGap(
                code="retrieval_generation_stale",
                blocking=True,
            ),
        ),
    )


def _scope_stale(
    request: RetrievalToolRequest,
    code: str,
) -> RetrievalToolResult:
    return _result(
        request,
        status=RetrievalStatus.BLOCKED,
        scope_is_current=False,
        gaps=(RetrievalToolGap(code=code, blocking=True),),
    )


__all__ = [
    "_BoundRetrievedItem",
    "_FINGERPRINT_MISMATCH",
    "_ReadyFileBinding",
    "_active_generation",
    "_blocked",
    "_bounded_gaps",
    "_budget",
    "_consistent_fingerprints",
    "_context_gaps",
    "_file_item_is_authorized",
    "_file_request",
    "_file_retrieve_contexts",
    "_gap_code",
    "_generation_stale",
    "_history_item_is_authorized",
    "_history_request",
    "_history_scope_is_stale",
    "_history_source_types",
    "_optional_fingerprint",
    "_pack_file_candidates",
    "_port_evidence",
    "_require_foundation",
    "_result",
    "_scope_stale",
    "_session_document_item_is_authorized",
    "_session_document_request",
    "_session_picture_item_is_authorized",
    "_session_picture_request",
]
