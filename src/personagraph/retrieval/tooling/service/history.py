"""冻结 Session/长期记忆范围上的 Retrieval Tool 查询。"""

from __future__ import annotations

from ...contracts import ContextStatus, SourceAvailability, SourceType
from ...query_guard import QueryGuardError
from ..contracts import (
    RetrievalStatus,
    RetrievalToolGap,
    RetrievalToolRequest,
    RetrievalToolResult,
    parse_current_session_cutoff,
)
from .audit import _QueryAudit, _safe_retrieval_exception_code
from .ports import RetrievalFoundationReadPort
from .projection import (
    _active_generation,
    _blocked,
    _bounded_gaps,
    _budget,
    _context_gaps,
    _generation_stale,
    _history_item_is_authorized,
    _history_request,
    _history_scope_is_stale,
    _history_source_types,
    _port_evidence,
    _result,
    _scope_stale,
)


def retrieve_history(
    request: RetrievalToolRequest,
    *,
    foundation: RetrievalFoundationReadPort | None,
    audit: "_QueryAudit | None" = None,
) -> RetrievalToolResult:
    if foundation is None:
        return _blocked(request, "history_retrieval_not_configured")
    generation = _active_generation(foundation.data_version_provider)
    if generation != request.retrieval_data_version:
        return _generation_stale(request, generation)
    try:
        context = foundation.service.retrieve_context_unrecorded(
            _history_request(request),
            _budget(request),
        )
    except QueryGuardError:
        return _blocked(request, "retrieval_query_rejected")
    except Exception as exc:
        if audit is not None:
            audit.diagnostics.append(
                {
                    "stage": "service",
                    "code": _safe_retrieval_exception_code(exc),
                    "status": "failed",
                    "blocking": True,
                    "known_count": 1,
                }
            )
        return _blocked(request, "retrieval_backend_unavailable")
    if audit is not None:
        audit.contexts.append(context)
    if context.retrieval_data_version != request.retrieval_data_version:
        return _generation_stale(request, context.retrieval_data_version)
    generation_after = _active_generation(foundation.data_version_provider)
    if generation_after != request.retrieval_data_version:
        return _generation_stale(request, generation_after)
    if _history_scope_is_stale(request, context):
        return _scope_stale(request, "history_scope_snapshot_stale")

    allowed_sources = _history_source_types(request.history_scopes)
    cutoff = (
        parse_current_session_cutoff(request.current_session_cutoff)
        if request.current_session_cutoff is not None
        else None
    )
    for item in context.items:
        if item.ref.source_type not in allowed_sources:
            return _blocked(request, "retrieval_authority_mismatch")
        if not _history_item_is_authorized(
            request,
            item,
            cutoff=cutoff,
        ):
            if item.ref.source_type is SourceType.CURRENT_SESSION:
                return _result(
                    request,
                    status=RetrievalStatus.PARTIAL,
                    gaps=(
                        RetrievalToolGap(
                            code="current_session_cutoff_unproven",
                            blocking=True,
                            source_type=SourceType.CURRENT_SESSION,
                        ),
                    ),
                )
            return _blocked(request, "retrieval_authority_mismatch")

    gaps = _context_gaps(
        context,
        requested_sources=tuple(allowed_sources),
    )
    authority_blocked = any(
        context.source_outcomes[source_type].availability
        in {
            SourceAvailability.BLOCKED,
            SourceAvailability.UNAVAILABLE,
            SourceAvailability.NOT_IMPLEMENTED,
        }
        for source_type in allowed_sources
    )
    evidence = tuple(
        _port_evidence(item, rank=rank)
        for rank, item in enumerate(context.items[: request.limit], start=1)
    )
    adapter_truncated = len(context.items) > request.limit
    if adapter_truncated:
        gaps.append(
            RetrievalToolGap(
                code="port_item_limit_exceeded",
                blocking=False,
                known_count=len(context.items) - request.limit,
            )
        )
    gaps = _bounded_gaps(gaps)
    truncated = bool(context.truncated or adapter_truncated)
    status = (
        RetrievalStatus.BLOCKED
        if context.status is ContextStatus.BLOCKED or authority_blocked
        else (
            RetrievalStatus.PARTIAL
            if context.status is ContextStatus.PARTIAL or gaps or truncated
            else RetrievalStatus.COMPLETE
        )
    )
    if status is RetrievalStatus.BLOCKED:
        evidence = ()
    return _result(
        request,
        status=status,
        evidence=evidence,
        gaps=gaps,
        truncated=truncated,
        encoder_fingerprint=context.encoder_fingerprint or None,
        reranker_fingerprint=context.reranker_fingerprint or None,
    )


__all__ = ["retrieve_history"]
