"""冻结文件候选范围上的 Retrieval Tool 查询。"""

from __future__ import annotations

from ..contracts import (
    FileRetrievalReadinessResult,
    FileRetrievalReadinessStatus,
)
from ...contracts import ContextStatus, RetrievedContext, SourceType
from ...execution import checkpoint
from ...ports import RetrievalCancelled
from ...query_guard import QueryGuardError
from ..contracts import (
    FileRetrievalScope,
    RetrievalStatus,
    RetrievalToolGap,
    RetrievalToolRequest,
    RetrievalToolResult,
)
from .audit import _QueryAudit, _safe_retrieval_exception_code
from .ports import FileRetrievalReadinessPort, RetrievalFoundationReadPort
from .projection import (
    _BoundRetrievedItem,
    _FINGERPRINT_MISMATCH,
    _ReadyFileBinding,
    _active_generation,
    _blocked,
    _bounded_gaps,
    _budget,
    _consistent_fingerprints,
    _context_gaps,
    _file_item_is_authorized,
    _file_request,
    _file_retrieve_contexts,
    _gap_code,
    _generation_stale,
    _optional_fingerprint,
    _pack_file_candidates,
    _result,
    _scope_stale,
    _session_document_item_is_authorized,
    _session_document_request,
    _session_picture_item_is_authorized,
    _session_picture_request,
)


def retrieve_file_corpus(
    request: RetrievalToolRequest,
    *,
    foundation: RetrievalFoundationReadPort | None,
    file_readiness: FileRetrievalReadinessPort | None,
    file_readonly: bool,
    max_file_service_calls: int,
    audit: "_QueryAudit | None" = None,
) -> RetrievalToolResult:
    checkpoint()
    if foundation is None:
        return _blocked(request, "file_retrieval_not_configured")
    if request.file_scope is FileRetrievalScope.SESSION_CORPUS:
        return _retrieve_session_document_corpus(
            request,
            foundation=foundation,
            max_file_service_calls=max_file_service_calls,
            audit=audit,
        )
    if file_readiness is None:
        return _blocked(request, "file_readiness_not_configured")

    gaps: list[RetrievalToolGap] = []
    if not request.file_inventory_complete:
        gaps.append(
            RetrievalToolGap(
                code="picture_file_inventory_incomplete",
                blocking=False,
                source_type=SourceType.PICTURE,
            )
        )
    ready: list[_ReadyFileBinding] = []
    blocked_count = 0
    service_bindings = request.file_bindings[: max_file_service_calls]
    omitted_bindings = len(request.file_bindings) - len(service_bindings)
    if omitted_bindings:
        gaps.append(
            RetrievalToolGap(
                code="file_scope_service_limit",
                blocking=False,
                source_type=SourceType.DOCUMENT,
                known_count=omitted_bindings,
            )
        )

    for binding in service_bindings:
        checkpoint()
        try:
            readiness = file_readiness.ensure_ready(request, binding)
        except QueryGuardError as exc:
            return _blocked(request, exc.code.value)
        except RetrievalCancelled:
            raise
        except Exception as exc:
            if audit is not None:
                audit.diagnostics.append(
                    {
                        "stage": "readiness",
                        "code": _safe_retrieval_exception_code(exc),
                        "status": "failed",
                        "blocking": True,
                        "file_id": binding.source_id,
                        "known_count": 1,
                    }
                )
            readiness = FileRetrievalReadinessResult(
                status=FileRetrievalReadinessStatus.BLOCKED,
                file_id=binding.source_id,
                reason_code="file_readiness_unavailable",
            )
        if not isinstance(readiness, FileRetrievalReadinessResult):
            return _blocked(request, "file_readiness_contract_violation")
        if readiness.file_id != binding.source_id:
            return _scope_stale(request, "file_scope_stale")
        if readiness.status is FileRetrievalReadinessStatus.STALE:
            return _scope_stale(request, "file_scope_stale")
        if readiness.status is FileRetrievalReadinessStatus.READY:
            if readiness.retrieval_data_version != request.retrieval_data_version:
                return _generation_stale(
                    request,
                    readiness.retrieval_data_version,
                )
            if not readiness.document_id or not readiness.document_version_id:
                return _blocked(request, "file_readiness_contract_violation")
            if (
                binding.expected_source_revision is not None
                and readiness.document_version_id
                != binding.expected_source_revision
            ):
                return _scope_stale(request, "file_scope_stale")
            ready.append(
                _ReadyFileBinding(
                    binding=binding,
                    document_id=readiness.document_id,
                    document_version_id=readiness.document_version_id,
                )
            )
            continue
        if binding.file_id is not None and binding.file_version_id is not None:
            # Picture is an independent File source.  A non-document image can be
            # searchable even when no Document preparation route exists.
            if readiness.status is FileRetrievalReadinessStatus.PENDING:
                gaps.append(
                    RetrievalToolGap(
                        code="file_indexing_pending",
                        blocking=False,
                        source_type=SourceType.DOCUMENT,
                        authority_id=binding.authority_id,
                        known_count=1,
                    )
                )
            else:
                blocked_count += 1
                gaps.append(
                    RetrievalToolGap(
                        code=_gap_code(
                            readiness.reason_code,
                            "file_retrieval_blocked",
                        ),
                        blocking=True,
                        source_type=SourceType.DOCUMENT,
                        authority_id=binding.authority_id,
                        known_count=1,
                    )
                )
            ready.append(_ReadyFileBinding(binding=binding))
            continue
        if readiness.status is FileRetrievalReadinessStatus.PENDING:
            gaps.append(
                RetrievalToolGap(
                    code="file_indexing_pending",
                    blocking=False,
                    source_type=SourceType.DOCUMENT,
                    authority_id=binding.authority_id,
                    known_count=1,
                )
            )
            continue
        blocked_count += 1
        gaps.append(
            RetrievalToolGap(
                code=_gap_code(
                    readiness.reason_code,
                    "file_retrieval_blocked",
                ),
                blocking=True,
                source_type=SourceType.DOCUMENT,
                authority_id=binding.authority_id,
                known_count=1,
            )
        )

    if not ready:
        status = (
            RetrievalStatus.BLOCKED
            if blocked_count
            else RetrievalStatus.PARTIAL
        )
        return _result(
            request,
            status=status,
            gaps=gaps,
        )

    # Readiness is read-only. Search only the exact active generation established
    # by explicit File preparation; retrieval never publishes a new generation.
    generation = _active_generation(foundation.data_version_provider)
    if generation != request.retrieval_data_version:
        return _generation_stale(request, generation)

    contexts: list[tuple[_ReadyFileBinding, RetrievedContext]] = []
    failed_calls = 0
    try:
        batch_contexts = _file_retrieve_contexts(foundation.service)(
            tuple(_file_request(request, binding) for binding in ready),
            _budget(request), result_limit=request.limit, readonly=file_readonly,
        )
        if len(batch_contexts) != len(ready):
            return _blocked(request, "retrieval_backend_contract_violation")
    except QueryGuardError as exc:
        return _blocked(request, exc.code.value)
    except RetrievalCancelled:
        raise
    except Exception as exc:
        if audit is not None:
            audit.diagnostics.append({"stage": "service", "status": "failed", "blocking": True,
                                      "code": _safe_retrieval_exception_code(exc)})
        return _blocked(request, "retrieval_backend_unavailable")
    for scope_index, ready_binding in enumerate(ready):
        binding = ready_binding.binding
        requested_sources = tuple(
            source_type
            for source_type, available in (
                (
                    SourceType.DOCUMENT,
                    ready_binding.document_id is not None
                    and ready_binding.document_version_id is not None,
                ),
                (
                    SourceType.PICTURE,
                    binding.file_id is not None
                    and binding.file_version_id is not None,
                ),
            )
            if available
        )
        try:
            context = batch_contexts[scope_index]
        except QueryGuardError as exc:
            return _blocked(request, exc.code.value)
        except RetrievalCancelled:
            raise
        except Exception as exc:
            failed_calls += 1
            if audit is not None:
                audit.diagnostics.append(
                    {
                        "stage": "service",
                        "code": _safe_retrieval_exception_code(exc),
                        "status": "failed",
                        "blocking": True,
                        "file_id": binding.source_id,
                        "source_type": (
                            requested_sources[0].value
                            if len(requested_sources) == 1
                            else "mixed_file"
                        ),
                        "known_count": 1,
                    }
                )
            gaps.extend(
                RetrievalToolGap(
                    code="retrieval_backend_unavailable",
                    blocking=True,
                    source_type=source_type,
                    authority_id=binding.authority_id,
                    known_count=1,
                )
                for source_type in requested_sources
            )
            continue
        if audit is not None:
            audit.contexts.append(context)
        if context.retrieval_data_version != request.retrieval_data_version:
            return _generation_stale(
                request,
                context.retrieval_data_version,
            )
        contexts.append((ready_binding, context))
        gaps.extend(
            _context_gaps(
                context,
                requested_sources=requested_sources,
                authority_id=binding.authority_id,
            )
        )

    generation_after = _active_generation(foundation.data_version_provider)
    if generation_after != request.retrieval_data_version:
        return _generation_stale(request, generation_after)
    if not contexts:
        return _result(
            request,
            status=RetrievalStatus.BLOCKED,
            gaps=gaps,
        )

    candidates: list[_BoundRetrievedItem] = []
    fingerprints: list[tuple[str, str]] = []
    any_partial = bool(failed_calls or omitted_bindings or blocked_count)
    any_truncated = False
    for binding_index, (ready_binding, context) in enumerate(contexts):
        binding = ready_binding.binding
        fingerprints.append(
            (context.encoder_fingerprint, context.reranker_fingerprint)
        )
        any_partial = any_partial or context.status is not ContextStatus.COMPLETE
        any_truncated = any_truncated or context.truncated
        if context.status is ContextStatus.BLOCKED:
            continue
        for item in context.items:
            if not _file_item_is_authorized(item, ready_binding):
                return _blocked(request, "retrieval_authority_mismatch")
            candidates.append(
                _BoundRetrievedItem(
                    item=item,
                    authority_id=binding.authority_id,
                    binding_index=binding_index,
                )
            )

    encoder, reranker = _consistent_fingerprints(fingerprints)
    if encoder is _FINGERPRINT_MISMATCH or reranker is _FINGERPRINT_MISMATCH:
        return _blocked(request, "retrieval_backend_contract_violation")
    evidence, pack_gaps, pack_truncated = _pack_file_candidates(
        request,
        candidates,
    )
    _record_query_ranking_diagnostics(audit, candidates)
    gaps.extend(pack_gaps)
    any_truncated = any_truncated or pack_truncated
    gaps = _bounded_gaps(gaps)
    status = (
        RetrievalStatus.PARTIAL
        if any_partial or gaps or any_truncated
        else RetrievalStatus.COMPLETE
    )
    return _result(
        request,
        status=status,
        evidence=evidence,
        gaps=gaps,
        truncated=any_truncated,
        encoder_fingerprint=_optional_fingerprint(encoder),
        reranker_fingerprint=_optional_fingerprint(reranker),
    )


def _retrieve_session_document_corpus(
    request: RetrievalToolRequest,
    *,
    foundation: RetrievalFoundationReadPort,
    max_file_service_calls: int,
    audit: "_QueryAudit | None",
) -> RetrievalToolResult:
    """Search Session Documents plus exact authorized Picture file versions."""

    generation = _active_generation(foundation.data_version_provider)
    if generation != request.retrieval_data_version:
        return _generation_stale(request, generation)

    picture_call_limit = max(0, max_file_service_calls - 1)
    picture_bindings = request.session_file_bindings[:picture_call_limit]
    omitted_picture_bindings = len(request.session_file_bindings) - len(
        picture_bindings
    )
    calls = [
        (
            SourceType.DOCUMENT,
            None,
            _session_document_request(request),
        ),
        *(
            (
                SourceType.PICTURE,
                binding,
                _session_picture_request(request, binding, ordinal=ordinal),
            )
            for ordinal, binding in enumerate(
                picture_bindings,
                start=1,
            )
        ),
    ]
    contexts: list[tuple[SourceType, object, RetrievedContext]] = []
    gaps: list[RetrievalToolGap] = []
    failed_calls = 0
    if not request.file_inventory_complete:
        gaps.append(
            RetrievalToolGap(
                code="picture_file_inventory_incomplete",
                blocking=False,
                source_type=SourceType.PICTURE,
            )
        )
    if omitted_picture_bindings:
        gaps.append(
            RetrievalToolGap(
                code="picture_scope_service_limit",
                blocking=False,
                source_type=SourceType.PICTURE,
                known_count=omitted_picture_bindings,
            )
        )
    try:
        batch_contexts = _file_retrieve_contexts(foundation.service)(
            tuple(retrieval_request for _, _, retrieval_request in calls),
            _budget(request), result_limit=request.limit, readonly=True,
        )
        if len(batch_contexts) != len(calls):
            return _blocked(request, "retrieval_backend_contract_violation")
    except QueryGuardError as exc:
        return _blocked(request, exc.code.value)
    except RetrievalCancelled:
        raise
    except Exception as exc:
        if audit is not None:
            audit.diagnostics.append({"stage": "service", "status": "failed", "blocking": True,
                                      "code": _safe_retrieval_exception_code(exc)})
        return _blocked(request, "retrieval_backend_unavailable")
    for scope_index, (source_type, binding, retrieval_request) in enumerate(calls):
        try:
            context = batch_contexts[scope_index]
        except QueryGuardError as exc:
            return _blocked(request, exc.code.value)
        except RetrievalCancelled:
            raise
        except Exception as exc:
            failed_calls += 1
            if audit is not None:
                audit.diagnostics.append(
                    {
                        "stage": "service",
                        "code": _safe_retrieval_exception_code(exc),
                        "status": "failed",
                        "blocking": True,
                        "source_type": source_type.value,
                        "known_count": 1,
                    }
                )
            gaps.append(
                RetrievalToolGap(
                    code="retrieval_backend_unavailable",
                    blocking=True,
                    source_type=source_type,
                    known_count=1,
                )
            )
            continue
        if audit is not None:
            audit.contexts.append(context)
        if context.retrieval_data_version != request.retrieval_data_version:
            return _generation_stale(request, context.retrieval_data_version)
        contexts.append((source_type, binding, context))
        gaps.extend(_context_gaps(context, requested_sources=(source_type,)))

    generation_after = _active_generation(foundation.data_version_provider)
    if generation_after != request.retrieval_data_version:
        return _generation_stale(request, generation_after)

    if not contexts:
        return _result(
            request,
            status=RetrievalStatus.BLOCKED,
            gaps=gaps,
        )

    candidates: list[_BoundRetrievedItem] = []
    fingerprints: list[tuple[str, str]] = []
    any_partial = bool(
        failed_calls
        or not request.file_inventory_complete
        or omitted_picture_bindings
    )
    any_truncated = False
    for binding_index, (source_type, binding, context) in enumerate(contexts):
        fingerprints.append(
            (context.encoder_fingerprint, context.reranker_fingerprint)
        )
        any_partial = any_partial or context.status is not ContextStatus.COMPLETE
        any_truncated = any_truncated or context.truncated
        if context.status is ContextStatus.BLOCKED:
            continue
        for item in context.items:
            authorized = (
                _session_document_item_is_authorized(item)
                if source_type is SourceType.DOCUMENT
                else _session_picture_item_is_authorized(item, binding)
            )
            if not authorized:
                return _blocked(request, "retrieval_authority_mismatch")
            candidates.append(
                _BoundRetrievedItem(
                    item=item,
                    authority_id=None,
                    binding_index=binding_index,
                )
            )

    encoder, reranker = _consistent_fingerprints(fingerprints)
    if encoder is _FINGERPRINT_MISMATCH or reranker is _FINGERPRINT_MISMATCH:
        return _blocked(request, "retrieval_backend_contract_violation")
    evidence, pack_gaps, pack_truncated = _pack_file_candidates(
        request,
        candidates,
    )
    _record_query_ranking_diagnostics(audit, candidates)
    gaps.extend(pack_gaps)
    gaps = _bounded_gaps(gaps)
    truncated = any_truncated or pack_truncated
    status = (
        RetrievalStatus.PARTIAL
        if any_partial or gaps or truncated
        else RetrievalStatus.COMPLETE
    )
    return _result(
        request,
        status=status,
        evidence=evidence,
        gaps=gaps,
        truncated=truncated,
        encoder_fingerprint=_optional_fingerprint(encoder),
        reranker_fingerprint=_optional_fingerprint(reranker),
    )


def _record_query_ranking_diagnostics(
    audit: "_QueryAudit | None",
    candidates: list[_BoundRetrievedItem],
) -> None:
    if audit is None or not candidates:
        return
    scores_by_query: dict[int, list[float | None]] = {}
    for candidate in candidates:
        item = candidate.item
        if item.query_matches:
            for match in item.query_matches:
                scores_by_query.setdefault(match.query_index, []).append(match.reranker_score)
        else:
            scores_by_query.setdefault(item.query_index, []).append(item.reranker_score)
    for query_index, scores in sorted(scores_by_query.items()):
        used = all(score is not None for score in scores)
        audit.diagnostics.append({
            "stage": "ranking", "query_index": query_index,
            "code": "query_rerank_used" if used else "query_rerank_unavailable_rrf_fallback",
            "status": "used" if used else "degraded", "blocking": False,
            "known_count": len(scores),
        })


__all__ = ["retrieve_file_corpus"]
