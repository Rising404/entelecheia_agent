"""正文无关的 Retrieval query trajectory 审计投影。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
import re

from ...contracts import (
    FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
    FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY,
    MAX_FILE_RETRIEVAL_QUERIES,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievedContext,
    RetrievedItem,
    SourceAvailability,
    SourceRetrievalStatus,
    SourceType,
)
from ...execution import RetrievalExecution
from ..contracts import (
    RetrievalCorpus,
    RetrievalToolRequest,
    RetrievalToolResult,
)


MAX_FILE_QUERY_AUDIT_DIAGNOSTICS = 256
MAX_FILE_QUERY_AUDIT_CANDIDATES = 4 * 3 * 128
MAX_FILE_QUERY_AUDIT_CANDIDATES_PER_LANE = 128
MAX_FILE_QUERY_AUDIT_CANDIDATES_PER_CONTEXT = 4 * 128

_SAFE_AUDIT_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")


@dataclass(slots=True)
class _QueryAudit:
    """只在一次检索查询内存活的无正文审计收集器。"""

    contexts: list[RetrievedContext] = field(default_factory=list)
    diagnostics: list[Mapping[str, object]] = field(default_factory=list)
    execution: RetrievalExecution | None = None


@dataclass(slots=True)
class _DeferredQueryAudit:
    """把唯一 trajectory 写入推迟到最外层公开投影完成。"""

    request: RetrievalToolRequest
    result: RetrievalToolResult
    audit: _QueryAudit
    turn_id: str | None
    duration_ms: int
    _recorded: bool = False

    def record(
        self,
        *,
        final_projection: Mapping[str, object] | None = None,
        diagnostics: Sequence[Mapping[str, object]] = (),
    ) -> None:
        if self._recorded:
            return
        self._recorded = True
        try:
            _record_query_audit(
                request=self.request,
                result=self.result,
                audit=self.audit,
                turn_id=self.turn_id,
                duration_ms=self.duration_ms,
                final_projection=final_projection,
                diagnostics=diagnostics,
            )
        except Exception:
            return


def _record_query_audit(
    *,
    request: RetrievalToolRequest,
    result: RetrievalToolResult,
    audit: _QueryAudit,
    turn_id: str | None,
    duration_ms: int,
    final_projection: Mapping[str, object] | None = None,
    diagnostics: Sequence[Mapping[str, object]] = (),
) -> None:
    """把一次检索查询投影为 trajectory；不复制正文或私有路径。"""

    from ....trajectory import record_retrieval

    method_outcomes: list[dict[str, object]] = []
    reranker_outcomes: list[dict[str, object]] = []
    item_by_ref: dict[
        tuple[SourceType, str, str, str],
        RetrievedItem,
    ] = {}
    encoder_fingerprints: set[str] = set()
    fusion_candidates: list[dict[str, object]] = []
    fusion_candidate_count = 0
    remaining_lane_candidate_budget = MAX_FILE_QUERY_AUDIT_CANDIDATES
    remaining_fusion_candidate_budget = MAX_FILE_QUERY_AUDIT_CANDIDATES
    remaining_reranker_candidate_budget = MAX_FILE_QUERY_AUDIT_CANDIDATES
    audit_diagnostics: list[Mapping[str, object]] = [
        *diagnostics,
        *audit.diagnostics,
    ]
    for context_index, context in enumerate(audit.contexts):
        lane_candidates_by_key: dict[
            tuple[SourceType, RetrievalMethod, int],
            list[RetrievalCandidate],
        ] = {}
        for candidate in context.lane_candidates:
            lane_candidates_by_key.setdefault(
                (
                    candidate.ref.source_type,
                    candidate.method,
                    candidate.query_index,
                ),
                [],
            ).append(candidate)
        reranker_candidates_by_source: dict[
            SourceType,
            list[RetrievalCandidate],
        ] = {}
        for candidate in context.reranker_candidates:
            reranker_candidates_by_source.setdefault(
                candidate.ref.source_type,
                [],
            ).append(candidate)
        if context.encoder_fingerprint:
            encoder_fingerprints.add(
                _audit_fingerprint_ref(context.encoder_fingerprint)
            )
        for outcome in context.method_outcomes:
            lane_candidates = sorted(
                lane_candidates_by_key.get(
                    (outcome.source_type, outcome.method, outcome.query_index),
                    (),
                ),
                key=lambda candidate: (
                    candidate.rank,
                    candidate.ref.source_unit_id,
                    candidate.ref.source_revision,
                ),
            )
            recorded_lane_candidates = lane_candidates[
                : min(
                    MAX_FILE_QUERY_AUDIT_CANDIDATES_PER_LANE,
                    remaining_lane_candidate_budget,
                )
            ]
            remaining_lane_candidate_budget -= len(recorded_lane_candidates)
            method_reason = _audit_reason_code(
                outcome.reason_code,
                fallback="retrieval_method_failure_unclassified",
            )
            failure_stage = _audit_reason_code(
                outcome.failure_stage,
                fallback="retrieval_method_stage_unclassified",
            )
            method_projection: dict[str, object] = {
                "context_index": context_index,
                "method": outcome.method.value,
                "source_type": outcome.source_type.value,
                "query_index": outcome.query_index,
                "status": outcome.status.value,
                "candidate_count": outcome.candidate_count,
                "reason_code": method_reason,
                "degraded_from": [
                    method.value for method in outcome.degraded_from
                ],
                "attempt": outcome.attempt,
                "infrastructure_attempts": outcome.infrastructure_attempts,
                "recorded_candidate_count": len(recorded_lane_candidates),
                "candidates_truncated": (
                    outcome.candidate_count > len(recorded_lane_candidates)
                    or len(lane_candidates) > len(recorded_lane_candidates)
                ),
                "candidates": [
                    _lane_candidate_audit(candidate)
                    for candidate in recorded_lane_candidates
                ],
            }
            if failure_stage is not None:
                method_projection["failure_stage"] = failure_stage
            if outcome.attempt_failures:
                method_projection["attempt_failures"] = [
                    {
                        "attempt": attempt,
                        "reason_code": _audit_reason_code(
                            reason_code,
                            fallback="retrieval_method_failure_unclassified",
                        ),
                        "failure_stage": _audit_reason_code(
                            failure_stage,
                            fallback="retrieval_method_stage_unclassified",
                        ),
                    }
                    for attempt, reason_code, failure_stage in outcome.attempt_failures
                ]
            method_outcomes.append(method_projection)
            if outcome.status.value != "used":
                audit_diagnostics.append(
                    {
                        "stage": "method",
                        "code": method_reason or "method_degraded",
                        "status": outcome.status.value,
                        "failure_stage": failure_stage,
                        "context_index": context_index,
                        "source_type": outcome.source_type.value,
                        "method": outcome.method.value,
                        "query_index": outcome.query_index,
                        "known_count": outcome.candidate_count,
                    }
                )
        for outcome in context.reranker_outcomes:
            reranker_candidates = sorted(
                (
                    candidate for candidate in reranker_candidates_by_source.get(outcome.source_type, ())
                    if outcome.query_index is None or candidate.query_index == outcome.query_index
                ),
                key=lambda candidate: (
                    candidate.rerank_rank is None,
                    candidate.rerank_rank or candidate.rank,
                    candidate.rank,
                    candidate.ref.source_unit_id,
                    candidate.ref.source_revision,
                ),
            )
            recorded_reranker_candidates = reranker_candidates[
                : min(
                    MAX_FILE_QUERY_AUDIT_CANDIDATES_PER_CONTEXT,
                    remaining_reranker_candidate_budget,
                )
            ]
            remaining_reranker_candidate_budget -= len(recorded_reranker_candidates)
            reranker_reason = _audit_reason_code(
                outcome.reason_code,
                fallback="reranker_failure_unclassified",
            )
            reranker_outcomes.append(
                {
                    "context_index": context_index,
                    "source_type": outcome.source_type.value,
                    "status": outcome.status.value,
                    "candidate_count": outcome.candidate_count,
                    "scored_candidate_count": outcome.scored_candidate_count,
                    "query_index": outcome.query_index,
                    "reason_code": reranker_reason,
                    "fingerprint": (
                        _audit_fingerprint_ref(context.reranker_fingerprint)
                        if context.reranker_fingerprint
                        else None
                    ),
                    "recorded_candidate_count": len(recorded_reranker_candidates),
                    "candidates_truncated": (
                        outcome.candidate_count > len(recorded_reranker_candidates)
                        or len(reranker_candidates)
                        > len(recorded_reranker_candidates)
                    ),
                    "candidates": [
                        _reranker_candidate_audit(candidate)
                        for candidate in recorded_reranker_candidates
                    ],
                }
            )
            if outcome.status.value == "degraded":
                audit_diagnostics.append(
                    {
                        "stage": "reranker",
                        "code": reranker_reason or "reranker_degraded",
                        "status": "degraded",
                        "context_index": context_index,
                        "source_type": outcome.source_type.value,
                        "known_count": max(
                            0,
                            outcome.candidate_count
                            - outcome.scored_candidate_count,
                        ),
                    }
                )
        for item in context.items:
            item_by_ref[
                (
                    item.ref.source_type,
                    item.ref.source_unit_id,
                    item.ref.source_revision,
                    item.ref.indexed_content_hash,
                )
            ] = item
        context_fusion_candidates = sorted(
            context.fusion_candidates,
            key=lambda candidate: (
                candidate.ref.source_type.value,
                candidate.rank,
                candidate.query_index,
                candidate.ref.source_unit_id,
                candidate.ref.source_revision,
            ),
        )
        fusion_candidate_count += len(context_fusion_candidates)
        recorded_fusion_candidates = context_fusion_candidates[
            : min(
                MAX_FILE_QUERY_AUDIT_CANDIDATES_PER_CONTEXT,
                remaining_fusion_candidate_budget,
            )
        ]
        remaining_fusion_candidate_budget -= len(recorded_fusion_candidates)
        fusion_candidates.extend(
            _fusion_candidate_audit(candidate, context_index=context_index)
            for candidate in recorded_fusion_candidates
        )
        audit_diagnostics.extend(
            _context_audit_diagnostics(context, context_index=context_index)
        )

    audit_diagnostics.extend(
        {
            "stage": "port",
            "code": gap.code,
            "status": "blocked" if gap.blocking else "partial",
            "source_type": (
                gap.source_type.value if gap.source_type is not None else None
            ),
            "file_id": next(
                (
                    binding.source_id
                    for binding in request.file_bindings
                    if binding.authority_id == gap.authority_id
                ),
                None,
            ),
            "blocking": gap.blocking,
            "known_count": gap.known_count,
        }
        for gap in result.gaps
    )

    packed_refs = (
        _final_projection_evidence(
            final_projection,
            result=result,
            item_by_ref=item_by_ref,
        )
        if final_projection is not None
        else _port_projection_evidence(result, item_by_ref=item_by_ref)
    )
    bounded_diagnostics = _bounded_audit_diagnostics(audit_diagnostics)
    final_status = (
        str(final_projection.get("status") or result.status.value)
        if final_projection is not None
        else result.status.value
    )
    final_reason = _final_projection_reason(final_projection)
    fusion: dict[str, object] = {
        "algorithm": "rrf",
        "context_count": len(audit.contexts),
        "contributor_ranks_recorded": True,
        "encoder_fingerprints": sorted(encoder_fingerprints),
        "candidate_count": fusion_candidate_count,
        "recorded_candidate_count": len(fusion_candidates),
        "candidates_truncated": fusion_candidate_count > len(fusion_candidates),
        "candidates": fusion_candidates,
    }
    if audit.execution is not None:
        fusion["execution"] = audit.execution.snapshot()
    if final_projection is not None:
        fusion.update(
            {
                "scope": "query_rrf_then_joint_rerank_exact_source_union",
                "result_limit": request.limit,
                "port_packed_count": len(result.evidence),
                "final_injected_count": len(packed_refs),
                **(
                    {
                        "v2_projection_drop_count": sum(
                            1
                            for item in bounded_diagnostics
                            if item.get("stage") == "v2_projection"
                            and item.get("status") == "dropped"
                        )
                    }
                    if request.corpus is RetrievalCorpus.FILES
                    else {
                        "public_projection_drop_count": max(
                            0,
                            len(result.evidence) - len(packed_refs),
                        )
                    }
                ),
            }
        )

    record_retrieval(
        queries=request.queries,
        source_scope=(
            "MODEL_RETRIEVE_FILES_V1"
            if request.corpus is RetrievalCorpus.FILES
            else "MODEL_RETRIEVE_HISTORY_V1"
        ),
        evidence=packed_refs,
        method_outcomes=method_outcomes,
        reranker_outcomes=reranker_outcomes,
        diagnostics=bounded_diagnostics,
        fusion=fusion,
        outcome=final_status,
        reason_code=final_reason
        or next((gap.code for gap in result.gaps if gap.blocking), None),
        duration_ms=duration_ms,
        session_id=request.session_id,
        turn_id=turn_id,
    )


def _port_projection_evidence(
    result: RetrievalToolResult,
    *,
    item_by_ref: Mapping[tuple[SourceType, str, str, str], RetrievedItem],
) -> list[dict[str, object]]:
    packed_refs: list[dict[str, object]] = []
    for packed in result.evidence:
        item = item_by_ref.get(
            (
                packed.source_type,
                packed.source_unit_id,
                packed.source_revision,
                packed.indexed_content_hash,
            )
        )
        producer_chunk_id = packed.citation.get("producer_chunk_id")
        packed_refs.append(
            {
                "source_type": packed.source_type.value,
                "source_unit_id": packed.source_unit_id,
                "source_revision": packed.source_revision,
                "indexed_content_hash": packed.indexed_content_hash,
                "producer_chunk_id": (
                    producer_chunk_id
                    if isinstance(producer_chunk_id, str)
                    else None
                ),
                "packed_rank": packed.rank,
                "query_index": packed.query_index,
                "fused_rank": item.fused_rank if item is not None else None,
                "reranked_rank": (
                    item.reranked_rank if item is not None else None
                ),
                "fusion_score": (
                    _finite_audit_score(item.fusion_score)
                    if item is not None
                    else None
                ),
                "reranker_score": (
                    _finite_audit_score(item.reranker_score)
                    if item is not None
                    else None
                ),
            }
        )
    return packed_refs


def _final_projection_evidence(
    projection: Mapping[str, object],
    *,
    result: RetrievalToolResult,
    item_by_ref: Mapping[tuple[SourceType, str, str, str], RetrievedItem],
) -> list[dict[str, object]]:
    evidence = projection.get("evidence")
    if not isinstance(evidence, list):
        return []
    packed_by_rank = {item.rank: item for item in result.evidence}
    projected: list[dict[str, object]] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        rank = item.get("rank")
        packed = packed_by_rank.get(rank) if isinstance(rank, int) else None
        output = {
            key: item.get(key)
            for key in (
                "handle",
                "file_id",
                "file_version_id",
                "document_id",
                "document_version_id",
                "chunk_id",
                "picture_id",
                "picture_unit_id",
                "observation_id",
                "chunk_sequence",
                "content_sha256",
                "source_revision_fingerprint",
                "rank",
            )
        }
        output["projection_status"] = "injected"
        query_index = item.get("query_index")
        if type(query_index) is int and 0 <= query_index < MAX_FILE_RETRIEVAL_QUERIES:
            output["query_index"] = query_index
        if "query_matches" in item:
            output["query_matches"] = _bounded_query_matches(item.get("query_matches"))
        if packed is not None:
            retrieved = item_by_ref.get(
                (
                    packed.source_type,
                    packed.source_unit_id,
                    packed.source_revision,
                    packed.indexed_content_hash,
                )
            )
            producer_chunk_id = packed.citation.get("producer_chunk_id")
            output.update(
                {
                    "source_type": packed.source_type.value,
                    "source_unit_id": packed.source_unit_id,
                    "source_revision": packed.source_revision,
                    "indexed_content_hash": packed.indexed_content_hash,
                    "producer_chunk_id": (
                        producer_chunk_id
                        if isinstance(producer_chunk_id, str)
                        else None
                    ),
                    "packed_rank": packed.rank,
                    "fused_rank": (
                        retrieved.fused_rank if retrieved is not None else None
                    ),
                    "reranked_rank": (
                        retrieved.reranked_rank if retrieved is not None else None
                    ),
                    "fusion_score": (
                        _finite_audit_score(retrieved.fusion_score)
                        if retrieved is not None
                        else None
                    ),
                    "reranker_score": (
                        _finite_audit_score(retrieved.reranker_score)
                        if retrieved is not None
                        else None
                    ),
                }
            )
        projected.append(output)
    return projected


def _bounded_query_matches(value: object) -> list[dict[str, object]]:
    """只保留最终工具实际发布的数值关联；绝不复制任意额外字段或正文。"""

    if not isinstance(value, (list, tuple)):
        return []
    matches: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in value[:MAX_FILE_RETRIEVAL_QUERIES]:
        if not isinstance(raw, Mapping):
            continue
        query_index, rank, fused_rank = (raw.get(name) for name in ("query_index", "rank", "fused_rank"))
        if (
            type(query_index) is not int or not 0 <= query_index < MAX_FILE_RETRIEVAL_QUERIES
            or type(rank) is not int or not 1 <= rank <= FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY
            or type(fused_rank) is not int or not 1 <= fused_rank <= FILE_RETRIEVAL_CANDIDATES_PER_QUERY
            or query_index in seen
        ):
            continue
        seen.add(query_index)
        matches.append({
            "query_index": query_index, "rank": rank, "fused_rank": fused_rank,
            "fusion_score": _finite_audit_score(raw.get("fusion_score")),
            "reranker_score": _finite_audit_score(raw.get("reranker_score")),
        })
    return matches


def _lane_candidate_audit(candidate: RetrievalCandidate) -> dict[str, object]:
    return {
        "source_unit_id": candidate.ref.source_unit_id,
        "source_revision": candidate.ref.source_revision,
        "indexed_content_hash": candidate.ref.indexed_content_hash,
        "rank": candidate.rank,
        "raw_score": _finite_audit_score(candidate.raw_score),
    }


def _fusion_candidate_audit(
    candidate: RetrievalCandidate,
    *,
    context_index: int,
) -> dict[str, object]:
    return {
        "context_index": context_index,
        "source_type": candidate.ref.source_type.value,
        "source_unit_id": candidate.ref.source_unit_id,
        "source_revision": candidate.ref.source_revision,
        "indexed_content_hash": candidate.ref.indexed_content_hash,
        "query_index": candidate.query_index,
        "source_fused_rank": candidate.rank,
        "query_fused_rank": candidate.query_fused_rank,
        "fusion_score": _finite_audit_score(candidate.fusion_score),
        "contributors": [
            {
                "method": method.value,
                "rank": rank,
                "raw_score": _finite_audit_score(raw_score),
            }
            for method, rank, raw_score in candidate.fusion_contributors
        ],
    }


def _reranker_candidate_audit(candidate: RetrievalCandidate) -> dict[str, object]:
    return {
        "source_unit_id": candidate.ref.source_unit_id,
        "source_revision": candidate.ref.source_revision,
        "indexed_content_hash": candidate.ref.indexed_content_hash,
        "query_index": candidate.query_index,
        "pre_rank": candidate.rank,
        "post_rank": candidate.rerank_rank,
        "score": _finite_audit_score(candidate.rerank_score),
    }


def _finite_audit_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    projected = float(value)
    return projected if math.isfinite(projected) else None


def _final_projection_reason(
    projection: Mapping[str, object] | None,
) -> str | None:
    if projection is None:
        return None
    gaps = projection.get("gaps")
    if not isinstance(gaps, list):
        return None
    for gap in gaps:
        if not isinstance(gap, Mapping) or gap.get("blocking") is not True:
            continue
        code = gap.get("code")
        if isinstance(code, str) and _SAFE_AUDIT_TOKEN.fullmatch(code):
            return code
    return None


def _context_audit_diagnostics(
    context: RetrievedContext,
    *,
    context_index: int,
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for source_type, outcome in context.source_outcomes.items():
        healthy = (
            outcome.availability is SourceAvailability.READY
            and outcome.retrieval
            in {SourceRetrievalStatus.MATCHED, SourceRetrievalStatus.NO_MATCH}
        ) or (
            outcome.availability is SourceAvailability.EMPTY
            and outcome.retrieval is SourceRetrievalStatus.NOT_RUN
        ) or (
            outcome.availability is SourceAvailability.DISABLED
            and outcome.retrieval is SourceRetrievalStatus.NOT_RUN
        )
        if healthy:
            continue
        diagnostics.append(
            {
                "stage": "source",
                "code": outcome.reason_code or "source_retrieval_incomplete",
                "status": outcome.retrieval.value,
                "context_index": context_index,
                "source_type": source_type.value,
                "blocking": outcome.availability
                in {SourceAvailability.BLOCKED, SourceAvailability.UNAVAILABLE},
            }
        )
    for limitation in context.coverage_limitations:
        diagnostics.append(
            {
                "stage": "coverage",
                "code": limitation.reason_code,
                "status": "partial",
                "context_index": context_index,
                "source_type": limitation.source_type.value,
            }
        )
    for omission in context.pack_omissions:
        diagnostics.append(
            {
                "stage": "packing",
                "code": omission.reason.value,
                "status": "omitted",
                "context_index": context_index,
                "source_type": omission.source_type.value,
                "known_count": omission.count,
            }
        )
    for drop in context.verification_drops:
        diagnostics.append(
            {
                "stage": "verification",
                "code": drop.reason.value,
                "status": "dropped",
                "context_index": context_index,
                "source_type": drop.source_type.value,
                "known_count": drop.count,
            }
        )
    diagnostics.extend(
        {
            "stage": "context",
            "code": code,
            "status": "partial",
            "context_index": context_index,
        }
        for code in context.diagnostic_codes
    )
    if context.truncated:
        diagnostics.append(
            {
                "stage": "packing",
                "code": "context_truncated",
                "status": "partial",
                "context_index": context_index,
                "known_count": 1,
            }
        )
    return diagnostics


_AUDIT_TOKEN_FIELDS = frozenset(
    {"stage", "code", "status", "source_type", "method", "failure_stage"}
)
_AUDIT_IDENTITY_FIELDS = frozenset(
    {
        "file_id",
        "source_unit_id",
        "source_revision",
        "indexed_content_hash",
        "producer_chunk_id",
    }
)
_AUDIT_INTEGER_FIELDS = frozenset(
    {"context_index", "query_index", "known_count", "packed_rank"}
)
_SAFE_AUDIT_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_SAFE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _bounded_audit_diagnostics(
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    safe: list[dict[str, object]] = []
    for value in values:
        item: dict[str, object] = {}
        for key in _AUDIT_TOKEN_FIELDS:
            field_value = value.get(key)
            if isinstance(field_value, str) and _SAFE_AUDIT_TOKEN.fullmatch(
                field_value
            ):
                item[key] = field_value
        for key in _AUDIT_IDENTITY_FIELDS:
            field_value = value.get(key)
            validator = (
                _SAFE_SHA256
                if key == "indexed_content_hash"
                else _SAFE_AUDIT_IDENTITY
            )
            if isinstance(field_value, str) and validator.fullmatch(field_value):
                item[key] = field_value
        for key in _AUDIT_INTEGER_FIELDS:
            field_value = value.get(key)
            if (
                isinstance(field_value, int)
                and not isinstance(field_value, bool)
                and field_value >= 0
            ):
                item[key] = field_value
        if isinstance(value.get("blocking"), bool):
            item["blocking"] = value["blocking"]
        item.setdefault("stage", "retrieval")
        item.setdefault("code", "retrieval_diagnostic_unavailable")
        item.setdefault("status", "partial")
        safe.append(item)
    if len(safe) <= MAX_FILE_QUERY_AUDIT_DIAGNOSTICS:
        return safe
    omitted = len(safe) - (MAX_FILE_QUERY_AUDIT_DIAGNOSTICS - 1)
    return [
        *safe[: MAX_FILE_QUERY_AUDIT_DIAGNOSTICS - 1],
        {
            "stage": "trajectory",
            "code": "retrieval_diagnostics_truncated",
            "status": "omitted",
            "known_count": omitted,
        },
    ]


def _safe_retrieval_exception_code(exc: Exception) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", type(exc).__name__.lower()).strip("_")
    candidate = f"retrieval_exception_{name or 'unknown'}"
    return candidate[:96]


def _audit_reason_code(value: object, *, fallback: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and _SAFE_AUDIT_TOKEN.fullmatch(value):
        return value
    return fallback


def _audit_fingerprint_ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "_DeferredQueryAudit",
    "_QueryAudit",
    "_record_query_audit",
    "_safe_retrieval_exception_code",
]
