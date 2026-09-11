"""query 内融合取候选，一次批量按各自 query 评分，最终合并去重并限总条数。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math

from ..contracts import (
    FILE_RETRIEVAL_CANDIDATES_PER_QUERY,
    FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY,
    MAX_FILE_RETRIEVAL_ITEMS,
    MAX_FILE_RETRIEVAL_QUERIES,
    ContextPackOmission,
    ContextPackOmissionReason,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalQueryMatch,
    RetrievalRequest,
    RetrievedItem,
    RerankerOutcome,
    RerankerRunStatus,
    SourcePlan,
    SourceType,
    SourceUnitRef,
)
from ..ports import RetrievalCancelled, RetrievalPolicyPort, RerankerPort, TokenEstimator
from ..execution import RerankingDeadlineReached, checkpoint, measure
from ..query_guard import QueryGuard
from .packing import ContextPackResult
from .ranking import fuse_candidates_by_rrf
from .recovery import RetrievalRecoveryController
from .reranking import RerankerUnavailable
from .source_execution import SourceExecution, SourceExecutionState, _merge_method_hits
from .selection import ResolvedCandidate


@dataclass(slots=True)
class FileQueryScope:
    request: RetrievalRequest
    source_plans: tuple[SourcePlan, ...]
    recovery: RetrievalRecoveryController
    state: SourceExecutionState
    items: list[RetrievedItem] = field(default_factory=list)
    omissions: list[ContextPackOmission] = field(default_factory=list)
    reranker_candidates: list[RetrievalCandidate] = field(default_factory=list)


def execute_file_query_batch(
    requests: Sequence[RetrievalRequest],
    budget: RetrievalBudget,
    *,
    result_limit: int,
    execution: SourceExecution,
    policy: RetrievalPolicyPort,
    query_guard: QueryGuard,
    generation: str | None,
    max_attempts: int,
    rrf_k: int,
    reranker: RerankerPort | None,
    token_estimator: TokenEstimator,
) -> tuple[tuple[FileQueryScope, ...], tuple[RerankerOutcome, ...], str]:
    """先核验授权，再 query 内 RRF，批量评分后按精确来源去重并保留总前 K。"""

    queries = _validate_requests(requests, budget, result_limit, query_guard)
    scopes: list[FileQueryScope] = []
    for request in requests:
        checkpoint()
        plan = policy.compile(request, budget)
        recovery = RetrievalRecoveryController(max_attempts=max_attempts)
        state = execution.execute(
            source_plans=plan.source_plans,
            queries=queries,
            candidate_limit=budget.candidate_limit_per_source,
            recovery=recovery,
            retrieval_data_version_id=generation,
            expected_source_snapshots=request.boundary.expected_source_snapshots,
        )
        scopes.append(FileQueryScope(request, plan.source_plans, recovery, state))

    _select_global_candidates(
        scopes, len(queries), budget.candidate_limit_per_source, rrf_k
    )
    with measure("fetch_verify"):
        resolved, owners = _fetch_verified_candidates(scopes, execution)
    items, outcomes, fingerprint = _rerank_queries(
        queries,
        resolved,
        reranker,
        token_estimator,
    )
    for candidate, _ in items:
        scopes[
            owners[(candidate.query_index, candidate.ref)]
        ].reranker_candidates.append(candidate)
    for outcome in outcomes:
        if outcome.status is RerankerRunStatus.DEGRADED:
            for scope in scopes:
                if outcome.source_type in scope.state.source_outcomes:
                    scope.state.diagnostics.append(
                        f"reranker_degraded:{outcome.source_type.value}:query_{outcome.query_index}:"
                        f"{outcome.reason_code or 'reranker_unavailable'}"
                    )
    checkpoint()
    with measure("packing"):
        packed = pack_query_items(
            [item for _, item in items],
            result_limit=result_limit,
            token_limit=budget.context_token_limit,
        )
    for item in packed.items:
        scopes[owners[(item.query_index, item.ref)]].items.append(item)
    for omission in packed.omissions:
        next(
            scope
            for scope in scopes
            if omission.source_type in scope.state.source_outcomes
        ).omissions.append(omission)
    return tuple(scopes), tuple(outcomes), fingerprint


def _fetch_verified_candidates(
    scopes: Sequence[FileQueryScope],
    execution: SourceExecution,
) -> tuple[
    dict[SourceType, list[ResolvedCandidate]], dict[tuple[int, SourceUnitRef], int]
]:
    """按各自冻结范围读取，返回每个 query/ref 的确切授权 owner。"""
    resolved: dict[SourceType, list[ResolvedCandidate]] = defaultdict(list)
    owners: dict[tuple[int, SourceUnitRef], int] = {}
    for scope_index, scope in enumerate(scopes):
        checkpoint()
        if scope.state.blocked:
            continue
        values = execution.fetch_and_verify(
            request=scope.request,
            source_plans=scope.source_plans,
            state=scope.state,
            recovery=scope.recovery,
        )
        execution.revalidate_expected_snapshots_after_fetch(
            expected_source_snapshots=scope.request.boundary.expected_source_snapshots,
            state=scope.state,
            resolved_by_source=values,
            recovery=scope.recovery,
        )
        for source_type, candidates in values.items():
            for candidate, unit in candidates:
                owners[(candidate.query_index, candidate.ref)] = scope_index
                resolved[source_type].append((candidate, unit))

    return resolved, owners


def _validate_requests(
    requests: Sequence[RetrievalRequest],
    budget: RetrievalBudget,
    result_limit: int,
    query_guard: QueryGuard,
) -> tuple[str, ...]:
    if not requests:
        raise ValueError("file query batch requires at least one authorized scope")
    if (
        isinstance(result_limit, bool)
        or not isinstance(result_limit, int)
        or not 1 <= result_limit <= MAX_FILE_RETRIEVAL_ITEMS
    ):
        raise ValueError("result_limit must be between 1 and 96")
    queries = query_guard.validate(requests[0].query_proposal)
    if not 1 <= len(queries) <= MAX_FILE_RETRIEVAL_QUERIES:
        raise ValueError("file query batch accepts one to four queries")
    if (
        budget.candidate_limit_per_source != FILE_RETRIEVAL_CANDIDATES_PER_QUERY
        or budget.max_items != result_limit
    ):
        raise ValueError("file batch requires recall=128 per query and total result_limit")
    for request in requests:
        if query_guard.validate(request.query_proposal) != queries:
            raise ValueError("file query batch scopes must have the same queries")
        if set(request.boundary.source_filters) - {
            SourceType.DOCUMENT,
            SourceType.PICTURE,
        }:
            raise ValueError("file query batch cannot retrieve history sources")
    return queries


def _select_global_candidates(
    scopes: Sequence[FileQueryScope],
    query_count: int,
    candidate_limit: int,
    rrf_k: int,
) -> None:
    """先跨文件合并同方法排序，再 query 内 RRF；无跨 query 的 seen 集合。"""

    lanes = defaultdict(list)
    owners = {}
    for scope_index, scope in enumerate(scopes):
        # execute 的二次权限/新鲜度校验会清空失效来源；不能拿原始召回把它复活。
        allowed = {
            source
            for source, candidates in scope.state.candidates_by_source.items()
            if candidates
        }
        for candidate in scope.state.lane_candidates:
            if scope.state.blocked or candidate.ref.source_type not in allowed:
                continue
            if candidate.ref in scope.request.excluded_direct_refs:
                continue
            lanes[(candidate.query_index, candidate.method)].append(candidate)
            owners.setdefault((candidate.query_index, candidate.ref), scope_index)
        scope.state.candidates_by_source.clear()
        scope.state.fusion_candidates.clear()
    for query_index in range(query_count):
        checkpoint()
        hits = [
            candidate
            for (lane_query, _), lane in lanes.items()
            if lane_query == query_index
            for candidate in _merge_method_hits(lane, limit=candidate_limit)
        ]
        for candidate in fuse_candidates_by_rrf(hits, rrf_k=rrf_k)[:FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY]:
            scope = scopes[owners[(query_index, candidate.ref)]]
            scope.state.candidates_by_source.setdefault(
                candidate.ref.source_type, []
            ).append(candidate)
            scope.state.fusion_candidates.append(candidate)


def _rerank_queries(
    queries: Sequence[str],
    resolved: Mapping[SourceType, Sequence[ResolvedCandidate]],
    reranker: RerankerPort | None,
    token_estimator: TokenEstimator,
) -> tuple[list[tuple[RetrievalCandidate, RetrievedItem]], list[RerankerOutcome], str]:
    """只调一次 scorer，但每个文本对仍使用召回它的 query，不拼接不同问题。

    任何评分失败使整个批次保留融合顺序；取消必须上抛，不能伪装重排降级。
    按 Source/query 的 outcome 是审计分组，不代表调用了多次重排模型。
    """
    pairs_by_key = {}
    for candidates in resolved.values():
        for candidate, unit in candidates:
            key = (candidate.query_index, candidate.ref)
            if key in pairs_by_key and pairs_by_key[key][1].content != unit.content:
                raise ValueError("conflicting content for the same exact query/source")
            pairs_by_key.setdefault(key, (candidate, unit))
    pairs = sorted(pairs_by_key.values(), key=lambda pair: (
        pair[0].query_index, pair[0].query_fused_rank or pair[0].rank,
        pair[0].ref.source_type.value, pair[0].ref.source_unit_id, pair[0].ref.source_revision,
    ))
    scores = None
    reason = None
    fingerprint = ""
    if reranker is not None and pairs:
        checkpoint()
        try:
            fingerprint = reranker.fingerprint()
            raw = tuple(float(score) for score in reranker.score(tuple(
                (queries[candidate.query_index], unit.content) for candidate, unit in pairs
            )))
            if len(raw) != len(pairs) or any(not math.isfinite(score) for score in raw):
                raise ValueError("reranker_returned_unaligned_scores")
            scores = raw
        except RetrievalCancelled:
            raise
        except RerankingDeadlineReached:
            reason = "reranker_deadline_reached"
        except RerankerUnavailable as exc:
            # 适配器只在此领域异常中传递安全原因码，不透传底层异常正文。
            reason = str(exc) or "reranker_unavailable"
        except Exception as exc:
            reason = f"reranker_failed:{type(exc).__name__}"
    checkpoint()
    scored_pairs = []
    query_lanes = defaultdict(list)
    for index, (candidate, unit) in enumerate(pairs):
        query_lanes[candidate.query_index].append((candidate, unit, scores[index] if scores is not None else None))
    for lane in query_lanes.values():
        lane.sort(key=lambda value: (
            -value[2] if value[2] is not None else value[0].query_fused_rank or value[0].rank,
            value[0].query_fused_rank or value[0].rank, value[0].ref.source_unit_id,
        ))
        for rank, (candidate, unit, score) in enumerate(lane, 1):
            scored_pairs.append((replace(candidate, rerank_score=score,
                                         rerank_rank=rank if score is not None else None), unit))
    counts = Counter((candidate.ref.source_type, candidate.query_index) for candidate, _ in pairs)
    outcomes = [RerankerOutcome(
        source_type=source, query_index=query_index, candidate_count=count,
        scored_candidate_count=count if scores is not None else 0,
        status=RerankerRunStatus.USED if scores is not None else (
            RerankerRunStatus.DEGRADED if reason else RerankerRunStatus.NOT_RUN
        ), reason_code=reason,
    ) for (source, query_index), count in sorted(counts.items())]
    items = [(candidate, RetrievedItem(
        ref=unit.ref, content=unit.content, citation=unit.citation,
        estimated_tokens=token_estimator(unit.content),
        fused_rank=candidate.query_fused_rank or candidate.rank,
        query_index=candidate.query_index, fusion_score=candidate.fusion_score,
        reranked_rank=candidate.rerank_rank, reranker_score=candidate.rerank_score,
    )) for candidate, unit in scored_pairs]
    return items, outcomes, fingerprint


def pack_query_items(
    items: Sequence[RetrievedItem],
    *,
    result_limit: int,
    token_limit: int,
) -> ContextPackResult:
    """同块保留全部 query 关系、取最高匹配分，按全局前 K 打包；原始证据不变。"""

    if type(result_limit) is not int or not 1 <= result_limit <= MAX_FILE_RETRIEVAL_ITEMS:
        raise ValueError("result_limit must be between 1 and 96")

    lanes = defaultdict(dict)
    contents = {}
    selection_complete = bool(items) and all(item.query_matches for item in items)
    if any(item.query_matches for item in items) and not selection_complete:
        raise ValueError("cannot mix selected query results with unselected candidates")
    for item in items:
        if item.ref in contents and contents[item.ref] != item.content:
            raise ValueError("conflicting content for the same exact source")
        contents[item.ref] = item.content
        matches = item.query_matches or (
            RetrievalQueryMatch(
                query_index=item.query_index,
                rank=item.reranked_rank or item.fused_rank,
                fused_rank=item.fused_rank,
                fusion_score=item.fusion_score,
                reranker_score=item.reranker_score,
            ),
        )
        for match in matches:
            lanes[match.query_index].setdefault(item.ref, (item, match))
    retained, omitted = {}, Counter()
    scored = bool(lanes) and all(
        match.reranker_score is not None for lane in lanes.values() for _, match in lane.values()
    )
    if not scored and any(
        match.reranker_score is not None for lane in lanes.values() for _, match in lane.values()
    ):
        # 部分评分不能作为整体重排。清除分数及其排名，防止无分项被截掉后，
        # 第二次打包把剩余候选误认作完整评分批次并改变顺序。
        lanes = {
            query_index: {
                ref: (
                    replace(item, reranker_score=None, reranked_rank=None),
                    replace(match, reranker_score=None, rank=match.fused_rank),
                )
                for ref, (item, match) in lane.items()
            }
            for query_index, lane in lanes.items()
        }
        selection_complete = False
    for query_index, lane in sorted(lanes.items()):
        ordered = sorted(
            lane.values(),
            key=lambda value: (
                value[1].rank
                if selection_complete
                else -value[1].reranker_score
                if scored
                else value[1].fused_rank,
                value[1].fused_rank,
                value[0].ref.source_unit_id,
                value[0].ref.source_revision,
            ),
        )
        retained[query_index] = [
            (item, match if selection_complete else replace(match, rank=rank))
            for rank, (item, match) in enumerate(ordered, 1)
        ]

    by_ref, matches_by_ref = {}, defaultdict(list)
    events = sorted(
        (value for lane in retained.values() for value in lane),
        key=lambda value: (
            -value[1].reranker_score if scored else value[1].fused_rank,
            value[1].rank, value[1].query_index,
            value[0].ref.source_type.value, value[0].ref.source_unit_id,
            value[0].ref.source_revision,
        ),
    )
    for item, match in events:
        by_ref.setdefault(item.ref, (item, match))
        matches_by_ref[item.ref].append(match)
    selected, used = [], 0
    for index, (ref, (item, first_match)) in enumerate(by_ref.items()):
        if index >= result_limit:
            omitted[(ref.source_type, ContextPackOmissionReason.MAX_ITEMS)] += 1
            continue
        if used + item.estimated_tokens > token_limit:
            omitted[(ref.source_type, ContextPackOmissionReason.TOKEN_LIMIT)] += 1
            continue
        selected.append(
            replace(
                item,
                query_index=first_match.query_index,
                fused_rank=first_match.fused_rank,
                fusion_score=first_match.fusion_score,
                reranker_score=first_match.reranker_score,
                reranked_rank=first_match.rank
                if first_match.reranker_score is not None
                else None,
                query_matches=tuple(
                    sorted(matches_by_ref[ref], key=lambda match: match.query_index)
                ),
            )
        )
        used += item.estimated_tokens
    return ContextPackResult(
        tuple(selected),
        used,
        (),
        tuple(
            ContextPackOmission(source, reason, count)
            for (source, reason), count in sorted(
                omitted.items(),
                key=lambda value: (value[0][0].value, value[0][1].value),
            )
        ),
    )
