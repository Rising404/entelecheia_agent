"""Source 本地执行、回退及权威内容验证。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    ContextCoverageLimitation,
    ContextVerificationDrop,
    ContextVerificationDropReason,
    MethodOutcome,
    MethodRunStatus,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalRequest,
    SourceAccess,
    SourceAvailability,
    SourceDependency,
    SourceOutcome,
    SourcePlan,
    SourceFilter,
    SourceRetrievalStatus,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from ..ports import (
    RetrievalMethodPort,
    SourceAccessRevalidator,
    SourceCoverageProbe,
    SourceRetrievalAdapter,
)
from ..execution import checkpoint, current_execution, measure
from .recovery import RetrievalRecoveryController
from .ranking import fuse_candidates_by_rrf, round_robin_query_candidates


@dataclass(slots=True)
class SourceExecutionState:
    """执行与验证共享的可变单请求来源状态。"""

    source_outcomes: dict[SourceType, SourceOutcome] = field(default_factory=dict)
    source_access: dict[SourceType, SourceAccess] = field(default_factory=dict)
    method_outcomes: list[MethodOutcome] = field(default_factory=list)
    lane_candidates: list[RetrievalCandidate] = field(default_factory=list)
    fusion_candidates: list[RetrievalCandidate] = field(default_factory=list)
    candidates_by_source: dict[SourceType, list[RetrievalCandidate]] = field(default_factory=dict)
    verification_drops: list[ContextVerificationDrop] = field(default_factory=list)
    coverage_limitations: list[ContextCoverageLimitation] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    blocked: bool = False


class SourceExecution:
    """运行受信来源计划，但不接管上下文打包。"""

    def __init__(
        self,
        *,
        sources: Mapping[SourceType, SourceRetrievalAdapter],
        methods: Mapping[RetrievalMethod, RetrievalMethodPort],
        coverage_probes: Mapping[SourceType, SourceCoverageProbe] | None = None,
        access_revalidators: Mapping[SourceType, SourceAccessRevalidator] | None = None,
        rrf_k: int,
    ) -> None:
        self._sources = dict(sources)
        self._methods: dict[RetrievalMethod, RetrievalMethodPort] = dict(methods)
        self._coverage_probes = dict(coverage_probes or {})
        self._access_revalidators = dict(access_revalidators or {})
        for source_type, probe in self._coverage_probes.items():
            if not isinstance(source_type, SourceType):
                raise ValueError("coverage probe keys must be SourceType values")
            if getattr(probe, "source_type", None) is not source_type:
                raise ValueError("coverage probes must declare the matching SourceType")
        for source_type, revalidator in self._access_revalidators.items():
            if not isinstance(source_type, SourceType):
                raise ValueError("source access revalidator keys must be SourceType values")
            if getattr(revalidator, "source_type", None) is not source_type:
                raise ValueError("source access revalidators must declare the matching SourceType")
        self._rrf_k = rrf_k

    def execute(
        self,
        *,
        source_plans: Sequence[SourcePlan],
        queries: Sequence[str],
        candidate_limit: int,
        recovery: RetrievalRecoveryController,
        retrieval_data_version_id: str | None,
        expected_source_snapshots: Mapping[SourceType, SourceAccess] | None = None,
    ) -> SourceExecutionState:
        state = SourceExecutionState()
        frozen_snapshots = dict(expected_source_snapshots or {})
        for source_plan in source_plans:
            checkpoint()
            source_type = source_plan.source_type
            if not source_plan.should_run:
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.DISABLED,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code=source_plan.skipped_reason,
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    SourceAvailability.DISABLED,
                )
                continue

            adapter = self._sources.get(source_type)
            if adapter is None:
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.NOT_IMPLEMENTED,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code="source_adapter_not_installed",
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    SourceAvailability.NOT_IMPLEMENTED,
                )
                continue

            access_result = recovery.run(
                ("source_availability", source_type.value),
                lambda: _open_source_access(
                    adapter,
                    expected_source_type=source_type,
                    source_filter=source_plan.source_filter,
                ),
            )
            if not access_result.succeeded:
                availability = SourceAvailability.UNAVAILABLE
                reason_code = (
                    "source_availability_circuit_open"
                    if access_result.circuit_open
                    else f"source_availability_failed:{type(access_result.error).__name__}"
                )
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=availability,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code=reason_code,
                )
                state.diagnostics.append(f"{reason_code}:{source_type.value}")
                state.blocked |= _blocks_context(source_type, source_plan.dependency, availability)
                continue

            access = access_result.value
            if not isinstance(access, SourceAccess):
                raise TypeError("source access must be a SourceAccess")
            state.source_access[source_type] = access
            expected_snapshot = frozen_snapshots.get(source_type)
            if expected_snapshot is not None and not _source_access_matches_expected(
                access,
                expected_snapshot,
            ):
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.BLOCKED,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code="history_scope_snapshot_stale",
                    source_snapshot_id=expected_snapshot.source_snapshot_id,
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    SourceAvailability.BLOCKED,
                )
                continue
            availability = access.availability
            if availability is not SourceAvailability.READY:
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=availability,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code=access.reason_code or f"source_{availability.value}",
                    source_snapshot_id=access.source_snapshot_id,
                    coverage_facts=access.coverage_facts,
                )
                state.blocked |= _blocks_context(source_type, source_plan.dependency, availability)
                continue

        # 在搜索前以及即将发布结果前分别检查派生目录。两次观察可防止 Outbox 消费者在
        # 搜索期间填补同版本缺口后，追溯性地把该旧搜索变成完整无匹配结果。
            initial_coverage_limitation = self._check_source_coverage(
                source_type=source_type,
                access=access,
                recovery=recovery,
                retrieval_data_version_id=retrieval_data_version_id,
            )
            (
                source_candidates,
                source_method_outcomes,
                source_lane_candidates,
            ) = self._search_source(
                queries=queries,
                source_plan=source_plan,
                catalog_source_filters=_catalog_source_filters(adapter, access),
                candidate_limit=candidate_limit,
                recovery=recovery,
                retrieval_data_version_id=retrieval_data_version_id,
            )
            state.method_outcomes.extend(source_method_outcomes)
            state.lane_candidates.extend(source_lane_candidates)
            state.fusion_candidates.extend(source_candidates)
            if any(outcome.status is MethodRunStatus.FAILED for outcome in source_method_outcomes):
                state.diagnostics.append(f"retrieval_method_failed:{source_type.value}")
            state.candidates_by_source[source_type] = source_candidates

            final_access, revalidation_limitation = self._revalidate_source_access(
                source_type=source_type,
                access=access,
                recovery=recovery,
            )
            if revalidation_limitation is not None:
                state.coverage_limitations.append(revalidation_limitation)
                state.candidates_by_source[source_type] = []
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.UNAVAILABLE,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code=revalidation_limitation.reason_code,
                    source_snapshot_id=access.source_snapshot_id,
                    coverage_facts=access.coverage_facts,
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    SourceAvailability.UNAVAILABLE,
                )
                continue
            assert isinstance(final_access, SourceAccess)
            state.source_access[source_type] = final_access
            if expected_snapshot is not None and not _source_access_matches_expected(
                final_access,
                expected_snapshot,
            ):
                state.candidates_by_source[source_type] = []
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=SourceAvailability.BLOCKED,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code="history_scope_snapshot_stale",
                    source_snapshot_id=expected_snapshot.source_snapshot_id,
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    SourceAvailability.BLOCKED,
                )
                continue
            if final_access.availability is not SourceAvailability.READY:
                state.candidates_by_source[source_type] = []
                state.source_outcomes[source_type] = SourceOutcome(
                    source_type=source_type,
                    availability=final_access.availability,
                    retrieval=SourceRetrievalStatus.NOT_RUN,
                    dependency=source_plan.dependency,
                    reason_code=(
                        final_access.reason_code
                        or f"source_{final_access.availability.value}"
                    ),
                    source_snapshot_id=final_access.source_snapshot_id,
                    coverage_facts=final_access.coverage_facts,
                )
                state.blocked |= _blocks_context(
                    source_type,
                    source_plan.dependency,
                    final_access.availability,
                )
                continue

            final_coverage_limitation = self._check_source_coverage(
                source_type=source_type,
                access=final_access,
                recovery=recovery,
                retrieval_data_version_id=retrieval_data_version_id,
            )
            source_limitations = [
                limitation
                for limitation in (
                    initial_coverage_limitation,
                    _source_access_change_limitation(source_type, access, final_access),
                    final_coverage_limitation,
                )
                if limitation is not None
            ]
            state.coverage_limitations.extend(source_limitations)
            source_outcome = _source_outcome_after_search(
                source_plan=source_plan,
                candidates=source_candidates,
                method_outcomes=source_method_outcomes,
                source_snapshot_id=final_access.source_snapshot_id,
                coverage_facts=final_access.coverage_facts,
            )
            if (
                source_limitations
                and source_outcome.retrieval
                in {SourceRetrievalStatus.MATCHED, SourceRetrievalStatus.NO_MATCH}
            ):
                source_outcome = _with_source_retrieval_outcome(
                    source_outcome,
                    retrieval=SourceRetrievalStatus.PARTIAL,
                    reason_code=source_limitations[0].reason_code,
                )
            state.source_outcomes[source_type] = source_outcome
            if source_outcome.retrieval is SourceRetrievalStatus.FAILED:
                state.diagnostics.append(f"retrieval_methods_failed:{source_type.value}")
        return state

    def _check_source_coverage(
        self,
        *,
        source_type: SourceType,
        access: SourceAccess,
        recovery: RetrievalRecoveryController,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None:
        """运行可选的跨存储覆盖证明，不扩大 Source 作用域。"""

        probe = self._coverage_probes.get(source_type)
        if probe is None:
            return None
        result = recovery.run(
            ("source_coverage", source_type.value),
            lambda: _check_coverage_probe(
                probe,
                source_type=source_type,
                access=access,
                retrieval_data_version_id=retrieval_data_version_id,
            ),
        )
        if not result.succeeded:
            return ContextCoverageLimitation(
                source_type=source_type,
                reason_code="source_index_coverage_check_unavailable",
            )
        return result.value

    def _revalidate_source_access(
        self,
        *,
        source_type: SourceType,
        access: SourceAccess,
        recovery: RetrievalRecoveryController,
    ) -> tuple[SourceAccess | None, ContextCoverageLimitation | None]:
        """仅当组合配置要求时重新检查来源。

        重新校验器绝不会收到原始用户查询文本，且必须保留原始受信过滤器。最终检查失败时
        会丢弃候选：报告上下文不完整，比发布来源新鲜度已未知的数据更安全。
        """

        revalidator = self._access_revalidators.get(source_type)
        if revalidator is None:
            return access, None
        result = recovery.run(
            ("source_access_revalidation", source_type.value),
            lambda: _validate_revalidated_source_access(
                revalidator,
                source_type=source_type,
                access=access,
            ),
        )
        if not result.succeeded:
            return None, ContextCoverageLimitation(
                source_type=source_type,
                reason_code="source_access_revalidation_unavailable",
            )
        return result.value, None

    @measure("source_fetch_verify")
    def fetch_and_verify(
        self,
        *,
        request: RetrievalRequest,
        source_plans: Sequence[SourcePlan],
        state: SourceExecutionState,
        recovery: RetrievalRecoveryController,
    ) -> dict[SourceType, list[tuple[RetrievalCandidate, SourceUnit]]]:
        resolved: dict[SourceType, list[tuple[RetrievalCandidate, SourceUnit]]] = {}
        plan_by_source = {source_plan.source_type: source_plan for source_plan in source_plans}
        for source_type, candidates in state.candidates_by_source.items():
            checkpoint()
            plan_by_source[source_type]
            adapter = self._sources[source_type]
            access = state.source_access[source_type]
            requested = [
                candidate
                for candidate in candidates
                if candidate.ref not in request.excluded_direct_refs
            ]
            if not requested:
                continue
            # 同一精确来源可参与多个 query 的独立评分，权威正文只需读取一次。
            refs = list(dict.fromkeys(candidate.ref for candidate in requested))
            result = recovery.run(
                ("source_fetch", source_type.value),
                lambda: tuple(adapter.fetch_units(access, refs)),
            )
            if not result.succeeded:
                state.diagnostics.append(
                    (
                        f"source_fetch_circuit_open:{source_type.value}"
                        if result.circuit_open
                        else f"source_fetch_failed:{source_type.value}:{type(result.error).__name__}"
                    )
                )
                _mark_source_retrieval_outcome(
                    state.source_outcomes,
                    source_type=source_type,
                    retrieval=SourceRetrievalStatus.FAILED,
                    reason_code="source_fetch_failed",
                )
                state.verification_drops.append(
                    ContextVerificationDrop(
                        source_type=source_type,
                        reason=ContextVerificationDropReason.SOURCE_FETCH_FAILED,
                        count=len(requested),
                    )
                )
                continue
            fetched = result.value
            assert isinstance(fetched, tuple)
            by_ref = {unit.ref: unit for unit in fetched}
            verified: list[tuple[RetrievalCandidate, SourceUnit]] = []
            drop_counts: dict[ContextVerificationDropReason, int] = {}
            for candidate in requested:
                unit = by_ref.get(candidate.ref)
                if unit is None:
                    state.diagnostics.append(f"source_unit_missing:{source_type.value}")
                    _increment_drop_count(
                        drop_counts,
                        ContextVerificationDropReason.SOURCE_UNIT_MISSING,
                    )
                    continue
                if not unit.retrievable:
                    state.diagnostics.append(f"source_unit_not_retrievable:{source_type.value}")
                    _increment_drop_count(
                        drop_counts,
                        ContextVerificationDropReason.SOURCE_UNIT_NOT_RETRIEVABLE,
                    )
                    continue
                if unit.ref != candidate.ref:
                    state.diagnostics.append(f"source_unit_ref_mismatch:{source_type.value}")
                    _increment_drop_count(
                        drop_counts,
                        ContextVerificationDropReason.SOURCE_UNIT_REF_MISMATCH,
                    )
                    continue
                verified.append((candidate, unit))
            state.verification_drops.extend(
                ContextVerificationDrop(source_type=source_type, reason=reason, count=count)
                for reason, count in sorted(drop_counts.items(), key=lambda item: item[0].value)
            )
            if len(verified) != len(requested):
                _mark_source_retrieval_outcome(
                    state.source_outcomes,
                    source_type=source_type,
                    retrieval=SourceRetrievalStatus.PARTIAL,
                    reason_code="source_candidate_verification_partial",
                )
            if verified:
                resolved[source_type] = verified
        return resolved

    def revalidate_expected_snapshots_after_fetch(
        self,
        *,
        expected_source_snapshots: Mapping[SourceType, SourceAccess],
        state: SourceExecutionState,
        resolved_by_source: dict[
            SourceType,
            list[tuple[RetrievalCandidate, SourceUnit]],
        ],
        recovery: RetrievalRecoveryController,
    ) -> None:
        """封闭读取权威正文期间发生的生命周期竞争。"""

        for source_type, expected in expected_source_snapshots.items():
            access = state.source_access.get(source_type)
            revalidator = self._access_revalidators.get(source_type)
            if access is None or revalidator is None:
                _mark_expected_snapshot_stale(
                    state,
                    resolved_by_source,
                    source_type=source_type,
                    expected=expected,
                )
                continue
            result = recovery.run(
                ("source_access_post_fetch_revalidation", source_type.value),
                lambda: _validate_revalidated_source_access(
                    revalidator,
                    source_type=source_type,
                    access=access,
                ),
            )
            if (
                not result.succeeded
                or not isinstance(result.value, SourceAccess)
                or not _source_access_matches_expected(result.value, expected)
            ):
                _mark_expected_snapshot_stale(
                    state,
                    resolved_by_source,
                    source_type=source_type,
                    expected=expected,
                )
                continue
            state.source_access[source_type] = result.value

    def _search_source(
        self,
        *,
        queries: Sequence[str],
        source_plan: SourcePlan,
        catalog_source_filters: Sequence[SourceFilter],
        candidate_limit: int,
        recovery: RetrievalRecoveryController,
        retrieval_data_version_id: str | None,
    ) -> tuple[
        list[RetrievalCandidate],
        list[MethodOutcome],
        list[RetrievalCandidate],
    ]:
        outcomes: list[MethodOutcome] = []
        lane_candidates: list[RetrievalCandidate] = []
        fused_candidates_by_query: list[list[RetrievalCandidate]] = []
        for query_index, query in enumerate(queries):
            checkpoint()
            query_candidates: list[RetrievalCandidate] = []
            failed_primary_methods: list[RetrievalMethod] = []
            primary_method_succeeded = False
            for method_name in source_plan.methods:
                hits, outcome = self._run_method(
                    method_name=method_name,
                    query=query,
                    query_index=query_index,
                    source_plan=source_plan,
                    catalog_source_filters=catalog_source_filters,
                    candidate_limit=candidate_limit,
                    recovery=recovery,
                    retrieval_data_version_id=retrieval_data_version_id,
                )
                outcomes.append(outcome)
                query_candidates.extend(hits)
                lane_candidates.extend(hits)
                if outcome.status is MethodRunStatus.FAILED:
                    failed_primary_methods.append(method_name)
                else:
                    primary_method_succeeded = True

            bm25_failed = RetrievalMethod.BM25 in failed_primary_methods
            if (
                failed_primary_methods
                and RetrievalMethod.BM25 not in source_plan.methods
            ):
                # 旧调用方仍可只声明语义方法。只在其中至少一项技术失败时，才补跑未被
                # 声明为正式 lane 的 BM25；一个成功但无命中的主路径仍是有效 no-match。
                bm25_hits, bm25_outcome = self._run_method(
                    method_name=RetrievalMethod.BM25,
                    query=query,
                    query_index=query_index,
                    source_plan=source_plan,
                    catalog_source_filters=catalog_source_filters,
                    candidate_limit=candidate_limit,
                    degraded_from=tuple(failed_primary_methods),
                    attempt=2,
                    recovery=recovery,
                    retrieval_data_version_id=retrieval_data_version_id,
                )
                outcomes.append(bm25_outcome)
                query_candidates.extend(bm25_hits)
                lane_candidates.extend(bm25_hits)
                bm25_failed = bm25_outcome.status is MethodRunStatus.FAILED

            # 字面兜底只处理“所有正式 lane 与 BM25 都不可用”的基础设施故障。只要某个
            # lane 成功完成搜索，即使没有命中，也不能把有效 no-match 改写成另一种查询。
            if bm25_failed and not primary_method_succeeded:
                literal_hits, literal_outcome = self._run_method(
                    method_name=RetrievalMethod.LITERAL_BOOLEAN,
                    query=query,
                    query_index=query_index,
                    source_plan=source_plan,
                    catalog_source_filters=catalog_source_filters,
                    candidate_limit=candidate_limit,
                    degraded_from=(RetrievalMethod.BM25,),
                    attempt=3,
                    recovery=recovery,
                    retrieval_data_version_id=retrieval_data_version_id,
                )
                outcomes.append(literal_outcome)
                query_candidates.extend(literal_hits)
                lane_candidates.extend(literal_hits)

            fused_candidates_by_query.append(
                list(fuse_candidates_by_rrf(query_candidates, rrf_k=self._rrf_k))
            )

        return (
            round_robin_query_candidates(fused_candidates_by_query, candidate_limit),
            outcomes,
            lane_candidates,
        )

    def _run_method(
        self,
        *,
        method_name: RetrievalMethod,
        query: str,
        query_index: int,
        source_plan: SourcePlan,
        catalog_source_filters: Sequence[SourceFilter],
        candidate_limit: int,
        degraded_from: tuple[RetrievalMethod, ...] = (),
        attempt: int = 1,
        recovery: RetrievalRecoveryController,
        retrieval_data_version_id: str | None,
    ) -> tuple[list[RetrievalCandidate], MethodOutcome]:
        method = self._methods.get(method_name)
        if method is None:
            return [], MethodOutcome(
                method=method_name,
                source_type=source_plan.source_type,
                query_index=query_index,
                status=MethodRunStatus.FAILED,
                reason_code="retrieval_method_not_installed",
                degraded_from=degraded_from,
                attempt=attempt,
            )
        def search_authorized_filters() -> tuple[RetrievalCandidate, ...]:
            hits: list[RetrievalCandidate] = []
            for catalog_filter in catalog_source_filters:
                checkpoint()
                with measure(f"recall_{method_name.value}_total"):
                    found = method.search(
                        query,
                        query_index=query_index,
                        source_filter=catalog_filter,
                        limit=candidate_limit,
                        retrieval_data_version_id=retrieval_data_version_id,
                    )
                checkpoint()
                hits.extend(found)
            execution = current_execution()
            if execution is not None:
                execution.increment(f"recall_{method_name.value}_candidates", len(hits))
            return tuple(hits)

        result = recovery.run(
            ("method", source_plan.source_type.value, method_name.value),
            search_authorized_filters,
        )
        if not result.succeeded:
            reason_code, failure_stage = _method_failure_diagnostic(
                result.error,
                circuit_open=result.circuit_open,
            )
            return [], MethodOutcome(
                method=method_name,
                source_type=source_plan.source_type,
                query_index=query_index,
                status=MethodRunStatus.FAILED,
                reason_code=reason_code,
                failure_stage=failure_stage,
                degraded_from=degraded_from,
                attempt=attempt,
                infrastructure_attempts=max(1, result.attempts),
                attempt_failures=_method_attempt_failures(result.attempt_errors),
            )
        hits = result.value
        assert isinstance(hits, tuple)
        valid_hits = [
            hit
            for hit in hits
            if hit.method is method_name
            and hit.query_index == query_index
            and hit.ref.source_type is source_plan.source_type
        ]
        if len(valid_hits) != len(hits):
            return [], MethodOutcome(
                method=method_name,
                source_type=source_plan.source_type,
                query_index=query_index,
                status=MethodRunStatus.FAILED,
                candidate_count=len(valid_hits),
                reason_code="method_returned_invalid_candidate_boundary",
                degraded_from=degraded_from,
                attempt=attempt,
                infrastructure_attempts=result.attempts,
                attempt_failures=_method_attempt_failures(result.attempt_errors),
            )
        valid_hits = _merge_method_hits(valid_hits, limit=candidate_limit)
        return valid_hits, MethodOutcome(
            method=method_name,
            source_type=source_plan.source_type,
            query_index=query_index,
            status=MethodRunStatus.DEGRADED if degraded_from else MethodRunStatus.USED,
            candidate_count=len(valid_hits),
            degraded_from=degraded_from,
            attempt=attempt,
            infrastructure_attempts=result.attempts,
            attempt_failures=_method_attempt_failures(result.attempt_errors),
        )


def _method_failure_diagnostic(
    error: Exception | None,
    *,
    circuit_open: bool,
) -> tuple[str, str | None]:
    """保留 encoder/method 已生成的安全分类，而不是统一吞成异常类名。"""

    if circuit_open:
        return "method_circuit_open", "recovery"
    safe_error_code = getattr(error, "safe_error_code", None)
    failure_stage = getattr(error, "stage", None)
    reason_code = (
        safe_error_code
        if isinstance(safe_error_code, str) and safe_error_code.strip()
        else f"method_exception:{type(error).__name__}"
    )
    return (
        reason_code,
        failure_stage
        if isinstance(failure_stage, str) and failure_stage.strip()
        else "method_search",
    )


def _method_attempt_failures(
    errors: Sequence[Exception],
) -> tuple[tuple[int, str, str | None], ...]:
    return tuple(
        (attempt, *_method_failure_diagnostic(error, circuit_open=False))
        for attempt, error in enumerate(errors, start=1)
    )


def _catalog_source_filters(
    adapter: SourceRetrievalAdapter,
    access: SourceAccess,
) -> tuple[SourceFilter, ...]:
    """返回派生存储过滤器，不削弱已打开许可。

    大多数 Source 会持久化与其授权相同的结构作用域。项目 Document 适配器是刻意设置的
    例外：Session 挂载拥有访问权，而共享项目索引存储一个 ``doc_id`` 作用域的 Unit。
    """

    resolver = getattr(adapter, "catalog_source_filters", None)
    if not callable(resolver):
        return (access.source_filter,)
    filters = tuple(resolver(access))
    if any(
        not isinstance(source_filter, SourceFilter)
        or source_filter.source_type is not access.source_type
        for source_filter in filters
    ):
        raise ValueError("catalog source filters must retain the SourceType")
    return filters


def _merge_method_hits(
    hits: Sequence[RetrievalCandidate],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    """将每份文档的方法排序合并为一个有界 Source 排序。"""

    best_by_ref: dict[SourceUnitRef, RetrievalCandidate] = {}
    for hit in hits:
        current = best_by_ref.get(hit.ref)
        if current is None or (hit.raw_score, -hit.rank) > (
            current.raw_score,
            -current.rank,
        ):
            best_by_ref[hit.ref] = hit
    ordered = sorted(
        best_by_ref.values(),
        key=lambda hit: (
            -hit.raw_score,
            hit.ref.source_unit_id,
            hit.ref.source_revision,
        ),
    )[:limit]
    return [
        RetrievalCandidate(
            ref=hit.ref,
            method=hit.method,
            query_index=hit.query_index,
            rank=rank,
            raw_score=hit.raw_score,
            fusion_score=hit.fusion_score,
        )
        for rank, hit in enumerate(ordered, start=1)
    ]


def _source_outcome_after_search(
    *,
    source_plan: SourcePlan,
    candidates: Sequence[RetrievalCandidate],
    method_outcomes: Sequence[MethodOutcome],
    source_snapshot_id: str | None,
    coverage_facts: Mapping[str, str],
) -> SourceOutcome:
    if candidates:
        retrieval = SourceRetrievalStatus.MATCHED
        reason_code = None
    elif method_outcomes and all(
        outcome.status is MethodRunStatus.FAILED for outcome in method_outcomes
    ):
        retrieval = SourceRetrievalStatus.FAILED
        reason_code = "all_retrieval_methods_failed_or_rejected"
    else:
        retrieval = SourceRetrievalStatus.NO_MATCH
        reason_code = None
    return SourceOutcome(
        source_type=source_plan.source_type,
        availability=SourceAvailability.READY,
        retrieval=retrieval,
        dependency=source_plan.dependency,
        reason_code=reason_code,
        source_snapshot_id=source_snapshot_id,
        coverage_facts=coverage_facts,
    )


def _blocks_context(
    source_type: SourceType,
    dependency: SourceDependency,
    availability: SourceAvailability,
) -> bool:
    if source_type is SourceType.CURRENT_SESSION and availability not in {
        SourceAvailability.READY,
        SourceAvailability.EMPTY,
    }:
        return True
    return dependency is SourceDependency.REQUIRED and availability not in {
        SourceAvailability.READY,
        SourceAvailability.EMPTY,
    }


def _mark_source_retrieval_outcome(
    source_outcomes: dict[SourceType, SourceOutcome],
    *,
    source_type: SourceType,
    retrieval: SourceRetrievalStatus,
    reason_code: str,
) -> None:
    previous = source_outcomes[source_type]
    source_outcomes[source_type] = _with_source_retrieval_outcome(
        previous,
        retrieval=retrieval,
        reason_code=reason_code,
    )


def _with_source_retrieval_outcome(
    previous: SourceOutcome,
    *,
    retrieval: SourceRetrievalStatus,
    reason_code: str,
) -> SourceOutcome:
    return SourceOutcome(
        source_type=previous.source_type,
        availability=previous.availability,
        retrieval=retrieval,
        dependency=previous.dependency,
        reason_code=reason_code,
        source_snapshot_id=previous.source_snapshot_id,
        lane_outcomes=previous.lane_outcomes,
        coverage_facts=previous.coverage_facts,
    )


def _check_coverage_probe(
    probe: SourceCoverageProbe,
    *,
    source_type: SourceType,
    access: SourceAccess,
    retrieval_data_version_id: str | None,
) -> ContextCoverageLimitation | None:
    """在探针结果影响公开结果前对其进行校验。"""

    if getattr(probe, "source_type", None) is not source_type:
        raise ValueError("coverage probe source type mismatch")
    limitation = probe.check_source_coverage(
        access,
        retrieval_data_version_id=retrieval_data_version_id,
    )
    if limitation is None:
        return None
    if not isinstance(limitation, ContextCoverageLimitation):
        raise TypeError("coverage probe must return ContextCoverageLimitation or None")
    if limitation.source_type is not source_type:
        raise ValueError("coverage limitation must retain its SourceType")
    return limitation


def _validate_revalidated_source_access(
    revalidator: SourceAccessRevalidator,
    *,
    source_type: SourceType,
    access: SourceAccess,
) -> SourceAccess:
    """在最终来源许可替换原许可前对其进行校验。"""

    if getattr(revalidator, "source_type", None) is not source_type:
        raise ValueError("source access revalidator source type mismatch")
    final_access = revalidator.revalidate_retrieval_access(access)
    if not isinstance(final_access, SourceAccess):
        raise TypeError("source access revalidator must return a SourceAccess")
    if (
        final_access.source_type is not source_type
        or final_access.source_filter != access.source_filter
    ):
        raise ValueError("source access revalidator must preserve the trusted source scope")
    return final_access


def _source_access_change_limitation(
    source_type: SourceType,
    initial_access: SourceAccess,
    final_access: SourceAccess,
) -> ContextCoverageLimitation | None:
    """来源 revision 在读取期间变化时，将搜索标记为部分完成。

    快照 ID 是审计标识符，在重新打开时可以合理变化；只有来源所有的 revision 清单才能
    确定重新校验前搜索的材料是否仍能代表最终来源状态。
    """

    if initial_access.source_revision_map == final_access.source_revision_map:
        return None
    return ContextCoverageLimitation(
        source_type=source_type,
        reason_code=f"{source_type.value}_source_changed_during_retrieval",
    )


def _source_access_matches_expected(
    observed: SourceAccess,
    expected: SourceAccess,
) -> bool:
    """只比较不可变的 Source 快照权威事实。"""

    return (
        observed.source_type is expected.source_type
        and observed.source_filter == expected.source_filter
        and observed.availability is expected.availability
        and observed.source_snapshot_id == expected.source_snapshot_id
        and observed.source_revision_map == expected.source_revision_map
    )


def _mark_expected_snapshot_stale(
    state: SourceExecutionState,
    resolved_by_source: dict[
        SourceType,
        list[tuple[RetrievalCandidate, SourceUnit]],
    ],
    *,
    source_type: SourceType,
    expected: SourceAccess,
) -> None:
    outcome = state.source_outcomes.get(source_type)
    dependency = (
        outcome.dependency if outcome is not None else SourceDependency.OPTIONAL
    )
    state.candidates_by_source[source_type] = []
    resolved_by_source.pop(source_type, None)
    state.source_outcomes[source_type] = SourceOutcome(
        source_type=source_type,
        availability=SourceAvailability.BLOCKED,
        retrieval=SourceRetrievalStatus.NOT_RUN,
        dependency=dependency,
        reason_code="history_scope_snapshot_stale",
        source_snapshot_id=expected.source_snapshot_id,
    )
    state.blocked |= _blocks_context(
        source_type,
        dependency,
        SourceAvailability.BLOCKED,
    )


def _open_source_access(
    adapter: SourceRetrievalAdapter,
    *,
    expected_source_type: SourceType,
    source_filter: SourceFilter,
) -> SourceAccess:
    """在恢复边界内校验适配器的读取许可。"""

    access = adapter.open_retrieval_access(source_filter)
    if not isinstance(access, SourceAccess):
        raise TypeError("source access must be a SourceAccess")
    if access.source_type is not expected_source_type or access.source_filter != source_filter:
        raise ValueError("source access must preserve the trusted source scope")
    return access


def _increment_drop_count(
    counts: dict[ContextVerificationDropReason, int],
    reason: ContextVerificationDropReason,
) -> None:
    counts[reason] = counts.get(reason, 0) + 1
