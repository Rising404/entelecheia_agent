"""在线 Retrieval 的稳定服务门面与最终上下文 owner。

Service 接收已冻结的请求和显式注入的 ports，协调 query 准入、generation 捕获、Source access、
方法执行/降级、权威内容复核、可选重排、配额选择与预算打包，最终返回一个带来源状态和缺口的
``RetrievedContext``。具体阶段算法位于 ``orchestration``，本模块只维持公开读取生命周期及其
跨阶段不变量。

它不直接拥有 SQLite schema、Source 正文、模型调用、工具注册或 L0/L1/L2 状态推进。Tool 与
Runtime 应调用这个门面或其有界 wrapper，而不是重新排列内部阶段。作为所有在线实现的稳定
入口，它保留在包根。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from math import ceil

from .contracts import (
    ContextCoverageLimitation,
    ContextStatus,
    LongTermMemoryWriteGuard,
    LongTermMemoryWriteGuardStatus,
    RetrievalBudget,
    RetrievedContext,
    RetrievalMethod,
    RetrievalRequest,
    RerankerRunStatus,
    SourceAccess,
    SourceAvailability,
    SourceDependency,
    SourceFilter,
    SourceOutcome,
    SourcePlan,
    SourceRetrievalStatus,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from .orchestration.packing import ContextPacker
from .orchestration.file_query_batch import execute_file_query_batch
from .orchestration.source_execution import SourceExecution
from .ports import (
    RerankerPort,
    RetrievalCancelled,
    RetrievalDataVersionProvider,
    RetrievalMethodPort,
    RetrievalPolicyPort,
    SourceAccessRevalidator,
    SourceCoverageProbe,
    SourceRetrievalAdapter,
    TokenEstimator,
)
from .query_guard import QueryGuard, QueryGuardError
from .orchestration.recovery import RetrievalRecoveryController
from .orchestration.reranking import rerank_verified_candidates
from .orchestration.selection import SourceSelectionQuotas


def _default_token_estimator(text: str) -> int:
    """保守的临时估算器；模型最终 token 计数仍由外部负责。"""

    return max(1, ceil(len(text) / 4))


def _hint_names(proposal: object) -> list[str]:
    """列出被拒提案指向的来源，但不包含其内部信息。"""

    hints = getattr(proposal, "source_hints", None) or ()
    return sorted(getattr(hint, "value", str(hint)) for hint in hints)


class RetrievalService:
    """运行受信检索，然后返回已验证且受预算约束的上下文。

    Source 本地执行、方法回退和内容验证归 ``SourceExecution`` 所有。本门面拥有公开请求
    生命周期、跨来源汇合、最终预算打包和 ``RetrievedContext`` 契约，但不取得底层数据所有权。
    """

    def __init__(
        self,
        *,
        policy: RetrievalPolicyPort,
        query_guard: QueryGuard,
        sources: Mapping[SourceType, SourceRetrievalAdapter],
        methods: Mapping[RetrievalMethod, RetrievalMethodPort],
        source_coverage_probes: Mapping[SourceType, SourceCoverageProbe] | None = None,
        source_access_revalidators: Mapping[SourceType, SourceAccessRevalidator] | None = None,
        readonly_sources: Mapping[SourceType, SourceRetrievalAdapter] | None = None,
        readonly_source_coverage_probes: (
            Mapping[SourceType, SourceCoverageProbe] | None
        ) = None,
        readonly_source_access_revalidators: (
            Mapping[SourceType, SourceAccessRevalidator] | None
        ) = None,
        data_version_provider: RetrievalDataVersionProvider | None = None,
        token_estimator: TokenEstimator = _default_token_estimator,
        rrf_k: int = 60,
        max_infrastructure_attempts: int = 3,
        source_selection_quotas: SourceSelectionQuotas | None = None,
        encoder_fingerprint: str = "",
        reranker: RerankerPort | None = None,
        reranker_candidate_limit_per_source: int = 32,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if max_infrastructure_attempts <= 0:
            raise ValueError("max_infrastructure_attempts must be greater than zero")
        if reranker_candidate_limit_per_source <= 1:
            raise ValueError("reranker_candidate_limit_per_source must be greater than one")
        quotas = source_selection_quotas or SourceSelectionQuotas()
        self._policy = policy
        self._query_guard = query_guard
        # 盖印到每项结果上，使模型或索引版本变化后，召回仍可归因于为其排序的编码器。
        self._encoder_fingerprint = encoder_fingerprint
        self._reranker = reranker
        self._reranker_candidate_limit_per_source = reranker_candidate_limit_per_source
        self._reranker_fingerprint = _reranker_fingerprint(reranker)
        # 保留这些属性作为现有 Foundation 连线和测试的兼容级检查点；执行现已委托给下方
        # 职责集中的内部组件。
        self._sources = dict(sources)
        self._methods = dict(methods)
        self._token_estimator = token_estimator
        self._rrf_k = rrf_k
        self._source_selection_quotas = quotas
        self._data_version_provider = data_version_provider
        self._source_execution = SourceExecution(
            sources=self._sources,
            methods=self._methods,
            coverage_probes=source_coverage_probes,
            access_revalidators=source_access_revalidators,
            rrf_k=self._rrf_k,
        )
        # File Corpus 候选读取必须真正无副作用。因此它会获得单独组合的来源执行路径，
        # 其文档适配器执行只读新鲜度检查，而不写入持久检索审计快照。这里刻意不回退到
        # 普通执行路径：请求此契约的调用方若未由组合层安装该路径，就必须以关闭方式失败。
        self._readonly_source_execution = (
            SourceExecution(
                sources=readonly_sources,
                methods=self._methods,
                coverage_probes=readonly_source_coverage_probes,
                access_revalidators=readonly_source_access_revalidators,
                rrf_k=self._rrf_k,
            )
            if readonly_sources is not None
            else None
        )
        self._context_packer = ContextPacker(
            token_estimator=self._token_estimator,
            source_selection_quotas=self._source_selection_quotas,
        )
        self._max_infrastructure_attempts = max_infrastructure_attempts

    def reread_authoritative_units(
        self,
        *,
        source_filter: SourceFilter,
        refs: Sequence[SourceUnitRef],
    ) -> tuple[SourceUnit, ...]:
        """从对应 Source 重新读取精确指针，不查询排序数据。

        这是此前返回的检索句柄跨越后续持久边界时使用的窄权威查找。派生目录可以定位指针，
        但只有 Source 适配器可以提供正文。读取前、读取中或读取后若生命周期或 revision
        发生变化，就不返回任何 Unit，使调用方可以以关闭方式失败。
        """

        if not isinstance(source_filter, SourceFilter):
            raise TypeError("source_filter must be a SourceFilter")
        normalized_refs = tuple(refs)
        if not normalized_refs:
            return ()
        if any(
            not isinstance(ref, SourceUnitRef)
            or ref.source_type is not source_filter.source_type
            for ref in normalized_refs
        ):
            raise ValueError("authority refs must match the trusted SourceFilter")
        if len(set(normalized_refs)) != len(normalized_refs):
            raise ValueError("authority refs must be unique")

        adapter = self._sources.get(source_filter.source_type)
        if adapter is None:
            return ()
        access = adapter.open_retrieval_access(source_filter)
        if (
            not isinstance(access, SourceAccess)
            or access.source_type is not source_filter.source_type
            or access.source_filter != source_filter
            or access.availability is not SourceAvailability.READY
        ):
            return ()
        fetched = tuple(adapter.fetch_units(access, normalized_refs))
        if any(not isinstance(unit, SourceUnit) for unit in fetched):
            return ()

        revalidate = getattr(adapter, "revalidate_retrieval_access", None)
        if not callable(revalidate):
            return ()
        final_access = revalidate(access)
        if (
            not isinstance(final_access, SourceAccess)
            or final_access.source_type is not source_filter.source_type
            or final_access.source_filter != source_filter
            or final_access.availability is not SourceAvailability.READY
            or final_access.source_revision_map != access.source_revision_map
        ):
            return ()

        by_ref = {unit.ref: unit for unit in fetched if unit.retrievable}
        verified: list[SourceUnit] = []
        for ref in normalized_refs:
            unit = by_ref.get(ref)
            if (
                unit is None
                or unit.ref != ref
                or hashlib.sha256(unit.content.encode("utf-8")).hexdigest()
                != ref.indexed_content_hash
            ):
                return ()
            verified.append(unit)
        return tuple(verified)

    def retrieve_context(
        self,
        request: RetrievalRequest,
        budget: RetrievalBudget,
    ) -> RetrievedContext:
        """执行检索，并同时记录请求内容和返回内容。

        此处采用包装而非内联埋点有两个原因：被阻止路径会提前返回，而守卫通过抛出异常
        拒绝。被拒提案更值得保留——运行时已经记录守卫作出拒绝，却从未记录它拒绝了什么。
        """

        from ..trajectory import record_rejected_output, record_retrieval

        started = time.monotonic()
        proposal = getattr(request, "query_proposal", None)
        queries = tuple(getattr(proposal, "queries", ()) or ())
        try:
            result = self._retrieve_context(request, budget)
        except QueryGuardError as exc:
            record_rejected_output(
                stage=f"query_guard:{request.model_call_purpose}",
                rejected=json.dumps(
                    {"queries": list(queries), "source_hints": _hint_names(proposal)},
                    ensure_ascii=False,
                ),
                reason_code=str(getattr(exc, "code", "query_guard_rejected")),
            )
            raise
        record_retrieval(
            queries=queries,
            source_scope=request.model_call_purpose,
            evidence=[
                {
                    "source_type": item.ref.source_type.value,
                    "source_unit_id": item.ref.source_unit_id,
                    "fused_rank": item.fused_rank,
                    "estimated_tokens": item.estimated_tokens,
                }
                for item in result.items
            ],
            outcome=result.status.value,
            reason_code=None,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        )
        return result

    def retrieve_context_readonly(
        self,
        request: RetrievalRequest,
        budget: RetrievalBudget,
    ) -> RetrievedContext:
        """从显式只读的来源组合执行检索。

        与 :meth:`retrieve_context` 不同，此方法既不记录轨迹遥测，也不允许来源适配器写入
        单查询审计状态。它服务于效果配置严格只读的模型侧契约；普通检索则通过
        ``retrieve_context`` 保留原有审计和可观测行为。
        """

        execution = self._readonly_source_execution
        if execution is None:
            raise RuntimeError("readonly retrieval execution is not configured")
        return self._retrieve_context(
            request,
            budget,
            source_execution=execution,
        )

    def retrieve_context_unrecorded(
        self,
        request: RetrievalRequest,
        budget: RetrievalBudget,
    ) -> RetrievedContext:
        """执行普通 Source 路径，但把唯一 trajectory 时机交给外层公开投影。"""

        return self._retrieve_context(request, budget)

    def retrieve_file_query_batch(
        self,
        requests: Sequence[RetrievalRequest],
        budget: RetrievalBudget,
        *,
        result_limit: int,
        readonly: bool,
    ) -> tuple[RetrievedContext, ...]:
        """多个精确文件授权范围共用 query 内候选/重排预算；历史入口不变。

        返回项与输入范围一一对应，便于外层继续按原始 File authority 复查。
        这里不记录轨迹，最终公开文件投影是该请求唯一的审计写入点。
        """

        execution = self._readonly_source_execution if readonly else self._source_execution
        if execution is None:
            raise RuntimeError("readonly retrieval execution is not configured")
        generation = self._capture_retrieval_data_version_id()
        if self._data_version_provider is not None and generation is None:
            return tuple(self._retrieval_data_version_unavailable_context(
                request=request, budget=budget,
                source_plans=self._policy.compile(request, budget).source_plans,
            ) for request in requests)
        scopes, outcomes, fingerprint = execute_file_query_batch(
            tuple(requests), budget, result_limit=result_limit,
            execution=execution, policy=self._policy, query_guard=self._query_guard,
            generation=generation, max_attempts=self._max_infrastructure_attempts,
            rrf_k=self._rrf_k, reranker=self._reranker, token_estimator=self._token_estimator,
        )
        contexts = []
        for scope_index, scope in enumerate(scopes):
            state = scope.state
            source_outcomes = self._complete_source_outcomes(
                request=scope.request, observed=state.source_outcomes,
            )
            contexts.append(RetrievedContext(
                status=(ContextStatus.BLOCKED if state.blocked else self._context_status(
                    source_outcomes, state.diagnostics, truncated=bool(scope.omissions),
                )),
                items=tuple(scope.items), source_outcomes=source_outcomes,
                method_outcomes=tuple(state.method_outcomes),
                configured_token_limit=budget.context_token_limit,
                packed_tokens=sum(item.estimated_tokens for item in scope.items),
                truncated=bool(scope.omissions), pack_omissions=tuple(scope.omissions),
                verification_drops=tuple(state.verification_drops),
                coverage_limitations=self._coverage_limitations(
                    source_outcomes, additional=state.coverage_limitations,
                ),
                diagnostic_codes=tuple(state.diagnostics),
                lane_candidates=tuple(state.lane_candidates),
                fusion_candidates=tuple(state.fusion_candidates),
                reranker_candidates=(
                    tuple(candidate for part in scopes for candidate in part.reranker_candidates)
                    if scope_index == 0 else ()
                ),
                retrieval_data_version=generation or "",
                encoder_fingerprint=self._encoder_fingerprint,
                reranker_fingerprint=fingerprint,
                reranker_outcomes=outcomes if scope_index == 0 else (),
                long_term_memory_write_guard=self._long_term_memory_write_guard(
                    source_outcomes=source_outcomes, diagnostics=state.diagnostics,
                    evaluation_complete=not state.blocked,
                ),
            ))
        return tuple(contexts)

    def _retrieve_context(
        self,
        request: RetrievalRequest,
        budget: RetrievalBudget,
        *,
        source_execution: SourceExecution | None = None,
    ) -> RetrievedContext:
        execution = source_execution or self._source_execution
        queries = self._query_guard.validate(request.query_proposal)
        plan = self._policy.compile(request, budget)
        recovery = RetrievalRecoveryController(max_attempts=self._max_infrastructure_attempts)
        retrieval_data_version_id = self._capture_retrieval_data_version_id()
        if (
            self._data_version_provider is not None
            and retrieval_data_version_id is None
        ):
            return self._retrieval_data_version_unavailable_context(
                request=request,
                budget=budget,
                source_plans=plan.source_plans,
            )
        state = execution.execute(
            source_plans=plan.source_plans,
            queries=queries,
            candidate_limit=budget.candidate_limit_per_source,
            recovery=recovery,
            retrieval_data_version_id=retrieval_data_version_id,
            expected_source_snapshots=request.boundary.expected_source_snapshots,
        )
        source_outcomes = self._complete_source_outcomes(
            request=request,
            observed=state.source_outcomes,
        )
        if state.blocked:
            return RetrievedContext(
                status=ContextStatus.BLOCKED,
                items=(),
                source_outcomes=source_outcomes,
                method_outcomes=tuple(state.method_outcomes),
                configured_token_limit=budget.context_token_limit,
                packed_tokens=0,
                coverage_limitations=self._coverage_limitations(
                    source_outcomes,
                    additional=state.coverage_limitations,
                ),
                diagnostic_codes=tuple(state.diagnostics + ["required_source_blocked"]),
                lane_candidates=tuple(state.lane_candidates),
                fusion_candidates=tuple(state.fusion_candidates),
                retrieval_data_version=retrieval_data_version_id or "",
                encoder_fingerprint=self._encoder_fingerprint,
                reranker_fingerprint=self._reranker_fingerprint,
                long_term_memory_write_guard=self._long_term_memory_write_guard(
                    source_outcomes=source_outcomes,
                    diagnostics=state.diagnostics,
                    evaluation_complete=False,
                ),
            )

        resolved_by_source = execution.fetch_and_verify(
            request=request,
            source_plans=plan.source_plans,
            state=state,
            recovery=recovery,
        )
        execution.revalidate_expected_snapshots_after_fetch(
            expected_source_snapshots=request.boundary.expected_source_snapshots,
            state=state,
            resolved_by_source=resolved_by_source,
            recovery=recovery,
        )
        rerank_result = rerank_verified_candidates(
            queries=queries,
            resolved_by_source=resolved_by_source,
            reranker=self._reranker,
            candidate_limit_per_source=self._reranker_candidate_limit_per_source,
        )
        state.diagnostics.extend(
            f"reranker_degraded:{outcome.source_type.value}:"
            f"{outcome.reason_code or 'reranker_unavailable'}"
            for outcome in rerank_result.outcomes
            if outcome.status is RerankerRunStatus.DEGRADED
        )
        pack_result = self._context_packer.pack(
            source_order=tuple(source_plan.source_type for source_plan in plan.source_plans),
            resolved_by_source=rerank_result.resolved_by_source,
            budget=budget,
        )
        state.diagnostics.extend(pack_result.diagnostics)
        source_outcomes = self._complete_source_outcomes(
            request=request,
            observed=state.source_outcomes,
        )
        return RetrievedContext(
            status=self._context_status(
                source_outcomes,
                state.diagnostics,
                truncated=bool(pack_result.omissions),
            ),
            items=pack_result.items,
            source_outcomes=source_outcomes,
            method_outcomes=tuple(state.method_outcomes),
            configured_token_limit=budget.context_token_limit,
            packed_tokens=pack_result.packed_tokens,
            truncated=bool(pack_result.omissions),
            pack_omissions=pack_result.omissions,
            verification_drops=tuple(state.verification_drops),
            coverage_limitations=self._coverage_limitations(
                source_outcomes,
                additional=state.coverage_limitations,
            ),
            diagnostic_codes=tuple(state.diagnostics),
            lane_candidates=tuple(state.lane_candidates),
            fusion_candidates=tuple(state.fusion_candidates),
            reranker_candidates=tuple(
                candidate
                for source_plan in plan.source_plans
                for candidate, _ in rerank_result.resolved_by_source.get(
                    source_plan.source_type,
                    (),
                )
            ),
            retrieval_data_version=retrieval_data_version_id or "",
            encoder_fingerprint=self._encoder_fingerprint,
            reranker_fingerprint=rerank_result.reranker_fingerprint,
            reranker_outcomes=rerank_result.outcomes,
            long_term_memory_write_guard=self._long_term_memory_write_guard(
                source_outcomes=source_outcomes,
                diagnostics=state.diagnostics,
                evaluation_complete=True,
            ),
        )

    def _retrieval_data_version_unavailable_context(
        self,
        *,
        request: RetrievalRequest,
        budget: RetrievalBudget,
        source_plans: Sequence[SourcePlan],
    ) -> RetrievedContext:
        """精确 generation 不可用时，不把 ``None`` 降级成任意 ACTIVE 查询。"""

        observed = {
            source_plan.source_type: SourceOutcome(
                source_type=source_plan.source_type,
                availability=(
                    SourceAvailability.UNAVAILABLE
                    if source_plan.should_run
                    else SourceAvailability.DISABLED
                ),
                retrieval=SourceRetrievalStatus.NOT_RUN,
                dependency=source_plan.dependency,
                reason_code=(
                    "retrieval_data_version_unavailable"
                    if source_plan.should_run
                    else source_plan.skipped_reason
                ),
            )
            for source_plan in source_plans
        }
        source_outcomes = self._complete_source_outcomes(
            request=request,
            observed=observed,
        )
        blocked = any(
            source_plan.should_run
            and source_plan.dependency is SourceDependency.REQUIRED
            for source_plan in source_plans
        )
        diagnostics = ("retrieval_data_version_unavailable",)
        return RetrievedContext(
            status=ContextStatus.BLOCKED if blocked else ContextStatus.PARTIAL,
            items=(),
            source_outcomes=source_outcomes,
            method_outcomes=(),
            configured_token_limit=budget.context_token_limit,
            packed_tokens=0,
            coverage_limitations=self._coverage_limitations(source_outcomes),
            diagnostic_codes=diagnostics,
            retrieval_data_version="",
            encoder_fingerprint=self._encoder_fingerprint,
            reranker_fingerprint=self._reranker_fingerprint,
            long_term_memory_write_guard=self._long_term_memory_write_guard(
                source_outcomes=source_outcomes,
                diagnostics=diagnostics,
                evaluation_complete=False,
            ),
        )

    def _capture_retrieval_data_version_id(self) -> str | None:
        """捕获本次逻辑读取使用的唯一派生数据 generation。

        缺失或出错的 provider 会被刻意转为 ``None``。随后 Document 覆盖会安全失败，
        而不会把未固定版本的查找转换为误导性的完整 ``no_match``；没有 provider 的旧版
        调用方仍保留既有方法存储兼容行为。
        """

        if self._data_version_provider is None:
            return None
        try:
            data_version_id = self._data_version_provider.active_retrieval_data_version_id()
        except RetrievalCancelled:
            raise
        except Exception:
            return None
        if not isinstance(data_version_id, str) or not data_version_id.strip():
            return None
        return data_version_id

    @staticmethod
    def _complete_source_outcomes(
        *,
        request: RetrievalRequest,
        observed: Mapping[SourceType, SourceOutcome],
    ) -> dict[SourceType, SourceOutcome]:
        """返回一次逻辑调用固定的四 Source 结果骨架。

        受信边界中省略的 Source 会被刻意标为禁用，而非静默缺席。反之，若策略丢弃允许的
        Source，就违反了请求契约；调用方会收到显式失败结果，而非误导性的 ``not_run``。
        """

        completed: dict[SourceType, SourceOutcome] = {}
        allowed_sources = set(request.boundary.allowed_sources)
        for source_type in SourceType:
            outcome = observed.get(source_type)
            if outcome is None:
                dependency = request.boundary.dependency_for(source_type)
                if source_type in allowed_sources:
                    outcome = SourceOutcome(
                        source_type=source_type,
                        availability=SourceAvailability.UNAVAILABLE,
                        retrieval=SourceRetrievalStatus.FAILED,
                        dependency=dependency,
                        reason_code="retrieval_policy_omitted_allowed_source",
                    )
                else:
                    outcome = SourceOutcome(
                        source_type=source_type,
                        availability=SourceAvailability.DISABLED,
                        retrieval=SourceRetrievalStatus.NOT_RUN,
                        dependency=dependency,
                        reason_code="source_excluded_by_trusted_boundary",
                    )
            completed[source_type] = outcome
        return completed

    @staticmethod
    def _coverage_limitations(
        source_outcomes: Mapping[SourceType, SourceOutcome],
        *,
        additional: Sequence[ContextCoverageLimitation] = (),
    ) -> tuple[ContextCoverageLimitation, ...]:
        """投影安全的来源覆盖事实，不泄露检索内部信息。"""

        limitations: list[ContextCoverageLimitation] = []
        seen: set[tuple[SourceType, str]] = set()

        def append(limitation: ContextCoverageLimitation) -> None:
            identity = (limitation.source_type, limitation.reason_code)
            if identity not in seen:
                seen.add(identity)
                limitations.append(limitation)

        additional_by_source: dict[SourceType, list[ContextCoverageLimitation]] = {}
        for limitation in additional:
            if isinstance(limitation, ContextCoverageLimitation):
                additional_by_source.setdefault(limitation.source_type, []).append(limitation)
        for source_type in SourceType:
            for limitation in sorted(
                additional_by_source.get(source_type, ()),
                key=lambda item: item.reason_code,
            ):
                append(limitation)
            outcome = source_outcomes.get(source_type)
            if outcome is None:
                continue
            if outcome.availability is not SourceAvailability.READY:
                append(
                    ContextCoverageLimitation(
                        source_type=source_type,
                        reason_code=(
                            outcome.reason_code
                            or f"source_availability_{outcome.availability.value}"
                        ),
                    )
                )
                continue
            if outcome.retrieval in {
                SourceRetrievalStatus.PARTIAL,
                SourceRetrievalStatus.FAILED,
                SourceRetrievalStatus.NOT_RUN,
            }:
                append(
                    ContextCoverageLimitation(
                        source_type=source_type,
                        reason_code=(
                            outcome.reason_code
                            or f"source_retrieval_{outcome.retrieval.value}"
                        ),
                    )
                )
            coverage_gap = outcome.coverage_facts.get("coverage_gap")
            if coverage_gap in {"true", "unknown"}:
                append(
                    ContextCoverageLimitation(
                        source_type=source_type,
                        reason_code=(
                            f"{source_type.value}_processing_coverage_partial"
                            if coverage_gap == "true"
                            else f"{source_type.value}_processing_coverage_unknown"
                        ),
                    )
                )
        return tuple(limitations)

    @staticmethod
    def _context_status(
        source_outcomes: Mapping[SourceType, SourceOutcome],
        diagnostics: Sequence[str],
        *,
        truncated: bool = False,
    ) -> ContextStatus:
        if diagnostics or truncated:
            return ContextStatus.PARTIAL
        if any(
            outcome.availability
            in {
                SourceAvailability.UNAVAILABLE,
                SourceAvailability.BLOCKED,
                SourceAvailability.NOT_IMPLEMENTED,
            }
            or outcome.retrieval in {SourceRetrievalStatus.PARTIAL, SourceRetrievalStatus.FAILED}
            or outcome.coverage_facts.get("coverage_gap") in {"true", "unknown"}
            for outcome in source_outcomes.values()
        ):
            return ContextStatus.PARTIAL
        return ContextStatus.COMPLETE

    @staticmethod
    def _long_term_memory_write_guard(
        *,
        source_outcomes: Mapping[SourceType, SourceOutcome],
        diagnostics: Sequence[str],
        evaluation_complete: bool,
    ) -> LongTermMemoryWriteGuard:
        """根据最终 Source 结果推导双记忆禁止信号。"""

        long_term_sources = (
            SourceType.LONG_TERM_USER,
            SourceType.LONG_TERM_TASK,
        )
    # 固定的四 Source 响应骨架包含已禁用 Source。它们是显式可见性事实，而非已完成的
    # 长期记忆评估；若视其为故障，每个聚焦检索配方都会产生虚假写入禁令。
        evaluated = tuple(
            source
            for source in long_term_sources
            if source in source_outcomes
            and source_outcomes[source].availability is not SourceAvailability.DISABLED
        )
        blocking: list[SourceType] = []
        reasons: list[str] = []
        for source_type in evaluated:
            outcome = source_outcomes[source_type]
            if outcome.availability not in {SourceAvailability.READY, SourceAvailability.EMPTY}:
                blocking.append(source_type)
                reasons.append(f"{source_type.value}:availability_{outcome.availability.value}")
            elif outcome.retrieval in {
                SourceRetrievalStatus.PARTIAL,
                SourceRetrievalStatus.FAILED,
            }:
                blocking.append(source_type)
                reasons.append(f"{source_type.value}:retrieval_{outcome.retrieval.value}")

        for source_type in evaluated:
            if any(
                diagnostic.startswith(
                    (
                        f"source_fetch_circuit_open:{source_type.value}",
                        f"source_fetch_failed:{source_type.value}",
                        f"source_unit_missing:{source_type.value}",
                        f"source_unit_not_retrievable:{source_type.value}",
                        f"source_unit_ref_mismatch:{source_type.value}",
                    )
                )
                for diagnostic in diagnostics
            ) and source_type not in blocking:
                blocking.append(source_type)
                reasons.append(f"{source_type.value}:source_verification_partial")

        if blocking:
            return LongTermMemoryWriteGuard(
                status=LongTermMemoryWriteGuardStatus.BLOCKED,
                evaluated_sources=evaluated,
                blocking_sources=tuple(blocking),
                reason_codes=tuple(reasons),
            )
        if not evaluation_complete or len(evaluated) != len(long_term_sources):
            return LongTermMemoryWriteGuard(
                status=LongTermMemoryWriteGuardStatus.NOT_EVALUATED,
                evaluated_sources=evaluated,
            )
        return LongTermMemoryWriteGuard(
            status=LongTermMemoryWriteGuardStatus.CLEAR,
            evaluated_sources=evaluated,
        )


def _reranker_fingerprint(reranker: RerankerPort | None) -> str:
    if reranker is None:
        return ""
    try:
        fingerprint = reranker.fingerprint()
    except Exception as exc:
        return f"reranker_fingerprint_unavailable:{type(exc).__name__}"
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        return "reranker_fingerprint_unavailable"
    return fingerprint
