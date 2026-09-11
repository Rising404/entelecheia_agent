"""与具体检索实现无关、且不保存正文的相关性评测合同。

语料所有者提供 query、人工判定的精确 ``SourceUnitRef`` 和可选发布阈值；本模块只根据排序后的
引用计算 MRR、Recall@K、Hit@K、NDCG@K，并产生透明的 gate violation。它不执行检索、不读取
Source 内容，也不会把某一语料或模型的阈值自动套到产品路径。

该模块留在包根，是因为指标用于比较 lexical、dense、hybrid、reranker 及不同 Source adapter，
不属于某一种 online orchestration 或运维命令。目前只有这一组公共评测合同，单独建立一层
``evaluation/`` 子包会增加路径而没有形成新的职责边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence

from .contracts import SourceUnitRef


@dataclass(frozen=True, slots=True)
class RelevanceCase:
    """一个由外部判定的查询；相关性绑定到精确 Source 引用。"""

    case_id: str
    query: str
    relevant_refs: frozenset[SourceUnitRef]

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.query.strip():
            raise ValueError("case_id and query must not be empty")
        if not self.relevant_refs:
            raise ValueError("a relevance case must have at least one judged relevant ref")


@dataclass(frozen=True, slots=True)
class RelevanceCaseMetrics:
    """一个经判定查询的仅含指针指标。"""

    case_id: str
    ranked_count: int
    first_relevant_rank: int | None
    reciprocal_rank: float
    recall_at_k: Mapping[int, float]
    hit_at_k: Mapping[int, float]
    ndcg_at_k: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class RelevanceReport:
    """语料和路径的聚合指标，不保留来源正文。"""

    route_id: str
    case_count: int
    missing_result_case_ids: tuple[str, ...]
    mean_reciprocal_rank: float
    mean_recall_at_k: Mapping[int, float]
    hit_rate_at_k: Mapping[int, float]
    mean_ndcg_at_k: Mapping[int, float]
    cases: tuple[RelevanceCaseMetrics, ...]


@dataclass(frozen=True, slots=True)
class RelevanceQualityGate:
    """由调用方所有、用于一次基准发布决策的透明最低要求。

    所有字段均可选，使同一个运行器可在产品决定发布阈值前先收集基线。缺失阈值刻意不会
    根据其他语料或检索路径推断。
    """

    minimum_mrr: float | None = None
    minimum_recall_at_k: Mapping[int, float] | None = None
    minimum_hit_rate_at_k: Mapping[int, float] | None = None
    minimum_ndcg_at_k: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        _validate_fraction("minimum_mrr", self.minimum_mrr)
        for name, mapping in (
            ("minimum_recall_at_k", self.minimum_recall_at_k),
            ("minimum_hit_rate_at_k", self.minimum_hit_rate_at_k),
            ("minimum_ndcg_at_k", self.minimum_ndcg_at_k),
        ):
            for cutoff, value in (mapping or {}).items():
                if int(cutoff) <= 0:
                    raise ValueError(f"{name} cutoffs must be greater than zero")
                _validate_fraction(f"{name}[{cutoff}]", value)


@dataclass(frozen=True, slots=True)
class RelevanceGateResult:
    passed: bool
    violations: tuple[str, ...]


def evaluate_ranked_retrieval(
    *,
    route_id: str,
    cases: Sequence[RelevanceCase],
    ranked_refs_by_case_id: Mapping[str, Sequence[SourceUnitRef]],
    cutoffs: Sequence[int] = (1, 3, 5),
) -> RelevanceReport:
    """基于排序引用列表计算 Recall@K、Hit@K、MRR 和 nDCG@K。

    重复引用在首次出现后会被忽略，以匹配检索上下文的稳定去重语义。未知结果键会被拒绝，
    防止过期或拼写错误的基准案例被静默遗漏。
    """

    normalized_cutoffs = _normalize_cutoffs(cutoffs)
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("relevance case ids must be unique")
    unknown_case_ids = set(ranked_refs_by_case_id) - set(case_by_id)
    if unknown_case_ids:
        raise ValueError(f"ranked results include unknown case ids: {sorted(unknown_case_ids)}")
    if not route_id.strip():
        raise ValueError("route_id must not be empty")

    case_metrics: list[RelevanceCaseMetrics] = []
    missing: list[str] = []
    for case in cases:
        ranked = tuple(ranked_refs_by_case_id.get(case.case_id, ()))
        if case.case_id not in ranked_refs_by_case_id:
            missing.append(case.case_id)
        case_metrics.append(_evaluate_case(case, ranked, normalized_cutoffs))

    denominator = len(case_metrics)
    return RelevanceReport(
        route_id=route_id,
        case_count=denominator,
        missing_result_case_ids=tuple(missing),
        mean_reciprocal_rank=_mean(item.reciprocal_rank for item in case_metrics),
        mean_recall_at_k={
            cutoff: _mean(item.recall_at_k[cutoff] for item in case_metrics)
            for cutoff in normalized_cutoffs
        },
        hit_rate_at_k={
            cutoff: _mean(item.hit_at_k[cutoff] for item in case_metrics)
            for cutoff in normalized_cutoffs
        },
        mean_ndcg_at_k={
            cutoff: _mean(item.ndcg_at_k[cutoff] for item in case_metrics)
            for cutoff in normalized_cutoffs
        },
        cases=tuple(case_metrics),
    )


def evaluate_relevance_gate(
    report: RelevanceReport,
    gate: RelevanceQualityGate,
) -> RelevanceGateResult:
    """只使用调用方提供的阈值比较报告。"""

    violations: list[str] = []
    if gate.minimum_mrr is not None and report.mean_reciprocal_rank < gate.minimum_mrr:
        violations.append(
            f"mrr_below_minimum:{report.mean_reciprocal_rank:.6f}<{gate.minimum_mrr:.6f}"
        )
    _check_cutoff_minimums(
        name="recall",
        actual=report.mean_recall_at_k,
        minimums=gate.minimum_recall_at_k,
        violations=violations,
    )
    _check_cutoff_minimums(
        name="hit_rate",
        actual=report.hit_rate_at_k,
        minimums=gate.minimum_hit_rate_at_k,
        violations=violations,
    )
    _check_cutoff_minimums(
        name="ndcg",
        actual=report.mean_ndcg_at_k,
        minimums=gate.minimum_ndcg_at_k,
        violations=violations,
    )
    if report.missing_result_case_ids:
        violations.append(f"missing_result_cases:{','.join(report.missing_result_case_ids)}")
    return RelevanceGateResult(passed=not violations, violations=tuple(violations))


def _evaluate_case(
    case: RelevanceCase,
    ranked_refs: Sequence[SourceUnitRef],
    cutoffs: tuple[int, ...],
) -> RelevanceCaseMetrics:
    deduplicated = _deduplicate_refs(ranked_refs)
    first_relevant_rank = next(
        (rank for rank, ref in enumerate(deduplicated, start=1) if ref in case.relevant_refs),
        None,
    )
    return RelevanceCaseMetrics(
        case_id=case.case_id,
        ranked_count=len(deduplicated),
        first_relevant_rank=first_relevant_rank,
        reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        recall_at_k={
            cutoff: len(set(deduplicated[:cutoff]) & case.relevant_refs) / len(case.relevant_refs)
            for cutoff in cutoffs
        },
        hit_at_k={
            cutoff: float(any(ref in case.relevant_refs for ref in deduplicated[:cutoff]))
            for cutoff in cutoffs
        },
        ndcg_at_k={
            cutoff: _ndcg_at_k(deduplicated, case.relevant_refs, cutoff)
            for cutoff in cutoffs
        },
    )


def _normalize_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in cutoffs}))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("cutoffs must contain positive integers")
    return normalized


def _deduplicate_refs(refs: Sequence[SourceUnitRef]) -> tuple[SourceUnitRef, ...]:
    seen: set[SourceUnitRef] = set()
    deduplicated: list[SourceUnitRef] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduplicated.append(ref)
    return tuple(deduplicated)


def _ndcg_at_k(
    ranked_refs: Sequence[SourceUnitRef],
    relevant_refs: frozenset[SourceUnitRef],
    cutoff: int,
) -> float:
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, ref in enumerate(ranked_refs[:cutoff], start=1)
        if ref in relevant_refs
    )
    ideal_count = min(len(relevant_refs), cutoff)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg


def _mean(values) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def _check_cutoff_minimums(
    *,
    name: str,
    actual: Mapping[int, float],
    minimums: Mapping[int, float] | None,
    violations: list[str],
) -> None:
    for cutoff, minimum in (minimums or {}).items():
        cutoff = int(cutoff)
        if cutoff not in actual:
            violations.append(f"{name}_cutoff_not_reported:{cutoff}")
        elif actual[cutoff] < minimum:
            violations.append(f"{name}_below_minimum@{cutoff}:{actual[cutoff]:.6f}<{minimum:.6f}")


def _validate_fraction(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
