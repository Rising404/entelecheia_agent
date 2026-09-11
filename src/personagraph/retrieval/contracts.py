"""Retrieval 各子域共享的纯类型与不变量。

这里定义请求边界、来源身份、候选、预算、来源结果和最终 ``RetrievedContext``，但不执行
数据库、模型、文件或网络 I/O。它处在依赖图最底部，Indexing、Sources、Orchestration、
Tooling 与 Runtime adapter 都可以依赖它，合同本身不能反向依赖这些实现。

最重要的信任分层是：

* ``SourceUnitRef`` 只定位一份权威内容及其不可变修订；
* ``SourceFilter`` 是 Host 冻结、并在 Top-K 之前应用的授权候选范围；
* ``RetrievalCandidate`` 只表达派生索引命中，不自动成为可信证据；
* ``SourceUnit`` 是按引用回到权威来源并复核后取得的内容。

检索数据库可以保存指针、哈希、范围与派生表示，但绝不能把 ``SourceUnit.content`` 保存成
第二权威源。模型提供的 query、source hint 或完成声明也不能修改这些 Host 权威字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping


MAX_FILE_RETRIEVAL_QUERIES = 4
FILE_RETRIEVAL_CANDIDATES_PER_QUERY = 128
FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY = 64
MAX_FILE_RETRIEVAL_ITEMS = 96


class CorpusKey(StrEnum):
    """共享 Retrieval 内核的独立派生索引空间。"""

    FILE = "file"
    HISTORY = "history"


class SourceType(StrEnum):
    CURRENT_SESSION = "current_session"
    LONG_TERM_USER = "long_term_user"
    LONG_TERM_TASK = "long_term_task"
    DOCUMENT = "document"
    PICTURE = "picture"


# ``SourceFilter`` 默认仍是精确结构选择器。下方唯一的封闭范围运算符仅供 Host 使用，
# 使冻结 Turn 能搜索接受时刻的当前 Session 历史。已索引 Unit 带有
# ``assistant_turn_idx``；请求可以携带 ``*_lte`` 键。
CURRENT_SESSION_TURN_INDEX_SCOPE_KEY = "assistant_turn_idx"
CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY = "assistant_turn_idx_lte"


class RetrievalMethod(StrEnum):
    DENSE = "dense"
    LEARNED_SPARSE = "learned_sparse"
    BM25 = "bm25"
    LITERAL_BOOLEAN = "literal_boolean"


class RetrievalStatus(StrEnum):
    """与 Unit 目录一同存储的生命周期可见性。

    索引构建状态属于同步诊断，不应复用公开检索生命周期枚举来表达。
    """

    ACTIVE = "active"
    TRASHED = "trashed"


class SourceAvailability(StrEnum):
    READY = "ready"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    NOT_IMPLEMENTED = "not_implemented"
    BLOCKED = "blocked"


_SOURCE_COVERAGE_FACT_KEYS = frozenset({
    "processing_status",
    "processing_diagnostic_codes",
    "needs_vision",
    "coverage_gap",
})
DOCUMENT_PROCESSING_COVERAGE_CODES = frozenset({
    "fallback_reader_used",
    "page_empty",
    "page_needs_vision",
    "parser_partial",
})


def _freeze_source_coverage_facts(facts: Mapping[str, str]) -> Mapping[str, str]:
    """冻结可能进入模型 Prompt 的有界公开事实。"""

    if not isinstance(facts, Mapping):
        raise ValueError("coverage_facts must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in facts.items():
        if key not in _SOURCE_COVERAGE_FACT_KEYS:
            raise ValueError(f"unsupported source coverage fact: {key}")
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError("source coverage fact values must be bounded non-empty strings")
        if any(ord(character) < 32 for character in value):
            raise ValueError("source coverage fact values must not contain control characters")
        normalized[key] = value
    if normalized:
        if set(normalized) != _SOURCE_COVERAGE_FACT_KEYS:
            raise ValueError("source coverage facts must provide the complete bounded shape")
        status = normalized["processing_status"]
        if status not in {"complete", "partial", "legacy_unknown"}:
            raise ValueError("processing_status coverage fact is invalid")
        if normalized["needs_vision"] not in {"true", "false", "unknown"}:
            raise ValueError("needs_vision coverage fact is invalid")
        expected_gap = {
            "complete": "false",
            "partial": "true",
            "legacy_unknown": "unknown",
        }[status]
        if normalized["coverage_gap"] != expected_gap:
            raise ValueError("coverage_gap must match processing_status")
        try:
            diagnostic_codes = json.loads(normalized["processing_diagnostic_codes"])
        except json.JSONDecodeError as exc:
            raise ValueError("processing_diagnostic_codes must be canonical JSON") from exc
        if (
            not isinstance(diagnostic_codes, list)
            or any(
                not isinstance(code, str)
                or code not in DOCUMENT_PROCESSING_COVERAGE_CODES
                for code in diagnostic_codes
            )
            or len(set(diagnostic_codes)) != len(diagnostic_codes)
        ):
            raise ValueError("processing_diagnostic_codes contains unsupported values")
        normalized["processing_diagnostic_codes"] = json.dumps(
            sorted(diagnostic_codes),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if status == "complete" and diagnostic_codes:
            raise ValueError("complete processing coverage cannot carry gap codes")
        if status == "partial" and not diagnostic_codes:
            raise ValueError("partial processing coverage requires a gap code")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class SourceAccess:
    """每次调用用于搜索和读取一个 Source 的权威许可。

    ``source_filter`` 仍是检索前编译的受信作用域。``source_snapshot_id`` 和
    ``source_revision_map`` 则捕获打开本次读取时观察到的事实。它们绝不是模型输入，
    检索策略或排序代码也不会解释它们。文档 Source 当前把 revision 映射用作
    ``doc_id -> source_version_id``；其他 Source 可以将其留空。
    """

    source_type: SourceType
    source_filter: SourceFilter
    availability: SourceAvailability
    reason_code: str | None = None
    source_snapshot_id: str | None = None
    source_revision_map: Mapping[str, str] = field(default_factory=dict)
    coverage_facts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SourceType):
            raise ValueError("source_type must be a SourceType")
        if not isinstance(self.source_filter, SourceFilter):
            raise ValueError("source_filter must be a SourceFilter")
        if self.source_filter.source_type is not self.source_type:
            raise ValueError("source_filter must match source_type")
        if not isinstance(self.availability, SourceAvailability):
            raise ValueError("availability must be a SourceAvailability")
        for name, value in (
            ("reason_code", self.reason_code),
            ("source_snapshot_id", self.source_snapshot_id),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if not isinstance(self.source_revision_map, Mapping):
            raise ValueError("source_revision_map must be a mapping")
        revision_map = dict(self.source_revision_map)
        for container_id, revision in revision_map.items():
            if not isinstance(container_id, str) or not container_id.strip():
                raise ValueError("source_revision_map keys must be non-empty strings")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("source_revision_map values must be non-empty strings")
        object.__setattr__(self, "source_revision_map", MappingProxyType(revision_map))
        object.__setattr__(
            self,
            "coverage_facts",
            _freeze_source_coverage_facts(self.coverage_facts),
        )


@dataclass(frozen=True, slots=True)
class SourceIndexBinding:
    """派生索引中预期存在的一项仅含指针的权威绑定。

    此结构只省略 ``source_type``，因为外层 Source 拥有该边界。内容哈希属于绑定的一部分：
    仅凭 revision ID 无法证明目录索引的是权威源所暴露的精确字节。
    """

    source_unit_id: str
    source_revision: str
    indexed_content_hash: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_unit_id", self.source_unit_id),
            ("source_revision", self.source_revision),
            ("indexed_content_hash", self.indexed_content_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


def source_index_binding_manifest(
    bindings: tuple[SourceIndexBinding, ...] | list[SourceIndexBinding],
) -> tuple[str, Mapping[str, str]]:
    """返回 Source 绑定的规范且不含内容的身份。

    清单只包含持久指针、revision 和权威字节的哈希。因此可安全持久化到冻结运行时作用域，
    而无需把记忆内容复制进 Runtime 状态。
    """

    normalized = tuple(bindings)
    if any(not isinstance(binding, SourceIndexBinding) for binding in normalized):
        raise ValueError("bindings must contain SourceIndexBinding values")
    ordered = sorted(
        (
            binding.source_unit_id,
            binding.source_revision,
            binding.indexed_content_hash,
        )
        for binding in normalized
    )
    encoded = json.dumps(
        ordered,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, MappingProxyType(
        {
            "binding_manifest_sha256": digest,
            "binding_count": str(len(ordered)),
        }
    )


@dataclass(frozen=True, slots=True)
class SourceIndexBindingSnapshot:
    """用于验证派生覆盖、由来源所有且不含内容的视图。

    如果 Source 在 ``SourceAccess`` 打开后发生变化、消失或离开受信作用域，
    ``source_snapshot_is_current`` 为 false。此时调用方必须把派生索引覆盖视为不完整，
    而不能假定为正常的无匹配结果。
    """

    source_snapshot_is_current: bool
    bindings: tuple[SourceIndexBinding, ...] = ()
    binding_enumeration_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot_is_current, bool):
            raise ValueError("source_snapshot_is_current must be a bool")
        if not isinstance(self.binding_enumeration_complete, bool):
            raise ValueError("binding_enumeration_complete must be a bool")
        normalized = tuple(self.bindings)
        if any(not isinstance(binding, SourceIndexBinding) for binding in normalized):
            raise ValueError("bindings must contain SourceIndexBinding values")
        identities = {(binding.source_unit_id, binding.source_revision) for binding in normalized}
        if len(identities) != len(normalized):
            raise ValueError("bindings must not contain duplicate source identities")
        object.__setattr__(self, "bindings", normalized)


class SourceRetrievalStatus(StrEnum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    PARTIAL = "partial"
    NOT_RUN = "not_run"
    FAILED = "failed"


class MethodRunStatus(StrEnum):
    USED = "used"
    NOT_RUN = "not_run"
    FAILED = "failed"
    DEGRADED = "degraded"


class RerankerRunStatus(StrEnum):
    """可选第二阶段相关性模型的结果。"""

    USED = "used"
    NOT_RUN = "not_run"
    DEGRADED = "degraded"


class SourceDependency(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class ContextStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class LongTermMemoryWriteGuardStatus(StrEnum):
    """本次检索调用观察到的长期记忆召回健康状况。

    ``CLEAR`` 刻意不表示写入授权。它只说明本次 RAG 调用观察了两个长期 Source，且未发现
    召回故障。每项实际写入决策仍归 Runtime 和 Memory 领域所有。
    """

    CLEAR = "clear"
    BLOCKED = "blocked"
    NOT_EVALUATED = "not_evaluated"


class ContextPackOmissionReason(StrEnum):
    """已验证内容未被打包进本次检索结果的原因。"""

    MAX_ITEMS = "max_items"
    TOKEN_LIMIT = "token_limit"
    SOURCE_POOL_ITEM_QUOTA = "source_pool_item_quota"


class ContextVerificationDropReason(StrEnum):
    """来源验证后无法返回轻量命中项的原因。"""

    SOURCE_FETCH_FAILED = "source_fetch_failed"
    SOURCE_UNIT_MISSING = "source_unit_missing"
    SOURCE_UNIT_NOT_RETRIEVABLE = "source_unit_not_retrievable"
    SOURCE_UNIT_REF_MISMATCH = "source_unit_ref_mismatch"


@dataclass(frozen=True, slots=True)
class SourceUnitRef:
    """指向一个权威且可独立检索的 Source Unit 的指针。"""

    source_type: SourceType
    source_unit_id: str
    source_revision: str
    indexed_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SourceType):
            raise ValueError("source_type must be a SourceType")
        for name, value in (
            ("source_unit_id", self.source_unit_id),
            ("source_revision", self.source_revision),
            ("indexed_content_hash", self.indexed_content_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SourceFilter:
    """在检索 Top-K 前编译的受信封闭结构过滤器。"""

    source_type: SourceType
    scope: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """即使面对直接调用方，也保持 SQL/JSON 过滤边界封闭。

        ``from_mapping`` 是常规构造路径，但 dataclass 仍向带类型适配器和夹具公开。在此
        校验可阻止直接调用方提供形似 JSON 路径的键，避免其之后变成数据库错误或语义略有
        不同的选择器；规范化也使相等和哈希语义不受原始映射顺序影响。
        """

        if not isinstance(self.source_type, SourceType):
            raise ValueError("source_type must be a SourceType")
        normalized_scope: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for pair in self.scope:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("scope must contain (key, value) string pairs")
            key, value = pair
            if not isinstance(key, str) or not _SCOPE_KEY.fullmatch(key):
                raise ValueError("scope keys must be simple identifier names")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("scope values must be non-empty strings")
            if any(ord(character) < 32 for character in value):
                raise ValueError("scope values must not contain control characters")
            if key in seen_keys:
                raise ValueError("scope keys must be unique")
            if key in {
                CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
                CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
            }:
                if self.source_type is not SourceType.CURRENT_SESSION:
                    raise ValueError(
                        "Turn-index scope keys are valid only for current-session retrieval"
                    )
                if not value.isascii() or not value.isdigit():
                    raise ValueError("Turn-index scope values must be non-negative decimals")
                if len(value) > 18:
                    raise ValueError("Turn-index scope value exceeds the supported range")
            seen_keys.add(key)
            normalized_scope.append((key, value))
        if {
            CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
            CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
        }.issubset(seen_keys):
            raise ValueError("a SourceFilter cannot be both an indexed Unit scope and a cutoff")
        object.__setattr__(self, "scope", tuple(sorted(normalized_scope)))

    @classmethod
    def from_mapping(
        cls,
        source_type: SourceType,
        scope: Mapping[str, str] | None = None,
    ) -> "SourceFilter":
        pairs = tuple(sorted((str(key), str(value)) for key, value in (scope or {}).items()))
        if any(not key or not value for key, value in pairs):
            raise ValueError("scope keys and values must not be empty")
        return cls(source_type=source_type, scope=pairs)

    def as_mapping(self) -> dict[str, str]:
        return dict(self.scope)

    def selects(self, indexed_scope: "SourceFilter") -> bool:
        """返回此受信边界是否允许一个已索引 Unit 作用域。

        检索请求可以刻意比单个 Unit 的存储作用域更宽。例如，请求一个 Session 中挂载的
        所有 Documents 时只携带 ``session_id``，而每个已索引 Document Unit 还携带自身
        ``doc_id``。请求仍保持封闭：它提供的每个键都必须精确匹配。

        这只是一种结构关系。它不授予跨 Source 类型访问权限，也绝不会解释模型提供的
        元数据。
        """

        if self.source_type is not indexed_scope.source_type:
            return False
        candidate = indexed_scope.as_mapping()
        for key, value in self.scope:
            if key == CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY:
                indexed_value = candidate.get(CURRENT_SESSION_TURN_INDEX_SCOPE_KEY)
                if (
                    indexed_value is None
                    or not indexed_value.isascii()
                    or not indexed_value.isdigit()
                    or int(indexed_value) > int(value)
                ):
                    return False
                continue
            if candidate.get(key) != value:
                return False
        return True


_SCOPE_KEY = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class RetrievalUnit:
    """仅含指针的 Unit 目录记录；按设计不含内容字段。"""

    ref: SourceUnitRef
    retrieval_data_version: str
    retrieval_status: RetrievalStatus = RetrievalStatus.ACTIVE
    source_filter: SourceFilter | None = None

    def __post_init__(self) -> None:
        if not self.retrieval_data_version.strip():
            raise ValueError("retrieval_data_version must not be empty")
        if self.source_filter is not None and self.source_filter.source_type != self.ref.source_type:
            raise ValueError("source_filter must match SourceUnitRef.source_type")


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """Source 适配器批量读取后返回的权威内容。"""

    ref: SourceUnitRef
    content: str
    citation: Mapping[str, str] = field(default_factory=dict)
    retrievable: bool = True

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("SourceUnit.content must not be empty")


@dataclass(frozen=True, slots=True)
class QueryProposal:
    """模型建议的语义查询，带有非权威来源提示。"""

    queries: tuple[str, ...]
    source_hints: frozenset[SourceType] = frozenset()

    def __post_init__(self) -> None:
        """在 Guard 审查前保持不受信提案的结构化类型。

        空白、重复、过长或低层查询刻意继续由 QueryGuard 负责：不同 Runtime 用途各自拥有
        其失败行为。本契约只拒绝根本不可能来自带类型提案通道的值。
        """

        if not isinstance(self.queries, tuple) or any(
            not isinstance(query, str) for query in self.queries
        ):
            raise ValueError("queries must be a tuple of strings")
        if not isinstance(self.source_hints, frozenset) or any(
            not isinstance(source_type, SourceType) for source_type in self.source_hints
        ):
            raise ValueError("source_hints must be a frozenset of SourceType values")


@dataclass(frozen=True, slots=True)
class TrustedRetrievalBoundary:
    """Runtime 所有的访问边界；绝不接受模型提供的自由格式过滤器。"""

    source_filters: Mapping[SourceType, SourceFilter]
    source_dependencies: Mapping[SourceType, SourceDependency] = field(default_factory=dict)
    expected_source_snapshots: Mapping[SourceType, SourceAccess] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_filters, Mapping):
            raise ValueError("source_filters must be a mapping")
        if not isinstance(self.source_dependencies, Mapping):
            raise ValueError("source_dependencies must be a mapping")
        if not isinstance(self.expected_source_snapshots, Mapping):
            raise ValueError("expected_source_snapshots must be a mapping")

        # 复制每个受信叶节点以及两层外部映射。若仅包装调用方的 dict，此边界仍会与调用方
        # 共享嵌套的 ``SourceFilter`` 对象。SourceFilter 只有值字段，因此重建它就是此处
        # 所需的窄而显式的深快照，无需使用通用 deepcopy。
        source_filters: dict[SourceType, SourceFilter] = {}
        for source_type, source_filter in self.source_filters.items():
            if not isinstance(source_type, SourceType):
                raise ValueError("source filter keys must be SourceType values")
            if not isinstance(source_filter, SourceFilter):
                raise ValueError("source filter values must be SourceFilter values")
            if source_type != source_filter.source_type:
                raise ValueError("source filter mapping key must match its SourceType")
            source_filters[source_type] = SourceFilter(
                source_type=source_filter.source_type,
                scope=tuple(source_filter.scope),
            )

        source_dependencies: dict[SourceType, SourceDependency] = {}
        for source_type, dependency in self.source_dependencies.items():
            if not isinstance(source_type, SourceType):
                raise ValueError("source dependency keys must be SourceType values")
            if not isinstance(dependency, SourceDependency):
                raise ValueError("source dependency values must be SourceDependency values")
            source_dependencies[source_type] = dependency
        unknown = set(source_dependencies) - set(source_filters)
        if unknown:
            raise ValueError(f"dependencies declared for unavailable Sources: {sorted(unknown)}")
        expected_source_snapshots: dict[SourceType, SourceAccess] = {}
        for source_type, access in self.expected_source_snapshots.items():
            if not isinstance(source_type, SourceType):
                raise ValueError("expected snapshot keys must be SourceType values")
            if not isinstance(access, SourceAccess):
                raise ValueError("expected snapshots must be SourceAccess values")
            if (
                access.source_type is not source_type
                or access.source_filter != source_filters.get(source_type)
                or access.availability
                not in {SourceAvailability.READY, SourceAvailability.EMPTY}
                or access.source_snapshot_id is None
            ):
                raise ValueError(
                    "expected snapshots must be frozen ready/empty permits for their filters"
                )
            expected_source_snapshots[source_type] = SourceAccess(
                source_type=access.source_type,
                source_filter=access.source_filter,
                availability=access.availability,
                source_snapshot_id=access.source_snapshot_id,
                source_revision_map=access.source_revision_map,
            )
        unknown_snapshots = set(expected_source_snapshots) - set(source_filters)
        if unknown_snapshots:
            raise ValueError(
                f"snapshots declared for unavailable Sources: {sorted(unknown_snapshots)}"
            )
        # 冻结 dataclass 并不会冻结调用方提供的 dict。保留私有快照，使后续变更无法扩大
        # 已验证边界。
        object.__setattr__(self, "source_filters", MappingProxyType(source_filters))
        object.__setattr__(self, "source_dependencies", MappingProxyType(source_dependencies))
        object.__setattr__(
            self,
            "expected_source_snapshots",
            MappingProxyType(expected_source_snapshots),
        )

    @property
    def allowed_sources(self) -> tuple[SourceType, ...]:
        return tuple(self.source_filters)

    def dependency_for(self, source_type: SourceType) -> SourceDependency:
        return self.source_dependencies.get(source_type, SourceDependency.OPTIONAL)

    def expected_snapshot_for(self, source_type: SourceType) -> SourceAccess | None:
        return self.expected_source_snapshots.get(source_type)


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """单次调用的资源上限，而非模型全局产品预算策略。"""

    candidate_limit_per_source: int
    context_token_limit: int
    max_items: int

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_limit_per_source", self.candidate_limit_per_source),
            ("context_token_limit", self.context_token_limit),
            ("max_items", self.max_items),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """由受信运行时适配器传给检索层的带类型请求。"""

    request_id: str
    model_call_purpose: str
    query_proposal: QueryProposal
    boundary: TrustedRetrievalBoundary
    excluded_direct_refs: frozenset[SourceUnitRef] = frozenset()

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("model_call_purpose", self.model_call_purpose),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.query_proposal, QueryProposal):
            raise ValueError("query_proposal must be a QueryProposal")
        if not isinstance(self.boundary, TrustedRetrievalBoundary):
            raise ValueError("boundary must be a TrustedRetrievalBoundary")
        if not isinstance(self.excluded_direct_refs, frozenset) or any(
            not isinstance(ref, SourceUnitRef) for ref in self.excluded_direct_refs
        ):
            raise ValueError("excluded_direct_refs must be a frozenset of SourceUnitRef values")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """轻量方法命中项，刻意不含内容。"""

    ref: SourceUnitRef
    method: RetrievalMethod
    query_index: int
    rank: int
    raw_score: float
    fusion_score: float | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    # ``rank`` 在多查询轮询后表示 Source 内融合顺序；这里单独保留该候选在原查询
    # RRF lane 内的名次，以及形成其 RRF 分数的各方法名次。三元组只含方法、名次和
    # 原始数值分，不复制 Source 正文。
    query_fused_rank: int | None = None
    fusion_contributors: tuple[tuple[RetrievalMethod, int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.query_index < 0:
            raise ValueError("query_index must be non-negative")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.rerank_score is not None and not math.isfinite(self.rerank_score):
            raise ValueError("rerank_score must be finite when provided")
        if self.rerank_rank is not None and self.rerank_rank <= 0:
            raise ValueError("rerank_rank must be positive when provided")
        if self.query_fused_rank is not None and self.query_fused_rank <= 0:
            raise ValueError("query_fused_rank must be positive when provided")
        for contributor in self.fusion_contributors:
            if (
                not isinstance(contributor, tuple)
                or len(contributor) != 3
                or not isinstance(contributor[0], RetrievalMethod)
                or not isinstance(contributor[1], int)
                or isinstance(contributor[1], bool)
                or contributor[1] <= 0
                or not isinstance(contributor[2], (int, float))
                or isinstance(contributor[2], bool)
            ):
                raise ValueError("fusion_contributors must contain method/rank/score triples")


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    method: RetrievalMethod
    source_type: SourceType
    query_index: int
    status: MethodRunStatus
    candidate_count: int = 0
    reason_code: str | None = None
    failure_stage: str | None = None
    degraded_from: tuple[RetrievalMethod, ...] = ()
    attempt: int = 1
    infrastructure_attempts: int = 1
    attempt_failures: tuple[tuple[int, str, str | None], ...] = ()

    def __post_init__(self) -> None:
        if self.attempt <= 0:
            raise ValueError("attempt must be positive")
        if self.infrastructure_attempts <= 0:
            raise ValueError("infrastructure_attempts must be positive")
        if self.failure_stage is not None and (
            not isinstance(self.failure_stage, str)
            or not self.failure_stage.strip()
        ):
            raise ValueError("failure_stage must be non-empty when provided")
        for failed_attempt, reason_code, failure_stage in self.attempt_failures:
            if (
                isinstance(failed_attempt, bool)
                or not isinstance(failed_attempt, int)
                or failed_attempt <= 0
                or failed_attempt > self.infrastructure_attempts
                or not isinstance(reason_code, str)
                or not reason_code.strip()
                or (
                    failure_stage is not None
                    and (
                        not isinstance(failure_stage, str)
                        or not failure_stage.strip()
                    )
                )
            ):
                raise ValueError("attempt_failures contains an invalid safe diagnostic")


@dataclass(frozen=True, slots=True)
class RerankerOutcome:
    """一次 Source 本地重排流程的不含内容可观测信息。"""

    source_type: SourceType
    status: RerankerRunStatus
    candidate_count: int
    scored_candidate_count: int = 0
    reason_code: str | None = None
    query_index: int | None = None

    def __post_init__(self) -> None:
        if self.query_index is not None and (
            isinstance(self.query_index, bool)
            or not isinstance(self.query_index, int)
            or self.query_index < 0
        ):
            raise ValueError("reranker query_index must be non-negative")
        if self.candidate_count < 0 or self.scored_candidate_count < 0:
            raise ValueError("reranker candidate counts must be non-negative")
        if self.scored_candidate_count > self.candidate_count:
            raise ValueError("reranker scored count cannot exceed candidate count")
        if self.status is RerankerRunStatus.USED and self.scored_candidate_count < 1:
            raise ValueError("a used reranker must score at least one candidate")
        if self.status is RerankerRunStatus.DEGRADED and not self.reason_code:
            raise ValueError("a degraded reranker outcome requires a reason_code")


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    source_type: SourceType
    availability: SourceAvailability
    retrieval: SourceRetrievalStatus
    dependency: SourceDependency
    reason_code: str | None = None
    source_snapshot_id: str | None = None
    lane_outcomes: Mapping[str, str] = field(default_factory=dict)
    coverage_facts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_snapshot_id is not None and (
            not isinstance(self.source_snapshot_id, str) or not self.source_snapshot_id.strip()
        ):
            raise ValueError("source_snapshot_id must be a non-empty string when provided")
        object.__setattr__(
            self,
            "coverage_facts",
            _freeze_source_coverage_facts(self.coverage_facts),
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryWriteGuard:
    """由 RAG 产生的禁止信号；绝不是 Memory 写入结果。

    任一长期 Source 召回故障都会阻止该 Turn 的用户记忆和任务记忆写入。``CLEAR`` 只表示
    在完整的双 Source 评估中未观察到此类故障，并不授予任何权限。
    """

    status: LongTermMemoryWriteGuardStatus
    evaluated_sources: tuple[SourceType, ...] = ()
    blocking_sources: tuple[SourceType, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalQueryMatch:
    """精确来源与原 query 的评分关系；合并可取最高分，但不丢失各 query 身份。"""

    query_index: int
    rank: int
    fused_rank: int
    fusion_score: float | None = None
    reranker_score: float | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("query_index", self.query_index, 0),
            ("rank", self.rank, 1),
            ("fused_rank", self.fused_rank, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"query match {name} is invalid")
        if self.query_index >= MAX_FILE_RETRIEVAL_QUERIES:
            raise ValueError("query match query_index must be below 4")
        for value in (self.fusion_score, self.reranker_score):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("query match scores must be finite")


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    ref: SourceUnitRef
    content: str
    citation: Mapping[str, str]
    estimated_tokens: int
    fused_rank: int
    query_index: int = 0
    fusion_score: float | None = None
    reranked_rank: int | None = None
    reranker_score: float | None = None
    query_matches: tuple[RetrievalQueryMatch, ...] = ()

    def __post_init__(self) -> None:
        matches = tuple(self.query_matches)
        if len(matches) > MAX_FILE_RETRIEVAL_QUERIES:
            raise ValueError("query_matches accepts at most four queries")
        if any(not isinstance(match, RetrievalQueryMatch) for match in matches):
            raise ValueError("query_matches must contain RetrievalQueryMatch values")
        if len({match.query_index for match in matches}) != len(matches):
            raise ValueError("query_matches must be unique by query_index")
        object.__setattr__(self, "query_matches", matches)
        if (
            isinstance(self.query_index, bool)
            or not isinstance(self.query_index, int)
            or self.query_index < 0
        ):
            raise ValueError("RetrievedItem.query_index must be non-negative")
        if self.reranker_score is not None and (
            isinstance(self.reranker_score, bool)
            or not isinstance(self.reranker_score, (int, float))
            or not math.isfinite(self.reranker_score)
        ):
            raise ValueError("RetrievedItem.reranker_score must be finite")


@dataclass(frozen=True, slots=True)
class ContextPackOmission:
    """上下文打包造成的可计数遗漏，绝不是未知缺口。"""

    source_type: SourceType
    reason: ContextPackOmissionReason
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("ContextPackOmission.count must be positive")


@dataclass(frozen=True, slots=True)
class ContextVerificationDrop:
    """返回权威 Source 读取时丢失的可计数命中项。"""

    source_type: SourceType
    reason: ContextVerificationDropReason
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("ContextVerificationDrop.count must be positive")


@dataclass(frozen=True, slots=True)
class ContextCoverageLimitation:
    """无法安全计数其不可见范围的来源级限制。"""

    source_type: SourceType
    reason_code: str


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """可供 Prompt Builder 序列化的最终检索结果。"""

    status: ContextStatus
    items: tuple[RetrievedItem, ...]
    source_outcomes: Mapping[SourceType, SourceOutcome]
    method_outcomes: tuple[MethodOutcome, ...]
    configured_token_limit: int
    packed_tokens: int
    truncated: bool = False
    pack_omissions: tuple[ContextPackOmission, ...] = ()
    verification_drops: tuple[ContextVerificationDrop, ...] = ()
    coverage_limitations: tuple[ContextCoverageLimitation, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()
    # 以下三个集合仅供 Host trajectory 审计。Prompt serializer 不读取它们；候选只含
    # 稳定引用、排序和分数，不包含正文。
    lane_candidates: tuple[RetrievalCandidate, ...] = ()
    fusion_candidates: tuple[RetrievalCandidate, ...] = ()
    reranker_candidates: tuple[RetrievalCandidate, ...] = ()
    # 生成这些条目的索引。缺少此信息，结果就会与为其排序的编码器脱离；模型或索引版本
    # 变化后便无法回答“为什么召回了它”，而这正是过期召回引发的问题。空值表示调用方
    # 未提供该信息，绝不表示它不存在。
    retrieval_data_version: str = ""
    encoder_fingerprint: str = ""
    reranker_fingerprint: str = ""
    reranker_outcomes: tuple[RerankerOutcome, ...] = ()
    long_term_memory_write_guard: LongTermMemoryWriteGuard = field(
        default_factory=lambda: LongTermMemoryWriteGuard(
            status=LongTermMemoryWriteGuardStatus.NOT_EVALUATED
        )
    )

    def __post_init__(self) -> None:
        expected_sources = set(SourceType)
        actual_sources = set(self.source_outcomes)
        if actual_sources != expected_sources:
            raise ValueError("RetrievedContext must include an outcome for every SourceType")
        if any(
            source_type is not outcome.source_type
            for source_type, outcome in self.source_outcomes.items()
        ):
            raise ValueError("RetrievedContext source outcome keys must match their SourceType")
        if self.configured_token_limit <= 0 or self.packed_tokens < 0:
            raise ValueError("RetrievedContext token counts are invalid")
        if self.pack_omissions and not self.truncated:
            raise ValueError("pack omissions require RetrievedContext.truncated")
        if any(not isinstance(outcome, RerankerOutcome) for outcome in self.reranker_outcomes):
            raise ValueError("reranker_outcomes must contain RerankerOutcome values")
        for field_name, candidates in (
            ("lane_candidates", self.lane_candidates),
            ("fusion_candidates", self.fusion_candidates),
            ("reranker_candidates", self.reranker_candidates),
        ):
            if any(not isinstance(candidate, RetrievalCandidate) for candidate in candidates):
                raise ValueError(f"{field_name} must contain RetrievalCandidate values")
        outcome_sources = [
            (outcome.source_type, outcome.query_index) for outcome in self.reranker_outcomes
        ]
        if len(outcome_sources) != len(set(outcome_sources)):
            raise ValueError("reranker_outcomes must contain at most one outcome per Source/query")
        if not isinstance(self.long_term_memory_write_guard, LongTermMemoryWriteGuard):
            raise ValueError(
                "RetrievedContext.long_term_memory_write_guard must be a LongTermMemoryWriteGuard"
            )


@dataclass(frozen=True, slots=True)
class SourcePlan:
    source_type: SourceType
    source_filter: SourceFilter
    dependency: SourceDependency
    methods: tuple[RetrievalMethod, ...]
    should_run: bool = True
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    source_plans: tuple[SourcePlan, ...]
