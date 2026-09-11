"""Retrieval 核心依赖外部能力时使用的窄 Protocol。

这些端口把在线 Service 与具体 Source 数据库、派生索引、reranker、generation provider 和
token estimator 解耦。接口只描述 Retrieval 完成一次读取所需的最小能力；实现分别位于
``sources``、``indexing``、``lifecycle`` 或外层 composition，不应把 Store、Runtime 或供应商
对象泄漏进合同。

本模块是依赖叶子而不是 adapter 集合：新增实现应放到所属子域，只有多个实现共同依赖且语义
稳定的 Protocol 才应加入这里。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from .contracts import (
    ContextCoverageLimitation,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalPlan,
    RetrievalRequest,
    SourceAccess,
    SourceFilter,
    SourceIndexBindingSnapshot,
    SourceUnit,
    SourceUnitRef,
    SourceType,
)


class RetrievalCancelled(Exception):
    """调用方撤销本次检索；不是后端不可用，不可降级或透明重试。"""


class RetrievalCancellationPort(Protocol):
    """由外层适配的调用级控制，不含 Tool、Turn 或模型参数。"""

    def checkpoint(self) -> None:
        """已取消时抛 RetrievalCancelled；单次 native forward 不能被强杀。"""
        ...

    def remaining_seconds(self) -> float | None: ...

    def snapshot(self) -> Mapping[str, object]: ...

    def on_settled(self, observer: Callable[[str], None]) -> None:
        """处理器退出且工具终态确定后通知审计；不阻塞调用方的超时返回。"""
        ...


class SourceRetrievalAdapter(Protocol):
    source_type: SourceType

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess: ...

    def fetch_units(
        self,
        access: SourceAccess,
        refs: Sequence[SourceUnitRef],
    ) -> Sequence[SourceUnit]: ...


class SourceAccessRevalidator(Protocol):
    """重新检查一个已打开的 Source 读取，不改变受信作用域。"""

    source_type: SourceType

    def revalidate_retrieval_access(self, access: SourceAccess) -> SourceAccess: ...


class SourceCoverageProbe(Protocol):
    """一个就绪 Source 已被派生数据覆盖的只读证明。"""

    source_type: SourceType

    def check_source_coverage(
        self,
        access: SourceAccess,
        *,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None: ...


class SourceIndexBindingReaderPort(Protocol):
    """仅供派生索引覆盖探针使用、由 Source 所有的指针快照。"""

    def get_current_index_binding_snapshot(
        self,
        access: SourceAccess,
        *,
        maximum_bindings: int,
    ) -> SourceIndexBindingSnapshot: ...


class RetrievalDataVersionProvider(Protocol):
    """为一次逻辑调用捕获一个检索数据版本身份。"""

    def active_retrieval_data_version_id(self) -> str | None: ...


class RetrievalMethodPort(Protocol):
    method: RetrievalMethod

    def search(
        self,
        query: str,
        *,
        query_index: int,
        source_filter: SourceFilter,
        limit: int,
        retrieval_data_version_id: str | None = None,
    ) -> Sequence[RetrievalCandidate]: ...


class RetrievalPolicyPort(Protocol):
    def compile(self, request: RetrievalRequest, budget: RetrievalBudget) -> RetrievalPlan: ...


TokenEstimator = Callable[[str], int]


class BgeM3EncoderPort(Protocol):
    """一次 BGE-M3 流程为一段文本提供全部表示。"""

    def encode(self, texts: Sequence[str]): ...

    def token_ids(self, text: str) -> Sequence[int]: ...

    def tokenizer_fingerprint(self) -> str: ...

    def fingerprint(self) -> str: ...


class RerankerPort(Protocol):
    """在本地为已授权的 Query/SourceUnit 内容对评分。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...

    def fingerprint(self) -> str: ...
