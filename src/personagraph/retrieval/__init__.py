"""Retrieval 领域的稳定公共入口。

根级 API 只暴露跨实现仍稳定的合同、查询准入、策略、在线服务、Prompt 投影和离线相关性
评测。具体实现按职责分布在 ``indexing``、``sources``、``lifecycle``、``orchestration``、
``tooling`` 与 ``operations``；调用方不应绕过本入口去拼装第二条在线检索链。

Retrieval 刻意不依赖 Runtime 图、Session 状态机或模型工具注册。Runtime 可以把已冻结的
Host 范围投影为带类型请求，但来源授权、generation 身份、候选验证与上下文预算仍由本领域
合同约束。``__all__`` 是承诺稳定的公共面；未列出的具体 Store、adapter 和 composition helper
应从其 canonical 子模块显式导入。
"""

from .contracts import (
    CorpusKey,
    ContextStatus,
    ContextCoverageLimitation,
    ContextPackOmission,
    ContextPackOmissionReason,
    ContextVerificationDrop,
    ContextVerificationDropReason,
    LongTermMemoryWriteGuard,
    LongTermMemoryWriteGuardStatus,
    QueryProposal,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievedContext,
    RetrievalMethod,
    RetrievalRequest,
    SourceAvailability,
    SourceDependency,
    SourceFilter,
    SourceAccess,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from .policy import DefaultRetrievalPolicy
from .prompt_context import RetrievedContextPromptSerializer, SerializedRetrievedContext
from .query_guard import QueryGuard, QueryGuardConfig, QueryGuardError, QueryGuardErrorCode
from .relevance import (
    RelevanceCase,
    RelevanceCaseMetrics,
    RelevanceGateResult,
    RelevanceQualityGate,
    RelevanceReport,
    evaluate_ranked_retrieval,
    evaluate_relevance_gate,
)
from .service import RetrievalService
from .indexing.token_estimation import (
    BgeM3RetrievalTokenEstimator,
    RetrievalTokenEstimatorSnapshot,
)

__all__ = [
    "CorpusKey",
    "ContextStatus",
    "ContextCoverageLimitation",
    "ContextPackOmission",
    "ContextPackOmissionReason",
    "ContextVerificationDrop",
    "ContextVerificationDropReason",
    "BgeM3RetrievalTokenEstimator",
    "DefaultRetrievalPolicy",
    "LongTermMemoryWriteGuard",
    "LongTermMemoryWriteGuardStatus",
    "QueryGuard",
    "QueryGuardConfig",
    "QueryGuardError",
    "QueryGuardErrorCode",
    "QueryProposal",
    "RelevanceCase",
    "RelevanceCaseMetrics",
    "RelevanceGateResult",
    "RelevanceQualityGate",
    "RelevanceReport",
    "RetrievalBudget",
    "RetrievalCandidate",
    "RetrievedContext",
    "RetrievedContextPromptSerializer",
    "RetrievalMethod",
    "RetrievalRequest",
    "RetrievalService",
    "RetrievalTokenEstimatorSnapshot",
    "SourceAvailability",
    "SourceDependency",
    "SourceFilter",
    "SourceAccess",
    "SourceIndexBinding",
    "SourceIndexBindingSnapshot",
    "SourceType",
    "SourceUnit",
    "SourceUnitRef",
    "SerializedRetrievedContext",
    "evaluate_ranked_retrieval",
    "evaluate_relevance_gate",
]
