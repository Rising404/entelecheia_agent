"""把已验证请求确定性编译为 Retrieval 执行计划。

Policy 只消费 ``RetrievalRequest`` 中已经由 Host 冻结的边界，并为每个允许来源选择方法与
依赖语义。它不调用模型、不访问数据库、不执行召回，也不能根据 query hint 新增来源或扩大
``SourceFilter``。实际预算消耗、方法降级、排序和打包由在线 Service/Orchestration 负责。

该职责是所有具体来源和索引实现之上的稳定决策边界，因此保留在包根并实现 ``ports`` 中的
``RetrievalPolicyPort``。
"""

from __future__ import annotations

from .contracts import (
    RetrievalBudget,
    RetrievalMethod,
    RetrievalPlan,
    RetrievalRequest,
    SourcePlan,
)


DEFAULT_PRIMARY_METHODS = (
    RetrievalMethod.DENSE,
    RetrievalMethod.LEARNED_SPARSE,
)


class DefaultRetrievalPolicy:
    """只把受信边界字段编译进 Source 计划。

    查询来源提示在此处会被刻意忽略。它们之后可以辅助诊断或有界排序启发式逻辑，但绝不能
    关闭默认 Source 或扩大受信边界。
    """

    def __init__(
        self,
        *,
        default_methods: tuple[RetrievalMethod, ...] = DEFAULT_PRIMARY_METHODS,
    ) -> None:
        if not default_methods:
            raise ValueError("at least one default retrieval method is required")
        self._default_methods = default_methods

    def compile(self, request: RetrievalRequest, budget: RetrievalBudget) -> RetrievalPlan:
        del budget  # 预算由服务消耗；策略目前不负责动态分配。
        plans: list[SourcePlan] = []
        for source_type in request.boundary.allowed_sources:
            source_filter = request.boundary.source_filters[source_type]
            plans.append(
                SourcePlan(
                    source_type=source_type,
                    source_filter=source_filter,
                    dependency=request.boundary.dependency_for(source_type),
                    methods=self._default_methods,
                )
            )
        return RetrievalPlan(source_plans=tuple(plans))
