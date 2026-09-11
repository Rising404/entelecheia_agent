"""由新 Runtime entry 路径持有的可信输入 policy。

本模块有意不依赖旧版代码。``ingress_adapter`` 仍会读取旧 graph 的 LangGraph
checkpoint 与旧版 operation receipt；从新 entry 导入它既违反 `062` §2.4 的硬切换，
也会将 ``langgraph`` 拉入新路径的导入图。

此处全部是基于已可信 host 事实的纯 policy：能力上限、权威控制快照，以及为一次
entry 模型调用有界组装先前对话材料。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....context_budget.token_counter import estimate_tokens
from .contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    IngressDisposition,
)
from ...turn_events import RuntimeErrorCode


# 一次 entry 模型调用的 system prompt、已组装上下文和自身有界输出，都必须装入
# 确定性 guard 保护的窗口。L2 文档/附件回答可能合理地需要超过历史 800-token
# provider 默认值。通过预留 ``entry.response.model.generate_response`` 使用的相同最大输出
# 配额，保证输入组装器如实计算。
ENTRY_PROMPT_RESERVE_TOKENS = 4096

# 061 §8.5 尚未冻结 summary 和 history 的优先级。在此之前，summary
# 获得固定份额，避免它抹去所有近期 turn。附件只进入有界元数据 manifest，
# 内容预算由后续 candidate Tool 调用承担。
ENTRY_SUMMARY_MAX_SHARE = 0.25

# 061 §3.3：在 token 上限之外，最多保留五个已提交 pair。
ENTRY_MAX_HISTORY_PAIRS = 5

@dataclass(frozen=True, slots=True)
class EntryIngressShortCircuitOutcome:
    """Entry 之下 ingress 停止的现有正式回复事实。"""

    error_code: RuntimeErrorCode
    reply: str


def select_entry_ingress_short_circuit_outcome(
    disposition: object,
) -> EntryIngressShortCircuitOutcome:
    """将不可路由 disposition 映射到 Entry 现有公开回复事实。

    ``disposition`` 有意保持为 ``object`` 而非封闭 enum 注解：Entry 历来将两个已知
    enum 标识之外的所有值视为通用不支持输入回退。为 adapter 与未来 enum 扩展保留
    这项兼容规则。
    """

    if disposition is IngressDisposition.OVERFLOW:
        return EntryIngressShortCircuitOutcome(
            error_code=RuntimeErrorCode.INPUT_INVALID,
            reply="输入超过当前安全上限，请缩短后重试。",
        )
    if disposition is IngressDisposition.REJECT:
        return EntryIngressShortCircuitOutcome(
            error_code=RuntimeErrorCode.INGRESS_REJECTED,
            reply="输入为空或格式无效，请补充后重试。",
        )
    return EntryIngressShortCircuitOutcome(
        error_code=RuntimeErrorCode.INGRESS_REJECTED,
        reply="当前版本还不支持该输入类型。",
    )


def build_entry_runtime_snapshot(session_id: str | None) -> AuthoritativeRuntimeSnapshot:
    """读取当前 Runtime 持有的权威控制事实。

    在 `insession_task`/WorkRun 持久化落地前，没有可报告的 Runtime 控制状态，
    因此空快照是当前权威结果。
    """

    del session_id  # 由 WorkRun 切片恢复，并读取新表。
    return AuthoritativeRuntimeSnapshot()


def entry_capability_ceiling(features: Mapping[str, Any]) -> CapabilityCeiling:
    """推导 host policy 上限；自由文本和模型输出绝不参与。"""

    allow_tools = bool(features.get("tools_loop", True))
    return CapabilityCeiling(
        allow_model=True,
        allow_tools=allow_tools,
        # Session history、任务状态与项目文件输出属于核心产品持久化。具体写入仍须通过
        # 此 host 上限之下的工具 policy、路径权威、批准与幂等 guard。
        allow_protected_writes=allow_tools,
        allow_persistence=True,
        # 当前 registry 可以暴露只读网络工具。这只是能力上限，不构成操作授权，
        # 也不表示该工具已挂载。
        allow_external_network=allow_tools,
        allow_delegation=False,
    )


def select_session_summary(
    summary: str | None,
    *,
    context_hard_limit: int,
    user_text_tokens: int,
) -> str | None:
    """仅在为近期 turn 留有空间时保留持久化 summary。

    061 §4 要求 summary 问题绝不能阻塞对话，因此会丢弃超大 summary，而非允许其
    耗尽预算。
    """

    if not summary:
        return None
    available = context_hard_limit - user_text_tokens - ENTRY_PROMPT_RESERVE_TOKENS
    if available <= 0:
        return None
    if estimate_tokens(summary) > int(available * ENTRY_SUMMARY_MAX_SHARE):
        return None
    return summary


def entry_history_budget(
    *,
    context_hard_limit: int,
    user_text_tokens: int,
    summary_tokens: int,
    attachment_tokens: int = 0,
) -> int:
    """本次 entry 模型调用仍可用于先前 turn 的 token 数。"""

    return max(
        0,
        context_hard_limit
        - user_text_tokens
        - summary_tokens
        - attachment_tokens
        - ENTRY_PROMPT_RESERVE_TOKENS,
    )


def entry_task_catalog_budget(
    *,
    context_hard_limit: int,
    user_text_tokens: int,
    summary_tokens: int,
    attachment_tokens: int,
    history_tokens: int,
) -> int:
    """只将未使用的全局输入预算用于动态 Task catalog。

    此处有意不虚构 catalog 专用产品上限。现有硬上下文限制是唯一上限；省略会作为
    可信 ``truncated`` 事实返回，并强制采用保守 L2 路由。
    """

    return max(
        0,
        context_hard_limit
        - user_text_tokens
        - summary_tokens
        - attachment_tokens
        - history_tokens
        - ENTRY_PROMPT_RESERVE_TOKENS,
    )


def select_history_within_budget(
    pairs: Sequence[Mapping[str, Any]],
    *,
    budget_tokens: int,
    max_pairs: int = ENTRY_MAX_HISTORY_PAIRS,
) -> tuple[dict[str, str], ...]:
    """按 061 §3 组装近期已提交 pair。

    pair 按最新优先添加且绝不拆分：若下一个完整 pair 放不下，组装会立即停止，而不
    跳过它寻找更小的旧 pair，从而使投影 history 保持连续。
    """

    selected: list[dict[str, str]] = []
    remaining = max(0, budget_tokens)
    for pair in reversed(list(pairs)):
        if len(selected) >= max_pairs:
            break
        user_content = str(pair.get("user_content") or "")
        assistant_content = str(pair.get("assistant_content") or "")
        cost = estimate_tokens(user_content) + estimate_tokens(assistant_content)
        if cost > remaining:
            break
        remaining -= cost
        selected.append({"user": user_content, "assistant": assistant_content})
    selected.reverse()
    return tuple(selected)
