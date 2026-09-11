"""应用功能门的类型合同。"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class FeatureFlags(TypedDict):
    user_interaction_mode: NotRequired[Literal["interactive", "closed_world"]]
    model_control_transport: NotRequired[Literal["auto", "native", "prompt_json"]]
    # L1 是否可见并可调度公共 Web 搜索/抓取工具；
    # 默认产品 GUI 保持开启。
    l1_external_web_tools_enabled: NotRequired[bool]
    l1_max_attempts: NotRequired[int]
    l1_max_tool_calls_per_attempt: NotRequired[int]
    l1_semantic_verification_mode: NotRequired[
        Literal["off", "conditional", "always"]
    ]
    execution_findings_enabled: NotRequired[bool]
    turn_wall_clock_budget_s: NotRequired[float]
    file_retrieval_write_enabled: NotRequired[bool]
    file_retrieval_read_enabled: NotRequired[bool]
    history_retrieval_write_enabled: NotRequired[bool]
    history_retrieval_read_enabled: NotRequired[bool]
    l1_retrieval_tools_enabled: NotRequired[bool]
    l2_aux_retrieval_tools_enabled: NotRequired[bool]
    l2_task_retrieval_tools_enabled: NotRequired[bool]
    session_context_repair_apply_enabled: NotRequired[bool]  # API 受控替换的紧急停用开关
    tools_loop: bool               # 多轮工具环路（模型驱动调用）
    context_guard_limit: NotRequired[int]  # C2：provider envelope 的硬准入上限
