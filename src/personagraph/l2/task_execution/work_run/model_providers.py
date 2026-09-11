"""WorkRun 运行时的生产形态结构化模型适配器。

纯 Attempt 与 Verification 端口刻意不选择提供商限制。本模块是将这些端口绑定到 Entelecheia
已配置结构化补全网关的狭窄组装边界。Mock 模式仍适用于离线端到端演示，但其回复由精确提示词
载荷推导，而不是来自第二条运行时路径。
"""

from __future__ import annotations

import json
from typing import Any

from ....persistent_turn_content.evidence import (
    EmptySupportJustification,
    EmptySupportReason,
)
from ....model_io.gateway import ModelResult, complete_structured, prepare_complete_structured
from ..attempts.decision import AttemptDecisionStructuredProvider
from ..verification.decision import NodeVerificationStructuredProvider
from ....model_io.tier_bindings import ModelTier, effective_model_tier_binding
from .model_profile import WorkRunStructuredModelProfile


def build_attempt_structured_provider(
    profile: WorkRunStructuredModelProfile,
) -> AttemptDecisionStructuredProvider:
    """将 Attempt 端口绑定到已配置的模型网关。

    tier 会在每次调用时解析，而不是在构造时捕获，因此设置变更会在下一个 Turn 生效，
    而无需等到下次重启。这与该网关读取其他所有配置的行为一致。
    """

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        binding = effective_model_tier_binding(
            ModelTier.ATTEMPT,
            allowed_scoped_tiers=(ModelTier.ATTEMPT, ModelTier.ARCHITECT),
        )
        return complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: _mock_attempt_decision(user_content),
            max_tokens=profile.attempt_max_output_tokens,
            timeout_s=profile.timeout_s,
            json_mode=True,
            model_call_id=model_call_id,
            purpose=purpose,
            binding=binding,
        )

    def prepare(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ):
        binding = effective_model_tier_binding(
            ModelTier.ATTEMPT,
            allowed_scoped_tiers=(ModelTier.ATTEMPT, ModelTier.ARCHITECT),
        )
        return prepare_complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: _mock_attempt_decision(user_content),
            max_tokens=profile.attempt_max_output_tokens,
            timeout_s=profile.timeout_s,
            json_mode=True,
            purpose=purpose,
            binding=binding,
            repair_messages=repair_messages,
        )

    provider.prepare = prepare  # type: ignore[attr-defined]
    return provider


def build_verification_structured_provider(
    profile: WorkRunStructuredModelProfile,
) -> NodeVerificationStructuredProvider:
    """将语义验证端口绑定到已配置的模型网关。"""

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        binding = effective_model_tier_binding(ModelTier.NODE_VERIFICATION)
        return complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: _mock_verification(user_content),
            max_tokens=profile.verification_max_output_tokens,
            timeout_s=profile.timeout_s,
            json_mode=True,
            model_call_id=model_call_id,
            purpose=purpose,
            binding=binding,
        )

    def prepare(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ):
        binding = effective_model_tier_binding(ModelTier.NODE_VERIFICATION)
        return prepare_complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: _mock_verification(user_content),
            max_tokens=profile.verification_max_output_tokens,
            timeout_s=profile.timeout_s,
            json_mode=True,
            purpose=purpose,
            binding=binding,
            repair_messages=repair_messages,
        )

    provider.prepare = prepare  # type: ignore[attr-defined]
    return provider


def _mock_attempt_decision(user_content: str) -> dict[str, Any]:
    payload = _require_object(user_content, purpose="Attempt")
    node = _require_mapping(payload.get("node"), name="node")
    acceptances = _require_list(node.get("acceptances"), name="node.acceptances")
    acceptance_updates = _mock_completed_acceptance_updates(
        acceptances,
        reason_code=EmptySupportReason.CANDIDATE_IS_PRIMARY_ARTIFACT,
        explanation=(
            "The deterministic mock candidate is the primary artifact under "
            "review, so it has no external ToolResult to cite."
        ),
    )

    user_input = payload.get("user_input")
    current_input = (
        user_input.get("content")
        if isinstance(user_input, dict)
        else payload.get("current_user_input")
    )
    if not isinstance(current_input, str) or not current_input.strip():
        current_input = str(node.get("objective") or node.get("title") or "已完成当前任务")
    raw_task_graph_contract = payload.get("task_graph_proposal_contract")
    if raw_task_graph_contract is not None:
        task_graph_contract = _require_mapping(
            raw_task_graph_contract,
            name="task_graph_proposal_contract",
        )
        allowed_source_anchor_ids = _require_string_list(
            task_graph_contract.get("allowed_source_anchor_ids"),
            name="task_graph_proposal_contract.allowed_source_anchor_ids",
        )
        required_source_anchor_ids = _require_string_list(
            task_graph_contract.get("required_source_anchor_ids"),
            name="task_graph_proposal_contract.required_source_anchor_ids",
        )
        if (
            not allowed_source_anchor_ids
            or not required_source_anchor_ids
            or not set(required_source_anchor_ids).issubset(
                allowed_source_anchor_ids
            )
        ):
            raise ValueError(
                "TaskGraph mock requires valid allowed/required source anchors"
            )
        title = _required_string(node, "title")
        objective = _required_string(node, "objective")
        action = {
            "kind": "submit_task_graph",
            "proposal": {
                "schema_version": "insession-task-graph-revision-v2",
                "root": {
                    "root_key": "root",
                    "nodes": [
                        {
                            "node_key": "root",
                            "node_kind": "root",
                            "title": title,
                            "objective": objective,
                            "source_anchor_ids": allowed_source_anchor_ids,
                            "acceptance_criteria": [
                                {
                                    "acceptance_id": "root_complete",
                                    "criterion": (
                                        "交付内容完整满足当前根任务目标"
                                    ),
                                    # 确定性的终态守卫会将节点使用的每个证据锚点提升为
                                    # 必需的 Acceptance 覆盖范围。让离线提供商忠实遵守该生产契约。
                                    "source_anchor_ids": allowed_source_anchor_ids,
                                }
                            ],
                        }
                    ],
                },
            },
        }
        raw_revision_base = payload.get("task_graph_revision_base")
        if raw_revision_base is not None:
            revision_base = _require_mapping(
                raw_revision_base,
                name="task_graph_revision_base",
            )
            base_nodes = [
                _require_mapping(item, name="task_graph_revision_base.nodes[]")
                for item in _require_list(
                    revision_base.get("nodes"),
                    name="task_graph_revision_base.nodes",
                )
            ]
            base_roots = [
                item
                for item in base_nodes
                if item.get("parent_node_alias") is None
            ]
            if len(base_roots) != 1:
                raise ValueError(
                    "TaskGraph revision mock requires exactly one base root"
                )
            base_root_alias = _required_string(
                base_roots[0],
                "node_alias",
            )
            proposal_root = action["proposal"]["root"]["nodes"][0]
            proposal_root["acceptance_criteria"][0]["criterion"] = (
                "交付内容完整满足当前根任务目标，并明确修复整体验证失败。"
            )
            action["lineage"] = [
                {
                    "proposal_node_key": proposal_root["node_key"],
                    "disposition": "revise",
                    "base_node_alias": base_root_alias,
                }
            ]
    else:
        raw_dependencies = payload.get("dependency_deliveries")
        dependencies = (
            [
                _require_mapping(item, name="dependency_delivery")
                for item in raw_dependencies
            ]
            if isinstance(raw_dependencies, list)
            else []
        )
        if dependencies:
            acceptance_updates = _mock_completed_acceptance_updates(
                acceptances,
                reason_code=(
                    EmptySupportReason.DEPENDENCY_DELIVERY_SUFFICIENT
                ),
                explanation=(
                    "The deterministic synthesis is based on the verified "
                    "dependency deliveries supplied by the Host rather than "
                    "ordinary ToolResult IDs."
                ),
            )
            child_sections = []
            for dependency in dependencies:
                output = _require_mapping(
                    dependency.get("output_window"),
                    name="dependency_delivery.output_window",
                )
                child_sections.append(_required_string(output, "content"))
            content = (
                "# 论文证据综合（确定性链路演示）\n\n"
                "以下内容严格来自已通过验证的子节点 Delivery；"
                "离线 mock 不宣称新增语义判断。\n\n"
                + "\n\n---\n\n".join(child_sections)
            )
        else:
            content = (
                "## Entelecheia WorkRun 演示结果\n\n"
                f"{current_input.strip()}\n\n"
                "此回复由离线 mock WorkRun 完整经过 OutputWindow 与语义验证后交付。"
            )
        action = {
            "kind": "submit_output_window",
            "content": content,
            "format": "markdown",
        }
    return {
        "acceptance_updates": acceptance_updates,
        "action": action,
    }


def _mock_completed_acceptance_updates(
    acceptances: list[Any],
    *,
    reason_code: EmptySupportReason,
    explanation: str,
) -> list[dict[str, Any]]:
    """仅为离线 mock 构建符合当前契约的完成声明。"""

    justification = EmptySupportJustification(
        reason_code=reason_code,
        explanation=explanation,
    ).model_dump(mode="json")
    return [
        {
            "acceptance_id": _required_string(
                _require_mapping(item, name="acceptance"),
                "acceptance_id",
            ),
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
            "empty_support_justification": justification,
        }
        for item in acceptances
    ]


def _mock_verification(user_content: str) -> dict[str, Any]:
    payload = _require_object(user_content, purpose="Verification")
    node = _require_mapping(payload.get("node"), name="node")
    acceptances = _require_list(node.get("acceptances"), name="node.acceptances")
    response: dict[str, Any] = {
        "acceptance_results": [
            {
                "acceptance_id": _required_string(
                    _require_mapping(value, name="acceptance"),
                    "acceptance_id",
                ),
                "verdict": "passed",
                "finding": "离线演示验证器确认锁定正文覆盖该验收条件。",
                "missing_requirements": [],
            }
            for value in acceptances
        ]
    }
    return response


def _require_object(serialized: str, *, purpose: str) -> dict[str, Any]:
    try:
        value = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{purpose} prompt payload is not JSON") from exc
    return _require_mapping(value, name=f"{purpose} payload")


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return value


def _require_string_list(value: Any, *, name: str) -> list[str]:
    items = _require_list(value, name=name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return items


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item


__all__ = [
    'WorkRunStructuredModelProfile',
    "build_attempt_structured_provider",
    "build_verification_structured_provider",
]
