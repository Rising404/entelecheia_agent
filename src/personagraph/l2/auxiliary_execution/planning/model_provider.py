"""配置了结构化的 Provider 适配器用于 图 Architect。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphRevisionProposalDisposition,
    AuxiliaryGraphRevisionProposal,
    AuxiliaryGraphRevisionReason,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
)
from personagraph.l2.task_graph import InSessionTaskAcceptanceProposal
from personagraph.model_io.gateway import ModelResult, complete_structured, prepare_complete_structured
from .architect import AuxiliaryGraphArchitectStructuredProvider
from .profiles import (
    MODEL_ANALYSIS_CAPABILITY,
    MOUNTED_DOCUMENT_READ_CAPABILITY,
    MOUNTED_VISUAL_READ_CAPABILITY,
)

from personagraph.model_io.tier_bindings import ModelTier, effective_model_tier_binding
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS



@dataclass(frozen=True, slots=True)
class AuxiliaryArchitectModelProfile:
    max_output_tokens: int = L2_MAX_OUTPUT_TOKENS
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_output_tokens <= L2_MAX_OUTPUT_TOKENS:
            raise ValueError(
                "Architect max_output_tokens must be within "
                f"1..{L2_MAX_OUTPUT_TOKENS}"
            )
        if self.timeout_s <= 0:
            raise ValueError("Architect timeout_s must be positive")


def build_auxiliary_architect_structured_provider(
    profile: AuxiliaryArchitectModelProfile = (
        AuxiliaryArchitectModelProfile()
    ),
) -> AuxiliaryGraphArchitectStructuredProvider:
    """将类型化的 Architect 端口绑定到 Entelecheia 配置的网关。"""

    if not isinstance(profile, AuxiliaryArchitectModelProfile):
        raise TypeError("profile must be AuxiliaryArchitectModelProfile")

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        binding = effective_model_tier_binding(ModelTier.ARCHITECT)
        return complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: build_mock_auxiliary_architect_proposal(
                user_content
            ),
            max_tokens=profile.max_output_tokens,
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
        binding = effective_model_tier_binding(ModelTier.ARCHITECT)
        return prepare_complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: build_mock_auxiliary_architect_proposal(
                user_content
            ),
            max_tokens=profile.max_output_tokens,
            timeout_s=profile.timeout_s,
            json_mode=True,
            purpose=purpose,
            binding=binding,
            repair_messages=repair_messages,
        )

    provider.prepare = prepare  # type: ignore[attr-defined]
    return provider


def build_mock_auxiliary_architect_proposal(
    user_content: str,
) -> dict[str, Any]:
    """从精确提示派生一个离线提案，而不是一个固定图。

    模拟模式故意保持简朴：每个挂载文档一个有限的 Host 感知节点，一个无工具分析节点，然后是终端规划器。它锻炼了真实的多节点持久化/执行路径，但不就生产模型质量做出语义声明。
    """

    try:
        payload = json.loads(user_content)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Architect mock input must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Architect mock input must be a JSON object")
    goal = _mapping(payload.get("goal"), label="goal")
    authority = _mapping(payload.get("authority"), label="authority")
    catalog = _mapping(payload.get("capabilities"), label="capabilities")
    current = _mapping(payload.get("current_revision"), label="current_revision")
    authorization_aliases = _string_tuple(
        goal.get("authorization_aliases"),
        label="goal.authorization_aliases",
    )
    if not authorization_aliases:
        raise ValueError("Architect mock requires authorization authority")
    cards = _list(authority.get("cards"), label="authority.cards")
    document_aliases = tuple(
        sorted(
            str(card["alias"])
            for card in cards
            if isinstance(card, dict)
            and card.get("authority_class") == "evidence"
            and card.get("source_kind") == "document"
            and isinstance(card.get("alias"), str)
        )
    )
    visual_aliases = tuple(
        sorted(
            str(card["alias"])
            for card in cards
            if isinstance(card, dict)
            and card.get("authority_class") == "evidence"
            and card.get("source_kind") == "visual"
            and isinstance(card.get("alias"), str)
        )
    )
    available_capabilities = {
        str(item["capability_alias"])
        for item in _list(catalog.get("capabilities"), label="capabilities")
        if isinstance(item, dict)
        and item.get("available") is True
        and isinstance(item.get("capability_alias"), str)
    }
    if MODEL_ANALYSIS_CAPABILITY not in available_capabilities:
        raise ValueError("Architect mock requires the model-analysis capability")
    if document_aliases and (
        MOUNTED_DOCUMENT_READ_CAPABILITY not in available_capabilities
    ):
        raise ValueError("mounted Documents have no available read capability")
    if visual_aliases and (
        MOUNTED_VISUAL_READ_CAPABILITY not in available_capabilities
    ):
        raise ValueError("mounted visual units have no available read capability")

    current_structure = _mapping(
        current.get("structure"), label="current_revision.structure"
    )
    current_revision = current.get("auxiliary_graph_revision")
    if isinstance(current_revision, bool) or not isinstance(current_revision, int):
        raise ValueError("Architect mock requires a positive current revision")
    terminal_origin = current_structure.get("terminal_node_key")
    if not isinstance(terminal_origin, str) or not terminal_origin:
        raise ValueError("Architect mock current terminal alias is invalid")
    current_nodes = _list(
        current_structure.get("nodes"),
        label="current_revision.structure.nodes",
    )
    current_node_keys = {
        str(node["node_key"])
        for node in current_nodes
        if isinstance(node, dict) and isinstance(node.get("node_key"), str)
    }
    raw_replan_trigger = payload.get("replan_trigger")
    if raw_replan_trigger is not None and not isinstance(raw_replan_trigger, dict):
        raise ValueError("Architect mock replan_trigger must be an object or null")
    raw_task_graph_trigger = payload.get("task_graph_revision_trigger")
    if raw_task_graph_trigger is not None and not isinstance(
        raw_task_graph_trigger, dict
    ):
        raise ValueError(
            "Architect mock task_graph_revision_trigger must be an object or null"
        )
    raw_task_graph_route = payload.get("task_graph_revision_route")
    if raw_task_graph_route is not None and not isinstance(
        raw_task_graph_route,
        dict,
    ):
        raise ValueError(
            "Architect mock task_graph_revision_route must be an object or null"
        )
    verification_replan = (
        raw_replan_trigger is not None or raw_task_graph_trigger is not None
    )
    semantic_blocked_replan = bool(
        raw_replan_trigger is not None
        and raw_replan_trigger.get("semantic_host_disposition") == "blocked"
    )
    task_graph_blocked_replan = bool(
        raw_task_graph_route is not None
        and raw_task_graph_route.get("requires_user_gate") is True
    )
    blocked_replan = semantic_blocked_replan or task_graph_blocked_replan
    if raw_replan_trigger is not None:
        # 语义重规划的权威状态还携带供审阅结论使用的观察卡片。
        # 这些卡片是证据，而不是已挂载资源：若把每个具有文档形态的观察
        # 都当作一次新的 Host 读取，不仅会虚构无效的资源别名，
        # 还可能在实际 Provider 尚未分发前就超出单节点来源限制。
        # 此处仅允许执行已绑定到当前图中的资源。
        current_resource_aliases = {
            alias
            for node in current_nodes
            if isinstance(node, dict)
            for alias in _string_tuple(
                node.get("input_resource_aliases"),
                label="current_revision.node.input_resource_aliases",
            )
        }
        document_aliases = tuple(
            alias for alias in document_aliases if alias in current_resource_aliases
        )
        visual_aliases = tuple(
            alias for alias in visual_aliases if alias in current_resource_aliases
        )

    def replan_origin(node_key: str) -> str | None:
        return node_key if verification_replan and node_key in current_node_keys else None

    all_sources = tuple(
        sorted(
            {
                *authorization_aliases,
                *document_aliases,
                *visual_aliases,
            }
        )
    )
    nodes: list[AuxiliaryNodeProposal] = []
    for ordinal, document_alias in enumerate(document_aliases, start=1):
        suffix = f"{ordinal:02d}"
        node_sources = tuple(sorted({*authorization_aliases, document_alias}))
        nodes.append(
            AuxiliaryNodeProposal(
                node_key=f"observe_document_{suffix}",
                node_kind=AuxiliaryNodeKind.OBSERVE,
                executor_kind=AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
                title=f"查阅已挂载的文档 {ordinal}",
                objective=(
                    "读取被冻结的那一份文档资源，产出有来源支撑的规划上下文"
                    "产物。"
                ),
                acceptance_criteria=(
                    InSessionTaskAcceptanceProposal(
                        acceptance_id=f"document_{suffix}_grounded",
                        criterion=(
                            "文档观察结果记录了可读证据、覆盖范围，以及明确列出的"
                            "遗留缺口。"
                        ),
                        source_anchor_ids=node_sources,
                    ),
                ),
                capability_profile_id=MOUNTED_DOCUMENT_READ_CAPABILITY,
                input_resource_aliases=(document_alias,),
                source_anchor_ids=node_sources,
                output_contract="planning_context_artifact_v1",
                origin_node_alias=replan_origin(f"observe_document_{suffix}"),
            )
        )

    for ordinal, visual_alias in enumerate(visual_aliases, start=1):
        suffix = f"{ordinal:03d}"
        node_sources = tuple(sorted({*authorization_aliases, visual_alias}))
        nodes.append(
            AuxiliaryNodeProposal(
                node_key=f"observe_visual_{suffix}",
                node_kind=AuxiliaryNodeKind.OBSERVE,
                executor_kind=AuxiliaryNodeExecutorKind.HOST_PRIMITIVE,
                title=f"查阅已挂载的视觉单元 {ordinal}",
                objective=(
                    "经由 Host 的视觉边界解析被冻结的那一个视觉单元，并保留披露"
                    "或供应方层面的缺口。"
                ),
                acceptance_criteria=(
                    InSessionTaskAcceptanceProposal(
                        acceptance_id=f"visual_{suffix}_grounded",
                        criterion=(
                            "视觉观察结果记录了受约束的语义证据，或一个明确的、未"
                            "解决的定类缺口。"
                        ),
                        source_anchor_ids=node_sources,
                    ),
                ),
                capability_profile_id=MOUNTED_VISUAL_READ_CAPABILITY,
                input_resource_aliases=(visual_alias,),
                source_anchor_ids=node_sources,
                output_contract="planning_context_artifact_v1",
                origin_node_alias=replan_origin(f"observe_visual_{suffix}"),
            )
        )

    if blocked_replan:
        if task_graph_blocked_replan:
            questions = _string_tuple(
                raw_task_graph_route.get("blocking_questions"),
                label="task_graph_revision_route.blocking_questions",
            )
            if not questions:
                raise ValueError(
                    "TaskGraph clarification route requires blocking questions"
                )
        else:
            questions = (
                "Obtain the missing user-authorized information identified "
                "by the frozen semantic verification findings.",
            )
        for ordinal, question in enumerate(questions, start=1):
            suffix = f"{ordinal:02d}"
            user_gate_key = f"clarify_blocking_question_{suffix}"
            nodes.append(
                AuxiliaryNodeProposal(
                    node_key=user_gate_key,
                    node_kind=AuxiliaryNodeKind.CLARIFY,
                    executor_kind=AuxiliaryNodeExecutorKind.USER_GATE,
                    title=f"澄清阻塞信息 {ordinal}",
                    objective=question,
                    acceptance_criteria=(
                        InSessionTaskAcceptanceProposal(
                            acceptance_id=(
                                f"blocking_question_{suffix}_clarified"
                            ),
                            criterion=(
                                "用户的回答解决了这个阻塞性的语义证据缺口，或明确"
                                "地把它保留了下来。"
                            ),
                            source_anchor_ids=authorization_aliases,
                        ),
                    ),
                    input_resource_aliases=(),
                    source_anchor_ids=authorization_aliases,
                    output_contract="user_response_v1",
                    origin_node_alias=replan_origin(user_gate_key),
                )
            )

    analysis_key = "analyze_verified_context"
    nodes.append(
        AuxiliaryNodeProposal(
            node_key=analysis_key,
            node_kind=AuxiliaryNodeKind.ANALYZE,
            executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            title="分析已验证的规划上下文",
            objective=(
                "把已验证的上游观察结果，归拢成终结规划器所需的事实、约束与"
                "缺口。"
            ),
            acceptance_criteria=(
                InSessionTaskAcceptanceProposal(
                    acceptance_id="analysis_grounded",
                    criterion=(
                        "分析区分了有证据支撑的事实与缺失或不完整的证据，并保留了"
                        "用户的目标。"
                    ),
                    source_anchor_ids=all_sources,
                ),
            ),
            capability_profile_id=MODEL_ANALYSIS_CAPABILITY,
            input_resource_aliases=(),
            source_anchor_ids=all_sources,
            output_contract="planning_analysis_v1",
            origin_node_alias=replan_origin(analysis_key),
        )
    )
    terminal_key = "synthesize_task_graph"
    nodes.append(
        AuxiliaryNodeProposal(
            node_key=terminal_key,
            node_kind=AuxiliaryNodeKind.SYNTHESIZE,
            executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
            title="形成正式的 TaskGraph",
            objective=(
                "仅依据已验证的依赖产出与冻结的授权，产出完整可执行的 TaskGraph "
                "提案。"
            ),
            acceptance_criteria=(
                InSessionTaskAcceptanceProposal(
                    acceptance_id="task_graph_ready",
                    criterion=(
                        "该提案以可观察的验收标准和真实的依赖关系，完整覆盖了被授权"
                        "的目标。"
                    ),
                    source_anchor_ids=all_sources,
                ),
            ),
            input_resource_aliases=(),
            source_anchor_ids=all_sources,
            output_contract="task_graph_revision_proposal_v2",
            origin_node_alias=(
                replan_origin(terminal_key)
                if verification_replan
                else terminal_origin
            ),
        )
    )
    edges = [
        AuxiliaryGraphEdgeProposal(
            source_node_key=node.node_key,
            target_node_key=analysis_key,
        )
        for node in nodes[:-2]
    ]
    edges.append(
        AuxiliaryGraphEdgeProposal(
            source_node_key=analysis_key,
            target_node_key=terminal_key,
        )
    )
    proposal = AuxiliaryGraphRevisionProposal(
        disposition=AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION,
        expected_current_auxiliary_graph_revision=current_revision,
        revision_reason=(
            AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
            if verification_replan
            else AuxiliaryGraphRevisionReason.MANUAL_REPLAN
        ),
        structure=AuxiliaryGraphStructureProposal(
            terminal_node_key=terminal_key,
            nodes=tuple(nodes),
            edges=tuple(edges),
        ),
        explanation=(
            (
                "Revise the current planning graph in response to the frozen "
                "semantic verification findings."
                if verification_replan
                else "Replace the authority bootstrap with a finite perception, "
                "analysis, and terminal-synthesis graph."
            )
        ),
    )
    return proposal.model_dump(mode="json")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Architect mock {label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Architect mock {label} must be an array")
    return value


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    values = _list(value, label=label)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"Architect mock {label} contains an invalid alias")
    return tuple(values)


__all__ = [
    "AuxiliaryArchitectModelProfile",
    "build_auxiliary_architect_structured_provider",
    "build_mock_auxiliary_architect_proposal",
]
