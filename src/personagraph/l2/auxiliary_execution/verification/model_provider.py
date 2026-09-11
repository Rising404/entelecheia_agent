"""为 Auxiliary TaskGraph 语义审查配置结构化 Provider 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityClass,
    TaskGraphSemanticVerificationDimension,
)
from personagraph.model_io.gateway import ModelResult, complete_structured, prepare_complete_structured
from .task_graph_semantic import (
    TaskGraphSemanticVerificationStructuredProvider,
)

from personagraph.model_io.tier_bindings import ModelTier, effective_model_tier_binding
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS



@dataclass(frozen=True, slots=True)
class AuxiliarySemanticModelProfile:
    max_output_tokens: int = L2_MAX_OUTPUT_TOKENS
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_output_tokens <= L2_MAX_OUTPUT_TOKENS:
            raise ValueError(
                "semantic reviewer max_output_tokens must be within "
                f"1..{L2_MAX_OUTPUT_TOKENS}"
            )
        if self.timeout_s <= 0:
            raise ValueError("semantic reviewer timeout_s must be positive")


def build_auxiliary_semantic_structured_provider(
    profile: AuxiliarySemanticModelProfile = (
        AuxiliarySemanticModelProfile()
    ),
) -> TaskGraphSemanticVerificationStructuredProvider:
    """将语义审查绑定到 Entelecheia 的配置结构化网关。"""

    if not isinstance(profile, AuxiliarySemanticModelProfile):
        raise TypeError("profile must be AuxiliarySemanticModelProfile")

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        binding = effective_model_tier_binding(ModelTier.FINAL_GATE)
        return complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: build_mock_auxiliary_semantic_result(
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
        binding = effective_model_tier_binding(ModelTier.FINAL_GATE)
        return prepare_complete_structured(
            system_prompt,
            user_content,
            mock_payload=lambda: build_mock_auxiliary_semantic_result(
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


def build_mock_auxiliary_semantic_result(
    user_content: str,
) -> dict[str, Any]:
    """构建一个离线的完整维度审查，基于精确的提示构建。

    模拟模式是一个编排框架，不是质量声明。它镜像了完整的节点/证据/空白命名空间，以确定性语义保护和持久化重放来执行相同的合同，就像一个配置的真实模型一样。
    """

    try:
        payload = json.loads(user_content)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("semantic mock input must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic mock input must be a JSON object")
    authority = _mapping(payload.get("authority"), label="authority")
    proposal = _mapping(
        payload.get("task_graph_proposal"),
        label="task_graph_proposal",
    )
    root = _mapping(proposal.get("root"), label="task_graph_proposal.root")
    nodes = tuple(
        _mapping(item, label="task_graph_proposal.root.nodes[]")
        for item in _list(root.get("nodes"), label="task_graph_proposal.root.nodes")
    )
    if not nodes:
        raise ValueError("semantic mock requires at least one TaskGraph node")
    node_keys = tuple(
        sorted(_string(item.get("node_key"), label="node_key") for item in nodes)
    )
    if len(node_keys) != len(set(node_keys)):
        raise ValueError("semantic mock TaskGraph node keys must be unique")

    evidence_aliases = {
        _string(card.get("alias"), label="authority.cards[].alias")
        for card in (
            _mapping(item, label="authority.cards[]")
            for item in _list(authority.get("cards"), label="authority.cards")
        )
        if card.get("authority_class")
        == PlanningAuthorityClass.EVIDENCE.value
    }
    referenced_aliases: set[str] = set()
    for node in nodes:
        referenced_aliases.update(
            _string_list(
                node.get("source_anchor_ids"),
                label="node.source_anchor_ids",
            )
        )
        for acceptance in (
            _mapping(item, label="node.acceptance_criteria[]")
            for item in _list(
                node.get("acceptance_criteria"),
                label="node.acceptance_criteria",
            )
        ):
            referenced_aliases.update(
                _string_list(
                    acceptance.get("source_anchor_ids"),
                    label="acceptance.source_anchor_ids",
                )
            )
    required_evidence = tuple(sorted(evidence_aliases & referenced_aliases))

    gaps: list[tuple[str, bool]] = []
    for artifact in (
        _mapping(item, label="context_artifacts[]")
        for item in _list(payload.get("context_artifacts"), label="context_artifacts")
    ):
        for gap in (
            _mapping(item, label="context_artifact.gaps[]")
            for item in _list(
                artifact.get("gaps"),
                label="context_artifact.gaps",
            )
        ):
            blocking = gap.get("blocking")
            if not isinstance(blocking, bool):
                raise ValueError("semantic mock gap blocking must be a boolean")
            gaps.append(
                (
                    _string(gap.get("gap_alias"), label="gap_alias"),
                    blocking,
                )
            )
    gap_aliases = [alias for alias, _blocking in gaps]
    if len(gap_aliases) != len(set(gap_aliases)):
        raise ValueError("semantic mock gap aliases must be unique")
    has_blocking_gap = any(blocking for _alias, blocking in gaps)

    return {
        "items": [
            {
                "dimension": dimension.value,
                "verdict": (
                    "insufficient_evidence"
                    if has_blocking_gap
                    and dimension
                    is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                    else "pass"
                ),
                "failure_scope": (
                    "missing_authority"
                    if has_blocking_gap
                    and dimension
                    is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                    else None
                ),
                "finding": (
                    "Blocking typed gaps require additional evidence."
                    if has_blocking_gap
                    and dimension
                    is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                    else f"{dimension.value} passed the frozen mock review."
                ),
                "affected_node_keys": list(node_keys),
                "evidence_aliases": (
                    list(required_evidence)
                    if dimension
                    is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                    else []
                ),
                "gap_aliases": (
                    sorted(gap_aliases)
                    if dimension
                    is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                    else []
                ),
            }
            for dimension in TaskGraphSemanticVerificationDimension
        ]
    }


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"semantic mock {label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"semantic mock {label} must be an array")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"semantic mock {label} must be a non-empty string")
    return value


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(_string(item, label=f"{label}[]") for item in _list(value, label=label))


__all__ = [
    "AuxiliarySemanticModelProfile",
    "build_auxiliary_semantic_structured_provider",
    "build_mock_auxiliary_semantic_result",
]
