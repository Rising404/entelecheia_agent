"""整 Task 交付验证的已配置结构化 Provider。"""

from __future__ import annotations

import json

from personagraph.l2.task_graph import TaskDeliveryValidationDimension
from personagraph.model_io.gateway import (
    ModelResult,
    complete_structured,
    prepare_complete_structured,
)
from .model_contracts import TaskDeliveryValidationStructuredProvider

from personagraph.model_io.tier_bindings import ModelTier, effective_model_tier_binding
from personagraph.l2.model_output_budgets import L2_MAX_OUTPUT_TOKENS

def build_task_delivery_validation_structured_provider(
    *,
    max_output_tokens: int = L2_MAX_OUTPUT_TOKENS,
    timeout_s: float = 300.0,
) -> TaskDeliveryValidationStructuredProvider:
    if not 1 <= max_output_tokens <= L2_MAX_OUTPUT_TOKENS:
        raise ValueError("Task delivery validation token limit is invalid")
    if timeout_s <= 0:
        raise ValueError("Task delivery validation timeout must be positive")

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
            mock_payload=lambda: _mock_pass(user_content),
            max_tokens=max_output_tokens,
            timeout_s=timeout_s,
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
            mock_payload=lambda: _mock_pass(user_content),
            max_tokens=max_output_tokens,
            timeout_s=timeout_s,
            json_mode=True,
            purpose=purpose,
            binding=binding,
            repair_messages=repair_messages,
        )

    provider.prepare = prepare  # type: ignore[attr-defined]
    return provider


def _mock_pass(user_content: str) -> dict[str, object]:
    payload = json.loads(user_content)
    if not isinstance(payload, dict):
        raise ValueError("Task delivery validation mock input must be an object")
    nodes = payload.get("nodes")
    anchors = payload.get("source_anchors")
    if not isinstance(nodes, list) or not nodes or not isinstance(anchors, list):
        raise ValueError("Task delivery validation mock input is incomplete")
    findings = [
        {
            "dimension": dimension.value,
            "verdict": "pass",
            "fault_domain": "none",
            "finding": f"{dimension.value} passed the frozen mock review.",
            "affected_node_ids": [],
            "evidence_anchor_ids": [],
        }
        for dimension in TaskDeliveryValidationDimension
    ]
    return {
        "findings": findings,
        "summary": "The mutable root candidate passed whole-Task mock review.",
        "execution_repair_objective": None,
        "task_graph_revision_objective": None,
        "blocking_questions": [],
    }


__all__ = ["build_task_delivery_validation_structured_provider"]
