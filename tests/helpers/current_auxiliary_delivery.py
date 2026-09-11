"""Current Auxiliary TaskGraph delivery fixtures.

These helpers deliberately exercise the production graph-commit and whole-Task
candidate-settlement path.  They must not synthesize commit receipts or PASS
settlements directly in SQLite.
"""

from __future__ import annotations

import json
from typing import Literal

import pytest

from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryResult,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.session import store
from personagraph.l2.task_graph import TaskDeliveryValidationDimension
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_facts,
)
from personagraph.model_io.gateway import ModelResult
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.runtime.test_auxiliary_task_delivery_composition import (
    _commit_task_graph,
)


CandidateRoute = Literal["pass", "replan_task_graph"]


def _candidate_payload(
    *,
    route: CandidateRoute,
    root_node_id: str,
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for dimension in TaskDeliveryValidationDimension:
        is_gap = dimension is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
        if route == "pass" or not is_gap:
            verdict = "pass"
            fault_domain = "none"
            affected_node_ids: list[str] = []
        else:
            verdict = "fail"
            fault_domain = "task_graph_design"
            affected_node_ids = [root_node_id]
        findings.append(
            {
                "dimension": dimension.value,
                "verdict": verdict,
                "fault_domain": fault_domain,
                "finding": f"{dimension.value}: {route}",
                "affected_node_ids": affected_node_ids,
                "evidence_anchor_ids": [],
            }
        )
    return {
        "findings": findings,
        "summary": f"candidate route: {route}",
        "execution_repair_objective": None,
        "task_graph_revision_objective": (
            "重建遗漏必要子任务的任务图。"
            if route == "replan_task_graph"
            else None
        ),
        "blocking_questions": [],
    }


def settle_current_auxiliary_candidate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: CandidateRoute,
) -> tuple[str, str, str, AuxiliaryTaskDeliveryResult]:
    """Commit a current TaskGraph and settle its root candidate through runtime."""

    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    configured_provider, configured_model = configured_structured_model_facts()

    def candidate_provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(route=route, root_node_id=task_id),
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    result = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            task_candidate_validation_provider=as_prepared_test_provider(
                candidate_provider
            ),
        ),
    )
    return session_id, turn_id, task_id, result
