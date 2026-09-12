from __future__ import annotations

import hashlib
import json

from tests.helpers.auxiliary_project import (
    auxiliary_project_authority,  # noqa: F401
    authorized_auxiliary_documents,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider

from personagraph.l2.auxiliary_graph import (
    AuxiliaryReplanTriggerReason,
    PlanningAuthorityClass,
    PlanningObservationStatus,
    TaskGraphSemanticVerificationDimension,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.planning.resource_perception import (
    PlanningResourceCoverage,
    PlanningResourceEvidenceKind,
    PlanningResourceEvidenceUnit,
    PlanningResourceGapReason,
    PlanningResourceReadOutcome,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from tests.runtime.test_auxiliary_planning_controller import (
    _ingest_planning_document,
    _seed_task,
    _virtual_mounted_freshness,
)
from tests.runtime.test_auxiliary_semantic_verification_controller import (
    _authority_factory,
)


class _PartialInformationReadPort:
    def read_frozen_resource(self, request):
        statement = "The available excerpt omits the user-selected comparison period."
        return PlanningResourceReadOutcome(
            status=PlanningObservationStatus.PARTIAL,
            observed_resource_version=request.resource.resource_version,
            observed_content_sha256=request.resource.content_sha256,
            observed_coverage=PlanningResourceCoverage.PARTIAL,
            evidence=(
                PlanningResourceEvidenceUnit(
                    source_unit_id="partial_information_excerpt",
                    statement=statement,
                    locator="resource:mounted_document_01#page=1&chunk=0",
                    content_sha256=hashlib.sha256(
                        statement.encode("utf-8")
                    ).hexdigest(),
                    evidence_kind=PlanningResourceEvidenceKind.DOCUMENT_TEXT,
                ),
            ),
            gap_reasons=(PlanningResourceGapReason.INCOMPLETE_COVERAGE,),
        )


class _MissingInformationSemanticProvider:
    """带类型的常规信息停止，绝不能替代授权。"""

    def __init__(self, *, failure_scope: str = "missing_information") -> None:
        self.calls: list[str] = []
        self.failure_scope = failure_scope

    def __call__(
        self,
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        payload = json.loads(user_content)
        self.calls.append(model_call_id)
        node_key = payload["task_graph_proposal"]["root"]["root_key"]
        evidence_aliases = {
            item["alias"]
            for item in payload["authority"]["cards"]
            if item["authority_class"] == PlanningAuthorityClass.EVIDENCE.value
        }
        referenced_aliases = {
            alias
            for node in payload["task_graph_proposal"]["root"]["nodes"]
            for alias in (
                *node["source_anchor_ids"],
                *(
                    source_alias
                    for acceptance in node["acceptance_criteria"]
                    for source_alias in acceptance["source_anchor_ids"]
                ),
            )
        }
        required_evidence = sorted(evidence_aliases & referenced_aliases)
        gap_aliases = sorted(
            gap["gap_alias"]
            for artifact in payload["context_artifacts"]
            for gap in artifact["gaps"]
        )
        assert gap_aliases
        items = []
        for dimension in TaskGraphSemanticVerificationDimension:
            missing_information = (
                dimension
                is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
            )
            items.append(
                {
                    "dimension": dimension.value,
                    "verdict": (
                        "insufficient_evidence" if missing_information else "pass"
                    ),
                    "failure_scope": (
                        self.failure_scope if missing_information else None
                    ),
                    "finding": (
                        "The ordinary comparison-period choice is missing from the "
                        "frozen evidence and must be clarified with the user."
                        if missing_information
                        else f"{dimension.value} passed frozen review."
                    ),
                    "affected_node_keys": [node_key],
                    "evidence_aliases": (
                        required_evidence
                        if dimension
                        is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                        else []
                    ),
                    "gap_aliases": (
                        gap_aliases if missing_information else []
                    ),
                }
            )
        return ModelResult(
            reply=json.dumps({"items": items}, ensure_ascii=False),
            provider="semantic_test_provider",
            model="semantic_test_model",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )


def test_terminal_semantic_information_blocked_replans_to_durable_user_gate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "personagraph.workspace.documents.application.check_mounted_document_freshness",
        _virtual_mounted_freshness,
    )
    session_id, turn_id, task_id = _seed_task()
    ingested_document = _ingest_planning_document(
        session_id=session_id,
        path="/private/evidence/incomplete-paper.pdf",
        title="Incomplete Paper",
        content="The paper compares two systems but does not select a time period.",
    )
    physical_planner = build_auxiliary_architect_structured_provider()
    planning_payloads: list[dict[str, object]] = []

    def planning_provider(system_prompt, user_content, **kwargs):
        planning_payloads.append(json.loads(user_content))
        return physical_planner(system_prompt, user_content, **kwargs)

    semantic_provider = _MissingInformationSemanticProvider()
    semantic_authority_factory, _bindings = _authority_factory()
    model_profile = AuxiliaryApplicationPorts(
        model_ledger_store=store,
        emit=lambda _event: None
    ).work_run_model_profile
    physical_attempt = build_attempt_structured_provider(model_profile)

    def attempt_provider(system_prompt, user_content, **kwargs):
        payload = json.loads(user_content)
        gate = payload.get("user_gate_contract")
        if gate is None:
            return physical_attempt(system_prompt, user_content, **kwargs)
        assert gate["phase"] == "ask"
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": gate["expected_question"],
                    },
                },
                ensure_ascii=False,
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=kwargs["model_call_id"],
            purpose=kwargs["purpose"],
        )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=16,
        ),
        ports=AuxiliaryApplicationPorts(
            task_document_scope=authorized_auxiliary_documents(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                document_ids=(str(ingested_document["doc_id"]),),
            ),
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(planning_provider),
            attempt_provider=as_prepared_test_provider(attempt_provider),
            semantic_provider=as_prepared_test_provider(semantic_provider),
            semantic_model_call_authority_factory=semantic_authority_factory,
            resource_read_port=_PartialInformationReadPort(),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.WAITING_USER, (
        result.reason_code,
        result.last_driver_action,
    )
    assert result.requested_user_question
    assert result.replan_result is not None
    assert result.replan_result.trigger is not None
    assert result.replan_result.architect_decision is not None
    assert (
        result.replan_result.architect_decision.proposal.requested_user_question
        is None
    )
    assert (
        result.replan_result.trigger.trigger_reason
        is AuxiliaryReplanTriggerReason.SEMANTIC_EVIDENCE_BLOCKED
    )
    trigger = result.replan_result.trigger
    settlement = semantic_store.get_auxiliary_semantic_quorum_settlement(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=trigger.auxiliary_graph_id,
        goal_id=trigger.goal_id,
        auxiliary_graph_revision=trigger.source_auxiliary_graph_revision,
        frozen_prompt_payload_sha256=trigger.semantic_prompt_payload_sha256,
    )
    assert settlement is not None
    assert settlement.settlement_id == trigger.semantic_settlement_id
    assert settlement.settlement_sha256 == trigger.semantic_settlement_sha256
    assert settlement.host_disposition.value == "blocked"
    assert len(planning_payloads) == 2
    assert planning_payloads[0]["replan_trigger"] is None
    blocked_trigger = planning_payloads[1]["replan_trigger"]
    assert isinstance(blocked_trigger, dict)
    assert blocked_trigger["semantic_host_disposition"] == "blocked"
    # 候选审查与持久法定人数稳定共享同一个逻辑调用；持久化稳定结果必须重放，
    # 而不是重新分发。
    assert len(semantic_provider.calls) == 1

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.auxiliary_graph_revision == 3
    assert details.parent_auxiliary_graph_revision == 2
    assert details.reason == "verification_failed"
    assert any(node.executor_kind == "user_gate" for node in details.nodes)
    pending = continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert pending is not None
    assert pending.question == result.requested_user_question
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None

    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision is None
    with store._connect() as conn:
        published_task_deliveries = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchone()[0]
        )
    assert published_task_deliveries == 0


def test_terminal_semantic_missing_authority_cannot_become_textual_user_gate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "personagraph.workspace.documents.application.check_mounted_document_freshness",
        _virtual_mounted_freshness,
    )
    session_id, turn_id, task_id = _seed_task()
    ingested_document = _ingest_planning_document(
        session_id=session_id,
        path="/private/evidence/authority-bound-paper.pdf",
        title="Authority-bound Paper",
        content="The protected source requires formal access authority.",
    )
    semantic_provider = _MissingInformationSemanticProvider(
        failure_scope="missing_authority"
    )
    semantic_authority_factory, _bindings = _authority_factory()

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=16,
        ),
        ports=AuxiliaryApplicationPorts(
            task_document_scope=authorized_auxiliary_documents(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                document_ids=(str(ingested_document["doc_id"]),),
            ),
            model_ledger_store=store,
            emit=lambda _event: None,
            semantic_provider=as_prepared_test_provider(semantic_provider),
            semantic_model_call_authority_factory=semantic_authority_factory,
            resource_read_port=_PartialInformationReadPort(),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert result.reason_code == "terminal_candidate_semantic_blocked"
    assert result.requested_user_question is None
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None and details.auxiliary_graph_revision == 2
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=details.goal_id,
    ) == 0
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision is None
    with store._connect() as conn:
        settlement_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_semantic_quorum_settlements "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchone()[0]
        )
        published_task_deliveries = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchone()[0]
        )
    assert settlement_count == 0
    assert published_task_deliveries == 0
