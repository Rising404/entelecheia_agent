from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from personagraph.context_budget import ContextBudgetExceeded
from personagraph.l2.task_execution.work_run import turn_controller as controller_module
from personagraph.l2.task_execution.attempts.decision import (
    AttemptDecisionInputLimits,
    AttemptDecisionInputUnsupported,
    RequiredPriorToolResultsUnavailable,
)
from personagraph.l2.task_execution.verification.decision import NodeVerificationInputLimits
from personagraph.l2.task_execution.paper_prompt_context import PaperAttemptContext
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyInputLimits,
)
from personagraph.l2.task_execution.task_node.model_authority import (
    create_task_node_work_run_model_call_authority,
)
from personagraph.l2.task_execution.work_run.turn_controller import (
    WorkRunTurnApplicationRequest,
    WorkRunTurnAuthorityUnavailable,
    WorkRunTurnResumeRequest,
    WorkRunTurnStableIdPlan,
    WorkRunTurnVerificationRecoveryRequest,
    WorkRunTurnWaitingUserContinuationRequest,
    continue_waiting_user_task_node_work_run as _continue_waiting_user_task_node_work_run,
    recover_task_node_work_run_verification as _recover_task_node_work_run_verification,
    resume_active_task_node_work_run as _resume_active_task_node_work_run,
    run_new_task_node_work_run as _run_new_task_node_work_run,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    ProtectedToolDispatchRequest,
    RuntimeProtectedToolDispatcher,
)
# 运行时导入先于模型网关，以保留包当前的初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.l2.task_graph.paper_resource_contracts import (
    PaperDocumentBinding,
    PaperResourceSnapshot,
)
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.time.date_tools import build_date_tool_registrations
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.policy import (
    AuthorityFacts,
    ProtectedToolExecutionAuthority,
    ScopeGrant,
)
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration
from personagraph.l2.work_run import (
    PendingUserQuestion,
    TaskNodeSubject,
    ToolResultStatus,
    WorkRunStatus,
)
from tests.helpers.evidence_submission import empty_support_justification
from tests.helpers.prepared_model_provider import as_prepared_test_provider


_USER_TEXT = "产出一份可直接使用的旅行计划"
_CallResultT = TypeVar("_CallResultT")


def _call_with_prepared_test_providers(
    call: Callable[..., _CallResultT],
    /,
    *args: object,
    **kwargs: Any,
) -> _CallResultT:
    """Adapt test providers at the prepared-only production call boundary."""

    for name in ("attempt_provider", "verification_provider"):
        provider = kwargs.get(name)
        if not callable(provider):
            raise TypeError(f"{name} must be callable in WorkRun tests")
        kwargs[name] = as_prepared_test_provider(provider)
    return call(*args, **kwargs)


def run_new_task_node_work_run(*args: object, **kwargs: Any) -> Any:
    return _call_with_prepared_test_providers(
        _run_new_task_node_work_run,
        *args,
        **kwargs,
    )


def continue_waiting_user_task_node_work_run(*args: object, **kwargs: Any) -> Any:
    return _call_with_prepared_test_providers(
        _continue_waiting_user_task_node_work_run,
        *args,
        **kwargs,
    )


def recover_task_node_work_run_verification(*args: object, **kwargs: Any) -> Any:
    return _call_with_prepared_test_providers(
        _recover_task_node_work_run_verification,
        *args,
        **kwargs,
    )


def resume_active_task_node_work_run(*args: object, **kwargs: Any) -> Any:
    return _call_with_prepared_test_providers(
        _resume_active_task_node_work_run,
        *args,
        **kwargs,
    )


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _dependency_limits() -> TaskNodeDependencyInputLimits:
    return TaskNodeDependencyInputLimits(
        profile_id="sqlite-work-run-dependencies-v1",
        max_items=32,
        max_serialized_utf8_bytes=256_000,
    )


def _paper_resources(*, session_id: str, task_id: str) -> PaperAttemptContext:
    return PaperAttemptContext.from_snapshot(
        PaperResourceSnapshot.create(
            session_id=session_id,
            task_id=task_id,
            bound_graph_revision=1,
            bound_task_state_version=1,
            retrieval_data_version_id="rdv-controller-paper",
            retrieval_generation_fingerprint="retrieval-controller-paper",
            encoder_fingerprint="deterministic-lexical@1",
            documents=(
                PaperDocumentBinding(
                    paper_key="P1",
                    document_id="private-controller-document",
                    source_version_id="private-controller-version",
                    title="Controller paper",
                    source_sha256="c" * 64,
                    processing_status="complete",
                    admitted_chunk_count=3,
                    chunk_manifest_sha256="d" * 64,
                    admitted_text_page_start=1,
                    admitted_text_page_end=6,
                ),
            ),
        )
    )


def _seed_ready_node(*, suffix: str) -> tuple[str, str, TaskNodeSubject]:
    session_id = store.create_session("Entelecheia", title=f"controller-{suffix}")
    user_text = _USER_TEXT
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"request-{suffix}",
        source="runtime_test",
        user_text=user_text,
        lease_owner="work-run-turn-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_id = f"task-{suffix}"
    node_id = f"node-{suffix}"
    now = "2026-08-14T00:00:00+00:00"
    source_anchors_json = json.dumps(
        [
            {
                "anchor_id": "request",
                "source_turn_id": turn_id,
                "source_kind": "current_user_instruction",
                "start": 0,
                "end": len(user_text),
                "excerpt": user_text,
            }
        ],
        ensure_ascii=False,
    )
    acceptance_json = json.dumps(
        [
            {
                "acceptance_id": "deliverable",
                "criterion": "提供完整可读的旅行计划",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, '旅行计划', '产出旅行计划', ?, ?, ?)",
            (task_id, session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, ?, '[\"request\"]', '[\"request\"]', ?)",
            (task_id, turn_id, f"proposal-{suffix}", source_anchors_json, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, '旅行计划', '产出旅行计划', "
            "'[\"request\"]', ?, '[]', ?)",
            (task_id, node_id, acceptance_json, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, node_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, turn_id, task_id, now),
        )
    return session_id, turn_id, TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )


def _insert_additional_ready_node(
    *,
    session_id: str,
    turn_id: str,
    suffix: str,
) -> TaskNodeSubject:
    task_id = f"task-{suffix}"
    node_id = f"node-{suffix}"
    now = "2026-08-14T00:01:00+00:00"
    source_anchors_json = json.dumps(
        [
            {
                "anchor_id": "request",
                "source_turn_id": turn_id,
                "source_kind": "current_user_instruction",
                "start": 0,
                "end": len(_USER_TEXT),
                "excerpt": _USER_TEXT,
            }
        ],
        ensure_ascii=False,
    )
    acceptance_json = json.dumps(
        [
            {
                "acceptance_id": "deliverable",
                "criterion": "提供完整可读的旅行计划",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, '另一旅行计划', '产出另一计划', ?, ?, ?)",
            (task_id, session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, ?, '[\"request\"]', '[\"request\"]', ?)",
            (task_id, turn_id, f"proposal-{suffix}", source_anchors_json, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, '另一旅行计划', '产出另一计划', "
            "'[\"request\"]', ?, '[]', ?)",
            (task_id, node_id, acceptance_json, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, node_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, turn_id, task_id, now),
        )
    return TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )


def _request(
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    *,
    paper_resources: PaperAttemptContext | None = None,
) -> WorkRunTurnApplicationRequest:
    details = task_graph_store.get_insession_task_details(session_id, subject.task_id)
    assert details is not None
    node = next(
        item
        for item in details.nodes
        if item["insession_task_node_id"] == subject.node_id
    )
    return WorkRunTurnApplicationRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=details.task_state_version,
        expected_node_state_version=int(node["state_version"]),
        expected_window_revision=_window_revision(session_id),
        attempt_input_limits=AttemptDecisionInputLimits(
            profile_id="sqlite-work-run-attempt-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=_dependency_limits(),
            max_serialized_utf8_bytes=1_000_000,
        ),
        verification_input_limits=NodeVerificationInputLimits(
            profile_id="sqlite-work-run-verification-v1",
            max_acceptance_items=64,
            max_supporting_tool_result_items=256,
            dependency_delivery_limits=_dependency_limits(),
            max_serialized_utf8_bytes=1_000_000,
        ),
        paper_resources=paper_resources,
    )


def _continuation_request(
    *,
    session_id: str,
    turn_id: str,
    pending: PendingUserQuestion,
    paper_resources: PaperAttemptContext | None = None,
) -> WorkRunTurnWaitingUserContinuationRequest:
    return WorkRunTurnWaitingUserContinuationRequest(
        session_id=session_id,
        turn_id=turn_id,
        pending_question=pending,
        expected_window_revision=_window_revision(session_id),
        attempt_input_limits=AttemptDecisionInputLimits(
            profile_id="sqlite-waiting-continuation-attempt-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=_dependency_limits(),
            max_serialized_utf8_bytes=1_000_000,
        ),
        verification_input_limits=NodeVerificationInputLimits(
            profile_id="sqlite-waiting-continuation-verification-v1",
            max_acceptance_items=64,
            max_supporting_tool_result_items=256,
            dependency_delivery_limits=_dependency_limits(),
            max_serialized_utf8_bytes=1_000_000,
        ),
        paper_resources=paper_resources,
    )


def _finalize_question_and_accept_answer(
    *,
    session_id: str,
    question_turn_id: str,
    subject: TaskNodeSubject,
    pending: PendingUserQuestion,
    suffix: str,
    answer: str,
) -> str:
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content=pending.question,
        post_commit_job_kinds=(),
    )
    final_window = finalized["window"]
    assert isinstance(final_window, dict)
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(final_window["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"answer-{suffix}",
        source="runtime_test",
        user_text=answer,
        lease_owner="work-run-turn-controller-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                answer_turn_id,
                subject.task_id,
                "2026-08-14T01:00:00+00:00",
            ),
        )
    return answer_turn_id


def _prepare_waiting_continuation(
    *,
    suffix: str,
    answer: str = "九月十日出发，九月十五日返程。",
    with_paper_resources: bool = False,
    first_attempt_payloads: list[dict[str, object]] | None = None,
) -> tuple[
    str,
    TaskNodeSubject,
    WorkRunTurnStableIdPlan,
    PendingUserQuestion,
    WorkRunTurnWaitingUserContinuationRequest,
]:
    session_id, question_turn_id, subject = _seed_ready_node(suffix=suffix)
    id_plan = WorkRunTurnStableIdPlan(namespace=f"waiting-{suffix}")
    paper_resources = (
        _paper_resources(session_id=session_id, task_id=subject.task_id)
        if with_paper_resources
        else None
    )

    def first_attempt_provider(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        if first_attempt_payloads is not None:
            first_attempt_payloads.append(json.loads(user_content))
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请提供旅行日期。",
                },
            },
            str(kwargs["model_call_id"]),
        )

    first = run_new_task_node_work_run(
        _request(
            session_id,
            question_turn_id,
            subject,
            paper_resources=paper_resources,
        ),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=first_attempt_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("request_user must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )
    assert first.outcome == "waiting_user"
    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert len(pending) == 1
    answer_turn_id = _finalize_question_and_accept_answer(
        session_id=session_id,
        question_turn_id=question_turn_id,
        subject=subject,
        pending=pending[0],
        suffix=suffix,
        answer=answer,
    )
    return (
        session_id,
        subject,
        id_plan,
        pending[0],
        _continuation_request(
            session_id=session_id,
            turn_id=answer_turn_id,
            pending=pending[0],
            paper_resources=paper_resources,
        ),
    )


def _model_result(reply: dict[str, object], model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=json.dumps(reply, ensure_ascii=False),
        provider="sqlite-controller-provider",
        model="sqlite-controller-model",
        latency_ms=1,
        model_call_id=model_call_id,
    )


def _advancing_clock(*, step: float = 1.0):
    current = 0.0

    def read() -> float:
        nonlocal current
        current += step
        return current

    return read


def _readonly_registration(tool_id: str, *, handler) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="contract-1",
            name=f"Test {tool_id}",
            description="Controlled read-only recovery test registration.",
            input_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            catalog_tags=("read",),
        ),
        implementation_version="implementation-1",
        source=ToolSourceDescriptor(ToolSourceKind.LOCAL, "controller-recovery-test"),
        handler=handler,
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.MEMORY,
                    EffectAction.READ,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(),
    )


def _readonly_runtime(
    id_plan: WorkRunTurnStableIdPlan,
    *registrations: ToolRegistration,
) -> tuple[CatalogSnapshot, tuple[ToolSpec, ...], SqliteWorkRunToolBridge]:
    catalog = ToolCatalog()
    for registration in registrations:
        catalog.register(registration)
    snapshot = catalog.snapshot()
    allowed_tools = tuple(
        entry.registration.spec for entry in snapshot.exposed()
    )
    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=id_plan.tool_bridge_persistence_plan,
    )
    return snapshot, allowed_tools, bridge


def _protected_registration(tool_id: str, *, handler) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="contract-1",
            name=f"Test {tool_id}",
            description="Controlled protected recovery test registration.",
            input_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            catalog_tags=("research",),
        ),
        implementation_version="1+protected@1",
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "controller-protected-recovery-test",
        ),
        handler=handler,
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    data_egress=DataEgress.CONTENT,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(),
    )


def _protected_runtime(
    id_plan: WorkRunTurnStableIdPlan,
    registration: ToolRegistration,
) -> tuple[CatalogSnapshot, tuple[ToolSpec, ...], SqliteWorkRunToolBridge]:
    catalog = ToolCatalog()
    catalog.register(registration)
    snapshot = catalog.snapshot()
    allowed_tools = tuple(
        entry.registration.spec for entry in snapshot.exposed()
    )
    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=id_plan.tool_bridge_persistence_plan,
        authority=AuthorityFacts(
            approval_grants=(
                ScopeGrant(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    "*",
                ),
            )
        ),
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={
            (registration.tool_id, registration.contract_version): (
                ProtectedToolExecutionAuthority(
                    approval_receipt_ids=("protected-runtime-approval",),
                    execution_backend_identity_sha256="c" * 64,
                    revalidate=lambda: True,
                )
            )
        },
    )
    assert bridge.supports_protected_recovery is True
    return snapshot, allowed_tools, bridge


def _acceptance_update() -> list[dict[str, object]]:
    return [
        {
            "acceptance_id": "deliverable",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
            "empty_support_justification": empty_support_justification(),
        }
    ]


def _verification_reply(*, passed: bool) -> dict[str, object]:
    return {
        "acceptance_results": [
            {
                "acceptance_id": "deliverable",
                "verdict": "passed" if passed else "not_satisfied",
                "finding": "内容完整。" if passed else "缺少逐日安排。",
                "missing_requirements": [] if passed else ["补充逐日安排"],
            }
        ]
    }


def _assert_no_assistant_transcript(session_id: str) -> None:
    with store._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM session_turns WHERE session_id=? AND role='assistant'",
            (session_id,),
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 0


def test_stable_tool_call_ids_include_the_attempt_ordinal() -> None:
    plan = WorkRunTurnStableIdPlan(namespace="tool-ids")

    first = plan.tool_call_id(plan.attempt_id(1), 1)
    second = plan.tool_call_id(plan.attempt_id(2), 1)

    assert first == "tool-ids:tool-call:1:1"
    assert second == "tool-ids:tool-call:2:1"
    assert first != second

    first_resume = plan.resume_active_attempt_apply_id(1, "turn-one")
    replay_resume = plan.resume_active_attempt_apply_id(1, "turn-one")
    next_turn_resume = plan.resume_active_attempt_apply_id(1, "turn-two")
    assert first_resume == replay_resume
    assert first_resume != next_turn_resume
    assert len(first_resume) <= 200


def test_attempt_and_verifier_receive_the_same_node_scoped_source_context() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="source-context")
    attempt_payloads: list[dict[str, object]] = []
    verifier_payloads: list[dict[str, object]] = []

    def attempt_provider(
        _system: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        attempt_payloads.append(json.loads(user_content))
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "一份完整可读的旅行计划。",
                    "format": "plain_text",
                },
            },
            model_call_id,
        )

    def verification_provider(
        _system: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_task_node_semantic_verification"
        verifier_payloads.append(json.loads(user_content))
        return _model_result(_verification_reply(passed=True), model_call_id)

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="source-context"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert len(attempt_payloads) == len(verifier_payloads) == 1
    attempt_source = attempt_payloads[0]["source_context"]
    verifier_source = verifier_payloads[0]["source_context"]
    assert attempt_source == verifier_source
    assert attempt_source["anchors"] == [
        {
            "anchor_id": "request",
            "source_turn_id": turn_id,
            "source_kind": "current_user_instruction",
            "start": 0,
            "end": len(_USER_TEXT),
            "excerpt": _USER_TEXT,
            "excerpt_sha256": hashlib.sha256(
                _USER_TEXT.encode("utf-8")
            ).hexdigest(),
            "authorization": True,
            "required": True,
        }
    ]


def test_unknown_node_source_anchor_fails_before_attempt_provider() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="unknown-attempt-source"
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_nodes "
            "SET source_anchor_ids_json='[\"request\",\"unknown\"]' "
            "WHERE insession_task_id=? AND graph_revision=1 "
            "AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        )
    provider_calls = 0

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("unknown source anchor reached Attempt provider")

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda *_args, **_kwargs: pytest.fail(
            "unknown source anchor reached verifier"
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="unknown-attempt-source"),
        monotonic_clock=_advancing_clock(),
    )

    assert provider_calls == 0
    assert result.outcome != "delivery_ready"


def test_source_authority_drift_after_attempt_fails_before_verifier_provider() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="unknown-verifier-source"
    )
    verifier_calls = 0

    def attempt_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_task_graph_nodes "
                "SET source_anchor_ids_json='[\"request\",\"unknown\"]' "
                "WHERE insession_task_id=? AND graph_revision=1 "
                "AND insession_task_node_id=?",
                (subject.task_id, subject.node_id),
            )
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "一份完整可读的旅行计划。",
                    "format": "plain_text",
                },
            },
            model_call_id,
        )

    def verification_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal verifier_calls
        verifier_calls += 1
        raise AssertionError("drifted source authority reached verifier")

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="unknown-verifier-source"),
        monotonic_clock=_advancing_clock(),
    )

    assert verifier_calls == 0
    assert result.outcome != "delivery_ready"


def test_missing_task_or_node_cas_fails_before_creating_a_work_run() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="missing-cas")
    request = _request(session_id, turn_id, subject).model_copy(
        update={"expected_task_state_version": None}
    )

    result = run_new_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing CAS must fail before the provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing CAS must fail before the verifier")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="missing-cas"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "failed_closed"
    assert result.failure_code == "missing_authority_cas"
    assert work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    ) == ()


def test_attempt_input_limits_have_no_implicit_runtime_default() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="missing-input-limits")
    payload = _request(session_id, turn_id, subject).model_dump(mode="python")
    payload.pop("attempt_input_limits")

    with pytest.raises(ValidationError, match="attempt_input_limits"):
        WorkRunTurnApplicationRequest.model_validate(payload)


def test_verification_input_limits_have_no_implicit_runtime_default() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="missing-verification-input-limits"
    )
    payload = _request(session_id, turn_id, subject).model_dump(mode="python")
    payload.pop("verification_input_limits")

    with pytest.raises(ValidationError, match="verification_input_limits"):
        WorkRunTurnApplicationRequest.model_validate(payload)


def test_exposed_tools_without_a_bridge_fail_before_the_attempt_provider() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="missing-bridge")
    catalog = ToolCatalog()
    for registration in build_date_tool_registrations():
        catalog.register(registration)
    snapshot = catalog.snapshot()
    allowed_tools = tuple(entry.registration.spec for entry in snapshot.exposed())
    assert allowed_tools

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing Tool Bridge must fail before the provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing Tool Bridge must fail before the verifier")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="missing-bridge"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "failed_closed"
    assert result.failure_code == "tool_bridge_unavailable"
    assert work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    ) == ()


def test_prior_tool_result_byte_limit_interrupts_before_attempt_provider() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="prior-result-limit")
    request = _request(session_id, turn_id, subject).model_copy(
        update={
            "attempt_input_limits": AttemptDecisionInputLimits(
                profile_id="sqlite-work-run-attempt-tiny-prior-v1",
                max_prior_tool_result_items=0,
                max_prior_tool_results_serialized_utf8_bytes=1,
                dependency_delivery_limits=_dependency_limits(),
                max_serialized_utf8_bytes=1_000_000,
            )
        }
    )
    provider_calls = 0

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit history must stop before the provider")

    result = run_new_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("over-limit Attempt input must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="prior-result-limit"),
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.outcome == "internal_interrupted"
    assert result.failure_code is None
    assert result.interruption_reason == "attempt_prior_tool_results_input_too_large"
    assert result.work_run_id == "prior-result-limit:workrun"
    assert result.current_attempt_id == "prior-result-limit:attempt:1"
    assert result.window_revision == int(window["state_version"])
    assert provider_calls == 0


def test_complete_attempt_input_limit_interrupts_with_authoritative_cursor() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="attempt-input-limit")
    request = _request(session_id, turn_id, subject).model_copy(
        update={
            "attempt_input_limits": AttemptDecisionInputLimits(
                profile_id="sqlite-work-run-attempt-tiny-total-v1",
                max_prior_tool_result_items=32,
                max_prior_tool_results_serialized_utf8_bytes=128_000,
                dependency_delivery_limits=_dependency_limits(),
                max_serialized_utf8_bytes=1,
            )
        }
    )
    provider_calls = 0
    events = []

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit complete input must stop before the provider")

    result = run_new_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("over-limit complete input must not invoke verification")
        ),
        emit=events.append,
        id_plan=WorkRunTurnStableIdPlan(namespace="attempt-input-limit"),
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.outcome == "internal_interrupted"
    assert result.failure_code is None
    assert result.interruption_reason == "attempt_model_input_too_large"
    assert result.current_attempt_id == "attempt-input-limit:attempt:1"
    assert result.window_revision == int(window["state_version"])
    assert provider_calls == 0
    assert events == []


def test_unsupported_attempt_input_interrupts_with_authoritative_cursor(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="attempt-input-unsupported")
    provider_calls = 0
    events = []

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("unsupported input must stop before the provider")

    monkeypatch.setattr(
        controller_module,
        "run_started_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttemptDecisionInputUnsupported(profile_id="unsupported-test-v1")
        ),
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported input must not invoke verification")
        ),
        emit=events.append,
        id_plan=WorkRunTurnStableIdPlan(namespace="attempt-input-unsupported"),
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.outcome == "internal_interrupted"
    assert result.interruption_reason == "attempt_model_input_unsupported"
    assert result.current_attempt_id == "attempt-input-unsupported:attempt:1"
    assert result.window_revision == int(window["state_version"])
    assert provider_calls == 0
    assert events == []


def test_unavailable_supporting_result_returns_resumable_authoritative_cursor(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="missing-support")
    provider_calls = 0

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("missing required history must stop before the provider")

    monkeypatch.setattr(
        controller_module,
        "select_bounded_prior_tool_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RequiredPriorToolResultsUnavailable(("missing-result",))
        ),
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing required history must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="missing-support"),
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.outcome == "internal_interrupted"
    assert result.failure_code is None
    assert result.interruption_reason == "attempt_supporting_tool_result_unavailable"
    assert result.work_run_id == "missing-support:workrun"
    assert result.current_attempt_id == "missing-support:attempt:1"
    assert result.window_revision == int(window["state_version"])
    assert provider_calls == 0


def test_verifier_input_limit_interrupts_submit_before_verification_provider() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="verifier-input-limit")
    request = _request(session_id, turn_id, subject).model_copy(
        update={
            "verification_input_limits": NodeVerificationInputLimits(
                profile_id="sqlite-work-run-verification-tiny-v1",
                max_acceptance_items=64,
                max_supporting_tool_result_items=256,
                dependency_delivery_limits=_dependency_limits(),
                max_serialized_utf8_bytes=1,
            )
        }
    )
    verification_provider_calls = 0

    def attempt_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "# 可交付旅行计划\n第一天抵达，第二天参观。",
                    "format": "markdown",
                },
            },
            model_call_id,
        )

    def verification_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal verification_provider_calls
        verification_provider_calls += 1
        raise AssertionError("over-limit verifier input must not call provider")

    result = run_new_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="verifier-input-limit"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "verification_interrupted"
    assert result.interruption_reason == "verification_input_too_large"
    assert verification_provider_calls == 0
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verifier-input-limit:verification:1",
    )
    assert record.request.status.value == "interrupted"
    assert record.request.technical_error_code == "verification_input_too_large"
    assert record.result is None


def test_same_turn_write_then_submit_pass_returns_internal_delivery_only() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="pass")
    attempt_calls = 0
    body = "# 上海旅行计划\n\n第一天抵达并入住；第二天参观博物馆。"

    def attempt_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal attempt_calls
        assert purpose == "runtime_work_run_attempt_decision"
        attempt_calls += 1
        action = (
            {
                "kind": "write_output_window",
                "content": body,
                "format": "markdown",
            }
            if attempt_calls == 1
            else {
                "kind": "submit_output_window",
                "content": body,
                "format": "markdown",
            }
        )
        return _model_result(
            {
                "acceptance_updates": _acceptance_update() if attempt_calls == 1 else [],
                "action": action,
            },
            model_call_id,
        )

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="pass"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert result.delivery_id == "pass:delivery:2"
    assert attempt_calls == 2
    assert result.work_run_id is not None
    delivery = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=result.delivery_id,
    )
    assert delivery.delivery.work_run_id == result.work_run_id
    assert delivery.output_window.content == body
    assert work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=result.work_run_id,
    ).work_run.status is WorkRunStatus.COMPLETED
    _assert_no_assistant_transcript(session_id)


def test_nonpass_feedback_continues_with_the_store_started_next_attempt() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="nonpass")
    attempt_payloads: list[dict[str, object]] = []
    verification_calls = 0

    def attempt_provider(
        _system: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        payload = json.loads(user_content)
        attempt_payloads.append(payload)
        improved = len(attempt_payloads) == 2
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": (
                        "第一天抵达。第二天参观。第三天返程。"
                        if improved
                        else "旅行计划草稿。"
                    ),
                    "format": "plain_text",
                },
            },
            model_call_id,
        )

    def verification_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal verification_calls
        assert purpose == "runtime_task_node_semantic_verification"
        verification_calls += 1
        return _model_result(
            _verification_reply(passed=verification_calls == 2),
            model_call_id,
        )

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="nonpass"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert verification_calls == 2
    assert len(attempt_payloads) == 2
    assert attempt_payloads[0]["verification_feedback"] is None
    assert attempt_payloads[1]["verification_feedback"]["acceptance_results"][0][
        "verdict"
    ] == "not_satisfied"
    assert result.work_run_id is not None
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=result.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    _assert_no_assistant_transcript(session_id)


def test_request_user_returns_typed_wait_without_verifier_or_transcript() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="waiting")
    verification_calls = 0

    def verification_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal verification_calls
        verification_calls += 1
        raise AssertionError("request_user must not invoke the verifier")

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请提供旅行日期。",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="waiting"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "waiting_user"
    assert result.pending_user_question == "请提供旅行日期。"
    assert verification_calls == 0
    assert result.work_run_id is not None
    assert work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=result.work_run_id,
    ).work_run.status is WorkRunStatus.WAITING_USER
    _assert_no_assistant_transcript(session_id)


def test_waiting_user_continuation_projects_answer_and_delivers() -> None:
    answer = "九月十日出发，九月十五日返程。"
    first_attempt_payloads: list[dict[str, object]] = []
    session_id, _subject, id_plan, pending, request = (
        _prepare_waiting_continuation(
            suffix="continue-pass",
            answer=answer,
            with_paper_resources=True,
            first_attempt_payloads=first_attempt_payloads,
        )
    )
    attempt_payloads: list[dict[str, object]] = []

    def attempt_provider(
        _system: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        attempt_payloads.append(json.loads(user_content))
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "九月十日出发，九月十五日返程的旅行计划。",
                    "format": "plain_text",
                },
            },
            model_call_id,
        )

    result = continue_waiting_user_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert result.delivery_id == id_plan.delivery_id(2)
    assert len(attempt_payloads) == 1
    assert attempt_payloads[0]["user_input"] == {
        "content": answer,
        "prior_waiting_user_question": pending.question,
    }
    assert request.paper_resources is not None
    expected_paper_resources = request.paper_resources.to_dict()
    assert first_attempt_payloads[0]["paper_resources"] == expected_paper_resources
    assert attempt_payloads[0]["paper_resources"] == expected_paper_resources
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=pending.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert stored.attempts[1].input_turn_id == request.turn_id
    assert (
        stored.attempts[1].predecessor_question_attempt_id
        == pending.question_attempt_id
    )
    assert stored.work_run.status is WorkRunStatus.COMPLETED


def test_waiting_user_continuation_response_loss_replays_exact_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _subject, id_plan, pending, request = (
        _prepare_waiting_continuation(suffix="continue-response-loss")
    )
    original = continuation_store.continue_waiting_user_work_run_and_start_attempt
    continuation_calls = 0

    def continue_then_lose_first_response(**kwargs: object):
        nonlocal continuation_calls
        continuation_calls += 1
        mutation = original(**kwargs)
        if continuation_calls == 1:
            raise RuntimeError("injected continuation response loss")
        return mutation

    monkeypatch.setattr(
        continuation_store,
        "continue_waiting_user_work_run_and_start_attempt",
        continue_then_lose_first_response,
    )
    attempt_calls = 0

    def attempt_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal attempt_calls
        assert purpose == "runtime_work_run_attempt_decision"
        attempt_calls += 1
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "可验证的旅行计划。",
                    "format": "plain_text",
                },
            },
            model_call_id,
        )

    result = continue_waiting_user_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=attempt_provider,
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert continuation_calls == 2
    assert attempt_calls == 1
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=pending.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    with store._connect() as conn:
        receipt_count = conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id=?",
            (
                id_plan.continue_waiting_user_apply_id(
                    pending.question_attempt_ordinal
                ),
            ),
        ).fetchone()[0]
    assert receipt_count == 1


def test_two_lost_waiting_continuation_responses_reconcile_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _subject, id_plan, pending, request = (
        _prepare_waiting_continuation(suffix="continue-double-loss")
    )
    original = continuation_store.continue_waiting_user_work_run_and_start_attempt
    continuation_calls = 0

    def continue_or_replay_then_lose_response(**kwargs: object):
        nonlocal continuation_calls
        continuation_calls += 1
        original(**kwargs)
        raise RuntimeError("injected loss after continuation commit or replay")

    monkeypatch.setattr(
        continuation_store,
        "continue_waiting_user_work_run_and_start_attempt",
        continue_or_replay_then_lose_response,
    )
    result = continue_waiting_user_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("two lost responses must stop before the provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("two lost responses must stop before verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "internal_interrupted"
    assert result.interruption_reason == "waiting_user_continuation_response_lost"
    assert result.current_attempt_id == id_plan.attempt_id(2)
    assert continuation_calls == 2
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=pending.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert stored.current_attempt_id == id_plan.attempt_id(2)


def test_stale_waiting_user_question_fails_before_any_continuation_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _subject, id_plan, pending, request = (
        _prepare_waiting_continuation(suffix="continue-stale")
    )
    stale = request.model_copy(
        update={
            "pending_question": pending.model_copy(
                update={"question_attempt_id": "stale-question-attempt"}
            )
        }
    )
    window_revision_before = _window_revision(session_id)
    monkeypatch.setattr(
        continuation_store,
        "continue_waiting_user_work_run_and_start_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale authority must fail before the Store command")
        ),
    )

    result = continue_waiting_user_task_node_work_run(
        stale,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale question must fail before the provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale question must fail before verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "failed_closed"
    assert result.failure_code == "waiting_user_continuation_failed"
    assert _window_revision(session_id) == window_revision_before
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=pending.work_run_id,
    )
    assert stored.work_run.status is WorkRunStatus.WAITING_USER
    assert len(stored.attempts) == 1
    assert continuation_store.list_pending_user_questions(session_id=session_id) == (pending,)
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE operation='continue_waiting_user_and_start_attempt'"
        ).fetchone()[0] == 0


def test_verifier_transport_failure_returns_the_durable_interruption() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="verify-interrupt")

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelGatewayError(
                "MODEL_UNAVAILABLE",
                "verification transport unavailable",
                retryable=False,
            )
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="verify-interrupt"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "verification_interrupted"
    assert result.interruption_reason == "verification_unavailable"
    assert result.work_run_id is not None
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=result.work_run_id,
    )
    assert stored.work_run.status is WorkRunStatus.INTERRUPTED
    assert stored.work_run.reason == "verification_technical_failure"
    assert stored.current_verification_request_id == "verify-interrupt:verification:1"
    _assert_no_assistant_transcript(session_id)


def _accept_verification_recovery_turn(
    *,
    session_id: str,
    interrupted_turn_id: str,
    subject: TaskNodeSubject,
    window_revision: int,
    suffix: str,
) -> str:
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=interrupted_turn_id,
        expected_window_revision=window_revision,
        stage="VERIFICATION",
        interruption_reason="verification_process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=interrupted_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="verification_process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"verification-recovery-{suffix}",
        source="runtime_test",
        user_text="继续验证",
        lease_owner="work-run-turn-controller-test",
    )
    recovery_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                recovery_turn_id,
                subject.task_id,
                "2026-08-14T02:00:00+00:00",
            ),
        )
    return recovery_turn_id


def _verification_recovery_request(
    *,
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    id_plan: WorkRunTurnStableIdPlan,
    base_request: WorkRunTurnApplicationRequest,
) -> WorkRunTurnVerificationRecoveryRequest:
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    verification_request_id = stored.current_verification_request_id
    assert verification_request_id is not None
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=verification_request_id,
    )
    return WorkRunTurnVerificationRecoveryRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        work_run_id=id_plan.work_run_id,
        submitted_attempt_id=record.request.submitted_attempt_id,
        verification_request_id=verification_request_id,
        expected_work_run_revision=stored.work_run.revision,
        expected_verification_request_revision=record.request.revision,
        expected_window_revision=_window_revision(session_id),
        attempt_input_limits=base_request.attempt_input_limits,
        verification_input_limits=base_request.verification_input_limits,
        paper_resources=base_request.paper_resources,
    )


def _seed_interrupted_verification_recovery(
    *,
    suffix: str,
    clock_step: float = 1.0,
) -> tuple[
    str,
    TaskNodeSubject,
    WorkRunTurnStableIdPlan,
    WorkRunTurnVerificationRecoveryRequest,
]:
    session_id, first_turn_id, subject = _seed_ready_node(suffix=suffix)
    base_request = _request(session_id, first_turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(namespace=suffix)
    interrupted = run_new_task_node_work_run(
        base_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "旅行计划草稿。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelGatewayError(
                "MODEL_UNAVAILABLE",
                "verification unavailable",
                retryable=False,
            )
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(step=clock_step),
    )
    assert interrupted.outcome == "verification_interrupted"
    recovery_turn_id = _accept_verification_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        subject=subject,
        window_revision=interrupted.window_revision,
        suffix=suffix,
    )
    return (
        session_id,
        subject,
        id_plan,
        _verification_recovery_request(
            session_id=session_id,
            turn_id=recovery_turn_id,
            subject=subject,
            id_plan=id_plan,
            base_request=base_request,
        ),
    )


def test_pending_verification_recovery_rebinds_same_request_and_exactly_replays_commit(
    monkeypatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="pending-verification-recovery"
    )
    base_request = _request(session_id, first_turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(
        namespace="pending-verification-recovery"
    )
    original_invoke = controller_module._invoke_node_verification

    def prepare_then_lose(request, **_kwargs):  # type: ignore[no-untyped-def]
        verification_store.prepare_task_node_verification(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_progress_revision=request.expected_progress_revision,
            expected_output_revision=request.expected_output_revision,
            expected_window_revision=request.expected_window_revision,
            apply_id=request.prepare_apply_id,
            verification_request_id=request.verification_request_id,
        )
        raise RuntimeError("injected process loss after verification prepare")

    monkeypatch.setattr(
        controller_module,
        "_invoke_node_verification",
        prepare_then_lose,
    )
    attempt_source_contexts: list[dict[str, object]] = []

    def submit_for_stranded_verification(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        attempt_source_contexts.append(json.loads(user_content)["source_context"])
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        )

    stranded = run_new_task_node_work_run(
        base_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=submit_for_stranded_verification,
        verification_provider=lambda *_args, **_kwargs: pytest.fail(
            "stranded prepare must not reach the provider"
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )
    assert stranded.outcome == "internal_interrupted"
    record_before = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=id_plan.verification_request_id(1),
    )
    assert record_before.request.status.value == "pending"
    assert record_before.request.revision == 1

    recovery_turn_id = _accept_verification_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        subject=subject,
        window_revision=stranded.window_revision,
        suffix="pending",
    )
    recovery_request = _verification_recovery_request(
        session_id=session_id,
        turn_id=recovery_turn_id,
        subject=subject,
        id_plan=id_plan,
        base_request=base_request,
    )
    monkeypatch.setattr(
        controller_module,
        "_invoke_node_verification",
        original_invoke,
    )
    original_commit = verification_store.commit_task_node_verification_result
    commit_calls = 0
    provider_calls = 0

    def commit_then_lose_response(**kwargs: object):
        nonlocal commit_calls
        commit_calls += 1
        mutation = original_commit(**kwargs)
        if commit_calls == 1:
            raise RuntimeError("injected recovery commit response loss")
        return mutation

    def passing_verifier(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        assert json.loads(user_content)["source_context"] == (
            attempt_source_contexts[0]
        )
        return _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        )

    monkeypatch.setattr(
        verification_store,
        "commit_task_node_verification_result",
        commit_then_lose_response,
    )
    recovered = recover_task_node_work_run_verification(
        recovery_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: pytest.fail(
            "passing recovery must not start another Attempt"
        ),
        verification_provider=passing_verifier,
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "delivery_ready"
    assert recovered.delivery_id == id_plan.delivery_id(1)
    assert provider_calls == 1
    assert commit_calls == 1
    record_after = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=id_plan.verification_request_id(1),
    )
    assert record_after.request.request_turn_id == first_turn_id
    assert record_after.request.revision == 3
    assert record_after.result is not None
    assert record_after.result.verification_request_revision == 2
    with store._connect() as conn:
        apply_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT apply_id FROM insession_work_run_apply_receipts "
                "WHERE work_run_id=?",
                (id_plan.work_run_id,),
            ).fetchall()
        }
    assert id_plan.verification_commit_apply_id(1, 2) in apply_ids


def test_verification_success_before_commit_replays_typed_result_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="verification-success-precommit-loss"
    )
    base_request = _request(session_id, first_turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(
        namespace="verification-success-precommit-loss"
    )
    original_commit = verification_store.commit_task_node_verification_result
    commit_calls = 0
    provider_calls = 0

    def durable_model_result(
        payload: dict[str, object],
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        return ModelResult(
            reply=json.dumps(payload, ensure_ascii=False),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    def lose_before_verification_commit(**_kwargs: object):
        nonlocal commit_calls
        commit_calls += 1
        raise RuntimeError("injected process loss before verification commit")

    def passing_verifier(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        assert purpose == "runtime_task_node_semantic_verification"
        return durable_model_result(
            _verification_reply(passed=True),
            model_call_id=model_call_id,
            purpose=purpose,
        )

    def submit_attempt(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        return durable_model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            model_call_id=model_call_id,
            purpose=purpose,
        )

    monkeypatch.setattr(
        verification_store,
        "commit_task_node_verification_result",
        lose_before_verification_commit,
    )
    stranded = run_new_task_node_work_run(
        base_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=submit_attempt,
        verification_provider=passing_verifier,
        emit=lambda _event: None,
        id_plan=id_plan,
        model_call_authority_factory=(
            partial(create_task_node_work_run_model_call_authority, ledger_store=store)
        ),
        monotonic_clock=_advancing_clock(),
    )

    assert stranded.outcome == "internal_interrupted"
    assert provider_calls == 1
    # 控制器在移交轮次前会精确重试待处理请求一次。两次提交尝试都在打开 Store
    # 事务前失败，而成功的带类型提供方结果已经持久化。
    assert commit_calls == 2
    verification_request_id = id_plan.verification_request_id(1)
    record_before = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=verification_request_id,
    )
    assert record_before.request.status.value == "pending"
    assert record_before.result is None
    logical_call_id = id_plan.verification_model_call_id(1)
    with store._connect() as conn:
        physical_before = conn.execute(
            "SELECT physical_ordinal, started_turn_id, status FROM "
            "insession_runtime_model_physical_attempts "
            "WHERE session_id=? AND logical_call_id=? "
            "ORDER BY physical_ordinal",
            (session_id, logical_call_id),
        ).fetchall()
    assert [tuple(row) for row in physical_before] == [
        (1, first_turn_id, "succeeded")
    ]

    recovery_turn_id = _accept_verification_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        subject=subject,
        window_revision=stranded.window_revision,
        suffix="typed-success",
    )
    recovery_request = _verification_recovery_request(
        session_id=session_id,
        turn_id=recovery_turn_id,
        subject=subject,
        id_plan=id_plan,
        base_request=base_request,
    )
    monkeypatch.setattr(
        verification_store,
        "commit_task_node_verification_result",
        original_commit,
    )

    def provider_must_not_run(
        *_args: object,
        **_kwargs: object,
    ) -> ModelResult:
        raise AssertionError("durable typed success must skip Provider I/O")

    recovered = recover_task_node_work_run_verification(
        recovery_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: pytest.fail(
            "passing verification recovery must not start another Attempt"
        ),
        verification_provider=provider_must_not_run,
        emit=lambda _event: None,
        id_plan=id_plan,
        model_call_authority_factory=(
            partial(create_task_node_work_run_model_call_authority, ledger_store=store)
        ),
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "delivery_ready"
    assert recovered.delivery_id == id_plan.delivery_id(1)
    assert provider_calls == 1
    record_after = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=verification_request_id,
    )
    assert record_after.request.request_turn_id == first_turn_id
    assert record_after.result is not None
    assert record_after.result.verification_request_revision == 2
    with store._connect() as conn:
        physical_after = conn.execute(
            "SELECT physical_ordinal, started_turn_id, status FROM "
            "insession_runtime_model_physical_attempts "
            "WHERE session_id=? AND logical_call_id=? "
            "ORDER BY physical_ordinal",
            (session_id, logical_call_id),
        ).fetchall()
    assert [tuple(row) for row in physical_after] == [
        (1, first_turn_id, "succeeded")
    ]


def test_interrupted_verification_recovery_nonpass_continues_attempt_loop() -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="interrupted-verification-recovery"
    )
    base_request = _request(session_id, first_turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(
        namespace="interrupted-verification-recovery"
    )
    interrupted = run_new_task_node_work_run(
        base_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "旅行计划草稿。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelGatewayError(
                "MODEL_UNAVAILABLE",
                "verification unavailable",
                retryable=False,
            )
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )
    assert interrupted.outcome == "verification_interrupted"
    recovery_turn_id = _accept_verification_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        subject=subject,
        window_revision=interrupted.window_revision,
        suffix="interrupted",
    )
    recovery_request = _verification_recovery_request(
        session_id=session_id,
        turn_id=recovery_turn_id,
        subject=subject,
        id_plan=id_plan,
        base_request=base_request,
    )
    attempt_payloads: list[dict[str, object]] = []
    verification_calls = 0

    def improved_attempt(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        attempt_payloads.append(json.loads(user_content))
        return _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        )

    def verifier(
        _system: str,
        _user: str,
        **kwargs: object,
    ) -> ModelResult:
        nonlocal verification_calls
        verification_calls += 1
        return _model_result(
            _verification_reply(passed=verification_calls == 2),
            str(kwargs["model_call_id"]),
        )

    recovered = recover_task_node_work_run_verification(
        recovery_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=improved_attempt,
        verification_provider=verifier,
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "delivery_ready"
    assert recovered.delivery_id == id_plan.delivery_id(2)
    assert verification_calls == 2
    assert len(attempt_payloads) == 1
    assert attempt_payloads[0]["verification_feedback"] is not None
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)


def test_verification_recovery_provider_failure_returns_recoverable_interruption() -> None:
    session_id, _subject, id_plan, recovery_request = (
        _seed_interrupted_verification_recovery(
            suffix="verification-recovery-interrupt"
        )
    )

    recovered = recover_task_node_work_run_verification(
        recovery_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: pytest.fail(
            "interrupted recovery must not start another Attempt"
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelGatewayError(
                "MODEL_UNAVAILABLE",
                "verification remains unavailable",
                retryable=False,
            )
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "verification_interrupted"
    assert recovered.interruption_reason == "verification_unavailable"
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=recovery_request.verification_request_id,
    )
    assert record.request.status.value == "interrupted"
    assert record.request.revision == 4
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id=?",
            (id_plan.verification_interrupt_apply_id(1, 3),),
        ).fetchone()[0] == 1


def test_verification_recovery_hard_limit_projects_work_run_failed() -> None:
    session_id, _subject, id_plan, recovery_request = (
        _seed_interrupted_verification_recovery(
            suffix="verification-recovery-limit",
            clock_step=449.5,
        )
    )
    assert work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    ).work_run.budget.active_seconds_consumed == 899

    recovered = recover_task_node_work_run_verification(
        recovery_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: pytest.fail(
            "hard-limit recovery must not start another Attempt"
        ),
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "work_run_failed", recovered
    assert recovered.delivery_id is None
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert stored.work_run.status is WorkRunStatus.FAILED
    assert stored.work_run.reason == "work_run_limit_reached"
    assert stored.work_run.budget.active_seconds_consumed == 900


def test_attempt_context_budget_failure_propagates_without_provider_dispatch() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="context-budget")
    provider_calls = 0
    prepare_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("budget rejection must stop before provider dispatch")

    def prepare(*_args: object, **_kwargs: object):
        nonlocal prepare_calls
        prepare_calls += 1
        raise ContextBudgetExceeded(limit=1000, estimated_tokens=1001)

    provider.prepare = prepare  # type: ignore[attr-defined]

    with pytest.raises(ContextBudgetExceeded):
        run_new_task_node_work_run(
            _request(session_id, turn_id, subject),
            catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
            allowed_tools=(),
            attempt_provider=provider,
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("attempt budget failure must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=WorkRunTurnStableIdPlan(namespace="context-budget"),
            monotonic_clock=_advancing_clock(),
        )

    assert prepare_calls == 1
    assert provider_calls == 0


def test_existing_nonterminal_run_and_attempt_provider_failure_are_typed() -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="fail-closed")
    request = _request(session_id, turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(namespace="fail-closed")

    failed = run_new_task_node_work_run(
        request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider unavailable")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("attempt failure must not invoke the verifier")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert failed.outcome == "internal_interrupted"
    assert failed.failure_code is None
    assert failed.interruption_reason == "attempt_decision_interrupted"
    assert failed.current_attempt_id == id_plan.attempt_id(1)
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert failed.window_revision == int(window["state_version"])

    replay_as_new = run_new_task_node_work_run(
        request.model_copy(update={"expected_window_revision": failed.window_revision}),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing WorkRun must fail before the provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing WorkRun must fail before the verifier")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="second-new-run"),
        monotonic_clock=_advancing_clock(),
    )

    assert replay_as_new.outcome == "failed_closed"
    assert replay_as_new.failure_code == "existing_nonterminal_work_run"


def test_attempt_model_bad_response_propagates_to_the_entry_boundary() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="model-bad-response-propagates"
    )
    expected_error = ModelGatewayError(
        "MODEL_BAD_RESPONSE",
        "typed Attempt decision retries exhausted",
        retryable=False,
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="model-bad-response-propagates"
    )

    with pytest.raises(ModelGatewayError) as raised:
        run_new_task_node_work_run(
            _request(session_id, turn_id, subject),
            catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
            allowed_tools=(),
            attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                expected_error
            ),
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("invalid Attempt output must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=id_plan,
            monotonic_clock=_advancing_clock(),
        )

    assert raised.value is expected_error
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["window_state"] == "active"
    assert window["current_attempt_id"] == id_plan.attempt_id(1)


def test_detached_waiting_lane_does_not_block_a_fresh_other_task_lane() -> None:
    session_id, turn_id, first_subject = _seed_ready_node(
        suffix="detached-first-lane"
    )
    first_plan = WorkRunTurnStableIdPlan(namespace="detached-first-lane")
    waiting = run_new_task_node_work_run(
        _request(session_id, turn_id, first_subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请补充第一个任务的日期。",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("waiting lane must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=first_plan,
        monotonic_clock=_advancing_clock(),
    )
    assert waiting.outcome == "waiting_user"
    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=first_plan.work_run_id,
        expected_window_revision=waiting.window_revision,
        apply_id="detach-first-waiting-lane",
    )
    second_subject = _insert_additional_ready_node(
        session_id=session_id,
        turn_id=turn_id,
        suffix="detached-second-lane",
    )
    second_provider_calls = 0

    def ask_for_second_task(_system, _user, **kwargs):
        nonlocal second_provider_calls
        second_provider_calls += 1
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请补充第二个任务的预算。",
                },
            },
            str(kwargs["model_call_id"]),
        )

    second = run_new_task_node_work_run(
        _request(session_id, turn_id, second_subject).model_copy(
            update={"expected_window_revision": detached.window_state_version}
        ),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=ask_for_second_task,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("second waiting lane must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="detached-second-lane"),
        monotonic_clock=_advancing_clock(),
    )

    assert second.outcome == "waiting_user"
    assert second_provider_calls == 1
    candidates = work_run_store.list_turn_linked_nonterminal_work_runs(
        session_id=session_id,
        turn_id=turn_id,
    )
    assert {candidate.subject for candidate in candidates} == {
        first_subject,
        second_subject,
    }


def test_generic_task_node_retry_continues_same_logical_call_on_new_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="generic-ledger-resume"
    )
    first_request = _request(session_id, first_turn_id, subject)
    id_plan = WorkRunTurnStableIdPlan(namespace="generic-ledger-resume")
    first_payloads: list[dict[str, object]] = []

    class StopAfterOnePhysicalAttempt:
        checks = 0

        def expired(self) -> bool:
            self.checks += 1
            return self.checks > 1

        def remaining_s(self) -> float:
            return 1.0

    def retryable_provider(
        _system_prompt: str,
        user_content: str,
        **_kwargs: object,
    ) -> ModelResult:
        first_payloads.append(json.loads(user_content))
        raise ModelGatewayError(
            "MODEL_UNAVAILABLE",
            "retry on a later Turn",
            retryable=True,
        )

    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    interrupted = run_new_task_node_work_run(
        first_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=retryable_provider,
        verification_provider=lambda *_args, **_kwargs: pytest.fail(
            "undecided Attempt must not invoke verification"
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        deadline=StopAfterOnePhysicalAttempt(),  # type: ignore[arg-type]
        model_call_authority_factory=(
            partial(create_task_node_work_run_model_call_authority, ledger_store=store)
        ),
        monotonic_clock=_advancing_clock(),
    )

    assert interrupted.outcome == "internal_interrupted"
    assert len(first_payloads) == 1
    logical_call_id = id_plan.attempt_model_call_id(1)
    with store._connect() as conn:
        origin = conn.execute(
            "SELECT invocation_turn_id FROM "
            "insession_runtime_model_logical_calls "
            "WHERE session_id=? AND logical_call_id=?",
            (session_id, logical_call_id),
        ).fetchone()
        first_physical = conn.execute(
            "SELECT physical_ordinal, started_turn_id, status FROM "
            "insession_runtime_model_physical_attempts "
            "WHERE session_id=? AND logical_call_id=?",
            (session_id, logical_call_id),
        ).fetchall()
    assert origin is not None
    assert str(origin["invocation_turn_id"]) == first_turn_id
    assert [tuple(row) for row in first_physical] == [
        (1, first_turn_id, "retryable_failure")
    ]

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=interrupted.window_revision,
        stage="L2_PLAN",
        interruption_reason="provider_retry_handoff",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="provider_retry_handoff",
        error_code="TURN_DEADLINE_EXCEEDED",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="generic-ledger-resume-new-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-24T00:30:00+00:00",
            ),
        )
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    resumed_payloads: list[dict[str, object]] = []

    def resumed_provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        resumed_payloads.append(json.loads(user_content))
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充旅行日期。",
                    },
                }
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    resumed = resume_active_task_node_work_run(
        WorkRunTurnResumeRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            subject=subject,
            work_run_id=id_plan.work_run_id,
            attempt_id=id_plan.attempt_id(1),
            expected_work_run_revision=stored.work_run.revision,
            expected_window_revision=_window_revision(session_id),
            attempt_input_limits=first_request.attempt_input_limits,
            verification_input_limits=first_request.verification_input_limits,
        ),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=resumed_provider,
        verification_provider=lambda *_args, **_kwargs: pytest.fail(
            "request_user must not invoke verification"
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        model_call_authority_factory=(
            partial(create_task_node_work_run_model_call_authority, ledger_store=store)
        ),
        monotonic_clock=_advancing_clock(),
    )

    assert resumed.outcome == "waiting_user"
    assert resumed_payloads == first_payloads
    with store._connect() as conn:
        logical_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND logical_call_id=?",
                (session_id, logical_call_id),
            ).fetchone()[0]
        )
        physical = conn.execute(
            "SELECT physical_ordinal, started_turn_id, status FROM "
            "insession_runtime_model_physical_attempts "
            "WHERE session_id=? AND logical_call_id=? "
            "ORDER BY physical_ordinal",
            (session_id, logical_call_id),
        ).fetchall()
    assert logical_count == 1
    assert [tuple(row) for row in physical] == [
        (1, first_turn_id, "retryable_failure"),
        (2, second_turn_id, "succeeded"),
    ]


def test_active_undecided_attempt_resumes_on_new_turn_with_a_fresh_timer(
    monkeypatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(suffix="resume-active")
    paper_resources = _paper_resources(
        session_id=session_id,
        task_id=subject.task_id,
    )
    first_request = _request(
        session_id,
        first_turn_id,
        subject,
        paper_resources=paper_resources,
    )
    id_plan = WorkRunTurnStableIdPlan(namespace="resume-active")
    first_payloads: list[dict[str, object]] = []

    def interrupted_provider(
        _system: str,
        user_content: str,
        **_kwargs: object,
    ) -> ModelResult:
        first_payloads.append(json.loads(user_content))
        raise RuntimeError("process lost with an undecided Attempt")

    interrupted = run_new_task_node_work_run(
        first_request,
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=interrupted_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("undecided failure must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(step=100.0),
    )
    assert interrupted.outcome == "internal_interrupted"
    assert interrupted.current_attempt_id == id_plan.attempt_id(1)
    before_handoff = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert before_handoff.work_run.budget.active_seconds_consumed == 0

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=interrupted.window_revision,
        stage="L2_PLAN",
        interruption_reason="process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-active-new-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:30:00+00:00",
            ),
        )
    resume_inputs = _request(session_id, second_turn_id, subject)
    stored_before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    original_resume = work_run_store.resume_active_work_run_attempt
    resume_calls = 0

    def resume_then_lose_first_response(**kwargs: object):
        nonlocal resume_calls
        resume_calls += 1
        mutation = original_resume(**kwargs)
        if resume_calls == 1:
            raise RuntimeError("injected response loss after Attempt rebind")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "resume_active_work_run_attempt",
        resume_then_lose_first_response,
    )
    other_subject = _insert_additional_ready_node(
        session_id=session_id,
        turn_id=second_turn_id,
        suffix="resume-active-other-lane",
    )
    authoritative_candidates = work_run_store.list_turn_linked_nonterminal_work_runs

    def candidates_with_unrelated_waiting_lane(**kwargs):
        exact = authoritative_candidates(**kwargs)
        return exact + (
            work_run_store.TurnLinkedNonterminalWorkRunCandidate(
                session_id=session_id,
                work_run_id="unrelated-detached-waiting-run",
                subject=other_subject,
                status=WorkRunStatus.WAITING_USER,
                reason="user_input_required",
                work_run_revision=4,
                created_turn_id=first_turn_id,
                updated_turn_id=second_turn_id,
            ),
        )

    monkeypatch.setattr(
        work_run_store,
        "list_turn_linked_nonterminal_work_runs",
        candidates_with_unrelated_waiting_lane,
    )
    provider_payloads: list[dict[str, object]] = []

    def resumed_provider(
        _system: str,
        user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        provider_payloads.append(json.loads(user))
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请补充旅行日期。",
                },
            },
            model_call_id,
        )

    resumed = resume_active_task_node_work_run(
        WorkRunTurnResumeRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            subject=subject,
            work_run_id=id_plan.work_run_id,
            attempt_id=id_plan.attempt_id(1),
            expected_work_run_revision=stored_before_resume.work_run.revision,
            expected_window_revision=resume_inputs.expected_window_revision,
            attempt_input_limits=resume_inputs.attempt_input_limits,
            verification_input_limits=resume_inputs.verification_input_limits,
            paper_resources=paper_resources,
        ),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=resumed_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("request_user must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(step=2.0),
    )

    assert resumed.outcome == "waiting_user"
    assert resumed.pending_user_question == "请补充旅行日期。"
    assert resume_calls == 2
    assert len(provider_payloads) == 1
    assert provider_payloads[0]["user_input"] == {
        "content": _USER_TEXT,
        "prior_waiting_user_question": None,
    }
    expected_paper_resources = paper_resources.to_dict()
    assert first_payloads[0]["paper_resources"] == expected_paper_resources
    assert provider_payloads[0]["paper_resources"] == expected_paper_resources
    stored_after_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert len(stored_after_resume.attempts) == 1
    assert stored_after_resume.attempts[0].turn_id == second_turn_id
    assert stored_after_resume.related_turn_ids == (first_turn_id, second_turn_id)
    # 旧进程未提交的 100 秒尾段已丢弃；只有新计时器的 2 秒区间被原子稳定。
    assert stored_after_resume.work_run.budget.active_seconds_consumed == 2.0
    _assert_no_assistant_transcript(session_id)


def test_decided_readonly_tool_batch_resumes_after_process_death_without_repeating_final_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="resume-decided-readonly"
    )
    id_plan = WorkRunTurnStableIdPlan(namespace="resume-decided-readonly")
    invocations: list[str] = []

    def read_first(payload: dict[str, str]) -> dict[str, str]:
        invocations.append("first")
        return {"value": payload["value"].upper()}

    def read_second(payload: dict[str, str]) -> dict[str, str]:
        invocations.append("second")
        return {"value": payload["value"].upper()}

    snapshot, allowed_tools, bridge = _readonly_runtime(
        id_plan,
        _readonly_registration("read.first", handler=read_first),
        _readonly_registration("read.second", handler=read_second),
    )
    original_append = work_run_store.append_work_run_tool_result
    append_calls = 0

    def persist_first_result_then_die(**kwargs: object):
        nonlocal append_calls
        append_calls += 1
        mutation = original_append(**kwargs)
        if append_calls == 1:
            raise SystemExit("injected process death after the first durable ToolResult")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "append_work_run_tool_result",
        persist_first_result_then_die,
    )
    with pytest.raises(SystemExit, match="injected process death"):
        run_new_task_node_work_run(
            _request(session_id, first_turn_id, subject),
            catalog_snapshot=snapshot,
            allowed_tools=allowed_tools,
            attempt_provider=lambda _system, _user, **kwargs: _model_result(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": "read.first",
                                "arguments": {"value": "one"},
                            },
                            {
                                "tool_id": "read.second",
                                "arguments": {"value": "two"},
                            },
                        ],
                    },
                },
                str(kwargs["model_call_id"]),
            ),
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("call_tools must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=id_plan,
            tool_bridge=bridge,
            monotonic_clock=_advancing_clock(),
        )

    stranded = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert invocations == ["first"]
    assert stranded.current_attempt_id == id_plan.attempt_id(1)
    assert stranded.attempts[-1].action == "call_tools"
    assert stranded.attempts[-1].decision is not None
    assert len(stranded.tool_calls) == 2
    assert len(stranded.tool_results) == 1

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="TOOL",
        interruption_reason="process_lost_mid_readonly_batch",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-decided-readonly-new-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:30:00+00:00",
            ),
        )
    resume_inputs = _request(session_id, second_turn_id, subject)
    before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    original_resume = work_run_store.resume_decided_readonly_tool_attempt
    resume_calls = 0

    def rebind_then_lose_first_response(**kwargs: object):
        nonlocal resume_calls
        resume_calls += 1
        mutation = original_resume(**kwargs)
        if resume_calls == 1:
            raise RuntimeError("injected decided-tool rebind response loss")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "resume_decided_readonly_tool_attempt",
        rebind_then_lose_first_response,
    )

    resumed = resume_active_task_node_work_run(
        WorkRunTurnResumeRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            subject=subject,
            work_run_id=id_plan.work_run_id,
            attempt_id=id_plan.attempt_id(1),
            expected_work_run_revision=before_resume.work_run.revision,
            expected_window_revision=resume_inputs.expected_window_revision,
            attempt_input_limits=resume_inputs.attempt_input_limits,
            verification_input_limits=resume_inputs.verification_input_limits,
        ),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("decided Attempt recovery must not call the model provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered call_tools must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
    # 在恢复批次后立即强制执行安全轮次上限停止，使该测试将恢复与下一次语义尝试隔离。
        monotonic_clock=_advancing_clock(step=800.0),
    )

    assert resumed.outcome == "turn_limit_reached"
    assert resume_calls == 2
    assert invocations == ["first", "second"]
    recovered = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert len(recovered.attempts) == 1
    assert recovered.attempts[0].turn_id == second_turn_id
    assert recovered.attempts[0].input_turn_id == first_turn_id
    assert recovered.attempts[0].attempt.status.value == "closed"
    assert len(recovered.tool_results) == 2
    assert recovered.related_turn_ids == (first_turn_id, second_turn_id)
    _assert_no_assistant_transcript(session_id)


def test_protected_tool_same_turn_replay_uses_ledger_without_resending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="protected-same-turn-replay"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="protected-same-turn-replay"
    )
    invocations = 0

    def transmit(payload: dict[str, str]) -> dict[str, str]:
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"].upper()}

    snapshot, allowed_tools, bridge = _protected_runtime(
        id_plan,
        _protected_registration("vision.analyze", handler=transmit),
    )
    original_append = work_run_store.append_work_run_tool_result
    append_calls = 0

    def lose_first_result_before_persist(**kwargs: object):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            raise RuntimeError("injected result response loss")
        return original_append(**kwargs)

    monkeypatch.setattr(
        work_run_store,
        "append_work_run_tool_result",
        lose_first_result_before_persist,
    )
    provider_calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        action: dict[str, object]
        if provider_calls == 1:
            action = {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "vision.analyze",
                        "arguments": {"value": "chart"},
                    }
                ],
            }
        else:
            action = {
                "kind": "request_user_input",
                "question": "请确认是否继续。",
            }
        return _model_result(
            {"acceptance_updates": [], "action": action},
            str(kwargs["model_call_id"]),
        )

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tool/ask path must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "waiting_user"
    assert append_calls == 2
    assert provider_calls == 2
    assert invocations == 1
    logical = store.get_runtime_tool_logical_call(
        session_id=session_id,
        logical_tool_call_id=id_plan.tool_call_id(id_plan.attempt_id(1), 1),
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1


@pytest.mark.parametrize(
    "recovery_case",
    ("before_ledger", "after_success", "uncertain_result"),
)
def test_protected_tool_cross_turn_recovery_uses_ledger_without_resending(
    monkeypatch: pytest.MonkeyPatch,
    recovery_case: str,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix=f"protected-cross-turn-recovery-{recovery_case}"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace=f"protected-cross-turn-recovery-{recovery_case}"
    )
    invocations = 0

    def transmit(payload: dict[str, str]) -> dict[str, str]:
        nonlocal invocations
        invocations += 1
        if recovery_case == "uncertain_result":
            raise RuntimeError("provider response was lost")
        return {"value": payload["value"].upper()}

    snapshot, allowed_tools, bridge = _protected_runtime(
        id_plan,
        _protected_registration("vision.analyze", handler=transmit),
    )
    original_append = work_run_store.append_work_run_tool_result
    original_close = work_run_store.close_work_run_attempt
    assert bridge._protected_dispatcher is not None
    original_protected_dispatch = bridge._protected_dispatcher.dispatch

    def die_before_ledger(*_args: object, **_kwargs: object):
        raise SystemExit("injected process death before protected ledger reserve")

    def die_before_result_persist(**_kwargs: object):
        raise SystemExit("injected process death after protected provider success")

    def die_after_uncertain_result(**_kwargs: object):
        raise SystemExit("injected process death after uncertain ToolResult")

    if recovery_case == "before_ledger":
        monkeypatch.setattr(
            bridge._protected_dispatcher,
            "dispatch",
            die_before_ledger,
        )
        death_pattern = "before protected ledger reserve"
    elif recovery_case == "uncertain_result":
        monkeypatch.setattr(
            work_run_store,
            "close_work_run_attempt",
            die_after_uncertain_result,
        )
        death_pattern = "after uncertain ToolResult"
    else:
        monkeypatch.setattr(
            work_run_store,
            "append_work_run_tool_result",
            die_before_result_persist,
        )
        death_pattern = "after protected provider success"
    with pytest.raises(SystemExit, match=death_pattern):
        run_new_task_node_work_run(
            _request(session_id, first_turn_id, subject),
            catalog_snapshot=snapshot,
            allowed_tools=allowed_tools,
            attempt_provider=lambda _system, _user, **kwargs: _model_result(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": "vision.analyze",
                                "arguments": {"value": "chart"},
                            }
                        ],
                    },
                },
                str(kwargs["model_call_id"]),
            ),
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("call_tools must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=id_plan,
            tool_bridge=bridge,
            monotonic_clock=_advancing_clock(),
        )

    stranded = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert invocations == (0 if recovery_case == "before_ledger" else 1)
    assert len(stranded.tool_results) == int(
        recovery_case == "uncertain_result"
    )
    if recovery_case == "uncertain_result":
        assert (
            stranded.tool_results[0].status
            is ToolResultStatus.COMPLETION_UNCONFIRMED
        )
    logical = store.get_runtime_tool_logical_call(
        session_id=session_id,
        logical_tool_call_id=id_plan.tool_call_id(id_plan.attempt_id(1), 1),
    )
    if recovery_case == "before_ledger":
        assert logical is None
    else:
        assert logical is not None
        assert logical.request.invocation_turn_id == first_turn_id
        assert len(logical.physical_attempts) == 1

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="TOOL",
        interruption_reason="process_lost_after_protected_success",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-protected-cross-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:32:00+00:00",
            ),
        )
    resume_inputs = _request(session_id, second_turn_id, subject)
    before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    monkeypatch.setattr(work_run_store, "append_work_run_tool_result", original_append)
    monkeypatch.setattr(work_run_store, "close_work_run_attempt", original_close)

    resumed_dispatch_turns: list[tuple[str, str]] = []

    def record_resumed_protected_dispatch(
        request: ProtectedToolDispatchRequest,
        **kwargs: object,
    ):
        resumed_dispatch_turns.append(
            (request.turn_id, request.invocation_turn_id)
        )
        return original_protected_dispatch(request, **kwargs)

    monkeypatch.setattr(
        bridge._protected_dispatcher,
        "dispatch",
        record_resumed_protected_dispatch,
    )
    resume_request = WorkRunTurnResumeRequest(
        session_id=session_id,
        turn_id=second_turn_id,
        subject=subject,
        work_run_id=id_plan.work_run_id,
        attempt_id=id_plan.attempt_id(1),
        expected_work_run_revision=before_resume.work_run.revision,
        expected_window_revision=resume_inputs.expected_window_revision,
        attempt_input_limits=resume_inputs.attempt_input_limits,
        verification_input_limits=resume_inputs.verification_input_limits,
    )
    bridge_without_protected_recovery = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=id_plan.tool_bridge_persistence_plan,
        authority=AuthorityFacts(
            approval_grants=(
                ScopeGrant(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    "*",
                ),
            )
        ),
    )
    denied = resume_active_task_node_work_run(
        resume_request,
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported recovery must fail before a model call")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported recovery must fail before verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge_without_protected_recovery,
        monotonic_clock=_advancing_clock(step=800.0),
    )
    assert denied.outcome == "failed_closed"
    assert denied.failure_code == "active_attempt_resume_failed"

    resumed = resume_active_task_node_work_run(
        resume_request,
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("decided protected recovery must not call the model")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered call_tools must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(step=800.0),
    )

    assert resumed.outcome == (
        "waiting_external"
        if recovery_case == "uncertain_result"
        else "turn_limit_reached"
    )
    assert resumed_dispatch_turns == (
        []
        if recovery_case == "uncertain_result"
        else [(second_turn_id, first_turn_id)]
    )
    assert invocations == 1
    recovered = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert recovered.attempts[0].turn_id == second_turn_id
    assert len(recovered.tool_results) == 1
    logical_after = store.get_runtime_tool_logical_call(
        session_id=session_id,
        logical_tool_call_id=id_plan.tool_call_id(id_plan.attempt_id(1), 1),
    )
    assert logical_after is not None
    assert logical_after.request.invocation_turn_id == first_turn_id
    assert len(logical_after.physical_attempts) == 1


def test_readonly_tool_close_response_loss_replays_close_and_never_repeats_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="readonly-close-response-loss"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="readonly-close-response-loss"
    )
    invocations = 0

    def read_once(payload: dict[str, str]) -> dict[str, str]:
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"].upper()}

    snapshot, allowed_tools, bridge = _readonly_runtime(
        id_plan,
        _readonly_registration("read.once", handler=read_once),
    )
    original_close = work_run_store.close_work_run_attempt
    close_calls = 0

    def close_then_lose_first_response(**kwargs: object):
        nonlocal close_calls
        close_calls += 1
        mutation = original_close(**kwargs)
        if close_calls == 1:
            raise RuntimeError("injected close response loss")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "close_work_run_attempt",
        close_then_lose_first_response,
    )
    provider_calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            reply = {
                "acceptance_updates": [],
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "read.once",
                            "arguments": {"value": "one"},
                        }
                    ],
                },
            }
        else:
            reply = {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请确认是否继续。",
                },
            }
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tool/ask path must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "waiting_user"
    assert close_calls == 2
    assert invocations == 1
    assert provider_calls == 2
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert len(stored.tool_results) == 1


def test_readonly_tool_close_at_attempt_budget_projects_turn_limit_without_idle_strand() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="readonly-close-attempt-limit"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="readonly-close-attempt-limit"
    )
    handler_calls = 0
    provider_calls = 0

    def read_once(payload: dict[str, str]) -> dict[str, str]:
        nonlocal handler_calls
        handler_calls += 1
        return {"value": payload["value"]}

    snapshot, allowed_tools, bridge = _readonly_runtime(
        id_plan,
        _readonly_registration("read.once", handler=read_once),
    )

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "read.once",
                            "arguments": {"value": str(provider_calls)},
                        }
                    ],
                },
            },
            str(kwargs["model_call_id"]),
        )

    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("call_tools must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "turn_limit_reached"
    assert provider_calls == 32
    assert handler_calls == 32
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert stored.work_run.status is WorkRunStatus.TURN_LIMIT_REACHED
    assert stored.work_run.reason == "turn_limit_reached"
    assert stored.current_attempt_id is None
    assert len(stored.attempts) == 32
    assert stored.attempts[-1].close_reason == "turn_limit_reached"


def test_readonly_tool_close_process_death_resumes_idle_run_on_new_turn_without_repeating_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="readonly-close-process-death"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="readonly-close-process-death"
    )
    invocations = 0

    def read_once(payload: dict[str, str]) -> dict[str, str]:
        nonlocal invocations
        invocations += 1
        return {"value": payload["value"].upper()}

    snapshot, allowed_tools, bridge = _readonly_runtime(
        id_plan,
        _readonly_registration("read.once", handler=read_once),
    )
    original_close = work_run_store.close_work_run_attempt

    def close_then_die(**kwargs: object):
        original_close(**kwargs)
        raise SystemExit("injected process death after durable close")

    monkeypatch.setattr(work_run_store, "close_work_run_attempt", close_then_die)
    with pytest.raises(SystemExit, match="after durable close"):
        run_new_task_node_work_run(
            _request(session_id, first_turn_id, subject),
            catalog_snapshot=snapshot,
            allowed_tools=allowed_tools,
            attempt_provider=lambda _system, _user, **kwargs: _model_result(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": "read.once",
                                "arguments": {"value": "one"},
                            }
                        ],
                    },
                },
                str(kwargs["model_call_id"]),
            ),
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("call_tools must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=id_plan,
            tool_bridge=bridge,
            monotonic_clock=_advancing_clock(),
        )
    closed = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert invocations == 1
    assert closed.work_run.status is WorkRunStatus.ACTIVE
    assert closed.current_attempt_id is None
    assert closed.attempts[-1].attempt.status.value == "closed"

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="TOOL",
        interruption_reason="process_lost_after_readonly_close",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-readonly-close-new-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:35:00+00:00",
            ),
        )
    resume_inputs = _request(session_id, second_turn_id, subject)
    before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    original_idle_resume = (
        work_run_store.resume_idle_readonly_work_run_and_start_attempt
    )
    resume_calls = 0

    def resume_then_lose_first_response(**kwargs: object):
        nonlocal resume_calls
        resume_calls += 1
        mutation = original_idle_resume(**kwargs)
        if resume_calls == 1:
            raise RuntimeError("injected idle resume response loss")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "resume_idle_readonly_work_run_and_start_attempt",
        resume_then_lose_first_response,
    )
    result = resume_active_task_node_work_run(
        WorkRunTurnResumeRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            subject=subject,
            work_run_id=id_plan.work_run_id,
            attempt_id=id_plan.attempt_id(1),
            expected_work_run_revision=before_resume.work_run.revision,
            expected_window_revision=resume_inputs.expected_window_revision,
            attempt_input_limits=resume_inputs.attempt_input_limits,
            verification_input_limits=resume_inputs.verification_input_limits,
        ),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请确认是否继续。",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered ask path must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "waiting_user"
    assert resume_calls == 2
    assert invocations == 1
    recovered = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in recovered.attempts) == (1, 2)
    assert recovered.attempts[0].turn_id == first_turn_id
    assert recovered.attempts[0].input_turn_id == first_turn_id
    assert recovered.attempts[1].turn_id == second_turn_id
    assert recovered.attempts[1].input_turn_id == second_turn_id
    assert len(recovered.tool_results) == 1
    late_replay = original_idle_resume(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id=id_plan.work_run_id,
        closed_attempt_id=id_plan.attempt_id(1),
        next_attempt_id=id_plan.attempt_id(2),
        expected_work_run_revision=before_resume.work_run.revision,
        expected_window_revision=resume_inputs.expected_window_revision,
        catalog_snapshot=snapshot.to_descriptor(),
        apply_id=id_plan.resume_idle_work_run_apply_id(1, second_turn_id),
    )
    assert late_replay.status == "replayed"
    assert late_replay.current_attempt_id == id_plan.attempt_id(2)


def test_idle_readonly_recovery_preserves_latest_nonpass_verification_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, subject = _seed_ready_node(
        suffix="readonly-close-verification-feedback"
    )
    id_plan = WorkRunTurnStableIdPlan(
        namespace="readonly-close-verification-feedback"
    )
    snapshot, allowed_tools, bridge = _readonly_runtime(
        id_plan,
        _readonly_registration(
            "read.once",
            handler=lambda payload: {"value": payload["value"]},
        ),
    )
    provider_calls = 0

    def initial_provider(
        _system: str,
        _user: str,
        **kwargs: object,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            action: dict[str, object] = {
                "kind": "submit_output_window",
                "content": "旅行计划草稿。",
                "format": "plain_text",
            }
            updates = _acceptance_update()
        else:
            action = {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "read.once",
                        "arguments": {"value": "补充事实"},
                    }
                ],
            }
            updates = []
        return _model_result(
            {"acceptance_updates": updates, "action": action},
            str(kwargs["model_call_id"]),
        )

    original_close = work_run_store.close_work_run_attempt

    def close_then_die(**kwargs: object):
        original_close(**kwargs)
        raise SystemExit("injected process death after feedback-bound close")

    monkeypatch.setattr(work_run_store, "close_work_run_attempt", close_then_die)
    with pytest.raises(SystemExit, match="feedback-bound close"):
        run_new_task_node_work_run(
            _request(session_id, first_turn_id, subject),
            catalog_snapshot=snapshot,
            allowed_tools=allowed_tools,
            attempt_provider=initial_provider,
            verification_provider=lambda _system, _user, **kwargs: _model_result(
                _verification_reply(passed=False),
                str(kwargs["model_call_id"]),
            ),
            emit=lambda _event: None,
            id_plan=id_plan,
            tool_bridge=bridge,
            monotonic_clock=_advancing_clock(),
        )
    stranded = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    feedback_id = stranded.attempts[-1].input_verification_request_id
    assert feedback_id is not None
    assert stranded.attempts[-1].attempt.ordinal == 2

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="TOOL",
        interruption_reason="process_lost_after_feedback_bound_close",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-readonly-feedback-new-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-run-turn-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:40:00+00:00",
            ),
        )
    resume_inputs = _request(session_id, second_turn_id, subject)
    before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    resumed_payloads: list[dict[str, object]] = []

    def resumed_provider(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        resumed_payloads.append(json.loads(user_content))
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请确认是否继续。",
                },
            },
            str(kwargs["model_call_id"]),
        )

    result = resume_active_task_node_work_run(
        WorkRunTurnResumeRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            subject=subject,
            work_run_id=id_plan.work_run_id,
            attempt_id=id_plan.attempt_id(2),
            expected_work_run_revision=before_resume.work_run.revision,
            expected_window_revision=resume_inputs.expected_window_revision,
            attempt_input_limits=resume_inputs.attempt_input_limits,
            verification_input_limits=resume_inputs.verification_input_limits,
        ),
        catalog_snapshot=snapshot,
        allowed_tools=allowed_tools,
        attempt_provider=resumed_provider,
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered ask path must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        tool_bridge=bridge,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "waiting_user"
    recovered = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert recovered.attempts[-1].attempt.ordinal == 3
    assert recovered.attempts[-1].input_verification_request_id == feedback_id
    assert recovered.attempts[-1].input_checkpoint_id == feedback_id
    assert resumed_payloads[0]["verification_feedback"] is not None


def test_output_commit_response_loss_returns_the_latest_authoritative_cursor(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="write-response-loss")
    original_commit = work_run_store.commit_work_run_output_action

    def commit_then_lose_response(**kwargs: object):
        original_commit(**kwargs)
        raise RuntimeError("injected response loss after durable output commit")

    monkeypatch.setattr(
        work_run_store,
        "commit_work_run_output_action",
        commit_then_lose_response,
    )
    id_plan = WorkRunTurnStableIdPlan(namespace="write-response-loss")
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "write_output_window",
                    "content": "已持久化但响应丢失的工作稿。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("write response loss must not invoke verification")
        ),
        emit=lambda _event: None,
        id_plan=id_plan,
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.outcome == "internal_interrupted"
    assert result.interruption_reason == "attempt_decision_interrupted"
    assert result.window_revision == int(window["state_version"])
    assert result.window_revision == 4
    assert result.current_attempt_id is None
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert stored.current_attempt_id is None
    assert stored.output_window.content == "已持久化但响应丢失的工作稿。"
    assert stored.attempts[-1].action == "write_output_window"


def test_reconciliation_never_returns_another_turns_window_cursor(monkeypatch) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="window-drift")
    authoritative_getter = store.get_turn_execution_window

    def provider_then_drift(
        *_args: object,
        **_kwargs: object,
    ) -> ModelResult:
        current = authoritative_getter(session_id)
        assert current is not None
        drifted = {
            **current,
            "turn_id": "a-newer-turn",
            "state_version": int(current["state_version"]) + 1,
        }
        monkeypatch.setattr(
            store,
            "get_turn_execution_window",
            lambda _session_id: drifted,
        )
        raise RuntimeError("provider failed after Window ownership drift")

    with pytest.raises(
        WorkRunTurnAuthorityUnavailable,
        match="originating Turn",
    ):
        run_new_task_node_work_run(
            _request(session_id, turn_id, subject),
            catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
            allowed_tools=(),
            attempt_provider=provider_then_drift,
            verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Window drift must not invoke verification")
            ),
            emit=lambda _event: None,
            id_plan=WorkRunTurnStableIdPlan(namespace="window-drift"),
            monotonic_clock=_advancing_clock(),
        )


def test_verification_commit_response_loss_replays_the_same_request_once(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="verify-response-loss")
    original_commit = verification_store.commit_task_node_verification_result
    commit_calls = 0
    provider_calls = 0

    def commit_then_lose_first_response(**kwargs: object):
        nonlocal commit_calls
        commit_calls += 1
        mutation = original_commit(**kwargs)
        if commit_calls == 1:
            raise RuntimeError("injected response loss after verification commit")
        return mutation

    def verification_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        assert purpose == "runtime_task_node_semantic_verification"
        provider_calls += 1
        return _model_result(_verification_reply(passed=True), model_call_id)

    monkeypatch.setattr(
        verification_store,
        "commit_task_node_verification_result",
        commit_then_lose_first_response,
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=verification_provider,
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="verify-response-loss"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert result.delivery_id == "verify-response-loss:delivery:1"
    assert provider_calls == 1
    # 即使响应丢失，首次提交仍已持久化。恢复会重放小型稳定回执，而不是重新提交
    # 语义结果或重新计量活跃时间。
    assert commit_calls == 1


def test_delivery_getter_failure_after_pass_does_not_hide_committed_delivery(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="delivery-read-loss")
    original_getter = verification_store.get_task_node_delivery

    monkeypatch.setattr(
        verification_store,
        "get_task_node_delivery",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected delivery projection failure")
        ),
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="delivery-read-loss"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert result.delivery_id == "delivery-read-loss:delivery:1"
    delivery = original_getter(
        session_id=session_id,
        delivery_id=result.delivery_id,
    )
    assert delivery.delivery.work_run_id == "delivery-read-loss:workrun"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert result.window_revision == int(window["state_version"])


def test_missing_pass_projection_delivery_reconciles_from_committed_authority(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="missing-pass-projection")
    original_invoke = controller_module._invoke_node_verification

    def invoke_then_drop_delivery(*args: object, **kwargs: object):
        committed = original_invoke(*args, **kwargs)
        assert committed.outcome == "passed"
        return SimpleNamespace(
            outcome="passed",
            store_projection=SimpleNamespace(
                delivery_id=None,
                window_revision=committed.store_projection.window_revision,
            ),
            next_attempt_mutation=None,
        )

    monkeypatch.setattr(
        controller_module,
        "_invoke_node_verification",
        invoke_then_drop_delivery,
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一天抵达，第二天参观，第三天返程。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="missing-pass-projection"),
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "delivery_ready"
    assert result.delivery_id == "missing-pass-projection:delivery:1"


def test_reused_namespace_for_another_task_cannot_return_the_old_delivery() -> None:
    session_id, turn_id, first_subject = _seed_ready_node(
        suffix="namespace-first"
    )
    reused_plan = WorkRunTurnStableIdPlan(namespace="reused-namespace")
    first = run_new_task_node_work_run(
        _request(session_id, turn_id, first_subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": _acceptance_update(),
                "action": {
                    "kind": "submit_output_window",
                    "content": "第一份任务的已验证交付。",
                    "format": "plain_text",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda _system, _user, **kwargs: _model_result(
            _verification_reply(passed=True),
            str(kwargs["model_call_id"]),
        ),
        emit=lambda _event: None,
        id_plan=reused_plan,
        monotonic_clock=_advancing_clock(),
    )
    assert first.outcome == "delivery_ready"

    second_subject = _insert_additional_ready_node(
        session_id=session_id,
        turn_id=turn_id,
        suffix="namespace-second",
    )
    second = run_new_task_node_work_run(
        _request(session_id, turn_id, second_subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("create collision must fail before the Attempt provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("create collision must fail before the verifier")
        ),
        emit=lambda _event: None,
        id_plan=reused_plan,
        monotonic_clock=_advancing_clock(),
    )

    assert second.outcome == "failed_closed"
    assert second.failure_code == "work_run_create_failed"
    assert second.delivery_id is None
    assert second.pending_user_question is None


def test_create_response_loss_requires_and_accepts_only_exact_store_replay(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="create-replay")
    original_create = work_run_store.create_task_node_work_run
    create_calls = 0

    def create_then_lose_first_response(**kwargs: object):
        nonlocal create_calls
        create_calls += 1
        mutation = original_create(**kwargs)
        if create_calls == 1:
            raise RuntimeError("injected response loss after WorkRun create")
        return mutation

    monkeypatch.setattr(
        work_run_store,
        "create_task_node_work_run",
        create_then_lose_first_response,
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda _system, _user, **kwargs: _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请提供旅行日期。",
                },
            },
            str(kwargs["model_call_id"]),
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("request_user must not invoke the verifier")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="create-replay"),
        monotonic_clock=_advancing_clock(),
    )

    assert create_calls == 2
    assert result.outcome == "waiting_user"
    assert result.work_run_id == "create-replay:workrun"


def test_two_lost_create_responses_reconcile_to_internal_interruption(
    monkeypatch,
) -> None:
    session_id, turn_id, subject = _seed_ready_node(suffix="create-double-loss")
    original_create = work_run_store.create_task_node_work_run
    create_calls = 0

    def create_or_replay_then_lose_response(**kwargs: object):
        nonlocal create_calls
        create_calls += 1
        original_create(**kwargs)
        raise RuntimeError("injected loss after create commit or exact replay")

    monkeypatch.setattr(
        work_run_store,
        "create_task_node_work_run",
        create_or_replay_then_lose_response,
    )
    result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lost create responses must stop before Attempt provider")
        ),
        verification_provider=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lost create responses must stop before verifier")
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(namespace="create-double-loss"),
        monotonic_clock=_advancing_clock(),
    )

    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert create_calls == 2
    assert result.outcome == "internal_interrupted"
    assert result.work_run_id == "create-double-loss:workrun"
    assert result.current_attempt_id is None
    assert result.interruption_reason == "work_run_create_response_lost"
    assert result.window_revision == int(window["state_version"])
    assert window["current_work_run_id"] == result.work_run_id
