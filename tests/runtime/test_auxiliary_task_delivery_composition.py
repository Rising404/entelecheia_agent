from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from personagraph.l2.auxiliary_graph import AuxiliaryGraphStructureProposal
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.task_graph import (
    InSessionTaskStatus,
    TaskDeliveryValidationDimension,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.l2.auxiliary_execution.delivery import composition as delivery
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectGuardError,
    AuxiliaryGraphArchitectRequest,
    validate_auxiliary_graph_architect_proposal,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryStatus,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunProfile,
    TaskGraphWorkRunResult,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
    _seed_task,
)
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _settled_positive_base,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def test_default_delivery_model_factories_share_the_injected_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, object]] = []
    node_result = object()
    candidate_result = object()

    def create_node(
        _binding: object,
        *,
        rederive_state_guard_sha256: object,
        ledger_store: object,
    ) -> object:
        assert callable(rederive_state_guard_sha256)
        observed.append(("node", ledger_store))
        return node_result

    def create_candidate(
        _authority: object,
        *,
        rederive_state_guard_sha256: object,
        ledger_store: object,
    ) -> object:
        assert callable(rederive_state_guard_sha256)
        observed.append(("candidate", ledger_store))
        return candidate_result

    monkeypatch.setattr(
        delivery,
        "create_task_node_work_run_model_call_authority",
        create_node,
    )
    monkeypatch.setattr(
        delivery,
        "create_task_delivery_candidate_model_call_authority",
        create_candidate,
    )
    ports = AuxiliaryTaskDeliveryPorts(
        model_ledger_store=store,
        emit=lambda _event: None,
    )

    node_factory = delivery._task_node_model_call_authority_factory(ports)
    candidate_factory = (
        delivery._task_candidate_model_call_authority_factory(ports)
    )

    assert node_factory is not None
    assert node_factory(
        object(),
        rederive_state_guard_sha256=lambda: "node-guard",
    ) is node_result
    assert candidate_factory(
        object(),
        rederive_state_guard_sha256=lambda: "candidate-guard",
    ) is candidate_result
    assert observed == [("node", store), ("candidate", store)]


def test_custom_delivery_model_factories_remain_exactly_injectable() -> None:
    def custom_node(*_args: object, **_kwargs: object) -> object:
        return object()

    def custom_candidate(*_args: object, **_kwargs: object) -> object:
        return object()

    ports = AuxiliaryTaskDeliveryPorts(
        model_ledger_store=store,
        emit=lambda _event: None,
        task_node_model_call_authority_factory=custom_node,
        task_candidate_validation_model_call_authority_factory=(
            custom_candidate
        ),
    )

    assert (
        delivery._task_node_model_call_authority_factory(ports)
        is custom_node
    )
    assert (
        delivery._task_candidate_model_call_authority_factory(ports)
        is custom_candidate
    )


def test_delivery_ports_reject_an_incomplete_model_ledger() -> None:
    with pytest.raises(
        TypeError,
        match="model_ledger_store.reserve_runtime_model_logical_call",
    ):
        AuxiliaryTaskDeliveryPorts(
            model_ledger_store=object(),  # type: ignore[arg-type]
            emit=lambda _event: None,
        )


def _commit_task_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str, str]:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
    )
    assert result.status is AuxiliaryApplicationStatus.COMMITTED
    return session_id, turn_id, task_id


def _task_execution_counts(session_id: str, task_id: str) -> tuple[int, int]:
    with store._connect() as conn:
        work_runs = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_runs "
                "WHERE session_id=? AND insession_task_id=? "
                "AND subject_kind='task_node'",
                (session_id, task_id),
            ).fetchone()[0]
        )
        deliveries = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchone()[0]
        )
    return work_runs, deliveries


def test_production_ports_use_full_task_node_model_envelope() -> None:
    ports = AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None)

    assert ports.profile.attempt_max_output_tokens == 131_072
    assert ports.profile.verification_max_output_tokens == 131_072
    assert ports.profile.model_timeout_s == 180.0

    # 通用配置采用相同的生产输出上限。
    demo = TaskGraphWorkRunProfile()
    assert demo.attempt_max_output_tokens == 131_072
    assert demo.verification_max_output_tokens == 131_072
    assert demo.model_timeout_s == 45.0


def test_committed_graph_runs_to_verified_body_and_reentry_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    structured = TaskGraphWorkRunProfile().structured_model_profile()
    default_attempt = build_attempt_structured_provider(structured)
    default_verification = build_verification_structured_provider(structured)
    model_calls: list[str] = []
    source_contexts: list[dict[str, object]] = []

    def attempt_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        model_calls.append(purpose)
        source_contexts.append(json.loads(user_content)["source_context"])
        return default_attempt(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    def verification_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        model_calls.append(purpose)
        source_contexts.append(json.loads(user_content)["source_context"])
        return default_verification(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    request = AuxiliaryTaskDeliveryRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    first = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(attempt_provider),
            verification_provider=as_prepared_test_provider(
                verification_provider
            ),
        ),
    )

    assert first.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert first.reason_code == "whole_task_candidate_pass"
    assert first.final_delivery_id is not None
    assert first.publication_body is not None
    assert first.publication_body.strip()
    assert first.publication_format == "markdown"
    assert first.replayed_delivery is False
    assert first.task_candidate_settlement is not None
    assert first.task_candidate_settlement.settlement.disposition.value == "pass"
    assert model_calls == [
        "runtime_work_run_attempt_decision",
        "runtime_task_node_semantic_verification",
    ]
    assert len(source_contexts) == 2
    assert source_contexts[0] == source_contexts[1]
    assert source_contexts[0]["session_id"] == session_id
    assert source_contexts[0]["task_id"] == task_id
    assert source_contexts[0]["node_source_anchor_ids"] == [
        "task_creation_source"
    ]
    assert source_contexts[0]["anchors"][0]["excerpt"]
    assert len(source_contexts[0]["authority_sha256"]) == 64
    resolved = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=first.final_delivery_id,
    )
    assert first.publication_body == resolved.output_window.content
    assert resolved.delivery.subject.task_id == task_id
    assert resolved.delivery.subject.graph_revision == 1
    before_reentry = _task_execution_counts(session_id, task_id)
    assert before_reentry == (1, 1)
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM session_turns WHERE role='assistant'"
            ).fetchone()[0]
        ) == 0

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("completed Delivery replay must not invoke a model")

    monkeypatch.setattr(store, "get_turn_execution_window", lambda _session_id: None)
    replayed = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )

    assert replayed.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert replayed.reason_code == "whole_task_candidate_pass_replayed"
    assert replayed.final_delivery_id == first.final_delivery_id
    assert replayed.publication_body == first.publication_body
    assert replayed.replayed_delivery is True
    assert _task_execution_counts(session_id, task_id) == before_reentry
    assert model_calls == [
        "runtime_work_run_attempt_decision",
        "runtime_task_node_semantic_verification",
    ]


def test_default_task_node_attempt_and_verification_use_runtime_model_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    request = AuxiliaryTaskDeliveryRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )

    first = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert first.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT logical.logical_call_id, logical.call_kind, "
            "logical.execution_subject_id, logical.auxiliary_graph_id, "
            "logical.goal_id, physical.physical_ordinal, physical.status "
            "FROM insession_runtime_model_logical_calls AS logical "
            "JOIN insession_runtime_model_physical_attempts AS physical "
            "ON physical.session_id=logical.session_id "
            "AND physical.logical_call_id=logical.logical_call_id "
            "WHERE logical.session_id=? AND logical.insession_task_id=? "
            "AND logical.auxiliary_graph_id IS NULL "
            "AND logical.goal_id IS NULL "
            "AND logical.call_kind IN "
            "('attempt_decision', 'node_verification') "
            "ORDER BY logical.call_kind",
            (session_id, task_id),
        ).fetchall()
    call_kinds = [str(row["call_kind"]) for row in rows]
    assert call_kinds.count("attempt_decision") >= 1
    assert call_kinds.count("attempt_decision") == call_kinds.count(
        "node_verification"
    )
    assert all(
        str(row["execution_subject_id"]).startswith("execsubject-")
        for row in rows
    )
    assert all(row["auxiliary_graph_id"] is None for row in rows)
    assert all(row["goal_id"] is None for row in rows)
    assert all(int(row["physical_ordinal"]) == 1 for row in rows)
    assert all(str(row["status"]) == "succeeded" for row in rows)
    logical_ids = tuple(str(row["logical_call_id"]) for row in rows)

    replayed = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert replayed.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert replayed.replayed_delivery is True
    with store._connect() as conn:
        after = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT logical_call_id FROM "
                "insession_runtime_model_logical_calls WHERE session_id=? "
                "AND auxiliary_graph_id IS NULL AND goal_id IS NULL "
                "AND call_kind IN ('attempt_decision', 'node_verification') "
                "ORDER BY call_kind",
                (session_id,),
            ).fetchall()
        )
    assert after == logical_ids


def test_completed_task_without_candidate_settlement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    completed_task = task.model_copy(
        update={"status": InSessionTaskStatus.COMPLETED}
    )
    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        lambda _session_id, _task_id: completed_task,
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_task_delivery_candidate_settlement",
        lambda **_kwargs: None,
    )
    request = AuxiliaryTaskDeliveryRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("missing candidate settlement reached a Provider")

    rejected = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
            task_candidate_validation_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )

    assert rejected.status is AuxiliaryTaskDeliveryStatus.FAILED
    assert (
        rejected.reason_code
        == "task_delivery_candidate_settlement_missing_for_completed_task"
    )
    assert rejected.final_delivery_id is None
    assert rejected.publication_body is None


def test_revision_trigger_without_candidate_settlement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    monkeypatch.setattr(
        task_delivery_store,
        "get_task_delivery_candidate_settlement",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda **_kwargs: SimpleNamespace(
            base_graph_revision=1,
            target_graph_revision=2,
        ),
    )

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("orphaned revision trigger reached a Provider")

    rejected = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
            task_candidate_validation_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )

    assert rejected.status is AuxiliaryTaskDeliveryStatus.FAILED
    assert (
        rejected.reason_code
        == "task_delivery_candidate_settlement_missing_for_revision_trigger"
    )


def test_root_delivery_is_withheld_when_whole_task_model_requires_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)

    def task_validator(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        payload = {
            "findings": [
                {
                    "dimension": dimension.value,
                    "verdict": (
                        "fail"
                        if dimension
                        is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
                        else "pass"
                    ),
                    "fault_domain": (
                        "task_graph_design"
                        if dimension
                        is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
                        else "none"
                    ),
                    "finding": (
                        "The final body omitted the required release step."
                        if dimension
                        is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
                        else f"{dimension.value} passed."
                    ),
                    "affected_node_ids": [],
                    "evidence_anchor_ids": [],
                }
                for dimension in TaskDeliveryValidationDimension
            ],
            "summary": "The final answer is incomplete.",
            "execution_repair_objective": None,
            "task_graph_revision_objective": (
                "Preserve verified content and add the required release step."
            ),
            "blocking_questions": [],
        }
        return ModelResult(
            reply=json.dumps(payload),
            provider="mock",
            model="mock-structured",
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
                task_validator
            ),
        ),
    )

    assert result.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert result.reason_code == "whole_task_candidate_replan_task_graph"
    assert result.final_delivery_id is None
    assert result.publication_body is None
    assert result.publication_format is None
    assert result.task_candidate_settlement is not None
    trigger = result.task_candidate_settlement.trigger
    assert trigger is not None
    assert trigger.base_graph_revision == 1
    assert trigger.target_graph_revision == 2
    assert "release step" in trigger.revision_objective
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.status.value == "active"
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) == trigger
    before_counts = _task_execution_counts(session_id, task_id)
    before_version = task.task_state_version

    def must_not_call_model(*_args, **_kwargs):
        raise AssertionError("REVISE replay must not dispatch another model call")

    replayed = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(must_not_call_model),
            verification_provider=as_prepared_test_provider(
                must_not_call_model
            ),
            task_candidate_validation_provider=as_prepared_test_provider(
                must_not_call_model
            ),
        ),
    )

    assert replayed.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert replayed.reason_code == "whole_task_candidate_replan_task_graph_replayed"
    assert replayed.publication_body is None
    replayed_task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert replayed_task is not None
    assert replayed_task.task_state_version == before_version
    assert _task_execution_counts(session_id, task_id) == before_counts


def test_whole_task_insufficient_evidence_waits_for_user_with_exact_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    questions = (
        "Which release environment should the final procedure target?",
    )
    candidate_calls = 0

    def task_validator(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal candidate_calls
        candidate_calls += 1
        payload = {
            "findings": [
                {
                    "dimension": dimension.value,
                    "verdict": (
                        "insufficient_evidence"
                        if dimension
                        is TaskDeliveryValidationDimension.EVIDENCE_GROUNDING
                        else "pass"
                    ),
                    "fault_domain": (
                        "missing_information"
                        if dimension
                        is TaskDeliveryValidationDimension.EVIDENCE_GROUNDING
                        else "none"
                    ),
                    "finding": (
                        "The available evidence does not identify the target "
                        "environment."
                        if dimension
                        is TaskDeliveryValidationDimension.EVIDENCE_GROUNDING
                        else f"{dimension.value} passed."
                    ),
                    "affected_node_ids": [],
                    "evidence_anchor_ids": [],
                }
                for dimension in TaskDeliveryValidationDimension
            ],
            "summary": "Publication needs one user-owned fact.",
            "execution_repair_objective": None,
            "task_graph_revision_objective": None,
            "blocking_questions": list(questions),
        }
        return ModelResult(
            reply=json.dumps(payload),
            provider="mock",
            model="mock-structured",
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
                task_validator
            ),
        ),
    )

    assert result.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert result.reason_code == "whole_task_candidate_blocked_replan"
    assert result.requested_user_questions == ()
    assert result.final_delivery_id is None
    assert result.publication_body is None
    assert result.task_candidate_settlement is not None
    assert result.task_candidate_settlement.intent.result.blocking_questions == questions
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.status.value == "active"
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    assert trigger.base_graph_revision == 1
    assert trigger.target_graph_revision == 2
    assert result.task_candidate_settlement.trigger == trigger

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("durable candidate BLOCKED replay reached a Provider")

    replayed = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
            task_candidate_validation_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )
    assert replayed.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert replayed.reason_code == "whole_task_candidate_blocked_replan_replayed"
    assert replayed.requested_user_questions == ()
    assert replayed.task_candidate_settlement is not None
    assert replayed.task_candidate_settlement.trigger == trigger
    assert candidate_calls == 1

    physical_planner = build_auxiliary_architect_structured_provider()
    physical_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    planner_calls = 0

    def planner_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal planner_calls
        planner_calls += 1
        prompt = json.loads(str(args[1]))
        route = prompt["task_graph_revision_route"]
        assert route["disposition"] == "blocked"
        assert route["route_kind"] == "missing_information_clarification"
        assert route["requires_user_gate"] is True
        assert route["blocking_questions"] == list(questions)
        return physical_planner(*args, **kwargs)  # type: ignore[arg-type]

    def user_gate_attempt_provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        payload = json.loads(user_content)
        contract = payload.get("user_gate_contract")
        if contract is None:
            return physical_attempt(
                _system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
            )
        phase = contract["phase"]
        question = payload["node"]["objective"]
        assert question in questions
        if phase == "ask":
            action = {
                "kind": "request_user_input",
                "question": question,
            }
            acceptance_updates: list[dict[str, object]] = []
        else:
            assert phase == "consume_answer"
            action = {
                "kind": "submit_output_window",
                "content": "accept_answer",
                "format": "plain_text",
            }
            acceptance_updates = [
                {
                    "acceptance_id": "blocking_question_01_clarified",
                    "model_claimed_satisfied": True,
                }
            ]
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": acceptance_updates,
                    "action": action,
                }
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    boundary = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(planner_provider),
            attempt_provider=as_prepared_test_provider(
                user_gate_attempt_provider
            ),
        ),
    )

    assert boundary.status is AuxiliaryApplicationStatus.WAITING_USER
    assert boundary.requested_user_question in questions
    assert planner_calls == 1
    assert candidate_calls == 1
    planned = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert planned is not None
    gates = tuple(
        node
        for node in planned.nodes
        if node.executor_kind == "user_gate" and node.required
    )
    assert {node.objective for node in gates} == set(questions)

    assert boundary.planning_result is not None
    decision = boundary.planning_result.architect_decision
    assert decision is not None
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=decision.logical_call_id,
    )
    assert logical is not None
    architect_request = AuxiliaryGraphArchitectRequest.model_validate_json(
        logical.request.request_json
    )
    route = architect_request.prompt_payload.task_graph_revision_route
    assert route is not None and route.requires_user_gate is True
    assert route.route_sha256
    proposal = decision.proposal
    assert proposal.structure is not None
    non_gate_keys = {
        node.node_key
        for node in proposal.structure.nodes
        if node.executor_kind.value != "user_gate"
    }
    invalid_structure = AuxiliaryGraphStructureProposal(
        terminal_node_key=proposal.structure.terminal_node_key,
        nodes=tuple(
            node
            for node in proposal.structure.nodes
            if node.node_key in non_gate_keys
        ),
        edges=tuple(
            edge
            for edge in proposal.structure.edges
            if edge.source_node_key in non_gate_keys
            and edge.target_node_key in non_gate_keys
        ),
    )
    with pytest.raises(AuxiliaryGraphArchitectGuardError, match="user_gate"):
        validate_auxiliary_graph_architect_proposal(
            request=architect_request,
            proposal=proposal.model_copy(update={"structure": invalid_structure}),
        )

    answer = "Target the staging environment."
    current_window = store.get_turn_execution_window(session_id)
    assert current_window is not None
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(current_window["state_version"]),
        stage="RESPONSE",
        interruption_reason="WAITING_USER",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="host_stopped",
        error_code="WAITING_USER",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="candidate-blocked-user-answer",
        source="auxiliary_v2_candidate_blocked_test",
        user_text=answer,
        lease_owner="auxiliary-v2-candidate-blocked-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    answer_window = store.get_turn_execution_window(session_id)
    assert answer_window is not None
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=answer_turn_id,
        apply_id="candidate-blocked-user-answer-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": answer,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=int(answer_window["state_version"]),
    )
    assert task_delivery_store.get_task_delivery_candidate_settlement(
        session_id=session_id,
        task_id=task_id,
        graph_revision=1,
    ) is not None
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) == trigger
    committed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=answer_turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "settled clarification plan reached planning Provider again"
                )
            ),
            attempt_provider=as_prepared_test_provider(
                user_gate_attempt_provider
            ),
        ),
    )
    assert committed.status is AuxiliaryApplicationStatus.COMMITTED, (
        committed.reason_code,
        committed.last_driver_action,
    )
    revised_task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert revised_task is not None
    assert revised_task.current_graph_revision == 2
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None

    delivered = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=answer_turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
    )
    assert delivered.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert delivered.final_delivery_id is not None
    assert delivered.final_delivery_id != trigger.root_delivery_id
    assert delivered.publication_body


def test_whole_task_missing_formal_authority_fails_closed_without_user_gate_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)

    def task_validator(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        return ModelResult(
            reply=json.dumps(
                {
                    "findings": [
                        {
                            "dimension": dimension.value,
                            "verdict": (
                                "insufficient_evidence"
                                if dimension
                                is TaskDeliveryValidationDimension.CONSTRAINT_FIDELITY
                                else "pass"
                            ),
                            "fault_domain": (
                                "missing_authority"
                                if dimension
                                is TaskDeliveryValidationDimension.CONSTRAINT_FIDELITY
                                else "none"
                            ),
                            "finding": (
                                "The protected release operation has no formal "
                                "Authorization/Approval/Receipt."
                                if dimension
                                is TaskDeliveryValidationDimension.CONSTRAINT_FIDELITY
                                else f"{dimension.value} passed."
                            ),
                            "affected_node_ids": [],
                            "evidence_anchor_ids": [],
                        }
                        for dimension in TaskDeliveryValidationDimension
                    ],
                    "summary": "A protected effect lacks formal authority.",
                    "execution_repair_objective": None,
                    "task_graph_revision_objective": None,
                    "blocking_questions": [
                        "Provide the formal protected-operation authorization receipt."
                    ],
                }
            ),
            provider="mock",
            model="mock-structured",
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
                task_validator
            ),
        ),
    )

    assert result.status is AuxiliaryTaskDeliveryStatus.FAILED
    assert result.reason_code == "verification_application_interrupted"
    assert task_delivery_store.get_task_delivery_candidate_settlement(
        session_id=session_id,
        task_id=task_id,
        graph_revision=1,
    ) is None
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.status.value == "active"


def test_positive_base_revision_two_runs_to_verified_body_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, _base_root_id = _settled_positive_base("delivery-positive-base")
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert committed.previous_graph_revision == 1
    assert committed.committed_graph_revision == 2
    assert committed.carry_receipt_ids == ()

    structured = TaskGraphWorkRunProfile().structured_model_profile()
    attempt_provider = build_attempt_structured_provider(structured)
    verification_provider = build_verification_structured_provider(structured)
    model_calls: list[str] = []

    def counted_attempt(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        model_calls.append(purpose)
        return attempt_provider(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    def counted_verification(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        model_calls.append(purpose)
        return verification_provider(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    request = AuxiliaryTaskDeliveryRequest(
        session_id=command.session_id,
        turn_id=command.source_turn_id,
        task_id=command.task_id,
    )
    first = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(counted_attempt),
            verification_provider=as_prepared_test_provider(
                counted_verification
            ),
        ),
    )

    assert first.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert first.reason_code == "whole_task_candidate_pass"
    assert first.final_delivery_id is not None
    assert first.publication_body is not None
    resolved = verification_store.get_task_node_delivery(
        session_id=command.session_id,
        delivery_id=first.final_delivery_id,
    )
    assert resolved.delivery.subject.task_id == command.task_id
    assert resolved.delivery.subject.graph_revision == 2
    assert _task_execution_counts(command.session_id, command.task_id) == (2, 2)
    assert model_calls == [
        "runtime_work_run_attempt_decision",
        "runtime_task_node_semantic_verification",
        "runtime_work_run_attempt_decision",
        "runtime_task_node_semantic_verification",
    ]

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("completed Revision 2 replay must not invoke a model")

    before_reentry = _task_execution_counts(command.session_id, command.task_id)
    monkeypatch.setattr(store, "get_turn_execution_window", lambda _session_id: None)
    replayed = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )

    assert replayed.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert replayed.reason_code == "whole_task_candidate_pass_replayed"
    assert replayed.final_delivery_id == first.final_delivery_id
    assert replayed.publication_body == first.publication_body
    assert replayed.replayed_delivery is True
    assert _task_execution_counts(command.session_id, command.task_id) == (
        before_reentry
    )


def test_waiting_user_is_durable_and_never_exposes_an_output_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    provider_calls = 0

    def ask_provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充完成任务所需的关键信息。",
                    },
                },
                ensure_ascii=False,
            ),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    request = AuxiliaryTaskDeliveryRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    waiting = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(ask_provider),
            verification_provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("a user question must not be verified")
                )
            ),
        ),
    )

    assert waiting.status is AuxiliaryTaskDeliveryStatus.WAITING_USER
    assert waiting.pending_question_attempt_ids
    assert waiting.final_delivery_id is None
    assert waiting.publication_body is None
    assert provider_calls == 1
    assert _task_execution_counts(session_id, task_id) == (1, 0)

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("waiting-user replay must not invoke a model")

    replayed = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )

    assert replayed.status is AuxiliaryTaskDeliveryStatus.WAITING_USER
    assert replayed.pending_question_attempt_ids == (
        waiting.pending_question_attempt_ids
    )
    assert replayed.publication_body is None
    assert provider_calls == 1


@pytest.mark.parametrize(
    ("task_graph_status", "expected", "reason"),
    (
        (
            "waiting_external",
            AuxiliaryTaskDeliveryStatus.WAITING_EXTERNAL,
            "waiting_external",
        ),
        (
            "turn_limit_reached",
            AuxiliaryTaskDeliveryStatus.TURN_LIMIT,
            "host_work_run_limit_reached",
        ),
        (
            "failed_closed",
            AuxiliaryTaskDeliveryStatus.FAILED,
            "injected_failed_closed",
        ),
        (
            "completed",
            AuxiliaryTaskDeliveryStatus.FAILED,
            "task_delivery_candidate_settlement_missing_after_graph_completion",
        ),
    ),
)
def test_maps_existing_task_graph_boundaries_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    task_graph_status: str,
    expected: AuxiliaryTaskDeliveryStatus,
    reason: str,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    revision = int(window["state_version"])
    task_graph_result = TaskGraphWorkRunResult(
        status=task_graph_status,
        task_id=task_id,
        final_delivery_id=(
            "missing-candidate-delivery"
            if task_graph_status == "completed"
            else None
        ),
        window_state_version=revision,
        last_work_run_outcome=(
            reason if task_graph_status != "failed_closed" else None
        ),
        failure_code=(
            reason if task_graph_status == "failed_closed" else None
        ),
    )
    monkeypatch.setattr(
        delivery,
        "run_task_graph_work_runs",
        lambda *_args, **_kwargs: task_graph_result,
    )

    result = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
    )

    assert result.status is expected
    assert result.reason_code == reason
    assert result.final_delivery_id is None
    assert result.publication_body is None
