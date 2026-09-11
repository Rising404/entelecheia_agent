from __future__ import annotations

from collections.abc import Callable
import hashlib
import json

import pytest

from personagraph.l2.task_graph import TaskDeliveryValidationDimension
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.task_execution.verification.controller import (
    SqliteNodeVerificationApplicationStore,
    run_node_verification,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryStatus,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunProfile,
)
from personagraph.l2.task_execution.delivery.candidate_gate import (
    TaskDeliveryCandidateAuthority,
    build_task_delivery_candidate_gate_factory,
    create_task_delivery_candidate_model_call_authority,
)
from personagraph.l2.task_execution.delivery.model_contracts import PURPOSE
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_facts,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.persistence.l2.work_run.work_execution import (
    WorkExecutionPersistenceError,
)
from personagraph.l2.work_run import DownstreamVerificationDisposition
from tests.runtime.test_node_verification_controller_sqlite import (
    _advancing_clock,
    _application_request,
    _model_result,
    _reply,
    _seed_submitted_node,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _settled_positive_base,
)


def _candidate_payload(
    *,
    route: str,
    root_node_id: str,
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for dimension in TaskDeliveryValidationDimension:
        is_target = dimension is TaskDeliveryValidationDimension.GOAL_COMPLETENESS
        if route == "pass" or not is_target:
            verdict = "pass"
            fault_domain = "none"
            affected_node_ids: list[str] = []
        elif route == "retry_execution":
            verdict = "fail"
            fault_domain = "execution_output"
            affected_node_ids = [root_node_id]
        elif route == "replan_task_graph":
            verdict = "fail"
            fault_domain = "task_graph_design"
            affected_node_ids = [root_node_id]
        else:
            verdict = "insufficient_evidence"
            fault_domain = "missing_information"
            affected_node_ids = []
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
        "execution_repair_objective": (
            "保留正确内容并补齐根交付的必要步骤。"
            if route == "retry_execution"
            else None
        ),
        "task_graph_revision_objective": (
            "重建遗漏必要子任务的任务图。"
            if route == "replan_task_graph"
            else None
        ),
        "blocking_questions": (
            ["请提供缺失的目标范围信息。"] if route == "blocked" else []
        ),
    }


def _candidate_model_call_authority_factory(
    authority: TaskDeliveryCandidateAuthority,
    *,
    rederive_state_guard_sha256: Callable[[], str],
):
    return create_task_delivery_candidate_model_call_authority(
        authority,
        rederive_state_guard_sha256=rederive_state_guard_sha256,
        ledger_store=store,
    )


def test_sqlite_root_candidate_retry_is_durable_and_stays_in_same_work_run():
    seeded = _seed_submitted_node(suffix="candidate-v2-retry")
    request = _application_request(seeded, suffix="candidate-v2-retry")
    candidate_calls: list[tuple[str, str]] = []
    configured_provider, configured_model = configured_structured_model_facts()

    def node_provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    def candidate_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        candidate_calls.append((model_call_id, purpose))
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(
                    route="retry_execution",
                    root_node_id=seeded.task_id,
                ),
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=3,
            model_call_id=model_call_id,
        )

    factory = build_task_delivery_candidate_gate_factory(
        provider=as_prepared_test_provider(candidate_provider),
        emit=lambda _event: None,
        model_call_authority_factory=(
            _candidate_model_call_authority_factory
        ),
    )
    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(node_provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
        downstream_gate=factory(request),
    )

    assert result.outcome == "not_passed"
    assert result.next_attempt_mutation is not None
    assert candidate_calls[0][1] == PURPOSE
    loaded = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert loaded.current_attempt_id == request.next_attempt.attempt_id
    assert len(loaded.attempts) == 2
    feedback = loaded.attempts[-1].input_verification_result
    assert feedback is not None
    assert feedback.downstream_results[0].disposition is (
        DownstreamVerificationDisposition.RETRY_ATTEMPT
    )
    assert feedback.downstream_results[0].affected_subject_ids == (
        seeded.task_id,
    )
    with store._connect() as conn:
        logical = conn.execute(
            "SELECT call_kind, purpose, typed_result_contract "
            "FROM insession_runtime_model_logical_calls WHERE session_id=?",
            (seeded.session_id,),
        ).fetchall()
        delivery_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE work_run_id=?",
                (seeded.work_run_id,),
            ).fetchone()[0]
        )
        frozen_at = conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows "
            "WHERE work_run_id=?",
            (seeded.work_run_id,),
        ).fetchone()[0]
    assert [tuple(row) for row in logical] == [
        (
            "task_delivery_candidate_validation",
            PURPOSE,
            "task-delivery-validation-model-envelope-v2",
        )
    ]
    assert delivery_count == 0
    assert frozen_at is None


def test_node_verifier_nonpass_never_dispatches_whole_task_candidate_review():
    seeded = _seed_submitted_node(suffix="candidate-v2-node-nonpass")
    request = _application_request(seeded, suffix="candidate-v2-node-nonpass")
    candidate_calls = 0

    def candidate_must_not_run(*_args, **_kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        raise AssertionError("node non-pass must not reach the downstream gate")

    factory = build_task_delivery_candidate_gate_factory(
        provider=as_prepared_test_provider(candidate_must_not_run),
        emit=lambda _event: None,
        model_call_authority_factory=(
            _candidate_model_call_authority_factory
        ),
    )
    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(first_verdict="not_satisfied"),
                str(kwargs["model_call_id"]),
            )
        ),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
        downstream_gate=factory(request),
    )

    assert result.outcome == "not_passed"
    assert candidate_calls == 0


def test_sqlite_unfinished_child_suppresses_candidate_review_and_fails_stale_commit():
    seeded = _seed_submitted_node(suffix="candidate-v2-unfinished-child")
    request = _application_request(seeded, suffix="candidate-v2-unfinished-child")
    details = task_graph_store.get_insession_task_details(seeded.session_id, seeded.task_id)
    assert details is not None
    root = details.nodes[0]
    child_id = "child-candidate-v2-unfinished"

    candidate_calls = 0

    def candidate_must_not_run(*_args, **_kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        raise AssertionError("unfinished child must suppress whole-Task review")

    factory = build_task_delivery_candidate_gate_factory(
        provider=as_prepared_test_provider(candidate_must_not_run),
        emit=lambda _event: None,
        model_call_authority_factory=(
            _candidate_model_call_authority_factory
        ),
    )

    def node_provider_after_late_unfinished_child(
        _system: str,
        _user: str,
        **kwargs: object,
    ) -> ModelResult:
    # 此时请求已经准备好。在下游门禁重新投影整项任务权威前加入未完成依赖，
    # 以模拟过期/无效的调度器观察。门禁不得基于变化后的图状态分发审查器。
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO insession_task_graph_nodes "
                "(insession_task_id, graph_revision, insession_task_node_id, "
                "node_revision, node_kind, ordinal, title, objective, "
                "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
                "created_at) VALUES (?, 1, ?, 1, 'subtask', 1, '待完成子节点', "
                "'先完成子节点', ?, ?, '[]', ?)",
                (
                    seeded.task_id,
                    child_id,
                    json.dumps(root["source_anchor_ids"], ensure_ascii=False),
                    json.dumps(root["acceptance_criteria"], ensure_ascii=False),
                    "2026-08-24T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO insession_task_graph_edges "
                "(insession_task_id, graph_revision, "
                "parent_insession_task_node_id, child_insession_task_node_id, "
                "ordinal) VALUES (?, 1, ?, ?, 0)",
                (seeded.task_id, seeded.task_id, child_id),
            )
            conn.execute(
                "INSERT INTO insession_task_node_states "
                "(insession_task_id, insession_task_node_id, node_revision, "
                "status, state_version, updated_at) "
                "VALUES (?, ?, 1, 'proposed', 1, ?)",
                (
                    seeded.task_id,
                    child_id,
                    "2026-08-24T00:00:00+00:00",
                ),
            )
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    with pytest.raises(WorkExecutionPersistenceError):
        run_node_verification(
            request,
            store=SqliteNodeVerificationApplicationStore(),
            provider=as_prepared_test_provider(
                node_provider_after_late_unfinished_child
            ),
            emit=lambda _event: None,
            monotonic_clock=_advancing_clock(),
            downstream_gate=factory(request),
        )
    assert candidate_calls == 0


def _commit_candidate_test_graph(suffix: str) -> tuple[str, str, str]:
    command, _base_root_id = _settled_positive_base(suffix)
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert committed.committed_graph_revision == 2
    return command.session_id, command.source_turn_id, command.task_id


def test_composition_retries_root_candidate_in_same_work_run_then_delivers():
    session_id, turn_id, task_id = _commit_candidate_test_graph(
        "candidate-v2-composition-retry"
    )
    configured_provider, configured_model = configured_structured_model_facts()
    default_attempt = build_attempt_structured_provider(
        TaskGraphWorkRunProfile().structured_model_profile()
    )
    attempt_prompts: list[dict[str, object]] = []
    candidate_prompts: list[dict[str, object]] = []
    candidate_system_prompts: list[str] = []
    candidate_routes = ["retry_execution", "pass"]

    def attempt_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        attempt_prompts.append(json.loads(user_content))
        return default_attempt(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    def candidate_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        candidate_system_prompts.append(system_prompt)
        candidate_prompts.append(json.loads(user_content))
        route = candidate_routes.pop(0)
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(route=route, root_node_id=task_id),
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=2,
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
            attempt_provider=as_prepared_test_provider(attempt_provider),
            task_candidate_validation_provider=as_prepared_test_provider(
                candidate_provider
            ),
        ),
    )

    assert result.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert result.final_delivery_id is not None
    assert candidate_routes == []
    assert len(candidate_prompts) == 2
    first_attestations = candidate_prompts[0]["node_verification_attestations"]
    final_attestations = candidate_prompts[1]["node_verification_attestations"]
    assert len(first_attestations) == len(final_attestations) == 2
    attestation_by_node = {item["node_id"]: item for item in final_attestations}
    assert set(attestation_by_node) == {
        item["insession_task_node_id"]
        for item in task_graph_store.get_insession_task_details(session_id, task_id).nodes
    }
    root_attestation = attestation_by_node[task_id]
    assert root_attestation["verification_source"] == "root_candidate"
    child_attestation = next(
        item for node_id, item in attestation_by_node.items() if node_id != task_id
    )
    assert child_attestation["verification_source"] == "direct_delivery"
    first_child_attestation = next(
        item for item in first_attestations if item["node_id"] != task_id
    )
    assert first_child_attestation == child_attestation
    first_root_attestation = next(
        item for item in first_attestations if item["node_id"] == task_id
    )
    assert first_root_attestation["verification_request_id"] != (
        root_attestation["verification_request_id"]
    )
    for attestation in (*first_attestations, *final_attestations):
        assert attestation["schema_version"] == (
            "task-delivery-validation-node-verification-attestation-v1"
        )
        assert attestation["all_pass"] is True
        assert attestation["acceptance_ids"]
        assert "output_body" not in attestation
        assert "tool_result_bodies" not in attestation
    assert all(
        "node_verification_attestations" in system_prompt
        and "missing_information" in system_prompt
        for system_prompt in candidate_system_prompts
    )
    child_projection = candidate_prompts[0]["child_delivery_projection"]
    assert child_projection["schema_version"] == (
        "task-delivery-validation-child-delivery-projection-v1"
    )
    assert child_projection == candidate_prompts[1]["child_delivery_projection"]
    child_deliveries = child_projection["deliveries"]
    assert len(child_deliveries) == 1
    child = child_deliveries[0]
    assert child["node_id"] != task_id
    assert child["output_body"]
    assert child["output_sha256"] == hashlib.sha256(
        child["output_body"].encode("utf-8")
    ).hexdigest()
    assert len(child["binding_sha256"]) == 64
    assert len(result.work_run_ids) == 2
    loaded_runs = tuple(
        work_run_store.get_work_run(session_id=session_id, work_run_id=work_run_id)
        for work_run_id in result.work_run_ids
    )
    root_run = next(
        item
        for item in loaded_runs
        if item.work_run.subject.node_id == task_id
    )
    assert len(root_run.attempts) == 2
    root_prompts = tuple(
        item
        for item in attempt_prompts
        if item["node"]["subject"]["node_id"] == task_id
    )
    assert len(root_prompts) == 2
    downstream = root_prompts[1]["verification_feedback"][
        "downstream_results"
    ]
    assert downstream[0]["gate_id"] == "whole_task_delivery_v2"
    assert downstream[0]["disposition"] == "retry_attempt"
    with store._connect() as conn:
        delivery_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE work_run_id=?",
                (root_run.work_run.work_run_id,),
            ).fetchone()[0]
        )
        candidate_logical_calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind="
                "'task_delivery_candidate_validation'",
                (session_id,),
            ).fetchone()[0]
        )
    assert delivery_count == 1
    assert candidate_logical_calls == 2


@pytest.mark.parametrize(
    ("route", "expected_status", "expected_reason"),
    [
        (
            "replan_task_graph",
            AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
            "whole_task_candidate_replan_task_graph",
        ),
        (
            "blocked",
            AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
            "whole_task_candidate_blocked_replan",
        ),
    ],
)
def test_composition_persists_candidate_route_without_publication(
    route: str,
    expected_status: AuxiliaryTaskDeliveryStatus,
    expected_reason: str,
):
    session_id, turn_id, task_id = _commit_candidate_test_graph(
        f"candidate-v2-composition-{route}"
    )
    configured_provider, configured_model = configured_structured_model_facts()

    def candidate_provider(
        _system: str,
        _user: str,
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
            latency_ms=2,
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

    assert result.status is expected_status
    assert result.reason_code == expected_reason
    assert result.final_delivery_id is None
    assert result.publication_body is None
    if route == "blocked":
        assert result.requested_user_questions == ()
    with store._connect() as conn:
        root_delivery_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=? "
                "AND insession_task_node_id=?",
                (session_id, task_id, task_id),
            ).fetchone()[0]
        )
    assert root_delivery_count == 1


def test_composition_settles_replan_with_frozen_root_and_exact_trigger():
    session_id, turn_id, task_id = _commit_candidate_test_graph(
        "candidate-v2-composition-replan-settlement"
    )
    configured_provider, configured_model = configured_structured_model_facts()
    candidate_calls = 0

    def candidate_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal candidate_calls
        candidate_calls += 1
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(
                    route="replan_task_graph",
                    root_node_id=task_id,
                ),
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=2,
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
            task_candidate_validation_provider=as_prepared_test_provider(
                candidate_provider
            ),
        ),
    )

    assert first.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert first.reason_code == "whole_task_candidate_replan_task_graph"
    assert first.final_delivery_id is None
    assert first.publication_body is None
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.status.value == "active"
    assert task.current_graph_revision == 2
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    assert trigger.base_graph_revision == 2
    assert trigger.target_graph_revision == 3
    assert trigger.revision_objective == "重建遗漏必要子任务的任务图。"
    with store._connect() as conn:
        root_deliveries = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT delivery_id FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=? "
                "AND insession_task_node_id=?",
                (session_id, task_id, task_id),
            ).fetchall()
        )
        candidate_logical_calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind="
                "'task_delivery_candidate_validation'",
                (session_id,),
            ).fetchone()[0]
        )
    assert root_deliveries == (trigger.root_delivery_id,)
    assert candidate_logical_calls == 1
    assert candidate_calls == 1

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("settled V2 candidate replay reached a Provider")

    replayed = run_auxiliary_committed_task_to_delivery(
        request,
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            attempt_provider=as_prepared_test_provider(model_must_not_run),
            verification_provider=as_prepared_test_provider(model_must_not_run),
            task_candidate_validation_provider=as_prepared_test_provider(
                model_must_not_run
            ),
        ),
    )
    assert replayed.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert replayed.reason_code == "whole_task_candidate_replan_task_graph_replayed"
    assert replayed.final_delivery_id is None
    assert candidate_calls == 1


@pytest.mark.parametrize("route", ["replan_task_graph", "blocked"])
def test_root_candidate_routes_freeze_delivery_and_settle_atomically(route: str):
    seeded = _seed_submitted_node(suffix=f"candidate-v2-{route}")
    request = _application_request(seeded, suffix=f"candidate-v2-{route}")
    configured_provider, configured_model = configured_structured_model_facts()

    def candidate_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(route=route, root_node_id=seeded.task_id),
                ensure_ascii=False,
            ),
            provider=configured_provider,
            model=configured_model,
            latency_ms=3,
            model_call_id=model_call_id,
        )

    factory = build_task_delivery_candidate_gate_factory(
        provider=as_prepared_test_provider(candidate_provider),
        emit=lambda _event: None,
        model_call_authority_factory=(
            _candidate_model_call_authority_factory
        ),
    )
    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(), str(kwargs["model_call_id"])
            )
        ),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
        downstream_gate=factory(request),
    )

    assert result.outcome == "passed"
    assert result.store_projection.delivery_id == request.delivery_id
    with store._connect() as conn:
        delivery_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE work_run_id=?",
                (seeded.work_run_id,),
            ).fetchone()[0]
        )
        frozen_at = conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows "
            "WHERE work_run_id=?",
            (seeded.work_run_id,),
        ).fetchone()[0]
    assert delivery_count == 1
    assert frozen_at is not None
    task = task_graph_store.get_insession_task_details(seeded.session_id, seeded.task_id)
    assert task is not None
    assert task.status.value == "active"
    settled = task_delivery_store.get_task_delivery_candidate_settlement(
        session_id=seeded.session_id,
        task_id=seeded.task_id,
        graph_revision=1,
    )
    assert settled is not None
    assert settled.settlement.disposition.value == route
    assert settled.trigger is not None
    assert settled.trigger.base_graph_revision == 1
    assert settled.trigger.target_graph_revision == 2
