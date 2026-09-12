from __future__ import annotations

import json

import pytest

from tests.helpers.prepared_model_provider import as_prepared_test_provider


from personagraph.l2.task_graph import (
    TaskDeliveryValidationDimension,
    TaskDeliveryValidationFaultDomain,
    TaskDeliveryValidationFinding,
    TaskDeliveryValidationVerdict,
    TaskGraphRevisionTrigger,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.positive_controller import (
    AuxiliaryPositivePlanningError,
    AuxiliaryPositivePlanningRequest,
    AuxiliaryPositivePlanningStatus,
    run_positive_base_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryStatus,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
)
from tests.runtime.test_auxiliary_task_delivery_composition import (
    _commit_task_graph,
)
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _commit_command,
)


def _seed_positive_base() -> tuple[str, str, str, object]:
    command = _commit_command("positive-architect-entry")
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    task = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert task is not None
    assert task.current_graph_revision == 1
    assert task.status.value == "active"
    assert committed.committed_graph_revision == 1
    return command.session_id, command.source_turn_id, command.task_id, task


def _trigger(
    session_id: str,
    turn_id: str,
    task_id: str,
    *,
    task_state_version: int,
    trigger_id: str = "taskgraph-positive-trigger",
    base_revision: int = 1,
) -> TaskGraphRevisionTrigger:
    return TaskGraphRevisionTrigger.create(
        trigger_id=trigger_id,
        create_apply_id=f"{trigger_id}-create",
        session_id=session_id,
        task_id=task_id,
        base_graph_revision=base_revision,
        target_graph_revision=base_revision + 1,
        root_delivery_id="positive-trigger-root-delivery",
        verification_request_id="positive-trigger-verification-request",
        request_binding_sha256="1" * 64,
        verification_result_id="positive-trigger-verification-result",
        result_sha256="2" * 64,
        settlement_id="positive-trigger-settlement",
        settlement_sha256="3" * 64,
        revision_objective=(
            "Preserve the valid base graph and repair the omitted release step."
        ),
        gap_diagnosis=(
            TaskDeliveryValidationFinding(
                dimension=(
                    TaskDeliveryValidationDimension.GOAL_COMPLETENESS
                ),
                verdict=TaskDeliveryValidationVerdict.FAIL,
                fault_domain=TaskDeliveryValidationFaultDomain.TASK_GRAPH_DESIGN,
                finding="The final delivery omitted the required release step.",
                affected_node_ids=(task_id,),
                evidence_anchor_ids=(),
            ),
        ),
        reopened_task_state_version=task_state_version,
        created_turn_id=turn_id,
    )


def _settle_turn(*, session_id: str, turn_id: str) -> None:
    window = store.get_turn_execution_window(session_id)
    assert window is not None and window["turn_id"] == turn_id
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(window["state_version"]),
        stage="RESPONSE",
        interruption_reason="TEST_CONTINUATION",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="host_stopped",
        error_code="TEST_CONTINUATION",
    )


def _accept_continuation_turn(
    *,
    session_id: str,
    prior_turn_id: str,
    task_id: str,
    execute_current: bool,
) -> str:
    _settle_turn(session_id=session_id, turn_id=prior_turn_id)
    user_text = "继续执行已有任务"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=(
            "positive-planning-execute-lane"
            if execute_current
            else "positive-planning-readonly-lane"
        ),
        source="runtime_test",
        user_text=user_text,
        lease_owner="positive-planning-invocation-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    current_window = store.get_turn_execution_window(session_id)
    assert current_window is not None and current_window["turn_id"] == turn_id
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id=f"positive-planning-lane-{turn_id}",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": user_text,
                        "execute_current": execute_current,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=int(current_window["state_version"]),
    )
    return turn_id


def _revision_required_task_candidate_validator(
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
                    "The final delivery omitted the required release step."
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


def test_delivery_revise_trigger_drives_application_through_taskgraph_revision_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    validation = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            task_candidate_validation_provider=(
                as_prepared_test_provider(_revision_required_task_candidate_validator)
            ),
        ),
    )
    assert validation.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert validation.final_delivery_id is None
    assert validation.publication_body is None
    assert validation.task_candidate_settlement is not None
    trigger = validation.task_candidate_settlement.trigger
    assert trigger is not None
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) == trigger

    physical = build_auxiliary_architect_structured_provider()
    default_attempt = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=8_192,
            verification_max_output_tokens=4_096,
            timeout_s=60.0,
        )
    )
    architect_calls = 0

    def architect_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal architect_calls
        architect_calls += 1
        payload = json.loads(str(args[1]))
        assert payload["goal"]["objective"] == trigger.revision_objective
        assert payload["task_graph_revision_trigger"] == trigger.model_dump(
            mode="json"
        )
        assert payload["task_graph_semantic_base"][
            "base_task_graph_revision"
        ] == 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    def attempt_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        payload = json.loads(user_content)
        if payload.get("task_graph_revision_base") is None:
            return default_attempt(
                system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
            )
        base = payload["task_graph_revision_base"]
        base_root = next(
            node
            for node in base["nodes"]
            if node["node_alias"] == base["root_node_alias"]
        )
        reply = {
            "acceptance_updates": [
                {
                    "acceptance_id": acceptance["acceptance_id"],
                    "model_claimed_satisfied": True,
                }
                for acceptance in payload["node"]["acceptances"]
            ],
            "action": {
                "kind": "submit_task_graph",
                "proposal": {
                    "schema_version": "insession-task-graph-revision-v2",
                    "root": {
                        "root_key": "root",
                        "nodes": [
                            {
                                "node_key": "root",
                                "node_kind": "root",
                                "parent_node_key": None,
                                "title": base_root["title"],
                                "objective": (
                                    f'{base_root["objective"]} Include the '
                                    "verified release step."
                                ),
                                "source_anchor_ids": base_root[
                                    "source_anchor_aliases"
                                ],
                                "acceptance_criteria": base_root[
                                    "acceptance_criteria"
                                ],
                                "constraints": base_root["constraints"],
                            },
                            {
                                "node_key": "release",
                                "node_kind": "subtask",
                                "parent_node_key": "root",
                                "title": "Release",
                                "objective": (
                                    "Complete the omitted release step."
                                ),
                                "source_anchor_ids": [
                                    "task_creation_source"
                                ],
                                "acceptance_criteria": [
                                    {
                                        "acceptance_id": "release_ready",
                                        "criterion": (
                                            "The release result is complete "
                                            "and verifiable."
                                        ),
                                        "source_anchor_ids": [
                                            "task_creation_source"
                                        ],
                                    }
                                ],
                                "constraints": [],
                            },
                        ],
                    },
                },
                "lineage": [
                        {
                            "proposal_node_key": "root",
                            "disposition": "revise",
                            "base_node_alias": base["root_node_alias"],
                        },
                    {
                        "proposal_node_key": "release",
                        "disposition": "new",
                        "base_node_alias": None,
                    },
                ],
            },
        }
        return ModelResult(
            reply=json.dumps(reply),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    completed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=16,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(architect_provider),
            attempt_provider=as_prepared_test_provider(attempt_provider),
        ),
    )

    assert completed.status is AuxiliaryApplicationStatus.COMMITTED
    assert completed.planning_result is not None
    assert (
        completed.planning_result.status
        is AuxiliaryPositivePlanningStatus.PLANNED
    )
    assert architect_calls == 1
    revised = task_graph_store.get_insession_task_details(session_id, task_id)
    assert revised is not None
    assert revised.current_graph_revision == 2
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None


def test_positive_planner_model_sees_trigger_objective_and_exact_base_then_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    trigger = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
    )
    def active_trigger(*, session_id: str, task_id: str):
        assert session_id == trigger.session_id
        assert task_id == trigger.task_id
        return trigger

    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        active_trigger,
        raising=False,
    )
    physical = build_auxiliary_architect_structured_provider()
    calls: list[dict[str, object]] = []

    def provider(*args: object, **kwargs: object) -> ModelResult:
        payload = json.loads(str(args[1]))
        calls.append(payload)
        assert payload["goal"]["objective"] == trigger.revision_objective
        assert payload["task_graph_revision_trigger"] == trigger.model_dump(
            mode="json"
        )
        semantic_base = payload["task_graph_semantic_base"]
        assert semantic_base["base_task_graph_revision"] == 1
        assert semantic_base["root_node_alias"] == "base_node_000"
        assert semantic_base["nodes"][0]["title"] == task.nodes[0]["title"]
        result = physical(*args, **kwargs)  # type: ignore[arg-type]
        proposal = json.loads(result.reply)
        assert proposal["revision_reason"] == "verification_failed"
        return result

    request = AuxiliaryPositivePlanningRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        trigger=trigger,
    )
    planned = run_positive_base_auxiliary_planning(
        request,
        ledger_store=store,
        emit=lambda _event: None,
        provider=as_prepared_test_provider(provider),
    )

    assert planned.status is AuxiliaryPositivePlanningStatus.PLANNED
    assert planned.bootstrapped is True
    assert planned.bootstrap_commit is not None
    assert planned.revision_commit is not None
    assert planned.details.base_task_graph_revision == 1
    assert planned.details.target_task_graph_revision == 2
    assert planned.details.goal_objective == trigger.revision_objective
    assert planned.details.reason == "verification_failed"
    assert [item.local_node_key for item in planned.details.nodes] == [
        "analyze_verified_context",
        "synthesize_task_graph",
    ]

    # 在相同活跃触发器和正向目标仍具权威时，主机可以推进外壳元数据。恢复必须
    # 依据这些精确权威，而不能永久固定引导前的任务状态版本。
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_tasks SET state_version=state_version+1 "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        )
    replayed = run_positive_base_auxiliary_planning(
        request,
        ledger_store=store,
        emit=lambda _event: None,
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "completed positive planning reached Provider"
            )
        ),
    )
    assert (
        replayed.status
        is AuxiliaryPositivePlanningStatus.ALREADY_PLANNED
    )
    assert replayed.details == planned.details
    assert replayed.model_replayed is True
    assert replayed.revision_commit is not None
    assert replayed.revision_commit.status == "replayed"
    assert len(calls) == 1

    # 进程可能在终态提案封印后、TaskGraph 提交前退出。正向规划重新进入时，
    # 必须从这个提交就绪的后代恢复已稳定的架构师计划，而不是再次尝试引导或
    # 分发架构师。
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_graph_goals SET status='proposal_ready', "
            "state_version=state_version+1 WHERE goal_id=?",
            (planned.details.goal_id,),
        )
        conn.execute(
            "UPDATE insession_auxiliary_graph_revision_states_v2 "
            "SET status='proposal_ready', state_version=state_version+1 "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
            (
                planned.details.auxiliary_graph_id,
                planned.details.auxiliary_graph_revision,
            ),
        )
        conn.execute(
            "UPDATE insession_tasks SET state_version=state_version+1 "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        )
    commit_ready_replay = run_positive_base_auxiliary_planning(
        request,
        ledger_store=store,
        emit=lambda _event: None,
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "commit-ready positive plan reached Provider"
            )
        ),
    )
    assert (
        commit_ready_replay.status
        is AuxiliaryPositivePlanningStatus.ALREADY_PLANNED
    )
    assert commit_ready_replay.details.goal_status == "proposal_ready"
    assert commit_ready_replay.details.revision_status == "proposal_ready"
    assert commit_ready_replay.model_replayed is True
    assert len(calls) == 1


def test_application_prioritizes_active_taskgraph_trigger_over_initial_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    trigger = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
        trigger_id="taskgraph-application-positive-trigger",
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: trigger,
        raising=False,
    )
    planning_calls = 0
    physical = build_auxiliary_architect_structured_provider()

    def provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal planning_calls
        planning_calls += 1
        payload = json.loads(str(args[1]))
        assert payload["task_graph_revision_trigger"]["trigger_id"] == (
            trigger.trigger_id
        )
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(provider),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert result.effect_steps == 1
    assert result.planning_result is not None
    assert (
        result.planning_result.status
        is AuxiliaryPositivePlanningStatus.PLANNED
    )
    assert result.planning_result.details.base_task_graph_revision == 1
    assert result.planning_result.details.target_task_graph_revision == 2
    assert planning_calls == 1


def test_positive_plan_committed_before_turn_handoff_replays_then_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已提交的正向计划保留其来源，而执行会获得新租约。"""

    session_id, source_turn_id, task_id = _commit_task_graph(monkeypatch)
    validation = run_auxiliary_committed_task_to_delivery(
        AuxiliaryTaskDeliveryRequest(
            session_id=session_id,
            turn_id=source_turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryTaskDeliveryPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            task_candidate_validation_provider=(
                as_prepared_test_provider(_revision_required_task_candidate_validator)
            ),
        ),
    )
    assert validation.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    assert validation.task_candidate_settlement is not None
    trigger = validation.task_candidate_settlement.trigger
    assert trigger is not None

    architect_provider = build_auxiliary_architect_structured_provider()
    architect_calls = 0

    def count_architect(*args: object, **kwargs: object) -> ModelResult:
        nonlocal architect_calls
        architect_calls += 1
        return architect_provider(*args, **kwargs)  # type: ignore[arg-type]

    planned = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=source_turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(count_architect),
        ),
    )
    assert planned.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert planned.planning_result is not None
    assert planned.planning_result.status is AuxiliaryPositivePlanningStatus.PLANNED
    assert architect_calls == 1
    planned_details = planned.planning_result.details
    assert all(node.status == "proposed" for node in planned_details.nodes)

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=True,
    )
    profile = WorkRunStructuredModelProfile(
        attempt_max_output_tokens=8_192,
        verification_max_output_tokens=4_096,
        timeout_s=60.0,
    )
    resumed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=continuation_turn_id,
            task_id=task_id,
            max_effect_steps=1,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "committed positive Architect plan reached Provider after handoff"
                )
            ),
            attempt_provider=as_prepared_test_provider(build_attempt_structured_provider(profile)),
            verification_provider=as_prepared_test_provider(build_verification_structured_provider(profile)),
        ),
    )

    assert resumed.status is AuxiliaryApplicationStatus.STEP_LIMIT_REACHED
    assert resumed.effect_steps == 1
    assert resumed.planning_result is not None
    assert (
        resumed.planning_result.status
        is AuxiliaryPositivePlanningStatus.ALREADY_PLANNED
    )
    assert resumed.planning_result.model_replayed is True
    assert architect_calls == 1
    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert any(node.status != "proposed" for node in after.nodes)


def test_positive_planner_replays_settled_model_across_turn_after_precommit_response_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    trigger = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
        trigger_id="taskgraph-positive-response-loss-trigger",
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: trigger,
        raising=False,
    )


    physical = build_auxiliary_architect_structured_provider()
    provider_calls = 0

    def provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision

    def lose_before_architect_commit(**kwargs: object):
        if str(kwargs["apply_id"]).endswith(":architect"):
            raise RuntimeError("simulated response loss before Architect commit")
        return real_commit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        lose_before_architect_commit,
    )
    request = AuxiliaryPositivePlanningRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        trigger=trigger,
    )
    with pytest.raises(RuntimeError, match="simulated response loss"):
        run_positive_base_auxiliary_planning(
            request,
            ledger_store=store,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(provider),
        )
    bootstrap = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert bootstrap is not None
    assert [node.local_node_key for node in bootstrap.nodes] == [
        "positive_base_bootstrap"
    ]
    assert provider_calls == 1

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        real_commit,
    )
    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=turn_id,
        task_id=task_id,
        execute_current=True,
    )
    recovered = run_positive_base_auxiliary_planning(
        AuxiliaryPositivePlanningRequest(
            session_id=session_id,
            turn_id=continuation_turn_id,
            task_id=task_id,
            trigger=trigger,
        ),
        ledger_store=store,
        emit=lambda _event: None,
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "settled Architect model call reached Provider again"
            )
        ),
    )

    assert recovered.status is AuxiliaryPositivePlanningStatus.PLANNED
    assert recovered.model_replayed is True
    assert recovered.bootstrapped is False
    assert recovered.revision_commit is not None
    assert recovered.revision_commit.status == "applied"
    assert provider_calls == 1


def test_positive_planner_rejects_wrong_base_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    wrong = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
        trigger_id="taskgraph-wrong-base-trigger",
        base_revision=2,
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: wrong,
        raising=False,
    )

    with pytest.raises(
        AuxiliaryPositivePlanningError,
        match="verifier-reopened trigger authority",
    ):
        run_positive_base_auxiliary_planning(
            AuxiliaryPositivePlanningRequest(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                trigger=wrong,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "wrong base reached Provider"
                )
            ),
        )


def test_positive_planner_rejects_self_consistent_but_inactive_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    active = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
    )
    forged = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
        trigger_id="taskgraph-forged-trigger",
    )
    assert forged.trigger_sha256 != active.trigger_sha256
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: active,
        raising=False,
    )

    with pytest.raises(
        AuxiliaryPositivePlanningError,
        match="exact active Store authority",
    ):
        run_positive_base_auxiliary_planning(
            AuxiliaryPositivePlanningRequest(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                trigger=forged,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "forged trigger reached Provider"
                )
            ),
        )


def test_positive_planner_rejects_non_running_invocation_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    trigger = _trigger(
        session_id,
        turn_id,
        task_id,
        task_state_version=task.task_state_version,
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: trigger,
        raising=False,
    )
    _settle_turn(session_id=session_id, turn_id=turn_id)

    with pytest.raises(
        AuxiliaryPositivePlanningError,
        match="no active Runtime Turn",
    ):
        run_positive_base_auxiliary_planning(
            AuxiliaryPositivePlanningRequest(
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                trigger=trigger,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "non-running invocation reached Provider"
                )
            ),
        )


def test_positive_planner_rejects_durable_lane_without_execution_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, trigger_turn_id, task_id, task = _seed_positive_base()
    trigger = _trigger(
        session_id,
        trigger_turn_id,
        task_id,
        task_state_version=task.task_state_version,
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda *, session_id, task_id: trigger,
        raising=False,
    )
    invocation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=trigger_turn_id,
        task_id=task_id,
        execute_current=False,
    )
    assert invocation_turn_id != trigger.created_turn_id

    with pytest.raises(
        AuxiliaryPositivePlanningError,
        match="did not request this Task lane",
    ):
        run_positive_base_auxiliary_planning(
            AuxiliaryPositivePlanningRequest(
                session_id=session_id,
                turn_id=invocation_turn_id,
                task_id=task_id,
                trigger=trigger,
            ),
            ledger_store=store,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail(
                    "non-executing durable lane reached Provider"
                )
            ),
        )
