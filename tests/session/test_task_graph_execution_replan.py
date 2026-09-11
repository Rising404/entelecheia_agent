from __future__ import annotations

import hashlib
import json
import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionReason,
    TaskGraphSemanticLineageProjection,
)
from personagraph.l2.task_graph import InSessionTaskGraphRevisionProposal
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunRequest,
    run_task_graph_work_runs,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.planning.positive_controller import (
    AuxiliaryPositivePlanningStatus,
)
from personagraph.l2.task_execution.work_run.turn_controller import (
    WorkRunTurnStableIdPlan,
    run_new_task_node_work_run,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.work_run.contracts import (
    HostAcceptedAttemptDecision,
    RequestTaskGraphRevisionAction,
    TaskGraphExecutionReplanApplication,
    TaskGraphExecutionReplanReason,
    TaskGraphExecutionReplanRequest,
    TaskNodeSubject,
)
from tests.runtime.test_work_run_turn_controller_sqlite import (
    _advancing_clock,
    _model_result,
    _request,
    _seed_ready_node,
)
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
)
from tests.runtime.test_auxiliary_positive_planning_controller import (
    _seed_positive_base,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _revision_proposal,
    _settled_positive_base,
)
from tests.session.test_auxiliary_goal_supersede_persistence import (
    _command as _goal_supersede_command,
    _commit_task_graph_one,
    _seed_goal,
)


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _settle_execution_request(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    node_id: str,
    prefix: str,
) -> tuple[TaskGraphExecutionReplanRequest, object, dict[str, object]]:
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision is not None
    node = next(
        item
        for item in task.nodes
        if item["insession_task_node_id"] == node_id
    )
    subject = TaskNodeSubject(
        task_id=task_id,
        graph_revision=task.current_graph_revision,
        node_id=node_id,
        node_revision=int(node["node_revision"]),
    )
    created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=task.task_state_version,
        expected_node_state_version=int(node["state_version"]),
        expected_window_revision=_window_revision(session_id),
        apply_id=f"{prefix}-create-run",
        work_run_id=f"{prefix}-run",
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id=f"{prefix}-start-attempt",
        input_checkpoint_id=None,
        catalog_snapshot={"revision": 1, "tools": []},
        attempt_id=f"{prefix}-attempt",
    )
    decision = HostAcceptedAttemptDecision(
        acceptance_updates=(),
        action=RequestTaskGraphRevisionAction(
            reason=(
                TaskGraphExecutionReplanReason.TASK_DECOMPOSITION_INCOMPLETE
            ),
            diagnosis=(
                "当前节点把调查、综合和最终交付压在一个节点中，无法可靠执行。"
            ),
            revision_objective=(
                "把调查与综合拆成有依赖关系的节点，并修订当前源节点。"
            ),
            supporting_tool_result_ids=(),
        ),
    )
    settled = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=created.work_run_id,
        attempt_id=started.current_attempt_id,
        decision=decision,
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id=f"{prefix}-request-revision",
        active_seconds_delta=1.0,
    )
    request = work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    )
    assert request is not None
    return request, settled, {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": created.work_run_id,
        "attempt_id": started.current_attempt_id,
        "decision": decision,
        "expected_work_run_revision": started.work_run_revision,
        "expected_progress_revision": started.acceptance_progress_revision,
        "expected_window_revision": started.window_state_version,
        "apply_id": f"{prefix}-request-revision",
        "active_seconds_delta": 1.0,
    }


def _execution_positive_base(
    prefix: str,
    *,
    reuse_requesting_node: bool = False,
    goal_objective: str = (
        "把调查与综合拆成有依赖关系的节点，并修订当前源节点。"
    ),
):
    holder: list[
        tuple[TaskGraphExecutionReplanRequest, dict[str, object]]
    ] = []

    def prepare_base(first_command, _first, base_node_id: str) -> None:
        request, _settled, replay_kwargs = _settle_execution_request(
            session_id=first_command.session_id,
            turn_id=first_command.source_turn_id,
            task_id=first_command.task_id,
            node_id=base_node_id,
            prefix=f"{prefix}-execution",
        )
        holder.append((request, replay_kwargs))

    if reuse_requesting_node:
        root = _revision_proposal().root.nodes[0]
        proposal = InSessionTaskGraphRevisionProposal.model_validate(
            {
                "root": {
                    "root_key": "root",
                    "nodes": [root.model_dump(mode="json")],
                }
            }
        )
        lineage = (
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="reuse",
                base_node_alias="base_node_000",
            ),
        )
    else:
        payload = _revision_proposal().model_dump(mode="json")
        payload["root"]["nodes"][0]["objective"] = (
            "按分离后的调查与综合职责交付第二版执行计划"
        )
        proposal = InSessionTaskGraphRevisionProposal.model_validate(payload)
        lineage = (
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="revise",
                base_node_alias="base_node_000",
            ),
            TaskGraphSemanticLineageProjection(
                proposal_node_key="release",
                disposition="new",
            ),
        )
    command, root_id = _settled_positive_base(
        prefix,
        prepare_base=prepare_base,
        revision_proposal=proposal,
        revision_lineage=lineage,
        goal_objective=goal_objective,
        revision_reason="verification_failed",
    )
    assert len(holder) == 1
    request, replay_kwargs = holder[0]
    return (
        command.model_copy(
            update={
                "task_graph_execution_replan_request_id": request.request_id,
                "expected_task_graph_execution_replan_request_sha256": (
                    request.request_sha256
                ),
            }
        ),
        request,
        root_id,
        replay_kwargs,
    )


def _production_execution_positive_command(
    monkeypatch,
    *,
    prefix: str,
    reuse_requesting_node: bool = False,
):
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    request, _settled, replay_kwargs = _settle_execution_request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        node_id=str(task.nodes[0]["insession_task_node_id"]),
        prefix=f"{prefix}-request",
    )
    physical_architect = build_auxiliary_architect_structured_provider()
    default_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )

    def architect_provider(*args: object, **kwargs: object):
        payload = json.loads(str(args[1]))
        assert payload["task_graph_revision_trigger"] == request.model_dump(
            mode="json"
        )
        assert payload["task_graph_revision_route"] is None
        return physical_architect(*args, **kwargs)  # type: ignore[arg-type]

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
        root_objective = (
            base_root["objective"]
            if reuse_requesting_node
            else (
                f'{base_root["objective"]} Separate investigation and '
                "synthesis responsibilities."
            )
        )
        nodes = [
            {
                "node_key": "root",
                "node_kind": "root",
                "parent_node_key": None,
                "title": base_root["title"],
                "objective": root_objective,
                "source_anchor_ids": base_root["source_anchor_aliases"],
                "acceptance_criteria": base_root["acceptance_criteria"],
                "constraints": base_root["constraints"],
            }
        ]
        lineage = [
            {
                "proposal_node_key": "root",
                "disposition": "reuse" if reuse_requesting_node else "revise",
                "base_node_alias": base["root_node_alias"],
            }
        ]
        if not reuse_requesting_node:
            nodes.append(
                {
                    "node_key": "release",
                    "node_kind": "subtask",
                    "parent_node_key": "root",
                    "title": "Synthesis",
                    "objective": "Synthesize the separately investigated result.",
                    "source_anchor_ids": ["task_creation_source"],
                    "acceptance_criteria": [
                        {
                            "acceptance_id": "synthesis_ready",
                            "criterion": "The synthesis is complete and verifiable.",
                            "source_anchor_ids": ["task_creation_source"],
                        }
                    ],
                    "constraints": [],
                }
            )
            lineage.append(
                {
                    "proposal_node_key": "release",
                    "disposition": "new",
                    "base_node_alias": None,
                }
            )
        return ModelResult(
            reply=json.dumps(
                {
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
                            "schema_version": (
                                "insession-task-graph-revision-v2"
                            ),
                            "root": {"root_key": "root", "nodes": nodes},
                        },
                        "lineage": lineage,
                    },
                }
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    captured: list[object] = []
    real_commit = terminal_store.commit_auxiliary_task_graph_proposal

    def capture_commit(*, command):
        captured.append(command)
        raise terminal_store.AuxiliaryTaskGraphCommitPersistenceError(
            terminal_store.AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "captured before TaskGraph commit",
        )

    monkeypatch.setattr(
        terminal_store,
        "commit_auxiliary_task_graph_proposal",
        capture_commit,
    )
    with pytest.raises(terminal_store.AuxiliaryTaskGraphCommitPersistenceError):
        run_auxiliary_application_to_boundary(
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
    monkeypatch.setattr(
        terminal_store,
        "commit_auxiliary_task_graph_proposal",
        real_commit,
    )
    assert len(captured) == 1
    return captured[0], request, replay_kwargs


def test_typed_action_routes_from_work_run_into_task_graph_revision_boundary() -> None:
    session_id, turn_id, subject = _seed_ready_node(
        suffix="execution-replan-route"
    )

    def attempt_provider(
        _system: str,
        _user: str,
        *,
        model_call_id: str,
        **_kwargs: object,
    ):
        return _model_result(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_task_graph_revision",
                    "reason": "task_decomposition_incomplete",
                    "diagnosis": "单一节点无法可靠完成调查和综合。",
                    "revision_objective": "拆分调查和综合节点后重建任务图。",
                    "supporting_tool_result_ids": [],
                },
            },
            model_call_id,
        )

    work_run_result = run_new_task_node_work_run(
        _request(session_id, turn_id, subject),
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
        allowed_tools=(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "execution replan must not enter node verification"
            )
        ),
        emit=lambda _event: None,
        id_plan=WorkRunTurnStableIdPlan(
            namespace="execution-replan-route"
        ),
        monotonic_clock=_advancing_clock(),
    )

    assert work_run_result.outcome == "task_graph_revision_requested"
    assert work_run_result.work_run_id is not None
    active = work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=subject.task_id,
    )
    assert active is not None
    routed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=subject.task_id,
            expected_window_revision=work_run_result.window_revision,
            initial_work_run_result=work_run_result,
        ),
        monotonic_clock=_advancing_clock(),
    )
    assert routed.status == "revision_required"
    assert routed.work_run_ids == (work_run_result.work_run_id,)
    assert routed.last_work_run_outcome == "task_graph_revision_requested"


def test_default_application_prioritizes_active_execution_replan_request(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    node_id = str(task.nodes[0]["insession_task_node_id"])
    request, _settled, _replay_kwargs = _settle_execution_request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        node_id=node_id,
        prefix="v74-default-application-route",
    )
    physical = build_auxiliary_architect_structured_provider()
    default_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    calls = 0

    def provider(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        payload = json.loads(str(args[1]))
        assert payload["task_graph_revision_trigger"] == request.model_dump(
            mode="json"
        )
        assert payload["task_graph_revision_route"] is None
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
                                    f'{base_root["objective"]} Separate the '
                                    "investigation and synthesis responsibilities."
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
                                "title": "Synthesis",
                                "objective": (
                                    "Synthesize the separately investigated "
                                    "material into a verified result."
                                ),
                                "source_anchor_ids": ["task_creation_source"],
                                "acceptance_criteria": [
                                    {
                                        "acceptance_id": "synthesis_ready",
                                        "criterion": (
                                            "The separated synthesis result is "
                                            "complete and verifiable."
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

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=16,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=as_prepared_test_provider(provider),
            attempt_provider=as_prepared_test_provider(attempt_provider),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.COMMITTED
    assert result.planning_result is not None
    assert (
        result.planning_result.status
        is AuxiliaryPositivePlanningStatus.PLANNED
    )
    assert result.planning_result.trigger == request
    assert result.terminal_result is not None
    assert result.terminal_result.task_graph_commit is not None
    assert (
        result.terminal_result.task_graph_commit.committed_graph_revision
        == request.target_graph_revision
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None and task.current_graph_revision == 2
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    ) is None
    with store._connect() as conn:
        application_row = conn.execute(
            "SELECT task_graph_commit_apply_id FROM "
            "insession_task_graph_execution_replan_applications "
            "WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
    assert application_row is not None
    assert str(application_row["task_graph_commit_apply_id"]) == (
        result.terminal_result.task_graph_commit.apply_id
    )
    assert calls == 1


def test_application_rejects_pending_goal_supersede_with_execution_replan(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id, task = _seed_positive_base()
    _no_mounted_documents(monkeypatch, session_id)
    node_id = str(task.nodes[0]["insession_task_node_id"])
    request, _settled, _replay_kwargs = _settle_execution_request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        node_id=node_id,
        prefix="v74-supersede-authority-conflict",
    )
    monkeypatch.setattr(
        planning_store,
        "get_pending_auxiliary_goal_supersede_receipt",
        lambda **_kwargs: object(),
    )

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_effect_steps=16,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            planning_provider=lambda *_args, **_kwargs: pytest.fail(
                "conflicting successor authority reached the planner"
            ),
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert result.reason_code == "auxiliary_v2_application_rejected"
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    ) == request


def test_goal_supersede_rejects_active_execution_replan_authority() -> None:
    session_id, turn_id, task_id, details = _seed_goal(
        "v74-request-before-supersede"
    )
    _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    request, _settled, _replay_kwargs = _settle_execution_request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        node_id=str(task.nodes[-1]["insession_task_node_id"]),
        prefix="v74-request-before-supersede",
    )
    current_task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert current_task is not None
    command = _goal_supersede_command(
        prefix="v74-request-before-supersede",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
        expected_task_state_version=current_task.task_state_version,
        observed_task_graph_revision=current_task.current_graph_revision,
    )

    with pytest.raises(
        planning_store.AuxiliaryGoalSupersedeStaleAuthority,
        match="TaskGraph revision authority",
    ):
        planning_store.supersede_auxiliary_planning_goal(command=command)

    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    ) == request
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.goal_status == "active"
    assert current.revision_status == "active"


def test_execution_replan_rejects_pending_goal_supersede_authority() -> None:
    session_id, turn_id, task_id, details = _seed_goal(
        "v74-supersede-before-request"
    )
    _commit_task_graph_one(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    current_task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert current_task is not None
    superseded = planning_store.supersede_auxiliary_planning_goal(
        command=_goal_supersede_command(
            prefix="v74-supersede-before-request",
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            details=details,
            reason=planning_store.PlanningGoalSupersedeReason.BASE_DRIFT,
            expected_task_state_version=current_task.task_state_version,
            observed_task_graph_revision=current_task.current_graph_revision,
        )
    )
    assert superseded.status == "applied"

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="pending Auxiliary goal supersede",
    ):
        _settle_execution_request(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            node_id=str(task.nodes[-1]["insession_task_node_id"]),
            prefix="v74-supersede-before-request",
        )

    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert planning_store.get_pending_auxiliary_goal_supersede_receipt(
        session_id=session_id,
        task_id=task_id,
    ) == superseded.receipt


def test_execution_replan_commit_consumes_atomically_and_exactly_replays(
    monkeypatch,
) -> None:
    command, request, replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix="v74-execution-atomic",
    )

    applied = terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert applied.committed_graph_revision == request.target_graph_revision
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=command.session_id,
        task_id=command.task_id,
    ) is None
    with store._connect() as conn:
        row = conn.execute(
            "SELECT * FROM "
            "insession_task_graph_execution_replan_applications "
            "WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
        assert row is not None
        application = TaskGraphExecutionReplanApplication.model_validate_json(
            str(row["receipt_json"])
        )
        assert application.request_sha256 == request.request_sha256
        assert application.task_graph_commit_apply_id == command.apply_id
        assert application.committed_graph_revision == request.target_graph_revision
    assert terminal_store.commit_auxiliary_task_graph_proposal(
        command=command
    ) == applied.model_copy(update={"status": "replayed"})
    assert work_run_store.commit_work_run_attempt_decision(
        **replay_kwargs
    ).status == "replayed"


def test_execution_replan_commit_reexecutes_requesting_node_without_carry(
    monkeypatch,
) -> None:
    command, request, _replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix="v74-execution-reuse",
        reuse_requesting_node=True,
    )

    applied = terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert applied.status == "applied"
    assert applied.committed_graph_revision == request.target_graph_revision
    assert applied.carry_receipt_ids == ()
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=command.session_id,
        task_id=command.task_id,
    ) is None
    details = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None
    requesting_node = next(
        item
        for item in details.nodes
        if item["insession_task_node_id"] == request.requesting_subject.node_id
    )
    assert (
        requesting_node["node_revision"]
        == request.requesting_subject.node_revision + 1
    )
    assert requesting_node["status"] == "proposed"


def test_execution_replan_commit_rejects_unrelated_auxiliary_goal() -> None:
    command, request, _root_id, _replay_kwargs = _execution_positive_base(
        "v74-execution-goal-mismatch",
        goal_objective="为另一个用户目标构建无关的后续执行图。",
    )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert rejected.value.code == "authority_not_current"
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=command.session_id,
        task_id=command.task_id,
    ) == request


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    (("missing", "authority_not_current"), ("null", "stored_authority_corrupt")),
)
def test_execution_replan_completion_cannot_be_downgraded(
    monkeypatch,
    replacement: str,
    expected_code: str,
) -> None:
    command, request, _replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix=f"v74-execution-completion-{replacement}",
    )
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT apply_id, result_json FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE session_id=? AND insession_task_id=? "
            "AND operation='append_goal_revision' "
            "AND result_json LIKE '%positive_planning_completion%'",
            (command.session_id, command.task_id),
        ).fetchall()
        assert len(rows) == 1
        result = json.loads(str(rows[0]["result_json"]))
        if replacement == "missing":
            result.pop("positive_planning_completion")
        else:
            result["positive_planning_completion"] = None
        result_json = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_graph_revision_apply_receipts_v2 "
            "SET result_json=?, result_sha256=? WHERE apply_id=?",
            (
                result_json,
                hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                str(rows[0]["apply_id"]),
            ),
        )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert rejected.value.code == expected_code
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=command.session_id,
        task_id=command.task_id,
    ) == request


def test_execution_replan_rejects_unauthorized_auxiliary_descendant(
    monkeypatch,
) -> None:
    command, request, _replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix="v74-execution-unauthorized-aux-revision",
    )
    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=command.session_id,
        insession_task_id=command.task_id,
    )
    task = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None and details.budget is not None and task is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_graph_goals SET status='active' "
            "WHERE auxiliary_graph_id=? AND goal_id=?",
            (details.auxiliary_graph_id, details.goal_id),
        )
        conn.execute(
            "UPDATE insession_auxiliary_graph_revision_states_v2 "
            "SET status='active' WHERE auxiliary_graph_id=? "
            "AND auxiliary_graph_revision=?",
            (details.auxiliary_graph_id, details.auxiliary_graph_revision),
        )
    key_by_id = {
        node.auxiliary_node_id: node.local_node_key for node in details.nodes
    }
    appended = auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=command.session_id,
        turn_id=command.source_turn_id,
        insession_task_id=command.task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=details.base_task_graph_revision,
        expected_control_state_version=details.control_state_version,
        expected_current_auxiliary_graph_revision=(
            details.auxiliary_graph_revision
        ),
        apply_id="v74-execution-unauthorized-aux-revision-append",
        goal_objective=details.goal_objective,
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason=AuxiliaryGraphRevisionReason.VERIFICATION_FAILED,
            terminal_node_key=key_by_id[details.terminal_auxiliary_node_id],
            nodes=tuple(
                auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                    local_node_key=node.local_node_key,
                    node_kind=node.node_kind,
                    executor_kind=node.executor_kind,
                    title=node.title,
                    objective=node.objective,
                    source_anchor_ids=node.source_anchor_ids,
                    acceptance_criteria=node.acceptance_criteria,
                    output_contract=node.output_contract,
                    capability_profile_id=node.capability_profile_id,
                    input_resource_aliases=node.input_resource_aliases,
                    required=node.required,
                    origin_node_alias=node.local_node_key,
                )
                for node in details.nodes
            ),
            edges=tuple(
                auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                    dependency_node_key=key_by_id[
                        edge.dependency_auxiliary_node_id
                    ],
                    consumer_node_key=key_by_id[
                        edge.consumer_auxiliary_node_id
                    ],
                    required=edge.required,
                )
                for edge in details.edges
            ),
        ),
        authority_context={"anchors": []},
        budget_profile=details.budget.base_profile.model_dump(mode="json"),
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
    )
    assert appended.committed_auxiliary_graph_revision == (
        details.auxiliary_graph_revision + 1
    )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert rejected.value.code == "stored_authority_corrupt"
    assert work_run_store.get_active_task_graph_execution_replan_request(
        session_id=command.session_id,
        task_id=command.task_id,
    ) == request


def test_execution_replan_commit_rejects_lost_active_pointer_without_authority(
    monkeypatch,
) -> None:
    command, request, _replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix="v74-execution-pointer-loss",
    )
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM "
            "insession_active_task_graph_execution_replan_requests "
            "WHERE request_id=?",
            (request.request_id,),
        )
    omitted = command.model_copy(
        update={
            "task_graph_execution_replan_request_id": None,
            "expected_task_graph_execution_replan_request_sha256": None,
        }
    )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=omitted)

    assert rejected.value.code == "authority_not_current"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=2",
            (command.task_id,),
        ).fetchone()[0] == 0


def test_execution_replan_replay_rejects_application_sql_mirror_tamper(
    monkeypatch,
) -> None:
    command, request, _replay_kwargs = _production_execution_positive_command(
        monkeypatch,
        prefix="v74-execution-app-tamper",
    )
    terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_execution_replan_applications "
            "SET request_sha256=? WHERE request_id=?",
            ("f" * 64, request.request_id),
        )

    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)

    assert rejected.value.code == "stored_authority_corrupt"


@pytest.mark.parametrize("consumed", [False, True])
def test_session_purge_handles_active_and_consumed_execution_replan(
    consumed: bool,
    monkeypatch,
) -> None:
    if not consumed:
        session_id, turn_id, subject = _seed_ready_node(
            suffix="execution-replan-purge-active"
        )
        _settle_execution_request(
            session_id=session_id,
            turn_id=turn_id,
            task_id=subject.task_id,
            node_id=subject.node_id,
            prefix="v74-execution-purge-active",
        )
    else:
        command, _request, _replay_kwargs = _production_execution_positive_command(
            monkeypatch,
            prefix="v74-execution-purge-consumed",
        )
        terminal_store.commit_auxiliary_task_graph_proposal(command=command)
        session_id = command.session_id

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        for table in (
            "insession_active_task_graph_execution_replan_requests",
            "insession_task_graph_execution_replan_applications",
            "insession_task_graph_execution_replan_requests",
        ):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?",
                (session_id,),
            ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
