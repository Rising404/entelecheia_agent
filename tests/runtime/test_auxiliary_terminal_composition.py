from __future__ import annotations

import hashlib
import json
from functools import partial
from types import SimpleNamespace

import pytest

from tests.helpers.auxiliary_project import auxiliary_project_authority  # noqa: F401

from personagraph.l2.auxiliary_graph import (
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationDisposition,
)
from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
)
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_work_run_model_call_authority,
)
from personagraph.l2.auxiliary_execution.planning.controller import (
    AuxiliaryInitialPlanningStatus,
    run_initial_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    build_auxiliary_execution_capability_catalogs,
)
from personagraph.l2.auxiliary_execution.terminal.composition import (
    AuxiliaryTerminalCompositionStatus,
    _semantic_revision_reason_code,
    run_auxiliary_terminal_composition,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    AuxiliaryWorkRunProfile,
    derive_auxiliary_work_run_ids,
    run_auxiliary_model_node,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunRequest,
    run_task_graph_work_runs,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.l2.work_run import (
    CurrentTaskNodeDeliveryResolutionKind,
    TaskNodeSubject,
)
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
    _seed_task,
)
from tests.runtime.test_auxiliary_semantic_verification_controller import (
    _SemanticProvider,
    _authority_factory,
)
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _acceptance,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _completed_child_positive_base,
    _commit_command,
    _default_revision_lineage,
    _insert_base_release_node,
    _revision_proposal,
    _settle_task_revision_trigger,
    _submit_revision,
)


def _complete_planned_model_nodes(monkeypatch):
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    planned = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )
    assert planned.status is AuxiliaryInitialPlanningStatus.PLANNED
    model_profile = WorkRunStructuredModelProfile(
        attempt_max_output_tokens=8_192,
        verification_max_output_tokens=4_096,
        timeout_s=30.0,
    )
    attempt_provider = build_attempt_structured_provider(model_profile)
    verification_provider = build_verification_structured_provider(model_profile)
    capability_catalogs = build_auxiliary_execution_capability_catalogs()
    creation_source = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=task_id,
    )
    for expected_executor in ("model_work_run", "terminal_planner"):
        frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
        )
        candidate = frontier.ready_fresh[0]
        assert candidate.executor_kind.value == expected_executor
        validation_context = None
        if expected_executor == "terminal_planner":
            validation_context = InSessionTaskGraphRevisionValidationContext(
                session_id=session_id,
                source_turn_id=turn_id,
                target_insession_task_id=task_id,
                expected_current_graph_revision=None,
                source_anchors=(creation_source,),
                authorization_anchor_ids=(creation_source.anchor_id,),
                required_anchor_ids=(creation_source.anchor_id,),
            )
        result = run_auxiliary_model_node(
            AuxiliaryWorkRunRequest(
                session_id=session_id,
                turn_id=turn_id,
                subject=candidate.subject,
                executor_kind=candidate.executor_kind,
                initial_driver_state_guard_sha256=(
                    canonical_auxiliary_graph_driver_state_guard(frontier)
                ),
                id_plan=derive_auxiliary_work_run_ids(
                    session_id=session_id,
                    subject=candidate.subject,
                ),
                task_graph_validation_context=validation_context,
            ),
            profile=AuxiliaryWorkRunProfile(),
            capability_catalogs=capability_catalogs,
            attempt_provider=attempt_provider,
            verification_provider=verification_provider,
            emit=lambda _event: None,
            monotonic_clock=iter(range(1, 500)).__next__,
            model_call_authority_factory=(
                partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
            ),
        )
        assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    return session_id, turn_id, task_id


def _complete_current_positive_terminal(
    *,
    prefix: str,
    session_id: str,
    turn_id: str,
    task_id: str,
    proposal: InSessionTaskGraphRevisionProposal,
    lineage: tuple[TaskGraphSemanticLineageProjection, ...],
    goal_objective: str = (
        "Reuse the completed child through one more revision."
    ),
    revision_reason: str = "manual_replan",
):
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert task is not None and task.current_graph_revision is not None
    assert current is not None
    base_revision = task.current_graph_revision
    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="Revise the carried TaskGraph",
        objective="Produce the next complete TaskGraph snapshot.",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(_acceptance(),),
        output_contract="task_graph_revision_proposal_v2",
    )
    auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=task.task_state_version,
        expected_base_task_graph_revision=base_revision,
        expected_control_state_version=current.control_state_version,
        expected_current_auxiliary_graph_revision=(
            current.auxiliary_graph_revision
        ),
        apply_id=f"{prefix}-aux-revision",
        goal_objective=goal_objective,
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason=revision_reason,
            terminal_node_key="synthesize",
            nodes=(terminal,),
            edges=(),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=current.auxiliary_graph_id,
        goal_id=f"{prefix}-goal",
    )
    base = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=task_id,
        graph_revision=base_revision,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    candidate = frontier.ready_fresh[0]
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    completed = run_auxiliary_model_node(
        AuxiliaryWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            subject=candidate.subject,
            executor_kind=candidate.executor_kind,
            initial_driver_state_guard_sha256=(
                canonical_auxiliary_graph_driver_state_guard(frontier)
            ),
            id_plan=derive_auxiliary_work_run_ids(
                session_id=session_id,
                subject=candidate.subject,
            ),
            task_graph_validation_context=context,
            task_graph_semantic_base_snapshot=base.snapshot,
        ),
        profile=AuxiliaryWorkRunProfile(),
        capability_catalogs={},
        attempt_provider=as_prepared_test_provider(
            _ReplyProvider(
                [_submit_revision(proposal, lineage)],
                provider="mock",
                model="mock-structured",
            )
        ),
        verification_provider=as_prepared_test_provider(
            _PassVerifier(
                provider="mock",
                model="mock-structured",
            )
        ),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 200)).__next__,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )
    assert (
        completed.status is AuxiliaryWorkRunStatus.COMPLETED
    ), completed.model_dump(mode="json")
    return base


def _commit_completed_delivery_carry_chain(prefix: str):
    command, root_id, child_id = _completed_child_positive_base(
        f"{prefix}-revision-two"
    )
    revision_two = terminal_store.commit_auxiliary_task_graph_proposal(
        command=command
    )
    assert revision_two.committed_graph_revision == 2
    assert len(revision_two.carry_receipt_ids) == 1

    proposal_payload = _revision_proposal().model_dump(mode="json")
    proposal_payload["root"]["nodes"][0]["objective"] = (
        "交付一份再次吸收已验证发布结果的第三版执行计划"
    )
    proposal = InSessionTaskGraphRevisionProposal.model_validate(
        proposal_payload
    )
    lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="revise",
            base_node_alias="base_node_000",
        ),
        TaskGraphSemanticLineageProjection(
            proposal_node_key="release",
            disposition="reuse",
            base_node_alias="base_node_001",
        ),
    )
    base_projection = _complete_current_positive_terminal(
        prefix=f"{prefix}-revision-three",
        session_id=command.session_id,
        turn_id=command.source_turn_id,
        task_id=command.task_id,
        proposal=proposal,
        lineage=lineage,
    )
    semantic_provider = _SemanticProvider()
    authority_factory, _bindings = _authority_factory()
    revision_three = run_auxiliary_terminal_composition(
        session_id=command.session_id,
        turn_id=command.source_turn_id,
        task_id=command.task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(semantic_provider),
        semantic_model_call_authority_factory=authority_factory,
    )
    assert revision_three.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    assert revision_three.task_graph_commit is not None
    assert revision_three.task_graph_commit.previous_graph_revision == 2
    assert revision_three.task_graph_commit.committed_graph_revision == 3
    assert len(revision_three.task_graph_commit.carry_receipt_ids) == 1
    return (
        command,
        revision_two,
        revision_three,
        root_id,
        child_id,
        base_projection,
    )


@pytest.mark.parametrize(
    ("disposition", "expected"),
    (
        (
            TaskGraphSemanticVerificationDisposition.REVISE,
            "semantic_review_requires_revision",
        ),
        (
            TaskGraphSemanticVerificationDisposition.BLOCKED,
            "semantic_review_blocked_by_insufficient_evidence",
        ),
    ),
)
def test_non_pass_terminal_reason_codes_are_distinct(disposition, expected) -> None:
    assert _semantic_revision_reason_code(disposition) == expected


def test_positive_terminal_composition_binds_exact_active_revision_trigger(
    monkeypatch,
) -> None:
    session_id = "trigger-terminal-session"
    turn_id = "trigger-terminal-turn"
    task_id = "trigger-terminal-task"
    details = SimpleNamespace(
        auxiliary_graph_id="trigger-terminal-aux",
        goal_id="trigger-terminal-goal",
        auxiliary_graph_revision=2,
        structure_sha256="a" * 64,
        goal_status="proposal_ready",
        revision_status="proposal_ready",
        base_task_graph_revision=1,
        target_task_graph_revision=2,
    )
    task = SimpleNamespace(
        current_graph_revision=1,
        task_state_version=17,
    )
    trigger = SimpleNamespace(
        trigger_id="trigger-terminal-revision-trigger",
        trigger_sha256="b" * 64,
        session_id=session_id,
        task_id=task_id,
        base_graph_revision=1,
        target_graph_revision=2,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        lambda _session_id, _task_id: task,
    )
    monkeypatch.setattr(
        auxiliary_graph_store,
        "get_auxiliary_graph_for_task",
        lambda **_kwargs: details,
    )
    monkeypatch.setattr(
        task_graph_store,
        "project_task_graph_semantic_base",
        lambda **_kwargs: SimpleNamespace(
            snapshot=object(),
            base_node_alias_bindings=(),
        ),
    )
    monkeypatch.setattr(
        auxiliary_graph_store,
        "project_auxiliary_graph_execution_frontier",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "personagraph.l2.auxiliary_execution.terminal.composition."
        "decide_auxiliary_graph_driver_step",
        lambda _frontier: SimpleNamespace(
            action=AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL
        ),
    )
    monkeypatch.setattr(
        terminal_store,
        "get_auxiliary_terminal_proposal_receipt",
        lambda **_kwargs: SimpleNamespace(
            terminal_proposal_receipt_id="trigger-terminal-receipt"
        ),
    )
    monkeypatch.setattr(
        store,
        "get_turn_execution_window",
        lambda _session_id: {"turn_id": turn_id, "state_version": 23},
    )
    monkeypatch.setattr(
        task_delivery_store,
        "get_active_task_graph_revision_trigger",
        lambda **_kwargs: trigger,
    )

    def commit(*, command):
        captured["command"] = command
        return SimpleNamespace(status="applied")

    monkeypatch.setattr(
        terminal_store,
        "commit_auxiliary_task_graph_proposal",
        commit,
    )

    result = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
    )

    assert result.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    command = captured["command"]
    assert isinstance(command, terminal_store.CommitAuxiliaryTaskGraphProposalCommand)
    assert command.task_graph_revision_trigger_id == trigger.trigger_id
    assert (
        command.expected_task_graph_revision_trigger_sha256
        == trigger.trigger_sha256
    )


def test_base_null_terminal_composition_commits_and_replays(monkeypatch) -> None:
    session_id, turn_id, task_id = _complete_planned_model_nodes(monkeypatch)

    committed = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
    )

    assert (
        committed.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    ), (committed.reason_code, committed.semantic_result)
    assert committed.semantic_result is not None
    assert committed.terminal_seal is not None
    assert committed.task_graph_commit is not None
    assert committed.task_graph_commit.committed_graph_revision == 1
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == 1
    assert len(task.nodes) == 1

    replayed = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
    )

    assert (
        replayed.status
        is AuxiliaryTerminalCompositionStatus.ALREADY_COMMITTED
    )
    assert replayed.id_plan == committed.id_plan


def test_positive_base_terminal_composition_reviews_lineage_and_commits_revision_two(
    monkeypatch,
) -> None:
    first_command = _commit_command("runtime-positive-terminal-base")
    first = terminal_store.commit_auxiliary_task_graph_proposal(command=first_command)
    task = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=first_command.session_id,
        insession_task_id=first_command.task_id,
    )
    assert task is not None and task.current_graph_revision == 1
    assert current is not None

    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="Revise the TaskGraph",
        objective="Produce the complete next TaskGraph snapshot.",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(_acceptance(),),
        output_contract="task_graph_revision_proposal_v2",
    )
    auxiliary_graph_store.commit_auxiliary_graph_revision(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        insession_task_id=first_command.task_id,
        expected_task_state_version=first.task_state_version,
        expected_base_task_graph_revision=1,
        expected_control_state_version=current.control_state_version,
        expected_current_auxiliary_graph_revision=(
            current.auxiliary_graph_revision
        ),
        apply_id="runtime-positive-terminal-aux-revision",
        goal_objective="Preserve the root identity and add a release step.",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason="manual_replan",
            terminal_node_key="synthesize",
            nodes=(terminal,),
            edges=(),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=current.auxiliary_graph_id,
        goal_id="runtime-positive-terminal-goal-2",
    )
    base = task_graph_store.project_task_graph_semantic_base(
        session_id=first_command.session_id,
        task_id=first_command.task_id,
        graph_revision=1,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        insession_task_id=first_command.task_id,
    )
    candidate = frontier.ready_fresh[0]
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=first_command.session_id,
        invocation_turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
    )
    completed = run_auxiliary_model_node(
        AuxiliaryWorkRunRequest(
            session_id=first_command.session_id,
            turn_id=first_command.source_turn_id,
            subject=candidate.subject,
            executor_kind=candidate.executor_kind,
            initial_driver_state_guard_sha256=(
                canonical_auxiliary_graph_driver_state_guard(frontier)
            ),
            id_plan=derive_auxiliary_work_run_ids(
                session_id=first_command.session_id,
                subject=candidate.subject,
            ),
            task_graph_validation_context=context,
            task_graph_semantic_base_snapshot=base.snapshot,
        ),
        profile=AuxiliaryWorkRunProfile(),
        capability_catalogs={},
        attempt_provider=as_prepared_test_provider(
            _ReplyProvider(
                [
                    _submit_revision(
                        _revision_proposal(),
                        _default_revision_lineage(),
                    )
                ],
                provider="mock",
                model="mock-structured",
            )
        ),
        verification_provider=as_prepared_test_provider(
            _PassVerifier(
                provider="mock",
                model="mock-structured",
            )
        ),
        emit=lambda _event: None,
        monotonic_clock=iter(range(1, 200)).__next__,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )
    assert (
        completed.status is AuxiliaryWorkRunStatus.COMPLETED
    ), completed.model_dump(mode="json")

    semantic_provider = _SemanticProvider()
    authority_factory, bindings = _authority_factory()
    committed = run_auxiliary_terminal_composition(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(semantic_provider),
        semantic_model_call_authority_factory=authority_factory,
    )

    assert (
        committed.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    ), (committed.reason_code, committed.semantic_result)
    assert committed.task_graph_commit is not None
    assert committed.task_graph_commit.previous_graph_revision == 1
    assert committed.task_graph_commit.committed_graph_revision == 2
    assert len(bindings) == 1
    bound_request = json.loads(bindings[0].request_json)["verification_request"]
    prompt = bound_request["prompt_payload"]
    assert prompt["goal"]["objective"] == (
        "Preserve the root identity and add a release step."
    )
    assert prompt["base_task_graph"] == base.snapshot.model_dump(mode="json")
    assert tuple(item["proposal_node_key"] for item in prompt["lineage"]) == (
        "root",
        "release",
    )
    current_task = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    assert current_task is not None
    assert current_task.current_graph_revision == 2

    replayed = run_auxiliary_terminal_composition(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(semantic_provider),
        semantic_model_call_authority_factory=authority_factory,
    )
    assert replayed.status is AuxiliaryTerminalCompositionStatus.ALREADY_COMMITTED
    assert len(semantic_provider.calls) == 1


def test_positive_terminal_composition_atomically_consumes_delivery_trigger() -> None:
    first_command = _commit_command("runtime-trigger-terminal-base")
    terminal_store.commit_auxiliary_task_graph_proposal(command=first_command)
    base = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    assert base is not None and base.current_graph_revision == 1
    root_id = str(base.nodes[0]["insession_task_node_id"])
    child_id = "runtime-trigger-terminal-release-node"
    _insert_base_release_node(
        task_id=first_command.task_id,
        base_node_id=root_id,
        child_id=child_id,
    )
    # Candidate review is part of the root's delivery gate. Run execution and
    # settlement together; an already completed Task cannot be reviewed later.
    trigger = _settle_task_revision_trigger(
        prefix="runtime-trigger-terminal",
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
    )
    settled_base = task_graph_store.get_insession_task_details(
        first_command.session_id, first_command.task_id
    )
    assert settled_base is not None
    assert settled_base.status.value == "active"
    assert any(
        node["insession_task_node_id"] == child_id and node["status"] == "completed"
        for node in settled_base.nodes
    )
    proposal_payload = _revision_proposal().model_dump(mode="json")
    proposal_payload["root"]["nodes"][0]["objective"] = (
        "交付一份修复完整交付缺口的第二版执行计划"
    )
    proposal = InSessionTaskGraphRevisionProposal.model_validate(
        proposal_payload
    )
    lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="revise",
            base_node_alias="base_node_000",
        ),
        TaskGraphSemanticLineageProjection(
            proposal_node_key="release",
            disposition="reuse",
            base_node_alias="base_node_001",
        ),
    )
    _complete_current_positive_terminal(
        prefix="runtime-trigger-terminal",
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        proposal=proposal,
        lineage=lineage,
        goal_objective=trigger.revision_objective,
        revision_reason="verification_failed",
    )
    semantic_provider = _SemanticProvider()
    authority_factory, _bindings = _authority_factory()

    committed = run_auxiliary_terminal_composition(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(semantic_provider),
        semantic_model_call_authority_factory=authority_factory,
    )

    assert committed.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    assert committed.task_graph_commit is not None
    assert committed.task_graph_commit.previous_graph_revision == 1
    assert committed.task_graph_commit.committed_graph_revision == 2
    assert len(committed.task_graph_commit.carry_receipt_ids) == 1
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=first_command.session_id,
        task_id=first_command.task_id,
    ) is None
    current = task_graph_store.get_insession_task_details(
        first_command.session_id,
        first_command.task_id,
    )
    assert current is not None and current.current_graph_revision == 2
    by_id = {
        str(item["insession_task_node_id"]): item for item in current.nodes
    }
    assert int(by_id[root_id]["node_revision"]) == 2
    assert str(by_id[root_id]["status"]) == "proposed"
    assert int(by_id[child_id]["node_revision"]) == 1
    assert str(by_id[child_id]["status"]) == "completed"
    with store._connect() as conn:
        application = conn.execute(
            "SELECT trigger_sha256, base_graph_revision, "
            "committed_graph_revision, task_graph_commit_apply_id "
            "FROM insession_task_graph_revision_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()
        assert application is not None
        assert str(application["trigger_sha256"]) == trigger.trigger_sha256
        assert int(application["base_graph_revision"]) == 1
        assert int(application["committed_graph_revision"]) == 2
        assert (
            str(application["task_graph_commit_apply_id"])
            == committed.id_plan.task_graph_commit_apply_id
        )
    semantic_call_count = len(semantic_provider.calls)
    assert semantic_call_count == 2
    replayed = run_auxiliary_terminal_composition(
        session_id=first_command.session_id,
        turn_id=first_command.source_turn_id,
        task_id=first_command.task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(semantic_provider),
        semantic_model_call_authority_factory=authority_factory,
    )
    assert replayed.status is AuxiliaryTerminalCompositionStatus.ALREADY_COMMITTED
    assert len(semantic_provider.calls) == semantic_call_count


def test_completed_delivery_carries_from_revision_one_through_three() -> None:
    command, revision_two, revision_three, root_id, child_id, _base_projection = (
        _commit_completed_delivery_carry_chain("runtime-carry-chain")
    )
    details = task_graph_store.get_insession_task_details(command.session_id, command.task_id)
    assert details is not None and details.current_graph_revision == 3
    node_by_id = {
        str(item["insession_task_node_id"]): item for item in details.nodes
    }
    assert int(node_by_id[root_id]["node_revision"]) == 3
    assert str(node_by_id[root_id]["status"]) == "proposed"
    assert int(node_by_id[child_id]["node_revision"]) == 1
    assert str(node_by_id[child_id]["status"]) == "completed"

    root_subject = TaskNodeSubject(
        task_id=command.task_id,
        graph_revision=3,
        node_id=root_id,
        node_revision=3,
    )
    dependencies = work_run_store.get_current_task_node_dependency_deliveries(
        session_id=command.session_id,
        subject=root_subject,
    )
    assert len(dependencies) == 1
    carried = dependencies[0]
    assert carried.resolution_kind is CurrentTaskNodeDeliveryResolutionKind.CARRIED
    assert carried.target_subject == TaskNodeSubject(
        task_id=command.task_id,
        graph_revision=3,
        node_id=child_id,
        node_revision=1,
    )
    assert carried.source_delivery.delivery.subject.graph_revision == 1
    assert carried.delivery_id == carried.source_delivery.delivery.delivery_id
    assert carried.carry_authority is not None
    assert carried.carry_authority.base_task_graph_revision == 2
    assert (
        carried.carry_authority.carry_receipt_id
        == revision_three.task_graph_commit.carry_receipt_ids[0]
    )

    with store._connect() as conn:
        carry_rows = conn.execute(
            "SELECT target_task_graph_revision, source_delivery_id "
            "FROM insession_auxiliary_v2_task_graph_node_carry_receipts "
            "WHERE insession_task_id=? AND insession_task_node_id=? "
            "ORDER BY target_task_graph_revision",
            (command.task_id, child_id),
        ).fetchall()
        assert [int(row["target_task_graph_revision"]) for row in carry_rows] == [
            2,
            3,
        ]
        assert len({str(row["source_delivery_id"]) for row in carry_rows}) == 1
        assert conn.execute(
            "SELECT graph_revision FROM insession_task_node_deliveries "
            "WHERE delivery_id=?",
            (str(carry_rows[0]["source_delivery_id"]),),
        ).fetchone()[0] == 1
        assert [
            int(row["graph_revision"])
            for row in conn.execute(
                "SELECT graph_revision FROM insession_task_graph_revisions "
                "WHERE insession_task_id=? ORDER BY graph_revision",
                (command.task_id,),
            ).fetchall()
        ] == [1, 2, 3]

    window = store.get_turn_execution_window(command.session_id)
    assert window is not None
    executed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=command.session_id,
            turn_id=command.source_turn_id,
            task_id=command.task_id,
            expected_window_revision=int(window["state_version"]),
        ),
        monotonic_clock=iter(range(1, 500)).__next__,
    )
    assert executed.status == "completed"
    assert executed.final_delivery_id is not None


def test_carry_chain_predecessor_tamper_rejects_resolution_and_commit_replay() -> None:
    command, _revision_two, revision_three, root_id, child_id, base_projection = (
        _commit_completed_delivery_carry_chain("runtime-carry-chain-tamper")
    )
    assert revision_three.task_graph_commit is not None
    with store._connect() as conn:
        commit_row = conn.execute(
            "SELECT source_turn_id, terminal_proposal_receipt_id, "
            "expected_task_state_version, expected_window_revision "
            "FROM insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE apply_id=?",
            (revision_three.id_plan.task_graph_commit_apply_id,),
        ).fetchone()
        predecessor = conn.execute(
            "SELECT carry_receipt_id, receipt_json FROM "
            "insession_auxiliary_v2_task_graph_node_carry_receipts "
            "WHERE insession_task_id=? AND target_task_graph_revision=2 "
            "AND insession_task_node_id=?",
            (command.task_id, child_id),
        ).fetchone()
        assert commit_row is not None and predecessor is not None
        receipt = json.loads(str(predecessor["receipt_json"]))
        receipt["dependency_delivery_ids"] = ["forged-dependency"]
        receipt_json = json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_v2_task_graph_node_carry_receipts "
            "SET dependency_delivery_ids_json='[\"forged-dependency\"]', "
            "receipt_json=?, receipt_sha256=? WHERE carry_receipt_id=?",
            (
                receipt_json,
                hashlib.sha256(receipt_json.encode("utf-8")).hexdigest(),
                str(predecessor["carry_receipt_id"]),
            ),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="source",
    ):
        work_run_store.get_current_task_node_dependency_deliveries(
            session_id=command.session_id,
            subject=TaskNodeSubject(
                task_id=command.task_id,
                graph_revision=3,
                node_id=root_id,
                node_revision=3,
            ),
        )

    replay_command = terminal_store.CommitAuxiliaryTaskGraphProposalCommand(
        apply_id=revision_three.id_plan.task_graph_commit_apply_id,
        session_id=command.session_id,
        source_turn_id=str(commit_row["source_turn_id"]),
        task_id=command.task_id,
        terminal_proposal_receipt_id=str(
            commit_row["terminal_proposal_receipt_id"]
        ),
        expected_base_task_graph_revision=2,
        expected_task_state_version=int(
            commit_row["expected_task_state_version"]
        ),
        expected_window_revision=int(commit_row["expected_window_revision"]),
        base_node_alias_bindings=base_projection.base_node_alias_bindings,
    )
    with pytest.raises(
        terminal_store.AuxiliaryTaskGraphCommitPersistenceError
    ) as rejected:
        terminal_store.commit_auxiliary_task_graph_proposal(command=replay_command)
    assert rejected.value.code == "stored_authority_corrupt"


def test_semantic_revise_stops_before_terminal_seal_and_exactly_replays(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _complete_planned_model_nodes(monkeypatch)
    provider = _SemanticProvider(failed_reviewer_ordinal=1)
    authority_factory, bindings = _authority_factory()

    revision_required = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(provider),
        semantic_model_call_authority_factory=authority_factory,
    )

    assert (
        revision_required.status
        is AuxiliaryTerminalCompositionStatus.REVISION_REQUIRED
    )
    assert revision_required.reason_code == "semantic_review_requires_revision"
    assert revision_required.semantic_result is not None
    assert revision_required.semantic_result.settlement is not None
    assert revision_required.terminal_seal is None
    assert revision_required.task_graph_commit is None
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision is None
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_v2_terminal_seal_apply_receipts"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_v2_task_graph_commit_receipts"
        ).fetchone()[0] == 0

    replayed = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
        semantic_provider=as_prepared_test_provider(provider),
        semantic_model_call_authority_factory=authority_factory,
    )

    assert replayed.status is AuxiliaryTerminalCompositionStatus.REVISION_REQUIRED
    assert replayed.reason_code == revision_required.reason_code
    assert replayed.semantic_result is not None
    assert replayed.semantic_result.settlement == (
        revision_required.semantic_result.settlement
    )
    assert replayed.semantic_result.replayed is True
    assert len(provider.calls) == 1
    assert len(bindings) == 1


def test_response_loss_after_terminal_seal_resumes_at_commit(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _complete_planned_model_nodes(monkeypatch)
    real_commit = terminal_store.commit_auxiliary_task_graph_proposal

    def lose_before_commit(*, command):
        raise RuntimeError("simulated response loss after terminal seal")

    monkeypatch.setattr(
        terminal_store,
        "commit_auxiliary_task_graph_proposal",
        lose_before_commit,
    )
    try:
        run_auxiliary_terminal_composition(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            model_ledger_store=store,
            emit=lambda _event: None,
        )
    except RuntimeError as exc:
        assert "response loss" in str(exc)
    else:
        raise AssertionError("simulated response loss did not escape")

    monkeypatch.setattr(
        terminal_store,
        "commit_auxiliary_task_graph_proposal",
        real_commit,
    )
    resumed = run_auxiliary_terminal_composition(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        model_ledger_store=store,
        emit=lambda _event: None,
    )

    assert resumed.status is AuxiliaryTerminalCompositionStatus.COMMITTED
    assert resumed.semantic_result is None
    assert resumed.terminal_seal is None
    assert resumed.task_graph_commit is not None
