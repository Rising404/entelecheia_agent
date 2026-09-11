from __future__ import annotations

import json

import pytest

from personagraph.l2.auxiliary_graph import (
    PlanningEpisodeBudgetDisposition,
    PlanningEpisodeBudget,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticFailureScope,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationItem,
    TaskGraphSemanticVerificationResult,
    TaskGraphSemanticVerificationVerdict,
)
from personagraph.l2.auxiliary_graph import PlanningObservationStatus
from personagraph.session import store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_terminal_seal
from personagraph.l2.work_run import (
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
)
from tests.runtime.test_auxiliary_work_run_controller import (
    _PassVerifier,
    _ReplyProvider,
    _commit_graph,
    _request,
    _run,
    _seed_task,
    _subject,
    _task_graph_submit,
)
from tests.session.test_auxiliary_semantic_verification_persistence import (
    _catalog,
    _complete_terminal_with_host_artifact,
    _complete_terminal_only,
    _freeze_command,
    _pass_items,
    _request_command,
    _result,
    _result_command,
    _semantic_request,
    _settlement_command,
    _terminal_proposal,
)


def _seal_command(
    *,
    prefix: str,
    session_id: str,
    turn_id: str,
    task_id: str,
    details,
    semantic_settlement_id: str,
    semantic_prompt_payload_sha256: str,
):
    with store._connect() as conn:
        task_state_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (task_id,),
            ).fetchone()[0]
        )
        completion = conn.execute(
            "SELECT completion_id, auxiliary_node_id, node_revision "
            "FROM insession_auxiliary_node_completions_v2 "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND auxiliary_node_id=?",
            (
                details.auxiliary_graph_id,
                details.auxiliary_graph_revision,
                details.terminal_auxiliary_node_id,
            ),
        ).fetchone()
    assert completion is not None
    return terminal_store.SealAuxiliaryTerminalProposalCommand(
        apply_id=f"{prefix}-seal-apply",
        finish_gate_receipt_id=f"{prefix}-finish-gate",
        terminal_proposal_receipt_id=f"{prefix}-terminal-proposal",
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        terminal_auxiliary_node_id=str(completion["auxiliary_node_id"]),
        terminal_node_revision=int(completion["node_revision"]),
        terminal_completion_id=str(completion["completion_id"]),
        semantic_settlement_id=semantic_settlement_id,
        semantic_prompt_payload_sha256=semantic_prompt_payload_sha256,
        expected_base_task_graph_revision=details.base_task_graph_revision,
        expected_task_state_version=task_state_version,
        expected_control_state_version=details.control_state_version,
        expected_goal_state_version=details.goal_state_version,
        expected_revision_state_version=details.revision_state_version,
        expected_budget_state_version=details.budget_state_version,
        expected_structure_sha256=details.structure_sha256,
        expected_budget_snapshot_sha256=details.budget.snapshot_sha256,
    )


def _settled_terminal(prefix: str):
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_only(prefix)
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(
            session_id,
            turn_id,
            task_id,
            details,
            catalog,
        )
    )
    request = _semantic_request(
        request_id=f"{prefix}-semantic-request",
        logical_call_id=f"{prefix}-semantic-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    result = _result(request, f"{prefix}-semantic-result")
    semantic_store.commit_auxiliary_semantic_verification_result(
        command=_result_command(session_id, turn_id, details, result)
    )
    settlement = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            session_id,
            turn_id,
            details,
            (request,),
            (result,),
            f"{prefix}-semantic-settlement",
        )
    ).settlement
    command = _seal_command(
        prefix=prefix,
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        semantic_settlement_id=settlement.settlement_id,
        semantic_prompt_payload_sha256=(settlement.frozen_prompt_payload_sha256),
    )
    return command, settlement


def test_terminal_seal_is_atomic_exactly_replayable_and_only() -> None:
    command, settlement = _settled_terminal("terminal-seal-pass")

    applied = terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert applied.status == "applied"
    assert applied.readiness_status == "proposal_ready"
    assert applied.semantic_settlement_sha256 == settlement.settlement_sha256
    assert terminal_store.seal_auxiliary_terminal_proposal(
        command=command
    ) == applied.model_copy(update={"status": "replayed"})

    receipt = terminal_store.get_auxiliary_terminal_proposal_receipt(
        session_id=command.session_id,
        terminal_proposal_receipt_id=command.terminal_proposal_receipt_id,
    )
    assert receipt.proposal_sha256 == applied.proposal_sha256
    assert receipt.output_window.content
    with store._connect() as conn:
        goal = conn.execute(
            "SELECT status, state_version FROM "
            "insession_auxiliary_graph_goals WHERE goal_id=?",
            (command.goal_id,),
        ).fetchone()
        revision = conn.execute(
            "SELECT status, state_version FROM "
            "insession_auxiliary_graph_revision_states_v2 "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
            (command.auxiliary_graph_id, command.auxiliary_graph_revision),
        ).fetchone()
        assert tuple(goal) == (
            "proposal_ready",
            command.expected_goal_state_version + 1,
        )
        assert tuple(revision) == (
            "proposal_ready",
            command.expected_revision_state_version + 1,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_terminal_seal_accepts_unchanged_window_resubmitted_by_later_attempt() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    runtime_request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="terminal-seal-unchanged-retry",
    )
    gate_calls = 0

    def gate(_context, _result):
        nonlocal gate_calls
        gate_calls += 1
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_semantic",
                disposition=(
                    DownstreamVerificationDisposition.RETRY_ATTEMPT
                    if gate_calls == 1
                    else DownstreamVerificationDisposition.PASS
                ),
                finding=(
                    "请重试并明确依赖已经完成的观察节点。"
                    if gate_calls == 1
                    else "终端提案通过独立语义审查。"
                ),
                repair_objective=(
                    "保留合法结构并明确依赖关系。"
                    if gate_calls == 1
                    else None
                ),
                source_result_id=f"terminal-seal-unchanged-review-{gate_calls}",
                source_result_sha256=str(gate_calls) * 64,
                affected_subject_ids=(subject.node_id,) if gate_calls == 1 else (),
            ),
        )

    completed = _run(
        runtime_request,
        attempt_provider=_ReplyProvider(
            [_task_graph_submit(), _task_graph_submit()]
        ),
        verifier=_PassVerifier(),
        terminal_downstream_gate=gate,
    )
    assert completed.status.value == "completed"
    assert completed.attempt_id.endswith(":attempt-2")

    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    semantic_request = _semantic_request(
        request_id="terminal-seal-unchanged-semantic-request",
        logical_call_id="terminal-seal-unchanged-semantic-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=(),
    )
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(
            session_id,
            turn_id,
            details,
            semantic_request,
        )
    )
    semantic_result = _result(
        semantic_request,
        "terminal-seal-unchanged-semantic-result",
    )
    semantic_store.commit_auxiliary_semantic_verification_result(
        command=_result_command(
            session_id,
            turn_id,
            details,
            semantic_result,
        )
    )
    settlement = semantic_store.settle_auxiliary_semantic_verification_quorum(
        command=_settlement_command(
            session_id,
            turn_id,
            details,
            (semantic_request,),
            (semantic_result,),
            "terminal-seal-unchanged-semantic-settlement",
        )
    ).settlement
    command = _seal_command(
        prefix="terminal-seal-unchanged-retry",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        semantic_settlement_id=settlement.settlement_id,
        semantic_prompt_payload_sha256=(
            settlement.frozen_prompt_payload_sha256
        ),
    )

    with store._connect() as conn:
        provenance = conn.execute(
            "SELECT completion.submitted_attempt_id, "
            "output.updated_attempt_id, attempt.submitted_output_revision, "
            "completion.output_revision, output.snapshot_json, "
            "output.snapshot_hash, completion.work_run_id, "
            "completion.completion_json, completion.completion_sha256 "
            "FROM insession_auxiliary_node_completions_v2 AS completion "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=completion.work_run_id "
            "JOIN insession_work_run_attempts AS attempt "
            "ON attempt.work_run_id=completion.work_run_id "
            "AND attempt.attempt_id=completion.submitted_attempt_id "
            "WHERE completion.completion_id=?",
            (command.terminal_completion_id,),
        ).fetchone()
    assert provenance is not None
    assert provenance["submitted_attempt_id"] != provenance["updated_attempt_id"]
    assert provenance["submitted_output_revision"] == provenance["output_revision"]

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts "
            "SET submitted_output_revision=NULL WHERE attempt_id=?",
            (provenance["submitted_attempt_id"],),
        )
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as corrupt_submitter:
        terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert corrupt_submitter.value.code == "terminal_completion_invalid"
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts "
            "SET submitted_output_revision=? WHERE attempt_id=?",
            (
                provenance["output_revision"],
                provenance["submitted_attempt_id"],
            ),
        )

    tampered_snapshot = json.loads(provenance["snapshot_json"])
    tampered_snapshot["updated_attempt_id"] = provenance["submitted_attempt_id"]
    tampered_snapshot_json = (
        auxiliary_terminal_seal.OutputWindow.model_validate(
            tampered_snapshot
        ).model_dump_json()
    )
    tampered_snapshot_hash = auxiliary_terminal_seal._sha256_text(
        tampered_snapshot_json
    )
    tampered_completion = json.loads(provenance["completion_json"])
    tampered_completion["output_snapshot_sha256"] = tampered_snapshot_hash
    tampered_completion_json = auxiliary_graphs._canonical_json(
        tampered_completion
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows "
            "SET updated_attempt_id=?, snapshot_json=?, snapshot_hash=? "
            "WHERE work_run_id=?",
            (
                provenance["submitted_attempt_id"],
                tampered_snapshot_json,
                tampered_snapshot_hash,
                provenance["work_run_id"],
            ),
        )
        conn.execute(
            "UPDATE insession_auxiliary_node_completions_v2 "
            "SET completion_json=?, completion_sha256=? "
            "WHERE completion_id=?",
            (
                tampered_completion_json,
                auxiliary_graphs._text_hash(tampered_completion_json),
                command.terminal_completion_id,
            ),
        )
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as false_output_author:
        terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert false_output_author.value.code == "terminal_completion_invalid"
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows "
            "SET updated_attempt_id=?, snapshot_json=?, snapshot_hash=? "
            "WHERE work_run_id=?",
            (
                provenance["updated_attempt_id"],
                provenance["snapshot_json"],
                provenance["snapshot_hash"],
                provenance["work_run_id"],
            ),
        )
        conn.execute(
            "UPDATE insession_auxiliary_node_completions_v2 "
            "SET completion_json=?, completion_sha256=? "
            "WHERE completion_id=?",
            (
                provenance["completion_json"],
                provenance["completion_sha256"],
                command.terminal_completion_id,
            ),
        )

    applied = terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert applied.status == "applied"


def test_terminal_output_material_is_base_discriminated_and_canonical() -> None:
    proposal = _terminal_proposal()
    lineage = (
        TaskGraphSemanticLineageProjection(
            proposal_node_key="root",
            disposition="revise",
            base_node_alias="base_node_000",
        ),
    )
    candidate = TaskGraphRevisionCandidate(
        proposal=proposal,
        lineage=lineage,
    )

    assert auxiliary_terminal_seal._parse_terminal_output_material(
        proposal.model_dump_json(),
        expected_base_task_graph_revision=None,
    ) == (proposal, ())
    assert auxiliary_terminal_seal._parse_terminal_output_material(
        candidate.model_dump_json(),
        expected_base_task_graph_revision=1,
    ) == (proposal, lineage)

    with pytest.raises(ValueError):
        auxiliary_terminal_seal._parse_terminal_output_material(
            proposal.model_dump_json(),
            expected_base_task_graph_revision=1,
        )
    with pytest.raises(ValueError):
        auxiliary_terminal_seal._parse_terminal_output_material(
            candidate.model_dump_json(),
            expected_base_task_graph_revision=None,
        )
    with pytest.raises(ValueError, match="not canonical"):
        auxiliary_terminal_seal._parse_terminal_output_material(
            json.dumps(proposal.model_dump(mode="json"), indent=2),
            expected_base_task_graph_revision=None,
        )


def test_terminal_seal_rejects_collision_tamper_and_base_drift() -> None:
    command, _settlement = _settled_terminal("terminal-seal-guards")
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as base_drift:
        terminal_store.seal_auxiliary_terminal_proposal(
            command=command.model_copy(
                update={"expected_base_task_graph_revision": 1}
            )
        )
    assert base_drift.value.code == "base_task_graph_drift"

    applied = terminal_store.seal_auxiliary_terminal_proposal(command=command)

    with pytest.raises(terminal_store.AuxiliaryTerminalSealIdentityCollision):
        terminal_store.seal_auxiliary_terminal_proposal(
            command=command.model_copy(
                update={"finish_gate_receipt_id": "different-finish-gate"}
            )
        )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_terminal_proposal_receipts "
            "SET proposal_json='{}' WHERE terminal_proposal_receipt_id=?",
            (applied.terminal_proposal_receipt_id,),
        )
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as corrupt:
        terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert corrupt.value.code == "stored_authority_corrupt"


def test_terminal_seal_rejects_missing_semantic_and_stale_versions() -> None:
    command, _settlement = _settled_terminal("terminal-seal-fail-closed")
    missing = command.model_copy(
        update={
            "semantic_settlement_id": "missing-semantic-settlement",
            "semantic_prompt_payload_sha256": "f" * 64,
        }
    )
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as no_semantic:
        terminal_store.seal_auxiliary_terminal_proposal(command=missing)
    assert no_semantic.value.code == "semantic_settlement_missing"

    stale = command.model_copy(
        update={
            "apply_id": "terminal-seal-stale-apply",
            "expected_revision_state_version": (
                command.expected_revision_state_version + 1
            ),
        }
    )
    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as stale_error:
        terminal_store.seal_auxiliary_terminal_proposal(command=stale)
    assert stale_error.value.code == "state_version_conflict"


def test_terminal_seal_rejects_persisted_semantic_failure() -> None:
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_only("terminal-seal-semantic-fail")
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    request = _semantic_request(
        request_id="terminal-seal-failed-semantic-request",
        logical_call_id="terminal-seal-failed-semantic-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    failed_items = tuple(
        TaskGraphSemanticVerificationItem(
            dimension=item.dimension,
            verdict=(
                TaskGraphSemanticVerificationVerdict.FAIL
                if item.dimension
                is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                else item.verdict
            ),
            failure_scope=(
                TaskGraphSemanticFailureScope.TERMINAL_PROPOSAL
                if item.dimension
                is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                else None
            ),
            finding=(
                "The frozen proposal does not cover the requested goal."
                if item.dimension
                is TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
                else item.finding
            ),
        )
        for item in _pass_items()
    )
    result = TaskGraphSemanticVerificationResult.create(
        verification_result_id="terminal-seal-failed-semantic-result",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=request.reviewer_ordinal,
        required_reviewer_count=request.required_reviewer_count,
        items=failed_items,
    )
    assert result.host_disposition == "revise"
    semantic_store.commit_auxiliary_semantic_verification_result(
        command=_result_command(session_id, turn_id, details, result)
    )
    command = _seal_command(
        prefix="terminal-seal-semantic-fail",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        semantic_settlement_id="terminal-seal-no-pass-settlement",
        semantic_prompt_payload_sha256=request.prompt_payload.payload_sha256,
    )

    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as rejected:
        terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert rejected.value.code == "semantic_settlement_not_pass"


def test_terminal_seal_rejects_authenticated_blocking_gap(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_with_host_artifact(
            tmp_path,
            "terminal-seal-blocking-gap",
            observation_status=PlanningObservationStatus.BLOCKED,
        )
    )
    catalog = _catalog(protected=False)
    semantic_store.freeze_auxiliary_semantic_capability_catalog(
        command=_freeze_command(session_id, turn_id, task_id, details, catalog)
    )
    request = _semantic_request(
        request_id="terminal-seal-blocking-gap-request",
        logical_call_id="terminal-seal-blocking-gap-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    assert request.blocking_gap_aliases
    semantic_store.commit_auxiliary_semantic_verification_request(
        command=_request_command(session_id, turn_id, details, request)
    )
    command = _seal_command(
        prefix="terminal-seal-blocking-gap",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        details=details,
        semantic_settlement_id="terminal-seal-no-gap-settlement",
        semantic_prompt_payload_sha256=request.prompt_payload.payload_sha256,
    )

    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as blocked:
        terminal_store.seal_auxiliary_terminal_proposal(command=command)
    assert blocked.value.code == "blocking_gap"


def test_terminal_seal_rejects_current_hard_budget() -> None:
    command, _settlement = _settled_terminal("terminal-seal-hard-budget")
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=command.session_id,
        insession_task_id=command.task_id,
    )
    assert details is not None and details.budget is not None
    budget = details.budget
    hard_budget = PlanningEpisodeBudget.create(
        budget_ledger_id=budget.budget_ledger_id,
        goal_id=budget.goal_id,
        base_profile=budget.base_profile,
        usage=budget.usage.model_copy(
            update={
                "logical_model_calls": (
                    budget.effective_profile.hard_logical_model_calls
                )
            }
        ),
        extensions=budget.extensions,
        state_version=budget.state_version + 1,
    )
    assert (
        hard_budget.assessment.disposition
        is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
    )
    usage_json = auxiliary_graphs._model_json(hard_budget.usage)
    snapshot_json = auxiliary_graphs._model_json(hard_budget)
    with store._connect() as conn:
        updated = conn.execute(
            "UPDATE insession_auxiliary_goal_budgets SET usage_json=?, "
            "usage_sha256=?, snapshot_json=?, snapshot_sha256=?, "
            "state_version=? WHERE goal_id=? AND state_version=?",
            (
                usage_json,
                auxiliary_graphs._text_hash(usage_json),
                snapshot_json,
                hard_budget.snapshot_sha256,
                hard_budget.state_version,
                command.goal_id,
                budget.state_version,
            ),
        )
        assert updated.rowcount == 1
    hard_command = command.model_copy(
        update={
            "expected_budget_state_version": hard_budget.state_version,
            "expected_budget_snapshot_sha256": hard_budget.snapshot_sha256,
        }
    )

    with pytest.raises(
        terminal_store.AuxiliaryTerminalSealPersistenceError
    ) as exhausted:
        terminal_store.seal_auxiliary_terminal_proposal(command=hard_command)
    assert exhausted.value.code == "budget_hard_limit"
