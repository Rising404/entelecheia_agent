from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.verification.decision import (
    NodeVerificationContext,
    NodeVerificationInputLimits,
    NodeVerificationResult,
    request_node_verification,
)
from personagraph.l2.task_execution.verification.controller import (
    NodeVerificationApplicationRequest,
    NodeVerificationCommit,
    NodeVerificationControllerStateConflict,
    NodeVerificationNextAttemptPlan,
    NodeVerificationResumeRequest,
    NodeVerificationResumeStoreCommand,
    SqliteNodeVerificationApplicationStore,
    resume_and_run_node_verification,
    run_node_verification,
)
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyInputLimits,
)
# 运行时导入先于模型网关，以保留包当前的初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.l2.work_run import (
    AcceptanceUpdate,
    AcceptanceVerificationFeedback,
    AttemptDecision,
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    OutputWindowFormat,
    SubmitOutputWindowAction,
    TaskNodeSubject,
    VerificationVerdict,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


ACCEPTANCES = (
    InSessionTaskAcceptanceProposal(
        acceptance_id="deliverable",
        criterion="提供完整可读的交付文本",
        source_anchor_ids=("request",),
    ),
    InSessionTaskAcceptanceProposal(
        acceptance_id="quality",
        criterion="交付文本满足明确质量要求",
        source_anchor_ids=("request",),
    ),
)


def _input_limits(
    *,
    max_serialized_utf8_bytes: int = 1_000_000,
) -> NodeVerificationInputLimits:
    return NodeVerificationInputLimits(
        profile_id="sqlite-node-verification-v1",
        max_acceptance_items=64,
        max_supporting_tool_result_items=256,
        dependency_delivery_limits=TaskNodeDependencyInputLimits(
            profile_id="sqlite-verification-dependencies-v1",
            max_items=32,
            max_serialized_utf8_bytes=256_000,
        ),
        max_serialized_utf8_bytes=max_serialized_utf8_bytes,
    )


@dataclass(frozen=True)
class _SubmittedNode:
    session_id: str
    turn_id: str
    task_id: str
    work_run_id: str
    submitted_attempt_id: str
    work_run_revision: int
    progress_revision: int
    output_revision: int
    window_revision: int


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _catalog_snapshot() -> dict[str, object]:
    return {"revision": 1, "entries": []}


def _advancing_clock(*, step: float = 1.0):
    current = 0.0

    def read() -> float:
        nonlocal current
        current += step
        return current

    return read


def _seed_submitted_node(*, suffix: str) -> _SubmittedNode:
    session_id = store.create_session("Entelecheia")
    user_text = "执行当前任务节点"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"request-{suffix}",
        source="runtime_test",
        user_text=user_text,
        lease_owner="node-verification-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_id = f"task-{suffix}"
    # 生产图提交会把唯一根节点绑定到任务身份。让 SQLite 垂直夹具保持规范，
    # 还能在验证通过后测试窄范围的单节点任务 FinishGate。
    node_id = task_id
    work_run_id = f"workrun-{suffix}"
    submitted_attempt_id = f"submit-attempt-{suffix}"
    now = "2026-08-14T00:00:00+00:00"
    acceptance_json = json.dumps(
        [item.model_dump(mode="json") for item in ACCEPTANCES],
        ensure_ascii=False,
    )
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
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, '测试任务', '完成测试任务', ?, ?, ?)",
            (task_id, session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, 'proposal-hash', ?, '[\"request\"]', '[\"request\"]', ?)",
            (task_id, turn_id, source_anchors_json, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, '测试节点', '完成节点', "
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

    subject = TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )
    work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=1,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"create-{suffix}",
        work_run_id=work_run_id,
    )
    work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=1,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"start-{suffix}",
        catalog_snapshot=_catalog_snapshot(),
        attempt_id=submitted_attempt_id,
    )
    submitted = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=submitted_attempt_id,
        decision=AttemptDecision(
            acceptance_updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=item.acceptance_id,
                    model_claimed_satisfied=True,
                )
                for item in ACCEPTANCES
            ),
            action=SubmitOutputWindowAction(
                content="# 旅行计划\n2026-09-01 出发，2026-09-08 返程。",
                format=OutputWindowFormat.MARKDOWN,
            ),
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"submit-{suffix}",
    )
    return _SubmittedNode(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        work_run_id=work_run_id,
        submitted_attempt_id=submitted_attempt_id,
        work_run_revision=submitted.work_run_revision,
        progress_revision=submitted.acceptance_progress_revision,
        output_revision=submitted.output_window_revision,
        window_revision=submitted.window_state_version,
    )


def _application_request(
    seeded: _SubmittedNode,
    *,
    suffix: str,
) -> NodeVerificationApplicationRequest:
    return NodeVerificationApplicationRequest(
        session_id=seeded.session_id,
        turn_id=seeded.turn_id,
        work_run_id=seeded.work_run_id,
        submitted_attempt_id=seeded.submitted_attempt_id,
        verification_request_id=f"verification-{suffix}",
        expected_work_run_revision=seeded.work_run_revision,
        expected_progress_revision=seeded.progress_revision,
        expected_output_revision=seeded.output_revision,
        expected_window_revision=seeded.window_revision,
        prepare_apply_id=f"prepare-verification-{suffix}",
        commit_apply_id=f"commit-verification-{suffix}",
        interrupt_apply_id=f"interrupt-verification-{suffix}",
        delivery_id=f"delivery-{suffix}",
        input_limits=_input_limits(),
        next_attempt=NodeVerificationNextAttemptPlan(
            attempt_id=f"attempt-after-verification-{suffix}",
            apply_id=f"start-after-verification-{suffix}",
            catalog_snapshot=_catalog_snapshot(),
        ),
    )


def _reply(*, first_verdict: str = "passed") -> str:
    return json.dumps(
        {
            "acceptance_results": [
                {
                    "acceptance_id": "quality",
                    "verdict": "passed",
                    "finding": "交付正文清晰且可直接使用。",
                    "missing_requirements": [],
                },
                {
                    "acceptance_id": "deliverable",
                    "verdict": first_verdict,
                    "finding": (
                        "交付正文完整。"
                        if first_verdict == "passed"
                        else "正文缺少可执行细节。"
                    ),
                    "missing_requirements": (
                        [] if first_verdict == "passed" else ["补充逐日行程"]
                    ),
                },
            ]
        },
        ensure_ascii=False,
    )


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="sqlite-test-provider",
        model="sqlite-test-model",
        latency_ms=2,
        model_call_id=model_call_id,
    )


def _old_result(context: NodeVerificationContext) -> NodeVerificationResult:
    return NodeVerificationResult(
        verification_request_id=context.verification_request_id,
        verification_request_revision=context.verification_request_revision,
        work_run_id=context.work_run_id,
        locked_work_run_revision=context.locked_work_run_revision,
        submitted_attempt_id=context.submitted_attempt_id,
        acceptance_progress_revision=context.acceptance_progress_revision,
        subject=context.subject,
        output_revision=context.locked_output_window.output_revision,
        acceptance_results=tuple(
            AcceptanceVerificationFeedback(
                acceptance_id=item.acceptance_id,
                verdict=VerificationVerdict.PASSED,
                finding="旧调用认为该条件通过。",
            )
            for item in context.acceptances
        ),
        all_pass=True,
    )


def test_sqlite_pass_creates_resolved_delivery_and_exact_replay_skips_provider():
    seeded = _seed_submitted_node(suffix="pass")
    request = _application_request(seeded, suffix="pass")
    adapter = SqliteNodeVerificationApplicationStore()
    provider_calls: list[str] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        provider_calls.append(str(kwargs["model_call_id"]))
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    result = run_node_verification(
        request,
        store=adapter,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "passed"
    assert result.store_projection.status == "applied"
    assert result.store_projection.delivery_id == request.delivery_id
    assert result.next_attempt_mutation is None
    delivery = verification_store.get_task_node_delivery(
        session_id=seeded.session_id,
        delivery_id=request.delivery_id,
    )
    assert delivery.delivery.verification_request_id == request.verification_request_id
    assert delivery.delivery.created_turn_id == seeded.turn_id
    assert delivery.output_window.content.startswith("# 旅行计划")
    details = task_graph_store.get_insession_task_details(seeded.session_id, seeded.task_id)
    assert details is not None
    assert details.status.value == "completed"
    with store._connect() as conn:
        completed_task_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (seeded.task_id,),
            ).fetchone()[0]
        )

    replay_provider_calls = 0

    def must_not_run(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal replay_provider_calls
        replay_provider_calls += 1
        raise AssertionError("settled verification replay must skip the provider")

    replay = run_node_verification(
        request,
        store=adapter,
        provider=as_prepared_test_provider(must_not_run),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert replay.outcome == "passed"
    assert replay.replayed_without_model is True
    assert replay.store_projection.status == "replayed"
    assert replay.store_projection.delivery_id == request.delivery_id
    assert replay_provider_calls == 0
    assert len(provider_calls) == 1
    with store._connect() as conn:
        assert conn.execute(
            "SELECT state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (seeded.task_id,),
        ).fetchone()[0] == completed_task_version


def test_sqlite_nonpass_starts_next_attempt_with_store_owned_unique_feedback():
    seeded = _seed_submitted_node(suffix="nonpass")
    request = _application_request(seeded, suffix="nonpass")

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
    )

    assert result.outcome == "not_passed"
    assert result.next_attempt_mutation is not None
    assert result.next_attempt_mutation.status == "applied"
    loaded = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert loaded.current_attempt_id == request.next_attempt.attempt_id
    next_attempt = next(
        item
        for item in loaded.attempts
        if item.attempt.attempt_id == request.next_attempt.attempt_id
    )
    assert next_attempt.input_checkpoint_id == request.verification_request_id
    assert (
        next_attempt.input_verification_request_id
        == request.verification_request_id
    )
    assert next_attempt.input_verification_result == result.store_projection.resolved_result
    assert sum(
        item.input_verification_result is not None for item in loaded.attempts
    ) == 1
    record = verification_store.get_task_node_verification_record(
        session_id=seeded.session_id,
        verification_request_id=request.verification_request_id,
    )
    assert record.result == next_attempt.input_verification_result


def test_sqlite_downstream_retry_keeps_candidate_unfrozen_and_starts_same_work_run_attempt():
    seeded = _seed_submitted_node(suffix="downstream-retry")
    request = _application_request(seeded, suffix="downstream-retry")

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    def downstream_gate(_prepared, node_result):
        return (
            DownstreamVerificationFeedback(
                gate_id="whole_task_delivery",
                disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
                finding="根正文遗漏发布步骤。",
                repair_objective="保留正确内容并补齐发布步骤。",
                source_result_id="candidate-review-downstream-retry",
                source_result_sha256="f" * 64,
                affected_subject_ids=(node_result.subject.node_id,),
            ),
        )

    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
        downstream_gate=downstream_gate,
    )

    assert result.outcome == "not_passed"
    assert result.next_attempt_mutation is not None
    loaded = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert loaded.current_attempt_id == request.next_attempt.attempt_id
    next_attempt = next(
        item
        for item in loaded.attempts
        if item.attempt.attempt_id == request.next_attempt.attempt_id
    )
    assert next_attempt.input_verification_result is not None
    assert next_attempt.input_verification_result.acceptance_results[0].verdict is (
        VerificationVerdict.PASSED
    )
    assert next_attempt.input_verification_result.downstream_results[0].gate_id == (
        "whole_task_delivery"
    )
    with store._connect() as conn:
        frozen_at = conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows "
            "WHERE work_run_id=?",
            (seeded.work_run_id,),
        ).fetchone()[0]
        deliveries = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_task_node_deliveries "
                "WHERE work_run_id=?",
                (seeded.work_run_id,),
            ).fetchone()[0]
        )
    assert frozen_at is None
    assert deliveries == 0
    details = task_graph_store.get_insession_task_details(seeded.session_id, seeded.task_id)
    assert details is not None and details.status.value == "active"
    with store._connect() as conn:
        receipt = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id=?",
                (request.commit_apply_id,),
            ).fetchone()[0]
        )
    assert "acceptance_results" not in receipt
    assert "catalog_snapshot" not in receipt


def test_sqlite_nonpass_commit_start_gap_is_completed_without_second_model_call():
    seeded = _seed_submitted_node(suffix="gap")
    request = _application_request(seeded, suffix="gap")

    class CrashBeforeStart(SqliteNodeVerificationApplicationStore):
        def start_next_attempt(self, command):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated process loss before next Attempt")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        run_node_verification(
            request,
            store=CrashBeforeStart(),
            provider=as_prepared_test_provider(
                lambda _system, _user, **kwargs: _model_result(
                    _reply(first_verdict="insufficient_evidence"),
                    str(kwargs["model_call_id"]),
                )
            ),
            emit=lambda _event: None,
            monotonic_clock=_advancing_clock(),
        )

    record = verification_store.get_task_node_verification_record(
        session_id=seeded.session_id,
        verification_request_id=request.verification_request_id,
    )
    assert record.result is not None and record.result.all_pass is False
    before_recovery = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert before_recovery.current_attempt_id is None
    provider_calls = 0

    def must_not_run(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("completed request recovery must skip the provider")

    recovered = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(must_not_run),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert recovered.outcome == "not_passed"
    assert recovered.replayed_without_model is True
    assert recovered.next_attempt_mutation is not None
    assert recovered.next_attempt_mutation.status == "applied"
    assert provider_calls == 0
    loaded = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert loaded.current_attempt_id == request.next_attempt.attempt_id
    recovered_attempt = next(
        item
        for item in loaded.attempts
        if item.attempt.attempt_id == request.next_attempt.attempt_id
    )
    assert recovered_attempt.input_checkpoint_id == request.verification_request_id
    assert recovered_attempt.input_verification_request_id == (
        request.verification_request_id
    )
    assert recovered_attempt.input_verification_result == record.result


def test_sqlite_technical_failure_requires_explicit_new_turn_resume_and_fences_old_result():
    seeded = _seed_submitted_node(suffix="resume")
    request = _application_request(seeded, suffix="resume")
    adapter = SqliteNodeVerificationApplicationStore()
    first_payload: str | None = None
    old_context: NodeVerificationContext | None = None

    def failing_provider(_system: str, user: str, **_kwargs: object) -> ModelResult:
        nonlocal first_payload
        first_payload = user
        raise ModelGatewayError(
            "MODEL_CONFIGURATION_ERROR",
            "provider unavailable",
            retryable=False,
        )

    def capture_then_verify(context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal old_context
        old_context = context
        return request_node_verification(context, **kwargs)

    interrupted = run_node_verification(
        request,
        store=adapter,
        provider=as_prepared_test_provider(failing_provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
        verifier=capture_then_verify,
    )

    assert interrupted.outcome == "interrupted"
    assert interrupted.interruption_reason == "verification_unavailable"
    assert old_context is not None
    assert first_payload is not None
    blocked_provider_calls = 0

    def blocked_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal blocked_provider_calls
        blocked_provider_calls += 1
        raise AssertionError("interrupted requests require explicit resume")

    with pytest.raises(
        NodeVerificationControllerStateConflict,
        match="explicit resume",
    ):
        run_node_verification(
            request,
            store=adapter,
            provider=as_prepared_test_provider(blocked_provider),
            emit=lambda _event: None,
            monotonic_clock=_advancing_clock(),
        )
    assert blocked_provider_calls == 0

    marked = store.mark_turn_execution_interrupted(
        session_id=seeded.session_id,
        turn_id=seeded.turn_id,
        expected_window_revision=interrupted.store_projection.window_revision,
        stage="VERIFICATION",
        interruption_reason=interrupted.interruption_reason,
    )
    store.settle_interrupted_turn_execution(
        session_id=seeded.session_id,
        turn_id=seeded.turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason=interrupted.interruption_reason,
        error_code="MODEL_CONFIGURATION_FAILURE",
    )
    accepted = store.accept_turn_execution(
        session_id=seeded.session_id,
        client_request_id="resume-node-verification",
        source="runtime_test",
        user_text="继续",
        lease_owner="node-verification-controller-test",
    )
    resumed_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                seeded.session_id,
                resumed_turn_id,
                seeded.task_id,
                "2026-08-14T00:30:00+00:00",
            ),
        )
    resume_request = NodeVerificationResumeRequest(
        session_id=seeded.session_id,
        turn_id=resumed_turn_id,
        work_run_id=seeded.work_run_id,
        submitted_attempt_id=seeded.submitted_attempt_id,
        verification_request_id=request.verification_request_id,
        expected_work_run_revision=interrupted.store_projection.work_run_revision,
        expected_verification_request_revision=(
            interrupted.store_projection.verification_request_revision
        ),
        expected_window_revision=_window_revision(seeded.session_id),
        resume_apply_id="resume-verification-resume",
        commit_apply_id="commit-verification-after-resume",
        interrupt_apply_id="interrupt-verification-after-resume",
        delivery_id=request.delivery_id,
        input_limits=request.input_limits,
        next_attempt=request.next_attempt,
    )
    resumed_preparation = adapter.resume_node_verification(
        NodeVerificationResumeStoreCommand(
            session_id=resume_request.session_id,
            turn_id=resume_request.turn_id,
            work_run_id=resume_request.work_run_id,
            submitted_attempt_id=resume_request.submitted_attempt_id,
            verification_request_id=resume_request.verification_request_id,
            expected_work_run_revision=resume_request.expected_work_run_revision,
            expected_verification_request_revision=(
                resume_request.expected_verification_request_revision
            ),
            expected_window_revision=resume_request.expected_window_revision,
            apply_id=resume_request.resume_apply_id,
            commit_settlement_apply_id=resume_request.commit_apply_id,
            interrupt_settlement_apply_id=resume_request.interrupt_apply_id,
            delivery_id=resume_request.delivery_id,
            input_limits=resume_request.input_limits,
        )
    )
    assert resumed_preparation.status == "ready"
    before_late = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert old_context is not None
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="verification request revision conflict",
    ):
        adapter.commit_node_verification(
            NodeVerificationCommit(
                session_id=seeded.session_id,
                turn_id=resumed_turn_id,
                work_run_id=seeded.work_run_id,
                verification_request_id=request.verification_request_id,
                expected_work_run_revision=(
                    resumed_preparation.context.work_run.revision
                ),
                expected_verification_request_revision=(
                    old_context.verification_request_revision
                ),
                expected_window_revision=resumed_preparation.window_revision,
                verification_result=_old_result(old_context),
                active_seconds_delta=1,
                apply_id="late-old-verification-result",
                delivery_id="late-delivery-must-not-exist",
            )
        )
    after_late = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert after_late == before_late
    resumed_payload: str | None = None

    def passing_provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal resumed_payload
        resumed_payload = user
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    resumed = resume_and_run_node_verification(
        resume_request,
        store=adapter,
        provider=as_prepared_test_provider(passing_provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert resumed.outcome == "passed"
    assert resumed.store_projection.verification_request_id == (
        request.verification_request_id
    )
    assert resumed_payload == first_payload
    record = verification_store.get_task_node_verification_record(
        session_id=seeded.session_id,
        verification_request_id=request.verification_request_id,
    )
    assert record.request.request_turn_id == seeded.turn_id
    assert record.request.revision == 4
    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        verification_store.get_task_node_delivery(
            session_id=seeded.session_id,
            delivery_id="late-delivery-must-not-exist",
        )

    replay_provider_calls = 0

    def replay_must_not_run(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal replay_provider_calls
        replay_provider_calls += 1
        raise AssertionError("settled resumed request must skip the provider")

    replay = resume_and_run_node_verification(
        resume_request,
        store=adapter,
        provider=as_prepared_test_provider(replay_must_not_run),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )
    assert replay.outcome == "passed"
    assert replay.replayed_without_model is True
    assert replay.store_projection.status == "replayed"
    assert replay_provider_calls == 0


def test_sqlite_over_limit_input_interrupts_without_provider_or_verification_result():
    seeded = _seed_submitted_node(suffix="input-too-large")
    request = _application_request(seeded, suffix="input-too-large").model_copy(
        update={"input_limits": _input_limits(max_serialized_utf8_bytes=1)}
    )
    provider_calls = 0

    def must_not_run(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit verifier input must not call provider")

    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(must_not_run),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "interrupted"
    assert result.interruption_reason == "verification_input_too_large"
    assert result.model_call is None
    assert result.store_projection.resolved_result is None
    assert provider_calls == 0
    record = verification_store.get_task_node_verification_record(
        session_id=seeded.session_id,
        verification_request_id=request.verification_request_id,
    )
    assert record.request.status.value == "interrupted"
    assert record.request.technical_error_code == "verification_input_too_large"
    assert record.result is None
    loaded = work_run_store.get_work_run(
        session_id=seeded.session_id,
        work_run_id=seeded.work_run_id,
    )
    assert loaded is not None
    assert loaded.work_run.status.value == "interrupted"


def test_sqlite_exhausted_invalid_structured_output_interrupts_with_exact_reason():
    seeded = _seed_submitted_node(suffix="invalid")
    request = _application_request(seeded, suffix="invalid")
    model_call_ids: list[str] = []

    def malformed_provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        model_call_ids.append(str(kwargs["model_call_id"]))
        return _model_result("not-json", str(kwargs["model_call_id"]))

    result = run_node_verification(
        request,
        store=SqliteNodeVerificationApplicationStore(),
        provider=as_prepared_test_provider(malformed_provider),
        emit=lambda _event: None,
        monotonic_clock=_advancing_clock(),
    )

    assert result.outcome == "interrupted"
    assert result.interruption_reason == "verification_output_invalid"
    assert len(model_call_ids) == 6
    assert len(set(model_call_ids)) == 1
    record = verification_store.get_task_node_verification_record(
        session_id=seeded.session_id,
        verification_request_id=request.verification_request_id,
    )
    assert record.request.status.value == "interrupted"
    assert record.request.technical_error_code == "verification_output_invalid"
    assert record.result is None
