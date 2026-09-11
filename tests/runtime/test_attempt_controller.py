from __future__ import annotations

import json

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.attempts.controller import (
    AttemptActiveTimeMeter,
    AttemptControllerStateConflict,
    AttemptToolBridgePreflightRequest,
    AttemptToolBridgeRequest,
    project_authoritative_attempt_user_input,
    run_started_attempt as _run_started_attempt,
)
from personagraph.l2.task_execution.attempts.decision import (
    AcceptanceVerificationFeedback,
    AttemptDecisionContext,
    AttemptDecisionInputLimits,
    AttemptVerificationFeedback,
    PriorToolResultProjection,
    PriorToolResultsProjection,
    VerificationVerdict,
)
# 先导入运行时，以维持模型网关当前的模块初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
)
from personagraph.session import store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.tools.contracts import ToolSpec
from personagraph.l2.work_run import (
    AttemptDecision,
    AcceptanceUpdate,
    AttemptStatus,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
    NodeVerificationResult,
    OutputWindowFormat,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    WriteOutputWindowAction,
    WorkRunStatus,
)
from tests.helpers.task_node_source_context import task_node_source_context
from tests.helpers.evidence_submission import empty_support_justification
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


def _active_time_meter(delta: float = 1.0) -> AttemptActiveTimeMeter:
    readings = iter((0.0, delta))
    return AttemptActiveTimeMeter.starting_now(lambda: next(readings))


def run_started_attempt(*args, active_time_meter=None, **kwargs):
    provider = kwargs.get("provider")
    if not callable(provider):
        raise TypeError("provider must be callable in Attempt controller tests")
    kwargs["provider"] = as_prepared_test_provider(provider)
    return _run_started_attempt(
        *args,
        active_time_meter=active_time_meter or _active_time_meter(),
        **kwargs,
    )


def test_active_time_meter_freezes_one_positive_delta_without_rereading_clock():
    readings = iter((10.0, 15.5, 999.0))
    meter = AttemptActiveTimeMeter.starting_now(lambda: next(readings))

    assert meter.freeze() == 5.5
    assert meter.freeze() == 5.5
    assert next(readings) == 999.0


@pytest.mark.parametrize("stopped_at", [10.0, 9.0, float("nan")])
def test_active_time_meter_rejects_nonpositive_or_nonfinite_delta(
    stopped_at: float,
) -> None:
    readings = iter((10.0, stopped_at))
    meter = AttemptActiveTimeMeter.starting_now(lambda: next(readings))

    with pytest.raises(RuntimeError, match="finite and positive|timestamp must be finite"):
        meter.freeze()


def _tool_spec() -> ToolSpec:
    return ToolSpec(
        tool_id="read_test",
        contract_version="1.0.0",
        name="Read test",
        description="Read deterministic test input.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        catalog_tags=("read",),
    )



def _catalog_snapshot(
    *,
    include_tool: bool,
    tool_spec: ToolSpec | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    if include_tool:
        spec = tool_spec or _tool_spec()
        entries.append(
            {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "status": "active",
                "created_revision": 1,
                "updated_revision": 1,
                "registration": {
                    "spec": spec.to_dict(),
                    "implementation_version": "implementation-1",
                    "source": {"kind": "local", "source_id": "test"},
                    "effect_count": 1,
                    "execution": {},
                },
            }
        )
    return {"revision": 1, "entries": entries}


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="test-provider",
        model="test-model",
        latency_ms=2,
        model_call_id=model_call_id,
    )


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_started_attempt(
    *,
    work_run_id: str = "workrun-controller",
    attempt_id: str = "attempt-controller-1",
    include_tool: bool = False,
    tool_spec: ToolSpec | None = None,
) -> tuple[str, str, TaskNodeSubject]:
    session_id = store.create_session("Entelecheia")
    user_text = "执行当前任务节点"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"request-{work_run_id}",
        source="runtime_test",
        user_text=user_text,
        lease_owner="attempt-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_id = f"task-{work_run_id}"
    node_id = f"node-{work_run_id}"
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
        apply_id=f"create-{work_run_id}",
        work_run_id=work_run_id,
    )
    work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=1,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"start-{attempt_id}",
        catalog_snapshot=_catalog_snapshot(
            include_tool=include_tool,
            tool_spec=tool_spec,
        ),
        attempt_id=attempt_id,
    )
    return session_id, turn_id, subject


def _context(
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    *,
    work_run_id: str = "workrun-controller",
    prior_tool_results: PriorToolResultsProjection | None = None,
    include_tool: bool = False,
    tool_spec: ToolSpec | None = None,
) -> AttemptDecisionContext:
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=work_run_id)
    assert loaded.current_attempt_id is not None
    current = next(
        item
        for item in loaded.attempts
        if item.attempt.attempt_id == loaded.current_attempt_id
    )
    return AttemptDecisionContext(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        work_run_revision=loaded.work_run.revision,
        attempt_id=current.attempt.attempt_id,
        attempt_ordinal=current.attempt.ordinal,
        user_input=project_authoritative_attempt_user_input(
            stored=loaded,
            current_attempt=current,
        ),
        subject=subject,
        node_title="测试节点",
        node_objective="完成节点",
        acceptances=ACCEPTANCES,
        acceptance_progress=loaded.acceptance_progress,
        output_window=loaded.output_window,
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        source_context=task_node_source_context(
            session_id=session_id,
            subject=subject,
            acceptances=ACCEPTANCES,
        ),
        prior_tool_results=prior_tool_results or PriorToolResultsProjection(),
        input_limits=AttemptDecisionInputLimits(
            profile_id="attempt-controller-test-v1",
            max_prior_tool_result_items=32,
            max_prior_tool_results_serialized_utf8_bytes=128_000,
            dependency_delivery_limits=TaskNodeDependencyInputLimits(
                profile_id="attempt-controller-test-dependencies-v1",
                max_items=32,
                max_serialized_utf8_bytes=256_000,
            ),
            max_serialized_utf8_bytes=1_000_000,
        ),
        allowed_tools=(
            (tool_spec or _tool_spec(),)
            if include_tool
            else ()
        ),
    )


def _provider(reply: dict[str, object], calls: list[str] | None = None):
    def complete(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        if calls is not None:
            calls.append(model_call_id)
        return _model_result(json.dumps(reply, ensure_ascii=False), model_call_id)

    return complete


def _verification_result(
    context: AttemptDecisionContext,
    *,
    request_id: str = "verification-authority",
    all_pass: bool = False,
) -> NodeVerificationResult:
    verdict = (
        VerificationVerdict.PASSED
        if all_pass
        else VerificationVerdict.NOT_SATISFIED
    )
    return NodeVerificationResult(
        verification_request_id=request_id,
        verification_request_revision=1,
        work_run_id=context.work_run_id,
        locked_work_run_revision=context.work_run_revision - 1,
        submitted_attempt_id="prior-submit-attempt",
        acceptance_progress_revision=context.acceptance_progress.revision,
        subject=context.subject,
        output_revision=context.output_window.output_revision,
        acceptance_results=tuple(
            AcceptanceVerificationFeedback(
                acceptance_id=item.acceptance_id,
                verdict=verdict,
                finding=(
                    "验证门确认已满足。"
                    if all_pass
                    else "验证门确认仍需修正。"
                ),
                missing_requirements=(
                    () if all_pass else (f"补足 {item.acceptance_id}",)
                ),
            )
            for item in context.acceptances
        ),
        all_pass=all_pass,
    )


def _project_verification_binding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored,
    request_id: str | None,
    result: NodeVerificationResult | None,
    ordinal: int = 2,
    progress_revision_delta: int = 0,
):
    current_id = stored.current_attempt_id
    assert current_id is not None
    projected_attempts = []
    for item in stored.attempts:
        if item.attempt.attempt_id != current_id:
            projected_attempts.append(item)
            continue
        projected_attempts.append(
            item.model_copy(
                update={
                    "attempt": item.attempt.model_copy(update={"ordinal": ordinal}),
                    "input_verification_request_id": request_id,
                    "input_verification_result": result,
                }
            )
        )
    projected = stored.model_copy(
        update={
            "attempts": tuple(projected_attempts),
            "acceptance_progress": stored.acceptance_progress.model_copy(
                update={
                    "revision": (
                        stored.acceptance_progress.revision
                        + progress_revision_delta
                    )
                }
            ),
        }
    )
    monkeypatch.setattr(work_run_store, "get_work_run", lambda **_kwargs: projected)
    return projected


def test_controller_whole_replaces_output_through_real_atomic_store():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)

    result = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-write-output",
        provider=_provider(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "write_output_window",
                    "content": "# 工作稿\n\n第一版内容。",
                    "format": "markdown",
                },
            }
        ),
        emit=lambda _event: None,
    )

    assert result.action == "write_output_window"
    assert result.apply_id == "controller-write-output"
    assert result.model_call.provider == "test-provider"
    assert result.model_call.physical_attempts == 1
    assert result.mutation.output_window_revision == 2
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.output_window.content == "# 工作稿\n\n第一版内容。"
    assert loaded.output_window.output_revision == 2
    assert loaded.acceptance_progress.evaluated_output_revision == 2
    assert loaded.current_attempt_id is None
    assert loaded.attempts[-1].attempt.status is AttemptStatus.CLOSED
    assert loaded.attempts[-1].action == "write_output_window"
    assert not hasattr(loaded.attempts[-1].decision.action, "content")


def test_controller_submits_exact_output_revision_through_real_atomic_store():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    updates = [
        {
            "acceptance_id": item.acceptance_id,
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
            "empty_support_justification": empty_support_justification(),
        }
        for item in ACCEPTANCES
    ]

    result = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-submit-output",
        provider=_provider(
            {
                "acceptance_updates": updates,
                "action": {
                    "kind": "submit_output_window",
                    "content": "最终交付文本",
                    "format": "plain_text",
                },
            }
        ),
        emit=lambda _event: None,
    )

    assert result.action == "submit_output_window"
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.work_run.status is WorkRunStatus.ACTIVE
    assert loaded.work_run.reason == "verification_pending"
    assert loaded.output_window.content == "最终交付文本"
    assert all(
        item.model_claimed_satisfied
        for item in loaded.acceptance_progress.items
    )
    assert loaded.attempts[-1].attempt.submitted_output_revision == 2
    assert loaded.attempts[-1].committed_output_revision == 2



def test_controller_materializes_request_user_and_commits_real_store():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)

    result = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-request-user",
        provider=_provider(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请提供必须采用的交付格式。",
                },
            }
        ),
        emit=lambda _event: None,
    )

    assert result.action == "request_user_input"
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.work_run.status is WorkRunStatus.WAITING_USER
    assert loaded.work_run.reason == "needs_input"
    assert loaded.pending_user_question == "请提供必须采用的交付格式。"
    assert loaded.output_window.output_revision == 1
    assert loaded.attempts[-1].attempt.status is AttemptStatus.CLOSED


def test_waiting_continuation_projects_exact_question_and_answer_to_model():
    session_id, question_turn_id, subject = _seed_started_attempt()
    first_context = _context(session_id, question_turn_id, subject)
    first = run_started_attempt(
        first_context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-first-question",
        provider=_provider(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "请提供旅行日期。",
                },
            }
        ),
        emit=lambda _event: None,
    )
    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert len(pending) == 1
    question = pending[0]

    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=first.mutation.window_state_version,
        processing_level="L2",
        assistant_content=question.question,
        post_commit_job_kinds=(),
    )
    finalized_window = finalized["window"]
    assert isinstance(finalized_window, dict)
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized_window["state_version"]),
    )

    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="controller-answer-turn",
        source="runtime_test",
        user_text="九月十日出发，九月十五日返程。",
        lease_owner="attempt-controller-test",
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
    continued = continuation_store.continue_waiting_user_work_run_and_start_attempt(
        session_id=session_id,
        turn_id=answer_turn_id,
        work_run_id=first_context.work_run_id,
        subject=subject,
        question_attempt_id=question.question_attempt_id,
        expected_work_run_revision=question.work_run_revision,
        expected_task_state_version=question.task_state_version,
        expected_node_state_version=question.node_state_version,
        expected_progress_revision=question.acceptance_progress_revision,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-continue-question",
        catalog_snapshot=_catalog_snapshot(include_tool=False),
        attempt_id="controller-answer-attempt",
    )
    context = _context(session_id, answer_turn_id, subject)
    received_payloads: list[dict[str, object]] = []

    def capture_provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        received_payloads.append(json.loads(user_content))
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请确认年份。",
                    },
                },
                ensure_ascii=False,
            ),
            model_call_id,
        )

    run_started_attempt(
        context,
        expected_window_revision=continued.window_state_version,
        apply_id="controller-answer-question",
        provider=capture_provider,
        emit=lambda _event: None,
    )

    assert len(received_payloads) == 1
    assert received_payloads[0]["user_input"] == {
        "content": "九月十日出发，九月十五日返程。",
        "prior_waiting_user_question": "请提供旅行日期。",
    }


def test_stale_context_is_rejected_before_provider_call():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    stale = context.model_copy(update={"work_run_revision": 1})
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match="work_run_revision"):
        run_started_attempt(
            stale,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-stale",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.current_attempt_id == context.attempt_id
    assert loaded.attempts[-1].attempt.status is AttemptStatus.ACTIVE


def test_tampered_user_input_is_rejected_before_provider_call():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    tampered = context.model_copy(
        update={
            "user_input": context.user_input.model_copy(
                update={"content": "调用方伪造的用户指令"}
            )
        }
    )
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match="user_input_authority"):
        run_started_attempt(
            tampered,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-tampered-user-input",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


@pytest.mark.parametrize(
    ("binding", "mutate"),
    (
        ("node_title", lambda context: context.model_copy(update={"node_title": "伪造标题"})),
        (
            "node_objective",
            lambda context: context.model_copy(update={"node_objective": "伪造目标"}),
        ),
        (
            "node_acceptances",
            lambda context: context.model_copy(
                update={
                    "acceptances": (
                        context.acceptances[0].model_copy(
                            update={"criterion": "被调用方篡改的验收条件"}
                        ),
                        *context.acceptances[1:],
                    )
                }
            ),
        ),
    ),
)
def test_tampered_node_semantics_are_rejected_before_provider_or_write(
    binding: str,
    mutate,
):
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    tampered = mutate(context)
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match=binding):
        run_started_attempt(
            tampered,
            expected_window_revision=_window_revision(session_id),
            apply_id=f"controller-tampered-{binding}",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.work_run.revision == context.work_run_revision
    assert loaded.current_attempt_id == context.attempt_id
    assert loaded.attempts[-1].decision is None


def test_unverifiable_verification_feedback_is_rejected_before_provider_call():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    feedback = AttemptVerificationFeedback(
        submitted_output_revision=context.output_window.output_revision,
        acceptance_results=tuple(
            AcceptanceVerificationFeedback(
                acceptance_id=item.acceptance_id,
                verdict=VerificationVerdict.NOT_SATISFIED,
                finding="调用方提供、但当前 Store 无法证明的反馈。",
            )
            for item in ACCEPTANCES
        ),
    )
    unproven = context.model_copy(update={"verification_feedback": feedback})
    provider_calls: list[str] = []

    with pytest.raises(
        AttemptControllerStateConflict,
        match="verification_feedback_authority",
    ):
        run_started_attempt(
            unproven,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-unproven-feedback",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


def test_store_joined_nonpass_feedback_is_exactly_projected_into_attempt_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    session_id, turn_id, subject = _seed_started_attempt()
    original_context = _context(session_id, turn_id, subject)
    result = _verification_result(original_context)
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=original_context.work_run_id,
    )
    _project_verification_binding(
        monkeypatch,
        stored=stored,
        request_id=result.verification_request_id,
        result=result,
    )
    context = _context(session_id, turn_id, subject).model_copy(
        update={
            "verification_feedback": AttemptVerificationFeedback(
                submitted_output_revision=result.output_revision,
                acceptance_results=result.acceptance_results,
            )
        }
    )
    provider_payloads: list[dict[str, object]] = []

    def provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        provider_payloads.append(json.loads(user_content))
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充缺失要求。",
                    },
                },
                ensure_ascii=False,
            ),
            model_call_id,
        )

    outcome = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-authoritative-feedback",
        provider=provider,
        emit=lambda _event: None,
    )

    assert outcome.action == "request_user_input"
    assert provider_payloads[0]["verification_feedback"] == (
        context.verification_feedback.model_dump(mode="json")
    )


def test_nonpass_feedback_survives_progress_change_when_output_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    session_id, turn_id, subject = _seed_started_attempt()
    original_context = _context(session_id, turn_id, subject)
    result = _verification_result(original_context)
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=original_context.work_run_id,
    )
    _project_verification_binding(
        monkeypatch,
        stored=stored,
        request_id=result.verification_request_id,
        result=result,
        ordinal=3,
        progress_revision_delta=1,
    )
    context = _context(session_id, turn_id, subject).model_copy(
        update={
            "verification_feedback": AttemptVerificationFeedback(
                submitted_output_revision=result.output_revision,
                acceptance_results=result.acceptance_results,
            )
        }
    )
    provider_calls = 0

    def unavailable(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        assert purpose == "runtime_work_run_attempt_decision"
        raise ModelGatewayError(
            "MODEL_CONFIGURATION_ERROR",
            "stop after proving the guarded prompt is callable",
            retryable=False,
        )

    with pytest.raises(ModelGatewayError, match="proving the guarded prompt"):
        run_started_attempt(
            context,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-feedback-after-progress",
            provider=unavailable,
            emit=lambda _event: None,
        )

    assert context.acceptance_progress.revision == (
        result.acceptance_progress_revision + 1
    )
    assert context.output_window.output_revision == result.output_revision
    assert provider_calls == 1


@pytest.mark.parametrize(
    ("binding", "request_id", "all_pass", "tamper_context", "ordinal"),
    [
        (
            "attempt_verification_binding",
            "different-request-id",
            False,
            False,
            2,
        ),
        ("attempt_verification_binding", "verification-authority", True, False, 2),
        (
            "verification_feedback_authority",
            "verification-authority",
            False,
            True,
            2,
        ),
        (
            "first_attempt_verification_feedback",
            "verification-authority",
            False,
            False,
            1,
        ),
    ],
    ids=("request-id", "pass-result", "caller-tamper", "first-attempt"),
)
def test_invalid_store_joined_verification_feedback_fails_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    request_id: str,
    all_pass: bool,
    tamper_context: bool,
    ordinal: int,
):
    session_id, turn_id, subject = _seed_started_attempt()
    original_context = _context(session_id, turn_id, subject)
    result = _verification_result(original_context, all_pass=all_pass)
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=original_context.work_run_id,
    )
    _project_verification_binding(
        monkeypatch,
        stored=stored,
        request_id=request_id,
        result=result,
        ordinal=ordinal,
    )
    feedback = AttemptVerificationFeedback(
        submitted_output_revision=result.output_revision,
        acceptance_results=result.acceptance_results,
    )
    if tamper_context:
        feedback = feedback.model_copy(
            update={
                "acceptance_results": (
                    feedback.acceptance_results[0].model_copy(
                        update={"finding": "调用方篡改的验证结论。"}
                    ),
                    *feedback.acceptance_results[1:],
                )
            }
        )
    context = _context(session_id, turn_id, subject).model_copy(
        update={"verification_feedback": feedback}
    )
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match=binding):
        run_started_attempt(
            context,
            expected_window_revision=_window_revision(session_id),
            apply_id=f"controller-invalid-feedback-{binding}",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


def test_tampered_allowed_tool_spec_is_rejected_before_provider_call():
    session_id, turn_id, subject = _seed_started_attempt(include_tool=True)
    context = _context(session_id, turn_id, subject, include_tool=True)
    original = context.allowed_tools[0]
    tampered_tool = ToolSpec(
        tool_id=original.tool_id,
        contract_version=original.contract_version,
        name=original.name,
        description="伪造后会误导模型的工具描述。",
        input_schema=original.input_schema,
        output_schema=original.output_schema,
        catalog_tags=original.catalog_tags,
    )
    tampered = context.model_copy(update={"allowed_tools": (tampered_tool,)})
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match="allowed_tools"):
        run_started_attempt(
            tampered,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-tampered-tools",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


def test_unknown_prior_tool_result_is_rejected_before_provider_call():
    session_id, turn_id, subject = _seed_started_attempt()
    forged = PriorToolResultsProjection(
        items=(
            PriorToolResultProjection(
                tool_id="read_test",
                tool_version="1.0.0",
                result=ToolResult(
                    status=ToolResultStatus.SUCCEEDED,
                    tool_result_id="unknown-result",
                    tool_call_id="unknown-call",
                    attempt_id="older-attempt",
                    ordinal=1,
                    output={"value": "not stored"},
                ),
            ),
        ),
        truncated=True,
    )
    context = _context(
        session_id,
        turn_id,
        subject,
        prior_tool_results=forged,
    )
    provider_calls: list[str] = []

    with pytest.raises(AttemptControllerStateConflict, match="prior_tool_results"):
        run_started_attempt(
            context,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-forged-history",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


def test_progress_support_must_remain_visible_in_truncated_prior_projection():
    session_id, turn_id, subject = _seed_started_attempt(include_tool=True)
    work_run_id = "workrun-controller"
    first_attempt_id = "attempt-controller-1"
    call_id = "historical-call"
    result_id = "historical-result"

    mutation = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=first_attempt_id,
        decision=HostAcceptedAttemptDecision(
            action=HostMaterializedCallToolsAction(
                calls=(
                    HostMaterializedToolCall(
                        tool_call_id=call_id,
                        tool_id="read_test",
                        tool_version="implementation-1",
                        arguments={"value": "x"},
                        modifies_environment=False,
                    ),
                )
            )
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="historical-decision",
    )
    mutation = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id=result_id,
            tool_call_id=call_id,
            attempt_id=first_attempt_id,
            ordinal=1,
            output={"value": "evidence"},
        ),
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id="historical-result-append",
    )
    mutation = work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=first_attempt_id,
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id="historical-close",
    )
    mutation = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id="supporting-start-two",
        catalog_snapshot=_catalog_snapshot(include_tool=True),
        attempt_id="attempt-controller-2",
    )
    mutation = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id="attempt-controller-2",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=True,
                    supporting_tool_result_ids=(result_id,),
                ),
            ),
            action=WriteOutputWindowAction(
                content="",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_output_revision=mutation.output_window_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id="supporting-progress-write",
    )
    work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=mutation.work_run_revision,
        expected_progress_revision=mutation.acceptance_progress_revision,
        expected_window_revision=mutation.window_state_version,
        apply_id="supporting-start-three",
        catalog_snapshot=_catalog_snapshot(include_tool=True),
        attempt_id="attempt-controller-3",
    )
    context = _context(
        session_id,
        turn_id,
        subject,
        include_tool=True,
        prior_tool_results=PriorToolResultsProjection(truncated=True),
    )
    provider_calls: list[str] = []

    with pytest.raises(
        AttemptControllerStateConflict,
        match="supporting_tool_result_projection",
    ):
        run_started_attempt(
            context,
            expected_window_revision=_window_revision(session_id),
            apply_id="controller-hidden-support",
            provider=_provider(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "不会被调用",
                    },
                },
                provider_calls,
            ),
            emit=lambda _event: None,
        )

    assert provider_calls == []


def test_malformed_model_exhaustion_leaves_started_attempt_active():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    revision_before = _window_revision(session_id)
    provider_calls: list[str] = []

    def malformed(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        provider_calls.append(model_call_id)
        return _model_result("not-json", model_call_id)

    with pytest.raises(ModelGatewayError) as exc_info:
        run_started_attempt(
            context,
            expected_window_revision=revision_before,
            apply_id="controller-malformed",
            provider=malformed,
            emit=lambda _event: None,
        )

    assert exc_info.value.code == "MODEL_BAD_RESPONSE"
    assert len(provider_calls) == 6
    assert len(set(provider_calls)) == 1
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.work_run.revision == context.work_run_revision
    assert loaded.acceptance_progress == context.acceptance_progress
    assert loaded.output_window == context.output_window
    assert loaded.current_attempt_id == context.attempt_id
    assert loaded.attempts[-1].attempt.status is AttemptStatus.ACTIVE
    assert loaded.attempts[-1].decision is None
    assert _window_revision(session_id) == revision_before


def test_store_cas_rejects_race_after_provider_without_partial_attempt_commit():
    session_id, turn_id, subject = _seed_started_attempt()
    context = _context(session_id, turn_id, subject)
    expected_window_revision = _window_revision(session_id)

    def racing_provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        with store._connect() as conn:
            conn.execute(
                "UPDATE turn_execution_windows SET state_version=state_version+1 "
                "WHERE session_id=?",
                (session_id,),
            )
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "write_output_window",
                        "content": "竞争写入不会提交",
                        "format": "plain_text",
                    },
                },
                ensure_ascii=False,
            ),
            model_call_id,
        )

    with pytest.raises(store.TurnExecutionWindowRevisionConflict):
        run_started_attempt(
            context,
            expected_window_revision=expected_window_revision,
            apply_id="controller-window-race",
            provider=racing_provider,
            emit=lambda _event: None,
        )

    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.work_run.revision == context.work_run_revision
    assert loaded.current_attempt_id == context.attempt_id
    assert loaded.output_window == context.output_window
    assert loaded.attempts[-1].attempt.status is AttemptStatus.ACTIVE


def test_call_tools_is_delegated_once_and_never_uses_output_store(monkeypatch):
    session_id, turn_id, subject = _seed_started_attempt(include_tool=True)
    context = _context(session_id, turn_id, subject, include_tool=True)
    preflight_requests: list[AttemptToolBridgePreflightRequest] = []
    bridge_requests: list[AttemptToolBridgeRequest] = []

    def reject_output_path(**_kwargs: object):
        raise AssertionError("call_tools must not enter the OutputWindow Store path")

    monkeypatch.setattr(
        work_run_store,
        "commit_work_run_output_action",
        reject_output_path,
    )

    class Bridge:
        def preflight(self, request: AttemptToolBridgePreflightRequest):
            preflight_requests.append(request)
            proposal = request.decision.action.calls[0]
            return HostAcceptedAttemptDecision(
                acceptance_updates=request.decision.acceptance_updates,
                action=HostMaterializedCallToolsAction(
                    calls=(
                        HostMaterializedToolCall(
                            tool_call_id=request.tool_call_ids[0],
                            tool_id=proposal.tool_id,
                            tool_version="1.0.0",
                            arguments=proposal.arguments,
                            modifies_environment=False,
                        ),
                    )
                ),
            )

        def dispatch(
            self,
            request: AttemptToolBridgeRequest,
            *,
            active_time_meter: AttemptActiveTimeMeter,
        ):
            bridge_requests.append(request)
            return work_run_store.commit_work_run_attempt_decision(
                session_id=request.session_id,
                turn_id=request.turn_id,
                work_run_id=request.work_run_id,
                attempt_id=request.attempt_id,
                decision=request.decision,
                expected_work_run_revision=request.expected_work_run_revision,
                expected_progress_revision=request.expected_progress_revision,
                expected_window_revision=request.expected_window_revision,
                apply_id=request.apply_id,
            )

    result = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-call-tools",
        provider=_provider(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {"tool_id": "read_test", "arguments": {"value": "x"}}
                    ],
                },
            }
        ),
        emit=lambda _event: None,
        tool_bridge=Bridge(),
        tool_call_id_factory=lambda attempt_id, ordinal: (
            f"{attempt_id}-call-{ordinal}"
        ),
    )

    assert result.action == "call_tools"
    assert len(preflight_requests) == 1
    assert len(bridge_requests) == 1
    request = bridge_requests[0]
    assert request.attempt_id == context.attempt_id
    assert request.expected_output_revision == context.output_window.output_revision
    assert result.mutation.new_tool_call_ids == (
        "attempt-controller-1-call-1",
    )
    loaded = work_run_store.get_work_run(session_id=session_id, work_run_id=context.work_run_id)
    assert loaded.current_attempt_id == context.attempt_id
    assert loaded.attempts[-1].action == "call_tools"
    assert [item.call.tool_call_id for item in loaded.tool_calls] == [
        "attempt-controller-1-call-1"
    ]


def test_tool_preflight_rejection_repairs_inside_one_logical_model_request():
    session_id, turn_id, subject = _seed_started_attempt(include_tool=True)
    context = _context(session_id, turn_id, subject, include_tool=True)
    provider_call_ids: list[str] = []
    preflight_values: list[str] = []
    allocated_ordinals: list[int] = []
    dispatch_count = 0

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_work_run_attempt_decision"
        provider_call_ids.append(model_call_id)
        value = "wrong-type" if len(provider_call_ids) == 1 else "valid"
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {"tool_id": "read_test", "arguments": {"value": value}}
                        ],
                    },
                }
            ),
            model_call_id,
        )

    class RepairingBridge:
        def preflight(self, request: AttemptToolBridgePreflightRequest):
            proposal = request.decision.action.calls[0]
            value = str(proposal.arguments["value"])
            preflight_values.append(value)
            if value == "wrong-type":
                raise ModelOutputValidationError("invalid tool input")
            return HostAcceptedAttemptDecision(
                action=HostMaterializedCallToolsAction(
                    calls=(
                        HostMaterializedToolCall(
                            tool_call_id=request.tool_call_ids[0],
                            tool_id=proposal.tool_id,
                            tool_version="implementation-1",
                            arguments=proposal.arguments,
                            modifies_environment=False,
                        ),
                    )
                )
            )

        def dispatch(
            self,
            request: AttemptToolBridgeRequest,
            *,
            active_time_meter: AttemptActiveTimeMeter,
        ):
            nonlocal dispatch_count
            dispatch_count += 1
            return work_run_store.commit_work_run_attempt_decision(
                session_id=request.session_id,
                turn_id=request.turn_id,
                work_run_id=request.work_run_id,
                attempt_id=request.attempt_id,
                decision=request.decision,
                expected_work_run_revision=request.expected_work_run_revision,
                expected_progress_revision=request.expected_progress_revision,
                expected_window_revision=request.expected_window_revision,
                apply_id=request.apply_id,
            )

    def allocate_call_id(attempt_id: str, ordinal: int) -> str:
        allocated_ordinals.append(ordinal)
        return f"{attempt_id}-repair-call-{ordinal}"

    result = run_started_attempt(
        context,
        expected_window_revision=_window_revision(session_id),
        apply_id="controller-call-tools-repair",
        provider=provider,
        emit=lambda _event: None,
        tool_bridge=RepairingBridge(),
        tool_call_id_factory=allocate_call_id,
    )

    assert result.action == "call_tools"
    assert result.model_call.physical_attempts == 2
    assert len(set(provider_call_ids)) == 1
    assert preflight_values == ["wrong-type", "valid"]
    assert allocated_ordinals == [1]
    assert dispatch_count == 1
