from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.attempts.decision import (
    AcceptanceVerificationFeedback,
    AttemptDecisionContext,
    AttemptDecisionInputLimits,
    AttemptUserInput,
    PriorToolResultsProjection,
    VerificationVerdict,
    build_attempt_prompt_payload,
    request_attempt_decision,
)
from personagraph.l2.task_execution.verification.decision import (
    NodeVerificationContext,
    NodeVerificationInputLimits,
    NodeVerificationInputTooLarge,
    NodeVerificationInputUnsupported,
    NodeVerificationProposal,
    NodeVerificationResult,
    SupportingToolResults,
    _parse_and_guard_node_verification,
    build_node_verification_prompt_payload,
    node_verification_serialized_utf8_bytes,
    request_node_verification,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDelivery,
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
    TaskNodeDependencyInputTooLarge,
)
# 先导入运行时端口，以维持模型网关当前的模块初始化顺序。
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.work_run import (
    AcceptanceProgressItem,
    AcceptanceProgressSnapshot,
    AttemptStatus,
    Attempt,
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    OutputWindow,
    ResolvedCurrentTaskNodeDelivery,
    ResolvedTaskNodeDelivery,
    SupportingToolResult,
    TaskNodeDelivery,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    WorkRunStatus,
    WorkRun,
)
from tests.helpers.task_node_source_context import task_node_source_context
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider,
    repair_feedback_from_provider_kwargs,
)


def _dependency_deliveries() -> TaskNodeDependencyDeliveries:
    items = []
    for child_ordinal, node_id, body in (
        (1, "child-a", "子节点甲完整正文🙂"),
        (2, "child-b", "子节点乙完整正文"),
    ):
        child_run_id = f"work-run-{node_id}"
        attempt_id = f"attempt-{node_id}"
        items.append(
            TaskNodeDependencyDelivery(
                child_ordinal=child_ordinal,
                resolved_current_delivery=ResolvedCurrentTaskNodeDelivery.direct(
                    ResolvedTaskNodeDelivery(
                        delivery=TaskNodeDelivery(
                        delivery_id=f"delivery-{node_id}",
                        session_id="session-1",
                        work_run_id=child_run_id,
                        subject=TaskNodeSubject(
                            task_id="task-1",
                            graph_revision=3,
                            node_id=node_id,
                            node_revision=1,
                        ),
                        verification_request_id=f"verification-{node_id}",
                        submitted_attempt_id=attempt_id,
                        output_revision=2,
                        created_turn_id="turn-6",
                    ),
                        output_window=OutputWindow(
                        work_run_id=child_run_id,
                        output_revision=2,
                        format="markdown",
                        content=body,
                        updated_turn_id="turn-6",
                        updated_attempt_id=attempt_id,
                        ),
                    )
                ),
            )
        )
    return TaskNodeDependencyDeliveries(items=tuple(items))


def _attempt_context_with_dependencies(
    dependencies: TaskNodeDependencyDeliveries,
    *,
    dependency_limits: TaskNodeDependencyInputLimits | None = None,
) -> AttemptDecisionContext:
    verification = _context()
    return AttemptDecisionContext(
        session_id=verification.session_id,
        turn_id=verification.invocation_turn_id,
        work_run_id=verification.work_run_id,
        work_run_revision=verification.work_run.revision,
        attempt_id="attempt-5",
        attempt_ordinal=5,
        user_input=AttemptUserInput(content="综合直接子节点交付。"),
        subject=verification.subject,
        node_title=verification.node_title,
        node_objective=verification.node_objective,
        acceptances=verification.acceptances,
        acceptance_progress=verification.acceptance_progress,
        output_window=verification.locked_output_window,
        dependency_deliveries=dependencies,
        source_context=verification.source_context,
        prior_tool_results=PriorToolResultsProjection(),
        input_limits=AttemptDecisionInputLimits(
            profile_id="attempt-dependency-parity-v1",
            max_prior_tool_result_items=0,
            max_prior_tool_results_serialized_utf8_bytes=1_024,
            dependency_delivery_limits=(
                dependency_limits
                or TaskNodeDependencyInputLimits(
                    profile_id="attempt-dependency-parity-items-v1",
                    max_items=8,
                    max_serialized_utf8_bytes=100_000,
                )
            ),
            max_serialized_utf8_bytes=1_000_000,
        ),
    )


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
    )


def _input_limits(
    *,
    max_acceptance_items: int = 64,
    max_supporting_tool_result_items: int = 256,
    max_serialized_utf8_bytes: int = 1_000_000,
) -> NodeVerificationInputLimits:
    return NodeVerificationInputLimits(
        profile_id="test-node-verification-v1",
        max_acceptance_items=max_acceptance_items,
        max_supporting_tool_result_items=max_supporting_tool_result_items,
        dependency_delivery_limits=TaskNodeDependencyInputLimits(
            profile_id="verification-test-dependencies",
            max_items=32,
            max_serialized_utf8_bytes=256_000,
        ),
        max_serialized_utf8_bytes=max_serialized_utf8_bytes,
    )


def _context() -> NodeVerificationContext:
    subject = TaskNodeSubject(
        task_id="task-1",
        graph_revision=3,
        node_id="node-2",
        node_revision=4,
    )
    acceptances = (
        InSessionTaskAcceptanceProposal(
            acceptance_id="accurate_route",
            criterion="路线前后衔接且可执行",
            source_anchor_ids=("anchor_1",),
        ),
        InSessionTaskAcceptanceProposal(
            acceptance_id="includes_dates",
            criterion="包含出发与返程日期",
            source_anchor_ids=("anchor_1",),
        ),
    )
    return NodeVerificationContext(
        session_id="session-1",
        request_turn_id="turn-8",
        invocation_turn_id="turn-9",
        verification_request_id="verification-1",
        verification_request_revision=1,
        locked_work_run_revision=10,
        work_run=WorkRun(
            work_run_id="work-run-9",
            subject=subject,
            revision=11,
            status=WorkRunStatus.ACTIVE,
            reason="verification_pending",
        ),
        submitted_attempt=Attempt(
            attempt_id="attempt-submit",
            work_run_id="work-run-9",
            ordinal=4,
            status=AttemptStatus.CLOSED,
            submitted_output_revision=5,
        ),
        acceptance_progress=AcceptanceProgressSnapshot(
            work_run_id="work-run-9",
            subject=subject,
            revision=7,
            evaluated_output_revision=5,
            items=(
                AcceptanceProgressItem(
                    acceptance_id="accurate_route",
                    model_claimed_satisfied=True,
                    supporting_tool_result_ids=("result-1",),
                ),
                AcceptanceProgressItem(
                    acceptance_id="includes_dates",
                    model_claimed_satisfied=True,
                    empty_support_justification={
                        "reason_code": "provided_context_sufficient",
                        "explanation": (
                            "The locked source context directly provides the dates."
                        ),
                    },
                ),
            ),
        ),
        node_title="产出旅行计划",
        node_objective="形成一份包含明确日期与路线的旅行计划。",
        acceptances=acceptances,
        locked_output_window=OutputWindow(
            work_run_id="work-run-9",
            output_revision=5,
            format="markdown",
            content="# 上海旅行计划\n9 月 1 日出发，9 月 8 日返程。",
            updated_turn_id="turn-7",
            updated_attempt_id="attempt-write",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        source_context=task_node_source_context(
            session_id="session-1",
            subject=subject,
            acceptances=acceptances,
        ),
        supporting_tool_results=SupportingToolResults(
            items=(
                SupportingToolResult(
                    work_run_id="work-run-9",
                    tool_id="read_itinerary_source",
                    tool_version="1.0.0",
                    result=ToolResult(
                        status=ToolResultStatus.SUCCEEDED,
                        tool_result_id="result-1",
                        tool_call_id="call-1",
                        attempt_id="attempt-tool",
                        ordinal=1,
                        output={
                            "departure": "2026-09-01",
                            "return": "2026-09-08",
                        },
                    ),
                ),
            )
        ),
        input_limits=_input_limits(),
    )


def _feedback(
    acceptance_id: str,
    verdict: str = "passed",
) -> dict[str, object]:
    return {
        "acceptance_id": acceptance_id,
        "verdict": verdict,
        "finding": f"{acceptance_id} 的条件已核验。",
        "missing_requirements": (
            [] if verdict == "passed" else [f"仍需满足 {acceptance_id}"]
        ),
    }


def test_findings_mutation_receipt_is_not_admissible_node_evidence() -> None:
    payload = _context().model_dump(mode="json")
    payload["supporting_tool_results"]["items"][0]["tool_id"] = (
        "record_execution_findings"
    )

    with pytest.raises(
        ValidationError,
        match="execution-findings mutation receipt",
    ):
        NodeVerificationContext.model_validate(payload)


def _valid_reply(*, second_verdict: str = "passed") -> str:
    return json.dumps(
        {
            "acceptance_results": [
                _feedback("accurate_route"),
                _feedback("includes_dates", second_verdict),
            ]
        },
        ensure_ascii=False,
    )


def test_prompt_projects_exact_locked_submission_and_supporting_results():
    received: dict[str, object] = {}
    events = []

    def complete(system_prompt: str, user_content: str, **kwargs: object) -> ModelResult:
        received["system_prompt"] = system_prompt
        received["payload"] = json.loads(user_content)
        received["kwargs"] = kwargs
        return _model_result(_valid_reply(), str(kwargs["model_call_id"]))

    context = _context()
    result = request_node_verification(
        context,
        provider=as_prepared_test_provider(complete),
        emit=events.append,
    )

    payload = received["payload"]
    assert isinstance(payload, dict)
    assert payload["bindings"] == {
        "session_id": "session-1",
        "request_turn_id": "turn-8",
        "verification_request_id": "verification-1",
        "work_run_id": "work-run-9",
        "locked_work_run_revision": 10,
        "submitted_attempt_id": "attempt-submit",
        "acceptance_progress_revision": 7,
        "subject": {
            "kind": "task_node",
            "task_id": "task-1",
            "graph_revision": 3,
            "node_id": "node-2",
            "node_revision": 4,
        },
        "task_id": "task-1",
        "graph_revision": 3,
        "node_id": "node-2",
        "node_revision": 4,
        "output_revision": 5,
    }
    assert payload["node"] == {
        "title": "产出旅行计划",
        "objective": "形成一份包含明确日期与路线的旅行计划。",
        "acceptances": [
            {
                "acceptance_id": "accurate_route",
                "criterion": "路线前后衔接且可执行",
                "source_anchor_ids": ["anchor_1"],
            },
            {
                "acceptance_id": "includes_dates",
                "criterion": "包含出发与返程日期",
                "source_anchor_ids": ["anchor_1"],
            },
        ],
    }
    assert payload["locked_output_window"] == {
        "format": "markdown",
        "content": "# 上海旅行计划\n9 月 1 日出发，9 月 8 日返程。",
    }
    assert payload["supporting_tool_results"] == [
        {
            "tool_id": "read_itinerary_source",
            "tool_version": "1.0.0",
            "status": "succeeded",
            "tool_result_id": "result-1",
            "tool_call_id": "call-1",
            "attempt_id": "attempt-tool",
            "ordinal": 1,
            "output": {
                "departure": "2026-09-01",
                "return": "2026-09-08",
            },
            "error_code": None,
            "error_message": None,
        }
    ]
    assert payload["completion_submission"] == [
        {
            "acceptance_id": "accurate_route",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": ["result-1"],
            "empty_support_justification": None,
        },
        {
            "acceptance_id": "includes_dates",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
            "empty_support_justification": {
                "schema_version": "empty-support-justification-v1",
                "reason_code": "provided_context_sufficient",
                "explanation": (
                    "The locked source context directly provides the dates."
                ),
            },
        },
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "hash" not in serialized
    assert "evidence_ref" not in serialized
    assert "empty_support_justification" in serialized
    assert "overall" not in serialized
    assert "all_pass" not in serialized
    assert "graph_action" not in serialized
    assert "test-node-verification-v1" not in serialized
    assert "一轮完整重验" in str(received["system_prompt"])
    assert "host_output_repair_feedback" not in str(received["system_prompt"])
    assert received["kwargs"] == {
        "model_call_id": result.model_call_id,
        "purpose": "runtime_task_node_semantic_verification",
    }
    assert "max_tokens" not in received["kwargs"]
    assert result.value.work_run_id == "work-run-9"
    assert result.value.verification_request_id == "verification-1"
    assert result.value.verification_request_revision == 1
    assert result.value.locked_work_run_revision == 10
    assert result.value.submitted_attempt_id == "attempt-submit"
    assert result.value.acceptance_progress_revision == 7
    assert result.value.output_revision == 5
    assert result.value.subject == context.subject
    assert result.value.all_pass is True
    assert [event.stage.value for event in events] == ["VERIFICATION", "VERIFICATION"]
    assert [event.status.value for event in events] == ["started", "completed"]


def test_exact_canonical_utf8_input_is_measured_then_sent_without_truncation():
    received: list[str] = []
    context = _context()
    expected_bytes = node_verification_serialized_utf8_bytes(context)
    exact_context = context.model_copy(
        update={
            "input_limits": _input_limits(
                max_acceptance_items=len(context.acceptances),
                max_supporting_tool_result_items=len(
                    context.supporting_tool_results.items
                ),
                max_serialized_utf8_bytes=expected_bytes,
            )
        }
    )

    request_node_verification(
        exact_context,
        provider=as_prepared_test_provider(
            lambda _system, user, **kwargs: (
                received.append(user)
                or _model_result(_valid_reply(), str(kwargs["model_call_id"]))
            )
        ),
        emit=lambda _event: None,
    )

    assert len(received) == 1
    assert len(received[0].encode("utf-8")) == expected_bytes
    payload = json.loads(received[0])
    assert len(payload["node"]["acceptances"]) == len(context.acceptances)
    assert len(payload["supporting_tool_results"]) == len(
        context.supporting_tool_results.items
    )


@pytest.mark.parametrize(
    "limits",
    [
        _input_limits(max_acceptance_items=1),
        _input_limits(max_supporting_tool_result_items=0),
        _input_limits(max_serialized_utf8_bytes=1),
    ],
    ids=("acceptance-count", "supporting-result-count", "serialized-bytes"),
)
def test_complete_input_over_any_host_limit_fails_before_provider(
    limits: NodeVerificationInputLimits,
):
    provider_calls = 0
    events = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit verifier input must not reach provider")

    with pytest.raises(NodeVerificationInputTooLarge) as error:
        request_node_verification(
            _context().model_copy(update={"input_limits": limits}),
            provider=as_prepared_test_provider(provider),
            emit=events.append,
        )

    assert error.value.limits == limits
    assert error.value.acceptance_count == 2
    assert error.value.supporting_tool_result_count == 1
    assert error.value.serialized_utf8_bytes > 1
    assert provider_calls == 0
    assert events == []


def test_attempt_and_verifier_receive_the_same_ordered_dependency_bodies():
    dependencies = _dependency_deliveries()
    attempt_context = _attempt_context_with_dependencies(dependencies)
    verification_context = _context().model_copy(
        update={"dependency_deliveries": dependencies}
    )

    attempt_payload = build_attempt_prompt_payload(attempt_context)
    verification_payload = build_node_verification_prompt_payload(
        verification_context
    )

    assert attempt_payload["dependency_deliveries"] == (
        verification_payload["dependency_deliveries"]
    )
    assert [
        item["subject"]["node_id"]
        for item in attempt_payload["dependency_deliveries"]
    ] == ["child-a", "child-b"]
    assert [
        item["output_window"]["content"]
        for item in attempt_payload["dependency_deliveries"]
    ] == ["子节点甲完整正文🙂", "子节点乙完整正文"]


def test_dependency_sublimit_stops_both_model_boundaries_before_provider():
    dependencies = _dependency_deliveries()
    tiny = TaskNodeDependencyInputLimits(
        profile_id="dependency-one-item-v1",
        max_items=1,
        max_serialized_utf8_bytes=100_000,
    )
    attempt_context = _attempt_context_with_dependencies(
        dependencies,
        dependency_limits=tiny,
    )
    verification_base = _context()
    verification_context = verification_base.model_copy(
        update={
            "dependency_deliveries": dependencies,
            "input_limits": verification_base.input_limits.model_copy(
                update={"dependency_delivery_limits": tiny}
            ),
        }
    )
    attempt_provider_calls = 0
    verification_provider_calls = 0

    def attempt_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal attempt_provider_calls
        attempt_provider_calls += 1
        raise AssertionError("over-limit dependency input reached Attempt provider")

    def verification_provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal verification_provider_calls
        verification_provider_calls += 1
        raise AssertionError("over-limit dependency input reached verifier")

    with pytest.raises(TaskNodeDependencyInputTooLarge):
        request_attempt_decision(
            attempt_context,
            provider=as_prepared_test_provider(attempt_provider),
            emit=lambda _event: None,
        )
    with pytest.raises(TaskNodeDependencyInputTooLarge):
        request_node_verification(
            verification_context,
            provider=as_prepared_test_provider(verification_provider),
            emit=lambda _event: None,
        )

    assert attempt_provider_calls == 0
    assert verification_provider_calls == 0


def test_non_utf8_json_input_is_typed_unsupported_before_provider():
    context = _context()
    unsupported_window = context.locked_output_window.model_copy(
        update={"content": "unsupported-surrogate-\ud800"}
    )
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("unsupported verifier input must not reach provider")

    with pytest.raises(NodeVerificationInputUnsupported) as error:
        request_node_verification(
            context.model_copy(update={"locked_output_window": unsupported_window}),
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert error.value.reason == "not_canonical_json_utf8"
    assert provider_calls == 0


def test_malformed_structured_output_retries_with_one_logical_model_call_id():
    calls = 0
    model_call_ids: list[str] = []

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        model_call_ids.append(str(kwargs["model_call_id"]))
        reply = "not-json" if calls == 1 else _valid_reply()
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert result.attempts == 2
    assert model_call_ids == [result.model_call_id, result.model_call_id]


def test_rejected_node_review_receives_safe_structured_repair_feedback():
    secret_marker = "PRIVATE_REJECTED_NODE_REVIEW"
    invalid = json.loads(_valid_reply())
    invalid.pop("acceptance_results")
    invalid["private_extra"] = secret_marker
    seen_payloads: list[dict[str, object]] = []
    seen_feedback: list[dict[str, object]] = []

    def complete(_system: str, user: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user)
        assert isinstance(payload, dict)
        seen_payloads.append(payload)
        feedback = repair_feedback_from_provider_kwargs(kwargs)
        if feedback is not None:
            seen_feedback.append(feedback)
        reply = (
            json.dumps(invalid, ensure_ascii=False)
            if len(seen_payloads) == 1
            else _valid_reply()
        )
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert "host_output_repair_feedback" not in seen_payloads[0]
    assert seen_payloads[1] == seen_payloads[0]
    assert len(seen_feedback) == 1
    feedback = seen_feedback[0]
    assert set(feedback) == {"current_issues"}
    explanations = {issue["safe_explanation"] for issue in feedback["current_issues"]}
    assert "目标合同要求此位置必须存在。" in explanations
    assert "该位置含有目标合同未声明的额外字段。" in explanations
    assert secret_marker not in json.dumps(feedback, ensure_ascii=False)


def test_prepared_node_review_repair_uses_four_message_context():
    secret_marker = "PRIVATE_REJECTED_NODE_REVIEW"
    invalid = json.dumps(
        {"private_extra": secret_marker},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    prepared_calls: list[dict[str, object]] = []

    class Prepared:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return _model_result(self.reply, model_call_id)

    def complete(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared provider must not use its legacy path")

    def prepare(
        system_prompt: str,
        user_content: str,
        **kwargs: object,
    ) -> Prepared:
        prepared_calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                **kwargs,
            }
        )
        return Prepared(invalid if len(prepared_calls) == 1 else _valid_reply())

    complete.prepare = prepare  # type: ignore[attr-defined]

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert result.attempts == 2
    assert len(prepared_calls) == 2
    first, repair = prepared_calls
    assert "repair_messages" not in first
    assert first["system_prompt"] == repair["system_prompt"]
    assert first["user_content"] == repair["user_content"]
    assert "host_output_repair_feedback" not in str(first["system_prompt"])
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == first["system_prompt"]
    assert messages[1]["content"] == first["user_content"]
    assert messages[2]["content"] == invalid
    repair_instruction = messages[3]["content"]
    assert invalid not in repair_instruction
    envelope = json.loads(repair_instruction.split("Host 修复清单：", 1)[1])
    assert set(envelope) == {"current_issues"}
    assert "当前清单可能不完整" not in repair_instruction
    issues_by_path = {
        tuple(issue["paths"]): issue for issue in envelope["current_issues"]
    }
    assert ("/acceptance_results",) in issues_by_path
    assert ("",) in issues_by_path
    assert secret_marker not in json.dumps(envelope, ensure_ascii=False)


def test_prepared_legal_verification_failure_is_not_output_repair():
    prepared_calls: list[dict[str, object]] = []

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return _model_result(
                _valid_reply(second_verdict="insufficient_evidence"),
                model_call_id,
            )

    def complete(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared provider must not use its legacy path")

    def prepare(
        _system_prompt: str,
        _user_content: str,
        **kwargs: object,
    ) -> Prepared:
        prepared_calls.append(kwargs)
        return Prepared()

    complete.prepare = prepare  # type: ignore[attr-defined]

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert result.attempts == 1
    assert result.value.all_pass is False
    assert len(prepared_calls) == 1
    assert "repair_messages" not in prepared_calls[0]


def test_repair_feedback_does_not_mutate_frozen_node_input():
    context = _context()
    exact_bytes = node_verification_serialized_utf8_bytes(context)
    exact_context = context.model_copy(
        update={
            "input_limits": context.input_limits.model_copy(
                update={"max_serialized_utf8_bytes": exact_bytes}
            )
        }
    )
    prepared_calls: list[dict[str, object]] = []

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        prepared_calls.append(
            {
                "user": _user,
                "repair_messages": kwargs.get("repair_messages"),
            }
        )
        reply = "not-json" if len(prepared_calls) == 1 else _valid_reply()
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_node_verification(
        exact_context,
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert prepared_calls[0]["user"] == prepared_calls[1]["user"]
    assert prepared_calls[0]["repair_messages"] is None
    assert isinstance(prepared_calls[1]["repair_messages"], list)


@pytest.mark.parametrize(
    "invalid_results",
    [
        [_feedback("accurate_route")],
        [_feedback("accurate_route"), _feedback("accurate_route")],
        [_feedback("accurate_route"), _feedback("unknown_acceptance")],
    ],
    ids=("missing", "duplicate", "unknown"),
)
def test_acceptance_id_set_or_order_mismatch_retries_full_review(
    invalid_results: list[dict[str, object]],
):
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        reply = (
            json.dumps({"acceptance_results": invalid_results}, ensure_ascii=False)
            if calls == 1
            else _valid_reply()
        )
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert result.attempts == 2
    assert [item.acceptance_id for item in result.value.acceptance_results] == [
        "accurate_route",
        "includes_dates",
    ]


def test_node_review_host_guard_reports_all_current_paths_with_complete_coverage():
    reply = json.dumps(
        {
            "acceptance_results": [
                _feedback("accurate_route"),
                _feedback("unknown_acceptance"),
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ModelOutputValidationError) as captured:
        _parse_and_guard_node_verification(reply, context=_context())

    error = captured.value
    assert error.repair_issue_coverage.value == "complete"
    assert error.omitted_repair_issue_count == 0
    assert {
        path
        for issue in error.repair_issues or ()
        for path in issue.paths
    } == {
        "/acceptance_results",
        "/acceptance_results/1/acceptance_id",
    }


def test_reordered_complete_result_is_canonicalized_without_model_retry():
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            json.dumps(
                {
                    "acceptance_results": [
                        _feedback("includes_dates"),
                        _feedback("accurate_route"),
                    ]
                },
                ensure_ascii=False,
            ),
            str(kwargs["model_call_id"]),
        )

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert calls == 1
    assert result.attempts == 1
    assert [item.acceptance_id for item in result.value.acceptance_results] == [
        "accurate_route",
        "includes_dates",
    ]


def test_model_cannot_supply_host_aggregate_or_extra_actions():
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        proposal = json.loads(_valid_reply())
        if calls == 1:
            proposal["overall"] = "passed"
        elif calls == 2:
            proposal["graph_action"] = "revise"
        return _model_result(
            json.dumps(proposal, ensure_ascii=False),
            str(kwargs["model_call_id"]),
        )

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert calls == 3
    assert result.attempts == 3
    assert result.value.all_pass is True


@pytest.mark.parametrize(
    ("second_verdict", "expected_all_pass"),
    [
        ("passed", True),
        ("not_satisfied", False),
        ("insufficient_evidence", False),
    ],
)
def test_host_derives_all_pass_from_every_acceptance(
    second_verdict: str,
    expected_all_pass: bool,
):
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            _valid_reply(second_verdict=second_verdict),
            str(kwargs["model_call_id"]),
        )

    result = request_node_verification(
        _context(),
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert calls == 1
    assert result.attempts == 1
    assert result.value.all_pass is expected_all_pass


def test_host_result_rejects_a_caller_supplied_incorrect_aggregate():
    context = _context()

    with pytest.raises(ValidationError, match="Host-derived aggregate"):
        NodeVerificationResult(
            verification_request_id=context.verification_request_id,
            verification_request_revision=context.verification_request_revision,
            work_run_id=context.work_run_id,
            locked_work_run_revision=context.locked_work_run_revision,
            submitted_attempt_id=context.submitted_attempt_id,
            acceptance_progress_revision=context.acceptance_progress_revision,
            subject=context.subject,
            output_revision=context.locked_output_window.output_revision,
            acceptance_results=(
                AcceptanceVerificationFeedback(
                    acceptance_id="accurate_route",
                    verdict=VerificationVerdict.NOT_SATISFIED,
                    finding="路线仍不完整。",
                    missing_requirements=("补齐路线",),
                ),
            ),
            all_pass=True,
        )


def test_downstream_retry_is_separate_from_passed_node_acceptance():
    context = _context()
    acceptance_results = tuple(
        AcceptanceVerificationFeedback(
            acceptance_id=item.acceptance_id,
            verdict=VerificationVerdict.PASSED,
            finding="节点自身验收已通过。",
        )
        for item in context.acceptances
    )
    downstream = DownstreamVerificationFeedback(
        gate_id="whole_task_delivery",
        disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
        finding="根正文遗漏了用户要求的发布步骤。",
        repair_objective="保留正确内容并补齐发布步骤。",
        source_result_id="candidate-review-result-1",
        source_result_sha256="a" * 64,
        affected_subject_ids=(context.subject.node_id,),
    )

    result = NodeVerificationResult(
        verification_request_id=context.verification_request_id,
        verification_request_revision=context.verification_request_revision,
        work_run_id=context.work_run_id,
        locked_work_run_revision=context.locked_work_run_revision,
        submitted_attempt_id=context.submitted_attempt_id,
        acceptance_progress_revision=context.acceptance_progress_revision,
        subject=context.subject,
        output_revision=context.locked_output_window.output_revision,
        acceptance_results=acceptance_results,
        downstream_results=(downstream,),
        all_pass=False,
    )

    assert all(
        item.verdict is VerificationVerdict.PASSED
        for item in result.acceptance_results
    )
    assert result.downstream_results == (downstream,)
    assert result.all_pass is False


def test_downstream_feedback_shape_is_host_validated():
    with pytest.raises(ValidationError, match="RETRY_ATTEMPT requires"):
        DownstreamVerificationFeedback(
            gate_id="task_graph_semantic",
            disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
            finding="终端提案仍需局部订正。",
            source_result_id="semantic-result-1",
            source_result_sha256="b" * 64,
        )

    with pytest.raises(ValidationError, match="BLOCKED requires"):
        DownstreamVerificationFeedback(
            gate_id="task_graph_semantic",
            disposition=DownstreamVerificationDisposition.BLOCKED,
            finding="缺少用户授权。",
            repair_objective="不能用执行订正替代授权。",
            source_result_id="semantic-result-2",
            source_result_sha256="c" * 64,
        )


def test_contracts_are_frozen_and_forbid_unknown_fields():
    context = _context()

    with pytest.raises(ValidationError, match="frozen"):
        context.locked_work_run_revision = 12

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NodeVerificationProposal.model_validate(
            {
                "acceptance_results": [
                    _feedback("accurate_route"),
                    _feedback("includes_dates"),
                ],
                "overall": "passed",
            }
        )


def test_context_fails_closed_on_cross_workrun_or_duplicate_inputs():
    context = _context()
    payload = context.model_dump()
    payload["locked_output_window"] = OutputWindow(
        work_run_id="other-run",
        output_revision=5,
        format="markdown",
        content="# other",
        updated_turn_id="turn-7",
        updated_attempt_id="attempt-write",
    ).model_dump()

    with pytest.raises(ValidationError, match="does not belong"):
        NodeVerificationContext.model_validate(payload)

    duplicated = context.acceptances + (context.acceptances[0],)
    with pytest.raises(ValidationError, match="Acceptance IDs must be unique"):
        NodeVerificationContext.model_validate(
            {**context.model_dump(), "acceptances": duplicated}
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["submitted_attempt"].update(
                {"submitted_output_revision": 4}
            ),
            "submitted OutputWindow",
        ),
        (
            lambda payload: payload["acceptance_progress"].update(
                {"evaluated_output_revision": 4}
            ),
            "current OutputWindow",
        ),
        (
            lambda payload: payload["acceptance_progress"].update(
                {
                    "subject": {
                        "kind": "task_node",
                        "task_id": "task-1",
                        "graph_revision": 3,
                        "node_id": "other-node",
                        "node_revision": 4,
                    }
                }
            ),
            "subject",
        ),
        (
            lambda payload: payload["acceptance_progress"]["items"][0].update(
                {"supporting_tool_result_ids": ["missing-result"]}
            ),
            "supporting ToolResult",
        ),
    ],
)
def test_context_rejects_mismatched_submit_progress_or_supporting_bindings(
    mutate,
    message: str,
):
    payload = _context().model_dump(mode="json")
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        NodeVerificationContext.model_validate(payload)


def test_context_rejects_unreferenced_supporting_result():
    payload = _context().model_dump(mode="json")
    payload["acceptance_progress"]["items"][0]["supporting_tool_result_ids"] = []

    with pytest.raises(ValidationError, match="supporting ToolResult"):
        NodeVerificationContext.model_validate(payload)


def test_supporting_result_bindings_must_be_unique():
    item = _context().supporting_tool_results.items[0]

    with pytest.raises(ValidationError, match="ToolResult IDs must be unique"):
        SupportingToolResults(items=(item, item))
