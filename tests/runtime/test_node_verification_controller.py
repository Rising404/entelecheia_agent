from __future__ import annotations

import json
from dataclasses import replace

import pytest

from personagraph.context_budget import ContextBudgetExceeded
from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.attempts.decision import VerificationVerdict
from personagraph.l2.task_execution.verification.decision import (
    NodeVerificationContext,
    NodeVerificationInputLimits,
    NodeVerificationInputUnsupported,
    NodeVerificationResult,
    request_node_verification,
)
from personagraph.l2.task_execution.verification.controller import (
    NodeVerificationApplicationRequest,
    NodeVerificationCommit,
    NodeVerificationInterrupt,
    NodeVerificationNextAttemptPlan,
    NodeVerificationPrepareStoreCommand,
    NodeVerificationResumeRequest,
    NodeVerificationResumeStoreCommand,
    NodeVerificationStartNextAttempt,
    NodeVerificationStoreProjection,
    PreparedNodeVerification,
    SettledNodeVerification,
    resume_and_run_node_verification,
    run_node_verification,
)
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
)
# 运行时导入先于模型网关，以保留包当前的初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.l2.work_run import (
    AcceptanceProgressItem,
    AcceptanceProgressSnapshot,
    AttemptStatus,
    Attempt,
    DownstreamVerificationDisposition,
    WorkExecutionMutationResult,
    DownstreamVerificationFeedback,
    OutputWindow,
    TaskNodeSubject,
    WorkRunBudgetDisposition,
    WorkRunBudget,
    WorkRunStatus,
    WorkRun,
    charge_work_run_active_seconds,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.helpers.task_node_source_context import task_node_source_context


class FakeStoreConflict(RuntimeError):
    pass


class FakeMonotonicClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values or (100.0, 101.0))
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._values)


def _clock(*values: float) -> FakeMonotonicClock:
    return FakeMonotonicClock(*values)


def _input_limits(
    *,
    max_serialized_utf8_bytes: int = 1_000_000,
) -> NodeVerificationInputLimits:
    return NodeVerificationInputLimits(
        profile_id="test-node-verification-v1",
        max_acceptance_items=64,
        max_supporting_tool_result_items=256,
        dependency_delivery_limits=TaskNodeDependencyInputLimits(
            profile_id="controller-test-dependencies",
            max_items=32,
            max_serialized_utf8_bytes=256_000,
        ),
        max_serialized_utf8_bytes=max_serialized_utf8_bytes,
    )


def _context(
    *,
    invocation_turn_id: str = "turn-1",
    verification_request_revision: int = 1,
    live_work_run_revision: int = 4,
    input_limits: NodeVerificationInputLimits | None = None,
) -> NodeVerificationContext:
    subject = TaskNodeSubject(
        task_id="task-1",
        graph_revision=2,
        node_id="node-1",
        node_revision=4,
    )
    acceptances = (
        InSessionTaskAcceptanceProposal(
            acceptance_id="deliverable",
            criterion="正文包含日期和路线",
            source_anchor_ids=("anchor-1",),
        ),
    )
    return NodeVerificationContext(
        session_id="session-1",
        request_turn_id="turn-1",
        invocation_turn_id=invocation_turn_id,
        verification_request_id="verification-1",
        verification_request_revision=verification_request_revision,
        locked_work_run_revision=3,
        work_run=WorkRun(
            work_run_id="work-run-1",
            subject=subject,
            revision=live_work_run_revision,
            status=WorkRunStatus.ACTIVE,
            reason="verification_pending",
        ),
        submitted_attempt=Attempt(
            attempt_id="submit-attempt",
            work_run_id="work-run-1",
            ordinal=2,
            status=AttemptStatus.CLOSED,
            submitted_output_revision=2,
        ),
        acceptance_progress=AcceptanceProgressSnapshot(
            work_run_id="work-run-1",
            subject=subject,
            revision=2,
            evaluated_output_revision=2,
            items=(
                AcceptanceProgressItem(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=True,
                ),
            ),
        ),
        node_title="旅行计划",
        node_objective="给出可直接使用的旅行计划。",
        acceptances=acceptances,
        locked_output_window=OutputWindow(
            work_run_id="work-run-1",
            output_revision=2,
            format="markdown",
            content="# 旅行计划\n2026-09-01 出发。",
            updated_turn_id="turn-1",
            updated_attempt_id="submit-attempt",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        source_context=task_node_source_context(
            session_id="session-1",
            subject=subject,
            acceptances=acceptances,
        ),
        input_limits=input_limits or _input_limits(),
    )


def _next_attempt_plan() -> NodeVerificationNextAttemptPlan:
    return NodeVerificationNextAttemptPlan(
        attempt_id="attempt-after-verification",
        apply_id="start-after-verification",
        catalog_snapshot={"tools": []},
    )


def _request() -> NodeVerificationApplicationRequest:
    return NodeVerificationApplicationRequest(
        session_id="session-1",
        turn_id="turn-1",
        work_run_id="work-run-1",
        submitted_attempt_id="submit-attempt",
        verification_request_id="verification-1",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=10,
        prepare_apply_id="prepare-verification",
        commit_apply_id="commit-verification",
        interrupt_apply_id="interrupt-verification",
        delivery_id="delivery-1",
        input_limits=_input_limits(),
        next_attempt=_next_attempt_plan(),
    )


def _resume_request() -> NodeVerificationResumeRequest:
    return NodeVerificationResumeRequest(
        session_id="session-1",
        turn_id="turn-2",
        work_run_id="work-run-1",
        submitted_attempt_id="submit-attempt",
        verification_request_id="verification-1",
        expected_work_run_revision=5,
        expected_verification_request_revision=2,
        expected_window_revision=12,
        resume_apply_id="resume-verification",
        commit_apply_id="commit-verification-after-resume",
        interrupt_apply_id="interrupt-verification-after-resume",
        delivery_id="delivery-1",
        input_limits=_input_limits(),
        next_attempt=_next_attempt_plan(),
    )


def _reply(verdict: str = "passed") -> str:
    return json.dumps(
        {
            "acceptance_results": [
                {
                    "acceptance_id": "deliverable",
                    "verdict": verdict,
                    "finding": "已检查正文。",
                    "missing_requirements": (
                        [] if verdict == "passed" else ["缺少返程日期"]
                    ),
                }
            ]
        },
        ensure_ascii=False,
    )


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="fake-provider",
        model="fake-model",
        latency_ms=2,
        model_call_id=model_call_id,
    )


class FakeVerificationStore:
    """各方法模拟独立短事务的功能替身。"""

    def __init__(self) -> None:
        self.in_transaction = False
        self.calls: list[str] = []
        self.context = _context()
        self.request_revision = 1
        self.work_run_revision = 3
        self.window_revision = 10
        self.prepared_once = False
        self.interrupted = False
        self.settled: NodeVerificationStoreProjection | None = None
        self.started: WorkExecutionMutationResult | None = None
        self.prepared: PreparedNodeVerification | None = None
        self.prepare_error: Exception | None = None
        self.settlement_disposition = WorkRunBudgetDisposition.WITHIN_LIMIT
        self.prepare_commands: list[NodeVerificationPrepareStoreCommand] = []
        self.resume_commands: list[NodeVerificationResumeStoreCommand] = []
        self.commit_commands: list[NodeVerificationCommit] = []
        self.interrupt_commands: list[NodeVerificationInterrupt] = []
        self.start_commands: list[NodeVerificationStartNextAttempt] = []

    def _begin(self, call: str) -> None:
        assert not self.in_transaction
        self.in_transaction = True
        self.calls.append(call)

    def _end(self) -> None:
        assert self.in_transaction
        self.in_transaction = False

    def prepare_node_verification(
        self,
        command: NodeVerificationPrepareStoreCommand,
    ):
        self._begin("prepare")
        try:
            self.prepare_commands.append(command)
            if self.prepare_error is not None:
                raise self.prepare_error
            assert command.verification_request_id == "verification-1"
            if self.settled is not None:
                return SettledNodeVerification(
                    projection=self.settled.model_copy(update={"status": "replayed"})
                )
            if self.interrupted:
                raise FakeStoreConflict("interrupted requests require explicit resume")
            if not self.prepared_once:
                self.prepared_once = True
                self.work_run_revision += 1
                self.window_revision += 1
                self.context = _context(
                    verification_request_revision=self.request_revision,
                    live_work_run_revision=self.work_run_revision,
                    input_limits=command.input_limits,
                )
            self.prepared = PreparedNodeVerification(
                verification_request_id="verification-1",
                verification_request_revision=self.request_revision,
                window_revision=self.window_revision,
                context=self.context,
            )
            return self.prepared
        finally:
            self._end()

    def resume_node_verification(
        self,
        command: NodeVerificationResumeStoreCommand,
    ):
        self._begin("resume")
        try:
            self.resume_commands.append(command)
            if self.settled is not None:
                return SettledNodeVerification(
                    projection=self.settled.model_copy(update={"status": "replayed"})
                )
            if not self.interrupted:
                raise FakeStoreConflict("request is not interrupted")
            if (
                command.verification_request_id != "verification-1"
                or command.expected_verification_request_revision
                != self.request_revision
            ):
                raise FakeStoreConflict("stale verification request revision")
            self.interrupted = False
            self.request_revision += 1
            self.work_run_revision += 1
            self.window_revision += 1
            self.context = _context(
                invocation_turn_id=command.turn_id,
                verification_request_revision=self.request_revision,
                live_work_run_revision=self.work_run_revision,
                input_limits=command.input_limits,
            )
            self.prepared = PreparedNodeVerification(
                verification_request_id="verification-1",
                verification_request_revision=self.request_revision,
                window_revision=self.window_revision,
                rebound_for_recovery=True,
                context=self.context,
            )
            return self.prepared
        finally:
            self._end()

    def commit_node_verification(
        self,
        command: NodeVerificationCommit,
    ) -> NodeVerificationStoreProjection:
        self._begin("commit")
        try:
            self.commit_commands.append(command)
            if (
                command.expected_verification_request_revision
                != self.request_revision
            ):
                raise FakeStoreConflict("late verification result")
            assert self.prepared is not None
            self.request_revision += 1
            self.work_run_revision += 1
            self.window_revision += 1
            transition = self._budget_transition(command.active_seconds_delta)
            hard = (
                transition.disposition
                is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
            )
            outcome = (
                "work_run_limit_reached"
                if hard
                else "passed"
                if command.verification_result.all_pass
                else "not_passed"
            )
            projection = self._projection(
                outcome=outcome,
                prepared=self.prepared,
                result=None if hard else command.verification_result,
                budget_transition=transition,
            )
            self.settled = projection
            return projection
        finally:
            self._end()

    def interrupt_node_verification(
        self,
        command: NodeVerificationInterrupt,
    ) -> NodeVerificationStoreProjection:
        self._begin("interrupt")
        try:
            self.interrupt_commands.append(command)
            if (
                command.expected_verification_request_revision
                != self.request_revision
            ):
                raise FakeStoreConflict("late verification interruption")
            assert self.prepared is not None
            self.request_revision += 1
            self.work_run_revision += 1
            self.window_revision += 1
            transition = self._budget_transition(command.active_seconds_delta)
            hard = (
                transition.disposition
                is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
            )
            self.interrupted = not hard
            projection = self._projection(
                outcome=("work_run_limit_reached" if hard else "interrupted"),
                prepared=self.prepared,
                result=None,
                technical_reason=(
                    "work_run_limit_reached" if hard else command.reason
                ),
                budget_transition=transition,
            )
            if hard:
                self.settled = projection
            return projection
        finally:
            self._end()

    def start_next_attempt(
        self,
        command: NodeVerificationStartNextAttempt,
    ) -> WorkExecutionMutationResult:
        self._begin("start_next_attempt")
        try:
            self.start_commands.append(command)
            if self.started is not None:
                return self.started.model_copy(update={"status": "replayed"})
            if (
                command.expected_work_run_revision != self.work_run_revision
                or command.expected_window_revision != self.window_revision
            ):
                raise FakeStoreConflict("stale next-Attempt CAS")
            self.work_run_revision += 1
            self.window_revision += 1
            attempt = Attempt(
                attempt_id=command.plan.attempt_id,
                work_run_id=command.work_run_id,
                ordinal=3,
            )
            self.started = WorkExecutionMutationResult(
                status="applied",
                work_run_id=command.work_run_id,
                work_run_revision=self.work_run_revision,
                work_run_status=WorkRunStatus.ACTIVE,
                current_attempt_id=command.plan.attempt_id,
                acceptance_progress_revision=command.expected_progress_revision,
                output_window_revision=command.expected_output_revision,
                attempt=attempt,
                window_state_version=self.window_revision,
                budget_transition=None,
            )
            return self.started
        finally:
            self._end()

    def _projection(
        self,
        *,
        outcome: str,
        prepared: PreparedNodeVerification,
        result: NodeVerificationResult | None,
        technical_reason: str | None = None,
        budget_transition=None,
    ) -> NodeVerificationStoreProjection:
        passed = result is not None and result.all_pass
        interrupted = outcome == "interrupted"
        hard = outcome == "work_run_limit_reached"
        soft_nonpass = (
            outcome == "not_passed"
            and budget_transition is not None
            and budget_transition.disposition
            is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
        )
        return NodeVerificationStoreProjection(
            status="applied",
            outcome=outcome,
            verification_request_id="verification-1",
            verified_verification_request_revision=(
                prepared.verification_request_revision
            ),
            verification_request_revision=self.request_revision,
            verification_request_status=(
                "interrupted" if interrupted or hard else "completed"
            ),
            work_run_id="work-run-1",
            locked_work_run_revision=prepared.context.locked_work_run_revision,
            verified_acceptance_progress_revision=(
                prepared.context.acceptance_progress_revision
            ),
            verified_output_revision=(
                prepared.context.locked_output_window.output_revision
            ),
            work_run_revision=self.work_run_revision,
            work_run_status=(
                "failed"
                if hard
                else "interrupted"
                if interrupted
                else "completed"
                if passed
                else "turn_limit_reached"
                if soft_nonpass
                else "active"
            ),
            work_run_reason=(
                technical_reason
                if interrupted or hard
                else "verification_passed"
                if passed
                else "turn_limit_reached"
                if soft_nonpass
                else None
            ),
            submitted_attempt_id="submit-attempt",
            acceptance_progress_revision=(
                prepared.context.acceptance_progress_revision
            ),
            output_revision=prepared.context.locked_output_window.output_revision,
            window_revision=self.window_revision,
            delivery_id="delivery-1" if passed else None,
            resolved_result=result,
            budget_transition=budget_transition,
        )

    def _budget_transition(self, delta: float):
        before = {
            WorkRunBudgetDisposition.WITHIN_LIMIT: 0.0,
            WorkRunBudgetDisposition.SOFT_LIMIT_REACHED: 719.0,
            WorkRunBudgetDisposition.HARD_LIMIT_REACHED: 899.0,
        }[self.settlement_disposition]
        return charge_work_run_active_seconds(
            WorkRunBudget(active_seconds_consumed=before),
            active_seconds_delta=delta,
        )


def test_prepare_model_commit_are_separate_and_physical_repair_is_one_logical_call():
    store = FakeVerificationStore()
    provider_calls: list[str] = []
    repair_messages_seen: list[object] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        assert store.in_transaction is False
        provider_calls.append(str(kwargs["model_call_id"]))
        repair_messages_seen.append(kwargs.get("repair_messages"))
        reply = "not-json" if len(provider_calls) == 1 else _reply()
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert result.outcome == "passed"
    assert result.model_call is not None
    assert result.model_call.physical_attempts == 2
    assert provider_calls == [result.model_call.model_call_id] * 2
    assert repair_messages_seen[0] is None
    repair_messages = repair_messages_seen[1]
    assert isinstance(repair_messages, list)
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert store.calls == ["prepare", "commit"]
    assert len(store.commit_commands) == 1
    assert store.interrupt_commands == []
    assert store.start_commands == []
    prepare_payload = store.prepare_commands[0].model_dump(mode="json")
    assert "next_attempt" not in prepare_payload
    assert "catalog_snapshot" not in json.dumps(prepare_payload)
    commit_payload = store.commit_commands[0].model_dump(mode="json")
    assert "locked_output_window" not in commit_payload
    assert "supporting_tool_results" not in commit_payload
    assert "# 旅行计划" not in json.dumps(commit_payload, ensure_ascii=False)


def test_non_pass_persists_typed_feedback_then_starts_next_attempt_without_caller_feedback():
    store = FakeVerificationStore()

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        assert store.in_transaction is False
        return _model_result(_reply("insufficient_evidence"), str(kwargs["model_call_id"]))

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert result.outcome == "not_passed"
    assert result.store_projection.resolved_result is not None
    feedback = result.store_projection.resolved_result.acceptance_results[0]
    assert feedback.verdict is VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert result.next_attempt_mutation is not None
    assert result.next_attempt_mutation.current_attempt_id == (
        "attempt-after-verification"
    )
    assert store.calls == ["prepare", "commit", "start_next_attempt"]
    start = store.start_commands[0]
    assert start.expected_work_run_revision == result.store_projection.work_run_revision
    assert start.expected_window_revision == result.store_projection.window_revision
    assert "feedback" not in start.model_dump(mode="json")


def test_passed_node_acceptance_can_be_routed_to_same_work_run_next_attempt_by_downstream_gate():
    store = FakeVerificationStore()
    gate_calls: list[NodeVerificationResult] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(_reply("passed"), str(kwargs["model_call_id"]))

    def downstream_gate(prepared, node_result):
        gate_calls.append(node_result)
        assert prepared.context.subject == node_result.subject
        assert node_result.all_pass is True
        return (
            DownstreamVerificationFeedback(
                gate_id="whole_task_delivery",
                disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
                finding="根正文遗漏发布步骤。",
                repair_objective="保留正确内容并补齐发布步骤。",
                source_result_id="candidate-review-result-1",
                source_result_sha256="e" * 64,
                affected_subject_ids=(node_result.subject.node_id,),
            ),
        )

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        downstream_gate=downstream_gate,
    )

    assert len(gate_calls) == 1
    assert result.outcome == "not_passed"
    assert result.next_attempt_mutation is not None
    committed = store.commit_commands[0].verification_result
    assert committed.acceptance_results[0].verdict is VerificationVerdict.PASSED
    assert committed.downstream_results[0].disposition is (
        DownstreamVerificationDisposition.RETRY_ATTEMPT
    )
    assert store.calls == ["prepare", "commit", "start_next_attempt"]


def test_settled_non_pass_replay_skips_model_and_replays_same_next_attempt():
    store = FakeVerificationStore()

    def non_pass(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(_reply("not_satisfied"), str(kwargs["model_call_id"]))

    first = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(non_pass),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )
    provider_calls = 0

    def must_not_run(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("settled replay must not call the provider")

    replay = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(must_not_run),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert first.outcome == replay.outcome == "not_passed"
    assert replay.replayed_without_model is True
    assert replay.model_call is None
    assert replay.store_projection.status == "replayed"
    assert replay.next_attempt_mutation is not None
    assert replay.next_attempt_mutation.status == "replayed"
    assert provider_calls == 0
    assert [command.plan.apply_id for command in store.start_commands] == [
        "start-after-verification",
        "start-after-verification",
    ]


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("malformed", "verification_output_invalid"),
        ("provider", "verification_unavailable"),
        ("module", "runtime_module_error"),
    ],
)
def test_verifier_failure_interrupts_same_request_without_starting_attempt(
    failure: str,
    expected_reason: str,
):
    store = FakeVerificationStore()

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        assert store.in_transaction is False
        if failure == "provider":
            raise ModelGatewayError(
                "MODEL_CONFIGURATION_ERROR",
                "provider unavailable",
                retryable=False,
            )
        return _model_result("not-json", str(kwargs["model_call_id"]))

    def module_failure(*_args: object, **_kwargs: object):
        raise RuntimeError("verifier module crashed")

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        verifier=module_failure if failure == "module" else request_node_verification,
    )

    assert result.outcome == "interrupted"
    assert result.interruption_reason == expected_reason
    assert result.model_call is None
    assert result.next_attempt_mutation is None
    assert store.interrupted is True
    assert store.settled is None
    assert len(store.interrupt_commands) == 1
    assert store.interrupt_commands[0].verification_request_id == "verification-1"
    assert store.context.submitted_attempt_id == "submit-attempt"


def test_context_budget_failure_interrupts_verification_then_propagates() -> None:
    store = FakeVerificationStore()
    provider_calls = 0

    def over_budget(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise ContextBudgetExceeded(limit=1000, estimated_tokens=1001)

    with pytest.raises(ContextBudgetExceeded):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(over_budget),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
        )

    assert provider_calls == 1
    assert store.interrupted is True
    assert store.settled is None
    assert len(store.interrupt_commands) == 1
    assert store.interrupt_commands[0].reason == "context_budget_exceeded"
    assert store.start_commands == []


def test_downstream_context_budget_failure_interrupts_then_propagates() -> None:
    store = FakeVerificationStore()
    provider_calls = 0
    gate_calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(_reply("passed"), str(kwargs["model_call_id"]))

    def downstream_gate(
        _prepared: PreparedNodeVerification,
        _node_result: NodeVerificationResult,
    ) -> tuple[DownstreamVerificationFeedback, ...]:
        nonlocal gate_calls
        gate_calls += 1
        raise ContextBudgetExceeded(limit=1_000, estimated_tokens=1_001)

    with pytest.raises(ContextBudgetExceeded):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
            downstream_gate=downstream_gate,
        )

    assert provider_calls == 1
    assert gate_calls == 1
    assert store.calls == ["prepare", "interrupt"]
    assert store.interrupted is True
    assert store.settled is None
    assert len(store.interrupt_commands) == 1
    assert store.interrupt_commands[0].reason == "context_budget_exceeded"
    assert store.start_commands == []


def test_over_limit_input_interrupts_without_provider_or_semantic_result():
    store = FakeVerificationStore()
    request = _request().model_copy(
        update={"input_limits": _input_limits(max_serialized_utf8_bytes=1)}
    )
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit input must not reach the provider")

    result = run_node_verification(
        request,
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert result.outcome == "interrupted"
    assert result.interruption_reason == "verification_input_too_large"
    assert result.model_call is None
    assert result.store_projection.resolved_result is None
    assert provider_calls == 0
    assert store.calls == ["prepare", "interrupt"]
    assert store.commit_commands == []
    assert store.interrupt_commands[0].reason == "verification_input_too_large"


def test_unsupported_input_interrupts_without_provider_or_semantic_result():
    store = FakeVerificationStore()
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("unsupported input must not reach the provider")

    def unsupported(context: NodeVerificationContext, **_kwargs: object):
        raise NodeVerificationInputUnsupported(
            reason="not_canonical_json_utf8",
            limits=context.input_limits,
        )

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        verifier=unsupported,
    )

    assert result.outcome == "interrupted"
    assert result.interruption_reason == "verification_input_unsupported"
    assert result.store_projection.resolved_result is None
    assert provider_calls == 0
    assert store.commit_commands == []
    assert store.interrupt_commands[0].reason == "verification_input_unsupported"


def test_request_recovery_rebinds_the_same_closed_submit_across_turns():
    store = FakeVerificationStore()

    def unavailable(_system: str, _user: str, **_kwargs: object) -> ModelResult:
        raise ModelGatewayError("MODEL_CALL_TIMEOUT", "timeout", retryable=False)

    interrupted = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(unavailable),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )
    assert interrupted.outcome == "interrupted"
    with pytest.raises(FakeStoreConflict, match="explicit resume"):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail("provider called")
            ),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
        )

    resumed_provider_ids: list[str] = []
    resumed_payloads: list[dict[str, object]] = []

    def passes(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        assert store.in_transaction is False
        resumed_provider_ids.append(str(kwargs["model_call_id"]))
        resumed_payloads.append(json.loads(user_content))
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    resumed = resume_and_run_node_verification(
        _resume_request(),
        store=store,
        provider=as_prepared_test_provider(passes),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert resumed.outcome == "passed"
    assert resumed.store_projection.submitted_attempt_id == "submit-attempt"
    assert resumed.model_call is not None
    assert resumed_provider_ids == [resumed.model_call.model_call_id]
    assert resumed_payloads[0]["bindings"]["request_turn_id"] == "turn-1"
    assert resumed_payloads[0]["bindings"]["locked_work_run_revision"] == 3
    assert "invocation_turn_id" not in resumed_payloads[0]["bindings"]
    assert store.calls == ["prepare", "interrupt", "prepare", "resume", "commit"]
    assert store.prepared is not None
    assert store.prepared.rebound_for_recovery is True
    assert store.resume_commands[0].input_limits == _resume_request().input_limits
    assert store.prepared.context.input_limits == _resume_request().input_limits


def test_stale_prepare_fails_before_provider_and_without_interrupt_write():
    store = FakeVerificationStore()
    store.prepare_error = FakeStoreConflict("stale WorkRun revision")
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError

    with pytest.raises(FakeStoreConflict, match="stale WorkRun"):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
        )

    assert provider_calls == 0
    assert store.calls == ["prepare"]
    assert store.interrupt_commands == []


def test_late_model_result_loses_request_revision_fence_without_interrupt_fallback():
    store = FakeVerificationStore()

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        assert store.in_transaction is False
    # 模拟旧物理响应仍在传输时，另一项调用中断并恢复当前请求。
        store.request_revision += 1
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    with pytest.raises(FakeStoreConflict, match="late verification result"):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
        )

    assert store.calls == ["prepare", "commit"]
    assert len(store.commit_commands) == 1
    assert store.interrupt_commands == []
    assert store.settled is None


def test_misbound_late_model_result_fails_closed_without_interrupt_fallback():
    store = FakeVerificationStore()

    def stale_verifier(context, **kwargs):  # type: ignore[no-untyped-def]
        requested = request_node_verification(
            context,
            provider=as_prepared_test_provider(
                lambda _system, _user, **provider_kwargs: _model_result(
                    _reply(),
                    str(provider_kwargs["model_call_id"]),
                )
            ),
            emit=kwargs["emit"],
            deadline=kwargs.get("deadline"),
        )
        return replace(
            requested,
            value=requested.value.model_copy(
                update={
                    "verification_request_revision": (
                        context.verification_request_revision + 1
                    )
                }
            ),
        )

    with pytest.raises(
        RuntimeError,
        match="semantic verifier returned a stale or misbound result",
    ):
        run_node_verification(
            _request(),
            store=store,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: pytest.fail("provider called")
            ),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
            verifier=stale_verifier,
        )

    assert store.calls == ["prepare"]
    assert store.commit_commands == []
    assert store.interrupt_commands == []


def test_pass_does_not_start_attempt_or_publish_a_delivery():
    store = FakeVerificationStore()

    result = run_node_verification(
        _request(),
        store=store,
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(), str(kwargs["model_call_id"])
            )
        ),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert result.outcome == "passed"
    assert result.next_attempt_mutation is None
    assert store.start_commands == []
    assert not hasattr(store, "publish")
