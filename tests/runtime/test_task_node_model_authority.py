from __future__ import annotations

import json

import pytest

from personagraph.model_io.gateway import ModelResult, PreparedModelCall
from personagraph.runtime.model_calls import (
    DurableModelCallStateGuardRejected,
)
from personagraph.runtime.model_calls import request_model_with_retry
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
)
from personagraph.runtime.model_calls import (
    RuntimeModelCallWaitingExternal,
)
from personagraph.l2.task_execution.task_node.model_authority import (
    TaskNodeBoundModelCall,
    TaskNodeModelAuthorityFactoryError,
    create_task_node_work_run_model_call_authority,
    task_node_durable_provider_prompt,
)
from personagraph.runtime.turn_events import RuntimeStage
from personagraph.l2.work_run import (
    AcceptanceVerificationFeedback,
    AttemptDecision,
    NodeVerificationResult,
    OutputWindowFormat,
    SubmitOutputWindowAction,
    TaskNodeSubject,
    VerificationVerdict,
)
from tests.runtime.test_runtime_model_call_authority import _MemoryLedger


SHA_A = "a" * 64
SHA_B = "b" * 64
CURRENT_REPAIR_PROTOCOL = (
    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
)


def _subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task_task_node_ledger_01",
        graph_revision=3,
        node_id="node_task_node_ledger_01",
        node_revision=2,
    )


def _attempt_user_content(*, turn_id: str, work_run_revision: int) -> str:
    return json.dumps(
        {
            "bindings": {
                "session_id": "session_task_node_ledger_01",
                "turn_id": turn_id,
                "work_run_id": "work_run_task_node_ledger_01",
                "work_run_revision": work_run_revision,
                "attempt_id": "attempt_task_node_ledger_01",
                "attempt_ordinal": 1,
            },
            "node": {
                "subject": _subject().model_dump(mode="json"),
                "objective": "produce one bounded result",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _attempt_binding(
    *,
    turn_id: str = "turn_task_node_ledger_01",
    work_run_revision: int = 2,
    state_guard_sha256: str = SHA_A,
    system_prompt: str = "task node attempt system",
) -> TaskNodeBoundModelCall:
    return TaskNodeBoundModelCall.create(
        call_kind="attempt_decision",
        logical_call_id="task_node_attempt_model_call_01",
        session_id="session_task_node_ledger_01",
        subject=_subject(),
        request_turn_id="turn_task_node_ledger_01",
        invocation_turn_id=turn_id,
        work_run_id="work_run_task_node_ledger_01",
        dispatch_work_run_revision=work_run_revision,
        attempt_id="attempt_task_node_ledger_01",
        attempt_ordinal=1,
        verification_request_id=None,
        verification_request_revision=None,
        locked_work_run_revision=None,
        system_prompt=system_prompt,
        user_content=_attempt_user_content(
            turn_id=turn_id,
            work_run_revision=work_run_revision,
        ),
        state_guard_sha256=state_guard_sha256,
    )


def _verification_binding(
    *,
    turn_id: str = "turn_task_node_ledger_02",
    dispatch_work_run_revision: int = 7,
    verification_request_revision: int = 2,
    state_guard_sha256: str = SHA_A,
    system_prompt: str = "task node verification system",
) -> TaskNodeBoundModelCall:
    subject = _subject()
    user_content = json.dumps(
        {
            "bindings": {
                "session_id": "session_task_node_ledger_01",
                "request_turn_id": "turn_task_node_ledger_01",
                "verification_request_id": "verification_task_node_ledger_01",
                "work_run_id": "work_run_task_node_ledger_01",
                "locked_work_run_revision": 5,
                "submitted_attempt_id": "attempt_task_node_ledger_01",
                "acceptance_progress_revision": 2,
                "subject": subject.model_dump(mode="json"),
                "output_revision": 2,
                "task_id": subject.task_id,
                "graph_revision": subject.graph_revision,
                "node_id": subject.node_id,
                "node_revision": subject.node_revision,
            },
            "node": {"acceptances": [{"acceptance_id": "acceptance_01"}]},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return TaskNodeBoundModelCall.create(
        call_kind="node_verification",
        logical_call_id="task_node_verification_model_call_01",
        session_id="session_task_node_ledger_01",
        subject=subject,
        request_turn_id="turn_task_node_ledger_01",
        invocation_turn_id=turn_id,
        work_run_id="work_run_task_node_ledger_01",
        dispatch_work_run_revision=dispatch_work_run_revision,
        attempt_id="attempt_task_node_ledger_01",
        attempt_ordinal=1,
        verification_request_id="verification_task_node_ledger_01",
        verification_request_revision=verification_request_revision,
        locked_work_run_revision=5,
        system_prompt=system_prompt,
        user_content=user_content,
        state_guard_sha256=state_guard_sha256,
    )


@pytest.mark.parametrize(
    "binding",
    (
        _attempt_binding(system_prompt="精确的 TaskNode Attempt 系统提示。"),
        _verification_binding(
            system_prompt="精确的 TaskNode Verification 系统提示。"
        ),
    ),
)
def test_new_task_node_call_freezes_v2_protocol_and_exact_prompt(
    binding: TaskNodeBoundModelCall,
) -> None:
    authority = create_task_node_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )

    logical = authority.logical_request
    assert logical.output_repair_protocol is CURRENT_REPAIR_PROTOCOL
    assert logical.structured_prompt is not None
    assert logical.structured_prompt.system_prompt == binding.system_prompt
    assert logical.structured_prompt.user_content == binding.user_content
    assert task_node_durable_provider_prompt(
        durable=authority,
        invocation_turn_id=binding.invocation_turn_id,
        system_prompt="不得使用的升级后系统提示。",
        user_content='{"changed":true}',
    ) == (binding.system_prompt, binding.user_content)


def test_retryable_attempt_continuation_keeps_logical_origin_and_opens_new_physical() -> None:
    ledger = _MemoryLedger()
    origin = _attempt_binding()
    authority = create_task_node_work_run_model_call_authority(
        origin,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    authority.reserve(turn_id=origin.invocation_turn_id)
    first = authority.begin_physical_attempt(
        turn_id=origin.invocation_turn_id,
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    authority.settle_physical_attempt(
        turn_id=origin.invocation_turn_id,
        physical=first,
        outcome="retryable_failure",
        result_fingerprint=SHA_B,
        error_code="provider_retryable",
    )

    continued = _attempt_binding(
        turn_id="turn_task_node_ledger_02",
        work_run_revision=4,
        state_guard_sha256=SHA_B,
    )
    resumed = create_task_node_work_run_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_B,
        ledger_store=ledger,
    )
    second = resumed.begin_physical_attempt(
        turn_id=continued.invocation_turn_id,
        max_physical_attempts=6,
        output_repair_enabled=True,
    )

    assert resumed.logical_request == authority.logical_request
    assert second.logical_call_id == first.logical_call_id
    assert second.physical_ordinal == 2
    assert second.started_turn_id == continued.invocation_turn_id
    assert second.physical_attempt_id != first.physical_attempt_id


@pytest.mark.parametrize("outcome", ["pending", "uncertain"])
def test_pending_or_uncertain_attempt_never_blindly_redispatches(
    outcome: str,
) -> None:
    ledger = _MemoryLedger()
    binding = _attempt_binding()
    authority = create_task_node_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    authority.reserve(turn_id=binding.invocation_turn_id)
    physical = authority.begin_physical_attempt(
        turn_id=binding.invocation_turn_id,
        max_physical_attempts=6,
        output_repair_enabled=True,
    )
    if outcome == "uncertain":
        authority.settle_physical_attempt(
            turn_id=binding.invocation_turn_id,
            physical=physical,
            outcome="uncertain",
            result_fingerprint=SHA_B,
            provider_request_id="provider_request_uncertain_01",
            error_code="provider_outcome_uncertain",
        )

    with pytest.raises(RuntimeModelCallWaitingExternal):
        request_model_with_retry(
            turn_id="turn_task_node_ledger_02",
            session_id=binding.session_id,
            purpose=binding.purpose,
            stage=RuntimeStage.L2_PLAN,
            prepare_request=lambda: PreparedModelCall(
                _dispatch=lambda _model_call_id: pytest.fail(
                    "pending/uncertain TaskNode call reached Provider"
                )
            ),
            prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
                _dispatch=lambda _model_call_id: pytest.fail(
                    "pending/uncertain TaskNode call reached Provider"
                )
            ),
            validate=lambda result: AttemptDecision.model_validate_json(
                result.reply
            ),
            emit=lambda _event: None,
            durable_call=create_task_node_work_run_model_call_authority(
                _attempt_binding(
                    turn_id="turn_task_node_ledger_02",
                    work_run_revision=4,
                    state_guard_sha256=SHA_B,
                ),
                rederive_state_guard_sha256=lambda: SHA_B,
                ledger_store=ledger,
            ),
        )
    assert ledger.logical is not None
    assert len(ledger.logical.physical_attempts) == 1


def test_attempt_typed_success_replays_without_provider_and_state_drift_fails_closed() -> None:
    ledger = _MemoryLedger()
    binding = _attempt_binding()
    decision = AttemptDecision(
        action=SubmitOutputWindowAction(
            content="durable task node output",
            format=OutputWindowFormat.MARKDOWN,
        )
    )
    provider_calls = 0

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=decision.model_dump_json(),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    fresh_authority = create_task_node_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    fresh = request_model_with_retry(
        turn_id=binding.invocation_turn_id,
        session_id=binding.session_id,
        purpose=binding.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(_dispatch=provider),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=provider
        ),
        validate=lambda result: AttemptDecision.model_validate_json(
            result.reply
        ),
        emit=lambda _event: None,
        durable_call=fresh_authority,
    )
    continued = _attempt_binding(
        turn_id="turn_task_node_ledger_02",
        work_run_revision=4,
        state_guard_sha256=SHA_B,
    )
    replay_authority = create_task_node_work_run_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_B,
        ledger_store=ledger,
    )
    replay = request_model_with_retry(
        turn_id=continued.invocation_turn_id,
        session_id=continued.session_id,
        purpose=continued.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail(
                "typed replay reached Provider"
            )
        ),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail(
                "typed replay reached Provider"
            )
        ),
        validate=lambda result: AttemptDecision.model_validate_json(
            result.reply
        ),
        emit=lambda _event: None,
        durable_call=replay_authority,
    )

    assert fresh.value == replay.value == decision
    assert replay.replayed is True
    assert provider_calls == 1
    stale = create_task_node_work_run_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    with pytest.raises(DurableModelCallStateGuardRejected):
        stale.require_current_state()


def test_verification_typed_replay_persists_only_provider_envelope() -> None:
    subject = _subject()
    user_content = json.dumps(
        {
            "bindings": {
                "session_id": "session_task_node_ledger_01",
                "request_turn_id": "turn_task_node_ledger_01",
                "verification_request_id": "verification_task_node_ledger_01",
                "work_run_id": "work_run_task_node_ledger_01",
                "locked_work_run_revision": 5,
                "submitted_attempt_id": "attempt_task_node_ledger_01",
                "acceptance_progress_revision": 2,
                "subject": subject.model_dump(mode="json"),
                "output_revision": 2,
                "task_id": subject.task_id,
                "graph_revision": subject.graph_revision,
                "node_id": subject.node_id,
                "node_revision": subject.node_revision,
            },
            "node": {"acceptances": [{"acceptance_id": "acceptance_01"}]},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    binding = TaskNodeBoundModelCall.create(
        call_kind="node_verification",
        logical_call_id="task_node_verification_model_call_01",
        session_id="session_task_node_ledger_01",
        subject=subject,
        request_turn_id="turn_task_node_ledger_01",
        invocation_turn_id="turn_task_node_ledger_02",
        work_run_id="work_run_task_node_ledger_01",
        dispatch_work_run_revision=7,
        attempt_id="attempt_task_node_ledger_01",
        attempt_ordinal=1,
        verification_request_id="verification_task_node_ledger_01",
        verification_request_revision=2,
        locked_work_run_revision=5,
        system_prompt="task node verification system",
        user_content=user_content,
        state_guard_sha256=SHA_A,
    )
    authority = create_task_node_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=_MemoryLedger(),
    )
    result = NodeVerificationResult(
        verification_request_id="verification_task_node_ledger_01",
        verification_request_revision=2,
        work_run_id="work_run_task_node_ledger_01",
        locked_work_run_revision=5,
        submitted_attempt_id="attempt_task_node_ledger_01",
        acceptance_progress_revision=2,
        subject=subject,
        output_revision=2,
        acceptance_results=(
            AcceptanceVerificationFeedback(
                acceptance_id="acceptance_01",
                verdict=VerificationVerdict.PASSED,
                finding="supported",
            ),
        ),
        all_pass=True,
    )
    payload = authority.typed_result_payload(
        model_result=ModelResult(
            reply="{}",
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id="physical_task_node_verification_01",
        ),
        value=result,
    )

    assert payload == {
        "acceptance_results": [
            result.acceptance_results[0].model_dump(mode="json")
        ]
    }


def test_continuation_rejects_semantic_prompt_drift() -> None:
    ledger = _MemoryLedger()
    origin = _attempt_binding()
    create_task_node_work_run_model_call_authority(
        origin,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    ).reserve(turn_id=origin.invocation_turn_id)
    changed = _attempt_binding(
        turn_id="turn_task_node_ledger_02",
        work_run_revision=4,
        state_guard_sha256=SHA_B,
    )
    payload = json.loads(changed.user_content)
    payload["node"]["objective"] = "silently changed semantic objective"
    changed = TaskNodeBoundModelCall.create(
        call_kind=changed.call_kind,
        logical_call_id=changed.logical_call_id,
        session_id=changed.session_id,
        subject=changed.subject,
        request_turn_id=changed.request_turn_id,
        invocation_turn_id=changed.invocation_turn_id,
        work_run_id=changed.work_run_id,
        dispatch_work_run_revision=changed.dispatch_work_run_revision,
        attempt_id=changed.attempt_id,
        attempt_ordinal=changed.attempt_ordinal,
        verification_request_id=None,
        verification_request_revision=None,
        locked_work_run_revision=None,
        system_prompt=changed.system_prompt,
        user_content=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        state_guard_sha256=changed.state_guard_sha256,
    )
    with pytest.raises(TaskNodeModelAuthorityFactoryError, match="dispatch lease"):
        create_task_node_work_run_model_call_authority(
            changed,
            rederive_state_guard_sha256=lambda: SHA_B,
            ledger_store=ledger,
        )
