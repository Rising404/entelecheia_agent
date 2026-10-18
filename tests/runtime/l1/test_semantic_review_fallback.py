"""审查不可用时提交原候选；保留来源校验、调用账本与提交幂等性。"""

import json
from dataclasses import replace

import pytest

from personagraph.model_io.api_quota_controller import (
    ApiQuotaWaitTimeout,
    PreparedApiQuotaRequest,
)
from personagraph.model_io.api_quota_queue import QuotaLimits
from personagraph.model_io.gateway import (
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,
)
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.l1 import model, semantic_verification
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.session import store
from tests.helpers.prepared_model_provider import as_prepared_test_provider


CANDIDATE = "二加二等于四。"


def _result(payload, call_id):
    return ModelResult(
        reply=json.dumps(payload, ensure_ascii=False),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(call_id),
    )


@pytest.fixture
def candidate_turn(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = store.create_session("Entelecheia", working_dir=str(workspace))
    decisions = []

    def classify(*_args, **kwargs):
        return _result({"processing_level": "L1"}, kwargs["model_call_id"])

    def decide(*_args, **kwargs):
        decisions.append(kwargs["model_call_id"])
        return _result(
            {
                "plan": {
                    "objective": "回答算术问题",
                    "acceptances": [{"criterion": "回答二加二"}],
                },
                "action": {"kind": "submit_final_reply", "reply": CANDIDATE},
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        ingress_model, "complete_structured", as_prepared_test_provider(classify)
    )
    monkeypatch.setattr(
        model,
        "complete_structured",
        as_prepared_test_provider(decide, add_l1_notes=True),
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )

    def run():
        return entry.run_entry_turn(
            user_input="二加二等于几？",
            session_id=session_id,
            client_request_id="review-unavailable",
            features={
                "context_guard_limit": 24000,
                "l1_semantic_verification_mode": "always",
            },
            routing_policy=freeze_turn_routing_policy(
                TurnRoutingPolicy(l1_enabled=True), source="request_override"
            ),
            store=store,
        )

    return session_id, decisions, run


def _assert_delivered(candidate_turn, error_code):
    session_id, decisions, run = candidate_turn
    result = run()
    assert result.status == "completed", result
    assert result.reply == CANDIDATE
    assert len(decisions) == 1
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution["run"]["status"] == "completed"
    assert execution["state"]["attempts_started"] == 1
    receipt = json.loads(execution["state"]["semantic_verification_report_json"])
    assert receipt["disposition"] == "unavailable"
    assert receipt["unavailable_error_code"] == error_code
    assert receipt["reviewer_result"] is None
    assert receipt["reviewer_result_hash"] is None
    messages = store.get_history_messages(session_id)
    assert (
        sum(
            message["role"] == "assistant" and message["content"] == CANDIDATE
            for message in messages
        )
        == 1
    )
    assert run() == result
    assert len(decisions) == 1
    assert store.get_history_messages(session_id) == messages
    return store.get_runtime_model_logical_call(
        session_id=session_id, logical_call_id=receipt["reviewer_logical_call_id"]
    )


@pytest.mark.parametrize("wait_timeout", [False, True])
def test_review_quota_failure_delivers_without_provider_dispatch(
    candidate_turn, monkeypatch, tmp_path, wait_timeout
):
    if wait_timeout:

        def timed_out(*_args, **_kwargs):
            raise ApiQuotaWaitTimeout("synthetic quota wait timeout")

        monkeypatch.setattr(PreparedApiQuotaRequest, "acquire", timed_out)

    def provider(*_args, **_kwargs):
        pytest.fail("quota-rejected review must not reach the provider")

    def prepare(*_args, **_kwargs):
        return PreparedModelCall(
            _dispatch=provider,
            api_quota=PreparedApiQuotaRequest(
                scope_hash="a" * 64,
                limits=QuotaLimits(10, 300000, 1000000000, 2),
                token_reservation=327541,
                provider_timeout_seconds=30,
                database_path=str(tmp_path / "quota.sqlite3"),
                request_identity="b" * 64,
            ),
        )

    provider.prepare = prepare
    monkeypatch.setattr(semantic_verification, "complete_structured", provider)
    logical = _assert_delivered(
        candidate_turn,
        "MODEL_QUOTA_WAIT_TIMEOUT" if wait_timeout else "MODEL_QUOTA_ADMISSION_FAILED",
    )
    assert logical.physical_attempts == ()


@pytest.mark.parametrize(
    "code,status,retryable",
    [
        ("MODEL_CALL_TIMEOUT", None, True),
        ("MODEL_CALL_FAILED", 429, True),
        ("MODEL_CALL_FAILED", 401, False),
        ("MODEL_CALL_FAILED", 503, True),
    ],
)
def test_review_provider_failure_delivers_original_candidate(
    candidate_turn, monkeypatch, code, status, retryable
):
    def fail(*_args, **_kwargs):
        raise ModelGatewayError(
            code,
            "synthetic provider failure",
            retryable=retryable,
            details={"status_code": status},
        )

    monkeypatch.setattr(
        semantic_verification, "complete_structured", as_prepared_test_provider(fail)
    )
    logical = _assert_delivered(candidate_turn, code)
    assert logical.physical_attempts[-1].settlement.outcome.value == "terminal_failure"
    assert logical.physical_attempts[-1].settlement.error_code == code


def test_review_exhausted_output_repairs_are_unavailable_not_passed(
    candidate_turn, monkeypatch
):
    def malformed(*_args, **kwargs):
        return replace(_result({}, kwargs["model_call_id"]), reply="not JSON")

    monkeypatch.setattr(
        semantic_verification,
        "complete_structured",
        as_prepared_test_provider(malformed),
    )
    logical = _assert_delivered(candidate_turn, "MODEL_BAD_RESPONSE")
    assert len(logical.physical_attempts) > 1
    assert all(
        attempt.settlement.typed_result is None for attempt in logical.physical_attempts
    )


@pytest.mark.parametrize(
    "code", ["MODEL_CONFIGURATION_FAILURE", "MODEL_QUOTA_RECONCILIATION_REQUIRED"]
)
def test_host_authority_errors_do_not_become_review_fallback(
    candidate_turn, monkeypatch, code
):
    def fail(*_args, **_kwargs):
        raise ModelGatewayError(code, "synthetic authority failure", retryable=False)

    monkeypatch.setattr(
        semantic_verification, "complete_structured", as_prepared_test_provider(fail)
    )
    session_id, _, run = candidate_turn
    result = run()
    assert result.error_code is not None
    assert result.reply != CANDIDATE
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution["state"]["semantic_verification_report_json"] is None
    assert all(
        message["content"] != CANDIDATE
        for message in store.get_history_messages(session_id)
    )


def test_store_rejects_disguising_valid_review_as_unavailable(
    candidate_turn, monkeypatch
):
    def passed(*_args, **kwargs):
        return _result({"issues": []}, kwargs["model_call_id"])

    monkeypatch.setattr(
        semantic_verification, "complete_structured", as_prepared_test_provider(passed)
    )
    original = store.commit_l1_attempt_decision

    def tamper(**kwargs):
        receipt = json.loads(kwargs["semantic_verification_report_json"])
        receipt.update(
            disposition="unavailable",
            reviewer_result=None,
            reviewer_result_hash=None,
            unavailable_error_code="MODEL_QUOTA_ADMISSION_FAILED",
        )
        kwargs.update(
            semantic_verification_report_json=canonical_json(receipt),
            semantic_verification_report_hash=sha256_json(receipt),
        )
        return original(**kwargs)

    monkeypatch.setattr(store, "commit_l1_attempt_decision", tamper)
    _, _, run = candidate_turn
    result = run()
    assert result.status == "incomplete"
    assert result.reply is None
