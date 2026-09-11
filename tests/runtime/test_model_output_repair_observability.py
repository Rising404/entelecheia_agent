from __future__ import annotations

import hashlib
from importlib import import_module
import json

import personagraph.trajectory as trajectory_api
import pytest

from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssueCoverage,
    RuntimeModelOutputRepairIssue,
)
from personagraph.runtime.model_calls import (
    request_model_with_retry,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.runtime.turn_events import RuntimeStage


@pytest.mark.parametrize("module_name,function_name,expected", [
    ("personagraph.runtime.model_calls.observability", "runtime_error_code", "MODEL_TRANSPORT_FAILURE"),
    ("personagraph.runtime.entry.application", "_public_model_error_code", "MODEL_TRANSPORT_FAILURE"),
    ("personagraph.runtime.model_calls.session_summary", "_summary_model_failure_code", "SUMMARY_MODEL_TRANSPORT_FAILURE"),
    ("personagraph.api.service.views", "model_error_http_status", 503),
])
def test_terminal_transport_is_not_reclassified_as_configuration_failure(
    module_name, function_name, expected,
):
    error = ModelGatewayError(
        "MODEL_CALL_FAILED", "provider unavailable", retryable=False,
        details={"physical_error_retryable": True},
    )
    classify = getattr(import_module(module_name), function_name)

    assert classify(error) == expected
    assert error.retryable is False


def _rejection_observation(values: dict[str, object]) -> dict[str, object]:
    return json.loads(str(values["rejected"]))


def test_host_rejection_trajectory_records_safe_metadata_not_response_body(
    monkeypatch,
) -> None:
    rejected_body = '{"secret_user_excerpt":"不要在诊断副本中重复我"}'
    recorded: list[dict[str, object]] = []
    provider_calls = 0

    monkeypatch.setattr(
        trajectory_api,
        "record_rejected_output",
        lambda **values: recorded.append(values),
    )

    class Prepared:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            nonlocal provider_calls
            provider_calls += 1
            return ModelResult(
                reply=self.reply,
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    def prepare_repair_request(
        _feedback: RuntimeModelOutputRepairFeedback,
        _rejected_response_text: str,
    ) -> Prepared:
        return Prepared('{"ok":true}')

    def validate(result: ModelResult) -> bool:
        if result.reply == rejected_body:
            raise ModelOutputValidationError(
                "missing required field",
                repair_code="test.missing",
                safe_repair_reason="目标合同要求 /ok 字段。",
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category="schema",
                        code="schema.missing",
                        paths=("/ok",),
                        safe_explanation="目标合同要求此位置必须存在。",
                    ),
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.COMPLETE
                ),
            )
        return bool(json.loads(result.reply)["ok"])

    result = request_model_with_retry(
        turn_id="turn-observe-rejection",
        session_id="session-observe-rejection",
        purpose="observe-rejection",
        stage=RuntimeStage.VERIFICATION,
        prepare_request=lambda: Prepared(rejected_body),
        prepare_repair_request=prepare_repair_request,
        repair_target_contract="test-contract-v1",
        validate=validate,
        emit=lambda _event: None,
    )

    assert result.value is True
    assert provider_calls == 2
    assert len(recorded) == 1
    observation = json.loads(str(recorded[0]["rejected"]))
    assert observation == {
        "schema_version": "runtime-model-output-rejection-observation-v1",
        "purpose": "observe-rejection",
        "logical_model_call_id": result.model_call_id,
        "physical_ordinal": 1,
        "rejected_response_sha256": hashlib.sha256(
            rejected_body.encode("utf-8")
        ).hexdigest(),
        "repair_scheduled": True,
        "target_contract": "test-contract-v1",
        "issue_coverage": "complete",
        "omitted_issue_count": 0,
        "issues": [
            {
                "category": "schema",
                "code": "schema.missing",
                "paths": ["/ok"],
                "safe_explanation": "目标合同要求此位置必须存在。",
            }
        ],
    }
    assert rejected_body not in str(recorded[0]["rejected"])


def test_final_allowed_attempt_does_not_claim_another_repair_is_scheduled(
    monkeypatch,
) -> None:
    recorded: list[dict[str, object]] = []
    provider_calls = 0

    monkeypatch.setattr(
        trajectory_api,
        "record_rejected_output",
        lambda **values: recorded.append(values),
    )

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            nonlocal provider_calls
            provider_calls += 1
            return ModelResult(
                reply='{"ok":false}',
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-final-rejection",
            session_id="session-final-rejection",
            purpose="observe-final-rejection",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=Prepared,
            prepare_repair_request=lambda _feedback, _body: Prepared(),
            repair_target_contract="test-contract-v1",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError("still invalid")
            ),
            emit=lambda _event: None,
            max_attempts=1,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert provider_calls == 1
    assert len(recorded) == 1
    assert _rejection_observation(recorded[0])["repair_scheduled"] is False


def test_expired_deadline_does_not_claim_another_repair_is_scheduled(
    monkeypatch,
) -> None:
    recorded: list[dict[str, object]] = []
    provider_calls = 0

    monkeypatch.setattr(
        trajectory_api,
        "record_rejected_output",
        lambda **values: recorded.append(values),
    )

    class Deadline:
        is_expired = False

        def expired(self) -> bool:
            return self.is_expired

        def remaining_s(self) -> float:
            return 0.0 if self.is_expired else 60.0

    deadline = Deadline()

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            nonlocal provider_calls
            provider_calls += 1
            deadline.is_expired = True
            return ModelResult(
                reply='{"ok":false}',
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-expired-after-rejection",
            session_id="session-expired-after-rejection",
            purpose="observe-expired-rejection",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=Prepared,
            prepare_repair_request=lambda _feedback, _body: Prepared(),
            repair_target_contract="test-contract-v1",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError("still invalid")
            ),
            emit=lambda _event: None,
            deadline=deadline,  # type: ignore[arg-type]
        )

    assert captured.value.code == "TURN_DEADLINE_EXCEEDED"
    assert provider_calls == 1
    assert len(recorded) == 1
    assert _rejection_observation(recorded[0])["repair_scheduled"] is False


def test_non_retryable_rejection_does_not_claim_repair_is_scheduled(
    monkeypatch,
) -> None:
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        trajectory_api,
        "record_rejected_output",
        lambda **values: recorded.append(values),
    )

    class Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply='{"ok":false}',
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )

    with pytest.raises(ModelGatewayError) as captured:
        request_model_with_retry(
            turn_id="turn-non-retryable-rejection",
            session_id="session-non-retryable-rejection",
            purpose="observe-non-retryable-rejection",
            stage=RuntimeStage.VERIFICATION,
            prepare_request=Prepared,
            prepare_repair_request=lambda _feedback, _body: Prepared(),
            repair_target_contract="test-contract-v1",
            validate=lambda _result: (_ for _ in ()).throw(
                ModelOutputValidationError("not repairable", retryable=False)
            ),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert captured.value.retryable is False
    assert len(recorded) == 1
    assert _rejection_observation(recorded[0])["repair_scheduled"] is False
