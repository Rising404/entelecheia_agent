from __future__ import annotations

from copy import deepcopy
import json

import pytest

from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskNodeKind,
    InSessionTaskSourceAnchor,
    TaskDeliveryValidationDimension,
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationNode,
    TaskDeliveryValidationPrompt,
    TaskDeliveryValidationRequest,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
from personagraph.runtime.model_calls import requests as model_requests
from personagraph.runtime.model_calls import MAX_MODEL_ATTEMPTS
from personagraph.l2.task_execution.delivery.validation import (
    PURPOSE,
    request_task_delivery_validation,
)
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider,
    repair_feedback_from_provider_kwargs,
)


def _request(
    *,
    include_child: bool = False,
    objective: str = "Produce a release-ready answer.",
    root_output_body: str = "Draft answer missing its final release step.",
) -> TaskDeliveryValidationRequest:
    anchor = InSessionTaskSourceAnchor(
        anchor_id="goal",
        source_turn_id="turn-1",
        source_kind="current_user_instruction",
        start=0,
        end=len(objective),
        excerpt=objective,
    )
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="done",
        criterion="The answer is complete and directly usable.",
        source_anchor_ids=("goal",),
    )
    nodes = [
        TaskDeliveryValidationNode(
            node_id="task-root",
            node_revision=1,
            node_kind=InSessionTaskNodeKind.ROOT,
            ordinal=0,
            title="Prepare the answer",
            objective=objective,
            source_anchor_ids=("goal",),
            acceptance_criteria=(acceptance,),
        )
    ]
    if include_child:
        nodes.append(
            TaskDeliveryValidationNode(
                node_id="child-node",
                node_revision=1,
                node_kind=InSessionTaskNodeKind.SUBTASK,
                parent_node_id="task-root",
                ordinal=1,
                title="Draft the release step",
                objective="Draft the evidence-grounded final release step.",
                source_anchor_ids=("goal",),
                acceptance_criteria=(acceptance,),
            )
        )
    prompt = TaskDeliveryValidationPrompt.create(
        session_id="session-1",
        task_id="task-root",
        graph_revision=1,
        task_state_version=4,
        title="Prepare the answer",
        objective=objective,
        source_anchors=(anchor,),
        nodes=tuple(nodes),
        root_delivery_id="delivery-root",
        root_output_format="markdown",
        root_output_body=root_output_body,
    )
    return TaskDeliveryValidationRequest.create(
        verification_request_id="validation-request-v2",
        verification_result_id="validation-result-v2",
        logical_call_id="validation-call-v2",
        verification_profile_id="whole-task-delivery-v2",
        invocation_turn_id="turn-1",
        prompt=prompt,
    )


def _payload() -> dict[str, object]:
    return {
        "findings": [
            {
                "dimension": dimension.value,
                "verdict": (
                    "fail"
                    if dimension
                    is TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY
                    else "pass"
                ),
                "fault_domain": (
                    "execution_output"
                    if dimension
                    is TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY
                    else "none"
                ),
                "finding": (
                    "The root answer omitted the final release step."
                    if dimension
                    is TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY
                    else f"{dimension.value} passed."
                ),
                "affected_node_ids": (
                    ["task-root"]
                    if dimension
                    is TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY
                    else []
                ),
                "evidence_anchor_ids": [],
            }
            for dimension in TaskDeliveryValidationDimension
        ],
        "summary": "The current graph is sound, but the root output needs repair.",
        "execution_repair_objective": (
            "Add the final release step without changing the TaskGraph."
        ),
        "task_graph_revision_objective": None,
        "blocking_questions": [],
    }


def _strictly_invalid_json_reply(
    payload: dict[str, object],
    *,
    defect: str,
) -> str:
    reply = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if defect == "duplicate_top_level_key":
        return reply[:-1] + ',"summary":"duplicate summary"}'
    if defect == "duplicate_nested_key":
        return reply.replace(
            '"dimension":',
            '"dimension":"duplicate","dimension":',
            1,
        )
    if defect in {"NaN", "Infinity", "-Infinity"}:
        return reply.replace(
            '"summary": "',
            f'"summary": {defect}, "summary_remainder": "',
            1,
        )
    raise AssertionError(f"unexpected strict JSON defect: {defect}")


def test_provider_parses_fault_domain_and_host_derives_execution_retry() -> None:
    request = _request()
    seen: dict[str, str] = {}

    def provider(
        system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        seen["system_prompt"] = system_prompt
        seen["purpose"] = purpose
        return ModelResult(
            reply=json.dumps(_payload()),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    requested = request_task_delivery_validation(
        request,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.value.disposition is (
        TaskDeliveryValidationDisposition.RETRY_EXECUTION
    )
    assert requested.value.execution_repair_objective is not None
    assert requested.value.task_graph_revision_objective is None
    assert seen["purpose"] == PURPOSE
    assert "不要输出 disposition" in seen["system_prompt"]
    assert "execution_output" in seen["system_prompt"]
    assert "task_graph_design" in seen["system_prompt"]
    assert "canonical root" in seen["system_prompt"]
    assert "child node" in seen["system_prompt"]
    assert "host_output_repair_feedback" not in seen["system_prompt"]
    assert "node_verification_attestations" in seen["system_prompt"]
    assert "不得重复执行节点级证据验收" in seen["system_prompt"]
    assert "未重复投影原始工具正文" in seen["system_prompt"]
    assert "不是 missing_information" in seen["system_prompt"]


def test_reviewer_is_told_to_audit_literals_and_same_clause_ambiguity() -> None:
    objective = "报告 evaluation milestone question 的数量。"
    root_output_body = (
        "evaluation milestone 数量: 1（500 adjudicated questions）"
    )
    request = _request(
        objective=objective,
        root_output_body=root_output_body,
    )
    payload = _payload()
    for finding in payload["findings"]:
        if finding["dimension"] == "final_delivery_quality":
            finding.update(
                verdict="pass",
                fault_domain="none",
                finding="The prose is otherwise directly usable.",
                affected_node_ids=[],
            )
        elif finding["dimension"] == "factual_correctness":
            finding.update(
                verdict="fail",
                fault_domain="execution_output",
                finding=(
                    "The same clause ambiguously labels one milestone while "
                    "also presenting 500 as the requested question count."
                ),
                affected_node_ids=["task-root"],
            )
    payload["summary"] = "The root must disambiguate the requested count."
    payload["execution_repair_objective"] = (
        "State the number of milestone questions unambiguously."
    )
    seen: dict[str, str] = {}

    def provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        seen["system_prompt"] = system_prompt
        seen["user_content"] = user_content
        return ModelResult(
            reply=json.dumps(payload),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    requested = request_task_delivery_validation(
        request,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.value.disposition is (
        TaskDeliveryValidationDisposition.RETRY_EXECUTION
    )
    projected = json.loads(seen["user_content"])
    assert projected["objective"] == objective
    assert projected["root_output_body"] == root_output_body
    assert "逐字面事实核对" in seen["system_prompt"]
    assert (
        "每一个数字、日期、百分比、计数、比较符和量纲"
        in seen["system_prompt"]
    )
    assert "同一句或同一字段" in seen["system_prompt"]
    assert "数量所指的对象" in seen["system_prompt"]
    assert "根正文与子交付重复同一错误" in seen["system_prompt"]


def test_provider_retries_child_execution_scope_as_task_graph_design() -> None:
    invalid = _payload()
    invalid_finding = next(
        finding
        for finding in invalid["findings"]  # type: ignore[union-attr]
        if finding["fault_domain"] == "execution_output"
    )
    invalid_finding["affected_node_ids"] = ["child-node"]

    corrected = deepcopy(invalid)
    corrected_finding = next(
        finding
        for finding in corrected["findings"]  # type: ignore[union-attr]
        if finding["fault_domain"] == "execution_output"
    )
    corrected_finding["fault_domain"] = "task_graph_design"
    corrected["execution_repair_objective"] = None
    corrected["task_graph_revision_objective"] = (
        "Revise the graph so the defective child work is regenerated."
    )
    calls = 0

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal calls
        calls += 1
        return ModelResult(
            reply=json.dumps(invalid if calls == 1 else corrected),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    requested = request_task_delivery_validation(
        _request(include_child=True),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.value.disposition is (
        TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
    )
    assert requested.value.task_graph_revision_objective is not None


def test_repair_feedback_survives_a_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    secret_marker = "PRIVATE_REJECTED_TASK_DELIVERY_REPLY"
    invalid = _payload()
    invalid.pop("summary")
    invalid["private_extra"] = secret_marker
    seen_payloads: list[dict[str, object]] = []
    seen_feedback: list[dict[str, object] | None] = []
    model_call_ids: list[str] = []

    def provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
        **kwargs: object,
    ) -> ModelResult:
        payload = json.loads(user_content)
        assert isinstance(payload, dict)
        seen_payloads.append(payload)
        seen_feedback.append(repair_feedback_from_provider_kwargs(kwargs))
        model_call_ids.append(model_call_id)
        if len(seen_payloads) == 1:
            reply = json.dumps(invalid, ensure_ascii=False)
        elif len(seen_payloads) == 2:
            raise ModelGatewayError(
                "MODEL_RATE_LIMITED",
                "temporary provider limit",
                retryable=True,
            )
        else:
            reply = json.dumps(_payload(), ensure_ascii=False)
        return ModelResult(
            reply=reply,
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    requested = request_task_delivery_validation(
        _request(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 3
    assert model_call_ids == [requested.model_call_id] * 3
    assert "host_output_repair_feedback" not in seen_payloads[0]
    repair_after_rejection = seen_feedback[1]
    repair_after_transport = seen_feedback[2]
    assert repair_after_transport == repair_after_rejection
    assert isinstance(repair_after_rejection, dict)
    assert set(repair_after_rejection) == {"current_issues"}
    explanations = {
        issue["safe_explanation"] for issue in repair_after_rejection["current_issues"]
    }
    assert "目标合同要求此位置必须存在。" in explanations
    assert "该位置含有目标合同未声明的额外字段。" in explanations
    assert secret_marker not in json.dumps(
        repair_after_rejection,
        ensure_ascii=False,
    )


def test_prepared_delivery_repair_uses_four_message_contract() -> None:
    invalid = _payload()
    invalid.pop("summary")
    invalid["private_extra"] = "PRIVATE_DELIVERY_VALUE"
    invalid_reply = json.dumps(invalid, ensure_ascii=False)
    valid_reply = json.dumps(_payload(), ensure_ascii=False)
    prepared_calls: list[dict[str, object]] = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared delivery provider used direct dispatch")

    def prepare(
        system_prompt: str,
        user_content: str,
        **kwargs: object,
    ) -> PreparedModelCall:
        prepared_calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                **kwargs,
            }
        )
        reply = invalid_reply if len(prepared_calls) == 1 else valid_reply

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return ModelResult(
                reply=reply,
                provider="mock",
                model="mock-structured",
                latency_ms=1,
                model_call_id=model_call_id,
                purpose=PURPOSE,
            )

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare  # type: ignore[attr-defined]
    requested = request_task_delivery_validation(
        _request(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert requested.value.disposition is (
        TaskDeliveryValidationDisposition.RETRY_EXECUTION
    )
    assert len(prepared_calls) == 2
    initial, repair = prepared_calls
    assert "host_output_repair_feedback" not in str(initial["system_prompt"])
    assert initial["system_prompt"] == repair["system_prompt"]
    assert initial["user_content"] == repair["user_content"]
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == invalid_reply
    repair_envelope = json.loads(
        messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert set(repair_envelope) == {"current_issues"}
    assert "当前清单可能不完整" not in messages[3]["content"]
    assert {
        tuple(issue["paths"])
        for issue in repair_envelope["current_issues"]
    } == {("",), ("/summary",)}
    assert "PRIVATE_DELIVERY_VALUE" not in messages[3]["content"]


def test_provider_rejects_model_owned_disposition() -> None:
    invalid = _payload()
    invalid["disposition"] = "retry_execution"
    calls = 0

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal calls
        calls += 1
        return ModelResult(
            reply=json.dumps(invalid),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    with pytest.raises(ModelGatewayError) as captured:
        request_task_delivery_validation(
            _request(),
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert calls == MAX_MODEL_ATTEMPTS


@pytest.mark.parametrize(
    "defect",
    (
        "duplicate_top_level_key",
        "duplicate_nested_key",
        "NaN",
        "Infinity",
        "-Infinity",
    ),
)
def test_validation_lifecycle_retries_strict_json_failures(
    defect: str,
) -> None:
    calls = 0
    reply = _strictly_invalid_json_reply(_payload(), defect=defect)

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal calls
        calls += 1
        return ModelResult(
            reply=reply,
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    with pytest.raises(ModelGatewayError) as captured:
        request_task_delivery_validation(
            _request(),
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.code == "MODEL_BAD_RESPONSE"
    assert calls == MAX_MODEL_ATTEMPTS
