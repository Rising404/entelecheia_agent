from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph import (
    AuxiliaryPlanningGoal,
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningContextArtifactProjection,
    PlanningContextFactProjection,
    PlanningContextGapProjection,
    PlanningEpisodeBudget,
    PlanningGoalPromptContext,
    PlanningObservationStatus,
    TaskGraphSemanticReviewPolicy,
    TaskGraphSemanticFailureScope,
    TaskGraphSemanticTerminalRoute,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationPromptPayload,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    derive_task_graph_semantic_terminal_route,
)
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
    InSessionTaskNodeProposal,
    InSessionTaskRootGraphProposal,
)
# 在模型之前导入运行时端口，以匹配围绕旧版 runtime.events 模块既有的包初始化顺序。
from personagraph.runtime.model_calls import requests as model_requests
from personagraph.runtime.model_calls import DurableModelCallTerminalState
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.turn_deadline import (
    TurnDeadline,
    TurnDeadlineExceeded,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_RESULT_JSON_UTF8_BYTES,
    TaskGraphSemanticVerificationInputTooLarge,
    TaskGraphSemanticVerificationInputUnsupported,
    request_task_graph_semantic_verification,
    serialize_task_graph_semantic_verification_prompt,
    task_graph_semantic_verification_prompt_utf8_bytes,
)
import personagraph.l2.auxiliary_execution.verification.task_graph_semantic as verifier_module
from personagraph.model_io.gateway import (
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)


def _request() -> TaskGraphSemanticVerificationRequest:
    authority = PlanningAuthorityProjection.create(
        authority_snapshot_id="authority_snapshot_01",
        authority_snapshot_sha256=SHA_A,
        cards=(
            PlanningAuthoritySourceCard(
                alias="gap_01",
                authority_class=PlanningAuthorityClass.GAP,
                source_kind=PlanningAuthoritySourceKind.GAP,
                source_label="Unavailable optional appendix",
                excerpt="One optional appendix remains unresolved.",
                projection_sha256=SHA_A,
            ),
            PlanningAuthoritySourceCard(
                alias="obs_01",
                authority_class=PlanningAuthorityClass.EVIDENCE,
                source_kind=PlanningAuthoritySourceKind.DOCUMENT,
                source_label="Mounted document",
                excerpt="The document establishes the facts used by this proposal.",
                projection_sha256=SHA_B,
            ),
            PlanningAuthoritySourceCard(
                alias="obs_unused",
                authority_class=PlanningAuthorityClass.EVIDENCE,
                source_kind=PlanningAuthoritySourceKind.DOCUMENT,
                source_label="Unreferenced appendix",
                excerpt="This evidence is available but is not cited by the proposal.",
                projection_sha256=SHA_D,
            ),
            PlanningAuthoritySourceCard(
                alias="src_01",
                authority_class=PlanningAuthorityClass.AUTHORIZATION,
                source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
                source_label="Current user instruction",
                excerpt="Interpret the document and produce a grounded result.",
                projection_sha256=SHA_C,
            ),
        ),
    )
    artifact = PlanningContextArtifactProjection.create(
        artifact_alias="artifact_01",
        artifact_id="private_artifact_01",
        artifact_sha256=SHA_B,
        producer_node_alias="observe_01",
        facts=(
            PlanningContextFactProjection(
                fact_alias="fact_01",
                statement="The document supports the proposed deliverable.",
                evidence_aliases=("obs_01",),
            ),
        ),
        gaps=(
            PlanningContextGapProjection(
                gap_alias="gap_01",
                observation_status=PlanningObservationStatus.PARTIAL,
                description="One optional appendix is unavailable.",
                blocking=False,
                affected_obligations=("accept_delivery",),
                evidence_aliases=("obs_01",),
                resolution_hint="Proceed without claiming appendix coverage.",
            ),
        ),
    )
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="accept_delivery",
        criterion="The authorized document analysis is complete and verifiable.",
        source_anchor_ids=("obs_01", "src_01"),
    )
    proposal = InSessionTaskGraphRevisionProposal(
        root=InSessionTaskRootGraphProposal(
            root_key="root",
            nodes=(
                InSessionTaskNodeProposal(
                    node_key="root",
                    node_kind=InSessionTaskNodeKind.ROOT,
                    title="Deliver the document analysis",
                    objective="Produce the authorized evidence-grounded result.",
                    source_anchor_ids=("obs_01", "src_01"),
                    acceptance_criteria=(acceptance,),
                ),
            ),
        )
    )
    payload = TaskGraphSemanticVerificationPromptPayload.create(
        goal=PlanningGoalPromptContext(
            goal_id="goal_01",
            objective="Interpret the supplied document correctly.",
            desired_output="A grounded result satisfying the user request.",
            authorization_aliases=("src_01",),
        ),
        authority=authority,
        base_task_graph=None,
        context_artifacts=(artifact,),
        capabilities=PlanningCapabilityCatalogProjection.create(
            capability_catalog_snapshot_id="capability_catalog_01",
            capability_catalog_snapshot_sha256=SHA_D,
            capabilities=(
                PlanningCapabilityDescriptor(
                    capability_alias="document_analysis",
                    label="Document analysis",
                    description="Read a document and synthesize a grounded result.",
                    available=True,
                    effect=PlanningCapabilityEffect.READ_ONLY,
                    supported_operations=("read", "synthesize"),
                    supported_resource_kinds=("pdf",),
                ),
            ),
        ),
        budget=PlanningEpisodeBudget.create(
            budget_ledger_id="budget_01",
            goal_id="goal_01",
        ),
        task_graph_proposal=proposal,
    )
    return TaskGraphSemanticVerificationRequest.create(
        verification_request_id="semantic_request_01",
        logical_call_id="logical_call_01",
        verification_profile_id="semantic_verifier_v1",
        reviewer_ordinal=1,
        required_reviewer_count=1,
        goal=AuxiliaryPlanningGoal(
            session_id="session_01",
            task_id="task_01",
            auxiliary_graph_id="aux_graph_01",
            goal_id="goal_01",
            base_task_graph_revision=None,
            target_task_graph_revision=1,
            creation_turn_id="turn_01",
            authorization_manifest_id="authorization_manifest_01",
            budget_ledger_id="budget_01",
        ),
        auxiliary_graph_revision=1,
        auxiliary_graph_structure_sha256=SHA_D,
        prompt_payload=payload,
        review_policy=TaskGraphSemanticReviewPolicy.create(
            policy_source_sha256=SHA_A,
            distinct_document_count=1,
            has_visual_input=False,
            requires_protected_effect=False,
            modifies_executed_task_graph=False,
        ),
    )


def _items(
    *,
    failed_dimension: TaskGraphSemanticVerificationDimension | None = None,
    insufficient_dimension: TaskGraphSemanticVerificationDimension | None = None,
    failed_scope: str = "terminal_proposal",
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for dimension in TaskGraphSemanticVerificationDimension:
        verdict = (
            "insufficient_evidence"
            if dimension is insufficient_dimension
            else "fail"
            if dimension is failed_dimension
            else "pass"
        )
        items.append(
            {
                "dimension": dimension.value,
                "verdict": verdict,
                "failure_scope": (
                    "missing_authority"
                    if verdict == "insufficient_evidence"
                    else failed_scope
                    if verdict == "fail"
                    else None
                ),
                "finding": f"{dimension.value} dimension has been reviewed.",
                "affected_node_keys": ["root"],
                "evidence_aliases": (
                    ["obs_01"]
                    if dimension
                    is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                    else []
                ),
                "gap_aliases": (
                    ["gap_01"]
                    if dimension
                    is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                    or dimension is insufficient_dimension
                    else []
                ),
            }
        )
    return items


def _reply(items: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {"items": _items() if items is None else items},
        ensure_ascii=False,
    )


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
    )


def test_prompt_payload_only_provider_protocol_and_host_result_binding() -> None:
    received: dict[str, object] = {}
    events = []

    def provider(system: str, user: str, **kwargs: object) -> ModelResult:
        received.update(system=system, user=user, kwargs=kwargs)
        return _model_result(_reply(), str(kwargs["model_call_id"]))

    request = _request()
    requested = request_task_graph_semantic_verification(
        request,
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=events.append,
    )

    expected_payload = request.to_prompt_payload().model_dump(mode="json")
    assert json.loads(str(received["user"])) == expected_payload
    assert str(received["user"]) == json.dumps(
        expected_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert request.verification_request_id not in str(received["user"])
    assert request.logical_call_id not in str(received["user"])
    assert request.verification_profile_id not in str(received["user"])
    assert "独立 TaskGraph 语义验证器" in str(received["system"])
    assert "不得输出 overall" in str(received["system"])
    assert "精确覆盖" in str(received["system"])
    assert "blocking=true" in str(received["system"])
    assert "必须输出 insufficient_evidence" in str(received["system"])
    assert "Host 先执行叶子" in str(received["system"])
    assert "root=C" in str(received["system"])
    assert "空协调壳" in str(received["system"])
    assert "完成回执" in str(received["system"])
    assert "host_output_repair_feedback" not in str(received["system"])
    assert received["kwargs"] == {
        "model_call_id": requested.model_call_id,
        "purpose": "runtime_task_graph_semantic_verification",
    }
    assert requested.model_call_id == request.logical_call_id
    assert {event.model_call_id for event in events} == {request.logical_call_id}
    assert requested.value.verification_result_id == "semantic_result_01"
    assert requested.value.verification_request_id == request.verification_request_id
    assert requested.value.request_binding_sha256 == request.binding_sha256
    assert requested.value.all_pass is True
    assert (
        requested.value.host_disposition
        is TaskGraphSemanticVerificationDisposition.PASS
    )
    assert tuple(item.dimension for item in requested.value.items) == tuple(
        TaskGraphSemanticVerificationDimension
    )
    assert [event.stage.value for event in events] == [
        "VERIFICATION",
        "VERIFICATION",
    ]
    assert [event.status.value for event in events] == ["started", "completed"]


def test_canonical_prompt_utf8_size_is_the_exact_sent_string() -> None:
    request = _request()
    serialized = serialize_task_graph_semantic_verification_prompt(request)

    assert task_graph_semantic_verification_prompt_utf8_bytes(request) == len(
        serialized.encode("utf-8")
    )
    assert "Interpret the supplied document correctly." in serialized
    assert "semantic_request_01" not in serialized


def test_prompt_over_frozen_byte_limit_fails_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    exact_bytes = task_graph_semantic_verification_prompt_utf8_bytes(request)
    monkeypatch.setattr(
        verifier_module,
        "TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_PROMPT_JSON_UTF8_BYTES",
        exact_bytes - 1,
    )
    provider_calls = 0
    events: list[object] = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("over-limit prompt reached provider")

    with pytest.raises(TaskGraphSemanticVerificationInputTooLarge) as captured:
        request_task_graph_semantic_verification(
            request,
            invocation_turn_id="turn_verify_01",
            verification_result_id="semantic_result_01",
            provider=as_prepared_test_provider(provider),
            emit=events.append,
        )

    assert captured.value.serialized_utf8_bytes == exact_bytes
    assert captured.value.max_serialized_utf8_bytes == exact_bytes - 1
    assert provider_calls == 0
    assert events == []


def test_non_utf8_prompt_payload_fails_before_provider() -> None:
    request = _request()
    bad_goal = request.prompt_payload.goal.model_copy(
        update={"objective": "unsupported-surrogate-\ud800"}
    )
    bad_payload = request.prompt_payload.model_copy(update={"goal": bad_goal})
    bad_request = request.model_copy(update={"prompt_payload": bad_payload})
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("non-UTF-8 prompt reached provider")

    with pytest.raises(TaskGraphSemanticVerificationInputUnsupported) as captured:
        request_task_graph_semantic_verification(
            bad_request,
            invocation_turn_id="turn_verify_01",
            verification_result_id="semantic_result_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert captured.value.reason == "not_canonical_json_utf8"
    assert provider_calls == 0


def test_malformed_shape_retries_under_one_logical_model_call() -> None:
    valid = {"items": _items()}
    missing = deepcopy(valid)
    missing["items"].pop()  # type: ignore[union-attr]
    duplicate = deepcopy(valid)
    duplicate["items"][-1]["dimension"] = duplicate["items"][0]["dimension"]  # type: ignore[index]
    unknown = deepcopy(valid)
    unknown["items"][-1]["dimension"] = "unknown_dimension"  # type: ignore[index]
    extra = deepcopy(valid)
    extra["overall_pass"] = True
    missing_field = deepcopy(valid)
    missing_field["items"][0].pop("gap_aliases")  # type: ignore[index]
    replies = [missing, duplicate, unknown, extra, missing_field, valid]
    call_ids: list[str] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        call_ids.append(str(kwargs["model_call_id"]))
        return _model_result(
            json.dumps(replies[len(call_ids) - 1], ensure_ascii=False),
            str(kwargs["model_call_id"]),
        )

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 6
    assert call_ids == [requested.model_call_id] * 6


def test_dimension_guard_reports_duplicate_and_missing_in_one_pass() -> None:
    items = _items()
    items[-1]["dimension"] = items[0]["dimension"]

    with pytest.raises(ModelOutputValidationError) as captured:
        verifier_module._parse_and_guard_task_graph_semantic_verification(
            _reply(items)
        )

    error = captured.value
    assert error.repair_issue_coverage.value == "complete"
    assert error.omitted_repair_issue_count == 0
    assert [issue.code for issue in error.repair_issues or ()] == [
        "host_guard.task_graph_semantic.dimension_duplicate",
        (
            "host_guard.task_graph_semantic.dimension_missing."
            "base_authority_preservation"
        ),
    ]
    assert (error.repair_issues or ())[0].paths == (
        "/items/0/dimension",
        "/items/7/dimension",
    )
    assert (error.repair_issues or ())[1].paths == ("/items",)


def test_rejected_output_receives_bounded_host_repair_feedback() -> None:
    secret_marker = "PRIVATE_REJECTED_SEMANTIC_REPLY"
    invalid_items = _items()
    invalid_items[0].pop("finding")
    invalid = {"items": invalid_items, "private_extra": secret_marker}
    seen_payloads: list[dict[str, object]] = []
    seen_repair_messages: list[list[dict[str, str]]] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user)
        assert isinstance(payload, dict)
        seen_payloads.append(payload)
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is not None:
            assert isinstance(repair_messages, list)
            seen_repair_messages.append(repair_messages)
        reply = invalid if len(seen_payloads) == 1 else {"items": _items()}
        return _model_result(
            json.dumps(reply, ensure_ascii=False),
            str(kwargs["model_call_id"]),
        )

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_repair",
        verification_result_id="semantic_result_repair",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert "host_output_repair_feedback" not in seen_payloads[0]
    assert seen_payloads[1] == seen_payloads[0]
    assert len(seen_repair_messages) == 1
    repair_messages = seen_repair_messages[0]
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert set(feedback) == {"current_issues"}
    explanations = {issue["safe_explanation"] for issue in feedback["current_issues"]}
    assert "目标合同要求此位置必须存在。" in explanations
    assert "该位置含有目标合同未声明的额外字段。" in explanations
    assert secret_marker not in repair_messages[3]["content"]


def test_prepared_semantic_repair_uses_four_message_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_items = _items()
    invalid_items[0].pop("finding")
    invalid_reply = json.dumps(
        {"items": invalid_items, "private_extra": "PRIVATE_SEMANTIC_VALUE"},
        ensure_ascii=False,
    )
    valid_reply = _reply()
    prepared_calls: list[dict[str, object]] = []
    frozen_system_prompt = "FROZEN_SEMANTIC_SYSTEM"
    frozen_user_content = '{"frozen_semantic_prompt":true}'

    monkeypatch.setattr(
        verifier_module,
        "durable_structured_provider_prompt",
        lambda _durable_call, **_current: (
            frozen_system_prompt,
            frozen_user_content,
        ),
    )

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared semantic provider used direct dispatch")

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
            return _model_result(reply, model_call_id)

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare  # type: ignore[attr-defined]
    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_prepared_repair",
        verification_result_id="semantic_result_prepared_repair",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert requested.value.all_pass is True
    assert len(prepared_calls) == 2
    initial, repair = prepared_calls
    assert initial["system_prompt"] == frozen_system_prompt
    assert initial["user_content"] == frozen_user_content
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
    assert messages[:2] == [
        {"role": "system", "content": frozen_system_prompt},
        {"role": "user", "content": frozen_user_content},
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
    } == {("",), ("/items",), ("/items/0/finding",)}
    assert "PRIVATE_SEMANTIC_VALUE" not in messages[3]["content"]


def test_repair_feedback_does_not_mutate_frozen_semantic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    exact_bytes = task_graph_semantic_verification_prompt_utf8_bytes(request)
    monkeypatch.setattr(
        verifier_module,
        "TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_PROMPT_JSON_UTF8_BYTES",
        exact_bytes,
    )
    prepared_calls: list[dict[str, object]] = []

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        prepared_calls.append(
            {
                "user": _user,
                "repair_messages": kwargs.get("repair_messages"),
            }
        )
        reply = "not-json" if len(prepared_calls) == 1 else _reply()
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        request,
        invocation_turn_id="turn_verify_repair_limit",
        verification_result_id="semantic_result_repair_limit",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert prepared_calls[0]["user"] == prepared_calls[1]["user"]
    assert prepared_calls[0]["repair_messages"] is None
    assert isinstance(prepared_calls[1]["repair_messages"], list)


def test_durable_repair_feedback_is_recovered_across_turn_boundary() -> None:
    request = _request()
    secret_marker = "PRIVATE_REJECTED_DURABLE_SEMANTIC_REPLY"
    invalid_reply = json.dumps(
        {"items": _items(), "private_extra": secret_marker},
        ensure_ascii=False,
    )
    state: dict[str, object] = {
        "ordinal": 0,
        "feedback": None,
        "rejected_response": None,
        "pause_before_next_attempt": False,
    }
    begun_feedback: list[object | None] = []
    seen_payloads: list[dict[str, object]] = []
    resumed_prepared_calls: list[dict[str, object]] = []

    class DurableCall:
        semantic_call_id = request.logical_call_id
        logical_request = SimpleNamespace(
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=(
                    verifier_module._TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT
                ),
                user_content=serialize_task_graph_semantic_verification_prompt(
                    request
                ),
            ),
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract=verifier_module._RESULT_CONTRACT,
        )

        def require_current_state(self) -> None:
            if state["pause_before_next_attempt"]:
                state["pause_before_next_attempt"] = False
                raise DurableModelCallTerminalState(
                    "simulated process boundary after durable rejection"
                )

        def reserve(self, *, turn_id: str) -> object:
            assert turn_id in {"turn_verify_first", "turn_verify_resume"}
            return object()

        def replay_succeeded_result(self) -> None:
            return None

        def recover_output_repair_feedback(
            self,
        ) -> RuntimeModelOutputRepairFeedback | None:
            feedback = state["feedback"]
            assert feedback is None or isinstance(
                feedback,
                RuntimeModelOutputRepairFeedback,
            )
            return feedback

        def recover_output_repair_response(
            self,
            feedback: RuntimeModelOutputRepairFeedback,
        ) -> str:
            assert feedback == state["feedback"]
            rejected_response = state["rejected_response"]
            assert isinstance(rejected_response, str)
            return rejected_response

        def begin_physical_attempt(self, **values: object) -> object:
            assert values["output_repair_enabled"] is True
            begun_feedback.append(values["output_repair_feedback"])
            state["ordinal"] = int(state["ordinal"]) + 1
            ordinal = int(state["ordinal"])
            return type(
                "Physical",
                (),
                {
                    "physical_attempt_id": f"semantic-physical-{ordinal}",
                    "physical_ordinal": ordinal,
                    "provider": "test",
                    "model": "test",
                    "model_call_id": (
                        f"{request.logical_call_id}:physical:{ordinal}"
                    ),
                },
            )()

        def settle_physical_attempt(self, **values: object) -> None:
            if values["outcome"] == "retryable_failure":
                feedback = values["next_output_repair_feedback"]
                assert isinstance(feedback, RuntimeModelOutputRepairFeedback)
                rejected_response = values["rejected_response_text"]
                assert isinstance(rejected_response, str)
                state["feedback"] = feedback
                state["rejected_response"] = rejected_response
                state["pause_before_next_attempt"] = True

        def typed_result_payload(
            self, *, model_result: object, value: object
        ) -> object:
            assert isinstance(model_result, ModelResult)
            return value

        def success_fingerprint(self, _result: object) -> str:
            return "semantic-success"

        def failure_fingerprint(self, **_values: object) -> str:
            return "semantic-failure"

        def terminal_state_error(self, message: str) -> DurableModelCallTerminalState:
            return DurableModelCallTerminalState(message)

    def first_provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared semantic provider used direct dispatch")

    def prepare_first_provider(
        _system: str,
        user: str,
        **kwargs: object,
    ) -> PreparedModelCall:
        assert "repair_messages" not in kwargs
        payload = json.loads(user)
        assert isinstance(payload, dict)
        seen_payloads.append(payload)

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return _model_result(invalid_reply, model_call_id)

        return PreparedModelCall(_dispatch=dispatch)

    first_provider.prepare = prepare_first_provider  # type: ignore[attr-defined]

    with pytest.raises(DurableModelCallTerminalState, match="process boundary"):
        request_task_graph_semantic_verification(
            request,
            invocation_turn_id="turn_verify_first",
            verification_result_id="semantic_result_repair",
            provider=first_provider,  # type: ignore[arg-type]
            emit=lambda _event: None,
            durable_call=DurableCall(),  # type: ignore[arg-type]
        )

    def resumed_provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared resumed provider used direct dispatch")

    def prepare_resumed_provider(
        system: str,
        user: str,
        **kwargs: object,
    ) -> PreparedModelCall:
        resumed_prepared_calls.append(
            {
                "system_prompt": system,
                "user_content": user,
                **kwargs,
            }
        )

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return _model_result(_reply(), model_call_id)

        return PreparedModelCall(_dispatch=dispatch)

    resumed_provider.prepare = prepare_resumed_provider  # type: ignore[attr-defined]

    resumed = request_task_graph_semantic_verification(
        request,
        invocation_turn_id="turn_verify_resume",
        verification_result_id="semantic_result_repair",
        provider=resumed_provider,
        emit=lambda _event: None,
        durable_call=DurableCall(),  # type: ignore[arg-type]
    )

    assert resumed.attempts == 2
    assert begun_feedback[0] is None
    assert isinstance(begun_feedback[1], RuntimeModelOutputRepairFeedback)
    assert "host_output_repair_feedback" not in seen_payloads[0]
    assert len(resumed_prepared_calls) == 2
    initial, repair = resumed_prepared_calls
    assert "repair_messages" not in initial
    repair_messages = repair["repair_messages"]
    assert isinstance(repair_messages, list)
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert repair_messages[:2] == [
        {
            "role": "system",
            "content": initial["system_prompt"],
        },
        {
            "role": "user",
            "content": initial["user_content"],
        },
    ]
    assert repair_messages[2]["content"] == invalid_reply
    repaired = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert set(repaired) == {"current_issues"}
    assert repaired["current_issues"]
    assert secret_marker not in repair_messages[3]["content"]


def test_duplicate_json_object_keys_are_not_silently_collapsed() -> None:
    calls = 0
    items_json = json.dumps(_items(), ensure_ascii=False)

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        reply = (
            f'{{"items":{items_json},"items":{items_json}}}'
            if calls == 1
            else _reply()
        )
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2


def test_reordered_complete_dimensions_are_host_canonicalized() -> None:
    items = list(reversed(_items()))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(items), str(kwargs["model_call_id"])
            )
        ),
        emit=lambda _event: None,
    )

    assert requested.attempts == 1
    assert tuple(item.dimension for item in requested.value.items) == tuple(
        TaskGraphSemanticVerificationDimension
    )


def test_node_evidence_gap_and_exact_manifest_guards_retry() -> None:
    valid = _items()
    unknown_node = deepcopy(valid)
    unknown_node[0]["affected_node_keys"] = ["unknown_node"]
    unknown_evidence = deepcopy(valid)
    evidence_index = list(TaskGraphSemanticVerificationDimension).index(
        TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
    )
    unknown_evidence[evidence_index]["evidence_aliases"] = ["unknown_evidence"]
    unknown_gap = deepcopy(valid)
    gap_index = list(TaskGraphSemanticVerificationDimension).index(
        TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
    )
    unknown_gap[gap_index]["gap_aliases"] = ["unknown_gap"]
    missing_evidence_manifest = deepcopy(valid)
    missing_evidence_manifest[evidence_index]["evidence_aliases"] = []
    missing_gap_manifest = deepcopy(valid)
    missing_gap_manifest[gap_index]["gap_aliases"] = []
    replies = [
        unknown_node,
        unknown_evidence,
        unknown_gap,
        missing_evidence_manifest,
        missing_gap_manifest,
        valid,
    ]
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(replies[calls])
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 6
    assert requested.attempts == 6


def test_duplicate_item_references_retry() -> None:
    duplicate = _items()
    duplicate[0]["affected_node_keys"] = ["root", "root"]
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(duplicate if calls == 0 else _items())
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2


def test_known_but_unreferenced_evidence_over_report_retries() -> None:
    over_reported = _items()
    evidence_index = list(TaskGraphSemanticVerificationDimension).index(
        TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
    )
    over_reported[evidence_index]["evidence_aliases"] = ["obs_01", "obs_unused"]
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(over_reported if calls == 0 else _items())
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2


def test_insufficient_evidence_requires_typed_gap_and_host_derives_blocked() -> None:
    insufficient_dimension = TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
    invalid = _items(insufficient_dimension=insufficient_dimension)
    invalid[0]["gap_aliases"] = []
    valid = _items(insufficient_dimension=insufficient_dimension)
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(invalid if calls == 0 else valid)
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.value.all_pass is False
    assert (
        requested.value.host_disposition
        is TaskGraphSemanticVerificationDisposition.BLOCKED
    )


def test_host_derives_revise_from_any_failed_dimension() -> None:
    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(
                    _items(
                        failed_dimension=(
                            TaskGraphSemanticVerificationDimension.EDGE_DEPENDENCY_VALIDITY
                        )
                    )
                ),
                str(kwargs["model_call_id"]),
            )
        ),
        emit=lambda _event: None,
    )

    assert requested.value.all_pass is False
    assert (
        requested.value.host_disposition
        is TaskGraphSemanticVerificationDisposition.REVISE
    )
    assert (
        requested.value.host_terminal_route
        is TaskGraphSemanticTerminalRoute.RETRY_TERMINAL_ATTEMPT
    )


def test_host_routes_auxiliary_investigation_scope_to_auxiliary_replan() -> None:
    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(
                    _items(
                        failed_dimension=(
                            TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                        ),
                        failed_scope="auxiliary_investigation",
                    )
                ),
                str(kwargs["model_call_id"]),
            )
        ),
        emit=lambda _event: None,
    )

    assert (
        requested.value.host_terminal_route
        is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
    )


def test_host_routes_missing_authority_scope_and_insufficient_evidence_to_blocked() -> None:
    missing = _items(
        failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE,
        failed_scope="missing_authority",
    )
    missing[0]["gap_aliases"] = ["gap_01"]
    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(
            lambda _system, _user, **kwargs: _model_result(
                _reply(missing), str(kwargs["model_call_id"])
            )
        ),
        emit=lambda _event: None,
    )

    assert (
        requested.value.host_terminal_route
        is TaskGraphSemanticTerminalRoute.BLOCKED
    )

    request = _request()
    insufficient = TaskGraphSemanticVerificationResult.create(
        verification_result_id="semantic_result_02",
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        logical_call_id=request.logical_call_id,
        verification_profile_id=request.verification_profile_id,
        reviewer_ordinal=1,
        required_reviewer_count=1,
        items=_items(
            insufficient_dimension=(
                TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
            )
        ),
    )
    assert (
        insufficient.host_terminal_route
        is TaskGraphSemanticTerminalRoute.BLOCKED
    )


def test_quorum_route_uses_blocked_then_auxiliary_then_terminal_precedence() -> None:
    request = _request()

    def result(
        result_id: str,
        items: list[dict[str, object]],
    ) -> TaskGraphSemanticVerificationResult:
        return TaskGraphSemanticVerificationResult.create(
            verification_result_id=result_id,
            verification_request_id=request.verification_request_id,
            request_binding_sha256=request.binding_sha256,
            logical_call_id=request.logical_call_id,
            verification_profile_id=request.verification_profile_id,
            reviewer_ordinal=1,
            required_reviewer_count=1,
            items=items,
        )

    terminal = result(
        "semantic_result_terminal",
        _items(
            failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
        ),
    )
    auxiliary = result(
        "semantic_result_auxiliary",
        _items(
            failed_dimension=(
                TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
            ),
            failed_scope="auxiliary_investigation",
        ),
    )
    blocked_items = _items(
        failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE,
        failed_scope="missing_authority",
    )
    blocked_items[0]["gap_aliases"] = ["gap_01"]
    blocked = result("semantic_result_blocked", blocked_items)

    assert derive_task_graph_semantic_terminal_route((terminal,)) is (
        TaskGraphSemanticTerminalRoute.RETRY_TERMINAL_ATTEMPT
    )
    assert derive_task_graph_semantic_terminal_route(
        (terminal, auxiliary)
    ) is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY
    assert derive_task_graph_semantic_terminal_route(
        (terminal, auxiliary, blocked)
    ) is TaskGraphSemanticTerminalRoute.BLOCKED


def test_model_response_must_explicitly_include_failure_scope() -> None:
    missing_scope = _items()
    for item in missing_scope:
        item.pop("failure_scope")
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(missing_scope if calls == 0 else _items())
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2
    assert all(item.failure_scope is None for item in requested.value.items)


def test_failed_model_response_requires_typed_failure_scope() -> None:
    failed_without_scope = _items(
        failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
    )
    failed_without_scope[0]["failure_scope"] = None
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(failed_without_scope if calls == 0 else _items())
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2
    assert requested.value.all_pass


def test_failure_scope_contract_rejects_pass_scope_and_missing_authority_without_gap() -> None:
    with pytest.raises(ValueError):
        TaskGraphSemanticFailureScope("unknown")

    invalid_pass = _items()
    invalid_pass[0]["failure_scope"] = "terminal_proposal"
    invalid_missing = _items(
        failed_dimension=TaskGraphSemanticVerificationDimension.GOAL_COVERAGE,
        failed_scope="missing_authority",
    )

    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        candidate = invalid_pass if calls == 0 else invalid_missing if calls == 1 else _items()
        calls += 1
        return _model_result(_reply(candidate), str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 3
    assert requested.attempts == 3


def test_private_binding_echo_is_rejected_by_formal_host_validator() -> None:
    request = _request()
    private_echo = _items()
    private_echo[0]["finding"] = (
        f"The private {request.logical_call_id} binding was exposed."
    )
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = _reply(private_echo if calls == 0 else _items())
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        request,
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2


def test_non_utf8_output_is_terminal_because_repair_requires_exact_utf8() -> None:
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            "invalid-surrogate-\ud800",
            str(kwargs["model_call_id"]),
        )

    with pytest.raises(ModelGatewayError) as captured:
        request_task_graph_semantic_verification(
            _request(),
            invocation_turn_id="turn_verify_01",
            verification_result_id="semantic_result_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert calls == 1
    assert captured.value.code == "MODEL_BAD_RESPONSE"


def test_frozen_result_byte_guard_retries() -> None:
    calls = 0

    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        reply = (
            " " * (TASK_GRAPH_SEMANTIC_VERIFICATION_MAX_RESULT_JSON_UTF8_BYTES + 1)
            + _reply()
            if calls == 0
            else _reply()
        )
        calls += 1
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_task_graph_semantic_verification(
        _request(),
        invocation_turn_id="turn_verify_01",
        verification_result_id="semantic_result_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.attempts == 2


def test_expired_turn_deadline_stops_before_provider() -> None:
    provider_calls = 0
    events = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("expired verification reached provider")

    with pytest.raises(TurnDeadlineExceeded):
        request_task_graph_semantic_verification(
            _request(),
            invocation_turn_id="turn_verify_01",
            verification_result_id="semantic_result_01",
            provider=as_prepared_test_provider(provider),
            emit=events.append,
            deadline=TurnDeadline.starting_now(0),
        )

    assert provider_calls == 0
    assert len(events) == 1
    assert events[0].stage.value == "VERIFICATION"
    assert events[0].status.value == "failed"


def test_invalid_host_result_identity_fails_before_provider() -> None:
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("invalid Host binding reached provider")

    with pytest.raises(ValidationError):
        request_task_graph_semantic_verification(
            _request(),
            invocation_turn_id="turn_verify_01",
            verification_result_id="not allowed whitespace",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
        )

    assert provider_calls == 0


def test_durable_call_identity_mismatch_fails_before_provider() -> None:
    provider_calls = 0

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("mismatched durable authority reached provider")

    with pytest.raises(ValueError, match="must match its durable call"):
        request_task_graph_semantic_verification(
            _request(),
            invocation_turn_id="turn_verify_01",
            verification_result_id="semantic_result_01",
            provider=as_prepared_test_provider(provider),
            emit=lambda _event: None,
            durable_call=SimpleNamespace(  # type: ignore[arg-type]
                semantic_call_id="different_logical_call"
            ),
        )

    assert provider_calls == 0
