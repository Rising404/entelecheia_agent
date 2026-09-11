from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_execution.attempts.decision import (
    AcceptanceVerificationFeedback,
    AttemptDecisionContext,
    AttemptDecisionInputLimits,
    AttemptDecisionInputTooLarge,
    AttemptDecisionInputUnsupported,
    AttemptUserInput,
    AttemptVerificationFeedback,
    PriorToolResultProjection,
    PriorToolResultsProjection,
    RequiredPriorToolResultsUnavailable,
    VerificationVerdict,
    attempt_prompt_serialized_utf8_bytes,
    build_prior_tool_results_prompt_payload,
    prior_tool_results_serialized_utf8_bytes,
    request_attempt_decision,
    select_bounded_prior_tool_results,
    serialize_attempt_prompt_payload,
)
from personagraph.l2.task_execution.paper_prompt_context import PaperAttemptContext
from personagraph.l2.task_execution.work_run.execution_findings import (
    work_run_tool_result_sha256,
)
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider,
    repair_feedback_from_provider_kwargs,
)
from personagraph.persistent_turn_content.findings import (
    ExecutionFindingsActiveProjection,
    ExecutionFindingsOwnerKind,
    derive_execution_findings_ledger_id,
    sha256_json,
)
from personagraph.tools.findings.execution_findings_tools import (
    build_execution_findings_tool_registrations,
)
from personagraph.l2.task_graph.paper_resource_contracts import (
    PaperDocumentBinding,
    PaperResourceSnapshot,
)
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
)
# 先导入运行时端口，以维持模型网关当前的模块初始化顺序。
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
from personagraph.tools.contracts import ToolSpec
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    OutputWindow,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    initialize_acceptance_progress,
)
from tests.helpers.task_node_source_context import task_node_source_context


def _model_result(reply: str, model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=reply,
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
    )


def _empty_support(
    reason_code: str = "candidate_is_primary_artifact",
) -> dict[str, str]:
    return {
        "schema_version": "empty-support-justification-v1",
        "reason_code": reason_code,
        "explanation": "The submitted artifact is the relevant review object.",
    }


def _input_limits(
    *,
    max_prior_tool_result_items: int = 32,
    max_prior_tool_results_serialized_utf8_bytes: int = 128_000,
    max_serialized_utf8_bytes: int = 1_000_000,
) -> AttemptDecisionInputLimits:
    return AttemptDecisionInputLimits(
        profile_id="attempt-test-profile",
        max_prior_tool_result_items=max_prior_tool_result_items,
        max_prior_tool_results_serialized_utf8_bytes=(
            max_prior_tool_results_serialized_utf8_bytes
        ),
        dependency_delivery_limits=TaskNodeDependencyInputLimits(
            profile_id="attempt-test-dependencies",
            max_items=32,
            max_serialized_utf8_bytes=256_000,
        ),
        max_serialized_utf8_bytes=max_serialized_utf8_bytes,
    )


def _paper_resources(
    *,
    session_id: str = "session-1",
    task_id: str = "task-1",
) -> PaperAttemptContext:
    snapshot = PaperResourceSnapshot.create(
        session_id=session_id,
        task_id=task_id,
        bound_graph_revision=3,
        bound_task_state_version=7,
        retrieval_data_version_id="rdv-paper-alpha",
        retrieval_generation_fingerprint="retrieval-generation-alpha",
        encoder_fingerprint="deterministic-lexical@1",
        documents=(
            PaperDocumentBinding(
                paper_key="P1",
                document_id="private-document-alpha",
                source_version_id="private-version-alpha",
                title="A bounded paper title",
                source_sha256="a" * 64,
                processing_status="complete",
                admitted_chunk_count=4,
                chunk_manifest_sha256="b" * 64,
                admitted_text_page_start=1,
                admitted_text_page_end=8,
            ),
        ),
    )
    return PaperAttemptContext.from_snapshot(snapshot)


def _context(
    *,
    with_feedback: bool = False,
    include_tools: bool = True,
    allow_user_input: bool = True,
) -> AttemptDecisionContext:
    subject = TaskNodeSubject(
        task_id="task-1",
        graph_revision=3,
        node_id="node-2",
        node_revision=4,
    )
    acceptance_ids = ("accurate_route", "includes_dates")
    feedback = (
        AttemptVerificationFeedback(
            submitted_output_revision=2,
            acceptance_results=(
                AcceptanceVerificationFeedback(
                    acceptance_id="accurate_route",
                    verdict=VerificationVerdict.PASSED,
                    finding="路线可执行。",
                ),
                AcceptanceVerificationFeedback(
                    acceptance_id="includes_dates",
                    verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
                    finding="尚未给出返程日期。",
                    missing_requirements=("补充返程日期",),
                ),
            ),
        )
        if with_feedback
        else None
    )
    acceptances = tuple(
        InSessionTaskAcceptanceProposal(
            acceptance_id=acceptance_id,
            criterion=criterion,
            source_anchor_ids=("anchor_1",),
        )
        for acceptance_id, criterion in (
            ("accurate_route", "路线前后衔接且可执行"),
            ("includes_dates", "包含出发与返程日期"),
        )
    )
    return AttemptDecisionContext(
        session_id="session-1",
        turn_id="turn-7",
        work_run_id="work-run-9",
        work_run_revision=6,
        attempt_id="attempt-3",
        attempt_ordinal=3,
        user_input=AttemptUserInput(
            content="我是九月1日出发，返程日期还没定。",
            prior_waiting_user_question="请提供出发日期。",
        ),
        subject=subject,
        node_title="产出旅行计划",
        node_objective="形成一份包含明确日期与路线的旅行计划。",
        acceptances=acceptances,
        acceptance_progress=initialize_acceptance_progress(
            work_run_id="work-run-9",
            subject=subject,
            acceptance_ids=acceptance_ids,
            evaluated_output_revision=2,
        ),
        output_window=OutputWindow(
            work_run_id="work-run-9",
            output_revision=2,
            format="markdown",
            content="# 当前草稿",
            updated_turn_id="turn-6",
            updated_attempt_id="attempt-2",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        source_context=task_node_source_context(
            session_id="session-1",
            subject=subject,
            acceptances=acceptances,
        ),
        prior_tool_results=PriorToolResultsProjection(
            items=(
                PriorToolResultProjection(
                    tool_id="read_itinerary_source",
                    tool_version="1.0.0",
                    result=ToolResult(
                        status=ToolResultStatus.SUCCEEDED,
                        tool_result_id="result-1",
                        tool_call_id="call-1",
                        attempt_id="attempt-2",
                        ordinal=1,
                        output={"departure": "2026-09-01"},
                    ),
                ),
            ),
            truncated=False,
        ),
        input_limits=_input_limits(),
        allowed_tools=(
            (
                ToolSpec(
                    tool_id="read_itinerary_source",
                    contract_version="1.0.0",
                    name="Read itinerary source",
                    description="Read one already-authorized itinerary source.",
                    input_schema={
                        "type": "object",
                        "properties": {"source_id": {"type": "string"}},
                        "required": ["source_id"],
                    },
                    output_schema={"type": "object"},
                    catalog_tags=("read",),
                ),
            )
            if include_tools
            else ()
        ),
        allow_user_input=allow_user_input,
        verification_feedback=feedback,
    )


def test_context_json_dump_thaws_frozen_tool_schemas_for_state_guards():
    context = _context()

    payload = context.model_dump(mode="json")

    assert payload["allowed_tools"][0]["input_schema"] == {
        "type": "object",
        "properties": {"source_id": {"type": "string"}},
        "required": ["source_id"],
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)



def _auxiliary_context() -> AttemptDecisionContext:
    current = _context(include_tools=False)
    subject = AuxiliaryNodeSubject(
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        auxiliary_graph_revision=1,
        node_id="planning-node",
        node_revision=1,
    )
    acceptances = (
        InSessionTaskAcceptanceProposal(
            acceptance_id="graph_proposal_ready",
            criterion="输出完整、来源绑定且可执行的 TaskGraph v2",
            source_anchor_ids=("task_creation_source",),
        ),
    )
    return current.model_copy(
        update={
            "subject": subject,
            "acceptances": acceptances,
            "acceptance_progress": initialize_acceptance_progress(
                work_run_id=current.work_run_id,
                subject=subject,
                acceptance_ids=("graph_proposal_ready",),
                evaluated_output_revision=current.output_window.output_revision,
            ),
            "source_context": None,
        }
    )


def test_auxiliary_attempt_projects_exact_complex_task_graph_contract() -> None:
    received: dict[str, object] = {}

    def provider(system: str, user: str, **kwargs: object) -> ModelResult:
        received["system"] = system
        received["payload"] = json.loads(user)
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "交付应优先优化时延还是成本？",
                    },
                },
                ensure_ascii=False,
            ),
            str(kwargs["model_call_id"]),
        )

    requested = request_attempt_decision(
        _auxiliary_context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.value.action.kind == "request_user_input"
    system = str(received["system"])
    assert "TaskGraph 规划器" in system
    assert "最少充分节点" in system
    assert "最短必要依赖链" in system
    assert "无法安全并入" in system
    assert "8–20" not in system
    assert "3–6" not in system
    assert "constraints 必须为空" in system
    assert "host_repair_feedback" not in system
    payload = received["payload"]
    assert isinstance(payload, dict)
    contract = payload["task_graph_proposal_contract"]
    assert contract["schema_version"] == "insession-task-graph-revision-v2"
    assert contract["action_kind"] == "submit_task_graph"
    assert contract["proposal_encoding"] == "direct_json_object"
    assert contract["allowed_source_anchor_ids"] == ["task_creation_source"]
    assert contract["required_source_anchor_ids"] == ["task_creation_source"]
    assert contract["limits"] == {
        "max_nodes_per_task": 64,
        "max_depth": 12,
        "max_acceptances_per_node": 64,
    }
    quality_target = contract["complex_task_quality_target"]
    assert "recommended_node_count" not in quality_target
    assert "recommended_depth" not in quality_target
    assert [8, 20] not in quality_target.values()
    assert [3, 6] not in quality_target.values()


def test_auxiliary_typed_graph_action_is_host_materialized_to_plain_text() -> None:
    def provider(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [
                        {
                            "acceptance_id": "graph_proposal_ready",
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": [],
                            "empty_support_justification": _empty_support(),
                        }
                    ],
                    "action": {
                        "kind": "submit_task_graph",
                        "proposal": {
                            "schema_version": "insession-task-graph-revision-v2",
                            "root": {
                                "root_key": "root",
                                "nodes": [
                                    {
                                        "node_key": "root",
                                        "node_kind": "root",
                                        "parent_node_key": None,
                                        "title": "交付系统",
                                        "objective": "综合所有子阶段的成果",
                                        "source_anchor_ids": [
                                            "task_creation_source"
                                        ],
                                        "acceptance_criteria": [
                                            {
                                                "acceptance_id": "root_done",
                                                "criterion": "完整成果可被验收",
                                                "source_anchor_ids": [
                                                    "task_creation_source"
                                                ],
                                            }
                                        ],
                                        "constraints": [],
                                    }
                                ],
                            },
                        },
                    },
                },
                ensure_ascii=False,
            ),
            str(kwargs["model_call_id"]),
        )

    requested = request_attempt_decision(
        _auxiliary_context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.value.action.kind == "submit_output_window"
    assert requested.value.action.format.value == "plain_text"
    proposal = json.loads(requested.value.action.content)
    assert proposal["schema_version"] == "insession-task-graph-revision-v2"
    assert proposal["root"]["nodes"][0]["node_key"] == "root"


def test_auxiliary_legacy_final_submit_is_rejected_with_repair_feedback() -> None:
    calls = 0
    rejected_reply: str | None = None

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal calls, rejected_reply
        calls += 1
        if calls == 1:
            action = {
                "kind": "submit_output_window",
                "format": "plain_text",
                "content": "{}",
            }
        else:
            feedback = repair_feedback_from_provider_kwargs(kwargs)
            assert feedback is not None
            assert set(feedback) == {"current_issues"}
            assert feedback["current_issues"]
            assert rejected_reply is not None
            assert kwargs["repair_messages"][2]["content"] == rejected_reply
            assert "previous_response_excerpt" not in feedback
            assert rejected_reply not in user
            action = {
                "kind": "request_user_input",
                "question": "请补充唯一会改变图结构的关键条件。",
            }
        reply = json.dumps(
            {
                "acceptance_updates": [],
                "action": action,
            },
            ensure_ascii=False,
        )
        if calls == 1:
            rejected_reply = reply
        return _model_result(reply, str(kwargs["model_call_id"]))

    requested = request_attempt_decision(
        _auxiliary_context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert calls == 2
    assert requested.value.action.kind == "request_user_input"


def test_task_node_contract_repair_feedback_exposes_safe_field_location() -> None:
    calls = 0
    model_call_ids: list[str] = []
    rejected_reply: str | None = None
    seen_feedback: list[dict[str, object]] = []
    secret_bad_input = "PRIVATE_BAD_INPUT_VALUE"

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal calls, rejected_reply
        calls += 1
        model_call_id = str(kwargs["model_call_id"])
        model_call_ids.append(model_call_id)
        feedback = repair_feedback_from_provider_kwargs(kwargs)
        if feedback is not None:
            seen_feedback.append(feedback)
        if calls == 1:
            reply = json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "PRIVATE_UNKNOWN_FIELD": secret_bad_input,
                    },
                }
            )
            rejected_reply = reply
        else:
            reply = json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "返程日期是哪一天？",
                    },
                },
                ensure_ascii=False,
            )
        return _model_result(reply, model_call_id)

    requested = request_attempt_decision(
        _context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert requested.value.action.kind == "request_user_input"
    assert len(set(model_call_ids)) == 1
    assert len(seen_feedback) == 1
    feedback = seen_feedback[0]
    issues = feedback["current_issues"]
    assert isinstance(issues, list)
    explanations = {issue["safe_explanation"] for issue in issues}
    assert "目标合同要求此位置必须存在。" in explanations
    assert "该位置含有目标合同未声明的额外字段。" in explanations
    assert secret_bad_input not in json.dumps(feedback, ensure_ascii=False)
    assert "PRIVATE_UNKNOWN_FIELD" not in json.dumps(feedback, ensure_ascii=False)
    assert rejected_reply is not None
    assert set(feedback) == {"current_issues"}


def test_prepared_attempt_repair_uses_four_message_contract() -> None:
    invalid_reply = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "request_user_input",
                "PRIVATE_UNKNOWN_FIELD": "PRIVATE_ATTEMPT_VALUE",
            },
        },
        ensure_ascii=False,
    )
    valid_reply = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "request_user_input",
                "question": "返程日期是哪一天？",
            },
        },
        ensure_ascii=False,
    )
    prepared_calls: list[dict[str, object]] = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared Attempt provider used direct dispatch")

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
    requested = request_attempt_decision(
        _context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert requested.value.action.kind == "request_user_input"
    assert len(prepared_calls) == 2
    initial, repair = prepared_calls
    assert "host_repair_feedback" not in str(initial["system_prompt"])
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
    } == {("/action",), ("/action/question",)}
    assert "PRIVATE_ATTEMPT_VALUE" not in messages[3]["content"]


def test_prepared_attempt_guard_reports_all_unexposed_tool_paths_once() -> None:
    invalid_reply = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "call_tools",
                "calls": [
                    {"tool_id": "unknown-a", "arguments": {}},
                    {"tool_id": "unknown-b", "arguments": {}},
                ],
            },
        }
    )
    valid_reply = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "request_user_input",
                "question": "请确认返程日期。",
            },
        },
        ensure_ascii=False,
    )
    prepared_calls: list[dict[str, object]] = []

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared Attempt provider used direct dispatch")

    def prepare(
        system_prompt: str,
        user_content: str,
        **kwargs: object,
    ) -> PreparedModelCall:
        prepared_calls.append({**kwargs})
        reply = invalid_reply if len(prepared_calls) == 1 else valid_reply

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return _model_result(reply, model_call_id)

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare  # type: ignore[attr-defined]
    requested = request_attempt_decision(
        _context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    messages = prepared_calls[1]["repair_messages"]
    assert isinstance(messages, list)
    repair_envelope = json.loads(
        messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert "当前清单可能不完整" in messages[3]["content"]
    assert [
        tuple(issue["paths"])
        for issue in repair_envelope["current_issues"]
    ] == [
        ("/action/calls/0/tool_id",),
        ("/action/calls/1/tool_id",),
    ]



def test_attempt_prompt_projects_authenticated_downstream_retry_feedback() -> None:
    base = _context()
    downstream = DownstreamVerificationFeedback(
        gate_id="whole_task_delivery",
        disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
        finding="根正文遗漏发布步骤。",
        repair_objective="保留正确内容并补齐发布步骤。",
        source_result_id="candidate-review-result-1",
        source_result_sha256="d" * 64,
        affected_subject_ids=(base.subject.node_id,),
    )
    feedback = AttemptVerificationFeedback(
        submitted_output_revision=base.output_window.output_revision,
        acceptance_results=tuple(
            AcceptanceVerificationFeedback(
                acceptance_id=item.acceptance_id,
                verdict=VerificationVerdict.PASSED,
                finding="节点自身验收已通过。",
            )
            for item in base.acceptances
        ),
        downstream_results=(downstream,),
    )
    context = base.model_copy(update={"verification_feedback": feedback})

    payload = json.loads(serialize_attempt_prompt_payload(context))

    assert payload["verification_feedback"]["downstream_results"] == [
        downstream.model_dump(mode="json")
    ]
    assert payload["verification_feedback"]["acceptance_results"][0][
        "verdict"
    ] == "passed"



def test_prompt_projects_exact_attempt_inputs_without_budget_or_hidden_state():
    received: dict[str, object] = {}
    events = []

    def complete(system_prompt: str, user_content: str, **kwargs: object) -> ModelResult:
        received["system_prompt"] = system_prompt
        received["payload"] = json.loads(user_content)
        received["kwargs"] = kwargs
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "返程日期是哪一天？",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    context = _context(with_feedback=True)
    result = request_attempt_decision(
        context,
        emit=events.append,
        provider=as_prepared_test_provider(complete),
    )

    payload = received["payload"]
    assert isinstance(payload, dict)
    assert payload["bindings"] == {
        "session_id": "session-1",
        "turn_id": "turn-7",
        "work_run_id": "work-run-9",
        "work_run_revision": 6,
        "attempt_id": "attempt-3",
        "attempt_ordinal": 3,
    }
    assert payload["user_input"] == {
        "content": "我是九月1日出发，返程日期还没定。",
        "prior_waiting_user_question": "请提供出发日期。",
    }
    assert payload["node"] == {
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
    assert payload["acceptance_progress"]["evaluated_output_revision"] == 2
    assert payload["output_window"]["content"] == "# 当前草稿"
    assert payload["prior_tool_results"] == {
        "items": [
                {
                    "tool_id": "read_itinerary_source",
                    "tool_version": "1.0.0",
                    "result_sha256": work_run_tool_result_sha256(
                        context.prior_tool_results.items[0].result
                    ),
                "status": "succeeded",
                "tool_result_id": "result-1",
                "tool_call_id": "call-1",
                "attempt_id": "attempt-2",
                "ordinal": 1,
                "output": {"departure": "2026-09-01"},
                "error_code": None,
                "error_message": None,
            }
        ],
        "truncated": False,
    }
    assert payload["allowed_tools"][0]["tool_id"] == "read_itinerary_source"
    assert "paper_resources" not in payload
    assert payload["verification_feedback"]["acceptance_results"][1] == {
        "acceptance_id": "includes_dates",
        "verdict": "insufficient_evidence",
        "finding": "尚未给出返程日期。",
        "missing_requirements": ["补充返程日期"],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "budget" not in serialized
    assert context.input_limits.profile_id not in serialized
    assert "max_attempts" not in serialized
    assert "reasoning" not in serialized
    assert "ready_for_verification" not in str(received["system_prompt"])
    assert "cannot_progress" not in str(received["system_prompt"])
    assert "Host 会先把全部 Acceptance 自评重置为 false" in str(
        received["system_prompt"]
    )
    assert "未列项将保持 false" in str(received["system_prompt"])
    assert received["kwargs"] == {
        "model_call_id": result.model_call_id,
        "purpose": "runtime_work_run_attempt_decision",
    }
    assert "max_tokens" not in received["kwargs"]
    assert result.value.action.kind == "request_user_input"
    assert [event.status.value for event in events] == ["started", "completed"]


def test_prompt_projects_only_public_paper_resources_and_applies_exact_byte_limit():
    paper_resources = _paper_resources()
    context = _context().model_copy(
        update={"paper_resources": paper_resources}
    )
    received: dict[str, object] = {}

    def complete(system_prompt: str, user_content: str, **kwargs: object) -> ModelResult:
        received["system_prompt"] = system_prompt
        received["payload"] = json.loads(user_content)
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "需要继续读取哪一节？",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    request_attempt_decision(
        context,
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    payload = received["payload"]
    assert isinstance(payload, dict)
    assert payload["paper_resources"] == paper_resources.to_dict()
    serialized_paper_resources = json.dumps(
        payload["paper_resources"],
        ensure_ascii=False,
    )
    for private_value in (
        "session-1",
        "task-1",
        "private-document-alpha",
        "private-version-alpha",
        "a" * 64,
    ):
        assert private_value not in serialized_paper_resources
    assert "path" not in serialized_paper_resources.lower()
    assert "paper_resources 是 Host 冻结的论文范围" in str(
        received["system_prompt"]
    )
    assert "成功 ToolResult" in str(received["system_prompt"])
    assert "[P#:C#]" in str(received["system_prompt"])
    assert "不得声称 Host 已机械验证 claim 覆盖" in str(
        received["system_prompt"]
    )

    exact_bytes = attempt_prompt_serialized_utf8_bytes(context)
    exact_context = context.model_copy(
        update={
            "input_limits": _input_limits(
                max_serialized_utf8_bytes=exact_bytes,
            )
        }
    )
    assert (
        len(serialize_attempt_prompt_payload(exact_context).encode("utf-8"))
        == exact_bytes
    )
    with pytest.raises(AttemptDecisionInputTooLarge) as error:
        serialize_attempt_prompt_payload(
            exact_context.model_copy(
                update={
                    "input_limits": _input_limits(
                        max_serialized_utf8_bytes=exact_bytes - 1,
                    )
                }
            )
        )
    assert error.value.serialized_utf8_bytes == exact_bytes


@pytest.mark.parametrize(
    ("paper_resources", "message"),
    [
        (_paper_resources(session_id="another-session"), "another Session"),
        (_paper_resources(task_id="another-task"), "another Task"),
    ],
)
def test_task_attempt_rejects_cross_authority_paper_resources(
    paper_resources: PaperAttemptContext,
    message: str,
):
    base = _context()
    payload = {
        name: getattr(base, name)
        for name in AttemptDecisionContext.model_fields
    }
    payload["paper_resources"] = paper_resources

    with pytest.raises(ValidationError, match=message):
        AttemptDecisionContext.model_validate(payload)


def test_auxiliary_attempt_rejects_paper_resources_even_for_the_same_task():
    base = _context()
    subject = AuxiliaryNodeSubject(
        task_id=base.subject.task_id,
        auxiliary_graph_id="auxiliary-graph-1",
        auxiliary_graph_revision=1,
        node_id="planning-node",
        node_revision=1,
    )
    payload = {
        name: getattr(base, name)
        for name in AttemptDecisionContext.model_fields
    }
    payload["subject"] = subject
    payload["acceptance_progress"] = base.acceptance_progress.model_copy(
        update={"subject": subject}
    )
    payload["paper_resources"] = _paper_resources()

    with pytest.raises(ValidationError, match="AuxiliaryNode"):
        AttemptDecisionContext.model_validate(payload)


def test_bounded_prior_results_keep_support_and_newest_optional_in_stable_order():
    base = _context().prior_tool_results.items[0]

    def item(ordinal: int) -> PriorToolResultProjection:
        return base.model_copy(
            update={
                "result": base.result.model_copy(
                    update={
                        "tool_result_id": f"result-{ordinal}",
                        "tool_call_id": f"call-{ordinal}",
                        "attempt_id": f"attempt-{ordinal}",
                        "output": {"ordinal": ordinal},
                    }
                )
            }
        )

    oldest, middle, newest = item(1), item(2), item(3)
    expected = PriorToolResultsProjection(
        items=(oldest, newest),
        truncated=True,
    )
    limits = _input_limits(
        max_prior_tool_result_items=2,
        max_prior_tool_results_serialized_utf8_bytes=(
            prior_tool_results_serialized_utf8_bytes(expected)
        ),
    )

    bounded = select_bounded_prior_tool_results(
        (oldest, middle, newest),
        required_result_ids=(oldest.result.tool_result_id,),
        limits=limits,
    )

    assert bounded == expected
    assert tuple(item.result.tool_result_id for item in bounded.items) == (
        "result-1",
        "result-3",
    )


def test_work_run_prompt_compacts_historical_findings_snapshots() -> None:
    old_claim = "this evicted finding must not re-enter the model view"
    result = ToolResult(
        status=ToolResultStatus.SUCCEEDED,
        tool_result_id="result-findings",
        tool_call_id="call-findings",
        attempt_id="attempt-findings",
        ordinal=1,
        output={
            "schema_version": "execution-findings-tool-result-v1",
            "status": "applied",
            "ledger_revision": 1,
            "affected_entry_ids": ["finding-1"],
            "claim": old_claim,
            "ledger": {"entry_revisions": [{"claim": old_claim}]},
            "active_projection": {
                "projection_sha256": "c" * 64,
                "active_entries": [{"claim": old_claim}],
            },
            "replayed": False,
        },
    )
    projection = PriorToolResultsProjection(
        items=(
            PriorToolResultProjection(
                tool_id="record_execution_findings",
                tool_version="1.3.0",
                result=result,
            ),
        )
    )

    payload = build_prior_tool_results_prompt_payload(projection)

    assert payload["items"][0]["result_sha256"] == sha256_json(result)
    assert payload["items"][0]["output"]["schema_version"] == (
        "execution-findings-tool-result-reference-v1"
    )
    assert payload["items"][0]["output"]["source_schema_version"] == (
        "execution-findings-tool-result-v1"
    )
    assert payload["items"][0]["output"]["active_projection"] == {
        "schema_version": "execution-findings-active-projection-reference-v1",
        "compacted": True,
        "projection_sha256": "c" * 64,
    }
    assert "claim" not in payload["items"][0]["output"]
    assert "ledger" not in payload["items"][0]["output"]
    assert old_claim not in json.dumps(payload)



def test_bounded_prior_results_fail_when_required_history_is_missing_or_too_large():
    base = _context().prior_tool_results.items[0]
    required = base.model_copy(
        update={
            "result": base.result.model_copy(
                update={"output": {"城市": "上海"}}
            )
        }
    )

    with pytest.raises(RequiredPriorToolResultsUnavailable) as missing:
        select_bounded_prior_tool_results(
            (required,),
            required_result_ids=("missing-result",),
            limits=_input_limits(max_prior_tool_result_items=1),
        )
    assert missing.value.result_ids == ("missing-result",)

    mandatory_projection = PriorToolResultsProjection(
        items=(required,),
        truncated=False,
    )
    mandatory_bytes = prior_tool_results_serialized_utf8_bytes(
        mandatory_projection
    )
    serialized_text = json.dumps(
        {
            "items": [
                    {
                        "tool_id": required.tool_id,
                        "tool_version": required.tool_version,
                        "result_sha256": work_run_tool_result_sha256(
                            required.result
                        ),
                        **required.result.model_dump(mode="json"),
                }
            ],
            "truncated": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert mandatory_bytes == len(serialized_text.encode("utf-8"))
    assert mandatory_bytes > len(serialized_text)
    with pytest.raises(AttemptDecisionInputTooLarge) as too_large:
        select_bounded_prior_tool_results(
            (required,),
            required_result_ids=(required.result.tool_result_id,),
            limits=_input_limits(
                max_prior_tool_result_items=1,
                max_prior_tool_results_serialized_utf8_bytes=mandatory_bytes - 1,
            ),
        )
    assert too_large.value.entry_count == 1
    assert too_large.value.serialized_utf8_bytes == mandatory_bytes


def test_attempt_input_limit_rejects_before_model_or_events():
    context = _context().model_copy(
        update={
            "input_limits": _input_limits(max_prior_tool_result_items=0)
        }
    )
    provider_calls: list[str] = []
    events = []

    with pytest.raises(AttemptDecisionInputTooLarge) as error:
        request_attempt_decision(
            context,
            emit=events.append,
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: provider_calls.append("called")
            ),
        )

    assert error.value.entry_count == 1
    assert error.value.limits.max_prior_tool_result_items == 0
    assert provider_calls == []
    assert events == []


def test_complete_attempt_input_uses_exact_compact_utf8_boundary_and_one_payload():
    base = _context()
    exact_bytes = attempt_prompt_serialized_utf8_bytes(base)
    context = base.model_copy(
        update={
            "input_limits": _input_limits(
                max_serialized_utf8_bytes=exact_bytes,
            )
        }
    )
    received: list[str] = []

    def complete(_system: str, user: str, **kwargs: object) -> ModelResult:
        received.append(user)
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充日期。",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    request_attempt_decision(
        context,
        provider=as_prepared_test_provider(complete),
        emit=lambda _event: None,
    )

    assert len(received) == 1
    assert len(received[0].encode("utf-8")) == exact_bytes

    provider_calls: list[str] = []
    events = []
    with pytest.raises(AttemptDecisionInputTooLarge) as error:
        request_attempt_decision(
            context.model_copy(
                update={
                    "input_limits": _input_limits(
                        max_serialized_utf8_bytes=exact_bytes - 1,
                    )
                }
            ),
            provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: provider_calls.append("called")
            ),
            emit=events.append,
        )

    assert error.value.code == "attempt_model_input_too_large"
    assert error.value.serialized_utf8_bytes == exact_bytes
    assert provider_calls == []
    assert events == []


def test_complete_attempt_input_bounds_output_window_and_allowed_tool_schema():
    base = _context()
    baseline_bytes = attempt_prompt_serialized_utf8_bytes(base)
    huge_user_input = base.model_copy(
        update={
            "user_input": base.user_input.model_copy(
                update={
                    "content": "当前回答" * 10_000,
                    "prior_waiting_user_question": "先前问题" * 10_000,
                }
            ),
            "input_limits": _input_limits(
                max_serialized_utf8_bytes=baseline_bytes,
            ),
        }
    )
    huge_output = base.model_copy(
        update={
            "output_window": base.output_window.model_copy(
                update={"content": "旅行计划" * 10_000}
            ),
            "input_limits": _input_limits(
                max_serialized_utf8_bytes=baseline_bytes,
            ),
        }
    )
    tool = base.allowed_tools[0]
    huge_schema = base.model_copy(
        update={
            "allowed_tools": (
                ToolSpec(
                    tool_id=tool.tool_id,
                    contract_version=tool.contract_version,
                    name=tool.name,
                    description=tool.description,
                    input_schema={
                        "type": "object",
                        "description": "schema" * 10_000,
                    },
                    output_schema=tool.output_schema,
                    catalog_tags=tool.catalog_tags,
                ),
            ),
            "input_limits": _input_limits(
                max_serialized_utf8_bytes=baseline_bytes,
            ),
        }
    )

    for context in (huge_user_input, huge_output, huge_schema):
        provider_calls: list[str] = []
        events = []
        with pytest.raises(AttemptDecisionInputTooLarge):
            request_attempt_decision(
                context,
                provider=as_prepared_test_provider(
                    lambda *_args, **_kwargs: provider_calls.append("called")
                ),
                emit=events.append,
            )
        assert provider_calls == []
        assert events == []


def test_unsupported_attempt_json_fails_typed_before_model_or_events():
    base = _context()
    unsupported_utf8 = base.model_copy(
        update={
            "output_window": base.output_window.model_copy(
                update={"content": "\ud800"}
            )
        }
    )
    tool = base.allowed_tools[0]
    unsupported_json = base.model_copy(
        update={
            "allowed_tools": (
                ToolSpec(
                    tool_id=tool.tool_id,
                    contract_version=tool.contract_version,
                    name=tool.name,
                    description=tool.description,
                    input_schema={"type": "number", "minimum": float("nan")},
                    output_schema=tool.output_schema,
                    catalog_tags=tool.catalog_tags,
                ),
            )
        }
    )

    for context in (unsupported_utf8, unsupported_json):
        provider_calls: list[str] = []
        events = []
        with pytest.raises(AttemptDecisionInputUnsupported) as error:
            request_attempt_decision(
                context,
                provider=as_prepared_test_provider(
                    lambda *_args, **_kwargs: provider_calls.append("called")
                ),
                emit=events.append,
            )
        assert error.value.code == "attempt_model_input_unsupported"
        assert error.value.profile_id == context.input_limits.profile_id
        assert provider_calls == []
        assert events == []


def test_submit_repairs_a_missing_empty_support_justification() -> None:
    calls = 0
    feedbacks: list[dict[str, object]] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        if calls > 1:
            feedback = repair_feedback_from_provider_kwargs(kwargs)
            assert feedback is not None
            feedbacks.append(feedback)
        empty_item: dict[str, object] = {
            "acceptance_id": "includes_dates",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": [],
        }
        if calls > 1:
            empty_item["empty_support_justification"] = _empty_support(
                "provided_context_sufficient"
            )
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [
                        {
                            "acceptance_id": "accurate_route",
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": ["result-1"],
                        },
                        empty_item,
                    ],
                    "action": {
                        "kind": "submit_output_window",
                        "content": "# Complete itinerary",
                        "format": "markdown",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    requested = request_attempt_decision(
        _context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert [issue["paths"] for issue in feedbacks[0]["current_issues"]] == [
        ["/acceptance_updates/1/empty_support_justification"]
    ]
    assert requested.value.acceptance_updates[1].empty_support_justification


def test_submit_repairs_a_findings_mutation_receipt_used_as_evidence() -> None:
    base = _context()
    context = base.model_copy(
        update={
            "prior_tool_results": base.prior_tool_results.model_copy(
                update={
                    "items": (
                        base.prior_tool_results.items[0].model_copy(
                            update={"tool_id": "record_execution_findings"}
                        ),
                    )
                }
            )
        }
    )
    calls = 0
    feedback_explanations: list[str] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        if calls > 1:
            feedback = repair_feedback_from_provider_kwargs(kwargs)
            assert feedback is not None
            feedback_explanations.extend(
                issue["safe_explanation"] for issue in feedback["current_issues"]
            )
        cite_findings = calls == 1
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [
                        {
                            "acceptance_id": acceptance_id,
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": (
                                ["result-1"]
                                if cite_findings and acceptance_id == "accurate_route"
                                else []
                            ),
                            **(
                                {}
                                if cite_findings
                                and acceptance_id == "accurate_route"
                                else {
                                    "empty_support_justification": _empty_support(
                                        "provided_context_sufficient"
                                    )
                                }
                            ),
                        }
                        for acceptance_id in (
                            "accurate_route",
                            "includes_dates",
                        )
                    ],
                    "action": {
                        "kind": "submit_output_window",
                        "content": "# Complete itinerary",
                        "format": "markdown",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    requested = request_attempt_decision(
        context,
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert feedback_explanations == [
        "该 ID 是执行发现账本的变更回执，不是任务证据。"
    ]


def test_submit_repairs_dependency_delivery_misused_as_tool_result_id() -> None:
    calls = 0
    feedbacks: list[dict[str, object]] = []

    def provider(_system: str, user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        if calls > 1:
            feedback = repair_feedback_from_provider_kwargs(kwargs)
            assert feedback is not None
            feedbacks.append(feedback)
        route_update: dict[str, object] = {
            "acceptance_id": "accurate_route",
            "model_claimed_satisfied": True,
            "supporting_tool_result_ids": (
                ["delivery-child-1"] if calls == 1 else []
            ),
        }
        if calls > 1:
            route_update["empty_support_justification"] = _empty_support(
                "dependency_delivery_sufficient"
            )
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [
                        route_update,
                        {
                            "acceptance_id": "includes_dates",
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": ["result-1"],
                        },
                    ],
                    "action": {
                        "kind": "submit_output_window",
                        "content": "# 当前草稿",
                        "format": "markdown",
                    },
                },
                ensure_ascii=False,
            ),
            str(kwargs["model_call_id"]),
        )

    requested = request_attempt_decision(
        _context(),
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
    )

    assert requested.attempts == 2
    assert len(feedbacks) == 1
    issues = feedbacks[0]["current_issues"]
    assert len(issues) == 1
    assert "supporting_tool_result_ids" in str(issues[0]["paths"])
    assert "dependency delivery IDs are not ToolResult IDs" in str(
        issues[0]["safe_explanation"]
    )
    assert "dependency_delivery_sufficient" in str(
        issues[0]["safe_explanation"]
    )
    repaired = requested.value.acceptance_updates[0]
    assert repaired.supporting_tool_result_ids == ()
    assert repaired.empty_support_justification is not None


@pytest.mark.parametrize(
    "decision",
    [
        {
            "acceptance_updates": [],
            "action": {
                "kind": "write_output_window",
                "content": "# 新工作稿",
                "format": "markdown",
            },
        },
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "accurate_route",
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": ["result-1"],
                },
                {
                    "acceptance_id": "includes_dates",
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": [],
                    "empty_support_justification": _empty_support(
                        "provided_context_sufficient"
                    ),
                },
            ],
            "action": {
                "kind": "submit_output_window",
                "content": "# 最终旅行计划\n出发：9 月 1 日；返程：9 月 8 日。",
                "format": "markdown",
            },
        },
        {
            "acceptance_updates": [],
            "action": {
                "kind": "request_user_input",
                "question": "请提供返程日期。",
            },
        },
    ],
)
def test_valid_output_and_request_user_actions_are_returned_without_mutation(
    decision: dict[str, object],
):
    context = _context()

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        return _model_result(json.dumps(decision), str(kwargs["model_call_id"]))

    result = request_attempt_decision(
        context,
        emit=lambda _event: None,
        provider=as_prepared_test_provider(complete),
    )

    assert result.value.action.kind == decision["action"]["kind"]
    assert context.output_window.output_revision == 2
    assert context.output_window.content == "# 当前草稿"
    assert all(
        not item.model_claimed_satisfied
        for item in context.acceptance_progress.items
    )


@pytest.mark.parametrize(
    "status",
    (
        ToolResultStatus.REJECTED,
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.COMPLETION_UNCONFIRMED,
    ),
)
def test_attempt_cannot_claim_non_succeeded_prior_tool_result_as_support(
    status: ToolResultStatus,
) -> None:
    base = _context()
    failed_result = ToolResult(
        status=status,
        tool_result_id="result-1",
        tool_call_id="call-1",
        attempt_id="attempt-2",
        ordinal=1,
        output=None,
        error_code="tool_did_not_succeed",
        error_message="The tool did not produce successful evidence.",
    )
    context = base.model_copy(
        update={
            "prior_tool_results": PriorToolResultsProjection(
                items=(
                    PriorToolResultProjection(
                        tool_id="read_itinerary_source",
                        tool_version="1.0.0",
                        result=failed_result,
                    ),
                ),
                truncated=False,
            )
        }
    )
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [
                        {
                            "acceptance_id": "accurate_route",
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": ["result-1"],
                        }
                    ],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充返程日期。",
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    with pytest.raises(ModelGatewayError) as error:
        request_attempt_decision(
            context,
            emit=lambda _event: None,
            provider=as_prepared_test_provider(complete),
        )

    assert error.value.code == "MODEL_BAD_RESPONSE"
    assert calls == 6


def test_unknown_tool_is_rejected_then_valid_exposed_tool_is_retried():
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        tool_id = "hidden_tool" if calls == 1 else "read_itinerary_source"
        return _model_result(
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": tool_id,
                                "arguments": {"source_id": "source-1"},
                            }
                        ],
                    },
                }
            ),
            str(kwargs["model_call_id"]),
        )

    result = request_attempt_decision(
        _context(),
        emit=lambda _event: None,
        provider=as_prepared_test_provider(complete),
    )

    assert calls == 2
    assert result.attempts == 2
    assert result.value.action.kind == "call_tools"
    assert result.value.action.calls[0].tool_id == "read_itinerary_source"


def test_unknown_tool_exhaustion_fails_as_invalid_model_output():
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        return _model_result(
            '{"acceptance_updates":[],"action":{"kind":"call_tools",'
            '"calls":[{"tool_id":"hidden_tool","arguments":{}}]}}',
            str(kwargs["model_call_id"]),
        )

    with pytest.raises(ModelGatewayError) as error:
        request_attempt_decision(
            _context(),
            emit=lambda _event: None,
            provider=as_prepared_test_provider(complete),
        )

    assert calls == 6
    assert error.value.code == "MODEL_BAD_RESPONSE"


def test_request_user_path_allows_an_empty_tool_projection():
    received_payload: dict[str, object] = {}

    def complete(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        received_payload.update(json.loads(user_content))
        return _model_result(
            '{"acceptance_updates":[],"action":'
            '{"kind":"request_user_input","question":"请提供返程日期。"}}',
            str(kwargs["model_call_id"]),
        )

    result = request_attempt_decision(
        _context(include_tools=False),
        emit=lambda _event: None,
        provider=as_prepared_test_provider(complete),
    )

    assert received_payload["allowed_tools"] == []
    assert result.value.action.kind == "request_user_input"


def test_closed_world_attempt_repairs_request_user_before_returning_decision():
    calls = 0
    system_prompts: list[str] = []
    repair_feedback: list[dict[str, object]] = []

    def complete(system: str, user_content: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        system_prompts.append(system)
        feedback = repair_feedback_from_provider_kwargs(kwargs)
        if feedback is not None:
            repair_feedback.append(feedback)
        action: dict[str, object]
        updates: list[dict[str, object]]
        if calls == 1:
            updates = []
            action = {
                "kind": "request_user_input",
                "question": "请提供表格截图。",
            }
        else:
            updates = [
                {
                    "acceptance_id": acceptance_id,
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": [],
                    "empty_support_justification": _empty_support(
                        "provided_context_sufficient"
                    ),
                }
                for acceptance_id in ("accurate_route", "includes_dates")
            ]
            action = {
                "kind": "submit_output_window",
                "content": "# 最佳证据输出\n当前授权材料不足以确定返程日期。",
                "format": "markdown",
            }
        return _model_result(
            json.dumps(
                {"acceptance_updates": updates, "action": action},
                ensure_ascii=False,
            ),
            str(kwargs["model_call_id"]),
        )

    result = request_attempt_decision(
        _context(allow_user_input=False),
        emit=lambda _event: None,
        provider=as_prepared_test_provider(complete),
    )

    assert calls == 2
    assert result.value.action.kind == "submit_output_window"
    assert all("request_user_input 被禁止" in item for item in system_prompts)
    assert len(repair_feedback) == 1
    assert [
        issue["paths"] for issue in repair_feedback[0]["current_issues"]
    ] == [["/action/kind"]]
    assert "闭卷执行禁止请求用户输入" in repair_feedback[0]["current_issues"][0]["safe_explanation"]


def test_malformed_structured_output_retries_through_shared_model_wrapper():
    calls = 0

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal calls
        calls += 1
        reply = (
            "not-json"
            if calls == 1
            else '{"acceptance_updates":[],"action":'
            '{"kind":"request_user_input","question":"需要哪一天返程？"}}'
        )
        return _model_result(reply, str(kwargs["model_call_id"]))

    result = request_attempt_decision(
        _context(),
        emit=lambda _event: None,
        provider=as_prepared_test_provider(complete),
    )

    assert calls == 2
    assert result.attempts == 2
    assert result.value.action.kind == "request_user_input"


def test_model_cannot_supply_host_bindings_or_legacy_ready_action():
    replies = iter(
        (
            '{"work_run_id":"other-run","acceptance_updates":[],"action":'
            '{"kind":"request_user_input","question":"问题？"}}',
            '{"acceptance_updates":[],"action":{"kind":"ready_for_verification"}}',
        )
    )

    def complete(_system: str, _user: str, **kwargs: object) -> ModelResult:
        try:
            reply = next(replies)
        except StopIteration:
            reply = '{"acceptance_updates":[],"action":'
            reply += '{"kind":"ready_for_verification"}}'
        return _model_result(reply, str(kwargs["model_call_id"]))

    with pytest.raises(ModelGatewayError) as error:
        request_attempt_decision(
            _context(),
            emit=lambda _event: None,
            provider=as_prepared_test_provider(complete),
        )

    assert error.value.code == "MODEL_BAD_RESPONSE"


def test_context_rejects_cross_workrun_progress_before_provider_call():
    context = _context()

    with pytest.raises(ValidationError, match="AcceptanceProgress"):
        AttemptDecisionContext(
            session_id=context.session_id,
            turn_id=context.turn_id,
            work_run_id="other-run",
            work_run_revision=context.work_run_revision,
            attempt_id=context.attempt_id,
            attempt_ordinal=context.attempt_ordinal,
            user_input=context.user_input,
            subject=context.subject,
            node_title=context.node_title,
            node_objective=context.node_objective,
            acceptances=context.acceptances,
            acceptance_progress=context.acceptance_progress,
            output_window=context.output_window,
            dependency_deliveries=context.dependency_deliveries,
            prior_tool_results=context.prior_tool_results,
            input_limits=context.input_limits,
            allowed_tools=context.allowed_tools,
            verification_feedback=context.verification_feedback,
        )


def test_findings_tools_require_the_exact_work_run_projection() -> None:
    context = _context(include_tools=False)
    registrations = build_execution_findings_tool_registrations()
    payload = context.model_dump(mode="python")
    payload["allowed_tools"] = tuple(item.spec for item in registrations)

    with pytest.raises(ValidationError, match="active projection"):
        AttemptDecisionContext.model_validate(payload)

    projection_payload = {
        "schema_version": "execution-findings-active-projection-v1",
        "ledger_id": derive_execution_findings_ledger_id(
            owner_kind=ExecutionFindingsOwnerKind.WORK_RUN,
            execution_owner_id=context.work_run_id,
        ),
        "ledger_revision": 0,
        "active_entries": (),
        "omitted_active_count": 0,
        "omitted_entry_ids_sha256": None,
        "remaining_durable_revisions": 256,
        "remaining_durable_utf8_bytes": 256_000,
    }
    payload["execution_findings"] = ExecutionFindingsActiveProjection(
        **projection_payload,
        projection_sha256=sha256_json(projection_payload),
    )
    admitted = AttemptDecisionContext.model_validate(payload)
    assert admitted.execution_findings is not None
    assert admitted.execution_findings.ledger_id == projection_payload["ledger_id"]


def test_first_attempt_cannot_claim_a_prior_waiting_user_question():
    payload = _context(include_tools=False).model_dump(mode="python")
    payload["attempt_ordinal"] = 1

    with pytest.raises(ValidationError, match="first Attempt"):
        AttemptDecisionContext.model_validate(payload)
