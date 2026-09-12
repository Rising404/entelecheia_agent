from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.l2.auxiliary_execution import (
    production_chain as auxiliary_production_chain,
)
from personagraph.runtime import entry
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import freeze_turn_routing_policy
from personagraph.runtime.turn.contracts import (
    TurnRoutingPolicy,
    TurnRoutingPolicySnapshot,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainPorts,
    AuxiliaryProductionChainRequest,
    AuxiliaryProductionChainStatus,
    run_auxiliary_to_verified_delivery,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
)
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.turn_events import RuntimeErrorCode, RuntimeStage
from personagraph.l2.task_execution.work_run.model_providers import (
    build_attempt_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
    _seed_task,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider
from tests.documents._authority import bound_project_document_authority


@pytest.fixture
def entry_project_documents(tmp_path):
    with bound_project_document_authority(tmp_path) as authority:
        yield authority


def _l2_routing_policy() -> TurnRoutingPolicySnapshot:
    return freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=False, l2_enabled=True),
        source="request_override",
    )


def _seed_task_shells(*, count: int = 1) -> tuple[str, tuple[str, ...]]:
    session_id = store.create_session("Entelecheia")
    user_text = (
        "创建材料分析任务" if count == 1 else "创建甲任务和乙任务"
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-entry-seed",
        source="runtime_test",
        user_text=user_text,
        lease_owner="aux-v2-entry-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    matches = (
        [
            {
                "match_type": "new_root",
                "local_key": "analysis",
                "title": "材料分析",
                "objective": "理解材料并输出有依据的结论",
                "source_excerpt": user_text,
            }
        ]
        if count == 1
        else [
            {
                "match_type": "new_root",
                "local_key": "alpha",
                "title": "甲任务",
                "objective": "完成甲任务",
                "source_excerpt": "甲任务",
            },
            {
                "match_type": "new_root",
                "local_key": "beta",
                "title": "乙任务",
                "objective": "完成乙任务",
                "source_excerpt": "乙任务",
            },
        ]
    )
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-entry-seed-tasks",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {"task_matches": matches}
        ),
        exposed_catalog_ids=(),
        expected_window_revision=int(window["state_version"]),
    )
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(window["state_version"]),
        processing_level="L2",
        assistant_content="任务已建立。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    ordered_ids = tuple(
        applied.created_insession_task_ids_by_local_key[key]
        for key in (["analysis"] if count == 1 else ["alpha", "beta"])
    )
    return session_id, ordered_ids


def _classify_one(task_id: str):
    def classify(*_args, **_kwargs) -> EntryClassification:
        return EntryClassification.model_validate(
            {
                "processing_level": "L2",
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": "继续执行已有任务",
                        "execute_current": True,
                    }
                ],
            }
        )

    return classify


def _install_classifier_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str,
    source_excerpt: str,
) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []

    def complete_classifier(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
        **_kwargs: object,
    ) -> ModelResult:
        assert purpose == "runtime_entry_classify"
        payload = json.loads(user_content)
        requests.append(payload)
        assert any(
            item["insession_task_id"] == task_id
            for item in payload["task_catalog"]
        )
        return ModelResult(
            reply=json.dumps(
                {
                    "processing_level": "L2",
                    "task_matches": [
                        {
                            "match_type": "existing_root",
                            "insession_task_id": task_id,
                            "source_excerpt": source_excerpt,
                            "execute_current": True,
                        }
                    ],
                }
            ),
            provider="mock",
            model="entry-classifier-boundary",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(complete_classifier),
    )
    return requests


def _settle_interrupted_turn(
    *,
    session_id: str,
    turn_id: str,
    interruption_reason: str = "TURN_DEADLINE_EXCEEDED",
) -> None:
    window = store.get_turn_execution_window(session_id)
    assert window is not None and window["turn_id"] == turn_id
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(window["state_version"]),
        stage=RuntimeStage.RESPONSE.value,
        interruption_reason=interruption_reason,
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="host_stopped",
        error_code=interruption_reason,
    )


def test_entry_ready_uses_verified_finalizer_and_http_replay_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    entry_project_documents,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    classifier_requests = _install_classifier_provider(
        monkeypatch,
        task_id=task_id,
        source_excerpt="继续执行已有任务",
    )
    physical_chain_calls = 0
    real_chain = auxiliary_production_chain.run_auxiliary_to_verified_delivery

    def counted_chain(*args, **kwargs):
        nonlocal physical_chain_calls
        physical_chain_calls += 1
        result = real_chain(*args, **kwargs)
        if result.status is AuxiliaryProductionChainStatus.DELIVERY_READY:
            # 入口必须忽略这个便利投影，并通过权威终结器重新加载冻结交付。
            return replace(result, publication_body="untrusted-chain-projection")
        return result

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        counted_chain,
    )
    request = dict(
        user_input="继续执行已有任务",
        features={
            "context_guard_limit": 24_000,
        },
        session_id=session_id,
        client_request_id="aux-v2-entry-ready",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    first = entry.run_entry_turn(**request)

    assert first.status == "completed"
    assert first.processing_level == "L2"
    assert first.reply
    assert first.related_insession_task_ids == (task_id,)
    assert physical_chain_calls == 1
    assert len(classifier_requests) == 1
    delivery_id = work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    )
    delivery = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=delivery_id,
    )
    assert first.reply == delivery.output_window.content
    assert first.reply != "untrusted-chain-projection"

    replayed = entry.run_entry_turn(**request)

    assert replayed.status == "completed"
    assert replayed.turn_id == first.turn_id
    assert replayed.reply == first.reply
    assert physical_chain_calls == 1
    assert len(classifier_requests) == 1


def test_entry_waiting_user_is_durable_incomplete_and_never_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    classifier_requests = _install_classifier_provider(
        monkeypatch,
        task_id=task_id,
        source_excerpt="继续执行已有任务",
    )
    chain_calls = 0

    def waiting_chain(*_args, **_kwargs):
        nonlocal chain_calls
        chain_calls += 1
        return SimpleNamespace(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            final_delivery_id=None,
            publication_body=None,
            requested_user_questions=(),
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        waiting_chain,
    )
    request = dict(
        user_input="继续执行已有任务",
        features={},
        session_id=session_id,
        client_request_id="aux-v2-entry-waiting",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    waiting = entry.run_entry_turn(**request)

    assert waiting.status == "incomplete"
    assert waiting.reply is None
    assert waiting.processing_level == "L2"
    assert waiting.window_state == "interrupted"
    assert waiting.error_code == RuntimeErrorCode.TRANSITION_DENIED.value
    assert chain_calls == 1
    assert len(classifier_requests) == 1
    with store._connect() as conn:
        assistant_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_turns "
                "WHERE session_id=? AND role='assistant'",
                (session_id,),
            ).fetchone()[0]
        )
    assert assistant_count == 1  # 只有种子回复；等待中的轮次为私有状态。

    replayed = entry.run_entry_turn(**request)

    # 现有入口重放契约会把持久中断标记公开为仍在运行的轮次，直到下一输入审计
    # 将其稳定。此处重要的重放不变量是不得产生第二次链路效果或发布。
    assert replayed.status == "running"
    assert replayed.reply is None
    assert replayed.turn_id == waiting.turn_id
    assert replayed.window_state == "interrupted"
    assert chain_calls == 1
    assert len(classifier_requests) == 1


def test_entry_model_bad_response_is_settled_as_model_output_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    _install_classifier_provider(
        monkeypatch,
        task_id=task_id,
        source_excerpt="继续执行已有任务",
    )

    def invalid_model_output(*_args, **_kwargs):
        raise ModelGatewayError(
            "MODEL_BAD_RESPONSE",
            "typed Attempt decision retries exhausted",
            retryable=False,
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        invalid_model_output,
    )

    result = entry.run_entry_turn(
        user_input="继续执行已有任务",
        features={},
        session_id=session_id,
        client_request_id="aux-v2-entry-model-output-invalid",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L2"
    assert result.reply is None
    assert result.end_reason == "provider_unavailable"
    assert result.error_code == RuntimeErrorCode.MODEL_OUTPUT_INVALID.value
    assert result.window_state == "interrupted"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["interruption_reason"] == (
        RuntimeErrorCode.MODEL_OUTPUT_INVALID.value
    )


def test_entry_durable_user_gate_exact_answer_resumes_same_work_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    _no_mounted_documents(monkeypatch, session_id)
    question = "请明确希望计划覆盖最近几年。"
    answer = "覆盖最近三年。"
    classified_inputs: list[str] = []

    def classify(context, *_args, **_kwargs) -> EntryClassification:
        user_text = context.envelope.user_text
        classified_inputs.append(user_text)
        if user_text == answer:
            catalog_item = next(
                item
                for item in context.task_catalog.items
                if item.insession_task_id == task_id
            )
            assert catalog_item.pending_user_question == question
        return EntryClassification.model_validate(
            {
                "processing_level": "L2",
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": user_text,
                        "execute_current": True,
                    }
                ],
            }
        )

    monkeypatch.setattr(entry_application, "classify_turn", classify)
    physical_planning = build_auxiliary_architect_structured_provider()
    physical_attempt = build_attempt_structured_provider(
        AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None
        ).work_run_model_profile
    )
    planning_calls = 0
    gate_phases: list[str] = []
    terminal_calls = 0

    def planning_provider(*args, **kwargs):
        nonlocal planning_calls
        planning_calls += 1
        result = physical_planning(*args, **kwargs)
        proposal = json.loads(result.reply)
        node = proposal["structure"]["nodes"][0]
        node.update(
            {
                "node_kind": "clarify",
                "executor_kind": "user_gate",
                "title": "Clarify the blocking planning input",
                "objective": question,
                "acceptance_criteria": [
                    {
                        "acceptance_id": "user_answer_bound",
                        "criterion": "The exact user answer is durably bound.",
                        "source_anchor_ids": node["source_anchor_ids"],
                    }
                ],
                "capability_profile_id": None,
                "input_resource_aliases": [],
                "output_contract": "user_response_v1",
            }
        )
        return replace(
            result,
            reply=json.dumps(proposal, ensure_ascii=False),
        )

    def attempt_provider(system_prompt, user_content, **kwargs):
        nonlocal terminal_calls
        payload = json.loads(user_content)
        contract = payload.get("user_gate_contract")
        if contract is None:
            terminal_calls += 1
            return physical_attempt(system_prompt, user_content, **kwargs)
        phase = contract["phase"]
        gate_phases.append(phase)
        reply = (
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": question,
                },
            }
            if phase == "ask"
            else {
                "acceptance_updates": [
                    {
                        "acceptance_id": "user_answer_bound",
                        "model_claimed_satisfied": True,
                    }
                ],
                "action": {
                    "kind": "submit_output_window",
                    "content": "accept_answer",
                    "format": "plain_text",
                },
            }
        )
        return ModelResult(
            reply=json.dumps(reply, ensure_ascii=False),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=kwargs["model_call_id"],
            finish_reason="stop",
        )

    real_chain = auxiliary_production_chain.run_auxiliary_to_verified_delivery
    chain_calls = 0
    chain_results: list[tuple[str, str]] = []

    def controlled_chain(request, *, ports):
        nonlocal chain_calls
        chain_calls += 1
        result = real_chain(
            request,
            ports=replace(
                ports,
                auxiliary=replace(
                    ports.auxiliary,
                    planning_provider=as_prepared_test_provider(
                        planning_provider
                    ),
                    attempt_provider=as_prepared_test_provider(attempt_provider),
                ),
            ),
        )
        chain_results.append((result.status.value, result.reason_code))
        return result

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        controlled_chain,
    )
    features = {}
    question_request = dict(
        user_input="继续执行已有任务",
        features=features,
        session_id=session_id,
        client_request_id="aux-v2-entry-user-gate-question",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    waiting = entry.run_entry_turn(**question_request)

    assert waiting.status == "completed"
    assert waiting.reply == question
    pending = continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert pending is not None and pending.question == question
    gate_work_run_id = pending.work_run_id
    assert tuple(
        item.attempt.ordinal
        for item in work_run_store.get_work_run(
            session_id=session_id,
            work_run_id=gate_work_run_id,
        ).attempts
    ) == (1,)
    waiting_task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert waiting_task is not None
    assert waiting_task.current_graph_revision is None
    assert waiting_task.status.value == "awaiting_user"
    with store._connect() as conn:
        assistant_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_turns "
                "WHERE session_id=? AND role='assistant'",
                (session_id,),
            ).fetchone()[0]
        )
    assert assistant_count == 2  # 种子回复加上一条精确门禁问题。

    waiting_replay = entry.run_entry_turn(**question_request)

    assert waiting_replay.status == "completed"
    assert waiting_replay.reply == question
    assert chain_calls == 1
    assert gate_phases == ["ask"]
    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="entry-user-gate-test-worker",
        lease_seconds=60,
    )
    for job in claimed:
        store.mark_turn_post_commit_job_applied(
            job_id=str(job["job_id"]),
            worker_id="entry-user-gate-test-worker",
        )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=waiting.turn_id,
        expected_window_revision=waiting.window_revision,
    )

    answer_request = dict(
        user_input=answer,
        features=features,
        session_id=session_id,
        client_request_id="aux-v2-entry-user-gate-answer",
        routing_policy=_l2_routing_policy(),
        store=store,
    )
    completed = entry.run_entry_turn(**answer_request)

    assert completed.status == "completed"
    assert completed.reply
    assert completed.related_insession_task_ids == (task_id,)
    assert classified_inputs == ["继续执行已有任务", answer]
    assert chain_calls == 2
    assert [status for status, _reason in chain_results] == [
        "waiting_user",
        "delivery_ready",
    ]
    assert planning_calls == 1
    assert gate_phases == ["ask", "consume_answer"]
    assert terminal_calls == 1
    gate_work_run = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=gate_work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in gate_work_run.attempts) == (1, 2)
    assert continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    ) is None
    assert work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    )

    replayed = entry.run_entry_turn(**answer_request)

    assert replayed.status == "completed"
    assert replayed.turn_id == completed.turn_id
    assert replayed.reply == completed.reply
    assert chain_calls == 2
    assert gate_phases == ["ask", "consume_answer"]


def test_entry_task_node_question_is_published_and_exact_answer_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已提交的 TaskGraph 问题是公开且可恢复的交互。"""

    session_id, (task_id,) = _seed_task_shells()
    _no_mounted_documents(monkeypatch, session_id)
    question = "请明确最终回答面向专家还是普通读者。"
    answer = "面向普通读者。"
    classified_inputs: list[str] = []

    def classify(context, *_args, **_kwargs) -> EntryClassification:
        user_text = context.envelope.user_text
        classified_inputs.append(user_text)
        if user_text == answer:
            catalog_item = next(
                item
                for item in context.task_catalog.items
                if item.insession_task_id == task_id
            )
            assert catalog_item.pending_user_question == question
        return EntryClassification.model_validate(
            {
                "processing_level": "L2",
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": user_text,
                        "execute_current": True,
                    }
                ],
            }
        )

    monkeypatch.setattr(entry_application, "classify_turn", classify)
    real_chain = auxiliary_production_chain.run_auxiliary_to_verified_delivery
    task_attempt_payloads: list[dict[str, object]] = []
    chain_statuses: list[str] = []

    def controlled_chain(request, *, ports):
        def task_attempt_provider(system_prompt, user_content, **kwargs):
            payload = json.loads(user_content)
            task_attempt_payloads.append(payload)
            if payload["user_input"]["prior_waiting_user_question"] is None:
                return ModelResult(
                    reply=json.dumps(
                        {
                            "acceptance_updates": [],
                            "action": {
                                "kind": "request_user_input",
                                "question": question,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    provider="mock",
                    model="mock-structured",
                    latency_ms=1,
                    model_call_id=kwargs["model_call_id"],
                    finish_reason="stop",
                )
            return ModelResult(
                reply=json.dumps(
                    {
                        "acceptance_updates": [
                            {
                                "acceptance_id": item["acceptance_id"],
                                "model_claimed_satisfied": True,
                                "supporting_tool_result_ids": [],
                                "empty_support_justification": {
                                    "schema_version": (
                                        "empty-support-justification-v1"
                                    ),
                                    "reason_code": (
                                        "candidate_is_primary_artifact"
                                    ),
                                    "explanation": (
                                        "本次提交正文就是该条件要求的主要交付物。"
                                    ),
                                },
                            }
                            for item in payload["node"]["acceptances"]
                        ],
                        "action": {
                            "kind": "submit_output_window",
                            "content": "已按普通读者所需的表达方式完成回答。",
                            "format": "plain_text",
                        },
                    },
                    ensure_ascii=False,
                ),
                provider="mock",
                model="mock-structured",
                latency_ms=1,
                model_call_id=kwargs["model_call_id"],
                finish_reason="stop",
            )

        result = real_chain(
            request,
            ports=replace(
                ports,
                delivery=replace(
                    ports.delivery,
                    attempt_provider=as_prepared_test_provider(
                        task_attempt_provider
                    ),
                ),
            ),
        )
        chain_statuses.append(result.status.value)
        return result

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        controlled_chain,
    )
    features = {}
    waiting = entry.run_entry_turn(
        user_input="继续执行已有任务",
        features=features,
        session_id=session_id,
        client_request_id="aux-v2-entry-task-node-question",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert waiting.status == "completed"
    assert waiting.reply == question
    pending = tuple(
        item
        for item in continuation_store.list_pending_user_questions(session_id=session_id)
        if getattr(item.subject, "task_id", None) == task_id
    )
    assert len(pending) == 1
    work_run_id = pending[0].work_run_id
    assert tuple(
        item.attempt.ordinal
        for item in work_run_store.get_work_run(
            session_id=session_id,
            work_run_id=work_run_id,
        ).attempts
    ) == (1,)

    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="entry-task-node-question-test-worker",
        lease_seconds=60,
    )
    for job in claimed:
        store.mark_turn_post_commit_job_applied(
            job_id=str(job["job_id"]),
            worker_id="entry-task-node-question-test-worker",
        )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=waiting.turn_id,
        expected_window_revision=waiting.window_revision,
    )

    completed = entry.run_entry_turn(
        user_input=answer,
        features=features,
        session_id=session_id,
        client_request_id="aux-v2-entry-task-node-answer",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert completed.status == "completed"
    assert completed.reply
    assert completed.reply != question
    assert classified_inputs == ["继续执行已有任务", answer]
    assert chain_statuses == ["waiting_user", "delivery_ready"]
    assert [payload["user_input"] for payload in task_attempt_payloads] == [
        {
            "content": "继续执行已有任务",
            "prior_waiting_user_question": None,
        },
        {
            "content": answer,
            "prior_waiting_user_question": question,
        },
    ]
    work_run = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in work_run.attempts) == (1, 2)
    assert continuation_store.list_pending_user_questions(session_id=session_id) == ()


def test_entry_explicit_target_change_enters_production_chain_with_durable_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    user_input = "请把材料分析目标改成只比较实验设计"
    excerpt = "目标改成只比较实验设计"
    replacement = "只比较实验设计"
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        lambda *_args, **_kwargs: EntryClassification.model_validate(
            {
                "processing_level": "L2",
                "task_matches": [
                    {
                        "match_type": "existing_root_target_change",
                        "insession_task_id": task_id,
                        "replacement_objective": replacement,
                        "source_excerpt": excerpt,
                        "execute_current": True,
                    }
                ],
            }
        ),
    )
    chain_task_ids: list[str] = []

    def waiting_chain(request, **_kwargs):
        chain_task_ids.append(request.task_id)
        return SimpleNamespace(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            final_delivery_id=None,
            publication_body=None,
            requested_user_questions=(),
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        waiting_chain,
    )

    result = entry.run_entry_turn(
        user_input=user_input,
        features={},
        session_id=session_id,
        client_request_id="aux-v2-entry-target-change",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "incomplete"
    assert chain_task_ids == [task_id]
    manifest = task_graph_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    lane_match = manifest.lanes[0].matches[0]
    assert lane_match.match_type == "existing_root_target_change"
    assert lane_match.replacement_objective == replacement


def test_entry_new_turn_resumes_persisted_active_graph_without_demo_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)

    stopped = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=first_turn_id,
            task_id=task_id,
            max_auxiliary_effect_steps=1,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=lambda _event: None
            ),
            delivery=AuxiliaryTaskDeliveryPorts(
                model_ledger_store=store,
                emit=lambda _event: None
            ),
        ),
    )

    assert stopped.status is AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED
    active_graph = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert active_graph is not None
    assert active_graph.goal_status == "active"
    _settle_interrupted_turn(
        session_id=session_id,
        turn_id=first_turn_id,
    )

    classifier_requests = _install_classifier_provider(
        monkeypatch,
        task_id=task_id,
        source_excerpt="继续执行已有任务",
    )
    request = dict(
        user_input="继续执行已有任务",
        features={},
        session_id=session_id,
        client_request_id="aux-v2-cross-turn-resume",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    resumed = entry.run_entry_turn(**request)

    assert resumed.status == "completed"
    assert resumed.processing_level == "L2"
    assert resumed.related_insession_task_ids == (task_id,)
    assert len(classifier_requests) == 1
    assert task_graph_store.get_insession_task_details(
        session_id,
        task_id,
    ).status.value == "completed"
    assert work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    )

    # 即使原始执行从规划轮次不同于当前回答轮次的图开始，HTTP 重试也只进行投影。
    replayed = entry.run_entry_turn(**request)
    assert replayed.status == "completed"
    assert replayed.turn_id == resumed.turn_id
    assert replayed.reply == resumed.reply


def test_entry_cross_turn_resumes_active_n_plus_one_trigger_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.current_auxiliary_delivery import (
        settle_current_auxiliary_candidate,
    )

    session_id, _turn_id, task_id, delivery_result = settle_current_auxiliary_candidate(
        monkeypatch,
        route="replan_task_graph",
    )
    assert delivery_result.status.value == "revision_required"
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    _settle_interrupted_turn(
        session_id=session_id,
        turn_id=trigger.created_turn_id,
        interruption_reason="INTERNAL_FAILURE",
    )
    classifier_requests = _install_classifier_provider(
        monkeypatch,
        task_id=task_id,
        source_excerpt="继续执行已有任务",
    )
    request = dict(
        user_input="继续执行已有任务",
        features={},
        session_id=session_id,
        client_request_id="aux-v2-cross-turn-positive-base",
        routing_policy=_l2_routing_policy(),
        store=store,
    )
    result = entry.run_entry_turn(**request)

    assert result.status == "completed"
    assert result.reply
    assert result.turn_id != trigger.created_turn_id
    assert len(classifier_requests) == 1
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == trigger.target_graph_revision == 2
    assert task.status.value == "completed"
    delivery_id = work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    )
    delivery = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=delivery_id,
    )
    assert delivery.delivery.subject.graph_revision == 2
    assert result.reply == delivery.output_window.content
    candidate_validation = task_delivery_store.get_task_delivery_candidate_settlement(
        session_id=session_id,
        task_id=task_id,
        graph_revision=2,
    )
    assert candidate_validation is not None
    assert candidate_validation.intent.result.disposition.value == "pass"
    assert candidate_validation.intent.invocation_turn_id == result.turn_id
    assert (
        candidate_validation.intent.review_request.invocation_turn_id
        == result.turn_id
    )
    assert candidate_validation.settlement.created_turn_id == result.turn_id
    assert candidate_validation.settlement.root_delivery_id == delivery_id
    assert (
        candidate_validation.settlement.completed_task_state_version
        == task.task_state_version
        == candidate_validation.settlement.settled_task_state_version
    )
    assert (
        candidate_validation.settlement.verification_result_id
        == candidate_validation.intent.result.verification_result_id
    )
    assert (
        candidate_validation.settlement.result_sha256
        == candidate_validation.intent.result.result_sha256
    )
    with store._connect() as conn:
        application = conn.execute(
            "SELECT application.base_graph_revision, "
            "application.committed_graph_revision, "
            "application.consumed_turn_id, trigger.created_turn_id FROM "
            "insession_task_graph_revision_trigger_applications AS application "
            "JOIN insession_task_graph_revision_triggers AS trigger "
            "ON trigger.trigger_id=application.trigger_id "
            "WHERE application.trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone()
        positive_model_call = conn.execute(
            "SELECT invocation_turn_id FROM "
            "insession_runtime_model_logical_calls WHERE session_id=? "
            "AND logical_call_id LIKE 'auxv2positive_%:model'",
            (session_id,),
        ).fetchone()
        candidate_call_count_before_replay = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind="
                "'task_delivery_candidate_validation'",
                (session_id,),
            ).fetchone()[0]
        )
    assert application is not None
    assert int(application["base_graph_revision"]) == 1
    assert int(application["committed_graph_revision"]) == 2
    assert str(application["created_turn_id"]) == trigger.created_turn_id
    assert str(application["consumed_turn_id"]) == result.turn_id
    assert positive_model_call is not None
    assert str(positive_model_call["invocation_turn_id"]) == result.turn_id

    replayed = entry.run_entry_turn(**request)

    assert replayed.status == "completed"
    assert replayed.turn_id == result.turn_id
    assert replayed.reply == result.reply
    assert len(classifier_requests) == 1
    assert task_delivery_store.get_task_delivery_candidate_settlement(
        session_id=session_id,
        task_id=task_id,
        graph_revision=2,
    ) == candidate_validation
    with store._connect() as conn:
        candidate_call_count_after_replay = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind="
                "'task_delivery_candidate_validation'",
                (session_id,),
            ).fetchone()[0]
        )
    assert candidate_call_count_after_replay == candidate_call_count_before_replay


def test_entry_cross_turn_selector_fails_closed_on_unbound_user_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_tasks SET current_status='awaiting_user', "
            "state_version=state_version+1 WHERE session_id=? "
            "AND insession_task_id=?",
            (session_id, task_id),
        )
    assert not tuple(
        question
        for question in continuation_store.list_pending_user_questions(
            session_id=session_id
        )
        if getattr(getattr(question, "subject", None), "task_id", None)
        == task_id
    )
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        _classify_one(task_id),
    )
    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        lambda *_args, **_kwargs: pytest.fail(
            "an unbound UserGate must not enter the Auxiliary chain"
        ),
    )
    monkeypatch.setattr(
        entry_application,
        "generate_response",
        lambda *_args, **_kwargs: pytest.fail(
            "corrupt L2 authority must not degrade to a plain reply"
        ),
    )

    result = entry.run_entry_turn(
        user_input="继续执行已有任务",
        features={},
        session_id=session_id,
        client_request_id="aux-v2-unbound-user-gate",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L2"
    assert result.end_reason == "persistence_error"
    assert result.error_code == "PERSIST_FAILED"
    assert result.window_state == "interrupted"


def test_entry_hard_cut_defaults_on_for_one_authenticated_existing_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        _classify_one(task_id),
    )
    chain_calls: list[AuxiliaryProductionChainRequest] = []

    def waiting_chain(request, **_kwargs):
        chain_calls.append(request)
        return SimpleNamespace(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            final_delivery_id=None,
            publication_body=None,
            requested_user_questions=("请补充完成任务所需的信息。",),
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        waiting_chain,
    )

    result = entry.run_entry_turn(
        user_input="继续执行已有任务",
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="aux-v2-entry-default-off",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "completed"
    assert result.processing_level == "L2"
    assert result.reply == "请补充完成任务所需的信息。"
    assert len(chain_calls) == 1
    assert chain_calls[0].task_id == task_id


def test_entry_hard_cut_executes_one_new_root_from_guarded_lane_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = store.create_session("Entelecheia")
    user_input = "创建材料分析任务"
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        lambda *_args, **_kwargs: EntryClassification.model_validate(
            {
                "processing_level": "L2",
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "analysis",
                        "title": "材料分析",
                        "objective": "理解材料并输出有依据的结论",
                        "source_excerpt": user_input,
                    }
                ],
            }
        ),
    )
    chain_calls: list[AuxiliaryProductionChainRequest] = []

    def waiting_chain(request, **_kwargs):
        chain_calls.append(request)
        return SimpleNamespace(
            status=AuxiliaryProductionChainStatus.WAITING_USER,
            final_delivery_id=None,
            publication_body=None,
            requested_user_questions=("请提供需要分析的材料。",),
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        waiting_chain,
    )

    result = entry.run_entry_turn(
        user_input=user_input,
        features={},
        session_id=session_id,
        client_request_id="aux-v2-entry-new-root-hard-cut",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "completed"
    assert result.processing_level == "L2"
    assert result.reply == "请提供需要分析的材料。"
    assert len(chain_calls) == 1
    assert result.related_insession_task_ids == (chain_calls[0].task_id,)
    manifest = task_graph_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert len(manifest.lanes) == 1
    assert manifest.lanes[0].insession_task_id == chain_calls[0].task_id
    assert manifest.lanes[0].matches[0].match_type == "new_root"


@pytest.mark.parametrize("shape", ("none", "multiple"))
def test_entry_ambiguous_task_selection_falls_back_without_running_chain(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    session_id, task_ids = _seed_task_shells(count=2 if shape == "multiple" else 1)
    user_input = (
        "继续甲任务并继续乙任务" if shape == "multiple" else "只分析当前消息"
    )
    matches = (
        [
            {
                "match_type": "existing_root",
                "insession_task_id": task_ids[0],
                "source_excerpt": "继续甲任务",
                "execute_current": True,
            },
            {
                "match_type": "existing_root",
                "insession_task_id": task_ids[1],
                "source_excerpt": "继续乙任务",
                "execute_current": True,
            },
        ]
        if shape == "multiple"
        else []
    )
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        lambda *_args, **_kwargs: EntryClassification.model_validate(
            {"processing_level": "L2", "task_matches": matches}
        ),
    )
    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        lambda *_args, **_kwargs: pytest.fail("ambiguous Entry invoked "),
    )

    result = entry.run_entry_turn(
        user_input=user_input,
        features={},
        session_id=session_id,
        client_request_id=f"aux-v2-entry-{shape}",
        routing_policy=_l2_routing_policy(),
        store=store,
    )

    assert result.status == "completed"
    assert result.processing_level == "L2"
    assert result.reply
