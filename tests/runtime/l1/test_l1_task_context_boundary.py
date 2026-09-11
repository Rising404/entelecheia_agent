"""L1 的当前请求、验收与工具视图不消费长期 Task 上下文。"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from personagraph.model_io.gateway import ModelGatewayError
from personagraph.output_protocol.l1 import L1PlanProposal, materialize_l1_plan
from personagraph.persistent_turn_content.findings import ExecutionFindingsOwnerKind
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.routing.policy import snapshot_sha256
from personagraph.runtime.l1 import controller, entry_lane, model, semantic_verification
from personagraph.runtime.l1.context import L1TurnContext
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.turn.contracts import EntryExecutionSnapshot
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.session import store as session_store


def test_l1_model_view_requires_no_task_catalog() -> None:
    context = SimpleNamespace(
        history_pairs=({"user": "之前的问题", "assistant": "之前的答复"},),
        session_summary="会话摘要",
        attachments=(),
    )
    payload = controller._model_payload(
        accepted=SimpleNamespace(user_input="回答当前问题", input_message_id="input-1"),
        context=context,
        execution={"attempts": []},
        state={"max_tool_calls_per_attempt": 3},
        current_plan=None,
        tool_runtime=SimpleNamespace(
            attachment_file_catalog=(),
            model_catalog=lambda: (),
        ),
        remaining=3,
        deadline=TurnDeadline.starting_now(60),
        finalization_required=False,
        findings_snapshot=None,
    )

    assert "task_catalog" not in L1TurnContext.__annotations__
    assert "task_references" not in payload
    assert "task_references" not in model._l1_system_prompt()
    assert payload["current_user_text"] == "回答当前问题"
    assert payload["history_pairs"] == list(context.history_pairs)
    assert payload["session_summary"] == context.session_summary
    assert payload["plan"] is None
    assert "acceptance_progress" not in payload
    assert payload["prior_tool_results"] == []


def test_l1_reviewer_does_not_propagate_task_references() -> None:
    user_text = "回答当前问题"
    plan = materialize_l1_plan(
        L1PlanProposal(
            objective=user_text,
            acceptances=(
                {
                    "criterion": user_text,
                },
            ),
        ),
        input_message_id="input-1",
        user_text=user_text,
    )
    model_view = {
        "current_user_text": user_text,
        "history_pairs": [],
        "session_summary": "会话摘要",
        "attachments": [],
        "task_references": [{"insession_task_id": "excluded-task-sentinel"}],
        "prior_tool_results": [],
        "verification_feedback": None,
    }
    review = semantic_verification._semantic_review_view(
        plan=plan,
        reply="还需要补充信息。",
        references=(),
        model_view=model_view,
        execution={"tool_calls": []},
    )

    assert review["request_context"] == {
        key: model_view[key]
        for key in (
            "current_user_text",
            "history_pairs",
            "session_summary",
            "attachments",
        )
    }
    assert "excluded-task-sentinel" not in canonical_json(review)
    assert "task_references" not in semantic_verification._L1_SEMANTIC_SYSTEM_PROMPT
    assert model_view["task_references"] == [
        {"insession_task_id": "excluded-task-sentinel"}
    ]


@pytest.mark.parametrize(
    "function",
    [
        entry_lane.run_l1_entry_lane,
        controller.run_l1_turn,
        controller._run_l1_turn_impl,
    ],
)
def test_l1_execution_interface_has_no_related_task_ids(function) -> None:
    assert "related_insession_task_ids" not in inspect.signature(function).parameters


@pytest.mark.parametrize(
    "task_references", [[], [{"insession_task_id": "task-1"}], None]
)
def test_pending_task_context_stops_before_model_io_without_rewriting_checkpoint(
    monkeypatch,
    tmp_path,
    task_references,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = session_store.create_session("L1 boundary", working_dir=str(workspace))
    features = {"l1_max_attempts": 3}
    accepted = entry_application.accept_entry_turn(
        user_input="回答当前问题",
        session_id=session_id,
        client_request_id="pending-task-context",
        attachment_ids=(),
        execution_snapshot=EntryExecutionSnapshot.create(
            features=features,
            post_commit_job_kinds=(),
        ),
        store=session_store,
    )
    context = SimpleNamespace(
        history_pairs=(),
        session_summary=None,
        attachments=SimpleNamespace(items=()),
    )
    deadline = TurnDeadline.starting_now(300)
    lease_owner = entry_application._runtime_entry_lease_owner()
    prepared = entry_lane.prepare_l1_entry_lane(
        accepted=accepted,
        context=context,
        features=features,
        initial_turn_window_revision=accepted.window_revision,
        routing_policy_snapshot_hash=snapshot_sha256(accepted.routing_policy),
        deadline=deadline,
        store=session_store,
        lease_owner=lease_owner,
    )
    session_store.create_execution_findings_ledger(
        session_id=session_id,
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=prepared.l1_turn_run_id,
    )
    payload = {
        "current_user_text": accepted.user_input,
        "task_references": task_references,
        "execution_limits": {"finalization_required": False},
    }
    request_json = canonical_json(payload)
    request_hash = sha256_json(payload)
    started = session_store.start_l1_attempt(
        session_id=session_id,
        turn_id=accepted.turn_id,
        l1_turn_run_id=prepared.l1_turn_run_id,
        request_json=request_json,
        request_hash=request_hash,
        expected_window_revision=prepared.turn_window_revision,
        expected_lease_owner=lease_owner,
    )

    def unexpected_model_io(*_args, **_kwargs):
        raise AssertionError("pending Task context reached model authority or provider")

    monkeypatch.setattr(
        model, "create_l1_attempt_model_call_authority", unexpected_model_io
    )
    monkeypatch.setattr(model, "complete_structured", unexpected_model_io)
    with pytest.raises(ModelGatewayError, match="unsupported Task context") as caught:
        controller.run_l1_turn(
            accepted=accepted,
            context=context,
            l1_turn_run_id=prepared.l1_turn_run_id,
            initial_turn_window_revision=started["window"]["state_version"],
            deadline=deadline,
            emit=lambda _event: None,
            store=session_store,
            lease_owner=lease_owner,
            execution_features=features,
            frozen_tool_runtime=prepared.tool_runtime,
            frozen_corpus_manifest=prepared.corpus_manifest,
        )

    assert caught.value.code == "MODEL_CONFIGURATION_FAILURE"
    assert caught.value.retryable is False
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "MODEL_CONFIGURATION_FAILURE"
    attempt = execution["attempts"][0]
    assert attempt["request_json"] == started["attempt"]["request_json"]
    assert attempt["request_hash"] == started["attempt"]["request_hash"]
    frozen = json.loads(attempt["request_json"])
    assert frozen["attempt_id"] == attempt["attempt_id"]
    assert frozen["attempt_ordinal"] == 1
    assert {
        key: value
        for key, value in frozen.items()
        if key not in {"attempt_id", "attempt_ordinal"}
    } == payload
    assert (
        session_store.get_runtime_model_logical_call(
            session_id=session_id,
            logical_call_id=attempt["logical_model_call_id"],
        )
        is None
    )
