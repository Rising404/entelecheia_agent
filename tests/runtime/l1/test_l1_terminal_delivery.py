"""L1 收尾交付走真实 Session 事务，失败不暗增决策次数或发布被拒稿。"""

import json

import pytest

from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.l1 import model as l1_model
from personagraph.runtime.l1 import semantic_verification
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.session import store
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _setup(monkeypatch, tmp_path, *, verdict="revise", failure_code=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = store.create_session("Entelecheia", working_dir=str(workspace))
    seen = {"attempt": [], "review": []}

    def result(value, kwargs):
        return ModelResult(
            reply=json.dumps(value),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=kwargs.get("model_call_id"),
        )

    def classifier(*_args, **kwargs):
        return result({"processing_level": "L1", "task_matches": []}, kwargs)

    def attempt(_system, user, **kwargs):
        view = json.loads(user)
        seen["attempt"].append(view)
        if failure_code:
            raise ModelGatewayError(
                failure_code, "PRIVATE_FAILURE_BODY", retryable=False
            )
        return result(
            {
                "plan": {
                    "objective": "回答问题",
                    "acceptances": [
                        {
                            "criterion": "回答问题",
                        }
                    ],
                }
                if view["plan"] is None
                else None,
                "note": "当前材料不足，提交诚实的限制说明。",
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "未经确认的候选：目前无法回答问题。",
                },
            },
            kwargs,
        )

    def review(_system, user, **kwargs):
        view = json.loads(user)
        seen["review"].append(view)
        return result(
            {
                "issues": []
                if verdict == "pass"
                else [{"message": "候选中的理由需要纠正。"}],
            },
            kwargs,
        )

    for module, provider in (
        (ingress_model, classifier),
        (l1_model, attempt),
        (semantic_verification, review),
    ):
        monkeypatch.setattr(
            module,
            "complete_structured",
            as_prepared_test_provider(
                provider,
                add_l1_notes=True,
            ),
        )
    return session_id, seen


def _run(session_id, request_id="terminal-notification"):
    return entry.run_entry_turn(
        user_input="回答问题",
        session_id=session_id,
        client_request_id=request_id,
        features={
            "context_guard_limit": 24000,
            "l1_max_attempts": 24,
            "l1_semantic_verification_mode": "always",
        },
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
            source="request_override",
        ),
        store=store,
    )


def test_last_rejected_candidate_delivers_notice_without_25th_attempt(
    monkeypatch, tmp_path
):
    session_id, seen = _setup(monkeypatch, tmp_path)
    result = _run(session_id)
    assert result.status == "completed"
    assert result.reply.startswith("本轮未完成")
    assert "未经确认的候选" not in result.reply
    assert result.error_code == "VERIFICATION_FAILED"
    assert result.end_reason == "l1_terminal_notification"
    assert len(seen["attempt"]) == len(seen["review"]) == 24
    stop = seen["review"][-1]["execution_context"]["stop"]
    assert stop["candidate_attempt_ordinal"] == 24
    assert stop["attempts_remaining_after_candidate"] == 0
    assert stop["must_finalize"] is True
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution["state"]["attempts_started"] == 24
    assert execution["state"]["failure_code"] == "VERIFICATION_FAILED"
    replayed = _run(session_id)
    assert replayed.reply == result.reply
    assert replayed.error_code == result.error_code
    assert replayed.end_reason == result.end_reason
    assert len(seen["attempt"]) == 24


def test_honest_incomplete_reply_can_pass_without_satisfying_acceptance(
    monkeypatch, tmp_path
):
    session_id, seen = _setup(monkeypatch, tmp_path, verdict="pass")
    result = _run(session_id)
    assert result.status == "completed"
    assert "目前无法回答" in result.reply
    assert result.error_code is None
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert "acceptance_progress_json" not in execution["state"]
    receipt = json.loads(execution["state"]["semantic_verification_report_json"])
    assert receipt["reviewer_result"] == {"issues": []}
    assert len(seen["attempt"]) == len(seen["review"]) == 1
    assert (
        seen["review"][0]["execution_context"]["candidate_note"]
        == "当前材料不足，提交诚实的限制说明。"
    )


@pytest.mark.parametrize(
    "failure_code, public_code",
    [
        ("MODEL_CALL_TIMEOUT", "MODEL_TIMEOUT"),
        ("TURN_DEADLINE_EXCEEDED", "TURN_DEADLINE_EXCEEDED"),
    ],
)
def test_model_failure_has_persisted_safe_notice(
    monkeypatch,
    tmp_path,
    failure_code,
    public_code,
):
    session_id, seen = _setup(monkeypatch, tmp_path, failure_code=failure_code)
    result = _run(session_id)
    assert result.status == "completed"
    assert result.reply.startswith("本轮未完成")
    assert "PRIVATE_FAILURE_BODY" not in result.reply
    assert result.error_code == public_code
    assert len(seen["attempt"]) == 1
    assert seen["review"] == []
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution["state"]["failure_code"] == failure_code


def test_terminal_notification_recovers_lost_commit_reply_without_republishing(
    monkeypatch,
    tmp_path,
):
    session_id, seen = _setup(monkeypatch, tmp_path, failure_code="MODEL_CALL_TIMEOUT")
    finalize = store.finalize_turn_execution
    commits = []

    def commit_then_lose_reply(**kwargs):
        committed = finalize(**kwargs)
        commits.append(committed)
        raise RuntimeError("simulated response loss after commit")

    monkeypatch.setattr(store, "finalize_turn_execution", commit_then_lose_reply)
    result = _run(session_id)
    assert result.status == "completed"
    assert result.reply.startswith("本轮未完成")
    assert result.error_code == "MODEL_TIMEOUT"
    assert result.end_reason == "l1_terminal_notification"
    assert len(commits) == 1
    assert len(seen["attempt"]) == 1
    assert seen["review"] == []
    replayed = _run(session_id)
    assert replayed.reply == result.reply
    assert replayed.error_code == result.error_code
    assert replayed.end_reason == result.end_reason
    assert len(commits) == len(seen["attempt"]) == 1


def test_host_deadline_exhaustion_delivers_notice_without_starting_an_attempt(
    monkeypatch,
    tmp_path,
):
    session_id, seen = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "personagraph.runtime.l1.controller._deadline_from_utc",
        lambda _value: TurnDeadline.starting_now(0),
    )
    result = _run(session_id)
    assert result.status == "completed"
    assert result.reply.startswith("本轮未完成")
    assert result.error_code == "TURN_DEADLINE_EXCEEDED"
    execution = store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "TURN_DEADLINE_EXCEEDED"
    assert execution["attempts"] == []
    assert seen == {"attempt": [], "review": []}


def test_failed_notification_commit_does_not_claim_success(monkeypatch, tmp_path):
    session_id, seen = _setup(monkeypatch, tmp_path, failure_code="MODEL_CALL_TIMEOUT")
    finalize = store.finalize_turn_execution

    def fail_before_commit(**_kwargs):
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(store, "finalize_turn_execution", fail_before_commit)
    result = _run(session_id)
    assert result.status == "incomplete"
    assert result.error_code == "PERSIST_FAILED"
    assert not result.reply
    assert len(seen["attempt"]) == 1
    assert store.get_committed_turn_pair(session_id, f"commit_{result.turn_id}") is None
    monkeypatch.setattr(store, "finalize_turn_execution", finalize)
    replayed = _run(session_id)
    assert replayed.status != "completed"
    assert replayed.window_state == "interrupted"
    assert not replayed.reply
    assert (
        entry.resume_active_l1_entry_turn(
            session_id=session_id,
            features={},
            store=store,
        )
        == "not_waiting"
    )
    assert len(seen["attempt"]) == 1
    assert store.get_committed_turn_pair(session_id, f"commit_{result.turn_id}") is None


@pytest.mark.parametrize("resume_path", ["same_request", "startup_worker"])
def test_failed_run_recovers_notification_after_process_loss_before_commit(
    monkeypatch,
    tmp_path,
    resume_path,
):
    session_id, seen = _setup(monkeypatch, tmp_path, failure_code="MODEL_CALL_TIMEOUT")
    finalize = store.finalize_turn_execution

    def lose_before_commit(**_kwargs):
        raise KeyboardInterrupt("process lost after failure persisted")

    monkeypatch.setattr(store, "finalize_turn_execution", lose_before_commit)
    with pytest.raises(KeyboardInterrupt, match="after failure persisted"):
        _run(session_id)
    inspected = store.inspect_turn_execution(session_id)
    turn_id = inspected["turn"]["turn_id"]
    execution = store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "MODEL_CALL_TIMEOUT"
    assert store.get_committed_turn_pair(session_id, f"commit_{turn_id}") is None
    monkeypatch.setattr(store, "finalize_turn_execution", finalize)
    result = (
        _run(session_id)
        if resume_path == "same_request"
        else entry.resume_active_l1_entry_turn(
            session_id=session_id, features={}, store=store
        )
    )
    assert not isinstance(result, str)
    assert result.status == "completed"
    assert result.reply.startswith("本轮未完成")
    assert result.error_code == "MODEL_TIMEOUT"
    assert result.end_reason == "l1_terminal_notification"
    assert result.turn_id == turn_id
    assert seen["review"] == []
    assert len(seen["attempt"]) == 1
    execution = store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert execution["run"]["status"] == "failed"
