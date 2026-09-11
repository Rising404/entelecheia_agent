"""真实 L1/SQLite/dispatcher 串行多工具；模型使用本地替身，故障点显式注入。"""

from __future__ import annotations

import json

import pytest

from personagraph.model_io.gateway import ModelResult
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.l1 import model as l1_model
from personagraph.runtime.l1.protected_tool_dispatch import ToolExecutor
from personagraph.session import store as session_store
from personagraph.tools.contracts import ExecutionStatus
from personagraph.trajectory import StepKind, active_store
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _model_result(payload, model_call_id):
    return ModelResult(
        reply=json.dumps(payload, ensure_ascii=False),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(model_call_id),
    )


def _write_call(path: str, content: str) -> dict:
    return {
        "tool_id": "write_workspace_file",
        "arguments": {"path": path, "content": content},
    }


def _assert_unconfirmed(response, turn_id):
    assert response.status == "incomplete", response
    assert response.processing_level == "L1"
    assert response.error_code == "TOOL_COMPLETION_UNCONFIRMED"
    assert response.end_reason == "tool_completion_unconfirmed"
    assert response.window_state == "interrupted"
    assert response.turn_id == turn_id


@pytest.fixture
def serial_batch(monkeypatch, tmp_path, bound_partitioned_session):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    model_inputs = []
    executions = []
    original_execute = ToolExecutor.execute

    def execute(self, invocation):
        identity = (invocation.registration.tool_id, dict(invocation.arguments))
        executions.append(("start", identity))
        outcome = original_execute(self, invocation)
        executions.append(("end", identity))
        return outcome

    monkeypatch.setattr(ToolExecutor, "execute", execute)
    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: _model_result(
                {"processing_level": "L1", "task_matches": []},
                kwargs["model_call_id"],
            ),
        ),
    )

    def configure(calls, expected_statuses, *, allow_workspace_write=True):
        if allow_workspace_write:
            session_store.grant_session_workspace_write_authority(session_id)

        def decide(_system, user_content, **kwargs):
            payload = json.loads(user_content)
            model_inputs.append(payload)
            if len(model_inputs) == 1:
                decision = {
                    "note": "按声明顺序执行独立工具，并如实汇报结果。",
                    "plan": {
                        "objective": "执行多个工具",
                        "acceptances": [
                            {"criterion": "逐项如实报告工具执行情况"},
                        ],
                    },
                    "action": {"kind": "call_tools", "calls": calls},
                }
            else:
                assert len(model_inputs) == 2, (
                    "工具批次不应进入模型格式修复或插入额外模型调用"
                )
                results = payload["prior_tool_results"]
                assert [item["tool_id"] for item in results] == [
                    call["tool_id"] for call in calls
                ]
                assert [item["status"] for item in results] == expected_statuses
                decision = {
                    "note": "已收到全部工具结果；保留失败信息，不把失败当成功。",
                    "references": [
                        {"tool_result_id": item["tool_result_id"]}
                        for item in results
                        if item["status"] == "succeeded"
                    ],
                    "action": {
                        "kind": "submit_final_reply",
                        "reply": "已按实际结果完成工具执行检查。",
                    },
                }
            return _model_result(decision, kwargs["model_call_id"])

        monkeypatch.setattr(
            l1_model, "complete_structured", as_prepared_test_provider(decide)
        )
        return dict(
            user_input="按顺序执行这些独立工具，并如实报告结果。",
            features={
                "context_guard_limit": 24000,
                "l1_max_attempts": 4,
                "l1_semantic_verification_mode": "off",
            },
            session_id=session_id,
            client_request_id="serial-protected-batch",
            routing_policy=freeze_turn_routing_policy(
                TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
                source="request_override",
            ),
            store=session_store,
        )

    return workspace, session_id, model_inputs, executions, configure


@pytest.mark.parametrize("second_tool", ["write_workspace_file", "create_output_file"])
def test_multiple_protected_calls_execute_serially_and_replay_without_redispatch(
    serial_batch,
    second_tool,
):
    workspace, session_id, model_inputs, executions, configure = serial_batch
    calls = [
        _write_call("first.txt", "first"),
        {"tool_id": "get_today", "arguments": {}},
        _write_call("second.txt", "second"),
    ]
    calls[2]["tool_id"] = second_tool
    request = configure(calls, ["succeeded"] * 3)
    response = entry.run_entry_turn(**request)
    assert response.status == "completed", (
        response,
        [item.get("prior_tool_results") for item in model_inputs],
        executions,
    )
    assert len(model_inputs) == 2
    assert "max_protected_calls_per_attempt" not in model_inputs[0]["execution_limits"]
    assert "protected_tool_ids" not in model_inputs[0]["execution_limits"]
    assert (workspace / "first.txt").read_text() == "first"
    second_root = (
        workspace / "output" if second_tool == "create_output_file" else workspace
    )
    assert (second_root / "second.txt").read_text() == "second"
    assert executions == [
        (phase, (call["tool_id"], call["arguments"]))
        for call in calls
        for phase in ("start", "end")
    ]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=response.turn_id
    )
    stored = execution["tool_calls"]
    assert len(stored) == 3
    assert len({item["tool_call_id"] for item in stored}) == 3
    assert len({item["attempt_id"] for item in stored}) == 1
    protected = [
        item for item in stored if item["execution_class"] == "protected_effect"
    ]
    assert len(protected) == 2
    assert len({item["physical_attempt_id"] for item in protected}) == 2
    for item in protected:
        assert item["status"] == item["protected_phase"] == "succeeded"
        receipt = json.loads(item["protected_receipt_json"])
        assert receipt["physical_attempt_id"] == item["physical_attempt_id"]
        assert receipt["outcome_status"] == "succeeded"
    assert (
        len(
            [
                step
                for step in active_store().steps_for_turn(response.turn_id)
                if step.kind is StepKind.TOOL_CALL
            ]
        )
        == 3
    )
    assert entry.run_entry_turn(**request).status == "completed"
    assert len(model_inputs) == 2 and len(executions) == 6


@pytest.mark.parametrize("failure", ["schema", "business", "authorization"])
def test_one_tool_failure_does_not_erase_other_serial_results(serial_batch, failure):
    workspace, session_id, model_inputs, executions, configure = serial_batch
    calls = [
        _write_call("first.txt", "first"),
        _write_call("blocked.txt", "blocked"),
        _write_call("last.txt", "last"),
    ]
    expected_status = "rejected"
    error_code = "invalid_tool_input"
    if failure == "schema":
        del calls[1]["arguments"]["content"]
    elif failure == "business":
        calls[1]["arguments"]["path"] = "missing-parent/blocked.txt"
        expected_status, error_code = "failed", "parent_directory_missing"
    else:
        # Output 新文件默认允许，但普通 workspace 写入仍需要它自己的授权。
        calls[0]["tool_id"] = calls[2]["tool_id"] = "create_output_file"
        error_code = "tool_policy_authorization_required"
    request = configure(
        calls,
        ["succeeded", expected_status, "succeeded"],
        allow_workspace_write=failure != "authorization",
    )
    response = entry.run_entry_turn(**request)
    assert response.status == "completed", (
        response,
        [item.get("prior_tool_results") for item in model_inputs],
    )
    root = workspace / "output" if failure == "authorization" else workspace
    assert (root / "first.txt").read_text() == "first"
    assert (root / "last.txt").read_text() == "last"
    assert not (workspace / calls[1]["arguments"]["path"]).exists()
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=response.turn_id
    )
    stored = execution["tool_calls"]
    assert [item["status"] for item in stored] == [
        "succeeded",
        expected_status,
        "succeeded",
    ]
    error = model_inputs[1]["prior_tool_results"][1]["error"]
    assert error["code"] == error_code and error["message"]
    assert json.loads(stored[1]["outcome_json"])["error"] == error
    assert len(executions) == (6 if failure == "business" else 4)
    assert (
        len(
            [
                step
                for step in active_store().steps_for_turn(response.turn_id)
                if step.kind is StepKind.TOOL_CALL and step.reason_code == error_code
            ]
        )
        == 1
    )
    assert entry.run_entry_turn(**request).status == "completed"
    assert len(model_inputs) == 2
    assert len(executions) == (6 if failure == "business" else 4)


@pytest.mark.parametrize("interruption", ["after_first_settle", "after_second_reserve"])
def test_serial_batch_resumes_after_settled_tool_without_repeating_it(
    serial_batch,
    monkeypatch,
    interruption,
):
    workspace, session_id, model_inputs, executions, configure = serial_batch
    calls = [_write_call("first.txt", "first"), _write_call("second.txt", "second")]
    request = configure(calls, ["succeeded", "succeeded"])
    original_settle = session_store.settle_l1_protected_tool_dispatch
    original_reserve = session_store.reserve_l1_tool_call
    interrupted = False

    def settle(**kwargs):
        nonlocal interrupted
        settled = original_settle(**kwargs)
        if interruption == "after_first_settle" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt(interruption)
        return settled

    def reserve(**kwargs):
        nonlocal interrupted
        reserved = original_reserve(**kwargs)
        if (
            interruption == "after_second_reserve"
            and kwargs["call_ordinal"] == 2
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt(interruption)
        return reserved

    monkeypatch.setattr(session_store, "settle_l1_protected_tool_dispatch", settle)
    monkeypatch.setattr(session_store, "reserve_l1_tool_call", reserve)
    with pytest.raises(KeyboardInterrupt, match=interruption):
        entry.run_entry_turn(**request)
    assert len(model_inputs) == 1 and len(executions) == 2
    assert (workspace / "first.txt").read_text() == "first"
    assert not (workspace / "second.txt").exists()
    turn_id = str(session_store.inspect_turn_execution(session_id)["turn"]["turn_id"])
    before = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert before["attempts"][0]["status"] == "active"
    assert before["tool_calls"][0]["protected_phase"] == "succeeded"
    if interruption == "after_second_reserve":
        assert before["tool_calls"][1]["status"] == "pending"
        assert before["tool_calls"][1]["protected_phase"] == "ready"
    response = entry.run_entry_turn(**request)
    if interruption == "after_second_reserve":
        # 当前恢复边界对 pending 一律保守停止，不在这次放开批次数量时扩权限。
        _assert_unconfirmed(response, turn_id)
        assert len(model_inputs) == 1 and len(executions) == 2
        assert not (workspace / "second.txt").exists()
        after = session_store.get_l1_turn_execution(
            session_id=session_id, turn_id=turn_id
        )
        for previous, current in zip(before["tool_calls"], after["tool_calls"]):
            for key in (
                "tool_call_id",
                "status",
                "protected_phase",
                "outcome_json",
                "physical_attempt_id",
            ):
                assert previous[key] == current[key]
        _assert_unconfirmed(entry.run_entry_turn(**request), turn_id)
        assert len(model_inputs) == 1 and len(executions) == 2
        return
    assert response.status == "completed", response
    assert response.turn_id == turn_id
    assert len(model_inputs) == 2 and len(executions) == 4
    assert (workspace / "second.txt").read_text() == "second"
    after = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    first_before, first_after = before["tool_calls"][0], after["tool_calls"][0]
    for key in (
        "tool_call_id",
        "arguments_hash",
        "policy_json",
        "physical_attempt_id",
        "protected_operation_binding_sha256",
        "outcome_hash",
        "outcome_json",
        "protected_receipt_json",
    ):
        assert first_before[key] == first_after[key]
    assert len({item["attempt_id"] for item in after["tool_calls"]}) == 1
    assert len({item["physical_attempt_id"] for item in after["tool_calls"]}) == 2
    assert entry.run_entry_turn(**request).status == "completed"
    assert len(model_inputs) == 2 and len(executions) == 4


def test_second_protected_dispatch_with_uncertain_completion_is_never_repeated(
    serial_batch,
    monkeypatch,
):
    workspace, session_id, model_inputs, executions, configure = serial_batch
    calls = [
        _write_call("first.txt", "first"),
        _write_call("second.txt", "once"),
        _write_call("third.txt", "third"),
    ]
    calls[1]["arguments"]["mode"] = "append"
    request = configure(calls, ["succeeded", "completion_unconfirmed", "succeeded"])
    original_execute = ToolExecutor.execute
    interrupted = False

    def execute(self, invocation):
        nonlocal interrupted
        outcome = original_execute(self, invocation)
        if invocation.arguments.get("path") == "second.txt" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("second_write_completed_before_settle")
        return outcome

    monkeypatch.setattr(ToolExecutor, "execute", execute)
    with pytest.raises(KeyboardInterrupt, match="second_write_completed_before_settle"):
        entry.run_entry_turn(**request)
    assert len(model_inputs) == 1 and len(executions) == 4
    assert (workspace / "second.txt").read_text() == "once"
    assert not (workspace / "third.txt").exists()
    turn_id = str(session_store.inspect_turn_execution(session_id)["turn"]["turn_id"])
    before = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert [item["protected_phase"] for item in before["tool_calls"]] == [
        "succeeded",
        "dispatching",
    ]
    assert before["tool_calls"][1]["status"] == "pending"
    response = entry.run_entry_turn(**request)
    _assert_unconfirmed(response, turn_id)
    assert len(model_inputs) == 1 and len(executions) == 4
    assert (workspace / "second.txt").read_text() == "once", (
        "append 不得重放导致重复副作用"
    )
    assert not (workspace / "third.txt").exists()
    after = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert [item["status"] for item in after["tool_calls"]] == ["succeeded", "pending"]
    second_before, second_after = before["tool_calls"][1], after["tool_calls"][1]
    assert second_before["tool_call_id"] == second_after["tool_call_id"]
    assert second_before["physical_attempt_id"] == second_after["physical_attempt_id"]
    assert second_after["protected_phase"] == "dispatching"
    _assert_unconfirmed(entry.run_entry_turn(**request), turn_id)
    assert len(model_inputs) == 1 and len(executions) == 4


def test_uncertain_protected_outcome_stops_before_next_serial_tool(
    serial_batch, monkeypatch
):
    workspace, session_id, model_inputs, executions, configure = serial_batch
    calls = [
        _write_call("first.txt", "first"),
        _write_call("uncertain.txt", "unknown"),
        _write_call("must-not-start.txt", "third"),
    ]
    request = configure(calls, ["succeeded", "completion_unconfirmed", "succeeded"])
    original_execute = ToolExecutor._execute

    def execute(self, invocation):
        if invocation.arguments.get("path") == "uncertain.txt":
            # 与有副作用同步 handler 超时同一 outcome 构造；保留外层 executor 的 trajectory。
            return self._interrupted_outcome(
                invocation.registration,
                "deadline_exceeded",
                "Worker completion is not confirmed.",
                {},
            )
        return original_execute(self, invocation)

    monkeypatch.setattr(ToolExecutor, "_execute", execute)
    response = entry.run_entry_turn(**request)
    _assert_unconfirmed(response, response.turn_id)
    assert len(model_inputs) == 1 and len(executions) == 4
    assert (workspace / "first.txt").read_text() == "first"
    assert not (workspace / "must-not-start.txt").exists()
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=response.turn_id
    )
    assert [item["status"] for item in execution["tool_calls"]] == [
        "succeeded",
        ExecutionStatus.COMPLETION_UNCONFIRMED.value,
    ]
    uncertain = execution["tool_calls"][1]
    receipt = json.loads(uncertain["protected_receipt_json"])
    assert receipt["physical_attempt_id"] == uncertain["physical_attempt_id"]
    assert receipt["outcome_status"] == "completion_unconfirmed"
    assert json.loads(uncertain["outcome_json"])["error"]["code"] == "deadline_exceeded"
    assert (
        len(
            [
                step
                for step in active_store().steps_for_turn(response.turn_id)
                if step.kind is StepKind.TOOL_CALL
                and step.reason_code == "deadline_exceeded"
            ]
        )
        == 1
    )
    _assert_unconfirmed(entry.run_entry_turn(**request), response.turn_id)
    assert len(model_inputs) == 1 and len(executions) == 4
