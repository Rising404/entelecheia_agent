"""真实 Entry/ToolExecutor/SQLite/后台索引等待；模型仅用本地脚本替身。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
import threading

import pytest
from PIL import Image

from personagraph.model_io.gateway import ModelResult
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.l1 import model as l1_model
from personagraph.runtime.l1 import recovery as l1_recovery
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session import store as session_store
from personagraph.tools.execution import ToolExecutor
from personagraph.tools.files.file_tools import derive_file_preparation_tool_request_id
from personagraph.trajectory import StepKind, active_store
from personagraph.workspace.ingestion.composition import build_document_maintenance_lifecycle
from personagraph.workspace.ingestion.worker import DocumentMaintenanceWorker
from personagraph.workspace.storage.context import connect_current
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _model_result(payload, model_call_id):
    return ModelResult(
        reply=json.dumps(payload, ensure_ascii=False),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(model_call_id),
    )


@pytest.mark.parametrize("interrupt_before_settle", [False, True])
def test_background_preparation_settles_original_call_without_polling_model_attempts(
    monkeypatch, tmp_path, bound_partitioned_session, interrupt_before_settle,
):
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "northstar.txt").write_text(
        "Northstar validation accuracy is 71 percent.", encoding="utf-8",
    )
    session_id = bound_partitioned_session(working_dir=workspace)
    entered_indexing = threading.Event()
    release_indexing = threading.Event()
    model_inputs = []
    tool_executions = []
    indexed_jobs = []
    original_index = DocumentMaintenanceWorker._index_and_prove
    original_execute = ToolExecutor.execute

    def index(self, job):
        indexed_jobs.append(job.job_id)
        entered_indexing.set()
        assert release_indexing.wait(10), "test did not release the real background indexer"
        return original_index(self, job)

    def execute(self, invocation):
        if invocation.registration.tool_id == "prepare_files":
            assert callable(invocation.continuation_check)
            tool_executions.append(invocation.logical_tool_call_id)
        outcome = original_execute(self, invocation)
        if interrupt_before_settle and len(tool_executions) == 1:
            raise KeyboardInterrupt("prepared_before_runtime_settle")
        return outcome

    monkeypatch.setattr(DocumentMaintenanceWorker, "_index_and_prove", index)
    monkeypatch.setattr(ToolExecutor, "execute", execute)
    monkeypatch.setattr(
        ingress_model, "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: _model_result(
                {"processing_level": "L1", "task_matches": []}, kwargs["model_call_id"],
            ),
        ),
    )

    def decide(_system, user_content, **kwargs):
        payload = json.loads(user_content)
        model_inputs.append(payload)
        if len(model_inputs) == 1:
            decision = {
                "note": "准备指定文件，等待工具报告真实完成状态。",
                "plan": {
                    "objective": "将 northstar.txt 准备为可检索文件",
                    "acceptances": [{"criterion": "文件准备完成并如实报告状态"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [{
                        "tool_id": "prepare_files",
                        "arguments": {"files": [{"path": "northstar.txt"}]},
                    }],
                },
            }
        else:
            assert len(model_inputs) == 2, "后台等待不应新增模型轮询或格式修复调用"
            results = payload["prior_tool_results"]
            assert len(results) == 1 and results[0]["tool_id"] == "prepare_files"
            assert results[0]["status"] == "succeeded", results
            prepared = results[0]["result"]
            assert prepared["ready_indices"] == [0], prepared
            assert prepared["not_ready_indices"] == []
            assert prepared["results"][0]["status"] == "ready"
            assert prepared["results"][0]["document_version_id"]
            decision = {
                "note": "已收到同一工具调用返回的准备完成结果。",
                "references": [{"tool_result_id": results[0]["tool_result_id"]}],
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "northstar.txt 已准备完成，可以检索。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr(
        l1_model, "complete_structured", as_prepared_test_provider(decide),
    )
    features = {
        "context_guard_limit": 24000,
        "l1_max_attempts": 4,
        "l1_semantic_verification_mode": "off",
        "file_retrieval_write_enabled": True,
        "file_retrieval_read_enabled": True,
        "l1_retrieval_tools_enabled": True,
    }
    request = dict(
        user_input="请准备 northstar.txt，完成后告诉我是否可检索。",
        features=features,
        session_id=session_id,
        client_request_id="background-file-preparation",
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
            source="request_override",
        ),
        store=session_store,
    )
    lifecycle = build_document_maintenance_lifecycle(profile=DocumentRetrievalProfile.lexical())
    assert lifecycle.start() is True
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(copy_context().run, entry.run_entry_turn, **request)
            try:
                assert entered_indexing.wait(30), (
                    future.result() if future.done() else "background indexing did not start"
                )
                assert not future.done()
                assert len(model_inputs) == 1
                inspection = session_store.inspect_turn_execution(session_id)
                turn_id = inspection["turn"]["turn_id"]
                assert inspection["window"]["window_state"] == "active"
                before = session_store.get_l1_turn_execution(
                    session_id=session_id, turn_id=turn_id,
                )
                assert len(before["attempts"]) == 1
                assert before["attempts"][0]["status"] == "active"
                assert len(before["tool_calls"]) == 1
                pending_call = before["tool_calls"][0]
                assert pending_call["status"] == "pending"
                assert pending_call["execution_class"] == "protected_effect"
                assert pending_call["protected_phase"] == "dispatching"
                assert pending_call["outcome_json"] is None
                assert tool_executions == [pending_call["tool_call_id"]]
                request_id = "file-tool-request:" + hashlib.sha256(
                    f"{pending_call['tool_call_id']}:0".encode(),
                ).hexdigest()
                with connect_current() as conn:
                    requests = list(conn.execute("SELECT * FROM document_ingest_requests"))
                    assert len(requests) == 1
                    assert requests[0]["request_id"] == request_id
                    assert requests[0]["session_id"] == session_id
                    assert requests[0]["delivery_status"] == "pending"
                    job = conn.execute("SELECT * FROM document_ingest_jobs").fetchone()
                    assert job["status"] == "processing" and job["stage"] == "indexing"
                    assert job["attempts"] == 1
            finally:
                release_indexing.set()
            if interrupt_before_settle:
                with pytest.raises(KeyboardInterrupt, match="prepared_before_runtime_settle"):
                    future.result(timeout=10)
            else:
                response = future.result(timeout=10)
    finally:
        release_indexing.set()
        assert lifecycle.stop(timeout_seconds=3) is True

    if interrupt_before_settle:
        interrupted = session_store.get_l1_turn_execution(
            session_id=session_id, turn_id=turn_id,
        )
        assert len(model_inputs) == 1
        assert len(interrupted["attempts"]) == 1
        assert interrupted["tool_calls"][0]["status"] == "pending"
        assert interrupted["tool_calls"][0]["outcome_json"] is None
        # 重建 Entry/工具 runtime；新 owner 从持久 request 读取已完成后台工作的结果。
        response = entry.run_entry_turn(**request)

    assert response.status == "completed", response
    assert response.processing_level == "L1" and response.error_code is None
    assert response.reply == "northstar.txt 已准备完成，可以检索。"
    assert len(model_inputs) == 2
    after = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert len(after["attempts"]) == 2 and len(after["tool_calls"]) == 1
    settled_call = after["tool_calls"][0]
    assert settled_call["tool_call_id"] == pending_call["tool_call_id"]
    assert settled_call["attempt_id"] == pending_call["attempt_id"]
    assert settled_call["physical_attempt_id"] == pending_call["physical_attempt_id"]
    assert after["state"]["deadline_at"] == before["state"]["deadline_at"]
    assert settled_call["status"] == "succeeded"
    outcome = json.loads(settled_call["outcome_json"])
    assert outcome["result"]["ready_indices"] == [0]
    with connect_current() as conn:
        requests = list(conn.execute("SELECT * FROM document_ingest_requests"))
        assert len(requests) == 1 and requests[0]["request_id"] == request_id
        assert requests[0]["delivery_status"] == "mounted"
        jobs = list(conn.execute("SELECT * FROM document_ingest_jobs"))
        assert len(jobs) == 1 and jobs[0]["status"] == "applied"
        assert jobs[0]["attempts"] == 1
    assert indexed_jobs == [jobs[0]["job_id"]]
    runtime = build_l1_tool_runtime(session_id, execution_features=features)
    matched = runtime.registrations_by_tool_id["retrieve_files"].handler({
        "queries": ["Northstar validation accuracy"],
    })
    assert matched["outcome"] == "matched", matched
    assert "71 percent" in matched["evidence"][0]["snippet"]
    expected_dispatches = 2 if interrupt_before_settle else 1
    # 记录两次真实的观察调用，不伪装成零成本；持久 ToolCall/physical receipt 仍唯一。
    assert len([
        step for step in active_store().steps_for_turn(turn_id)
        if step.kind is StepKind.TOOL_CALL and step.purpose == "prepare_files"
    ]) == expected_dispatches
    assert entry.run_entry_turn(**request).status == "completed"
    assert len(model_inputs) == 2
    assert tool_executions == [pending_call["tool_call_id"]] * expected_dispatches


@pytest.mark.parametrize("tool_id", ["prepare_files", "get_today"])
def test_pending_image_preparation_and_other_tools_are_not_automatically_replayed(
    monkeypatch, tmp_path, bound_partitioned_session, tool_id,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Image.new("RGB", (24, 16)).save(workspace / "chart.png")
    session_id = bound_partitioned_session(working_dir=workspace)
    model_calls = []
    tool_calls = []
    original_execute = ToolExecutor.execute
    monkeypatch.setattr(
        ingress_model, "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: _model_result(
                {"processing_level": "L1", "task_matches": []}, kwargs["model_call_id"],
            ),
        ),
    )

    def decide(_system, _user_content, **kwargs):
        model_calls.append(kwargs["model_call_id"])
        assert len(model_calls) == 1, "未确认工具不得通过再问模型绕过恢复边界"
        return _model_result({
            "note": "调用指定工具并如实报告状态。",
            "plan": {"objective": "执行工具", "acceptances": [{"criterion": "报告真实结果"}]},
            "action": {"kind": "call_tools", "calls": [{
                "tool_id": tool_id,
                "arguments": {"files": [{"path": "chart.png"}]} if tool_id == "prepare_files" else {},
            }]},
        }, kwargs["model_call_id"])

    def execute(self, invocation):
        tool_calls.append(invocation.logical_tool_call_id)
        outcome = original_execute(self, invocation)
        assert outcome.status.value == "succeeded"
        raise KeyboardInterrupt("unbound_tool_before_settle")

    monkeypatch.setattr(l1_model, "complete_structured", as_prepared_test_provider(decide))
    monkeypatch.setattr(ToolExecutor, "execute", execute)
    request = dict(
        user_input="执行指定工具。", session_id=session_id,
        client_request_id="unbound-tool-recovery",
        features={
            "context_guard_limit": 24000, "l1_semantic_verification_mode": "off",
            "file_retrieval_write_enabled": True, "file_retrieval_read_enabled": True,
            "l1_retrieval_tools_enabled": True,
        },
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True, l2_enabled=False), source="request_override",
        ),
        store=session_store,
    )
    with pytest.raises(KeyboardInterrupt, match="unbound_tool_before_settle"):
        entry.run_entry_turn(**request)
    inspected = session_store.inspect_turn_execution(session_id)
    turn_id = inspected["turn"]["turn_id"]
    before = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert before["tool_calls"][0]["status"] == "pending"
    with connect_current() as conn:
        assert conn.execute("SELECT count(*) FROM document_ingest_requests").fetchone()[0] == 0
    for _ in range(2):
        result = entry.run_entry_turn(**request)
        assert result.status == "incomplete"
        assert result.error_code == "TOOL_COMPLETION_UNCONFIRMED"
        assert result.end_reason == "tool_completion_unconfirmed"
    after = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
    assert len(model_calls) == len(tool_calls) == 1
    assert after["tool_calls"] == before["tool_calls"]


@pytest.mark.parametrize("corruption", [
    "foreign_session", "unknown_completion", "unknown_phase", "unstarted",
    "invalid_arguments", "arguments_hash", "partially_bound_batch",
])
def test_file_preparation_resume_requires_exact_pending_call_and_every_input(
    monkeypatch, corruption,
):
    arguments = {"files": [{"path": "first.txt"}, {"path": "second.txt"}]}
    call = {
        "session_id": "session", "tool_id": "prepare_files", "tool_call_id": "call",
        "status": "pending", "execution_class": "protected_effect",
        "protected_phase": "dispatching", "outcome_json": None,
        "arguments_json": canonical_json(arguments), "arguments_hash": sha256_json(arguments),
    }
    looked_up = []

    def request_exists(*, session_id, request_id):
        looked_up.append((session_id, request_id))
        return not (
            corruption == "partially_bound_batch"
            and request_id == derive_file_preparation_tool_request_id("call", 1)
        )

    monkeypatch.setattr(l1_recovery, "has_file_preparation_request", request_exists)
    if corruption == "foreign_session":
        call["session_id"] = "another-session"
    elif corruption == "unknown_completion":
        call["status"] = "completion_unconfirmed"
    elif corruption == "unknown_phase":
        call["protected_phase"] = "invented"
    elif corruption == "unstarted":
        call["protected_phase"] = "ready"
    elif corruption == "invalid_arguments":
        invalid = {"files": []}
        call["arguments_json"] = canonical_json(invalid)
        call["arguments_hash"] = sha256_json(invalid)
    elif corruption == "arguments_hash":
        call["arguments_hash"] = "0" * 64

    assert l1_recovery.can_resume_file_preparation_call(
        session_id="session", tool_call=call,
    ) is False
    if corruption == "partially_bound_batch":
        assert looked_up == [
            ("session", derive_file_preparation_tool_request_id("call", index))
            for index in range(2)
        ]
    else:
        assert looked_up == []
