"""另外两个关键入口留下的记录。

事件流已经能够说明工具调用失败或守卫拒绝，却无法说明请求了什么，而这恰是
解释二者的唯一信息。这些测试覆盖参数、查询，以及目前最大的缺口——被拒绝
内容本身。
"""

from __future__ import annotations

import json

import pytest

from personagraph.trajectory import PartRole, StepKind, StepOutcome, TrajectoryStore
from personagraph.trajectory import recorder as recorder_module


@pytest.fixture
def store(tmp_path, monkeypatch) -> TrajectoryStore:
    from personagraph.trajectory import store as store_module

    isolated = TrajectoryStore(tmp_path / "trajectory.sqlite")
    monkeypatch.setattr(store_module, "_ACTIVE", isolated)
    return isolated


# --- 工具调用 --------------------------------------------------------------------


def _tool_registration(tool_id: str, handler):
    """映射 tests/tools 中的构建器；保留本地副本以允许二者独立演进。"""

    from personagraph.tools.contracts import (
        ToolSourceDescriptor,
        ToolSourceKind,
        ToolSpec,
    )
    from personagraph.tools.effects import (
        EffectAction,
        EffectDescriptor,
        EffectResource,
        EffectScopeKind,
        ToolEffectProfile,
    )
    from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration

    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="1.0.0",
            name=tool_id,
            description="A probe tool.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["content"],
                "properties": {"content": {"type": "string"}},
            },
            catalog_tags=("read",),
        ),
        implementation_version="impl-1",
        source=ToolSourceDescriptor(ToolSourceKind.LOCAL, "test"),
        handler=handler,
        effect_profile=ToolEffectProfile(
            (EffectDescriptor(EffectResource.MEMORY, EffectAction.READ, EffectScopeKind.LOCAL),)
        ),
        execution_profile=ToolExecutionProfile(),
    )


def _run(registration, arguments):
    from personagraph.tools.execution import ResolvedInvocation, ToolExecutor

    return ToolExecutor().execute(
        ResolvedInvocation(registration=registration, arguments=arguments)
    )


def test_a_successful_tool_call_keeps_both_sides(store):
    registration = _tool_registration("probe_read", lambda payload: {"content": "hi"})
    outcome = _run(registration, {"path": "/tmp/x"})
    assert outcome.status.value == "succeeded"

    step = next(s for s in _all_steps(store) if s.kind is StepKind.TOOL_CALL)
    kept = {part.role: part.blob.text for part in step.parts}
    assert '"path"' in kept[PartRole.TOOL_ARGUMENTS]
    assert "hi" in kept[PartRole.TOOL_RESULT]
    assert step.purpose == "probe_read"
    assert step.outcome is StepOutcome.OK


def test_executor_records_frozen_arguments_and_nested_results_as_json(store):
    from personagraph.tools.contracts import ToolCallProposal

    result = {
        "content": "prepared",
        "results": [{"status": "ready", "error": None, "ready_indices": [0, 1]}],
        "observations": [{"text": "图表内容", "uncertainty": None}],
        "empty": {},
    }
    proposal = ToolCallProposal(tool_id="prepare_files", arguments={"path": "report.pdf"})
    outcome = _run(_tool_registration("prepare_files", lambda payload: result), proposal.arguments)
    assert outcome.status.value == "succeeded"

    step, = _all_steps(store)
    parts = {part.role: json.loads(part.blob.text) for part in step.parts}
    assert parts[PartRole.TOOL_ARGUMENTS] == {"path": "report.pdf"}
    assert parts[PartRole.TOOL_RESULT] == result == outcome.to_dict()["result"]
    # Logging receives a JSON copy; the authority result remains frozen.
    with pytest.raises(TypeError):
        outcome.result["results"][0]["status"] = "changed"


def test_invalid_circular_arguments_still_return_rejection_when_recording(store):
    arguments = {}
    arguments["path"] = arguments
    outcome = _run(_tool_registration("probe_read", lambda payload: {"content": "hi"}), arguments)
    assert outcome.status.value == "rejected"
    assert outcome.error.code == "invalid_tool_input"
    step, = _all_steps(store)
    assert step.outcome is StepOutcome.REJECTED


def test_successful_empty_tool_result_is_preserved_as_an_object(store):
    recorder_module.record_tool_call(
        tool_id="probe_empty", arguments={}, status="succeeded", result={},
        error_code=None, duration_ms=0, store=store,
    )
    step, = _all_steps(store)
    result, = [part for part in step.parts if part.role is PartRole.TOOL_RESULT]
    assert json.loads(result.blob.text) == {}


def test_executor_preserves_visual_failure_details_in_trajectory(store):
    from personagraph.tools.execution import ToolBusinessFailure

    details = {
        "reason_code": "vision_request_timeout",
        "background_wait_active": False,
        "automatic_retries": 0,
        "failure_diagnostics": {
            "phase": "request", "exception_type": "TimeoutError",
            "elapsed_ms": 120001, "timeout_s": 120.0,
            "completion_uncertain": True,
        },
    }

    def failed(_payload):
        raise ToolBusinessFailure("visual_completion_unconfirmed", "No usable receipt.", details)

    outcome = _run(_tool_registration("analyze_pdf_page", failed), {"path": "report.pdf"})
    step, = _all_steps(store)
    result_parts = [part for part in step.parts if part.role is PartRole.TOOL_RESULT]
    assert len(result_parts) == 1
    assert json.loads(result_parts[0].blob.text)["error"] == outcome.error.to_dict()
    assert json.loads(result_parts[0].blob.text)["error"]["details"] == details


def test_arguments_are_kept_even_when_the_call_never_ran(store):
    """输入被拒绝时，参数本身就能解释全部原因。"""

    registration = _tool_registration("probe_read", lambda payload: {"content": "hi"})
    outcome = _run(registration, {"wrong_field": 1})
    assert outcome.status.value == "rejected"

    step = next(s for s in _all_steps(store) if s.kind is StepKind.TOOL_CALL)
    assert step.outcome is StepOutcome.REJECTED
    assert step.reason_code == "invalid_tool_input"
    assert "wrong_field" in {part.role: part.blob.text for part in step.parts}[
        PartRole.TOOL_ARGUMENTS
    ]


@pytest.mark.parametrize("result", [None, {"observations": []}])
def test_host_tool_rejection_keeps_schema_violations_and_turn_linkage(store, result):
    error = {
        "code": "invalid_tool_input",
        "message": "Tool input does not satisfy its JSON Schema.",
        "details": {
            "violations": [{
                "path": ["pages"],
                "validator": "uniqueItems",
                "message": "[2, 2] has non-unique elements",
            }],
        },
    }
    recorder_module.record_tool_call(
        tool_id="analyze_pdf_page",
        arguments={
            "path": "report.pdf", "pages": [2, 2], "purpose": "general",
            "detail": "standard", "region": "page",
        },
        status="rejected",
        result=result,
        error=error,
        error_code=error["code"],
        duration_ms=0,
        session_id="session-batch",
        turn_id="turn-batch",
        step_id="l1tool-rejected-1",
        store=store,
    )

    step, = _all_steps(store)
    assert step.kind is StepKind.TOOL_CALL
    assert step.outcome is StepOutcome.REJECTED
    assert step.reason_code == error["code"]
    assert (step.session_id, step.turn_id, step.step_id) == (
        "session-batch", "turn-batch", "l1tool-rejected-1",
    )
    results = [part for part in step.parts if part.role is PartRole.TOOL_RESULT]
    assert len(results) == 1
    assert json.loads(results[0].blob.text) == {"result": result, "error": error}


def test_replaying_a_host_tool_rejection_keeps_one_immutable_step(store):
    error = {
        "code": "invalid_tool_input",
        "message": "Tool input does not satisfy its JSON Schema.",
        "details": {
            "violations": [{
                "path": ["pages"],
                "validator": "minItems",
                "message": "[] should be non-empty",
            }],
        },
    }
    arguments = {
        "path": "report.pdf", "pages": [], "purpose": "general",
        "detail": "standard", "region": "page",
    }
    for replay_index in range(2):
        recorder_module.record_tool_call(
            tool_id="analyze_pdf_page",
            arguments=arguments,
            status="rejected",
            result=None,
            error=error,
            error_code=error["code"],
            duration_ms=0,
            session_id="session-batch",
            turn_id="turn-batch",
            step_id="l1tool-rejected-replay",
            store=store,
        )
        if replay_index == 0:
            first = _all_steps(store)[0]

    step, = _all_steps(store)
    assert step == first
    assert step.step_id == "l1tool-rejected-replay"
    assert json.loads(next(
        part.blob.text for part in step.parts if part.role is PartRole.TOOL_RESULT
    )) == {"result": None, "error": error}


def test_a_handler_that_raised_is_still_recorded(store):
    def _boom(payload):
        raise RuntimeError("nope")

    _run(_tool_registration("probe_read", _boom), {"path": "/tmp/x"})
    step = next(s for s in _all_steps(store) if s.kind is StepKind.TOOL_CALL)
    assert step.outcome is StepOutcome.FAILED
    assert step.reason_code


def test_generic_retrieval_tool_step_does_not_duplicate_chunk_bodies(store):
    recorder_module.record_tool_call(
        tool_id="retrieve_files",
        arguments={"queries": ["revenue"], "file_ids": ["file_1"]},
        status="succeeded",
        result={
            "status": "complete",
            "evidence": [
                {
                    "file_id": "file_1",
                    "chunk_id": "chunk_1",
                    "content_sha256": "a" * 64,
                    "snippet": "document body must not be duplicated",
                    "locator": "/private/report.pdf",
                }
            ],
        },
        error_code=None,
        duration_ms=1,
        store=store,
    )

    step = next(s for s in _all_steps(store) if s.kind is StepKind.TOOL_CALL)
    result_part = next(
        part for part in step.parts if part.role is PartRole.TOOL_RESULT
    )
    projected = json.loads(result_part.blob.text)
    assert projected["evidence"] == [
        {
            "file_id": "file_1",
            "chunk_id": "chunk_1",
            "content_sha256": "a" * 64,
        }
    ]


def test_visual_tool_step_keeps_the_selected_image_and_observation_identities(store):
    """图片使用事实属于 Vision tool 结果，而不是模型消息的伪外部 Blob。"""

    recorder_module.record_tool_call(
        tool_id="read_file_visuals",
        arguments={
            "requests": [
                {
                    "file_id": "file_1",
                    "file_version_id": "file_version_1",
                    "visual_unit_id": "visual_unit_1",
                    "purpose": "chart",
                    "detail": "standard",
                    "region": "detected",
                }
            ]
        },
        status="succeeded",
        result={
            "results": [
                {
                    "file_id": "file_1",
                    "file_version_id": "file_version_1",
                    "visual_unit_id": "visual_unit_1",
                    "picture_id": "picture_1",
                    "picture_unit_id": "picture_unit_1",
                    "observation_id": "observation_1",
                    "status": "completed",
                    "observation": "柱状图显示收入增长。",
                }
            ]
        },
        error_code=None,
        duration_ms=12,
        store=store,
    )

    step = next(s for s in _all_steps(store) if s.kind is StepKind.TOOL_CALL)
    kept = {part.role: json.loads(part.blob.text) for part in step.parts}
    selected = kept[PartRole.TOOL_ARGUMENTS]["requests"][0]
    observed = kept[PartRole.TOOL_RESULT]["results"][0]
    assert selected == {
        "detail": "standard",
        "file_id": "file_1",
        "file_version_id": "file_version_1",
        "purpose": "chart",
        "region": "detected",
        "visual_unit_id": "visual_unit_1",
    }
    assert {
        key: observed[key]
        for key in (
            "file_id",
            "file_version_id",
            "visual_unit_id",
            "picture_id",
            "picture_unit_id",
            "observation_id",
        )
    } == {
        "file_id": "file_1",
        "file_version_id": "file_version_1",
        "visual_unit_id": "visual_unit_1",
        "picture_id": "picture_1",
        "picture_unit_id": "picture_unit_1",
        "observation_id": "observation_1",
    }
    assert step.purpose == "read_file_visuals"


# --- 被拒绝的输出 ----------------------------------------------------------------


def test_a_refusal_keeps_the_thing_that_was_refused(store):
    """运行时只记录守卫拒绝，却从不记录它拒绝了什么。"""

    recorder_module.record_rejected_output(
        stage="query_guard:task_graph_create",
        rejected='{"queries": ["q1", "q2", "q3"], "source_hints": ["document"]}',
        reason_code="too_many_queries",
    )
    step = next(s for s in _all_steps(store) if s.outcome is StepOutcome.REJECTED)
    assert step.reason_code == "too_many_queries"
    assert "q3" in step.parts[0].blob.text
    assert step.parts[0].role is PartRole.REJECTED_OUTPUT


# --- 检索 -----------------------------------------------------------------------


def test_the_queries_a_generator_actually_produced_are_kept(store):
    """生成器提示经过长期调优，却一直无法查看其输出。"""

    recorder_module.record_retrieval(
        queries=("季度报告 Figure 2 数值", "quarterly report figure 2 values"),
        source_scope="task_graph_create",
        evidence=[{"source_unit_id": "u1", "fused_rank": 1}],
        outcome="ok",
        reason_code=None,
        duration_ms=42,
    )
    step = next(s for s in _all_steps(store) if s.kind is StepKind.RETRIEVAL)
    queries = [p.blob.text for p in step.parts if p.role is PartRole.QUERY]
    assert queries == ["季度报告 Figure 2 数值", "quarterly report figure 2 values"]
    assert step.metrics == {"queries": 2, "items": 1}


def test_retrieval_audit_keeps_lane_fusion_reranker_and_refs_without_bodies(store):
    """评测可证明实际检索链，而不把分块正文复制进 trajectory。"""

    recorder_module.record_retrieval(
        queries=("quarterly revenue",),
        source_scope="MODEL_RETRIEVE_FILES_V1",
        evidence=[
            {
                "source_type": "document",
                "source_unit_id": "unit-1",
                "source_revision": "revision-1",
                "indexed_content_hash": "a" * 64,
                "producer_chunk_id": "chunk-1",
                "packed_rank": 1,
                "fused_rank": 2,
                "reranked_rank": 1,
            }
        ],
        method_outcomes=[
            {
                "method": "dense",
                "source_type": "document",
                "query_index": 0,
                "status": "used",
                "candidate_count": 20,
                "reason_code": None,
                "attempt": 1,
                "infrastructure_attempts": 1,
            },
            {
                "method": "learned_sparse",
                "source_type": "document",
                "query_index": 0,
                "status": "used",
                "candidate_count": 20,
                "reason_code": None,
                "attempt": 1,
                "infrastructure_attempts": 1,
            },
        ],
        reranker_outcomes=[
            {
                "source_type": "document",
                "status": "used",
                "candidate_count": 30,
                "scored_candidate_count": 30,
                "reason_code": None,
                "fingerprint": "bge-reranker-fingerprint",
            }
        ],
        fusion={
            "algorithm": "rrf",
            "context_count": 1,
            "contributor_ranks_recorded": False,
        },
        outcome="complete",
        reason_code=None,
        duration_ms=18,
        session_id="session-1",
        turn_id="turn-1",
    )

    step = next(s for s in _all_steps(store) if s.kind is StepKind.RETRIEVAL)
    assert step.outcome is StepOutcome.OK
    assert step.reason_code is None
    assert step.session_id == "session-1"
    assert step.turn_id == "turn-1"
    assert step.metrics == {
        "queries": 1,
        "items": 1,
        "method_runs": 2,
        "reranker_runs": 1,
    }
    by_role = {}
    for part in step.parts:
        by_role.setdefault(part.role, []).append(part.blob.text)
    assert len(by_role[PartRole.RETRIEVAL_METHOD]) == 2
    assert json.loads(by_role[PartRole.FUSION][0])["algorithm"] == "rrf"
    assert json.loads(by_role[PartRole.RERANKER][0])["status"] == "used"
    evidence = json.loads(by_role[PartRole.EVIDENCE][0])
    assert evidence["producer_chunk_id"] == "chunk-1"
    encoded = "\n".join(
        text for values in by_role.values() for text in values if text is not None
    )
    assert "document body must never be stored" not in encoded


def test_a_retrieval_that_found_nothing_is_still_a_recorded_step(store):
    """若没有该记录，零命中与“从未运行”看起来完全相同。"""

    recorder_module.record_retrieval(
        queries=("吞吐量",),
        source_scope="l0_probe",
        evidence=[],
        outcome="no_match",
        reason_code=None,
        duration_ms=7,
    )
    step = next(s for s in _all_steps(store) if s.kind is StepKind.RETRIEVAL)
    assert step.metrics["items"] == 0
    assert step.reason_code == "no_match"


def test_retrieval_failures_are_kept_as_body_free_stage_diagnostics(store):
    recorder_module.record_retrieval(
        queries=("quarterly revenue",),
        source_scope="MODEL_RETRIEVE_FILES_V1",
        evidence=[],
        diagnostics=[
            {
                "stage": "method",
                "code": "dense_query_encoding_failed",
                "status": "failed",
                "method": "dense",
                "known_count": 1,
            },
            {
                "stage": "tool_projection",
                "code": "evidence_identity_mismatch",
                "status": "dropped",
                "indexed_content_hash": "a" * 64,
                "known_count": 1,
            },
        ],
        outcome="partial",
        reason_code="evidence_identity_mismatch",
        duration_ms=9,
    )

    step = next(s for s in _all_steps(store) if s.kind is StepKind.RETRIEVAL)
    diagnostics = [
        json.loads(part.blob.text)
        for part in step.parts
        if part.role is PartRole.RETRIEVAL_DIAGNOSTIC
    ]
    assert [item["stage"] for item in diagnostics] == [
        "method",
        "tool_projection",
    ]
    assert step.metrics["diagnostics"] == 2
    assert "document body" not in "\n".join(
        part.blob.text or "" for part in step.parts
    )


# --- 这些也不能拖垮调用 -----------------------------------------------------------


def test_a_broken_recorder_does_not_break_a_tool_call(store, monkeypatch):
    monkeypatch.setattr(
        recorder_module, "recording_enabled", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    registration = _tool_registration("probe_read", lambda payload: {"content": "hi"})
    with pytest.raises(RuntimeError):
        recorder_module.recording_enabled()
    # 工具调用自己应当照常成功
    monkeypatch.setattr(recorder_module, "_tool_outcome", lambda status: 1 / 0)
    monkeypatch.setattr(recorder_module, "recording_enabled", lambda: True)
    assert _run(registration, {"path": "/tmp/x"}).status.value == "succeeded"


def _all_steps(store: TrajectoryStore):
    import sqlite3

    connection = sqlite3.connect(store.path)
    connection.row_factory = sqlite3.Row
    ids = [str(r["step_id"]) for r in connection.execute("SELECT step_id FROM trajectory_steps")]
    connection.close()
    return [store.get(step_id) for step_id in ids]
