"""当前 L1 的 output 创建、受保护分派及 File 精读链路。"""

import json

from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session import store as session_store
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.storage.context import require_current
from tests.runtime.test_l1_tools_plan_revision_verification import (
    _install_l1_classifier, _model_result, _run_l1,
)


def test_l1_creates_output_without_workspace_update_and_can_prepare_and_read_it(
    tmp_path, monkeypatch, bound_partitioned_session,
):
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    arguments = {"path": "reports/summary.md", "content": "# Result\n\nOutput accuracy is 84.25 percent."}
    runtime = build_l1_tool_runtime(session_id)
    prepared = runtime.prepare(
        tool_id="create_output_file", arguments=arguments, remaining_tool_calls=1,
    )
    assert prepared.rejected_outcome is None
    assert prepared.requires_protected_dispatch
    assert prepared.protected_authority.revalidate()
    assert runtime.execute_prepared(prepared, deadline_monotonic=10_000_000).outcome.error.code == (
        "protected_tool_dispatch_required"
    )
    assert not (workspace / "output").exists()
    write = runtime.prepare(
        tool_id="write_workspace_file", arguments={"path": "ordinary.txt", "content": "blocked"},
        remaining_tool_calls=1,
    )
    assert write.rejected_outcome.error.code == "tool_policy_authorization_required"
    _install_l1_classifier(monkeypatch)
    created = {}
    calls = []

    def decide(_system, user_content, **kwargs):
        payload = json.loads(user_content)
        calls.append(payload)
        plan = None
        if len(calls) == 1:
            assert "create_output_file" in {item["tool_id"] for item in payload["tool_catalog"]}
            plan = {
                "objective": "创建并精读产物", "acceptances": [{
                    "criterion": "产物已创建并精读确认",
                }],
            }
            action = {"kind": "call_tools", "calls": [{
                "tool_id": "create_output_file", "arguments": arguments,
            }]}
        elif len(calls) == 2:
            result = payload["prior_tool_results"][-1]
            assert result["status"] == "succeeded"
            created.update(result["result"])
            assert created["relative_path"] == "output/reports/summary.md"
            assert created["file_id"] and created["file_version_id"]
            assert "document_ref" not in created and "file_ref" not in created
            action = {"kind": "call_tools", "calls": [{
                "tool_id": "prepare_files", "arguments": {"files": [{
                    "file_id": created["file_id"],
                    "file_version_id": created["file_version_id"],
                }]},
            }]}
        elif len(calls) == 3:
            result = payload["prior_tool_results"][-1]
            assert result["status"] == "succeeded"
            item = result["result"]["results"][0]
            assert item["status"] == "ready"
            assert item["file_id"] == created["file_id"]
            action = {"kind": "call_tools", "calls": [{
                "tool_id": "read_file_chunks", "arguments": {"targets": [{
                    "file_id": item["file_id"], "document_version_id": item["document_version_id"],
                    "chunk_sequences": [0],
                }]},
            }]}
        else:
            exact = payload["prior_tool_results"][-1]
            assert exact["status"] == "succeeded"
            assert "84.25 percent" in exact["result"]["results"][0]["chunks"][0]["content"]
            return _model_result({
                "plan": None,  "action": {"kind": "submit_final_reply", "reply": "产物已创建并确认。"},
            }, kwargs["model_call_id"])
        return _model_result({
            "plan": plan,  "action": action,
        }, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    result = _run_l1(
        session_id=session_id, user_text="请在 output 目录创建并读取产物。", request_id="output-create-read",
        features={
            "file_retrieval_write_enabled": True, "file_retrieval_read_enabled": True,
            "l1_retrieval_tools_enabled": True, "l1_semantic_verification_mode": "off",
        },
    )
    assert result.status == "completed", (result, len(calls), calls[-1].get("prior_tool_results"))
    assert (workspace / created["relative_path"]).read_text() == arguments["content"]
    execution = session_store.get_l1_turn_execution(session_id=session_id, turn_id=result.turn_id)
    output_call = next(call for call in execution["tool_calls"] if call["tool_id"] == "create_output_file")
    canonical_output = json.loads(output_call["outcome_json"])["result"]
    record = WorkspaceFileAuthority(require_current()).get_file(canonical_output["file_id"])
    assert record.source is FileSource.AGENT_OUTPUT
    prepare_call = next(call for call in execution["tool_calls"] if call["tool_id"] == "prepare_files")
    assert json.loads(prepare_call["arguments_json"])["files"] == [{
        "file_id": canonical_output["file_id"], "file_version_id": canonical_output["file_version_id"],
    }]
    read_call = next(call for call in execution["tool_calls"] if call["tool_id"] == "read_file_chunks")
    read_target = json.loads(read_call["arguments_json"])["targets"][0]
    assert read_target["file_id"] == canonical_output["file_id"]
    assert read_target["document_version_id"]
    assert "document_ref" not in read_target
    assert output_call["tool_id"] == "create_output_file"
    assert output_call["execution_class"] == "protected_effect"
    assert output_call["protected_phase"] == "succeeded"
