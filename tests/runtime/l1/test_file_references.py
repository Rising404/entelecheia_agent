"""文件短引用跨步骤稳定；模型参数还原后仍走原生工具合同。"""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from personagraph.output_protocol.l1 import L1AttemptDecisionProposal, L1PlanProposal, materialize_l1_plan
from personagraph.runtime.l1.controller import _admit_decision, L1ControllerFailure
from personagraph.runtime.l1.model_view.file_references import (
    FileReferenceAdmissionError, FileReferences, build_file_reference_catalog, project_file_tool_schema,
)
from personagraph.runtime.l1.model_view.projection import project_attempt_view
from personagraph.tools.composition.default_catalog import build_production_default_catalog_seeds


def _file(identity="file-a", version="version-a", path="a.pdf"):
    return {"file_id": identity, "file_version_id": version, "relative_path": path, "name": path}


def _execution(rows, *, tool_id="prepare_files"):
    raw = json.dumps({"status": "succeeded", "result": {"results": rows}})
    return {
        "attempts": [{"attempt_id": "step-1", "ordinal": 1}],
        "tool_calls": [{"tool_call_id": "call-1", "attempt_id": "step-1", "call_ordinal": 1,
                        "tool_id": tool_id, "status": "succeeded", "outcome_json": raw,
                        "outcome_hash": hashlib.sha256(raw.encode()).hexdigest()}],
    }


def _files():
    return FileReferences(build_file_reference_catalog([_file()], {"tool_calls": []}))


def _catalog():
    return {seed.definition.spec.tool_id: seed.definition.spec.to_dict()
            for seed in build_production_default_catalog_seeds()}


def test_files_rebuild_from_frozen_sources_and_ledger_without_reordering_or_mutating():
    attachments = [_file("b", "b1", "b.pdf"), _file()]
    execution = _execution([_file(), _file("workspace", "v1", "c.pdf"), _file("file-a", "version-new")])
    before = deepcopy((attachments, execution))
    catalog = build_file_reference_catalog(attachments, execution)
    assert [(row["file_ref"], row["file_id"], row["file_version_id"]) for row in catalog] == [
        ("f1", "file-a", "version-a"), ("f2", "b", "b1"),
        ("f3", "workspace", "v1"), ("f4", "file-a", "version-new"),
    ]
    assert build_file_reference_catalog(list(reversed(attachments)), execution) == catalog
    assert (attachments, execution) == before
    assert _files().materialize_arguments("prepare_files", {"files": [{"file_ref": "f1"}]}, path="") == {
        "files": [{"file_id": "file-a", "file_version_id": "version-a"}],
    }


def test_file_reference_source_cannot_use_corrupt_or_pending_tool_outcome():
    execution = _execution([_file()])
    execution["tool_calls"][0]["outcome_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_file_reference_catalog([], execution)
    execution["tool_calls"][0]["status"] = "pending"
    assert build_file_reference_catalog([], execution) == []


@pytest.mark.parametrize(("tool_id", "arguments", "expected"), [
    ("prepare_files", {"files": [{"file_ref": "f1"}]}, {"files": [{"file_id": "file-a", "file_version_id": "version-a"}]}),
    ("check_files_state", {"files": [{"path": "work/new.pdf"}]}, {"files": [{"path": "work/new.pdf"}]}),
    ("retrieve_files", {"queries": ["term"], "file_refs": ["f1"]}, {"queries": ["term"], "file_ids": ["file-a"]}),
    ("read_file_chunks", {"targets": [{"file_ref": "f1", "document_version_id": "doc", "chunk_sequences": [0]}]},
     {"targets": [{"file_id": "file-a", "document_version_id": "doc", "chunk_sequences": [0]}]}),
    ("read_file_chunks", {"targets": [{"chunk_ids": ["chunk-a"]}]}, {"targets": [{"chunk_ids": ["chunk-a"]}]}),
    ("inspect_file_chunks", {"targets": [{"file_ref": "f1", "document_version_id": "doc"}]},
     {"targets": [{"file_id": "file-a", "document_version_id": "doc"}]}),
    ("search_file_text", {"targets": [{"file_ref": "f1", "document_version_id": "doc"}], "text": "file_id"},
     {"targets": [{"file_id": "file-a", "document_version_id": "doc"}], "text": "file_id"}),
    ("list_file_visuals", {"file_ref": "f1"}, {"file_id": "file-a", "file_version_id": "version-a"}),
])
def test_model_selectors_materialize_to_existing_tool_contract(tool_id, arguments, expected):
    canonical = _catalog()[tool_id]
    before = deepcopy((canonical, arguments))
    model = project_file_tool_schema(canonical)
    Draft202012Validator.check_schema(model["input_schema"])
    assert not list(Draft202012Validator(model["input_schema"]).iter_errors(arguments))
    bound = _files().materialize_arguments(tool_id, arguments, path="/action/calls/0/arguments")
    assert bound == expected
    assert not list(Draft202012Validator(canonical["input_schema"]).iter_errors(bound))
    assert _files().project_arguments(tool_id, bound) == arguments
    assert (canonical, arguments) == before


@pytest.mark.parametrize("arguments", [
    {"files": [{"file_ref": "f99"}]}, {"files": [{"file_ref": "f01"}]},
    {"files": [{"file_ref": "c1.1"}]}, {"files": [{"file_ref": []}]},
    {"files": [{"file_id": "file-a"}]},
    {"files": [{"file_ref": "f1", "file_version_id": "forged-version"}]},
])
def test_unknown_or_raw_file_identity_is_rejected_before_business_dispatch(arguments):
    with pytest.raises(FileReferenceAdmissionError) as rejected:
        _files().materialize_arguments("prepare_files", arguments, path="/action/calls/0/arguments")
    assert rejected.value.available == ("f1",)
    assert rejected.value.path.startswith("/action/calls/0/arguments")


def test_file_ref_always_resolves_inside_current_run_and_version():
    other = FileReferences(build_file_reference_catalog([_file("other", "other-version")], {"tool_calls": []}))
    args = {"files": [{"file_ref": "f1"}]}
    assert other.materialize_arguments("prepare_files", args, path="")["files"][0] == {
        "file_id": "other", "file_version_id": "other-version",
    }
    assert _files().materialize_arguments("prepare_files", args, path="")["files"][0]["file_id"] == "file-a"


def test_projection_uses_same_ref_in_catalog_results_and_progress_without_touching_body():
    row = {**_file(), "status": "ready", "content": {"file_id": "user-authored-field", "text": "file-a"}}
    catalog = build_file_reference_catalog([_file()], _execution([row]))
    payload = {
        "file_references": catalog, "file_catalog": [_file()],
        "prior_tool_results": [{"tool_id": "prepare_files", "result": {"results": [row]},
                                "arguments": {"files": [{"file_id": "file-a", "file_version_id": "version-a"}]}}],
        "reading_progress": {"results": [{"tool_id": "prepare_files", "arguments": {
            "files": [{"file_id": "file-a", "file_version_id": "version-a"}]}}]},
    }
    before = deepcopy(payload)
    projected = project_attempt_view(payload)
    assert projected["file_catalog"] == [{"file_ref": "f1", "name": "a.pdf", "relative_path": "a.pdf"}]
    result = projected["prior_tool_results"][0]
    assert result["arguments"] == {"files": [{"file_ref": "f1"}]}
    assert result["result"]["results"][0]["file_ref"] == "f1"
    assert result["result"]["results"][0]["content"] == row["content"]
    assert "file_id" not in result["result"]["results"][0]
    assert projected["reading_progress"]["results"][0]["arguments"] == result["arguments"]
    assert payload == before


def test_host_admission_binds_file_selector_and_reports_unknown_ref_exactly():
    catalog = _catalog()
    runtime = SimpleNamespace(
        attachment_file_catalog=(_file(),), model_catalog=lambda: tuple(catalog.values()),
        knows_tool=lambda tool_id: tool_id in catalog, is_execution_findings_tool=lambda _: False,
    )
    plan = materialize_l1_plan(L1PlanProposal(objective="Read PDF", acceptances=[{"criterion": "Use evidence"}]),
                               input_message_id="message", user_text="Read PDF")

    def admit(ref):
        proposal = L1AttemptDecisionProposal.model_validate({"note": "Prepare the source", "action": {
            "kind": "call_tools", "calls": [{"tool_id": "prepare_files", "arguments": {"files": [{"file_ref": ref}]}}],
        }})
        return _admit_decision(proposal, current_plan=plan, accepted=SimpleNamespace(),
                               l1_turn_run_id="run", tool_runtime=runtime, execution={"tool_calls": []},
                               finalization_required=False, max_tool_calls_per_attempt=8)

    decision, _ = admit("f1")
    assert decision.action.calls[0].model_dump(mode="json")["arguments"] == {
        "files": [{"file_id": "file-a", "file_version_id": "version-a"}],
    }
    with pytest.raises(L1ControllerFailure) as rejected:
        admit("f99")
    assert rejected.value.repair_issue.code == "host_guard.l1_unavailable_file_reference"
    assert rejected.value.repair_issue.paths == ("/action/calls/0/arguments/files/0/file_ref",)


def test_history_projects_file_refs_before_pagination_and_keeps_original_bytes(monkeypatch):
    from personagraph.persistent_turn_content.tool_results import ToolResultRecord
    from personagraph.runtime.l1.tool_context import history_binding

    execution = _execution([_file(), _file("later-file", "later-version", "later.pdf")])
    call = execution["tool_calls"][0]
    frozen = call["outcome_json"]
    port = SimpleNamespace(read_result=lambda **_: ToolResultRecord.model_validate({
        "source": {"tool_call_id": "call-1", "tool_id": "prepare_files", "status": "succeeded",
                   "call_ref": "c1.1", "result_sha256": call["outcome_hash"]},
        "outcome": json.loads(frozen),
    }))
    monkeypatch.setattr(history_binding.session_store, "build_l1_tool_history_reader", lambda **_: port)
    monkeypatch.setattr(history_binding.session_store, "get_l1_turn_execution", lambda **_: deepcopy(execution))
    runtime = history_binding.bind_l1_tool_history(session_id="session", turn_id="turn", attachment_catalog=[_file()])
    page = runtime.read_result({"call_ref": "c1.1", "path": "/result/results/0"})
    assert page["value"]["file_ref"] == "f1"
    assert not {"file_id", "file_version_id"} & page["value"].keys()
    assert runtime.read_result({"call_ref": "c1.1", "path": "/result/results/1"})["value"]["file_ref"] == "f2"
    assert execution["tool_calls"][0]["outcome_json"] == frozen


@pytest.mark.parametrize("protocol", ["l1-attempt-model-view-v6", "l1-attempt-model-view-v7"])
def test_old_frozen_model_request_cannot_run_under_current_reference_interface(protocol):
    from personagraph.model_io.gateway import ModelGatewayError
    from personagraph.runtime.l1.model import request_l1_attempt_decision

    with pytest.raises(ModelGatewayError, match="Unsupported frozen L1 protocol"):
        request_l1_attempt_decision(
            turn_id="turn", session_id="session", logical_model_call_id="call",
            payload={"schema_version": protocol}, emit=None, deadline=None,
            l1_turn_run_id="run", attempt_id="attempt", state_guard_hash="state", store=None, model_binding=None,
        )
