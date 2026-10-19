"""L1 原生工具身份与模型短引用：稳定恢复、精确作用域、历史回读和准入。"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from personagraph.output_protocol.l1 import L1AttemptDecisionProposal, L1PlanProposal, materialize_l1_plan
from personagraph.persistent_turn_content.tool_results import ToolResultRecord, ToolResultHistoryPage
from personagraph.runtime.l1.controller import L1ControllerFailure, _admit_decision
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.model_view.file_references import FileReferences, build_file_reference_catalog
from personagraph.runtime.l1.model_view.projection import project_attempt_view
from personagraph.runtime.l1.model_view.reference_bindings import ToolReferenceAdmissionError
from personagraph.runtime.l1.model_view.tool_references import ToolReferences, build_tool_reference_catalog
from personagraph.tools.composition.default_catalog import build_production_default_catalog_seeds

ENTRY = "efentry_" + "a" * 64
UNIT = "ntu_" + "b" * 24
VISUAL_CURSOR = "visualcursor_20_" + "c" * 64
DIRECTORY_CURSOR = "workspacedircursor_" + "d" * 40
FILE = {"file_id": "file-a", "file_version_id": "version-a", "name": "a.pdf", "relative_path": "a.pdf"}


def _append(execution, tool_id, arguments, result):
    ordinal = len(execution["attempts"]) + 1
    attempt_id = f"step-{ordinal}"
    outcome = {"status": "succeeded", "result": result}
    execution["attempts"].append({"attempt_id": attempt_id, "ordinal": ordinal})
    call = {"tool_call_id": f"call-{ordinal}", "attempt_id": attempt_id, "call_ordinal": 1,
            "tool_id": tool_id, "status": "succeeded",
            "arguments_json": canonical_json(arguments), "arguments_hash": sha256_json(arguments),
            "outcome_json": canonical_json(outcome), "outcome_hash": sha256_json(outcome)}
    execution["tool_calls"].append(call)
    return call


def _execution():
    execution = {"attempts": [], "tool_calls": []}
    _append(execution, "list_file_visuals", {**FILE, "limit": 20}, {
        **FILE, "visuals": [{"visual_unit_id": UNIT, "pages": [2], "kind": "figure"}],
        "next_cursor": VISUAL_CURSOR,
    })
    _append(execution, "list_workspace_directory", {"subpath": "docs"}, {
        "scope": {"path": "docs"}, "items": [], "next_cursor": DIRECTORY_CURSOR,
    })
    return execution


def _catalog(execution=None):
    return build_tool_reference_catalog(execution or _execution(), entry_revisions=[
        SimpleNamespace(entry_id=ENTRY, sequence=1),
    ])


def _native_catalog():
    return [seed.definition.spec.to_dict() for seed in build_production_default_catalog_seeds()]


def _model_view(execution, **payload):
    return project_attempt_view({
        "file_references": build_file_reference_catalog([FILE], execution),
        "tool_references": _catalog(execution), "tool_catalog": _native_catalog(), **payload,
    })


def test_full_history_rebuild_is_stable_across_eviction_revision_retraction_and_recovery():
    execution = _execution()
    catalog = _catalog(execution)
    before = deepcopy(execution)
    # 新文件版本上的同名视觉单元不能复用旧版本的引用。
    _append(execution, "list_file_visuals", {}, {
        **FILE, "file_version_id": "version-b", "visuals": [{"visual_unit_id": UNIT}], "next_cursor": None,
    })
    # 回执重放/重新列出同一资源也不应重编号。
    _append(execution, "list_file_visuals", {}, json.loads(before["tool_calls"][0]["outcome_json"])["result"])
    revisions = [SimpleNamespace(entry_id=ENTRY, sequence=1),
                 SimpleNamespace(entry_id="second-entry", sequence=2),
                 SimpleNamespace(entry_id=ENTRY, sequence=3)]
    rebuilt = build_tool_reference_catalog(execution, entry_revisions=revisions)
    assert rebuilt["entries"][:1] == catalog["entries"]
    assert rebuilt["visuals"][:1] == catalog["visuals"]
    assert len(rebuilt["visuals"]) == 2
    assert rebuilt["cursors"] == catalog["cursors"]
    assert build_tool_reference_catalog(json.loads(canonical_json(execution)), entry_revisions=revisions) == rebuilt
    # 只显示一个笔记和最近结果时，映射仍来自全账本而非可见顺序。
    view = project_attempt_view({"tool_references": rebuilt, "execution_findings": {
        "notes": [{"entry_id": "second-entry", "summary": "Visible"}],
    }})
    assert view["execution_findings"]["notes"][0]["entry_ref"] == "e2"
    assert execution["tool_calls"][:2] == before["tool_calls"]


@pytest.mark.parametrize("tool_id,model,native", [
    ("revise_execution_finding", {"items": [{"operation": "retract", "entry_ref": "e1", "reason": "Incorrect"}]},
     {"items": [{"operation": "retract", "entry_id": ENTRY, "reason": "Incorrect"}]}),
    ("read_file_visuals", {"requests": [{"file_ref": "f1", "visual_ref": "v1", "purpose": "general", "detail": "high", "region": "page"}]},
     {"requests": [{"file_id": "file-a", "file_version_id": "version-a", "visual_unit_id": UNIT,
                    "purpose": "general", "detail": "high", "region": "page"}]}),
    ("list_file_visuals", {"file_ref": "f1", "cursor_ref": "p1"},
     {"file_id": "file-a", "file_version_id": "version-a", "cursor": VISUAL_CURSOR}),
    ("list_workspace_directory", {"subpath": "docs", "cursor_ref": "p2"},
     {"subpath": "docs", "cursor": DIRECTORY_CURSOR}),
])
def test_model_contract_round_trip_preserves_native_contract_and_other_arguments(tool_id, model, native):
    execution = _execution()
    refs = ToolReferences(_catalog(execution))
    files = FileReferences(build_file_reference_catalog([FILE], execution))
    schemas = {entry["tool_id"]: entry["input_schema"] for entry in _model_view(execution)["tool_catalog"]}
    Draft202012Validator(schemas[tool_id]).validate(model)
    Draft202012Validator(schemas[tool_id]).check_schema(schemas[tool_id])
    restored = refs.materialize_arguments(tool_id, files.materialize_arguments(tool_id, model, path=""), path="")
    assert restored == native
    assert files.project_arguments(tool_id, refs.project_arguments(tool_id, restored)) == model
    original = next(entry for entry in _native_catalog() if entry["tool_id"] == tool_id)
    Draft202012Validator(original["input_schema"]).validate(restored)
    assert list(Draft202012Validator(original["input_schema"]).iter_errors(model))
    assert list(Draft202012Validator(schemas[tool_id]).iter_errors(native))


@pytest.mark.parametrize("tool_id,arguments,field", [
    ("revise_execution_finding", {"items": [{"entry_id": ENTRY}]}, "entry_id"),
    ("revise_execution_finding", {"items": [{"entry_ref": "v1"}]}, "entry_ref"),
    ("revise_execution_finding", {"items": [{"entry_ref": "e99"}]}, "entry_ref"),
    ("read_file_visuals", {"requests": [{**FILE, "visual_unit_id": UNIT}]}, "visual_unit_id"),
    ("read_file_visuals", {"requests": [{**FILE, "file_version_id": "other", "visual_ref": "v1"}]}, "visual_ref"),
    ("read_file_visuals", {"requests": [{**FILE, "file_id": "other", "visual_ref": "v1"}]}, "visual_ref"),
    ("list_file_visuals", {**FILE, "cursor_ref": "p2"}, "cursor_ref"),
    ("list_file_visuals", {**FILE, "file_version_id": "other", "cursor_ref": "p1"}, "cursor_ref"),
    ("list_workspace_directory", {"cursor_ref": "p1"}, "cursor_ref"),
    ("list_workspace_directory", {"cursor": DIRECTORY_CURSOR}, "cursor"),
    ("list_workspace_directory", {"cursor_ref": ["p2"]}, "cursor_ref"),
])
def test_unknown_raw_cross_type_and_cross_file_references_are_rejected(tool_id, arguments, field):
    refs = ToolReferences(_catalog())
    with pytest.raises(ToolReferenceAdmissionError) as rejected:
        refs.materialize_arguments(tool_id, arguments, path="/action/calls/0/arguments")
    assert rejected.value.path.endswith("/" + field)
    assert ENTRY not in str(rejected.value) and VISUAL_CURSOR not in str(rejected.value)


def test_fresh_execution_cannot_resolve_another_execution_references():
    refs = ToolReferences(build_tool_reference_catalog({"tool_calls": []}))
    with pytest.raises(ToolReferenceAdmissionError):
        refs.materialize_arguments("revise_execution_finding", {"items": [{"entry_ref": "e1"}]}, path="")
    with pytest.raises(ToolReferenceAdmissionError):
        refs.materialize_arguments("list_workspace_directory", {"cursor_ref": "p2"}, path="")


def test_corrupt_or_unfinished_results_cannot_allocate_references():
    execution = _execution()
    execution["tool_calls"][0]["outcome_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _catalog(execution)
    execution["tool_calls"][0]["status"] = "pending"
    catalog = _catalog(execution)
    assert catalog["visuals"] == []
    assert all(row["value"] != VISUAL_CURSOR for row in catalog["cursors"])


def test_projection_covers_results_arguments_progress_and_metadata_without_touching_body():
    execution = _execution()
    raw_result = json.loads(execution["tool_calls"][0]["outcome_json"])["result"]
    literal = {"entry_id": ENTRY, "visual_unit_id": UNIT, "cursor": VISUAL_CURSOR}
    raw_result["visuals"][0]["caption"] = literal
    payload = {
        "prior_tool_results": [{"call_ref": "c1.1", "tool_id": "list_file_visuals", "result": raw_result,
                                "arguments": {**FILE, "cursor": VISUAL_CURSOR},
                                "metadata": {"next_cursor": VISUAL_CURSOR}},
                               {"call_ref": "c3.1", "tool_id": "revise_execution_finding",
                                "arguments": {"items": [{"entry_id": ENTRY, "claim": literal}]},
                                "result": {"affected_entry_ids": [ENTRY]}}],
        "execution_findings": {"notes": [{"entry_id": ENTRY, "summary": "a note"}]},
        "reading_progress": {"results": [{"tool_id": "list_file_visuals", "arguments": FILE,
                                          "next_cursor": VISUAL_CURSOR}]},
    }
    before = deepcopy(payload)
    view = _model_view(execution, **payload)
    first, finding = view["prior_tool_results"]
    assert first["result"]["visuals"][0] == {"visual_ref": "v1", "pages": [2], "kind": "figure", "caption": literal}
    assert first["result"]["next_cursor_ref"] == "p1"
    assert first["arguments"]["cursor_ref"] == "p1"
    assert first["metadata"] == {"next_cursor_ref": "p1"}
    assert finding["result"]["affected_entry_refs"] == ["e1"]
    assert finding["arguments"]["items"][0] == {"entry_ref": "e1", "claim": literal}
    assert view["reading_progress"]["results"][0]["next_cursor_ref"] == "p1"
    assert view["execution_findings"]["notes"] == [{"entry_ref": "e1", "summary": "a note"}]
    assert "tool_references" not in view
    assert payload == before


def test_history_projects_refs_before_json_pointer_selection_and_pagination(monkeypatch):
    from personagraph.runtime.l1.tool_context import history_binding

    execution = _execution()
    finding = _append(execution, "record_execution_findings", {}, {"affected_entry_ids": [ENTRY]})
    _append(execution, "list_workspace_directory", {"subpath": "docs", "cursor": DIRECTORY_CURSOR}, {"next_cursor": None})
    _append(execution, "read_file_visuals", {"requests": [{**FILE, "visual_unit_id": UNIT}]}, {"results": []})
    before = deepcopy(execution)

    def read_result(call_ref):
        call = execution["tool_calls"][int(call_ref[1:].split(".")[0]) - 1]
        return ToolResultRecord.model_validate({
            "source": {"tool_call_id": call["tool_call_id"], "tool_id": call["tool_id"], "status": "succeeded",
                       "call_ref": call_ref, "result_sha256": call["outcome_hash"]},
            "outcome": json.loads(call["outcome_json"]),
        })

    def list_results(**_):
        return ToolResultHistoryPage(results=[{
            "source": read_result(f"c{i}.1").source, "attempt_ordinal": i, "call_ordinal": 1,
            "arguments_summary": call["arguments_json"][:12], "arguments_sha256": call["arguments_hash"],
            "arguments_truncated": True, "total_characters": len(call["outcome_json"]),
        } for i, call in enumerate(execution["tool_calls"], start=1)],
            offset=0, total_results=5, partial=False)

    monkeypatch.setattr(history_binding.session_store, "build_l1_tool_history_reader", lambda **_: SimpleNamespace(
        read_result=read_result, list_results=list_results,
    ))
    monkeypatch.setattr(history_binding.session_store, "get_l1_turn_execution", lambda **_: execution)
    monkeypatch.setattr(history_binding.session_store, "get_execution_findings_ledger_for_owner", lambda **_: SimpleNamespace(
        ledger=SimpleNamespace(entry_revisions=[SimpleNamespace(entry_id=ENTRY, sequence=1)]),
    ))
    runtime = history_binding.bind_l1_tool_history(session_id="session", turn_id="turn", attachment_catalog=[FILE])
    for call_ref, path, expected in [
        ("c1.1", "/result/visuals/0/visual_ref", "v1"),
        ("c1.1", "/result/next_cursor_ref", "p1"),
        ("c2.1", "/result/next_cursor_ref", "p2"),
        ("c3.1", "/result/affected_entry_refs/0", "e1"),
    ]:
        assert runtime.read_result({"call_ref": call_ref, "path": path})["value"] == expected
    page = runtime.list_results({})
    assert json.loads(page["results"][3]["arguments_summary"])["cursor_ref"] == "p2"
    assert json.loads(page["results"][4]["arguments_summary"])["requests"][0]["visual_ref"] == "v1"
    assert not page["results"][3]["arguments_truncated"]
    assert DIRECTORY_CURSOR not in canonical_json(page) and UNIT not in canonical_json(page)
    assert execution == before
    assert ENTRY in finding["outcome_json"]


def test_native_cursor_snapshot_checks_still_reject_stale_scope():
    from personagraph.runtime.l1.model_view.reference_bindings import append_binding
    from personagraph.tools.execution import ToolBusinessFailure
    from personagraph.tools.visual.file_visual_adapter import _cursor, _cursor_offset
    from personagraph.tools.workspace.workspace_directory_listing import _cursor_for_offset, _cursor_offset as directory_offset

    source = SimpleNamespace(snapshot_id="a" * 64, catalog=tuple(range(4)))
    cursor = _cursor(source, 2)
    directory = _cursor_for_offset(session_id="s", scope_path="docs", snapshot_sha256="b" * 64, offset=2)
    rows = {}
    append_binding(rows, prefix="p", value=cursor, scope=("list_file_visuals", "file-a", "version-a"))
    append_binding(rows, prefix="p", value=directory, scope=("list_workspace_directory",))
    refs = ToolReferences({"cursors": list(rows.values())})
    restored = refs.materialize_arguments("list_file_visuals", {**FILE, "cursor_ref": "p1"}, path="")["cursor"]
    assert _cursor_offset(restored, source) == 2
    source.snapshot_id = "c" * 64
    with pytest.raises(ToolBusinessFailure, match="cursor"):
        _cursor_offset(restored, source)
    restored = refs.materialize_arguments("list_workspace_directory", {"cursor_ref": "p2"}, path="")["cursor"]
    assert directory_offset(restored, session_id="s", scope_path="docs", snapshot_sha256="b" * 64, item_count=4) == 2
    for scope, snapshot in [("other", "b" * 64), ("docs", "c" * 64)]:
        with pytest.raises(ValueError, match="cursor"):
            directory_offset(restored, session_id="s", scope_path=scope, snapshot_sha256=snapshot, item_count=4)


def test_controller_admission_returns_native_identity_and_precise_repair_path():
    catalog = {item["tool_id"]: item for item in _native_catalog()}
    runtime = SimpleNamespace(attachment_file_catalog=(FILE,), model_catalog=lambda: tuple(catalog.values()),
                              knows_tool=lambda tool_id: tool_id in catalog, is_execution_findings_tool=lambda _: False)
    plan = materialize_l1_plan(L1PlanProposal(objective="Read PDF", acceptances=[{"criterion": "Use evidence"}]),
                               input_message_id="message", user_text="Read PDF")

    def admit(ref):
        proposal = L1AttemptDecisionProposal.model_validate({"note": "Read the next page", "action": {
            "kind": "call_tools", "calls": [{"tool_id": "list_file_visuals", "arguments": {"file_ref": "f1", "cursor_ref": ref}}],
        }})
        return _admit_decision(proposal, current_plan=plan, accepted=SimpleNamespace(), l1_turn_run_id="run",
                               tool_runtime=runtime, execution=_execution(), finalization_required=False,
                               max_tool_calls_per_attempt=8, tool_reference_catalog=_catalog())

    decision, _ = admit("p1")
    assert decision.action.calls[0].arguments == {"file_id": "file-a", "file_version_id": "version-a", "cursor": VISUAL_CURSOR}
    with pytest.raises(L1ControllerFailure) as rejected:
        admit("p2")
    assert rejected.value.repair_issue.code == "host_guard.l1_unavailable_tool_reference"
    assert rejected.value.repair_issue.paths == ("/action/calls/0/arguments/cursor_ref",)


def test_structured_text_cursors_and_unrelated_tool_arguments_stay_unchanged():
    refs = ToolReferences(_catalog())
    text = {"next_cursor": {"element_offset": 4, "character_offset": 10}, "content": UNIT}
    assert refs.project_result("read_pdf_text", text) == text
    malformed = {"items": None}
    assert refs.materialize_arguments("revise_execution_finding", malformed, path="") == malformed
    arguments = {"path": "a.txt", "content": ENTRY}
    assert refs.materialize_arguments("write_workspace_file", arguments, path="") == arguments


def test_semantic_review_uses_the_same_visual_and_cursor_refs_as_the_actor():
    from personagraph.runtime.l1.semantic_verification import _semantic_review_view

    execution = _execution()
    _append(execution, "read_file_visuals", {"requests": [{**FILE, "visual_unit_id": UNIT}]}, {
        "results": [{**FILE, "visual_unit_id": UNIT, "text": "A chart"}],
    })
    before = deepcopy(execution)
    plan = materialize_l1_plan(L1PlanProposal(objective="Read PDF", acceptances=[{"criterion": "Use evidence"}]),
                               input_message_id="message", user_text="Read PDF")
    view = _semantic_review_view(plan=plan, reply="A chart", references=(), execution=execution,
                                 model_view={"tool_references": _catalog(execution)})
    history = view["execution_context"]["tool_calls"]
    assert history[0]["reading_scope"]["next_cursor_ref"] == "p1"
    assert history[1]["reading_scope"]["next_cursor_ref"] == "p2"
    assert history[2]["arguments"]["requests"][0]["visual_ref"] == "v1"
    body = canonical_json(view)
    assert all(raw not in body for raw in (UNIT, VISUAL_CURSOR, DIRECTORY_CURSOR))
    assert execution == before
