"""Current L1 file and history retrieval integration; no retired candidate protocol."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import time
import uuid

import pytest

from personagraph.session.attachments.application import accept_upload
from personagraph.workspace.storage.context import connect_current
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.session.composition import build_session_retrieval_composition
from personagraph.retrieval.sources.session.lifecycle import ensure_session_retrieval_ready
from personagraph.session import store as session_store
from personagraph.tools.execution import ToolBusinessFailure, ToolExecutor, ResolvedInvocation
from personagraph.tools.model_interface import project_tool_result
from personagraph.tools.schema_validation import ToolSchemaCompiler
from tests.helpers.session_records import complete_test_turn_execution

pytestmark = pytest.mark.usefixtures("partitioned_project_state")


@pytest.fixture(autouse=True)
def _direct_l1_request_scope(tmp_path, monkeypatch, partitioned_project_state):
    original = session_store.create_session
    scopes = ExitStack()

    def create(*args, **kwargs):
        if kwargs.get("working_dir") is None:
            root = tmp_path / "project"
            root.mkdir(exist_ok=True)
            kwargs["working_dir"] = str(root)
        session_id = original(*args, **kwargs)
        scopes.enter_context(session_store.session_database_scope(session_id))
        return session_id

    monkeypatch.setattr(session_store, "create_session", create)
    try:
        yield
    finally:
        scopes.close()


def _features(*, files=False, current_session=False):
    return {
        "file_retrieval_write_enabled": files, "file_retrieval_read_enabled": files,
        "history_retrieval_write_enabled": current_session,
        "history_retrieval_read_enabled": current_session,
        "l1_retrieval_tools_enabled": files or current_session,
    }


def _registration(runtime, tool_id):
    return runtime.registrations_by_tool_id[tool_id]


def _call(runtime, tool_id, payload):
    registration = _registration(runtime, tool_id)
    outcome = ToolExecutor().execute(ResolvedInvocation(
        registration=registration, arguments=payload,
        deadline_monotonic=time.monotonic() + 30,
        logical_tool_call_id=f"test-call-{uuid.uuid4().hex}",
    ))
    assert outcome.status.value == "succeeded", outcome.error
    result = outcome.to_dict()["result"]
    ToolSchemaCompiler().compile(registration.spec.output_schema, role="output").validate(result)
    return result


def test_l1_global_file_definitions_remain_native_but_unavailable_tools_are_hidden():
    session_id = session_store.create_session("Entelecheia")
    runtime = build_l1_tool_runtime(session_id)
    names = {item["tool_id"] for item in runtime.model_catalog()}
    definitions = {item.identity.tool_id for item in runtime.definitions}
    assert {"check_files_state", "prepare_files", "retrieve_files", "read_file_chunks",
            "inspect_file_chunks", "search_file_text"} <= definitions
    assert names == set(runtime.registrations_by_tool_id) - runtime.disabled_tool_ids
    assert not {"list_file_candidates", "select_file_candidates", "prepare_file_candidates",
                "retrieve_file_candidates"} & names
    assert "retrieve_files" not in runtime.registrations_by_tool_id
    assert "retrieve_files" not in names


def test_l1_current_session_retrieval_uses_frozen_cutoff_and_shared_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.runtime.l1 import history_retrieval_composition

    session_id = session_store.create_session("Entelecheia")
    first_result = complete_test_turn_execution(
        session_id,
        1,
        user_content="Where is the aurora ledger?",
        assistant_content="The aurora ledger is in cabinet seven.",
    )
    first = first_result["pair"]
    composition = build_session_retrieval_composition(
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        encoder=DeterministicLexicalEncoder(),
        reranker=None,
        store=session_store,
    )
    monkeypatch.setattr(
        history_retrieval_composition,
        "build_session_retrieval_composition",
        lambda: composition,
    )

    runtime = build_l1_tool_runtime(
        session_id,
        execution_features=_features(current_session=True),
    )
    retrieve = _registration(runtime, "retrieve_history")
    assert tuple(
        retrieve.spec.input_schema["properties"]["scopes"]["items"]["enum"]
    ) == ("current_session", "long_term_user", "current_task")

    with pytest.raises(ToolBusinessFailure) as raised:
        retrieve.handler(
            {"query": "private memory", "scopes": ["long_term_user"]}
        )
    assert raised.value.error.code == "history_scope_outside_frozen_scope"

    matched = retrieve.handler(
        {
            "query": "aurora ledger cabinet",
            "scopes": ["current_session"],
            "limit": 4,
        }
    )
    assert matched["outcome"] == "matched"
    assert "cabinet seven" in matched["evidence"][0]["text"]
    assert matched["evidence"][0]["locator"]["assistant_turn_index"] == first[
        "assistant_turn_idx"
    ]

    from personagraph.tools.catalog.binding import BoundToolRegistration
    from personagraph.tools.retrieval.history_retrieval_catalog import (
        build_history_retrieval_tool_definition_manifest,
    )

    source = history_retrieval_composition.build_l1_history_retrieval_tool_source(
        session_id=session_id,
        turn_id="private-turn-id",
        session_retrieval_assistant_turn_cutoff=int(first["assistant_turn_idx"]),
    )
    assert source is not None
    assert len(source.history_retrieval_bindings) == 1
    binding = source.history_retrieval_bindings[0]
    definition = build_history_retrieval_tool_definition_manifest()[0].definition
    assert BoundToolRegistration(definition, binding).descriptor() == (
        source.registrations[0].descriptor()
    )
    binding_json = json.dumps(binding.descriptor(), ensure_ascii=False)
    assert session_id not in binding_json
    assert f"assistant_turn_idx:{first['assistant_turn_idx']}" not in binding_json
    assert "private-turn-id" not in binding_json

    later_result = complete_test_turn_execution(
        session_id,
        2,
        user_content="What is the future-only password?",
        assistant_content="The future-only password is indigo-orbit.",
    )
    later = later_result["pair"]
    ensure_session_retrieval_ready(
        composition,
        session_id=session_id,
        assistant_turn_cutoff=int(later["assistant_turn_idx"]),
    )
    frozen = retrieve.handler(
        {
            "query": "future-only indigo-orbit",
            "scopes": ["current_session"],
            "limit": 4,
        }
    )
    assert all("indigo-orbit" not in item["text"] for item in frozen["evidence"])
    assert session_id not in json.dumps(
        {"matched": matched, "frozen": frozen},
        ensure_ascii=False,
    )



def test_l1_direct_path_prepare_retrieve_and_exact_read_survives_new_turn(tmp_path, monkeypatch):
    from personagraph.workspace.ingestion.composition import build_document_maintenance_lifecycle

    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    workspace = tmp_path / "files"
    workspace.mkdir()
    (workspace / "northstar.txt").write_text("Northstar validation accuracy is 71 percent.")
    (workspace / "untouched.txt").write_text("Southstar accuracy is 62 percent.")
    session_id = session_store.create_session("Entelecheia", working_dir=str(workspace))
    runtime = build_l1_tool_runtime(session_id, execution_features=_features(files=True))

    checked = _call(runtime, "check_files_state", {"files": [{"path": "northstar.txt"}]})
    assert checked["results"][0]["status"] == "not_ingested"
    assert checked["results"][0]["file_id"] is None
    with connect_current() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0

    lifecycle = build_document_maintenance_lifecycle(profile=DocumentRetrievalProfile.lexical())
    assert lifecycle.start() is True
    try:
        prepared = _call(runtime, "prepare_files", {"files": [{"path": "northstar.txt"}]})
    finally:
        assert lifecycle.stop(timeout_seconds=2.0) is True
    item = prepared["results"][0]
    assert item["status"] == "ready", prepared
    assert prepared["ready_indices"] == [0]
    with connect_current() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
    checked = _call(runtime, "check_files_state", {"files": [{"file_id": item["file_id"]}]})
    assert checked["results"][0]["file_version_id"] == item["file_version_id"]
    assert checked["results"][0]["status"] == "ready"

    # New runtime has no candidate ledger and no attachment; Session mounts are enough.
    next_runtime = build_l1_tool_runtime(session_id, execution_features=_features(files=True))
    matched = _call(next_runtime, "retrieve_files", {
        "queries": ["Northstar accuracy", "validation percent"], "result_limit": 16,
    })
    assert matched["outcome"] == "matched", matched
    evidence = next(e for e in matched["evidence"] if e["file_id"] == item["file_id"])
    assert evidence["source"]["file_name"] == "northstar.txt"
    assert evidence["source"]["relative_path"] == "northstar.txt"
    assert evidence["query_index"] in {0, 1}
    assert evidence["document_version_id"] == item["document_version_id"]
    model_result = project_tool_result("retrieve_files", matched)
    file = next(row for row in model_result["files"] if row["file_id"] == item["file_id"])
    assert file["document_version_id"] == evidence["document_version_id"]
    assert file["source"]["relative_path"] == "northstar.txt"
    assert not {"rank", "query_matches", "content_sha256"} & file["chunks"][0].keys()
    read = _call(next_runtime, "read_file_chunks", {"targets": [{
        "chunk_ids": [evidence["chunk_id"]],
    }]})
    assert "71 percent" in read["results"][0]["chunks"][0]["content"]
    target = {"file_id": evidence["file_id"], "document_version_id": evidence["document_version_id"]}
    inventory = _call(next_runtime, "inspect_file_chunks", {"targets": [target]})["results"][0]
    assert inventory["total_chunk_count"] >= 1
    assert evidence["chunk_id"] in {chunk["chunk_id"] for chunk in inventory["chunks"]}
    literal = _call(next_runtime, "search_file_text", {"targets": [target], "text": "71 percent"})["results"][0]
    assert literal["scan_complete"] is True and literal["total_match_count"] == 1
    assert literal["matches"][0]["text"] == "71 percent"
    selected = _call(next_runtime, "retrieve_files", {
        "queries": ["accuracy"], "file_ids": [item["file_id"]],
    })
    assert all(e["file_id"] == item["file_id"] for e in selected["evidence"])
    denied = _call(next_runtime, "retrieve_files", {
        "queries": ["accuracy"], "file_ids": ["unknown-file"],
    })
    assert denied["status"] == "blocked" and denied["evidence"] == []

    (workspace / "northstar.txt").write_text("Northstar validation accuracy is 80 percent.")
    changed = _call(next_runtime, "check_files_state", {"files": [{"file_id": item["file_id"]}]})
    assert changed["results"][0]["status"] == "changed"
    stale = _call(next_runtime, "read_file_chunks", {"targets": [{
        "chunk_ids": [evidence["chunk_id"]],
    }]})
    assert stale["unavailable_targets"]
    for tool_id, payload in (
        ("inspect_file_chunks", {"targets": [target]}),
        ("search_file_text", {"targets": [target], "text": "71 percent"}),
    ):
        rejected = _call(next_runtime, tool_id, payload)["results"][0]
        assert rejected["status"] == "unavailable"
    updated = _call(next_runtime, "prepare_files", {"files": [{"file_id": item["file_id"]}]})
    assert updated["results"][0]["status"] == "ready"
    assert updated["results"][0]["file_id"] == item["file_id"]
    assert updated["results"][0]["file_version_id"] != item["file_version_id"]
    retired = _call(next_runtime, "read_file_chunks", {"targets": [{"chunk_ids": [evidence["chunk_id"]]}]})
    assert retired["results"] == [] and retired["unavailable_targets"]


def test_l1_attachment_ids_prepare_only_explicitly_requested_file(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    session_id = session_store.create_session("Entelecheia")
    uploads = [accept_upload(
        session_id=session_id, raw_name=name, declared_media_type="text/plain",
        payload=text, store=session_store,
    ) for name, text in (
        ("chosen.txt", b"Chosen file has the violet lantern."),
        ("untouched.txt", b"Unselected document must not be parsed."),
    )]
    accepted = session_store.accept_turn_execution(
        session_id=session_id, client_request_id="native-file-turn", source="runtime_test",
        user_text="Read chosen.txt", attachment_ids=tuple(u.attachment_id for u in uploads),
        lease_owner="native-file-test",
    )
    runtime = build_l1_tool_runtime(
        session_id, turn_id=str(accepted["turn"]["turn_id"]), execution_features=_features(files=True),
    )
    catalog = {item["name"]: item for item in runtime.attachment_file_catalog}
    selected = catalog["chosen.txt"]
    prepared = _call(runtime, "prepare_files", {"files": [{
        "file_id": selected["file_id"], "file_version_id": selected["file_version_id"],
    }]})
    assert prepared["results"][0]["status"] == "ready", prepared
    with connect_current() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    matched = _call(runtime, "retrieve_files", {"queries": ["violet lantern"]})
    assert matched["outcome"] == "matched"
    assert {item["file_id"] for item in matched["evidence"]} == {selected["file_id"]}


def test_l1_literal_count_uses_source_text_before_chunk_heading_repetition(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    workspace = tmp_path / "source-text-project"
    workspace.mkdir()
    text = "# Observatory\n\n" + "\n\n".join(
        f"Paragraph {index}: " + "A modest instrument records the changing sky. " * 90
        for index in range(4)
    )
    (workspace / "notes.md").write_text(text)
    session_id = session_store.create_session("Entelecheia", working_dir=str(workspace))
    runtime = build_l1_tool_runtime(session_id, execution_features=_features(files=True))
    prepared = _call(runtime, "prepare_files", {"files": [{"path": "notes.md"}]})["results"][0]
    assert prepared["status"] == "ready"
    target = {"file_id": prepared["file_id"], "document_version_id": prepared["document_version_id"]}
    inventory = _call(runtime, "inspect_file_chunks", {"targets": [target]})["results"][0]
    assert inventory["total_chunk_count"] > 1
    read = _call(runtime, "read_file_chunks", {"targets": [{**target, "chunk_sequences": [0, 1]}]})
    assert sum(chunk["content"].count("Observatory") for chunk in read["results"][0]["chunks"]) >= 2
    matched = _call(runtime, "search_file_text", {"targets": [target], "text": "Observatory"})["results"][0]
    assert matched["scan_complete"] is True and matched["total_match_count"] == 1
