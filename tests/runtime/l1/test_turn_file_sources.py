from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from personagraph.input_processing.files import FileKind
import personagraph.runtime.l1.turn_file_sources as turn_file_sources
from personagraph.runtime.l1.tool_catalog_snapshot import (
    _model_catalog_items_from_definitions,
)
from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)
from personagraph.workspace.files.turn_inputs import TurnInputFileAuthorityError


def test_turn_sources_delegate_the_complete_receipt_set_and_session_scope(monkeypatch):
    records = [{"attachment_id": "attachment-one"}, {"attachment_id": "attachment-two"}]
    resolved = (object(), object())
    calls = []
    monkeypatch.setattr(turn_file_sources.store, "session_database_scope", lambda session_id: nullcontext())
    monkeypatch.setattr(turn_file_sources.store, "list_turn_attachments", lambda session_id, turn_id: records)

    def resolve(supplied, *, session_id, turn_id):
        calls.append((supplied, session_id, turn_id))
        return resolved

    monkeypatch.setattr(turn_file_sources, "resolve_turn_input_files", resolve)
    assert turn_file_sources.freeze_turn_attachment_sources(session_id="s", turn_id="t") is resolved
    assert calls == [(records, "s", "t")]

    def reject(*args, **kwargs):
        raise TurnInputFileAuthorityError("file_version_stale")

    monkeypatch.setattr(turn_file_sources, "resolve_turn_input_files", reject)
    with pytest.raises(TurnInputFileAuthorityError, match="file_version_stale"):
        turn_file_sources.freeze_turn_attachment_sources(session_id="s", turn_id="t")


def test_attachment_catalog_exposes_real_file_versions_without_private_paths():
    source = SimpleNamespace(
        file_id="file-one", file_version_id="version-one", original_name="report.pdf",
        relative_path="附件/turn-one/report.pdf",
        media_type="application/pdf", size_bytes=12, kind=FileKind.DOCUMENT,
        canonical_path="/private/uploads/report.pdf", attachment_id="upload-secret",
    )
    assert turn_file_sources.attachment_file_catalog((source,)) == ({
        "file_id": "file-one", "file_version_id": "version-one", "name": "report.pdf",
        "relative_path": "附件/turn-one/report.pdf",
        "media_type": "application/pdf", "size_bytes": 12, "kind": "document",
        "origin": "user_upload",
    },)


def test_compact_l1_catalog_keeps_workspace_relative_path_guidance():
    definitions = tuple(
        seed.definition for seed in build_production_default_catalog_seeds()
    )
    compact = {
        item["tool_id"]: item
        for item in _model_catalog_items_from_definitions(definitions)
    }

    batch_path_tools = {"check_files_state", "prepare_files"}
    format_path_tools = {
        "read_text",
        "read_pdf_text",
        "read_word",
        "read_slides",
        "inspect_image",
        "analyze_image",
        "analyze_pdf_page",
    }
    assert "render_pdf_pages" not in compact
    for tool_id in format_path_tools:
        descriptor = compact[tool_id]
        assert "工作区相对路径" in descriptor["description"]
        path_schema = descriptor["input_schema"]["properties"]["path"]
        assert "工作区相对路径" in path_schema["description"]
        assert "name" in path_schema["description"]
        assert "output_schema" not in descriptor

    for tool_id in batch_path_tools:
        descriptor = compact[tool_id]
        assert "工作区相对路径" in descriptor["description"]
        path_schema = descriptor["input_schema"]["properties"]["files"]["items"]
        path_description = path_schema["properties"]["path"]["description"]
        assert "工作区相对路径" in path_description
        assert "name" in path_description
        assert "output_schema" not in descriptor
