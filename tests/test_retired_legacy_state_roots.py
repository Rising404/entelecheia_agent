from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from personagraph.workspace.storage.context import bind as bind_project_documents
from personagraph.workspace.storage.database import DocumentDatabase
from personagraph.workspace.files import attachments as attachment_storage
from personagraph.retrieval import sqlite_store as retrieval_store


PNG = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 64)
LEGACY_STATE_ROOTS = frozenset(
    {"memory", "retrieval_indexes", "session_files", "session_history", "trajectory"}
)


def test_cold_start_creates_partitioned_authorities_without_legacy_roots(
    tmp_path,
) -> None:
    state_dir = tmp_path / "var"
    project_root = tmp_path / "project"
    project_root.mkdir()
    script = r'''
import json
from pathlib import Path

from personagraph.session.attachments.application import accept_upload
from personagraph.session import store

project_root = Path(__import__("os").environ["ENTELECHEIA_TEST_PROJECT_ROOT"])
session_id = store.create_session(
    "Entelecheia",
    title="cold-start",
    working_dir=str(project_root),
)
with store.session_database_scope(session_id):
    accepted = accept_upload(
        session_id=session_id,
        raw_name="evidence.txt",
        declared_media_type="text/plain",
        payload=b"cold-start project evidence\n",
        store=store,
    )
    attachment = store.get_attachment(accepted.attachment_id)
    assert attachment is not None
session = store.get_session(session_id)
print(json.dumps({
    "session_id": session_id,
    "project_id": session["project_id"],
    "stored_rel_path": attachment["stored_rel_path"],
}))
'''
    environment = os.environ.copy()
    environment["PERSONAGRAPH_STATE_DIR"] = str(state_dir)
    environment["ENTELECHEIA_TEST_PROJECT_ROOT"] = str(project_root)
    source_root = Path(__file__).resolve().parents[1] / "src"
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(source_root), inherited_pythonpath)
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    catalog_path = state_dir / "project_catalog.sqlite"
    session_path = state_dir / "sessions" / result["session_id"] / "session.sqlite"
    documents_path = (
        state_dir / "projects" / result["project_id"] / "documents.sqlite"
    )
    assert catalog_path.is_file()
    assert session_path.is_file()
    assert documents_path.is_file()
    assert (project_root / result["stored_rel_path"]).read_bytes() == (
        b"cold-start project evidence\n"
    )
    assert not (project_root / "input").exists()
    assert not (project_root / "output").exists()
    assert LEGACY_STATE_ROOTS.isdisjoint(
        {path.name for path in state_dir.iterdir() if path.is_dir()}
    )

    with sqlite3.connect(catalog_path) as connection:
        catalog_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "sessions" in catalog_tables
    assert "session_turns" not in catalog_tables
    assert "runtime_turns" not in catalog_tables

    with sqlite3.connect(documents_path) as connection:
        origin = connection.execute("SELECT origin FROM files").fetchone()[0]
        assert origin == "user_upload"
        assert connection.execute("SELECT COUNT(*) FROM file_versions").fetchone()[0] == 1
        document_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "retrieval_update_outbox",
            "retrieval_outbox_manual_actions",
        } <= document_tables

    with sqlite3.connect(session_path) as connection:
        attachment = connection.execute(
            "SELECT project_id, file_id, file_version_id FROM session_attachments"
        ).fetchone()
        assert tuple(attachment)[0] == result["project_id"]
        assert all(value for value in attachment)
        session_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "session_workspace_read_grants",
            "session_workspace_root_observations",
            "session_workspace_write_grants",
        } <= session_tables
        assert {
            "retrieval_update_outbox",
            "retrieval_outbox_manual_actions",
        }.isdisjoint(session_tables)


def test_default_file_retrieval_requires_project_context_without_creating_a_directory(
    tmp_path,
) -> None:
    default_path = tmp_path / "var" / "retrieval_indexes" / "retrieval.sqlite"

    with pytest.raises(retrieval_store.RetrievalProjectContextRequired):
        retrieval_store.SqliteRetrievalCatalog()

    assert not default_path.parent.exists()


def test_project_retrieval_uses_documents_database_not_legacy_index_root(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "var" / "retrieval_indexes" / "retrieval.sqlite"
    project_root = tmp_path / "project"
    project_root.mkdir()
    database = DocumentDatabase(
        "project-retrieval",
        project_root,
        tmp_path / "var" / "projects" / "project-retrieval" / "documents.sqlite",
    )

    with bind_project_documents(database):
        catalog = retrieval_store.SqliteRetrievalCatalog()
        catalog.initialize()

    assert catalog.db_path == database.db_path
    assert database.db_path.is_file()
    assert not legacy_path.parent.exists()


def test_unbound_upload_fails_closed_without_creating_session_files(
    tmp_path,
) -> None:
    default_root = tmp_path / "var" / "session_files"

    with pytest.raises(attachment_storage.AttachmentProjectContextRequired):
        attachment_storage.store_attachment(
            session_id="session-1",
            attachment_id="attachment-1",
            raw_name="evidence.png",
            payload=PNG,
        )

    assert not default_root.exists()


def test_project_upload_has_one_project_copy_and_no_input_output_tree(
    tmp_path,
) -> None:
    legacy_root = tmp_path / "var" / "session_files"
    project_root = tmp_path / "project"
    project_root.mkdir()
    database = DocumentDatabase(
        "project-upload",
        project_root,
        tmp_path / "var" / "projects" / "project-upload" / "documents.sqlite",
    )

    with bind_project_documents(database):
        stored = attachment_storage.store_attachment(
            session_id="session-1",
            attachment_id="attachment-1",
            raw_name="evidence.png",
            payload=PNG,
        )

    assert stored.absolute_path.read_bytes() == PNG
    assert stored.absolute_path.is_relative_to(project_root / "附件")
    assert not (project_root / "input").exists()
    assert not (project_root / "output").exists()
    assert not legacy_root.exists()
