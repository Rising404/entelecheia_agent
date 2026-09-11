from __future__ import annotations

import sqlite3

import pytest

from personagraph.workspace.storage.context import (
    ProjectDocumentContextError,
    bind,
    current,
)
from personagraph.workspace.storage.database import DocumentDatabase
from personagraph.retrieval.contracts import (
    RetrievalStatus,
    RetrievalUnit,
    SourceFilter,
    SourceType,
    SourceUnitRef,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


def _database(tmp_path, project_id: str) -> DocumentDatabase:
    project_root = tmp_path / "roots" / project_id
    project_root.mkdir(parents=True)
    return DocumentDatabase(
        project_id,
        project_root,
        tmp_path / "projects" / project_id / "documents.sqlite",
    )


def _seed_ready_document_unit(catalog: SqliteRetrievalCatalog, project_id: str) -> None:
    document_id = f"doc-{project_id}"
    catalog.create_data_version(
        version_id="generation-v1",
        fingerprint=f"fingerprint-{project_id}",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    stored = catalog.upsert_pending_unit(
        RetrievalUnit(
            ref=SourceUnitRef(
                SourceType.DOCUMENT,
                f"document-v3:{document_id}:chunk-{project_id}",
                "version-1",
                ("a" if project_id == "project-a" else "b") * 64,
            ),
            retrieval_data_version="generation-v1",
            retrieval_status=RetrievalStatus.ACTIVE,
            source_filter=SourceFilter.from_mapping(
                SourceType.DOCUMENT,
                {"doc_id": document_id},
            ),
        )
    )
    catalog.mark_unit_index_ready(stored.unit_id)


def test_default_catalog_uses_the_bound_project_and_keeps_projects_isolated(tmp_path):
    project_a = _database(tmp_path, "project-a")
    project_b = _database(tmp_path, "project-b")

    with bind(project_a):
        catalog_a = SqliteRetrievalCatalog()
        assert catalog_a.db_path == project_a.db_path
        _seed_ready_document_unit(catalog_a, project_a.project_id)

    with bind(project_b):
        catalog_b = SqliteRetrievalCatalog()
        assert catalog_b.db_path == project_b.db_path
        _seed_ready_document_unit(catalog_b, project_b.project_id)

    units_a = catalog_a.active_units("generation-v1")
    units_b = catalog_b.active_units("generation-v1")
    assert [unit.unit.ref.source_unit_id for unit in units_a] == [
        "document-v3:doc-project-a:chunk-project-a"
    ]
    assert [unit.unit.ref.source_unit_id for unit in units_b] == [
        "document-v3:doc-project-b:chunk-project-b"
    ]
    assert catalog_a.db_path != catalog_b.db_path

    for database in (project_a, project_b):
        with sqlite3.connect(database.db_path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
            assert conn.execute(
                "SELECT schema_name FROM schema_meta"
            ).fetchone()[0] == "project_documents"


def test_project_binding_rejects_nesting_and_restores_after_failure(tmp_path):
    project_a = _database(tmp_path, "project-a")
    project_b = _database(tmp_path, "project-b")

    assert current() is None
    with pytest.raises(RuntimeError, match="sentinel"):
        with bind(project_a):
            assert current() is project_a
            with pytest.raises(ProjectDocumentContextError, match="already bound"):
                with bind(project_b):
                    pass
            raise RuntimeError("sentinel")
    assert current() is None


def test_explicit_path_overrides_project_context_and_uses_legacy_schema(tmp_path):
    project = _database(tmp_path, "project-a")
    legacy_path = tmp_path / "explicit-retrieval.sqlite"

    with bind(project):
        catalog = SqliteRetrievalCatalog(legacy_path)
        catalog.initialize()

    assert catalog.db_path == legacy_path
    with catalog.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone() is None


def test_project_catalog_logical_read_does_not_refresh_capability_rows(tmp_path):
    project = _database(tmp_path, "project-a")

    with bind(project):
        catalog = SqliteRetrievalCatalog()
        catalog.initialize()
        with project.connect() as conn:
            conn.execute(
                "UPDATE retrieval_capabilities SET checked_at='sentinel'"
            )

        assert catalog.active_data_version() is None

        with project.connect() as conn:
            checked_at = {
                str(row[0])
                for row in conn.execute(
                    "SELECT checked_at FROM retrieval_capabilities"
                ).fetchall()
            }
        assert checked_at == {"sentinel"}
