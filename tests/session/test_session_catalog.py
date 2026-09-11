import sqlite3

import pytest

from personagraph.session import project_catalog as session_projects
from personagraph.session import catalog as catalog_module
from personagraph.session.catalog import (
    SCHEMA_VERSION,
    SessionCatalog,
    catalog_db_path,
    session_db_path,
    validate_session_id,
)


def test_session_database_paths_are_physically_partitioned(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)

    path_a = catalog.session_db_path("session-A")
    path_b = catalog.session_db_path("session_B")

    assert path_a == tmp_path / "sessions" / "session-A" / "session.sqlite"
    assert path_b == tmp_path / "sessions" / "session_B" / "session.sqlite"
    assert path_a != path_b
    assert path_a.parent.parent == path_b.parent.parent == tmp_path / "sessions"


def test_default_catalog_uses_the_project_catalog_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        tmp_path / "project_catalog.sqlite",
    )

    catalog = SessionCatalog()

    assert catalog.db_path == tmp_path / "project_catalog.sqlite"
    assert catalog.session_db_path("session-A") == (
        tmp_path / "sessions" / "session-A" / "session.sqlite"
    )


@pytest.mark.parametrize(
    "session_id",
    (
        "",
        ".",
        "..",
        "../outside",
        "nested/session",
        r"nested\session",
        "/absolute",
        " leading",
        "trailing ",
        "session.id",
        "会话",
    ),
)
def test_session_id_rejects_unsafe_or_noncanonical_values(tmp_path, session_id):
    with pytest.raises(ValueError, match="session_id"):
        validate_session_id(session_id)
    with pytest.raises(ValueError, match="session_id"):
        session_db_path(session_id, state_dir=tmp_path)


def test_session_database_path_rejects_symlink_escape(tmp_path):
    sessions_dir = tmp_path / "sessions"
    outside = tmp_path / "outside"
    sessions_dir.mkdir()
    outside.mkdir()
    (sessions_dir / "session-A").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        session_db_path("session-A", state_dir=tmp_path)


def test_shared_catalog_path_rejects_file_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    state_dir = tmp_path / "state"
    outside.mkdir()
    state_dir.mkdir(exist_ok=True)
    escaped = outside / "escaped.sqlite"
    (state_dir / "project_catalog.sqlite").symlink_to(escaped)

    with pytest.raises(ValueError, match="project catalog"):
        catalog_db_path(state_dir=state_dir)
    assert not escaped.exists()


def test_catalog_crud_keeps_only_routing_and_lifecycle_metadata(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    at_1 = "2026-08-30T12:00:00+00:00"
    at_2 = "2026-08-30T12:01:00+00:00"

    folder = catalog.create_folder(folder_id="research", name="Research", occurred_at=at_1)
    session_a = catalog.create_session(
        session_id="session-A",
        project_id="project-1",
        folder_id=folder["id"],
        title="First",
        persona_id="default",
        occurred_at=at_1,
    )
    catalog.create_session(
        session_id="session-B",
        project_id="project-2",
        title="Second",
        occurred_at=at_1,
    )

    assert session_a["db_path"] == "sessions/session-A/session.sqlite"
    assert catalog.get_session("session-A") == session_a
    assert [row["id"] for row in catalog.list_sessions(project_id="project-1")] == ["session-A"]

    updated = catalog.update_session(
        "session-A",
        title="Renamed",
        folder_id=None,
        status="archived",
        last_active_at=at_2,
        occurred_at=at_2,
    )
    assert updated is not None
    assert updated["title"] == "Renamed"
    assert updated["folder_id"] is None
    assert updated["status"] == "archived"
    assert updated["last_active_at"] == at_2
    assert updated["archived_at"] == at_2
    assert [row["id"] for row in catalog.list_sessions(status="archived")] == ["session-A"]

    assert not catalog.session_db_path("session-A").exists()
    assert not catalog.session_db_path("session-B").exists()


def test_catalog_allows_a_projectless_conversation(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)

    created = catalog.create_session(
        session_id="projectless-session",
        project_id=None,
    )

    assert created["project_id"] is None


def test_catalog_schema_coexists_with_projects_and_contains_no_session_payload_tables(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "workspace"
    project_root.mkdir()
    shared_db = tmp_path / "project_catalog.sqlite"
    monkeypatch.setattr(session_projects, "DB_PATH", shared_db)
    project = session_projects.remember(str(project_root))
    catalog = SessionCatalog(state_dir=tmp_path)
    catalog.initialize()

    with sqlite3.connect(catalog.db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

    assert version == SCHEMA_VERSION
    assert tables == {
        "projects",
        "sessions",
        "session_folders",
        "session_purge_tombstones",
        "session_creation_requests",
    }
    assert catalog.create_session(
        session_id="session-A",
        project_id=project.project_id,
    )["project_id"] == project.project_id
    forbidden = ("message", "prompt", "output", "attachment", "turn", "chunk", "document")
    assert not any(token in table for table in tables for token in forbidden)


def test_project_catalog_can_initialize_after_session_tables(tmp_path, monkeypatch):
    catalog = SessionCatalog(state_dir=tmp_path)
    catalog.initialize()
    monkeypatch.setattr(session_projects, "DB_PATH", catalog.db_path)
    project_root = tmp_path / "workspace"
    project_root.mkdir()

    project = session_projects.remember(str(project_root))

    assert project.project_id
    assert catalog.create_session(
        session_id="session-A",
        project_id=project.project_id,
    )["project_id"] == project.project_id


def test_folder_update_rejects_hierarchy_cycle(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    catalog.create_folder(folder_id="parent", name="Parent")
    catalog.create_folder(folder_id="child", name="Child", parent_id="parent")

    with pytest.raises(ValueError, match="descendants"):
        catalog.update_folder("parent", parent_id="child")

    assert catalog.get_folder("parent")["parent_id"] is None
