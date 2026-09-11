from __future__ import annotations

import pytest

from personagraph.workspace.storage.context import (
    bind as bind_project_documents,
    current as current_project_documents,
)
from personagraph.session import store as session_store


def _session(tmp_path, name: str) -> str:
    root = tmp_path / name
    root.mkdir()
    return session_store.create_session(
        "Entelecheia",
        working_dir=str(root.resolve()),
    )


def _database_for(session_id: str):
    with session_store.session_database_scope(session_id):
        database = current_project_documents()
        assert database is not None
        return database


def test_route_scope_binds_only_the_session_under_the_existing_project(
    tmp_path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id = _session(tmp_path, "project-a")
    database = _database_for(session_id)

    with bind_project_documents(database):
        with session_store.session_database_route_scope(session_id) as path:
            assert session_store.current_session_id() == session_id
            assert session_store.current_session_database_path() == path
            assert current_project_documents() is database
            assert session_store.get_session(session_id)["id"] == session_id


def test_route_scope_rejects_a_session_from_another_project(
    tmp_path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    first = _session(tmp_path, "project-a")
    second = _session(tmp_path, "project-b")
    first_database = _database_for(first)

    with bind_project_documents(first_database):
        with pytest.raises(session_store.SessionStoreError, match="same project"):
            with session_store.session_database_route_scope(second):
                pass


def test_route_scope_rejects_missing_project_and_cross_session_nesting(
    tmp_path,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    first = _session(tmp_path, "project-a")
    second = session_store.create_session(
        "Entelecheia",
        working_dir=str((tmp_path / "project-a").resolve()),
    )
    first_database = _database_for(first)

    with pytest.raises(session_store.SessionStoreError, match="bound project"):
        with session_store.session_database_route_scope(first):
            pass

    with bind_project_documents(first_database):
        with session_store.session_database_route_scope(first):
            with pytest.raises(session_store.SessionStoreError, match="active scope"):
                with session_store.session_database_route_scope(second):
                    pass
