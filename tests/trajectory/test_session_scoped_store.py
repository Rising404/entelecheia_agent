from __future__ import annotations

import sqlite3

import pytest

from personagraph.configuration import paths
from personagraph.session import catalog as catalog_module
from personagraph.session import store as session_store
from personagraph.session.catalog import SessionCatalog
from personagraph.trajectory import (
    Part,
    PartRole,
    Step,
    StepKind,
    TrajectoryStore,
    text_blob,
)


def _step(step_id: str, session_id: str) -> Step:
    return Step(
        step_id=step_id,
        kind=StepKind.MODEL_CALL,
        occurred_at="2026-08-30T00:00:00+00:00",
        parts=(Part(PartRole.USER, text_blob("session-owned")),),
        session_id=session_id,
    )


@pytest.fixture
def partitioned_trajectory_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "var"
    shared_catalog = state_dir / "project_catalog.sqlite"
    retired_global_path = state_dir / "trajectory" / "trajectory.sqlite"

    monkeypatch.setattr(session_store, "DB_PATH", session_store._DEFAULT_DB_PATH)
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        shared_catalog,
    )
    monkeypatch.setattr(paths, "SESSIONS_DIR", state_dir / "sessions")
    session_store._INITIALIZED_PATHS.clear()
    return state_dir, retired_global_path


def test_default_store_fails_closed_without_a_session_scope(
    partitioned_trajectory_state,
) -> None:
    _state_dir, retired_global_path = partitioned_trajectory_state

    with pytest.raises(
        session_store.SessionStoreError,
        match="no session database is bound",
    ):
        TrajectoryStore().record(_step("outside", "missing"))

    assert not retired_global_path.exists()


def test_default_store_uses_and_isolates_each_bound_session_database(
    partitioned_trajectory_state,
) -> None:
    _state_dir, retired_global_path = partitioned_trajectory_state
    session_a = session_store.create_session("Entelecheia", title="A")
    session_b = session_store.create_session("Entelecheia", title="B")
    catalog = SessionCatalog()
    path_a = catalog.session_db_path(session_a)
    path_b = catalog.session_db_path(session_b)
    store = TrajectoryStore()

    with session_store.session_database_scope(session_a):
        assert session_store.current_session_database_path() == path_a
        assert store.path == path_a
        assert store.record(_step("step-a", session_a)) is True

    with session_store.session_database_scope(session_b):
        assert session_store.current_session_database_path() == path_b
        assert store.path == path_b
        assert store.record(_step("step-b", session_b)) is True

    with sqlite3.connect(path_a) as connection:
        assert connection.execute(
            "SELECT step_id FROM trajectory_steps"
        ).fetchall() == [("step-a",)]
    with sqlite3.connect(path_b) as connection:
        assert connection.execute(
            "SELECT step_id FROM trajectory_steps"
        ).fetchall() == [("step-b",)]

    assert not retired_global_path.exists()
    with pytest.raises(session_store.SessionStoreError):
        session_store.current_session_database_path()


def test_session_delete_cascades_trajectory_steps_and_parts() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    from personagraph.session.persistence.schema import initialize_schema

    initialize_schema(connection)
    connection.execute(
        "INSERT INTO sessions (id, persona_id) VALUES ('s1', 'Entelecheia')"
    )
    connection.execute(
        "INSERT INTO trajectory_blobs "
        "(sha256, byte_count, text, truncated, first_seen_at) "
        "VALUES (?, 1, 'x', 0, 'now')",
        ("a" * 64,),
    )
    connection.execute(
        "INSERT INTO trajectory_steps "
        "(step_id, kind, occurred_at, session_id, outcome) "
        "VALUES ('step', 'model_call', 'now', 's1', 'ok')"
    )
    connection.execute(
        "INSERT INTO trajectory_parts (step_id, seq, role, blob_sha256) "
        "VALUES ('step', 0, 'user', ?)",
        ("a" * 64,),
    )

    connection.execute("DELETE FROM sessions WHERE id='s1'")

    assert connection.execute(
        "SELECT COUNT(*) FROM trajectory_steps"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM trajectory_parts"
    ).fetchone()[0] == 0
