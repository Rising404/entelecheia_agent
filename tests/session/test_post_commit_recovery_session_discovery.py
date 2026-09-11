"""恢复枚举只读取目录身份，坏会话数据库不能阻断其余会话。"""

import sqlite3

import pytest

from personagraph.session import store
from personagraph.session.catalog import SessionCatalog
from tests.helpers.session_records import complete_test_turn_execution


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_partitioned_recovery_ids_survive_a_damaged_session_database(
    partitioned_project_state, tmp_path, monkeypatch, damage,
):
    healthy = store.create_session("Entelecheia", title="healthy")
    broken = store.create_session("Entelecheia", title="broken")
    archived = store.create_session("Entelecheia", title="archived")
    trashed = store.create_session("Entelecheia", title="trashed")
    catalog = SessionCatalog()
    catalog.update_session(archived, status="archived")
    catalog.update_session(trashed, status="trashed")
    with store.session_database_scope(healthy):
        complete_test_turn_execution(
            healthy, 0, post_commit_job_kinds=("session_retrieval_index",),
        )
    broken_database = catalog.session_db_path(broken)
    if damage == "missing":
        broken_database.unlink()
    else:
        broken_database.write_bytes(b"not a SQLite database")

    # The GUI's full listing really does fail; recovery must not share that path.
    with pytest.raises((store.SessionStoreError, sqlite3.DatabaseError)):
        store.list_sessions(status="all")
    identifiers = store.list_session_ids_for_post_commit_recovery()
    assert set(identifiers) == {healthy, broken, archived}

    from personagraph.runtime.post_commit import lifecycle

    scheduled = []
    monkeypatch.setattr(
        lifecycle, "schedule_turn_post_commit_jobs",
        lambda **kwargs: scheduled.append(kwargs["session_id"]),
    )
    discovery = lifecycle.build_turn_post_commit_lifecycle(store=store)
    # Discover the broken entry first, so failure cannot be masked by ordering.
    discovery._discover_session(broken)
    discovery._discover_session(healthy)
    assert scheduled == [healthy]
    if damage == "missing":
        assert not broken_database.exists(), "read-only recovery discovery must not recreate a missing source"


def test_explicit_single_database_recovery_ids_do_not_open_the_catalog(monkeypatch):
    first = store.create_session("Entelecheia", title="first")
    second = store.create_session("Entelecheia", title="second")
    trashed = store.create_session("Entelecheia", title="trashed")
    store.archive_session(second)
    store.trash_session(trashed)
    monkeypatch.setattr(store, "_catalog", lambda: pytest.fail("single database must not read catalog"))

    assert store._uses_explicit_database_override()
    assert set(store.list_session_ids_for_post_commit_recovery()) == {first, second}


def test_partitioned_recovery_identity_enumeration_does_not_open_session_payloads(
    partitioned_project_state, monkeypatch,
):
    session_id = store.create_session("Entelecheia")
    monkeypatch.setattr(
        store, "_read_catalog_session",
        lambda _: pytest.fail("recovery identity enumeration must not expand a session database"),
    )
    assert store.list_session_ids_for_post_commit_recovery() == (session_id,)
