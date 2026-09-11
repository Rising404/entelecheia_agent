import sqlite3

import pytest

from personagraph.session import store


def test_connect_wraps_directory_creation_error(tmp_path, monkeypatch):
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("blocks mkdir", encoding="utf-8")
    monkeypatch.setattr(store, "DB_PATH", blocking_file / "sessions.sqlite")

    with pytest.raises(store.SessionStoreError, match="Failed to open session database") as exc_info:
        store._connect()

    assert isinstance(exc_info.value.__cause__, OSError)


def test_init_wraps_schema_sqlite_error_without_caching_path(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.sqlite"
    monkeypatch.setattr(store, "DB_PATH", db_path)
    store._INITIALIZED_PATHS.discard(str(db_path))

    def _broken_schema(_conn):
        raise sqlite3.OperationalError("schema write failed")

    monkeypatch.setattr(store, "initialize_schema", _broken_schema)

    with pytest.raises(
        store.SessionStoreError,
        match="Failed to initialize session database schema",
    ) as exc_info:
        store.init_db()

    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)
    assert str(db_path) not in store._INITIALIZED_PATHS
