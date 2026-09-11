import sqlite3

import pytest

from personagraph.session.catalog import (
    CreationRequestConflict,
    CreationRequestInProgress,
    SessionCatalog,
)


def test_creation_receipt_survives_catalog_reopen(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    assert catalog.reserve_creation_request("request-1", "a" * 64) is None
    catalog.create_session(session_id="session-1", creation_request_id="request-1")

    reopened = SessionCatalog(state_dir=tmp_path)
    assert reopened.reserve_creation_request("request-1", "a" * 64) == "session-1"
    assert reopened.release_unpublished_creation("request-1") is False


def test_session_publication_rolls_back_if_receipt_is_not_reserved(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    with pytest.raises(CreationRequestConflict):
        catalog.create_session(session_id="session-1", creation_request_id="missing")
    assert catalog.get_session("session-1") is None


def test_pending_request_survives_catalog_reopen(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    catalog.reserve_creation_request("request-1", "a" * 64)
    with pytest.raises(CreationRequestInProgress):
        SessionCatalog(state_dir=tmp_path).reserve_creation_request("request-1", "a" * 64)


def test_additive_request_schema_preserves_existing_catalog(tmp_path):
    catalog = SessionCatalog(state_dir=tmp_path)
    catalog.create_session(session_id="existing")
    with sqlite3.connect(catalog.db_path) as conn:
        conn.execute("DROP TABLE session_creation_requests")
    catalog.initialize()
    catalog.initialize()
    assert catalog.get_session("existing")["id"] == "existing"
    assert catalog.reserve_creation_request("request-1", "a" * 64) is None
