"""会话范围文档挂载的工作区会话契约。"""

import pytest

from personagraph.api import router, service
from personagraph.session import store as session_store
from tests.documents._authority import ingest_registered_document_fixture


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


def _route_payload(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


def test_documents_filter_to_their_existing_session_mounts(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_a = session_store.create_session(
        "Entelecheia",
        title="A",
        working_dir=str(workspace),
    )
    session_b = session_store.create_session(
        "Entelecheia",
        title="B",
        working_dir=str(workspace),
    )
    source = workspace / "workspace-notes.md"
    source.write_text("项目内文档", encoding="utf-8")
    with session_store.session_database_scope(session_a):
        ingested = ingest_registered_document_fixture(
            source,
            session_id=session_a,
        )
    assert ingested["ok"] is True
    document_id = ingested["doc_id"]

    assert [document["id"] for document in _route_payload("GET", f"/api/documents?session_id={session_a}")["documents"]] == [document_id]
    assert _route_payload("GET", f"/api/documents?session_id={session_b}")["documents"] == []
    with pytest.raises(service.ApiError) as missing_scope:
        _route_payload("GET", "/api/documents")
    assert missing_scope.value.code == "MISSING_FIELD"

    with pytest.raises(service.ApiError) as exc:
        _route_payload("GET", "/api/documents?session_id=missing")
    assert exc.value.code == "SESSION_NOT_FOUND"
