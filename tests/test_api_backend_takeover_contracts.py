"""会话范围文档解绑的后端接管契约。"""
from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.api import router, service, workspace_documents
from personagraph.workspace.documents import application as docstore
from personagraph.session import store as session_store
from tests.documents._authority import ingest_registered_document_fixture


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


@pytest.fixture(autouse=True)
def _project_sessions(tmp_path, monkeypatch, partitioned_project_state):
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_ids = iter(("s1", "other"))
    monkeypatch.setattr(session_store, "_new_id", lambda: next(session_ids))
    assert session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    ) == "s1"
    assert session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    ) == "other"


def _route_payload(method: str, path: str, body: dict | None = None):
    return router.dispatch_response(method, path, body or {}).payload


def _ingest_doc(session_id: str = "s1") -> str:
    session = session_store.get_session(session_id)
    source = Path(str(session["working_dir"])) / "rules.md"
    source.write_text("评审规则" * 80, encoding="utf-8")
    with session_store.session_database_scope(session_id):
        result = ingest_registered_document_fixture(
            source,
            session_id=session_id,
        )
    assert result["ok"] is True
    return result["doc_id"]


def test_document_detach_keeps_document_index():
    doc_id = _ingest_doc("s1")

    result = _route_payload("POST", f"/api/documents/{doc_id}/detach", {"session_id": "s1"})

    assert result == {"ok": True, "doc_id": doc_id, "session_id": "s1", "detached": True}
    with session_store.session_database_scope("s1"):
        assert docstore.mounted_docs("s1") == []
    assert _route_payload(
        "GET",
        f"/api/documents/{doc_id}?session_id=s1",
    )["document"]["id"] == doc_id


def test_document_detach_missing_mount_is_structured_404():
    doc_id = _ingest_doc("s1")

    with pytest.raises(service.ApiError) as exc:
        _route_payload("POST", f"/api/documents/{doc_id}/detach", {"session_id": "other"})

    assert exc.value.code == "DOCUMENT_MOUNT_NOT_FOUND"
    assert exc.value.status == 404


def test_workspace_documents_reports_domain_reason_before_http_mapping():
    with session_store.session_database_scope("s1"):
        result = workspace_documents.detach_document("missing-doc", "s1")

    assert result == {
        "ok": False,
        "reason": "document_not_found",
        "doc_id": "missing-doc",
    }
