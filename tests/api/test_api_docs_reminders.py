"""文档域 API：列表、编辑摘要与删除。"""
from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.api import router, service, workspace_documents
from personagraph.api.service.document_ingest import _ingest_failure_hint
from personagraph.workspace.documents import application as docstore
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)
from personagraph.session import store as ss
from tests.documents._authority import ingest_registered_document_fixture


pytestmark = pytest.mark.usefixtures("partitioned_project_state")


@pytest.fixture(autouse=True)
def _project_session(tmp_path, monkeypatch, partitioned_project_state):
    del partitioned_project_state
    workspace = tmp_path / "seed-workspace"
    workspace.mkdir()
    original_new_id = ss._new_id
    monkeypatch.setattr(ss, "_new_id", lambda: "s1")
    assert ss.create_session(
        "Entelecheia",
        title="test",
        working_dir=str(workspace),
    ) == "s1"
    monkeypatch.setattr(ss, "_new_id", original_new_id)


def _d(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


def _ingest(title="评审规则"):
    session = ss.get_session("s1")
    source = Path(str(session["working_dir"])) / "fixture.md"
    source.write_text(
        "\n\n".join(f"第{i}节内容" + "文字" * 50 for i in range(1, 4)),
        encoding="utf-8",
    )
    with ss.session_database_scope("s1"):
        ingested = ingest_registered_document_fixture(
            source,
            session_id="s1",
        )
    assert ingested["ok"] is True
    document_id = ingested["doc_id"]
    _d(
        "PATCH",
        f"/api/documents/{document_id}",
        {"session_id": "s1", "summary": "旧摘要", "title": title},
    )
    return document_id


def test_document_list_edit_delete():
    did = _ingest()
    listed = _d("GET", "/api/documents?session_id=s1")
    assert [d["id"] for d in listed["documents"]] == [did]
    # 编辑摘要
    r = _d(
        "PATCH",
        f"/api/documents/{did}",
        {"session_id": "s1", "summary": "华为竞赛评审维度"},
    )
    assert r["document"]["summary"] == "华为竞赛评审维度"
    # 删除
    assert _d("DELETE", f"/api/documents/{did}?session_id=s1")["deleted"]
    with pytest.raises(service.ApiError) as ei:
        _d("GET", f"/api/documents/{did}?session_id=s1")
    assert ei.value.code == "DOCUMENT_NOT_FOUND"


def test_document_delete_targets_the_current_active_retrieval_generation(monkeypatch):
    did = _ingest()
    calls: list[str] = []
    provider_calls: list[tuple[str | None, str | None]] = []
    original_remove = docstore.remove
    original_provider = (
        workspace_documents._document_retrieval_cleanup_data_version_id
    )
    with ss.session_database_scope("s1"):
        catalog = SqliteRetrievalCatalog()
        versions = catalog.list_data_versions()
        active = next(
            (
                version
                for version in versions
                if version.role is RetrievalDataVersionRole.ACTIVE
            ),
            None,
        )
        if active is None:
            staging = [
                version
                for version in versions
                if version.role is RetrievalDataVersionRole.STAGING
            ]
            if not staging:
                staging = [
                    catalog.create_data_version(
                        version_id="document-delete-current",
                        fingerprint="document-delete-current",
                        role=RetrievalDataVersionRole.STAGING,
                        state=RetrievalDataVersionState.BUILDING,
                    )
                ]
            assert len(staging) == 1
            ready = staging[0]
            if ready.state is RetrievalDataVersionState.BUILDING:
                ready = catalog.mark_data_version_ready(ready.id)
            assert ready.state is RetrievalDataVersionState.READY
            active = catalog.activate_data_version(ready.id)
        active_version_id = active.id

    def select(conn, staging_candidate):
        selected = original_provider(conn, staging_candidate)
        provider_calls.append((staging_candidate, selected))
        return selected

    monkeypatch.setattr(
        workspace_documents,
        "_document_retrieval_cleanup_data_version_id",
        select,
    )

    def remove(
        document_id: str,
        *,
        retrieval_data_version: str | None = None,
        retrieval_data_version_provider=None,
        document_index_port=None,
    ):
        calls.append(document_id)
        return original_remove(
            document_id,
            retrieval_data_version=retrieval_data_version,
            retrieval_data_version_provider=retrieval_data_version_provider,
            document_index_port=document_index_port,
        )

    monkeypatch.setattr(docstore, "remove", remove)

    assert _d("DELETE", f"/api/documents/{did}?session_id=s1")["deleted"] is True
    assert calls == [did]
    assert provider_calls == [(None, active_version_id)]


def test_document_cleanup_reopens_its_unpublished_ready_generation(
    tmp_path,
    monkeypatch,
):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="document-staging-v1",
        fingerprint="document-generation-v1",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.BUILDING,
    )
    catalog.mark_data_version_ready("document-staging-v1")
    monkeypatch.setattr(
        "personagraph.retrieval.sqlite_store.SqliteRetrievalCatalog",
        lambda: catalog,
    )

    with catalog.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        selected = workspace_documents._document_retrieval_cleanup_data_version_id(
            conn,
            "document-staging-v1",
        )

    assert selected == "document-staging-v1"
    reopened = catalog.get_data_version(selected)
    assert reopened is not None
    assert reopened.role is RetrievalDataVersionRole.STAGING
    assert reopened.state is RetrievalDataVersionState.BUILDING


def test_document_patch_requires_field():
    did = _ingest()
    with pytest.raises(service.ApiError) as ei:
        _d("PATCH", f"/api/documents/{did}", {"session_id": "s1"})
    assert ei.value.code == "NO_EDITABLE_FIELDS"


def test_document_ingest_error_carries_actionable_hint():
    """持久收录拒绝必须带可操作提示，而非只有机器 reason。"""
    assert "找不到" in _ingest_failure_hint("not_a_file")
    assert "格式" in _ingest_failure_hint("unsupported_format")
    assert "安全策略" in _ingest_failure_hint("denied_sensitive_path")
    assert _ingest_failure_hint(None)  # 兜底非空
