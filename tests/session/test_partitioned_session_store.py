from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import threading

import pytest

from personagraph.session import project_catalog as session_projects
from personagraph.api import router
from personagraph.api.contracts import BodySchema
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.storage import context as document_context
from personagraph.configuration import paths
from personagraph.session.attachments.application import accept_upload
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.documents import application as docstore
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session.local_file_authority import (
    LocalFileAction,
    SqliteSessionFileAuthority,
)
from personagraph.session import catalog as catalog_module
from personagraph.session import service as session_service
from personagraph.session import store
from personagraph.session.catalog import SessionCatalog
from personagraph.session.context import store as context_store
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.contracts import SourceFilter, SourceType
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.session.composition import (
    SESSION_RETRIEVAL_DB_NAME,
    build_session_retrieval_composition,
)
from personagraph.retrieval.sources.session.lifecycle import (
    ensure_committed_session_pair_retrieval_ready,
)
from tests.helpers.session_records import append_test_turn


@pytest.fixture
def partitioned_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "var"
    state_dir.mkdir()
    shared_catalog = state_dir / "project_catalog.sqlite"
    retired_shared_path = state_dir / "retired-shared-session.sqlite"
    monkeypatch.setattr(store, "_DEFAULT_DB_PATH", retired_shared_path)
    monkeypatch.setattr(store, "DB_PATH", retired_shared_path)
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        shared_catalog,
    )
    monkeypatch.setattr(paths, "SESSIONS_DIR", state_dir / "sessions")
    monkeypatch.setattr(paths, "PROJECTS_DIR", state_dir / "projects")
    monkeypatch.setattr(session_projects, "DB_PATH", shared_catalog)
    store._INITIALIZED_PATHS.clear()
    return state_dir


def _turn_contents(database_path: Path) -> list[str]:
    with sqlite3.connect(database_path) as connection:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT content FROM session_turns ORDER BY turn_idx"
            ).fetchall()
        ]


def test_create_read_list_and_search_use_two_physical_session_databases(
    partitioned_state,
):
    session_a = store.create_session("Entelecheia", title="Alpha")
    session_b = store.create_session("Entelecheia", title="Beta")
    catalog = SessionCatalog()
    path_a = catalog.session_db_path(session_a)
    path_b = catalog.session_db_path(session_b)

    assert store.DB_PATH == store._DEFAULT_DB_PATH
    assert path_a != path_b
    assert path_a.is_file() and path_b.is_file()
    assert catalog.get_session(session_a)["project_id"] is None
    assert catalog.get_session(session_b)["project_id"] is None

    with store.session_database_scope(session_a):
        append_test_turn(session_a, "user", "alpha-only evidence")
    with store.session_database_scope(session_b):
        append_test_turn(session_b, "user", "beta-only evidence")

    assert _turn_contents(path_a) == ["alpha-only evidence"]
    assert _turn_contents(path_b) == ["beta-only evidence"]
    assert store.get_session(session_a)["title"] == "Alpha"
    assert {row["id"] for row in store.list_sessions()} == {session_a, session_b}
    assert [row["id"] for row in store.search_sessions("alpha-only")] == [session_a]
    assert store.search_sessions("alpha-only", limit=0) == []
    with pytest.raises(store.SessionStoreError, match="session_database_scope"):
        append_test_turn(session_a, "user", "must not use a shared fallback")


def test_context_scopes_isolate_concurrent_session_writes(partitioned_state):
    session_a = store.create_session("Entelecheia", title="A")
    session_b = store.create_session("Entelecheia", title="B")
    barrier = threading.Barrier(2)

    def write(session_id: str, content: str) -> None:
        with store.session_database_scope(session_id):
            barrier.wait(timeout=5)
            append_test_turn(session_id, "user", content)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write, session_a, "only-a")
        second = executor.submit(write, session_b, "only-b")
        first.result(timeout=10)
        second.result(timeout=10)

    catalog = SessionCatalog()
    assert _turn_contents(catalog.session_db_path(session_a)) == ["only-a"]
    assert _turn_contents(catalog.session_db_path(session_b)) == ["only-b"]
    assert store.DB_PATH == store._DEFAULT_DB_PATH


def test_default_file_authority_is_isolated_in_each_session_database(
    partitioned_state,
    tmp_path,
):
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    file_a = workspace_a / "a.txt"
    file_b = workspace_b / "b.txt"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")
    session_a = store.create_session(
        "Entelecheia",
        title="A",
        working_dir=str(workspace_a),
    )
    session_b = store.create_session(
        "Entelecheia",
        title="B",
        working_dir=str(workspace_b),
    )
    catalog = SessionCatalog()
    authority = SqliteSessionFileAuthority()

    with store.session_database_scope(session_a):
        assert authority.path == catalog.session_db_path(session_a)
        assert [grant.session_id for grant in authority.list_grants(session_id=session_a)] == [
            session_a
        ]
        assert authority.authorize(
            session_id=session_a,
            candidate=file_a,
            working_root=workspace_a,
            action=LocalFileAction.READ,
        ).allowed
        with pytest.raises(store.SessionStoreError, match="active session scope"):
            authority.list_grants(session_id=session_b)

    with store.session_database_scope(session_b):
        assert authority.path == catalog.session_db_path(session_b)
        assert [grant.session_id for grant in authority.list_grants(session_id=session_b)] == [
            session_b
        ]
        assert authority.authorize(
            session_id=session_b,
            candidate=file_b,
            working_root=workspace_b,
            action=LocalFileAction.READ,
        ).allowed

    with pytest.raises(store.SessionStoreError, match="session_database_scope"):
        authority.list_grants(session_id=session_a)

    for session_id in (session_a, session_b):
        with sqlite3.connect(catalog.session_db_path(session_id)) as connection:
            assert connection.execute(
                "SELECT DISTINCT session_id FROM session_workspace_read_grants"
            ).fetchall() == [(session_id,)]
    assert not (partitioned_state / "session_file_authorizations.sqlite").exists()


def test_session_context_uses_the_bound_session_database(partitioned_state):
    session_a = store.create_session("Entelecheia", title="A")
    session_b = store.create_session("Entelecheia", title="B")

    with store.session_database_scope(session_a):
        assert context_store.context_revision(session_a) == 0
    with store.session_database_scope(session_b):
        assert context_store.context_revision(session_b) == 0

    assert not store.DB_PATH.exists()


def test_router_scopes_handlers_from_path_body_and_query_session_ids(
    partitioned_state,
    monkeypatch,
):
    session_a = store.create_session("Entelecheia", title="A")
    session_b = store.create_session("Entelecheia", title="B")
    session_c = store.create_session("Entelecheia", title="C")

    path_route = router.Route(
        "POST",
        ("api", "probe", "{session_id}"),
        lambda params, _body, _query: {
            "turn_idx": append_test_turn(params["session_id"], "user", "path-bound")
        },
    )
    monkeypatch.setattr(router, "ROUTES", (path_route,))
    router.dispatch_response("POST", f"/api/probe/{session_a}", {})

    body_route = router.Route(
        "POST",
        ("api", "probe"),
        lambda _params, body, _query: {
            "turn_idx": append_test_turn(body["session_id"], "user", "body-bound")
        },
        BodySchema({"session_id": str}),
    )
    monkeypatch.setattr(router, "ROUTES", (body_route,))
    router.dispatch_response("POST", "/api/probe", {"session_id": session_b})

    query_route = router.Route(
        "GET",
        ("api", "probe"),
        lambda _params, _body, query: {
            "turn_idx": append_test_turn(query["session_id"], "user", "query-bound")
        },
    )
    monkeypatch.setattr(router, "ROUTES", (query_route,))
    router.dispatch_response("GET", f"/api/probe?session_id={session_c}", {})

    catalog = SessionCatalog()
    assert _turn_contents(catalog.session_db_path(session_a)) == ["path-bound"]
    assert _turn_contents(catalog.session_db_path(session_b)) == ["body-bound"]
    assert _turn_contents(catalog.session_db_path(session_c)) == ["query-bound"]


def test_router_rejects_disagreeing_body_and_query_session_ids(
    partitioned_state,
    monkeypatch,
):
    session_a = store.create_session("Entelecheia", title="A")
    session_b = store.create_session("Entelecheia", title="B")
    route = router.Route(
        "POST",
        ("api", "probe"),
        lambda _params, _body, _query: {"ok": True},
        BodySchema({"session_id": str}),
    )
    monkeypatch.setattr(router, "ROUTES", (route,))

    with pytest.raises(router.service.ApiError) as exc:
        router.dispatch_response(
            "POST",
            f"/api/probe?session_id={session_b}",
            {"session_id": session_a},
        )

    assert getattr(exc.value, "code", None) == "SESSION_ID_MISMATCH"


def test_router_scope_binds_catalog_project_documents_and_is_reentrant(
    partitioned_state,
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    session_id = store.create_session(
        "Entelecheia",
        title="Project session",
        working_dir=str(workspace),
    )
    observed = []

    def handler(params, _body, _query):
        database = document_context.current()
        observed.append(database)
    # get_session 会经过同一会话的嵌套作用域；它必须复用两个活跃 ContextVar，
    # 而不是遮蔽它们。
        return {"session_id": store.get_session(params["session_id"])["id"]}

    route = router.Route(
        "GET",
        ("api", "probe", "{session_id}"),
        handler,
    )
    monkeypatch.setattr(router, "ROUTES", (route,))

    assert router.dispatch_response("GET", f"/api/probe/{session_id}", {}).payload == {
        "session_id": session_id
    }
    assert observed[0] is not None
    assert observed[0].project_id == SessionCatalog().get_session(session_id)["project_id"]
    assert observed[0].project_root == workspace.resolve()
    assert observed[0].db_path.parent.parent == partitioned_state / "projects"
    assert document_context.current() is None


def test_projectless_session_scope_has_no_project_document_binding(partitioned_state):
    session_id = store.create_session("Entelecheia", title="Projectless")

    with store.session_database_scope(session_id):
        assert document_context.current() is None

    assert document_context.current() is None


def test_workspace_creation_records_project_locator_and_local_identity(
    partitioned_state,
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Authority:
        def grant_workspace_root(self, **_kwargs):
            return SimpleNamespace(grant_id="grant-1")

        def revoke(self, **_kwargs):
            return 1

    monkeypatch.setattr(store, "_workspace_file_authority", lambda: Authority())

    session_id = store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
        title="Workspace",
    )

    project = session_projects.remember(str(workspace))
    catalog_record = SessionCatalog().get_session(session_id)
    local_record = store.get_session(session_id)
    assert catalog_record["project_id"] == project.project_id
    assert local_record["project_id"] == project.project_id
    assert local_record["working_dir"] == str(workspace.resolve())


def test_partitioned_lifecycle_and_folder_metadata_stay_mirrored(
    partitioned_state,
):
    source_folder = store.create_folder("Source")
    target_folder = store.create_folder("Target")
    session_id = store.create_session(
        "Entelecheia",
        title="Before",
        folder_id=source_folder,
    )

    assert store.rename_session(session_id, "After") is True
    assert store.move_session(session_id, target_folder) is True
    assert store.archive_session(session_id) is True
    assert store.trash_session(session_id) is True
    assert store.restore_session(session_id) is True

    catalog_record = SessionCatalog().get_session(session_id)
    visible_record = store.get_session(session_id)
    assert catalog_record is not None and visible_record is not None
    assert catalog_record["title"] == visible_record["title"] == "After"
    assert catalog_record["folder_id"] == visible_record["folder_id"] == target_folder
    assert catalog_record["status"] == visible_record["status"] == "archived"
    assert catalog_record["previous_status"] is None

    assert store.unarchive_session(session_id) is True
    assert SessionCatalog().get_session(session_id)["status"] == "active"
    tree = store.folder_tree(status="all")
    by_id = {row["id"]: row for row in tree}
    assert by_id[source_folder]["session_count"] == 0
    assert by_id[target_folder]["session_count"] == 1


def test_product_trash_cleans_session_index_and_restore_can_rebuild(
    partitioned_state,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PERSONAGRAPH_RETRIEVAL_PROFILE", "lexical")
    workspace = tmp_path / "retrieval-lifecycle-project"
    workspace.mkdir()
    session_id = store.create_session(
        "Entelecheia",
        title="Retrieval lifecycle",
        working_dir=str(workspace),
    )
    with store.session_database_scope(session_id):
        accepted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id="retrieval-lifecycle-turn",
            source="runtime_test",
            user_text="Remember the ochre key.",
        )
        finalized = store.finalize_turn_execution(
            session_id=session_id,
            turn_id=str(accepted["turn"]["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(
                accepted["window"]["state_version"]  # type: ignore[index]
            ),
            processing_level="L0",
            assistant_content="The ochre key is in drawer six.",
            post_commit_job_kinds=(),
        )
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=str(accepted["turn"]["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(
                finalized["window"]["state_version"]  # type: ignore[index]
            ),
        )
        accepted_second = store.accept_turn_execution(
            session_id=session_id,
            client_request_id="retrieval-lifecycle-turn-2",
            source="runtime_test",
            user_text="Remember the teal key too.",
        )
        finalized_second = store.finalize_turn_execution(
            session_id=session_id,
            turn_id=str(accepted_second["turn"]["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(
                accepted_second["window"]["state_version"]  # type: ignore[index]
            ),
            processing_level="L0",
            assistant_content="The teal key is in drawer seven.",
            post_commit_job_kinds=(),
        )
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=str(accepted_second["turn"]["turn_id"]),  # type: ignore[index]
            expected_window_revision=int(
                finalized_second["window"]["state_version"]  # type: ignore[index]
            ),
        )
        composition = build_session_retrieval_composition(
            profile=DocumentRetrievalProfile.lexical(),
            encoder=DeterministicLexicalEncoder(),
            reranker=None,
            store=store,
        )
        pair = store.list_committed_turn_pairs(session_id, limit=1)[0]
        ensure_committed_session_pair_retrieval_ready(composition, pair=pair)
        retrieval_db_path = composition.foundation.catalog.db_path
        assert retrieval_db_path.name == SESSION_RETRIEVAL_DB_NAME

    assert session_service.trash_session_fully(session_id) is True
    assert store.get_session(session_id)["status"] == "trashed"
    assert composition.foundation.catalog.list_stored_units() == ()
    assert session_service.restore_session_fully(session_id) is True

    with store.session_database_scope(session_id):
        restored_composition = build_session_retrieval_composition(
            retrieval_db_path=retrieval_db_path,
            profile=DocumentRetrievalProfile.lexical(),
            encoder=DeterministicLexicalEncoder(),
            reranker=None,
            store=store,
        )
        restored_pair = store.list_committed_turn_pairs(session_id, limit=1)[0]
        ensure_committed_session_pair_retrieval_ready(
            restored_composition,
            pair=restored_pair,
        )
        indexed_turn_count = (
            restored_composition.foundation.catalog.active_ready_distinct_scope_value_count(
                data_version_id=restored_composition.generation_spec.version_id,
                source_filter=SourceFilter.from_mapping(
                    SourceType.CURRENT_SESSION,
                    {"session_id": session_id},
                ),
                scope_key="assistant_turn_idx",
            )
        )
        assert indexed_turn_count == 2


def test_restore_remains_successful_when_derived_rebuild_is_temporarily_unavailable(
    partitioned_state,
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "restore-best-effort-project"
    workspace.mkdir()
    session_id = store.create_session(
        "Entelecheia",
        title="Restore best effort",
        working_dir=str(workspace),
    )
    assert session_service.trash_session_fully(session_id) is True

    def fail_rebuild(_session_id: str):
        raise RuntimeError("simulated encoder outage")

    monkeypatch.setattr(
        session_service,
        "_rebuild_session_retrieval_derived_state",
        fail_rebuild,
    )

    assert session_service.restore_session_fully(session_id) is True
    assert store.get_session(session_id)["status"] == "active"


def test_partitioned_working_directory_has_no_public_rebind_or_unbind_operation(
    partitioned_state,
    tmp_path,
):
    workspace = tmp_path / "bound-project"
    workspace.mkdir()
    session_id = store.create_session(
        "Entelecheia",
        title="Bound at creation",
        working_dir=str(workspace),
    )
    project = session_projects.get_by_path(str(workspace))
    assert project is not None
    catalog_record = SessionCatalog().get_session(session_id)
    visible_record = store.get_session(session_id)
    assert catalog_record["project_id"] == visible_record["project_id"] == project.project_id
    assert visible_record["working_dir"] == str(workspace)
    assert not hasattr(store, "bind_session_workspace_layout")
    assert not hasattr(store, "set_session_working_dir")


def test_partitioned_purge_records_tombstone_and_never_deletes_project_files(
    partitioned_state,
    tmp_path,
):
    workspace = tmp_path / "project-files"
    workspace.mkdir()
    retained_file = workspace / "retained.txt"
    retained_file.write_text("project-owned", encoding="utf-8")
    session_id = store.create_session(
        "Entelecheia",
        title="Disposable session",
        working_dir=str(workspace),
    )
    catalog = SessionCatalog()
    session_database = catalog.session_db_path(session_id)
    assert session_database.is_file()

    assert router.dispatch_response(
        "DELETE", f"/api/sessions/{session_id}", {}
    ).payload == {
        "purged": True,
        "session_id": session_id,
    }

    assert store.get_session(session_id) is None
    assert catalog.get_session(session_id) is None
    tombstone = catalog.get_session_purge_tombstone(session_id)
    assert tombstone is not None
    assert tombstone["state"] == "completed"
    assert tombstone["db_path"] == f"sessions/{session_id}/session.sqlite"
    assert not session_database.parent.exists()
    assert retained_file.read_text(encoding="utf-8") == "project-owned"


def test_document_mounts_and_snapshots_are_session_local_but_documents_are_shared(
    partitioned_state,
    tmp_path,
):
    workspace = tmp_path / "shared-document-project"
    workspace.mkdir()
    source = workspace / "shared.txt"
    source.write_text("shared project evidence", encoding="utf-8")
    session_a = store.create_session(
        "Entelecheia",
        title="A",
        working_dir=str(workspace),
    )
    session_b = store.create_session(
        "Entelecheia",
        title="B",
        working_dir=str(workspace),
    )
    project = session_projects.get_by_path(str(workspace))
    assert project is not None

    with store.session_database_scope(session_a):
        project_documents = document_context.current()
        assert project_documents is not None
        registered = WorkspaceFileAuthority(project_documents).register_path(
            "shared.txt",
            source=FileSource.WORKSPACE_EXISTING,
        )
        document = docstore.ingest(
            str(source.resolve()),
            source.stem,
            "text/plain",
            [{"content": "shared project evidence", "loc": "L1"}],
            session_id=session_a,
            file_id=registered.file.file_id,
            file_version_id=registered.version.file_version_id,
            source_fingerprint=fingerprint_file(source),
        )
        snapshot_a = docstore.preflight_mounted_documents(session_a)
        assert snapshot_a["ok"] is True

    with store.session_database_scope(session_b):
        assert docstore.mounted_docs(session_b) == []
        remounted = docstore.ingest(
            str(source.resolve()),
            source.stem,
            "text/plain",
            [{"content": "shared project evidence", "loc": "L1"}],
            session_id=session_b,
            file_id=registered.file.file_id,
            file_version_id=registered.version.file_version_id,
            source_fingerprint=fingerprint_file(source),
        )
        assert remounted["doc_id"] == document["doc_id"]
        assert docstore.get_retrieval_snapshot(
            snapshot_a["snapshot_id"],
            session_b,
        ) is None

    catalog = SessionCatalog()
    with sqlite3.connect(project.documents_db_path) as connection:
        project_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE id=?",
            (document["doc_id"],),
        ).fetchone()[0] == 1
    assert "doc_mounts" not in project_tables
    assert "doc_retrieval_snapshots" not in project_tables

    for session_id in (session_a, session_b):
        with sqlite3.connect(catalog.session_db_path(session_id)) as connection:
            assert connection.execute(
                "SELECT doc_id FROM doc_mounts WHERE session_id=?",
                (session_id,),
            ).fetchone()[0] == document["doc_id"]
    with sqlite3.connect(catalog.session_db_path(session_a)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM doc_retrieval_snapshots WHERE session_id=?",
            (session_a,),
        ).fetchone()[0] == 1
    with sqlite3.connect(catalog.session_db_path(session_b)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM doc_retrieval_snapshots WHERE session_id=?",
            (session_b,),
        ).fetchone()[0] == 0

    assert store.purge_session(session_a) is True
    assert not catalog.session_db_path(session_a).exists()
    with store.session_database_scope(session_b):
        assert [item["id"] for item in docstore.mounted_docs(session_b)] == [
            document["doc_id"]
        ]
    with sqlite3.connect(project.documents_db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE id=?",
            (document["doc_id"],),
        ).fetchone()[0] == 1


def test_partitioned_purge_failure_leaves_a_fail_closed_retryable_tombstone(
    partitioned_state,
    monkeypatch,
):
    session_id = store.create_session("Entelecheia", title="Pending purge")
    catalog = SessionCatalog()
    session_database = catalog.session_db_path(session_id)

    def fail_removal(_path):
        raise OSError("simulated partition removal failure")

    monkeypatch.setattr(store.shutil, "rmtree", fail_removal)

    with pytest.raises(OSError, match="simulated"):
        store.purge_session(session_id)

    assert catalog.get_session(session_id) is None
    assert all(row["id"] != session_id for row in catalog.list_sessions())
    tombstone = catalog.get_session_purge_tombstone(session_id)
    assert tombstone is not None and tombstone["state"] == "pending"
    assert session_database.is_file()


def test_current_turn_upload_has_one_file_authority_and_session_local_order(
    partitioned_state,
    tmp_path,
):
    workspace = tmp_path / "project-with-upload"
    workspace.mkdir()
    (workspace / "existing.txt").write_text(
        "existing project evidence",
        encoding="utf-8",
    )
    session_id = store.create_session(
        "Entelecheia",
        title="Upload authority",
        working_dir=str(workspace),
    )

    with store.session_database_scope(session_id):
        upload = accept_upload(
            session_id=session_id,
            raw_name="brief.txt",
            declared_media_type="text/plain",
            payload=b"current turn upload evidence",
            store=store,
        )
        accepted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id="upload-authority-request",
            source="runtime_test",
            user_text="Read both files",
            attachment_ids=(upload.attachment_id,),
            lease_owner="test-host",
        )
        turn_id = str(accepted["turn"]["turn_id"])
        runtime = build_l1_tool_runtime(
            session_id,
            turn_id=turn_id,
            execution_features={
                "file_retrieval_write_enabled": True,
                "file_retrieval_read_enabled": True,
                "history_retrieval_write_enabled": False,
                "history_retrieval_read_enabled": False,
                "l1_retrieval_tools_enabled": True,
            },
        )
        later_runtime = build_l1_tool_runtime(
            session_id,
            execution_features={
                "file_retrieval_write_enabled": True,
                "file_retrieval_read_enabled": True,
                "history_retrieval_write_enabled": False,
                "history_retrieval_read_enabled": False,
                "l1_retrieval_tools_enabled": True,
            },
        )

    assert runtime.workspace_corpus_authority is not None
    assert not hasattr(runtime.workspace_corpus_authority, 'candidates')
    assert len(runtime.attachment_file_catalog) == 1
    assert runtime.attachment_file_catalog[0]['file_id'] == upload.attachment_id
    assert later_runtime.attachment_file_catalog == ()

    catalog = SessionCatalog()
    with sqlite3.connect(catalog.session_db_path(session_id)) as connection:
        connection.row_factory = sqlite3.Row
        file_ref = connection.execute(
            "SELECT ordinal, project_id, file_id, file_version_id "
            "FROM runtime_turn_input_file_refs"
        ).fetchone()
        attachment = connection.execute(
            "SELECT stored_rel_path, project_id, file_id, file_version_id "
            "FROM session_attachments WHERE attachment_id=?",
            (upload.attachment_id,),
        ).fetchone()
    assert file_ref is not None and attachment is not None
    assert int(file_ref["ordinal"]) == 0
    assert tuple(file_ref[key] for key in ("project_id", "file_id", "file_version_id")) == tuple(
        attachment[key] for key in ("project_id", "file_id", "file_version_id")
    )
    assert str(attachment["stored_rel_path"]).startswith("附件/")
    assert runtime.attachment_file_catalog[0]["file_version_id"] == file_ref["file_version_id"]

    project = session_projects.get_by_path(str(workspace))
    assert project is not None
    with sqlite3.connect(project.documents_db_path) as connection:
        file_record = connection.execute(
            "SELECT origin, relative_path FROM files WHERE id=?",
            (upload.attachment_id,),
        ).fetchone()
    assert file_record == ("user_upload", attachment["stored_rel_path"])
