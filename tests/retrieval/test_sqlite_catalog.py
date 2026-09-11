from __future__ import annotations

from threading import Event, Thread

import pytest

from personagraph.retrieval.contracts import RetrievalStatus, RetrievalUnit, SourceFilter, SourceType, SourceUnitRef
from personagraph.retrieval.sqlite_store import (
    RetrievalCatalogError,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
    UnitIndexState,
)


def _catalog(tmp_path) -> SqliteRetrievalCatalog:
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="v1",
        fingerprint="test-fingerprint-v1",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    return catalog


def _unit(
    *,
    revision: str = "r1",
    content_hash: str = "h1",
    data_version_id: str = "v1",
) -> RetrievalUnit:
    ref = SourceUnitRef(SourceType.CURRENT_SESSION, "turn-pair-1", revision, content_hash)
    return RetrievalUnit(
        ref=ref,
        retrieval_data_version=data_version_id,
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
    )


def test_catalog_is_pointer_only_and_only_publishes_ready_active_units(tmp_path):
    catalog = _catalog(tmp_path)
    stored = catalog.upsert_pending_unit(_unit())

    assert stored.index_state is UnitIndexState.PENDING
    assert catalog.active_units("v1") == []

    catalog.mark_unit_index_ready(stored.unit_id)
    active = catalog.active_units("v1")
    assert len(active) == 1
    assert active[0].unit.ref.source_unit_id == "turn-pair-1"
    assert active[0].unit.source_filter is not None
    assert active[0].unit.source_filter.as_mapping() == {"session_id": "s1"}

    with catalog.connect() as conn:
        column_names = {row["name"] for row in conn.execute("PRAGMA table_info(retrieval_units)").fetchall()}
    assert "content" not in column_names


def test_catalog_trash_excludes_without_destroying_then_purge_deletes(tmp_path):
    catalog = _catalog(tmp_path)
    stored = catalog.upsert_pending_unit(_unit())
    catalog.mark_unit_index_ready(stored.unit_id)
    assert catalog.active_units("v1")

    catalog.set_unit_retrieval_status(stored.unit.ref, "v1", RetrievalStatus.TRASHED)
    assert catalog.active_units("v1") == []
    trashed = catalog.get_unit(stored.unit.ref, "v1")
    assert trashed is not None
    assert trashed.unit.retrieval_status is RetrievalStatus.TRASHED

    catalog.delete_unit(stored.unit_id)
    assert catalog.get_unit(stored.unit.ref, "v1") is None


def test_ready_staging_can_be_invalidated_before_source_authority_changes(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="document-bootstrap-v1",
        fingerprint="document-generation-v1",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.BUILDING,
    )
    catalog.mark_data_version_ready("document-bootstrap-v1")

    reopened = catalog.invalidate_ready_staging_data_version(
        "document-bootstrap-v1"
    )

    assert reopened.role is RetrievalDataVersionRole.STAGING
    assert reopened.state is RetrievalDataVersionState.BUILDING
    assert catalog.invalidate_ready_staging_data_version(
        "document-bootstrap-v1"
    ) == reopened


def test_generation_cleanup_can_share_a_caller_owned_write_transaction(tmp_path):
    catalog = SqliteRetrievalCatalog(tmp_path / "retrieval.sqlite")
    catalog.initialize()
    catalog.create_data_version(
        version_id="document-bootstrap-v1",
        fingerprint="document-generation-v1",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.BUILDING,
    )
    catalog.mark_data_version_ready("document-bootstrap-v1")

    with catalog.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert catalog.active_data_version_in_transaction(conn) is None
        target = catalog.get_data_version_in_transaction(
            conn,
            "document-bootstrap-v1",
        )
        assert target is not None
        assert target.state is RetrievalDataVersionState.READY
        reopened = catalog.invalidate_ready_staging_data_version_in_transaction(
            conn,
            target.id,
        )
        assert reopened.state is RetrievalDataVersionState.BUILDING

    assert catalog.get_data_version("document-bootstrap-v1") == reopened


def test_transaction_scoped_generation_api_rejects_an_unfenced_connection(tmp_path):
    catalog = _catalog(tmp_path)

    with catalog.connect() as conn:
        with pytest.raises(RuntimeError, match="active caller-owned transaction"):
            catalog.active_data_version_in_transaction(conn)


def test_catalog_new_content_revision_creates_another_immutable_binding(tmp_path):
    catalog = _catalog(tmp_path)
    first = catalog.upsert_pending_unit(_unit())
    second = catalog.upsert_pending_unit(_unit(revision="r2", content_hash="h2"))

    assert first.unit_id != second.unit_id
    assert first.unit.ref.source_revision == "r1"
    assert second.unit.ref.source_revision == "r2"


def test_catalog_checks_exact_ready_source_bindings_without_scanning_units(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    first = catalog.upsert_pending_unit(_unit())
    second = catalog.upsert_pending_unit(_unit(revision="r2", content_hash="h2"))
    catalog.mark_unit_index_ready(first.unit_id)

    # 覆盖探针具有精确指针身份。随着来源集合增长，不得退化成全目录诊断扫描。
    monkeypatch.setattr(
        catalog,
        "active_units",
        lambda data_version_id: (_ for _ in ()).throw(AssertionError("unexpected catalog scan")),
    )
    monkeypatch.setattr(
        catalog,
        "list_stored_units",
        lambda data_version_id=None: (_ for _ in ()).throw(AssertionError("unexpected catalog scan")),
    )

    assert catalog.active_ready_source_binding_keys(
        data_version_id="v1",
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
        keys=(
            ("turn-pair-1", "r1", "h1"),
            ("turn-pair-1", "r2", "h2"),
            ("missing", "r1", "missing-hash"),
        ),
    ) == frozenset({("turn-pair-1", "r1", "h1")})

    catalog.mark_unit_index_ready(second.unit_id)
    assert catalog.active_ready_source_binding_keys(
        data_version_id="v1",
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
        keys=(("turn-pair-1", "r1", "h1"), ("turn-pair-1", "r2", "h2")),
    ) == frozenset({("turn-pair-1", "r1", "h1"), ("turn-pair-1", "r2", "h2")})

    catalog.set_unit_retrieval_status(second.unit.ref, "v1", RetrievalStatus.TRASHED)
    assert catalog.active_ready_source_binding_keys(
        data_version_id="v1",
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
        keys=(("turn-pair-1", "r2", "h2"),),
    ) == frozenset()


def test_catalog_coverage_does_not_accept_the_wrong_indexed_content_hash(tmp_path):
    catalog = _catalog(tmp_path)
    stored = catalog.upsert_pending_unit(_unit(content_hash="actual-hash"))
    catalog.mark_unit_index_ready(stored.unit_id)

    assert catalog.active_ready_source_binding_keys(
        data_version_id="v1",
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "s1"}),
        keys=(("turn-pair-1", "r1", "wrong-hash"),),
    ) == frozenset()
    with pytest.raises(ValueError, match="indexed_content_hash"):
        catalog.active_ready_source_binding_keys(
            data_version_id="v1",
            source_filter=SourceFilter.from_mapping(
                SourceType.CURRENT_SESSION,
                {"session_id": "s1"},
            ),
            keys=(("turn-pair-1", "r1"),),  # type: ignore[arg-type]
        )


def test_catalog_coverage_lookup_rechecks_the_trusted_structural_scope(tmp_path):
    catalog = _catalog(tmp_path)
    stored = catalog.upsert_pending_unit(_unit())
    catalog.mark_unit_index_ready(stored.unit_id)

    assert catalog.active_ready_source_binding_keys(
        data_version_id="v1",
        source_filter=SourceFilter.from_mapping(SourceType.CURRENT_SESSION, {"session_id": "other"}),
        keys=(("turn-pair-1", "r1", "h1"),),
    ) == frozenset()


def test_catalog_upgrades_v3_database_with_the_exact_coverage_lookup_index(tmp_path):
    catalog = _catalog(tmp_path)
    with catalog.connect() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_retrieval_units_source_coverage")
        conn.execute("PRAGMA user_version = 3")

    catalog.initialize()

    with catalog.connect() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(retrieval_units)").fetchall()
        }
    assert version == 4
    assert "idx_retrieval_units_source_coverage" in indexes


def test_ready_staging_version_can_atomically_replace_active_version(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.create_data_version(
        version_id="v2",
        fingerprint="test-fingerprint-v2",
        role=RetrievalDataVersionRole.STAGING,
        state=RetrievalDataVersionState.READY,
    )

    catalog.activate_data_version("v2")

    assert catalog.get_data_version("v2").role is RetrievalDataVersionRole.ACTIVE
    assert catalog.get_data_version("v1").role is RetrievalDataVersionRole.PREVIOUS


def test_data_version_ensure_is_exact_idempotent_and_fingerprint_addressable(tmp_path):
    catalog = _catalog(tmp_path)

    created, replayed = catalog.ensure_staging_data_version(
        version_id="v2",
        fingerprint="generation-v2",
    )
    exact, exact_replayed = catalog.ensure_staging_data_version(
        version_id="v2",
        fingerprint="generation-v2",
    )

    assert replayed is False
    assert exact_replayed is True
    assert exact == created
    assert created.role is RetrievalDataVersionRole.STAGING
    assert created.state is RetrievalDataVersionState.BUILDING
    assert catalog.get_data_version_by_fingerprint("generation-v2") == created


def test_data_version_ensure_rejects_identity_collisions_and_a_second_staging(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")

    with pytest.raises(RetrievalCatalogError, match="version id collision"):
        catalog.ensure_staging_data_version(version_id="v2", fingerprint="other")
    with pytest.raises(RetrievalCatalogError, match="fingerprint collision"):
        catalog.ensure_staging_data_version(version_id="other", fingerprint="generation-v2")
    with pytest.raises(RetrievalCatalogError, match="staging version already exists"):
        catalog.ensure_staging_data_version(version_id="v3", fingerprint="generation-v3")


def test_data_version_publication_only_accepts_staging_building_then_ready(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")

    ready = catalog.mark_data_version_ready("v2")
    assert ready.role is RetrievalDataVersionRole.STAGING
    assert ready.state is RetrievalDataVersionState.READY
    assert catalog.mark_data_version_ready("v2") == ready

    active = catalog.activate_data_version("v2")
    assert active.role is RetrievalDataVersionRole.ACTIVE
    assert active.state is RetrievalDataVersionState.READY
    assert catalog.activate_data_version("v2") == active

    with pytest.raises(RetrievalCatalogError, match="staging building"):
        catalog.mark_data_version_ready("v1")
    with pytest.raises(RetrievalCatalogError, match="staging version"):
        catalog.activate_data_version("v1")


def test_failed_staging_generation_cannot_be_published(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")
    failed = catalog.mark_data_version_failed("v2")
    assert failed.state is RetrievalDataVersionState.FAILED
    assert failed.role is RetrievalDataVersionRole.PREVIOUS
    assert catalog.mark_data_version_failed("v2") == failed

    replacement, replayed = catalog.ensure_staging_data_version(
        version_id="v3",
        fingerprint="generation-v3",
    )
    assert replayed is False
    assert replacement.role is RetrievalDataVersionRole.STAGING

    with pytest.raises(RetrievalCatalogError, match="staging building"):
        catalog.mark_data_version_ready("v2")
    with pytest.raises(RetrievalCatalogError, match="ready staging"):
        catalog.activate_data_version("v2")


def test_catalog_only_accepts_writes_to_building_staging_or_ready_active_generations(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")

    staged = catalog.upsert_pending_unit(
        _unit(revision="staged", content_hash="staged", data_version_id="v2")
    )
    assert staged.unit.retrieval_data_version == "v2"

    catalog.mark_data_version_ready("v2")
    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.upsert_pending_unit(
            _unit(revision="frozen", content_hash="frozen", data_version_id="v2")
        )

    catalog.activate_data_version("v2")
    active = catalog.upsert_pending_unit(
        _unit(revision="active", content_hash="active", data_version_id="v2")
    )
    assert active.unit.retrieval_data_version == "v2"

    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.upsert_pending_unit(
            _unit(revision="previous", content_hash="previous", data_version_id="v1")
        )


def test_unit_upsert_serializes_writable_check_with_data_version_activation(
    tmp_path,
    monkeypatch,
):
    """激活操作不能插入可写检查与单元写入之间。"""

    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")
    catalog.mark_data_version_ready("v2")
    writable_checked = Event()
    release_upsert = Event()
    activation_finished = Event()
    upsert_errors: list[BaseException] = []
    activation_errors: list[BaseException] = []
    original_require = catalog._require_data_version

    def pause_after_writable_check(version_id, *, conn=None):
        if conn is None:
            original_require(version_id)
        else:
            original_require(version_id, conn=conn)
        writable_checked.set()
        if not release_upsert.wait(timeout=5):
            raise AssertionError("upsert test barrier timed out")

    monkeypatch.setattr(catalog, "_require_data_version", pause_after_writable_check)

    def upsert() -> None:
        try:
            catalog.upsert_pending_unit(_unit(revision="raced", content_hash="raced"))
        except BaseException as exc:  # pragma: no cover - 错误会在下方显式暴露
            upsert_errors.append(exc)

    def activate() -> None:
        try:
            catalog.activate_data_version("v2")
        except BaseException as exc:  # pragma: no cover - 错误会在下方显式暴露
            activation_errors.append(exc)
        finally:
            activation_finished.set()

    upsert_thread = Thread(target=upsert)
    activation_thread = Thread(target=activate)
    upsert_thread.start()
    assert writable_checked.wait(timeout=5)
    activation_thread.start()
    try:
        assert not activation_finished.wait(timeout=0.25)
    finally:
        release_upsert.set()
        upsert_thread.join(timeout=5)
        activation_thread.join(timeout=5)

    assert not upsert_thread.is_alive()
    assert not activation_thread.is_alive()
    assert upsert_errors == []
    assert activation_errors == []
    assert activation_finished.is_set()


def test_previous_and_frozen_generations_reject_publication_state_changes(tmp_path):
    catalog = _catalog(tmp_path)
    active_unit = catalog.upsert_pending_unit(_unit())
    catalog.mark_unit_index_ready(active_unit.unit_id)

    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")
    staged_unit = catalog.upsert_pending_unit(
        _unit(revision="staged", content_hash="staged", data_version_id="v2")
    )
    catalog.mark_data_version_ready("v2")

    for state_change in (
        catalog.mark_unit_index_pending,
        catalog.mark_unit_index_ready,
        catalog.mark_unit_index_failed,
    ):
        with pytest.raises(RetrievalCatalogError, match="not writable"):
            state_change(staged_unit.unit_id)
    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.set_unit_retrieval_status(
            staged_unit.unit.ref,
            "v2",
            RetrievalStatus.ACTIVE,
        )

    catalog.activate_data_version("v2")
    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.set_unit_retrieval_status(
            active_unit.unit.ref,
            "v1",
            RetrievalStatus.ACTIVE,
        )
    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.mark_unit_index_failed(active_unit.unit_id)

    # 对旧代次仍允许破坏性清理，使 PURGE 能退役过时方法数据及其指针行。
    trashed = catalog.set_unit_retrieval_status(
        active_unit.unit.ref,
        "v1",
        RetrievalStatus.TRASHED,
    )
    assert trashed is not None
    assert trashed.unit.retrieval_status is RetrievalStatus.TRASHED
    catalog.delete_unit(active_unit.unit_id)
    assert catalog.get_unit(active_unit.unit.ref, "v1") is None


def test_failed_generation_rejects_index_state_changes(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.ensure_staging_data_version(version_id="v2", fingerprint="generation-v2")
    staged_unit = catalog.upsert_pending_unit(
        _unit(revision="failed", content_hash="failed", data_version_id="v2")
    )
    catalog.mark_data_version_failed("v2")

    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.mark_unit_index_failed(staged_unit.unit_id)
    with pytest.raises(RetrievalCatalogError, match="not writable"):
        catalog.set_unit_retrieval_status(
            staged_unit.unit.ref,
            "v2",
            RetrievalStatus.ACTIVE,
        )
