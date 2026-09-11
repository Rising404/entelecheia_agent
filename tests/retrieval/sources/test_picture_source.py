from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import sqlite3

import pytest

from personagraph.retrieval.contracts import (
    SourceAvailability,
    SourceFilter,
    SourceType,
)
from personagraph.retrieval.lifecycle.outbox import (
    RetrievalUpdateEvent,
    RetrievalUpdateKind,
    SqliteRetrievalOutbox,
)
from personagraph.retrieval.sources.identity import (
    picture_observation_ref_and_content,
)
from personagraph.retrieval.sources.picture import (
    PictureFileAccessDecision,
    PictureObservationSourceAdapter,
)
from personagraph.retrieval.sources.picture_publication import (
    PictureObservationOutboxPublisher,
    PictureObservationPublicationUnavailable,
)
from personagraph.workspace.pictures.observations import (
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationRepository,
    PictureObservationService,
    PictureObservationWindowPolicy,
)
from personagraph.workspace.pictures.storage.schema import initialize_picture_schema


NOW = "2026-09-04T12:00:00+00:00"


@dataclass
class _MutablePictureFileAccessAuthority:
    allowed: bool = True
    authority_snapshot_id: str = "picture-access-v1"
    reason_code: str = "picture_file_access_revoked"
    failure: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []

    def authorize(
        self,
        *,
        session_id: str,
        file_id: str,
        file_version_id: str,
        picture_id: str | None,
    ) -> PictureFileAccessDecision:
        self.calls.append((session_id, file_id, file_version_id, picture_id))
        if self.failure is not None:
            raise self.failure
        return PictureFileAccessDecision(
            session_id=session_id,
            file_id=file_id,
            file_version_id=file_version_id,
            picture_id=picture_id,
            allowed=self.allowed,
            authority_snapshot_id=(
                self.authority_snapshot_id if self.allowed else None
            ),
            reason_code=None if self.allowed else self.reason_code,
        )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, current_version_id TEXT)")
    conn.execute(
        "CREATE TABLE file_versions ("
        "id TEXT PRIMARY KEY, file_id TEXT NOT NULL, content_sha256 TEXT NOT NULL, "
        "UNIQUE(id, file_id))"
    )
    conn.execute("BEGIN IMMEDIATE")
    initialize_picture_schema(conn)
    conn.commit()
    SqliteRetrievalOutbox().initialize(conn)
    _seed_picture(conn)
    return conn


def _seed_picture(conn: sqlite3.Connection) -> None:
    source_locator = json.dumps(
        {
            "kind": "embedded_asset",
            "payload": {"private_locator": "must-not-enter-citation" * 20},
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    unit_locator = json.dumps(
        {"kind": "full", "payload": {}},
        separators=(",", ":"),
        sort_keys=True,
    )
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('file-1', 'version-1')"
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) VALUES (?, ?, ?)",
        ("version-1", "file-1", _sha256("file-v1")),
    )
    conn.execute(
        "INSERT INTO pictures "
        "(picture_id, file_id, file_version_id, source_kind, source_locator_json, "
        "source_locator_sha256, source_content_sha256, source_media_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "picture-1",
            "file-1",
            "version-1",
            "embedded_asset",
            source_locator,
            _sha256(source_locator),
            _sha256("embedded-image"),
            "image/png",
            NOW,
        ),
    )
    conn.execute(
        "INSERT INTO picture_units "
        "(picture_unit_id, picture_id, unit_kind, unit_locator_json, "
        "unit_locator_sha256, producer_fingerprint, parent_picture_unit_id, "
        "pixel_sha256, media_type, width, height, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)",
        (
            "unit-1",
            "picture-1",
            "full",
            unit_locator,
            _sha256(unit_locator),
            "test-raster-v1",
            _sha256("pixels"),
            "image/png",
            20,
            10,
            NOW,
        ),
    )
    conn.commit()


def _draft(ordinal: int, text: str) -> PictureObservationDraft:
    return PictureObservationDraft(
        picture_id="picture-1",
        picture_unit_id="unit-1",
        logical_invocation_id=f"call-{ordinal}",
        request_ordinal=0,
        modality=PictureObservationModality.VLM,
        purpose="semantic-description",
        kind="caption",
        text=text,
        uncertainty=0.1,
        processor_fingerprint="test-vlm-v1",
        prompt_fingerprint="prompt-v1",
    )


def _commit(
    conn: sqlite3.Connection,
    service: PictureObservationService,
    draft: PictureObservationDraft,
):
    conn.execute("BEGIN IMMEDIATE")
    commit = service.commit_in_transaction(conn, draft, created_at=NOW)
    conn.commit()
    return commit


def _adapter(
    conn: sqlite3.Connection,
    *,
    policy: PictureObservationWindowPolicy,
    authority: _MutablePictureFileAccessAuthority | None = None,
) -> PictureObservationSourceAdapter:
    return PictureObservationSourceAdapter(
        connection_factory=lambda: nullcontext(conn),
        access_authority=(
            authority
            if authority is not None
            else _MutablePictureFileAccessAuthority()
        ),
        window_policy=policy,
    )


def _scope() -> SourceFilter:
    return SourceFilter.from_mapping(
        SourceType.PICTURE,
        {
            "session_id": "session-1",
            "file_id": "file-1",
            "file_version_id": "version-1",
        },
    )


def test_picture_source_only_exposes_nonblank_entries_in_the_active_fifo_window() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=2)
    service = PictureObservationService(PictureObservationRepository(), policy)
    first = _commit(conn, service, _draft(1, "old semantics"))
    second = _commit(conn, service, _draft(2, "  current two  "))
    third = _commit(conn, service, _draft(3, "current three"))
    adapter = _adapter(conn, policy=policy)

    backfill = adapter.list_indexable_units_for_backfill(
        SourceFilter.from_mapping(SourceType.PICTURE)
    )
    assert [unit.content for unit in backfill] == ["current two", "current three"]

    access = adapter.open_retrieval_access(_scope())
    assert access.availability is SourceAvailability.READY
    assert adapter.catalog_source_filters(access) == (
        SourceFilter.from_mapping(
            SourceType.PICTURE,
            {"file_id": "file-1", "file_version_id": "version-1"},
        ),
    )
    assert set(access.source_revision_map) == {
        second.observation.observation_id,
        third.observation.observation_id,
    }

    first_ref, _ = picture_observation_ref_and_content(
        observation_id=first.observation.observation_id,
        payload_sha256=first.observation.payload_sha256,
        text=first.observation.draft.text,
    )
    second_ref, _ = picture_observation_ref_and_content(
        observation_id=second.observation.observation_id,
        payload_sha256=second.observation.payload_sha256,
        text=second.observation.draft.text,
    )
    third_ref, _ = picture_observation_ref_and_content(
        observation_id=third.observation.observation_id,
        payload_sha256=third.observation.payload_sha256,
        text=third.observation.draft.text,
    )
    assert adapter.read_current_for_reconcile(first_ref) is None
    assert adapter.read_current_for_reconcile(second_ref) is not None

    fetched = adapter.fetch_units(access, (first_ref, second_ref, third_ref))
    assert [unit.content for unit in fetched] == ["current two", "current three"]
    assert all("source_locator_json" not in unit.citation for unit in fetched)
    assert all(
        "must-not-enter-citation" not in "".join(unit.citation.values())
        for unit in fetched
    )
    assert all(
        unit.ref.source_revision
        in {second.observation.payload_sha256, third.observation.payload_sha256}
        for unit in fetched
    )
    forged_hash = replace(second_ref, indexed_content_hash="0" * 64)
    forged_revision = replace(second_ref, source_revision="f" * 64)
    assert adapter.fetch_units(access, (forged_hash, forged_revision)) == ()
    assert adapter.read_current_for_reconcile(forged_hash) is None
    assert adapter.read_current_for_reconcile(forged_revision) is None

    bounded = adapter.get_current_index_binding_snapshot(access, maximum_bindings=1)
    assert bounded.source_snapshot_is_current is True
    assert bounded.binding_enumeration_complete is False
    assert len(bounded.bindings) == 1


def test_question_observation_indexes_and_returns_question_with_answer() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=8)
    service = PictureObservationService(PictureObservationRepository(), policy)
    draft = replace(
        _draft(1, "The legend contains four entries."),
        purpose="question",
        question="How many entries are in the legend?",
    )
    commit = _commit(conn, service, draft)
    adapter = _adapter(conn, policy=policy)

    expected_content = (
        "问题：How many entries are in the legend?\n\n"
        "回答：The legend contains four entries."
    )
    ref, content = picture_observation_ref_and_content(
        observation_id=commit.observation.observation_id,
        payload_sha256=commit.observation.payload_sha256,
        question=draft.question,
        text=draft.text,
    )
    assert content == expected_content
    assert ref.indexed_content_hash == _sha256(expected_content)

    access = adapter.open_retrieval_access(_scope())
    fetched = adapter.fetch_units(access, (ref,))
    assert fetched[0].content == expected_content
    assert fetched[0].citation["purpose"] == "question"
    assert fetched[0].citation["question"] == draft.question


def test_picture_transaction_backfill_requires_an_active_caller_transaction() -> None:
    conn = _connection()
    adapter = _adapter(
        conn,
        policy=PictureObservationWindowPolicy(max_active_entries=8),
    )

    with pytest.raises(ValueError, match="active transaction"):
        adapter.list_indexable_units_for_backfill_in_transaction(
            conn,
            SourceFilter.from_mapping(SourceType.PICTURE),
        )


def test_picture_transaction_backfill_sees_uncommitted_observation_without_reopening() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=8)
    service = PictureObservationService(PictureObservationRepository(), policy)

    def forbidden_connection_factory():
        raise AssertionError("transaction backfill must not open another connection")

    adapter = PictureObservationSourceAdapter(
        connection_factory=forbidden_connection_factory,
        window_policy=policy,
    )
    conn.execute("BEGIN IMMEDIATE")
    service.commit_in_transaction(
        conn,
        _draft(1, "uncommitted current semantics"),
        created_at=NOW,
    )

    units = adapter.list_indexable_units_for_backfill_in_transaction(
        conn,
        SourceFilter.from_mapping(SourceType.PICTURE),
    )

    assert [unit.content for unit in units] == ["uncommitted current semantics"]
    conn.rollback()


def test_picture_source_fails_closed_after_file_version_or_fifo_changes() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=1)
    service = PictureObservationService(PictureObservationRepository(), policy)
    first = _commit(conn, service, _draft(1, "first"))
    adapter = _adapter(conn, policy=policy)
    access = adapter.open_retrieval_access(_scope())
    first_ref, _ = picture_observation_ref_and_content(
        observation_id=first.observation.observation_id,
        payload_sha256=first.observation.payload_sha256,
        text=first.observation.draft.text,
    )

    _commit(conn, service, _draft(2, "second"))
    assert adapter.read_current_for_reconcile(first_ref) is None
    assert (
        adapter.revalidate_retrieval_access(access).reason_code
        == "picture_source_changed_during_retrieval"
    )

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) VALUES (?, ?, ?)",
        ("version-2", "file-1", _sha256("file-v2")),
    )
    conn.execute(
        "UPDATE files SET current_version_id = 'version-2' WHERE id = 'file-1'"
    )
    conn.commit()
    assert adapter.fetch_units(access, (first_ref,)) == ()
    assert adapter.open_retrieval_access(_scope()).reason_code == (
        "picture_file_version_not_current"
    )


def test_picture_publication_is_atomic_idempotent_and_emits_fifo_trash() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=2)
    repository = PictureObservationRepository()
    service = PictureObservationService(repository, policy)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        repository=repository,
        window_policy=policy,
    )

    commits = []
    results = []
    for ordinal in (1, 2, 3):
        conn.execute("BEGIN IMMEDIATE")
        commit = service.commit_in_transaction(
            conn,
            _draft(ordinal, f"semantics {ordinal}"),
            created_at=NOW,
        )
        result = publisher.publish_in_transaction(
            conn,
            commit,
            occurred_at=NOW,
        )
        conn.commit()
        commits.append(commit)
        results.append(result)

    assert [len(result.publication_ids) for result in results] == [1, 1, 2]
    rows = conn.execute(
        "SELECT kind, source_unit_id, source_revision, indexed_content_hash "
        "FROM retrieval_update_outbox ORDER BY event_id"
    ).fetchall()
    assert [str(row[0]) for row in rows].count(RetrievalUpdateKind.UPSERT.value) == 3
    assert [str(row[0]) for row in rows].count(RetrievalUpdateKind.TRASH.value) == 1
    first_ref, _ = picture_observation_ref_and_content(
        observation_id=commits[0].observation.observation_id,
        payload_sha256=commits[0].observation.payload_sha256,
        text=commits[0].observation.draft.text,
    )
    trash = next(row for row in rows if row[0] == RetrievalUpdateKind.TRASH.value)
    assert tuple(trash[1:]) == (
        first_ref.source_unit_id,
        first_ref.source_revision,
        first_ref.indexed_content_hash,
    )

    conn.execute("BEGIN IMMEDIATE")
    replay = service.commit_in_transaction(
        conn,
        _draft(3, "semantics 3"),
        created_at=NOW,
    )
    replay_publication = publisher.publish_in_transaction(
        conn,
        replay,
        occurred_at=NOW,
    )
    conn.commit()
    assert replay.inserted is False
    assert replay_publication.publication_ids == ()
    assert conn.execute(
        "SELECT count(*) FROM retrieval_update_outbox"
    ).fetchone()[0] == 4


def test_blank_insert_emits_no_upsert_but_retires_nonblank_fifo_predecessor() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=1)
    repository = PictureObservationRepository()
    service = PictureObservationService(repository, policy)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        repository=repository,
        window_policy=policy,
    )

    conn.execute("BEGIN IMMEDIATE")
    first = service.commit_in_transaction(conn, _draft(1, "visible"), created_at=NOW)
    publisher.publish_in_transaction(conn, first, occurred_at=NOW)
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    blank = service.commit_in_transaction(
        conn,
        replace(_draft(2, "placeholder"), text="  \n"),
        created_at=NOW,
    )
    result = publisher.publish_in_transaction(
        conn,
        blank,
        occurred_at=NOW,
    )
    conn.commit()

    assert len(result.publication_ids) == 1
    event_kind = conn.execute(
        "SELECT kind FROM retrieval_update_outbox WHERE event_id = ?",
        (result.publication_ids[0],),
    ).fetchone()
    assert event_kind is not None
    assert event_kind[0] == RetrievalUpdateKind.TRASH.value
    assert conn.execute(
        "SELECT count(*) FROM retrieval_update_outbox WHERE kind = 'upsert'"
    ).fetchone()[0] == 1
    adapter = _adapter(conn, policy=policy)
    assert adapter.list_indexable_units_for_backfill(
        SourceFilter.from_mapping(SourceType.PICTURE)
    ) == ()
    assert adapter.open_retrieval_access(_scope()).availability is SourceAvailability.EMPTY


def test_picture_observation_and_outbox_pointer_share_the_caller_transaction() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=8)
    repository = PictureObservationRepository()
    service = PictureObservationService(repository, policy)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        repository=repository,
        window_policy=policy,
    )

    conn.execute("BEGIN IMMEDIATE")
    commit = service.commit_in_transaction(conn, _draft(1, "rolled back"), created_at=NOW)
    publication = publisher.publish_in_transaction(
        conn,
        commit,
        occurred_at=NOW,
    )
    assert len(publication.publication_ids) == 1
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM retrieval_update_outbox").fetchone()[0] == 0


def test_missing_publication_target_fails_before_an_indexable_commit_can_escape() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=8)
    repository = PictureObservationRepository()
    service = PictureObservationService(repository, policy)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: None,
        repository=repository,
        window_policy=policy,
    )

    conn.execute("BEGIN IMMEDIATE")
    commit = service.commit_in_transaction(conn, _draft(1, "must stay atomic"), created_at=NOW)
    with pytest.raises(PictureObservationPublicationUnavailable):
        publisher.publish_in_transaction(
            conn,
            commit,
            occurred_at=NOW,
        )
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM retrieval_update_outbox").fetchone()[0] == 0


def test_picture_online_access_requires_authority_and_rechecks_revocation() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=2)
    service = PictureObservationService(PictureObservationRepository(), policy)
    commit = _commit(conn, service, _draft(1, "current semantics"))
    ref, _ = picture_observation_ref_and_content(
        observation_id=commit.observation.observation_id,
        payload_sha256=commit.observation.payload_sha256,
        text=commit.observation.draft.text,
    )
    authority = _MutablePictureFileAccessAuthority()
    adapter = _adapter(conn, policy=policy, authority=authority)

    access = adapter.open_retrieval_access(_scope())
    assert access.availability is SourceAvailability.READY
    assert adapter.fetch_units(access, (ref,))[0].content == "current semantics"
    assert authority.calls == [
        ("session-1", "file-1", "version-1", None),
        ("session-1", "file-1", "version-1", None),
    ]

    authority.allowed = False
    assert adapter.fetch_units(access, (ref,)) == ()
    final = adapter.revalidate_retrieval_access(access)
    assert final.availability is SourceAvailability.BLOCKED
    assert final.reason_code == "picture_source_changed_during_retrieval"

    unbound = PictureObservationSourceAdapter(
        connection_factory=lambda: nullcontext(conn),
        window_policy=policy,
    )
    unavailable = unbound.open_retrieval_access(_scope())
    assert unavailable.availability is SourceAvailability.BLOCKED
    assert unavailable.reason_code == "picture_file_access_authority_missing"
    sessionless = SourceFilter.from_mapping(
        SourceType.PICTURE,
        {"file_id": "file-1", "file_version_id": "version-1"},
    )
    assert adapter.open_retrieval_access(sessionless).reason_code == (
        "picture_scope_invalid"
    )


def test_picture_source_only_maps_open_failures_and_propagates_read_failures() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=2)
    service = PictureObservationService(PictureObservationRepository(), policy)
    commit = _commit(conn, service, _draft(1, "current semantics"))
    ref, _ = picture_observation_ref_and_content(
        observation_id=commit.observation.observation_id,
        payload_sha256=commit.observation.payload_sha256,
        text=commit.observation.draft.text,
    )
    authority = _MutablePictureFileAccessAuthority()
    adapter = _adapter(conn, policy=policy, authority=authority)
    access = adapter.open_retrieval_access(_scope())
    assert access.availability is SourceAvailability.READY

    authority.failure = sqlite3.OperationalError("authority unavailable")
    opened = adapter.open_retrieval_access(_scope())
    assert opened.availability is SourceAvailability.UNAVAILABLE
    assert opened.reason_code == "picture_authority_unavailable"
    with pytest.raises(sqlite3.OperationalError, match="authority unavailable"):
        adapter.fetch_units(access, (ref,))

    authority.failure = None
    conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        adapter.fetch_units(access, (ref,))
    with pytest.raises(sqlite3.ProgrammingError):
        adapter.read_for_index(
            RetrievalUpdateEvent(
                event_id="picture-failure-classification",
                kind=RetrievalUpdateKind.UPSERT,
                ref=ref,
                retrieval_data_version="retrieval-v1",
                occurred_at=NOW,
            )
        )
    with pytest.raises(sqlite3.ProgrammingError):
        adapter.read_current_for_reconcile(ref)
    with pytest.raises(sqlite3.ProgrammingError):
        adapter.list_indexable_units_for_backfill(
            SourceFilter.from_mapping(SourceType.PICTURE)
        )


def test_picture_publication_rejects_fifo_policy_drift() -> None:
    conn = _connection()
    policy = PictureObservationWindowPolicy(max_active_entries=2)
    repository = PictureObservationRepository()
    service = PictureObservationService(repository, policy)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        repository=repository,
    )

    conn.execute("BEGIN IMMEDIATE")
    commit = service.commit_in_transaction(
        conn,
        _draft(1, "must not cross policy"),
        created_at=NOW,
    )
    with pytest.raises(ValueError, match="FIFO policy"):
        publisher.publish_in_transaction(conn, commit, occurred_at=NOW)
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM retrieval_update_outbox").fetchone()[0] == 0


def test_default_fifo_keeps_eight_and_ninth_publication_trashes_first() -> None:
    conn = _connection()
    repository = PictureObservationRepository()
    service = PictureObservationService(repository)
    publisher = PictureObservationOutboxPublisher(
        retrieval_data_version_resolver=lambda _conn: "retrieval-v1",
        repository=repository,
    )
    commits = []
    ninth_publication = None

    for ordinal in range(1, 10):
        conn.execute("BEGIN IMMEDIATE")
        commit = service.commit_in_transaction(
            conn,
            _draft(ordinal, f"semantics {ordinal}"),
            created_at=NOW,
        )
        publication = publisher.publish_in_transaction(
            conn,
            commit,
            occurred_at=NOW,
        )
        conn.commit()
        commits.append(commit)
        if ordinal == 9:
            ninth_publication = publication

    assert len(commits[-1].active_window.observations) == 8
    assert [item.sequence for item in commits[-1].active_window.observations] == list(
        range(2, 10)
    )
    assert ninth_publication is not None
    assert len(ninth_publication.publication_ids) == 2
    first_ref, _ = picture_observation_ref_and_content(
        observation_id=commits[0].observation.observation_id,
        payload_sha256=commits[0].observation.payload_sha256,
        text=commits[0].observation.draft.text,
    )
    trash = conn.execute(
        "SELECT source_unit_id FROM retrieval_update_outbox "
        "WHERE event_id IN (?, ?) AND kind = 'trash'",
        ninth_publication.publication_ids,
    ).fetchone()
    assert trash is not None
    assert trash[0] == first_ref.source_unit_id
