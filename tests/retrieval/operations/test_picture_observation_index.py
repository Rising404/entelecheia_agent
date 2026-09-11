"""Cold FILE generation activation must follow durable picture outbox work."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import threading
import time

import pytest

from personagraph.retrieval.contracts import SourceFilter, SourceType
from personagraph.retrieval.lifecycle.outbox import OutboxStatus, SqliteRetrievalOutbox
from personagraph.retrieval.lifecycle.sync import SyncApplyStatus, SyncResult
from personagraph.retrieval.operations.document_maintenance import (
    build_document_retrieval_composition,
)
from personagraph.retrieval.operations.picture_index import PictureObservationIndex
from personagraph.retrieval.sources.events import build_picture_observation_upsert_event
from personagraph.retrieval.sources.picture_publication import (
    PictureObservationOutboxPublisher,
)
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
)
from personagraph.workspace.pictures.contracts import (
    PictureSourceLocator,
    PictureUnitLocator,
)
from personagraph.workspace.pictures.observations import (
    PictureObservationStructuredPayload,
)
from personagraph.workspace.pictures.project_publication import (
    ProjectPictureBinding,
    ProjectPictureObservationSpec,
    ProjectPicturePublicationCommand,
    ProjectPicturePublicationService,
    ProjectPictureUnitSpec,
)
from personagraph.workspace.storage.context import bind
from personagraph.workspace.storage.database import DocumentDatabase


NOW = "2026-09-08T12:00:00+00:00"
LATER = "2026-09-08T12:00:06+00:00"


@pytest.fixture
def picture_index(tmp_path):
    database = DocumentDatabase("project-1", tmp_path, tmp_path / "documents.sqlite")
    database.initialize()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO files (id,project_id,relative_path,origin,media_type,"
            "current_version_id,created_at,updated_at) VALUES "
            "('file-1','project-1','uploaded.pdf','user_upload','application/pdf',NULL,?,?)",
            (NOW, NOW),
        )
        conn.execute(
            "INSERT INTO file_versions (id,file_id,version_number,producer,"
            "content_sha256,size_bytes,source_mtime_ns,created_at) VALUES "
            "('version-1','file-1',1,'user_upload',?,100,1,?)",
            ("a" * 64, NOW),
        )
        conn.execute("UPDATE files SET current_version_id='version-1'")
    with bind(database):
        composition = build_document_retrieval_composition()
        index = PictureObservationIndex(
            composition=composition,
            connect_documents=database.open_connection,
        )
        publisher = ProjectPicturePublicationService(
            database=database,
            publication_port=PictureObservationOutboxPublisher(
                retrieval_data_version_resolver=index.resolve_target_in_transaction,
            ),
        )
        yield database, composition, index, publisher


def _command(call_id="call-1"):
    return ProjectPicturePublicationCommand(
        binding=ProjectPictureBinding(
            project_id="project-1",
            file_id="file-1",
            file_version_id="version-1",
            file_content_sha256="a" * 64,
            file_media_type="application/pdf",
        ),
        unit=ProjectPictureUnitSpec(
            source_locator=PictureSourceLocator.document_surface("pdf_page", 2),
            source_content_sha256="a" * 64,
            source_media_type="application/pdf",
            unit_locator=PictureUnitLocator.from_payload(
                "render",
                {"page": 2, "region": "full", "detail": "auto", "bbox": None},
            ),
            producer_fingerprint="raster-1",
            parent_picture_unit_id=None,
            pixel_sha256="b" * 64,
            media_type="image/png",
            width=20,
            height=20,
        ),
        logical_invocation_id=call_id,
        observations=(
            ProjectPictureObservationSpec(
                request_ordinal=0,
                purpose="question",
                question="Which planet is shown?",
                kind="question",
                text="Saturn has visible rings.",
                uncertainty=0.1,
                processor_fingerprint="fake-vlm",
                prompt_fingerprint="question-prompt",
                structured_payload=PictureObservationStructuredPayload.from_payload(
                    contract="test-observation-v1",
                    payload={},
                ),
            ),
        ),
        occurred_at=NOW,
    )


def _ids(publication):
    return tuple(
        item.observation.observation_id for item in publication.observation_commits
    )


def test_cold_pdf_question_activates_only_after_outbox_is_applied(picture_index):
    database, composition, index, publisher = picture_index
    catalog = composition.foundation.catalog
    assert catalog.active_data_version() is None
    target = index.prepare_target()
    assert catalog.get_data_version(target).state is RetrievalDataVersionState.BUILDING
    assert catalog.active_data_version() is None
    published = publisher.publish(_command())
    with database.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 0
        assert (
            conn.execute("SELECT status FROM retrieval_update_outbox").fetchone()[0]
            == "pending"
        )
    active = index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    assert active.role is RetrievalDataVersionRole.ACTIVE
    assert active.state is RetrievalDataVersionState.READY
    assert active.id == target
    with database.connect() as conn:
        assert (
            conn.execute("SELECT status FROM retrieval_update_outbox").fetchone()[0]
            == "applied"
        )
        assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 0
    results = composition.foundation.method_store.search_bm25(
        "Saturn",
        source_filter=SourceFilter.from_mapping(SourceType.PICTURE),
        retrieval_data_version_id=target,
        query_index=0,
        limit=5,
    )
    assert len(results) == 1


def test_picture_waits_for_an_event_owned_by_another_consumer(picture_index):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    outbox = SqliteRetrievalOutbox()
    claimed_at = datetime.now(timezone.utc).isoformat()
    with database.open_connection() as conn:
        (event,) = outbox.claim_due(
            conn, worker_id="background-owner", now=claimed_at, lease_seconds=30, limit=1,
        )
    completed = threading.Event()
    errors = []

    def background():
        try:
            time.sleep(0.08)
            with bind(database):
                applied = composition.foundation.sync_service.apply(event)
            assert applied.status is SyncApplyStatus.APPLIED, applied
            with database.open_connection() as conn:
                outbox.mark_applied(conn, event_id=event.event_id, worker_id="background-owner", now=claimed_at)
                conn.commit()
            completed.set()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=background)
    worker.start()
    try:
        result = index.synchronize(
            _ids(published), worker_id="visual-tool", now=claimed_at,
            deadline_monotonic=time.monotonic() + 5,
        )
        assert result.role is RetrievalDataVersionRole.ACTIVE
    finally:
        worker.join(timeout=5)
        assert errors == []
    assert errors == []
    assert completed.is_set()
    with database.open_connection() as conn:
        row = conn.execute("SELECT status,attempts FROM retrieval_update_outbox").fetchone()
        assert tuple(row) == ("applied", 1)


@pytest.mark.parametrize("mode", ["no_deadline", "timeout", "cancel", "terminal"])
def test_picture_wait_never_steals_a_live_lease_or_ignores_cancellation(picture_index, mode):
    database, _, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    now = datetime.now(timezone.utc).isoformat()
    with database.open_connection() as conn:
        (event,) = SqliteRetrievalOutbox().claim_due(
            conn, worker_id="background-owner", now=now, lease_seconds=30, limit=1,
        )
        if mode == "terminal":
            conn.execute("UPDATE retrieval_update_outbox SET status='terminal_failed' WHERE event_id=?", (event.event_id,))
            conn.commit()
    if mode != "terminal":
        assert index.confirm_ready(_ids(published), now=now) is None

    def cancelled():
        raise RuntimeError("caller_cancelled")

    expected = {
        "no_deadline": "picture_retrieval_outbox_incomplete",
        "timeout": "picture_retrieval_wait_timeout",
        "cancel": "caller_cancelled",
        "terminal": "picture_retrieval_outbox_terminal_failure",
    }[mode]
    with pytest.raises(RuntimeError, match=expected):
        index.synchronize(
            _ids(published), worker_id="foreground", now=now,
            deadline_monotonic=time.monotonic() + 0.03 if mode == "timeout" else None,
            checkpoint=cancelled if mode == "cancel" else None,
        )
    with database.open_connection() as conn:
        assert conn.execute("SELECT attempts FROM retrieval_update_outbox").fetchone()[0] == 1


def test_failed_index_is_not_activated_and_replay_recovers_original_event(
    picture_index, monkeypatch
):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    sync = composition.foundation.sync_service
    original_apply = sync.apply
    monkeypatch.setattr(
        sync,
        "apply",
        lambda event: SyncResult(
            event.event_id,
            SyncApplyStatus.RETRYABLE_FAILED,
            "transient-test-failure",
            failure_stage="index_write",
            safe_error_code="transient_test_failure",
        ),
    )
    with pytest.raises(RuntimeError, match="picture_retrieval_outbox_incomplete"):
        index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    assert composition.foundation.catalog.active_data_version() is None
    replay = publisher.publish(_command())
    assert replay.outbox_publications[0].publication_ids == ()
    monkeypatch.setattr(sync, "apply", original_apply)
    index.synchronize(_ids(replay), worker_id="recovery-index", now=LATER)
    with database.connect() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM picture_observations").fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM retrieval_update_outbox").fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT status FROM retrieval_update_outbox").fetchone()[0]
            == "applied"
        )


def test_foreign_active_configuration_is_refused(picture_index):
    _, composition, index, _ = picture_index
    composition.foundation.catalog.create_data_version(
        version_id="foreign",
        fingerprint="other-encoder",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    with pytest.raises(RuntimeError, match="active_generation_fingerprint_mismatch"):
        index.prepare_target()


def test_prepare_does_not_override_foreign_active_appearing_before_commit(
    picture_index,
):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    composition.foundation.catalog.create_data_version(
        version_id="foreign",
        fingerprint="other-encoder",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
    )
    with pytest.raises(RuntimeError, match="active_generation_fingerprint_mismatch"):
        publisher.publish(_command())
    with database.connect() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM picture_observations").fetchone()[0] == 0
        )


def test_synchronization_does_not_claim_foreign_generation_events(picture_index):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    composition.foundation.catalog.create_data_version(
        version_id="foreign",
        fingerprint="other-encoder",
        role=RetrievalDataVersionRole.PREVIOUS,
        state=RetrievalDataVersionState.READY,
    )
    event = build_picture_observation_upsert_event(
        observation=published.observation_commits[0].observation,
        retrieval_data_version="foreign",
        occurred_at=NOW,
    )
    with database.connect() as conn:
        SqliteRetrievalOutbox().enqueue(conn, event)
    index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    with database.connect() as conn:
        assert (
            SqliteRetrievalOutbox().get_status(conn, event.event_id)
            is OutboxStatus.PENDING
        )
    assert composition.foundation.catalog.list_stored_units("foreign") == ()


def test_missing_outbox_is_not_replaced_by_direct_backfill(picture_index):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    with database.connect() as conn:
        conn.execute("DELETE FROM retrieval_update_outbox")
    with pytest.raises(RuntimeError, match="picture_retrieval_outbox_missing"):
        index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    assert composition.foundation.catalog.active_data_version() is None


def test_same_active_generation_accepts_a_new_question(picture_index):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    first = publisher.publish(_command())
    index.synchronize(_ids(first), worker_id="visual-index", now=NOW)
    assert index.prepare_target() == composition.generation_spec.version_id
    second = publisher.publish(replace(_command("call-2"), occurred_at=LATER))
    index.synchronize(_ids(second), worker_id="visual-index", now=LATER)
    with database.connect() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM picture_observations").fetchone()[0] == 2
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_units WHERE index_state='ready'"
            ).fetchone()[0]
            == 2
        )


def test_ready_receipt_recovery_accepts_an_already_applied_fifo_removal(picture_index):
    database, composition, index, publisher = picture_index
    index.prepare_target()
    first = publisher.publish(_command())
    index.synchronize(_ids(first), worker_id="visual-index", now=NOW)
    for ordinal in range(2, 10):
        published = publisher.publish(_command(f"call-{ordinal}"))
        index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    # Simulate a lost session acknowledgement: Project/outbox completed before
    # the old READY receipt was resumed, and newer calls have evicted its answer.
    resumed = PictureObservationIndex(
        composition=composition,
        connect_documents=database.open_connection,
    )
    resumed.prepare_target()
    resumed.synchronize(_ids(first), worker_id="recovery-index", now=LATER)
    with database.connect() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM picture_observations").fetchone()[0] == 9
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_units WHERE retrieval_status='active'"
            ).fetchone()[0]
            == 8
        )


def test_foreign_active_appearing_before_bootstrap_is_never_replaced(
    picture_index, monkeypatch
):
    _, composition, index, publisher = picture_index
    index.prepare_target()
    published = publisher.publish(_command())
    original = index._authority.publish_bootstrap_generation
    catalog = composition.foundation.catalog

    def concurrent_activation():
        catalog.create_data_version(
            version_id="foreign",
            fingerprint="other-encoder",
            role=RetrievalDataVersionRole.ACTIVE,
            state=RetrievalDataVersionState.READY,
        )
        return original()

    monkeypatch.setattr(index._authority, "publish_bootstrap_generation", concurrent_activation)
    with pytest.raises(RuntimeError, match="active_generation_fingerprint_mismatch"):
        index.synchronize(_ids(published), worker_id="visual-index", now=NOW)
    assert catalog.active_data_version().id == "foreign"
    assert (
        catalog.get_data_version(composition.generation_spec.version_id).role
        is RetrievalDataVersionRole.STAGING
    )


def test_prepare_never_encodes_or_publishes_an_empty_corpus(picture_index, monkeypatch):
    _, composition, index, _ = picture_index

    def forbidden_encode(*args, **kwargs):
        pytest.fail("preparing an index target must not encode anything")

    monkeypatch.setattr(composition.encoder, "encode", forbidden_encode)
    index.prepare_target()
    assert composition.foundation.catalog.active_data_version() is None
    with pytest.raises(ValueError, match="observation_ids"):
        index.synchronize((), worker_id="visual-index", now=NOW)
