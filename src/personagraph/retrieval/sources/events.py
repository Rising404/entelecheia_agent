"""稳定 Source 生命周期转换的仅含指针 Outbox 写入。"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid

from ...workspace.documents.indexing import (
    DocumentIndexAuditError,
    DocumentIndexCoverageSnapshot,
    DocumentIndexEnqueueRequest,
)
from ...workspace.ingestion.storage.contracts import document_ingest_binding_digest
from ...workspace.storage.context import current as current_project_documents
from ...workspace.pictures.observations.contracts import PictureObservationRecord
from ..lifecycle.outbox import RetrievalUpdateEvent, RetrievalUpdateKind, SqliteRetrievalOutbox
from .identity import (
    mounted_document_chunk_ref_and_content,
    parse_mounted_document_chunk_source_unit_id,
    picture_observation_ref_and_content,
)


class SqliteDocumentIndexPort:
    """把 Workspace 的中性块事件写入共库 Retrieval outbox。"""

    def enqueue_in_transaction(
        self,
        conn: sqlite3.Connection,
        request: DocumentIndexEnqueueRequest,
    ) -> tuple[str, ...]:
        if not conn.in_transaction:
            raise ValueError("document index enqueue requires an active transaction")
        event_kind = RetrievalUpdateKind(request.kind.value)
        event_ids: list[str] = []
        for chunk in request.chunks:
            typed = chunk.producer_chunk_id is not None
            session_ids = (
                (request.typed_identity_session_id or "",)
                if typed
                else request.legacy_session_ids
            )
            for session_id in session_ids:
                event_ids.append(
                    enqueue_mounted_document_chunk_event(
                        conn,
                        kind=event_kind,
                        session_id=session_id,
                        chunk_id=chunk.storage_chunk_id,
                        source_version_id=chunk.source_version_id,
                        content=chunk.content,
                        doc_id=request.document_id if typed else None,
                        producer_chunk_id=chunk.producer_chunk_id,
                        retrieval_data_version=request.data_version_id,
                        occurred_at=request.occurred_at,
                    )
                )
        return tuple(event_ids)

    def validate_job_events_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        event_ids: tuple[str, ...],
        document_id: str,
        document_version_id: str,
        data_version_id: str,
    ) -> None:
        _require_active_transaction(conn)
        for event_id in event_ids:
            _validate_document_job_event(
                conn,
                event_id=event_id,
                document_id=document_id,
                document_version_id=document_version_id,
                data_version_id=data_version_id,
            )

    def inspect_job_coverage_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        event_ids: tuple[str, ...],
        document_id: str,
        document_version_id: str,
        data_version_id: str,
    ) -> DocumentIndexCoverageSnapshot:
        _require_active_transaction(conn)
        self.validate_job_events_in_transaction(
            conn,
            event_ids=event_ids,
            document_id=document_id,
            document_version_id=document_version_id,
            data_version_id=data_version_id,
        )
        bindings = _current_document_bindings(
            conn,
            document_id=document_id,
            document_version_id=document_version_id,
        )
        statuses = []
        for event_id in event_ids:
            row = conn.execute(
                "SELECT status FROM retrieval_update_outbox WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"retrieval outbox event not found: {event_id}")
            statuses.append(str(row["status"]))
        return DocumentIndexCoverageSnapshot(
            binding_digest=document_ingest_binding_digest(bindings),
            binding_count=len(bindings),
            mapped_event_count=len(event_ids),
            applied_event_count=sum(status == "applied" for status in statuses),
        )


def _require_active_transaction(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
        raise ValueError("document index audit requires an active transaction")


def _validate_document_job_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    document_id: str,
    document_version_id: str,
    data_version_id: str,
) -> None:
    event = conn.execute(
        "SELECT kind, source_type, source_unit_id, source_revision, "
        "indexed_content_hash, data_version_id "
        "FROM retrieval_update_outbox WHERE event_id=?",
        (event_id,),
    ).fetchone()
    if event is None:
        raise DocumentIndexAuditError(
            f"retrieval outbox event not found: {event_id}"
        )
    if str(event["source_type"]) != "document":
        raise DocumentIndexAuditError(
            "document ingest jobs may only map document retrieval events"
        )
    if str(event["data_version_id"]) != data_version_id:
        raise DocumentIndexAuditError(
            "retrieval event data_version_id does not match retrieval_data_version"
        )

    source_version_id = str(event["source_revision"])
    version_owner = conn.execute(
        "SELECT 1 FROM document_versions WHERE id=? AND doc_id=?",
        (source_version_id, document_id),
    ).fetchone()
    identity = parse_mounted_document_chunk_source_unit_id(
        str(event["source_unit_id"])
    )
    project_bound = current_project_documents() is not None
    identity_authorized = bool(
        identity is not None
        and (
            identity.is_project_scoped
            or (
                not project_bound
                and conn.execute(
                    "SELECT 1 FROM doc_mounts WHERE doc_id=? AND session_id=?",
                    (document_id, identity.session_id),
                ).fetchone()
                is not None
            )
        )
    )
    if (
        version_owner is None
        or identity is None
        or not identity_authorized
        or (identity.is_typed and identity.doc_id != document_id)
    ):
        raise DocumentIndexAuditError(
            "retrieval event must belong to the same document as its ingest job"
        )

    event_kind = str(event["kind"])
    if event_kind in {"upsert", "restore"} and source_version_id != document_version_id:
        raise DocumentIndexAuditError(
            "upsert and restore events must target the ingest job's current document version"
        )
    if event_kind not in {"upsert", "restore", "purge"}:
        raise DocumentIndexAuditError(
            "document ingest jobs may only map upsert, restore, or purge events"
        )
    if event_kind == "purge":
        return

    if identity.is_typed:
        chunk = conn.execute(
            "SELECT content FROM doc_chunks WHERE doc_id=? AND source_version_id=? "
            "AND producer_chunk_id=?",
            (document_id, source_version_id, identity.producer_chunk_id),
        ).fetchone()
    else:
        chunk = conn.execute(
            "SELECT content FROM doc_chunks WHERE id=? AND doc_id=? "
            "AND source_version_id=?",
            (identity.storage_chunk_id, document_id, source_version_id),
        ).fetchone()
    current_content_hash = (
        hashlib.sha256(str(chunk["content"]).strip().encode("utf-8")).hexdigest()
        if chunk is not None
        else None
    )
    if current_content_hash != str(event["indexed_content_hash"]):
        raise DocumentIndexAuditError(
            "upsert and restore events must identify a real current chunk and hash"
        )


def _current_document_bindings(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    document_version_id: str,
) -> tuple[tuple[str, str, str], ...]:
    current = conn.execute(
        "SELECT current_version_id FROM documents WHERE id=?",
        (document_id,),
    ).fetchone()
    if current is None or str(current["current_version_id"] or "") != document_version_id:
        raise DocumentIndexAuditError("document version is no longer current")
    chunks = conn.execute(
        "SELECT id, producer_chunk_id, content, source_version_id FROM doc_chunks "
        "WHERE doc_id=? AND source_version_id=? ORDER BY seq, id",
        (document_id, document_version_id),
    ).fetchall()
    if not chunks:
        raise DocumentIndexAuditError(
            "document coverage requires at least one usable chunk"
        )

    project_bound = current_project_documents() is not None
    if project_bound:
        bindings = []
        for chunk in chunks:
            if chunk["producer_chunk_id"] is None:
                raise DocumentIndexAuditError(
                    "project document coverage requires typed chunk identities"
                )
            ref, _ = mounted_document_chunk_ref_and_content(
                session_id="project-index",
                chunk_id=str(chunk["id"]),
                source_version_id=str(chunk["source_version_id"]),
                content=str(chunk["content"]),
                doc_id=document_id,
                producer_chunk_id=str(chunk["producer_chunk_id"]),
            )
            bindings.append(
                (
                    ref.source_unit_id,
                    ref.source_revision,
                    ref.indexed_content_hash,
                )
            )
        return tuple(sorted(bindings))

    mounts = conn.execute(
        "SELECT session_id FROM doc_mounts WHERE doc_id=? ORDER BY session_id",
        (document_id,),
    ).fetchall()
    if not mounts:
        raise DocumentIndexAuditError(
            "document coverage requires at least one mounted usable chunk"
        )
    bindings = []
    for chunk in chunks:
        producer_chunk_id = (
            str(chunk["producer_chunk_id"])
            if chunk["producer_chunk_id"] is not None
            else None
        )
        if producer_chunk_id is not None:
            ref, _ = mounted_document_chunk_ref_and_content(
                session_id="project-index",
                chunk_id=str(chunk["id"]),
                source_version_id=str(chunk["source_version_id"]),
                content=str(chunk["content"]),
                doc_id=document_id,
                producer_chunk_id=producer_chunk_id,
            )
            bindings.append(
                (
                    ref.source_unit_id,
                    ref.source_revision,
                    ref.indexed_content_hash,
                )
            )
            continue
        for mount in mounts:
            ref, _ = mounted_document_chunk_ref_and_content(
                session_id=str(mount["session_id"]),
                chunk_id=str(chunk["id"]),
                source_version_id=str(chunk["source_version_id"]),
                content=str(chunk["content"]),
            )
            bindings.append(
                (
                    ref.source_unit_id,
                    ref.source_revision,
                    ref.indexed_content_hash,
                )
            )
    return tuple(sorted(bindings))


def enqueue_mounted_document_chunk_event(
    conn: sqlite3.Connection,
    *,
    kind: RetrievalUpdateKind,
    session_id: str,
    chunk_id: str,
    source_version_id: str,
    content: str,
    doc_id: str | None = None,
    producer_chunk_id: str | None = None,
    retrieval_data_version: str,
    occurred_at: str,
) -> str:
    """在权威事务内追加一个仅含指针的 DocSet 生命周期事件。"""

    ref, _ = mounted_document_chunk_ref_and_content(
        session_id=session_id,
        chunk_id=chunk_id,
        source_version_id=source_version_id,
        content=content,
        doc_id=doc_id,
        producer_chunk_id=producer_chunk_id,
    )
    event = _event_for_ref(
        source_namespace="mounted-document-chunk",
        kind=kind,
        ref=ref,
        retrieval_data_version=retrieval_data_version,
        occurred_at=occurred_at,
    )
    SqliteRetrievalOutbox().enqueue(conn, event)
    return event.event_id


def build_picture_observation_upsert_event(
    *,
    observation: PictureObservationRecord,
    retrieval_data_version: str,
    occurred_at: str,
) -> RetrievalUpdateEvent | None:
    """Project one non-blank immutable observation into a deterministic UPSERT.

    The event remains pointer-only.  In particular, neither observation text nor a
    picture/source locator is copied into the authority-side outbox.
    """

    if not isinstance(observation, PictureObservationRecord):
        raise TypeError("observation must be PictureObservationRecord")
    if not observation.draft.text.strip():
        return None
    ref, _ = picture_observation_ref_and_content(
        observation_id=observation.observation_id,
        payload_sha256=observation.payload_sha256,
        text=observation.draft.text,
        question=observation.draft.question,
    )
    return _event_for_ref(
        source_namespace="picture-observation",
        kind=RetrievalUpdateKind.UPSERT,
        ref=ref,
        retrieval_data_version=retrieval_data_version,
        occurred_at=occurred_at,
    )


def build_picture_observation_trash_event(
    *,
    observation: PictureObservationRecord,
    evicted_by_observation_id: str,
    retrieval_data_version: str,
    occurred_at: str,
) -> RetrievalUpdateEvent | None:
    """Build the FIFO eviction transition for one formerly active observation."""

    if not isinstance(observation, PictureObservationRecord):
        raise TypeError("observation must be PictureObservationRecord")
    if not isinstance(evicted_by_observation_id, str) or not evicted_by_observation_id.strip():
        raise ValueError("evicted_by_observation_id must be non-empty")
    if not observation.draft.text.strip():
        return None
    ref, _ = picture_observation_ref_and_content(
        observation_id=observation.observation_id,
        payload_sha256=observation.payload_sha256,
        text=observation.draft.text,
        question=observation.draft.question,
    )
    return _event_for_ref(
        source_namespace="picture-observation",
        kind=RetrievalUpdateKind.TRASH,
        ref=ref,
        retrieval_data_version=retrieval_data_version,
        occurred_at=occurred_at,
        transition_nonce=f"fifo-evicted-by:{evicted_by_observation_id}",
    )


def _event_for_ref(
    *,
    source_namespace: str,
    kind: RetrievalUpdateKind,
    ref,
    retrieval_data_version: str,
    occurred_at: str,
    transition_nonce: str | None = None,
) -> RetrievalUpdateEvent:
    """创建一个幂等绑定事件或一次唯一生命周期转换。

    为同一不可变内容绑定和数据版本重新入队 UPSERT 属于幂等重放，因此刻意保留 v1
    确定性身份。相比之下，后续 TRASH、RESTORE 或 PURGE 可能在另一次权威状态转换后作用
    于完全相同的绑定。若把它视为重放，派生 Unit 会停留在错误状态。因此普通生命周期
    事件携带新的转换 nonce；拥有稳定转换身份（例如由某条新 FIFO 记录挤出旧记录）的
    调用方可以显式提供 nonce，使同一次转换的事务重试仍然幂等。
    """

    identity_fields = (
        kind.value,
        retrieval_data_version,
        ref.source_type.value,
        ref.source_unit_id,
        ref.source_revision,
        ref.indexed_content_hash,
    )
    if kind is RetrievalUpdateKind.UPSERT:
        material = "\x1f".join((f"retrieval-{source_namespace}-event-v1", *identity_fields))
    else:
        nonce = transition_nonce or uuid.uuid4().hex
        material = "\x1f".join(
            (
                f"retrieval-{source_namespace}-lifecycle-event-v2",
                *identity_fields,
                "transition_nonce",
                nonce,
            )
        )
    return RetrievalUpdateEvent(
        event_id=f"{source_namespace}:{kind.value}:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}",
        kind=kind,
        ref=ref,
        retrieval_data_version=retrieval_data_version,
        occurred_at=occurred_at,
    )
