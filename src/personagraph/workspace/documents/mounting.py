"""Session 文档挂载、来源时效性与检索快照应用能力。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

from personagraph.configuration.paths import deny_reason
from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    fingerprint_file,
)
from personagraph.workspace.storage.context import (
    connect_current,
    initialize_current,
)

from .contracts import (
    DocumentMountPort,
    MountedDocumentIndexBinding,
    MountedDocumentIndexBindingSnapshot,
    processing_coverage_projection,
    project_document_processing,
)
from .storage import chunks as chunk_repository
from .storage import metadata as metadata_repository


_CURRENT_MOUNT_PORT: ContextVar[DocumentMountPort | None] = ContextVar(
    "personagraph_document_mount_port",
    default=None,
)


class DocumentMountPortRequired(RuntimeError):
    """当前执行作用域没有绑定 Session 文档挂载能力。"""


@contextmanager
def bind_document_mount_port(
    port: DocumentMountPort,
) -> Iterator[DocumentMountPort]:
    if not isinstance(port, DocumentMountPort):
        raise TypeError("port must satisfy DocumentMountPort")
    current = _CURRENT_MOUNT_PORT.get()
    if current is port:
        yield port
        return
    if current is not None:
        raise DocumentMountPortRequired("document mount port is already bound")
    token = _CURRENT_MOUNT_PORT.set(port)
    try:
        yield port
    finally:
        _CURRENT_MOUNT_PORT.reset(token)


def _mount_port() -> DocumentMountPort:
    port = _CURRENT_MOUNT_PORT.get()
    if port is None:
        raise DocumentMountPortRequired(
            "document operations require a bound Session mount port"
        )
    return port


def mounted_document_ids(session_id: str) -> tuple[str, ...]:
    return tuple(
        str(row["doc_id"])
        for row in _mount_port().list_document_mounts(session_id)
    )


def mount_document(document_id: str, session_id: str | None) -> bool:
    if not session_id:
        return False
    return _mount_port().mount_document(document_id, session_id)


def mounted_docs(session_id: str | None) -> list[dict[str, Any]]:
    initialize_current()
    if not session_id:
        return []
    document_ids = mounted_document_ids(session_id)
    if not document_ids:
        return []
    documents_by_id: dict[str, dict[str, Any]] = {}
    with connect_current() as conn:
        for batch in _batches(document_ids, 300):
            documents_by_id.update(
                (
                    str(row["id"]),
                    project_document_processing(dict(row)),
                )
                for row in metadata_repository.get_documents(conn, batch)
            )
    return [
        documents_by_id[document_id]
        for document_id in document_ids
        if document_id in documents_by_id
    ]


def is_mounted(document_id: str, session_id: str | None) -> bool:
    """返回 Session 是否可使用文档；``None`` 表示内部受信读取。"""

    if session_id is None:
        return True
    initialize_current()
    return _mount_port().is_document_mounted(document_id, session_id)


def is_mounted_in_connection(
    conn: sqlite3.Connection,
    document_id: str,
    session_id: str,
) -> bool:
    """保留调用方 Document 事务，同时检查独立提交的 Session 挂载。"""

    del conn
    return _mount_port().is_document_mounted(document_id, session_id)


def detach(document_id: str, session_id: str) -> bool:
    initialize_current()
    port = _mount_port()
    if not port.is_document_mounted(document_id, session_id):
        return False
    return port.unmount_document(document_id, session_id)


def detach_from_current_session(document_id: str) -> bool:
    """清除当前 Session 的挂载；文档不存在时仍保持幂等。"""

    port = _mount_port()
    return port.unmount_document(document_id, port.current_session_id())


def current_mount_session_id() -> str:
    return _mount_port().current_session_id()


def preflight_documents(document_ids: set[str]) -> dict[str, Any]:
    """在检索前证明一个确定文档集合的来源时效性。"""

    if not document_ids:
        return {"ok": True, "status": "no_documents", "documents": []}
    initialize_current()
    ordered_ids = tuple(sorted(document_ids))
    with connect_current() as conn:
        rows = metadata_repository.get_document_freshness_rows(conn, ordered_ids)
    known = {str(row["id"]): dict(row) for row in rows}
    reports = [
        _source_freshness_report(document_id, known.get(document_id))
        for document_id in ordered_ids
    ]
    ok = all(item["status"] == "verified_current" for item in reports)
    return {
        "ok": ok,
        "status": "verified_current" if ok else "freshness_blocked",
        "documents": reports,
    }


def check_mounted_document_freshness(
    session_id: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    if document_id is not None:
        if not is_mounted(document_id, session_id):
            return {
                "ok": False,
                "status": "freshness_blocked",
                "documents": [
                    {"doc_id": document_id, "status": "not_mounted"}
                ],
            }
        return preflight_documents({document_id})
    return preflight_documents(
        {str(document["id"]) for document in mounted_docs(session_id)}
    )


def preflight_mounted_documents(
    session_id: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    """验证已挂载作用域并记录本次检索使用的精确版本清单。"""

    return _attach_retrieval_snapshot(
        session_id,
        check_mounted_document_freshness(session_id, document_id),
    )


def get_mounted_document_index_binding_snapshot(
    session_id: str,
    *,
    version_map: Mapping[str, str],
    maximum_bindings: int,
) -> MountedDocumentIndexBindingSnapshot:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(version_map, Mapping):
        raise ValueError("version_map must be a mapping")
    if (
        isinstance(maximum_bindings, bool)
        or not isinstance(maximum_bindings, int)
        or maximum_bindings <= 0
    ):
        raise ValueError("maximum_bindings must be a positive integer")
    expected_versions: dict[str, str] = {}
    for document_id, version_id in version_map.items():
        if (
            not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(version_id, str)
            or not version_id.strip()
        ):
            raise ValueError(
                "version_map must contain non-empty document and version IDs"
            )
        expected_versions[document_id] = version_id
    if not expected_versions:
        return MountedDocumentIndexBindingSnapshot(source_snapshot_is_current=True)
    if not set(expected_versions) <= set(mounted_document_ids(session_id)):
        return MountedDocumentIndexBindingSnapshot(source_snapshot_is_current=False)

    initialize_current()
    document_ids = tuple(sorted(expected_versions))
    with connect_current() as conn:
        current_versions: dict[str, str] = {}
        for batch in _batches(document_ids, 300):
            current_versions.update(
                {
                    str(row["id"]): str(row["current_version_id"] or "")
                    for row in chunk_repository.get_current_version_rows(conn, batch)
                }
            )
        if current_versions != expected_versions:
            return MountedDocumentIndexBindingSnapshot(
                source_snapshot_is_current=False
            )

        bindings: list[MountedDocumentIndexBinding] = []
        for batch in _batches(document_ids, 300):
            remaining_with_sentinel = maximum_bindings - len(bindings) + 1
            if remaining_with_sentinel <= 0:
                return MountedDocumentIndexBindingSnapshot(
                    source_snapshot_is_current=True,
                    binding_enumeration_complete=False,
                )
            rows = chunk_repository.get_index_binding_rows(
                conn,
                batch,
                maximum_rows=remaining_with_sentinel,
            )
            bindings.extend(
                MountedDocumentIndexBinding(
                    doc_id=str(row["doc_id"]),
                    chunk_id=str(row["id"]),
                    source_version_id=str(row["source_version_id"]),
                    indexed_content_hash=_indexed_content_hash(row),
                    producer_chunk_id=(
                        str(row["producer_chunk_id"])
                        if row["producer_chunk_id"] is not None
                        else None
                    ),
                )
                for row in rows
            )
            if len(bindings) > maximum_bindings:
                return MountedDocumentIndexBindingSnapshot(
                    source_snapshot_is_current=True,
                    binding_enumeration_complete=False,
                )
    return MountedDocumentIndexBindingSnapshot(
        source_snapshot_is_current=True,
        bindings=tuple(bindings),
    )


def get_retrieval_snapshot(
    snapshot_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    return _mount_port().get_document_retrieval_snapshot(snapshot_id, session_id)


def _indexed_content_hash(row: Mapping[str, object]) -> str:
    canonical_hash = hashlib.sha256(
        str(row["content"]).strip().encode("utf-8")
    ).hexdigest()
    persisted_hash = str(row["content_sha256"] or "")
    return persisted_hash if persisted_hash == canonical_hash else canonical_hash


def _attach_retrieval_snapshot(
    session_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    version_map = {
        str(item["doc_id"]): str(item["version_id"])
        for item in report.get("documents", [])
        if item.get("version_id")
    }
    if not version_map:
        return report
    manifest_json = json.dumps(
        version_map,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()[:16]
    snapshot_id = uuid.uuid4().hex[:8]
    _mount_port().create_document_retrieval_snapshot(
        snapshot_id=snapshot_id,
        session_id=session_id,
        manifest_hash=manifest_hash,
        manifest_json=manifest_json,
        reason="query_preflight",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return {**report, "snapshot_id": snapshot_id, "version_map": version_map}


def _source_freshness_report(
    document_id: str,
    document: dict[str, Any] | None,
) -> dict[str, Any]:
    if document is None:
        return {"doc_id": document_id, "status": "document_not_found"}
    base = {
        "doc_id": document_id,
        "version_id": document.get("current_version_id"),
        **processing_coverage_projection(
            document.get("processing_status"),
            document.get("diagnostics_json"),
        ),
    }
    expected_hash = str(document.get("source_sha256") or "")
    if not expected_hash:
        return {**base, "status": "legacy_unverified"}
    try:
        path = Path(str(document.get("path") or "")).expanduser().resolve()
    except Exception:
        return {**base, "status": "source_path_invalid"}
    if deny_reason(path):
        return {**base, "status": "source_path_denied"}
    if not path.is_file():
        return {**base, "status": "source_missing"}
    try:
        observed = fingerprint_file(path)
    except SourceChangedDuringReadError:
        return {**base, "status": "source_changed_during_check"}
    except OSError:
        return {**base, "status": "source_unreadable"}
    if observed.sha256 != expected_hash:
        return {**base, "status": "source_changed"}
    return {**base, "status": "verified_current"}


def _batches(
    values: tuple[str, ...],
    size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )
