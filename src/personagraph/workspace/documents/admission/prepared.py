"""已准备文档进入 Workspace 权威的事务编排。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from ....context_budget.token_counter import estimate_tokens
from personagraph.input_processing.documents import DocumentChunk
from personagraph.input_processing.documents import (
    DocumentElement,
    DocumentLocator,
    DocumentPageManifest,
)
from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    SourceFingerprint,
    fingerprint_file,
)
from personagraph.input_processing.documents.preparation import PreparedDocumentIngest
from personagraph.workspace.files import get_file_with_version_in_transaction
from personagraph.workspace.documents.contracts import (
    DOCUMENT_CHUNK_CONTRACT_VERSION,
    DocumentIngestCommitReceipt,
    validate_processing_coverage as _validate_processing_coverage,
)
from personagraph.workspace.documents.indexing.events import (
    capture_event_ids as _capture_retrieval_event_ids,
    enqueue_document_lifecycle_events as _enqueue_document_lifecycle_events_if_enabled,
    enqueue_document_upserts as _enqueue_document_upserts_if_enabled,
    validate_data_version as _validate_retrieval_data_version,
)
from personagraph.workspace.documents.indexing.ports import DocumentIndexPort
from personagraph.workspace.documents.mounting import (
    is_mounted_in_connection,
    mount_document,
)
from personagraph.workspace.documents.storage import commits as commit_repository
from personagraph.workspace.documents.storage import metadata as metadata_repository
from personagraph.workspace.documents.storage.commits import ChunkWrite as _ChunkWrite
from personagraph.workspace.storage.context import (
    connect_current as connect_document_db,
    current as current_project_document_database,
    initialize_current as init_document_db,
)

CHUNK_TOKENS = 1000       # 目标块大小
CHUNK_OVERLAP = 100

# 生成块的代码身份，与块所来自的文件版本相互独立。没有它，就无法区分解析器或切块器
# 升级与毫无变化，唯一安全的应对方式将是重新切分所有历史收录文档。输出形态变化时
# 应提升此版本。
CHUNKER_FINGERPRINT = "elementwise_token_aggregate@1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_document_id() -> str:
    return uuid.uuid4().hex[:8]


def chunk_text(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """忽略文档结构，将元素聚合到目标 token 数量。

    这是存储层为直接传入原始元素且不自行切块的调用方提供的回退方案。文档收录不会使用
    它：结构优先的切块逻辑位于 ``input_processing.documents``，因为只有该领域了解文档
    结构。两种方式通过 ``chunker_fingerprint`` 区分，因此切分出的材料仍可识别来源。
    """
    """elements: [{content, loc}] → chunks: [{seq, loc, content}]。按元素边界聚合到 ~CHUNK_TOKENS。"""
    chunks, buf, buf_locs, buf_tokens, seq = [], [], [], 0, 0
    def flush():
        nonlocal buf, buf_locs, buf_tokens, seq
        if buf:
            loc = buf_locs[0] if buf_locs[0] == buf_locs[-1] else f"{buf_locs[0]}-{buf_locs[-1]}"
            chunks.append({"seq": seq, "loc": loc, "content": "\n".join(buf)})
            seq += 1
            tail = buf[-1][-CHUNK_OVERLAP * 2:]
            buf, buf_locs, buf_tokens = [tail], [buf_locs[-1]], estimate_tokens(tail)
    for el in elements:
        text = (el.get("content") or "").strip()
        if not text:
            continue
        t = estimate_tokens(text)
        if buf_tokens + t > CHUNK_TOKENS and buf:
            flush()
        buf.append(text)
        buf_locs.append(str(el.get("loc", "")))
        buf_tokens += t
    if buf and (len(chunks) == 0 or len(buf) > 1):
        loc = buf_locs[0] if buf_locs[0] == buf_locs[-1] else f"{buf_locs[0]}-{buf_locs[-1]}"
        chunks.append({"seq": seq, "loc": loc, "content": "\n".join(buf)})
    return chunks


def ingest(
    path: str,
    title: str,
    mime: str,
    elements: list[dict[str, Any]],
    session_id: str | None = None,
    summary: str = "",
    *,
    file_id: str | None = None,
    file_version_id: str | None = None,
    source_fingerprint: SourceFingerprint | None = None,
    retrieval_data_version: str | None = None,
    processor_fingerprint: str | None = None,
    source_elements: Sequence[DocumentElement] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    document_chunks: Sequence[DocumentChunk] | None = None,
    chunker_fingerprint: str | None = None,
    processing_status: str | None = None,
    processing_diagnostics: Sequence[Mapping[str, object]] | None = None,
    page_manifest: DocumentPageManifest | None = None,
    document_index_port: DocumentIndexPort | None = None,
    _connection: sqlite3.Connection | None = None,
    _retrieval_event_ids_out: list[str] | None = None,
) -> dict[str, Any]:
    """收录一份文档，并原子替换已变化的物理来源。

    ``processor_fingerprint`` 标识生成 ``elements`` 的读取器。将它与切块器自身身份一同
    记录，才能在解析器升级后仅对受影响文档增量重切块，而非完整重建所有历史收录文档。

    为兼容起见，没有来源指纹的旧版调用方仍按内容哈希去重。按物理路径收录时则把规范
    路径视为逻辑文档：同一路径保留其 ID，发生变化的块会替换之前的当前 generation。
    这是 P0 过渡桥接，而非完整版本账本。
    """
    _validate_retrieval_data_version(retrieval_data_version)
    if retrieval_data_version is not None and document_index_port is None:
        raise RuntimeError(
            "retrieval_data_version requires an injected DocumentIndexPort"
        )
    raw = "\n".join((e.get("content") or "") for e in elements)
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    source_elements_json, source_elements_sha256 = (
        _prepare_source_elements_artifact(
            source_elements=source_elements,
            expected_raw=raw,
        )
    )
    # 切块归了解文档结构的一方负责。已经完成切块的调用方传入自己的单元和切块器身份；
    # 回退路径则让旧版调用方保持原样工作。
    if chunks is not None and document_chunks is not None:
        raise ValueError("chunks and document_chunks are mutually exclusive")
    if document_chunks is not None and source_fingerprint is None:
        raise ValueError("typed document_chunks require source_fingerprint")
    if page_manifest is not None and document_chunks is None:
        raise ValueError("page_manifest requires typed document_chunks")
    if page_manifest is not None:
        _validate_page_manifest_chunks(page_manifest, document_chunks or ())
    legacy_chunks = chunks if chunks is not None else (
        None if document_chunks is not None else chunk_text(elements)
    )
    chunk_writes = _prepare_chunk_writes(
        document_chunks=document_chunks,
        legacy_chunks=legacy_chunks,
    )
    chunker = chunker_fingerprint or CHUNKER_FINGERPRINT
    normalized_processing_status, diagnostics_json = _processing_provenance(
        processing_status=processing_status,
        processing_diagnostics=processing_diagnostics,
        typed=document_chunks is not None,
    )
    chunk_contract_version = (
        DOCUMENT_CHUNK_CONTRACT_VERSION if document_chunks is not None else None
    )
    page_manifest_json = (
        page_manifest.canonical_json() if page_manifest is not None else None
    )
    page_manifest_sha256 = (
        page_manifest.manifest_sha256 if page_manifest is not None else None
    )
    physical_page_count = (
        page_manifest.physical_page_count if page_manifest is not None else None
    )
    if _connection is None:
        init_document_db()
    connection_scope = (
        connect_document_db() if _connection is None else nullcontext(_connection)
    )
    project_bound = current_project_document_database() is not None
    with connection_scope as conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        _require_project_file_link(
            conn,
            path=path,
            source_fingerprint=source_fingerprint,
            file_id=file_id,
            file_version_id=file_version_id,
        )
        # 物理来源按路径标识。重新收录会在一个事务内替换旧 chunk，
        # 而非遗留陈旧挂载并新增第二条文档记录。
        path_row = None
        if source_fingerprint is not None:
            path_row = commit_repository.find_document_by_path(
                conn,
                path,
                include_file_link=project_bound,
            )
        if path_row:
            _require_existing_document_file_link(
                path_row,
                file_id=file_id,
                file_version_id=file_version_id,
            )
            existing_content_hash = path_row["content_hash"] or path_row["hash"]
            source_sha_matches = (
                bool(path_row["source_sha256"])
                and path_row["source_sha256"] == source_fingerprint.sha256
            )
            if existing_content_hash == content_hash and source_sha_matches:
                if path_row["current_page_manifest_sha256"] and page_manifest is None:
                    raise ValueError(
                        "same-source reindex cannot drop existing page manifest provenance"
                    )
                # Returning to earlier bytes still has a distinct file-version lineage.
                file_version_changed = (
                    file_version_id is not None
                    and path_row["file_version_id"] != file_version_id
                )
                if file_version_changed or _stored_processing_differs(
                    conn,
                    doc_id=str(path_row["id"]),
                    processor_fingerprint=processor_fingerprint,
                    chunker_fingerprint=chunker,
                    typed=document_chunks is not None,
                    processing_status=normalized_processing_status,
                    diagnostics_json=diagnostics_json,
                    page_manifest_json=page_manifest_json,
                    page_manifest_sha256=page_manifest_sha256,
                    physical_page_count=physical_page_count,
                    source_elements_json=source_elements_json,
                    source_elements_sha256=source_elements_sha256,
                    chunk_writes=chunk_writes,
                ):
                    _capture_retrieval_event_ids(
                        _retrieval_event_ids_out,
                        _enqueue_document_lifecycle_events_if_enabled(
                            conn,
                            doc_id=str(path_row["id"]),
                            kind="purge",
                            retrieval_data_version=retrieval_data_version,
                            occurred_at=now_utc(),
                            index_port=document_index_port,
                        ),
                    )
                    _replace_document_contents(
                        conn,
                        doc_id=path_row["id"],
                        path=path,
                        mime=mime,
                        content_hash=content_hash,
                        chunk_writes=chunk_writes,
                        session_id=session_id,
                        file_id=file_id,
                        file_version_id=file_version_id,
                        source_fingerprint=source_fingerprint,
                        processor_fingerprint=processor_fingerprint,
                        chunker_fingerprint=chunker,
                        processing_status=normalized_processing_status,
                        diagnostics_json=diagnostics_json,
                        chunk_contract_version=chunk_contract_version,
                        page_manifest_json=page_manifest_json,
                        page_manifest_sha256=page_manifest_sha256,
                        physical_page_count=physical_page_count,
                        source_elements_json=source_elements_json,
                        source_elements_sha256=source_elements_sha256,
                    )
                    mount_document(path_row["id"], session_id)
                    _capture_retrieval_event_ids(
                        _retrieval_event_ids_out,
                        _enqueue_document_upserts_if_enabled(
                            conn,
                            doc_id=str(path_row["id"]),
                            retrieval_data_version=retrieval_data_version,
                            index_port=document_index_port,
                        ),
                    )
                    return {
                        "doc_id": path_row["id"],
                        "n_chunks": len(chunk_writes),
                        "deduped": False,
                        "reindexed": True,
                    }
                _update_source_provenance(
                    conn,
                    path_row["id"],
                    source_fingerprint,
                    file_id=file_id,
                    file_version_id=file_version_id,
                )
                if mount_document(path_row["id"], session_id):
                    _capture_retrieval_event_ids(
                        _retrieval_event_ids_out,
                        _enqueue_document_lifecycle_events_if_enabled(
                            conn,
                            doc_id=str(path_row["id"]),
                            session_ids=(str(session_id),),
                            kind="restore",
                            retrieval_data_version=retrieval_data_version,
                            occurred_at=now_utc(),
                            index_port=document_index_port,
                        ),
                    )
                return {
                    "doc_id": path_row["id"],
                    "n_chunks": path_row["n_chunks"],
                    "deduped": True,
                    "reindexed": False,
                }
            _capture_retrieval_event_ids(
                _retrieval_event_ids_out,
                _enqueue_document_lifecycle_events_if_enabled(
                    conn,
                    doc_id=str(path_row["id"]),
                    kind="purge",
                    retrieval_data_version=retrieval_data_version,
                    occurred_at=now_utc(),
                    index_port=document_index_port,
                ),
            )
            _replace_document_contents(
                conn,
                doc_id=path_row["id"],
                path=path,
                mime=mime,
                content_hash=content_hash,
                chunk_writes=chunk_writes,
                session_id=session_id,
                file_id=file_id,
                file_version_id=file_version_id,
                source_fingerprint=source_fingerprint,
                processor_fingerprint=processor_fingerprint,
                chunker_fingerprint=chunker,
                processing_status=normalized_processing_status,
                diagnostics_json=diagnostics_json,
                chunk_contract_version=chunk_contract_version,
                page_manifest_json=page_manifest_json,
                page_manifest_sha256=page_manifest_sha256,
                physical_page_count=physical_page_count,
                source_elements_json=source_elements_json,
                source_elements_sha256=source_elements_sha256,
            )
            mount_document(path_row["id"], session_id)
            _capture_retrieval_event_ids(
                _retrieval_event_ids_out,
                _enqueue_document_upserts_if_enabled(
                    conn,
                    doc_id=str(path_row["id"]),
                    retrieval_data_version=retrieval_data_version,
                    index_port=document_index_port,
                ),
            )
            return {
                "doc_id": path_row["id"],
                "n_chunks": len(chunk_writes),
                "deduped": False,
                "reindexed": True,
            }

        # 显式旧版收录没有可持久化的物理标识，因此保留历史上的同内容复用行为。物理来源
        # 不同路径之间不能共享文档记录：引用/新鲜度必须有唯一明确的规范来源。
        if source_fingerprint is None:
            row = commit_repository.find_document_by_content_hash(
                conn,
                content_hash,
            )
            if row:
                if mount_document(row["id"], session_id):
                    _capture_retrieval_event_ids(
                        _retrieval_event_ids_out,
                        _enqueue_document_lifecycle_events_if_enabled(
                            conn,
                            doc_id=str(row["id"]),
                            session_ids=(str(session_id),),
                            kind="restore",
                            retrieval_data_version=retrieval_data_version,
                            occurred_at=now_utc(),
                            index_port=document_index_port,
                        ),
                    )
                return {"doc_id": row["id"], "n_chunks": row["n_chunks"], "deduped": True, "reindexed": False}
        doc_id = _new_document_id()
        dedupe_key = _unique_dedupe_key(conn, content_hash, doc_id)
        added_at = now_utc()
        commit_repository.insert_document(
            conn,
            document_id=doc_id,
            path=path,
            title=title,
            mime=mime,
            dedupe_key=dedupe_key,
            content_hash=content_hash,
            summary=summary,
            source_session=session_id,
            chunk_count=len(chunk_writes),
            added_at=added_at,
            source_sha256=(
                source_fingerprint.sha256 if source_fingerprint else None
            ),
            source_size=(
                source_fingerprint.size_bytes if source_fingerprint else None
            ),
            source_mtime_ns=(
                source_fingerprint.mtime_ns if source_fingerprint else None
            ),
            source_verified_at=added_at if source_fingerprint else None,
            freshness_status=(
                "verified_current" if source_fingerprint else "legacy_unverified"
            ),
            file_id=file_id,
        )
        version_id = _activate_document_version(
            conn,
            doc_id=doc_id,
            content_hash=content_hash,
            source_fingerprint=source_fingerprint,
            processing_status=normalized_processing_status,
            diagnostics_json=diagnostics_json,
            processor_fingerprint=processor_fingerprint,
            chunker_fingerprint=chunker,
            chunk_contract_version=chunk_contract_version,
            page_manifest_json=page_manifest_json,
            page_manifest_sha256=page_manifest_sha256,
            physical_page_count=physical_page_count,
            file_version_id=file_version_id,
            source_elements_json=source_elements_json,
            source_elements_sha256=source_elements_sha256,
        )
        _insert_chunk_generation(
            conn,
            doc_id=doc_id,
            version_id=version_id,
            chunk_writes=chunk_writes,
            processor_fingerprint=processor_fingerprint,
            chunker_fingerprint=chunker,
        )
        mount_document(doc_id, session_id)
        _capture_retrieval_event_ids(
            _retrieval_event_ids_out,
            _enqueue_document_upserts_if_enabled(
                conn,
                doc_id=doc_id,
                retrieval_data_version=retrieval_data_version,
                index_port=document_index_port,
            ),
        )
    return {"doc_id": doc_id, "n_chunks": len(chunk_writes), "deduped": False, "reindexed": False}


def ingest_prepared_in_transaction(
    conn: sqlite3.Connection,
    *,
    prepared: PreparedDocumentIngest,
    session_id: str | None = None,
    retrieval_data_version: str,
    document_index_port: DocumentIndexPort,
    file_id: str | None = None,
    file_version_id: str | None = None,
) -> DocumentIngestCommitReceipt:
    """在调用方现有的权威事务内提交已准备好的来源。

    此窄入口允许持久收录操作在一个由调用方所有的 Document 事务中，对文档、版本、块、
    仅含指针的检索事件及其自身阶段建立检查点。它绝不会提交或回滚所提供的连接。
    共享准备传入 ``session_id=None``，只提交项目来源；Session 挂载由独立请求交付。
    """

    if not isinstance(prepared, PreparedDocumentIngest):
        raise TypeError("prepared must be PreparedDocumentIngest")
    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
        raise ValueError("prepared ingest requires a caller-owned transaction")
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id.strip()
    ):
        raise ValueError("session_id must be non-empty when provided")
    if not isinstance(document_index_port, DocumentIndexPort):
        raise TypeError("document_index_port must satisfy DocumentIndexPort")
    if (
        not isinstance(retrieval_data_version, str)
        or not retrieval_data_version.strip()
    ):
        raise ValueError("retrieval_data_version must not be empty")

    event_ids: list[str] = []
    stored = ingest(
        prepared.canonical_path,
        prepared.title,
        prepared.mime,
        [dict(element) for element in prepared.elements],
        session_id=session_id,
        file_id=file_id,
        file_version_id=file_version_id,
        source_fingerprint=prepared.source_fingerprint,
        retrieval_data_version=retrieval_data_version,
        processor_fingerprint=prepared.processor_fingerprint,
        source_elements=prepared.source_elements,
        document_chunks=prepared.document_chunks,
        chunker_fingerprint=prepared.chunker_fingerprint,
        processing_status=prepared.processing_status,
        processing_diagnostics=tuple(
            dict(diagnostic) for diagnostic in prepared.processing_diagnostics
        ),
        page_manifest=prepared.page_manifest,
        document_index_port=document_index_port,
        _connection=conn,
        _retrieval_event_ids_out=event_ids,
    )
    current_version_id = metadata_repository.current_version_id(
        conn,
        str(stored["doc_id"]),
    )
    if (
        current_version_id is None
        or (
            session_id is not None
            and not is_mounted_in_connection(
                conn,
                str(stored["doc_id"]),
                session_id,
            )
        )
    ):
        raise RuntimeError("prepared ingest did not establish current document authority")
    return DocumentIngestCommitReceipt(
        document_id=str(stored["doc_id"]),
        document_version_id=current_version_id,
        retrieval_event_ids=tuple(dict.fromkeys(event_ids)),
        n_chunks=int(stored["n_chunks"]),
        deduped=bool(stored["deduped"]),
        reindexed=bool(stored["reindexed"]),
    )


def has_reusable_content_artifact(
    conn: sqlite3.Connection,
    *,
    canonical_path: str,
    source_sha256: str,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
) -> bool:
    """是否存在与精确字节和处理配方对应的持久带类型输出。

    这是生产 DocStore 查找，而非进程内解析缓存。可复用材料是一个已提交的当前
    ``document_version`` 及其带类型块。来源路径刻意不计入制品身份，但被排除的候选
    不能是目标路径：同一路径刷新仍走普通的新鲜度检查和重建索引路径。
    """

    artifact = _reusable_content_artifact_row(
        conn,
        canonical_path=canonical_path,
        source_sha256=source_sha256,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
    )
    if artifact is None or not _reusable_content_artifact_chunks(
        conn,
        artifact=artifact,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
    ):
        return False
    try:
        _validate_reusable_processing_artifact(artifact)
        _validate_reusable_page_manifest(
            page_manifest_json=_optional_row_text(artifact["page_manifest_json"]),
            page_manifest_sha256=_optional_row_text(
                artifact["page_manifest_sha256"]
            ),
            physical_page_count=(
                int(artifact["physical_page_count"])
                if artifact["physical_page_count"] is not None
                else None
            ),
        )
        _validate_reusable_source_elements(artifact)
    except (TypeError, ValueError):
        return False
    return True


def reuse_content_artifact_in_transaction(
    conn: sqlite3.Connection,
    *,
    canonical_path: str,
    session_id: str | None = None,
    source_fingerprint: SourceFingerprint,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
    retrieval_data_version: str,
    document_index_port: DocumentIndexPort,
    file_id: str | None = None,
    file_version_id: str | None = None,
) -> DocumentIngestCommitReceipt | None:
    """将精确的持久解析输出克隆到新的物理来源权威源。

    仅共享派生出的带类型载荷。目标会获得新的 Document ID、来源版本 ID、存储块 ID，
    以及路径或 Session 挂载；其来源指纹会根据目标文件重新盖印。因此，两个别名或路径
    绝不会折叠为同一权威或来源记录。

    调用方拥有事务，因此新权威、其仅含指针的 Retrieval 事件和收录任务检查点可以原子
    提交。制品缺失或不再有效时返回 ``None``，让工作器回退到规范解析器。
    """

    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
        raise ValueError("content artifact reuse requires a caller-owned transaction")
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id.strip()
    ):
        raise ValueError("session_id must be non-empty when provided")
    if not isinstance(document_index_port, DocumentIndexPort):
        raise TypeError("document_index_port must satisfy DocumentIndexPort")
    _validate_retrieval_data_version(retrieval_data_version)
    if not isinstance(source_fingerprint, SourceFingerprint):
        raise TypeError("source_fingerprint must be SourceFingerprint")
    _require_project_file_link(
        conn,
        path=canonical_path,
        source_fingerprint=source_fingerprint,
        file_id=file_id,
        file_version_id=file_version_id,
    )

    target_path = str(Path(canonical_path).resolve())
    # 绝不能绕过 ``ingest`` 中路径本地的更新语义。此辅助函数严格用于承载相同字节的
    # 第二个物理来源。
    if commit_repository.document_path_exists(conn, target_path):
        return None

    artifact = _reusable_content_artifact_row(
        conn,
        canonical_path=target_path,
        source_sha256=source_fingerprint.sha256,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
    )
    if artifact is None:
        return None
    chunk_writes = _reusable_content_artifact_chunks(
        conn,
        artifact=artifact,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
    )
    if not chunk_writes:
        return None

    try:
        processing_status, diagnostics_json = (
            _validate_reusable_processing_artifact(artifact)
        )
        page_manifest_json = _optional_row_text(artifact["page_manifest_json"])
        page_manifest_sha256 = _optional_row_text(artifact["page_manifest_sha256"])
        physical_page_count = (
            int(artifact["physical_page_count"])
            if artifact["physical_page_count"] is not None
            else None
        )
        _validate_reusable_page_manifest(
            page_manifest_json=page_manifest_json,
            page_manifest_sha256=page_manifest_sha256,
            physical_page_count=physical_page_count,
        )
        source_elements_json, source_elements_sha256 = (
            _validate_reusable_source_elements(artifact)
        )
    except (TypeError, ValueError):
        return None

    content_hash = str(artifact["content_hash"] or "").strip()
    if not content_hash:
        return None
    path = Path(target_path)
    document_id = _new_document_id()
    captured_at = now_utc()
    commit_repository.insert_document(
        conn,
        document_id=document_id,
        path=target_path,
        title=path.stem,
        mime=path.suffix.lstrip("."),
        dedupe_key=_unique_dedupe_key(conn, content_hash, document_id),
        content_hash=content_hash,
        summary="",
        source_session=session_id,
        chunk_count=len(chunk_writes),
        added_at=captured_at,
        source_sha256=source_fingerprint.sha256,
        source_size=source_fingerprint.size_bytes,
        source_mtime_ns=source_fingerprint.mtime_ns,
        source_verified_at=captured_at,
        freshness_status="verified_current",
        file_id=file_id,
    )
    document_version_id = _activate_document_version(
        conn,
        doc_id=document_id,
        content_hash=content_hash,
        source_fingerprint=source_fingerprint,
        processing_status=processing_status,
        diagnostics_json=diagnostics_json,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        page_manifest_json=page_manifest_json,
        page_manifest_sha256=page_manifest_sha256,
        physical_page_count=physical_page_count,
        file_version_id=file_version_id,
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
    )
    _insert_chunk_generation(
        conn,
        doc_id=document_id,
        version_id=document_version_id,
        chunk_writes=chunk_writes,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
    )
    mount_document(document_id, session_id)
    event_ids = _enqueue_document_upserts_if_enabled(
        conn,
        doc_id=document_id,
        retrieval_data_version=retrieval_data_version,
        index_port=document_index_port,
    )
    return DocumentIngestCommitReceipt(
        document_id=document_id,
        document_version_id=document_version_id,
        retrieval_event_ids=tuple(dict.fromkeys(event_ids)),
        n_chunks=len(chunk_writes),
        # 已创建新的权威源；只复用了其派生解析制品，因此旧版“同一 Document”去重必须
        # 保持为 false。
        deduped=False,
        reindexed=False,
    )


def _reusable_content_artifact_row(
    conn: sqlite3.Connection,
    *,
    canonical_path: str,
    source_sha256: str,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
) -> sqlite3.Row | None:
    path = Path(canonical_path)
    return commit_repository.find_reusable_content_artifact(
        conn,
        canonical_path=str(path.resolve()),
        source_sha256=source_sha256,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        file_extension=path.suffix.lstrip("."),
    )


def _reusable_content_artifact_chunks(
    conn: sqlite3.Connection,
    *,
    artifact: sqlite3.Row,
    processor_fingerprint: str,
    chunker_fingerprint: str,
    chunk_contract_version: int,
) -> tuple[_ChunkWrite, ...]:
    return commit_repository.load_reusable_content_chunks(
        conn,
        artifact=artifact,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
    )


def _validate_reusable_processing_artifact(
    artifact: sqlite3.Row,
) -> tuple[str, str | None]:
    status = str(artifact["processing_status"] or "")
    raw_diagnostics = artifact["diagnostics_json"]
    if raw_diagnostics is None:
        diagnostics: object = None
    else:
        try:
            diagnostics = json.loads(str(raw_diagnostics))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("reusable processing diagnostics are invalid") from exc
    normalized_status, normalized = _validate_processing_coverage(
        status,
        diagnostics,
    )
    normalized_json = (
        None if normalized is None else _canonical_json(list(normalized))
    )
    if normalized_json != _optional_row_text(raw_diagnostics):
        raise ValueError("reusable processing diagnostics are not canonical")
    return normalized_status, normalized_json


def _validate_reusable_page_manifest(
    *,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
) -> None:
    if page_manifest_json is None:
        if page_manifest_sha256 is not None or physical_page_count is not None:
            raise ValueError("reusable page manifest lineage is incomplete")
        return
    manifest = DocumentPageManifest.from_json(page_manifest_json)
    if (
        manifest.canonical_json() != page_manifest_json
        or manifest.manifest_sha256 != page_manifest_sha256
        or manifest.physical_page_count != physical_page_count
    ):
        raise ValueError("reusable page manifest lineage is invalid")


def _validate_reusable_source_elements(
    artifact: sqlite3.Row,
) -> tuple[str, str]:
    source_elements_json = _optional_row_text(artifact["source_elements_json"])
    source_elements_sha256 = _optional_row_text(
        artifact["source_elements_sha256"]
    )
    version_content_hash = str(artifact["version_content_hash"] or "")
    if version_content_hash != str(artifact["content_hash"] or ""):
        raise ValueError("reusable source elements use a conflicting content version")
    commit_repository.validate_source_elements_artifact(
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
        content_hash=version_content_hash,
        required=True,
    )
    assert source_elements_json is not None
    assert source_elements_sha256 is not None
    return source_elements_json, source_elements_sha256


def _optional_row_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _prepare_source_elements_artifact(
    *,
    source_elements: Sequence[DocumentElement] | None,
    expected_raw: str,
) -> tuple[str | None, str | None]:
    """冻结 reader 原始文本顺序；旧式调用方明确保留不可查询的空来源。"""

    if source_elements is None:
        return None, None
    if not source_elements:
        raise ValueError("source_elements must not be empty when provided")
    payload: list[dict[str, object]] = []
    identities: set[str] = set()
    contents: list[str] = []
    for element in source_elements:
        if not isinstance(element, DocumentElement) or not (element.text or "").strip():
            raise ValueError("source_elements must contain exact text elements")
        if element.element_id in identities:
            raise ValueError("source element identities must be unique")
        identities.add(element.element_id)
        content = element.text or ""
        contents.append(content)
        payload.append({
            "element_id": element.element_id,
            "content": content,
            "locator": element.locator.describe(),
            "source_pages": list(element.source_pages),
        })
    if "\n".join(contents) != expected_raw:
        raise ValueError("source_elements do not match admitted element content")
    source_elements_json = _canonical_json(payload)
    return (
        source_elements_json,
        hashlib.sha256(source_elements_json.encode("utf-8")).hexdigest(),
    )


def _prepare_chunk_writes(
    *,
    document_chunks: Sequence[DocumentChunk] | None,
    legacy_chunks: Sequence[Mapping[str, Any]] | None,
) -> tuple[_ChunkWrite, ...]:
    """在权威事务开始前校验并冻结一个 generation。"""

    if document_chunks is None:
        return tuple(
            _ChunkWrite(
                seq=int(chunk["seq"]),
                loc=str(chunk.get("loc") or ""),
                content=str(chunk.get("content") or ""),
            )
            for chunk in (legacy_chunks or ())
        )

    writes: list[_ChunkWrite] = []
    seen_producer_ids: set[str] = set()
    for seq, chunk in enumerate(document_chunks):
        if not isinstance(chunk, DocumentChunk):
            raise ValueError("document_chunks must contain DocumentChunk values")
        producer_chunk_id = chunk.chunk_id.strip()
        if not producer_chunk_id:
            raise ValueError("producer_chunk_id must not be empty")
        if producer_chunk_id in seen_producer_ids:
            raise ValueError(
                f"duplicate producer_chunk_id in document generation: {producer_chunk_id}"
            )
        seen_producer_ids.add(producer_chunk_id)
        writes.append(
            _ChunkWrite(
                seq=seq,
                loc=chunk.loc,
                content=chunk.text,
                producer_chunk_id=producer_chunk_id,
                span_json=_canonical_json({
                    "start": _locator_payload(chunk.span.start),
                    "end": _locator_payload(chunk.span.end),
                }),
                metadata_json=_canonical_json({
                    "section_path": list(chunk.section_path),
                    "element_ids": list(chunk.element_ids),
                    "token_count": chunk.token_count,
                    "kind": chunk.kind.value,
                    "was_split": chunk.was_split,
                }),
                content_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                chunk_contract_version=DOCUMENT_CHUNK_CONTRACT_VERSION,
                source_pages_json=_canonical_json(list(chunk.source_pages)),
            )
        )
    return tuple(writes)


def _locator_payload(locator: DocumentLocator) -> dict[str, object]:
    return {
        "page": locator.page,
        "ordinal": locator.ordinal,
        "section_path": list(locator.section_path),
        "bbox": list(locator.bbox) if locator.bbox is not None else None,
        "char_range": list(locator.char_range) if locator.char_range is not None else None,
    }


def _validate_page_manifest_chunks(
    manifest: DocumentPageManifest,
    chunks: Sequence[DocumentChunk],
) -> None:
    """在打开写入围栏前证明精确的元素到页面覆盖。"""

    element_pages: dict[str, set[int]] = {}
    for page in manifest.pages:
        for element_id in page.text_element_ids:
            element_pages.setdefault(element_id, set()).add(page.page_number)
    covered_ids: set[str] = set()
    for chunk in chunks:
        if not chunk.source_pages:
            raise ValueError("manifest-bound chunk source_pages must not be empty")
        expected_pages: set[int] = set()
        for element_id in chunk.element_ids:
            pages = element_pages.get(element_id)
            if pages is None:
                raise ValueError(
                    "manifest-bound chunk references an unknown text element"
                )
            covered_ids.add(element_id)
            expected_pages.update(pages)
        if tuple(sorted(expected_pages)) != chunk.source_pages:
            raise ValueError(
                "manifest-bound chunk source_pages are not the exact element page set"
            )
    if covered_ids != set(element_pages):
        raise ValueError("page manifest contains text elements omitted from chunks")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _processing_provenance(
    *,
    processing_status: str | None,
    processing_diagnostics: Sequence[Mapping[str, object]] | None,
    typed: bool,
) -> tuple[str, str | None]:
    status = processing_status or ("complete" if typed else "legacy_unknown")
    diagnostics_input: object = tuple(
        dict(item) for item in (processing_diagnostics or ())
    )
    if status == "legacy_unknown" and not diagnostics_input:
        diagnostics_input = None
    status, diagnostics = _validate_processing_coverage(
        status,
        diagnostics_input,
    )
    if typed and status == "legacy_unknown":
        raise ValueError("typed chunks cannot use legacy_unknown processing_status")
    if not typed and status != "legacy_unknown":
        raise ValueError("known processing_status requires typed document chunks")
    return status, None if diagnostics is None else _canonical_json(list(diagnostics))


def _unique_dedupe_key(conn, content_hash: str, doc_id: str) -> str:
    """在允许路径本地来源的同时保留旧版唯一 ``hash`` 列。"""

    return commit_repository.unique_dedupe_key(conn, content_hash, doc_id)


def _stored_processing_differs(
    conn,
    *,
    doc_id: str,
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    typed: bool,
    processing_status: str,
    diagnostics_json: str | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
    chunk_writes: Sequence[_ChunkWrite],
) -> bool:
    """已知处理器是否会替换当前块 generation。

    没有处理器身份的调用方属于旧版调用方，必须保留按内容哈希去重。带类型读取器则不能
    仅因文件字节未变就被视为空操作：新的解析代码可能从同一文件产生不同的位置、结构
    和块。
    """

    return commit_repository.stored_processing_differs(
        conn,
        document_id=doc_id,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        typed=typed,
        processing_status=processing_status,
        diagnostics_json=diagnostics_json,
        page_manifest_json=page_manifest_json,
        page_manifest_sha256=page_manifest_sha256,
        physical_page_count=physical_page_count,
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
        chunk_writes=chunk_writes,
    )


def _require_project_file_link(
    conn: sqlite3.Connection,
    *,
    path: str,
    source_fingerprint: SourceFingerprint | None,
    file_id: str | None,
    file_version_id: str | None,
) -> None:
    """证明一个 Document generation 属于已登记的项目文件。"""

    database = current_project_document_database()
    if database is None:
        if file_id is not None or file_version_id is not None:
            raise ValueError("file linkage requires a project documents database")
        return
    if source_fingerprint is None:
        raise ValueError("project document ingest requires a source fingerprint")
    if not file_id or not file_version_id:
        raise ValueError(
            "project document ingest requires file_id and file_version_id"
        )
    registered = get_file_with_version_in_transaction(
        conn,
        database,
        file_id=file_id,
        file_version_id=file_version_id,
    )
    if registered is None:
        raise ValueError("document file linkage is missing or not current")
    file, version = registered
    if file.current_version_id != file_version_id:
        raise ValueError("document file linkage is missing or not current")
    linked_path = (database.project_root / file.relative_path).resolve()
    if linked_path != Path(path).expanduser().resolve():
        raise ValueError("document path does not match its registered project file")
    if (
        version.content_sha256 != source_fingerprint.sha256
        or version.size_bytes != source_fingerprint.size_bytes
        or file.observed_mtime_ns != source_fingerprint.mtime_ns
    ):
        raise ValueError("document source does not match its registered file version")
    if fingerprint_file(linked_path) != source_fingerprint:
        raise SourceChangedDuringReadError(
            "document source changed after its project file version was registered"
        )


def _require_existing_document_file_link(
    row: sqlite3.Row,
    *,
    file_id: str | None,
    file_version_id: str | None,
) -> None:
    if file_id is None and file_version_id is None:
        return
    if not file_id or not file_version_id:
        raise ValueError("document file linkage must provide both identities")
    existing_file_id = row["file_id"]
    if existing_file_id is not None and str(existing_file_id) != file_id:
        raise ValueError("document path is already linked to another project file")


def _update_source_provenance(
    conn,
    doc_id: str,
    fingerprint: SourceFingerprint,
    *,
    file_id: str | None,
    file_version_id: str | None,
) -> None:
    commit_repository.update_source_provenance(
        conn,
        document_id=doc_id,
        source_sha256=fingerprint.sha256,
        source_size=fingerprint.size_bytes,
        source_mtime_ns=fingerprint.mtime_ns,
        verified_at=now_utc(),
        file_id=file_id,
        file_version_id=file_version_id,
    )


def _activate_document_version(
    conn,
    *,
    doc_id: str,
    content_hash: str,
    source_fingerprint: SourceFingerprint | None,
    processing_status: str,
    diagnostics_json: str | None,
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
    chunk_contract_version: int | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    file_version_id: str | None = None,
    source_elements_json: str | None = None,
    source_elements_sha256: str | None = None,
) -> str:
    """追加一个来源版本，并移动唯一的当前版本指针。"""

    version_id = _new_document_id()
    captured_at = now_utc()
    commit_repository.activate_document_version(
        conn,
        version_id=version_id,
        document_id=doc_id,
        content_hash=content_hash,
        source_sha256=(
            source_fingerprint.sha256 if source_fingerprint else None
        ),
        source_size=(
            source_fingerprint.size_bytes if source_fingerprint else None
        ),
        source_mtime_ns=(
            source_fingerprint.mtime_ns if source_fingerprint else None
        ),
        captured_at=captured_at,
        processing_status=processing_status,
        diagnostics_json=diagnostics_json,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_contract_version=chunk_contract_version,
        page_manifest_json=page_manifest_json,
        page_manifest_sha256=page_manifest_sha256,
        physical_page_count=physical_page_count,
        file_version_id=file_version_id,
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
    )
    return version_id


def _replace_document_contents(
    conn,
    *,
    doc_id: str,
    path: str,
    mime: str,
    content_hash: str,
    chunk_writes: Sequence[_ChunkWrite],
    session_id: str | None,
    file_id: str | None,
    file_version_id: str | None,
    source_fingerprint: SourceFingerprint,
    processor_fingerprint: str | None = None,
    chunker_fingerprint: str | None = None,
    processing_status: str,
    diagnostics_json: str | None,
    chunk_contract_version: int | None,
    page_manifest_json: str | None,
    page_manifest_sha256: str | None,
    physical_page_count: int | None,
    source_elements_json: str | None,
    source_elements_sha256: str | None,
) -> None:
    """在当前事务中替换一个路径的活动块。"""

    version_id = _new_document_id()
    commit_repository.replace_document_contents(
        conn,
        document_id=doc_id,
        path=path,
        mime=mime,
        content_hash=content_hash,
        chunk_writes=chunk_writes,
        source_session=session_id,
        file_id=file_id,
        source_sha256=source_fingerprint.sha256,
        source_size=source_fingerprint.size_bytes,
        source_mtime_ns=source_fingerprint.mtime_ns,
        verified_at=now_utc(),
        version_id=version_id,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint or CHUNKER_FINGERPRINT,
        processing_status=processing_status,
        diagnostics_json=diagnostics_json,
        chunk_contract_version=chunk_contract_version,
        page_manifest_json=page_manifest_json,
        page_manifest_sha256=page_manifest_sha256,
        physical_page_count=physical_page_count,
        file_version_id=file_version_id,
        source_elements_json=source_elements_json,
        source_elements_sha256=source_elements_sha256,
        chunk_ids=tuple(_new_document_id() for _ in chunk_writes),
    )


def _insert_chunk_generation(
    conn,
    *,
    doc_id: str,
    version_id: str,
    chunk_writes: Sequence[_ChunkWrite],
    processor_fingerprint: str | None,
    chunker_fingerprint: str,
) -> None:
    """通过单一 SQL 路径插入一个已经校验的块 generation。"""

    commit_repository.insert_chunk_generation(
        conn,
        document_id=doc_id,
        version_id=version_id,
        chunk_writes=chunk_writes,
        processor_fingerprint=processor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunk_ids=tuple(_new_document_id() for _ in chunk_writes),
    )
