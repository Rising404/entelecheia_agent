"""将一个 Turn 的上传文件绑定为已挂载 Document 权威状态。

聊天附件与已挂载 Document 有意采用不同 ingress 边界。上传首先接受不透明字节，而
AuxiliaryGraph 只能感知有类型且带来源指纹的 Document generation。本模块是这些
边界间由 Host 持有的 bridge。

此 bridge 只在 Session store 已将附件原子绑定到某 Turn 后运行。它准入当前支持的
有限文本/文档/图像格式，重新检查 Project 已登记用户文件与不可变上传 receipt，通过共享有类型
文档读取器解析，并在同一 Session 中挂载精确已准备 generation。不支持或不完整的
目标输入会以封闭原因码失败；绝不会从规划权威状态中静默省略。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
from typing import Any

from personagraph.workspace.files.attachments import MAX_ATTACHMENT_BYTES
from personagraph.workspace.files.turn_inputs import (
    ResolvedTurnInputFile,
    TurnInputFileAuthorityError,
    resolve_turn_input_files,
)
from personagraph.input_processing.documents import (
    ChunkingProfile,
    DocumentPrepareFailure,
    prepare_document_path,
    read_document,
)
from personagraph.input_processing.files import (
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
    fingerprint_file,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import (
    connect_current,
    initialize_current,
)
from personagraph.configuration.paths import deny_reason

from .contracts import (
    AttachmentDocumentAuthorityError,
    AttachmentDocumentBridgeResult,
    AttachmentDocumentGap,
    AttachmentDocumentGapReason,
    AttachmentDocumentMount,
    MountedDocumentStorePort,
    PrepareDocument,
    PreparedAttachmentAuthority,
    PreparedAttachmentVisualIngest,
    TurnAttachmentStorePort,
)


_SUPPORTED_MEDIA_BY_SUFFIX: Mapping[str, str] = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".doc": "application/msword",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
}
_PLAIN_TEXT_RESOURCE_SUFFIXES = frozenset({".txt", ".md", ".markdown"})




def mount_turn_document_attachments(
    *,
    session_id: str,
    turn_id: str,
    attachment_ids: Sequence[str] | None = None,
    attachment_store: TurnAttachmentStorePort | None = None,
    document_store: MountedDocumentStorePort = docstore,
    prepare: PrepareDocument | None = None,
    existing_mounts_only: bool = False,
) -> AttachmentDocumentBridgeResult:
    """挂载绑定到 ``turn_id`` 的所选受支持附件。

    非目标附件族仍由普通附件投影持有，并报告为 ignored。然而，一旦记录声明属于目标
    TXT/Markdown/PDF/image/Office 类型，任何路径、receipt、reader 或权威失败都会以
    显式有类型 gap 停止辅助图规划。``existing_mounts_only`` 是读取侧恢复模式：解析
    精确持久化挂载，否则失败；不会调用 parser 或提交数据。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if not isinstance(existing_mounts_only, bool):
        raise TypeError("existing_mounts_only must be a boolean")
    if attachment_store is None:
        from ....session import store as session_store

        attachment_store = session_store
    all_records = tuple(
        attachment_store.list_turn_attachments(session_id, turn_id)
    )
    _validate_record_set(all_records, session_id=session_id, turn_id=turn_id)
    records = _select_attachment_records(all_records, attachment_ids)
    try:
        resolved_files = (
            resolve_turn_input_files(
                all_records,
                session_id=session_id,
                turn_id=turn_id,
            )
            if all_records
            else ()
        )
    except TurnInputFileAuthorityError as exc:
        raise AttachmentDocumentAuthorityError(
            tuple(
                AttachmentDocumentGap(
                    attachment_id=str(record["attachment_id"]),
                    reason=_gap_reason_from_turn_file_authority(exc.reason),
                    detail_code=exc.reason[:160],
                )
                for record in records
            )
        ) from exc
    resolved_by_attachment_id = {
        resolved.attachment_id: resolved for resolved in resolved_files
    }
    expected_attachment_ids = {
        str(record["attachment_id"]) for record in all_records
    }
    if (
        len(resolved_by_attachment_id) != len(resolved_files)
        or set(resolved_by_attachment_id) != expected_attachment_ids
    ):
        raise AttachmentDocumentAuthorityError(
            tuple(
                AttachmentDocumentGap(
                    attachment_id=str(record["attachment_id"]),
                    reason=AttachmentDocumentGapReason.MALFORMED_BINDING,
                    detail_code="resolver_attachment_alignment_mismatch",
                )
                for record in records
            )
        )
    selected_files = tuple(
        resolved_by_attachment_id[str(record["attachment_id"])]
        for record in records
    )

    prepared_items: list[
        tuple[ResolvedTurnInputFile, PreparedAttachmentAuthority]
    ] = []
    mounts_by_attachment_id: dict[str, AttachmentDocumentMount] = {}
    ignored: list[str] = []
    gaps: list[AttachmentDocumentGap] = []
    prepare_fn = prepare

    for resolved_file in selected_files:
        attachment_id = resolved_file.attachment_id
        suffix = PurePosixPath(resolved_file.relative_path).suffix.lower()
        expected_media_type = _SUPPORTED_MEDIA_BY_SUFFIX.get(suffix)
        kind = resolved_file.kind.lower()
        if expected_media_type is None and kind not in {"document", "image"}:
            ignored.append(attachment_id)
            continue
        if expected_media_type is None:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.UNSUPPORTED_FORMAT,
                )
            )
            continue
        media_type = resolved_file.media_type.lower()
        expected_kinds = (
            {"text"}
            if suffix in _PLAIN_TEXT_RESOURCE_SUFFIXES
            else {"document", "image"}
        )
        if media_type != expected_media_type or kind not in expected_kinds:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.TYPE_BINDING_MISMATCH,
                )
            )
            continue
        target = resolved_file.canonical_path
        if deny_reason(target) is not None:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.SENSITIVE_PATH_DENIED,
                )
            )
            continue
        try:
            observed_source = _fingerprint_managed_attachment(
                target,
                max_bytes=MAX_ATTACHMENT_BYTES,
            )
        except SourceSizeLimitError:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.TOO_LARGE,
                )
            )
            continue
        except (OSError, SourceChangedDuringReadError, ValueError):
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=(
                        AttachmentDocumentGapReason.CONTENT_RECEIPT_MISMATCH
                    ),
                )
            )
            continue
        if (
            observed_source.size_bytes != resolved_file.size_bytes
            or observed_source.sha256
            != resolved_file.content_sha256
        ):
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=(
                        AttachmentDocumentGapReason.CONTENT_RECEIPT_MISMATCH
                    ),
                )
            )
            continue

        existing_mount = _find_exact_existing_mount(
            attachment_id=attachment_id,
            session_id=session_id,
            file_id=resolved_file.file_id,
            file_version_id=resolved_file.file_version_id,
            target=target,
            observed_source=observed_source,
            document_store=document_store,
        )
        if existing_mount is not None:
            mounts_by_attachment_id[attachment_id] = existing_mount
            continue
        if existing_mounts_only:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
                )
            )
            continue
        if prepare_fn is None:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.PROCESSING_INCOMPLETE,
                    detail_code="document_preparer_unavailable",
                )
            )
            continue
        try:
            prepared = prepare_fn(str(target))
        except SourceSizeLimitError:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.TOO_LARGE,
                )
            )
            continue
        except SourceChangedDuringReadError:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.SOURCE_CHANGED,
                )
            )
            continue
        except Exception as exc:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.PROCESSING_INCOMPLETE,
                    detail_code=type(exc).__name__,
                )
            )
            continue
        if isinstance(prepared, DocumentPrepareFailure):
            gaps.append(_gap_from_prepare_failure(attachment_id, prepared))
            continue
        if (
            prepared.canonical_path != str(target)
            or prepared.source_fingerprint.sha256
            != resolved_file.content_sha256
            or prepared.source_fingerprint.size_bytes != resolved_file.size_bytes
        ):
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.SOURCE_CHANGED,
                )
            )
            continue
        prepared_items.append((resolved_file, prepared))

    if gaps:
        raise AttachmentDocumentAuthorityError(gaps)

    committed_mounts = (
        _commit_prepared_batch(
            prepared_items,
            session_id=session_id,
            document_store=document_store,
        )
        if prepared_items
        else ()
    )
    mounts_by_attachment_id.update(
        (mount.attachment_id, mount) for mount in committed_mounts
    )
    return AttachmentDocumentBridgeResult(
        session_id=session_id,
        turn_id=turn_id,
        mounts=tuple(
            mounts_by_attachment_id[resolved_file.attachment_id]
            for resolved_file in selected_files
            if resolved_file.attachment_id in mounts_by_attachment_id
        ),
        ignored_attachment_ids=tuple(ignored),
    )


def _select_attachment_records(
    records: Sequence[Mapping[str, Any]],
    attachment_ids: Sequence[str] | None,
) -> tuple[Mapping[str, Any], ...]:
    """选择精确子集，而不改变已持久化 Turn 顺序。"""

    if attachment_ids is None:
        return tuple(records)
    if isinstance(attachment_ids, (str, bytes)):
        raise ValueError("attachment_ids must be a sequence of identifiers")
    requested = tuple(attachment_ids)
    if not requested or any(
        not isinstance(item, str) or not item.strip() for item in requested
    ):
        raise ValueError("attachment_ids must contain bounded identifiers")
    if len(requested) != len(set(requested)):
        raise ValueError("attachment_ids must be unique")
    requested_set = set(requested)
    available = {
        str(record["attachment_id"]): record for record in records
    }
    if not requested_set <= set(available):
        raise ValueError("attachment selection left the accepted Turn scope")
    return tuple(
        record
        for record in records
        if str(record["attachment_id"]) in requested_set
    )


def _find_exact_existing_mount(
    *,
    attachment_id: str,
    session_id: str,
    file_id: str,
    file_version_id: str,
    target: Path,
    observed_source: SourceFingerprint,
    document_store: MountedDocumentStorePort,
) -> AttachmentDocumentMount | None:
    """返回已证明的精确 generation，重放时不重新解析。"""

    try:
        mounted = document_store.mounted_docs(session_id)
    except Exception as exc:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
                detail_code=type(exc).__name__,
            ),
        )) from exc
    for document in mounted:
        try:
            same_source = (
                Path(str(document.get("path") or "")).expanduser().resolve()
                == target
                and document.get("file_id") == file_id
                and document.get("file_version_id") == file_version_id
                and str(document.get("source_sha256") or "")
                == observed_source.sha256
                and int(document.get("source_size"))
                == observed_source.size_bytes
            )
        except (OSError, TypeError, ValueError):
            same_source = False
        if not same_source:
            continue
        _require_source_fingerprint_current(
            target,
            observed_source,
            attachment_id=attachment_id,
        )
        # 元数据与最终资源读取必须属于同一版本；并发推进后由调用方重新取得权威。
        return _load_mount_from_store(
            attachment_id=attachment_id,
            session_id=session_id,
            document_id=str(document.get("id") or ""),
            expected_source_sha256=observed_source.sha256,
            expected_document_version_id=str(document.get("current_version_id") or ""),
            document_store=document_store,
        )
    return None


def _load_mount_from_store(
    *,
    attachment_id: str,
    session_id: str,
    document_id: str,
    expected_source_sha256: str,
    document_store: MountedDocumentStorePort,
    expected_document_version_id: str | None = None,
) -> AttachmentDocumentMount:
    try:
        snapshot = document_store.get_mounted_current_document_resource_snapshot(
            document_id,
            session_id=session_id,
            maximum_chunks=1,
        )
    except Exception as exc:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
                detail_code=type(exc).__name__,
            ),
        )) from exc
    version_changed = (
        snapshot is not None
        and expected_document_version_id is not None
        and str(getattr(snapshot, "document_version_id", ""))
        != expected_document_version_id
    )
    if (
        snapshot is None
        or str(getattr(snapshot, "source_sha256", ""))
        != expected_source_sha256
        or version_changed
    ):
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
                detail_code="document_version_changed" if version_changed else None,
            ),
        ))
    try:
        return AttachmentDocumentMount(
            attachment_id=attachment_id,
            document_id=str(getattr(snapshot, "document_id")),
            document_version_id=str(getattr(snapshot, "document_version_id")),
            source_sha256=str(getattr(snapshot, "source_sha256")),
            processing_status=str(getattr(snapshot, "processing_status")),
            processing_diagnostic_codes=tuple(
                str(item)
                for item in getattr(snapshot, "processing_diagnostic_codes", ())
            ),
        )
    except (TypeError, ValueError) as exc:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
                detail_code=type(exc).__name__,
            ),
        )) from exc


def _load_committed_mount_in_transaction(
    conn: sqlite3.Connection,
    *,
    attachment_id: str,
    session_id: str,
    document_id: str,
    file_id: str,
    file_version_id: str,
    prepared: PreparedAttachmentAuthority,
) -> AttachmentDocumentMount:
    """在调用方持有的事务提交前证明刚写入的挂载。"""

    row = conn.execute(
        "SELECT d.id AS document_id, d.path, d.current_version_id, d.file_id, "
        "d.source_sha256 AS document_source_sha256, "
        "d.source_size AS document_source_size, "
        "d.n_chunks, "
        "v.id AS document_version_id, v.file_version_id, "
        "v.source_sha256 AS version_source_sha256, "
        "v.source_size AS version_source_size, "
        "v.processing_status, v.diagnostics_json, "
        "v.chunk_contract_version, "
        "(SELECT COUNT(*) FROM doc_chunks AS c "
        " WHERE c.doc_id=d.id AND c.source_version_id=v.id) "
        "AS current_chunk_count "
        "FROM documents AS d "
        "JOIN document_versions AS v ON v.id=d.current_version_id "
        "WHERE d.id=?",
        (document_id,),
    ).fetchone()
    try:
        diagnostics = (
            json.loads(str(row["diagnostics_json"])) if row is not None else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        diagnostics = None
    expected_diagnostics = [dict(item) for item in prepared.processing_diagnostics]
    fingerprint = prepared.source_fingerprint
    valid = (
        row is not None
        and docstore.is_mounted_in_connection(conn, document_id, session_id)
        and str(row["document_id"]) == document_id
        and str(row["path"]) == prepared.canonical_path
        and row["file_id"] == file_id
        and row["file_version_id"] == file_version_id
        and row["current_version_id"] == row["document_version_id"]
        and row["document_source_sha256"] == fingerprint.sha256
        and row["version_source_sha256"] == fingerprint.sha256
        and row["document_source_size"] == fingerprint.size_bytes
        and row["version_source_size"] == fingerprint.size_bytes
        and row["processing_status"] == prepared.processing_status
        and row["processing_status"] in {"complete", "partial"}
        and diagnostics == expected_diagnostics
        and row["chunk_contract_version"]
        == docstore.DOCUMENT_CHUNK_CONTRACT_VERSION
        and row["n_chunks"] == len(prepared.document_chunks)
        and row["current_chunk_count"] == len(prepared.document_chunks)
    )
    if not valid:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
            ),
        ))
    assert isinstance(diagnostics, list)
    diagnostic_codes = tuple(
        str(item.get("code") or "")
        for item in diagnostics
        if isinstance(item, Mapping)
    )
    if len(diagnostic_codes) != len(diagnostics) or any(
        not code for code in diagnostic_codes
    ):
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING,
            ),
        ))
    return AttachmentDocumentMount(
        attachment_id=attachment_id,
        document_id=document_id,
        document_version_id=str(row["document_version_id"]),
        source_sha256=fingerprint.sha256,
        processing_status=str(row["processing_status"]),
        processing_diagnostic_codes=diagnostic_codes,
    )


def _require_prepared_source_current(
    prepared: PreparedAttachmentAuthority,
    *,
    attachment_id: str,
) -> None:
    _require_source_fingerprint_current(
        Path(prepared.canonical_path),
        prepared.source_fingerprint,
        attachment_id=attachment_id,
    )


def _require_source_fingerprint_current(
    path: Path,
    expected: SourceFingerprint,
    *,
    attachment_id: str,
) -> None:
    try:
        observed = _fingerprint_managed_attachment(
            path,
            max_bytes=MAX_ATTACHMENT_BYTES,
        )
    except SourceSizeLimitError as exc:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.TOO_LARGE,
            ),
        )) from exc
    except (OSError, SourceChangedDuringReadError, ValueError) as exc:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.SOURCE_CHANGED,
            ),
        )) from exc
    if observed != expected:
        raise AttachmentDocumentAuthorityError((
            AttachmentDocumentGap(
                attachment_id=attachment_id,
                reason=AttachmentDocumentGapReason.SOURCE_CHANGED,
            ),
        ))


def _commit_prepared_batch(
    prepared_items: Sequence[
        tuple[ResolvedTurnInputFile, PreparedAttachmentAuthority]
    ],
    *,
    session_id: str,
    document_store: MountedDocumentStorePort,
) -> tuple[AttachmentDocumentMount, ...]:
    """在一个权威事务中提交并证明生产批次。"""

    if not prepared_items:
        return ()
    if document_store is docstore:
        initialize_current()
        mounts: list[AttachmentDocumentMount] = []
        active_attachment_id = prepared_items[0][0].attachment_id
        try:
            with connect_current() as conn:
                for resolved_file, prepared in prepared_items:
                    active_attachment_id = resolved_file.attachment_id
                    _require_prepared_source_current(
                        prepared,
                        attachment_id=active_attachment_id,
                    )
                    stored = _commit_prepared_document(
                        document_store,
                        prepared,
                        resolved_file=resolved_file,
                        session_id=session_id,
                        connection=conn,
                    )
                    _require_prepared_source_current(
                        prepared,
                        attachment_id=active_attachment_id,
                    )
                    mounts.append(
                        _load_committed_mount_in_transaction(
                            conn,
                            attachment_id=active_attachment_id,
                            session_id=session_id,
                            document_id=str(stored["doc_id"]),
                            file_id=resolved_file.file_id,
                            file_version_id=resolved_file.file_version_id,
                            prepared=prepared,
                        )
                    )
                # 这有意作为 SQLite 上下文管理器提交前的最终操作。因此若文件在任何
                # 后续条目处理期间被替换，整个同 Turn 批次都会回滚。
                for resolved_file, prepared in prepared_items:
                    active_attachment_id = resolved_file.attachment_id
                    _require_prepared_source_current(
                        prepared,
                        attachment_id=active_attachment_id,
                    )
        except AttachmentDocumentAuthorityError:
            raise
        except Exception as exc:
            raise AttachmentDocumentAuthorityError((
                AttachmentDocumentGap(
                    attachment_id=active_attachment_id,
                    reason=AttachmentDocumentGapReason.COMMIT_FAILED,
                    detail_code=type(exc).__name__,
                ),
            )) from exc
        return tuple(mounts)

    # 提供的 port 是聚焦测试/应用 adapter。只有上方生产路径可在 Project documents.sqlite 中
    # 声明全有或全无的批次权威状态。
    mounts = []
    for resolved_file, prepared in prepared_items:
        attachment_id = resolved_file.attachment_id
        try:
            _require_prepared_source_current(
                prepared,
                attachment_id=attachment_id,
            )
            stored = _commit_prepared_document(
                document_store,
                prepared,
                resolved_file=resolved_file,
                session_id=session_id,
            )
            _require_prepared_source_current(
                prepared,
                attachment_id=attachment_id,
            )
            mounts.append(
                _load_mount_from_store(
                    attachment_id=attachment_id,
                    session_id=session_id,
                    document_id=str(stored["doc_id"]),
                    expected_source_sha256=prepared.source_fingerprint.sha256,
                    document_store=document_store,
                )
            )
        except AttachmentDocumentAuthorityError:
            raise
        except Exception as exc:
            raise AttachmentDocumentAuthorityError((
                AttachmentDocumentGap(
                    attachment_id=attachment_id,
                    reason=AttachmentDocumentGapReason.COMMIT_FAILED,
                    detail_code=type(exc).__name__,
                ),
            )) from exc
    return tuple(mounts)


def _commit_prepared_document(
    document_store: MountedDocumentStorePort,
    prepared: PreparedAttachmentAuthority,
    *,
    resolved_file: ResolvedTurnInputFile,
    session_id: str,
    connection: object | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "session_id": session_id,
        "source_fingerprint": prepared.source_fingerprint,
        "processor_fingerprint": prepared.processor_fingerprint,
        "source_elements": getattr(prepared, "source_elements", None),
        "document_chunks": prepared.document_chunks,
        "chunker_fingerprint": prepared.chunker_fingerprint,
        "processing_status": prepared.processing_status,
        "processing_diagnostics": tuple(
            dict(item) for item in prepared.processing_diagnostics
        ),
        "page_manifest": prepared.page_manifest,
        "retrieval_data_version": None,
    }
    if connection is not None:
        values["_connection"] = connection
    values.update(
        file_id=resolved_file.file_id,
        file_version_id=resolved_file.file_version_id,
    )
    return document_store.ingest(
        prepared.canonical_path,
        prepared.title,
        prepared.mime,
        [dict(element) for element in prepared.elements],
        **values,
    )


def prepare_attachment_document(
    path: str,
    *,
    chunking_profile: ChunkingProfile,
) -> PreparedAttachmentAuthority | DocumentPrepareFailure:
    """使用产品纯视觉准入 policy 准备一个附件。"""

    prepared = prepare_document_path(
        path,
        chunking_profile=chunking_profile,
    )
    if not isinstance(prepared, DocumentPrepareFailure):
        return prepared
    if prepared.reason not in {
        "document_processing_incomplete",
        "no_text_extracted",
    }:
        return prepared
    visual = _prepare_visual_only_attachment(
        path,
        chunking_profile=chunking_profile,
    )
    return visual if visual is not None else prepared


def _prepare_visual_only_attachment(
    path_str: str,
    *,
    chunking_profile: ChunkingProfile,
) -> PreparedAttachmentVisualIngest | None:
    """仅将精确无文本 PDF/PNG/JPEG 准入为未解析视觉权威状态。"""

    path = Path(path_str).expanduser().resolve()
    if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        return None
    try:
        before = fingerprint_file(path)
        result = read_document(path)
        after = fingerprint_file(path)
    except (
        OSError,
        RuntimeError,
        SourceChangedDuringReadError,
        SourceSizeLimitError,
        ValueError,
    ):
        return None
    if before != after or result.text_elements() or result.page_manifest is None:
        return None
    if not result.needs_vision or not any(
        unit.requires_visual_read
        for page in result.page_manifest.pages
        for unit in page.nontext_units
    ):
        return None
    # 页面 manifest 保留完整 reader 诊断，包括 OCR backend 失败。文档级 partial
    # 覆盖只携带 planner 所需的可持久化事实：视觉语义仍未读取。
    persisted_diagnostics = tuple(
        diagnostic.to_dict()
        for diagnostic in result.diagnostics
        if diagnostic.code.value
        in {"page_needs_vision", "page_empty", "fallback_reader_used"}
    )
    if not any(
        item.get("code") == "page_needs_vision"
        for item in persisted_diagnostics
    ):
        return None
    return PreparedAttachmentVisualIngest(
        canonical_path=str(path),
        title=path.stem,
        mime=path.suffix.lstrip("."),
        elements=(),
        document_chunks=(),
        source_fingerprint=after,
        processor_fingerprint=str(result.processor),
        chunker_fingerprint=chunking_profile.fingerprint(),
        processing_status="partial",
        processing_diagnostics=persisted_diagnostics,
        needs_vision=True,
        summary_preview="",
        page_manifest=result.page_manifest,
    )


def _fingerprint_managed_attachment(
    path: Path,
    *,
    max_bytes: int,
) -> SourceFingerprint:
    """在 inode guard 保护下，对 ``path`` 当前指向的常规文件执行哈希。

    共享文档指纹防护 size/mtime 变化。此 ingress 边界还会将已打开 descriptor 绑定到
    受管路径，因此哈希期间的 ``os.replace`` 无法在已接受路径已指向不同文件时，
    仍返回旧 descriptor 中的字节。
    """

    before_path = path.lstat()
    if not stat.S_ISREG(before_path.st_mode):
        raise SourceChangedDuringReadError(str(path))
    if before_path.st_size > max_bytes:
        raise SourceSizeLimitError(str(path))
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as source:
        opened_before = os.fstat(source.fileno())
        if _file_observation(opened_before) != _file_observation(before_path):
            raise SourceChangedDuringReadError(str(path))
        while block := source.read(1_048_576):
            observed_bytes += len(block)
            if observed_bytes > max_bytes:
                raise SourceSizeLimitError(str(path))
            digest.update(block)
        opened_after = os.fstat(source.fileno())
    after_path = path.lstat()
    if (
        _file_observation(opened_before) != _file_observation(opened_after)
        or _file_observation(opened_after) != _file_observation(after_path)
        or observed_bytes != after_path.st_size
    ):
        raise SourceChangedDuringReadError(str(path))
    return SourceFingerprint(
        sha256=digest.hexdigest(),
        size_bytes=after_path.st_size,
        mtime_ns=after_path.st_mtime_ns,
    )


def _file_observation(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_record_set(
    records: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    turn_id: str,
) -> None:
    seen: set[str] = set()
    gaps: list[AttachmentDocumentGap] = []
    for ordinal, record in enumerate(records):
        attachment_id = str(record.get("attachment_id") or "")
        if not attachment_id:
            attachment_id = f"malformed_attachment_{ordinal + 1}"
        bound_at = record.get("bound_at")
        untrusted_origin = (
            str(record.get("origin") or "") != "user_upload"
            or not isinstance(bound_at, str)
            or not bound_at
            or len(bound_at) > 80
        )
        malformed = (
            attachment_id in seen
            or len(attachment_id) > 160
            or str(record.get("session_id") or "") != session_id
            or str(record.get("turn_id") or "") != turn_id
            or not 1 <= len(str(record.get("stored_rel_path") or "")) <= 1024
            or not _is_sha256(str(record.get("content_hash") or ""))
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or int(record.get("size_bytes") or 0) < 0
        )
        seen.add(attachment_id)
        if malformed or untrusted_origin:
            gaps.append(
                AttachmentDocumentGap(
                    attachment_id=attachment_id[:160],
                    reason=(
                        AttachmentDocumentGapReason.MALFORMED_BINDING
                        if malformed
                        else AttachmentDocumentGapReason.UNTRUSTED_ORIGIN
                    ),
                )
            )
    if gaps:
        raise AttachmentDocumentAuthorityError(gaps)


def _is_sha256(value: str) -> bool:
    normalized = value.lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _gap_reason_from_turn_file_authority(
    reason: str,
) -> AttachmentDocumentGapReason:
    if reason in {
        "untrusted_attachment_origin",
        "project_file_source_mismatch",
    }:
        return AttachmentDocumentGapReason.UNTRUSTED_ORIGIN
    if reason in {"unsafe_project_path", "registered_path_mismatch"}:
        return AttachmentDocumentGapReason.PATH_OUTSIDE_SESSION_INPUT
    if reason in {
        "project_file_lookup_failed",
        "project_file_version_missing",
        "project_file_identity_mismatch",
        "bound_version_not_current",
        "project_file_version_incomplete",
        "session_receipt_mismatch",
        "file_content_changed",
    }:
        return AttachmentDocumentGapReason.CONTENT_RECEIPT_MISMATCH
    if reason == "file_too_large":
        return AttachmentDocumentGapReason.TOO_LARGE
    if reason == "project_context_missing":
        return AttachmentDocumentGapReason.MOUNT_AUTHORITY_MISSING
    return AttachmentDocumentGapReason.MALFORMED_BINDING


def _gap_from_prepare_failure(
    attachment_id: str,
    failure: DocumentPrepareFailure,
) -> AttachmentDocumentGap:
    reason = failure.reason
    if reason.startswith("denied_"):
        mapped = AttachmentDocumentGapReason.SENSITIVE_PATH_DENIED
    elif reason == "unsupported_legacy_office":
        mapped = AttachmentDocumentGapReason.UNSUPPORTED_LEGACY_OFFICE
    elif reason == "too_large":
        mapped = AttachmentDocumentGapReason.TOO_LARGE
    elif reason == "source_changed_during_ingest":
        mapped = AttachmentDocumentGapReason.SOURCE_CHANGED
    else:
        mapped = AttachmentDocumentGapReason.PROCESSING_INCOMPLETE
    return AttachmentDocumentGap(
        attachment_id=attachment_id,
        reason=mapped,
        detail_code=reason[:160],
    )


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be a 1..200 character identity")


__all__ = [
    "AttachmentDocumentAuthorityError",
    'AttachmentDocumentBridgeResult',
    'AttachmentDocumentGapReason',
    'AttachmentDocumentGap',
    'AttachmentDocumentMount',
    "PreparedAttachmentAuthority",
    "mount_turn_document_attachments",
    "prepare_attachment_document",
]
