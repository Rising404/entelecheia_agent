"""冻结当前工作区和附加事实，形成 L1 文集的权威状态。"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Any, TYPE_CHECKING
from .corpus_contracts import (
    FrozenL1CorpusManifest,
    L1CorpusAttachment,
    L1CorpusManifest,
    L1CorpusManifestError,
    L1CorpusWorkspace,
    freeze_l1_corpus_contract,
    l1_corpus_canonical_json,
)

if TYPE_CHECKING:
    from .tool_runtime import L1ToolRuntime


def freeze_l1_corpus_manifest(
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    tool_runtime: L1ToolRuntime,
    attachments: Any,
) -> FrozenL1CorpusManifest:
    """捕获文集标识而不持久化源文本或绝对路径。"""

    workspace = None
    authority = tool_runtime.workspace_corpus_authority
    if authority is not None:
        workspace = L1CorpusWorkspace(
            boundary_fingerprint=authority.boundary_fingerprint,
            scope_snapshot_sha256=authority.scope_snapshot_sha256,
        )
    projection_items = tuple(attachments.items)
    sources = tuple(tool_runtime.turn_attachment_sources)
    if len(projection_items) != len(sources):
        raise L1CorpusManifestError(
            "L1 attachment projection and FileVersion authority differ"
        )
    attachment_items = tuple(
        _attachment(index, source, projection)
        for index, (source, projection) in enumerate(
            zip(sources, projection_items, strict=True),
            start=1,
        )
    )
    return freeze_l1_corpus_contract(
        L1CorpusManifest(
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            catalog_snapshot_sha256=tool_runtime.catalog_snapshot_sha256,
            workspace=workspace,
            attachment_count=len(attachment_items),
            attachments=attachment_items,
        )
    )


def _attachment(
    index: int,
    source: Any,
    projection: Any,
) -> L1CorpusAttachment:
    kind = source.kind.value
    access = getattr(projection.access, "value", projection.access)
    projection_facts = (
        projection.attachment_id,
        projection.name,
        projection.media_type,
        projection.size_bytes,
        getattr(projection.kind, "value", projection.kind),
        access,
        projection.stored_rel_path,
        projection.content_hash,
    )
    source_facts = (
        source.attachment_id,
        source.original_name,
        source.media_type,
        source.size_bytes,
        kind,
        "on_demand",
        source.relative_path,
        source.content_sha256,
    )
    if projection_facts != source_facts or source.ordinal != index - 1:
        raise L1CorpusManifestError(
            "L1 attachment projection and FileVersion authority differ"
        )
    name = source.original_name
    projection_identity = {
        "contract": "l1-attachment-file-source-identity-v1",
        "attachment_id": source.attachment_id,
        "ordinal": source.ordinal,
        "input_message_id": source.input_message_id,
        "project_id": source.project_id,
        "file_id": source.file_id,
        "file_version_id": source.file_version_id,
        "name": name,
        "media_type": source.media_type,
        "size_bytes": source.size_bytes,
        "kind": kind,
        "access": "on_demand",
        "origin": "user_upload",
        "content_sha256": source.content_sha256,
    }
    return L1CorpusAttachment(
        alias=f"attachment_{index:03d}",
        attachment_id=source.attachment_id,
        ordinal=source.ordinal,
        input_message_id=source.input_message_id,
        project_id=source.project_id,
        file_id=source.file_id,
        file_version_id=source.file_version_id,
        name=name,
        media_type=source.media_type,
        suffix=PurePosixPath(name).suffix.lower(),
        size_bytes=source.size_bytes,
        kind=kind,
        content_sha256=source.content_sha256,
        access="on_demand",
        origin="user_upload",
        projection_identity_sha256=_sha256_json(projection_identity),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(l1_corpus_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["freeze_l1_corpus_manifest"]
