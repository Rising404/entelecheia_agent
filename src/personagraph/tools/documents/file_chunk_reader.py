"""Bounded exact Document reads using shared native File and chunk identities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib

from ...workspace.documents.reading import (
    CurrentFileChunk,
    CurrentFileDocument,
    get_current_chunk_document,
    get_current_file_document,
    read_current_file_chunk,
)
from ...workspace.files.access import AuthorizedFileSource
from ..effects import EffectScopeKind
from ..execution import ToolBusinessFailure
from ..retrieval.public_projection import sanitize_public_locator
from .file_chunk_tools import (
    MAX_READ_TARGETS, MAX_CHUNKS_PER_TARGET, MAX_CHUNK_CONTENT_CHARS, MAX_TOTAL_CONTENT_CHARS,
    READ_FILE_CHUNKS_CONTRACT_VERSION, build_read_file_chunks_registration,
)


@dataclass(frozen=True, slots=True)
class FileChunkReaderRuntime:
    session_id: str = field(repr=False)
    scope_id: str = field(repr=False)
    resolve_file: Callable[..., AuthorizedFileSource] = field(repr=False)
    revalidate: Callable[[AuthorizedFileSource], bool] = field(repr=False)
    read_document: Callable[..., CurrentFileDocument | None] = field(
        default=get_current_file_document, repr=False,
    )
    read_chunk: Callable[..., CurrentFileChunk | None] = field(
        default=read_current_file_chunk, repr=False,
    )
    read_chunk_document: Callable[..., CurrentFileDocument | None] = field(
        default=get_current_chunk_document, repr=False,
    )
    filesystem_scope_kind: EffectScopeKind = EffectScopeKind.WORKSPACE

    def __post_init__(self):
        if not self.session_id or not self.scope_id or not all(
            callable(value)
            for value in (self.resolve_file, self.revalidate, self.read_document,
                          self.read_chunk, self.read_chunk_document)
        ):
            raise ValueError("chunk reader requires a Session and explicit read authority")

    @property
    def effect_scope(self):
        return "file-corpus:" + hashlib.sha256(self.scope_id.encode()).hexdigest()

    def registration(self):
        return build_read_file_chunks_registration(
            handler=self.read_chunks, effect_scope=self.effect_scope,
            filesystem_scope_kind=self.filesystem_scope_kind,
        )

    def read_chunks(self, payload):
        targets, unavailable = self._resolve_targets(_parse_targets(payload))
        results = []
        remaining = MAX_TOTAL_CONTENT_CHARS
        for target in targets:
            result, gaps, used = self._read_target(target, remaining)
            results.append(result)
            unavailable.extend(gaps)
            remaining -= used
        return {
            "contract_version": READ_FILE_CHUNKS_CONTRACT_VERSION,
            "results": results,
            "unavailable_targets": unavailable,
            "truncated": any(
                item["reason_code"] == "content_budget_exhausted" for item in unavailable
            ) or any(
                chunk["content_truncated"] for item in results for chunk in item["chunks"]
            ),
        }

    def _resolve_targets(self, targets):
        """Native IDs supply lineage; sequence selectors retain explicit version context."""
        resolved, unavailable = [], []
        for target in targets:
            if "chunk_ids" not in target or all(
                field in target for field in ("file_id", "document_version_id")
            ):
                resolved.append(target)
                continue
            groups = {}
            for chunk_id in target["chunk_ids"]:
                document = self.read_chunk_document(chunk_id=chunk_id, session_id=self.session_id)
                if document is None or any(
                    field in target and target[field] != getattr(document, field)
                    for field in ("file_id", "document_version_id")
                ):
                    unavailable.append(_unavailable(target, "chunk_ids", chunk_id, "chunk_unavailable"))
                    continue
                key = (document.file_id, document.document_version_id)
                groups.setdefault(key, {
                    "file_id": document.file_id,
                    "document_version_id": document.document_version_id,
                    "chunk_ids": [],
                })["chunk_ids"].append(chunk_id)
            resolved.extend(groups.values())
        return resolved, unavailable

    def _read_target(self, target, remaining):
        result = {
            "file_id": target["file_id"],
            "file_version_id": None,
            "document_id": None,
            "document_version_id": target["document_version_id"],
            "status": "unavailable",
            "chunks": [],
        }
        key = "chunk_ids" if "chunk_ids" in target else "chunk_sequences"
        selectors = target[key]
        try:
            source = self.resolve_file(file_id=target["file_id"])
            if source is None or source.file_id != target["file_id"] or not self.revalidate(source):
                raise ValueError("File access unavailable")
            document = self.read_document(
                file_id=source.file_id,
                file_version_id=source.file_version_id,
                session_id=self.session_id,
            )
            if document is None or document.document_version_id != target["document_version_id"]:
                return result, [_unavailable(target, key, value, "document_version_unavailable") for value in selectors], 0
        except Exception:
            return result, [_unavailable(target, key, value, "file_access_unavailable") for value in selectors], 0
        gaps, chunks, used = [], [], 0
        for value in selectors:
            if used >= remaining:
                gaps.append(_unavailable(target, key, value, "content_budget_exhausted"))
                continue
            try:
                chunk = self.read_chunk(
                    file_id=source.file_id, file_version_id=source.file_version_id,
                    document_version_id=document.document_version_id, session_id=self.session_id,
                    **({"chunk_id": value} if key == "chunk_ids" else {"sequence": value}),
                )
                if chunk is None or chunk.document != document:
                    raise ValueError("chunk belongs to another Document generation")
                matches_selector = (
                    chunk.chunk_id == value if key == "chunk_ids" else chunk.sequence == value
                )
                if (
                    not matches_selector or not chunk.content
                    or hashlib.sha256(chunk.content.encode()).hexdigest() != chunk.content_sha256
                ):
                    raise ValueError("chunk identity or content mismatch")
                content = chunk.content[:min(MAX_CHUNK_CONTENT_CHARS, remaining - used)]
                chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "sequence": chunk.sequence,
                    "locator": sanitize_public_locator(chunk.locator),
                    "source_pages": list(chunk.source_pages),
                    "content": content,
                    "content_truncated": len(content) < len(chunk.content),
                    "content_sha256": chunk.content_sha256,
                })
                used += len(content)
            except Exception:
                gaps.append(_unavailable(target, key, value, "chunk_unavailable"))
        if not self.revalidate(source):
            return result, [_unavailable(target, key, value, "file_access_changed") for value in selectors], 0
        current_document = self.read_document(
            file_id=source.file_id,
            file_version_id=source.file_version_id,
            session_id=self.session_id,
        )
        if current_document != document:
            return result, [
                _unavailable(target, key, value, "document_version_changed") for value in selectors
            ], 0
        result.update({
            "file_version_id": source.file_version_id,
            "document_id": document.document_id,
            "chunks": chunks,
            "status": "partial" if gaps else "ready",
        })
        return result, gaps, used


def build_file_chunk_reader_runtime(**kwargs):
    return FileChunkReaderRuntime(**kwargs)


def _parse_targets(payload):
    if not isinstance(payload, Mapping) or set(payload) != {"targets"}:
        raise ToolBusinessFailure("invalid_request", "The chunk read request is invalid.")
    targets = payload["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_READ_TARGETS:
        raise ToolBusinessFailure("invalid_request", "targets must be a bounded non-empty list.")
    for target in targets:
        if not isinstance(target, Mapping) or set(target) - {"file_id", "document_version_id", "chunk_ids", "chunk_sequences"}:
            raise ToolBusinessFailure("invalid_request", "Invalid chunk target.")
        if any(key in target and not _identifier(target[key]) for key in ("file_id", "document_version_id")):
            raise ToolBusinessFailure("invalid_request", "File and Document version IDs must be valid.")
        if ("chunk_ids" in target) == ("chunk_sequences" in target):
            raise ToolBusinessFailure("invalid_request", "Use chunk_ids or chunk_sequences, not both.")
        if "chunk_sequences" in target and not all(
            key in target for key in ("file_id", "document_version_id")
        ):
            raise ToolBusinessFailure("invalid_request", "Sequence reads require exact File and Document version IDs.")
        values = target.get("chunk_ids", target.get("chunk_sequences"))
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_CHUNKS_PER_TARGET:
            raise ToolBusinessFailure("invalid_request", "Chunk selectors must be a bounded non-empty list.")
        valid = all(_identifier(value) for value in values) if "chunk_ids" in target else all(
            not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2**63 for value in values)
        if not valid or len(values) != len(set(values)):
            raise ToolBusinessFailure("invalid_request", "Chunk selectors must be valid and unique.")
    return targets


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and value == value.strip() and not any(ord(c) < 32 for c in value)


def _unavailable(target, key, value, reason):
    return {"file_id": target.get("file_id"), "document_version_id": target.get("document_version_id"),
            "chunk_id": value if key == "chunk_ids" else None,
            "chunk_sequence": value if key == "chunk_sequences" else None, "reason_code": reason}


__all__ = ["FileChunkReaderRuntime", "build_file_chunk_reader_runtime"]
