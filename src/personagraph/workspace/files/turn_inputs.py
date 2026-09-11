"""Resolve accepted Turn uploads through immutable Project file identities.

Session rows prove which files were accepted for a Turn.  Project
``documents.sqlite`` proves what those files and versions are.  This module joins
the two authorities without treating the Session's stored relative path as a
filesystem capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hmac
import os
from pathlib import Path
import stat
from typing import NoReturn

from personagraph.input_processing.files import FileKind
from personagraph.workspace.storage.context import current as current_document_database
from personagraph.workspace.files import (
    FileSource,
    ProjectFileChangedError,
    ProjectFilePathError,
    ProjectFileSizeLimitError,
    ProjectFileRecord,
    WorkspaceFileAuthority,
    ProjectFileVersionRecord,
    normalize_project_relative_path,
)
from personagraph.workspace.files.attachments import MAX_ATTACHMENT_BYTES
from personagraph.workspace.files.observation import observe_project_file


class TurnInputFileAuthorityError(RuntimeError):
    """The complete Turn-to-Project file binding could not be proven."""

    code = "turn_input_file_authority_unavailable"

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.code}:{reason}")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ResolvedTurnInputFile:
    """One ordered Turn input resolved to its exact current Project version."""

    attachment_id: str
    ordinal: int
    input_message_id: str
    project_id: str
    file_id: str
    file_version_id: str
    relative_path: str
    authority_path: Path = field(repr=False)
    canonical_path: Path = field(repr=False)
    original_name: str
    media_type: str
    kind: FileKind
    content_sha256: str
    size_bytes: int
    source_mtime_ns: int
    project_file: ProjectFileRecord = field(repr=False)
    project_file_version: ProjectFileVersionRecord = field(repr=False)


def resolve_turn_input_files(
    records: Sequence[Mapping[str, object]],
    *,
    session_id: str,
    turn_id: str,
) -> tuple[ResolvedTurnInputFile, ...]:
    """Resolve and strictly validate every file binding in persisted Turn order.

    ``records`` must be the complete result of Session
    ``list_turn_attachments``.  A missing join row, reordered ordinal, stale
    version, provenance mismatch, receipt mismatch, or unsafe path rejects the
    entire batch.
    """

    _require_scope_identifier(session_id, "session_id")
    _require_scope_identifier(turn_id, "turn_id")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        _fail("malformed_record_set")

    database = current_document_database()
    if database is None:
        _fail("project_context_missing")

    resolved: list[ResolvedTurnInputFile] = []
    attachment_ids: set[str] = set()
    file_bindings: set[tuple[str, str, str]] = set()
    input_message_id: str | None = None
    file_authority = WorkspaceFileAuthority(database)

    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("malformed_record")
        try:
            attachment_id = _required_text(record, "attachment_id", maximum=160)
            record_session_id = _required_text(record, "session_id", maximum=256)
            record_turn_id = _required_text(record, "turn_id", maximum=256)
            record_input_message_id = _required_text(
                record,
                "input_message_id",
                maximum=256,
            )
            original_name = _required_text(record, "original_name", maximum=1024)
            stored_rel_path = _required_text(
                record,
                "stored_rel_path",
                maximum=1024,
            )
            media_type = _required_text(record, "media_type", maximum=256)
            kind = _required_kind(record)
            origin = _required_text(record, "origin", maximum=64)
            _required_text(record, "bound_at", maximum=80)
            binding_ordinal = _required_non_negative_int(
                record,
                "binding_ordinal",
            )
            input_file_ordinal = _required_non_negative_int(
                record,
                "input_file_ordinal",
            )
            attachment_triple = (
                _required_text(record, "project_id", maximum=128),
                _required_text(record, "file_id", maximum=128),
                _required_text(record, "file_version_id", maximum=128),
            )
            input_triple = (
                _required_text(record, "input_project_id", maximum=128),
                _required_text(record, "input_file_id", maximum=128),
                _required_text(record, "input_file_version_id", maximum=128),
            )
            receipt_size = _required_non_negative_int(record, "size_bytes")
            receipt_sha256 = _required_sha256(record, "content_hash")
        except TurnInputFileAuthorityError:
            raise
        except Exception:
            _fail("malformed_record")

        if record_session_id != session_id or record_turn_id != turn_id:
            _fail("turn_scope_mismatch")
        if origin != FileSource.USER_UPLOAD.value:
            _fail("untrusted_attachment_origin")
        if binding_ordinal != ordinal or input_file_ordinal != ordinal:
            _fail("binding_order_mismatch")
        if attachment_triple != input_triple:
            _fail("input_file_ref_mismatch")
        if attachment_triple[0] != database.project_id:
            _fail("project_mismatch")
        if input_message_id is None:
            input_message_id = record_input_message_id
        elif input_message_id != record_input_message_id:
            _fail("input_message_mismatch")
        if attachment_id in attachment_ids or attachment_triple in file_bindings:
            _fail("duplicate_binding")
        attachment_ids.add(attachment_id)
        file_bindings.add(attachment_triple)

        try:
            pair = file_authority.get_file_with_version(
                attachment_triple[1],
                attachment_triple[2],
            )
        except Exception:
            _fail("project_file_lookup_failed")
        if pair is None:
            _fail("project_file_version_missing")
        project_file, project_file_version = pair
        if (
            project_file.project_id != database.project_id
            or project_file.file_id != attachment_triple[1]
            or project_file_version.file_id != project_file.file_id
            or project_file_version.file_version_id != attachment_triple[2]
        ):
            _fail("project_file_identity_mismatch")
        if (
            project_file.source is not FileSource.USER_UPLOAD
            or project_file_version.source is not FileSource.USER_UPLOAD
        ):
            _fail("project_file_source_mismatch")
        if project_file.current_version_id != project_file_version.file_version_id:
            _fail("bound_version_not_current")
        if project_file.relative_path != stored_rel_path:
            _fail("registered_path_mismatch")
        if (
            project_file_version.size_bytes != receipt_size
            or not hmac.compare_digest(
                project_file_version.content_sha256,
                receipt_sha256,
            )
        ):
            _fail("session_receipt_mismatch")
        if (
            isinstance(project_file.observed_mtime_ns, bool)
            or not isinstance(project_file.observed_mtime_ns, int)
            or project_file.observed_mtime_ns < 0
        ):
            _fail("project_file_version_incomplete")

        authority_path, canonical_path = _resolve_registered_regular_file(
            database.project_root,
            project_file.relative_path,
        )
        try:
            disk_observation = observe_project_file(
                database.project_root,
                project_file.relative_path,
                max_bytes=MAX_ATTACHMENT_BYTES,
            )
        except ProjectFileSizeLimitError:
            _fail("file_too_large")
        except ProjectFileChangedError:
            _fail("file_content_changed")
        except ProjectFilePathError:
            _fail("unsafe_project_path")
        if (
            disk_observation.size_bytes != project_file_version.size_bytes
            or disk_observation.size_bytes != receipt_size
            or not hmac.compare_digest(
                disk_observation.content_sha256,
                project_file_version.content_sha256,
            )
            or not hmac.compare_digest(
                disk_observation.content_sha256,
                receipt_sha256,
            )
        ):
            _fail("file_content_changed")
        resolved.append(
            ResolvedTurnInputFile(
                attachment_id=attachment_id,
                ordinal=ordinal,
                input_message_id=record_input_message_id,
                project_id=project_file.project_id,
                file_id=project_file.file_id,
                file_version_id=project_file_version.file_version_id,
                relative_path=project_file.relative_path,
                authority_path=authority_path,
                canonical_path=canonical_path,
                original_name=original_name,
                media_type=media_type,
                kind=kind,
                content_sha256=project_file_version.content_sha256,
                size_bytes=project_file_version.size_bytes,
                source_mtime_ns=project_file.observed_mtime_ns,
                project_file=project_file,
                project_file_version=project_file_version,
            )
        )

    return tuple(resolved)


def _resolve_registered_regular_file(
    project_root: Path,
    relative_path: str,
) -> tuple[Path, Path]:
    """Prove that the registered lexical path is the same regular file."""

    try:
        normalized_path = normalize_project_relative_path(relative_path)
        root = Path(os.path.abspath(project_root.expanduser()))
        root_status = root.lstat()
        if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
            _fail("unsafe_project_path")
        canonical_root = root.resolve(strict=True)
        authority_path = root.joinpath(*normalized_path.split("/"))
        current = root
        components = normalized_path.split("/")
        for index, component in enumerate(components):
            current = current / component
            observed = current.lstat()
            if stat.S_ISLNK(observed.st_mode):
                _fail("unsafe_project_path")
            if index < len(components) - 1:
                if not stat.S_ISDIR(observed.st_mode):
                    _fail("unsafe_project_path")
            elif not stat.S_ISREG(observed.st_mode):
                _fail("unsafe_project_path")
        canonical_path = authority_path.resolve(strict=True)
        canonical_relative = canonical_path.relative_to(canonical_root)
        if (
            not canonical_relative.parts
            or canonical_relative.as_posix() != normalized_path
        ):
            _fail("unsafe_project_path")
    except TurnInputFileAuthorityError:
        raise
    except (OSError, ProjectFilePathError, ValueError):
        _fail("unsafe_project_path")
    return authority_path, canonical_path


def _required_text(
    record: Mapping[str, object],
    key: str,
    *,
    maximum: int,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail("malformed_record")
    return value


def _required_non_negative_int(
    record: Mapping[str, object],
    key: str,
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("malformed_record")
    return value


def _required_sha256(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("malformed_record")
    return value


def _required_kind(record: Mapping[str, object]) -> FileKind:
    value = record.get("kind")
    if not isinstance(value, str):
        _fail("malformed_record")
    try:
        return FileKind(value)
    except ValueError:
        _fail("malformed_record")


def _require_scope_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail(f"invalid_{label}")


def _fail(reason: str) -> NoReturn:
    raise TurnInputFileAuthorityError(reason) from None


__all__ = [
    "ResolvedTurnInputFile",
    "TurnInputFileAuthorityError",
    "resolve_turn_input_files",
]
