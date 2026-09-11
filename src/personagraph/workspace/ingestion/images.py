"""按明确选择登记图片源；像素观察与外发由视觉工具另行负责。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from personagraph.input_processing.files import detect_type
from personagraph.workspace.files import ProjectFileError, WorkspaceFileAuthority
from personagraph.workspace.files.access import AuthorizedFileSource
from personagraph.workspace.storage.database import DocumentDatabase
from .contracts import FilePreparationResult, FilePreparationStatus


_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})


def check_image_file(
    *,
    source: AuthorizedFileSource,
    revalidate_source: Callable[[AuthorizedFileSource], bool],
) -> FilePreparationResult:
    """只查当前源是否可交给视觉工具，不登记、不解析、也不声称已有语义。"""
    rejected = _check_source(source, revalidate_source)
    if rejected is not None:
        return rejected
    if source.file_id is not None and source.file_version_id is None:
        return FilePreparationResult(
            status=FilePreparationStatus.STALE,
            file_id=source.file_id,
            reason_code="file_content_changed",
        )
    return FilePreparationResult(
        status=FilePreparationStatus.PENDING,
        file_id=source.file_id,
        file_version_id=source.file_version_id,
        reason_code=(
            "image_visual_ready"
            if source.file_id is not None and source.file_version_id is not None
            else "image_not_prepared"
        ),
    )


def prepare_image_file(
    *,
    source: AuthorizedFileSource,
    database: DocumentDatabase,
    revalidate_source: Callable[[AuthorizedFileSource], bool],
) -> FilePreparationResult:
    """显式登记单张图片的 File 身份，不启动文本摄取任务或视觉模型。"""
    rejected = _check_source(source, revalidate_source)
    if rejected is not None:
        return rejected
    if (
        source.project_id != database.project_id
        or Path(source.canonical_path) != database.project_root / source.relative_path
    ):
        return _blocked("image_project_binding_mismatch")
    try:
        registered = WorkspaceFileAuthority(database).ensure_current_path(
            source.relative_path,
            source=source.origin,
            file_id=source.file_id,
            media_type=source.media_type,
        )
    except (OSError, ValueError, ProjectFileError):
        return _blocked("file_registration_unavailable")
    if (
        registered.version.content_sha256 != source.fingerprint.sha256
        or registered.version.size_bytes != source.fingerprint.size_bytes
        or registered.file.observed_mtime_ns != source.fingerprint.mtime_ns
    ):
        return FilePreparationResult(
            status=FilePreparationStatus.STALE,
            reason_code="frozen_source_mismatch",
        )
    registered_source = replace(
        source,
        file_id=registered.file.file_id,
        file_version_id=registered.version.file_version_id,
    )
    return check_image_file(
        source=registered_source,
        revalidate_source=revalidate_source,
    )


def _check_source(source, revalidate_source) -> FilePreparationResult | None:
    if revalidate_source(source) is not True:
        return _blocked("file_authority_denied")
    if source.media_type not in _IMAGE_MEDIA_TYPES:
        return _blocked("image_format_unsupported")
    try:
        with Path(source.canonical_path).open("rb") as stream:
            detected = detect_type(stream.read(8192), source.file_name)
    except OSError:
        return _blocked("image_source_unavailable")
    if detected.media_type != source.media_type:
        return _blocked("image_source_type_mismatch")
    if revalidate_source(source) is not True:
        return _blocked("file_authority_denied")
    return None


def _blocked(reason: str) -> FilePreparationResult:
    return FilePreparationResult(status=FilePreparationStatus.BLOCKED, reason_code=reason)


__all__ = ["check_image_file", "prepare_image_file"]
