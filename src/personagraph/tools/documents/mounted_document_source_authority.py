"""Tool-source authority for mounted document generations.

This module owns only physical mounted-document selection, DocStore generation and
freshness checks, managed-input admission, private-workspace exclusion, and raw
document/visual bindings. It deliberately does not know about L1, L2 planning
resources, AuxiliaryGraph projections, prompt cards, or model execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from ...workspace.files.attachments import (
    SessionStorageArea,
    classify_session_path,
)
from personagraph.workspace.documents import application as docstore
from ...workspace.binding import (
    ReservedWorkspacePathError,
    is_reserved_workspace_path,
)
from ..visual.mounted_visual_source_authority import (
    FrozenMountedVisualBinding,
    MountedVisualDocumentSource,
    MountedVisualPlanningProjectionLimitExceeded,
    MountedVisualResourceFormat,
    freeze_mounted_visual_bindings,
)


MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES = 32
_SUPPORTED_PROCESSING_COVERAGE = frozenset({"complete", "partial", "rejected"})


class MountedDocumentPlanningAuthorityError(RuntimeError):
    """A mounted Document generation is unavailable, stale, or malformed."""


class MountedDocumentSourceFreshness(StrEnum):
    """Content-free classification of one physical DocStore source check."""

    CURRENT = "current"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class FrozenMountedVisualProjectionLimit:
    """Content-free evidence that the complete visual surface did not fit."""

    observed_visual_unit_count: int
    maximum_visual_units: int
    projection_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_visual_units, bool)
            or not isinstance(self.maximum_visual_units, int)
            or self.maximum_visual_units < 1
        ):
            raise ValueError("visual projection limit is invalid")
        if (
            isinstance(self.observed_visual_unit_count, bool)
            or not isinstance(self.observed_visual_unit_count, int)
            or self.observed_visual_unit_count <= self.maximum_visual_units
        ):
            raise ValueError("visual projection overflow count is invalid")
        if len(self.projection_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.projection_sha256
        ):
            raise ValueError("visual projection limit requires a canonical hash")


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentBinding:
    """Exact current Document generation without a lane-specific projection."""

    ordinal: int
    session_id: str
    resource_alias: str
    document_id: str = field(repr=False)
    document_version: str = field(repr=False)
    source_sha256: str = field(repr=False)
    resource_format: MountedVisualResourceFormat
    processing_status: str
    processing_diagnostic_codes: tuple[str, ...]
    total_chunk_count: int
    file_extension: str
    private_path: str = field(repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.ordinal <= MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES:
            raise ValueError("mounted Document ordinal is outside the Host limit")
        if not self.session_id or len(self.session_id) > 200:
            raise ValueError("session_id must be a bounded durable identity")
        if self.resource_alias != f"mounted_document_{self.ordinal:02d}":
            raise ValueError("mounted Document alias is outside source order")
        if not self.document_id or not self.document_version:
            raise ValueError("mounted Document requires exact generation identities")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("mounted Document requires a canonical source hash")
        if not isinstance(self.resource_format, MountedVisualResourceFormat):
            raise ValueError("mounted Document format is invalid")
        if self.processing_status not in _SUPPORTED_PROCESSING_COVERAGE:
            raise ValueError("mounted Document processing coverage is invalid")
        if self.total_chunk_count < 0:
            raise ValueError("mounted Document chunk count cannot be negative")
        if not self.file_extension or not self.private_path:
            raise ValueError("mounted Document physical source is unavailable")


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentAuthority:
    """Complete raw mounted-document and visual authority for one frozen scope."""

    session_id: str
    task_id: str | None
    bindings: tuple[FrozenMountedDocumentBinding, ...]
    visual_bindings: tuple[FrozenMountedVisualBinding, ...]
    visual_projection_limit: FrozenMountedVisualProjectionLimit | None = None

    def __post_init__(self) -> None:
        if not self.session_id or len(self.session_id) > 200:
            raise ValueError("session_id must be a bounded durable identity")
        if self.task_id is not None and (
            not self.task_id or len(self.task_id) > 200
        ):
            raise ValueError("task_id must be a bounded durable identity")
        if tuple(item.ordinal for item in self.bindings) != tuple(
            range(1, len(self.bindings) + 1)
        ):
            raise ValueError("mounted Document bindings must use source order")
        if any(item.session_id != self.session_id for item in self.bindings):
            raise ValueError("mounted Documents must belong to their Session")
        if tuple(item.ordinal for item in self.visual_bindings) != tuple(
            range(1, len(self.visual_bindings) + 1)
        ):
            raise ValueError("mounted visual bindings must use source order")
        if any(item.session_id != self.session_id for item in self.visual_bindings):
            raise ValueError("mounted visuals must belong to their Session")
        if self.visual_projection_limit is not None and self.visual_bindings:
            raise ValueError(
                "a limited visual projection cannot expose partial visual aliases"
            )


def freeze_mounted_document_authority(
    *,
    session_id: str,
    task_id: str | None = None,
    maximum_documents: int = MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES,
    allowed_managed_document_ids: tuple[str, ...] | None = None,
) -> FrozenMountedDocumentAuthority:
    """Freeze the complete physically admitted mounted Document scope."""

    if not isinstance(maximum_documents, int) or isinstance(maximum_documents, bool):
        raise TypeError("maximum_documents must be an integer")
    if not 1 <= maximum_documents <= MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES:
        raise ValueError("maximum_documents is outside the Host planning limit")
    mounted = _exclude_agent_private_workspace_mounts(
        session_id=session_id,
        mounted=tuple(docstore.mounted_docs(session_id)),
    )
    if allowed_managed_document_ids is not None:
        allowed = _validate_allowed_managed_document_ids(
            allowed_managed_document_ids
        )
        allowed_set = set(allowed)
        observed_allowed: set[str] = set()
        selected: list[dict[str, object]] = []
        for document in mounted:
            document_id = str(document.get("id") or "").strip()
            private_path = str(document.get("path") or "").strip()
            storage_area = (
                classify_session_path(session_id, Path(private_path))
                if private_path
                else None
            )
            if storage_area is SessionStorageArea.OUTPUT:
                continue
            if storage_area is SessionStorageArea.INPUT:
                if document_id in allowed_set:
                    observed_allowed.add(document_id)
                    selected.append(document)
                continue
            if document_id in allowed_set:
                raise MountedDocumentPlanningAuthorityError(
                    "allowed managed Document does not belong to Session input"
                )
            if storage_area is None:
                selected.append(document)
        if observed_allowed != allowed_set:
            raise MountedDocumentPlanningAuthorityError(
                "Task-scoped mounted Document authority is unavailable"
            )
        mounted = tuple(selected)
    if len(mounted) > maximum_documents:
        raise MountedDocumentPlanningAuthorityError(
            "mounted Document count exceeds the complete planning projection"
        )

    bindings: list[FrozenMountedDocumentBinding] = []
    visual_sources: list[MountedVisualDocumentSource] = []
    for ordinal, document in enumerate(mounted, start=1):
        document_id = str(document.get("id") or "").strip()
        try:
            snapshot = docstore.get_mounted_current_document_resource_snapshot(
                document_id,
                session_id=session_id,
                maximum_chunks=1,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document current-generation authority is unavailable"
            ) from exc
        if snapshot is None:
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document disappeared while freezing planning authority"
            )
        try:
            freshness = docstore.check_mounted_document_freshness(
                session_id,
                document_id,
            )
        except Exception as exc:
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document current source could not be verified"
            ) from exc
        if (
            classify_mounted_document_source_freshness(
                freshness,
                document_id=document_id,
                expected_version_id=snapshot.document_version_id,
            )
            is not MountedDocumentSourceFreshness.CURRENT
        ):
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document current source bytes changed or became unavailable"
            )
        try:
            resource_format = MountedVisualResourceFormat(
                snapshot.file_extension.lstrip(".").lower()
            )
            if snapshot.processing_status not in _SUPPORTED_PROCESSING_COVERAGE:
                raise ValueError("unsupported processing coverage")
        except ValueError as exc:
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document format or coverage is unsupported"
            ) from exc
        alias = f"mounted_document_{ordinal:02d}"
        private_path = str(document.get("path") or "").strip()
        if not private_path:
            raise MountedDocumentPlanningAuthorityError(
                "mounted Document private source path is unavailable"
            )
        bindings.append(
            FrozenMountedDocumentBinding(
                ordinal=ordinal,
                session_id=session_id,
                resource_alias=alias,
                document_id=snapshot.document_id,
                document_version=snapshot.document_version_id,
                source_sha256=snapshot.source_sha256,
                resource_format=resource_format,
                processing_status=snapshot.processing_status,
                processing_diagnostic_codes=tuple(
                    snapshot.processing_diagnostic_codes
                ),
                total_chunk_count=snapshot.total_chunk_count,
                file_extension=snapshot.file_extension,
                private_path=private_path,
            )
        )
        visual_sources.append(
            MountedVisualDocumentSource(
                session_id=session_id,
                document_ordinal=ordinal,
                document_alias=alias,
                document_id=snapshot.document_id,
                document_version=snapshot.document_version_id,
                source_sha256=snapshot.source_sha256,
                resource_format=resource_format,
                private_path=private_path,
            )
        )

    visual_projection_limit = None
    try:
        visual_bindings = freeze_mounted_visual_bindings(
            session_id=session_id,
            sources=tuple(visual_sources),
        )
    except MountedVisualPlanningProjectionLimitExceeded as exc:
        visual_bindings = ()
        visual_projection_limit = FrozenMountedVisualProjectionLimit(
            observed_visual_unit_count=exc.observed_visual_unit_count,
            maximum_visual_units=exc.maximum_visual_units,
            projection_sha256=exc.projection_sha256,
        )
    return FrozenMountedDocumentAuthority(
        session_id=session_id,
        task_id=task_id,
        bindings=tuple(bindings),
        visual_bindings=visual_bindings,
        visual_projection_limit=visual_projection_limit,
    )


def recover_managed_input_document_ids(
    *,
    session_id: str,
    allowed_document_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Intersect exact candidate identities with current managed-input mounts."""

    if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
        raise ValueError("session_id must be a bounded durable identity")
    if not isinstance(allowed_document_ids, tuple):
        raise TypeError("allowed_document_ids must be a tuple")
    allowed = set(allowed_document_ids)
    managed_mounted_ids: set[str] = set()
    for document in docstore.mounted_docs(session_id):
        document_id = str(document.get("id") or "").strip()
        private_path = str(document.get("path") or "").strip()
        if not document_id or not private_path:
            continue
        if (
            classify_session_path(session_id, Path(private_path))
            is SessionStorageArea.INPUT
        ):
            managed_mounted_ids.add(document_id)
    return tuple(sorted(allowed & managed_mounted_ids))


def classify_mounted_document_source_freshness(
    report: object,
    *,
    document_id: str,
    expected_version_id: str,
) -> MountedDocumentSourceFreshness:
    """Classify a private DocStore report without projecting its identity."""

    if not isinstance(report, Mapping):
        return MountedDocumentSourceFreshness.STALE
    documents = report.get("documents")
    if not isinstance(documents, list):
        return MountedDocumentSourceFreshness.STALE
    matching = tuple(
        item
        for item in documents
        if isinstance(item, Mapping)
        and str(item.get("doc_id") or "") == document_id
    )
    if len(matching) != 1:
        return MountedDocumentSourceFreshness.STALE
    document = matching[0]
    if document.get("status") in {"not_mounted", "document_not_found"}:
        return MountedDocumentSourceFreshness.UNAVAILABLE
    if (
        report.get("ok") is True
        and report.get("status") == "verified_current"
        and document.get("status") == "verified_current"
        and str(document.get("version_id") or "") == expected_version_id
    ):
        return MountedDocumentSourceFreshness.CURRENT
    return MountedDocumentSourceFreshness.STALE


def _validate_allowed_managed_document_ids(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError("allowed_managed_document_ids must be a tuple")
    if len(values) > MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES:
        raise ValueError("allowed managed Document scope exceeds the Host limit")
    if len(values) != len(set(values)):
        raise ValueError("allowed managed Document ids must be unique")
    if any(
        not isinstance(value, str) or not value or len(value) > 200
        for value in values
    ):
        raise ValueError("allowed managed Document ids must be bounded identities")
    return values


def _exclude_agent_private_workspace_mounts(
    *,
    session_id: str,
    mounted: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    workspace_root = _current_bound_workspace_root(session_id)
    if workspace_root is None:
        return mounted
    return tuple(
        document
        for document in mounted
        if not _is_agent_private_workspace_mount(
            workspace_root=workspace_root,
            document=document,
        )
    )


def _current_bound_workspace_root(session_id: str) -> Path | None:
    try:
        from ...session import store as session_store

        session = session_store.get_session(session_id)
        working_dir = None if session is None else session.get("working_dir")
        if not isinstance(working_dir, str) or not working_dir.strip():
            return None
        root = Path(working_dir).expanduser().resolve(strict=True)
        return root if root.is_dir() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _is_agent_private_workspace_mount(
    *,
    workspace_root: Path,
    document: dict[str, object],
) -> bool:
    private_path = str(document.get("path") or "").strip()
    if not private_path:
        return False
    try:
        return is_reserved_workspace_path(workspace_root, private_path)
    except ReservedWorkspacePathError as exc:
        return exc.code != "outside_bound_root"


__all__ = [
    "MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES",
    'FrozenMountedDocumentAuthority',
    'FrozenMountedDocumentBinding',
    'FrozenMountedVisualProjectionLimit',
    "MountedDocumentPlanningAuthorityError",
    'MountedDocumentSourceFreshness',
    "classify_mounted_document_source_freshness",
    "freeze_mounted_document_authority",
    "recover_managed_input_document_ids",
]
