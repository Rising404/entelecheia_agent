"""Resolve shared File versions without a separate per-Turn identity ledger.

File access owns path and Session authorization. This module only reads an exact
Document page manifest or a selected standalone image. It never parses documents,
renders pages, calls a provider, or grants external disclosure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re

from PIL import Image

from ...input_processing.documents.contracts import DocumentLocator, DocumentNonTextKind
from ...input_processing.vision.contracts import PixelSize, VisionPurpose
from ...workspace.documents import application as docstore
from ...workspace.files.access import AuthorizedFileSource
from .mounted_visual_source_authority import (
    MountedVisualDocumentSource,
    MountedVisualResourceFormat,
    freeze_mounted_visual_bindings,
    visual_unit_render_pages,
)
from .visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef


_PAGE_ID = re.compile(r"pdf_page:([1-9][0-9]{0,7})\Z")
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})


class FileVisualAuthorityError(RuntimeError):
    """A selected shared File version cannot be safely observed."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class FileVisualUnit:
    """Stable pre-render address; a PictureUnit only exists after rendering."""

    visual_unit_id: str
    source_pages: tuple[int, ...]
    kind: str
    allowed_purposes: tuple[str, ...]
    default_purpose: str

    def to_model_dict(self) -> dict[str, object]:
        return {
            "visual_unit_id": self.visual_unit_id,
            "pages": list(self.source_pages),
            "kind": self.kind,
            "allowed_purposes": list(self.allowed_purposes),
            "default_purpose": self.default_purpose,
        }


@dataclass(frozen=True, slots=True)
class FrozenFileVisualSource:
    source: AuthorizedFileSource = field(repr=False)
    session_id: str = field(repr=False)
    catalog: tuple[FileVisualUnit, ...]
    units: tuple[VisualUnitRef, ...] = field(repr=False)
    document_id: str | None = None
    document_version_id: str | None = None
    manifest_sha256: str | None = None
    _revalidate_source: Callable[[AuthorizedFileSource], bool] = field(
        repr=False, compare=False, default=lambda source: False,
    )

    @property
    def snapshot_id(self) -> str:
        """Bind pagination to shared content/manifest, never to a Session ID."""
        encoded = json.dumps({
            "file_id": self.source.file_id,
            "file_version_id": self.source.file_version_id,
            "document_version_id": self.document_version_id,
            "manifest_sha256": self.manifest_sha256,
            "units": [unit.to_model_dict() for unit in self.catalog],
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate_current(self) -> None:
        if self._revalidate_source(self.source) is not True:
            raise FileVisualAuthorityError("file_authority_stale")
        if self.document_id is None:
            return
        current = docstore.get_current_file_document(
            file_id=self.source.file_id,
            file_version_id=self.source.file_version_id,
            session_id=self.session_id,
        )
        if current is None or current.document_version_id != self.document_version_id:
            raise FileVisualAuthorityError("visual_document_stale")
        authority = docstore.get_current_document_page_authority(
            self.document_id,
            expected_version_id=self.document_version_id,
            session_id=self.session_id,
        )
        if (
            authority is None
            or authority.source_sha256 != self.source.fingerprint.sha256
            or authority.page_manifest.manifest_sha256 != self.manifest_sha256
        ):
            raise FileVisualAuthorityError("visual_manifest_stale")

    def resolve(self, visual_unit_id: str) -> tuple[FileVisualUnit, FrozenVisualToolBoundary]:
        self.validate_current()
        for public, unit in zip(self.catalog, self.units, strict=True):
            if public.visual_unit_id == visual_unit_id:
                return public, FrozenVisualToolBoundary(self.session_id, (unit,))
        raise FileVisualAuthorityError("visual_unit_outside_file")


def visual_page_number(visual_unit_id: str) -> int | None:
    match = _PAGE_ID.fullmatch(visual_unit_id)
    return int(match.group(1)) if match is not None else None


def freeze_file_visual_source(
    *,
    session_id: str,
    source: AuthorizedFileSource,
    revalidate_source: Callable[[AuthorizedFileSource], bool],
    source_pages: Sequence[int] | None = None,
) -> FrozenFileVisualSource | None:
    """Locate only the selected file; missing Document preparation stays missing."""
    if revalidate_source(source) is not True:
        raise FileVisualAuthorityError("file_authority_stale")
    if source.media_type in _IMAGE_TYPES:
        if source_pages is not None and any(page != 1 for page in source_pages):
            raise FileVisualAuthorityError("visual_page_outside_file")
        with Image.open(source.canonical_path) as image:
            size = PixelSize(width=image.width, height=image.height)
        unit = VisualUnitRef(
            unit_id=f"{source.file_version_id}:whole_file",
            kind=DocumentNonTextKind.FIGURE,
            image_path=str(source.canonical_path),
            source_sha256=source.fingerprint.sha256,
            image_sha256=source.fingerprint.sha256,
            locator=DocumentLocator(page=1),
            mime_type=source.media_type,
            pixel_size=size,
            byte_count=source.fingerprint.size_bytes,
        )
        frozen = FrozenFileVisualSource(
            source=source,
            session_id=session_id,
            catalog=(_public("whole_file", unit, (1,)),),
            units=(unit,),
            _revalidate_source=revalidate_source,
        )
    elif source.media_type == "application/pdf":
        document = docstore.get_current_file_document(
            file_id=source.file_id,
            file_version_id=source.file_version_id,
            session_id=session_id,
        )
        if document is None:
            return None
        if document.source_sha256 != source.fingerprint.sha256:
            raise FileVisualAuthorityError("visual_document_stale")
        authority = docstore.get_current_document_page_authority(
            document.document_id,
            expected_version_id=document.document_version_id,
            session_id=session_id,
        )
        if authority is None:
            return None
        bindings = freeze_mounted_visual_bindings(
            session_id=session_id,
            sources=(MountedVisualDocumentSource(
                session_id=session_id,
                document_ordinal=1,
                document_alias=document.document_id,
                document_id=document.document_id,
                document_version=document.document_version_id,
                source_sha256=source.fingerprint.sha256,
                resource_format=MountedVisualResourceFormat.PDF,
                private_path=str(source.canonical_path),
            ),),
            source_pages=source_pages,
        )
        catalog = []
        for binding in bindings:
            unit = binding.visual_unit
            identifier = unit.unit_id
            if unit.kind is DocumentNonTextKind.PAGE_VISUAL and unit.locator.page is not None:
                identifier = f"pdf_page:{unit.locator.page}"
            catalog.append(_public(identifier, unit, visual_unit_render_pages(binding.source_unit)))
        frozen = FrozenFileVisualSource(
            source=source,
            session_id=session_id,
            catalog=tuple(catalog),
            units=tuple(binding.visual_unit for binding in bindings),
            document_id=document.document_id,
            document_version_id=document.document_version_id,
            manifest_sha256=authority.page_manifest.manifest_sha256,
            _revalidate_source=revalidate_source,
        )
    else:
        return None
    frozen.validate_current()
    return frozen


def _public(identifier: str, unit: VisualUnitRef, pages: tuple[int, ...]) -> FileVisualUnit:
    purposes = tuple(purpose.value for purpose in unit.allowed_purposes)
    default = (
        VisionPurpose.FORMULA.value
        if unit.kind is DocumentNonTextKind.FORMULA
        else VisionPurpose.GENERAL.value
        if VisionPurpose.GENERAL.value in purposes
        else purposes[0]
    )
    return FileVisualUnit(identifier, pages, unit.kind.value, purposes, default)


__all__ = [
    "FileVisualAuthorityError", "FileVisualUnit", "FrozenFileVisualSource",
    "freeze_file_visual_source", "visual_page_number",
]
