"""Lane-neutral Host authority for mounted visual document units.

This module owns only physical source validation, page-manifest binding, renderable
unit construction, and currentness checks.  It deliberately does not know about
L1, AuxiliaryGraph, planning resources, prompt projections, or model execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from ...input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
    DocumentNonTextUnit,
)
from ...input_processing.files import fingerprint_file
from personagraph.workspace.documents import application as docstore
from ...input_processing.vision.contracts import PixelSize, VisionPurpose
from ...configuration.paths import deny_reason
from .visual_tool_boundary import VisualUnitRef


MAX_MOUNTED_VISUAL_PLANNING_RESOURCES = 1_000_000


class MountedVisualResourceFormat(StrEnum):
    """Document formats understood by the mounted-visual Host boundary."""

    PDF = "pdf"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    TXT = "txt"
    MD = "md"
    MARKDOWN = "markdown"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"


_RENDERABLE_FORMATS = frozenset(
    {
        MountedVisualResourceFormat.PDF,
        MountedVisualResourceFormat.JPG,
        MountedVisualResourceFormat.JPEG,
        MountedVisualResourceFormat.PNG,
    }
)


class MountedVisualPlanningAuthorityError(RuntimeError):
    """The mounted visual generation is unavailable, stale, or malformed."""

    code = "mounted_visual_planning_authority_unavailable"


class MountedVisualPlanningProjectionLimitExceeded(
    MountedVisualPlanningAuthorityError
):
    """The complete mounted visual surface exceeded its explicit Host limit."""

    code = "mounted_visual_planning_projection_limit_exceeded"

    def __init__(
        self,
        *,
        observed_visual_unit_count: int,
        maximum_visual_units: int,
        projection_sha256: str,
    ) -> None:
        if (
            isinstance(maximum_visual_units, bool)
            or not isinstance(maximum_visual_units, int)
            or maximum_visual_units < 1
        ):
            raise ValueError("visual projection limit is invalid")
        if (
            isinstance(observed_visual_unit_count, bool)
            or not isinstance(observed_visual_unit_count, int)
            or observed_visual_unit_count <= maximum_visual_units
        ):
            raise ValueError("visual projection overflow count is invalid")
        if len(projection_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in projection_sha256
        ):
            raise ValueError("visual projection overflow requires a canonical hash")
        self.observed_visual_unit_count = observed_visual_unit_count
        self.maximum_visual_units = maximum_visual_units
        self.projection_sha256 = projection_sha256
        super().__init__(
            "mounted visual count exceeds the complete planning projection"
        )


@dataclass(frozen=True, slots=True)
class MountedVisualDocumentSource:
    """Private current-generation input used to freeze visual units."""

    session_id: str
    document_ordinal: int
    document_alias: str
    document_id: str
    document_version: str
    source_sha256: str
    resource_format: MountedVisualResourceFormat
    private_path: str

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.document_alias.strip():
            raise ValueError("visual source requires a Session and document alias")
        if self.document_ordinal < 1:
            raise ValueError("visual source ordinal must be positive")
        if not self.document_id.strip() or not self.document_version.strip():
            raise ValueError("visual source requires exact Document identities")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_sha256
        ):
            raise ValueError("visual source requires a canonical source hash")
        if not isinstance(self.resource_format, MountedVisualResourceFormat):
            raise ValueError("visual source format is invalid")
        if not self.private_path.strip():
            raise ValueError("visual source path must remain available to the Host")


@dataclass(frozen=True, slots=True)
class FrozenMountedVisualBinding:
    """An exact mounted visual generation with no lane-specific projection."""

    ordinal: int
    session_id: str = field(repr=False)
    parent_document_alias: str
    parent_document_id: str = field(repr=False)
    parent_document_version: str = field(repr=False)
    page_manifest_sha256: str = field(repr=False)
    source_format: MountedVisualResourceFormat = field(repr=False)
    source_unit: DocumentNonTextUnit = field(repr=False)
    visual_unit: VisualUnitRef = field(repr=False)
    purpose: VisionPurpose

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("mounted visual ordinal must be positive")
        if not self.session_id.strip() or not self.parent_document_alias.strip():
            raise ValueError("mounted visual binding requires a Session and alias")
        if not self.parent_document_id.strip() or not self.parent_document_version.strip():
            raise ValueError("mounted visual binding requires exact Document identities")
        if len(self.page_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.page_manifest_sha256
        ):
            raise ValueError("visual binding requires a page-manifest hash")
        if not isinstance(self.source_format, MountedVisualResourceFormat):
            raise ValueError("mounted visual source format is invalid")
        if (
            self.source_unit.unit_id != self.visual_unit.unit_id
            or self.source_unit.kind is not self.visual_unit.kind
            or self.source_unit.locator != self.visual_unit.locator
        ):
            raise ValueError("mounted visual source and render unit differ")
        if self.purpose not in self.visual_unit.allowed_purposes:
            raise ValueError("visual purpose is not allowed for this unit kind")


def freeze_mounted_visual_bindings(
    *,
    session_id: str,
    sources: Sequence[MountedVisualDocumentSource],
    maximum_visual_units: int | None = None,
    source_pages: Sequence[int] | None = None,
    include_all_manifest_units: bool = False,
) -> tuple[FrozenMountedVisualBinding, ...]:
    """Freeze native visual units and optional exact-page fallback units."""

    if not isinstance(include_all_manifest_units, bool):
        raise TypeError("include_all_manifest_units must be a boolean")
    if maximum_visual_units is not None and (
        isinstance(maximum_visual_units, bool)
        or not isinstance(maximum_visual_units, int)
        or not 1
        <= maximum_visual_units
        <= MAX_MOUNTED_VISUAL_PLANNING_RESOURCES
    ):
        raise ValueError("maximum_visual_units is outside the optional diagnostic limit")
    if any(source.session_id != session_id for source in sources):
        raise MountedVisualPlanningAuthorityError(
            "mounted visual sources belong to another Session"
        )
    ordinals = tuple(source.document_ordinal for source in sources)
    document_ids = tuple(source.document_id for source in sources)
    if ordinals != tuple(range(1, len(sources) + 1)):
        raise MountedVisualPlanningAuthorityError(
            "mounted visual sources must use Document source order"
        )
    if len(document_ids) != len(set(document_ids)):
        raise MountedVisualPlanningAuthorityError(
            "mounted visual Document identities must be unique"
        )
    selected_pages = _normalize_visual_source_pages(source_pages)

    candidates: list[
        tuple[MountedVisualDocumentSource, str, DocumentNonTextUnit]
    ] = []
    for source in sources:
        try:
            authority = docstore.get_current_document_page_authority(
                source.document_id,
                expected_version_id=source.document_version,
                session_id=session_id,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise MountedVisualPlanningAuthorityError(
                "mounted visual page authority is unavailable"
            ) from exc
        if authority is None:
            continue
        if (
            authority.document_id != source.document_id
            or authority.document_version_id != source.document_version
            or authority.source_sha256 != source.source_sha256
        ):
            raise MountedVisualPlanningAuthorityError(
                "mounted visual page authority differs from its Document generation"
            )
        if selected_pages is not None and any(
            page > authority.page_manifest.physical_page_count
            for page in selected_pages
        ):
            raise MountedVisualPlanningAuthorityError(
                "selected visual page is outside the Document generation"
            )
        seen: dict[str, DocumentNonTextUnit] = {}
        for page in authority.page_manifest.pages:
            if (
                selected_pages is not None
                and not include_all_manifest_units
                and page.page_number not in selected_pages
            ):
                continue
            for unit in page.nontext_units:
                if not unit.requires_visual_read:
                    continue
                existing = seen.setdefault(unit.unit_id, unit)
                if existing != unit:
                    raise MountedVisualPlanningAuthorityError(
                        "mounted visual unit identity is ambiguous"
                    )
        if (
            selected_pages is not None
            and source.resource_format is MountedVisualResourceFormat.PDF
        ):
            pages_with_visuals = {
                page
                for unit in seen.values()
                for page in visual_unit_render_pages(unit)
            }
            for page_number in sorted(selected_pages - pages_with_visuals):
                fallback = derive_page_visual_fallback_unit(
                    document_id=source.document_id,
                    document_version=source.document_version,
                    page_manifest_sha256=authority.page_manifest.manifest_sha256,
                    page_number=page_number,
                )
                seen[fallback.unit_id] = fallback
        if source.resource_format not in _RENDERABLE_FORMATS:
            continue
        for unit in sorted(
            seen.values(),
            key=lambda item: (item.source_pages, item.unit_id),
        ):
            candidates.append(
                (source, authority.page_manifest.manifest_sha256, unit)
            )

    if maximum_visual_units is not None and len(candidates) > maximum_visual_units:
        for source in sources:
            if any(candidate[0] == source for candidate in candidates):
                _observe_current_visual_source(source)
        raise MountedVisualPlanningProjectionLimitExceeded(
            observed_visual_unit_count=len(candidates),
            maximum_visual_units=maximum_visual_units,
            projection_sha256=_sha256_value(
                {
                    "schema_version": "mounted-visual-overflow-projection-v1",
                    "maximum_visual_units": maximum_visual_units,
                    "candidates": [
                        {
                            "document_ordinal": source.document_ordinal,
                            "document_version": source.document_version,
                            "source_sha256": source.source_sha256,
                            "page_manifest_sha256": manifest_sha256,
                            "unit": unit.to_dict(),
                        }
                        for source, manifest_sha256, unit in candidates
                    ],
                }
            ),
        )

    source_observations: dict[str, tuple[Path, int]] = {}
    source_sizes: dict[tuple[str, int], PixelSize] = {}
    bindings: list[FrozenMountedVisualBinding] = []
    for ordinal, (source, manifest_sha256, unit) in enumerate(candidates, start=1):
        observed = source_observations.get(source.document_id)
        if observed is None:
            observed = _observe_current_visual_source(source)
            source_observations[source.document_id] = observed
        path, byte_count = observed
        page_number = unit.locator.page or unit.source_pages[0]
        source_size_key = (source.document_id, page_number)
        source_size = source_sizes.get(source_size_key)
        if source_size is None:
            source_size = _source_pixel_size(
                path,
                source.resource_format,
                page_number=page_number,
            )
            source_sizes[source_size_key] = source_size
        visual_ref = VisualUnitRef(
            unit_id=unit.unit_id,
            kind=unit.kind,
            image_path=str(path),
            source_sha256=source.source_sha256,
            image_sha256=source.source_sha256,
            locator=unit.locator,
            mime_type=_vision_media_type(source.resource_format),
            pixel_size=_visual_pixel_size(
                unit,
                fallback=source_size,
                use_locator_geometry=(
                    source.resource_format is MountedVisualResourceFormat.PDF
                ),
            ),
            byte_count=byte_count,
        )
        bindings.append(
            FrozenMountedVisualBinding(
                ordinal=ordinal,
                session_id=session_id,
                parent_document_alias=source.document_alias,
                parent_document_id=source.document_id,
                parent_document_version=source.document_version,
                page_manifest_sha256=manifest_sha256,
                source_format=source.resource_format,
                source_unit=unit,
                visual_unit=visual_ref,
                purpose=_default_purpose(unit.kind),
            )
        )
    return tuple(bindings)


def derive_page_visual_fallback_unit(
    *,
    document_id: str,
    document_version: str,
    page_manifest_sha256: str,
    page_number: int,
) -> DocumentNonTextUnit:
    """Derive a rebuildable full-page unit without persisting an overlay."""

    identity = _sha256_value(
        {
            "schema_version": "mounted-page-visual-fallback-v1",
            "document_id": document_id,
            "document_version": document_version,
            "page_manifest_sha256": page_manifest_sha256,
            "page_number": page_number,
        }
    )
    return DocumentNonTextUnit(
        unit_id="page_visual_" + identity[:32],
        kind=DocumentNonTextKind.PAGE_VISUAL,
        source_pages=(page_number,),
        locator=DocumentLocator(page=page_number),
        requires_visual_read=True,
    )


def visual_unit_render_pages(unit: DocumentNonTextUnit) -> tuple[int, ...]:
    """Return only pages the frozen unit can actually render."""

    if unit.locator.page is not None:
        return (unit.locator.page,)
    return tuple(unit.source_pages)


def mounted_visual_binding_is_current(binding: FrozenMountedVisualBinding) -> bool:
    """Reproduce a frozen visual generation before model-callable I/O."""

    if not isinstance(binding, FrozenMountedVisualBinding):
        return False
    return _visual_binding_is_current(binding)


def _normalize_visual_source_pages(
    source_pages: Sequence[int] | None,
) -> frozenset[int] | None:
    if source_pages is None:
        return None
    if isinstance(source_pages, (str, bytes)):
        raise ValueError("source_pages must be a sequence of positive integers")
    pages = tuple(source_pages)
    if len(pages) != len(set(pages)) or any(
        isinstance(page, bool) or not isinstance(page, int) or page < 1
        for page in pages
    ):
        raise ValueError("source_pages must contain unique positive integers")
    return frozenset(pages)


def _observe_current_visual_source(
    source: MountedVisualDocumentSource,
) -> tuple[Path, int]:
    try:
        path = Path(source.private_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise MountedVisualPlanningAuthorityError(
            "mounted visual source path is invalid"
        ) from exc
    if deny_reason(path) is not None or not path.is_file():
        raise MountedVisualPlanningAuthorityError(
            "mounted visual source path is unavailable"
        )
    try:
        fingerprint = fingerprint_file(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MountedVisualPlanningAuthorityError(
            "mounted visual source bytes are unavailable"
        ) from exc
    if fingerprint.sha256 != source.source_sha256:
        raise MountedVisualPlanningAuthorityError(
            "mounted visual source changed after Document ingest"
        )
    return path, fingerprint.size_bytes


def _source_pixel_size(
    path: Path,
    resource_format: MountedVisualResourceFormat,
    *,
    page_number: int,
) -> PixelSize:
    if resource_format is MountedVisualResourceFormat.PDF:
        try:
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(str(path))
            try:
                if not 1 <= page_number <= len(document):
                    raise ValueError("visual page is outside the mounted PDF")
                page = document[page_number - 1]
                try:
                    width, height = page.get_size()
                finally:
                    page.close()
            finally:
                document.close()
        except Exception as exc:
            raise MountedVisualPlanningAuthorityError(
                "mounted PDF geometry is unavailable"
            ) from exc
        return PixelSize(max(1, math.ceil(width)), max(1, math.ceil(height)))
    try:
        from PIL import Image

        with Image.open(path) as opened:
            return PixelSize(int(opened.width), int(opened.height))
    except Exception as exc:
        raise MountedVisualPlanningAuthorityError(
            "mounted image geometry is unavailable"
        ) from exc


def _visual_binding_is_current(binding: FrozenMountedVisualBinding) -> bool:
    if not docstore.is_mounted(binding.parent_document_id, binding.session_id):
        return False
    try:
        authority = docstore.get_current_document_page_authority(
            binding.parent_document_id,
            expected_version_id=binding.parent_document_version,
            session_id=binding.session_id,
        )
        documents = tuple(
            document
            for document in docstore.mounted_docs(binding.session_id)
            if str(document.get("id") or "") == binding.parent_document_id
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if (
        authority is None
        or authority.page_manifest.manifest_sha256 != binding.page_manifest_sha256
        or authority.source_sha256 != binding.visual_unit.source_sha256
        or len(documents) != 1
    ):
        return False
    try:
        current_source = MountedVisualDocumentSource(
            session_id=binding.session_id,
            document_ordinal=1,
            document_alias=binding.parent_document_alias,
            document_id=binding.parent_document_id,
            document_version=binding.parent_document_version,
            source_sha256=binding.visual_unit.source_sha256,
            resource_format=binding.source_format,
            private_path=str(documents[0].get("path") or ""),
        )
    except ValueError:
        return False
    units = {
        unit.unit_id: unit
        for page in authority.page_manifest.pages
        for unit in page.nontext_units
        if unit.requires_visual_read
    }
    unit = units.get(binding.visual_unit.unit_id)
    if (
        unit is None
        and binding.visual_unit.kind is DocumentNonTextKind.PAGE_VISUAL
        and binding.source_format is MountedVisualResourceFormat.PDF
        and binding.visual_unit.locator.page is not None
        and binding.visual_unit.locator.page
        <= authority.page_manifest.physical_page_count
    ):
        unit = derive_page_visual_fallback_unit(
            document_id=binding.parent_document_id,
            document_version=binding.parent_document_version,
            page_manifest_sha256=authority.page_manifest.manifest_sha256,
            page_number=binding.visual_unit.locator.page,
        )
    if unit is None or unit.kind is not binding.visual_unit.kind:
        return False
    try:
        path, byte_count = _observe_current_visual_source(current_source)
    except (MountedVisualPlanningAuthorityError, ValueError):
        return False
    page_number = unit.locator.page or unit.source_pages[0]
    try:
        source_size = _source_pixel_size(
            path,
            binding.source_format,
            page_number=page_number,
        )
    except MountedVisualPlanningAuthorityError:
        return False
    expected = VisualUnitRef(
        unit_id=unit.unit_id,
        kind=unit.kind,
        image_path=str(path),
        source_sha256=binding.visual_unit.source_sha256,
        image_sha256=binding.visual_unit.source_sha256,
        locator=unit.locator,
        mime_type=_vision_media_type(binding.source_format),
        pixel_size=_visual_pixel_size(
            unit,
            fallback=source_size,
            use_locator_geometry=(
                binding.source_format is MountedVisualResourceFormat.PDF
            ),
        ),
        byte_count=byte_count,
    )
    return expected == binding.visual_unit


def _visual_pixel_size(
    unit: DocumentNonTextUnit,
    *,
    fallback: PixelSize,
    use_locator_geometry: bool,
) -> PixelSize:
    bbox = unit.locator.bbox
    if bbox is None or not use_locator_geometry:
        return fallback
    width = max(1, math.ceil(abs(float(bbox[2]) - float(bbox[0]))))
    height = max(1, math.ceil(abs(float(bbox[3]) - float(bbox[1]))))
    return PixelSize(width, height)


def _default_purpose(kind: DocumentNonTextKind) -> VisionPurpose:
    if kind is DocumentNonTextKind.FORMULA:
        return VisionPurpose.FORMULA
    return VisionPurpose.GENERAL


def _vision_media_type(resource_format: MountedVisualResourceFormat) -> str:
    if resource_format in {
        MountedVisualResourceFormat.JPG,
        MountedVisualResourceFormat.JPEG,
    }:
        return "image/jpeg"
    return "image/png"


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    'FrozenMountedVisualBinding',
    "MAX_MOUNTED_VISUAL_PLANNING_RESOURCES",
    'MountedVisualDocumentSource',
    "MountedVisualPlanningAuthorityError",
    "MountedVisualPlanningProjectionLimitExceeded",
    'MountedVisualResourceFormat',
    "derive_page_visual_fallback_unit",
    "freeze_mounted_visual_bindings",
    "mounted_visual_binding_is_current",
    "visual_unit_render_pages",
]
