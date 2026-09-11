"""Project frozen mounted Host resources into L2 planning authority.

The document Tool source freezes physical document and visual generations without
knowing about L2. This module owns the L2 aliases, planning resources, authority
anchors, prompt source cards, perception requests, and read-port composition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthorityProjection,
    PlanningAuthoritySnapshot,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    planning_document_group_alias,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceFormat,
    PlanningResourcePerceptionRequest,
    PlanningResourceReadRequest,
)
from personagraph.l2.task_graph.contracts import InSessionTaskSourceAnchor
from personagraph.tools.documents.mounted_document_source_authority import (
    MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES,
    FrozenMountedDocumentBinding,
    MountedDocumentPlanningAuthorityError,
    freeze_mounted_document_authority,
    recover_managed_input_document_ids,
)
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
)

from .host_primitive_controller import (
    AuxiliaryHostPrimitiveDispatchContext,
)
from .mounted_document_resource_read_port import (
    MountedDocumentPlanningResourceReadPort,
)
from .mounted_visual_resource import (
    FrozenMountedVisualPlanningBinding,
    MOUNTED_VISUAL_READ_CAPABILITY,
    MountedDocumentAndVisualPlanningResourceReadPort,
    MountedVisualPlanningResourceReadPort,
    build_mounted_visual_perception_request,
    project_mounted_visual_planning_bindings,
    visual_binding_private_payload,
)


MAX_MOUNTED_DOCUMENT_PLANNING_RESOURCES = (
    MAX_MOUNTED_DOCUMENT_AUTHORITY_RESOURCES
)
MOUNTED_DOCUMENT_READ_CAPABILITY = "mounted_document_read"
_PENDING_AUTHORITY_SNAPSHOT_ID = "pending_authority_snapshot"
_MEDIA_TYPES = {
    PlanningResourceFormat.PDF: "application/pdf",
    PlanningResourceFormat.JPG: "image/jpeg",
    PlanningResourceFormat.JPEG: "image/jpeg",
    PlanningResourceFormat.PNG: "image/png",
    PlanningResourceFormat.TXT: "text/plain",
    PlanningResourceFormat.MD: "text/markdown",
    PlanningResourceFormat.MARKDOWN: "text/markdown",
    PlanningResourceFormat.DOC: "application/msword",
    PlanningResourceFormat.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    PlanningResourceFormat.PPT: "application/vnd.ms-powerpoint",
    PlanningResourceFormat.PPTX: (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
}


@dataclass(frozen=True, slots=True)
class FrozenMountedVisualPlanningProjectionLimit:
    """Content-free evidence that the visual alias set could not be complete."""

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
class FrozenMountedDocumentPlanningBinding:
    ordinal: int
    resource: FrozenPlanningResource
    frozen_document: FrozenMountedDocument
    authority_anchor: PlanningAuthorityAnchor
    source_card: PlanningAuthoritySourceCard

    def __post_init__(self) -> None:
        if not 1 <= self.ordinal <= MAX_MOUNTED_DOCUMENT_PLANNING_RESOURCES:
            raise ValueError("mounted Document ordinal is outside the Host limit")
        if self.resource.resource_alias != self.authority_anchor.projection_alias:
            raise ValueError("mounted resource and authority anchor aliases differ")
        if self.source_card.alias != self.resource.resource_alias:
            raise ValueError("mounted resource and prompt card aliases differ")
        if not _resource_matches_frozen_document(
            self.resource,
            self.frozen_document,
        ):
            raise ValueError("mounted resource and frozen document identities differ")
        if (
            self.authority_anchor.authority_class
            is not PlanningAuthorityClass.EVIDENCE
            or self.source_card.authority_class
            is not PlanningAuthorityClass.EVIDENCE
        ):
            raise ValueError("mounted Documents may only contribute evidence authority")
        if self.authority_anchor.projection_sha256 != self.source_card.projection_sha256:
            raise ValueError("mounted authority projection hashes differ")
        if (
            self.authority_anchor.freshness_binding_sha256
            != self.frozen_document.freshness_binding_sha256
        ):
            raise ValueError("mounted frozen document freshness binding differs")


@dataclass(frozen=True, slots=True)
class FrozenMountedDocumentPlanningAuthority:
    session_id: str
    task_id: str | None
    bindings: tuple[FrozenMountedDocumentPlanningBinding, ...]
    visual_bindings: tuple[FrozenMountedVisualPlanningBinding, ...]
    scope_snapshot_sha256: str
    visual_projection_limit: FrozenMountedVisualPlanningProjectionLimit | None = None

    def __post_init__(self) -> None:
        if not self.session_id or len(self.session_id) > 200:
            raise ValueError("session_id must be a bounded durable identity")
        if self.task_id is not None and (
            not self.task_id or len(self.task_id) > 200
        ):
            raise ValueError("task_id must be a bounded durable identity")
        aliases = tuple(item.resource.resource_alias for item in self.bindings)
        if len(aliases) != len(set(aliases)):
            raise ValueError("mounted Document aliases must be unique")
        visual_aliases = tuple(
            item.resource.resource_alias for item in self.visual_bindings
        )
        if (
            len(visual_aliases) != len(set(visual_aliases))
            or set(aliases) & set(visual_aliases)
        ):
            raise ValueError("mounted visual aliases must be globally unique")
        if any(
            item.resource.session_id != self.session_id
            for item in (*self.bindings, *self.visual_bindings)
        ):
            raise ValueError("mounted resources must belong to their Session")
        if tuple(item.ordinal for item in self.bindings) != tuple(
            range(1, len(self.bindings) + 1)
        ):
            raise ValueError("mounted Document bindings must use source order")
        if tuple(item.ordinal for item in self.visual_bindings) != tuple(
            range(1, len(self.visual_bindings) + 1)
        ):
            raise ValueError("mounted visual bindings must use source order")
        if self.visual_projection_limit is not None and self.visual_bindings:
            raise ValueError(
                "a limited visual projection cannot expose partial visual aliases"
            )
        expected = _sha256_value(
            _scope_payload(
                self.bindings,
                self.visual_bindings,
                session_id=self.session_id,
                task_id=self.task_id,
                visual_projection_limit=self.visual_projection_limit,
            )
        )
        if self.scope_snapshot_sha256 != expected:
            raise ValueError("mounted Document scope hash is invalid")

    @property
    def authority_context(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "anchors": [
                    item.authority_anchor.model_dump(mode="json")
                    for item in (*self.bindings, *self.visual_bindings)
                ]
            }
        )

    @property
    def resources_by_alias(self) -> Mapping[str, FrozenPlanningResource]:
        return MappingProxyType(
            {item.resource.resource_alias: item.resource for item in self.bindings}
        )

    @property
    def visual_resources_by_alias(self) -> Mapping[str, FrozenPlanningResource]:
        return MappingProxyType(
            {
                item.resource.resource_alias: item.resource
                for item in self.visual_bindings
            }
        )


def freeze_mounted_document_planning_authority(
    *,
    session_id: str,
    task_id: str | None = None,
    maximum_documents: int = MAX_MOUNTED_DOCUMENT_PLANNING_RESOURCES,
    allowed_managed_document_ids: tuple[str, ...] | None = None,
) -> FrozenMountedDocumentPlanningAuthority:
    """Freeze the Host scope and map it into exact L2 planning identities."""

    raw_authority = freeze_mounted_document_authority(
        session_id=session_id,
        task_id=task_id,
        maximum_documents=maximum_documents,
        allowed_managed_document_ids=allowed_managed_document_ids,
    )
    bindings = [
        _project_document_binding(binding) for binding in raw_authority.bindings
    ]
    visual_projection_limit = None
    if raw_authority.visual_projection_limit is None:
        visual_bindings = project_mounted_visual_planning_bindings(
            session_id=session_id,
            bindings=raw_authority.visual_bindings,
        )
    else:
        visual_bindings = ()
        raw_limit = raw_authority.visual_projection_limit
        visual_projection_limit = FrozenMountedVisualPlanningProjectionLimit(
            observed_visual_unit_count=raw_limit.observed_visual_unit_count,
            maximum_visual_units=raw_limit.maximum_visual_units,
            projection_sha256=raw_limit.projection_sha256,
        )
        bindings = [
            _mark_visual_projection_incomplete(binding) for binding in bindings
        ]
    payload = _scope_payload(
        tuple(bindings),
        visual_bindings,
        session_id=session_id,
        task_id=task_id,
        visual_projection_limit=visual_projection_limit,
    )
    return FrozenMountedDocumentPlanningAuthority(
        session_id=session_id,
        task_id=task_id,
        bindings=tuple(bindings),
        visual_bindings=visual_bindings,
        scope_snapshot_sha256=_sha256_value(payload),
        visual_projection_limit=visual_projection_limit,
    )


def recover_task_scoped_managed_document_ids(
    *,
    session_id: str,
    task_id: str,
    authority_snapshot: PlanningAuthoritySnapshot | None,
) -> tuple[str, ...]:
    """Recover the persisted managed-input document authority for one Task."""

    for name, value in (("session_id", session_id), ("task_id", task_id)):
        if not isinstance(value, str) or not value or len(value) > 200:
            raise ValueError(f"{name} must be a bounded durable identity")
    if authority_snapshot is None:
        return ()
    if (
        authority_snapshot.session_id != session_id
        or authority_snapshot.task_id != task_id
    ):
        raise MountedDocumentPlanningAuthorityError(
            "persisted document authority belongs to another Session or Task"
        )
    return recover_managed_input_document_ids(
        session_id=session_id,
        allowed_document_ids=tuple(
            anchor.origin_id
            for anchor in authority_snapshot.anchors
            if anchor.origin_kind
            is PlanningAuthorityOriginKind.WORKSPACE_RESOURCE
        ),
    )


def build_mounted_document_authority_projection(
    *,
    authority_snapshot: PlanningAuthoritySnapshot,
    task_creation_source: InSessionTaskSourceAnchor,
    mounted_authority: FrozenMountedDocumentPlanningAuthority,
) -> PlanningAuthorityProjection:
    """Bind prompt cards to the exact Store-authorized authority snapshot."""

    if (
        authority_snapshot.session_id != mounted_authority.session_id
        or (
            mounted_authority.task_id is not None
            and authority_snapshot.task_id != mounted_authority.task_id
        )
    ):
        raise MountedDocumentPlanningAuthorityError(
            "authority snapshot and mounted resources cross Session/Task authority"
        )
    stored_by_alias = {
        item.projection_alias: item for item in authority_snapshot.anchors
    }
    expected_aliases = {
        "task_creation_source",
        *(item.resource.resource_alias for item in mounted_authority.bindings),
        *(item.resource.resource_alias for item in mounted_authority.visual_bindings),
    }
    if set(stored_by_alias) != expected_aliases:
        raise MountedDocumentPlanningAuthorityError(
            "stored authority aliases differ from the frozen mounted scope"
        )

    task_anchor = stored_by_alias["task_creation_source"]
    task_excerpt_sha256 = _sha256_text(task_creation_source.excerpt)
    if (
        task_anchor.authority_class is not PlanningAuthorityClass.AUTHORIZATION
        or task_anchor.origin_kind
        is not PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN
        or task_anchor.origin_id != task_creation_source.source_turn_id
        or task_anchor.span_start != task_creation_source.start
        or task_anchor.span_end != task_creation_source.end
        or task_anchor.content_sha256 != task_excerpt_sha256
        or task_anchor.projection_sha256 != task_excerpt_sha256
    ):
        raise MountedDocumentPlanningAuthorityError(
            "stored Task creation authority differs from its source span"
        )

    cards: list[PlanningAuthoritySourceCard] = [
        PlanningAuthoritySourceCard(
            alias="task_creation_source",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
            source_label="Task creation instruction",
            excerpt=task_creation_source.excerpt,
            projection_sha256=task_excerpt_sha256,
        )
    ]
    for binding in mounted_authority.bindings:
        stored = stored_by_alias[binding.resource.resource_alias]
        expected = binding.authority_anchor.model_copy(
            update={
                "authority_snapshot_id": authority_snapshot.authority_snapshot_id,
            }
        )
        if stored != expected:
            raise MountedDocumentPlanningAuthorityError(
                "stored Document authority differs from its frozen generation"
            )
        cards.append(binding.source_card)
    for binding in mounted_authority.visual_bindings:
        stored = stored_by_alias[binding.resource.resource_alias]
        expected = binding.authority_anchor.model_copy(
            update={
                "authority_snapshot_id": authority_snapshot.authority_snapshot_id,
            }
        )
        if stored != expected:
            raise MountedDocumentPlanningAuthorityError(
                "stored visual authority differs from its frozen generation"
            )
        cards.append(binding.source_card)
    cards.sort(key=lambda item: item.alias)
    return PlanningAuthorityProjection.create(
        authority_snapshot_id=authority_snapshot.authority_snapshot_id,
        authority_snapshot_sha256=authority_snapshot.snapshot_sha256,
        cards=tuple(cards),
    )


def build_mounted_document_perception_request(
    *,
    context: AuxiliaryHostPrimitiveDispatchContext,
    input_resource_aliases: tuple[str, ...],
    mounted_authority: FrozenMountedDocumentPlanningAuthority,
) -> PlanningResourcePerceptionRequest:
    """Resolve one Architect-selected mounted alias for a Host node."""

    if context.session_id != mounted_authority.session_id:
        raise MountedDocumentPlanningAuthorityError(
            "Host primitive and mounted authority belong to different Sessions"
        )
    resources = mounted_authority.resources_by_alias
    selected = tuple(alias for alias in input_resource_aliases if alias in resources)
    if len(selected) != 1:
        raise MountedDocumentPlanningAuthorityError(
            "a mounted Document perception node must select exactly one resource alias"
        )
    binding = next(
        item
        for item in mounted_authority.bindings
        if item.resource.resource_alias == selected[0]
    )
    suffix = f"{binding.ordinal:02d}"
    return PlanningResourcePerceptionRequest(
        binding=context.artifact_binding(
            scope_snapshot_sha256=binding.authority_anchor.freshness_binding_sha256,
            alias_prefix=f"document_{suffix}",
            artifact_alias=f"document_context_{suffix}",
        ),
        read_request=PlanningResourceReadRequest(resource=binding.resource),
    )


def build_mounted_resource_perception_request(
    *,
    context: AuxiliaryHostPrimitiveDispatchContext,
    input_resource_aliases: tuple[str, ...],
    mounted_authority: FrozenMountedDocumentPlanningAuthority,
) -> PlanningResourcePerceptionRequest:
    """Dispatch a document or visual preview without expanding model access."""

    if context.capability_profile_id == MOUNTED_DOCUMENT_READ_CAPABILITY:
        return build_mounted_document_perception_request(
            context=context,
            input_resource_aliases=input_resource_aliases,
            mounted_authority=mounted_authority,
        )
    if context.capability_profile_id == MOUNTED_VISUAL_READ_CAPABILITY:
        return build_mounted_visual_perception_request(
            context=context,
            input_resource_aliases=input_resource_aliases,
            visual_bindings=mounted_authority.visual_bindings,
        )
    raise MountedDocumentPlanningAuthorityError(
        "selected Host capability is not a mounted resource perception profile"
    )


def build_mounted_resource_read_port(
    *,
    mounted_authority: FrozenMountedDocumentPlanningAuthority,
    vision_adapter=None,
    visual_call_ledger=None,
) -> MountedDocumentAndVisualPlanningResourceReadPort:
    """Compose the exact document and visual readers for one frozen scope."""

    return MountedDocumentAndVisualPlanningResourceReadPort(
        document_port=MountedDocumentPlanningResourceReadPort(
            tuple(item.frozen_document for item in mounted_authority.bindings)
        ),
        visual_port=MountedVisualPlanningResourceReadPort(
            mounted_authority.visual_bindings,
            adapter=vision_adapter,
            call_ledger=visual_call_ledger,
        ),
    )


def _project_document_binding(
    binding: FrozenMountedDocumentBinding,
) -> FrozenMountedDocumentPlanningBinding:
    resource_format = PlanningResourceFormat(binding.resource_format.value)
    coverage = PlanningResourceCoverage(binding.processing_status)
    resource = FrozenPlanningResource(
        session_id=binding.session_id,
        resource_alias=binding.resource_alias,
        resource_id=binding.document_id,
        resource_version=binding.document_version,
        content_sha256=binding.source_sha256,
        coverage=coverage,
        resource_format=resource_format,
        media_type=_MEDIA_TYPES[resource_format],
        file_extension=binding.file_extension,
    )
    frozen_document = FrozenMountedDocument(
        session_id=binding.session_id,
        resource_alias=binding.resource_alias,
        document_id=binding.document_id,
        document_version_id=binding.document_version,
        source_sha256=binding.source_sha256,
        processing_status=binding.processing_status,
        resource_format=resource_format.value,
        media_type=_MEDIA_TYPES[resource_format],
        file_extension=binding.file_extension,
        total_chunk_count=binding.total_chunk_count,
        processing_diagnostic_codes=binding.processing_diagnostic_codes,
    )
    excerpt = _document_card_excerpt(
        resource,
        total_chunk_count=binding.total_chunk_count,
    )
    projection_sha256 = _sha256_text(excerpt)
    freshness_sha256 = frozen_document.freshness_binding_sha256
    anchor = PlanningAuthorityAnchor(
        anchor_id="auxanchor_document_" + freshness_sha256[:32],
        authority_snapshot_id=_PENDING_AUTHORITY_SNAPSHOT_ID,
        projection_alias=binding.resource_alias,
        authority_class=PlanningAuthorityClass.EVIDENCE,
        origin_kind=PlanningAuthorityOriginKind.WORKSPACE_RESOURCE,
        origin_id=binding.document_id,
        source_revision=None,
        content_sha256=binding.source_sha256,
        item_ordinal=binding.ordinal,
        projection_sha256=projection_sha256,
        freshness_binding_sha256=freshness_sha256,
    )
    return FrozenMountedDocumentPlanningBinding(
        ordinal=binding.ordinal,
        resource=resource,
        frozen_document=frozen_document,
        authority_anchor=anchor,
        source_card=PlanningAuthoritySourceCard(
            alias=binding.resource_alias,
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            document_group_alias=planning_document_group_alias(
                session_id=binding.session_id,
                document_id=binding.document_id,
            ),
            source_label=(
                f"Mounted {resource_format.value.upper()} document {binding.ordinal}"
            ),
            excerpt=excerpt,
            projection_sha256=projection_sha256,
        ),
    )


def _document_card_excerpt(
    resource: FrozenPlanningResource,
    *,
    total_chunk_count: int,
) -> str:
    return (
        f"A Session-mounted {resource.resource_format.value.upper()} document is "
        f"available as {resource.resource_alias}; it has {total_chunk_count} "
        f"admitted text chunks and its admitted processing "
        f"coverage is {resource.coverage.value}."
    )


def _mark_visual_projection_incomplete(
    binding: FrozenMountedDocumentPlanningBinding,
) -> FrozenMountedDocumentPlanningBinding:
    excerpt = (
        f"{binding.source_card.excerpt} Its admitted text chunks remain "
        "available, but visual coverage is not complete because the bounded "
        "visual projection limit was exceeded."
    )
    projection_sha256 = _sha256_text(excerpt)
    return replace(
        binding,
        authority_anchor=binding.authority_anchor.model_copy(
            update={"projection_sha256": projection_sha256}
        ),
        source_card=binding.source_card.model_copy(
            update={
                "excerpt": excerpt,
                "projection_sha256": projection_sha256,
            }
        ),
    )


def _resource_payload(resource: FrozenPlanningResource) -> dict[str, object]:
    return {
        "session_id": resource.session_id,
        "resource_alias": resource.resource_alias,
        "resource_id": resource.resource_id,
        "resource_version": resource.resource_version,
        "content_sha256": resource.content_sha256,
        "coverage": resource.coverage.value,
        "resource_format": resource.resource_format.value,
        "media_type": resource.media_type,
        "file_extension": resource.file_extension,
    }


def _resource_matches_frozen_document(
    resource: FrozenPlanningResource,
    document: FrozenMountedDocument,
) -> bool:
    return bool(
        resource.session_id == document.session_id
        and resource.resource_alias == document.resource_alias
        and resource.resource_id == document.document_id
        and resource.resource_version == document.document_version_id
        and resource.content_sha256 == document.source_sha256
        and resource.coverage.value == document.processing_status
        and resource.resource_format.value == document.resource_format
        and resource.media_type == document.media_type
        and resource.file_extension == document.file_extension
    )


def _scope_payload(
    bindings: tuple[FrozenMountedDocumentPlanningBinding, ...],
    visual_bindings: tuple[FrozenMountedVisualPlanningBinding, ...],
    *,
    session_id: str,
    task_id: str | None = None,
    visual_projection_limit: FrozenMountedVisualPlanningProjectionLimit | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "frozen-mounted-document-planning-authority-v1",
        "session_id": session_id,
        "bindings": [
            {
                "ordinal": item.ordinal,
                "resource": _resource_payload(item.resource),
                "authority_anchor": item.authority_anchor.model_dump(mode="json"),
                "source_card": item.source_card.model_dump(mode="json"),
            }
            for item in bindings
        ],
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if visual_bindings:
        payload["visual_bindings"] = [
            visual_binding_private_payload(item) for item in visual_bindings
        ]
    if visual_projection_limit is not None:
        payload["visual_projection_limit"] = {
            "observed_visual_unit_count": (
                visual_projection_limit.observed_visual_unit_count
            ),
            "maximum_visual_units": visual_projection_limit.maximum_visual_units,
            "projection_sha256": visual_projection_limit.projection_sha256,
        }
    return payload


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    "MAX_MOUNTED_DOCUMENT_PLANNING_RESOURCES",
    "MOUNTED_DOCUMENT_READ_CAPABILITY",
    "FrozenMountedDocumentPlanningAuthority",
    "FrozenMountedDocumentPlanningBinding",
    "FrozenMountedVisualPlanningProjectionLimit",
    "MountedDocumentPlanningAuthorityError",
    "build_mounted_document_authority_projection",
    "build_mounted_document_perception_request",
    "build_mounted_resource_perception_request",
    "build_mounted_resource_read_port",
    "freeze_mounted_document_planning_authority",
    "recover_task_scoped_managed_document_ids",
]
