"""Stable definitions and contextual bindings for local format observation.

This module owns only the five local filesystem readers.  External semantic visual
analysis has a different provider, disclosure, effect, and identity lifecycle and
is intentionally excluded from this manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import re

from ..catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from ..contracts import ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration
from .format_observation_tools import (
    FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
    FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    build_format_observation_execution_profile,
    build_format_observation_tool_specs,
)


FORMAT_OBSERVATION_DEFINITION_MANIFEST_SCHEMA = (
    "format-observation-definition-manifest-v1"
)
FORMAT_OBSERVATION_BINDING_ASSERTION_SCHEMA = "format-observation-binding-assertion-v1"

_FORMAT_OBSERVATION_SOURCE_ID = "personagraph.input_processing.format_observation"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FormatObservationCatalogError(ValueError):
    """A local observation definition or binding drifted from its manifest."""


@dataclass(frozen=True, slots=True)
class FormatObservationToolDefinitionManifestItem:
    """One stable Definition plus its reviewed behavior declaration."""

    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_ref, str) or not self.implementation_ref:
            raise ValueError("implementation_ref must not be empty")
        if (
            not isinstance(self.declared_behavior_revision, str)
            or not self.declared_behavior_revision
        ):
            raise ValueError("declared_behavior_revision must not be empty")
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if self.definition.implementation_ref != self.implementation_ref:
            raise ValueError("definition implementation_ref does not match manifest")


@dataclass(frozen=True, slots=True)
class FormatObservationBindingFacts:
    """Secret-free hashes for one frozen local reader stack and authority."""

    boundary_sha256: str
    read_authority_sha256: str
    reader_stack_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.boundary_sha256, "boundary_sha256")
        _require_sha256(self.read_authority_sha256, "read_authority_sha256")
        _require_sha256(self.reader_stack_sha256, "reader_stack_sha256")


@dataclass(frozen=True, slots=True)
class _FormatObservationDeclaration:
    tool_id: str
    implementation_ref: str
    declared_behavior_revision: str
    implementation_version: str


_DECLARATIONS = (
    _FormatObservationDeclaration(
        "read_text",
        "builtin/read_text",
        "read-text-handler-2",
        FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    ),
    _FormatObservationDeclaration(
        "read_pdf_text",
        "builtin/read_pdf_text",
        "read-pdf-text-handler-2",
        FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    ),
    _FormatObservationDeclaration(
        "read_word",
        "builtin/read_word",
        "read-word-handler-2",
        FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    ),
    _FormatObservationDeclaration(
        "read_slides",
        "builtin/read_slides",
        "read-slides-handler-2",
        FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    ),
    _FormatObservationDeclaration(
        "inspect_image",
        "builtin/inspect_image",
        "inspect-image-handler-3",
        VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    ),
)


def build_format_observation_tool_definition_manifest() -> tuple[
    FormatObservationToolDefinitionManifestItem, ...
]:
    """Build the five context-free local reader Definitions in exposure order."""

    specs = build_format_observation_tool_specs()
    actual_ids = tuple(spec.tool_id for spec in specs)
    declared_ids = tuple(item.tool_id for item in _DECLARATIONS)
    if (
        actual_ids != FORMAT_OBSERVATION_LOCAL_TOOL_IDS
        or declared_ids != FORMAT_OBSERVATION_LOCAL_TOOL_IDS
        or len(set(actual_ids)) != len(actual_ids)
    ):
        raise FormatObservationCatalogError(
            "format observation definition order or identity drifted"
        )
    execution = build_format_observation_execution_profile()
    manifest = tuple(
        _manifest_item(spec, declaration, execution)
        for spec, declaration in zip(specs, _DECLARATIONS, strict=True)
    )
    return manifest


def build_format_observation_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: FormatObservationBindingFacts,
) -> tuple[ToolBinding, ...]:
    """Convert five frozen local registrations into exact bare bindings."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, FormatObservationBindingFacts):
        raise TypeError("facts must be FormatObservationBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != len(FORMAT_OBSERVATION_LOCAL_TOOL_IDS) or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise FormatObservationCatalogError(
            "format observation requires exactly five ToolRegistration values"
        )
    manifest = build_format_observation_tool_definition_manifest()
    actual_ids = tuple(item.tool_id for item in frozen)
    if actual_ids != FORMAT_OBSERVATION_LOCAL_TOOL_IDS:
        raise FormatObservationCatalogError(
            "format observation registrations are not in canonical order"
        )

    bindings: list[ToolBinding] = []
    for registration, item in zip(frozen, manifest, strict=True):
        definition = item.definition
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != definition.identity:
            raise FormatObservationCatalogError(
                f"format observation identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise FormatObservationCatalogError(
                f"format observation model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise FormatObservationCatalogError(
                f"format observation execution contract drifted: {registration.tool_id!r}"
            )
        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != _FORMAT_OBSERVATION_SOURCE_ID
            or source.display_name is not None
        ):
            raise FormatObservationCatalogError(
                f"format observation source drifted: {registration.tool_id!r}"
            )
        _require_sha256(
            source.fingerprint,
            f"{registration.tool_id} source fingerprint",
        )
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "schema_version": FORMAT_OBSERVATION_BINDING_ASSERTION_SCHEMA,
                "binding_kind": "frozen_workspace_format_observation",
                "boundary_sha256": facts.boundary_sha256,
                "read_authority_sha256": facts.read_authority_sha256,
                "reader_stack_sha256": facts.reader_stack_sha256,
                "source_fingerprint": source.fingerprint,
            },
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise FormatObservationCatalogError(
                f"format observation effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise FormatObservationCatalogError(
                f"format observation descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)
    return tuple(bindings)


def _manifest_item(
    spec: ToolSpec,
    declaration: _FormatObservationDeclaration,
    execution: ToolExecutionProfile,
) -> FormatObservationToolDefinitionManifestItem:
    effect_template = ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=EffectAction.READ,
                scope_kind=EffectScopeKind.WORKSPACE,
                default_scope="*",
                resource_argument="path",
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )
    implementation_digest = _declared_implementation_digest(
        spec=spec,
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        implementation_version=declaration.implementation_version,
        effect_template=effect_template,
        execution=execution,
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=declaration.implementation_version,
        implementation_ref=declaration.implementation_ref,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return FormatObservationToolDefinitionManifestItem(
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        definition=definition,
    )


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    implementation_ref: str,
    declared_behavior_revision: str,
    implementation_version: str,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    artifact = {
        "schema_version": FORMAT_OBSERVATION_DEFINITION_MANIFEST_SCHEMA,
        "implementation_ref": implementation_ref,
        "declared_behavior_revision": declared_behavior_revision,
        "identity": {
            "tool_id": spec.tool_id,
            "contract_version": spec.contract_version,
            "implementation_version": implementation_version,
        },
        "spec": spec.to_dict(),
        "effect_template": [effect.to_dict() for effect in effect_template.effects],
        "execution": _execution_descriptor(execution),
    }
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execution_descriptor(profile: ToolExecutionProfile) -> dict[str, object]:
    return {
        "default_timeout_s": profile.default_timeout_s,
        "hard_timeout_s": profile.hard_timeout_s,
        "max_output_bytes": profile.max_output_bytes,
        "max_transparent_retries": profile.max_transparent_retries,
        "execution_mode": profile.execution_mode.value,
        "cancellation_mode": profile.cancellation_mode.value,
        "isolation_requirement": profile.isolation_requirement.value,
        "concurrency_class": profile.concurrency_class,
    }


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FormatObservationCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


__all__ = [
    "FORMAT_OBSERVATION_BINDING_ASSERTION_SCHEMA",
    "FORMAT_OBSERVATION_DEFINITION_MANIFEST_SCHEMA",
    "FormatObservationBindingFacts",
    "FormatObservationCatalogError",
    "FormatObservationToolDefinitionManifestItem",
    "build_format_observation_tool_bindings",
    "build_format_observation_tool_definition_manifest",
]
