"""Stable definitions and contextual bindings for external visual analysis.

The two model contracts and handlers are owned by
``format_observation_tools``.  This module keeps their stable code identity
separate from the workspace authority, configured provider, capability snapshot,
disclosure receipts, and durable physical-call ledger selected by the Host.
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
    EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS,
    VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    build_external_visual_analysis_execution_profile,
    build_external_visual_analysis_tool_specs,
)


EXTERNAL_VISUAL_ANALYSIS_DEFINITION_MANIFEST_SCHEMA = (
    "external-visual-analysis-definition-manifest-v1"
)
EXTERNAL_VISUAL_ANALYSIS_BINDING_ASSERTION_SCHEMA = (
    "external-visual-analysis-binding-assertion-v1"
)

_EXTERNAL_VISUAL_ANALYSIS_SOURCE_ID = (
    "personagraph.input_processing.external_visual_analysis"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ExternalVisualAnalysisCatalogError(ValueError):
    """An external visual Definition or contextual Binding drifted."""


@dataclass(frozen=True, slots=True)
class ExternalVisualAnalysisToolDefinitionManifestItem:
    """One provider-neutral Definition and its reviewed behavior revision."""

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
class ExternalVisualAnalysisBindingFacts:
    """Secret-free identities for one authorized physical visual stack.

    The Host supplies only canonical hashes.  Consequently a persisted Binding
    can prove exactly which authority/provider/ledger was selected without
    serializing a Session ID, root path, grant receipt, endpoint, model name, or
    credential.
    """

    boundary_sha256: str
    read_authority_sha256: str
    provider_identity_sha256: str
    capability_snapshot_sha256: str
    egress_policy_sha256: str
    physical_call_ledger_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "boundary_sha256",
            "read_authority_sha256",
            "provider_identity_sha256",
            "capability_snapshot_sha256",
            "egress_policy_sha256",
            "physical_call_ledger_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class _ExternalVisualAnalysisDeclaration:
    tool_id: str
    implementation_ref: str
    declared_behavior_revision: str


_DECLARATIONS = (
    _ExternalVisualAnalysisDeclaration(
        "analyze_image",
        "builtin/analyze_image",
        "analyze-image-handler-3",
    ),
    _ExternalVisualAnalysisDeclaration(
        "analyze_pdf_page",
        "builtin/analyze_pdf_page",
        "analyze-pdf-page-batch-handler",
    ),
)


def build_external_visual_analysis_tool_definition_manifest() -> tuple[
    ExternalVisualAnalysisToolDefinitionManifestItem,
    ExternalVisualAnalysisToolDefinitionManifestItem,
]:
    """Build the two provider-neutral Definitions in exposure order."""

    specs = build_external_visual_analysis_tool_specs()
    actual_ids = tuple(spec.tool_id for spec in specs)
    declared_ids = tuple(item.tool_id for item in _DECLARATIONS)
    if (
        actual_ids != EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS
        or declared_ids != EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS
        or len(set(actual_ids)) != len(actual_ids)
    ):
        raise ExternalVisualAnalysisCatalogError(
            "external visual analysis definition order or identity drifted"
        )
    manifest = tuple(
        _manifest_item(
            spec, declaration, build_external_visual_analysis_execution_profile(spec.tool_id)
        )
        for spec, declaration in zip(specs, _DECLARATIONS, strict=True)
    )
    return (manifest[0], manifest[1])


def build_external_visual_analysis_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: ExternalVisualAnalysisBindingFacts,
) -> tuple[ToolBinding, ToolBinding]:
    """Convert two already-frozen external registrations into exact bindings."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, ExternalVisualAnalysisBindingFacts):
        raise TypeError("facts must be ExternalVisualAnalysisBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != len(EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS) or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise ExternalVisualAnalysisCatalogError(
            "external visual analysis requires exactly two ToolRegistration values"
        )
    manifest = build_external_visual_analysis_tool_definition_manifest()
    if tuple(item.tool_id for item in frozen) != EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS:
        raise ExternalVisualAnalysisCatalogError(
            "external visual analysis registrations are not in canonical order"
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
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis execution contract drifted: {registration.tool_id!r}"
            )
        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != _EXTERNAL_VISUAL_ANALYSIS_SOURCE_ID
            or source.display_name is not None
        ):
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis source drifted: {registration.tool_id!r}"
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
                "schema_version": EXTERNAL_VISUAL_ANALYSIS_BINDING_ASSERTION_SCHEMA,
                "binding_kind": "authorized_external_visual_analysis",
                "boundary_sha256": facts.boundary_sha256,
                "read_authority_sha256": facts.read_authority_sha256,
                "provider_identity_sha256": facts.provider_identity_sha256,
                "capability_snapshot_sha256": facts.capability_snapshot_sha256,
                "egress_policy_sha256": facts.egress_policy_sha256,
                "physical_call_ledger_sha256": facts.physical_call_ledger_sha256,
                "source_fingerprint": source.fingerprint,
            },
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise ExternalVisualAnalysisCatalogError(
                f"external visual analysis descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)
    return (bindings[0], bindings[1])


def _manifest_item(
    spec: ToolSpec,
    declaration: _ExternalVisualAnalysisDeclaration,
    execution: ToolExecutionProfile,
) -> ExternalVisualAnalysisToolDefinitionManifestItem:
    effect_template = _effect_template()
    implementation_digest = _declared_implementation_digest(
        spec=spec,
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        effect_template=effect_template,
        execution=execution,
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
        implementation_ref=declaration.implementation_ref,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return ExternalVisualAnalysisToolDefinitionManifestItem(
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        definition=definition,
    )


def _effect_template() -> ToolEffectProfile:
    return ToolEffectProfile(
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
            EffectDescriptor(
                resource=EffectResource.NETWORK,
                action=EffectAction.TRANSMIT,
                scope_kind=EffectScopeKind.SESSION,
                default_scope="*",
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.NOT_IDEMPOTENT,
                reversibility=Reversibility.IRREVERSIBLE,
            ),
            EffectDescriptor(
                resource=EffectResource.RUNTIME_STATE,
                action=EffectAction.UPDATE,
                scope_kind=EffectScopeKind.SESSION,
                default_scope="*",
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    implementation_ref: str,
    declared_behavior_revision: str,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    artifact = {
        "schema_version": EXTERNAL_VISUAL_ANALYSIS_DEFINITION_MANIFEST_SCHEMA,
        "implementation_ref": implementation_ref,
        "declared_behavior_revision": declared_behavior_revision,
        "identity": {
            "tool_id": spec.tool_id,
            "contract_version": spec.contract_version,
            "implementation_version": VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
        },
        "spec": spec.to_dict(),
        "effect_template": [
            effect.to_dict() for effect in effect_template.effects
        ],
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
        raise ExternalVisualAnalysisCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


__all__ = [
    "EXTERNAL_VISUAL_ANALYSIS_BINDING_ASSERTION_SCHEMA",
    "EXTERNAL_VISUAL_ANALYSIS_DEFINITION_MANIFEST_SCHEMA",
    "ExternalVisualAnalysisBindingFacts",
    "ExternalVisualAnalysisCatalogError",
    "ExternalVisualAnalysisToolDefinitionManifestItem",
    "build_external_visual_analysis_tool_bindings",
    "build_external_visual_analysis_tool_definition_manifest",
]
