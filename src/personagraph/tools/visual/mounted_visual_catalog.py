"""Stable Definition and external-only Binding for mounted visual reads.

The model contract and implementation identity are process-stable.  Session
scope, opaque alias projection, mounted-document authority, freshness checks,
provider capabilities, disclosure grants, and the durable physical-call ledger
belong to a contextual ``ToolBinding``.

The current production adapter is either unavailable or an external HTTP
provider.  Test-only local adapters have a different filesystem-read effect and
must not be made to look like the external Definition.  A future production
local provider therefore needs a separately reviewed effect-variant contract.
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
from ..contracts import ToolSourceKind
from ..registration import ToolExecutionProfile, ToolRegistration
from .mounted_visual_tools import (
    MOUNTED_VISUAL_IMPLEMENTATION_VERSION,
    MOUNTED_VISUAL_SOURCE_DISPLAY_NAME,
    MOUNTED_VISUAL_SOURCE_ID,
    MOUNTED_VISUAL_TOOL_ID,
    mounted_visual_tool_spec,
)
from .visual_tools import (
    visual_observation_effect_profile,
    visual_observation_execution_profile,
)


MOUNTED_VISUAL_DEFINITION_MANIFEST_SCHEMA = (
    "mounted-visual-definition-manifest-v1"
)
MOUNTED_VISUAL_BINDING_ASSERTION_SCHEMA = (
    "mounted-visual-binding-assertion-v1"
)
MOUNTED_VISUAL_SOURCE_FINGERPRINT_SCHEMA = (
    "mounted-visual-source-fingerprint-v1"
)

_IMPLEMENTATION_REF = "builtin/read_mounted_visuals"
_DECLARED_BEHAVIOR_REVISION = "mounted-visual-external-read-handler-1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MountedVisualCatalogError(ValueError):
    """The mounted visual Definition or contextual Binding drifted."""


@dataclass(frozen=True, slots=True)
class MountedVisualToolDefinitionManifestItem:
    """The stable external-read Definition and its reviewed declaration."""

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
class ExternalMountedVisualBindingFacts:
    """Secret-free commitments for one exact external mounted-visual route.

    The effective Session policy scope remains explicit in the Binding effect.
    Raw aliases, paths, provider names, receipt IDs, and ledger paths stay inside
    the existing Host closures; their authorities persist only as digests.
    """

    session_scope_sha256: str
    visual_scope_snapshot_sha256: str
    visual_alias_projection_sha256: str
    mounted_authority_sha256: str
    freshness_authority_sha256: str
    provider_identity_sha256: str
    capability_snapshot_sha256: str
    egress_policy_sha256: str
    physical_call_ledger_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "session_scope_sha256",
            "visual_scope_snapshot_sha256",
            "visual_alias_projection_sha256",
            "mounted_authority_sha256",
            "freshness_authority_sha256",
            "provider_identity_sha256",
            "capability_snapshot_sha256",
            "egress_policy_sha256",
            "physical_call_ledger_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)


def build_mounted_visual_tool_definition_manifest() -> tuple[
    MountedVisualToolDefinitionManifestItem,
]:
    """Build the context-free external mounted-visual Definition."""

    spec = mounted_visual_tool_spec()
    if spec.tool_id != MOUNTED_VISUAL_TOOL_ID:
        raise MountedVisualCatalogError("mounted visual definition identity drifted")
    effect_template = visual_observation_effect_profile("*", True)
    execution = visual_observation_execution_profile()
    implementation_digest = _canonical_sha256(
        {
            "schema_version": MOUNTED_VISUAL_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": _IMPLEMENTATION_REF,
            "declared_behavior_revision": _DECLARED_BEHAVIOR_REVISION,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": MOUNTED_VISUAL_IMPLEMENTATION_VERSION,
            },
            "spec": spec.to_dict(),
            "effect_template": [
                effect.to_dict() for effect in effect_template.effects
            ],
            "execution": _execution_descriptor(execution),
        }
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=MOUNTED_VISUAL_IMPLEMENTATION_VERSION,
        implementation_ref=_IMPLEMENTATION_REF,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return (
        MountedVisualToolDefinitionManifestItem(
            implementation_ref=_IMPLEMENTATION_REF,
            declared_behavior_revision=_DECLARED_BEHAVIOR_REVISION,
            definition=definition,
        ),
    )


def derive_mounted_visual_source_fingerprint(
    facts: ExternalMountedVisualBindingFacts,
) -> str:
    """Commit every private execution authority into the live source identity."""

    if not isinstance(facts, ExternalMountedVisualBindingFacts):
        raise TypeError("facts must be ExternalMountedVisualBindingFacts")
    return _canonical_sha256(
        {
            "schema_version": MOUNTED_VISUAL_SOURCE_FINGERPRINT_SCHEMA,
            **_facts_descriptor(facts),
        }
    )


def build_mounted_visual_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: ExternalMountedVisualBindingFacts,
) -> tuple[ToolBinding]:
    """Bind one exact external registration; reject local/unavailable shapes."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, ExternalMountedVisualBindingFacts):
        raise TypeError("facts must be ExternalMountedVisualBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != 1 or not isinstance(frozen[0], ToolRegistration):
        raise MountedVisualCatalogError(
            "mounted visuals require exactly one ToolRegistration value"
        )
    (registration,) = frozen
    if registration.tool_id != MOUNTED_VISUAL_TOOL_ID:
        raise MountedVisualCatalogError(
            "mounted visual registration is not the canonical tool"
        )

    (manifest_item,) = build_mounted_visual_tool_definition_manifest()
    definition = manifest_item.definition
    actual_identity = ToolIdentity(
        registration.tool_id,
        registration.contract_version,
        registration.implementation_version,
    )
    if actual_identity != definition.identity:
        raise MountedVisualCatalogError("mounted visual identity drifted")
    if registration.spec != definition.spec:
        raise MountedVisualCatalogError("mounted visual model contract drifted")
    if registration.execution_profile != definition.execution_profile:
        raise MountedVisualCatalogError("mounted visual execution contract drifted")

    effects = registration.effect_profile.effects
    if len(effects) != 1:
        raise MountedVisualCatalogError("mounted visual effects drifted")
    effective_scope = effects[0].default_scope
    if (
        effective_scope == "*"
        or registration.effect_profile
        != visual_observation_effect_profile(effective_scope, True)
        or _sha256_text(effective_scope) != facts.session_scope_sha256
    ):
        raise MountedVisualCatalogError(
            "mounted visual binding is not an exact external effect"
        )

    source = registration.source
    if (
        source.kind is not ToolSourceKind.LOCAL
        or source.source_id != MOUNTED_VISUAL_SOURCE_ID
        or source.display_name != MOUNTED_VISUAL_SOURCE_DISPLAY_NAME
    ):
        raise MountedVisualCatalogError("mounted visual source drifted")
    expected_source_fingerprint = derive_mounted_visual_source_fingerprint(facts)
    if source.fingerprint != expected_source_fingerprint:
        raise MountedVisualCatalogError("mounted visual source fingerprint drifted")

    binding = ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=source,
        handler=registration.handler,
        effect_profile=registration.effect_profile,
        binding_assertion={
            "schema_version": MOUNTED_VISUAL_BINDING_ASSERTION_SCHEMA,
            "binding_kind": "authorized_external_mounted_visual_read",
            **_facts_descriptor(facts),
            "source_fingerprint": expected_source_fingerprint,
        },
    )
    try:
        bound = BoundToolRegistration(definition, binding)
    except ValueError as exc:
        raise MountedVisualCatalogError("mounted visual effects drifted") from exc
    if bound.descriptor() != registration.descriptor():
        raise MountedVisualCatalogError("mounted visual descriptor drifted")
    return (binding,)


def _facts_descriptor(
    facts: ExternalMountedVisualBindingFacts,
) -> dict[str, str]:
    return {
        "session_scope_sha256": facts.session_scope_sha256,
        "visual_scope_snapshot_sha256": facts.visual_scope_snapshot_sha256,
        "visual_alias_projection_sha256": facts.visual_alias_projection_sha256,
        "mounted_authority_sha256": facts.mounted_authority_sha256,
        "freshness_authority_sha256": facts.freshness_authority_sha256,
        "provider_identity_sha256": facts.provider_identity_sha256,
        "capability_snapshot_sha256": facts.capability_snapshot_sha256,
        "egress_policy_sha256": facts.egress_policy_sha256,
        "physical_call_ledger_sha256": facts.physical_call_ledger_sha256,
    }


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
        raise MountedVisualCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MOUNTED_VISUAL_BINDING_ASSERTION_SCHEMA",
    "MOUNTED_VISUAL_DEFINITION_MANIFEST_SCHEMA",
    "MOUNTED_VISUAL_SOURCE_FINGERPRINT_SCHEMA",
    "ExternalMountedVisualBindingFacts",
    "MountedVisualCatalogError",
    "MountedVisualToolDefinitionManifestItem",
    "build_mounted_visual_tool_bindings",
    "build_mounted_visual_tool_definition_manifest",
    "derive_mounted_visual_source_fingerprint",
]
