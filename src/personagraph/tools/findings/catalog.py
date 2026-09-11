"""Stable Definitions and contextual Bindings for the execution findings ledger.

The two Definitions are safe to persist and expose to a model.  Their live
handler, exact execution scope, ledger, scope-key authority, dispatcher and
mutation store are supplied later as secret-free Binding facts by the Host.
This module does not read Runtime state and does not create a second dispatcher.
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
from ..effects import ToolEffectProfile
from ..registration import ToolExecutionProfile, ToolRegistration
from .contracts import (
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
    REVISE_EXECUTION_FINDING_TOOL_ID,
)
from .execution_findings_tools import (
    EXECUTION_FINDINGS_SOURCE_DISPLAY_NAME,
    EXECUTION_FINDINGS_SOURCE_ID,
    EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION,
    build_execution_findings_effect_profile,
    build_execution_findings_execution_profile,
    build_execution_findings_tool_specs,
)


EXECUTION_FINDINGS_DEFINITION_MANIFEST_SCHEMA = (
    "execution-findings-definition-manifest-v1"
)
EXECUTION_FINDINGS_BINDING_ASSERTION_SCHEMA = (
    "execution-findings-binding-assertion-v1"
)
EXECUTION_FINDINGS_SOURCE_FINGERPRINT_SCHEMA = (
    "execution-findings-source-fingerprint-v1"
)

_DECLARATIONS = (
    ("builtin/record_execution_findings", "record-execution-findings-handler-3"),
    ("builtin/revise_execution_finding", "revise-execution-finding-handler-3"),
)
_ORDERED_TOOL_IDS = (
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
    REVISE_EXECUTION_FINDING_TOOL_ID,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ExecutionFindingsCatalogError(ValueError):
    """The findings tools drifted from their reviewed Definition or Binding."""


@dataclass(frozen=True, slots=True)
class ExecutionFindingsToolDefinitionManifestItem:
    """One stable Definition plus its reviewed behavior declaration."""

    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition

    def __post_init__(self) -> None:
        if not self.implementation_ref:
            raise ValueError("implementation_ref must not be empty")
        if not self.declared_behavior_revision:
            raise ValueError("declared_behavior_revision must not be empty")
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if self.definition.implementation_ref != self.implementation_ref:
            raise ValueError(
                "definition implementation_ref does not match manifest"
            )


@dataclass(frozen=True, slots=True)
class ExecutionFindingsBindingFacts:
    """Secret-free identities for one exact execution-ledger write route."""

    effect_scope_sha256: str
    ledger_identity_sha256: str
    scope_key_authority_sha256: str
    dispatcher_identity_sha256: str
    mutation_store_identity_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "effect_scope_sha256",
            "ledger_identity_sha256",
            "scope_key_authority_sha256",
            "dispatcher_identity_sha256",
            "mutation_store_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)


def build_execution_findings_tool_definition_manifest() -> tuple[
    ExecutionFindingsToolDefinitionManifestItem,
    ExecutionFindingsToolDefinitionManifestItem,
]:
    """Build the context-free findings Definition pair in exposure order."""

    specs = build_execution_findings_tool_specs()
    if (
        tuple(spec.tool_id for spec in specs) != _ORDERED_TOOL_IDS
        or len(_DECLARATIONS) != len(specs)
    ):
        raise ExecutionFindingsCatalogError(
            "execution findings definition order or identity drifted"
        )
    effect_template = build_execution_findings_effect_profile(
        default_scope="*"
    )
    execution = build_execution_findings_execution_profile()
    items = tuple(
        _manifest_item(
            spec=spec,
            implementation_ref=implementation_ref,
            declared_behavior_revision=declared_behavior_revision,
            effect_template=effect_template,
            execution=execution,
        )
        for spec, (implementation_ref, declared_behavior_revision) in zip(
            specs,
            _DECLARATIONS,
            strict=True,
        )
    )
    return (items[0], items[1])


def derive_execution_findings_source_fingerprint(
    facts: ExecutionFindingsBindingFacts,
) -> str:
    """Derive the source identity advertised by matching live registrations."""

    if not isinstance(facts, ExecutionFindingsBindingFacts):
        raise TypeError("facts must be ExecutionFindingsBindingFacts")
    return _canonical_sha256(
        {
            "schema_version": EXECUTION_FINDINGS_SOURCE_FINGERPRINT_SCHEMA,
            **_facts_descriptor(facts),
        }
    )


def build_execution_findings_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: ExecutionFindingsBindingFacts,
) -> tuple[ToolBinding, ToolBinding]:
    """Bind an already-frozen findings pair to one exact Host write route."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError(
            "registrations must be a sequence of ToolRegistration values"
        )
    if not isinstance(facts, ExecutionFindingsBindingFacts):
        raise TypeError("facts must be ExecutionFindingsBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != len(_ORDERED_TOOL_IDS) or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise ExecutionFindingsCatalogError(
            "execution findings require exactly two ToolRegistration values"
        )
    if tuple(item.tool_id for item in frozen) != _ORDERED_TOOL_IDS:
        raise ExecutionFindingsCatalogError(
            "execution findings registrations are not in canonical order"
        )

    manifest = build_execution_findings_tool_definition_manifest()
    expected_fingerprint = derive_execution_findings_source_fingerprint(facts)
    bindings: list[ToolBinding] = []
    for registration, manifest_item in zip(frozen, manifest, strict=True):
        definition = manifest_item.definition
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != definition.identity:
            raise ExecutionFindingsCatalogError(
                f"execution findings identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise ExecutionFindingsCatalogError(
                f"execution findings model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise ExecutionFindingsCatalogError(
                "execution findings execution contract drifted: "
                f"{registration.tool_id!r}"
            )
        effects = registration.effect_profile.effects
        if len(effects) != 1:
            raise ExecutionFindingsCatalogError(
                f"execution findings effects drifted: {registration.tool_id!r}"
            )
        effective_scope = effects[0].default_scope
        if (
            effective_scope == "*"
            or registration.effect_profile
            != build_execution_findings_effect_profile(
                default_scope=effective_scope
            )
            or _sha256_text(effective_scope) != facts.effect_scope_sha256
        ):
            raise ExecutionFindingsCatalogError(
                f"execution findings effects drifted: {registration.tool_id!r}"
            )
        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != EXECUTION_FINDINGS_SOURCE_ID
            or source.display_name != EXECUTION_FINDINGS_SOURCE_DISPLAY_NAME
            or source.fingerprint != expected_fingerprint
        ):
            raise ExecutionFindingsCatalogError(
                f"execution findings source drifted: {registration.tool_id!r}"
            )
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "schema_version": EXECUTION_FINDINGS_BINDING_ASSERTION_SCHEMA,
                "binding_kind": "host_execution_findings",
                **_facts_descriptor(facts),
                "source_fingerprint": expected_fingerprint,
            },
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise ExecutionFindingsCatalogError(
                f"execution findings effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise ExecutionFindingsCatalogError(
                f"execution findings descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)
    return (bindings[0], bindings[1])


def _manifest_item(
    *,
    spec: ToolSpec,
    implementation_ref: str,
    declared_behavior_revision: str,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> ExecutionFindingsToolDefinitionManifestItem:
    implementation_digest = _canonical_sha256(
        {
            "schema_version": EXECUTION_FINDINGS_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": implementation_ref,
            "declared_behavior_revision": declared_behavior_revision,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": (
                    EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION
                ),
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
        implementation_version=EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION,
        implementation_ref=implementation_ref,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return ExecutionFindingsToolDefinitionManifestItem(
        implementation_ref=implementation_ref,
        declared_behavior_revision=declared_behavior_revision,
        definition=definition,
    )


def _facts_descriptor(facts: ExecutionFindingsBindingFacts) -> dict[str, str]:
    return {
        "effect_scope_sha256": facts.effect_scope_sha256,
        "ledger_identity_sha256": facts.ledger_identity_sha256,
        "scope_key_authority_sha256": facts.scope_key_authority_sha256,
        "dispatcher_identity_sha256": facts.dispatcher_identity_sha256,
        "mutation_store_identity_sha256": facts.mutation_store_identity_sha256,
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
        raise ExecutionFindingsCatalogError(
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
    "EXECUTION_FINDINGS_BINDING_ASSERTION_SCHEMA",
    "EXECUTION_FINDINGS_DEFINITION_MANIFEST_SCHEMA",
    "EXECUTION_FINDINGS_SOURCE_FINGERPRINT_SCHEMA",
    "ExecutionFindingsBindingFacts",
    "ExecutionFindingsCatalogError",
    "ExecutionFindingsToolDefinitionManifestItem",
    "build_execution_findings_tool_bindings",
    "build_execution_findings_tool_definition_manifest",
    "derive_execution_findings_source_fingerprint",
]
