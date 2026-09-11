from __future__ import annotations

import json
from dataclasses import replace

import pytest

from personagraph.tools.catalog import CatalogConflictError, CatalogError, ToolCatalog
from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from personagraph.tools.contracts import (
    ExecutionStatus,
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.policy import (
    AuthorityFacts,
    PolicyDisposition,
    ScopeGrant,
    PolicyRequest,
    ToolPolicyCore,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration


IMPLEMENTATION_DIGEST = "a" * 64


def _spec(*, description: str = "Read a bounded value.") -> ToolSpec:
    return ToolSpec(
        tool_id="read_value",
        contract_version="read-value-v1",
        name="Read value",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        catalog_tags=("read",),
    )


def _effects(*, scope: str = "*") -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.WORKSPACE,
                default_scope=scope,
            ),
        )
    )


def _definition(
    *,
    description: str = "Read a bounded value.",
    implementation_version: str = "1",
) -> ToolDefinition:
    return ToolDefinition(
        spec=_spec(description=description),
        implementation_version=implementation_version,
        implementation_ref="personagraph.tools.read_value",
        implementation_digest=IMPLEMENTATION_DIGEST,
        effect_template=_effects(),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=5,
            hard_timeout_s=10,
            max_output_bytes=4096,
            max_transparent_retries=0,
        ),
    )


def _binding(
    definition: ToolDefinition,
    *,
    scope: str = "/workspace/one",
    fingerprint: str = "workspace-one",
) -> ToolBinding:
    return ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "personagraph.workspace.read",
            fingerprint=fingerprint,
        ),
        handler=lambda payload: {"value": payload["key"]},
        effect_profile=_effects(scope=scope),
        binding_assertion={
            "scope_sha256": "b" * 64,
            "resources": {"workspace": scope},
        },
    )


def test_definition_is_canonical_json_and_excludes_dynamic_binding() -> None:
    definition = _definition()

    encoded = json.dumps(
        definition.descriptor(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    assert len(definition.digest) == 64
    assert definition.digest == _definition().digest
    assert "handler" not in encoded
    assert "/workspace/one" not in encoded
    assert definition.identity == ToolIdentity(
        "read_value",
        "read-value-v1",
        "1",
    )


def test_definition_identity_and_implementation_ref_are_canonicalized() -> None:
    identity = ToolIdentity(" read_value ", " read-value-v1 ", " 1 ")
    definition = replace(
        _definition(),
        implementation_version=" 1 ",
        implementation_ref=" personagraph.tools.read_value ",
    )

    assert identity == ToolIdentity("read_value", "read-value-v1", "1")
    assert definition.identity.implementation_version == "1"
    assert definition.descriptor()["implementation"]["ref"] == (
        "personagraph.tools.read_value"
    )


def test_binding_requires_an_exact_source_fingerprint() -> None:
    definition = _definition()

    with pytest.raises(ValueError, match="source fingerprint"):
        ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=ToolSourceDescriptor(
                ToolSourceKind.LOCAL,
                "personagraph.workspace.read",
            ),
            handler=lambda payload: payload,
            effect_profile=_effects(scope="/workspace/one"),
            binding_assertion={"scope_sha256": "b" * 64},
        )


def test_binding_digest_commits_scope_and_source_but_not_handler_object() -> None:
    definition = _definition()
    first = _binding(definition)
    equivalent = _binding(definition)
    another_scope = _binding(definition, scope="/workspace/two")
    another_source = _binding(definition, fingerprint="workspace-two")

    assert first.digest == equivalent.digest
    assert first.digest != another_scope.digest
    assert first.digest != another_source.digest
    assert "handler" not in first.descriptor()
    with pytest.raises(TypeError):
        first.binding_assertion["scope_sha256"] = "changed"  # type: ignore[index]


def test_bound_registration_rejects_definition_or_identity_drift() -> None:
    definition = _definition()
    binding = _binding(definition)

    with pytest.raises(ValueError, match="definition digest"):
        BoundToolRegistration(
            definition,
            ToolBinding(
                identity=definition.identity,
                definition_digest="c" * 64,
                source=binding.source,
                handler=binding.handler,
                effect_profile=binding.effect_profile,
                binding_assertion=binding.binding_assertion,
            ),
        )

    with pytest.raises(ValueError, match="identity"):
        BoundToolRegistration(
            definition,
            ToolBinding(
                identity=ToolIdentity(
                    "another_tool",
                    definition.identity.contract_version,
                    definition.identity.implementation_version,
                ),
                definition_digest=definition.digest,
                source=binding.source,
                handler=binding.handler,
                effect_profile=binding.effect_profile,
                binding_assertion=binding.binding_assertion,
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("resource", EffectResource.NETWORK),
        ("action", EffectAction.SEARCH),
        ("scope_kind", EffectScopeKind.SESSION),
        ("data_egress", DataEgress.CONTENT),
        ("idempotency", Idempotency.IDEMPOTENT),
        ("reversibility", Reversibility.REVERSIBLE),
    ),
)
def test_binding_cannot_change_declared_effect_shape(
    field_name: str,
    changed_value: object,
) -> None:
    definition = _definition()
    binding = _binding(definition)
    changed_effect = replace(
        binding.effect_profile.effects[0],
        **{field_name: changed_value},
    )
    changed_binding = replace(
        binding,
        effect_profile=ToolEffectProfile((changed_effect,)),
    )

    with pytest.raises(ValueError, match="effects"):
        BoundToolRegistration(definition, changed_binding)


def test_binding_can_only_specialize_a_wildcard_default_scope() -> None:
    wildcard_definition = _definition()
    BoundToolRegistration(wildcard_definition, _binding(wildcard_definition))

    fixed_definition = replace(
        wildcard_definition,
        effect_template=_effects(scope="/workspace/fixed"),
    )
    with pytest.raises(ValueError, match="scope exceeds"):
        BoundToolRegistration(fixed_definition, _binding(fixed_definition))


def test_bound_registration_keeps_unbound_registration_descriptor_shape() -> None:
    definition = _definition()
    binding = _binding(definition)
    bound = BoundToolRegistration(definition, binding)
    unbound = ToolRegistration(
        spec=definition.spec,
        implementation_version=definition.implementation_version,
        source=binding.source,
        handler=binding.handler,
        effect_profile=binding.effect_profile,
        execution_profile=definition.execution_profile,
    )

    assert bound.descriptor() == unbound.descriptor()
    assert bound.definition_digest == definition.digest
    assert bound.binding_digest == binding.digest
    assert bound.digest == BoundToolRegistration(definition, _binding(definition)).digest


def test_bound_registration_runs_through_policy_and_executor() -> None:
    definition = _definition()
    bound = BoundToolRegistration(definition, _binding(definition))
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.WORKSPACE,
                "/workspace/one",
            ),
        )
    )

    decision = ToolPolicyCore().evaluate(
        PolicyRequest.from_registration(
            bound,
            {"key": "expected"},
            authority=authority,
        )
    )
    outcome = ToolExecutor().execute(
        ResolvedInvocation(bound, {"key": "expected"})
    )

    assert decision.disposition is PolicyDisposition.ALLOW
    assert outcome.status is ExecutionStatus.SUCCEEDED
    assert outcome.result == {"value": "expected"}


def test_catalog_rejects_definition_drift_under_the_same_identity() -> None:
    original = _definition()
    drifted = _definition(description="A changed contract under the same identity.")
    catalog = ToolCatalog()
    catalog.register(BoundToolRegistration(original, _binding(original)))

    with pytest.raises(CatalogConflictError, match="ToolIdentity"):
        catalog.register(
            BoundToolRegistration(drifted, _binding(drifted)),
            replace=True,
            expected_revision=catalog.revision,
        )

    assert catalog.revision == 1
    assert catalog.snapshot().resolve("read_value").definition_digest == original.digest


@pytest.mark.parametrize("implementation_version", ("1", "2"))
def test_catalog_cannot_replace_a_bound_entry_with_unbound_registration(
    implementation_version: str,
) -> None:
    definition = _definition()
    binding = _binding(definition)
    catalog = ToolCatalog()
    catalog.register(BoundToolRegistration(definition, binding))
    unbound = ToolRegistration(
        spec=definition.spec,
        implementation_version=implementation_version,
        source=binding.source,
        handler=binding.handler,
        effect_profile=binding.effect_profile,
        execution_profile=definition.execution_profile,
    )

    with pytest.raises(CatalogConflictError, match="unbound ToolRegistration"):
        catalog.register(
            unbound,
            replace=True,
            expected_revision=catalog.revision,
        )

    assert catalog.revision == 1


def test_catalog_has_an_explicit_digest_preserving_bound_snapshot_projection() -> None:
    definition = _definition()
    bound = BoundToolRegistration(definition, _binding(definition))
    catalog = ToolCatalog()
    catalog.register(bound)

    descriptor_projection = catalog.snapshot().to_descriptor()
    bound_projection = catalog.snapshot().to_bound_descriptor()

    assert descriptor_projection["entries"][0]["registration"] == bound.descriptor()
    exact_registration = bound_projection["entries"][0]["registration"]
    assert exact_registration["definition_digest"] == bound.definition_digest
    assert exact_registration["binding_digest"] == bound.binding_digest


def test_bound_snapshot_projection_rejects_unbound_entries() -> None:
    definition = _definition()
    binding = _binding(definition)
    catalog = ToolCatalog()
    catalog.register(
        ToolRegistration(
            spec=definition.spec,
            implementation_version=definition.implementation_version,
            source=binding.source,
            handler=binding.handler,
            effect_profile=binding.effect_profile,
            execution_profile=definition.execution_profile,
        )
    )

    with pytest.raises(CatalogError, match="BoundToolRegistration"):
        catalog.snapshot().to_bound_descriptor()


def test_catalog_allows_a_new_implementation_identity_for_the_same_contract() -> None:
    original = _definition()
    replacement = _definition(implementation_version="2")
    catalog = ToolCatalog()
    catalog.register(BoundToolRegistration(original, _binding(original)))

    catalog.register(
        BoundToolRegistration(replacement, _binding(replacement)),
        replace=True,
        expected_revision=catalog.revision,
    )

    resolved = catalog.snapshot().resolve("read_value")
    assert resolved.identity.implementation_version == "2"
    assert catalog.revision == 2
