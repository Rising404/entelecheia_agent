from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from personagraph.tools.catalog.snapshots.attempt import (
    AttemptToolBindingOwnerKind,
    AttemptToolBindingOwner,
    AttemptToolCatalogProvenance,
    FrozenAttemptToolCatalog,
    encode_frozen_attempt_tool_catalog,
)
from personagraph.tools.catalog import CatalogStatus, ToolCatalog
from personagraph.tools.catalog.binding import (
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from personagraph.tools.catalog.persistence import (
    CatalogEntrySnapshot,
    CatalogRevisionSnapshot,
    DefaultCatalogResolutionState,
    DefaultProfileAvailability,
    DefaultProfileItem,
    DefaultProfileSnapshot,
    EmergencyRevocation,
    EmergencyRevocationSelector,
)
from personagraph.tools.contracts import (
    ToolCallProposal,
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.catalog.default_profile import (
    DefaultProfileMaterializationError,
    RequiredToolUnavailableError,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.catalog.revocation import (
    EmergencyRevocationDenied,
    EmergencyRevocationGuardStage,
)
from personagraph.tools.policy import TOOL_POLICY_VERSION
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration
from personagraph.tools.catalog.materialization import (
    RuntimeCatalogMaterializationError,
    RuntimeCatalogMaterializer,
)
from personagraph.tools.catalog.snapshots.bound import (
    FrozenBoundCatalog,
    FrozenBoundCatalogEntry,
)
from personagraph.tools.catalog.trusted_factories import (
    ToolFactoryUnavailableError,
    TrustedDefaultFactoryRegistry,
    TrustedStaticDefaultToolFactory,
)


_NOW = datetime(2026, 9, 3, 16, 30, tzinfo=timezone.utc)


class _Repository:
    def __init__(
        self,
        state: DefaultCatalogResolutionState,
        *,
        revocations: tuple[EmergencyRevocation, ...] = (),
    ) -> None:
        self.state = state
        self.revocations = revocations
        self.state_calls = 0
        self.revocation_calls = 0
        self.fail_state = False
        self.fail_revocations = False

    def default_resolution_state(self) -> DefaultCatalogResolutionState:
        self.state_calls += 1
        if self.fail_state:
            raise RuntimeError("ordinary catalog must not be read")
        return self.state

    def list_active_emergency_revocations(
        self,
    ) -> tuple[EmergencyRevocation, ...]:
        self.revocation_calls += 1
        if self.fail_revocations:
            raise RuntimeError("security state unavailable")
        return self.revocations


def _registration(
    tool_id: str,
    *,
    contract_version: str = "1",
) -> ToolRegistration:
    effects = ToolEffectProfile(
        (
            EffectDescriptor(
                EffectResource.RUNTIME_STATE,
                EffectAction.READ,
                EffectScopeKind.LOCAL,
                default_scope=tool_id,
            ),
        )
    )
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version=contract_version,
            name=f"Read {tool_id}",
            description=f"Read the bounded {tool_id} test value.",
            input_schema={"type": "object", "additionalProperties": False},
            output_schema={"type": "object"},
            catalog_tags=("read",),
        ),
        implementation_version="implementation-1",
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            f"test.{tool_id}.{contract_version}",
            fingerprint=f"source-{tool_id}-{contract_version}",
        ),
        handler=lambda _arguments: {"tool": tool_id},
        effect_profile=effects,
        execution_profile=ToolExecutionProfile(max_transparent_retries=0),
    )


def _factory(
    tool_id: str,
    *,
    contract_version: str = "1",
    registration_factory=None,
) -> TrustedStaticDefaultToolFactory:
    registration = _registration(tool_id, contract_version=contract_version)
    return TrustedStaticDefaultToolFactory.from_registration(
        implementation_ref=f"builtin/test-{tool_id}-{contract_version}",
        declared_behavior_revision=f"{tool_id}-{contract_version}-behavior-1",
        registration=registration,
        registration_factory=registration_factory,
    )


def _contextual_binding(
    definition: ToolDefinition,
    *,
    fingerprint: str | None = None,
) -> ToolBinding:
    selected_fingerprint = fingerprint or f"context-{definition.identity.tool_id}-1"
    return ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            f"context.{definition.identity.tool_id}",
            fingerprint=selected_fingerprint,
        ),
        handler=lambda _arguments: {"context": True},
        effect_profile=definition.effect_template,
        binding_assertion={"context_fingerprint": selected_fingerprint},
    )


def _state(
    selections: tuple[
        tuple[ToolDefinition, DefaultProfileAvailability],
        ...,
    ],
    *,
    current_statuses: dict[ToolIdentity, CatalogStatus | None] | None = None,
    current_digests: dict[ToolIdentity, str] | None = None,
) -> DefaultCatalogResolutionState:
    status_by_identity = current_statuses or {}
    digest_by_identity = current_digests or {}
    definitions = tuple(definition for definition, _ in selections)
    profile_entries = tuple(
        CatalogEntrySnapshot(
            definition.identity,
            definition.digest,
            CatalogStatus.ACTIVE,
        )
        for definition in definitions
    )
    current_entries = tuple(
        CatalogEntrySnapshot(
            definition.identity,
            digest_by_identity.get(definition.identity, definition.digest),
            status_by_identity.get(definition.identity, CatalogStatus.ACTIVE),
        )
        for definition in definitions
        if status_by_identity.get(definition.identity, CatalogStatus.ACTIVE)
        is not None
    )
    profile = DefaultProfileSnapshot(
        revision=7,
        catalog_revision=17,
        items=tuple(
            DefaultProfileItem(
                ordinal,
                definition.identity,
                definition.digest,
                availability,
            )
            for ordinal, (definition, availability) in enumerate(selections)
        ),
    )
    return DefaultCatalogResolutionState(
        current_catalog=CatalogRevisionSnapshot(41, current_entries),
        default_profile=profile,
        profile_catalog=CatalogRevisionSnapshot(17, profile_entries),
        definitions=definitions,
    )


def _revocation(tool_id: str) -> EmergencyRevocation:
    return EmergencyRevocation(
        revocation_id=f"revoke-{tool_id}",
        selector=EmergencyRevocationSelector(tool_id=tool_id),
        reason="test incident",
        issued_by="security",
        effective_at=_NOW,
        expires_at=None,
        created_at=_NOW,
    )


def _materializer(
    repository: _Repository,
    *factories: TrustedStaticDefaultToolFactory,
) -> RuntimeCatalogMaterializer:
    return RuntimeCatalogMaterializer(
        repository,
        TrustedDefaultFactoryRegistry(factories),
    )


def _new_mixed_catalog() -> tuple[
    _Repository,
    RuntimeCatalogMaterializer,
    ToolBinding,
]:
    static = _factory("zeta")
    contextual_definition = _factory("alpha").definition
    contextual = _contextual_binding(contextual_definition)
    repository = _Repository(
        _state(
            (
                (static.definition, DefaultProfileAvailability.REQUIRED),
                (
                    contextual_definition,
                    DefaultProfileAvailability.REQUIRED,
                ),
            )
        )
    )
    return repository, _materializer(repository, static), contextual


def test_new_materialization_uses_one_state_and_preserves_exposure_order() -> None:
    repository, materializer, contextual = _new_mixed_catalog()

    result = materializer.materialize_new(
        attempt_id="attempt-new",
        created_at=_NOW,
        contextual_bindings=(contextual,),
    )

    assert repository.state_calls == 1
    assert repository.revocation_calls == 2
    assert [item.identity.tool_id for item in result.exposed_definitions] == [
        "zeta",
        "alpha",
    ]
    assert result.exposure_order == tuple(
        item.identity for item in result.exposed_definitions
    )
    assert [entry.identity.tool_id for entry in result.frozen_attempt_catalog.frozen_catalog.entries] == [
        "alpha",
        "zeta",
    ]
    assert [entry.created_revision for entry in result.frozen_attempt_catalog.frozen_catalog.entries] == [
        1,
        2,
    ]
    assert result.frozen_attempt_catalog.frozen_catalog.revision == 2
    provenance = result.frozen_attempt_catalog.provenance
    assert provenance.source_catalog_revision == 41
    assert provenance.profile_catalog_revision == 17
    assert provenance.default_profile_revision == 7
    assert result.frozen_attempt_catalog.policy_version == TOOL_POLICY_VERSION
    assert [
        (owner.identity.tool_id, owner.kind)
        for owner in result.frozen_attempt_catalog.binding_owners
    ] == [
        ("alpha", AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE),
        ("zeta", AttemptToolBindingOwnerKind.TRUSTED_FACTORY),
    ]
    assert not hasattr(result, "snapshot")
    assert not hasattr(result, "registrations")
    assert all(not hasattr(item, "handler") for item in result.exposed_definitions)
    assert "handler" not in result.frozen_attempt_catalog.canonical_json


def test_contextual_candidates_cannot_expand_or_override_the_profile() -> None:
    selected = _factory("selected")
    outside = _factory("outside")
    repository = _Repository(
        _state(((selected.definition, DefaultProfileAvailability.REQUIRED),))
    )
    materializer = _materializer(repository, selected)

    with pytest.raises(DefaultProfileMaterializationError, match="outside"):
        materializer.materialize_new(
            attempt_id="attempt-outside",
            created_at=_NOW,
            contextual_bindings=(_contextual_binding(outside.definition),),
        )
    with pytest.raises(DefaultProfileMaterializationError, match="override"):
        materializer.materialize_new(
            attempt_id="attempt-override",
            created_at=_NOW,
            contextual_bindings=(_contextual_binding(selected.definition),),
        )
    with pytest.raises(TypeError, match="ToolBinding"):
        materializer.materialize_new(
            attempt_id="attempt-bound",
            created_at=_NOW,
            contextual_bindings=(
                selected.bind(selected.definition),  # type: ignore[arg-type]
            ),
        )


def test_duplicate_contextual_identity_and_profile_tool_name_fail_hard() -> None:
    contextual_definition = _factory("contextual").definition
    binding = _contextual_binding(contextual_definition)
    repository = _Repository(
        _state(
            ((contextual_definition, DefaultProfileAvailability.REQUIRED),)
        )
    )
    materializer = _materializer(repository)

    with pytest.raises(RuntimeCatalogMaterializationError, match="duplicate contextual"):
        materializer.materialize_new(
            attempt_id="attempt-duplicate-context",
            created_at=_NOW,
            contextual_bindings=(binding, binding),
        )

    first = _factory("duplicate", contract_version="1")
    second = _factory("duplicate", contract_version="2")
    duplicate_repository = _Repository(
        _state(
            (
                (first.definition, DefaultProfileAvailability.REQUIRED),
                (second.definition, DefaultProfileAvailability.REQUIRED),
            )
        )
    )
    with pytest.raises(RuntimeCatalogMaterializationError, match="duplicate tool_id"):
        _materializer(
            duplicate_repository,
            first,
            second,
        ).materialize_new(
            attempt_id="attempt-duplicate-name",
            created_at=_NOW,
        )
    assert duplicate_repository.revocation_calls == 0


@pytest.mark.parametrize(
    ("availability", "status", "expected_reason"),
    (
        (
            DefaultProfileAvailability.IF_AVAILABLE,
            None,
            "catalog_entry_absent",
        ),
        (
            DefaultProfileAvailability.IF_AVAILABLE,
            CatalogStatus.DISABLED,
            "catalog_status:disabled",
        ),
    ),
)
def test_optional_current_absence_or_inactivity_is_diagnostic_only(
    availability: DefaultProfileAvailability,
    status: CatalogStatus | None,
    expected_reason: str,
) -> None:
    factory = _factory("optional")
    repository = _Repository(
        _state(
            ((factory.definition, availability),),
            current_statuses={factory.identity: status},
        )
    )

    result = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-optional",
        created_at=_NOW,
    )

    assert result.exposed_definitions == ()
    assert result.unavailable[0].reason == expected_reason
    assert repository.revocation_calls == 0


@pytest.mark.parametrize("status", (None, CatalogStatus.DISABLED))
def test_required_current_absence_or_inactivity_fails(
    status: CatalogStatus | None,
) -> None:
    factory = _factory("required")
    repository = _Repository(
        _state(
            ((factory.definition, DefaultProfileAvailability.REQUIRED),),
            current_statuses={factory.identity: status},
        )
    )

    with pytest.raises(RequiredToolUnavailableError):
        _materializer(repository, factory).materialize_new(
            attempt_id="attempt-required",
            created_at=_NOW,
        )


def test_optional_factory_or_contextual_unavailability_is_diagnostic() -> None:
    def unavailable() -> ToolRegistration:
        raise ToolFactoryUnavailableError("dependency_unavailable")

    factory = _factory("factory-optional", registration_factory=unavailable)
    contextual_definition = _factory("context-optional").definition
    repository = _Repository(
        _state(
            (
                (factory.definition, DefaultProfileAvailability.IF_AVAILABLE),
                (
                    contextual_definition,
                    DefaultProfileAvailability.IF_AVAILABLE,
                ),
            )
        )
    )

    result = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-unavailable",
        created_at=_NOW,
    )

    assert result.exposed_definitions == ()
    assert [item.reason for item in result.unavailable] == [
        "dependency_unavailable",
        "contextual_binding_unavailable",
    ]


def test_digest_drift_is_never_downgraded_to_optional() -> None:
    factory = _factory("drift")
    repository = _Repository(
        _state(
            ((factory.definition, DefaultProfileAvailability.IF_AVAILABLE),),
            current_digests={factory.identity: "f" * 64},
        )
    )

    with pytest.raises(DefaultProfileMaterializationError, match="digest drifted"):
        _materializer(repository, factory).materialize_new(
            attempt_id="attempt-drift",
            created_at=_NOW,
        )


@pytest.mark.parametrize(
    "availability",
    (
        DefaultProfileAvailability.REQUIRED,
        DefaultProfileAvailability.IF_AVAILABLE,
    ),
)
def test_active_revoke_respects_availability_only_for_a_valid_match(
    availability: DefaultProfileAvailability,
) -> None:
    factory = _factory("revoked")
    repository = _Repository(
        _state(((factory.definition, availability),)),
        revocations=(_revocation("revoked"),),
    )
    materializer = _materializer(repository, factory)

    if availability is DefaultProfileAvailability.REQUIRED:
        with pytest.raises(RequiredToolUnavailableError, match="tool_emergency_revoked"):
            materializer.materialize_new(
                attempt_id="attempt-revoked-required",
                created_at=_NOW,
            )
    else:
        result = materializer.materialize_new(
            attempt_id="attempt-revoked-optional",
            created_at=_NOW,
        )
        assert result.exposed_definitions == ()
        assert result.unavailable[0].reason == "tool_emergency_revoked"


def test_unavailable_security_state_is_always_a_hard_denial() -> None:
    factory = _factory("security")
    repository = _Repository(
        _state(
            ((factory.definition, DefaultProfileAvailability.IF_AVAILABLE),)
        )
    )
    repository.fail_revocations = True

    with pytest.raises(EmergencyRevocationDenied) as caught:
        _materializer(repository, factory).materialize_new(
            attempt_id="attempt-security",
            created_at=_NOW,
        )
    assert caught.value.code == "emergency_revocation_state_unavailable"


def test_optional_tool_with_invalid_revocation_target_still_fails_hard() -> None:
    # Simulate a corrupt upstream value that bypassed ToolEffectProfile's normal
    # constructor. The materializer must not turn a security-target failure into
    # an IF_AVAILABLE omission.
    empty_effects = object.__new__(ToolEffectProfile)
    object.__setattr__(empty_effects, "effects", ())
    registration = replace(
        _registration("invalid-target"),
        effect_profile=empty_effects,
    )
    factory = TrustedStaticDefaultToolFactory.from_registration(
        implementation_ref="builtin/test-invalid-target",
        declared_behavior_revision="invalid-target-behavior-1",
        registration=registration,
    )
    repository = _Repository(
        _state(
            (
                (
                    factory.definition,
                    DefaultProfileAvailability.IF_AVAILABLE,
                ),
            )
        )
    )

    with pytest.raises(
        RuntimeCatalogMaterializationError,
        match="complete revocation target",
    ):
        _materializer(repository, factory).materialize_new(
            attempt_id="attempt-invalid-target",
            created_at=_NOW,
        )

    assert repository.revocation_calls == 0


def test_all_runtime_guard_entrypoints_reread_the_live_overlay() -> None:
    factory = _factory("guarded")
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    result = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-guarded",
        created_at=_NOW,
    )
    assert repository.revocation_calls == 1

    frozen_entry = result.resolve_for_proposal(
        ToolCallProposal(tool_id="guarded", arguments={})
    )
    assert isinstance(frozen_entry, FrozenBoundCatalogEntry)
    assert not hasattr(frozen_entry.binding, "handler")
    assert repository.revocation_calls == 2
    assert result.authorize_resume(factory.identity).stage is (
        EmergencyRevocationGuardStage.RESUME
    )
    assert repository.revocation_calls == 3
    assert result.authorize_execute(factory.identity).stage is (
        EmergencyRevocationGuardStage.EXECUTE
    )
    assert repository.revocation_calls == 4
    assert result.authorize_execute(factory.identity).stage is (
        EmergencyRevocationGuardStage.EXECUTE
    )
    assert repository.revocation_calls == 5

    repository.revocations = (_revocation("guarded"),)
    with pytest.raises(EmergencyRevocationDenied):
        result.resolve_for_proposal(
            ToolCallProposal(tool_id="guarded", arguments={})
        )
    with pytest.raises(EmergencyRevocationDenied):
        result.authorize_execute(factory.identity)
    assert repository.revocation_calls == 7


def test_exact_rebind_preserves_freeze_without_ordinary_or_resume_reads() -> None:
    repository, materializer, contextual = _new_mixed_catalog()
    original = materializer.materialize_new(
        attempt_id="attempt-rebind",
        created_at=_NOW,
        contextual_bindings=(contextual,),
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    repository.state_calls = 0
    repository.revocation_calls = 0
    repository.fail_state = True

    rebound = materializer.rebind_frozen(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
        expected_attempt_id="attempt-rebind",
        contextual_bindings=(contextual,),
    )

    assert repository.state_calls == 0
    assert repository.revocation_calls == 0
    assert rebound.frozen_attempt_catalog == original.frozen_attempt_catalog
    assert [item.identity for item in rebound.exposed_definitions] == list(
        original.exposure_order
    )
    assert rebound.authorize_resume(contextual.identity).stage is (
        EmergencyRevocationGuardStage.RESUME
    )
    assert repository.revocation_calls == 1


def test_frozen_trusted_owner_never_falls_back_to_contextual() -> None:
    factory = _factory("trusted-owner")
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-trusted-owner",
        created_at=_NOW,
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    exact_but_untrusted = factory.bind(factory.definition).binding

    with pytest.raises(RuntimeCatalogMaterializationError, match="no longer registered"):
        _materializer(repository).rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-trusted-owner",
            contextual_bindings=(exact_but_untrusted,),
        )


def test_frozen_contextual_owner_never_falls_forward_to_registry() -> None:
    factory = _factory("context-owner")
    contextual = _contextual_binding(factory.definition)
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository).materialize_new(
        attempt_id="attempt-context-owner",
        created_at=_NOW,
        contextual_bindings=(contextual,),
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )

    with pytest.raises(RuntimeCatalogMaterializationError, match="cannot replace"):
        _materializer(repository, factory).rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-context-owner",
            contextual_bindings=(contextual,),
        )


def test_frozen_contextual_binding_must_be_present_and_exact() -> None:
    factory = _factory("exact-context")
    contextual = _contextual_binding(factory.definition)
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository).materialize_new(
        attempt_id="attempt-exact-context",
        created_at=_NOW,
        contextual_bindings=(contextual,),
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    materializer = _materializer(repository)

    with pytest.raises(RuntimeCatalogMaterializationError, match="unavailable"):
        materializer.rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-exact-context",
        )
    with pytest.raises(RuntimeCatalogMaterializationError, match="exactly match"):
        materializer.rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-exact-context",
            contextual_bindings=(
                _contextual_binding(
                    factory.definition,
                    fingerprint="context-exact-context-2",
                ),
            ),
        )


def test_rebind_rejects_wrong_owner_policy_and_non_active_entries() -> None:
    factory = _factory("rebind-checks")
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-checks",
        created_at=_NOW,
    )

    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    with pytest.raises(RuntimeCatalogMaterializationError, match="different Attempt"):
        _materializer(repository, factory).rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-other",
        )

    historical = replace(
        original.frozen_attempt_catalog,
        policy_version="0.9.0",
    )
    historical_encoded = encode_frozen_attempt_tool_catalog(historical)
    with pytest.raises(RuntimeCatalogMaterializationError, match="policy version"):
        _materializer(repository, factory).rebind_frozen(
            canonical_json=historical_encoded.canonical_json,
            expected_sha256=historical_encoded.sha256,
            expected_attempt_id="attempt-checks",
        )

    disabled_entry = replace(
        original.frozen_attempt_catalog.frozen_catalog.entries[0],
        status=CatalogStatus.DISABLED,
    )
    disabled_catalog = replace(
        original.frozen_attempt_catalog.frozen_catalog,
        entries=(disabled_entry,),
    )
    disabled = replace(
        original.frozen_attempt_catalog,
        frozen_catalog=disabled_catalog,
    )
    disabled_encoded = encode_frozen_attempt_tool_catalog(disabled)
    with pytest.raises(RuntimeCatalogMaterializationError, match="must all be active"):
        _materializer(repository, factory).rebind_frozen(
            canonical_json=disabled_encoded.canonical_json,
            expected_sha256=disabled_encoded.sha256,
            expected_attempt_id="attempt-checks",
        )


def test_rebind_rejects_duplicate_tool_names_before_factory_binding() -> None:
    first = _factory("same-name", contract_version="1")
    second = _factory("same-name", contract_version="2")
    catalog = ToolCatalog()
    for factory in (first, second):
        catalog.register(factory.bind(factory.definition))
    frozen_catalog = FrozenBoundCatalog.from_catalog_snapshot(catalog.snapshot())
    frozen = FrozenAttemptToolCatalog(
        attempt_id="attempt-duplicate-rebind",
        created_at=_NOW,
        policy_version=TOOL_POLICY_VERSION,
        provenance=AttemptToolCatalogProvenance(
            source_catalog_revision=41,
            source_catalog_digest="a" * 64,
            default_profile_revision=7,
            default_profile_digest="b" * 64,
            profile_catalog_revision=17,
        ),
        frozen_catalog=frozen_catalog,
        exposure_order=(first.identity, second.identity),
        binding_owners=tuple(
            AttemptToolBindingOwner(
                identity=factory.identity,
                kind=AttemptToolBindingOwnerKind.TRUSTED_FACTORY,
            )
            for factory in (first, second)
        ),
    )
    encoded = encode_frozen_attempt_tool_catalog(frozen)
    repository = _Repository(_state(()))

    with pytest.raises(RuntimeCatalogMaterializationError, match="duplicate tool_id"):
        _materializer(repository, first, second).rebind_frozen(
            canonical_json=encoded.canonical_json,
            expected_sha256=encoded.sha256,
            expected_attempt_id="attempt-duplicate-rebind",
        )
    assert repository.state_calls == 0
    assert repository.revocation_calls == 0


def test_rebind_ignores_unselected_contextual_candidates_without_expansion() -> None:
    factory = _factory("selected-rebind")
    outside = _factory("outside-rebind")
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-extra-rebind",
        created_at=_NOW,
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )

    rebound = _materializer(repository, factory).rebind_frozen(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
        expected_attempt_id="attempt-extra-rebind",
        contextual_bindings=(_contextual_binding(outside.definition),),
    )

    assert [item.identity.tool_id for item in rebound.exposed_definitions] == [
        "selected-rebind"
    ]


def test_rebind_ignores_duplicate_unselected_contextual_candidates() -> None:
    factory = _factory("selected-rebind-duplicate-extra")
    outside = _factory("outside-rebind-duplicate-extra")
    repository = _Repository(
        _state(((factory.definition, DefaultProfileAvailability.REQUIRED),))
    )
    original = _materializer(repository, factory).materialize_new(
        attempt_id="attempt-duplicate-extra-rebind",
        created_at=_NOW,
    )
    encoded = encode_frozen_attempt_tool_catalog(
        original.frozen_attempt_catalog
    )
    duplicate_outside = _contextual_binding(outside.definition)

    rebound = _materializer(repository, factory).rebind_frozen(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
        expected_attempt_id="attempt-duplicate-extra-rebind",
        contextual_bindings=(duplicate_outside, duplicate_outside),
    )

    assert [item.identity.tool_id for item in rebound.exposed_definitions] == [
        "selected-rebind-duplicate-extra"
    ]


def test_repository_is_the_single_default_and_revocation_authority() -> None:
    class DefaultsOnly:
        def default_resolution_state(self):
            return _state(())

    with pytest.raises(TypeError, match="list_active_emergency_revocations"):
        RuntimeCatalogMaterializer(
            DefaultsOnly(),  # type: ignore[arg-type]
            TrustedDefaultFactoryRegistry(()),
        )
