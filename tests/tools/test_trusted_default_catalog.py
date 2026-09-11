from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from threading import Barrier

import pytest

from personagraph.tools.catalog.persistence import (
    CatalogRevisionSnapshot,
    DefaultCatalogResolutionState,
    DefaultProfileAvailability,
    DefaultProfileSnapshot,
    EmergencyRevocationSelector,
    ToolCatalogRepository,
    ToolCatalogSeed,
)
from personagraph.tools.catalog import CatalogConflictError, CatalogStatus
from personagraph.tools.composition.default_catalog import (
    ProductionDefaultProfileResolutionError,
    bootstrap_production_default_catalog,
    build_production_default_catalog_seeds,
    build_production_default_factory_registry,
    resolve_production_default_profile,
)
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from personagraph.tools.catalog.default_profile import (
    RequiredToolUnavailableError,
    materialize_default_catalog,
)
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration
from personagraph.tools.catalog.materialization import (
    RuntimeCatalogMaterializer,
)
from personagraph.tools.time.date_tools import DATE_TOOL_SOURCE_FINGERPRINT
from personagraph.tools.catalog.trusted_factories import (
    ToolFactoryDriftError,
    ToolFactoryUnavailableError,
    TrustedDefaultFactoryRegistry,
    TrustedStaticDefaultToolFactory,
    UnknownToolFactoryError,
)


def _registration(
    *,
    tool_id: str = "read_value",
    source: ToolSourceDescriptor | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="read-value-1",
            name="Read value",
            description="Read one deterministic value.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            output_schema={"type": "object"},
            catalog_tags=("read",),
        ),
        implementation_version="1",
        source=source
        or ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "personagraph.test.read_value",
            fingerprint="read-value-source-1",
        ),
        handler=lambda _payload: {"value": "ok"},
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    resource=EffectResource.RUNTIME_STATE,
                    action=EffectAction.READ,
                    scope_kind=EffectScopeKind.LOCAL,
                    default_scope="test_value",
                    idempotency=Idempotency.IDEMPOTENT,
                    reversibility=Reversibility.REVERSIBLE,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=1,
            hard_timeout_s=2,
            max_output_bytes=4096,
            max_transparent_retries=0,
        ),
    )


def _factory(
    registration: ToolRegistration | None = None,
    *,
    implementation_ref: str = "builtin/read_value",
    declared_behavior_revision: str = "read-value-handler-1",
    registration_factory=None,
) -> TrustedStaticDefaultToolFactory:
    manifest = registration or _registration()
    return TrustedStaticDefaultToolFactory.from_registration(
        implementation_ref=implementation_ref,
        declared_behavior_revision=declared_behavior_revision,
        registration=manifest,
        registration_factory=registration_factory,
    )


def _repository_with(
    tmp_path,
    factory: TrustedStaticDefaultToolFactory,
    availability: DefaultProfileAvailability = DefaultProfileAvailability.REQUIRED,
) -> ToolCatalogRepository:
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    repository.bootstrap((ToolCatalogSeed(factory.definition, availability),))
    return repository


def test_production_seed_separates_factory_and_contextual_owners() -> None:
    first = build_production_default_catalog_seeds()
    second = build_production_default_catalog_seeds()

    assert tuple(seed.definition.identity.tool_id for seed in first) == (
        "get_today",
        "date_after",
        "web_search",
        "web_fetch",
        "workspace_overview",
        "list_workspace_directory",
        "find_files",
        "search_text_files",
        "inspect_file",
        "read_text",
        "read_pdf_text",
        "read_word",
        "read_slides",
        "inspect_image",
        "analyze_image",
        "analyze_pdf_page",
        "retrieve_files",
        "check_files_state",
        "prepare_files",
        "read_file_chunks",
        "inspect_file_chunks",
        "search_file_text",
        "write_workspace_file",
        "create_output_file",
        "retrieve_history",
        "list_file_visuals",
        "read_file_visuals",
        "record_execution_findings",
        "revise_execution_finding",
        "list_tool_results",
        "read_tool_result",
    )
    assert tuple(seed.definition.digest for seed in first) == tuple(
        seed.definition.digest for seed in second
    )
    availability = tuple(item.availability for item in first)
    assert availability[:2] == (
        DefaultProfileAvailability.REQUIRED,
        DefaultProfileAvailability.REQUIRED,
    )
    assert set(availability[2:]) == {DefaultProfileAvailability.IF_AVAILABLE}
    factories = build_production_default_factory_registry().factories()
    assert tuple(factory.identity.tool_id for factory in factories) == (
        "get_today",
        "date_after",
        "web_search",
        "web_fetch",
    )
    assert tuple(factory.definition for factory in factories) == tuple(
        seed.definition for seed in first[:4]
    )
    static_registrations = {
        factory.identity.tool_id: factory.build_registration()
        for factory in factories
        if isinstance(factory, TrustedStaticDefaultToolFactory)
    }
    for seed in first:
        descriptor = seed.definition.descriptor()
        encoded = json.dumps(descriptor, sort_keys=True)
        assert "handler" not in encoded
        assert set(descriptor) == {
            "identity",
            "spec",
            "implementation",
            "effect_template",
            "execution",
        }
        assert seed.definition.implementation_ref.startswith("builtin/")
    assert static_registrations["get_today"].source.fingerprint == (
        DATE_TOOL_SOURCE_FINGERPRINT
    )
    assert static_registrations["date_after"].source.fingerprint == (
        DATE_TOOL_SOURCE_FINGERPRINT
    )


def test_production_catalog_restarts_into_the_same_bound_materialization(
    tmp_path,
) -> None:
    database_path = tmp_path / "catalog.sqlite"
    first_repository = ToolCatalogRepository(database_path)
    bootstrap = bootstrap_production_default_catalog(first_repository)
    first = RuntimeCatalogMaterializer(
        first_repository,
        build_production_default_factory_registry(),
    ).materialize_new(
        attempt_id="production-restart-test",
        created_at=datetime(2026, 9, 3, 17, 0, tzinfo=timezone.utc),
    )

    second_repository = ToolCatalogRepository(database_path)
    second_bootstrap = bootstrap_production_default_catalog(second_repository)
    second = RuntimeCatalogMaterializer(
        second_repository,
        build_production_default_factory_registry(),
    ).materialize_new(
        attempt_id="production-restart-test",
        created_at=datetime(2026, 9, 3, 17, 0, tzinfo=timezone.utc),
    )

    assert bootstrap.changed is True
    assert second_bootstrap.changed is False
    assert first.frozen_attempt_catalog == second.frozen_attempt_catalog
    provenance = first.frozen_attempt_catalog.provenance
    assert provenance.default_profile_revision == 1
    assert provenance.profile_catalog_revision == 1
    assert provenance.source_catalog_revision == 1
    assert tuple(item.identity.tool_id for item in first.unavailable) == (
        "workspace_overview",
        "list_workspace_directory",
        "find_files",
        "search_text_files",
        "inspect_file",
        "read_text",
        "read_pdf_text",
        "read_word",
        "read_slides",
        "inspect_image",
        "analyze_image",
        "analyze_pdf_page",
        "retrieve_files",
        "check_files_state",
        "prepare_files",
        "read_file_chunks",
        "inspect_file_chunks",
        "search_file_text",
        "write_workspace_file",
        "create_output_file",
        "retrieve_history",
        "list_file_visuals",
        "read_file_visuals",
        "record_execution_findings",
        "revise_execution_finding",
        "list_tool_results",
        "read_tool_result",
    )
    assert {item.reason for item in first.unavailable} == {
        "contextual_binding_unavailable"
    }
    assert tuple(item.identity.tool_id for item in first.exposed_definitions) == (
        "get_today",
        "date_after",
        "web_search",
        "web_fetch",
    )


def test_production_profile_hard_cut_converges_after_cas_race(
    tmp_path,
) -> None:
    database_path = tmp_path / "catalog.sqlite"
    seeds = build_production_default_catalog_seeds()
    custom = _factory().definition
    ToolCatalogRepository(database_path).bootstrap(
        (ToolCatalogSeed(custom),)
    )
    first_status_gate = Barrier(2)
    conflicts: list[CatalogConflictError] = []

    class RacingRepository(ToolCatalogRepository):
        def __init__(self) -> None:
            super().__init__(database_path)
            self._first_status = True

        def set_status(self, *args, **kwargs):
            if self._first_status:
                self._first_status = False
                first_status_gate.wait(timeout=5)
            try:
                return super().set_status(*args, **kwargs)
            except CatalogConflictError as exc:
                conflicts.append(exc)
                raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda _: bootstrap_production_default_catalog(
                    RacingRepository()
                ),
                range(2),
            )
        )

    repository = ToolCatalogRepository(database_path)
    snapshot = repository.current_snapshot()
    profile = repository.current_default_profile()
    assert conflicts
    assert snapshot.revision == len(seeds) + 2
    assert {entry.status for entry in snapshot.entries} == {CatalogStatus.ACTIVE}
    assert profile.revision == 2
    assert profile.catalog_revision == snapshot.revision
    assert tuple(item.identity for item in profile.items) == tuple(
        seed.definition.identity for seed in seeds
    )
    assert all(result.default_profile == profile for result in results)
    assert bootstrap_production_default_catalog(repository).changed is False


def test_reconciliation_reactivates_a_disabled_canonical_identity(
    tmp_path,
) -> None:
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds[:4])
    repository.set_status(
        seeds[0].definition.identity,
        CatalogStatus.DISABLED,
        expected_revision=1,
        actor="operator",
        reason="disable one predecessor default",
    )

    result = bootstrap_production_default_catalog(repository)

    assert result.catalog.revision == len(seeds)
    assert tuple(
        item.identity for item in result.default_profile.items
    ) == tuple(seed.definition.identity for seed in seeds)
    assert result.default_profile.catalog_revision == result.catalog.revision
    status_by_identity = {
        entry.identity: entry.status for entry in result.catalog.entries
    }
    assert all(
        status_by_identity[seed.definition.identity] is CatalogStatus.ACTIVE
        for seed in seeds
    )


def test_production_profile_resolver_reads_exact_handler_free_profile_once(
    tmp_path,
) -> None:
    class CountingRepository(ToolCatalogRepository):
        reads = 0

        def default_resolution_state(self):
            self.reads += 1
            return super().default_resolution_state()

    repository = CountingRepository(tmp_path / "catalog.sqlite")
    bootstrap_production_default_catalog(repository)
    repository.reads = 0

    profile = resolve_production_default_profile(repository)

    assert repository.reads == 1
    assert profile.catalog_revision == 1
    assert profile.profile_catalog_revision == 1
    assert profile.profile_revision == 1
    expected_count = len(build_production_default_catalog_seeds())
    assert len(profile.items) == expected_count
    assert tuple(item.ordinal for item in profile.items) == tuple(range(expected_count))
    assert profile.definitions == tuple(
        item.definition for item in profile.items
    )
    assert tuple(item.definition.identity.tool_id for item in profile.items)[-4:] == (
        "record_execution_findings",
        "revise_execution_finding",
        "list_tool_results",
        "read_tool_result",
    )
    encoded = json.dumps(
        [item.definition.descriptor() for item in profile.items],
        sort_keys=True,
    )
    assert "handler" not in encoded


def test_production_profile_resolver_rejects_a_custom_profile(tmp_path) -> None:
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds[:4])

    with pytest.raises(
        ProductionDefaultProfileResolutionError,
        match="not the canonical L1 production profile",
    ):
        resolve_production_default_profile(repository)


def test_reconciliation_retires_an_active_contract_implementation(
    tmp_path,
) -> None:
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds[:4])
    previous_profile = repository.current_default_profile()
    alternative = replace(
        seeds[4].definition,
        implementation_version="operator-implementation",
    )
    staged = repository.register_draft(
        alternative,
        expected_revision=1,
        actor="operator",
        reason="stage an alternative workspace implementation",
    )
    repository.set_status(
        alternative.identity,
        CatalogStatus.ACTIVE,
        expected_revision=staged.revision,
        actor="operator",
        reason="activate the alternative workspace implementation",
    )
    revocation = repository.issue_emergency_revocation(
        "keep-across-default-hard-cut",
        EmergencyRevocationSelector(tool_id=seeds[0].definition.identity.tool_id),
        issued_by="security",
        reason="deny remains authoritative during profile migration",
    )

    result = bootstrap_production_default_catalog(repository)

    assert result.catalog.revision == len(seeds) + 1
    assert result.default_profile != previous_profile
    assert tuple(
        item.identity for item in result.default_profile.items
    ) == tuple(seed.definition.identity for seed in seeds)
    status_by_identity = {
        entry.identity: entry.status for entry in result.catalog.entries
    }
    assert status_by_identity[alternative.identity] is CatalogStatus.RETIRED
    assert all(
        status_by_identity[seed.definition.identity] is CatalogStatus.ACTIVE
        for seed in seeds[4:]
    )
    audit_events = repository.list_audit_events()
    assert any(
        event.action == "set_status"
        and event.reason
        == (
            "retire active implementation superseded by canonical L1 "
            "production default"
        )
        and event.change["identity"] == alternative.identity.to_dict()
        for event in audit_events
    )
    assert audit_events[-1].action == "publish_default_profile"
    assert audit_events[-1].reason == "publish canonical L1 production defaults"
    assert repository.list_active_emergency_revocations() == (revocation,)


def test_reconciliation_fails_closed_for_a_retired_canonical_identity(
    tmp_path,
) -> None:
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    seeds = build_production_default_catalog_seeds()
    repository.bootstrap(seeds)
    retired = repository.set_status(
        seeds[0].definition.identity,
        CatalogStatus.RETIRED,
        expected_revision=1,
        actor="operator",
        reason="permanently retire canonical implementation",
    )
    audit_before = repository.list_audit_events()

    with pytest.raises(
        CatalogConflictError,
        match="retired canonical production ToolDefinition cannot be reactivated",
    ):
        bootstrap_production_default_catalog(repository)

    assert repository.current_snapshot() == retired
    assert repository.list_audit_events() == audit_before


def test_unknown_dotted_reference_never_triggers_dynamic_import(
    tmp_path,
    monkeypatch,
) -> None:
    original = _factory()
    unknown = replace(
        original.definition,
        spec=replace(original.definition.spec, tool_id="unknown_tool"),
        implementation_ref="os.system",
    )
    repository = ToolCatalogRepository(tmp_path / "catalog.sqlite")
    repository.bootstrap(
        (
            ToolCatalogSeed(
                unknown,
                DefaultProfileAvailability.IF_AVAILABLE,
            ),
        )
    )
    imported: list[str] = []
    real_import = importlib.import_module

    def recording_import(name: str, package: str | None = None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", recording_import)

    with pytest.raises(UnknownToolFactoryError, match="unknown trusted factory"):
        materialize_default_catalog(
            repository,
            TrustedDefaultFactoryRegistry((original,)),
        )

    assert imported == []


def test_registry_rejects_duplicate_factory_key_or_identity() -> None:
    first = _factory()
    duplicate_identity = _factory(implementation_ref="builtin/another_ref")

    with pytest.raises(ValueError, match="duplicate trusted factory key"):
        TrustedDefaultFactoryRegistry((first, first))
    with pytest.raises(ValueError, match="duplicate trusted factory identity"):
        TrustedDefaultFactoryRegistry((first, duplicate_identity))


def test_registry_ownership_is_read_only_and_identity_safe() -> None:
    factory = _factory()
    registry = TrustedDefaultFactoryRegistry((factory,))
    unknown = replace(
        factory.definition,
        spec=replace(factory.definition.spec, tool_id="unknown_tool"),
        implementation_ref="builtin/not_registered",
    )
    changed_ref = replace(
        factory.definition,
        implementation_ref="builtin/not_registered",
    )
    wrong_identity = replace(
        factory.definition,
        spec=replace(factory.definition.spec, tool_id="another_tool"),
    )

    assert registry.owns(factory.definition) is True
    assert registry.owns(unknown) is False
    with pytest.raises(ToolFactoryDriftError, match="implementation reference"):
        registry.owns(changed_ref)
    with pytest.raises(ToolFactoryDriftError, match="different tool identity"):
        registry.owns(wrong_identity)
    with pytest.raises(ToolFactoryDriftError, match="identity drifted"):
        registry.resolve(wrong_identity)


def test_declared_behavior_revision_drift_fails_after_restart(
    tmp_path,
) -> None:
    old_factory = _factory(declared_behavior_revision="handler-1")
    repository = _repository_with(tmp_path, old_factory)
    drifted_factory = _factory(declared_behavior_revision="handler-2")

    with pytest.raises(ToolFactoryDriftError, match="implementation digest"):
        materialize_default_catalog(
            repository,
            TrustedDefaultFactoryRegistry((drifted_factory,)),
        )


def test_disabled_optional_tool_cannot_hide_trusted_definition_drift(
    tmp_path,
) -> None:
    original = _factory(declared_behavior_revision="handler-1")
    repository = _repository_with(
        tmp_path,
        original,
        DefaultProfileAvailability.IF_AVAILABLE,
    )
    repository.set_status(
        original.identity,
        CatalogStatus.DISABLED,
        expected_revision=1,
        actor="test",
        reason="disable new use",
    )
    drifted = _factory(declared_behavior_revision="handler-2")

    with pytest.raises(ToolFactoryDriftError, match="implementation digest"):
        materialize_default_catalog(
            repository,
            TrustedDefaultFactoryRegistry((drifted,)),
        )


@pytest.mark.parametrize(
    ("changed_registration", "message"),
    (
        (
            replace(
                _registration(),
                spec=replace(
                    _registration().spec,
                    description="Drifted description.",
                ),
            ),
            "tool spec",
        ),
        (
            replace(
                _registration(),
                effect_profile=ToolEffectProfile(
                    (
                        replace(
                            _registration().effect_profile.effects[0],
                            default_scope="another_value",
                        ),
                    )
                ),
            ),
            "effect template",
        ),
        (
            replace(
                _registration(),
                execution_profile=replace(
                    _registration().execution_profile,
                    max_output_bytes=8192,
                ),
            ),
            "execution profile",
        ),
        (
            _registration(
                source=ToolSourceDescriptor(
                    ToolSourceKind.LOCAL,
                    "personagraph.test.read_value",
                    fingerprint="read-value-source-2",
                )
            ),
            "source fingerprint",
        ),
        (_registration(tool_id="another_tool"), "tool identity"),
    ),
)
def test_live_factory_drift_fails_closed(
    changed_registration: ToolRegistration,
    message: str,
) -> None:
    manifest = _registration()
    factory = _factory(
        manifest,
        registration_factory=lambda: changed_registration,
    )

    with pytest.raises(ToolFactoryDriftError, match=message):
        TrustedDefaultFactoryRegistry((factory,)).resolve(factory.definition)


@pytest.mark.parametrize(
    "availability",
    (
        DefaultProfileAvailability.REQUIRED,
        DefaultProfileAvailability.IF_AVAILABLE,
    ),
)
def test_profile_availability_handles_only_explicit_factory_unavailability(
    tmp_path,
    availability: DefaultProfileAvailability,
) -> None:
    def unavailable() -> ToolRegistration:
        raise ToolFactoryUnavailableError("optional dependency is not configured")

    factory = _factory(registration_factory=unavailable)
    repository = _repository_with(tmp_path, factory, availability)
    registry = TrustedDefaultFactoryRegistry((factory,))

    if availability is DefaultProfileAvailability.REQUIRED:
        with pytest.raises(RequiredToolUnavailableError, match="required default"):
            materialize_default_catalog(repository, registry)
    else:
        materialized = materialize_default_catalog(repository, registry)
        assert materialized.registrations == ()
        assert len(materialized.unavailable) == 1
        assert materialized.unavailable[0].reason == (
            "optional dependency is not configured"
        )


@pytest.mark.parametrize(
    "availability",
    (
        DefaultProfileAvailability.REQUIRED,
        DefaultProfileAvailability.IF_AVAILABLE,
    ),
)
def test_current_disabled_status_blocks_only_new_default_materialization(
    tmp_path,
    availability: DefaultProfileAvailability,
) -> None:
    factory = _factory()
    repository = _repository_with(tmp_path, factory, availability)
    repository.set_status(
        factory.identity,
        CatalogStatus.DISABLED,
        expected_revision=1,
        actor="test",
        reason="disable new use",
    )
    registry = TrustedDefaultFactoryRegistry((factory,))

    if availability is DefaultProfileAvailability.REQUIRED:
        with pytest.raises(RequiredToolUnavailableError, match="catalog_status:disabled"):
            materialize_default_catalog(repository, registry)
    else:
        materialized = materialize_default_catalog(repository, registry)
        assert materialized.registrations == ()
        assert materialized.catalog_revision == 2
        assert materialized.profile_catalog_revision == 1
        assert materialized.unavailable[0].reason == "catalog_status:disabled"

    # Exact rebind is lifecycle-neutral. Runtime's future frozen-snapshot path must
    # add its own emergency-revoke guard before using this registration.
    assert registry.resolve(factory.definition).tool_id == "read_value"


def test_atomic_resolution_state_rejects_a_current_revision_from_the_past() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        DefaultCatalogResolutionState(
            current_catalog=CatalogRevisionSnapshot(0, ()),
            default_profile=DefaultProfileSnapshot(0, 1, ()),
            profile_catalog=CatalogRevisionSnapshot(1, ()),
            definitions=(),
        )


@pytest.mark.parametrize("extra_tool_id", ("get_today", "unexpected"))
def test_production_manifest_rejects_duplicate_or_extra_family_registrations(
    monkeypatch,
    extra_tool_id: str,
) -> None:
    from personagraph.tools.composition import default_catalog

    original = default_catalog.build_date_tool_registrations()
    extra = replace(
        original[0],
        spec=replace(original[0].spec, tool_id=extra_tool_id),
    )
    monkeypatch.setattr(
        default_catalog,
        "build_date_tool_registrations",
        lambda: (*original, extra),
    )

    with pytest.raises(RuntimeError, match="registration family drifted"):
        default_catalog.build_production_default_catalog_seeds()


def test_bootstrap_retires_candidate_tool_ids_and_old_chunk_contracts(tmp_path):
    retired_ids = ('list_file_candidates', 'select_file_candidates',
                   'prepare_file_candidates', 'retrieve_file_candidates', 'read_file_chunks')
    obsolete = tuple(_factory(_registration(tool_id=tool_id),
                             implementation_ref=f'builtin/{tool_id}').definition
                     for tool_id in retired_ids)
    repository = ToolCatalogRepository(tmp_path / 'catalog.sqlite')
    repository.bootstrap(tuple(ToolCatalogSeed(definition) for definition in obsolete))
    result = bootstrap_production_default_catalog(repository)
    statuses = {entry.identity: entry.status for entry in result.catalog.entries}
    assert all(statuses[item.identity] is CatalogStatus.RETIRED for item in obsolete)
    assert {item.identity.tool_id for item in result.default_profile.items}.isdisjoint(retired_ids[:-1])
    assert any(item.identity.tool_id == 'read_file_chunks' for item in result.default_profile.items)
    assert bootstrap_production_default_catalog(repository).changed is False
