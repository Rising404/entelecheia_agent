from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest

from personagraph.configuration.paths import STATE_DIR
from personagraph.tools.catalog import CatalogConflictError, CatalogError, CatalogStatus
from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
)
from personagraph.tools.catalog.persistence import (
    CATALOG_DATABASE_SCHEMA_VERSION,
    DEFAULT_TOOL_CATALOG_DATABASE_PATH,
    CatalogCorruptionError,
    CatalogSchemaError,
    DefaultProfileAvailability,
    DefaultProfileSelection,
    EmergencyRevocation,
    EmergencyRevocationSelector,
    EmergencyRevocationTarget,
    ToolCatalogRepository,
    ToolCatalogSeed,
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
    ToolEffectProfile,
)
from personagraph.tools.registration import ToolExecutionProfile


_NOW = datetime(2026, 9, 3, 12, 30, tzinfo=timezone.utc)


def _definition(
    tool_id: str,
    *,
    description: str | None = None,
    implementation_version: str = "impl-1",
) -> ToolDefinition:
    return ToolDefinition(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version=f"{tool_id}-v1",
            name=tool_id.replace("_", " ").title(),
            description=description or f"Use the bounded {tool_id} capability.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            catalog_tags=("tooling", "read"),
        ),
        implementation_version=implementation_version,
        implementation_ref=f"personagraph.tools.factories.{tool_id}",
        implementation_digest="a" * 64,
        effect_template=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.MEMORY,
                    EffectAction.READ,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=5,
            hard_timeout_s=10,
            max_output_bytes=4096,
            max_transparent_retries=0,
        ),
    )


def _repository(path: Path) -> ToolCatalogRepository:
    return ToolCatalogRepository(path, clock=lambda: _NOW)


def _binding(
    definition: ToolDefinition,
    *,
    source_fingerprint: str = "local-source-v1",
) -> ToolBinding:
    return ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "test-provider",
            fingerprint=source_fingerprint,
        ),
        handler=lambda *_: None,
        effect_profile=definition.effect_template,
        binding_assertion={},
    )


def test_default_path_and_fresh_schema_are_global_and_versioned(tmp_path: Path) -> None:
    path = tmp_path / "tool_catalog.sqlite"
    repository = _repository(path)

    assert DEFAULT_TOOL_CATALOG_DATABASE_PATH == STATE_DIR / "tool_catalog.sqlite"
    assert repository.current_snapshot().revision == 0
    assert repository.current_snapshot().entries == ()
    assert repository.current_default_profile().revision == 0
    assert repository.current_default_profile().items == ()

    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == (
            CATALOG_DATABASE_SCHEMA_VERSION
        )
        assert str(connection.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"
    assert {
        "tool_catalog_control",
        "tool_definitions",
        "tool_catalog_entries",
        "tool_catalog_revisions",
        "tool_default_profile_revisions",
        "tool_default_profile_items",
        "tool_catalog_tombstones",
        "tool_emergency_revocations",
        "tool_catalog_audit_events",
        "tool_catalog_bootstrap_manifests",
    } <= tables


def test_concurrent_first_initialization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "tool_catalog.sqlite"
    worker_count = 12
    creation_gate = Barrier(worker_count)

    class SynchronizedRepository(ToolCatalogRepository):
        def _create_empty_database(self, connection: sqlite3.Connection) -> None:
            creation_gate.wait(timeout=10)
            super()._create_empty_database(connection)

    def initialize_one(_: int) -> None:
        SynchronizedRepository(path, clock=lambda: _NOW).initialize()

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(initialize_one, range(worker_count)))

    repository = _repository(path)
    assert repository.current_snapshot().revision == 0
    assert repository.current_default_profile().revision == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM tool_catalog_control"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM tool_catalog_revisions"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM tool_default_profile_revisions"
        ).fetchone() == (1,)


def test_initial_seed_survives_restart_and_preserves_profile_order(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    first = _definition("zeta")
    second = _definition("alpha")
    result = _repository(path).bootstrap(
        (
            ToolCatalogSeed(first, DefaultProfileAvailability.REQUIRED),
            ToolCatalogSeed(second),
        )
    )

    reopened = _repository(path)
    catalog = reopened.current_snapshot()
    profile = reopened.current_default_profile()

    assert result.changed is True
    assert catalog.revision == 1
    assert [entry.identity.tool_id for entry in catalog.entries] == ["alpha", "zeta"]
    assert {entry.status for entry in catalog.entries} == {CatalogStatus.ACTIVE}
    assert profile.revision == 1
    assert profile.catalog_revision == 1
    assert [item.identity.tool_id for item in profile.items] == ["zeta", "alpha"]
    assert [item.availability for item in profile.items] == [
        DefaultProfileAvailability.REQUIRED,
        DefaultProfileAvailability.IF_AVAILABLE,
    ]
    assert reopened.load_definition(first.identity).descriptor() == first.descriptor()
    assert reopened.load_definition(second.identity).digest == second.digest

    with sqlite3.connect(path) as connection:
        persisted = "\n".join(
            str(row[0])
            for row in connection.execute(
                "SELECT descriptor_json FROM tool_definitions ORDER BY tool_id"
            )
        )
    assert "handler" not in persisted
    assert "provider" not in persisted
    assert "credential" not in persisted
    assert "session_id" not in persisted
    assert "binding_assertion" not in persisted


def test_same_seed_manifest_is_a_true_no_op(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    seed = ToolCatalogSeed(_definition("stable"))
    repository = _repository(path)
    first = repository.bootstrap((seed,))
    audit_before = repository.list_audit_events()

    second = _repository(path).bootstrap((seed,))

    assert second.changed is False
    assert second.manifest_digest == first.manifest_digest
    assert second.catalog == first.catalog
    assert second.default_profile == first.default_profile
    assert repository.list_audit_events() == audit_before
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tool_catalog_bootstrap_manifests"
        ).fetchone()[0] == 1


def test_later_seed_identity_is_draft_and_does_not_rewrite_profile(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    stable = ToolCatalogSeed(_definition("stable"))
    later = ToolCatalogSeed(
        _definition("later"),
        DefaultProfileAvailability.REQUIRED,
    )
    repository.bootstrap((stable,))

    expanded = repository.bootstrap((stable, later))
    missing_old_seed = repository.bootstrap((later,))

    assert expanded.catalog.revision == 2
    assert {
        entry.identity.tool_id: entry.status for entry in expanded.catalog.entries
    } == {"later": CatalogStatus.DRAFT, "stable": CatalogStatus.ACTIVE}
    assert expanded.default_profile.revision == 1
    assert [
        item.identity.tool_id for item in expanded.default_profile.items
    ] == ["stable"]
    assert missing_old_seed.changed is False
    assert missing_old_seed.catalog == expanded.catalog
    assert missing_old_seed.default_profile == expanded.default_profile


def test_seed_rejects_definition_drift_under_the_same_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    original = _definition("stable")
    repository.bootstrap((ToolCatalogSeed(original),))
    drifted = _definition("stable", description="Changed without a new identity.")

    with pytest.raises(CatalogConflictError, match="existing ToolIdentity"):
        repository.bootstrap((ToolCatalogSeed(drifted),))

    assert repository.current_snapshot().revision == 1
    assert repository.load_definition(original.identity).digest == original.digest


def test_register_draft_uses_catalog_revision_cas(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    first_writer = _repository(path)
    stale_writer = _repository(path)

    published = first_writer.register_draft(
        _definition("first"),
        expected_revision=0,
        actor="operator-a",
        reason="stage first definition",
    )

    assert published.revision == 1
    assert published.entries[0].status is CatalogStatus.DRAFT
    with pytest.raises(CatalogConflictError, match="expected 0, current 1"):
        stale_writer.register_draft(
            _definition("second"),
            expected_revision=0,
            actor="operator-b",
            reason="stale update",
        )
    assert stale_writer.current_snapshot() == published
    assert [event.action for event in first_writer.list_audit_events()] == [
        "register_draft"
    ]


def test_bootstrap_rejects_catalog_mutated_before_first_manifest(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    definition = _definition("manually_staged")
    repository.register_draft(
        definition,
        expected_revision=0,
        actor="operator",
        reason="manual staging before bootstrap",
    )

    with pytest.raises(CatalogConflictError, match="requires a pristine catalog"):
        repository.bootstrap((ToolCatalogSeed(definition),))

    assert repository.current_snapshot().revision == 1
    assert repository.current_default_profile().revision == 0
    assert repository.current_default_profile().items == ()


def test_read_transaction_keeps_one_snapshot_across_concurrent_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite"
    definition = _definition("snapshot")
    reader = _repository(path)
    writer = _repository(path)
    reader.bootstrap((ToolCatalogSeed(definition),))

    with reader._read_transaction() as connection:
        before, before_profile = reader._read_current_state(connection)
        writer.set_status(
            definition.identity,
            CatalogStatus.DEPRECATED,
            expected_revision=1,
            actor="writer",
            reason="commit while reader holds a WAL snapshot",
        )
        after, after_profile = reader._read_current_state(connection)

    assert after == before
    assert after_profile == before_profile
    assert reader.current_snapshot().revision == 2
    assert reader.current_snapshot().entries[0].status is CatalogStatus.DEPRECATED


def test_consistency_read_apis_enter_explicit_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    repository.bootstrap((ToolCatalogSeed(_definition("transactional")),))
    original = repository._read_current_state
    observed: list[bool] = []

    def observe(connection: sqlite3.Connection):
        observed.append(connection.in_transaction)
        return original(connection)

    monkeypatch.setattr(repository, "_read_current_state", observe)
    repository.current_snapshot()
    repository.current_default_profile()
    repository.list_active_emergency_revocations()

    assert observed == [True, True, True]


def test_schema_validation_rejects_v0_future_and_drifted_databases(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE legacy_state(value TEXT)")
    with pytest.raises(CatalogSchemaError, match="version-zero"):
        _repository(legacy).initialize()

    future = tmp_path / "future.sqlite"
    with sqlite3.connect(future) as connection:
        connection.execute(
            f"PRAGMA user_version = {CATALOG_DATABASE_SCHEMA_VERSION + 1}"
        )
    with pytest.raises(CatalogSchemaError, match="newer application"):
        _repository(future).initialize()

    drifted = tmp_path / "drifted.sqlite"
    repository = _repository(drifted)
    repository.initialize()
    with sqlite3.connect(drifted) as connection:
        connection.execute("CREATE TABLE unexpected(value TEXT)")
    with pytest.raises(CatalogSchemaError, match="fingerprint has drifted"):
        repository.current_snapshot()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: " " + raw,
        lambda raw: raw[:-1] + ',"spec":{}}',
        lambda raw: json.dumps(
            {**json.loads(raw), "handler": "must-not-be-persisted"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        lambda raw: raw[:-1] + ',"unexpected_number":NaN}',
    ),
    ids=("noncanonical", "duplicate-key", "unknown-field", "nan"),
)
def test_definition_load_fails_closed_on_noncanonical_persisted_json(
    tmp_path: Path,
    mutate,
) -> None:
    path = tmp_path / "catalog.sqlite"
    definition = _definition("strict")
    repository = _repository(path)
    repository.bootstrap((ToolCatalogSeed(definition),))
    with sqlite3.connect(path) as connection:
        raw = str(
            connection.execute(
                "SELECT descriptor_json FROM tool_definitions"
            ).fetchone()[0]
        )
        connection.execute(
            "UPDATE tool_definitions SET descriptor_json=?",
            (mutate(raw),),
        )

    with pytest.raises(CatalogCorruptionError):
        repository.load_definition(definition.identity)


def test_definition_codec_rejects_unknown_nested_fields() -> None:
    definition = _definition("strict")
    descriptor = definition.descriptor()
    descriptor["execution"]["runtime_provider"] = "forbidden"

    with pytest.raises(ValueError, match="canonical contract"):
        ToolDefinition.from_descriptor(descriptor)

    assert ToolDefinition.from_descriptor(
        definition.descriptor()
    ).digest == definition.digest


def test_status_transition_is_cas_versioned_and_writes_retirement_tombstone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite"
    definition = _definition("lifecycle")
    repository = _repository(path)
    repository.bootstrap((ToolCatalogSeed(definition),))

    deprecated = repository.set_status(
        definition.identity,
        CatalogStatus.DEPRECATED,
        expected_revision=1,
        actor="operator",
        reason="replacement is ready",
    )
    unchanged = repository.set_status(
        definition.identity,
        CatalogStatus.DEPRECATED,
        expected_revision=2,
        actor="operator",
        reason="idempotent request",
    )

    assert deprecated.revision == 2
    assert deprecated.entries[0].status is CatalogStatus.DEPRECATED
    assert unchanged == deprecated
    assert repository.snapshot_at(1).entries[0].status is CatalogStatus.ACTIVE
    with pytest.raises(CatalogConflictError, match="expected 1, current 2"):
        repository.set_status(
            definition.identity,
            CatalogStatus.DISABLED,
            expected_revision=1,
            actor="stale",
            reason="must not overwrite",
        )

    retired = repository.set_status(
        definition.identity,
        CatalogStatus.RETIRED,
        expected_revision=2,
        actor="operator",
        reason="permanently retired",
    )
    assert retired.revision == 3
    with pytest.raises(CatalogError, match="invalid catalog lifecycle transition"):
        repository.set_status(
            definition.identity,
            CatalogStatus.ACTIVE,
            expected_revision=3,
            actor="operator",
            reason="retirement is irreversible",
        )
    with sqlite3.connect(path) as connection:
        tombstone = connection.execute(
            "SELECT retired_revision, retired_by, reason "
            "FROM tool_catalog_tombstones"
        ).fetchone()
    assert tombstone == (3, "operator", "permanently retired")
    assert [event.action for event in repository.list_audit_events()] == [
        "bootstrap_initialize",
        "set_status",
        "set_status",
    ]


def test_only_one_implementation_of_a_contract_can_be_active(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    active = _definition("replaceable")
    replacement = _definition("replaceable", implementation_version="impl-2")
    repository.bootstrap((ToolCatalogSeed(active),))
    staged = repository.register_draft(
        replacement,
        expected_revision=1,
        actor="operator",
        reason="stage replacement",
    )

    with pytest.raises(CatalogConflictError, match="already active"):
        repository.set_status(
            replacement.identity,
            CatalogStatus.ACTIVE,
            expected_revision=staged.revision,
            actor="operator",
            reason="cannot create ambiguous active implementation",
        )
    assert repository.current_snapshot() == staged


def test_default_profile_has_independent_cas_revision_and_order(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    first = _definition("first")
    second = _definition("second")
    repository = _repository(path)
    repository.bootstrap((ToolCatalogSeed(first), ToolCatalogSeed(second)))

    profile = repository.publish_default_profile(
        (
            DefaultProfileSelection(
                second.identity,
                DefaultProfileAvailability.REQUIRED,
            ),
            DefaultProfileSelection(first.identity),
        ),
        expected_catalog_revision=1,
        expected_profile_revision=1,
        actor="operator",
        reason="make second capability mandatory",
    )

    assert profile.revision == 2
    assert profile.catalog_revision == 1
    assert [item.identity for item in profile.items] == [
        second.identity,
        first.identity,
    ]
    assert repository.current_snapshot().revision == 1
    assert _repository(path).current_default_profile() == profile
    audit_count = len(repository.list_audit_events())
    assert repository.publish_default_profile(
        (
            DefaultProfileSelection(
                second.identity,
                DefaultProfileAvailability.REQUIRED,
            ),
            DefaultProfileSelection(first.identity),
        ),
        expected_catalog_revision=1,
        expected_profile_revision=2,
        actor="operator",
        reason="same exact profile",
    ) == profile
    assert len(repository.list_audit_events()) == audit_count
    with pytest.raises(CatalogConflictError, match="profile revision mismatch"):
        repository.publish_default_profile(
            (DefaultProfileSelection(first.identity),),
            expected_catalog_revision=1,
            expected_profile_revision=1,
            actor="stale",
            reason="must not overwrite profile revision two",
        )


def test_default_profile_rejects_non_active_definitions(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "catalog.sqlite")
    active = _definition("active")
    draft = _definition("draft")
    repository.bootstrap((ToolCatalogSeed(active),))
    repository.register_draft(
        draft,
        expected_revision=1,
        actor="operator",
        reason="stage draft",
    )

    with pytest.raises(CatalogError, match="only active"):
        repository.publish_default_profile(
            (DefaultProfileSelection(draft.identity),),
            expected_catalog_revision=2,
            expected_profile_revision=1,
            actor="operator",
            reason="draft must not be exposed",
        )


def test_emergency_revocation_selector_matches_one_complete_bound_target() -> None:
    effects = ToolEffectProfile(
        (
            EffectDescriptor(
                EffectResource.MEMORY,
                EffectAction.READ,
                EffectScopeKind.LOCAL,
            ),
            EffectDescriptor(
                EffectResource.FILESYSTEM,
                EffectAction.SEARCH,
                EffectScopeKind.WORKSPACE,
            ),
        )
    )
    definition = replace(_definition("matcher"), effect_template=effects)
    binding = _binding(definition, source_fingerprint="trusted-source")
    registration = BoundToolRegistration(definition, binding)
    target = EmergencyRevocationTarget.from_definition_and_binding(
        definition,
        binding,
    )

    assert target == EmergencyRevocationTarget.from_bound_registration(registration)
    assert EmergencyRevocationSelector(
        tool_id=definition.identity.tool_id,
        contract_version=definition.identity.contract_version,
        implementation_version=definition.identity.implementation_version,
        implementation_digest=definition.implementation_digest,
        source_fingerprint="trusted-source",
        effect_resource=EffectResource.FILESYSTEM,
        effect_action=EffectAction.SEARCH,
    ).matches(target)
    assert not EmergencyRevocationSelector(
        source_fingerprint="different-source",
    ).matches(target)
    assert not EmergencyRevocationSelector(
        effect_resource=EffectResource.MEMORY,
        effect_action=EffectAction.SEARCH,
    ).matches(target)


def test_emergency_revocation_matcher_rejects_incomplete_targets() -> None:
    definition = _definition("matcher")
    selector = EmergencyRevocationSelector(tool_id="matcher")

    with pytest.raises(TypeError, match="complete EmergencyRevocationTarget"):
        selector.matches(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_fingerprint"):
        EmergencyRevocationTarget(
            identity=definition.identity,
            implementation_digest=definition.implementation_digest,
            source_fingerprint=" ",
            effects=definition.effect_template.effects,
        )
    with pytest.raises(ValueError, match="non-empty tuple"):
        EmergencyRevocationTarget(
            identity=definition.identity,
            implementation_digest=definition.implementation_digest,
            source_fingerprint="source",
            effects=(),
        )


@pytest.mark.parametrize(
    "partial_clear",
    (
        {"cleared_at": _NOW},
        {"cleared_by": "operator", "clear_reason": "resolved"},
        {"cleared_at": _NOW, "cleared_by": "operator"},
    ),
)
def test_emergency_revocation_clear_fields_are_atomic(partial_clear) -> None:
    with pytest.raises(ValueError, match="present together"):
        EmergencyRevocation(
            revocation_id="incident",
            selector=EmergencyRevocationSelector(tool_id="tool"),
            reason="incident",
            issued_by="security",
            effective_at=_NOW - timedelta(minutes=1),
            expires_at=None,
            created_at=_NOW - timedelta(minutes=1),
            **partial_clear,
        )


def test_emergency_revocation_overlay_is_independent_audited_and_clearable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite"
    definition = _definition("sensitive_read")
    repository = _repository(path)
    repository.bootstrap((ToolCatalogSeed(definition),))
    selector = EmergencyRevocationSelector(
        tool_id=definition.identity.tool_id,
        source_fingerprint="compromised-source",
        effect_resource=EffectResource.MEMORY,
    )

    issued = repository.issue_emergency_revocation(
        "incident-1",
        selector,
        issued_by="security-operator",
        reason="source integrity incident",
        expires_at=_NOW + timedelta(hours=1),
    )
    repository.issue_emergency_revocation(
        "future-incident",
        EmergencyRevocationSelector(tool_id="future"),
        issued_by="security-operator",
        reason="scheduled deny",
        effective_at=_NOW + timedelta(hours=2),
    )
    repository.issue_emergency_revocation(
        "expired-incident",
        EmergencyRevocationSelector(tool_id="expired"),
        issued_by="security-operator",
        reason="historical deny",
        effective_at=_NOW - timedelta(hours=2),
        expires_at=_NOW - timedelta(hours=1),
    )

    assert repository.current_snapshot().revision == 1
    assert repository.current_default_profile().revision == 1
    assert repository.list_active_emergency_revocations(at=_NOW) == (issued,)
    assert _repository(path).list_active_emergency_revocations(at=_NOW) == (issued,)

    cleared = repository.clear_emergency_revocation(
        "incident-1",
        cleared_by="security-operator",
        reason="source replaced and verified",
        cleared_at=_NOW + timedelta(minutes=1),
    )
    audit_count = len(repository.list_audit_events())

    assert cleared.cleared_by == "security-operator"
    assert repository.list_active_emergency_revocations(at=_NOW) == (cleared,)
    assert repository.list_active_emergency_revocations(
        at=_NOW + timedelta(minutes=1)
    ) == ()
    assert repository.clear_emergency_revocation(
        "incident-1",
        cleared_by="another-operator",
        reason="idempotent clear",
        cleared_at=_NOW + timedelta(minutes=2),
    ) == cleared
    assert len(repository.list_audit_events()) == audit_count
    assert repository.current_snapshot().revision == 1
    assert [event.action for event in repository.list_audit_events()] == [
        "bootstrap_initialize",
        "issue_emergency_revocation",
        "issue_emergency_revocation",
        "issue_emergency_revocation",
        "clear_emergency_revocation",
    ]


def test_emergency_revocation_listing_fails_closed_on_selector_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite"
    repository = _repository(path)
    repository.initialize()
    repository.issue_emergency_revocation(
        "incident",
        EmergencyRevocationSelector(tool_id="tool"),
        issued_by="security",
        reason="test deny",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE tool_emergency_revocations SET selector_json=?",
            ('{"tool_id":"tool","unknown":true}',),
        )

    with pytest.raises(CatalogCorruptionError):
        repository.list_active_emergency_revocations(at=_NOW)


def test_database_path_rejects_a_preplanted_file_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    target.touch()
    link = tmp_path / "tool_catalog.sqlite"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="symbolic link"):
        ToolCatalogRepository(link)
