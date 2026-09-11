from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from personagraph.tools.catalog.revocation import (
    EmergencyRevocationDenied,
    EmergencyRevocationGuard,
    EmergencyRevocationGuardDisposition,
    EmergencyRevocationGuardReason,
    EmergencyRevocationGuardStage,
)
from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
)
from personagraph.tools.catalog.persistence import (
    EmergencyRevocationSelector,
    EmergencyRevocationTarget,
    ToolCatalogRepository,
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


_NOW = datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc)


def _effect(
    resource: EffectResource = EffectResource.MEMORY,
    action: EffectAction = EffectAction.READ,
) -> EffectDescriptor:
    return EffectDescriptor(resource, action, EffectScopeKind.LOCAL)


def _registration(
    *,
    effects: tuple[EffectDescriptor, ...] = (_effect(),),
) -> BoundToolRegistration:
    effect_profile = ToolEffectProfile(effects)
    definition = ToolDefinition(
        spec=ToolSpec(
            tool_id="guarded_tool",
            contract_version="guarded-tool-v1",
            name="Guarded Tool",
            description="Exercise the emergency revocation guard.",
            input_schema={"type": "object", "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
        implementation_version="implementation-1",
        implementation_ref="personagraph.tools.factories.guarded_tool",
        implementation_digest="a" * 64,
        effect_template=effect_profile,
        execution_profile=ToolExecutionProfile(max_transparent_retries=0),
    )
    binding = ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "test-provider",
            fingerprint="trusted-source",
        ),
        handler=lambda _arguments: {"ok": True},
        effect_profile=effect_profile,
        binding_assertion={},
    )
    return BoundToolRegistration(definition, binding)


def _repository(path: Path) -> ToolCatalogRepository:
    return ToolCatalogRepository(path, clock=lambda: _NOW)


def _target(
    registration: BoundToolRegistration | None = None,
) -> EmergencyRevocationTarget:
    return EmergencyRevocationTarget.from_bound_registration(
        registration or _registration()
    )


def _issue_matching_revocation(repository: ToolCatalogRepository) -> None:
    repository.issue_emergency_revocation(
        "incident",
        EmergencyRevocationSelector(tool_id="guarded_tool"),
        issued_by="security",
        reason="test active deny",
    )


@pytest.mark.parametrize("stage", tuple(EmergencyRevocationGuardStage))
def test_active_revocation_denies_each_current_stage(
    tmp_path: Path,
    stage: EmergencyRevocationGuardStage,
) -> None:
    repository = _repository(tmp_path / f"{stage.value}.sqlite")
    _issue_matching_revocation(repository)
    guard = EmergencyRevocationGuard(repository)

    decision = guard.check_current(stage=stage, target=_target())

    assert decision.disposition is EmergencyRevocationGuardDisposition.DENY
    assert decision.reason is EmergencyRevocationGuardReason.ACTIVE
    assert decision.code == "tool_emergency_revoked"
    assert decision.matched_revocation_ids == ("incident",)
    assert decision.to_dict() == {
        "stage": stage.value,
        "disposition": "deny",
        "code": "tool_emergency_revoked",
        "matched_revocation_ids": ["incident"],
    }
    with pytest.raises(EmergencyRevocationDenied) as caught:
        guard.enforce_current(stage=stage, target=_target())
    assert caught.value.code == decision.code
    assert caught.value.decision == decision


def test_effect_selector_dimensions_must_match_the_same_effect(
    tmp_path: Path,
) -> None:
    registration = _registration(
        effects=(
            _effect(EffectResource.MEMORY, EffectAction.READ),
            _effect(EffectResource.FILESYSTEM, EffectAction.SEARCH),
        )
    )
    repository = _repository(tmp_path / "effects.sqlite")
    repository.issue_emergency_revocation(
        "cross-effect-selector",
        EmergencyRevocationSelector(
            effect_resource=EffectResource.MEMORY,
            effect_action=EffectAction.SEARCH,
        ),
        issued_by="security",
        reason="must not combine separate effects",
    )
    guard = EmergencyRevocationGuard(repository)

    assert guard.check_current(
        stage=EmergencyRevocationGuardStage.EXECUTE,
        target=_target(registration),
    ).allowed

    repository.issue_emergency_revocation(
        "matching-effect-selector",
        EmergencyRevocationSelector(
            effect_resource=EffectResource.FILESYSTEM,
            effect_action=EffectAction.SEARCH,
        ),
        issued_by="security",
        reason="match one complete effect",
    )
    decision = guard.check_current(
        stage=EmergencyRevocationGuardStage.EXECUTE,
        target=_target(registration),
    )
    assert not decision.allowed
    assert decision.matched_revocation_ids == ("matching-effect-selector",)


def test_repository_filters_expired_and_cleared_revocations_before_guarding(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "inactive.sqlite")
    repository.issue_emergency_revocation(
        "expired",
        EmergencyRevocationSelector(tool_id="guarded_tool"),
        issued_by="security",
        reason="expired incident",
        effective_at=_NOW - timedelta(hours=2),
        expires_at=_NOW - timedelta(hours=1),
    )
    repository.issue_emergency_revocation(
        "cleared",
        EmergencyRevocationSelector(tool_id="guarded_tool"),
        issued_by="security",
        reason="resolved incident",
    )
    repository.clear_emergency_revocation(
        "cleared",
        cleared_by="security",
        reason="resolution verified",
    )

    assert repository.list_active_emergency_revocations() == ()
    decision = EmergencyRevocationGuard(repository).check_current(
        stage=EmergencyRevocationGuardStage.RESOLVE,
        target=_target(),
    )
    assert decision.allowed
    assert decision.reason is EmergencyRevocationGuardReason.NOT_MATCHED


def test_new_revocation_between_physical_attempts_blocks_the_second_attempt(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "retry.sqlite")
    guard = EmergencyRevocationGuard(repository)
    target = _target()
    physical_attempts: list[int] = []

    def run_physical_attempt() -> None:
        guard.enforce_current(
            stage=EmergencyRevocationGuardStage.EXECUTE,
            target=target,
        )
        physical_attempts.append(len(physical_attempts) + 1)

    run_physical_attempt()
    _issue_matching_revocation(repository)

    with pytest.raises(EmergencyRevocationDenied) as caught:
        run_physical_attempt()

    assert caught.value.code == "tool_emergency_revoked"
    assert physical_attempts == [1]


def test_reader_failure_is_a_stable_fail_closed_decision() -> None:
    class BrokenReader:
        calls = 0

        def list_active_emergency_revocations(self):
            self.calls += 1
            raise RuntimeError("private database path and driver details")

    reader = BrokenReader()
    guard = EmergencyRevocationGuard(reader)

    decision = guard.check_current(
        stage=EmergencyRevocationGuardStage.RESUME,
        target=_target(),
    )

    assert reader.calls == 1
    assert decision.disposition is EmergencyRevocationGuardDisposition.DENY
    assert decision.reason is EmergencyRevocationGuardReason.STATE_UNAVAILABLE
    assert decision.matched_revocation_ids == ()
    with pytest.raises(EmergencyRevocationDenied) as caught:
        guard.enforce_current(
            stage=EmergencyRevocationGuardStage.RESUME,
            target=_target(),
        )
    assert reader.calls == 2
    assert caught.value.code == "emergency_revocation_state_unavailable"
    assert "private database" not in str(caught.value)


def test_reader_requires_an_immutable_materialized_snapshot() -> None:
    class MutableReader:
        def list_active_emergency_revocations(self):
            return []

    decision = EmergencyRevocationGuard(MutableReader()).check_current(
        stage=EmergencyRevocationGuardStage.RESOLVE,
        target=_target(),
    )

    assert not decision.allowed
    assert decision.reason is EmergencyRevocationGuardReason.STATE_UNAVAILABLE


def test_guard_stage_contract_excludes_completed_result_replay() -> None:
    class EmptyReader:
        def list_active_emergency_revocations(self):
            return ()

    guard = EmergencyRevocationGuard(EmptyReader())

    assert tuple(stage.value for stage in EmergencyRevocationGuardStage) == (
        "resolve",
        "resume",
        "execute",
    )
    with pytest.raises(TypeError, match="EmergencyRevocationGuardStage"):
        guard.check_current(stage="replay", target=_target())  # type: ignore[arg-type]
