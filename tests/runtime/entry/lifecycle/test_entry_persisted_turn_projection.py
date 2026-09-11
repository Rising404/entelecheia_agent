"""Entry-owned acceptance and L1-recovery receipt projection tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.lifecycle import persisted_turn as projection
from personagraph.runtime.entry.lifecycle.persisted_turn import (
    accepted_entry_turn_from_l1_recovery_inspection,
    accepted_entry_turn_from_persisted_receipt,
)
from personagraph.runtime.entry.routing.policy import (
    canonical_snapshot_json,
    snapshot_sha256,
)
from personagraph.runtime.turn.contracts import (
    EntryExecutionSnapshot,
    EntryRecoveryProjection,
    TurnRoutingPolicy,
    TurnRoutingPolicySnapshot,
)
from personagraph.session.turn_execution_contracts import TurnExecutionBusyError


def _execution_snapshot() -> EntryExecutionSnapshot:
    return EntryExecutionSnapshot.create(features={}, post_commit_job_kinds=())


def _routing_snapshot() -> TurnRoutingPolicySnapshot:
    policy = TurnRoutingPolicy(l1_enabled=True, l2_enabled=False)
    return TurnRoutingPolicySnapshot(
        source="default",
        policy=policy,
        allowed_processing_levels=policy.allowed_processing_levels,
    )


def _receipt(**overrides: object) -> dict[str, object]:
    execution_snapshot = _execution_snapshot()
    routing_snapshot = _routing_snapshot()
    receipt: dict[str, object] = {
        "turn": {
            "turn_id": "turn-1",
            "source": "cli",
            "status": "running",
            "processing_level": "L2",
            "end_reason": 7,
            "error_code": False,
            "execution_snapshot_json": execution_snapshot.to_json(),
            "execution_snapshot_sha256": execution_snapshot.sha256,
        },
        "input_message": {
            "message_id": "message-1",
            "content": "continue the persisted Turn",
        },
        "window": {"state_version": "9", "window_state": "active"},
        "attachments": [
            {"attachment_id": "attachment-1"},
            {"attachment_id": 8},
            {"attachment_id": ""},
            {"attachment_id": 0},
            {"not_attachment_id": "ignored"},
            "ignored",
        ],
        "replayed": False,
        "routing_policy": {
            "snapshot_json": canonical_snapshot_json(routing_snapshot),
            "snapshot_hash": snapshot_sha256(routing_snapshot),
        },
    }
    receipt.update(overrides)
    return receipt


def _recovery() -> EntryRecoveryProjection:
    return EntryRecoveryProjection(
        turn_id="previous-turn",
        end_reason="host_stopped",
        error_code="INTERNAL_FAILURE",
        input_message_id="message-previous",
    )


def test_persisted_receipt_projects_acceptance_without_runtime_source() -> None:
    recovery = _recovery()
    recovery_supplier_calls = 0

    def recovery_supplier() -> EntryRecoveryProjection:
        nonlocal recovery_supplier_calls
        recovery_supplier_calls += 1
        return recovery

    accepted = accepted_entry_turn_from_persisted_receipt(
        persisted=_receipt(),
        session_id="session-1",
        client_request_id="request-1",
        recovery_projection_supplier=recovery_supplier,
    )

    assert accepted.session_id == "session-1"
    assert accepted.turn_id == "turn-1"
    assert not hasattr(accepted, "source")
    assert accepted.user_input == "continue the persisted Turn"
    assert accepted.input_message_id == "message-1"
    assert accepted.attachment_ids == ("attachment-1", "8")
    assert accepted.window_revision == 9
    assert accepted.replayed is False
    assert accepted.turn_status == "running"
    assert accepted.processing_level == "L2"
    assert accepted.end_reason == "7"
    assert accepted.error_code == "False"
    assert accepted.window_state == "active"
    assert accepted.execution_snapshot == _execution_snapshot()
    assert accepted.routing_policy == _routing_snapshot()
    assert accepted.recovery_projection is recovery
    assert recovery_supplier_calls == 1


@pytest.mark.parametrize(
    ("receipt", "error"),
    (
        (_receipt(turn=None), "missing persisted turn"),
        (_receipt(input_message=[]), "missing persisted input_message"),
        (_receipt(window=None), "accepted Turn is missing its execution window"),
        (_receipt(window=[]), "accepted Turn is missing its execution window"),
    ),
)
def test_persisted_receipt_rejects_missing_required_mappings_and_window(
    receipt: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=f"^{error}$"):
        accepted_entry_turn_from_persisted_receipt(
            persisted=receipt,
            session_id="session-1",
            client_request_id="request-1",
        )


@pytest.mark.parametrize(
    ("routing_policy", "error"),
    (
        (None, "accepted Turn is missing its routing-policy snapshot"),
        ({}, "accepted Turn routing-policy record is incomplete"),
    ),
)
def test_persisted_receipt_rejects_missing_current_routing_snapshot(
    routing_policy: object,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=f"^{error}$"):
        accepted_entry_turn_from_persisted_receipt(
            persisted=_receipt(routing_policy=routing_policy),
            session_id="session-1",
            client_request_id="request-1",
        )


@pytest.mark.parametrize(
    ("missing_fields", "error"),
    (
        (
            ("execution_snapshot_json", "execution_snapshot_sha256"),
            "accepted Turn is missing its execution snapshot",
        ),
        (
            ("execution_snapshot_sha256",),
            "accepted Turn execution snapshot is incomplete",
        ),
    ),
)
def test_persisted_receipt_rejects_missing_current_execution_snapshot(
    missing_fields: tuple[str, ...],
    error: str,
) -> None:
    receipt = _receipt()
    turn = dict(receipt["turn"])  # type: ignore[arg-type]
    for field in missing_fields:
        turn.pop(field)
    receipt["turn"] = turn

    with pytest.raises(RuntimeError, match=f"^{error}$"):
        accepted_entry_turn_from_persisted_receipt(
            persisted=receipt,
            session_id="session-1",
            client_request_id="request-1",
        )


def test_replayed_receipt_uses_bool_coercion_and_drops_prior_recovery() -> None:
    def unexpected_recovery_supplier() -> EntryRecoveryProjection:
        raise AssertionError("replayed receipts must not read audit recovery")

    accepted = accepted_entry_turn_from_persisted_receipt(
        persisted=_receipt(replayed="false"),
        session_id="session-1",
        client_request_id="request-1",
        recovery_projection_supplier=unexpected_recovery_supplier,
    )

    assert accepted.replayed is True
    assert accepted.recovery_projection is None


def test_entry_keeps_audit_recovery_lazy_for_replayed_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_sources: list[object] = []

    class ExplosiveAudit:
        accesses = 0

        @property
        def recovery_projection(self) -> EntryRecoveryProjection:
            self.accesses += 1
            raise AssertionError("replayed receipts must not read audit recovery")

    class ReplayAfterAuditStore:
        calls = 0

        def accept_turn_execution(self, **kwargs: object) -> dict[str, object]:
            self.calls += 1
            accepted_sources.append(kwargs.get("source"))
            if self.calls == 1:
                raise TurnExecutionBusyError(
                    {
                        "session_id": "session-1",
                        "turn_id": "turn-previous",
                        "window_state": "active",
                        "state_version": 4,
                    }
                )
            return _receipt(replayed=True)

    audit = ExplosiveAudit()
    store = ReplayAfterAuditStore()
    monkeypatch.setattr(
        entry_application,
        "audit_turn_window_before_accept",
        lambda **_kwargs: audit,
    )

    accepted = entry_application.accept_entry_turn(
        user_input="retry the accepted request",
        session_id="session-1",
        client_request_id="request-1",
        attachment_ids=(),
        store=store,  # type: ignore[arg-type]
        has_existing_local_execution=False,
    )

    assert accepted.replayed is True
    assert accepted.recovery_projection is None
    assert audit.accesses == 0
    assert store.calls == 2
    assert accepted_sources == ["api", "api"]


def test_l1_recovery_rebuilds_the_original_replayed_accepted_turn() -> None:
    inspected = _receipt()
    turn = dict(inspected["turn"])  # type: ignore[arg-type]
    turn.update({"session_id": "session-1", "client_request_id": "request-1"})
    inspected["turn"] = turn

    accepted = accepted_entry_turn_from_l1_recovery_inspection(inspected)

    assert accepted.session_id == "session-1"
    assert accepted.client_request_id == "request-1"
    assert accepted.turn_id == "turn-1"
    assert accepted.replayed is True


@pytest.mark.parametrize("field", ("session_id", "client_request_id"))
def test_l1_recovery_requires_the_original_request_identity(field: str) -> None:
    inspected = _receipt()
    turn = dict(inspected["turn"])  # type: ignore[arg-type]
    turn.update({"session_id": "session-1", "client_request_id": "request-1"})
    turn[field] = ""
    inspected["turn"] = turn

    with pytest.raises(
        RuntimeError,
        match="^L1 recovery inspection lost request identity$",
    ):
        accepted_entry_turn_from_l1_recovery_inspection(inspected)


def test_entry_projection_cold_imports_no_lane_or_storage_runtime() -> None:
    code = """
import json
import sys
import personagraph.runtime.entry.lifecycle.persisted_turn
allowed = {
    'personagraph',
    'personagraph.entry_execution_snapshot',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.lifecycle',
    'personagraph.runtime.entry.lifecycle.persisted_turn',
    'personagraph.runtime.entry.routing',
    'personagraph.runtime.entry.routing.policy',
    'personagraph.runtime.turn',
    'personagraph.runtime.turn.contracts',
    'personagraph.runtime.turn.persisted_projection',
}
blocked_prefixes = (
    'personagraph.l2',
    'personagraph.model_io',
    'personagraph.runtime.l1',
    'personagraph.session',
)
loaded = {
    name
    for name in sys.modules
    if name == 'personagraph' or name.startswith('personagraph.')
}
print(json.dumps({
    'blocked': sorted(
        name for name in sys.modules if name.startswith(blocked_prefixes)
    ),
    'unexpected': sorted(loaded - allowed),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"blocked": [], "unexpected": []}


def test_entry_projection_keeps_only_entry_owned_receipt_shapes() -> None:
    assert projection.__all__ == [
        "accepted_entry_turn_from_l1_recovery_inspection",
        "accepted_entry_turn_from_persisted_receipt",
    ]
    assert not hasattr(
        projection,
        "accepted_entry_turn_from_visual_disclosure_claim",
    )
    assert not {
        "execution_snapshot_from_persisted_turn",
        "optional_text",
        "require_entry_turn_status",
        "require_entry_window_state",
        "require_persisted_mapping",
        "require_processing_level",
    } & set(vars(projection))

    tree = ast.parse(Path(projection.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imports == {
        "__future__",
        "typing",
        "turn",
        "turn.contracts",
        "routing.policy",
    }
