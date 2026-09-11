"""轮次拥有的 L1 P1 聚合体的持久化边界。"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from personagraph.runtime.l1.corpus_contracts import (
    L1_CORPUS_MANIFEST_CONTRACT_VERSION,
    L1CorpusManifest,
    L1CorpusWorkspace,
    derive_l1_turn_run_id,
    freeze_l1_corpus_contract,
)
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    canonical_policy_json,
    canonical_snapshot_json,
    freeze_turn_routing_policy,
    policy_sha256,
    snapshot_sha256,
)
from personagraph.session import store as session_store


def _accept_l1(session_id: str, *, request_id: str = "request-l1"):
    policy = TurnRoutingPolicy(l1_enabled=True, l2_enabled=False)
    snapshot = freeze_turn_routing_policy(policy, source="request_override")
    receipt = session_store.accept_turn_execution(
        session_id=session_id,
        client_request_id=request_id,
        source="runtime_test",
        user_text="bounded work",
        lease_owner="worker-1",
        routing_policy_source=snapshot.source,
        routing_policy_snapshot_json=canonical_snapshot_json(snapshot),
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        session_routing_policy_json=canonical_policy_json(policy),
        session_routing_policy_hash=policy_sha256(policy),
    )
    return snapshot, receipt


def test_acceptance_persists_one_immutable_turn_snapshot_and_session_default() -> None:
    session_id = session_store.create_session("Entelecheia")
    snapshot, receipt = _accept_l1(session_id)

    assert receipt["routing_policy"]["snapshot_hash"] == snapshot_sha256(snapshot)  # type: ignore[index]
    stored = session_store.get_session_turn_routing_policy(session_id)
    assert stored is not None
    assert stored["policy_hash"] == policy_sha256(snapshot.policy)

    replay = session_store.accept_turn_execution(
        session_id=session_id,
        client_request_id="request-l1",
        source="runtime_test",
        user_text="bounded work",
        lease_owner="worker-1",
        routing_policy_source=snapshot.source,
        routing_policy_snapshot_json=canonical_snapshot_json(snapshot),
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
    )
    assert replay["replayed"] is True
    assert replay["turn"]["turn_id"] == receipt["turn"]["turn_id"]  # type: ignore[index]


def test_explicit_policy_change_is_a_request_id_collision() -> None:
    session_id = session_store.create_session("Entelecheia")
    _snapshot, _receipt = _accept_l1(session_id)
    changed = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=False, l2_enabled=False),
        source="request_override",
    )

    with pytest.raises(session_store.TurnExecutionRequestIdCollision):
        session_store.accept_turn_execution(
            session_id=session_id,
            client_request_id="request-l1",
            source="runtime_test",
            user_text="bounded work",
            lease_owner="worker-1",
            routing_policy_source=changed.source,
            routing_policy_snapshot_json=canonical_snapshot_json(changed),
            routing_policy_snapshot_hash=snapshot_sha256(changed),
        )


def test_l1_shell_replays_after_a_lost_response_and_requires_the_window_lease() -> None:
    session_id = session_store.create_session("Entelecheia")
    snapshot, receipt = _accept_l1(session_id)
    turn_id = str(receipt["turn"]["turn_id"])  # type: ignore[index]
    revision = int(receipt["window"]["state_version"])  # type: ignore[index]

    created = session_store.create_l1_turn_run(
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=revision,
        expected_lease_owner="worker-1",
    )
    replayed = session_store.create_l1_turn_run(
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=revision,
        expected_lease_owner="worker-1",
    )

    assert created["replayed"] is False
    assert replayed["replayed"] is True
    assert replayed["run"]["l1_turn_run_id"] == created["run"]["l1_turn_run_id"]  # type: ignore[index]
    assert int(replayed["window"]["state_version"]) == revision + 1  # type: ignore[index]
    findings = session_store.get_execution_findings_ledger_for_owner(
        owner_kind="l1_turn_run",
        execution_owner_id=str(created["run"]["l1_turn_run_id"]),  # type: ignore[index]
    )
    assert findings is not None
    assert findings.ledger.originating_turn_id == turn_id
    assert findings.ledger.revision == 0

    with sqlite3.connect(session_store.DB_PATH) as conn:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="execution lanes are mutually exclusive",
        ):
            conn.execute(
                "UPDATE turn_execution_windows SET current_work_run_id='work-1' "
                "WHERE session_id=?",
                (session_id,),
            )

    with pytest.raises(
        session_store.L1TurnRunPersistenceError,
        match="lease owner",
    ):
        session_store.create_l1_turn_run(
            session_id=session_id,
            turn_id=turn_id,
            routing_policy_snapshot_hash=snapshot_sha256(snapshot),
            expected_window_revision=revision + 1,
            expected_lease_owner="worker-2",
        )


def test_l1_run_and_bootstrap_state_can_be_created_atomically() -> None:
    session_id = session_store.create_session("Entelecheia")
    snapshot, receipt = _accept_l1(
        session_id,
        request_id="request-l1-atomic-bootstrap",
    )
    turn_id = str(receipt["turn"]["turn_id"])  # type: ignore[index]
    revision = int(receipt["window"]["state_version"])  # type: ignore[index]
    catalog_json = "{}"
    catalog_hash = hashlib.sha256(catalog_json.encode("utf-8")).hexdigest()
    execution_config_json = "{}"
    execution_config_hash = hashlib.sha256(
        execution_config_json.encode("utf-8")
    ).hexdigest()
    l1_turn_run_id = derive_l1_turn_run_id(
        session_id=session_id,
        turn_id=turn_id,
    )
    manifest = freeze_l1_corpus_contract(
        L1CorpusManifest(
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            catalog_snapshot_sha256=catalog_hash,
            attachment_count=0,
        )
    )
    initialization = {
        "l1_turn_run_id": l1_turn_run_id,
        "deadline_at": "2099-01-01T00:00:00+00:00",
        "max_attempts": 12,
        "max_tool_calls_per_attempt": 8,
        "catalog_snapshot_json": catalog_json,
        "catalog_snapshot_hash": catalog_hash,
        "execution_config_json": execution_config_json,
        "execution_config_hash": execution_config_hash,
        "corpus_manifest_contract_version": (
            L1_CORPUS_MANIFEST_CONTRACT_VERSION
        ),
        "corpus_manifest_json": manifest.manifest_json,
        "corpus_manifest_hash": manifest.manifest_sha256,
    }

    created = session_store.create_l1_turn_run(
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=revision,
        expected_lease_owner="worker-1",
        **initialization,
    )
    replayed = session_store.create_l1_turn_run(
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=revision,
        expected_lease_owner="worker-1",
        **initialization,
    )

    assert created["replayed"] is False
    assert created["run"]["status"] == "active"  # type: ignore[index]
    assert created["state"]["stage"] == "bootstrap"  # type: ignore[index]
    assert int(created["window"]["state_version"]) == revision + 1  # type: ignore[index]
    assert replayed["replayed"] is True
    assert replayed["state"]["deadline_at"] == initialization["deadline_at"]  # type: ignore[index]
    findings = session_store.get_execution_findings_ledger_for_owner(
        owner_kind="l1_turn_run",
        execution_owner_id=l1_turn_run_id,
    )
    assert findings is not None
    assert findings.ledger.originating_turn_id == turn_id

    with pytest.raises(
        session_store.L1TurnRunPersistenceError,
        match="initialization facts",
    ):
        session_store.create_l1_turn_run(
            session_id=session_id,
            turn_id=turn_id,
            routing_policy_snapshot_hash=snapshot_sha256(snapshot),
            expected_window_revision=revision,
            expected_lease_owner="worker-1",
            **{**initialization, "max_attempts": 13},
        )


    changed_manifest = freeze_l1_corpus_contract(
        L1CorpusManifest(
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            catalog_snapshot_sha256=catalog_hash,
            workspace=L1CorpusWorkspace(
                boundary_fingerprint="c" * 64,
                scope_snapshot_sha256="d" * 64,
            ),
            attachment_count=0,
        )
    )
    with pytest.raises(
        session_store.L1TurnRunPersistenceError,
        match="initialization facts",
    ):
        session_store.create_l1_turn_run(
            session_id=session_id,
            turn_id=turn_id,
            routing_policy_snapshot_hash=snapshot_sha256(snapshot),
            expected_window_revision=revision,
            expected_lease_owner="worker-1",
            **{
                **initialization,
                "corpus_manifest_json": changed_manifest.manifest_json,
                "corpus_manifest_hash": changed_manifest.manifest_sha256,
            },
        )

    request_json = "{}"
    request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    prepared = session_store.start_l1_attempt(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=l1_turn_run_id,
        request_json=request_json,
        request_hash=request_hash,
        expected_window_revision=int(created["window"]["state_version"]),  # type: ignore[index]
        expected_lease_owner="worker-1",
    )
    attempt_id = str(
        prepared["attempt"]["attempt_id"]  # type: ignore[index]
    )
    frozen_request = json.loads(prepared["attempt"]["request_json"])
    assert frozen_request == {
        "attempt_id": attempt_id,
        "attempt_ordinal": 1,
    }
    assert prepared["attempt"]["request_hash"] == hashlib.sha256(
        prepared["attempt"]["request_json"].encode("utf-8")
    ).hexdigest()
    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.execute(
            "UPDATE l1_turn_run_states SET corpus_manifest_hash=? "
            "WHERE l1_turn_run_id=?",
            ("f" * 64, l1_turn_run_id),
        )
    with pytest.raises(
        session_store.L1TurnRunPersistenceError,
        match="state guard hash changed",
    ):
        session_store.get_l1_attempt_state_guard(
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            attempt_id=attempt_id,
        )


def test_l1_creation_rolls_back_when_findings_companion_fails() -> None:
    session_id = session_store.create_session("Entelecheia")
    snapshot, receipt = _accept_l1(
        session_id,
        request_id="request-l1-findings-rollback",
    )
    turn_id = str(receipt["turn"]["turn_id"])  # type: ignore[index]
    revision = int(receipt["window"]["state_version"])  # type: ignore[index]
    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TRIGGER reject_test_l1_findings_companion
            BEFORE INSERT ON execution_findings_ledgers
            BEGIN
                SELECT RAISE(ABORT, 'injected L1 findings companion failure');
            END;
            """
        )

    with pytest.raises(
        session_store.L1TurnRunPersistenceError,
        match="findings companion",
    ):
        session_store.create_l1_turn_run(
            session_id=session_id,
            turn_id=turn_id,
            routing_policy_snapshot_hash=snapshot_sha256(snapshot),
            expected_window_revision=revision,
            expected_lease_owner="worker-1",
        )

    with sqlite3.connect(session_store.DB_PATH) as conn:
        run_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM l1_turn_runs WHERE turn_id=?",
                (turn_id,),
            ).fetchone()[0]
        )
        ledger_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM execution_findings_ledgers "
                "WHERE originating_turn_id=?",
                (turn_id,),
            ).fetchone()[0]
        )
        window = conn.execute(
            "SELECT current_l1_turn_run_id, state_version "
            "FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        ).fetchone()
    assert run_count == 0
    assert ledger_count == 0
    assert window == (None, revision)


def test_l1_resume_claim_fences_fresh_owner_and_reclaims_stale_owner() -> None:
    session_id = session_store.create_session("Entelecheia")
    snapshot, receipt = _accept_l1(session_id)
    turn_id = str(receipt["turn"]["turn_id"])  # type: ignore[index]
    created = session_store.create_l1_turn_run(
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=int(receipt["window"]["state_version"]),  # type: ignore[index]
        expected_lease_owner="worker-1",
    )
    l1_turn_run_id = str(created["run"]["l1_turn_run_id"])  # type: ignore[index]
    semantic_revision = int(created["window"]["state_version"])  # type: ignore[index]

    busy = session_store.claim_l1_turn_run_resume(
        session_id=session_id,
        turn_id=turn_id,
        lease_owner="worker-2",
        lease_seconds=90,
    )
    assert busy["status"] == "busy"
    assert busy["reason"] == "fresh_lease_owner"

    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.execute(
            "UPDATE turn_execution_windows SET heartbeat_at=? WHERE session_id=?",
            ("2000-01-01T00:00:00+00:00", session_id),
        )
        conn.commit()

    claimed = session_store.claim_l1_turn_run_resume(
        session_id=session_id,
        turn_id=turn_id,
        lease_owner="worker-2",
        lease_seconds=90,
    )
    assert claimed["status"] == "applied"
    assert claimed["window"]["lease_owner"] == "worker-2"  # type: ignore[index]
    # 租约隔离不是语义转换，必须保留包含窗口修订版的已准备模型状态守卫。
    assert int(claimed["window"]["state_version"]) == semantic_revision  # type: ignore[index]
    assert session_store.renew_l1_turn_run_resume_lease(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=l1_turn_run_id,
        lease_owner="worker-1",
    ) is False
    assert session_store.renew_l1_turn_run_resume_lease(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=l1_turn_run_id,
        lease_owner="worker-2",
    ) is True

    unrelated = session_store.claim_l1_turn_run_resume(
        session_id=session_id,
        turn_id="turn-not-owned-by-this-window",
        lease_owner="worker-3",
        lease_seconds=90,
    )
    assert unrelated["status"] == "not_resumable"
    assert unrelated["reason"] == "turn_does_not_own_window"
