"""运行通知复用正式提交事务，但不能冒充已验证的 L1 答案。"""

import hashlib

import pytest

from personagraph.persistent_turn_content.delivery import build_l1_terminal_notification
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy, canonical_policy_json, canonical_snapshot_json,
    freeze_turn_routing_policy, policy_sha256, snapshot_sha256,
)
from personagraph.runtime.l1.corpus_contracts import (
    L1_CORPUS_MANIFEST_CONTRACT_VERSION, L1CorpusManifest,
    derive_l1_turn_run_id, freeze_l1_corpus_contract,
)
from personagraph.session import store


def _run(*, failed=True, code="VERIFICATION_FAILED"):
    session_id = store.create_session("terminal delivery")
    policy = TurnRoutingPolicy(l1_enabled=True, l2_enabled=False)
    snapshot = freeze_turn_routing_policy(policy, source="request_override")
    accepted = store.accept_turn_execution(
        session_id=session_id, client_request_id="terminal-notice",
        source="runtime_test", user_text="请回答问题", lease_owner="worker",
        routing_policy_source=snapshot.source,
        routing_policy_snapshot_json=canonical_snapshot_json(snapshot),
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        session_routing_policy_json=canonical_policy_json(policy),
        session_routing_policy_hash=policy_sha256(policy),
    )
    turn_id = accepted["turn"]["turn_id"]
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    empty_hash = hashlib.sha256(b"{}").hexdigest()
    manifest = freeze_l1_corpus_contract(L1CorpusManifest(
        session_id=session_id, turn_id=turn_id, l1_turn_run_id=run_id,
        catalog_snapshot_sha256=empty_hash, attachment_count=0,
    ))
    created = store.create_l1_turn_run(
        session_id=session_id, turn_id=turn_id, l1_turn_run_id=run_id,
        routing_policy_snapshot_hash=snapshot_sha256(snapshot),
        expected_window_revision=accepted["window"]["state_version"],
        expected_lease_owner="worker", deadline_at="2099-01-01T00:00:00+00:00",
        max_attempts=24, max_tool_calls_per_attempt=8,
        catalog_snapshot_json="{}", catalog_snapshot_hash=empty_hash,
        execution_config_json="{}", execution_config_hash=empty_hash,
        corpus_manifest_contract_version=L1_CORPUS_MANIFEST_CONTRACT_VERSION,
        corpus_manifest_json=manifest.manifest_json,
        corpus_manifest_hash=manifest.manifest_sha256,
    )
    if failed:
        store.fail_l1_turn_run(
            session_id=session_id, turn_id=turn_id, l1_turn_run_id=run_id,
            failure_code=code,
        )
    notification = build_l1_terminal_notification(code)
    assert notification is not None
    return {
        "session_id": session_id, "turn_id": turn_id,
        "expected_window_revision": created["window"]["state_version"],
        "processing_level": "L1", "assistant_content": notification.reply,
        "post_commit_job_kinds": (), "expected_lease_owner": "worker",
        "l1_terminal_failure_code": code,
    }


def test_terminal_notice_commits_once_and_preserves_failed_run():
    arguments = _run()
    first = store.finalize_turn_execution(**arguments)
    replay = store.finalize_turn_execution(**(arguments | {"expected_lease_owner": None}))
    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert first["turn"]["status"] == "completed"
    assert first["turn"]["error_code"] == "VERIFICATION_FAILED"
    assert first["turn"]["end_reason"] == "l1_terminal_notification"
    execution = store.get_l1_turn_execution(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
    )
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "VERIFICATION_FAILED"
    assert execution["state"]["attempts_started"] == 0
    assert len(store.list_committed_turn_pairs(arguments["session_id"])) == 1
    assert [message["role"] for message in store.get_turns(arguments["session_id"])] == ["user", "assistant"]


@pytest.mark.parametrize("change", [
    {"assistant_content": "未经审查的候选答案"},
    {"l1_terminal_failure_code": "MODEL_OUTPUT_INVALID"},
    {"processing_level": "L0"},
    {"l1_terminal_failure_code": "UNKNOWN"},
    {"l1_terminal_failure_code": ""},
    {"expected_lease_owner": "another-worker"},
])
def test_terminal_notice_rejects_changed_authority_without_publishing(change):
    arguments = _run()
    with pytest.raises((ValueError, RuntimeError)):
        store.finalize_turn_execution(**(arguments | change))
    assert store.list_committed_turn_pairs(arguments["session_id"]) == []


def test_active_run_cannot_bypass_verification_with_terminal_notice():
    arguments = _run(failed=False)
    with pytest.raises((ValueError, RuntimeError)):
        store.finalize_turn_execution(**arguments)
    assert store.list_committed_turn_pairs(arguments["session_id"]) == []


@pytest.mark.parametrize("code", [None, "MODEL_OUTPUT_INVALID"])
def test_terminal_replay_cannot_erase_or_rewrite_failure_identity(code):
    arguments = _run()
    store.finalize_turn_execution(**arguments)
    with pytest.raises((ValueError, RuntimeError)):
        store.finalize_turn_execution(**(arguments | {
            "expected_lease_owner": None, "l1_terminal_failure_code": code,
        }))
    assert len(store.list_committed_turn_pairs(arguments["session_id"])) == 1


def test_terminal_notice_keeps_raw_failure_identity_but_publishes_safe_error_code():
    arguments = _run(code="MODEL_BAD_RESPONSE")
    receipt = store.finalize_turn_execution(**arguments)
    assert receipt["turn"]["error_code"] == "MODEL_OUTPUT_INVALID"
    execution = store.get_l1_turn_execution(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
    )
    assert execution["state"]["failure_code"] == "MODEL_BAD_RESPONSE"


def test_failed_run_claim_allows_delivery_only_without_restarting_execution():
    arguments = _run(code="MODEL_BAD_RESPONSE")
    claimed = store.claim_l1_turn_run_resume(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
        lease_owner="worker", lease_seconds=30,
    )
    assert claimed["status"] == "applied"
    assert claimed["terminal_failure_code"] == "MODEL_BAD_RESPONSE"
    assert claimed["run"]["status"] == "failed"
    assert claimed["state"]["stage"] == "failed"
    execution = store.get_l1_turn_execution(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
    )
    assert execution["state"]["attempts_started"] == 0
    assert store.list_committed_turn_pairs(arguments["session_id"]) == []


def test_active_run_claim_has_no_terminal_delivery_authority():
    arguments = _run(failed=False)
    claimed = store.claim_l1_turn_run_resume(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
        lease_owner="worker", lease_seconds=30,
    )
    assert claimed["status"] == "applied"
    assert claimed["terminal_failure_code"] is None


def test_failed_notification_claim_still_respects_a_fresh_foreign_lease():
    arguments = _run()
    claimed = store.claim_l1_turn_run_resume(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
        lease_owner="another-worker", lease_seconds=300,
    )
    assert claimed["status"] == "busy"
    assert store.list_committed_turn_pairs(arguments["session_id"]) == []


def test_failed_run_without_known_notification_cannot_be_claimed_for_delivery():
    arguments = _run(failed=False)
    execution = store.get_l1_turn_execution(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
    )
    store.fail_l1_turn_run(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
        l1_turn_run_id=execution["run"]["l1_turn_run_id"], failure_code="UNKNOWN",
    )
    claimed = store.claim_l1_turn_run_resume(
        session_id=arguments["session_id"], turn_id=arguments["turn_id"],
        lease_owner="worker", lease_seconds=30,
    )
    assert claimed["status"] == "not_resumable"
    assert claimed["reason"] == "l1_failure_has_no_terminal_notification"
