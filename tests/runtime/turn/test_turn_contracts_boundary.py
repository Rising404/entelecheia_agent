"""共享冷 Turn 契约的边界覆盖。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from personagraph.runtime.turn import contracts


def _execution_snapshot() -> contracts.EntryExecutionSnapshot:
    return contracts.EntryExecutionSnapshot.create(
        features={},
        post_commit_job_kinds=(),
    )


def test_turn_contracts_preserve_public_validation_and_recovery_binding() -> None:
    recovery = contracts.EntryRecoveryProjection(
        turn_id="turn-previous",
        end_reason="host_stopped",
        error_code="TURN_DEADLINE_EXCEEDED",
        input_message_id="message-previous",
    )
    accepted = contracts.AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="continue",
        attachment_ids=("attachment-1",),
        window_revision=3,
        replayed=True,
        execution_snapshot=_execution_snapshot(),
        recovery_projection=recovery,
    )
    result = contracts.EntryTurnResult(
        session_id="session-1",
        turn_id="turn-1",
        status="completed",
        processing_level="L2",
        window_state="post_commit_pending",
        window_revision=4,
        reply="durable reply",
    )

    assert accepted.recovery_projection is recovery
    assert accepted.turn_status == "running"
    assert result.reply == "durable reply"
    with pytest.raises(FrozenInstanceError):
        result.reply = "replacement"  # type: ignore[misc]
    with pytest.raises(ValueError, match="formal reply"):
        contracts.EntryTurnResult(
            session_id="session-1",
            turn_id="turn-1",
            status="completed",
            processing_level="L0",
            window_state="empty",
            window_revision=1,
        )
    with pytest.raises(ValueError, match="only completed"):
        contracts.EntryTurnResult(
            session_id="session-1",
            turn_id="turn-1",
            status="running",
            processing_level="L0",
            window_state="active",
            window_revision=1,
            reply="not yet public",
        )
    with pytest.raises(ValueError, match="window_revision"):
        contracts.AcceptedEntryTurn(
            session_id="session-1",
            turn_id="turn-1",
            client_request_id="request-1",
            user_input="continue",
            attachment_ids=(),
            window_revision=-1,
            replayed=False,
            execution_snapshot=_execution_snapshot(),
        )
    with pytest.raises(ValueError, match="stable Session"):
        contracts.AcceptedEntryTurn(
            session_id="",
            turn_id="turn-1",
            client_request_id="request-1",
            user_input="continue",
            attachment_ids=(),
            window_revision=1,
            replayed=False,
            execution_snapshot=_execution_snapshot(),
        )


def test_execution_snapshot_is_canonical_and_hash_authenticated() -> None:
    snapshot = contracts.EntryExecutionSnapshot.create(
        features={
            "file_retrieval_read_enabled": True,
            "turn_wall_clock_budget_s": 1248.0,
        },
        file_retrieval_data_version="file-generation-01",
        session_retrieval_data_version="session-generation-01",
        session_retrieval_assistant_turn_cutoff=5,
        post_commit_job_kinds=("session_summary",),
    )

    recovered = contracts.EntryExecutionSnapshot.from_json(
        snapshot.to_json(),
        expected_sha256=snapshot.sha256,
    )

    assert recovered == snapshot
    assert recovered.features["file_retrieval_read_enabled"] is True
    assert recovered.session_retrieval_data_version == "session-generation-01"
    assert recovered.session_retrieval_assistant_turn_cutoff == 5
    assert "history_retrieval_data_version" not in snapshot.to_json()
    retired_payload = json.dumps(
        {
            **json.loads(snapshot.to_json()),
            "history_retrieval_data_version": "retired-generation",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="invalid"):
        contracts.EntryExecutionSnapshot.from_json(
            retired_payload,
            expected_sha256=hashlib.sha256(retired_payload.encode()).hexdigest(),
        )
    with pytest.raises(ValueError, match="hash is invalid"):
        contracts.EntryExecutionSnapshot.from_json(
            snapshot.to_json().replace("1248.0", "1249.0"),
            expected_sha256=snapshot.sha256,
        )


def test_turn_contracts_import_does_not_load_entry_lanes_model_or_session() -> None:
    code = """
import json
import sys
import personagraph.runtime.turn.contracts
blocked_prefixes = (
    'personagraph.l2',
    'personagraph.model_io',
    'personagraph.runtime.entry',
    'personagraph.runtime.l1',
    'personagraph.session',
)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name.startswith(blocked_prefixes)
)))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_retired_entry_turn_contract_path_is_absent() -> None:
    assert importlib.util.find_spec(
        "personagraph.runtime.entry.turn_contracts"
    ) is None


def test_entry_package_is_a_standard_lazy_read_facade() -> None:
    code = """
import importlib
import json
import sys
from types import ModuleType

import personagraph.runtime.entry as entry

application_name = 'personagraph.runtime.entry.application'
application_loaded_before_read = application_name in sys.modules
resolved = entry.run_entry_turn
application = importlib.import_module(application_name)
print(json.dumps({
    'application_loaded_before_read': application_loaded_before_read,
    'is_standard_module': type(entry) is ModuleType,
    'resolved_from_application': resolved is application.run_entry_turn,
    'facade_did_not_cache_application_attribute': 'run_entry_turn' not in entry.__dict__,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "application_loaded_before_read": False,
        "is_standard_module": True,
        "resolved_from_application": True,
        "facade_did_not_cache_application_attribute": True,
    }
