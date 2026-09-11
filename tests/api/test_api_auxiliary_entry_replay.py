from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.api.service import sessions as session_service
from personagraph.l2.auxiliary_execution import (
    production_chain as auxiliary_production_chain,
)
from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainStatus,
)
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.turn_events import RuntimeErrorCode
from personagraph.session import store
from tests.runtime.test_auxiliary_entry_integration import (
    _classify_one,
    _seed_task_shells,
)


def test_api_client_request_replay_never_reenters_waiting_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, (task_id,) = _seed_task_shells()
    monkeypatch.setattr(
        entry_application,
        "classify_turn",
        _classify_one(task_id),
    )
    monkeypatch.setattr(
        session_service,
        "load_features",
        lambda _config: {
            "context_guard_limit": 24_000,
        },
    )
    chain_calls = 0

    def waiting_chain(*_args, **_kwargs):
        nonlocal chain_calls
        chain_calls += 1
        return SimpleNamespace(
            status=AuxiliaryProductionChainStatus.WAITING_EXTERNAL,
            final_delivery_id=None,
            publication_body=None,
        )

    monkeypatch.setattr(
        auxiliary_production_chain,
        "run_auxiliary_to_verified_delivery",
        waiting_chain,
    )
    payload = {
        "session_id": session_id,
        "message": "继续执行已有任务",
        "client_request_id": "api-aux-v2-waiting",
        "runtime_policy": {"l1_enabled": False, "l2_enabled": True},
    }

    first = session_service.chat_turn(payload)
    replayed = session_service.chat_turn(payload)

    assert first["result"]["status"] == "incomplete"
    assert first["result"]["reply"] is None
    assert first["result"]["error_code"] == (
        RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value
    )
    assert replayed["result"]["status"] == "running"
    assert replayed["result"]["reply"] is None
    assert replayed["result"]["turn_id"] == first["result"]["turn_id"]
    assert chain_calls == 1
    with store._connect() as conn:
        assistant_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_turns "
                "WHERE session_id=? AND role='assistant'",
                (session_id,),
            ).fetchone()[0]
        )
    assert assistant_count == 1  # 只发布了种子轮次。
