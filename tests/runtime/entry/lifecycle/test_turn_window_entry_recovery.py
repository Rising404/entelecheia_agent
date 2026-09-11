"""持久轮次执行窗口的入口级恢复不变量。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from personagraph.runtime import entry
from personagraph.runtime.entry.lifecycle import window_audit as audit_module
from personagraph.runtime.entry.lifecycle.window_audit import (
    TurnWindowBlockedError,
    audit_turn_window_before_accept,
)
from personagraph.runtime.turn.timing import (
    ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S,
)
from personagraph.session import store


def test_active_window_heartbeat_ttl_is_an_explicit_recovery_policy() -> None:
    assert ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S == 90.0


def test_turn_window_audit_is_owned_only_by_runtime_entry() -> None:
    runtime_root = Path(audit_module.__file__).resolve().parents[1]

    assert audit_module.__name__ == (
        "personagraph.runtime.entry.lifecycle.window_audit"
    )
    assert not (runtime_root / "turn_window_audit.py").exists()


def test_next_turn_settles_a_stale_active_window_before_it_is_accepted():
    """主机重启标记不得被误认为当前请求的租约。"""

    session_id = store.create_session("Entelecheia")
    abandoned = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="abandoned-request",
        source="runtime_test",
        user_text="上一轮在主机停止时中断",
        lease_owner="old-host",
    )
    abandoned_turn_id = str(abandoned["turn"]["turn_id"])  # type: ignore[index]
    with store._connect() as conn:
        conn.execute(
            "UPDATE turn_execution_windows SET heartbeat_at=? WHERE session_id=?",
            (
                (datetime.now(timezone.utc) - timedelta(
                    seconds=ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S + 1
                )).isoformat(),
                session_id,
            ),
        )

    result = entry.run_entry_turn(
        user_input="请继续新的问题",
        features={"context_guard_limit": 24000},
        session_id=session_id,
        client_request_id="next-request",
        store=store,
    )

    assert result.status == "completed"
    with store._connect() as conn:
        old = conn.execute(
            "SELECT status, end_reason FROM runtime_turns WHERE turn_id=?",
            (abandoned_turn_id,),
        ).fetchone()
    assert dict(old) == {"status": "incomplete", "end_reason": "process_lost"}


def test_next_turn_refuses_a_fresh_active_window_owned_by_another_runtime():
    """进程本地守卫不能使另一进程的新鲜租约失效。"""

    session_id = store.create_session("Entelecheia")
    active = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="runtime-a-request",
        source="runtime_test",
        user_text="Runtime A 仍在调用模型",
        lease_owner="runtime-a",
    )
    active_turn_id = str(active["turn"]["turn_id"])  # type: ignore[index]

    with pytest.raises(TurnWindowBlockedError) as caught:
        entry.run_entry_turn(
            user_input="Runtime B 的另一条消息",
            features={"context_guard_limit": 24000},
            session_id=session_id,
            client_request_id="runtime-b-request",
            store=store,
        )

    assert caught.value.details["reason"] == "active_remote_execution"
    assert caught.value.details["turn_id"] == active_turn_id
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["window_state"] == "active"
    assert window["turn_id"] == active_turn_id
    assert window["lease_owner"] == "runtime-a"
    assert [turn["content"] for turn in store.get_turns(session_id)] == [
        "Runtime A 仍在调用模型"
    ]


@pytest.mark.parametrize(
    ("marker", "expected_end_reason"),
    (
        ("TOOL_COMPLETION_UNCONFIRMED", "host_stopped"),
        ("TURN_DEADLINE_EXCEEDED", "host_stopped"),
        ("INTERNAL_FAILURE", "module_error"),
    ),
)
def test_next_input_audit_preserves_typed_no_public_stop_outcomes(
    marker: str,
    expected_end_reason: str,
) -> None:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"typed-stop-{marker}",
        source="runtime_test",
        user_text="上一轮没有可公开交付的安全停止",
        lease_owner="typed-stop-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])  # type: ignore[index]
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(accepted["window"]["state_version"]),  # type: ignore[index]
        stage="TOOL" if marker == "TOOL_COMPLETION_UNCONFIRMED" else "RESPONSE",
        interruption_reason=marker,
    )

    audit = audit_turn_window_before_accept(
        session_id=session_id,
        store=store,
        has_local_execution=False,
    )

    assert audit.recovery_projection is not None
    assert audit.recovery_projection.turn_id == turn_id
    assert audit.recovery_projection.end_reason == expected_end_reason
    assert audit.recovery_projection.error_code == marker
    with store._connect() as conn:
        turn = conn.execute(
            "SELECT status, end_reason, error_code FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
    assert dict(turn) == {
        "status": "incomplete",
        "end_reason": expected_end_reason,
        "error_code": marker,
    }
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert int(window["state_version"]) == int(marked["state_version"]) + 1
    assert window["window_state"] == "empty"


class _AdvanceWindowBeforeRecoveryMark:
    """模拟审计初次读取后发生另一项运行时阶段转换。"""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._advanced = False

    def inspect_turn_execution(self, session_id: str):
        return store.inspect_turn_execution(session_id)

    def mark_turn_execution_interrupted(self, **kwargs):
        if not self._advanced:
            self._advanced = True
            window = store.get_turn_execution_window(self._session_id)
            assert window is not None
            store.advance_turn_execution_window(
                session_id=self._session_id,
                turn_id=str(window["turn_id"]),
                expected_window_revision=int(window["state_version"]),
                stage="CLASSIFY",
                lease_owner=str(window["lease_owner"]),
            )
        return store.mark_turn_execution_interrupted(**kwargs)

    def settle_interrupted_turn_execution(self, **kwargs):
        return store.settle_interrupted_turn_execution(**kwargs)

    def release_turn_execution_window(self, **kwargs):
        return store.release_turn_execution_window(**kwargs)


def test_recovery_rechecks_a_window_changed_by_a_live_runtime_before_returning():
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="old-request",
        source="runtime_test",
        user_text="旧任务",
        lease_owner="runtime-a",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE turn_execution_windows SET heartbeat_at=? WHERE session_id=?",
            ("2000-01-01T00:00:00+00:00", session_id),
        )

    with pytest.raises(TurnWindowBlockedError) as caught:
        audit_turn_window_before_accept(
            session_id=session_id,
            store=_AdvanceWindowBeforeRecoveryMark(session_id),
            has_local_execution=False,
        )

    assert caught.value.details["reason"] == "active_remote_execution"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["turn_id"] == accepted["turn"]["turn_id"]  # type: ignore[index]
    assert window["window_state"] == "active"
