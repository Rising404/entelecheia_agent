from __future__ import annotations

import threading

import pytest

from personagraph.runtime import concurrency
from personagraph.runtime.concurrency import session_run_guard
from personagraph.runtime.concurrency import SessionRunBusyError


def test_same_session_rejects_parallel_run_but_other_session_is_independent():
    entered = threading.Event()
    release = threading.Event()

    def hold_session():
        with session_run_guard("s1"):
            entered.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=hold_session)
    thread.start()
    assert entered.wait(timeout=1)
    try:
        with pytest.raises(SessionRunBusyError) as caught:
            with session_run_guard("s1"):
                pass
        assert caught.value.details == {"session_id": "s1"}
        with session_run_guard("s2"):
            pass
    finally:
        release.set()
        thread.join(timeout=2)


def test_session_lock_identity_survives_release_for_safe_handoff():
    session_id = "stable-lock-session"
    with session_run_guard(session_id):
        first_lock = concurrency._LOCKS[session_id]

    with session_run_guard(session_id):
        assert concurrency._LOCKS[session_id] is first_lock
