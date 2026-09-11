from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import pytest

from personagraph.l2.entry_adapter.application import (
    L2TaskTargetUnavailableError,
    run_l2_task_lane,
)
from personagraph.runtime.turn_deadline import TurnDeadline


@dataclass(frozen=True)
class _Task:
    objective: str


class _TaskStore:
    def __init__(self, task: _Task | None) -> None:
        self.task = task
        self.calls: list[tuple[str, str]] = []

    def get_insession_task_details(
        self,
        session_id: str,
        insession_task_id: str,
    ) -> _Task | None:
        self.calls.append((session_id, insession_task_id))
        return self.task


def test_l2_entry_lane_passes_the_durable_task_objective_to_the_executor() -> None:
    store = _TaskStore(_Task(objective="生成经过验证的交付"))
    seen: dict[str, object] = {}
    expected = object()
    deadline = TurnDeadline.starting_now(60.0)

    def executor(**kwargs: object) -> object:
        seen.update(kwargs)
        return expected

    def emitted(_event: object) -> None:
        return None

    result = run_l2_task_lane(
        session_id="session-1",
        turn_id="turn-1",
        task_id="task-1",
        emit=emitted,
        deadline=deadline,
        features={"user_interaction_mode": "closed_world"},
        file_retrieval_data_version="file-generation-1",
        task_store=store,
        executor=executor,
    )

    assert result is expected
    assert store.calls == [("session-1", "task-1")]
    assert seen == {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "task_id": "task-1",
        "desired_output": "生成经过验证的交付",
        "emit": emitted,
        "deadline": deadline,
        "features": {"user_interaction_mode": "closed_world"},
        "file_retrieval_data_version": "file-generation-1",
    }


def test_l2_entry_lane_rejects_a_missing_task_before_executor_effects() -> None:
    store = _TaskStore(None)

    with pytest.raises(L2TaskTargetUnavailableError):
        run_l2_task_lane(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            emit=lambda _event: None,
            deadline=TurnDeadline.starting_now(60.0),
            task_store=store,
            executor=lambda **_kwargs: pytest.fail("executor must not run"),
        )


def test_l2_entry_lane_does_not_translate_store_or_executor_failures() -> None:
    store_error = RuntimeError("store unavailable")

    class _FailingStore:
        def get_insession_task_details(
            self,
            _session_id: str,
            _insession_task_id: str,
        ) -> None:
            raise store_error

    with pytest.raises(RuntimeError, match="store unavailable") as captured:
        run_l2_task_lane(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            emit=lambda _event: None,
            deadline=TurnDeadline.starting_now(60.0),
            task_store=_FailingStore(),
            executor=lambda **_kwargs: object(),
        )
    assert captured.value is store_error

    executor_error = RuntimeError("executor unavailable")

    def fail_executor(**_kwargs: object) -> object:
        raise executor_error

    with pytest.raises(RuntimeError, match="executor unavailable") as captured:
        run_l2_task_lane(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            emit=lambda _event: None,
            deadline=TurnDeadline.starting_now(60.0),
            task_store=_TaskStore(_Task(objective="目标")),
            executor=fail_executor,
        )
    assert captured.value is executor_error


def test_l2_entry_lane_resolves_production_dependencies_only_when_called(
    monkeypatch,
) -> None:
    from personagraph.l2.entry_adapter import executor as executor_module
    from personagraph.session.l2_store import task_graph

    expected = object()
    monkeypatch.setattr(
        task_graph,
        "get_insession_task_details",
        lambda _session_id, _task_id: _Task(objective="持久目标"),
    )
    monkeypatch.setattr(
        executor_module,
        "run_auxiliary_task_executor",
        lambda **_kwargs: expected,
    )

    assert run_l2_task_lane(
        session_id="session-1",
        turn_id="turn-1",
        task_id="task-1",
        emit=lambda _event: None,
        deadline=TurnDeadline.starting_now(60.0),
    ) is expected


def test_l2_entry_lane_cold_import_does_not_load_entry_store_or_executor() -> None:
    code = """
import json
import sys
import personagraph.l2.entry_adapter.application
blocked = {
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.application',
    'personagraph.session',
    'personagraph.session.store',
    'personagraph.l2.auxiliary_execution',
    'personagraph.l2.entry_adapter.executor',
}
print(json.dumps(sorted(blocked & set(sys.modules))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
