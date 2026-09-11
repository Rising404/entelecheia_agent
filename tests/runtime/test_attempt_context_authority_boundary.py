"""由 Store 支撑的尝试上下文权威的边界覆盖。"""

from __future__ import annotations

from types import SimpleNamespace

from personagraph.l2.task_execution.attempts import context_authority as authority


def test_context_authority_projects_immutable_turn_input(monkeypatch) -> None:
    """公开投影保持独立于模型或桥接器设置。"""

    monkeypatch.setattr(
        authority.session_store,
        "get_turn_execution_input",
        lambda *, session_id, turn_id: {
            "session_id": session_id,
            "turn_id": turn_id,
            "content": "authoritative request",
        },
    )
    stored = SimpleNamespace(session_id="session-1", attempts=())
    current_attempt = SimpleNamespace(
        input_turn_id="turn-input-1",
        predecessor_question_attempt_id=None,
    )

    projection = authority.project_authoritative_attempt_user_input(
        stored=stored,
        current_attempt=current_attempt,
    )

    assert projection.content == "authoritative request"
    assert projection.prior_waiting_user_question is None
