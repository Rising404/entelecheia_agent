from __future__ import annotations

import pytest

from personagraph.session import session_summary
from personagraph.session.session_summary import (
    SESSION_SUMMARY_SYSTEM_PROMPT,
    SessionSummaryGenerationError,
    SessionSummaryState,
    SessionSummaryStatus,
    SessionSummaryTurnPair,
    build_session_summary_update,
    mock_session_summary,
    render_session_summary_input,
)


def _pair(*, turn_id: str | None, ordinal: int) -> SessionSummaryTurnPair:
    return SessionSummaryTurnPair(
        turn_id=turn_id,
        user_turn_idx=ordinal * 2,
        assistant_turn_idx=ordinal * 2 + 1,
        user_content=f"user content {ordinal}",
        assistant_content=f"assistant content {ordinal}",
        created_at="2026-08-11T00:00:00+00:00",
    )


def _summary_state(*, status: SessionSummaryStatus) -> SessionSummaryState:
    return SessionSummaryState(
        session_id="session-summary-generation",
        running_summary="已有摘要",
        summarized_through_turn_id="turn-existing",
        status=status,
        state_version=1,
        updated_at="2026-08-11T00:00:00+00:00",
        last_error_code=(None if status is SessionSummaryStatus.OK else "SUMMARY_STALE"),
    )


def test_summary_input_budget_advances_only_through_complete_contiguous_pairs(
    monkeypatch,
) -> None:
    first = _pair(turn_id="turn-1", ordinal=0)
    second = _pair(turn_id="turn-2", ordinal=1)

    # 按字符计数，使边界精确且不依赖开发者机器上可选安装的 tokenizer。
    monkeypatch.setattr(session_summary, "estimate_tokens", len)
    one_pair_input = render_session_summary_input(None, (first,))
    monkeypatch.setattr(
        session_summary,
        "task_budget",
        lambda: len(SESSION_SUMMARY_SYSTEM_PROMPT) + len(one_pair_input),
    )

    selected = session_summary._select_summary_pairs_within_input_budget(
        None,
        (first, second),
    )

    assert selected == (first,)


def test_summary_update_binds_generated_text_to_the_last_bounded_pair() -> None:
    first = _pair(turn_id="turn-1", ordinal=0)
    observed: list[tuple[SessionSummaryTurnPair, ...]] = []

    update = build_session_summary_update(
        state=None,
        pairs=(first,),
        generate=lambda _state, pairs: observed.append(pairs) or "  摘要正文  ",
    )

    assert observed == [(first,)]
    assert update.running_summary == "摘要正文"
    assert update.summarized_through_turn_id == "turn-1"


def test_summary_update_rejects_unverifiable_source_or_output() -> None:
    with pytest.raises(SessionSummaryGenerationError) as degraded:
        build_session_summary_update(
            state=_summary_state(status=SessionSummaryStatus.STALE),
            pairs=(),
            generate=lambda *_args: "must not run",
        )
    assert degraded.value.code == "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE"
    assert degraded.value.retryable is False

    with pytest.raises(SessionSummaryGenerationError) as unbound:
        build_session_summary_update(
            state=None,
            pairs=(_pair(turn_id=None, ordinal=0),),
            generate=lambda *_args: "must not run",
        )
    assert unbound.value.code == "SUMMARY_BOUNDARY_UNAVAILABLE"
    assert unbound.value.retryable is False

    with pytest.raises(SessionSummaryGenerationError) as empty:
        build_session_summary_update(
            state=None,
            pairs=(_pair(turn_id="turn-1", ordinal=0),),
            generate=lambda *_args: "   ",
        )
    assert empty.value.code == "SUMMARY_EMPTY_OUTPUT"
    assert empty.value.retryable is True


def test_summary_rendering_and_mock_generation_preserve_the_existing_wire_text() -> None:
    state = _summary_state(status=SessionSummaryStatus.OK)
    pair = _pair(turn_id="turn-1", ordinal=0)

    assert render_session_summary_input(state, (pair,)) == (
        "已有摘要（已验证）：\n已有摘要\n\n"
        "较早完整回合：\n用户：user content 0\n助手：assistant content 0"
    )
    assert mock_session_summary(state, (pair,)) == (
        "已有摘要\n用户此前提出：user content 0"
    )
