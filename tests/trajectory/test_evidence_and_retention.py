"""首批四个步骤落地后确定的两项性质。

第一项是一致性：检索步骤按身份记录证据，因此携带相同证据的模型调用不得再
完整复制一份。第二项是上限：记录默认开启，因此必须具有容量边界。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personagraph.trajectory import (
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
    TrajectoryStore,
    text_blob,
)
from personagraph.trajectory import recorder as recorder_module


@pytest.fixture
def store(tmp_path, monkeypatch) -> TrajectoryStore:
    from personagraph.trajectory import store as store_module

    isolated = TrajectoryStore(tmp_path / "trajectory.sqlite")
    monkeypatch.setattr(store_module, "_ACTIVE", isolated)
    return isolated


# --- 证据只存引用 ----------------------------------------------------------------


def _real_prompt_with_evidence(body: str) -> str:
    """按照检索链路的实际方式构建提示。

    特意由拥有该格式的模块生成，而不是手写。下方剥离逻辑与该格式匹配，
    因此格式变化必须让此处失败，而不能静默地让轨迹再次膨胀。
    """

    from personagraph.retrieval.contracts import (
        ContextStatus,
        RetrievedContext,
        RetrievedItem,
        SourceAvailability,
        SourceDependency,
        SourceOutcome,
        SourceRetrievalStatus,
        SourceType,
        SourceUnitRef,
    )
    from personagraph.retrieval.prompt_context import RetrievedContextPromptSerializer

    context = RetrievedContext(
        status=ContextStatus.COMPLETE,
        items=(
            RetrievedItem(
                SourceUnitRef(SourceType.DOCUMENT, "unit-1", "v2", "a" * 64),
                body,
                {"doc_id": "doc-1", "chunk_id": "chunk-1"},
                max(1, len(body) // 4),
                1,
            ),
        ),
        source_outcomes={
            source_type: SourceOutcome(
                source_type=source_type,
                availability=SourceAvailability.DISABLED,
                retrieval=SourceRetrievalStatus.NOT_RUN,
                dependency=SourceDependency.OPTIONAL,
                reason_code="source_excluded_by_trusted_boundary",
            )
            for source_type in SourceType
        },
        method_outcomes=(),
        configured_token_limit=4096,
        packed_tokens=100,
    )
    return RetrievedContextPromptSerializer().serialize(context).text


def test_evidence_bodies_are_replaced_by_a_note_of_their_size():
    body = "US-EAST processed 938 requests per second. " * 400
    prompt = _real_prompt_with_evidence(body)

    stripped = recorder_module._strip_evidence_bodies(prompt)

    assert body not in stripped
    assert "未记录" in stripped
    assert str(len(body.encode("utf-8"))) in stripped
    assert len(stripped.encode("utf-8")) < len(prompt.encode("utf-8")) / 10


def test_the_identity_of_the_evidence_survives():
    """值得保留的是展示了哪些证据，以及它们的顺序。"""

    prompt = _real_prompt_with_evidence("正文" * 500)
    stripped = recorder_module._strip_evidence_bodies(prompt)

    assert "doc_id=doc-1" in stripped
    assert "chunk_id=chunk-1" in stripped
    assert 'position="1"' in stripped


def test_an_ordinary_message_is_left_alone():
    plain = "看看这份文档里的 Figure 2 说了什么"
    assert recorder_module._strip_evidence_bodies(plain) == plain


def test_stripping_applies_to_what_actually_gets_recorded(store):
    body = "机密合同条款：" + "违约金 100 万。" * 300
    recorder_module.record_model_call(
        model_call_id="mc-ev",
        purpose="probe",
        provider="p",
        model="m",
        payload={"messages": [{"role": "user", "content": _real_prompt_with_evidence(body)}]},
        response={"content": [{"type": "text", "text": "好的"}]},
        reply="好的",
        duration_ms=1,
        store=store,
    )
    recorded = "".join(
        part.blob.text or "" for part in store.steps_for_model_call("mc-ev")[0].parts
    )
    assert body not in recorded
    assert "doc_id=doc-1" in recorded


# --- 保留：A + D + E -------------------------------------------------------------


def _aged(step_id: str, days: int, *, outcome=StepOutcome.OK, session_id=None, shared=None):
    return Step(
        step_id=step_id,
        kind=StepKind.TOOL_CALL,
        occurred_at=(datetime.now(UTC) - timedelta(days=days)).isoformat(),
        parts=(shared,) if shared else (Part(PartRole.TOOL_ARGUMENTS, text_blob(step_id)),),
        session_id=session_id,
        outcome=outcome,
        reason_code=None if outcome is StepOutcome.OK else "boom",
    )


def test_old_successes_go_and_recent_ones_stay(store):
    store.record(_aged("old_ok", 40))
    store.record(_aged("new_ok", 1))

    report = store.prune(older_than_days=30)

    assert report["removed_steps"] == 1
    assert store.get("old_ok") is None
    assert store.get("new_ok") is not None


def test_failures_outlive_the_window(store):
    """人们事后回来查看的，正是未能正常工作的步骤。"""

    store.record(_aged("old_fail", 90, outcome=StepOutcome.FAILED))
    store.record(_aged("old_rejected", 90, outcome=StepOutcome.REJECTED))

    assert store.prune(older_than_days=30)["removed_steps"] == 0
    assert store.get("old_fail") is not None
    assert store.get("old_rejected") is not None


def test_failures_can_be_swept_too_when_explicitly_asked(store):
    store.record(_aged("old_fail", 90, outcome=StepOutcome.FAILED))
    assert store.prune(older_than_days=30, keep_failures=False)["removed_steps"] == 1


def test_a_blob_another_step_still_uses_is_not_reclaimed(store):
    shared = Part(PartRole.SYSTEM, text_blob("共用的系统提示"))
    store.record(_aged("old_ok", 40, shared=shared))
    store.record(_aged("new_ok", 1, shared=shared))

    report = store.prune(older_than_days=30)

    assert report["removed_steps"] == 1
    assert report["removed_blobs"] == 0
    assert store.get("new_ok").parts[0].blob.text == "共用的系统提示"


def test_timestamps_are_compared_by_parsing_not_by_spelling(store):
    """“Z”和“+00:00”按字符串排序不同，但表示同一时刻。"""

    same_instant = datetime.now(UTC) - timedelta(days=40)
    store.record(
        Step(
            step_id="z_form",
            kind=StepKind.TOOL_CALL,
            occurred_at=same_instant.strftime("%Y-%m-%dT%H:%M:%SZ"),
            parts=(Part(PartRole.TOOL_ARGUMENTS, text_blob("z")),),
        )
    )
    store.record(_aged("offset_form", 40))

    assert store.prune(older_than_days=30)["removed_steps"] == 2


def test_an_unreadable_timestamp_is_kept_rather_than_guessed(store):
    store.record(
        Step(
            step_id="broken",
            kind=StepKind.TOOL_CALL,
            occurred_at="不是时间",
            parts=(Part(PartRole.TOOL_ARGUMENTS, text_blob("x")),),
        )
    )
    assert store.prune(older_than_days=0)["removed_steps"] == 0
    assert store.get("broken") is not None


@pytest.mark.parametrize("bad", [-1, True, "30"])
def test_a_nonsense_window_is_refused(store, bad):
    with pytest.raises((TypeError, ValueError)):
        store.prune(older_than_days=bad)
