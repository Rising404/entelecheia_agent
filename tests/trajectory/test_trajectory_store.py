"""保存每个步骤的内容，同时避免重复保存。

该存储之所以存在，是因为运行时只记录步骤发生过，却不记录具体内容。这些测试
覆盖决定其成本与可信度的两个性质：相同内容无论出现多少次都只保存一份；
任何无法完整保存的内容都会明确说明。
"""

from __future__ import annotations

import pytest

from personagraph.trajectory import (
    MAX_BLOB_BYTES,
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
    TrajectoryStore,
    text_blob,
)


@pytest.fixture
def store(tmp_path) -> TrajectoryStore:
    return TrajectoryStore(tmp_path / "trajectory.sqlite")


def _model_step(step_id: str, *parts: Part, **overrides) -> Step:
    payload = {
        "step_id": step_id,
        "kind": StepKind.MODEL_CALL,
        "occurred_at": f"2026-08-22T00:00:{int(step_id[-1]):02d}Z",
        "parts": parts,
    }
    payload.update(overrides)
    return Step(**payload)


# --- 去重：这是它划不划算的全部 -------------------------------------------------


def test_a_prompt_resent_every_call_is_held_once(store):
    """系统提示会随每次调用发出；它只占一行，而不是多行。"""

    system = Part(PartRole.SYSTEM, text_blob("你是 PersonaGraph 的入口分类器。" * 40))
    for index in range(8):
        store.record(
            _model_step(
                f"step-{index}",
                system,
                Part(PartRole.USER, text_blob(f"第 {index} 个问题")),
                turn_id="turn-1",
            )
        )

    usage = store.usage()
    assert usage["parts"] == 16
    assert usage["blobs"] == 9  # 1 个系统提示 + 8 个不同的问题
    assert usage["stored_bytes"] < usage["referenced_bytes"] / 3


def test_the_two_byte_counts_are_the_same_unit(store):
    """只有两个数值都以字节计量时，节省量才有意义。"""

    chinese = Part(PartRole.USER, text_blob("吞吐量"))  # 3 个字符，9 字节
    store.record(_model_step("step-0", chinese))
    usage = store.usage()
    assert usage["stored_bytes"] == 9
    assert usage["referenced_bytes"] == 9


# --- 诚实：装不下的时候要说 ------------------------------------------------------


def test_content_too_large_is_cut_but_never_silently(store):
    oversized = "范" * MAX_BLOB_BYTES  # 每个字符 3 字节，明显超过上限
    blob = text_blob(oversized)

    assert blob.truncated is True
    assert blob.byte_count == len(oversized.encode("utf-8"))
    assert len(blob.text.encode("utf-8")) <= MAX_BLOB_BYTES
    # 截断点落在字符边界上，不是半个字
    assert blob.text == oversized[: len(blob.text)]

    store.record(_model_step("step-0", Part(PartRole.EVIDENCE, blob)))
    restored = store.get("step-0").parts[0].blob
    assert restored.truncated is True
    assert restored.byte_count == blob.byte_count


def test_a_hash_still_identifies_the_whole_content_not_the_kept_part(store):
    """共享前缀的两个不同长文档不得发生冲突。"""

    first = text_blob("同" * MAX_BLOB_BYTES + "甲")
    second = text_blob("同" * MAX_BLOB_BYTES + "乙")
    assert first.sha256 != second.sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sha256": "c" * 64, "byte_count": 1, "text": None},
        {"sha256": "c" * 64, "byte_count": 1, "text": 1},
        {"sha256": "nothex", "byte_count": 1, "text": "x"},
        {"sha256": "c" * 64, "byte_count": -1, "text": "x"},
    ],
    ids=["missing-text", "non-string-text", "bad-hash", "negative-size"],
)
def test_a_blob_must_hold_text_with_a_valid_content_identity(kwargs):
    from personagraph.trajectory import Blob

    with pytest.raises(ValueError):
        Blob(**kwargs)


# --- 写入语义 --------------------------------------------------------------------


def test_recording_the_same_step_twice_changes_nothing(store):
    store.record(_model_step("step-0", Part(PartRole.USER, text_blob("原始"))))
    assert store.record(_model_step("step-0", Part(PartRole.USER, text_blob("覆盖")))) is False

    parts = store.get("step-0").parts
    assert len(parts) == 1
    assert parts[0].blob.text == "原始"


def test_the_order_of_a_step_s_parts_survives(store):
    """消息列表若以错误顺序回读，就会变成另一份提示。"""

    roles = (PartRole.SYSTEM, PartRole.USER, PartRole.THINKING, PartRole.ASSISTANT)
    store.record(
        _model_step("step-0", *(Part(role, text_blob(role.value)) for role in roles))
    )
    assert tuple(part.role for part in store.get("step-0").parts) == roles


# --- 契约 -----------------------------------------------------------------------


def test_a_step_that_failed_must_say_why():
    with pytest.raises(ValueError):
        Step(
            step_id="step-0",
            kind=StepKind.TOOL_CALL,
            occurred_at="2026-08-22T00:00:00Z",
            outcome=StepOutcome.FAILED,
        )


# --- 读取面 ---------------------------------------------------------------------


def test_a_turn_reads_back_as_its_steps_in_order(store):
    for index in (2, 0, 1):
        store.record(
            _model_step(
                f"step-{index}",
                Part(PartRole.USER, text_blob(f"q{index}")),
                turn_id="turn-1",
                session_id="s1",
            )
        )
    store.record(_model_step("step-9", turn_id="turn-2", session_id="s1"))

    assert [s.step_id for s in store.steps_for_turn("turn-1")] == [
        "step-0", "step-1", "step-2",
    ]
    assert len(store.steps_for_session("s1")) == 4


def test_a_step_is_findable_by_the_id_the_event_stream_already_records(store):
    """model_call_id 是关联 Runtime Turn 事件台账的连接键。"""

    store.record(_model_step("step-0", model_call_id="mc-42"))
    assert [s.step_id for s in store.steps_for_model_call("mc-42")] == ["step-0"]


def test_metrics_round_trip(store):
    store.record(_model_step("step-0", metrics={"output_tokens": 203, "thinking_bytes": 521}))
    assert store.get("step-0").metrics == {"output_tokens": 203, "thinking_bytes": 521}


def test_an_absent_step_is_none_not_an_error(store):
    assert store.get("no-such-step") is None


# --- 清理 -----------------------------------------------------------------------


def test_forgetting_a_session_keeps_what_other_sessions_still_use(store):
    shared = Part(PartRole.SYSTEM, text_blob("共用的系统提示"))
    store.record(_model_step("step-0", shared, session_id="s1"))
    store.record(_model_step("step-1", shared, session_id="s2"))

    assert store.delete_session("s1") == 1
    assert store.get("step-0") is None
    assert store.collect_unreferenced_blobs() == 0  # s2 还在用
    assert store.get("step-1").parts[0].blob.text == "共用的系统提示"


def test_a_blob_nobody_points_at_is_collectable(store):
    store.record(_model_step("step-0", Part(PartRole.USER, text_blob("只此一处")), session_id="s1"))
    store.delete_session("s1")

    assert store.usage()["blobs"] == 1
    assert store.collect_unreferenced_blobs() == 1
    assert store.usage()["blobs"] == 0
