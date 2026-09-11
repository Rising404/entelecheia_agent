"""保留网关原本即将丢弃的内容。

网关会丢弃所有非文本块，而模型推理正位于其中。这些测试覆盖保留哪些内容、
如何标记，以及决定该功能能否安全常开的关键性质：记录绝不能把正常的模型
调用变成失败。
"""

from __future__ import annotations

import pytest

from personagraph.model_io import gateway as models
from personagraph.trajectory import (
    PartRole,
    StepKind,
    StepOutcome,
    TrajectoryStore,
    turn_linkage_scope,
)
from personagraph.trajectory import recorder as recorder_module


class _FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class _FakeClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return self.response


def _provider(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")


def _responds(monkeypatch, content, usage=None):
    payload = {
        "content": content,
        "usage": usage if usage is not None else {"input_tokens": 11, "output_tokens": 22},
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(
        models.httpx, "Client", lambda *a, **k: _FakeClient(_FakeResponse(payload))
    )


@pytest.fixture
def store(tmp_path, monkeypatch) -> TrajectoryStore:
    from personagraph.trajectory import store as store_module

    isolated = TrajectoryStore(tmp_path / "trajectory.sqlite")
    monkeypatch.setattr(store_module, "_ACTIVE", isolated)
    return isolated


THINKING = {"type": "thinking", "thinking": "先判断这是不是脑筋急转弯，再算数量。"}
ANSWER = {"type": "text", "text": "3只。"}


def test_the_thinking_the_gateway_discards_is_kept(store, monkeypatch):
    """_extract_anthropic_text 只保留文本块；这里保留其余内容。"""

    _provider(monkeypatch)
    _responds(monkeypatch, [THINKING, ANSWER])

    result = models.anthropic_compatible_chat([{"role": "user", "content": "几只猫？"}])
    assert result.reply == "3只。"  # 网关行为不变

    step = store.steps_for_model_call(result.model_call_id)[0]
    kept = {part.role: part.blob.text for part in step.parts}
    assert kept[PartRole.THINKING] == THINKING["thinking"]
    assert kept[PartRole.ASSISTANT] == "3只。"
    assert kept[PartRole.USER] == "几只猫？"
    assert step.metrics["thinking_bytes"] == len(THINKING["thinking"].encode("utf-8"))


def test_the_system_prompt_is_kept_as_its_own_part(store, monkeypatch):
    _provider(monkeypatch)
    _responds(monkeypatch, [ANSWER])

    result = models.anthropic_compatible_chat(
        [{"role": "system", "content": "你是分类器"}, {"role": "user", "content": "hi"}]
    )
    roles = [part.role for part in store.steps_for_model_call(result.model_call_id)[0].parts]
    assert roles == [PartRole.SYSTEM, PartRole.USER, PartRole.ASSISTANT]


# --- 思考内容是否存在 -------------------------------------------------------------


def test_no_thinking_block_records_no_thinking_part(store, monkeypatch):
    _provider(monkeypatch)
    _responds(monkeypatch, [ANSWER])

    result = models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])
    step = store.steps_for_model_call(result.model_call_id)[0]
    assert all(part.role is not PartRole.THINKING for part in step.parts)


def test_a_returned_thinking_block_is_kept_regardless_of_request_display(store):
    """轨迹保存提供方返回的内容，不根据请求猜测它属于哪类推理。"""

    payload = {"thinking": {"type": "adaptive", "display": "summarized"}}
    recorder_module.record_model_call(
        model_call_id="mc-1",
        purpose="probe",
        provider="anthropic",
        model="claude-opus-5",
        payload=payload,
        response={"content": [THINKING]},
        reply="",
        duration_ms=1,
        store=store,
    )
    step = store.steps_for_model_call("mc-1")[0]
    assert [(part.role, part.blob.text) for part in step.parts] == [
        (PartRole.THINKING, THINKING["thinking"]),
    ]


def test_an_inline_image_is_not_misrepresented_as_external_content(store):
    recorder_module.record_model_call(
        model_call_id="mc-image",
        purpose="vision",
        provider="anthropic-compatible",
        model="test-model",
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "A" * 5000,
                            },
                        },
                        {"type": "text", "text": "检查这张图片。"},
                    ],
                }
            ]
        },
        response={"content": [ANSWER]},
        reply="3只。",
        duration_ms=1,
        store=store,
    )

    step = store.steps_for_model_call("mc-image")[0]
    assert [(part.role, part.blob.text) for part in step.parts] == [
        (PartRole.USER, "检查这张图片。"),
        (PartRole.ASSISTANT, "3只。"),
    ]


# --- 记录绝不能拖垮调用 -----------------------------------------------------------


def test_a_broken_recorder_still_returns_the_model_s_answer(monkeypatch, store):
    """业务答案不受影响，但所属 Turn 留下一条最小失败标记。"""

    _provider(monkeypatch)
    _responds(monkeypatch, [THINKING, ANSWER])

    def _explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(recorder_module, "_record_model_call", _explode)

    with turn_linkage_scope(
        session_id="session-recording-failure",
        turn_id="turn-recording-failure",
    ):
        result = models.anthropic_compatible_chat(
            [{"role": "user", "content": "hi"}]
        )

    assert result.reply == "3只。"
    steps = store.steps_for_turn("turn-recording-failure")
    assert len(steps) == 1
    assert steps[0].kind is StepKind.RECORDING_FAILURE
    assert steps[0].outcome is StepOutcome.FAILED
    assert steps[0].purpose == "record_model_call"
    assert steps[0].reason_code == "trajectory_recording_failed:RuntimeError"


def test_a_broken_store_is_not_called_recursively(caplog):
    class BrokenStore:
        def __init__(self) -> None:
            self.calls = 0

        def record(self, _step) -> None:
            self.calls += 1
            raise RuntimeError("store unavailable")

    broken = BrokenStore()
    with turn_linkage_scope(session_id="session-1", turn_id="turn-1"):
        recorder_module.record_tool_call(
            tool_id="probe",
            arguments={},
            status="succeeded",
            result={"ok": True},
            error_code=None,
            duration_ms=1,
            store=broken,
        )

    assert broken.calls == 1
    assert "trajectory recording failed operation=record_tool_call" in caplog.text


def test_terminal_model_request_failure_is_a_body_free_failed_step(store):
    recorder_module.record_model_request_failure(
        model_call_id="model-terminal-1",
        purpose="l1_attempt",
        reason_code="MODEL_CALL_FAILED",
        attempts=3,
        duration_ms=1200,
        session_id="session-1",
        turn_id="turn-1",
        store=store,
    )

    step = store.steps_for_model_call("model-terminal-1")[0]
    assert step.kind is StepKind.MODEL_CALL
    assert step.outcome is StepOutcome.FAILED
    assert step.reason_code == "MODEL_CALL_FAILED"
    assert step.metrics == {"attempts": 3}
    assert step.duration_ms == 1200
    assert step.parts == ()


def test_recording_can_be_switched_off(monkeypatch, store):
    _provider(monkeypatch)
    _responds(monkeypatch, [THINKING, ANSWER])
    monkeypatch.setenv(recorder_module.RECORDING_ENV, "off")

    result = models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])
    assert result.reply == "3只。"
    assert store.usage()["steps"] == 0


@pytest.mark.parametrize("value", ["", "on", "1", "anything-else"])
def test_recording_is_on_unless_explicitly_disabled(monkeypatch, value):
    """需要靠人记得开启的记录器，往往恰在需要时处于关闭状态。"""

    monkeypatch.setenv(recorder_module.RECORDING_ENV, value)
    assert recorder_module.recording_enabled() is True


def test_a_mock_provider_records_nothing_by_design(monkeypatch, store):
    """预制回复不携带信息；记录它们只会产生噪声。"""

    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    models.complete_structured("sys", "user", mock_payload={"ok": True})
    assert store.usage()["steps"] == 0


# --- 去重跨调用生效 ---------------------------------------------------------------


def test_one_system_prompt_across_many_calls_is_stored_once(store, monkeypatch):
    _provider(monkeypatch)
    _responds(monkeypatch, [ANSWER])
    system = "你是 PersonaGraph 的入口分类器。" * 40

    for index in range(6):
        models.anthropic_compatible_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"第 {index} 问"},
            ]
        )

    usage = store.usage()
    assert usage["steps"] == 6
    assert usage["stored_bytes"] < usage["referenced_bytes"] / 3


# --- 流式那条路 -------------------------------------------------------------------
#
# 对话主路径是流式的。它的思考走的是另一个 delta 字段，文本提取器读不到——
# 不单独接住，最常用的那条路就是全空的。


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)


class _FakeStreamClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        return self.response


def _streams(monkeypatch, events):
    import json as _json

    lines = [f"data: {_json.dumps(event)}" for event in events]
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda *a, **k: _FakeStreamClient(_FakeStreamResponse(lines)),
    )


def test_streamed_reasoning_is_recorded_without_reaching_the_reply(store, monkeypatch):
    _provider(monkeypatch)
    _streams(
        monkeypatch,
        [
            {"type": "content_block_delta", "delta": {"thinking": "先想一下，"}},
            {"type": "content_block_delta", "delta": {"thinking": "这是脑筋急转弯。"}},
            {"type": "content_block_delta", "delta": {"text": "3只。"}},
            {"type": "message_delta", "usage": {"output_tokens": 40}},
        ],
    )

    result = models.anthropic_compatible_chat(
        [{"role": "user", "content": "几只猫？"}],
        stream=True,
        timeout_s=5,
        control_transport="prompt_json",
        model_call_id="mc-stream",
        purpose="probe",
    )

    # 思考绝不能混进回复
    assert result.reply == "3只。"

    step = store.steps_for_model_call("mc-stream")[0]
    kept = {part.role: part.blob.text for part in step.parts}
    assert kept[PartRole.THINKING] == "先想一下，这是脑筋急转弯。"
    assert kept[PartRole.ASSISTANT] == "3只。"


def test_a_stream_with_no_reasoning_records_no_thinking_part(store, monkeypatch):
    _provider(monkeypatch)
    _streams(
        monkeypatch,
        [
            {"type": "content_block_delta", "delta": {"text": "好的"}},
            {"type": "message_delta", "usage": {"output_tokens": 3}},
        ],
    )

    models.anthropic_compatible_chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
        timeout_s=5,
        control_transport="prompt_json",
        model_call_id="mc-quiet",
        purpose="probe",
    )
    step = store.steps_for_model_call("mc-quiet")[0]
    assert all(part.role is not PartRole.THINKING for part in step.parts)


# --- 重试：一次逻辑调用可以有多次物理尝试 ---------------------------------------------
#
# model_call_id 在重试循环外只铸一次，所有尝试共用它。按它归档会让第 2、3 次
# 尝试撞上第 1 次并被当成重放丢掉——而重试正是打开这个工具最想看的东西。


def test_every_attempt_of_a_retried_call_is_kept(store, monkeypatch):
    _provider(monkeypatch)

    replies = [
        [{"type": "text", "text": "第一次答错"}],
        [{"type": "text", "text": "第二次答对"}],
    ]

    def _next_client(*args, **kwargs):
        return _FakeClient(
            _FakeResponse(
                {
                    "content": replies.pop(0),
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            )
        )

    monkeypatch.setattr(models.httpx, "Client", _next_client)

    for _ in range(2):
        models.anthropic_compatible_chat(
            [{"role": "user", "content": "同一个问题"}], model_call_id="mc-retried"
        )

    steps = store.steps_for_model_call("mc-retried")
    assert len(steps) == 2
    answers = sorted(
        part.blob.text
        for step in steps
        for part in step.parts
        if part.role is PartRole.ASSISTANT
    )
    assert answers == ["第一次答错", "第二次答对"]


def test_the_same_attempt_written_twice_is_still_one_row(store, monkeypatch):
    """区分各次尝试不能破坏真实重放的幂等性。"""

    _provider(monkeypatch)
    _responds(monkeypatch, [ANSWER])

    for _ in range(2):
        recorder_module.record_model_call(
            model_call_id="mc-replay",
            purpose="probe",
            provider="p",
            model="m",
            payload={"messages": [{"role": "user", "content": "一样的输入"}]},
            response={"content": [ANSWER]},
            reply="3只。",
            duration_ms=1,
            store=store,
        )

    assert len(store.steps_for_model_call("mc-replay")) == 1


def test_both_protocol_families_record_the_same_shape(store, monkeypatch):
    """轨迹读取器不应被迫了解由哪个端点作答。"""

    from personagraph.model_io import gateway as models_module

    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-test")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "gpt-test")

    class _OpenAIClient:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def post(self, *a, **k):
            class _R:
                @staticmethod
                def raise_for_status(): return None
                @staticmethod
                def json():
                    return {
                        "choices": [{"message": {"content": "答案", "reasoning_content": "想了想"}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    }
            return _R()

    monkeypatch.setattr(models_module.httpx, "Client", lambda *a, **k: _OpenAIClient())
    result = models_module.openai_compatible_chat(
        [{"role": "system", "content": "系统"}, {"role": "user", "content": "问题"}],
        model_call_id="mc-openai",
    )

    step = store.steps_for_model_call(result.model_call_id)[0]
    kept = {part.role: part.blob.text for part in step.parts}
    assert kept[PartRole.SYSTEM] == "系统"
    assert kept[PartRole.THINKING] == "想了想"
    assert kept[PartRole.ASSISTANT] == "答案"
    assert step.metrics["input_tokens"] == 7
