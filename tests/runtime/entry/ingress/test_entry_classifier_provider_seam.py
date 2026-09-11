from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from personagraph.model_io.gateway import ModelResult
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.context.contracts import EntryContext
from personagraph.runtime.entry.ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    TrustedTurnEnvelope,
)
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)


def _context() -> EntryContext:
    return EntryContext(
        envelope=TrustedTurnEnvelope(
            turn_id="turn-provider-seam",
            session_id="session-provider-seam",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="请分析这个复杂任务",
        ),
        snapshot=AuthoritativeRuntimeSnapshot(),
        ceiling=CapabilityCeiling(),
        estimated_input_tokens=1,
        history_pairs=(),
        session_summary=None,
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=False, l2_enabled=True),
            source="request_override",
        ),
    )


def _classification_result(model_call_id: str) -> ModelResult:
    return ModelResult(
        reply=json.dumps(
            {"processing_level": "L2", "task_matches": []},
            ensure_ascii=False,
        ),
        provider="test",
        model="test",
        latency_ms=1,
        model_call_id=model_call_id,
        purpose="runtime_entry_classify",
    )


def test_classifier_uses_prepared_capability_when_provider_exposes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[object] = []

    class _Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            operations.append(("dispatch", model_call_id))
            return _classification_result(model_call_id)

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared provider must not use its legacy call path")

    def prepare(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        **_kwargs: object,
    ) -> _Prepared:
        operations.append(
            (
                "prepare",
                purpose,
                "allowed_processing_levels" in system_prompt,
                json.loads(user_content)["current_user_text"],
            )
        )
        return _Prepared()

    provider.prepare = prepare  # type: ignore[attr-defined]
    monkeypatch.setattr(ingress_model, "complete_structured", provider)

    result = ingress_model.classify_turn(_context(), lambda _event: None)

    assert result.processing_level == "L2"
    assert operations[0] == (
        "prepare",
        "runtime_entry_classify",
        True,
        "请分析这个复杂任务",
    )
    assert operations[1][0] == "dispatch"


def test_classifier_rejects_a_provider_without_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        **_kwargs: object,
    ) -> ModelResult:
        return _classification_result(model_call_id)

    monkeypatch.setattr(ingress_model, "complete_structured", provider)

    with pytest.raises(TypeError, match="must expose prepare"):
        ingress_model.classify_turn(_context(), lambda _event: None)


def test_classifier_prepared_repair_uses_latest_output_in_four_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_messages: list[object] = []
    rejected = json.dumps(
        {
            "processing_level": "L2",
            "task_matches": [
                {
                    "match_type": "new_root",
                    "local_key": "analysis",
                    "title": "复杂任务",
                    "objective": "分析任务",
                    "source_excerpt": "不存在的原文",
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    class _Prepared:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply=self.reply,
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
                purpose="runtime_entry_classify",
            )

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared provider must not use its legacy path")

    def prepare(
        _system_prompt: str,
        _user_content: str,
        *,
        repair_messages: object | None = None,
        **_kwargs: object,
    ) -> _Prepared:
        prepared_messages.append(repair_messages)
        return _Prepared(
            rejected
            if repair_messages is None
            else json.dumps(
                {"processing_level": "L2", "task_matches": []},
                ensure_ascii=False,
            )
        )

    provider.prepare = prepare  # type: ignore[attr-defined]
    monkeypatch.setattr(ingress_model, "complete_structured", provider)

    result = ingress_model.classify_turn(_context(), lambda _event: None)

    assert result.processing_level == "L2"
    assert prepared_messages[0] is None
    messages = prepared_messages[1]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == rejected
    feedback = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert feedback == {"current_issues": [{
        "paths": ["/task_matches"],
        "safe_explanation": "task_matches 未通过来源锚点、唯一性、数量或任务目录校验。",
    }]}
    assert "返回符合原请求的完整 JSON" in messages[3]["content"]
    assert rejected not in messages[3]["content"]
