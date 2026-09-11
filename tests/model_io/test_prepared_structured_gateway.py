from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from personagraph.model_io import endpoint_profiles as model_profiles
from personagraph.model_io import gateway as models
from personagraph.model_io import tier_bindings as model_tiers
from personagraph.model_io.endpoint_profiles import ModelProfileQuota
from personagraph.context_budget import ContextBudgetExceeded
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)


def _openai_binding() -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-5-test",
        api_key="test-secret",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        profile_id="prepared-structured-test",
        profile_name="Prepared structured test",
        request_dialect="openai-native",
    )


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }


def test_prepared_structured_request_is_admitted_before_exact_byte_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_opens: list[float] = []
    posts: list[dict[str, object]] = []

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            client_opens.append(timeout)

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> _Response:
            posts.append({"url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(models.httpx, "Client", _Client)
    monkeypatch.setattr(models, "record_model_call", lambda **_kwargs: None)

    prepared = models.prepare_complete_structured(
        "private system instruction",
        '{"private":"user payload"}',
        mock_payload={"ok": True},
        max_tokens=64,
        json_mode=True,
        purpose="prepared-structured-contract",
        binding=_openai_binding(),
        projection_epoch="turn-1:projection",
        projection_generation=4,
    )

    # 准备阶段可以解析配置并计量字节，但不得打开提供方客户端或消耗提供方 I/O 权限。
    assert client_opens == []
    assert posts == []
    assert prepared.context_budget is not None
    admitted_body = (
        prepared.context_budget.admitted_request.body_for_dispatch()
    )
    assert prepared.budget_metadata is not None
    assert prepared.budget_metadata.request_sha256 == sha256(
        admitted_body
    ).hexdigest()
    assert [
        message["role"]
        for message in json.loads(admitted_body)["messages"]
    ] == ["system", "user"]

    result = prepared.dispatch(model_call_id="physical-call-1")

    assert result.model_call_id == "physical-call-1"
    assert len(client_opens) == 1
    assert len(posts) == 1
    assert posts[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert "json" not in posts[0]
    assert posts[0]["content"] == admitted_body


def test_prepared_structured_repair_uses_exact_frozen_four_message_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[dict[str, object]] = []
    recorded_calls: list[dict[str, object]] = []
    rejected_response = '{"action":{"type":"bad"}}'

    class _RepairResponse(_Response):
        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok":true}',
                            "reasoning_content": (
                                "I compared the rejected response "
                                f"{rejected_response} before correcting it."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            assert timeout > 0

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> _Response:
            posts.append({"url": url, **kwargs})
            return _RepairResponse()

    monkeypatch.setattr(models.httpx, "Client", _Client)
    monkeypatch.setattr(
        models,
        "record_model_call",
        lambda **kwargs: recorded_calls.append(kwargs),
    )

    repair_messages = models.build_structured_repair_messages(
        "original system",
        '{"original":"user"}',
        rejected_response_text=rejected_response,
        repair_user_content="请重新生成完整 JSON。",
    )
    prepared = models.prepare_complete_structured(
        "original system",
        '{"original":"user"}',
        mock_payload={"ok": True},
        max_tokens=64,
        purpose="prepared-structured-repair",
        binding=_openai_binding(),
        repair_messages=repair_messages,
    )

    assert prepared.context_budget is not None
    admitted_body = prepared.context_budget.admitted_request.body_for_dispatch()
    assert json.loads(admitted_body)["messages"] == [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": '{"original":"user"}'},
        {
            "role": "assistant",
            "content": rejected_response,
        },
        {"role": "user", "content": "请重新生成完整 JSON。"},
    ]

    # 即使调用方之后修改原列表，已准入的请求体仍保持不可变。
    repair_messages[2]["content"] = "mutated"
    result = prepared.dispatch(model_call_id="physical-repair-2")

    assert result.reply == '{"ok":true}'
    assert len(posts) == 1
    assert posts[0]["content"] == admitted_body
    assert len(recorded_calls) == 1
    recorded_payload = recorded_calls[0]["payload"]
    assert isinstance(recorded_payload, dict)
    recorded_messages = recorded_payload["messages"]
    assert isinstance(recorded_messages, list)
    recorded_rejected = json.loads(recorded_messages[2]["content"])
    assert recorded_rejected == {
        "reference_kind": "trajectory-redacted-rejected-model-output",
        "response_sha256": sha256(rejected_response.encode("utf-8")).hexdigest(),
        "byte_count": len(rejected_response.encode("utf-8")),
        "body_owner": "runtime_model_rejected_output",
    }
    assert rejected_response not in json.dumps(
        recorded_calls,
        ensure_ascii=False,
    )
    recorded_reply = json.loads(recorded_calls[0]["reply"])
    provider_reply = '{"ok":true}'
    assert recorded_reply == {
        "reference_kind": "trajectory-redacted-structured-model-reply",
        "response_sha256": sha256(provider_reply.encode("utf-8")).hexdigest(),
        "byte_count": len(provider_reply.encode("utf-8")),
        "body_owner": "runtime_model_contract_layer",
    }
    assert provider_reply not in json.dumps(recorded_calls, ensure_ascii=False)
    recorded_response = recorded_calls[0]["response"]
    assert isinstance(recorded_response, dict)
    thinking_marker = json.loads(
        recorded_response["content"][0]["thinking"]
    )
    assert thinking_marker["reference_kind"] == (
        "trajectory-redacted-structured-model-thinking"
    )
    assert thinking_marker["body_owner"] == "provider_ephemeral_response"
    assert "I compared" not in json.dumps(recorded_calls, ensure_ascii=False)


@pytest.mark.parametrize(
    "repair_messages",
    [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "rejected"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "user"},
            {"role": "assistant", "content": "rejected"},
            {"role": "user", "content": "repair"},
        ],
        [
            {"role": "system", "content": "different system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "rejected"},
            {"role": "user", "content": "repair"},
        ],
        [
            {"role": "system", "content": "system", "extra": "forbidden"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "rejected"},
            {"role": "user", "content": "repair"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": {"not": "text"}},
            {"role": "user", "content": "repair"},
        ],
    ],
)
def test_prepared_structured_repair_rejects_any_noncanonical_message_shape(
    repair_messages: list[dict[str, object]],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        models.prepare_complete_structured(
            "system",
            "user",
            mock_payload={"ok": True},
            max_tokens=64,
            purpose="invalid-repair-shape",
            binding=_openai_binding(),
            repair_messages=repair_messages,  # type: ignore[arg-type]
        )


def test_structured_repair_context_rejection_covers_all_four_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client(**_kwargs: object) -> object:
        raise AssertionError("repair admission must precede Provider I/O")

    monkeypatch.setattr(models.httpx, "Client", forbidden_client)
    repair_messages = models.build_structured_repair_messages(
        "system",
        "user",
        rejected_response_text="rejected output " * 200,
        repair_user_content="请重新生成完整 JSON。",
    )

    with pytest.raises(ContextBudgetExceeded):
        models.prepare_complete_structured(
            "system",
            "user",
            mock_payload={"ok": True},
            max_tokens=64,
            purpose="repair-over-budget",
            binding=_openai_binding(),
            configured_input_limit_tokens=1,
            repair_messages=repair_messages,
        )


def test_prepared_structured_context_rejection_performs_no_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client(**_kwargs: object) -> object:
        raise AssertionError("context rejection must happen before Provider I/O")

    monkeypatch.setattr(models.httpx, "Client", forbidden_client)

    with pytest.raises(ContextBudgetExceeded):
        models.prepare_complete_structured(
            "system",
            "a payload that cannot fit inside one configured input token",
            mock_payload={"ok": True},
            max_tokens=64,
            json_mode=True,
            purpose="prepared-structured-over-budget",
            binding=_openai_binding(),
            configured_input_limit_tokens=1,
        )


def test_prepared_gateway_freezes_profile_quota_without_retaining_the_key() -> None:
    binding = replace(
        _openai_binding(),
        quota=ModelProfileQuota(
            requests_per_minute=10,
            tokens_per_minute=100_000,
            tokens_per_week=1_000_000_000,
            max_in_flight=2,
            quota_group="shared-test-account",
        ),
    )

    prepared = models.prepare_complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        max_tokens=64,
        binding=binding,
    )

    assert prepared.api_quota is not None
    assert prepared.api_quota.limits.requests_per_minute == 10
    assert prepared.api_quota.token_reservation > 64
    assert "test-secret" not in repr(prepared)
    assert "test-secret" not in repr(prepared.api_quota)


def test_global_tier_binding_refreshes_its_active_profile_quota_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全局层级保留选择语义，同时继续拥有配额。

    持久端点绑定的生命周期可能长于管理员收紧活跃配置的时点。准备后续物理尝试
    时必须使用当前策略，而不是用旧快照配置共享队列。
    """

    monkeypatch.delenv("PERSONAGRAPH_MODEL_PROVIDER", raising=False)
    profile_id = model_profiles.create_profile(
        kind="model",
        name="active quota owner",
        provider="openai-compatible",
        request_dialect="generic",
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
        quota={"requests_per_minute": 10},
    )["id"]
    binding = model_tiers.resolve_tier(ModelTier.ATTEMPT)

    assert binding.origin is EndpointOrigin.GLOBAL
    assert binding.profile_id is None
    assert binding.quota_profile_id == profile_id
    assert binding.quota.requests_per_minute == 10

    model_profiles.update_profile(
        profile_id,
        {"quota": {"requests_per_minute": 3}},
    )
    prepared = models.prepare_complete_structured(
        "system",
        "user",
        mock_payload={"ok": True},
        max_tokens=64,
        binding=binding,
    )

    assert prepared.api_quota is not None
    assert prepared.api_quota.limits.requests_per_minute == 3
