"""27/002 §4：六档模型配置的解析、回落与限额。

覆盖的是"哪次调用去哪个端点、开不开思考"这件事本身，不涉及任何真实 Provider。
"""
from __future__ import annotations

import json

import pytest

from personagraph.configuration import app_settings as runtime_config
from personagraph.model_io import endpoint_profiles as model_profiles
from personagraph.model_io import tier_bindings as model_tiers
from personagraph.api import router
from personagraph.api.service import ApiError
from personagraph.model_io.api_quota_controller import (
    MODEL_API_QUOTA_ENVIRONMENT_VARIABLE,
    model_profile_quota_environment_value,
)
from personagraph.model_io.tier_bindings import EndpointOrigin, ModelTier


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_config, "CONFIG_PATH", tmp_path / "app_config.json")
    monkeypatch.setattr(model_profiles, "CONFIG_PATH", tmp_path / "model_profiles.json")
    for tier in model_tiers.TIERS:
        for key in (
            model_tiers.profile_setting_key(tier),
            model_tiers.thinking_setting_key(tier),
            model_tiers.reasoning_effort_setting_key(tier),
        ):
            monkeypatch.delenv(runtime_config._CONFIG_ENV_MAP[key], raising=False)
    for var in ("PERSONAGRAPH_MODEL_PROVIDER", "PERSONAGRAPH_API_KEY",
                "PERSONAGRAPH_BASE_URL", "PERSONAGRAPH_MODEL",
                MODEL_API_QUOTA_ENVIRONMENT_VARIABLE):
        monkeypatch.delenv(var, raising=False)


def _make_profile(name: str, model: str) -> str:
    return model_profiles.create_profile(
        kind="model", name=name, provider="deepseek",
        base_url="https://example.invalid/anthropic", model=model, api_key="k-" + name,
    )["id"]


# provider 会先按真实契约构造 mock 回退payload，所以 user_content 必须是合法的
# Attempt/Verification 输入，不能随便传 "{}"。
_NODE_INPUT = json.dumps({
    "node": {
        "objective": "检查一个节点",
        "acceptances": [{"acceptance_id": "acc_1", "statement": "做完了"}],
    },
    "current_user_input": "跑一次",
})


def _d(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


# --- 默认值 ---------------------------------------------------------------

def test_thinking_defaults_off_for_every_tier():
    """27/002 §8.1：默认关，优先响应速度。这条会改变既有安装的行为。"""
    for tier in model_tiers.TIERS:
        assert model_tiers.resolve_tier(tier).thinking_enabled is False


def test_unconfigured_tier_inherits_global_endpoint():
    runtime_config.update_config({"provider": "deepseek", "model": "global-model"})
    binding = model_tiers.resolve_tier(ModelTier.ARCHITECT)
    assert binding.origin is EndpointOrigin.GLOBAL
    assert binding.model == "global-model"
    assert binding.configured is False


def test_openai_global_binding_materializes_gateway_defaults_before_dialect_resolution():
    runtime_config.update_config({
        "provider": "openai-compatible",
        "api_key": "k",
    })

    binding = model_tiers.resolve_tier(ModelTier.ARCHITECT)

    assert binding.base_url == "https://api.openai.com/v1"
    assert binding.model == "gpt-4o-mini"
    assert binding.request_dialect == "openai-native"


def test_raw_anthropic_global_binding_keeps_native_endpoint_and_no_guessed_model(
    monkeypatch,
):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "k")
    monkeypatch.delenv("PERSONAGRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)

    binding = model_tiers.resolve_tier(ModelTier.ARCHITECT)

    assert binding.provider == "anthropic-compatible"
    assert binding.base_url == "https://api.anthropic.com"
    assert binding.model == ""
    assert binding.request_dialect == "anthropic-native"


def test_invalid_hand_edited_global_provider_fails_closed_without_breaking_tier_view():
    runtime_config.update_config({"provider": "not-a-provider", "model": "unsafe"})

    tiers = model_tiers.redacted_tier_view()

    assert runtime_config.model_configured() is False
    assert all(row["provider"] == "mock" for row in tiers)


def test_invalid_hand_edited_global_dialect_also_fails_closed():
    runtime_config.update_config({
        "provider": "openai-compatible",
        "request_dialect": "made-up",
        "model": "unsafe",
    })
    assert all(
        row["provider"] == "mock" for row in model_tiers.redacted_tier_view()
    )


def test_tier_names_match_the_settings_key_namespace():
    """两边分别持有键名与语义；漂移会让某一档永远静默落回全局。"""
    assert tuple(t.value for t in model_tiers.TIERS) == runtime_config.TIER_SETTING_NAMES


# --- profile 指针 ---------------------------------------------------------

def test_tier_uses_its_own_profile_endpoint():
    profile_id = _make_profile("fast", "fast-model")
    runtime_config.update_config({
        "provider": "deepseek", "model": "global-model",
        "tier_node_verification_profile_id": profile_id,
    })
    binding = model_tiers.resolve_tier(ModelTier.NODE_VERIFICATION)
    assert binding.origin is EndpointOrigin.PROFILE
    assert (binding.model, binding.api_key) == ("fast-model", "k-fast")
    # 其它档不受影响
    assert model_tiers.resolve_tier(ModelTier.ARCHITECT).model == "global-model"


def test_several_tiers_may_share_one_profile():
    """允许复用是用户明确要求的形态：指针相同即可，不必复制四份凭据。"""
    profile_id = _make_profile("shared", "shared-model")
    runtime_config.update_config({
        f"tier_{name}_profile_id": profile_id
        for name in ("architect", "attempt", "node_verification")
    })
    models = {
        model_tiers.resolve_tier(t).model
        for t in (ModelTier.ARCHITECT, ModelTier.ATTEMPT,
                  ModelTier.NODE_VERIFICATION)
    }
    assert models == {"shared-model"}


def test_profile_quota_travels_with_every_tier_using_that_profile():
    profile_id = model_profiles.create_profile(
        kind="model",
        name="shared quota",
        provider="openai-compatible",
        request_dialect="generic",
        base_url="https://example.invalid/v1",
        model="shared-model",
        api_key="k-shared",
        quota={
            "requests_per_minute": 10,
            "tokens_per_minute": 100_000,
            "tokens_per_week": 1_000_000_000,
            "max_in_flight": 2,
            "quota_group": "shared-account",
        },
    )["id"]
    runtime_config.update_config({
        "tier_architect_profile_id": profile_id,
        "tier_attempt_profile_id": profile_id,
    })

    architect = model_tiers.resolve_tier(ModelTier.ARCHITECT)
    attempt = model_tiers.resolve_tier(ModelTier.ATTEMPT)

    assert architect.quota == attempt.quota
    assert architect.quota.requests_per_minute == 10
    assert architect.quota.max_in_flight == 2
    assert architect.quota.quota_group == "shared-account"


def test_global_binding_inherits_quota_only_from_the_matching_active_profile():
    profile_id = model_profiles.create_profile(
        kind="model",
        name="active quota",
        provider="openai-compatible",
        request_dialect="generic",
        base_url="https://example.invalid/v1",
        model="active-model",
        api_key="k-active",
        quota={"requests_per_minute": 10, "max_in_flight": 2},
    )["id"]
    assert model_profiles.resolve_active_profile("model").id == profile_id

    inherited = model_tiers.resolve_tier(ModelTier.ARCHITECT)
    assert inherited.origin is EndpointOrigin.GLOBAL
    assert inherited.quota.requests_per_minute == 10

    runtime_config.update_config({"api_key": "a-manually-replaced-key"})
    detached = model_tiers.resolve_tier(ModelTier.ARCHITECT)
    assert detached.quota == model_profiles.ModelProfileQuota()


def test_isolated_global_binding_inherits_the_parent_projected_quota(monkeypatch):
    quota = model_profiles.ModelProfileQuota(
        requests_per_minute=10,
        tokens_per_minute=300_000,
        tokens_per_week=1_000_000_000,
        max_in_flight=2,
        quota_group="docbench-shared-account",
    )
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "deepseek-chat")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "worker-secret")
    monkeypatch.setenv(
        MODEL_API_QUOTA_ENVIRONMENT_VARIABLE,
        model_profile_quota_environment_value(quota),
    )

    binding = model_tiers.resolve_tier(ModelTier.L1)

    assert binding.origin is EndpointOrigin.GLOBAL
    assert binding.quota == quota


def test_deleted_profile_falls_back_and_says_so():
    """陈旧档位指针仍可见，但已删除的活动凭据不能继续外呼。"""
    profile_id = _make_profile("temp", "temp-model")
    runtime_config.update_config({
        "provider": "deepseek", "model": "global-model",
        "tier_final_gate_profile_id": profile_id,
    })
    model_profiles.delete_profile(profile_id)

    binding = model_tiers.resolve_tier(ModelTier.FINAL_GATE)
    assert binding.origin is EndpointOrigin.STALE_PROFILE
    assert binding.provider == "mock"
    assert binding.profile_id == profile_id


def test_vision_profile_cannot_be_used_as_a_model_tier():
    vision_id = model_profiles.create_profile(
        kind="vision", name="eyes", provider="openai",
        base_url="https://example.invalid/v1", model="vision-model", api_key="k-v",
    )["id"]
    runtime_config.update_config({
        "provider": "deepseek", "model": "global-model",
        "tier_attempt_profile_id": vision_id,
    })
    binding = model_tiers.resolve_tier(ModelTier.ATTEMPT)
    assert binding.origin is EndpointOrigin.STALE_PROFILE
    assert binding.model == "global-model"


# --- 开关取值 -------------------------------------------------------------

@pytest.mark.parametrize("stored,expected", [
    ("on", True), ("1", True), ("true", True), ("YES", True),
    ("off", False), ("0", False), ("false", False),
    ("", False), ("gibberish", False),
])
def test_thinking_switch_parsing(stored, expected):
    """手改过的配置文件不该让一个 Turn 失败，无法辨认就按默认处理。"""
    runtime_config.update_config({"tier_l1_thinking": stored} if stored else {})
    assert model_tiers.read_thinking_enabled(ModelTier.L1) is expected


# --- 脱敏 -----------------------------------------------------------------

def test_redacted_tier_view_never_carries_the_key():
    _id = _make_profile("secret", "m")
    runtime_config.update_config({"tier_architect_profile_id": _id})
    for row in model_tiers.redacted_tier_view():
        assert "api_key" not in row
        assert row["has_api_key"] in (True, False)
    assert "k-secret" not in repr(model_tiers.resolve_tier(ModelTier.ARCHITECT))


def test_redacted_view_separates_configured_effort_from_the_wire_value():
    common = {
        "tier": ModelTier.ATTEMPT,
        "provider": "anthropic-compatible",
        "model": "reasoning-model",
        "api_key": "secret",
        "thinking_enabled": False,
        "reasoning_effort": model_tiers.ReasoningEffort.HIGH,
        "origin": EndpointOrigin.PROFILE,
    }
    deepseek = model_tiers.ModelTierBinding(
        **common,
        base_url="https://api.deepseek.com/anthropic",
        request_dialect="deepseek-anthropic",
    ).redacted()
    anthropic = model_tiers.ModelTierBinding(
        **common,
        base_url="https://api.anthropic.com",
        request_dialect="anthropic-native",
    ).redacted()

    assert deepseek["reasoning_effort"] == "high"
    assert deepseek["effective_reasoning_effort"] is None
    assert anthropic["reasoning_effort"] == "high"
    assert anthropic["effective_reasoning_effort"] == "high"


# --- 配置写入 API ---------------------------------------------------------

def test_config_api_rejects_a_profile_id_that_does_not_exist():
    """运行期为了不毁掉 Turn 会静默回落；保存设置时不行——选中的东西必须存在。"""
    with pytest.raises(ApiError) as exc:
        _d("POST", "/api/config", {"tier_architect_profile_id": "mp_nope"})
    assert exc.value.code == "PROFILE_NOT_FOUND"


def test_config_api_rejects_a_vision_profile_for_a_model_tier():
    vision_id = model_profiles.create_profile(
        kind="vision",
        name="eyes",
        provider="openai",
        base_url="https://example.invalid/v1",
        model="vision-model",
        api_key="k-v",
    )["id"]

    with pytest.raises(ApiError) as exc:
        _d("POST", "/api/config", {"tier_architect_profile_id": vision_id})

    assert exc.value.code == "PROFILE_KIND_MISMATCH"
    assert runtime_config.get_setting("tier_architect_profile_id") is None


def test_config_api_accepts_bool_and_on_off_for_thinking():
    profile_id = _make_profile("p", "m")
    _d("POST", "/api/config", {
        "tier_architect_profile_id": profile_id,
        "tier_architect_thinking": True,
        "tier_attempt_thinking": "on",
    })
    assert model_tiers.read_thinking_enabled(ModelTier.ARCHITECT) is True
    assert model_tiers.read_thinking_enabled(ModelTier.ATTEMPT) is True

    _d("POST", "/api/config", {"tier_architect_thinking": False})
    assert model_tiers.read_thinking_enabled(ModelTier.ARCHITECT) is False


def test_config_api_accepts_openai_reasoning_effort_levels():
    _d("POST", "/api/config", {
        "provider": "openai-compatible",
        "request_dialect": "openai",
        "base_url": "https://api.openai.com/v1",
        "tier_architect_reasoning_effort": "xhigh",
        "tier_final_gate_reasoning_effort": "none",
    })
    assert model_tiers.resolve_tier(ModelTier.ARCHITECT).reasoning_effort == "xhigh"
    assert model_tiers.resolve_tier(ModelTier.FINAL_GATE).reasoning_effort == "none"


def test_config_api_rejects_effort_left_over_from_an_incompatible_profile():
    openai_id = model_profiles.create_profile(
        kind="model",
        name="openai",
        provider="openai-compatible",
        request_dialect="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-test",
        api_key="k-openai",
    )["id"]
    anthropic_id = model_profiles.create_profile(
        kind="model",
        name="anthropic",
        provider="anthropic-compatible",
        request_dialect="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-test",
        api_key="k-anthropic",
    )["id"]
    _d("PUT", "/api/config", {
        "tier_architect_profile_id": openai_id,
        "tier_architect_reasoning_effort": "minimal",
    })

    with pytest.raises(ApiError) as exc:
        _d("PUT", "/api/config", {
            "tier_architect_profile_id": anthropic_id,
        })

    assert exc.value.code == "TIER_REASONING_EFFORT_INCOMPATIBLE"
    assert exc.value.details["tier"] == "architect"
    assert exc.value.details["reasoning_effort"] == "minimal"


def test_config_api_rejects_unknown_reasoning_effort():
    with pytest.raises(ApiError) as exc:
        _d("POST", "/api/config", {"tier_attempt_reasoning_effort": "turbo"})
    assert exc.value.code == "INVALID_REASONING_EFFORT"


def test_config_api_rejects_an_uninterpretable_switch():
    with pytest.raises(ApiError) as exc:
        _d("POST", "/api/config", {"tier_l1_thinking": "sometimes"})
    assert exc.value.code == "INVALID_TIER_THINKING"


def test_dash_clears_a_tier_back_to_the_global_endpoint():
    """空串沿用既有的"不改该字段"语义，所以清除需要一个显式记号。"""
    profile_id = _make_profile("p", "tier-model")
    runtime_config.update_config({"provider": "deepseek", "model": "global-model"})
    _d("POST", "/api/config", {"tier_l1_profile_id": profile_id})
    assert model_tiers.resolve_tier(ModelTier.L1).model == "tier-model"

    _d("POST", "/api/config",
       {"tier_l1_profile_id": model_tiers.CLEARED_PROFILE_ID})
    assert model_tiers.resolve_tier(ModelTier.L1).origin is EndpointOrigin.GLOBAL


def test_get_config_reports_the_endpoint_each_tier_will_actually_call():
    runtime_config.update_config({"provider": "deepseek", "model": "global-model"})
    tiers = _d("GET", "/api/config")["model_tiers"]
    assert {t["tier"] for t in tiers} == set(runtime_config.TIER_SETTING_NAMES)
    # 未配置的档也要报出继承来的端点，而不是留空
    assert all(t["model"] == "global-model" for t in tiers)
    assert all(t["origin"] == "global" for t in tiers)


def test_update_config_returns_the_same_effective_view_as_get_config():
    profile_id = _make_profile("architect", "architect-model")

    updated = _d(
        "PUT",
        "/api/config",
        {
            "tier_architect_profile_id": profile_id,
            "tier_architect_thinking": True,
        },
    )

    assert updated == _d("GET", "/api/config")
    architect = next(
        row for row in updated["model_tiers"] if row["tier"] == "architect"
    )
    assert architect["profile_id"] == profile_id
    assert architect["model"] == "architect-model"
    assert architect["thinking_enabled"] is True


# --- 前后端键名对齐 -------------------------------------------------------

def test_frontend_tier_keys_match_the_backend():
    """前端自己写了一份档位清单，漂了就会有一档永远存不进去。

    用 Python 测试读 JS，是因为这条约束跨语言：任何一边单独的测试都看不见它。
    """
    import re
    from pathlib import Path

    js = Path("frontend/src/composables/useConfig.js").read_text(encoding="utf-8")
    block = js.split("export const MODEL_TIERS = [")[1].split("];")[0]
    assert tuple(re.findall(r'key:\s*"([a-z0-9_]+)"', block)) == (
        runtime_config.TIER_SETTING_NAMES
    )
    cleared = re.search(r'TIER_CLEARED = "([^"]+)"', js).group(1)
    assert cleared == model_tiers.CLEARED_PROFILE_ID


def test_frontend_submits_every_tier_field_the_backend_accepts():
    """前端提交的键必须都在后端 writable_fields 里，否则会被静默丢弃。"""
    from pathlib import Path

    from personagraph.api.service.status import _TIER_WRITABLE_FIELDS

    js = Path("frontend/src/composables/useConfig.js").read_text(encoding="utf-8")
    for name in runtime_config.TIER_SETTING_NAMES:
        assert "tier_${key}_profile_id" in js or f"tier_{name}_profile_id" in js
    assert "tier_${key}_reasoning_effort" in js
    assert set(_TIER_WRITABLE_FIELDS) == {
        f"tier_{name}_{suffix}"
        for name in runtime_config.TIER_SETTING_NAMES
        for suffix in ("profile_id", "thinking", "reasoning_effort")
    }


# --- 绑定是否真的到了请求体 -----------------------------------------------

class _FakeResponse:
    status_code = 200

    @staticmethod
    def json():
        return {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }

    def raise_for_status(self):
        return None


class _CapturingClient:
    """记录 anthropic 网关实际发出的请求体。"""

    captured: list[dict] = []

    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, endpoint, headers=None, json=None, content=None, **__):
        payload = json or {}
        if content is not None:
            payload = __import__("json").loads(content)
        type(self).captured.append(
            {"endpoint": endpoint, "headers": headers or {}, "payload": payload}
        )
        return _FakeResponse()


@pytest.fixture
def _captured_requests(monkeypatch):
    import httpx

    from personagraph.model_io import gateway as models

    _CapturingClient.captured = []
    monkeypatch.setattr(models.httpx, "Client", _CapturingClient)
    monkeypatch.setattr(models, "record_model_call", lambda **_: None)
    monkeypatch.setattr(httpx, "Client", _CapturingClient, raising=False)
    return _CapturingClient.captured


def test_node_verification_provider_sends_its_tier_endpoint_and_switch(
    _captured_requests,
):
    """端到端：分档配置 → provider → 真实请求体。"""
    from personagraph.l2.task_execution.work_run.model_providers import (
        build_verification_structured_provider,
    )
    from personagraph.l2.task_execution.work_run.model_profile import (
        WorkRunStructuredModelProfile,
    )

    profile_id = _make_profile("verify", "cheap-model")
    runtime_config.update_config({
        "provider": "deepseek", "model": "global-model", "api_key": "global-key",
        "tier_node_verification_profile_id": profile_id,
        "tier_node_verification_thinking": "off",
    })

    provider = build_verification_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=32_768,
            verification_max_output_tokens=8_192,
            timeout_s=600.0,
        )
    )
    provider("sys", _NODE_INPUT, model_call_id="mc_1", purpose="runtime_x")

    assert len(_captured_requests) == 1
    sent = _captured_requests[0]
    assert sent["payload"]["model"] == "cheap-model"
    assert sent["payload"]["thinking"] == {"type": "disabled"}
    assert sent["headers"]["x-api-key"] == "k-verify"   # 该档自己的凭据
    assert "example.invalid" in sent["endpoint"]


def test_thinking_on_sends_enabled_and_keeps_the_generous_cap(_captured_requests):
    from personagraph.l2.task_execution.work_run.model_providers import (
        build_attempt_structured_provider,
    )
    from personagraph.l2.task_execution.work_run.model_profile import (
        WorkRunStructuredModelProfile,
    )

    runtime_config.update_config({
        "provider": "deepseek", "model": "deepseek-global-model", "api_key": "global-key",
        "tier_attempt_thinking": "on",
    })
    provider = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=32_768,
            verification_max_output_tokens=8_192,
            timeout_s=600.0,
        )
    )
    provider("sys", _NODE_INPUT, model_call_id="mc_2", purpose="runtime_y")

    payload = _captured_requests[0]["payload"]
    assert payload["thinking"] == {"type": "enabled"}
    # 开思考时上限不得收窄——收窄会把可用调用变成截断失败（§6.4 A2/A4）
    assert payload["max_tokens"] == 32_768
    # budget_tokens 实测无效，绝不能出现在请求里（§6.2）
    assert "budget_tokens" not in payload["thinking"]


def test_deepseek_anthropic_tier_sends_the_selected_effort(_captured_requests):
    from personagraph.l2.task_execution.work_run.model_providers import (
        build_attempt_structured_provider,
    )
    from personagraph.l2.task_execution.work_run.model_profile import (
        WorkRunStructuredModelProfile,
    )

    profile_id = _make_profile("deep", "deepseek-deep-model")
    runtime_config.update_config({
        "tier_attempt_profile_id": profile_id,
        "tier_attempt_thinking": "on",
        "tier_attempt_reasoning_effort": "xhigh",
    })
    provider = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=32_768,
            verification_max_output_tokens=8_192,
            timeout_s=600.0,
        )
    )

    provider("sys", _NODE_INPUT, model_call_id="mc_effort", purpose="runtime_effort")

    payload = _captured_requests[0]["payload"]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["output_config"] == {"effort": "high"}
    assert "temperature" not in payload


def test_thinking_off_keeps_the_configured_output_cap(_captured_requests):
    from personagraph.l2.task_execution.work_run.model_providers import (
        build_attempt_structured_provider,
    )
    from personagraph.l2.task_execution.work_run.model_profile import (
        WorkRunStructuredModelProfile,
    )

    runtime_config.update_config({
        # 使用已注册的模型家族配置，使该测试只验证思考开关/输出上限契约，
        # 而不触发未知模型的上下文准入（其保守窗口有意设为 32K）。
        "provider": "deepseek", "model": "deepseek-v4-pro", "api_key": "global-key",
    })
    provider = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=32_768,
            verification_max_output_tokens=8_192,
            timeout_s=600.0,
        )
    )
    provider("sys", _NODE_INPUT, model_call_id="mc_3", purpose="runtime_z")
    assert _captured_requests[0]["payload"]["max_tokens"] == 32_768
