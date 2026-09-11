"""保存多个端点配置，并在它们之间切换。

真正重要的两项性质并非“能够存储内容”：列表绝不能携带密钥；激活某个配置后，
应用其余部分读取的内容必须实际发生变化。
"""

from __future__ import annotations

import json
import stat

import pytest

from personagraph.model_io import endpoint_profiles as model_profiles
from personagraph.api import router
from personagraph.api.service.errors import ApiError


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


def _create(kind="model", name="主模型", key="sk-live", **overrides):
    payload = {
        "kind": kind, "name": name, "provider": "deepseek",
        "base_url": "https://api.deepseek.com/anthropic",
        "model": "deepseek-v4-pro", "api_key": key,
    }
    payload.update(overrides)
    if "base_url" not in overrides:
        provider = str(payload["provider"]).strip().lower()
        dialect = str(payload.get("request_dialect") or "auto").strip().lower()
        if provider == "openai-compatible":
            payload["base_url"] = "https://api.openai.com/v1"
        elif provider == "anthropic" or dialect == "anthropic":
            payload["base_url"] = "https://api.anthropic.com"
    return _route_payload("POST", "/api/model-profiles", payload)["profile"]


def _listing(kind):
    return _route_payload("GET", "/api/model-profiles", {})["profiles"][kind]


# --- 列表永远不带 key ------------------------------------------------------------


def test_a_listing_says_a_key_exists_without_carrying_it():
    """设置页会频繁加载此内容；若密钥随之返回，就会出现在每张截图以及所有
    曾渲染该页面的渲染器中。"""

    _create(key="sk-should-never-appear")
    body = json.dumps(_route_payload("GET", "/api/model-profiles", {}), ensure_ascii=False)

    assert "sk-should-never-appear" not in body
    assert _listing("model")["profiles"][0]["has_api_key"] is True


def test_reading_a_key_back_is_its_own_deliberate_request():
    profile = _create(key="sk-live")
    revealed = _route_payload("GET", f"/api/model-profiles/{profile['id']}/secret", {})
    assert revealed["api_key"] == "sk-live"


# --- 共享 API 配额 ---------------------------------------------------------------


def test_model_profile_quota_is_persisted_resolved_and_safe_to_list():
    quota = {
        "requests_per_minute": 10,
        "tokens_per_minute": 100_000,
        "tokens_per_week": 1_000_000_000,
        "max_in_flight": 2,
        "quota_group": "sjtu-models",
    }

    profile = _create(key="sk-never-list", quota=quota)
    listed = _listing("model")["profiles"][0]
    resolved = model_profiles.resolve_profile(profile["id"])

    assert profile["quota"] == quota
    assert listed["quota"] == quota
    assert "api_key" not in listed
    assert resolved is not None
    assert resolved.quota.requests_per_minute == 10
    assert resolved.quota.tokens_per_minute == 100_000
    assert resolved.quota.tokens_per_week == 1_000_000_000
    assert resolved.quota.max_in_flight == 2
    assert resolved.quota.quota_group == "sjtu-models"


def test_active_profile_lookup_returns_the_full_runtime_profile():
    quota = {"requests_per_minute": 10, "quota_group": "sjtu-models"}
    created = _create(key="sk-runtime-only", quota=quota)

    active = model_profiles.resolve_active_profile("model")

    assert active is not None
    assert active.id == created["id"]
    assert active.api_key == "sk-runtime-only"
    assert active.quota.requests_per_minute == 10
    assert active.quota.quota_group == "sjtu-models"
    assert model_profiles.resolve_active_profile("unknown") is None


def test_omitted_quota_means_every_dimension_is_unlimited():
    profile = _create()

    assert profile["quota"] == {
        "requests_per_minute": None,
        "tokens_per_minute": None,
        "tokens_per_week": None,
        "max_in_flight": None,
        "quota_group": None,
    }


def test_quota_patch_merges_fields_and_null_clears_one_dimension():
    profile = _create(quota={
        "requests_per_minute": 10,
        "tokens_per_minute": 100_000,
        "tokens_per_week": 1_000_000_000,
        "max_in_flight": 2,
        "quota_group": "sjtu-models",
    })

    updated = _route_payload(
        "PATCH",
        f"/api/model-profiles/{profile['id']}",
        {"quota": {"requests_per_minute": 12, "tokens_per_week": None}},
    )["profile"]

    assert updated["quota"] == {
        "requests_per_minute": 12,
        "tokens_per_minute": 100_000,
        "tokens_per_week": None,
        "max_in_flight": 2,
        "quota_group": "sjtu-models",
    }


def test_null_quota_patch_clears_the_whole_policy():
    profile = _create(quota={
        "requests_per_minute": 10,
        "quota_group": "sjtu-models",
    })

    updated = _route_payload(
        "PATCH", f"/api/model-profiles/{profile['id']}", {"quota": None}
    )["profile"]

    assert set(updated["quota"].values()) == {None}


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "10", (1 << 63)],
)
def test_quota_limits_must_be_positive_bounded_integers(value):
    with pytest.raises(ApiError) as raised:
        _create(quota={"requests_per_minute": value})

    assert raised.value.code == "PROFILE_QUOTA_VALUE_INVALID"


@pytest.mark.parametrize(
    "group",
    ["", "contains space", "contains/slash", 42, "x" * 129],
)
def test_quota_group_is_a_bounded_machine_safe_name(group):
    with pytest.raises(ApiError) as raised:
        _create(quota={"quota_group": group})

    assert raised.value.code == "PROFILE_QUOTA_GROUP_INVALID"


def test_quota_rejects_unknown_fields_instead_of_silently_ignoring_a_typo():
    with pytest.raises(ApiError) as raised:
        _create(quota={"request_per_minute": 10})

    assert raised.value.code == "PROFILE_QUOTA_FIELD_UNKNOWN"


def test_vision_profile_cannot_claim_task_model_quota():
    with pytest.raises(ApiError) as raised:
        _create(
            kind="vision",
            provider="highland-vl",
            quota={"requests_per_minute": 10},
        )

    assert raised.value.code == "PROFILE_QUOTA_KIND_UNSUPPORTED"


def test_quota_top_level_shape_is_checked_at_the_api_boundary():
    with pytest.raises(ApiError) as raised:
        _create(quota="10 RPM")

    assert raised.value.code == "INVALID_REQUEST_FIELD"


def test_same_credential_and_normalized_endpoint_cannot_disagree_on_limits():
    first = _create(
        name="first",
        key="sk-shared-secret",
        base_url="https://MODELS.example.test/v1/",
        quota={"requests_per_minute": 10, "max_in_flight": 2},
    )

    with pytest.raises(ApiError) as raised:
        _create(
            name="conflicting",
            key="sk-shared-secret",
            base_url="https://models.example.test/v1",
            quota={"requests_per_minute": 12, "max_in_flight": 2},
        )

    assert raised.value.code == "PROFILE_QUOTA_SCOPE_CONFLICT"
    assert "sk-shared-secret" not in str(raised.value)
    assert [item["id"] for item in _listing("model")["profiles"]] == [first["id"]]


def test_explicit_quota_group_shares_limits_across_different_credentials():
    _create(
        name="first",
        key="sk-one",
        base_url="https://models.example.test/v1",
        quota={"tokens_per_minute": 100_000, "quota_group": "shared-account"},
    )

    with pytest.raises(ApiError) as raised:
        _create(
            name="conflicting",
            key="sk-two",
            base_url="https://models.example.test/v1/",
            quota={"tokens_per_minute": 90_000, "quota_group": "shared-account"},
        )

    assert raised.value.code == "PROFILE_QUOTA_SCOPE_CONFLICT"
    assert "sk-one" not in str(raised.value)
    assert "sk-two" not in str(raised.value)


def test_configured_and_all_unset_limits_conflict_in_the_same_scope():
    _create(
        name="legacy-shaped",
        key="sk-shared",
        base_url="https://models.example.test/v1",
    )

    with pytest.raises(ApiError) as raised:
        _create(
            name="configured",
            key="sk-shared",
            base_url="https://models.example.test/v1",
            quota={"requests_per_minute": 10},
        )

    assert raised.value.code == "PROFILE_QUOTA_SCOPE_CONFLICT"


def test_same_scope_with_identical_limits_is_allowed():
    quota = {
        "requests_per_minute": 10,
        "tokens_per_minute": 100_000,
        "max_in_flight": 2,
        "quota_group": "shared-account",
    }

    _create(
        name="one",
        key="sk-one",
        base_url="https://models.example.test/v1",
        quota=quota,
    )
    second = _create(
        name="two",
        key="sk-two",
        base_url="https://models.example.test/v1/",
        quota=quota,
    )

    assert second["quota"] == {
        **quota,
        "tokens_per_week": None,
    }


def test_profile_update_cannot_introduce_a_quota_scope_conflict():
    first = _create(
        name="limited",
        key="sk-one",
        base_url="https://models.example.test/v1",
        quota={"requests_per_minute": 10, "quota_group": "shared-account"},
    )
    second = _create(
        name="other scope",
        key="sk-two",
        base_url="https://models.example.test/v1",
        quota={"requests_per_minute": 20, "quota_group": "other-account"},
    )

    with pytest.raises(ApiError) as raised:
        _route_payload(
            "PATCH",
            f"/api/model-profiles/{second['id']}",
            {"quota": {"quota_group": "shared-account"}},
        )

    assert raised.value.code == "PROFILE_QUOTA_SCOPE_CONFLICT"
    stored = {item["id"]: item for item in _listing("model")["profiles"]}
    assert stored[first["id"]]["quota"]["quota_group"] == "shared-account"
    assert stored[second["id"]]["quota"] == {
        "requests_per_minute": 20,
        "tokens_per_minute": None,
        "tokens_per_week": None,
        "max_in_flight": None,
        "quota_group": "other-account",
    }


def test_different_credentials_have_independent_implicit_quota_scopes():
    _create(
        name="one",
        key="sk-one",
        base_url="https://models.example.test/v1",
        quota={"requests_per_minute": 10},
    )
    second = _create(
        name="two",
        key="sk-two",
        base_url="https://models.example.test/v1",
        quota={"requests_per_minute": 20},
    )

    assert second["quota"]["requests_per_minute"] == 20


# --- 启用要真的改变全局 ----------------------------------------------------------


def test_activating_changes_what_the_rest_of_the_application_reads():
    from personagraph.configuration.app_settings import get_setting

    first = _create(name="A", model="model-a", key="sk-a")
    second = _create(name="B", model="model-b", key="sk-b")

    assert get_setting("model") == "model-a"  # 第一份自动启用
    assert first["active"] is True and second["active"] is False

    _route_payload("POST", f"/api/model-profiles/{second['id']}/activate", {})
    assert get_setting("model") == "model-b"
    assert get_setting("api_key") == "sk-b"


def test_the_two_kinds_do_not_share_an_active_slot():
    from personagraph.configuration.app_settings import get_setting

    _create(kind="model", name="对话", model="chat-model", key="sk-chat")
    _create(kind="vision", name="图片", model="vision-model", key="sk-vision")

    assert get_setting("model") == "chat-model"
    assert get_setting("vision_model") == "vision-model"
    assert get_setting("api_key") != get_setting("vision_api_key")


# --- 编辑 -----------------------------------------------------------------------


def test_an_empty_key_on_edit_means_keep_it_not_clear_it():
    """否则只保存一次表单就会静默抹除可用密钥。"""

    profile = _create(key="sk-original")
    _route_payload("PATCH", f"/api/model-profiles/{profile['id']}", {"name": "改个名"})

    assert _listing("model")["profiles"][0]["name"] == "改个名"
    assert _route_payload(
        "GET", f"/api/model-profiles/{profile['id']}/secret", {}
    )["api_key"] == "sk-original"


def test_editing_the_active_profile_takes_effect_immediately():
    from personagraph.configuration.app_settings import get_setting

    profile = _create(model="old-model")
    _route_payload("PATCH", f"/api/model-profiles/{profile['id']}", {"model": "new-model"})
    assert get_setting("model") == "new-model"


def test_request_dialect_defaults_to_auto_and_is_listed():
    profile = _create(provider="anthropic-compatible")

    assert profile["request_dialect"] == "auto"
    assert _listing("model")["profiles"][0]["request_dialect"] == "auto"


def test_request_dialect_is_persisted_edited_and_written_through_when_active():
    from personagraph.configuration.app_settings import get_setting

    profile = _create(
        provider="openai-compatible",
        request_dialect="openai",
    )
    assert profile["request_dialect"] == "openai"
    assert get_setting("request_dialect") == "openai"

    updated = _route_payload(
        "PATCH",
        f"/api/model-profiles/{profile['id']}",
        {
            "request_dialect": "  DeepSeek  ",
            "base_url": "https://api.deepseek.com/v1",
        },
    )["profile"]

    assert updated["request_dialect"] == "deepseek"
    assert get_setting("request_dialect") == "deepseek"


def test_same_provider_patch_preserves_an_omitted_explicit_request_dialect():
    """PATCH 中省略字段表示保留该字段，即使重复提供 provider 也是如此。"""

    from personagraph.configuration.app_settings import get_setting

    profile = _create(
        provider="openai-compatible",
        request_dialect="generic",
        base_url="https://proxy.example/v1",
    )

    updated = _route_payload(
        "PATCH",
        f"/api/model-profiles/{profile['id']}",
        {"provider": "  OPENAI-COMPATIBLE  ", "name": "same endpoint"},
    )["profile"]

    assert updated["request_dialect"] == "generic"
    assert _listing("model")["profiles"][0]["request_dialect"] == "generic"
    assert get_setting("request_dialect") == "generic"


def test_changed_base_url_reinfers_an_omitted_request_dialect():
    """真实端点变更后不得保留过时的厂商专属方言。"""

    profile = _create(
        provider="openai-compatible",
        request_dialect="openai",
        base_url="https://api.openai.com/v1",
    )

    updated = _route_payload(
        "PATCH",
        f"/api/model-profiles/{profile['id']}",
        {"base_url": "https://proxy.example/v1"},
    )["profile"]

    assert updated["request_dialect"] == "auto"


def test_vision_profile_keeps_the_safe_auto_dialect_default():
    profile = _create(kind="vision", provider="highland-vl")

    assert profile["request_dialect"] == "auto"


@pytest.mark.parametrize("request_dialect", ["responses", "claude-v2", "unknown"])
def test_profile_rejects_an_unknown_request_dialect(request_dialect):
    with pytest.raises(ApiError) as raised:
        _create(request_dialect=request_dialect)

    assert raised.value.code == "PROFILE_REQUEST_DIALECT_UNKNOWN"


@pytest.mark.parametrize(
    ("provider", "expected", "expected_dialect"),
    [
        ("  deepseek  ", "anthropic-compatible", "deepseek"),
        ("ANTHROPIC", "anthropic-compatible", "anthropic"),
        ("  OPENAI-COMPATIBLE ", "openai-compatible", "auto"),
    ],
)
def test_model_provider_is_canonicalized_when_created(
    provider, expected, expected_dialect
):
    profile = _create(provider=provider)

    assert profile["provider"] == expected
    assert profile["request_dialect"] == expected_dialect
    assert _listing("model")["profiles"][0]["provider"] == expected


def test_model_provider_is_canonicalized_when_updated():
    profile = _create(provider="openai-compatible")

    updated = _route_payload(
        "PATCH",
        f"/api/model-profiles/{profile['id']}",
        {
            "provider": "  DeepSeek  ",
            "base_url": "https://api.deepseek.com/anthropic",
        },
    )["profile"]

    assert updated["provider"] == "anthropic-compatible"
    assert updated["request_dialect"] == "deepseek"
    assert _listing("model")["profiles"][0]["provider"] == "anthropic-compatible"


def test_model_profile_rejects_a_dialect_for_the_other_protocol():
    with pytest.raises(ApiError) as raised:
        _create(provider="anthropic-compatible", request_dialect="openai")

    assert raised.value.code == "PROFILE_REQUEST_DIALECT_INCOMPATIBLE"


def test_creating_a_model_profile_rejects_an_unsupported_provider():
    with pytest.raises(ApiError) as raised:
        _create(provider="gpt-9")

    assert raised.value.code == "PROFILE_PROVIDER_UNKNOWN"


def test_updating_a_model_profile_rejects_an_unsupported_provider():
    profile = _create(provider="openai-compatible")

    with pytest.raises(ApiError) as raised:
        _route_payload(
            "PATCH",
            f"/api/model-profiles/{profile['id']}",
            {"provider": "gpt-9"},
        )

    assert raised.value.code == "PROFILE_PROVIDER_UNKNOWN"
    assert _listing("model")["profiles"][0]["provider"] == "openai-compatible"


def test_editing_a_referenced_profile_cannot_invalidate_its_tier_effort():
    _create(
        name="active-openai",
        provider="openai-compatible",
        request_dialect="openai",
        base_url="https://api.openai.com/v1",
    )
    referenced = _create(
        name="referenced-openai",
        provider="openai-compatible",
        request_dialect="openai",
        base_url="https://api.openai.com/v1",
    )
    _route_payload("PUT", "/api/config", {
        "tier_architect_profile_id": referenced["id"],
        "tier_architect_reasoning_effort": "minimal",
    })

    with pytest.raises(ApiError) as raised:
        _route_payload(
            "PATCH",
            f"/api/model-profiles/{referenced['id']}",
            {
                "provider": "anthropic-compatible",
                "request_dialect": "anthropic",
                "base_url": "https://api.anthropic.com",
            },
        )

    assert raised.value.code == "TIER_REASONING_EFFORT_INCOMPATIBLE"
    unchanged = next(
        row for row in _listing("model")["profiles"]
        if row["id"] == referenced["id"]
    )
    assert unchanged["provider"] == "openai-compatible"


def test_activating_a_profile_cannot_invalidate_inherited_tier_effort(
    monkeypatch,
):
    monkeypatch.delenv("PERSONAGRAPH_MODEL_PROVIDER", raising=False)
    _create(
        name="active-openai",
        provider="openai-compatible",
        request_dialect="openai",
        base_url="https://api.openai.com/v1",
    )
    anthropic = _create(
        name="anthropic",
        provider="anthropic-compatible",
        request_dialect="anthropic",
        base_url="https://api.anthropic.com",
    )
    _route_payload("PUT", "/api/config", {
        "tier_architect_profile_id": "-",
        "tier_architect_reasoning_effort": "minimal",
    })

    with pytest.raises(ApiError) as raised:
        _route_payload(
            "POST",
            f"/api/model-profiles/{anthropic['id']}/activate",
            {},
        )

    assert raised.value.code == "TIER_REASONING_EFFORT_INCOMPATIBLE"
    assert _listing("model")["active_id"] != anthropic["id"]


def test_vision_provider_remains_free_text():
    profile = _create(kind="vision", provider="  highland-vl  ")

    assert profile["provider"] == "highland-vl"


def test_a_hand_edited_unknown_model_provider_is_not_usable():
    stored = {
        "profiles": [{
            "id": "mp_bad",
            "kind": "model",
            "name": "手改配置",
            "provider": "gpt-9",
            "request_dialect": "auto",
            "base_url": "https://example.invalid",
            "model": "m",
            "api_key": "sk-bad",
        }],
        "active": {"model": "mp_bad"},
    }
    model_profiles._path().write_text(json.dumps(stored), encoding="utf-8")

    assert _listing("model") == {"active_id": None, "profiles": []}
    assert model_profiles.resolve_profile("mp_bad") is None
    with pytest.raises(ApiError) as raised:
        _route_payload("POST", "/api/model-profiles/mp_bad/activate", {})
    assert raised.value.code == "PROFILE_CORRUPT"


# --- 删除 -----------------------------------------------------------------------


def test_deleting_the_active_one_leaves_nothing_active_rather_than_guessing():
    """静默切换到另一端点，比没有端点更难察觉。"""

    first = _create(name="A")
    _create(name="B")
    _route_payload("DELETE", f"/api/model-profiles/{first['id']}", {})

    listing = _listing("model")
    assert listing["active_id"] is None
    assert [item["name"] for item in listing["profiles"]] == ["B"]


def test_deleting_active_model_profile_disables_its_write_through_endpoint(
    monkeypatch,
):
    """删除配额所有权后，不得让相同凭据继续通过全局端点以无限策略调用。"""

    from personagraph.configuration import app_settings as runtime_config

    monkeypatch.delenv("PERSONAGRAPH_MODEL_PROVIDER", raising=False)
    profile = _create(
        name="quota owner",
        key="sk-owned",
        quota={"requests_per_minute": 10},
    )
    assert runtime_config.model_configured() is True

    _route_payload("DELETE", f"/api/model-profiles/{profile['id']}", {})

    assert runtime_config.active_provider() == "mock"
    assert runtime_config.model_configured() is False


@pytest.mark.parametrize(
    "method, path",
    [
        ("PATCH", "/api/model-profiles/mp_missing"),
        ("DELETE", "/api/model-profiles/mp_missing"),
        ("POST", "/api/model-profiles/mp_missing/activate"),
        ("GET", "/api/model-profiles/mp_missing/secret"),
    ],
)
def test_an_unknown_profile_is_refused_with_a_reason(method, path):
    with pytest.raises(ApiError) as raised:
        _route_payload(method, path, {})
    assert raised.value.code == "PROFILE_NOT_FOUND"


# --- 存储 -----------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["name", "provider", "base_url", "model", "api_key"])
def test_a_profile_without_its_essentials_is_refused(missing):
    payload = {
        "kind": "model", "name": "x", "provider": "anthropic-compatible",
        "base_url": "u", "model": "m", "api_key": "k",
    }
    payload[missing] = ""
    with pytest.raises(ApiError) as raised:
        _route_payload("POST", "/api/model-profiles", payload)
    assert raised.value.code == "PROFILE_FIELD_REQUIRED"


def test_the_file_holding_keys_is_owner_only():
    _create()
    mode = stat.S_IMODE(model_profiles._path().stat().st_mode)
    assert mode == 0o600
    directory_mode = stat.S_IMODE(model_profiles._path().parent.stat().st_mode)
    assert directory_mode == 0o700


def test_a_damaged_file_does_not_take_the_application_down():
    """配置档案只是便利机制；直接配置的内容仍应正常工作。"""

    _create()
    model_profiles._path().write_text("{ broken", encoding="utf-8")
    assert _listing("model")["profiles"] == []
