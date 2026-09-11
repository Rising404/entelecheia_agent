"""FR-5 运行时配置：env 优先、文件兜底、脱敏、model_configured、config API。"""
from __future__ import annotations

import pytest

from personagraph.configuration import app_settings as runtime_config
from personagraph.api import router, service


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_config, "CONFIG_PATH", tmp_path / "app_config.json")
    for var in ("PERSONAGRAPH_MODEL_PROVIDER", "PERSONAGRAPH_API_KEY",
                "PERSONAGRAPH_BASE_URL", "PERSONAGRAPH_MODEL",
                "PERSONAGRAPH_REQUEST_DIALECT",
                "PERSONAGRAPH_MAX_TOKENS",
                "PERSONAGRAPH_LEGACY_OFFICE_SOFFICE"):
        monkeypatch.delenv(var, raising=False)


def _d(method, path, body=None):
    return router.dispatch_response(method, path, body or {}).payload


def test_default_is_mock_not_configured():
    assert runtime_config.get_setting("provider", "mock") == "mock"
    assert runtime_config.model_configured() is False
    assert runtime_config.redacted_view()["request_dialect"] == "auto"
    assert runtime_config.redacted_view()["legacy_office_soffice"] == {
        "configured": False,
        "available": False,
    }


def test_env_wins_over_file():
    runtime_config.update_config({"provider": "deepseek", "api_key": "file-key"})
    import os
    os.environ["PERSONAGRAPH_MODEL_PROVIDER"] = "mock"
    try:
        assert runtime_config.get_setting("provider") == "mock"   # env 覆盖文件
    finally:
        del os.environ["PERSONAGRAPH_MODEL_PROVIDER"]
    assert runtime_config.get_setting("provider") == "deepseek"    # 无 env 时回文件


def test_update_and_model_configured():
    runtime_config.update_config({"provider": "deepseek", "api_key": "sk-abc"})
    assert runtime_config.model_configured() is True


def test_request_dialect_is_writable_and_visible_without_being_secret():
    updated = _d(
        "PUT",
        "/api/config",
        {"provider": "openai-compatible", "request_dialect": "  OpenAI  "},
    )

    assert updated["config"]["request_dialect"] == "openai"
    assert runtime_config.get_setting("request_dialect") == "openai"


def test_openai_config_view_uses_the_same_default_model_as_the_gateway():
    runtime_config.update_config({"provider": "openai-compatible"})

    view = runtime_config.redacted_view()
    assert view["model"] == "gpt-4o-mini"
    assert view["base_url"] == "https://api.openai.com/v1"


def test_config_api_rejects_an_unknown_request_dialect():
    with pytest.raises(service.ApiError) as raised:
        _d("PUT", "/api/config", {"request_dialect": "responses-v9"})

    assert raised.value.code == "INVALID_REQUEST_DIALECT"
    assert runtime_config.get_setting("request_dialect") is None


def test_config_api_rejects_a_dialect_for_the_other_protocol():
    with pytest.raises(service.ApiError) as raised:
        _d(
            "PUT",
            "/api/config",
            {
                "provider": "anthropic-compatible",
                "request_dialect": "openai",
            },
        )

    assert raised.value.code == "INVALID_REQUEST_DIALECT"


def test_partial_legacy_anthropic_switch_cannot_reuse_deepseek_defaults():
    with pytest.raises(service.ApiError) as raised:
        _d("PUT", "/api/config", {"provider": "anthropic"})

    assert raised.value.code == "MODEL_ENDPOINT_INCOMPLETE"
    assert runtime_config.get_setting("provider") is None


def test_complete_legacy_anthropic_switch_uses_its_explicit_endpoint():
    updated = _d(
        "PUT",
        "/api/config",
        {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-test",
        },
    )

    assert updated["config"]["provider"] == "anthropic-compatible"
    assert updated["config"]["request_dialect"] == "anthropic"
    assert updated["config"]["base_url"] == "https://api.anthropic.com"


def test_raw_anthropic_environment_never_inherits_deepseek_defaults(
    monkeypatch,
):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-test")
    monkeypatch.delenv("PERSONAGRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("PERSONAGRAPH_MODEL", raising=False)

    view = runtime_config.redacted_view()

    assert view["provider"] == "anthropic-compatible"
    assert view["base_url"] == "https://api.anthropic.com"
    assert view["model"] == ""
    assert runtime_config.model_configured() is False

    monkeypatch.setenv("PERSONAGRAPH_MODEL", "claude-explicit")
    assert runtime_config.model_configured() is True


def test_protocol_change_cannot_keep_an_explicit_previous_vendor_endpoint():
    runtime_config.update_config(
        {
            "provider": "anthropic-compatible",
            "request_dialect": "deepseek",
            "base_url": "https://api.deepseek.com/anthropic",
            "model": "deepseek-v4-pro",
        }
    )

    with pytest.raises(service.ApiError) as raised:
        _d("PUT", "/api/config", {"provider": "openai-compatible"})

    assert raised.value.code == "MODEL_ENDPOINT_INCOMPLETE"
    assert runtime_config.get_setting("provider") == "anthropic-compatible"


@pytest.mark.parametrize(
    ("provider", "base_url", "request_dialect"),
    (
        ("unknown-provider", "https://example.test", "auto"),
        ("openai-compatible", "https://api.openai.com/v1", "bad-dialect"),
        ("anthropic-compatible", "http://[", "generic"),
    ),
)
def test_status_survives_invalid_environment_model_configuration(
    monkeypatch,
    provider,
    base_url,
    request_dialect,
):
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", provider)
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", base_url)
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")
    monkeypatch.setenv("PERSONAGRAPH_REQUEST_DIALECT", request_dialect)

    status = service.system_status()

    assert status["model_configured"] is False
    assert status["model_control"]["effective"] == "prompt_json"
    assert status["model_control"]["reason"] == "invalid_configuration"


def test_resaving_same_canonical_provider_preserves_explicit_dialect():
    runtime_config.update_config({
        "provider": "anthropic-compatible",
        "request_dialect": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-test",
    })

    updated = _d(
        "PUT",
        "/api/config",
        {"provider": "anthropic-compatible"},
    )

    assert updated["config"]["request_dialect"] == "anthropic"


def test_manually_unknown_provider_is_not_reported_as_configured():
    runtime_config.update_config({"provider": "gpt-9", "api_key": "sk-present"})

    assert runtime_config.model_configured() is False


def test_max_tokens_environment_setting_is_visible():
    import os

    os.environ["PERSONAGRAPH_MAX_TOKENS"] = "2048"
    try:
        assert runtime_config.get_setting("max_tokens") == "2048"
    finally:
        del os.environ["PERSONAGRAPH_MAX_TOKENS"]


def test_empty_field_does_not_clear_existing():
    runtime_config.update_config({"provider": "deepseek", "api_key": "sk-keep"})
    runtime_config.update_config({"provider": "deepseek", "api_key": ""})  # 空=不改
    assert runtime_config.get_setting("api_key") == "sk-keep"


def test_get_config_redacts_key():
    _d("PUT", "/api/config", {"provider": "deepseek", "api_key": "sk-secret"})
    view = _d("GET", "/api/config")
    assert view["config"]["has_key"] is True
    assert "sk-secret" not in str(view)          # 明文绝不外露
    assert view["model_configured"] is True


def test_status_exposes_model_configured():
    st = _d("GET", "/api/status")
    assert "model_configured" in st
    assert st["model_control"]["effective"] in {"native", "prompt_json"}
    assert "api_key" not in str(st["model_control"])


def test_invalid_provider_rejected():
    with pytest.raises(service.ApiError) as ei:
        _d("PUT", "/api/config", {"provider": "gpt-9"})
    assert ei.value.code == "INVALID_PROVIDER"


def test_config_api_accepts_executable_legacy_office_without_disclosing_path(
    tmp_path,
    monkeypatch,
):
    soffice = tmp_path / "soffice"
    soffice.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    soffice.chmod(0o700)
    monkeypatch.setattr(runtime_config.platform, "system", lambda: "Darwin")

    updated = _d(
        "PUT",
        "/api/config",
        {"legacy_office_soffice": str(soffice)},
    )
    status = updated["config"]["legacy_office_soffice"]

    assert status == {"configured": True, "available": True}
    assert str(soffice) not in str(updated)
    assert runtime_config.get_setting("legacy_office_soffice") == str(soffice)
    assert _d("GET", "/api/config")["config"][
        "legacy_office_soffice"
    ] == status


def test_config_api_does_not_mark_legacy_office_available_off_macos(
    tmp_path,
    monkeypatch,
):
    soffice = tmp_path / "soffice"
    soffice.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    soffice.chmod(0o700)
    monkeypatch.setattr(runtime_config.platform, "system", lambda: "Linux")

    updated = _d(
        "PUT",
        "/api/config",
        {"legacy_office_soffice": str(soffice)},
    )

    assert updated["config"]["legacy_office_soffice"] == {
        "configured": True,
        "available": False,
    }


def test_config_api_blank_legacy_office_path_preserves_existing(tmp_path):
    soffice = tmp_path / "soffice"
    soffice.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    soffice.chmod(0o700)
    runtime_config.update_config({"legacy_office_soffice": str(soffice)})

    _d("PUT", "/api/config", {"legacy_office_soffice": "   "})

    assert runtime_config.get_setting("legacy_office_soffice") == str(soffice)


@pytest.mark.parametrize(
    "kind",
    ("relative", "missing", "symlink", "directory", "not_executable", "non_string"),
)
def test_config_api_rejects_unavailable_legacy_office_executable(
    tmp_path,
    kind,
):
    executable = tmp_path / "real-soffice"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    if kind == "relative":
        candidate = "relative/soffice"
    elif kind == "missing":
        candidate = str(tmp_path / "missing-soffice")
    elif kind == "symlink":
        link = tmp_path / "linked-soffice"
        link.symlink_to(executable)
        candidate = str(link)
    elif kind == "directory":
        candidate = str(tmp_path)
    elif kind == "not_executable":
        executable.chmod(0o600)
        candidate = str(executable)
    else:
        candidate = 17

    with pytest.raises(service.ApiError) as exc_info:
        _d(
            "PUT",
            "/api/config",
            {"legacy_office_soffice": candidate},
        )

    assert exc_info.value.code == "INVALID_LEGACY_OFFICE_SOFFICE"
    assert runtime_config.get_setting("legacy_office_soffice") is None


def test_config_view_marks_manually_invalid_legacy_office_as_unavailable(
    tmp_path,
):
    missing = tmp_path / "missing-soffice"
    runtime_config.update_config({"legacy_office_soffice": str(missing)})

    view = runtime_config.redacted_view()

    assert view["legacy_office_soffice"] == {
        "configured": True,
        "available": False,
    }
    assert str(missing) not in str(view)


def test_config_file_permissions(tmp_path):
    runtime_config.update_config({"api_key": "sk-x"})
    import stat
    mode = runtime_config.CONFIG_PATH.stat().st_mode
    assert stat.S_IMODE(mode) == 0o600          # 仅属主可读写
    directory_mode = runtime_config.CONFIG_PATH.parent.stat().st_mode
    assert stat.S_IMODE(directory_mode) == 0o700
