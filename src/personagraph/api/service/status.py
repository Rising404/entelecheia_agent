from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ...configuration.features import load_features
from ...configuration import paths
from ...runtime.entry import run_entry_turn
from ...runtime.entry.routing.policy import (
    available_processing_levels,
    default_turn_routing_policy,
)
from ...session import store as session_store
from .errors import ApiError

def system_status() -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {
        "api": {
            "ok": True,
            "label": "HTTP API",
            "detail": "本地 API 正在响应",
        }
    }

    _check_session_db(components)
    _check_agent_gateway(components)

    from ...configuration import app_settings as runtime_config
    from ...model_io.capabilities import model_control_capability_summary

    features = load_features(None)
    default_routing_policy = default_turn_routing_policy()
    endpoint = runtime_config.resolve_global_model_endpoint()
    provider = endpoint.provider
    base_url = endpoint.base_url
    model = endpoint.model

    return {
        "ok": all(item.get("ok") for item in components.values()),
        "service": "personagraph-api",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "model_configured": runtime_config.model_configured(),
        "model_control": model_control_capability_summary(
            features.get("model_control_transport", "auto"),
        provider=provider,
        base_url=base_url,
        model=model,
        request_dialect=(
            runtime_config.get_setting("request_dialect", "auto") or "auto"
        ),
        ),
        "runtime_events": _runtime_entry_event_health(),
        "runtime_routing": {
            "schema_version": 1,
            "available_processing_levels": list(available_processing_levels()),
            "default_policy": default_routing_policy.to_dict(),
            "default_source": "default",
        },
    }

def _runtime_config_tier_names() -> tuple[str, ...]:
    from ...configuration import app_settings as runtime_config

    return runtime_config.TIER_SETTING_NAMES


def _config_response(config_view: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建配置读写共享的唯一脱敏响应形状。"""

    from ...model_io import tier_bindings as model_tiers
    from ...configuration import app_settings as runtime_config

    return {
        "config": (
            runtime_config.redacted_view() if config_view is None else config_view
        ),
        "model_tiers": model_tiers.redacted_tier_view(),
        "model_configured": runtime_config.model_configured(),
    }


def get_config() -> dict[str, Any]:
    """读当前运行时配置；密钥与私有可执行文件路径不回显。

    ``model_tiers`` 一项给的是**实际生效**的端点，包括某档没选 profile 时
    从全局继承来的那一份。设置页必须能显示这件事：回落是静默发生的，只给
    一个空下拉框会让用户以为该档没在跑（27/002 §7.4）。
    """
    return _config_response()


def update_config(payload: dict[str, Any]) -> dict[str, Any]:
    """写运行时配置；空字段=不改。回脱敏视图。"""
    from ...model_io import endpoint_profiles as model_profiles
    from ...model_io import tier_bindings as model_tiers
    from ...configuration import app_settings as runtime_config
    from personagraph.model_io.dialects import (
        reasoning_effort_is_compatible,
        resolve_request_dialect,
    )
    writable_fields = (
        "provider",
        "request_dialect",
        "base_url",
        "model",
        "api_key",
        "legacy_office_soffice",
        "default_projects_dir",
        "vision_provider",
        "vision_base_url",
        "vision_model",
        "vision_api_key",
        *_TIER_WRITABLE_FIELDS,
    )
    allowed = {k: payload.get(k) for k in writable_fields if k in payload}
    if "default_projects_dir" in allowed:
        from ...configuration.workspace import validate_workspace_directory

        try:
            allowed["default_projects_dir"] = str(validate_workspace_directory(
                allowed["default_projects_dir"], must_exist=False,
            ))
        except ValueError as exc:
            raise ApiError("INVALID_DEFAULT_PROJECTS_DIR", str(exc)) from exc
    if not allowed:
        raise ApiError(
            "NO_CONFIG_FIELDS",
            "无可写配置字段（provider/base_url/model/api_key/vision_*/tier_*/"
            "legacy_office_soffice/default_projects_dir）",
        )
    _normalize_tier_fields(allowed, model_profiles, model_tiers)
    existing_raw_provider = runtime_config.get_setting("provider", "mock")
    provider = allowed.get("provider")
    raw_provider = (
        provider
        if provider is not None and str(provider).strip()
        else runtime_config.get_setting("provider", "mock")
    )
    if provider is not None and str(provider).strip():
        # 归一化之后再校验：旧配置里存的 deepseek/anthropic 仍然是合法输入，
        # 它们指向的本来就是同一套协议适配器。
        normalized = runtime_config.normalize_provider(provider)
        if normalized not in runtime_config.SUPPORTED_PROVIDERS:
            raise ApiError("INVALID_PROVIDER", "provider 取值非法",
                           details={"provider": provider,
                                    "allowed": list(runtime_config.SUPPORTED_PROVIDERS)})
        allowed["provider"] = normalized
    request_dialect = allowed.get("request_dialect")
    if request_dialect is not None and str(request_dialect).strip():
        normalized_dialect = runtime_config.normalize_request_dialect(
            request_dialect
        )
        if normalized_dialect not in runtime_config.REQUEST_DIALECTS:
            raise ApiError(
                "INVALID_REQUEST_DIALECT",
                "request_dialect 取值非法",
                details={
                    "request_dialect": request_dialect,
                    "allowed": list(runtime_config.REQUEST_DIALECTS),
                },
            )
        allowed["request_dialect"] = normalized_dialect
    elif provider is not None and str(provider).strip():
        legacy_name = str(raw_provider or "").strip().lower()
    # 保留旧供应商别名携带的 vendor 信号。只有协议族实际变化时，规范供应商名称才重置为
    # auto。重新保存未变化的规范供应商不能抹掉先前显式选择的 vendor 方言。
        if legacy_name in {"deepseek", "anthropic"}:
            allowed["request_dialect"] = legacy_name
        elif runtime_config.normalize_provider(
            existing_raw_provider
        ) != runtime_config.normalize_provider(provider):
            allowed["request_dialect"] = "auto"

    if "provider" in allowed or "request_dialect" in allowed:
        effective_provider = runtime_config.normalize_provider(allowed.get(
            "provider", runtime_config.get_setting("provider", "mock")
        ))
        effective_base_url = allowed.get(
            "base_url", runtime_config.get_setting("base_url", "")
        )
        effective_dialect = allowed.get(
            "request_dialect",
            runtime_config.get_setting("request_dialect", "auto"),
        )
        try:
            resolve_request_dialect(
                effective_provider,
                effective_base_url,
                effective_dialect,
                legacy_provider=raw_provider,
            )
        except ValueError as exc:
            raise ApiError(
                "INVALID_REQUEST_DIALECT",
                "request_dialect 与 provider 不兼容",
                details={
                    "provider": effective_provider,
                    "request_dialect": effective_dialect,
                },
            ) from exc
    _require_explicit_endpoint_for_nondefault_dialect_transition(
        allowed,
        runtime_config=runtime_config,
        existing_raw_provider=existing_raw_provider,
    )
    _validate_tier_reasoning_compatibility(
        allowed,
        model_profiles=model_profiles,
        model_tiers=model_tiers,
        runtime_config=runtime_config,
        resolve_request_dialect=resolve_request_dialect,
        reasoning_effort_is_compatible=reasoning_effort_is_compatible,
    )
    legacy_office_soffice = allowed.get("legacy_office_soffice")
    if isinstance(legacy_office_soffice, str) and not legacy_office_soffice.strip():
        allowed["legacy_office_soffice"] = ""
    elif legacy_office_soffice is not None:
        try:
            allowed["legacy_office_soffice"] = (
                runtime_config.validate_legacy_office_soffice(
                    legacy_office_soffice,
                )
            )
        except ValueError as exc:
            raise ApiError(
                "INVALID_LEGACY_OFFICE_SOFFICE",
                "legacy Office 可执行文件必须是绝对路径指向的非符号链接可执行普通文件",
            ) from exc
    view = runtime_config.update_config(allowed)
    return _config_response(view)


# 六档各三个键：model profile 指针、思考开关与可移植 effort；实际能力
# 由所选 profile 的 request dialect 再校验。
_TIER_WRITABLE_FIELDS: tuple[str, ...] = tuple(
    key
    for name in _runtime_config_tier_names()
    for key in (
        f"tier_{name}_profile_id",
        f"tier_{name}_thinking",
        f"tier_{name}_reasoning_effort",
    )
)


def _normalize_tier_fields(
    allowed: dict[str, Any], model_profiles: Any, model_tiers: Any
) -> None:
    """Validate分档字段，并把开关写成稳定的 on/off 串。

    profile_id 指向一份不存在的配置时直接拒绝：解析期为了不让一个陈旧指针
    毁掉一整个 Turn 会静默回落到全局端点（见 ``model_tiers.resolve_tier``），
    但那是运行期的容错，不该变成保存设置时的默认行为——用户在设置页选中的
    东西必须真的存在。
    """

    for name in _runtime_config_tier_names():
        profile_key = f"tier_{name}_profile_id"
        if profile_key in allowed:
            raw = str(allowed[profile_key] or "").strip()
            # 空串沿用既有语义：不改该字段。清除本档请传 CLEARED_PROFILE_ID，
            # 它是能真正写进配置文件的值，因此可以覆盖掉先前的选择。
            if raw == model_tiers.CLEARED_PROFILE_ID:
                allowed[profile_key] = raw
            elif raw:
                profile = model_profiles.resolve_profile(raw)
                if profile is None:
                    raise ApiError(
                        "PROFILE_NOT_FOUND",
                        "分档指向的模型配置不存在",
                        details={"field": profile_key, "profile_id": raw},
                    )
                if profile.kind != "model":
                    raise ApiError(
                        "PROFILE_KIND_MISMATCH",
                        "分档只能使用任务模型配置",
                        details={
                            "field": profile_key,
                            "profile_id": raw,
                            "expected_kind": "model",
                            "actual_kind": profile.kind,
                        },
                    )
                allowed[profile_key] = raw
            else:
                allowed[profile_key] = raw

        thinking_key = f"tier_{name}_thinking"
        if thinking_key in allowed:
            value = allowed[thinking_key]
            if isinstance(value, bool):
                allowed[thinking_key] = "on" if value else "off"
            else:
                text = str(value or "").strip().lower()
                if not text:
                    # 与其它字段一致：空表示"不改"。
                    allowed.pop(thinking_key)
                elif text in {"1", "true", "yes", "on"}:
                    allowed[thinking_key] = "on"
                elif text in {"0", "false", "no", "off"}:
                    allowed[thinking_key] = "off"
                else:
                    raise ApiError(
                        "INVALID_TIER_THINKING",
                        "分档思考开关只能是布尔值或 on/off",
                        details={"field": thinking_key},
                    )

        reasoning_key = f"tier_{name}_reasoning_effort"
        if reasoning_key in allowed:
            text = str(allowed[reasoning_key] or "").strip().lower()
            if not text:
                allowed.pop(reasoning_key)
            elif text == "auto":
                allowed[reasoning_key] = text
            else:
                try:
                    allowed[reasoning_key] = model_tiers.ReasoningEffort(text).value
                except ValueError as exc:
                    raise ApiError(
                        "INVALID_REASONING_EFFORT",
                        "分档推理强度取值非法",
                        details={
                            "field": reasoning_key,
                            "allowed": [
                                item.value for item in model_tiers.ReasoningEffort
                            ],
                        },
                    ) from exc


def _validate_tier_reasoning_compatibility(
    allowed: dict[str, Any],
    *,
    model_profiles: Any,
    model_tiers: Any,
    runtime_config: Any,
    resolve_request_dialect: Any,
    reasoning_effort_is_compatible: Any,
) -> None:
    """拒绝所选端点无法解释的已存推理强度。

    tier 控制比 profile 选择存活更久。若无此规范化后检查，把携带 ``minimal`` 或 ``none``
    的 OpenAI tier 切换到原生 Anthropic profile，会产生被供应商拒绝，或与可见思考开关
    矛盾的请求。
    """

    global_endpoint_changed = bool(
        {"provider", "request_dialect", "base_url"} & allowed.keys()
    )
    for name in _runtime_config_tier_names():
        profile_key = f"tier_{name}_profile_id"
        reasoning_key = f"tier_{name}_reasoning_effort"
        profile_changed = profile_key in allowed
        effort_changed = reasoning_key in allowed

        profile_id = str(
            allowed.get(
                profile_key,
                runtime_config.get_setting(profile_key, ""),
            )
            or ""
        ).strip()
        inherits_global = profile_id in {"", model_tiers.CLEARED_PROFILE_ID}
        if not (
            profile_changed
            or effort_changed
            or (global_endpoint_changed and inherits_global)
        ):
            continue

        if inherits_global:
            raw_provider = allowed.get(
                "provider", runtime_config.get_setting("provider", "mock")
            )
            provider = runtime_config.normalize_provider(raw_provider)
            base_url = allowed.get(
                "base_url", runtime_config.get_setting("base_url", "")
            )
            configured_dialect = allowed.get(
                "request_dialect",
                runtime_config.get_setting("request_dialect", "auto"),
            )
            legacy_provider = raw_provider
        else:
            profile = model_profiles.resolve_profile(profile_id)
        # 预先存在的过期指针会在 Runtime 回退；本请求中的 profile 变化已在上方拒绝。
            if profile is None or profile.kind != "model":
                continue
            provider = runtime_config.normalize_provider(profile.provider)
            base_url = profile.base_url
            configured_dialect = profile.request_dialect
            legacy_provider = profile.provider

        dialect = resolve_request_dialect(
            provider,
            base_url,
            configured_dialect,
            legacy_provider=legacy_provider,
        )
        effort = allowed.get(
            reasoning_key,
            runtime_config.get_setting(reasoning_key, "auto"),
        )
        effort = str(effort or "auto").strip().lower()
        if effort == "auto":
            effort = None
        if reasoning_effort_is_compatible(dialect, effort):
            continue

        from personagraph.model_io.dialects import reasoning_capability

        _, options = reasoning_capability(dialect)
        raise ApiError(
            "TIER_REASONING_EFFORT_INCOMPATIBLE",
            "当前模型接口不支持该分档推理强度，请改为自动或选择兼容档位",
            details={
                "tier": name,
                "request_dialect": dialect.value,
                "reasoning_effort": effort,
                "allowed": ["auto", *options],
            },
        )


def _require_explicit_endpoint_for_nondefault_dialect_transition(
    allowed: dict[str, Any],
    *,
    runtime_config: Any,
    existing_raw_provider: Any,
) -> None:
    """不要把新选择的 vendor 方言与另一 vendor 的默认值组合。"""

    if "provider" not in allowed and "request_dialect" not in allowed:
        return
    provider = runtime_config.normalize_provider(
        allowed.get("provider", existing_raw_provider)
    )
    dialect = str(
        allowed.get(
            "request_dialect",
            runtime_config.get_setting("request_dialect", "auto"),
        )
        or "auto"
    ).strip().lower()
    previous_dialect = str(
        runtime_config.get_setting("request_dialect", "auto") or "auto"
    ).strip().lower()
    changed = (
        provider != runtime_config.normalize_provider(existing_raw_provider)
        or dialect != previous_dialect
    )
    protocol_changed = (
        provider != runtime_config.normalize_provider(existing_raw_provider)
    )
    if protocol_changed and any(
        str(runtime_config.get_setting(field, "") or "").strip()
        for field in ("base_url", "model")
    ):
    # 部分 PATCH 会与现有文件合并。具体端点保存后，若只更改协议，会保留旧 vendor 的
    # URL/model，并向错误路由生成看似合理的请求。因此这种情况要求完整替换。
        missing = [
            field
            for field in ("base_url", "model")
            if not str(allowed.get(field) or "").strip()
        ]
        if missing:
            raise ApiError(
                "MODEL_ENDPOINT_INCOMPLETE",
                "切换模型接口类型时必须同时明确填写接口地址和模型名",
                details={
                    "provider": provider,
                    "request_dialect": dialect,
                    "required": missing,
                },
            )
    protocol_default_choices = {
        "anthropic-compatible": {"auto", "deepseek"},
        "openai-compatible": {"auto", "openai"},
        "mock": {"auto"},
    }.get(provider, {"auto"})
    if not changed or dialect in protocol_default_choices:
        return
    missing = [
        field
        for field in ("base_url", "model")
        if not str(allowed.get(field) or "").strip()
    ]
    if missing:
        raise ApiError(
            "MODEL_ENDPOINT_INCOMPLETE",
            "切换厂商请求格式时必须同时明确填写接口地址和模型名",
            details={
                "provider": provider,
                "request_dialect": dialect,
                "required": missing,
            },
        )


def _check_session_db(components: dict[str, dict[str, Any]]) -> None:
    try:
        if not session_store.uses_partitioned_storage():
            session_store.init_db()
        active = len(session_store.list_sessions(status="active"))
        archived = len(session_store.list_sessions(status="archived"))
        trashed = len(session_store.list_sessions(status="trashed"))
        partitioned = session_store.uses_partitioned_storage()
        components["session_db"] = {
            "ok": True,
            "label": "Session Storage",
            "detail": (
                "project catalog + per-session session.sqlite 可读写"
                if partitioned
                else "显式 sessions.sqlite override 可读写"
            ),
            "path": str(
                paths.PROJECT_CATALOG_DB_PATH
                if partitioned
                else session_store.DB_PATH
            ),
            "counts": {
                "active": active,
                "archived": archived,
                "trashed": trashed,
            },
        }
    except Exception as exc:  # pragma: no cover - 防御性诊断
        partitioned = session_store.uses_partitioned_storage()
        components["session_db"] = {
            "ok": False,
            "label": "Session Storage",
            "detail": "session storage 不可用",
            "error": str(exc),
            "path": str(
                paths.PROJECT_CATALOG_DB_PATH
                if partitioned
                else session_store.DB_PATH
            ),
        }

def _check_agent_gateway(components: dict[str, dict[str, Any]]) -> None:
    try:
        features = load_features(None)
        components["agent"] = {
            "ok": callable(run_entry_turn),
            "label": "Runtime Entry",
            "detail": "runtime.entry.run_entry_turn 可调用",
            "run_entry_turn": callable(run_entry_turn),
            "features": sorted(features.keys()),
        }
    except Exception as exc:  # pragma: no cover - 防御性诊断
        components["agent"] = {
            "ok": False,
            "label": "Agent Gateway",
            "detail": "runtime 网关不可用",
            "error": str(exc),
        }


def _runtime_entry_event_health() -> dict[str, Any]:
    """报告新的 Runtime 事件 store，而非已退役 journal 标志。"""
    try:
        partitioned = session_store.uses_partitioned_storage()
        if not partitioned:
            session_store.init_db()
        else:
        # 列表操作会初始化/验证 catalog，但不创建共享 Session 数据库。Runtime 事件位于各
        # 现有 Session 自有数据库中。
            session_store.list_sessions()
        return {
            "enabled": True,
            "database_present": (
                paths.PROJECT_CATALOG_DB_PATH.exists()
                if partitioned
                else session_store.DB_PATH.exists()
            ),
            "degraded": False,
        }
    except Exception as exc:  # pragma: no cover - 防御性诊断
        partitioned = session_store.uses_partitioned_storage()
        return {
            "enabled": True,
            "database_present": (
                paths.PROJECT_CATALOG_DB_PATH.exists()
                if partitioned
                else session_store.DB_PATH.exists()
            ),
            "degraded": True,
            "error": str(exc),
        }


def _profile_call(action, *args, **kwargs) -> dict[str, Any]:
    """将 store 的类型化拒绝转换为 API 错误形状。"""
    from personagraph.model_io.endpoint_profiles import ProfileError
    try:
        return action(*args, **kwargs)
    except ProfileError as exc:
        raise ApiError(exc.code, str(exc), status=400) from exc


def list_model_profiles(query: dict[str, Any] | None = None) -> dict[str, Any]:
    from ...model_io import endpoint_profiles as model_profiles
    kind = (query or {}).get("kind")
    kind = kind if kind in model_profiles.KINDS else None
    return {"profiles": model_profiles.list_profiles(kind)}


def create_model_profile(payload: dict[str, Any]) -> dict[str, Any]:
    from ...model_io import endpoint_profiles as model_profiles
    return {"profile": _profile_call(
        model_profiles.create_profile,
        kind=str(payload.get("kind") or ""),
        name=payload.get("name"),
        provider=payload.get("provider"),
        request_dialect=payload.get("request_dialect"),
        base_url=payload.get("base_url"),
        model=payload.get("model"),
        api_key=payload.get("api_key"),
        quota=payload.get("quota"),
        precommit_validator=_validate_model_profile_tier_reasoning,
    )}


def update_model_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    from ...model_io import endpoint_profiles as model_profiles
    endpoint_changes = {"provider", "request_dialect", "base_url"} & payload.keys()
    return {"profile": _profile_call(
        model_profiles.update_profile,
        profile_id,
        payload,
        precommit_validator=(
            _validate_model_profile_tier_reasoning if endpoint_changes else None
        ),
    )}


def delete_model_profile(profile_id: str) -> dict[str, Any]:
    from ...model_io import endpoint_profiles as model_profiles
    return _profile_call(model_profiles.delete_profile, profile_id)


def activate_model_profile(profile_id: str) -> dict[str, Any]:
    from ...model_io import endpoint_profiles as model_profiles
    return {"profile": _profile_call(
        model_profiles.activate_profile,
        profile_id,
        precommit_validator=_validate_model_profile_tier_reasoning,
    )}


def _validate_model_profile_tier_reasoning(
    profile: Any,
    activates_global: bool,
) -> None:
    """防止 profile 编辑/激活使依赖的 tier 控制失效。"""

    if getattr(profile, "kind", None) != "model":
        return
    from ...model_io import tier_bindings as model_tiers
    from ...configuration import app_settings as runtime_config
    from personagraph.model_io.dialects import (
        reasoning_capability,
        reasoning_effort_is_compatible,
        resolve_request_dialect,
    )

    dialect = resolve_request_dialect(
        runtime_config.normalize_provider(profile.provider),
        profile.base_url,
        profile.request_dialect,
        legacy_provider=profile.provider,
    )
    for name in _runtime_config_tier_names():
        pointer = str(
            runtime_config.get_setting(f"tier_{name}_profile_id", "") or ""
        ).strip()
        directly_referenced = pointer == profile.id
        inherits_global = pointer in {"", model_tiers.CLEARED_PROFILE_ID}
        if not directly_referenced and not (activates_global and inherits_global):
            continue
        effort = str(
            runtime_config.get_setting(
                f"tier_{name}_reasoning_effort", "auto"
            )
            or "auto"
        ).strip().lower()
        if effort == "auto":
            effort = None
        if reasoning_effort_is_compatible(dialect, effort):
            continue
        _, options = reasoning_capability(dialect)
        raise ApiError(
            "TIER_REASONING_EFFORT_INCOMPATIBLE",
            "该模型配置被分档使用，修改后无法解释现有推理强度",
            details={
                "tier": name,
                "profile_id": profile.id,
                "request_dialect": dialect.value,
                "reasoning_effort": effort,
                "allowed": ["auto", *options],
            },
        )


def reveal_model_profile_secret(profile_id: str) -> dict[str, Any]:
    """刻意使用独立路由：列表绝不能携带密钥。"""
    from ...model_io import endpoint_profiles as model_profiles
    return _profile_call(model_profiles.reveal_secret, profile_id)
