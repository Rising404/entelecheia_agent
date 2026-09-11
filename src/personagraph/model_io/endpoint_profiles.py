"""用户可以保存、切换和编辑的具名端点配置。

这里使用两个独立集合，而不是一个带用途标签的集合：对话模型与图像模型的选择原因不同、
切换时机也不同；若共用一份列表，每次切换都必须先过滤。

激活 profile 会把它的值写入现有 Runtime 配置，因此每个消费者继续按原方式读取配置。
本文件是管理层，绝不是当前生效配置的第二事实来源。

密钥存储在这里，因为另一选择是浏览器 local storage；后者没有文件权限保护，renderer 中
任何脚本都能读取。列表绝不会包含密钥；回读密钥需要单独显式调用，因此普通设置页加载
完全不会携带密钥。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..configuration.paths import LOCAL_CONFIG_DIR as _LOCAL_CONFIG_DIR
from .dialects import resolve_request_dialect


ProfileKind = Literal["model", "vision"]
ProfilePrecommitValidator = Callable[["Profile", bool], None]
KINDS: tuple[ProfileKind, ...] = ("model", "vision")

# 每种类型激活时会写入哪些 Runtime 配置键。
_WRITE_THROUGH: dict[ProfileKind, dict[str, str]] = {
    "model": {
        "provider": "provider",
        "request_dialect": "request_dialect",
        "base_url": "base_url",
        "model": "model",
        "api_key": "api_key",
    },
    "vision": {
        "provider": "vision_provider",
        "base_url": "vision_base_url",
        "model": "vision_model",
        "api_key": "vision_api_key",
    },
}

_MAX_NAME = 80
_MAX_VALUE = 500
_MAX_QUOTA_GROUP = 128
_MAX_QUOTA_VALUE = (1 << 63) - 1
_QUOTA_INTEGER_FIELDS = (
    "requests_per_minute",
    "tokens_per_minute",
    "tokens_per_week",
    "max_in_flight",
)
_QUOTA_FIELDS = frozenset((*_QUOTA_INTEGER_FIELDS, "quota_group"))
_QUOTA_GROUP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProfileError(ValueError):
    """无法满足的 profile 请求，并携带稳定原因。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ModelProfileQuota:
    """所有使用某 profile 的 tier 共享的可选供应商账户限制。

    ``None`` 表示应用未配置该维度的限制，并不表示容量为零。dispatcher 可以根据端点和
    凭据材料派生私有的默认共享身份；如果账户限制跨多个凭据或 profile 记录，
    ``quota_group`` 可作为可选、非密钥的覆盖值。
    """

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_week: int | None = None
    max_in_flight: int | None = None
    quota_group: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_week": self.tokens_per_week,
            "max_in_flight": self.max_in_flight,
            "quota_group": self.quota_group,
        }


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    kind: ProfileKind
    name: str
    provider: str
    base_url: str
    model: str
    request_dialect: str = "auto"
    api_key: str = field(default="", repr=False)
    quota: ModelProfileQuota = field(default_factory=ModelProfileQuota)

    def redacted(self, *, active: bool) -> dict[str, Any]:
        """列表返回的数据形状，绝不携带密钥本身。"""

        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "provider": self.provider,
            "request_dialect": self.request_dialect,
            "base_url": self.base_url,
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "active": active,
        }
        if self.kind == "model":
            result["quota"] = self.quota.to_dict()
        return result


# 配置句柄可在测试中重定向，避免用例读写开发者的本地端点配置。
# 这是用户管理的 JSON 配置，不是运行时数据库。
CONFIG_PATH = _LOCAL_CONFIG_DIR / "model_profiles.json"


def _path() -> Path:
    return Path(CONFIG_PATH)


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"profiles": [], "active": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 损坏文件不能导致应用无法启动。profile 只是便利层；Runtime 配置中当前内容仍应可用。
        return {"profiles": [], "active": {}}
    profiles = data.get("profiles")
    active = data.get("active")
    return {
        "profiles": profiles if isinstance(profiles, list) else [],
        "active": active if isinstance(active, dict) else {},
    }


def _save(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
        os.chmod(path, 0o600)  # 含明文 key，仅属主可读写
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _row_to_profile(row: dict[str, Any]) -> Profile | None:
    try:
        kind = row["kind"]
        if kind not in KINDS:
            return None
        raw_provider = row.get("provider")
        provider = _normalize_profile_provider(kind, raw_provider)
        base_url = str(row.get("base_url") or "")
        return Profile(
            id=str(row["id"]),
            kind=kind,
            name=str(row.get("name") or ""),
            provider=provider,
            base_url=base_url,
            model=str(row.get("model") or ""),
            request_dialect=_normalize_profile_request_dialect(
                kind=kind,
                provider=provider,
                raw_provider=raw_provider,
                base_url=base_url,
                value=row.get("request_dialect"),
            ),
            api_key=str(row.get("api_key") or ""),
            quota=_normalize_quota(row.get("quota"), kind=kind),
        )
    except (KeyError, TypeError, ProfileError):
        return None


def _require_text(field_name: str, value: object, *, limit: int = _MAX_VALUE) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProfileError("PROFILE_FIELD_REQUIRED", f"{field_name} 不能为空")
    if len(text) > limit:
        raise ProfileError("PROFILE_FIELD_TOO_LONG", f"{field_name} 过长")
    return text


def _normalize_quota(
    value: object | None,
    *,
    kind: object,
    existing: ModelProfileQuota | None = None,
    partial: bool = False,
) -> ModelProfileQuota:
    """验证配额输入，并在需要时按 PATCH 语义合并字段。"""

    if value is None:
        return ModelProfileQuota()
    if not isinstance(value, dict):
        raise ProfileError("PROFILE_QUOTA_INVALID", "quota 必须是 JSON object")
    unknown = sorted(set(value) - _QUOTA_FIELDS)
    if unknown:
        raise ProfileError(
            "PROFILE_QUOTA_FIELD_UNKNOWN",
            f"quota 包含未知字段：{', '.join(unknown)}",
        )
    if kind != "model" and value:
        raise ProfileError(
            "PROFILE_QUOTA_KIND_UNSUPPORTED",
            "配额仅适用于任务模型配置",
        )

    base = existing if partial and existing is not None else ModelProfileQuota()
    normalized = base.to_dict()
    for name in _QUOTA_INTEGER_FIELDS:
        if name not in value:
            continue
        raw = value[name]
        if raw is None:
            normalized[name] = None
            continue
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw <= 0
            or raw > _MAX_QUOTA_VALUE
        ):
            raise ProfileError(
                "PROFILE_QUOTA_VALUE_INVALID",
                f"quota.{name} 必须是正整数或 null",
            )
        normalized[name] = raw

    if "quota_group" in value:
        raw_group = value["quota_group"]
        if raw_group is None:
            normalized["quota_group"] = None
        elif not isinstance(raw_group, str):
            raise ProfileError(
                "PROFILE_QUOTA_GROUP_INVALID",
                "quota.quota_group 必须是字符串或 null",
            )
        else:
            group = raw_group.strip()
            if (
                len(group) > _MAX_QUOTA_GROUP
                or _QUOTA_GROUP_PATTERN.fullmatch(group) is None
            ):
                raise ProfileError(
                    "PROFILE_QUOTA_GROUP_INVALID",
                    "quota.quota_group 只能包含字母、数字、点、下划线、冒号和连字符",
                )
            normalized["quota_group"] = group

    return ModelProfileQuota(
        requests_per_minute=normalized["requests_per_minute"],  # type: ignore[arg-type]
        tokens_per_minute=normalized["tokens_per_minute"],  # type: ignore[arg-type]
        tokens_per_week=normalized["tokens_per_week"],  # type: ignore[arg-type]
        max_in_flight=normalized["max_in_flight"],  # type: ignore[arg-type]
        quota_group=normalized["quota_group"],  # type: ignore[arg-type]
    )


def normalize_model_profile_quota(value: object | None) -> ModelProfileQuota:
    """将外部非密钥投影归一化为任务模型配额。

    Profile 文件与隔离 worker 的环境快照必须共享同一套字段和值域校验；后者不应为了
    不读取 GUI 配置而复制一份近似规则。
    """

    return _normalize_quota(value, kind="model")


def _quota_limit_identity(quota: ModelProfileQuota) -> tuple[int | None, ...]:
    """仅返回由同一共享供应商范围管理的限制。

    ``quota_group`` 用于选择范围，本身不是限制。明确保留这一区别，可避免仅因组标签一致
    就把同一账户的两个 profile 误判为兼容。
    """

    return (
        quota.requests_per_minute,
        quota.tokens_per_minute,
        quota.tokens_per_week,
        quota.max_in_flight,
    )


def _quota_scope_hash(profile: Profile) -> str | None:
    """在不保留 API key 的前提下派生私有共享身份。

    历史 profile 文件曾允许包含非绝对端点字符串，手工编辑时也可能包含空密钥。配额队列会
    正确拒绝此类输入，但 profile CRUD 仍须能读取和编辑这些旧记录。因此，在端点/凭据变得
    可用之前，它们不参与此一致性检查。
    """

    from .api_quota_queue import derive_quota_scope_hash

    try:
        return derive_quota_scope_hash(
            base_url=profile.base_url,
            credential=profile.api_key,
            quota_group=profile.quota.quota_group,
        )
    except ValueError:
        return None


def _validate_quota_scope_consistency(
    profile: Profile,
    stored_rows: list[dict[str, Any]],
    *,
    exclude_profile_id: str | None = None,
) -> None:
    """拒绝对同一共享配额定义不一致的两个模型 profile。

    检查发生在持久化之前。错误文本刻意不包含 profile 凭据或由凭据派生的指纹。
    """

    if profile.kind != "model":
        return
    scope_hash = _quota_scope_hash(profile)
    if scope_hash is None:
        return
    limits = _quota_limit_identity(profile.quota)
    for row in stored_rows:
        if str(row.get("id") or "") == exclude_profile_id:
            continue
        existing = _row_to_profile(row)
        if existing is None or existing.kind != "model":
            continue
        if _quota_scope_hash(existing) != scope_hash:
            continue
        if _quota_limit_identity(existing.quota) != limits:
            raise ProfileError(
                "PROFILE_QUOTA_SCOPE_CONFLICT",
                "共享同一 API 配额的模型配置必须使用完全相同的配额限制",
            )


def _normalize_profile_provider(kind: object, value: object) -> str:
    """验证模型协议，同时允许扩展视觉适配器名称。"""

    provider = _require_text("provider", value)
    if kind != "model":
        return provider

    from ..configuration.app_settings import SUPPORTED_PROVIDERS, normalize_provider

    normalized = normalize_provider(provider)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ProfileError(
            "PROFILE_PROVIDER_UNKNOWN",
            "任务模型 provider 只能是 anthropic-compatible、openai-compatible 或 mock",
        )
    return normalized


def _normalize_request_dialect(value: object) -> str:
    from ..configuration.app_settings import REQUEST_DIALECTS, normalize_request_dialect

    normalized = normalize_request_dialect(value)
    if normalized not in REQUEST_DIALECTS:
        raise ProfileError(
            "PROFILE_REQUEST_DIALECT_UNKNOWN",
            "request_dialect 只能是 auto、deepseek、openai、anthropic 或 generic",
        )
    return normalized


def _normalize_profile_request_dialect(
    *,
    kind: object,
    provider: str,
    raw_provider: object,
    base_url: str,
    value: object | None,
) -> str:
    """验证一个已存选项，并保留旧供应商名称。"""

    raw_dialect = str(value or "").strip()
    legacy_name = str(raw_provider or "").strip().lower()
    inferred = (
        legacy_name
        if not raw_dialect and legacy_name in {"deepseek", "anthropic"}
        else raw_dialect or "auto"
    )
    normalized = _normalize_request_dialect(inferred)
    if kind != "model":
        return normalized
    try:
        resolve_request_dialect(
            provider,
            base_url,
            normalized,
            legacy_provider=raw_provider,
        )
    except ValueError as exc:
        raise ProfileError(
            "PROFILE_REQUEST_DIALECT_INCOMPATIBLE",
            "request_dialect 与任务模型 provider 不兼容",
        ) from exc
    return normalized


def list_profiles(kind: ProfileKind | None = None) -> dict[str, Any]:
    """按类型分组返回所有已存 profile，并标记活动项。"""

    data = _load()
    stored = [item for item in (_row_to_profile(row) for row in data["profiles"]) if item]
    result: dict[str, Any] = {}
    for candidate in KINDS:
        if kind is not None and candidate != kind:
            continue
        requested_active_id = data["active"].get(candidate)
        active_id = (
            requested_active_id
            if any(
                item.kind == candidate and item.id == requested_active_id
                for item in stored
            )
            else None
        )
        result[candidate] = {
            "active_id": active_id,
            "profiles": [
                item.redacted(active=item.id == active_id)
                for item in stored
                if item.kind == candidate
            ],
        }
    return result


def create_profile(
    *, kind: str, name: str, provider: str, base_url: str, model: str,
    api_key: str, request_dialect: str | None = None,
    quota: object | None = None,
    precommit_validator: ProfilePrecommitValidator | None = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ProfileError("PROFILE_KIND_UNKNOWN", "配置类型只能是 model 或 vision")
    normalized_provider = _normalize_profile_provider(kind, provider)
    normalized_base_url = _require_text("base_url", base_url)
    profile = Profile(
        id=f"mp_{uuid.uuid4().hex[:12]}",
        kind=kind,  # type: ignore[arg-type]
        name=_require_text("名称", name, limit=_MAX_NAME),
        provider=normalized_provider,
        base_url=normalized_base_url,
        model=_require_text("model", model),
        request_dialect=_normalize_profile_request_dialect(
            kind=kind,
            provider=normalized_provider,
            raw_provider=provider,
            base_url=normalized_base_url,
            value=request_dialect,
        ),
        api_key=_require_text("api_key", api_key),
        quota=_normalize_quota(quota, kind=kind),
    )
    data = _load()
    will_activate = not data["active"].get(profile.kind)
    _validate_quota_scope_consistency(profile, data["profiles"])
    if precommit_validator is not None:
        precommit_validator(profile, will_activate)
    stored_profile = {
        "id": profile.id, "kind": profile.kind, "name": profile.name,
        "provider": profile.provider, "base_url": profile.base_url,
        "model": profile.model, "request_dialect": profile.request_dialect,
        "api_key": profile.api_key
    }
    if profile.kind == "model":
        stored_profile["quota"] = profile.quota.to_dict()
    data["profiles"].append(stored_profile)
    # 第一份自动启用：存了却没生效，是个没人想要的中间状态。
    if will_activate:
        data["active"][profile.kind] = profile.id
        _save(data)
        _write_through(profile)
        return profile.redacted(active=True)
    _save(data)
    return profile.redacted(active=False)


def update_profile(
    profile_id: str,
    patch: dict[str, Any],
    *,
    precommit_validator: ProfilePrecommitValidator | None = None,
) -> dict[str, Any]:
    data = _load()
    for row in data["profiles"]:
        if str(row.get("id")) != profile_id:
            continue
        existing = _row_to_profile(row)
        if existing is None:
            raise ProfileError("PROFILE_CORRUPT", "这份配置已损坏")
        candidate = dict(row)
        for key, limit in (
            ("name", _MAX_NAME), ("base_url", _MAX_VALUE),
            ("model", _MAX_VALUE),
        ):
            if key in patch and str(patch[key] or "").strip():
                candidate[key] = _require_text(key, patch[key], limit=limit)

        provider_supplied = bool(
            "provider" in patch and str(patch["provider"] or "").strip()
        )
        raw_provider = (
            patch["provider"] if provider_supplied else candidate.get("provider")
        )
        normalized_provider = _normalize_profile_provider(
            candidate.get("kind"), raw_provider
        )
        provider_changed = normalized_provider != existing.provider
        candidate["provider"] = normalized_provider

        base_url_changed = (
            str(candidate.get("base_url") or "") != existing.base_url
        )
        legacy_provider_hint = str(raw_provider or "").strip().lower()
        legacy_provider_hint_changed = (
            candidate.get("kind") == "model"
            and provider_supplied
            and legacy_provider_hint in {"deepseek", "anthropic"}
            and legacy_provider_hint != existing.request_dialect
        )

        dialect_changed = bool(
            "request_dialect" in patch
            and str(patch["request_dialect"] or "").strip()
        )
        if (
            provider_changed
            or base_url_changed
            or legacy_provider_hint_changed
            or dialect_changed
        ):
            candidate["request_dialect"] = _normalize_profile_request_dialect(
                kind=candidate.get("kind"),
                provider=str(candidate.get("provider") or ""),
                raw_provider=raw_provider,
                base_url=str(candidate.get("base_url") or ""),
                value=patch["request_dialect"] if dialect_changed else None,
            )
        else:
            # PATCH 省略字段时保留显式已存选项。仅重复同一供应商（包括大小写/空白变化），
            # 不能把已选择具体供应商方言的代理重新变为 ``auto``。
            candidate["request_dialect"] = existing.request_dialect

        # 空的 key 表示"不改"，不是"清空"——否则保存一次表单就会把 key 抹掉。
        if str(patch.get("api_key") or "").strip():
            candidate["api_key"] = _require_text("api_key", patch["api_key"])

        if "quota" in patch:
            candidate["quota"] = _normalize_quota(
                patch["quota"],
                kind=candidate.get("kind"),
                existing=existing.quota,
                partial=patch["quota"] is not None,
            ).to_dict()

        profile = _row_to_profile(candidate)
        if profile is None:
            raise ProfileError("PROFILE_CORRUPT", "这份配置已损坏")
        active = data["active"].get(profile.kind) == profile.id
        _validate_quota_scope_consistency(
            profile,
            data["profiles"],
            exclude_profile_id=profile.id,
        )
        if precommit_validator is not None:
            precommit_validator(profile, active)
        # 读取旧记录时可能规范化其协议并推断供应商方言；主动编辑该记录时同时持久化两者。
        candidate["provider"] = profile.provider
        candidate["request_dialect"] = profile.request_dialect
        if profile.kind == "model":
            candidate["quota"] = profile.quota.to_dict()
        row.clear()
        row.update(candidate)
        _save(data)
        if active:
            _write_through(profile)
        return profile.redacted(active=active)
    raise ProfileError("PROFILE_NOT_FOUND", "没有这份配置")


def delete_profile(profile_id: str) -> dict[str, Any]:
    data = _load()
    remaining = [row for row in data["profiles"] if str(row.get("id")) != profile_id]
    if len(remaining) == len(data["profiles"]):
        raise ProfileError("PROFILE_NOT_FOUND", "没有这份配置")
    data["profiles"] = remaining
    disabled_global_model = False
    # 删掉正在用的那份，就没有"当前配置"了。让指针空着而不是随便选一份顶上：
    # 悄悄换一个端点比没有端点更难察觉。
    for kind, active_id in list(data["active"].items()):
        if active_id == profile_id:
            data["active"].pop(kind, None)
            disabled_global_model = kind == "model"
    if disabled_global_model:
        # 激活曾把此 profile 的凭据复制到旧全局端点。删除其配额所有者后若仍让该端点生效，
        # 会把同一账户变成无限额账户。保留端点字段供日后主动重新配置，但现在通过将有效
        # 供应商切换为本地 mock 来保守失败。
        from ..configuration.app_settings import update_config

        update_config({"provider": "mock"})
    # 先禁用可调用端点。如果任一原子文件写入失败，唯一可能的局部状态仍是安全状态：
    # profile 可能保持活动，但外部模型分派已经禁用。
    _save(data)
    return {"deleted": True, "id": profile_id}


def activate_profile(
    profile_id: str,
    *,
    precommit_validator: ProfilePrecommitValidator | None = None,
) -> dict[str, Any]:
    data = _load()
    for row in data["profiles"]:
        if str(row.get("id")) != profile_id:
            continue
        profile = _row_to_profile(row)
        if profile is None:
            raise ProfileError("PROFILE_CORRUPT", "这份配置已损坏")
        if precommit_validator is not None:
            precommit_validator(profile, True)
        data["active"][profile.kind] = profile.id
        _save(data)
        _write_through(profile)
        return profile.redacted(active=True)
    raise ProfileError("PROFILE_NOT_FOUND", "没有这份配置")


def resolve_profile(profile_id: str) -> Profile | None:
    """为需要端点的调用方完整读取一个已存 profile。

    刻意不经由 :func:`activate_profile`：激活会把 profile 写入唯一的全局 Runtime 配置，
    而 per-tier 选择恰恰不能这样做——六个 tier 必须共存，而不能相互覆盖。未知 id 返回
    ``None``，使调用方可回退到全局端点，而不会因过期指针使 Turn 失败。
    """

    if not isinstance(profile_id, str) or not profile_id.strip():
        return None
    for row in _load()["profiles"]:
        if str(row.get("id")) == profile_id:
            return _row_to_profile(row)
    return None


def resolve_active_profile(kind: str) -> Profile | None:
    """返回某集合中完整的活动 profile，包括其密钥。

    这是内部 Runtime 查询，不是 API 列表形状。未知类型、过期活动指针和损坏记录都会解析为
    ``None``，使调用方能明确使用其现有全局端点回退路径。
    """

    if kind not in KINDS:
        return None
    data = _load()
    active_id = data["active"].get(kind)
    if not isinstance(active_id, str) or not active_id:
        return None
    for row in data["profiles"]:
        if str(row.get("id")) != active_id:
            continue
        profile = _row_to_profile(row)
        return profile if profile is not None and profile.kind == kind else None
    return None


def profile_exists(profile_id: str) -> bool:
    """在不读取密钥的情况下判断 id 是否指向已存 profile。"""

    return resolve_profile(profile_id) is not None


def reveal_secret(profile_id: str) -> dict[str, Any]:
    """仅在显式请求时，以明文返回一个已存密钥。

    密钥被刻意排除在所有列表之外。向机器所有者隐藏密钥没有任何保护作用——他们可以直接
    读取文件；但让密钥随每次设置加载传递，会使它出现在无人主动选择暴露的截图、屏幕共享
    和 renderer 内存中。
    """

    for row in _load()["profiles"]:
        if str(row.get("id")) == profile_id:
            return {"id": profile_id, "api_key": str(row.get("api_key") or "")}
    raise ProfileError("PROFILE_NOT_FOUND", "没有这份配置")


def _write_through(profile: Profile) -> None:
    """将已激活 profile 写入所有消费者读取的 Runtime 配置。"""

    from ..configuration.app_settings import update_config

    mapping = _WRITE_THROUGH[profile.kind]
    update_config({
        mapping["provider"]: profile.provider,
        mapping["base_url"]: profile.base_url,
        mapping["model"]: profile.model,
        mapping["api_key"]: profile.api_key,
        **(
            {mapping["request_dialect"]: profile.request_dialect}
            if "request_dialect" in mapping
            else {}
        ),
    })
