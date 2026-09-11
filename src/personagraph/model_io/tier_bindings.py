"""定义各模型调用点所使用的端点和思考策略。

Runtime 有六种调用点角色，其成本与风险特征不同。分别绑定 tier，允许验证与构图搜索
选择不同端点和思考策略，而不强制所有调用点共享全局 ``model`` 设置。

因此，一个 tier 会选择以下四项内容：

- 端点：指向已存储的 :mod:`.model_profiles` 条目。多个 tier 可以指向同一 profile；
  这是复用同一配置的预期方式，无须复制四套凭据。
- 具体的供应商请求方言：它独立于 Anthropic/OpenAI 信封，控制线上请求中精确的
  token 上限、思考和结构化输出字段。
- 当所选供应商提供显式开关时，是否启用思考；如果支持，还可选择分级的
  ``reasoning_effort``。支持的子集仍由具体模型决定，因此无效选项会在供应商处明确失败，
  而不会被静默近似。

解析过程绝不抛出异常。如果 tier 指向后来被删除的 profile，它会回退至全局端点并报告
这一情况；因过期设置指针丢失一个 Turn，比继续使用用户创建该 profile 之前的端点更糟。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator

from . import endpoint_profiles as model_profiles
from .dialects import (
    RequestDialect,
    build_request_controls,
    reasoning_capability,
    resolve_request_dialect,
)
from ..configuration.app_settings import (
    TIER_SETTING_NAMES,
    get_setting,
    normalize_provider,
    resolve_global_model_endpoint,
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ModelTier(StrEnum):
    """六种可以独立配置的调用点角色。"""

    ROUTER = "router"
    ARCHITECT = "architect"
    ATTEMPT = "attempt"
    NODE_VERIFICATION = "node_verification"
    FINAL_GATE = "final_gate"
    L1 = "l1"


class ReasoningEffort(StrEnum):
    """可投影到各具体供应商方言的可移植推理强度值。

    具体模型可能只支持其中一部分。用户显式选择值；不受支持的值会在供应商处明确失败，
    而不会被静默近似。
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


TIERS: tuple[ModelTier, ...] = tuple(ModelTier)

# app_settings 拥有设置键命名空间，且不能导入本模块（否则会形成循环），因此在这里
# 交叉检查两份列表。若只在一处新增 tier 而忘记另一处，该 tier 原本会永远静默解析到
# 全局端点。
if tuple(tier.value for tier in TIERS) != TIER_SETTING_NAMES:
    raise RuntimeError(
        "model tier names drifted from app_settings.TIER_SETTING_NAMES"
    )

# 每个 tier 的显式思考开关默认关闭；用户可按调用点的质量与延迟需求显式开启。
# 没有开关的方言会保持供应商默认行为，直到用户选择显式推理强度。
DEFAULT_THINKING_ENABLED = False

# ``app_settings.update_config`` 把空串当作"不改该字段"——这条语义存在的理由是
# 防止保存一次设置就把已配置的 api_key 抹掉，不该为了分档去动它。所以"清除本档、
# 回到全局端点"需要一个写得进去的显式记号。它在配置文件里也自解释。
CLEARED_PROFILE_ID = "-"


def profile_setting_key(tier: ModelTier) -> str:
    return f"tier_{tier.value}_profile_id"


def thinking_setting_key(tier: ModelTier) -> str:
    return f"tier_{tier.value}_thinking"


def reasoning_effort_setting_key(tier: ModelTier) -> str:
    return f"tier_{tier.value}_reasoning_effort"


class EndpointOrigin(StrEnum):
    """已解析 tier 端点的来源，供 UI 展示。"""

    PROFILE = "profile"
    GLOBAL = "global"
    # 已配置的 profile id 不再存在；改用全局端点运行。
    STALE_PROFILE = "stale_profile"


@dataclass(frozen=True, slots=True)
class ModelTierBinding:
    """一个 tier 的已解析端点及其思考策略。"""

    tier: ModelTier
    provider: str
    base_url: str
    model: str
    thinking_enabled: bool
    origin: EndpointOrigin
    profile_id: str | None = None
    profile_name: str | None = None
    # 配额归属独立于 tier 选择的 UI 状态。继承全局端点的 tier 仍有
    # ``profile_id=None``，但当该端点由活动 profile 写入时，这一身份可让后续物理重试
    # 重新加载当前运行配额策略。
    quota_profile_id: str | None = field(default=None, repr=False)
    reasoning_effort: ReasoningEffort | None = None
    # 具体请求语义，独立于 Anthropic/OpenAI 信封。手工构造的测试绑定仍可使用 ``auto``，
    # 并在网关边界解析；``resolve_tier`` 生成的绑定则是具体方言。
    request_dialect: RequestDialect | str = RequestDialect.AUTO
    api_key: str = field(default="", repr=False)
    # 供应商账户的运行限制随内存端点绑定传递。它们被刻意排除在持久模型调用账本之外
    # （账本绝不能保留凭据）；已准备的供应商请求会在排队前冻结不含内容的配额范围与限制。
    quota: model_profiles.ModelProfileQuota = field(
        default_factory=model_profiles.ModelProfileQuota,
        repr=False,
    )

    @property
    def configured(self) -> bool:
        """该 tier 是否解析到自身端点而非全局端点。"""

        return self.origin is EndpointOrigin.PROFILE

    def redacted(self) -> dict[str, Any]:
        """设置 API 返回的数据形状，绝不携带密钥本身。"""

        dialect = resolve_request_dialect(
            self.provider,
            self.base_url,
            self.request_dialect,
        )
        reasoning_control, reasoning_effort_options = reasoning_capability(dialect)
        request_controls = build_request_controls(
            provider=self.provider,
            dialect=dialect,
            thinking_enabled=self.thinking_enabled,
            reasoning_effort=self.reasoning_effort,
            max_tokens=1,
            temperature=None,
            json_mode=False,
        )
        configured_effort = (
            self.reasoning_effort.value
            if isinstance(self.reasoning_effort, ReasoningEffort)
            else self.reasoning_effort
        )

        return {
            "tier": self.tier.value,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "thinking_enabled": self.thinking_enabled,
            "request_dialect": dialect.value,
            "reasoning_control": reasoning_control,
            "reasoning_effort_options": list(reasoning_effort_options),
            # 保留已存选项，使思考开关切换不会在设置表单中抹掉它；另外公开实际线上值，
            # 避免状态把未启用的推理强度宣称为活动状态。
            "reasoning_effort": configured_effort,
            "effective_reasoning_effort": request_controls.fingerprint[
                "reasoning_effort"
            ],
            "origin": self.origin.value,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "has_api_key": bool(self.api_key),
            "quota": self.quota.to_dict(),
        }


# 一个持久逻辑调用拥有一个不可变内存绑定。共享重试包装器仅在调用物理供应商期间将它放入
# 此 ContextVar。使用 ContextVar（而非模块全局变量）可以隔离并发调用，并在异常后自动恢复
# 先前值。
_CURRENT_MODEL_TIER_BINDING: ContextVar[ModelTierBinding | None] = ContextVar(
    "personagraph_current_model_tier_binding",
    default=None,
)


def current_model_tier_binding() -> ModelTierBinding | None:
    """返回一次供应商分派中持久调用的绑定。"""

    return _CURRENT_MODEL_TIER_BINDING.get()


@contextmanager
def model_tier_binding_scope(
    binding: ModelTierBinding,
) -> Iterator[None]:
    """将一个精确 tier 绑定限定在一次物理供应商调用中。"""

    if not isinstance(binding, ModelTierBinding):
        raise TypeError("binding must be ModelTierBinding")
    token = _CURRENT_MODEL_TIER_BINDING.set(binding)
    try:
        yield
    finally:
        _CURRENT_MODEL_TIER_BINDING.reset(token)


def effective_model_tier_binding(
    default_tier: ModelTier,
    *,
    allowed_scoped_tiers: tuple[ModelTier, ...] | None = None,
) -> ModelTierBinding:
    """优先使用持久的作用域绑定，否则重新解析默认绑定。

    供应商适配器在其边界调用本函数。重试期间，作用域绑定绝不会被可变设置静默替换。
    ``allowed`` 还能捕获组合错误，例如通过节点验证供应商发送 final-gate authority。
    """

    if not isinstance(default_tier, ModelTier):
        raise TypeError("default_tier must be ModelTier")
    scoped = current_model_tier_binding()
    if scoped is None:
        return resolve_tier(default_tier)
    allowed = allowed_scoped_tiers or (default_tier,)
    if scoped.tier not in allowed:
        raise RuntimeError(
            "durable model tier does not match the selected provider role"
        )
    return scoped


def read_thinking_enabled(tier: ModelTier) -> bool:
    """该 tier 是否启用隐藏推理。

    无法识别的已存值按默认值处理而非报错：手工编辑的设置文件不应导致 Turn 失败。
    """

    raw = (get_setting(thinking_setting_key(tier)) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return DEFAULT_THINKING_ENABLED


def read_reasoning_effort(tier: ModelTier) -> ReasoningEffort | None:
    """读取显式 OpenAI 推理强度；未设置表示使用供应商默认值。"""

    raw = (
        get_setting(reasoning_effort_setting_key(tier)) or ""
    ).strip().lower()
    if not raw or raw == "auto":
        return None
    try:
        return ReasoningEffort(raw)
    except ValueError:
        # 手工写入的未来值或无效值不能导致 Turn 无法启动。
        return None


def _global_binding(
    tier: ModelTier, *, origin: EndpointOrigin, profile_id: str | None
) -> ModelTierBinding:
    endpoint = resolve_global_model_endpoint()
    raw_provider = endpoint.raw_provider
    provider = endpoint.provider
    base_url = endpoint.base_url
    model = endpoint.model
    if provider not in {"mock", "anthropic-compatible", "openai-compatible"}:
        # 手工编辑的全局供应商不能使设置读取或整个 L2 Turn 变成导入时/配置解析异常。
        # 以确定性 mock 绑定保守失败；脱敏后的全局配置仍会公开无效供应商和
        # 并公开为 model_configured=False。
        provider = "mock"
        base_url = ""
        model = ""
    try:
        dialect = resolve_request_dialect(
            provider,
            base_url,
            get_setting("request_dialect", "auto"),
            legacy_provider=raw_provider,
        )
    except ValueError:
        provider = "mock"
        base_url = ""
        model = ""
        dialect = RequestDialect.MOCK
    api_key = get_setting("api_key", "") or ""
    from .api_quota_controller import model_profile_quota_from_environment

    quota = (
        model_profile_quota_from_environment()
        or model_profiles.ModelProfileQuota()
    )
    quota_profile_id: str | None = None
    active_profile = model_profiles.resolve_active_profile("model")
    if (
        active_profile is not None
        and normalize_provider(active_profile.provider) == provider
        and active_profile.base_url.rstrip("/") == str(base_url or "").rstrip("/")
        and active_profile.api_key == api_key
    ):
        # 激活流程会将此 profile 写入全局端点。仅当端点与凭据仍匹配时才继承其配额；
        # 手工编辑的全局设置不能意外借用过期 profile 的账户限制。
        quota = active_profile.quota
        quota_profile_id = active_profile.id
    return ModelTierBinding(
        tier=tier,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        quota=quota,
        quota_profile_id=quota_profile_id,
        thinking_enabled=read_thinking_enabled(tier),
        reasoning_effort=read_reasoning_effort(tier),
        origin=origin,
        profile_id=profile_id,
        request_dialect=dialect,
    )


def resolve_tier(tier: ModelTier) -> ModelTierBinding:
    """将一个 tier 解析为具体端点和思考策略。"""

    profile_id = (get_setting(profile_setting_key(tier)) or "").strip()
    if not profile_id or profile_id == CLEARED_PROFILE_ID:
        return _global_binding(tier, origin=EndpointOrigin.GLOBAL, profile_id=None)

    profile = model_profiles.resolve_profile(profile_id)
    if profile is None or profile.kind != "model":
        # 指针比 profile 存活得更久（profile 已删除或文件被替换）。回退而不是失败，并记录
        # 此情况，以便设置页显示该 tier 当前没有有效指向。
        return _global_binding(
            tier, origin=EndpointOrigin.STALE_PROFILE, profile_id=profile_id
        )

    provider = normalize_provider(profile.provider)
    dialect = resolve_request_dialect(
        provider,
        profile.base_url,
        getattr(profile, "request_dialect", "auto"),
    )
    return ModelTierBinding(
        tier=tier,
        provider=provider,
        base_url=profile.base_url,
        model=profile.model,
        api_key=profile.api_key,
        thinking_enabled=read_thinking_enabled(tier),
        reasoning_effort=read_reasoning_effort(tier),
        origin=EndpointOrigin.PROFILE,
        profile_id=profile.id,
        profile_name=profile.name,
        quota_profile_id=profile.id,
        request_dialect=dialect,
        quota=profile.quota,
    )


def resolve_all_tiers() -> dict[str, ModelTierBinding]:
    return {tier.value: resolve_tier(tier) for tier in TIERS}


def redacted_tier_view() -> list[dict[str, Any]]:
    """供设置页使用的各 tier 有效端点。

    始终报告实际生效的端点，包括继承自全局配置的情况——002 §7.4 要求未设置的 tier
    显示它真正会调用的内容，而不是空白框。
    """

    return [resolve_tier(tier).redacted() for tier in TIERS]


__all__ = [
    "CLEARED_PROFILE_ID",
    "DEFAULT_THINKING_ENABLED",
    "EndpointOrigin",
    'ModelTierBinding',
    'ModelTier',
    'ReasoningEffort',
    "TIERS",
    "current_model_tier_binding",
    "effective_model_tier_binding",
    "model_tier_binding_scope",
    "profile_setting_key",
    "reasoning_effort_setting_key",
    "read_thinking_enabled",
    "read_reasoning_effort",
    "redacted_tier_view",
    "resolve_all_tiers",
    "resolve_tier",
    "thinking_setting_key",
]
