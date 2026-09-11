"""运行时配置单一入口（FR-5，2026-07-11）。

背景：模型 provider / api_key 等此前只能从环境变量读，没有应用内配置入口，
新用户开箱无法对话（默认 mock 返回空、真实调用无 key 直接报错）。本模块加一层持久配置：

    get_setting(key) 优先级： 环境变量 > 用户配置目录/app_config.json > 默认值

env 优先保证既有命令行/CI 工作流零改动；JSON 层供设置界面（PUT /api/config）写入。
配置文件落 LOCAL_CONFIG_DIR（默认是仓库外的 Entelecheia 用户配置目录），
含明文 key，绝不进入源码仓库。运行时状态使用独立的用户数据目录。
"""

from __future__ import annotations

import json
import os
import platform
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .paths import LOCAL_CONFIG_DIR
from personagraph.model_io.dialects import (
    PROFILE_DIALECT_CHOICES,
    default_base_url_for_provider,
    default_model_for_request_dialect,
    resolve_request_dialect,
)

CONFIG_PATH = LOCAL_CONFIG_DIR / "app_config.json"

# 允许通过配置文件覆盖的键 → 对应环境变量名。
_CONFIG_ENV_MAP: dict[str, str] = {
    "provider": "PERSONAGRAPH_MODEL_PROVIDER",
    "request_dialect": "PERSONAGRAPH_REQUEST_DIALECT",
    "api_key": "PERSONAGRAPH_API_KEY",
    "base_url": "PERSONAGRAPH_BASE_URL",
    "model": "PERSONAGRAPH_MODEL",
    # Entry 文本生成有自己的有界默认值，但运维人员必须能够在不手工编辑 app_config 的情况下
    # 降低该上限以控制成本。结构化 WorkRun profile 仍会显式传入精确上限，因此不继承此值。
    "max_tokens": "PERSONAGRAPH_MAX_TOKENS",
    # 视觉理解走独立的 provider（14/02 F3）。它与主模型平行而不是复用：主模型可以
    # 是纯文本的，视觉能力由工具在需要时调用另一个端点，两者的 key 与配额也应分开。
    # 四个都不配时，VisionModelAdapter 保持 unavailable，非文本单元继续留 typed gap。
    "vision_provider": "PERSONAGRAPH_VISION_PROVIDER",
    "vision_api_key": "PERSONAGRAPH_VISION_API_KEY",
    "vision_base_url": "PERSONAGRAPH_VISION_BASE_URL",
    "vision_model": "PERSONAGRAPH_VISION_MODEL",
    # 除非 Host 指定精确 LibreOffice 可执行文件，否则禁用二进制 .doc/.ppt 摄取。reader 还
    # 要求 OS sandbox；该路径绝不接受模型/工具输入，也不会搜索 PATH。
    "legacy_office_soffice": "PERSONAGRAPH_LEGACY_OFFICE_SOFFICE",
    "default_projects_dir": "PERSONAGRAPH_DEFAULT_PROJECTS_DIR",
}

# 六个调用点位分档（27/002 §4；Router 在真实链路诊断后补列）。每档存端点指针与 provider 能真实执行的
# 推理控制。具体字段由 profile 的请求方言决定：有的厂商同时支持 thinking
# 开关和 effort 档位，有的只支持 effort，通用代理则不擅自发送厂商字段。
# 端点凭据仍然只存在 profile 池里，这里不复制。
# 键名归本模块所有；`model_tiers` 在导入时断言它的枚举与这里一致，防止两边漂移。
TIER_SETTING_NAMES: tuple[str, ...] = (
    "router",
    "architect",
    "attempt",
    "node_verification",
    "final_gate",
    "l1",
)
for _tier in TIER_SETTING_NAMES:
    _CONFIG_ENV_MAP[f"tier_{_tier}_profile_id"] = (
        f"PERSONAGRAPH_TIER_{_tier.upper()}_PROFILE_ID"
    )
    _CONFIG_ENV_MAP[f"tier_{_tier}_thinking"] = (
        f"PERSONAGRAPH_TIER_{_tier.upper()}_THINKING"
    )
    _CONFIG_ENV_MAP[f"tier_{_tier}_reasoning_effort"] = (
        f"PERSONAGRAPH_TIER_{_tier.upper()}_REASONING_EFFORT"
    )
del _tier

# profile_id、thinking 与 reasoning_effort 都不是机密。凭据仍只在 profile
# 池中，因此这里不需要新增 secret 键。
_SECRET_KEYS = {"api_key", "vision_api_key"}


def _load_file() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_setting(key: str, default: str | None = None) -> str | None:
    """env 优先 → 配置文件 → 默认。key 使用 ``_CONFIG_ENV_MAP`` 中的逻辑名。"""
    env_name = _CONFIG_ENV_MAP.get(key)
    if env_name:
        env_val = os.getenv(env_name)
        if env_val is not None and env_val != "":
            return env_val
    file_val = _load_file().get(key)
    if isinstance(file_val, str) and file_val != "":
        return file_val
    return default


def update_config(patch: dict[str, Any]) -> dict[str, Any]:
    """合并写入配置文件。只接受已知键；空串/None 表示"不改该字段"（不覆盖既有）。

    返回脱敏后的当前配置视图（供接口回显）。原子写，避免半截文件。
    """
    current = _load_file()
    for key in _CONFIG_ENV_MAP:
        if key not in patch:
            continue
        value = patch.get(key)
        if value is None or (isinstance(value, str) and value == ""):
            continue  # 空 = 保留原值，不覆盖（尤其 api_key，避免清空已配置的 key）
        current[key] = str(value)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_PATH.parent, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(current, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
        try:
            os.chmod(CONFIG_PATH, 0o600)  # 含明文 key，仅属主可读写
        except OSError:
            pass
    finally:
        Path(tmp).unlink(missing_ok=True)
    return redacted_view()


# 这三个名字过去指向同一套协议适配器，代码里从来没有区分过它们——
# ModelResult 甚至一律把 provider 记成 "anthropic-compatible"。合并成一个名字，
# 并把旧值在读取时归一化，这样已经存好的配置不会因为改名而失效。
_LEGACY_PROVIDER_ALIASES = {
    "anthropic": "anthropic-compatible",
    "deepseek": "anthropic-compatible",
}

ANTHROPIC_COMPATIBLE = "anthropic-compatible"
OPENAI_COMPATIBLE = "openai-compatible"
SUPPORTED_PROVIDERS = ("mock", ANTHROPIC_COMPATIBLE, OPENAI_COMPATIBLE)
REQUEST_DIALECTS: tuple[str, ...] = PROFILE_DIALECT_CHOICES


def normalize_provider(value: object) -> str:
    """无论旧配置使用何种名称，每种协议统一为一个名称。"""

    name = str(value or "").strip().lower()
    if not name:
        return "mock"
    return _LEGACY_PROVIDER_ALIASES.get(name, name)


def active_provider() -> str:
    return normalize_provider(get_setting("provider", "mock"))


def normalize_request_dialect(value: object) -> str:
    """在不猜测的前提下规范化持久请求形状提示。"""

    name = str(value or "").strip().lower()
    return name or "auto"


@dataclass(frozen=True, slots=True)
class GlobalModelEndpointConfig:
    """全局模型端点的一份已解析、无密钥视图。"""

    raw_provider: str
    provider: str
    base_url: str
    model: str


def resolve_global_model_endpoint() -> GlobalModelEndpointConfig:
    """解析全局供应商、URL 和模型，同时保留供应商别名信息。

    ``anthropic-compatible`` 协议名同时用于 DeepSeek 的 Anthropic 形状端点和 Anthropic
    自身的 Messages API。因此，必须在规范化名称*之前*根据原始已配置供应商选择默认值。
    所有全局调用方均使用此投影，确保设置页、持久身份和实际 HTTP 请求不会彼此漂移。
    """

    raw_provider = str(get_setting("provider", "mock") or "mock").strip().lower()
    provider = normalize_provider(raw_provider)
    default_base_url = default_base_url_for_provider(raw_provider)
    base_url = str(
        get_setting("base_url", default_base_url) or default_base_url
    ).strip()
    try:
        dialect = resolve_request_dialect(
            provider,
            base_url,
            get_setting("request_dialect", "auto"),
            legacy_provider=raw_provider,
        )
        default_model = default_model_for_request_dialect(dialect)
    except ValueError:
        # 对于手工编辑的无效配置，仍让视图/状态保持可读；下方验证层会将其报告为未配置。
        default_model = ""
    return GlobalModelEndpointConfig(
        raw_provider=raw_provider,
        provider=provider,
        base_url=base_url,
        model=str(get_setting("model", default_model) or default_model).strip(),
    )


def redacted_view() -> dict[str, Any]:
    """脱敏配置视图：不回明文 key，只回 has_key 布尔。合成 env+文件的有效值。"""
    endpoint = resolve_global_model_endpoint()
    return {
        "provider": endpoint.provider,
        "request_dialect": normalize_request_dialect(
            get_setting("request_dialect", "auto")
        ),
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "has_key": bool(get_setting("api_key")),
        "default_projects_dir": default_projects_directory(),
        "default_projects_dir_managed": bool(os.getenv("PERSONAGRAPH_DEFAULT_PROJECTS_DIR")),
        "legacy_office_soffice": legacy_office_soffice_status(),
        # 视觉端点与主模型平行：主模型可以是纯文本的，图片由工具在需要时
        # 调用另一个端点。四个都空表示未配置，此时视觉能力保持 unavailable。
        "vision_provider": get_setting("vision_provider", ""),
        "vision_base_url": get_setting("vision_base_url", ""),
        "vision_model": get_setting("vision_model", ""),
        "has_vision_key": bool(get_setting("vision_api_key")),
    }


def default_projects_directory() -> str:
    """默认目录按当前配置解析，仅影响尚未创建的 Session。"""
    from . import paths

    return str(get_setting("default_projects_dir", str(paths.DEFAULT_SESSION_PROJECTS_DIR)))


def validate_legacy_office_soffice(value: object) -> str:
    """返回规范化可执行文件路径，或拒绝不安全的本地值。

    它被刻意设计得比 :func:`get_setting` 更严格：只有 API 边界会在持久化前调用它，而
    ``redacted_view`` 必须仍能描述来自环境变量或旧手工配置文件的无效值。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("legacy Office executable must be a non-empty path")
    normalized = value.strip()
    path = Path(normalized)
    if not path.is_absolute():
        raise ValueError("legacy Office executable path must be absolute")
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise ValueError("legacy Office executable is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError("legacy Office executable must not be a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("legacy Office executable must be a regular file")
    if not os.access(path, os.X_OK):
        raise ValueError("legacy Office executable is not executable")
    return normalized


def legacy_office_soffice_status() -> dict[str, bool]:
    """仅公开有效本地路径是否已设置且可用。

    路径本身是 Host 本地细节，绝不能进入公开配置视图。因此，无效环境/手工配置会报告
    ``configured=True`` 和 ``available=False``，而不会被误认为已验证的可执行文件。
    """
    configured_value = get_setting("legacy_office_soffice")
    configured = bool(configured_value and configured_value.strip())
    available = False
    if configured:
        try:
            validate_legacy_office_soffice(configured_value)
            available = platform.system() == "Darwin"
        except (OSError, ValueError):
            available = False
    return {"configured": configured, "available": available}


def model_configured() -> bool:
    """有效全局端点是否完整到足以发起调用。"""

    endpoint = resolve_global_model_endpoint()
    if endpoint.provider not in SUPPORTED_PROVIDERS or endpoint.provider == "mock":
        return False
    if not endpoint.base_url or not endpoint.model or not get_setting("api_key"):
        return False
    try:
        parsed = urlsplit(endpoint.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        resolve_request_dialect(
            endpoint.provider,
            endpoint.base_url,
            get_setting("request_dialect", "auto"),
            legacy_provider=endpoint.raw_provider,
        )
    except ValueError:
        return False
    return True
