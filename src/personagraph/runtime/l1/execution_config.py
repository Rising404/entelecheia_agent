"""单个 L1 TurnRun 所使用的不可变、凭证安全执行权威状态。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personagraph.model_io.tier_bindings import ModelTierBinding, ModelTier, resolve_tier
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)
from .identity import canonical_json


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class L1ExecutionConfigError(ValueError):
    """一个 L1 执行快照缺失、损坏或已无法解析。"""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class L1ModelEndpointSnapshot(_Contract):
    """非机密端点标识符，必须在整个 TurnRun 中保持稳定。"""

    tier: Literal["l1"] = "l1"
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=300)
    protocol: str = Field(min_length=1, max_length=200)
    control_profile_id: str = Field(pattern=_SHA256_PATTERN)
    endpoint_fingerprint: str = Field(pattern=_SHA256_PATTERN)


class L1ExecutionConfig(_Contract):
    """完整冻结所有非秘密的 Runtime 输入，冻结前基于 L1 提供者的 I/O。"""

    schema_version: Literal["l1-execution-config-v1"] = "l1-execution-config-v1"
    features: dict[str, Any]
    model_endpoint: L1ModelEndpointSnapshot

    @field_validator("features")
    @classmethod
    def _canonical_feature_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("L1 feature keys must be non-blank strings")
        # 这会立即拒绝非 JSON 值，避免其在部分初始化的运行提交前漏过检查；
        # 同时还会通过 canonical_json 拒绝 NaN/Infinity。
        canonical_json(value)
        return value


@dataclass(frozen=True, slots=True)
class FrozenL1ExecutionConfig:
    snapshot: L1ExecutionConfig
    snapshot_json: str
    snapshot_sha256: str
    model_binding: ModelTierBinding


def freeze_l1_execution_config(
    features: Mapping[str, Any],
) -> FrozenL1ExecutionConfig:
    """将 features 与 ModelTier.L1 的端点身份封成不可变执行快照。

    JSON 往返切断调用方可变容器的引用；持久快照只保存身份/指纹，API key 留在当前
    binding 凭据句柄中。L1 决策和 reviewer 都沿用该 binding，不能在每轮按全局默认重选。
    """

    try:
        # Freeze-by-value：不让调用方持有的嵌套容器在快照校验后继续改变 features。
        normalized_features = json.loads(canonical_json(dict(features)))
        binding = resolve_tier(ModelTier.L1)
        identity = configured_structured_model_endpoint_identity(binding)
        snapshot = L1ExecutionConfig(
            features=normalized_features,
            model_endpoint=L1ModelEndpointSnapshot(
                provider=identity.provider,
                model=identity.model,
                protocol=identity.protocol,
                control_profile_id=identity.control_profile_id,
                endpoint_fingerprint=identity.endpoint_fingerprint,
            ),
        )
        snapshot_json = canonical_json(snapshot.model_dump(mode="json"))
    except Exception as exc:
        raise L1ExecutionConfigError(
            "L1 execution configuration could not be frozen"
        ) from exc
    return FrozenL1ExecutionConfig(
        snapshot=snapshot,
        snapshot_json=snapshot_json,
        snapshot_sha256=_sha256_text(snapshot_json),
        model_binding=binding,
    )


def load_l1_execution_config(
    snapshot_json: object,
    snapshot_sha256: object,
    *,
    require_current_endpoint: bool = True,
) -> FrozenL1ExecutionConfig:
    """恢复执行快照：校验 hash / canonical JSON，再解析当前 L1 凭据句柄。

    快照不保存 API 密钥。默认还要求当前 endpoint 身份与接受时完全一致；
    require_current_endpoint=False 只跳过端点相等检查，不跳过快照完整性验证。
    """

    if not isinstance(snapshot_json, str) or not isinstance(snapshot_sha256, str):
        raise L1ExecutionConfigError("L1 execution configuration is unavailable")
    if not hmac.compare_digest(_sha256_text(snapshot_json), snapshot_sha256):
        raise L1ExecutionConfigError("L1 execution configuration hash changed")
    try:
        snapshot = L1ExecutionConfig.model_validate_json(snapshot_json)
        if canonical_json(snapshot.model_dump(mode="json")) != snapshot_json:
            raise ValueError("snapshot is not canonical JSON")
        binding = resolve_tier(ModelTier.L1)
        identity = configured_structured_model_endpoint_identity(binding)
    except Exception as exc:
        raise L1ExecutionConfigError(
            "L1 execution configuration failed validation"
        ) from exc
    expected_endpoint = snapshot.model_endpoint
    current_endpoint = L1ModelEndpointSnapshot(
        provider=identity.provider,
        model=identity.model,
        protocol=identity.protocol,
        control_profile_id=identity.control_profile_id,
        endpoint_fingerprint=identity.endpoint_fingerprint,
    )
    if require_current_endpoint and current_endpoint != expected_endpoint:
        raise L1ExecutionConfigError(
            "L1 model endpoint identity changed after Turn acceptance"
        )
    return FrozenL1ExecutionConfig(
        snapshot=snapshot,
        snapshot_json=snapshot_json,
        snapshot_sha256=snapshot_sha256,
        model_binding=binding,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    'FrozenL1ExecutionConfig',
    "L1ExecutionConfigError",
    'L1ExecutionConfig',
    'L1ModelEndpointSnapshot',
    "freeze_l1_execution_config",
    "load_l1_execution_config",
]
