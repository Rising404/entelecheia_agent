"""冷 AcceptedTurn 路由契约的验证与序列化。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from ...turn.contracts import (
    DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT,
    ProcessingLevel,
    RoutingPolicySource,
    TurnRoutingPolicySnapshot,
    TurnRoutingPolicy,
)


_POLICY_KEYS = {"schema_version", "l1_enabled", "l2_enabled"}
_SNAPSHOT_KEYS = {
    "schema_version",
    "source",
    "policy",
    "allowed_processing_levels",
}
_POLICY_SOURCES = {
    "default",
    "session_default",
    "request_override",
}


@dataclass(frozen=True, slots=True)
class ResolvedTurnRoutingPolicy:
    """已解析 Turn 快照，以及它是否更新 Session 默认值。"""

    snapshot: TurnRoutingPolicySnapshot
    persist_as_session_default: bool


class TurnRoutingPolicyError(ValueError):
    """请求或已存储路由策略无法安全准入。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def default_turn_routing_policy() -> TurnRoutingPolicy:
    """返回为 L1 激活所选择的产品默认值。"""

    return DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT.policy


def available_processing_levels() -> tuple[ProcessingLevel, ...]:
    """返回 Host 实现的处理层级；逐 Turn policy 决定其中哪些可用。"""

    return ("L0", "L1", "L2")


def freeze_turn_routing_policy(
    policy: TurnRoutingPolicy,
    *,
    source: RoutingPolicySource,
) -> TurnRoutingPolicySnapshot:
    """把已解析的开关与来源封成 Turn routing snapshot，供接受、重放和分类共用。

    allowed_processing_levels 是 Host 允许的选择集合，不是 classifier 的决定。
    本函数只构造值对象；持久化及完整性哈希的写入发生在 Turn 准入边界。
    """

    return TurnRoutingPolicySnapshot(
        source=source,
        policy=policy,
        allowed_processing_levels=policy.allowed_processing_levels,
    )


def resolve_turn_routing_policy(
    *,
    request_policy: object,
    request_policy_provided: bool,
    stored_session_policy_json: str | None,
) -> ResolvedTurnRoutingPolicy:
    """不使用语义路由，按请求 > Session > 产品默认值解析。

    显式请求覆盖会标记 persist_as_session_default，实际保存要等 Turn 接受成功；
    解析本身不写 Session。坏的已存策略直接报错，不能静默退回更宽松的产品默认值。
    """

    if request_policy_provided:
        policy = _parse_request_policy(request_policy)
        return ResolvedTurnRoutingPolicy(
            snapshot=freeze_turn_routing_policy(policy, source="request_override"),
            persist_as_session_default=True,
        )

    if stored_session_policy_json is not None:
        try:
            parsed = json.loads(stored_session_policy_json)
            policy = _policy_from_mapping(parsed, require_complete=True)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TurnRoutingPolicyError(
                "STORED_RUNTIME_POLICY_INVALID",
                "the stored Session routing policy is invalid",
            ) from exc
        return ResolvedTurnRoutingPolicy(
            snapshot=freeze_turn_routing_policy(policy, source="session_default"),
            persist_as_session_default=False,
        )

    return ResolvedTurnRoutingPolicy(
        snapshot=freeze_turn_routing_policy(
            default_turn_routing_policy(),
            source="default",
        ),
        persist_as_session_default=False,
    )


def parse_turn_routing_policy_snapshot(
    value: str | dict[str, object],
    *,
    expected_sha256: str | None = None,
) -> TurnRoutingPolicySnapshot:
    try:
        if expected_sha256 is not None:
            if not isinstance(value, str) or _sha256(value) != expected_sha256:
                raise ValueError("routing-policy snapshot hash changed")
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict) or set(parsed) != _SNAPSHOT_KEYS:
            raise ValueError("routing-policy snapshot has unexpected fields")
        source = parsed["source"]
        if not isinstance(source, str) or source not in _POLICY_SOURCES:
            raise ValueError("invalid routing-policy snapshot source")
        raw_levels = parsed["allowed_processing_levels"]
        if not isinstance(raw_levels, (list, tuple)) or any(
            type(level) is not str or level not in {"L0", "L1", "L2"}
            for level in raw_levels
        ):
            raise ValueError("invalid allowed_processing_levels")
        return TurnRoutingPolicySnapshot(
            schema_version=parsed["schema_version"],
            source=cast(RoutingPolicySource, source),
            policy=_policy_from_mapping(
                parsed["policy"],
                require_complete=True,
            ),
            allowed_processing_levels=cast(
                tuple[ProcessingLevel, ...], tuple(raw_levels)
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TurnRoutingPolicyError(
            "STORED_RUNTIME_POLICY_INVALID",
            "the accepted Turn routing-policy snapshot is invalid",
        ) from exc


def canonical_policy_json(policy: TurnRoutingPolicy) -> str:
    return _canonical_json(policy.to_dict())


def policy_sha256(policy: TurnRoutingPolicy) -> str:
    return _sha256(canonical_policy_json(policy))


def canonical_snapshot_json(snapshot: TurnRoutingPolicySnapshot) -> str:
    return _canonical_json(snapshot.to_dict())


def snapshot_sha256(snapshot: TurnRoutingPolicySnapshot) -> str:
    return _sha256(canonical_snapshot_json(snapshot))


def _parse_request_policy(value: object) -> TurnRoutingPolicy:
    try:
        return _policy_from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise TurnRoutingPolicyError(
            "INVALID_RUNTIME_POLICY",
            "runtime_policy does not satisfy the routing contract",
        ) from exc


def _policy_from_mapping(
    value: object,
    *,
    require_complete: bool = False,
) -> TurnRoutingPolicy:
    if not isinstance(value, dict):
        raise TypeError("routing policy must be an object")
    if not set(value).issubset(_POLICY_KEYS):
        raise ValueError("routing policy has unexpected fields")
    if require_complete and set(value) != _POLICY_KEYS:
        raise ValueError("stored routing policy is incomplete")
    return TurnRoutingPolicy(
        schema_version=value.get("schema_version", 1),
        l1_enabled=value.get("l1_enabled", True),
        l2_enabled=value.get("l2_enabled", False),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_TURN_ROUTING_POLICY_SNAPSHOT",
    "ProcessingLevel",
    "ResolvedTurnRoutingPolicy",
    "RoutingPolicySource",
    "TurnRoutingPolicyError",
    "TurnRoutingPolicySnapshot",
    "TurnRoutingPolicy",
    "canonical_policy_json",
    "canonical_snapshot_json",
    "available_processing_levels",
    "default_turn_routing_policy",
    "freeze_turn_routing_policy",
    "parse_turn_routing_policy_snapshot",
    "policy_sha256",
    "resolve_turn_routing_policy",
    "snapshot_sha256",
]
