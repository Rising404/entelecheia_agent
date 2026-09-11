"""Tool Platform 的封闭效果分类及精确事实推导。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class EffectResource(StrEnum):
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    TASK = "task"
    NETWORK = "network"
    PROCESS = "process"
    CREDENTIAL = "credential"
    UI = "ui"
    EXTERNAL_SERVICE = "external_service"
    RUNTIME_STATE = "runtime_state"


class EffectAction(StrEnum):
    READ = "read"
    SEARCH = "search"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    TRANSMIT = "transmit"


class EffectScopeKind(StrEnum):
    SESSION = "session"
    WORKSPACE = "workspace"
    CONFIGURED_ROOT = "configured_root"
    REMOTE_DOMAIN = "remote_domain"
    ACCOUNT = "account"
    PROCESS = "process"
    LOCAL = "local"
    EXECUTION = "execution"


class DataEgress(StrEnum):
    NONE = "none"
    METADATA = "metadata"
    CONTENT = "content"
    SENSITIVE_POSSIBLE = "sensitive_possible"


class Idempotency(StrEnum):
    IDEMPOTENT = "idempotent"
    DEDUPLICATED = "deduplicated"
    NOT_IDEMPOTENT = "not_idempotent"
    UNKNOWN = "unknown"


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    COMPENSATABLE = "compensatable"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\b(?:api[_-]?key|token|secret|password|passwd|authorization)\b\s*[:=]", re.I),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b1[3-9]\d{9}\b"),
)


@dataclass(frozen=True)
class EffectDescriptor:
    resource: EffectResource
    action: EffectAction
    scope_kind: EffectScopeKind
    default_scope: str = "*"
    data_egress: DataEgress = DataEgress.NONE
    idempotency: Idempotency = Idempotency.UNKNOWN
    reversibility: Reversibility = Reversibility.UNKNOWN
    resource_argument: str | None = None
    scope_argument: str | None = None
    egress_arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.default_scope.strip():
            raise ValueError("default_scope must not be empty")
        if self.resource is EffectResource.NETWORK and self.action is EffectAction.TRANSMIT:
            if self.data_egress is DataEgress.NONE:
                raise ValueError("network transmit must declare data egress")

    def to_dict(self) -> dict[str, Any]:
        """返回所有可能改变策略或执行权威信息的事实。"""

        return {
            "resource": self.resource.value,
            "action": self.action.value,
            "scope_kind": self.scope_kind.value,
            "default_scope": self.default_scope,
            "data_egress": self.data_egress.value,
            "idempotency": self.idempotency.value,
            "reversibility": self.reversibility.value,
            "resource_argument": self.resource_argument,
            "scope_argument": self.scope_argument,
            "egress_arguments": list(self.egress_arguments),
        }


@dataclass(frozen=True)
class ToolEffectProfile:
    effects: tuple[EffectDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.effects:
            raise ValueError("an effect profile must declare at least one effect")


@dataclass(frozen=True)
class EffectFact:
    resource: EffectResource
    action: EffectAction
    scope_kind: EffectScopeKind
    scope: str
    resource_id: str | None
    data_egress: DataEgress
    idempotency: Idempotency
    reversibility: Reversibility
    sensitive_egress: bool = False


def derive_effect_facts(profile: ToolEffectProfile, arguments: Mapping[str, Any]) -> tuple[EffectFact, ...]:
    """根据静态配置和规范化参数推导精确效果事实。"""
    facts: list[EffectFact] = []
    for descriptor in profile.effects:
        resource_id = _text_value(arguments.get(descriptor.resource_argument)) if descriptor.resource_argument else None
        scope = _text_value(arguments.get(descriptor.scope_argument)) if descriptor.scope_argument else None
        egress_values = [arguments.get(key) for key in descriptor.egress_arguments]
        facts.append(
            EffectFact(
                resource=descriptor.resource,
                action=descriptor.action,
                scope_kind=descriptor.scope_kind,
                scope=scope or descriptor.default_scope,
                resource_id=resource_id,
                data_egress=descriptor.data_egress,
                idempotency=descriptor.idempotency,
                reversibility=descriptor.reversibility,
                sensitive_egress=descriptor.data_egress is not DataEgress.NONE
                and any(_contains_sensitive_value(value) for value in egress_values),
            )
        )
    return tuple(facts)


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_sensitive_value(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_sensitive_value(item) for item in value)
    if not isinstance(value, (str, int, float)):
        return False
    text = str(value)
    return any(pattern.search(text) for pattern in _SENSITIVE_VALUE_PATTERNS)
