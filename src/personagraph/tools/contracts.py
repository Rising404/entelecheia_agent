"""独立于 Runtime 的 Tool Platform 契约。

本模块特意不从 ``personagraph.runtime``、LangGraph、Session 持久化或提供商 SDK 导入内容。
它定义未来 Runtime 可能持久化的值，但绝不创建其生命周期记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class ToolSourceKind(StrEnum):
    LOCAL = "local"
    MCP = "mcp"
    PROVIDER = "provider"


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    COMPLETION_UNCONFIRMED = "completion_unconfirmed"


CATALOG_TAGS = frozenset(
    {
        "artifact",
        "automation",
        "document",
        "execution",
        "file",
        "legacy",
        "memory",
        "quality",
        "research",
        "time",
        "tooling",
        "web",
        "read",
        "write",
        "search",
        "fetch",
        "list",
        "plan",
        "status",
    }
)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """返回适用于验证器和传输层的普通 JSON 形态副本。"""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_json(item) for item in value]
    return value


def _nonempty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class ToolSpec:
    """稳定且可向模型暴露的工具契约。

    它特意不含处理器、确认位、风险级别、能力字符串集合或副作用标志。这些概念要么属于
    注册/效果配置，要么属于逐次调用策略决策。
    """

    tool_id: str
    contract_version: str
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    catalog_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _nonempty(self.tool_id, "tool_id"))
        object.__setattr__(self, "contract_version", _nonempty(self.contract_version, "contract_version"))
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "description", _nonempty(self.description, "description"))
        if not isinstance(self.input_schema, Mapping):
            raise ValueError("input_schema must be a JSON object")
        if not isinstance(self.output_schema, Mapping):
            raise ValueError("output_schema must be a JSON object")
        tags = tuple(dict.fromkeys(str(tag).strip() for tag in self.catalog_tags if str(tag).strip()))
        unknown = set(tags) - CATALOG_TAGS
        if unknown:
            raise ValueError(f"unknown catalog tags: {sorted(unknown)!r}")
        object.__setattr__(self, "catalog_tags", tags)
        object.__setattr__(self, "input_schema", _freeze_json(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_json(self.output_schema))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "contract_version": self.contract_version,
            "name": self.name,
            "description": self.description,
            "input_schema": thaw_json(self.input_schema),
            "output_schema": thaw_json(self.output_schema),
            "catalog_tags": list(self.catalog_tags),
        }


@dataclass(frozen=True)
class ToolSourceDescriptor:
    kind: ToolSourceKind
    source_id: str
    fingerprint: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        if self.fingerprint is not None:
            object.__setattr__(self, "fingerprint", _nonempty(self.fingerprint, "fingerprint"))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind.value,
            "source_id": self.source_id,
            "fingerprint": self.fingerprint,
            "display_name": self.display_name,
        }


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _nonempty(self.code, "code"))
        object.__setattr__(self, "message", _nonempty(self.message, "message"))
        object.__setattr__(self, "details", _freeze_json(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": thaw_json(self.details)}


@dataclass(frozen=True)
class ToolCallProposal:
    """纯提案形态；ID、持久化及批准生命周期由 Runtime 持有。"""

    tool_id: str
    arguments: Mapping[str, Any]
    contract_version: str | None = None
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _nonempty(self.tool_id, "tool_id"))
        if self.contract_version is not None:
            object.__setattr__(self, "contract_version", _nonempty(self.contract_version, "contract_version"))
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a JSON object")
        object.__setattr__(self, "arguments", _freeze_json(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "contract_version": self.contract_version,
            "provider_call_id": self.provider_call_id,
            "arguments": thaw_json(self.arguments),
        }


@dataclass(frozen=True)
class ExecutionOutcome:
    """恰好一次已解析调用的结果，不含 Operation 记录。"""

    status: ExecutionStatus
    result: Mapping[str, Any] | None = None
    error: ToolError | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is ExecutionStatus.SUCCEEDED and self.result is None:
            raise ValueError("successful execution requires a result")
        if self.status is not ExecutionStatus.SUCCEEDED and self.error is None:
            raise ValueError("non-successful execution requires an error")
        if self.result is not None and not isinstance(self.result, Mapping):
            raise ValueError("execution result must be a JSON object")
        object.__setattr__(self, "result", _freeze_json(self.result) if self.result is not None else None)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @classmethod
    def succeeded(cls, result: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> "ExecutionOutcome":
        return cls(ExecutionStatus.SUCCEEDED, result=result, metadata=metadata or {})

    @classmethod
    def rejected(cls, error: ToolError, *, metadata: Mapping[str, Any] | None = None) -> "ExecutionOutcome":
        return cls(ExecutionStatus.REJECTED, error=error, metadata=metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "result": thaw_json(self.result) if self.result is not None else None,
            "error": self.error.to_dict() if self.error else None,
            "metadata": thaw_json(self.metadata),
        }
