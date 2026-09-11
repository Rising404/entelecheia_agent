"""稳定工具契约的实现绑定。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol

from .contracts import ToolSourceDescriptor, ToolSpec
from .effects import ToolEffectProfile
from .schema_validation import ToolSchemaCompiler


ToolHandler = Callable[[dict[str, Any]], Any]


class ExecutableToolRegistration(Protocol):
    """Policy、Catalog 与 Executor 共同消费的最小结构合同。"""

    @property
    def spec(self) -> ToolSpec: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def source(self) -> ToolSourceDescriptor: ...

    @property
    def handler(self) -> ToolHandler: ...

    @property
    def effect_profile(self) -> ToolEffectProfile: ...

    @property
    def execution_profile(self) -> "ToolExecutionProfile": ...

    @property
    def tool_id(self) -> str: ...

    @property
    def contract_version(self) -> str: ...

    def descriptor(self) -> dict[str, Any]: ...


class ExecutionMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
    SUBPROCESS = "subprocess"
    EXTERNAL = "external"


class CancellationMode(StrEnum):
    NONE = "none"
    COOPERATIVE = "cooperative"
    FORCEFUL = "forceful"


class IsolationRequirement(StrEnum):
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ToolExecutionProfile:
    default_timeout_s: float | None = None
    hard_timeout_s: float | None = None
    max_output_bytes: int = 1_000_000
    max_transparent_retries: int = 3
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    cancellation_mode: CancellationMode = CancellationMode.NONE
    isolation_requirement: IsolationRequirement = IsolationRequirement.IN_PROCESS
    concurrency_class: str = "default"

    def __post_init__(self) -> None:
        for name in ("default_timeout_s", "hard_timeout_s"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")
        if self.default_timeout_s and self.hard_timeout_s and self.default_timeout_s > self.hard_timeout_s:
            raise ValueError("default_timeout_s cannot exceed hard_timeout_s")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if (
            isinstance(self.max_transparent_retries, bool)
            or not isinstance(self.max_transparent_retries, int)
            or not 0 <= self.max_transparent_retries <= 3
        ):
            raise ValueError("max_transparent_retries must be an integer from 0 to 3")
        if not self.concurrency_class.strip():
            raise ValueError("concurrency_class must not be empty")


@dataclass(frozen=True)
class ToolRegistration:
    spec: ToolSpec
    implementation_version: str
    source: ToolSourceDescriptor
    handler: ToolHandler
    effect_profile: ToolEffectProfile
    execution_profile: ToolExecutionProfile

    def __post_init__(self) -> None:
        if not self.implementation_version.strip():
            raise ValueError("implementation_version must not be empty")
        if not callable(self.handler):
            raise ValueError("handler must be callable")
        compiler = ToolSchemaCompiler()
        compiler.compile(self.spec.input_schema, role="input")
        compiler.compile(self.spec.output_schema, role="output")

    @property
    def tool_id(self) -> str:
        return self.spec.tool_id

    @property
    def contract_version(self) -> str:
        return self.spec.contract_version

    def descriptor(self) -> dict[str, Any]:
        """特意排除处理器对象的可序列化视图。"""
        return _registration_descriptor(
            spec=self.spec,
            implementation_version=self.implementation_version,
            source=self.source,
            effect_profile=self.effect_profile,
            execution_profile=self.execution_profile,
        )


def _registration_descriptor(
    *,
    spec: ToolSpec,
    implementation_version: str,
    source: ToolSourceDescriptor,
    effect_profile: ToolEffectProfile,
    execution_profile: ToolExecutionProfile,
) -> dict[str, Any]:
    return {
        "spec": spec.to_dict(),
        "implementation_version": implementation_version,
        "source": source.to_dict(),
        "effect_count": len(effect_profile.effects),
        "effects": _effect_descriptor(effect_profile),
        "execution": _execution_descriptor(execution_profile),
    }


def _effect_descriptor(profile: ToolEffectProfile) -> list[dict[str, Any]]:
    return [descriptor.to_dict() for descriptor in profile.effects]


def _execution_descriptor(profile: ToolExecutionProfile) -> dict[str, Any]:
    return {
        "default_timeout_s": profile.default_timeout_s,
        "hard_timeout_s": profile.hard_timeout_s,
        "max_output_bytes": profile.max_output_bytes,
        "max_transparent_retries": profile.max_transparent_retries,
        "execution_mode": profile.execution_mode.value,
        "cancellation_mode": profile.cancellation_mode.value,
        "isolation_requirement": profile.isolation_requirement.value,
        "concurrency_class": profile.concurrency_class,
    }
