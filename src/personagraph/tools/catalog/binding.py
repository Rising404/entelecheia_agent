"""稳定 Tool 定义与动态执行绑定的合成合同。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
    thaw_json,
)
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import (
    CancellationMode,
    ExecutionMode,
    IsolationRequirement,
    ToolExecutionProfile,
    ToolHandler,
    _effect_descriptor,
    _execution_descriptor,
    _registration_descriptor,
)
from ..schema_validation import ToolSchemaCompiler


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, order=True)
class ToolIdentity:
    """一个不可变工具定义的完整版本身份。"""

    tool_id: str
    contract_version: str
    implementation_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "tool_id",
            "contract_version",
            "implementation_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value.strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_id": self.tool_id,
            "contract_version": self.contract_version,
            "implementation_version": self.implementation_version,
        }


@dataclass(frozen=True)
class ToolDefinition:
    """可持久化的稳定工具定义，不含运行时资源或可调用对象。"""

    spec: ToolSpec
    implementation_version: str
    implementation_ref: str
    implementation_digest: str
    effect_template: ToolEffectProfile
    execution_profile: ToolExecutionProfile

    def __post_init__(self) -> None:
        if (
            not isinstance(self.implementation_version, str)
            or not self.implementation_version.strip()
        ):
            raise ValueError("implementation_version must not be empty")
        if (
            not isinstance(self.implementation_ref, str)
            or not self.implementation_ref.strip()
        ):
            raise ValueError("implementation_ref must not be empty")
        object.__setattr__(
            self,
            "implementation_version",
            self.implementation_version.strip(),
        )
        object.__setattr__(
            self,
            "implementation_ref",
            self.implementation_ref.strip(),
        )
        _require_sha256(self.implementation_digest, "implementation_digest")
        compiler = ToolSchemaCompiler()
        compiler.compile(self.spec.input_schema, role="input")
        compiler.compile(self.spec.output_schema, role="output")
        # 立即验证整个持久描述符都是有限 JSON，而不是到写入时才失败。
        _canonical_sha256(self.descriptor())

    @property
    def identity(self) -> ToolIdentity:
        return ToolIdentity(
            self.spec.tool_id,
            self.spec.contract_version,
            self.implementation_version,
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "spec": self.spec.to_dict(),
            "implementation": {
                "ref": self.implementation_ref,
                "digest": self.implementation_digest,
            },
            "effect_template": _effect_descriptor(self.effect_template),
            "execution": _execution_descriptor(self.execution_profile),
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.descriptor())

    @classmethod
    def from_descriptor(cls, value: Mapping[str, Any]) -> "ToolDefinition":
        """Strictly rebuild a definition from its canonical persisted shape."""

        descriptor = _plain_json_object(value, "tool definition")
        _require_exact_keys(
            descriptor,
            {
                "identity",
                "spec",
                "implementation",
                "effect_template",
                "execution",
            },
            "tool definition",
        )
        identity_value = _required_object(descriptor, "identity")
        _require_exact_keys(
            identity_value,
            {"tool_id", "contract_version", "implementation_version"},
            "tool definition identity",
        )
        identity = ToolIdentity(
            _required_string(identity_value, "tool_id"),
            _required_string(identity_value, "contract_version"),
            _required_string(identity_value, "implementation_version"),
        )

        spec_value = _required_object(descriptor, "spec")
        _require_exact_keys(
            spec_value,
            {
                "tool_id",
                "contract_version",
                "name",
                "description",
                "input_schema",
                "output_schema",
                "catalog_tags",
            },
            "tool definition spec",
        )
        spec = ToolSpec(
            tool_id=_required_string(spec_value, "tool_id"),
            contract_version=_required_string(spec_value, "contract_version"),
            name=_required_string(spec_value, "name"),
            description=_required_string(spec_value, "description"),
            input_schema=_required_object(spec_value, "input_schema"),
            output_schema=_required_object(spec_value, "output_schema"),
            catalog_tags=_required_string_tuple(spec_value, "catalog_tags"),
        )

        implementation = _required_object(descriptor, "implementation")
        _require_exact_keys(
            implementation,
            {"ref", "digest"},
            "tool definition implementation",
        )
        execution_value = _required_object(descriptor, "execution")
        _require_exact_keys(
            execution_value,
            {
                "default_timeout_s",
                "hard_timeout_s",
                "max_output_bytes",
                "max_transparent_retries",
                "execution_mode",
                "cancellation_mode",
                "isolation_requirement",
                "concurrency_class",
            },
            "tool definition execution",
        )
        definition = cls(
            spec=spec,
            implementation_version=identity.implementation_version,
            implementation_ref=_required_string(implementation, "ref"),
            implementation_digest=_required_string(implementation, "digest"),
            effect_template=_effect_profile_from_descriptor(
                descriptor["effect_template"],
                label="tool definition effect_template",
            ),
            execution_profile=ToolExecutionProfile(
                default_timeout_s=_optional_number(
                    execution_value,
                    "default_timeout_s",
                ),
                hard_timeout_s=_optional_number(
                    execution_value,
                    "hard_timeout_s",
                ),
                max_output_bytes=_required_integer(
                    execution_value,
                    "max_output_bytes",
                ),
                max_transparent_retries=_required_integer(
                    execution_value,
                    "max_transparent_retries",
                ),
                execution_mode=ExecutionMode(
                    _required_string(execution_value, "execution_mode")
                ),
                cancellation_mode=CancellationMode(
                    _required_string(execution_value, "cancellation_mode")
                ),
                isolation_requirement=IsolationRequirement(
                    _required_string(execution_value, "isolation_requirement")
                ),
                concurrency_class=_required_string(
                    execution_value,
                    "concurrency_class",
                ),
            ),
        )
        if definition.identity != identity:
            raise ValueError("tool definition identity does not match its spec")
        if definition.descriptor() != descriptor:
            raise ValueError("tool definition descriptor is not canonical")
        return definition


@dataclass(frozen=True)
class ToolBinding:
    """把一个精确定义绑定到当前 handler、provider 与资源范围。"""

    identity: ToolIdentity
    definition_digest: str
    # source fingerprint 与安全 assertion 共同描述 provider/resource 绑定；
    # live provider 只可被 handler 闭包引用，绝不进入 descriptor。
    source: ToolSourceDescriptor
    handler: ToolHandler
    effect_profile: ToolEffectProfile
    binding_assertion: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(self.definition_digest, "definition_digest")
        if self.source.fingerprint is None:
            raise ValueError("tool binding source fingerprint must not be empty")
        if not callable(self.handler):
            raise ValueError("handler must be callable")
        if not isinstance(self.binding_assertion, Mapping):
            raise ValueError("binding_assertion must be a JSON object")
        object.__setattr__(
            self,
            "binding_assertion",
            _freeze_json(self.binding_assertion),
        )
        _canonical_sha256(self.descriptor())

    def descriptor(self) -> dict[str, Any]:
        """返回不含 handler、credential 或 live authority 的安全绑定声明。"""

        return {
            "identity": self.identity.to_dict(),
            "definition_digest": self.definition_digest,
            "source": self.source.to_dict(),
            "effects": _effect_descriptor(self.effect_profile),
            "binding_assertion": thaw_json(self.binding_assertion),
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.descriptor())


@dataclass(frozen=True)
class FrozenToolBinding:
    """Strict, handler-free form of a :class:`ToolBinding` descriptor."""

    identity: ToolIdentity
    definition_digest: str
    source: ToolSourceDescriptor
    effect_profile: ToolEffectProfile
    binding_assertion: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ToolIdentity):
            raise TypeError("frozen binding identity must be a ToolIdentity")
        _require_sha256(self.definition_digest, "definition_digest")
        if not isinstance(self.source, ToolSourceDescriptor):
            raise TypeError("frozen binding source must be a ToolSourceDescriptor")
        if self.source.fingerprint is None:
            raise ValueError("tool binding source fingerprint must not be empty")
        if not isinstance(self.effect_profile, ToolEffectProfile):
            raise TypeError("frozen binding effects must be a ToolEffectProfile")
        if not isinstance(self.binding_assertion, Mapping):
            raise ValueError("binding_assertion must be a JSON object")
        object.__setattr__(
            self,
            "binding_assertion",
            _freeze_json(self.binding_assertion),
        )
        _canonical_sha256(self.descriptor())

    @classmethod
    def from_binding(cls, binding: ToolBinding) -> "FrozenToolBinding":
        """Discard the live handler while preserving the exact descriptor."""

        if not isinstance(binding, ToolBinding):
            raise TypeError("binding must be a ToolBinding")
        return cls.from_descriptor(binding.descriptor())

    @classmethod
    def from_descriptor(
        cls,
        value: Mapping[str, Any],
    ) -> "FrozenToolBinding":
        descriptor = _plain_json_object(value, "tool binding")
        _require_exact_keys(
            descriptor,
            {
                "identity",
                "definition_digest",
                "source",
                "effects",
                "binding_assertion",
            },
            "tool binding",
        )
        identity_value = _required_object(descriptor, "identity")
        _require_exact_keys(
            identity_value,
            {"tool_id", "contract_version", "implementation_version"},
            "tool binding identity",
        )
        identity = ToolIdentity(
            _required_string(identity_value, "tool_id"),
            _required_string(identity_value, "contract_version"),
            _required_string(identity_value, "implementation_version"),
        )
        if identity.to_dict() != identity_value:
            raise ValueError("tool binding identity is not canonical")

        source = _source_from_descriptor(
            _required_object(descriptor, "source")
        )
        binding = cls(
            identity=identity,
            definition_digest=_required_string(
                descriptor,
                "definition_digest",
            ),
            source=source,
            effect_profile=_effect_profile_from_descriptor(
                descriptor["effects"],
                label="tool binding effects",
            ),
            binding_assertion=_required_object(
                descriptor,
                "binding_assertion",
            ),
        )
        if binding.descriptor() != descriptor:
            raise ValueError("tool binding descriptor is not canonical")
        return binding

    def descriptor(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "definition_digest": self.definition_digest,
            "source": self.source.to_dict(),
            "effects": _effect_descriptor(self.effect_profile),
            "binding_assertion": thaw_json(self.binding_assertion),
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.descriptor())

    def require_exact_live_binding(self, binding: ToolBinding) -> ToolBinding:
        """Return a trusted live binding only when it exactly matches this freeze."""

        if not isinstance(binding, ToolBinding):
            raise TypeError("binding must be a ToolBinding")
        if (
            binding.identity != self.identity
            or binding.definition_digest != self.definition_digest
            or binding.digest != self.digest
            or binding.descriptor() != self.descriptor()
        ):
            raise ValueError("live tool binding does not match frozen binding")
        return binding


@dataclass(frozen=True)
class BoundToolRegistration:
    """供现有执行器消费的 Definition/Binding 合成视图。"""

    definition: ToolDefinition
    binding: ToolBinding

    def __post_init__(self) -> None:
        if self.definition.identity != self.binding.identity:
            raise ValueError("tool binding identity does not match its definition")
        if self.definition.digest != self.binding.definition_digest:
            raise ValueError("tool binding definition digest does not match")
        _require_effect_binding(
            self.definition.effect_template,
            self.binding.effect_profile,
        )

    @property
    def identity(self) -> ToolIdentity:
        return self.definition.identity

    @property
    def spec(self) -> ToolSpec:
        return self.definition.spec

    @property
    def implementation_version(self) -> str:
        return self.definition.implementation_version

    @property
    def source(self) -> ToolSourceDescriptor:
        return self.binding.source

    @property
    def handler(self) -> ToolHandler:
        return self.binding.handler

    @property
    def effect_profile(self) -> ToolEffectProfile:
        return self.binding.effect_profile

    @property
    def execution_profile(self) -> ToolExecutionProfile:
        return self.definition.execution_profile

    @property
    def tool_id(self) -> str:
        return self.identity.tool_id

    @property
    def contract_version(self) -> str:
        return self.identity.contract_version

    @property
    def definition_digest(self) -> str:
        return self.definition.digest

    @property
    def binding_digest(self) -> str:
        return self.binding.digest

    def descriptor(self) -> dict[str, Any]:
        """保持现有 ``ToolRegistration.descriptor`` 的精确投影视图。"""

        return _registration_descriptor(
            spec=self.spec,
            implementation_version=self.implementation_version,
            source=self.source,
            effect_profile=self.effect_profile,
            execution_profile=self.execution_profile,
        )

    def bound_descriptor(self) -> dict[str, Any]:
        """返回供未来 execution snapshot 使用的精确合成身份。"""

        return {
            "definition": self.definition.descriptor(),
            "definition_digest": self.definition_digest,
            "binding": self.binding.descriptor(),
            "binding_digest": self.binding_digest,
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.bound_descriptor())


def _require_effect_binding(
    template: ToolEffectProfile,
    effective: ToolEffectProfile,
) -> None:
    if len(template.effects) != len(effective.effects):
        raise ValueError("tool binding effects do not match the definition template")
    for declared, bound in zip(template.effects, effective.effects, strict=True):
        declared_shape = declared.to_dict()
        bound_shape = bound.to_dict()
        declared_scope = declared_shape.pop("default_scope")
        bound_scope = bound_shape.pop("default_scope")
        if declared_shape != bound_shape:
            raise ValueError(
                "tool binding effects do not match the definition template"
            )
        if declared_scope != "*" and bound_scope != declared_scope:
            raise ValueError(
                "tool binding scope exceeds the definition effect template"
            )


def _effect_profile_from_descriptor(
    value: object,
    *,
    label: str,
) -> ToolEffectProfile:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    effects: list[EffectDescriptor] = []
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label} items must be JSON objects")
        _require_exact_keys(
            item,
            {
                "resource",
                "action",
                "scope_kind",
                "default_scope",
                "data_egress",
                "idempotency",
                "reversibility",
                "resource_argument",
                "scope_argument",
                "egress_arguments",
            },
            f"{label}[{position}]",
        )
        effects.append(
            EffectDescriptor(
                resource=EffectResource(_required_string(item, "resource")),
                action=EffectAction(_required_string(item, "action")),
                scope_kind=EffectScopeKind(
                    _required_string(item, "scope_kind")
                ),
                default_scope=_required_string(item, "default_scope"),
                data_egress=DataEgress(
                    _required_string(item, "data_egress")
                ),
                idempotency=Idempotency(
                    _required_string(item, "idempotency")
                ),
                reversibility=Reversibility(
                    _required_string(item, "reversibility")
                ),
                resource_argument=_optional_string(item, "resource_argument"),
                scope_argument=_optional_string(item, "scope_argument"),
                egress_arguments=_required_string_tuple(
                    item,
                    "egress_arguments",
                ),
            )
        )
    profile = ToolEffectProfile(tuple(effects))
    if _effect_descriptor(profile) != value:
        raise ValueError(f"{label} is not canonical")
    return profile


def _source_from_descriptor(value: dict[str, Any]) -> ToolSourceDescriptor:
    _require_exact_keys(
        value,
        {"kind", "source_id", "fingerprint", "display_name"},
        "tool binding source",
    )
    source = ToolSourceDescriptor(
        kind=ToolSourceKind(_required_string(value, "kind")),
        source_id=_required_string(value, "source_id"),
        fingerprint=_optional_string(value, "fingerprint"),
        display_name=_optional_string(value, "display_name"),
    )
    if source.to_dict() != value:
        raise ValueError("tool binding source is not canonical")
    return source


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical SHA-256 digest")


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("tool descriptor must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("binding_assertion keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("binding_assertion must contain only finite JSON values")


def _plain_json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a plain JSON object")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must contain only finite JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields do not match the canonical contract: "
            f"missing={missing!r}, unknown={unknown!r}"
        )


def _required_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value[key]
    if not isinstance(selected, dict):
        raise ValueError(f"{key} must be a JSON object")
    return selected


def _required_string(value: dict[str, Any], key: str) -> str:
    selected = value[key]
    if not isinstance(selected, str):
        raise ValueError(f"{key} must be a string")
    return selected


def _optional_string(value: dict[str, Any], key: str) -> str | None:
    selected = value[key]
    if selected is not None and not isinstance(selected, str):
        raise ValueError(f"{key} must be a string or null")
    return selected


def _required_string_tuple(value: dict[str, Any], key: str) -> tuple[str, ...]:
    selected = value[key]
    if not isinstance(selected, list) or any(
        not isinstance(item, str) for item in selected
    ):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(selected)


def _required_integer(value: dict[str, Any], key: str) -> int:
    selected = value[key]
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise ValueError(f"{key} must be an integer")
    return selected


def _optional_number(value: dict[str, Any], key: str) -> float | int | None:
    selected = value[key]
    if selected is None:
        return None
    if isinstance(selected, bool) or not isinstance(selected, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    if isinstance(selected, float) and not math.isfinite(selected):
        raise ValueError(f"{key} must be finite")
    return selected
