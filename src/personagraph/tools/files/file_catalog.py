"""文件检查/准备的全局定义与当前来源绑定；不持久化第二份文件身份。"""

from dataclasses import dataclass, replace
import hashlib
import json

from ..catalog.binding import BoundToolRegistration, ToolBinding, ToolDefinition
from ..effects import ToolEffectProfile
from .file_tools import CHECK_FILES_STATE_TOOL_ID, PREPARE_FILES_TOOL_ID, build_file_state_registration


@dataclass(frozen=True, slots=True)
class FileToolDefinitionManifestItem:
    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition


def file_tool_definition_manifest() -> tuple[FileToolDefinitionManifestItem, ...]:
    result = []
    for tool_id in (CHECK_FILES_STATE_TOOL_ID, PREPARE_FILES_TOOL_ID):
        registration = build_file_state_registration(
            tool_id=tool_id, handler=_definition_only, effect_scope="*",
        )
        ref = f"builtin/{tool_id}"
        revision = f"{tool_id}-handler-{registration.implementation_version}"
        digest = _digest({"revision": revision, "registration": registration.descriptor()})
        result.append(FileToolDefinitionManifestItem(ref, revision, ToolDefinition(
            spec=registration.spec, implementation_version=registration.implementation_version,
            implementation_ref=ref, implementation_digest=digest,
            effect_template=registration.effect_profile, execution_profile=registration.execution_profile,
        )))
    return tuple(result)


def bind_file_state_tools(
    registrations, *, authority_sha256: str, generation_sha256: str, scope_sha256: str,
):
    """权威/配方摘要进入 Binding，模型参数和 Host 路径不进入默认定义。"""
    for value in (authority_sha256, generation_sha256, scope_sha256):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("binding facts require sha256")
    bindings = []
    for registration, item in zip(registrations, file_tool_definition_manifest(), strict=True):
        definition = item.definition
        scopes = {effect.default_scope for effect in registration.effect_profile.effects}
        if len(scopes) != 1 or hashlib.sha256(next(iter(scopes)).encode()).hexdigest() != scope_sha256:
            raise ValueError("file state authority scope drifted")
        normalized = ToolEffectProfile(tuple(
            replace(effect, default_scope="*") for effect in registration.effect_profile.effects
        ))
        if (
            registration.spec != definition.spec
            or registration.implementation_version != definition.implementation_version
            or registration.execution_profile != definition.execution_profile
            or normalized != definition.effect_template
        ):
            raise ValueError("file state registration drifted from its definition")
        assertion = {"schema_version": "file-state-binding-v1",
                     "authority_sha256": authority_sha256, "generation_sha256": generation_sha256,
                     "scope_sha256": scope_sha256}
        binding = ToolBinding(
            identity=definition.identity, definition_digest=definition.digest,
            source=replace(registration.source, fingerprint=_digest(assertion)),
            handler=registration.handler, effect_profile=registration.effect_profile,
            binding_assertion=assertion,
        )
        BoundToolRegistration(definition, binding)
        bindings.append(binding)
    return tuple(bindings)


def _definition_only(_payload):
    raise RuntimeError("file tool definition is not executable")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
