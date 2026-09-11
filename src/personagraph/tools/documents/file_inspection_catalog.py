"""文档查询工具的持久定义与会话文件权限绑定。"""

from dataclasses import dataclass
import hashlib
import json

from ..catalog.binding import BoundToolRegistration, ToolBinding, ToolDefinition
from .file_inspection_tools import FILE_INSPECTION_TOOL_IDS, build_file_inspection_registration


@dataclass(frozen=True, slots=True)
class FileInspectionDefinition:
    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition


def file_inspection_definition_manifest():
    result = []
    for tool_id in FILE_INSPECTION_TOOL_IDS:
        surface = build_file_inspection_registration(tool_id=tool_id, handler=_definition_only, effect_scope="*")
        ref, revision = f"builtin/{tool_id}", f"{tool_id}-handler-1"
        result.append(FileInspectionDefinition(ref, revision, ToolDefinition(
            spec=surface.spec, implementation_version=surface.implementation_version,
            implementation_ref=ref,
            implementation_digest=hashlib.sha256(json.dumps(
                [revision, surface.descriptor()], sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest(),
            effect_template=surface.effect_profile, execution_profile=surface.execution_profile,
        )))
    return tuple(result)


def bind_file_inspection_tools(registrations, *, scope_sha256, authority_sha256):
    for value in (scope_sha256, authority_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("file inspection binding requires authority digests")
    result = []
    for registration, item in zip(registrations, file_inspection_definition_manifest(), strict=True):
        scopes = {effect.default_scope for effect in registration.effect_profile.effects}
        if len(scopes) != 1 or hashlib.sha256(next(iter(scopes)).encode()).hexdigest() != scope_sha256:
            raise ValueError("file inspection scope drifted")
        surface = build_file_inspection_registration(tool_id=registration.tool_id,
                                                    handler=_definition_only, effect_scope="*")
        if registration.source != surface.source:
            raise ValueError("file inspection source drifted")
        binding = ToolBinding(
            identity=item.definition.identity, definition_digest=item.definition.digest,
            source=registration.source, handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={"schema_version": "file-inspection-binding-v1",
                               "scope_sha256": scope_sha256, "authority_sha256": authority_sha256},
        )
        if BoundToolRegistration(item.definition, binding).descriptor() != registration.descriptor():
            raise ValueError("file inspection registration drifted")
        result.append(binding)
    return tuple(result)


def _definition_only(_payload):
    raise RuntimeError("a tool definition cannot execute")
