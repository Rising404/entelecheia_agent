"""工具历史的唯一持久定义与只读执行端口绑定。"""

from dataclasses import dataclass
from collections.abc import Iterable
import hashlib
import json
import re
from typing import Any, NoReturn

from ..catalog.binding import BoundToolRegistration, ToolBinding, ToolDefinition
from ..registration import ToolRegistration
from .definitions import TOOL_HISTORY_TOOL_IDS, build_tool_history_registration


@dataclass(frozen=True, slots=True)
class ToolHistoryDefinition:
    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition


def tool_history_definition_manifest() -> tuple[ToolHistoryDefinition, ...]:
    items = []
    for tool_id in TOOL_HISTORY_TOOL_IDS:
        surface = build_tool_history_registration(
            tool_id=tool_id, handler=_definition_only, effect_scope="*"
        )
        ref, revision = f"builtin/{tool_id}", f"{tool_id}-handler-2"
        items.append(
            ToolHistoryDefinition(
                ref,
                revision,
                ToolDefinition(
                    spec=surface.spec,
                    implementation_version=surface.implementation_version,
                    implementation_ref=ref,
                    implementation_digest=hashlib.sha256(
                        json.dumps(
                            [revision, surface.descriptor()],
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    effect_template=surface.effect_profile,
                    execution_profile=surface.execution_profile,
                ),
            )
        )
    return tuple(items)


def bind_tool_history_registrations(
    registrations: Iterable[ToolRegistration],
    *,
    authority_sha256: str,
) -> tuple[ToolBinding, ...]:
    if not isinstance(authority_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", authority_sha256
    ):
        raise ValueError("tool history binding requires a frozen authority digest")
    registrations = tuple(registrations)
    if tuple(item.tool_id for item in registrations) != TOOL_HISTORY_TOOL_IDS:
        raise ValueError("tool history registration order drifted")
    bindings = []
    for registration, item in zip(
        registrations, tool_history_definition_manifest(), strict=True
    ):
        scopes = {
            effect.default_scope for effect in registration.effect_profile.effects
        }
        if len(scopes) != 1 or "*" in scopes:
            raise ValueError("tool history must be bound to one execution")
        binding = ToolBinding(
            identity=item.definition.identity,
            definition_digest=item.definition.digest,
            source=registration.source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "schema_version": "tool-history-binding-v1",
                "authority_sha256": authority_sha256,
                "effect_scope_sha256": hashlib.sha256(
                    next(iter(scopes)).encode()
                ).hexdigest(),
            },
        )
        if (
            BoundToolRegistration(item.definition, binding).descriptor()
            != registration.descriptor()
        ):
            raise ValueError("tool history registration drifted")
        bindings.append(binding)
    return tuple(bindings)


def _definition_only(_payload: dict[str, Any]) -> NoReturn:
    raise RuntimeError("a tool definition cannot execute")
