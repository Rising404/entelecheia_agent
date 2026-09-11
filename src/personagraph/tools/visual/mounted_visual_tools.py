"""Stable model-visible Tool for reading a frozen mounted visual scope.

The owner is lane neutral: callers inject an immutable alias scope, a narrow
freshness port, and an adapter whose egress behavior is explicit.  L2 remains
responsible for deriving that scope from planning authority and for wrapping
external adapters with its durable physical-call ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Protocol, TypeVar

from ...input_processing.vision.providers import VisionModelAdapter
from ...input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionDetail,
    VisionPurpose,
    VisionRegion,
    VisionStatus,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..execution import ToolBusinessFailure
from ..registration import ToolRegistration
from .failure_reporting import failure_diagnostics_schema
from .question_contract import (
    parse_visual_question,
    visual_call_identity,
    visual_question_constraint,
    visual_question_schema,
)
from .visual_observation_service import (
    MAX_UNITS_PER_CALL,
    VisualObservationRequest,
    VisualObservationService,
)
from .visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef
from .visual_tools import (
    visual_observation_effect_profile,
    visual_observation_execution_profile,
)


MOUNTED_VISUAL_TOOL_ID = "read_mounted_visuals"
MOUNTED_VISUAL_CONTRACT_VERSION = "mounted-visual-cognition-v1"
MOUNTED_VISUAL_IMPLEMENTATION_VERSION = "1"
MOUNTED_VISUAL_SOURCE_ID = "personagraph.tools.visual.mounted-visual"
MOUNTED_VISUAL_SOURCE_DISPLAY_NAME = "Task-authorized mounted visuals"

_MOUNTED_VISUAL_ALIAS = re.compile(r"mounted_visual_[0-9]+\Z")
_VisionEnum = TypeVar(
    "_VisionEnum",
    VisionDetail,
    VisionPurpose,
    VisionRegion,
)


@dataclass(frozen=True, slots=True)
class MountedVisualToolBinding:
    """Private alias binding required by the stable Tool handler."""

    visual_alias: str
    document_alias: str
    visual_unit: VisualUnitRef

    def __post_init__(self) -> None:
        if not _MOUNTED_VISUAL_ALIAS.fullmatch(self.visual_alias):
            raise ValueError("visual_alias must be an opaque mounted visual alias")
        if not isinstance(self.document_alias, str) or not self.document_alias:
            raise ValueError("document_alias must not be empty")
        if not isinstance(self.visual_unit, VisualUnitRef):
            raise TypeError("visual_unit must be a VisualUnitRef")


@dataclass(frozen=True, slots=True)
class FrozenMountedVisualToolScope:
    """Complete lane-neutral visual scope injected by a Host adapter."""

    session_id: str
    scope_snapshot_sha256: str
    bindings: tuple[MountedVisualToolBinding, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or len(self.session_id) > 200
        ):
            raise ValueError("session_id must be a bounded durable identity")
        if (
            not isinstance(self.scope_snapshot_sha256, str)
            or len(self.scope_snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.scope_snapshot_sha256
            )
        ):
            raise ValueError(
                "scope_snapshot_sha256 must be a canonical SHA-256 digest"
            )
        if not isinstance(self.bindings, tuple) or any(
            not isinstance(binding, MountedVisualToolBinding)
            for binding in self.bindings
        ):
            raise TypeError("bindings must be a tuple of MountedVisualToolBinding")
        aliases = tuple(binding.visual_alias for binding in self.bindings)
        if len(aliases) != len(set(aliases)):
            raise ValueError("mounted visual aliases must be unique")


class MountedVisualFreshnessPort(Protocol):
    """Narrow Host-owned check for an injected frozen alias binding."""

    def is_current(self, binding: MountedVisualToolBinding) -> bool: ...


@dataclass(frozen=True, slots=True)
class MountedVisualToolSource:
    """Stable registration plus immutable execution metadata."""

    session_id: str
    registration: ToolRegistration
    scope_snapshot_sha256: str
    capabilities: VisionCapabilitySnapshot
    transmits_externally: bool


def mounted_visual_tool_spec() -> ToolSpec:
    """Return the scope-independent wire contract."""

    return ToolSpec(
        tool_id=MOUNTED_VISUAL_TOOL_ID,
        contract_version=MOUNTED_VISUAL_CONTRACT_VERSION,
        name="Read mounted visuals",
        description=(
            "按目录提供的 visual_alias 读取最多三个已挂载视觉区域，选择允许的用途。"
            "purpose=question 时在 question 中提供希望结合图像回答的具体自然语言问题；"
            "其它用途省略 question 或设为 null。首次观察不足时可提高 detail 或扩大 region。"
        ),
        input_schema=_input_schema(),
        output_schema=_output_schema(),
        catalog_tags=("document", "read"),
    )


def build_mounted_visual_tool_source(
    scope: FrozenMountedVisualToolScope,
    *,
    adapter: VisionModelAdapter,
    freshness: MountedVisualFreshnessPort,
    source_fingerprint: str | None = None,
) -> MountedVisualToolSource | None:
    """Build one registration without discovering Host or L2 state.

    ``source_fingerprint`` lets the Host commit all private execution facts to a
    contextual Tool Binding.  The legacy scope digest remains the default until
    that Binding is wired into every caller; neither value changes the stable
    implementation identity.
    """

    if not isinstance(scope, FrozenMountedVisualToolScope):
        raise TypeError("scope must be FrozenMountedVisualToolScope")
    if not callable(getattr(freshness, "is_current", None)):
        raise TypeError("freshness must implement MountedVisualFreshnessPort")
    if source_fingerprint is not None and (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in source_fingerprint)
    ):
        raise ValueError("source_fingerprint must be a canonical SHA-256 digest")
    if not scope.bindings:
        return None

    service = VisualObservationService(
        adapter=adapter,
    )
    registration = ToolRegistration(
        spec=mounted_visual_tool_spec(),
        implementation_version=MOUNTED_VISUAL_IMPLEMENTATION_VERSION,
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL,
            source_id=MOUNTED_VISUAL_SOURCE_ID,
            fingerprint=source_fingerprint or scope.scope_snapshot_sha256,
            display_name=MOUNTED_VISUAL_SOURCE_DISPLAY_NAME,
        ),
        handler=_MountedVisualToolRuntime(
            scope=scope,
            freshness=freshness,
            service=service,
        ).read,
        effect_profile=visual_observation_effect_profile(
            scope.session_id,
            service.transmits_externally,
        ),
        execution_profile=visual_observation_execution_profile(),
    )
    return MountedVisualToolSource(
        session_id=scope.session_id,
        registration=registration,
        scope_snapshot_sha256=scope.scope_snapshot_sha256,
        capabilities=service.capabilities,
        transmits_externally=service.transmits_externally,
    )


class _MountedVisualToolRuntime:
    def __init__(
        self,
        *,
        scope: FrozenMountedVisualToolScope,
        freshness: MountedVisualFreshnessPort,
        service: VisualObservationService,
    ) -> None:
        self._scope = scope
        self._freshness = freshness
        self._service = service
        self._bindings_by_alias = {
            binding.visual_alias: binding for binding in scope.bindings
        }

    def read(self, payload: dict[str, Any]) -> dict[str, Any]:
        selected = self._parse_bindings(payload)
        detail = _vision_enum(
            VisionDetail,
            payload.get("detail", VisionDetail.STANDARD.value),
            field_name="detail",
        )
        region = _vision_enum(
            VisionRegion,
            payload.get("region", VisionRegion.DETECTED.value),
            field_name="region",
        )

        results: list[dict[str, Any]] = []
        for _, purpose, _ in selected:
            visual_call_identity(purpose)
        resolved = 0
        for binding, purpose, question in selected:
            observation = self._service.observe(
                FrozenVisualToolBoundary(
                    session_id=self._scope.session_id,
                    units=(binding.visual_unit,),
                ),
                (
                    VisualObservationRequest(
                        unit_id=binding.visual_unit.unit_id,
                        purpose=purpose,
                        detail=detail,
                        region=region,
                        question=question,
                    ),
                ),
            ).results[0]
            page = binding.visual_unit.locator.page
            projected = {
                "visual_alias": binding.visual_alias,
                "document_alias": binding.document_alias,
                "source_pages": [page] if page is not None else [],
                "kind": binding.visual_unit.kind.value,
            }
            projected.update(
                {
                    str(key): value
                    for key, value in observation.to_dict().items()
                    if key != "unit_id"
                }
            )
            results.append(projected)
            resolved += int(observation.status is VisionStatus.COMPLETED)
        return {
            "results": results,
            "requested": len(results),
            "resolved": resolved,
        }

    def _parse_bindings(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[tuple[MountedVisualToolBinding, VisionPurpose, str | None], ...]:
        requested = payload.get("visuals")
        if not isinstance(requested, list) or not requested:
            raise ToolBusinessFailure(
                "invalid_request",
                "visuals must be a non-empty list of frozen visual aliases",
            )
        if len(requested) > MAX_UNITS_PER_CALL:
            raise ToolBusinessFailure(
                "invalid_request",
                f"at most {MAX_UNITS_PER_CALL} visuals may be read at once",
            )

        selected: list[tuple[MountedVisualToolBinding, VisionPurpose, str | None]] = []
        for item in requested:
            if not isinstance(item, Mapping):
                raise ToolBusinessFailure("invalid_request", "visual item is invalid")
            alias = str(item.get("visual_alias") or "")
            binding = self._bindings_by_alias.get(alias)
            if binding is None:
                raise ToolBusinessFailure(
                    "unknown_mounted_visual_alias",
                    "The requested visual alias is outside the frozen authority.",
                )
            if not self._freshness.is_current(binding):
                raise ToolBusinessFailure(
                    "mounted_visual_authority_drift",
                    "The frozen mounted visual changed after authority was frozen.",
                )
            purpose = _vision_enum(
                VisionPurpose,
                item.get("purpose"),
                field_name="purpose",
            )
            if purpose not in binding.visual_unit.allowed_purposes:
                allowed = ", ".join(
                    value.value for value in binding.visual_unit.allowed_purposes
                )
                raise ToolBusinessFailure(
                    "purpose_not_allowed",
                    f"{binding.visual_unit.unit_id} is a "
                    f"{binding.visual_unit.kind.value}; allowed purposes are "
                    f"{allowed}",
                )
            selected.append((binding, purpose, parse_visual_question(purpose, item.get("question"))))
        return tuple(selected)


def _vision_enum(
    enum_type: type[_VisionEnum],
    raw: object,
    *,
    field_name: str,
) -> _VisionEnum:
    try:
        return enum_type(str(raw or "").strip())
    except ValueError as exc:
        raise ToolBusinessFailure(
            "invalid_request",
            f"unknown visual {field_name}",
        ) from exc


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visuals"],
        "properties": {
            "visuals": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_UNITS_PER_CALL,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["visual_alias", "purpose"],
                    "allOf": [visual_question_constraint()],
                    "properties": {
                        "visual_alias": {
                            "type": "string",
                            "pattern": "^mounted_visual_[0-9]+$",
                        },
                        "purpose": {
                            "enum": [item.value for item in VisionPurpose]
                        },
                        "question": visual_question_schema(),
                    },
                },
            },
            "detail": {
                "enum": [item.value for item in VisionDetail],
                "default": VisionDetail.STANDARD.value,
            },
            "region": {
                "enum": [item.value for item in VisionRegion],
                "default": VisionRegion.DETECTED.value,
            },
        },
    }


def _output_schema() -> dict[str, Any]:
    result_properties: dict[str, Any] = {
        "visual_alias": {"type": "string"},
        "document_alias": {"type": "string"},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "kind": {"type": "string"},
        "purpose": {"enum": [item.value for item in VisionPurpose]},
        "question": visual_question_schema(),
        "region": {"enum": [item.value for item in VisionRegion]},
        "status": {"enum": [item.value for item in VisionStatus]},
        "at": {"type": "string"},
        "observation": {"type": "string"},
        "observation_id": {"type": "string"},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "failure_code": {"type": "string"},
        "failure_diagnostics": failure_diagnostics_schema(),
        "resampled": {"type": "boolean"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results", "requested", "resolved"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "visual_alias",
                        "document_alias",
                        "source_pages",
                        "kind",
                        "purpose",
                        "region",
                        "status",
                        "at",
                    ],
                    "properties": result_properties,
                },
            },
            "requested": {"type": "integer", "minimum": 1},
            "resolved": {"type": "integer", "minimum": 0},
        },
    }


__all__ = [
    "FrozenMountedVisualToolScope",
    "MOUNTED_VISUAL_CONTRACT_VERSION",
    "MOUNTED_VISUAL_IMPLEMENTATION_VERSION",
    "MOUNTED_VISUAL_SOURCE_ID",
    "MOUNTED_VISUAL_SOURCE_DISPLAY_NAME",
    "MOUNTED_VISUAL_TOOL_ID",
    "MountedVisualFreshnessPort",
    "MountedVisualToolBinding",
    "MountedVisualToolSource",
    "build_mounted_visual_tool_source",
    "mounted_visual_tool_spec",
]
