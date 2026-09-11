"""Adapt L2 mounted-visual authority to the lane-neutral visual Tool owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from ....input_processing.vision.providers import (
    VisionModelAdapter,
    vision_adapter_transmits_externally,
)
from ....input_processing.vision.contracts import VisionCapabilitySnapshot
from ....tools.visual.egress_policy import auto_visual_egress_receipt
from ....runtime.model_calls.vision import (
    DurableMountedVisionAdapter,
    SqliteMountedVisualCallLedger,
)
from ....tools.policy import (
    ProtectedToolExecutionAuthority,
)
from ....tools.effects import EffectAction, EffectResource, EffectScopeKind
from ....tools.catalog.binding import ToolBinding
from ....tools.policy import AuthorityFacts, ScopeGrant
from ....tools.registration import ToolRegistration
from ....tools.visual.mounted_visual_catalog import (
    ExternalMountedVisualBindingFacts,
    build_mounted_visual_tool_bindings,
    derive_mounted_visual_source_fingerprint,
)
from ....tools.visual.mounted_visual_tools import (
    FrozenMountedVisualToolScope,
    MountedVisualToolBinding,
    build_mounted_visual_tool_source,
)
from ....tools.visual.visual_observation_service import default_vision_adapter
from ...auxiliary_execution.planning.mounted_document_authority import (
    FrozenMountedDocumentPlanningAuthority,
)
from ...auxiliary_execution.planning.mounted_visual_resource import (
    FrozenMountedVisualPlanningBinding,
    mounted_visual_binding_is_current,
)


class MountedVisualToolRuntimeError(RuntimeError):
    """The frozen mounted-visual Tool runtime cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class SessionMountedVisualToolRuntime:
    """One registration plus the Host authority needed by the WorkRun bridge."""

    session_id: str
    registration: ToolRegistration
    authority: AuthorityFacts
    contextual_bindings: tuple[ToolBinding, ...] = ()
    protected_authority_by_key: Mapping[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = field(default_factory=lambda: MappingProxyType({}))


def build_session_mounted_visual_tool_runtime(
    authority: FrozenMountedDocumentPlanningAuthority,
    bindings: tuple[FrozenMountedVisualPlanningBinding, ...] | None = None,
    *,
    adapter: VisionModelAdapter | None = None,
    call_ledger: SqliteMountedVisualCallLedger | None = None,
) -> SessionMountedVisualToolRuntime | None:
    """Project exact L2 authority and add Host-only egress protections."""

    if not isinstance(authority, FrozenMountedDocumentPlanningAuthority):
        raise TypeError(
            "authority must be FrozenMountedDocumentPlanningAuthority"
        )
    selected = authority.visual_bindings if bindings is None else bindings
    if not isinstance(selected, tuple):
        raise TypeError("bindings must be a tuple")
    if not selected:
        return None

    frozen_by_alias = {
        item.resource.resource_alias: item for item in authority.visual_bindings
    }
    aliases = tuple(item.resource.resource_alias for item in selected)
    if len(aliases) != len(set(aliases)):
        raise MountedVisualToolRuntimeError(
            "mounted visual aliases must be unique"
        )
    if any(
        item.resource.session_id != authority.session_id
        or frozen_by_alias.get(item.resource.resource_alias) != item
        for item in selected
    ):
        raise MountedVisualToolRuntimeError(
            "mounted visual bindings are outside the frozen authority"
        )

    projected = tuple(
        MountedVisualToolBinding(
            visual_alias=binding.resource.resource_alias,
            document_alias=binding.parent_document_alias,
            visual_unit=binding.visual_unit,
        )
        for binding in selected
    )
    freshness = _PlanningMountedVisualFreshnessPort(
        MappingProxyType(
            {
                tool_binding.visual_alias: (tool_binding, planning_binding)
                for tool_binding, planning_binding in zip(
                    projected,
                    selected,
                    strict=True,
                )
            }
        )
    )
    scope = FrozenMountedVisualToolScope(
        session_id=authority.session_id,
        scope_snapshot_sha256=_source_scope_fingerprint(
            authority,
            selected,
        ),
        bindings=projected,
    )

    resolved_adapter = adapter if adapter is not None else default_vision_adapter()
    transmits = vision_adapter_transmits_externally(resolved_adapter)
    capabilities = resolved_adapter.capabilities()
    if not isinstance(capabilities, VisionCapabilitySnapshot):
        raise TypeError("vision adapter returned an invalid capability snapshot")
    capability_bound_adapter = _CapabilityBoundVisionAdapter(
        resolved_adapter,
        capabilities,
        transmits_externally=transmits,
    )
    resolved_call_ledger = (
        call_ledger
        if call_ledger is not None
        else SqliteMountedVisualCallLedger()
    )
    runtime_adapter: VisionModelAdapter = capability_bound_adapter
    if transmits:
        runtime_adapter = DurableMountedVisionAdapter(
            capability_bound_adapter,
            session_id=authority.session_id,
            ledger=resolved_call_ledger,
        )
    disclosure_receipt_ids = (
        _mounted_visual_egress_receipts(
            session_id=authority.session_id,
            bindings=selected,
            capabilities=capabilities,
        )
        if transmits and capabilities.available
        else None
    )
    binding_facts = (
        _mounted_visual_binding_facts(
            authority=authority,
            selected=selected,
            scope=scope,
            capabilities=capabilities,
            disclosure_receipt_ids=disclosure_receipt_ids,
            call_ledger=resolved_call_ledger,
        )
        if transmits
        and capabilities.available
        and disclosure_receipt_ids is not None
        else None
    )
    tool_source = build_mounted_visual_tool_source(
        scope,
        adapter=runtime_adapter,
        freshness=freshness,
        source_fingerprint=(
            None
            if binding_facts is None
            else derive_mounted_visual_source_fingerprint(binding_facts)
        ),
    )
    if tool_source is None:
        raise AssertionError("a non-empty mounted visual scope must build a Tool")
    if (
        tool_source.capabilities != capabilities
        or tool_source.transmits_externally is not transmits
    ):
        raise MountedVisualToolRuntimeError(
            "mounted visual capability snapshot drifted during composition"
        )
    contextual_bindings = (
        build_mounted_visual_tool_bindings(
            (tool_source.registration,),
            facts=binding_facts,
        )
        if binding_facts is not None
        else ()
    )
    runtime_authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.SESSION,
                authority.session_id,
            ),
        )
        if not transmits
        else (
            ScopeGrant(
                EffectResource.NETWORK,
                EffectAction.TRANSMIT,
                EffectScopeKind.SESSION,
                authority.session_id,
            ),
        )
        if transmits
        and disclosure_receipt_ids is not None
        else (),
    )
    protected_authority_by_key: Mapping[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = MappingProxyType({})
    if transmits and capabilities.available and disclosure_receipt_ids is not None:
        protected_authority_by_key = MappingProxyType(
            {
                (
                    tool_source.registration.tool_id,
                    tool_source.registration.contract_version,
                ): ProtectedToolExecutionAuthority(
                    approval_receipt_ids=disclosure_receipt_ids,
                    execution_backend_identity_sha256=(
                        _provider_identity_sha256(tool_source.capabilities)
                    ),
                    revalidate=_mounted_visual_authority_is_current(
                        session_id=authority.session_id,
                        bindings=selected,
                        capabilities=tool_source.capabilities,
                        expected_receipt_ids=disclosure_receipt_ids,
                    ),
                )
            }
        )
    return SessionMountedVisualToolRuntime(
        session_id=tool_source.session_id,
        registration=tool_source.registration,
        authority=runtime_authority,
        contextual_bindings=contextual_bindings,
        protected_authority_by_key=protected_authority_by_key,
    )


@dataclass(frozen=True, slots=True)
class _CapabilityBoundVisionAdapter:
    """Keep policy, Binding, disclosure, and execution on one capability read."""

    delegate: VisionModelAdapter
    capability_snapshot: VisionCapabilitySnapshot
    transmits_externally: bool

    def capabilities(self) -> VisionCapabilitySnapshot:
        return self.capability_snapshot

    def analyze(self, request):
        return self.delegate.analyze(request)


@dataclass(frozen=True, slots=True)
class _PlanningMountedVisualFreshnessPort:
    bindings_by_alias: Mapping[
        str,
        tuple[
            MountedVisualToolBinding,
            FrozenMountedVisualPlanningBinding,
        ],
    ]

    def is_current(self, binding: MountedVisualToolBinding) -> bool:
        pair = self.bindings_by_alias.get(binding.visual_alias)
        return bool(
            pair is not None
            and pair[0] == binding
            and mounted_visual_binding_is_current(pair[1])
        )


def _mounted_visual_egress_receipts(
    *,
    session_id: str,
    bindings: tuple[FrozenMountedVisualPlanningBinding, ...],
    capabilities: VisionCapabilitySnapshot,
) -> tuple[str, ...] | None:
    """Resolve every receipt needed by a model-selectable visual operation."""

    receipt_ids: list[str] = []
    for binding in bindings:
        unit = binding.visual_unit
        for purpose in unit.allowed_purposes:
            receipt_id = auto_visual_egress_receipt(
                session_id=session_id,
                source_sha256=unit.source_sha256,
                endpoint_identity=capabilities.endpoint_identity,
                model=capabilities.model,
                purpose=purpose,
            )
            if receipt_id not in receipt_ids:
                receipt_ids.append(receipt_id)
    return tuple(receipt_ids)


def _mounted_visual_authority_is_current(
    *,
    session_id: str,
    bindings: tuple[FrozenMountedVisualPlanningBinding, ...],
    capabilities: VisionCapabilitySnapshot,
    expected_receipt_ids: tuple[str, ...],
) -> Callable[[], bool]:
    def revalidate() -> bool:
        try:
            if any(
                not mounted_visual_binding_is_current(binding)
                for binding in bindings
            ):
                return False
            current_receipt_ids = _mounted_visual_egress_receipts(
                session_id=session_id,
                bindings=bindings,
                capabilities=capabilities,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return current_receipt_ids == expected_receipt_ids

    return revalidate


def _mounted_visual_binding_facts(
    *,
    authority: FrozenMountedDocumentPlanningAuthority,
    selected: tuple[FrozenMountedVisualPlanningBinding, ...],
    scope: FrozenMountedVisualToolScope,
    capabilities: VisionCapabilitySnapshot,
    disclosure_receipt_ids: tuple[str, ...] | None,
    call_ledger: SqliteMountedVisualCallLedger,
) -> ExternalMountedVisualBindingFacts:
    """Commit the exact private L2 execution route without exposing its values."""

    session_scope_sha256 = _sha256_text(scope.session_id)
    return ExternalMountedVisualBindingFacts(
        session_scope_sha256=session_scope_sha256,
        visual_scope_snapshot_sha256=scope.scope_snapshot_sha256,
        visual_alias_projection_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-alias-projection-binding-v1",
                "bindings": [
                    {
                        "visual_alias_sha256": _sha256_text(binding.visual_alias),
                        "document_alias_sha256": _sha256_text(
                            binding.document_alias
                        ),
                        "unit_id_sha256": _sha256_text(
                            binding.visual_unit.unit_id
                        ),
                    }
                    for binding in scope.bindings
                ],
            }
        ),
        mounted_authority_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-l2-authority-binding-v1",
                "session_scope_sha256": session_scope_sha256,
                "task_sha256": _optional_text_sha256(
                    getattr(authority, "task_id", None),
                    field_name="authority.task_id",
                ),
                "authority_scope_snapshot_sha256": (
                    authority.scope_snapshot_sha256
                ),
                "selected_scope_snapshot_sha256": scope.scope_snapshot_sha256,
                "selected_alias_sha256": [
                    _sha256_text(binding.resource.resource_alias)
                    for binding in selected
                ],
            }
        ),
        freshness_authority_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-freshness-authority-binding-v1",
                "freshness_recipe": "mounted-visual-binding-is-current@1",
                "bindings": [
                    _mounted_visual_freshness_payload(binding)
                    for binding in selected
                ],
            }
        ),
        provider_identity_sha256=_provider_identity_sha256(capabilities),
        capability_snapshot_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-capability-binding-v1",
                "available": capabilities.available,
                "provider": capabilities.provider,
                "model": capabilities.model,
                "endpoint_identity": capabilities.endpoint_identity,
                "processor_fingerprint": capabilities.processor_fingerprint,
                "supported_purposes": [
                    purpose.value
                    for purpose in capabilities.supported_purposes
                ],
                "reason_code": capabilities.reason_code,
            }
        ),
        egress_policy_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-disclosure-authority-binding-v1",
                "session_scope_sha256": session_scope_sha256,
                "egress_policy": "default-allow-with-exact-source-purpose",
                "authorization_complete": disclosure_receipt_ids is not None,
                "receipt_id_sha256": (
                    None
                    if disclosure_receipt_ids is None
                    else [
                        _sha256_text(receipt_id)
                        for receipt_id in disclosure_receipt_ids
                    ]
                ),
            }
        ),
        physical_call_ledger_sha256=_sha256_value(
            {
                "schema_version": "mounted-visual-physical-ledger-binding-v1",
                "session_scope_sha256": session_scope_sha256,
                "ledger_backend": _backend_name(call_ledger),
                "ledger_path_sha256": _sha256_text(
                    str(call_ledger.path_for(scope.session_id).resolve())
                ),
                "dispatch_recipe": "mounted-visual-provider-call-ledger@1",
            }
        ),
    )


def _mounted_visual_freshness_payload(
    binding: FrozenMountedVisualPlanningBinding,
) -> dict[str, object]:
    """Describe every value consumed by mounted-visual freshness checks."""

    unit = binding.visual_unit
    locator = unit.locator
    anchor = getattr(binding, "authority_anchor", None)
    source_format = getattr(binding, "source_format", None)
    source_format_value = getattr(source_format, "value", source_format)
    if source_format_value is not None and not isinstance(source_format_value, str):
        raise MountedVisualToolRuntimeError(
            "mounted visual source format identity is invalid"
        )
    return {
        "ordinal": getattr(binding, "ordinal", None),
        "visual_alias_sha256": _sha256_text(binding.resource.resource_alias),
        "parent_document_alias_sha256": _sha256_text(
            binding.parent_document_alias
        ),
        "parent_document_id_sha256": _optional_text_sha256(
            getattr(binding, "parent_document_id", None),
            field_name="binding.parent_document_id",
        ),
        "parent_document_version_sha256": _optional_text_sha256(
            getattr(binding, "parent_document_version", None),
            field_name="binding.parent_document_version",
        ),
        "page_manifest_sha256": getattr(binding, "page_manifest_sha256", None),
        "source_format": source_format_value,
        "planning_freshness_binding_sha256": getattr(
            anchor,
            "freshness_binding_sha256",
            None,
        ),
        "visual_unit": {
            "unit_id_sha256": _sha256_text(unit.unit_id),
            "kind": unit.kind.value,
            "image_path_sha256": _sha256_text(unit.image_path),
            "authorization_path_sha256": _sha256_text(
                unit.authorization_path
            ),
            "source_sha256": unit.source_sha256,
            "image_sha256": unit.image_sha256,
            "locator": {
                "page": locator.page,
                "ordinal": locator.ordinal,
                "section_path_sha256": [
                    _sha256_text(part) for part in locator.section_path
                ],
                "bbox": list(locator.bbox) if locator.bbox is not None else None,
                "char_range": (
                    list(locator.char_range)
                    if locator.char_range is not None
                    else None
                ),
            },
            "mime_type": unit.mime_type,
            "pixel_size": {
                "width": unit.pixel_size.width,
                "height": unit.pixel_size.height,
            },
            "byte_count": unit.byte_count,
            "allowed_purposes": [
                purpose.value for purpose in unit.allowed_purposes
            ],
        },
    }


def _provider_identity_sha256(
    capabilities: VisionCapabilitySnapshot,
) -> str:
    return _sha256_value(
        {
            "endpoint_identity": capabilities.endpoint_identity,
            "model": capabilities.model,
            "processor_fingerprint": capabilities.processor_fingerprint,
            "provider": capabilities.provider,
            "schema_version": "mounted-visual-provider-identity-v1",
        }
    )


def _backend_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _optional_text_sha256(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise MountedVisualToolRuntimeError(f"{field_name} is invalid")
    return _sha256_text(value)


def _source_scope_fingerprint(
    authority: FrozenMountedDocumentPlanningAuthority,
    bindings: tuple[FrozenMountedVisualPlanningBinding, ...],
) -> str:
    if bindings == authority.visual_bindings:
        return authority.scope_snapshot_sha256
    return _sha256_value(
        {
            "schema_version": "mounted-visual-cognition-source-v1",
            "authority_scope_snapshot_sha256": authority.scope_snapshot_sha256,
            "visual_aliases": [
                binding.resource.resource_alias for binding in bindings
            ],
        }
    )


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MountedVisualToolRuntimeError",
    'SessionMountedVisualToolRuntime',
    "build_session_mounted_visual_tool_runtime",
]
