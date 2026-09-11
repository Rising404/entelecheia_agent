"""Host-only composition around the lane-neutral mounted visual Tool owner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
)
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionPurpose,
    VisionResult,
    VisionStatus,
)
from personagraph.input_processing.vision.providers import (
    UnavailableVisionModelAdapter,
)
from personagraph.l2.task_execution.tool_bridge import (
    mounted_visual_adapter as adapter_module,
)
from personagraph.runtime.model_calls.vision import (
    SqliteMountedVisualCallLedger,
)
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.visual.mounted_visual_catalog import (
    build_mounted_visual_tool_definition_manifest,
)
from personagraph.tools.visual.mounted_visual_tools import (
    MOUNTED_VISUAL_SOURCE_ID,
    MOUNTED_VISUAL_TOOL_ID,
)
from personagraph.tools.visual.visual_tool_boundary import VisualUnitRef


SESSION_ID = "session-mounted-visual-adapter"
SCOPE_SHA256 = hashlib.sha256(b"planning visual scope").hexdigest()


@dataclass(frozen=True)
class _Resource:
    session_id: str
    resource_alias: str


@dataclass(frozen=True)
class _PlanningBinding:
    resource: _Resource
    parent_document_alias: str
    visual_unit: VisualUnitRef


@dataclass(frozen=True)
class _Authority:
    session_id: str
    visual_bindings: tuple[_PlanningBinding, ...]
    scope_snapshot_sha256: str


class _Adapter:
    def __init__(self, *, transmits: bool) -> None:
        self.transmits_externally = transmits
        self.seen: list[object] = []

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="adapter-test",
            model="fixture-vl",
            endpoint_identity="test:https://vision.invalid",
            processor_fingerprint="adapter-test@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="adapter-test",
            model="fixture-vl",
            endpoint_identity="test:https://vision.invalid",
            processor_fingerprint="adapter-test@1",
            input_sha256=request.image_sha256,
            observations=(
                VisionObservation(
                    observation_id="adapter-observation",
                    kind=request.purpose.value,
                    text="Four bars are visible.",
                    uncertainty=0.1,
                ),
            ),
        )


def _authority(picture: Path) -> _Authority:
    raw = picture.read_bytes()
    unit = VisualUnitRef(
        unit_id="private-unit-01",
        kind=DocumentNonTextKind.FIGURE,
        image_path=str(picture),
        source_sha256=hashlib.sha256(b"source document").hexdigest(),
        image_sha256=hashlib.sha256(raw).hexdigest(),
        locator=DocumentLocator(page=2),
        mime_type="image/png",
        pixel_size=PixelSize(64, 48),
        byte_count=len(raw),
    )
    return _Authority(
        session_id=SESSION_ID,
        visual_bindings=(
            _PlanningBinding(
                resource=_Resource(
                    session_id=SESSION_ID,
                    resource_alias="mounted_visual_001",
                ),
                parent_document_alias="mounted_document_01",
                visual_unit=unit,
            ),
        ),
        scope_snapshot_sha256=SCOPE_SHA256,
    )


@pytest.fixture
def authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Authority:
    picture = tmp_path / "mounted.png"
    Image.new("RGB", (64, 48), "white").save(picture)
    value = _authority(picture)
    monkeypatch.setattr(
        adapter_module,
        "FrozenMountedDocumentPlanningAuthority",
        _Authority,
    )
    monkeypatch.setattr(
        adapter_module,
        "mounted_visual_binding_is_current",
        lambda binding: binding in value.visual_bindings,
    )
    return value


def test_l2_projects_authority_into_the_tools_owned_source(
    authority: _Authority,
) -> None:
    adapter = _Adapter(transmits=False)

    runtime = adapter_module.build_session_mounted_visual_tool_runtime(
        authority,
        adapter=adapter,

    )

    assert runtime is not None
    assert runtime.registration.tool_id == MOUNTED_VISUAL_TOOL_ID
    assert runtime.registration.source.source_id == MOUNTED_VISUAL_SOURCE_ID
    result = runtime.registration.handler(
        {
            "visuals": [
                {"visual_alias": "mounted_visual_001", "purpose": "chart"}
            ]
        }
    )
    assert result["resolved"] == 1
    assert result["results"][0]["document_alias"] == "mounted_document_01"
    assert len(adapter.seen) == 1
    assert runtime.authority.grants
    assert runtime.contextual_bindings == ()
    assert runtime.protected_authority_by_key == {}


def test_l2_keeps_receipts_and_provider_binding_outside_tool_owner(
    authority: _Authority,
    tmp_path: Path,
) -> None:
    adapter = _Adapter(transmits=True)
    adapter.capabilities()
    authority.visual_bindings[0].visual_unit
    physical_ledger_path = tmp_path / "mounted-visual-calls.sqlite"

    runtime = adapter_module.build_session_mounted_visual_tool_runtime(
        authority,
        adapter=adapter,

        call_ledger=SqliteMountedVisualCallLedger(physical_ledger_path),
    )

    assert runtime is not None
    key = (
        runtime.registration.tool_id,
        runtime.registration.contract_version,
    )
    protected = runtime.protected_authority_by_key[key]
    assert all(item.startswith("auto_visual_egress_") for item in protected.approval_receipt_ids)
    assert protected.revalidate()
    assert runtime.authority.grants
    assert runtime.authority.approval_grants == ()
    (binding,) = runtime.contextual_bindings
    (manifest_item,) = build_mounted_visual_tool_definition_manifest()
    materialized = BoundToolRegistration(manifest_item.definition, binding)
    assert materialized.descriptor() == runtime.registration.descriptor()
    assert materialized.handler is runtime.registration.handler
    assertion_json = str(dict(binding.binding_assertion))
    assert "auto_visual_egress_" not in assertion_json
    assert str(physical_ledger_path) not in assertion_json
    assert not physical_ledger_path.exists(), "composition must not perform I/O"


def test_l2_external_binding_requires_no_disclosure_ledger(
    authority: _Authority,
    tmp_path: Path,
) -> None:
    runtime = adapter_module.build_session_mounted_visual_tool_runtime(
        authority,
        adapter=_Adapter(transmits=True),

        call_ledger=SqliteMountedVisualCallLedger(
            tmp_path / "missing-disclosure.sqlite"
        ),
    )

    assert runtime is not None
    assert runtime.contextual_bindings
    assert runtime.authority.approval_grants == ()
    assert runtime.protected_authority_by_key


def test_l2_local_or_unavailable_adapter_never_gets_external_binding(
    authority: _Authority,
) -> None:
    for adapter in (
        _Adapter(transmits=False),
        UnavailableVisionModelAdapter(),
    ):
        runtime = adapter_module.build_session_mounted_visual_tool_runtime(
            authority,
            adapter=adapter,

        )

        assert runtime is not None
        assert runtime.contextual_bindings == ()
        assert runtime.protected_authority_by_key == {}


def test_physical_ledger_binding_uses_stable_path_identity_not_object_repr(
    authority: _Authority,
    tmp_path: Path,
) -> None:
    _Adapter(transmits=True).capabilities()
    authority.visual_bindings[0].visual_unit
    ledger_path = tmp_path / "stable-ledger-identity.sqlite"

    first = adapter_module.build_session_mounted_visual_tool_runtime(
        authority,
        adapter=_Adapter(transmits=True),

        call_ledger=SqliteMountedVisualCallLedger(ledger_path),
    )
    second = adapter_module.build_session_mounted_visual_tool_runtime(
        authority,
        adapter=_Adapter(transmits=True),

        call_ledger=SqliteMountedVisualCallLedger(ledger_path),
    )

    assert first is not None and second is not None
    assert first.contextual_bindings[0].descriptor() == (
        second.contextual_bindings[0].descriptor()
    )
    assert first.registration.source.fingerprint == (
        second.registration.source.fingerprint
    )
    assert not ledger_path.exists()


def test_l2_rejects_adapter_without_egress_classification(
    authority: _Authority,
) -> None:
    class IncompleteAdapter:
        def capabilities(self):
            return _Adapter(transmits=False).capabilities()

        def analyze(self, request):
            raise AssertionError("an incomplete adapter must never execute")

    with pytest.raises(TypeError, match="transmits_externally"):
        adapter_module.build_session_mounted_visual_tool_runtime(
            authority,
            adapter=IncompleteAdapter(),

        )
