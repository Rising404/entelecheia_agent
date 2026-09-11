"""Stable catalog contracts for the external mounted-visual capability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
)
from personagraph.input_processing.vision.providers import (
    UnavailableVisionModelAdapter,
)
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionPurpose,
)
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.visual.mounted_visual_catalog import (
    ExternalMountedVisualBindingFacts,
    MountedVisualCatalogError,
    build_mounted_visual_tool_bindings,
    build_mounted_visual_tool_definition_manifest,
    derive_mounted_visual_source_fingerprint,
)
from personagraph.tools.visual.mounted_visual_tools import (
    MOUNTED_VISUAL_IMPLEMENTATION_VERSION,
    FrozenMountedVisualToolScope,
    MountedVisualToolBinding,
    build_mounted_visual_tool_source,
)
from personagraph.tools.visual.visual_tool_boundary import VisualUnitRef


SESSION_ID = "session-mounted-visual-catalog"


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _facts(*, provider: str = "provider-a") -> ExternalMountedVisualBindingFacts:
    return ExternalMountedVisualBindingFacts(
        session_scope_sha256=_sha256(SESSION_ID),
        visual_scope_snapshot_sha256=_sha256("visual-scope"),
        visual_alias_projection_sha256=_sha256("alias-projection"),
        mounted_authority_sha256=_sha256("mounted-authority"),
        freshness_authority_sha256=_sha256("freshness-authority"),
        provider_identity_sha256=_sha256(provider),
        capability_snapshot_sha256=_sha256(f"{provider}-capabilities"),
        egress_policy_sha256=_sha256("disclosure-authority"),
        physical_call_ledger_sha256=_sha256("physical-call-ledger"),
    )


class _Freshness:
    def is_current(self, binding: MountedVisualToolBinding) -> bool:
        return True


class _Adapter:
    def __init__(self, *, transmits: bool, provider: str = "provider-a") -> None:
        self.transmits_externally = transmits
        self._provider = provider

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider=self._provider,
            model="fixture-vl",
            endpoint_identity=f"test:https://{self._provider}.invalid",
            processor_fingerprint=f"{self._provider}@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):  # pragma: no cover - composition performs no I/O
        raise AssertionError("catalog composition must not invoke the adapter")


def _scope(picture: Path) -> FrozenMountedVisualToolScope:
    raw = picture.read_bytes()
    return FrozenMountedVisualToolScope(
        session_id=SESSION_ID,
        scope_snapshot_sha256=_sha256("visual-scope"),
        bindings=(
            MountedVisualToolBinding(
                visual_alias="mounted_visual_001",
                document_alias="mounted_document_01",
                visual_unit=VisualUnitRef(
                    unit_id="private-unit-id",
                    kind=DocumentNonTextKind.FIGURE,
                    image_path=str(picture),
                    source_sha256=_sha256("source-document"),
                    image_sha256=hashlib.sha256(raw).hexdigest(),
                    locator=DocumentLocator(page=2),
                    mime_type="image/png",
                    pixel_size=PixelSize(40, 30),
                    byte_count=len(raw),
                ),
            ),
        ),
    )


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    target = tmp_path / "mounted-visual.png"
    Image.new("RGB", (40, 30), "white").save(target)
    return target


def _source(
    picture: Path,
    *,
    adapter,
    facts: ExternalMountedVisualBindingFacts,
):
    source = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=adapter,
        freshness=_Freshness(),
        source_fingerprint=derive_mounted_visual_source_fingerprint(facts),
    )
    assert source is not None
    return source


def test_definition_is_stable_and_contains_no_contextual_provider_facts() -> None:
    (first,) = build_mounted_visual_tool_definition_manifest()
    (second,) = build_mounted_visual_tool_definition_manifest()

    assert first.definition == second.definition
    assert (
        first.definition.identity.implementation_version
        == MOUNTED_VISUAL_IMPLEMENTATION_VERSION
    )
    serialized = json.dumps(first.definition.descriptor(), sort_keys=True)
    assert SESSION_ID not in serialized
    assert "mounted_visual_001" not in serialized
    assert "provider-a" not in serialized
    assert "physical-call-ledger" not in serialized


def test_external_registration_binds_every_private_authority_hash(
    picture: Path,
) -> None:
    facts = _facts()
    source = _source(
        picture,
        adapter=_Adapter(transmits=True),
        facts=facts,
    )

    (binding,) = build_mounted_visual_tool_bindings(
        (source.registration,),
        facts=facts,
    )
    (manifest_item,) = build_mounted_visual_tool_definition_manifest()
    bound = BoundToolRegistration(manifest_item.definition, binding)

    assert bound.descriptor() == source.registration.descriptor()
    assert dict(binding.binding_assertion) == {
        "schema_version": "mounted-visual-binding-assertion-v1",
        "binding_kind": "authorized_external_mounted_visual_read",
        "session_scope_sha256": facts.session_scope_sha256,
        "visual_scope_snapshot_sha256": facts.visual_scope_snapshot_sha256,
        "visual_alias_projection_sha256": facts.visual_alias_projection_sha256,
        "mounted_authority_sha256": facts.mounted_authority_sha256,
        "freshness_authority_sha256": facts.freshness_authority_sha256,
        "provider_identity_sha256": facts.provider_identity_sha256,
        "capability_snapshot_sha256": facts.capability_snapshot_sha256,
        "egress_policy_sha256": facts.egress_policy_sha256,
        "physical_call_ledger_sha256": facts.physical_call_ledger_sha256,
        "source_fingerprint": derive_mounted_visual_source_fingerprint(facts),
    }
    serialized = json.dumps(binding.descriptor(), sort_keys=True)
    assert "mounted_visual_001" not in serialized
    assert "private-unit-id" not in serialized


@pytest.mark.parametrize(
    "adapter",
    [
        _Adapter(transmits=False),
        UnavailableVisionModelAdapter(),
    ],
)
def test_local_or_unavailable_registration_cannot_impersonate_external_definition(
    picture: Path,
    adapter,
) -> None:
    facts = _facts()
    source = _source(picture, adapter=adapter, facts=facts)

    with pytest.raises(MountedVisualCatalogError, match="external effect"):
        build_mounted_visual_tool_bindings(
            (source.registration,),
            facts=facts,
        )


def test_provider_changes_binding_not_stable_definition(picture: Path) -> None:
    facts_a = _facts(provider="provider-a")
    facts_b = _facts(provider="provider-b")
    source_a = _source(
        picture,
        adapter=_Adapter(transmits=True, provider="provider-a"),
        facts=facts_a,
    )
    source_b = _source(
        picture,
        adapter=_Adapter(transmits=True, provider="provider-b"),
        facts=facts_b,
    )

    (binding_a,) = build_mounted_visual_tool_bindings(
        (source_a.registration,),
        facts=facts_a,
    )
    (binding_b,) = build_mounted_visual_tool_bindings(
        (source_b.registration,),
        facts=facts_b,
    )
    (manifest_item,) = build_mounted_visual_tool_definition_manifest()

    assert source_a.registration.implementation_version == (
        source_b.registration.implementation_version
    )
    assert binding_a.definition_digest == manifest_item.definition.digest
    assert binding_b.definition_digest == manifest_item.definition.digest
    assert binding_a.digest != binding_b.digest


def test_wrong_source_fingerprint_and_noncanonical_hash_fail_closed(
    picture: Path,
) -> None:
    facts = _facts()
    source = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=_Adapter(transmits=True),
        freshness=_Freshness(),
    )
    assert source is not None

    with pytest.raises(MountedVisualCatalogError, match="fingerprint"):
        build_mounted_visual_tool_bindings(
            (source.registration,),
            facts=facts,
        )
    with pytest.raises(MountedVisualCatalogError, match="provider_identity"):
        replace(
            facts,
            provider_identity_sha256="not-a-hash",
        )
