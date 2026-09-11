"""Lane-neutral ownership locks for the mounted visual Tool."""

from __future__ import annotations

import ast
import hashlib
import json
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
    VisionObservation,
    VisionPurpose,
    VisionResult,
    VisionStatus,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectResource,
    EffectScopeKind,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.visual.mounted_visual_tools import (
    MOUNTED_VISUAL_SOURCE_ID,
    MOUNTED_VISUAL_TOOL_ID,
    FrozenMountedVisualToolScope,
    MountedVisualToolBinding,
    build_mounted_visual_tool_source,
)
from personagraph.tools.visual.visual_tool_boundary import VisualUnitRef


SESSION_ID = "session-mounted-visual-tools"
SCOPE_SHA256 = hashlib.sha256(b"mounted visual scope").hexdigest()


class _Freshness:
    def __init__(self, current: bool = True) -> None:
        self.current = current
        self.seen: list[MountedVisualToolBinding] = []

    def is_current(self, binding: MountedVisualToolBinding) -> bool:
        self.seen.append(binding)
        return self.current


class _LocalAdapter:
    transmits_externally = False

    def __init__(self) -> None:
        self.seen: list[object] = []

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="local-test",
            model="fixture",
            endpoint_identity="local",
            processor_fingerprint="local-test@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="local-test",
            model="fixture",
            endpoint_identity="local",
            processor_fingerprint="local-test@1",
            input_sha256=request.image_sha256,
            observations=(
                VisionObservation(
                    observation_id="observation-1",
                    kind=request.purpose.value,
                    text="A chart with four bars.",
                    uncertainty=0.1,
                ),
            ),
        )


def _binding(
    picture: Path,
    *,
    visual_alias: str = "mounted_visual_001",
    document_alias: str = "mounted_document_01",
) -> MountedVisualToolBinding:
    raw = picture.read_bytes()
    return MountedVisualToolBinding(
        visual_alias=visual_alias,
        document_alias=document_alias,
        visual_unit=VisualUnitRef(
            unit_id="private-provider-unit-id",
            kind=DocumentNonTextKind.FIGURE,
            image_path=str(picture),
            source_sha256=hashlib.sha256(b"source document").hexdigest(),
            image_sha256=hashlib.sha256(raw).hexdigest(),
            locator=DocumentLocator(page=4),
            mime_type="image/png",
            pixel_size=PixelSize(48, 32),
            byte_count=len(raw),
        ),
    )


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    target = tmp_path / "private-render.png"
    Image.new("RGB", (48, 32), "white").save(target)
    return target


def _scope(
    picture: Path,
    *,
    session_id: str = SESSION_ID,
    scope_sha256: str = SCOPE_SHA256,
    alias: str = "mounted_visual_001",
) -> FrozenMountedVisualToolScope:
    return FrozenMountedVisualToolScope(
        session_id=session_id,
        scope_snapshot_sha256=scope_sha256,
        bindings=(_binding(picture, visual_alias=alias),),
    )


def test_source_owns_stable_spec_without_exposing_private_identity(
    picture: Path,
) -> None:
    source = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=UnavailableVisionModelAdapter(),
        freshness=_Freshness(),
    )

    assert source is not None
    assert source.registration.tool_id == MOUNTED_VISUAL_TOOL_ID
    assert source.registration.source.source_id == MOUNTED_VISUAL_SOURCE_ID
    descriptor = json.dumps(source.registration.descriptor(), sort_keys=True)
    assert str(picture) not in descriptor
    assert "private-provider-unit-id" not in descriptor
    assert "mounted_visual_001" not in descriptor
    alias_schema = source.registration.spec.input_schema["properties"]["visuals"][
        "items"
    ]["properties"]["visual_alias"]
    assert alias_schema == {
        "type": "string",
        "pattern": "^mounted_visual_[0-9]+$",
    }


def test_tool_spec_is_identical_across_distinct_scopes(picture: Path) -> None:
    first = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=UnavailableVisionModelAdapter(),
        freshness=_Freshness(),
    )
    second = build_mounted_visual_tool_source(
        _scope(
            picture,
            session_id="session-other",
            scope_sha256=hashlib.sha256(b"other scope").hexdigest(),
            alias="mounted_visual_099",
        ),
        adapter=UnavailableVisionModelAdapter(),
        freshness=_Freshness(),
    )

    assert first is not None and second is not None
    assert first.registration.spec.to_dict() == second.registration.spec.to_dict()
    assert first.registration.source.fingerprint != second.registration.source.fingerprint


def test_handler_uses_only_alias_and_projects_public_identity(picture: Path) -> None:
    adapter = _LocalAdapter()
    freshness = _Freshness()
    source = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=adapter,
        freshness=freshness,
    )

    assert source is not None
    result = source.registration.handler(
        {
            "visuals": [
                {"visual_alias": "mounted_visual_001", "purpose": "chart"}
            ]
        }
    )

    assert result["requested"] == 1
    assert result["resolved"] == 1
    assert result["results"][0]["visual_alias"] == "mounted_visual_001"
    assert result["results"][0]["document_alias"] == "mounted_document_01"
    assert result["results"][0]["source_pages"] == [4]
    assert result["results"][0]["observation"] == "A chart with four bars."
    assert freshness.seen == list(_scope(picture).bindings)
    assert len(adapter.seen) == 1
    (effect,) = source.registration.effect_profile.effects
    assert effect.resource is EffectResource.FILESYSTEM
    assert effect.action is EffectAction.READ
    assert effect.scope_kind is EffectScopeKind.SESSION
    assert effect.default_scope == SESSION_ID


def test_unknown_or_stale_alias_never_reaches_adapter(picture: Path) -> None:
    adapter = _LocalAdapter()
    source = build_mounted_visual_tool_source(
        _scope(picture),
        adapter=adapter,
        freshness=_Freshness(current=False),
    )
    assert source is not None

    with pytest.raises(ToolBusinessFailure) as unknown:
        source.registration.handler(
            {"visuals": [{"visual_alias": "mounted_visual_999", "purpose": "chart"}]}
        )
    assert unknown.value.error.code == "unknown_mounted_visual_alias"
    with pytest.raises(ToolBusinessFailure) as stale:
        source.registration.handler(
            {"visuals": [{"visual_alias": "mounted_visual_001", "purpose": "chart"}]}
        )
    assert stale.value.error.code == "mounted_visual_authority_drift"
    assert adapter.seen == []


def test_scope_rejects_duplicate_aliases(picture: Path) -> None:
    binding = _binding(picture)
    with pytest.raises(ValueError, match="aliases"):
        FrozenMountedVisualToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            bindings=(binding, binding),
        )


def test_tool_owner_has_no_l2_or_runtime_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/tools/visual/mounted_visual_tools.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        module == "personagraph.l2" or module.startswith("personagraph.l2.")
        for module in imported_modules
    )
    assert not any(
        module == "personagraph.runtime"
        or module.startswith("personagraph.runtime.")
        for module in imported_modules
    )
    assert not any(
        node.level > 0
        and (node.module or "").split(".", 1)[0] in {"l2", "runtime"}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
