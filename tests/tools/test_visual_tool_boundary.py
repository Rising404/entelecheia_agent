"""冻结视觉工具权威的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
)
from personagraph.input_processing.vision.contracts import PixelSize, VisionPurpose
from personagraph.tools.visual import visual_tool_boundary


def _unit(
    unit_id: str = "figure-1",
    *,
    kind: DocumentNonTextKind = DocumentNonTextKind.FIGURE,
    disclosure_source_path: str | None = None,
) -> visual_tool_boundary.VisualUnitRef:
    return visual_tool_boundary.VisualUnitRef(
        unit_id=unit_id,
        kind=kind,
        image_path="/host-private/rendered/figure-1.png",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
        locator=DocumentLocator(page=1),
        mime_type="image/png",
        pixel_size=PixelSize(640, 480),
        byte_count=1024,
        disclosure_source_path=disclosure_source_path,
    )


def test_boundary_keeps_purpose_and_disclosure_path_authority() -> None:
    unit = _unit(disclosure_source_path="/host-private/source.pdf")

    assert unit.allowed_purposes == (
        VisionPurpose.CHART,
        VisionPurpose.CAPTION,
        VisionPurpose.GENERAL,
        VisionPurpose.QUESTION,
    )
    assert unit.authorization_path == "/host-private/source.pdf"
    boundary = visual_tool_boundary.FrozenVisualToolBoundary(
        session_id="session-1",
        units=(unit,),
    )
    assert boundary.unit("figure-1") is unit
    assert boundary.unit("unknown") is None


def test_boundary_allows_unresolved_visual_tables_and_refuses_duplicates() -> None:
    table = _unit(kind=DocumentNonTextKind.TABLE)
    assert table.allowed_purposes == (VisionPurpose.GENERAL, VisionPurpose.QUESTION)

    unit = _unit()
    with pytest.raises(ValueError, match="unit_id values must be unique"):
        visual_tool_boundary.FrozenVisualToolBoundary(
            session_id="session-1",
            units=(unit, unit),
        )
