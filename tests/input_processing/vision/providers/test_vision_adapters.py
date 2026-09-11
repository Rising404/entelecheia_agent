from __future__ import annotations

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.providers import UnavailableVisionModelAdapter
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionPurpose,
    VisionRequest,
    VisionStatus,
)


def test_unavailable_vision_adapter_is_stable_and_performs_no_io(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unavailable adapter attempted external I/O")

    monkeypatch.setattr("socket.socket", forbidden)
    request = VisionRequest(
        source_unit_id="figure-1",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
        locator=DocumentLocator(page=1),
        mime_type="image/jpeg",
        pixel_size=PixelSize(640, 480),
        byte_count=100,
        purpose=VisionPurpose.GENERAL,
        prompt_contract_version="vision-general-v1",
        image_path="/tmp/figure-1.jpg",
    )
    adapter = UnavailableVisionModelAdapter()

    capabilities = adapter.capabilities()
    result = adapter.analyze(request)

    assert capabilities.available is False
    assert capabilities.reason_code == "vision_provider_unavailable"
    assert result.status is VisionStatus.UNAVAILABLE
    assert result.failure_code == "vision_provider_unavailable"
    assert result.observations == ()
    assert result.unresolved_gap_refs == ("figure-1",)
