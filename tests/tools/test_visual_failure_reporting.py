"""Provider failures must reach the model without fabricated background waits."""

from types import SimpleNamespace

import pytest

from personagraph.input_processing.vision.contracts import (
    PixelSize, VisionDetail, VisionFailureDiagnostics, VisionPurpose, VisionRegion,
)
from personagraph.runtime.model_calls.vision import MountedVisualCallWaitingExternal
from personagraph.tools.documents.format_observation_tools import (
    _analyze_visual_source, _observation_schema,
)
from personagraph.tools.execution import ToolBusinessFailure


@pytest.mark.parametrize("uncertain", [True, False])
def test_analyze_page_keeps_safe_failure_facts(tmp_path, uncertain):
    facts = VisionFailureDiagnostics(
        phase="request", elapsed_ms=17, timeout_s=120,
        completion_uncertain=uncertain,
        exception_type="TimeoutError" if uncertain else "HTTPError",
        http_status=None if uncertain else 429,
    )

    def observe(**_kwargs):
        if uncertain:
            raise MountedVisualCallWaitingExternal(
                reason_code="vision_request_timeout", failure_diagnostics=facts,
            )
        return SimpleNamespace(to_dict=lambda: {
            "status": "failed", "at": "general",
            "failure_code": "vision_http_rate_limited",
            "failure_diagnostics": facts.to_dict(),
        }), None

    def call():
        return _analyze_visual_source(
            SimpleNamespace(session_id="session"), target=tmp_path / "page.png",
            source={"sha256": "a" * 64, "byte_count": 100}, page=1,
            pixel_size=PixelSize(20, 20), purpose=VisionPurpose.GENERAL,
            detail=VisionDetail.STANDARD, region=VisionRegion.PAGE,
            observation_service=SimpleNamespace(),
            publication_binding=SimpleNamespace(),
            visual_publisher=SimpleNamespace(observe=observe),
        )

    if uncertain:
        with pytest.raises(ToolBusinessFailure) as caught:
            call()
        assert caught.value.error.code == "visual_completion_unconfirmed"
        assert caught.value.error.details["background_wait_active"] is False
        assert caught.value.error.details["failure_diagnostics"] == facts.to_dict()
        assert "没有后台" in caught.value.error.message
    else:
        result = call()
        assert result["failure_diagnostics"] == facts.to_dict()
        import jsonschema
        jsonschema.validate(result, _observation_schema())
