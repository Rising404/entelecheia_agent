from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    OcrFailureCode,
    OcrLine,
    OcrRequest,
    OcrResult,
    OcrStatus,
    PixelSize,
    VisionFailureDiagnostics,
    VisionObservation,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
)


SOURCE_SHA = "a" * 64
IMAGE_SHA = "b" * 64


@pytest.mark.parametrize("uncertainty", [None, 0, 0.2, 1])
def test_vision_observation_preserves_absent_or_supplied_uncertainty(uncertainty):
    observation = VisionObservation("observation-1", "general", "A chart.", uncertainty)
    assert observation.uncertainty == uncertainty
    if uncertainty is not None:
        assert type(observation.uncertainty) is float


@pytest.mark.parametrize("uncertainty", [True, "0.3", -0.1, 1.1, float("nan"), float("inf")])
def test_vision_observation_rejects_invalid_supplied_uncertainty(uncertainty):
    with pytest.raises(ValueError, match="uncertainty"):
        VisionObservation("observation-1", "general", "A chart.", uncertainty)


def test_vision_failure_diagnostics_are_immutable_and_only_attach_to_failures():
    diagnostics = VisionFailureDiagnostics(
        phase="request", elapsed_ms=3711, timeout_s=120,
        completion_uncertain=False, exception_type="HTTPError", http_status=429,
        retry_after_s=7,
    )
    result = VisionResult(
        status=VisionStatus.FAILED, provider="test", model="test-vl",
        endpoint_identity="test", processor_fingerprint="test", input_sha256=IMAGE_SHA,
        unresolved_gap_refs=("image-1",), failure_code="vision_http_rate_limited",
        failure_diagnostics=diagnostics,
    )
    assert result.failure_diagnostics.to_dict()["retry_after_s"] == 7.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.http_status = 500
    with pytest.raises(ValueError, match="failure_diagnostics"):
        replace(result, status=VisionStatus.UNAVAILABLE)
    with pytest.raises(ValueError, match="failure_diagnostics"):
        replace(result, failure_diagnostics={"raw_body": "secret"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"phase": "secret"}, {"phase": []}, {"exception_type": "Error: secret"},
        {"cause_type": "a" * 129}, {"http_status": True}, {"http_status": 999},
        {"elapsed_ms": -1}, {"elapsed_ms": True}, {"timeout_s": float("nan")},
        {"timeout_s": 0}, {"retry_after_s": float("inf")}, {"retry_after_s": -1},
        {"completion_uncertain": "true"},
    ],
)
def test_vision_failure_diagnostics_reject_unsafe_or_invalid_fields(overrides):
    values = {
        "phase": "request", "elapsed_ms": 0, "timeout_s": 120.0,
        "completion_uncertain": True,
    }
    with pytest.raises(ValueError):
        VisionFailureDiagnostics(**{**values, **overrides})


def _request() -> OcrRequest:
    return OcrRequest(
        source_unit_id="image-1",
        source_sha256=SOURCE_SHA,
        image_sha256=IMAGE_SHA,
        page=1,
        pixel_size=PixelSize(800, 600),
        dpi=(144.0, 144.0),
        language_hints=("zh-Hans", "en-US"),
    )


def _line() -> OcrLine:
    return OcrLine(
        text="图表标题",
        confidence=0.97,
        bbox_norm=(0.1, 0.2, 0.7, 0.3),
        bbox_px=(80.0, 120.0, 560.0, 180.0),
    )


def test_ocr_contracts_are_immutable_and_bind_every_result_to_exact_input():
    request = _request()
    result = OcrResult.from_request(
        request,
        status=OcrStatus.SUCCESS,
        engine_fingerprint="ocrmac-vision@1.0.1:accurate",
        lines=(_line(),),
        elapsed_ms=12,
        warnings=("low_contrast",),
    )

    assert result.source_sha256 == SOURCE_SHA
    assert result.image_sha256 == IMAGE_SHA
    assert result.page == 1
    assert result.pixel_size == PixelSize(800, 600)
    assert result.dpi == (144.0, 144.0)
    assert result.language_hints == ("zh-Hans", "en-US")
    assert result.lines[0].bbox_norm == (0.1, 0.2, 0.7, 0.3)
    assert result.lines[0].bbox_px == (80.0, 120.0, 560.0, 180.0)
    with pytest.raises(FrozenInstanceError):
        result.status = OcrStatus.FAILED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "lines", "failure_code"),
    [
        (OcrStatus.SUCCESS, (), None),
        (OcrStatus.BLANK, (_line(),), None),
        (OcrStatus.FAILED, (), None),
        (OcrStatus.FAILED, (_line(),), OcrFailureCode.BACKEND_FAILED),
    ],
)
def test_ocr_status_cannot_conflate_success_blank_and_failure(
    status, lines, failure_code
):
    with pytest.raises(ValueError):
        OcrResult.from_request(
            _request(),
            status=status,
            engine_fingerprint="fake@1",
            lines=lines,
            elapsed_ms=0,
            failure_code=failure_code,
        )


def test_ocr_contract_rejects_untrusted_geometry_hashes_and_confidence():
    with pytest.raises(ValueError, match="sha256"):
        replace(_request(), source_sha256="not-a-hash")
    with pytest.raises(ValueError, match="confidence"):
        replace(_line(), confidence=1.1)
    with pytest.raises(ValueError, match="bbox_norm"):
        replace(_line(), bbox_norm=(-0.1, 0.2, 0.7, 0.3))
    with pytest.raises(ValueError, match="bbox_px"):
        OcrResult.from_request(
            _request(),
            status=OcrStatus.SUCCESS,
            engine_fingerprint="fake@1",
            lines=(replace(_line(), bbox_px=(80.0, 120.0, 801.0, 180.0)),),
            elapsed_ms=0,
        )


def test_vision_request_binds_source_image_and_disclosure_metadata():
    request = VisionRequest(
        source_unit_id="figure-1",
        source_sha256=SOURCE_SHA,
        image_sha256=IMAGE_SHA,
        locator=DocumentLocator(page=3, bbox=(0.1, 0.2, 0.9, 0.8)),
        mime_type="image/png",
        pixel_size=PixelSize(1024, 768),
        byte_count=1234,
        purpose=VisionPurpose.CHART,
        prompt_contract_version="vision-chart-v1",
        image_path="/tmp/figure-1.jpg",
        disclosure_receipt_id=None,
    )

    assert request.page == 3
    assert request.purpose is VisionPurpose.CHART
    with pytest.raises(ValueError, match="byte_count"):
        replace(request, byte_count=0)


def _question_request(**overrides: object) -> VisionRequest:
    values = {
        "source_unit_id": "image-question",
        "source_sha256": SOURCE_SHA,
        "image_sha256": IMAGE_SHA,
        "locator": DocumentLocator(page=1),
        "mime_type": "image/png",
        "pixel_size": PixelSize(800, 600),
        "byte_count": 123,
        "purpose": VisionPurpose.QUESTION,
        "prompt_contract_version": "vision-question-v1",
        "image_path": "/tmp/question.png",
        "question": "  请描述这张图片中左右两部分的关系。  ",
        "logical_tool_call_id": "l1tool_question_01",
    }
    return VisionRequest(**{**values, **overrides})


def test_question_request_normalizes_the_question_and_requires_host_call_identity():
    request = _question_request()
    assert request.question == "请描述这张图片中左右两部分的关系。"
    assert request.logical_tool_call_id == "l1tool_question_01"
    with pytest.raises(ValueError, match="logical_tool_call_id"):
        _question_request(logical_tool_call_id=None)


@pytest.mark.parametrize("question", [None, "", " \n\t", "问题\x00", 3, True, ["what?"]])
def test_question_mode_rejects_missing_empty_or_non_text_questions(question):
    with pytest.raises(ValueError, match="question"):
        _question_request(question=question)


@pytest.mark.parametrize(
    "purpose", [value for value in VisionPurpose if value is not VisionPurpose.QUESTION]
)
def test_legacy_modes_do_not_accept_a_hidden_question(purpose):
    with pytest.raises(ValueError, match="question"):
        _question_request(purpose=purpose)


@pytest.mark.parametrize("call_id", [None, "", " ", "a b", "a\n", "a\x00", "a" * 301])
def test_question_request_rejects_invalid_host_call_identity(call_id):
    with pytest.raises(ValueError, match="logical_tool_call_id"):
        _question_request(logical_tool_call_id=call_id)


def test_question_length_and_normalization_have_a_shared_boundary():
    from personagraph.input_processing.vision.contracts import (
        MAX_VISION_QUESTION_CHARS,
        normalize_vision_question,
    )

    assert normalize_vision_question(VisionPurpose.QUESTION, " 问题 ") == "问题"
    assert normalize_vision_question(VisionPurpose.GENERAL, None) is None
    assert _question_request(question="问" * MAX_VISION_QUESTION_CHARS)
    with pytest.raises(ValueError, match="question"):
        _question_request(question="问" * (MAX_VISION_QUESTION_CHARS + 1))
