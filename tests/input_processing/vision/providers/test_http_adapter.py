"""在无网络环境下验证 HTTP 视觉适配器。

传输层由外部注入，因此这些测试覆盖适配器自身的职责：把准备失败转换为带类型
结果、记录实际传输的内容，并绝不让提供方或文件系统细节成为证据。
"""

from __future__ import annotations

import hashlib
import base64
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionDetail,
    VisionPurpose,
    VisionRequest,
    VisionStatus,
)
from personagraph.input_processing.vision.providers.http import (
    PROMPT_CONTRACT_VERSION,
    HttpVisionModelAdapter,
    VisionProviderConfig,
    load_provider_config,
)
from personagraph.input_processing.vision.imaging import VisionPayload, prepare_payload


CONFIG = VisionProviderConfig(
    provider="test-provider",
    base_url="https://vision.test/",
    api_key="secret-key-value",
    model="test-vl",
)


def _reply(text: str):
    def transport(_config, _body):
        return {"choices": [{"message": {"content": text}}]}

    return transport


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    target = tmp_path / "figure.png"
    Image.new("RGB", (600, 400), "white").save(target)
    return target


def _request(path, detail: VisionDetail = VisionDetail.STANDARD) -> VisionRequest:
    raw = Path(path).read_bytes() if Path(path).is_file() else b"x"
    return VisionRequest(
        source_unit_id="figure-7",
        source_sha256="a" * 64,
        image_sha256=hashlib.sha256(raw).hexdigest(),
        locator=DocumentLocator(page=3),
        mime_type="image/png",
        pixel_size=PixelSize(600, 400),
        byte_count=max(1, len(raw)),
        purpose=VisionPurpose.CHART,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        image_path=str(path),
        detail=detail,
    )


def test_the_endpoint_identity_never_contains_the_key():
    assert CONFIG.api_key not in CONFIG.endpoint_identity
    snapshot = HttpVisionModelAdapter(CONFIG).capabilities()
    assert CONFIG.api_key not in snapshot.endpoint_identity
    assert CONFIG.api_key not in snapshot.processor_fingerprint


def test_a_configured_adapter_reports_itself_available():
    snapshot = HttpVisionModelAdapter(CONFIG).capabilities()
    assert snapshot.available is True
    assert set(snapshot.supported_purposes) == set(VisionPurpose)


def test_a_successful_read_records_what_was_transmitted(picture):
    adapter = HttpVisionModelAdapter(CONFIG, transport=_reply("Four bars; tallest is Q4."))
    result = adapter.analyze(_request(picture))

    assert result.status is VisionStatus.COMPLETED
    assert result.observations[0].text == "Four bars; tallest is Q4."
    assert result.observations[0].kind == "chart"
    assert result.observations[0].uncertainty is None  # Free text supplies no calibrated score.
    # 600×400 符合标准预算，因此传输的字节就是原文件。
    assert result.input_sha256 == hashlib.sha256(picture.read_bytes()).hexdigest()
    assert "600x400" in result.processor_fingerprint
    assert result.warnings == ()


def test_verified_payload_is_sent_from_memory_even_if_source_path_changes(picture):
    request = _request(picture)
    prepared = prepare_payload(request)
    assert isinstance(prepared, VisionPayload)
    bound = replace(request, prepared_payload=prepared)
    Image.new("RGB", (600, 400), "black").save(picture)
    sent: dict[str, bytes] = {}

    def capture(_config, body):
        data_url = body["messages"][0]["content"][1]["image_url"]["url"]
        sent["data"] = base64.b64decode(data_url.split(",", 1)[1])
        return {"choices": [{"message": {"content": "verified"}}]}

    result = HttpVisionModelAdapter(CONFIG, transport=capture).analyze(bound)

    assert sent["data"] == prepared.data
    assert result.input_sha256 == prepared.sent_sha256


def test_unprepared_request_refuses_a_replaced_source_before_transport(picture):
    request = _request(picture)
    Image.new("RGB", (600, 400), "black").save(picture)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("changed source bytes must not reach transport")

    result = HttpVisionModelAdapter(CONFIG, transport=forbidden).analyze(request)

    assert result.status is VisionStatus.FAILED
    assert result.failure_code == "image_source_changed"


def test_a_resampled_read_says_so_and_reports_the_sent_hash(tmp_path: Path):
    """记录的输入必须是模型实际看到的图片，而不是原文件。"""

    target = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "white").save(target)
    request = _request(target)

    adapter = HttpVisionModelAdapter(CONFIG, transport=_reply("A blank canvas."))
    result = adapter.analyze(request)

    assert result.status is VisionStatus.COMPLETED
    assert result.warnings == ("image_resampled",)
    assert result.input_sha256 != request.image_sha256
    assert "resampled" in result.processor_fingerprint


def test_a_missing_file_becomes_a_gap_rather_than_an_exception(tmp_path: Path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a failed preparation must not reach the network")

    adapter = HttpVisionModelAdapter(CONFIG, transport=forbidden)
    result = adapter.analyze(_request(tmp_path / "absent.png"))

    assert result.status is VisionStatus.FAILED
    assert result.failure_code == "image_path_not_a_file"
    assert result.unresolved_gap_refs == ("figure-7",)
    assert result.observations == ()


def test_a_transport_failure_becomes_a_gap_without_leaking_detail(picture):
    def broken(_config, _body):
        raise urllib.error.URLError("host vision.internal unreachable at 10.0.0.4")

    result = HttpVisionModelAdapter(CONFIG, transport=broken).analyze(_request(picture))

    assert result.status is VisionStatus.FAILED
    assert result.failure_code == "vision_connection_failed"
    assert "10.0.0.4" not in repr(result)


def test_a_prepared_transport_failure_keeps_the_exact_payload_identity(picture):
    request = _request(picture)
    prepared = prepare_payload(request)
    assert isinstance(prepared, VisionPayload)

    def broken(_config, _body):
        raise urllib.error.URLError("provider unavailable")

    result = HttpVisionModelAdapter(CONFIG, transport=broken).analyze(
        replace(request, prepared_payload=prepared)
    )

    assert result.status is VisionStatus.FAILED
    assert result.failure_code == "vision_connection_failed"
    assert result.input_sha256 == prepared.sent_sha256


@pytest.mark.parametrize(
    ("error", "code", "phase", "uncertain", "http_status", "cause_type"),
    [
        (urllib.error.HTTPError("https://private.test", 401, "secret", {}, None),
         "vision_http_authentication_failed", "request", False, 401, None),
        (urllib.error.HTTPError("https://private.test", 403, "secret", {}, None),
         "vision_http_authentication_failed", "request", False, 403, None),
        (urllib.error.HTTPError("https://private.test", 429, "secret", {"Retry-After": "7"}, None),
         "vision_http_rate_limited", "request", False, 429, None),
        (urllib.error.HTTPError("https://private.test", 503, "secret", {}, None),
         "vision_http_server_error", "request", False, 503, None),
        (urllib.error.HTTPError("https://private.test", 422, "secret", {}, None),
         "vision_http_request_rejected", "request", False, 422, None),
        (TimeoutError("secret"), "vision_request_timeout", "request", True, None, None),
        (urllib.error.URLError(TimeoutError("secret")),
         "vision_request_timeout", "request", True, None, "TimeoutError"),
        (urllib.error.URLError(OSError("secret")),
         "vision_connection_failed", "request", True, None, "OSError"),
        (ConnectionResetError("secret"),
         "vision_connection_failed", "request", True, None, None),
        (json.JSONDecodeError("secret", "secret", 0),
         "vision_response_invalid", "response_decode", False, None, None),
        (ValueError("secret"), "vision_request_invalid", "request", False, None, None),
    ],
)
def test_visual_failure_classification_is_safe_and_never_resends(
    picture, error, code, phase, uncertain, http_status, cause_type
):
    calls = 0

    def broken(_config, _body):
        nonlocal calls
        calls += 1
        raise error

    result = HttpVisionModelAdapter(CONFIG, transport=broken).analyze(_request(picture))

    assert calls == 1
    assert result.failure_code == code
    diagnostics = result.failure_diagnostics.to_dict()
    assert diagnostics["phase"] == phase
    assert diagnostics["completion_uncertain"] is uncertain
    assert diagnostics["exception_type"] == type(error).__name__
    assert diagnostics["cause_type"] == cause_type
    assert diagnostics["http_status"] == http_status
    assert diagnostics["retry_after_s"] == (7.0 if http_status == 429 else None)
    assert diagnostics["timeout_s"] == 120.0
    assert diagnostics["elapsed_ms"] >= 0
    assert set(diagnostics) == {
        "phase", "exception_type", "cause_type", "http_status", "retry_after_s",
        "elapsed_ms", "timeout_s", "completion_uncertain",
    }
    assert "secret" not in repr(result)
    assert "private.test" not in repr(result)


@pytest.mark.parametrize("body", [{}, [], {"choices": []}, {"choices": [{"message": None}]}])
def test_malformed_response_is_distinct_from_empty_text(picture, body):
    result = HttpVisionModelAdapter(CONFIG, transport=lambda *_: body).analyze(_request(picture))

    assert result.failure_code == "vision_response_invalid"
    assert result.failure_diagnostics.phase == "response_validate"
    assert result.failure_diagnostics.completion_uncertain is False


@pytest.mark.parametrize("raw", [b"not-json secret", b"\xff"])
def test_real_transport_response_decoding_is_classified_without_body_leak(
    picture, monkeypatch, raw
):
    import io

    calls = []

    def open_response(_request, *, timeout):
        calls.append(timeout)
        return io.BytesIO(raw)

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    result = HttpVisionModelAdapter(CONFIG).analyze(_request(picture))

    assert calls == [120.0]
    assert result.failure_code == "vision_response_invalid"
    assert result.failure_diagnostics.phase == "response_decode"
    assert "secret" not in repr(result)


def test_retry_after_supports_dates_and_rejects_unusable_values():
    from personagraph.input_processing.vision.providers.failures import parse_retry_after

    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    assert parse_retry_after("2.5", now=now) == 2.5
    assert parse_retry_after(format_datetime(now + timedelta(seconds=10)), now=now) == 10.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=10)), now=now) == 0.0
    for value in (None, "NaN", "inf", "-1", "secret", "9" * 2000):
        assert parse_retry_after(value, now=now) is None


def test_an_empty_provider_reply_is_a_failure_not_an_empty_observation(picture):
    result = HttpVisionModelAdapter(CONFIG, transport=_reply("   ")).analyze(_request(picture))
    assert result.status is VisionStatus.FAILED
    assert result.failure_code == "vision_response_empty"


def test_typed_content_parts_are_accepted(picture):
    def parts(_config, _body):
        return {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "Two axes, no labels."}]}}
            ]
        }

    result = HttpVisionModelAdapter(CONFIG, transport=parts).analyze(_request(picture))
    assert result.observations[0].text == "Two axes, no labels."


def test_the_purpose_selects_the_prompt_that_is_sent(picture):
    seen: dict = {}

    def capture(_config, body):
        seen["text"] = body["messages"][0]["content"][0]["text"]
        return {"choices": [{"message": {"content": "ok"}}]}

    request = _request(picture)
    formula = VisionRequest(
        **{
            **{f: getattr(request, f) for f in request.__dataclass_fields__},
            "purpose": VisionPurpose.FORMULA,
        }
    )
    HttpVisionModelAdapter(CONFIG, transport=capture).analyze(formula)
    assert "LaTeX" in seen["text"]


def test_general_prompt_transcribes_tables_without_inventing_metric_semantics(
    picture,
):
    seen: dict = {}

    def capture(_config, body):
        seen["text"] = body["messages"][0]["content"][0]["text"]
        return {"choices": [{"message": {"content": "ok"}}]}

    request = _request(picture)
    general = VisionRequest(
        **{
            **{f: getattr(request, f) for f in request.__dataclass_fields__},
            "purpose": VisionPurpose.GENERAL,
        }
    )

    HttpVisionModelAdapter(CONFIG, transport=capture).analyze(general)

    prompt = seen["text"]
    assert "column headers" in prompt
    assert "Do not calculate" in prompt
    assert "or infer what a separator such as X/Y means" in prompt
    assert "say that its definition is not visible" in prompt


def test_the_same_read_yields_the_same_observation_id(picture):
    adapter = HttpVisionModelAdapter(CONFIG, transport=_reply("stable"))
    first = adapter.analyze(_request(picture))
    second = adapter.analyze(_request(picture))
    assert first.observations[0].observation_id == second.observations[0].observation_id


def test_an_unconfigured_installation_has_no_provider(monkeypatch):
    monkeypatch.setattr(
        "personagraph.configuration.app_settings.get_setting",
        lambda *_a, **_k: "",
    )
    assert load_provider_config() is None


def test_a_partially_configured_installation_is_still_unconfigured(monkeypatch):
    """四项设置只具备三项，无法构成可用端点。"""

    values = {
        "vision_provider": "p",
        "vision_base_url": "https://x.test",
        "vision_model": "m",
    }
    monkeypatch.setattr(
        "personagraph.configuration.app_settings.get_setting",
        lambda key, default=None: values.get(key, ""),
    )
    assert load_provider_config() is None


def test_question_and_actual_image_are_transmitted_together(picture):
    seen: dict = {}

    def capture(_config, body):
        seen.update(body)
        return {"choices": [{"message": {"content": "左侧连接右侧。"}}]}

    question = "左边的形状和右边的形状之间有什么关系？\n请说明可见依据。"
    request = replace(
        _request(picture),
        purpose=VisionPurpose.QUESTION,
        question=question,
        logical_tool_call_id="l1tool_question_a",
    )
    result = HttpVisionModelAdapter(CONFIG, transport=capture).analyze(request)

    content = seen["messages"][0]["content"]
    assert question in content[0]["text"]
    assert "可见" in content[0]["text"]
    assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == picture.read_bytes()
    assert result.observations[0].kind == "question"
    assert result.observations[0].text == "左侧连接右侧。"


def test_question_observation_identity_includes_question_and_host_call(picture):
    adapter = HttpVisionModelAdapter(CONFIG, transport=_reply("answer"))
    request = replace(
        _request(picture),
        purpose=VisionPurpose.QUESTION,
        question="颜色是什么？",
        logical_tool_call_id="l1tool_question_a",
    )

    def observation_id(value):
        return adapter.analyze(value).observations[0].observation_id

    original = observation_id(request)
    assert observation_id(request) == original
    assert observation_id(replace(request, question="形状是什么？")) != original
    assert observation_id(replace(request, logical_tool_call_id="l1tool_question_b")) != original
