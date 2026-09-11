"""Free-form visual questions bind to one Host ToolCall, not an image cache."""

from dataclasses import replace
import hashlib
import json
import sqlite3

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionPurpose,
    VisionRequest,
)
from personagraph.input_processing.vision.imaging import (
    PreparedVisualArtifactBundle,
    prepare_payload_with_receipt,
)
from personagraph.input_processing.vision.providers.http import (
    HttpVisionModelAdapter,
    PROMPT_CONTRACT_VERSION,
    VisionProviderConfig,
)
from personagraph.runtime.model_calls.vision import (
    MountedVisualCallLedgerError,
    MountedVisualCallWaitingExternal,
    MountedVisualPictureLocator,
    MountedVisualProjectPublicationEnvelope,
    MountedVisualProjectPublicationTarget,
    SqliteMountedVisualCallLedger,
)


@pytest.fixture
def question_call(tmp_path):
    source = tmp_path / "visual.png"
    Image.new("RGB", (24, 16), "white").save(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    request = VisionRequest(
        source_unit_id="source_unit_1",
        source_sha256=digest,
        image_sha256=digest,
        locator=DocumentLocator(page=1),
        mime_type="image/png",
        pixel_size=PixelSize(24, 16),
        byte_count=len(raw),
        purpose=VisionPurpose.QUESTION,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        image_path=str(source),
        question="图像中有什么？",
        logical_tool_call_id="l1tool_question_1",
    )
    bundle = prepare_payload_with_receipt(request)
    assert isinstance(bundle, PreparedVisualArtifactBundle)
    calls = []

    def transport(_config, body):
        calls.append(body)
        return {"choices": [{"message": {"content": "白色背景。"}}]}

    adapter = HttpVisionModelAdapter(
        VisionProviderConfig("test", "https://test.invalid", "test-key", "test-vlm"),
        transport=transport,
    )
    ledger = SqliteMountedVisualCallLedger(tmp_path / "vision.sqlite")
    return bundle, adapter, ledger, calls


def _dispatch(ledger, adapter, request, **kwargs):
    return ledger.dispatch_with_receipt(
        session_id="session_question", adapter=adapter, request=request, **kwargs
    )


def test_same_question_new_tool_call_sends_again_but_original_call_replays(
    question_call,
):
    bundle, adapter, ledger, calls = question_call
    request = bundle.prepared_request
    first = _dispatch(ledger, adapter, request)
    second = _dispatch(
        ledger, adapter, replace(request, logical_tool_call_id="l1tool_question_2")
    )
    restored = SqliteMountedVisualCallLedger(ledger.path_for("session_question"))
    replay = _dispatch(restored, adapter, request)

    assert len(calls) == 2
    assert first.call_key != second.call_key
    assert (
        first.result.observations[0].observation_id
        != second.result.observations[0].observation_id
    )
    assert replay == replace(first, replayed=True)


def test_unscored_visual_result_persists_null_and_replays_without_resending(question_call):
    bundle, adapter, ledger, calls = question_call
    first = _dispatch(ledger, adapter, bundle.prepared_request)
    assert first.result.observations[0].uncertainty is None
    with sqlite3.connect(ledger.path_for("session_question")) as conn:
        payload = json.loads(conn.execute(
            "SELECT result_json FROM mounted_visual_provider_calls"
        ).fetchone()[0])
    assert payload["observations"][0]["uncertainty"] is None
    restored = SqliteMountedVisualCallLedger(ledger.path_for("session_question"))
    replay = _dispatch(restored, adapter, bundle.prepared_request)
    assert replay == replace(first, replayed=True)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "drift", ["question", "pixels", "source", "provider", "detail", "prompt", "purpose"]
)
def test_same_tool_call_rejects_request_drift_before_a_second_send(
    question_call, drift
):
    bundle, adapter, ledger, calls = question_call
    request = bundle.prepared_request
    _dispatch(ledger, adapter, request)
    changed = request
    capabilities = adapter.capabilities()
    if drift == "question":
        changed = replace(request, question="另一种颜色是什么？")
    elif drift == "pixels":
        payload = request.prepared_payload
        assert payload is not None
        data = payload.data + b"different"
        changed = replace(
            request,
            prepared_payload=replace(
                payload, data=data, sent_sha256=hashlib.sha256(data).hexdigest()
            ),
        )
    elif drift == "provider":
        capabilities = replace(capabilities, model="another-vlm")
    elif drift == "source":
        changed = replace(request, source_sha256="a" * 64)
    elif drift == "prompt":
        changed = replace(request, prompt_contract_version="changed-prompt")
    elif drift == "purpose":
        changed = replace(request, purpose=VisionPurpose.GENERAL, question=None)
    else:
        from personagraph.input_processing.vision.contracts import VisionDetail

        changed = replace(request, detail=VisionDetail.HIGH)
    with pytest.raises(
        MountedVisualCallLedgerError, match="immutable request authority"
    ):
        _dispatch(ledger, adapter, changed, capabilities=capabilities)
    assert len(calls) == 1


def test_pending_question_call_does_not_resend_after_process_loss(question_call):
    bundle, adapter, ledger, calls = question_call

    def lost(_config, body):
        calls.append(body)
        raise KeyboardInterrupt("simulated process loss")

    adapter._transport = lost
    with pytest.raises(KeyboardInterrupt):
        _dispatch(ledger, adapter, bundle.prepared_request)
    restored = SqliteMountedVisualCallLedger(ledger.path_for("session_question"))
    with pytest.raises(MountedVisualCallWaitingExternal):
        _dispatch(restored, adapter, bundle.prepared_request)
    assert len(calls) == 1


@pytest.mark.parametrize("uncertain", [False, True])
def test_provider_failure_keeps_classification_without_automatic_resend(question_call, uncertain):
    import urllib.error

    bundle, adapter, ledger, calls = question_call

    def failed(_config, body):
        calls.append(body)
        if uncertain:
            raise TimeoutError("sensitive message must not escape")
        raise urllib.error.HTTPError("https://private.invalid", 429, "private body", {}, None)

    adapter._transport = failed
    if uncertain:
        with pytest.raises(MountedVisualCallWaitingExternal) as caught:
            _dispatch(ledger, adapter, bundle.prepared_request)
        assert caught.value.reason_code == "vision_request_timeout"
        details = caught.value.diagnostic_details()
        assert details["failure_diagnostics"]["exception_type"] == "TimeoutError"
        assert details["background_wait_active"] is False
        with pytest.raises(MountedVisualCallWaitingExternal) as replay:
            _dispatch(ledger, adapter, bundle.prepared_request)
        assert replay.value.reason_code == "vision_request_timeout"
    else:
        first = _dispatch(ledger, adapter, bundle.prepared_request)
        assert first.result.failure_code == "vision_http_rate_limited"
        assert first.result.failure_diagnostics.http_status == 429
        replay = _dispatch(ledger, adapter, bundle.prepared_request)
        assert replay.result == first.result
        assert replay.replayed is True
    assert len(calls) == 1


@pytest.mark.parametrize(
    "purpose", [value for value in VisionPurpose if value is not VisionPurpose.QUESTION]
)
def test_legacy_modes_keep_their_existing_content_key_and_request_fields(
    question_call, purpose
):
    bundle, adapter, ledger, calls = question_call
    request = replace(bundle.prepared_request, purpose=purpose, question=None)
    first = _dispatch(ledger, adapter, request)
    replay = _dispatch(
        ledger, adapter, replace(request, logical_tool_call_id="l1tool_legacy_2")
    )
    assert replay == replace(first, replayed=True)
    assert len(calls) == 1
    with sqlite3.connect(ledger.path_for("session_question")) as conn:
        stored = json.loads(
            conn.execute(
                "SELECT request_json FROM mounted_visual_provider_calls"
            ).fetchone()[0]
        )
    assert stored["schema_version"] == "mounted-visual-bound-request-v2"
    assert "question" not in stored
    assert "logical_tool_call_id" not in stored


def test_existing_legacy_cache_cannot_hide_a_question_call_mode_change(question_call):
    bundle, adapter, ledger, calls = question_call
    question = bundle.prepared_request
    legacy = replace(question, purpose=VisionPurpose.GENERAL, question=None)
    _dispatch(ledger, adapter, replace(legacy, logical_tool_call_id="l1tool_legacy"))
    _dispatch(ledger, adapter, question)
    with pytest.raises(
        MountedVisualCallLedgerError, match="immutable request authority"
    ):
        _dispatch(ledger, adapter, legacy)
    assert len(calls) == 2


def test_question_publication_freezes_question_and_call_identity(question_call):
    bundle, adapter, ledger, calls = question_call
    request = bundle.prepared_request
    target = MountedVisualProjectPublicationTarget(
        project_id="project_question",
        file_id="file_question",
        file_version_id="version_question",
        file_content_sha256=request.source_sha256,
        file_media_type="image/png",
        purpose=request.purpose,
        prompt_contract_version=request.prompt_contract_version,
        picture_source_kind="whole_file",
        picture_source_locator=MountedVisualPictureLocator.from_payload(
            kind="whole_file", payload={}
        ),
        picture_unit_kind="full",
        picture_unit_locator=MountedVisualPictureLocator.from_payload(
            kind="full", payload={}
        ),
        prepared_artifact=bundle.receipt,
        question=request.question,
        logical_tool_call_id=request.logical_tool_call_id,
    )
    envelope = MountedVisualProjectPublicationEnvelope.from_target(target)
    assert envelope.target == target
    first = _dispatch(ledger, adapter, request, publication_envelope=envelope)
    ready = ledger.list_ready_publications(session_id="session_question")
    assert ready == (replace(first, replayed=True),)
    conflict = MountedVisualProjectPublicationEnvelope.from_target(
        replace(target, question="不同问题？")
    )
    with pytest.raises(MountedVisualCallLedgerError, match="does not bind"):
        _dispatch(ledger, adapter, request, publication_envelope=conflict)
    assert len(calls) == 1
    with sqlite3.connect(ledger.path_for("session_question")) as conn:
        stored = conn.execute(
            "SELECT request_json FROM mounted_visual_provider_calls"
        ).fetchone()[0]
    assert request.question in stored
    assert request.logical_tool_call_id in stored
