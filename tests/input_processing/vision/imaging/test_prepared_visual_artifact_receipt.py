"""Prepared visual receipts bind sent pixels without retaining source authority data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionDetail,
    VisionPurpose,
    VisionRequest,
)
from personagraph.input_processing.vision.imaging import (
    PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT,
    PREPARED_VISUAL_RENDER_RECIPE_CONTRACT,
    PreparedVisualArtifactBundle,
    PreparedVisualArtifactReceipt,
    VisionPayload,
    build_prepared_visual_artifact_receipt,
    prepare_payload_with_receipt,
)


def _prepared(tmp_path: Path) -> tuple[VisionRequest, VisionPayload, Path]:
    source = tmp_path / "private-source.png"
    image = Image.new("RGBA", (17, 11), (20, 40, 60, 128))
    image.save(source, "PNG")
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    request = VisionRequest(
        source_unit_id="visual_unit_receipt_01",
        source_sha256=source_sha256,
        image_sha256=source_sha256,
        locator=DocumentLocator(page=1, ordinal=2, section_path=("Chart",)),
        mime_type="image/png",
        pixel_size=PixelSize(17, 11),
        byte_count=len(raw),
        purpose=VisionPurpose.CHART,
        prompt_contract_version="vision-purpose-v1",
        image_path=str(source),
    )
    bundle = prepare_payload_with_receipt(request)
    assert isinstance(bundle, PreparedVisualArtifactBundle)
    return bundle.prepared_request, bundle.payload, source


def test_receipt_binds_actual_png_and_canonical_rgba_pixels_without_path(
    tmp_path: Path,
) -> None:
    request, payload, source = _prepared(tmp_path)

    receipt = build_prepared_visual_artifact_receipt(request, payload)

    with Image.open(source) as opened:
        rgba = opened.convert("RGB").convert("RGBA")
        pixel_domain = (
            b"personagraph-picture-rgba-pixels-v1\0"
            + b"17x11\0RGBA\0"
            + rgba.tobytes()
        )
    assert receipt.sent_sha256 == hashlib.sha256(payload.data).hexdigest()
    assert receipt.pixel_sha256 == hashlib.sha256(pixel_domain).hexdigest()
    assert (receipt.width, receipt.height, receipt.media_type) == (
        17,
        11,
        "image/png",
    )
    recipe = json.loads(receipt.canonical_render_recipe)
    assert recipe["contract_version"] == PREPARED_VISUAL_RENDER_RECIPE_CONTRACT
    assert recipe["result"]["sent_sha256"] == payload.sent_sha256
    assert recipe["implementation"]["pillow_version"]
    assert receipt.preparation_fingerprint == hashlib.sha256(
        json.dumps(
            recipe["implementation"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    serialized = json.dumps(receipt.as_payload(), sort_keys=True)
    assert PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT in serialized
    assert str(source) not in serialized
    assert payload.data.hex() not in serialized


def test_receipt_builder_never_reopens_the_source_and_is_deterministic(
    tmp_path: Path,
) -> None:
    request, payload, source = _prepared(tmp_path)
    source.unlink()

    first = build_prepared_visual_artifact_receipt(request, payload)
    second = build_prepared_visual_artifact_receipt(request, payload)

    assert first == second


def test_receipt_refuses_bytes_that_do_not_match_the_claimed_sent_hash(
    tmp_path: Path,
) -> None:
    request, payload, _source = _prepared(tmp_path)
    # Frozen DTOs prevent ordinary mutation. Bypass that guard deliberately to
    # prove the receipt builder independently rechecks the bytes at its boundary.
    object.__setattr__(payload, "data", payload.data + b"tampered")

    with pytest.raises(ValueError, match="sent hash"):
        build_prepared_visual_artifact_receipt(request, payload)


def test_canonical_bundle_records_the_effective_detail_override(tmp_path: Path) -> None:
    request, _payload, _source = _prepared(tmp_path)
    unprepared = replace(request, prepared_payload=None, detail=VisionDetail.STANDARD)

    bundle = prepare_payload_with_receipt(unprepared, detail=VisionDetail.LOW)

    assert isinstance(bundle, PreparedVisualArtifactBundle)
    assert bundle.prepared_request.detail is VisionDetail.LOW
    assert bundle.payload.prepared_detail is VisionDetail.LOW
    recipe = json.loads(bundle.receipt.canonical_render_recipe)
    assert recipe["request"]["detail"] == "low"


def test_receipt_rehydration_rejects_extra_fields(tmp_path: Path) -> None:
    request, payload, _source = _prepared(tmp_path)
    receipt = build_prepared_visual_artifact_receipt(request, payload)
    encoded = receipt.as_payload()
    encoded["result_text"] = "must not enter the pointer-only receipt"

    with pytest.raises(ValueError, match="unsupported fields"):
        PreparedVisualArtifactReceipt.from_payload(encoded)
