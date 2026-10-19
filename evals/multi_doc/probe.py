"""Minimal real requests through the configured task and vision adapters."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def probe_task() -> dict:
    from personagraph.model_io.gateway import (
        anthropic_compatible_chat, openai_compatible_chat,
    )
    from personagraph.model_io.tier_bindings import ModelTier, resolve_tier

    binding = resolve_tier(ModelTier.L1)
    gateways = {
        "openai-compatible": openai_compatible_chat,
        "anthropic-compatible": anthropic_compatible_chat,
    }
    started = time.monotonic()
    result = gateways[binding.provider](
        [{"role": "user", "content": "Reply with exactly API_OK."}],
        binding=binding, temperature=0, max_tokens=64, timeout_s=45,
        purpose="multi_doc_connectivity_probe",
    )
    return {
        "ok": bool(result.reply.strip()), "model": binding.model,
        "profile": binding.profile_name, "reply": result.reply,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def probe_vision(directory: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    from personagraph.input_processing.documents.contracts import DocumentLocator
    from personagraph.input_processing.vision.contracts import (
        PixelSize, VisionPurpose, VisionRequest, VisionStatus,
    )
    from personagraph.input_processing.vision.providers.http import (
        HttpVisionModelAdapter, load_provider_config,
    )

    config = load_provider_config()
    if config is None:
        return {"ok": False, "error_code": "vision_not_configured"}
    path = directory / "api_probe.png"
    picture = Image.new("RGB", (640, 180), "white")
    ImageDraw.Draw(picture).text(
        (45, 55), "VISION 731", fill="black", font=ImageFont.load_default(size=48),
    )
    picture.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    request = VisionRequest(
        source_unit_id="multi-doc-probe", source_sha256=digest, image_sha256=digest,
        locator=DocumentLocator(page=1), mime_type="image/png",
        pixel_size=PixelSize(640, 180), byte_count=path.stat().st_size,
        purpose=VisionPurpose.QUESTION, prompt_contract_version="multi-doc-probe-v1",
        image_path=str(path), question="Read the text in this image.",
        logical_tool_call_id="multi-doc-api-probe",
    )
    started = time.monotonic()
    result = HttpVisionModelAdapter(replace(config, timeout_s=45, max_tokens=128)).analyze(request)
    reply = "\n".join(item.text for item in result.observations)
    return {
        "ok": result.status == VisionStatus.COMPLETED and "731" in reply,
        "model": config.model, "reply": reply, "error_code": result.failure_code,
        "diagnostics": result.failure_diagnostics.to_dict() if result.failure_diagnostics else None,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def safe_probe(callback) -> dict:
    try:
        return callback()
    except Exception as exc:
        # Provider exception bodies can contain credentials or request content.
        details = getattr(exc, "details", {})
        return {
            "ok": False, "exception_type": type(exc).__name__,
            "error_code": getattr(exc, "code", None),
            "status_code": details.get("status_code"),
        }


def run_probes(directory: Path) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        task = pool.submit(safe_probe, probe_task)
        vision = pool.submit(safe_probe, lambda: probe_vision(directory))
        return {"task": task.result(), "vision": vision.result()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true", required=True)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="entelecheia-multi-doc-probe-") as directory:
        os.environ["PERSONAGRAPH_STATE_DIR"] = str(Path(directory) / "state")
        os.environ["PERSONAGRAPH_API_TOKEN_FILE"] = str(Path(directory) / "state" / "api_secret")
        os.environ["PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH"] = str(Path(directory) / "quota.sqlite3")
        from personagraph.configuration.paths import load_dotenv

        load_dotenv()
        result = run_probes(Path(directory))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if all(item["ok"] for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
