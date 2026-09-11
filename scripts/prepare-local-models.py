#!/usr/bin/env python3
"""Explicit preparation of the same pinned retrieval assets used at inference.

`check` is local-only. `download` is an opt-in public Hub transfer; neither
command loads model weights, opens application state, or calls an LLM provider.
Set HF_HOME / HF_HUB_CACHE consistently for preparation and application startup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Native model/tokenizer/projection files only, not duplicate ONNX exports or images.
DOWNLOAD_PATTERNS = (
    "*.json", "*.safetensors", "*.bin", "*.model", "*.txt",
    "sparse_linear.pt", "colbert_linear.pt",
)


def prepare(command: str) -> dict[str, object]:
    from personagraph.retrieval.indexing.model_assets import (
        BGE_M3_HYBRID_MANIFEST,
        BGE_M3_MODEL_ID,
        BGE_M3_MODEL_REVISION,
        BGE_V2_M3_RERANKER_MANIFEST,
        BGE_V2_M3_RERANKER_MODEL_ID,
        BGE_V2_M3_RERANKER_REVISION,
        LocalModelAssetRef,
        preflight_local_model_asset,
    )

    if command not in {"check", "download"}:
        raise ValueError("expected check or download")
    assets = (
        (
            LocalModelAssetRef.hub(BGE_M3_MODEL_ID, revision=BGE_M3_MODEL_REVISION),
            BGE_M3_HYBRID_MANIFEST,
        ),
        (
            LocalModelAssetRef.hub(
                BGE_V2_M3_RERANKER_MODEL_ID,
                revision=BGE_V2_M3_RERANKER_REVISION,
            ),
            BGE_V2_M3_RERANKER_MANIFEST,
        ),
    )
    failures = []
    if command == "download":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return {
                "status": "not_ready",
                "scope": "retrieval",
                "reason_code": "model_runtime_dependency_missing",
                "next_action": "run scripts/bootstrap-local-runtime.sh",
            }
        for reference, _manifest in assets:
            try:
                snapshot_download(
                    repo_id=reference.identifier,
                    revision=reference.revision,
                    local_files_only=False,
                    token=False,
                    allow_patterns=list(DOWNLOAD_PATTERNS),
                )
            except Exception as exc:
                # Do not echo URLs, headers, credentials, or local exception payloads.
                failures.append({
                    "asset_identity": reference.identity,
                    "reason_code": "model_download_failed",
                    "exception_type": type(exc).__name__,
                })
    capabilities = [
        preflight_local_model_asset(reference, manifest).diagnostic_snapshot()
        for reference, manifest in assets
    ]
    ready = not failures and all(item["ready"] for item in capabilities)
    return {
        "status": "ready" if ready else "not_ready",
        "scope": "retrieval",
        "command": command,
        "assets": capabilities,
        "download_failures": failures,
        "next_action": (
            "run the offline retrieval smoke; Docling layout assets are separate"
            if ready else
            "install the locked dependencies, then explicitly run this script with download"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "download"))
    arguments = parser.parse_args(argv)
    result = prepare(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
