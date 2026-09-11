from dataclasses import replace  # noqa: F401

import pytest
from PIL import Image

from personagraph.input_processing.files import fingerprint_file
from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot, VisionObservation, VisionPurpose, VisionResult, VisionStatus,
)
from personagraph.runtime.model_calls.vision import SqliteMountedVisualCallLedger
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.visual.file_visual_adapter import build_file_visual_runtime
from personagraph.workspace.files import FileSource
from personagraph.workspace.files.access import AuthorizedFileSource


class _LocalProvider:
    transmits_externally = False

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return VisionCapabilitySnapshot(
            available=True, provider="local-test", model="fixture", endpoint_identity="local",
            processor_fingerprint="local-test@1", supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):
        self.calls.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED, provider="local-test", model="fixture",
            endpoint_identity="local", processor_fingerprint="local-test@1",
            input_sha256=request.image_sha256,
            observations=(VisionObservation("provider-only-id", "chart", "Three bars", 0.1),),
        )


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (24, 16)).save(path)
    source = AuthorizedFileSource(
        project_id="project", file_id="file-1", file_version_id="version-1",
        canonical_path=str(path), relative_path=path.name, file_name=path.name,
        origin=FileSource.WORKSPACE_EXISTING, media_type="image/png",
        fingerprint=fingerprint_file(path),
    )
    provider = _LocalProvider()
    allowed = {"value": True}
    resolutions = []
    def resolve(**kwargs):
        resolutions.append(kwargs)
        return source
    runtime = build_file_visual_runtime(
        session_id="session", resolve_file=resolve,
        revalidate_source=lambda _: allowed["value"], adapter=provider,

        call_ledger=SqliteMountedVisualCallLedger(tmp_path / "calls.sqlite"),
    )
    return runtime, source, provider, allowed, resolutions


def _selection(source):
    return {"file_id": source.file_id, "file_version_id": source.file_version_id}


def _request(source, **overrides):
    return {**_selection(source), "visual_unit_id": "whole_file",
            "purpose": "chart", "detail": "standard", "region": "detected", **overrides}


def test_construction_and_list_never_call_provider_or_prepare_documents(setup):
    runtime, source, provider, _, resolutions = setup
    assert resolutions == []
    listed = runtime.list_file_visuals(_selection(source))
    assert listed["status"] == "ready"
    assert listed["visuals"][0]["visual_unit_id"] == "whole_file"
    assert provider.calls == []
    assert resolutions == [_selection(source)]


def test_read_uses_exact_file_version_and_does_not_advertise_provider_ids_as_persisted(setup):
    runtime, source, provider, _, _ = setup
    result = runtime.read_file_visuals({"requests": [_request(source)]})["results"][0]
    assert result["status"] == "completed"
    assert result["file_id"] == source.file_id
    assert result["file_version_id"] == source.file_version_id
    assert result["observation"] == "Three bars"
    assert "observation_id" not in result
    assert len(provider.calls) == 1


@pytest.mark.parametrize("second", [
    {"visual_unit_id": "not_a_unit"}, {"file_version_id": "wrong-version"},
    {"purpose": "formula"},
])
def test_complete_batch_is_validated_before_any_provider_call(setup, second):
    runtime, source, provider, _, _ = setup
    with pytest.raises(ToolBusinessFailure):
        runtime.read_file_visuals({"requests": [_request(source), _request(source, **second)]})
    assert provider.calls == []


def test_revoked_file_is_rejected_after_prior_listing(setup):
    runtime, source, provider, allowed, _ = setup
    runtime.list_file_visuals(_selection(source))
    allowed["value"] = False
    with pytest.raises(ToolBusinessFailure, match="File visual authority"):
        runtime.read_file_visuals({"requests": [_request(source)]})
    assert provider.calls == []


def test_wrong_cursor_and_duplicate_units_fail_closed(setup):
    runtime, source, provider, _, _ = setup
    with pytest.raises(ToolBusinessFailure, match="cursor"):
        runtime.list_file_visuals({**_selection(source), "cursor": "visualcursor_0_" + "0" * 64})
    with pytest.raises(ToolBusinessFailure, match="once per call"):
        runtime.read_file_visuals({"requests": [_request(source), _request(source)]})
    assert provider.calls == []


def test_external_visual_authority_is_exact_and_rechecked_before_dispatch(setup, tmp_path):
    _, source, provider, allowed, _ = setup
    provider.transmits_externally = True
    runtime = build_file_visual_runtime(
        session_id="session", resolve_file=lambda **_: source,
        revalidate_source=lambda _: allowed["value"], adapter=provider,

        call_ledger=SqliteMountedVisualCallLedger(tmp_path / "external-calls.sqlite"),
    )
    payload = {"requests": [_request(source)]}
    assert runtime.authority.approval_grants == ()
    prepared = runtime.prepare_read_authority(payload)
    assert prepared.protected_authority.approval_receipt_ids[0].startswith("auto_visual_egress_")
    assert prepared.protected_authority.revalidate()
    allowed["value"] = False
    assert prepared.protected_authority.revalidate() is False
    assert provider.calls == []
