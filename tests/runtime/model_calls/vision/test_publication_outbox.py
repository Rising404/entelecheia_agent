"""Crash-safe provider receipts and the pointer-only Project publication outbox."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from personagraph.input_processing.documents.contracts import DocumentLocator
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from personagraph.runtime.model_calls.vision import (
    MountedVisualCallLedgerError,
    MountedVisualCallWaitingExternal,
    MountedVisualPictureLocator,
    MountedVisualProjectPublicationEnvelope,
    MountedVisualProjectPublicationTarget,
    MountedVisualPublicationState,
    SqliteMountedVisualCallLedger,
)
from personagraph.input_processing.vision.imaging import (
    PreparedVisualArtifactBundle,
    VisionPayload,
    prepare_payload_with_receipt,
)


class _RecordingExternalAdapter:
    transmits_externally = True

    def __init__(
        self,
        *,
        result_status: VisionStatus = VisionStatus.COMPLETED,
        before_result: Callable[[], None] | None = None,
        raises: Exception | None = None,
        input_sha256: str | None = None,
        uncertainty: float | None = 0.1,
    ) -> None:
        self.calls = 0
        self.result_status = result_status
        self.before_result = before_result
        self.raises = raises
        self.input_sha256 = input_sha256
        self.uncertainty = uncertainty

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider="publication-test",
            model="vl-test",
            endpoint_identity="publication-test-endpoint",
            processor_fingerprint="publication-test-processor-v1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request: VisionRequest) -> VisionResult:
        self.calls += 1
        if self.before_result is not None:
            self.before_result()
        if self.raises is not None:
            raise self.raises
        prepared = request.prepared_payload
        assert prepared is not None
        common = {
            "provider": "publication-test",
            "model": "vl-test",
            "endpoint_identity": "publication-test-endpoint",
            "processor_fingerprint": "publication-test-processor-v1",
            "input_sha256": self.input_sha256 or prepared.sent_sha256,
        }
        if self.result_status is VisionStatus.COMPLETED:
            return VisionResult(
                status=VisionStatus.COMPLETED,
                observations=(
                    VisionObservation(
                        observation_id="provider_observation_01",
                        kind="chart",
                        text="The blue bar is taller than the gray bar.",
                        uncertainty=self.uncertainty,
                    ),
                ),
                **common,
            )
        return VisionResult(
            status=self.result_status,
            unresolved_gap_refs=(request.source_unit_id,),
            failure_code=f"visual_{self.result_status.value}",
            **common,
        )


def _request_and_envelope(
    tmp_path: Path,
) -> tuple[VisionRequest, MountedVisualProjectPublicationEnvelope, Path]:
    source = tmp_path / "visual-source.png"
    Image.new("RGB", (24, 16), (32, 64, 128)).save(source, "PNG")
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    unprepared = VisionRequest(
        source_unit_id="visual_unit_outbox_01",
        source_sha256=source_sha256,
        image_sha256=source_sha256,
        locator=DocumentLocator(page=1, ordinal=1, section_path=("Results",)),
        mime_type="image/png",
        pixel_size=PixelSize(24, 16),
        byte_count=len(raw),
        purpose=VisionPurpose.CHART,
        prompt_contract_version="vision-purpose-v1",
        image_path=str(source),
    )
    bundle = prepare_payload_with_receipt(unprepared)
    assert isinstance(bundle, PreparedVisualArtifactBundle)
    target = MountedVisualProjectPublicationTarget(
        project_id="project_01",
        file_id="file_01",
        file_version_id="file_version_01",
        file_content_sha256=source_sha256,
        file_media_type="image/png",
        purpose=unprepared.purpose,
        prompt_contract_version=unprepared.prompt_contract_version,
        picture_source_kind="whole_file",
        picture_source_locator=MountedVisualPictureLocator.from_payload(
            kind="whole_file",
            payload={},
        ),
        picture_unit_kind="full",
        picture_unit_locator=MountedVisualPictureLocator.from_payload(
            kind="full",
            payload={},
        ),
        prepared_artifact=bundle.receipt,
    )
    envelope = MountedVisualProjectPublicationEnvelope.from_target(target)
    return bundle.prepared_request, envelope, source


def _publication_state(path: Path) -> tuple[str, str, str | None]:
    with sqlite3.connect(path) as connection:
        provider = connection.execute(
            "SELECT status FROM mounted_visual_provider_calls"
        ).fetchone()[0]
        publication = connection.execute(
            "SELECT state, skip_reason FROM mounted_visual_project_publications"
        ).fetchone()
    return provider, publication[0], publication[1]


@pytest.mark.parametrize("uncertainty", [None, 0.1])
def test_reservation_and_settlement_create_a_ready_replayable_receipt(
    tmp_path: Path,
    uncertainty: float | None,
) -> None:
    request, envelope, source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "publication-ledger.sqlite"

    def assert_reserved() -> None:
        assert _publication_state(ledger_path) == (
            "pending",
            "awaiting_result",
            None,
        )

    adapter = _RecordingExternalAdapter(before_result=assert_reserved, uncertainty=uncertainty)
    ledger = SqliteMountedVisualCallLedger(ledger_path)

    first = ledger.dispatch_with_receipt(
        session_id="session_publication_01",
        adapter=adapter,
        request=request,
        publication_envelope=envelope,
    )
    source.unlink()
    ready = ledger.list_ready_publications(session_id="session_publication_01")
    replay = ledger.dispatch_with_receipt(
        session_id="session_publication_01",
        adapter=adapter,
        request=request,
        publication_envelope=envelope,
    )

    assert first.publication_state is MountedVisualPublicationState.READY
    assert first.result.observations[0].uncertainty == uncertainty
    assert first.replayed is False
    assert ready == (replace(first, replayed=True),)
    assert replay == replace(first, replayed=True)
    assert adapter.calls == 1
    assert _publication_state(ledger_path) == ("succeeded", "ready", None)
    with sqlite3.connect(ledger_path) as connection:
        envelope_json = connection.execute(
            "SELECT envelope_json FROM mounted_visual_project_publications"
        ).fetchone()[0]
    lowered = envelope_json.casefold()
    assert str(source) not in envelope_json
    assert "base64" not in lowered
    assert "secret" not in lowered


def test_published_and_skipped_are_terminal_and_idempotent(tmp_path: Path) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger = SqliteMountedVisualCallLedger(tmp_path / "terminal-ledger.sqlite")
    adapter = _RecordingExternalAdapter()
    ready = ledger.dispatch_with_receipt(
        session_id="session_terminal",
        adapter=adapter,
        request=request,
        publication_envelope=envelope,
    )

    published = ledger.mark_publication_published(ready)
    published_again = ledger.mark_publication_published(ready)

    assert published.publication_state is MountedVisualPublicationState.PUBLISHED
    assert published_again.publication_state is MountedVisualPublicationState.PUBLISHED
    assert ledger.list_ready_publications(session_id="session_terminal") == ()
    with pytest.raises(MountedVisualCallLedgerError, match="terminal state"):
        ledger.mark_publication_skipped(published, reason_code="project_binding_stale")


def test_skipped_is_terminal_and_idempotent_only_for_the_same_reason(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger = SqliteMountedVisualCallLedger(tmp_path / "skipped-terminal.sqlite")
    ready = ledger.dispatch_with_receipt(
        session_id="session_skipped_terminal",
        adapter=_RecordingExternalAdapter(),
        request=request,
        publication_envelope=envelope,
    )

    skipped = ledger.mark_publication_skipped(
        ready,
        reason_code="project_binding_stale",
    )
    skipped_again = ledger.mark_publication_skipped(
        ready,
        reason_code="project_binding_stale",
    )

    assert skipped.publication_state is MountedVisualPublicationState.SKIPPED
    assert skipped_again.publication_state is MountedVisualPublicationState.SKIPPED
    with pytest.raises(MountedVisualCallLedgerError, match="reason changed"):
        ledger.mark_publication_skipped(ready, reason_code="file_version_stale")
    with pytest.raises(MountedVisualCallLedgerError, match="terminal state"):
        ledger.mark_publication_published(skipped)


def test_conflicting_envelope_fails_closed_without_resending_provider(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger = SqliteMountedVisualCallLedger(tmp_path / "conflict-ledger.sqlite")
    adapter = _RecordingExternalAdapter()
    ledger.dispatch_with_receipt(
        session_id="session_conflict",
        adapter=adapter,
        request=request,
        publication_envelope=envelope,
    )
    conflict = MountedVisualProjectPublicationEnvelope.from_target(
        replace(envelope.target, file_version_id="file_version_02")
    )

    with pytest.raises(MountedVisualCallLedgerError, match="publication authority"):
        ledger.dispatch_with_receipt(
            session_id="session_conflict",
            adapter=adapter,
            request=request,
            publication_envelope=conflict,
        )

    assert adapter.calls == 1


def test_legacy_succeeded_call_attaches_outbox_without_provider_resend(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "legacy-ledger.sqlite"
    ledger = SqliteMountedVisualCallLedger(ledger_path)
    adapter = _RecordingExternalAdapter()

    legacy_result = ledger.dispatch(
        session_id="session_legacy",
        adapter=adapter,
        request=request,
    )
    # Simulate the provider-only schema that predates publication support. Schema
    # initialization must rebuild the child table without invalidating the call.
    with sqlite3.connect(ledger_path) as connection:
        connection.execute("DROP TABLE mounted_visual_project_publications")
    attached = ledger.dispatch_with_receipt(
        session_id="session_legacy",
        adapter=adapter,
        request=request,
        publication_envelope=envelope,
    )

    assert attached.result == legacy_result
    assert attached.replayed is True
    assert attached.publication_state is MountedVisualPublicationState.READY
    assert adapter.calls == 1
    assert _publication_state(ledger_path) == ("succeeded", "ready", None)


def test_receipt_payload_mismatch_fails_before_provider_or_reservation(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    alternate_buffer = io.BytesIO()
    Image.new("RGB", (24, 16), (255, 0, 0)).save(alternate_buffer, "PNG")
    alternate_bytes = alternate_buffer.getvalue()
    original = request.prepared_payload
    assert original is not None
    alternate = VisionPayload(
        data=alternate_bytes,
        mime_type="image/png",
        pixel_size=PixelSize(24, 16),
        sent_sha256=hashlib.sha256(alternate_bytes).hexdigest(),
        source_sha256=original.source_sha256,
        resampled=False,
        prepared_detail=request.detail,
    )
    mismatched_request = replace(request, prepared_payload=alternate)
    ledger_path = tmp_path / "payload-mismatch.sqlite"
    adapter = _RecordingExternalAdapter()

    with pytest.raises(
        MountedVisualCallLedgerError,
        match="does not bind the prepared request",
    ):
        SqliteMountedVisualCallLedger(ledger_path).dispatch_with_receipt(
            session_id="session_payload_mismatch",
            adapter=adapter,
            request=mismatched_request,
            publication_envelope=envelope,
        )

    assert adapter.calls == 0
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "target_change",
    [
        {"purpose": VisionPurpose.FORMULA},
        {"prompt_contract_version": "vision-purpose-v2"},
    ],
    ids=("purpose", "prompt-contract-version"),
)
def test_publication_semantics_must_match_request_before_provider_io(
    tmp_path: Path,
    target_change: dict[str, object],
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    conflict = MountedVisualProjectPublicationEnvelope.from_target(
        replace(envelope.target, **target_change)
    )
    adapter = _RecordingExternalAdapter()
    ledger_path = tmp_path / "semantic-mismatch.sqlite"

    with pytest.raises(
        MountedVisualCallLedgerError,
        match="does not bind the prepared request",
    ):
        SqliteMountedVisualCallLedger(ledger_path).dispatch_with_receipt(
            session_id="session_semantic_mismatch",
            adapter=adapter,
            request=request,
            publication_envelope=conflict,
        )

    assert adapter.calls == 0
    assert not ledger_path.exists()


def test_wrong_provider_input_hash_is_uncertain_and_never_ready_or_resent(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "wrong-provider-input.sqlite"
    ledger = SqliteMountedVisualCallLedger(ledger_path)
    adapter = _RecordingExternalAdapter(input_sha256="f" * 64)

    with pytest.raises(MountedVisualCallWaitingExternal):
        ledger.dispatch_with_receipt(
            session_id="session_wrong_provider_input",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )
    with pytest.raises(MountedVisualCallWaitingExternal):
        ledger.dispatch_with_receipt(
            session_id="session_wrong_provider_input",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )

    assert adapter.calls == 1
    assert _publication_state(ledger_path) == (
        "uncertain",
        "awaiting_result",
        None,
    )
    assert (
        ledger.list_ready_publications(
            session_id="session_wrong_provider_input",
        )
        == ()
    )


@pytest.mark.parametrize(
    "status, expected_reason",
    [
        (VisionStatus.FAILED, "visual_result_failed"),
        (VisionStatus.UNAVAILABLE, "visual_result_unavailable"),
    ],
)
def test_nonsemantic_provider_result_is_durably_skipped(
    tmp_path: Path,
    status: VisionStatus,
    expected_reason: str,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / f"skipped-{status.value}.sqlite"
    receipt = SqliteMountedVisualCallLedger(ledger_path).dispatch_with_receipt(
        session_id=f"session_{status.value}",
        adapter=_RecordingExternalAdapter(result_status=status),
        request=request,
        publication_envelope=envelope,
    )

    assert receipt.publication_state is MountedVisualPublicationState.SKIPPED
    assert _publication_state(ledger_path) == (
        "succeeded",
        "skipped",
        expected_reason,
    )


def test_provider_uncertainty_keeps_publication_awaiting_and_never_retries(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "uncertain-ledger.sqlite"
    ledger = SqliteMountedVisualCallLedger(ledger_path)
    adapter = _RecordingExternalAdapter(raises=RuntimeError("response lost"))

    with pytest.raises(MountedVisualCallWaitingExternal):
        ledger.dispatch_with_receipt(
            session_id="session_uncertain",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )
    with pytest.raises(MountedVisualCallWaitingExternal):
        ledger.dispatch_with_receipt(
            session_id="session_uncertain",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )

    assert adapter.calls == 1
    assert _publication_state(ledger_path) == (
        "uncertain",
        "awaiting_result",
        None,
    )
    assert ledger.list_ready_publications(session_id="session_uncertain") == ()


def test_publication_insert_failure_rolls_back_provider_reservation(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "reserve-atomic.sqlite"
    ledger = SqliteMountedVisualCallLedger(ledger_path)
    ledger.list_ready_publications(session_id="session_reserve_atomic")
    with sqlite3.connect(ledger_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_publication_insert "
            "BEFORE INSERT ON mounted_visual_project_publications "
            "BEGIN SELECT RAISE(ABORT, 'reject publication'); END"
        )
    adapter = _RecordingExternalAdapter()

    with pytest.raises(MountedVisualCallLedgerError):
        ledger.dispatch_with_receipt(
            session_id="session_reserve_atomic",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )

    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM mounted_visual_provider_calls"
        ).fetchone()[0] == 0
    assert adapter.calls == 0


def test_publication_transition_failure_rolls_back_provider_settlement(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "settle-atomic.sqlite"

    def install_rejecting_trigger() -> None:
        with sqlite3.connect(ledger_path) as connection:
            connection.execute(
                "CREATE TRIGGER reject_publication_ready "
                "BEFORE UPDATE OF state ON mounted_visual_project_publications "
                "WHEN NEW.state='ready' "
                "BEGIN SELECT RAISE(ABORT, 'reject ready'); END"
            )

    adapter = _RecordingExternalAdapter(before_result=install_rejecting_trigger)
    ledger = SqliteMountedVisualCallLedger(ledger_path)

    with pytest.raises(MountedVisualCallWaitingExternal):
        ledger.dispatch_with_receipt(
            session_id="session_settle_atomic",
            adapter=adapter,
            request=request,
            publication_envelope=envelope,
        )

    assert _publication_state(ledger_path) == (
        "pending",
        "awaiting_result",
        None,
    )
    assert adapter.calls == 1


def test_typed_locator_allows_package_parts_and_ordinary_secretary_text(
    tmp_path: Path,
) -> None:
    _request, envelope, _source = _request_and_envelope(tmp_path)
    locator = MountedVisualPictureLocator.from_payload(
        kind="embedded_asset",
        payload={
            "package_part": "word/media/secretary_chart.png",
            "relationship": "secretary",
        },
    )
    typed = MountedVisualProjectPublicationEnvelope.from_target(
        replace(
            envelope.target,
            picture_source_kind="embedded_asset",
            picture_source_locator=locator,
        )
    )

    assert typed.target.picture_source_locator == locator


@pytest.mark.parametrize(
    "payload",
    [
        {"image_path": "private.png"},
        {"package_part": "/Users/private/source.png"},
        {"result_text": "provider output must never become a locator"},
        {"raw_bytes": "not-even-content"},
    ],
    ids=("path-key", "absolute-path-value", "result-text", "bytes-key"),
)
def test_typed_locator_rejects_nonlocative_fields_and_filesystem_paths(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MountedVisualPictureLocator.from_payload(
            kind="embedded_asset",
            payload=payload,
        )


@pytest.mark.parametrize("extra_field", ["result_text", "image_path", "extension"])
def test_publication_target_rehydration_rejects_every_extra_root_field(
    tmp_path: Path,
    extra_field: str,
) -> None:
    _request, envelope, _source = _request_and_envelope(tmp_path)
    encoded = json.loads(envelope.canonical_json)
    encoded["target"][extra_field] = "must not enter the target"
    canonical = json.dumps(
        encoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        MountedVisualProjectPublicationEnvelope.from_canonical_json(
            contract_version=envelope.contract_version,
            canonical_json=canonical,
            envelope_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )


def test_ready_scan_rejects_cross_session_publication_corruption(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "cross-session-corruption.sqlite"
    ledger = SqliteMountedVisualCallLedger(ledger_path)
    ledger.dispatch_with_receipt(
        session_id="session_authority",
        adapter=_RecordingExternalAdapter(),
        request=request,
        publication_envelope=envelope,
    )
    with sqlite3.connect(ledger_path) as connection:
        connection.execute(
            "DROP TRIGGER trg_mounted_visual_publication_authority_update"
        )
        connection.execute(
            "UPDATE mounted_visual_project_publications "
            "SET session_id='crossed_session'"
        )

    with pytest.raises(
        MountedVisualCallLedgerError,
        match="crossed provider call authority",
    ):
        ledger.list_ready_publications(session_id="session_authority")


def test_fresh_schema_enforces_composite_publication_authority(
    tmp_path: Path,
) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "composite-authority.sqlite"
    SqliteMountedVisualCallLedger(ledger_path).dispatch_with_receipt(
        session_id="session_composite_authority",
        adapter=_RecordingExternalAdapter(),
        request=request,
        publication_envelope=envelope,
    )

    with sqlite3.connect(ledger_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(mounted_visual_project_publications)"
        ).fetchall()
        authority_columns = {
            (str(row[3]), str(row[4]))
            for row in foreign_keys
            if str(row[2]) == "mounted_visual_provider_calls"
        }
        assert authority_columns == {
            ("call_key", "call_key"),
            ("session_id", "session_id"),
        }
        connection.execute(
            "DROP TRIGGER trg_mounted_visual_publication_authority_update"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE mounted_visual_project_publications "
                "SET session_id='crossed_session'"
            )


def test_outbox_json_is_canonical_and_contains_no_result_text(tmp_path: Path) -> None:
    request, envelope, _source = _request_and_envelope(tmp_path)
    ledger_path = tmp_path / "pointer-only.sqlite"
    SqliteMountedVisualCallLedger(ledger_path).dispatch_with_receipt(
        session_id="session_pointer_only",
        adapter=_RecordingExternalAdapter(),
        request=request,
        publication_envelope=envelope,
    )

    with sqlite3.connect(ledger_path) as connection:
        stored = connection.execute(
            "SELECT envelope_json, envelope_sha256 "
            "FROM mounted_visual_project_publications"
        ).fetchone()
    assert stored is not None
    assert json.dumps(
        json.loads(stored[0]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == stored[0]
    assert hashlib.sha256(stored[0].encode("utf-8")).hexdigest() == stored[1]
    assert "The blue bar" not in stored[0]
