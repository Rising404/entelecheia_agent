from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import io
import os
from pathlib import Path
import zlib

import pytest

from personagraph.l2.auxiliary_graph import PlanningObservationStatus
from personagraph.session.attachments.application import accept_upload
from personagraph.workspace.files import attachments as storage
from personagraph.input_processing.documents import readers
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage.context import connect_current
from personagraph.workspace.documents.admission.turn_inputs import (
    AttachmentDocumentAuthorityError,
    AttachmentDocumentBridgeResult,
    AttachmentDocumentGapReason,
    AttachmentDocumentGap,
    mount_turn_document_attachments as _mount_turn_document_attachments,
    prepare_attachment_document,
)
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    MountedDocumentPlanningAuthorityError,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    build_auxiliary_execution_capability_catalogs,
    build_auxiliary_planning_capability_catalog,
)
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_CAPABILITY,
)
from personagraph.l2.task_execution.tool_bridge.mounted_document_adapter import (
    build_session_mounted_document_cognition_runtime,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_resource_read_port import (
    MountedDocumentPlanningResourceReadPort,
)
from personagraph.l2.planning.resource_perception import (
    PlanningResourceCoverage,
    PlanningResourceFormat,
    PlanningResourceReadRequest,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store


_PROJECT_ROOT: Path | None = None
_SESSION_SCOPES: ExitStack | None = None


def mount_turn_document_attachments(**kwargs):
    kwargs.setdefault(
        "prepare",
        lambda path: prepare_attachment_document(
            path,
            chunking_profile=durable_chunking_profile(),
        ),
    )
    return _mount_turn_document_attachments(**kwargs)


@pytest.fixture(autouse=True)
def _partitioned_attachment_authority_state(
    tmp_path: Path,
    partitioned_project_state,
):
    global _PROJECT_ROOT, _SESSION_SCOPES
    _PROJECT_ROOT = tmp_path / "project"
    _PROJECT_ROOT.mkdir()
    _SESSION_SCOPES = ExitStack()
    try:
        yield
    finally:
        _SESSION_SCOPES.close()
        _SESSION_SCOPES = None
        _PROJECT_ROOT = None


def _create_session() -> str:
    assert _PROJECT_ROOT is not None and _SESSION_SCOPES is not None
    session_id = store.create_session(
        "Entelecheia",
        working_dir=str(_PROJECT_ROOT),
    )
    _SESSION_SCOPES.enter_context(store.session_database_scope(session_id))
    return session_id


def _pdf(objects: list[bytes]) -> bytes:
    output = io.BytesIO()
    output.write(b"%PDF-1.7\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n".encode())
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return output.getvalue()


def _text_pdf_bytes(text: str = "Cedar final accuracy is 71 percent") -> bytes:
    stream = f"BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n({text}) Tj\nET".encode(
        "latin-1"
    )
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 4 0 R >>",
        ]
    )


def _png_bytes() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image_module.new("RGB", (32, 24), color=(32, 96, 160)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _scanned_pdf_bytes() -> bytes:
    raw = bytes([200, 100, 50] * 16)
    compressed = zlib.compress(raw)
    image = (
        b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        b"/Length "
        + str(len(compressed)).encode()
        + b" >>\nstream\n"
        + compressed
        + b"\nendstream"
    )
    stream = b"q\n200 0 0 200 100 400 cm\n/Im0 Do\nQ"
    content = (
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
            image,
            content,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im0 3 0 R >> >> /Contents 4 0 R >>",
        ]
    )


def _accepted_turn_with_upload(
    *,
    filename: str = "benchmark.pdf",
    payload: bytes | None = None,
) -> tuple[str, str, str]:
    session_id = _create_session()
    attachment = accept_upload(
        session_id=session_id,
        raw_name=filename,
        declared_media_type="application/pdf",
        payload=payload or _text_pdf_bytes(),
        store=store,
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"bridge-{attachment.attachment_id}",
        source="runtime_test",
        user_text="Read the attached benchmark.",
        attachment_ids=(attachment.attachment_id,),
        lease_owner="attachment-document-authority-test",
    )
    return (
        session_id,
        str(accepted["turn"]["turn_id"]),
        attachment.attachment_id,
    )


def _seed_task_shell(
    *,
    source_attachment_payload: bytes | None = None,
) -> tuple[str, str, str | None]:
    session_id = _create_session()
    source_attachment = (
        accept_upload(
            session_id=session_id,
            raw_name="creation-source.pdf",
            declared_media_type="application/pdf",
            payload=source_attachment_payload,
            store=store,
        )
        if source_attachment_payload is not None
        else None
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="bridge-entry-seed",
        source="runtime_test",
        user_text="Create a document-analysis task.",
        attachment_ids=(
            (source_attachment.attachment_id,)
            if source_attachment is not None
            else ()
        ),
        lease_owner="attachment-document-authority-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="bridge-entry-seed-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "analysis",
                        "title": "Document analysis",
                        "objective": "Read the attached evidence and report its exact fact.",
                        "source_excerpt": "document-analysis task",
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=int(window["state_version"]),
    )
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(window["state_version"]),
        processing_level="L2",
        assistant_content="Task created.",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    return (
        session_id,
        applied.created_insession_task_ids_by_local_key["analysis"],
        (
            source_attachment.attachment_id
            if source_attachment is not None
            else None
        ),
    )


def test_bound_pdf_becomes_exact_mounted_planning_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()

    result = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert isinstance(result, AttachmentDocumentBridgeResult)
    assert [item.attachment_id for item in result.mounts] == [attachment_id]
    assert result.mounts[0].processing_status == "complete"
    mounted = docstore.mounted_docs(session_id)
    assert [item["id"] for item in mounted] == [result.mounts[0].document_id]
    assert _PROJECT_ROOT is not None
    assert mounted[0]["path"].startswith(
        str(_PROJECT_ROOT / "附件")
    )
    attachment_record = store.list_turn_attachments(session_id, turn_id)[0]
    with connect_current() as connection:
        committed_identity = connection.execute(
            "SELECT document.file_id, version.file_version_id "
            "FROM documents AS document "
            "JOIN document_versions AS version "
            "ON version.id=document.current_version_id "
            "WHERE document.id=?",
            (result.mounts[0].document_id,),
        ).fetchone()
    assert committed_identity is not None
    assert tuple(committed_identity) == (
        attachment_record["input_file_id"],
        attachment_record["input_file_version_id"],
    )

    source = Path(mounted[0]["path"])
    observed = source.stat()
    os.utime(
        source,
        ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000_000),
    )
    replayed = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
        prepare=lambda path: pytest.fail("unchanged attachment must reuse its document"),
    )
    assert replayed.mounts == result.mounts

    planning = freeze_mounted_document_planning_authority(
        session_id=session_id
    )
    assert len(planning.bindings) == 1
    binding = planning.bindings[0]
    assert binding.resource.resource_format is PlanningResourceFormat.PDF
    assert binding.resource.content_sha256 == result.mounts[0].source_sha256
    assert "benchmark.pdf" not in binding.source_card.excerpt
    assert mounted[0]["path"] not in binding.source_card.excerpt


def test_attachment_replay_rejects_document_version_changed_after_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _ = _accepted_turn_with_upload()
    mount_turn_document_attachments(session_id=session_id, turn_id=turn_id)
    read_snapshot = docstore.get_mounted_current_document_resource_snapshot

    def changed_snapshot(*args, **kwargs):
        snapshot = read_snapshot(*args, **kwargs)
        assert snapshot is not None
        return replace(snapshot, document_version_id="concurrently-replaced-version")

    monkeypatch.setattr(
        docstore,
        "get_mounted_current_document_resource_snapshot",
        changed_snapshot,
    )
    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(session_id=session_id, turn_id=turn_id)

    assert caught.value.gaps[0].detail_code == "document_version_changed"


@pytest.mark.parametrize(
    "record_update",
    (
        {"origin": "agent_created"},
        {"origin": "agent_modified"},
        {"bound_at": None},
    ),
)
def test_only_bound_user_uploads_can_become_document_evidence(
    record_update: dict[str, object],
) -> None:
    session_id, turn_id, _attachment_id = _accepted_turn_with_upload()
    record = {
        **store.list_turn_attachments(session_id, turn_id)[0],
        **record_update,
    }

    class UntrustedAttachmentStore:
        def list_turn_attachments(
            self,
            requested_session_id: str,
            requested_turn_id: str,
        ) -> list[dict[str, object]]:
            assert requested_session_id == session_id
            assert requested_turn_id == turn_id
            return [record]

    with pytest.raises(AttachmentDocumentAuthorityError) as rejected:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
            attachment_store=UntrustedAttachmentStore(),
        )

    assert tuple(item.reason for item in rejected.value.gaps) == (
        AttachmentDocumentGapReason.UNTRUSTED_ORIGIN,
    )
    assert docstore.mounted_docs(session_id) == []


def test_missing_turn_input_file_ref_is_a_typed_fail_closed_gap() -> None:
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    with store._connect() as connection:
        connection.execute(
            "DELETE FROM runtime_turn_input_file_refs WHERE message_id=?",
            (record["input_message_id"],),
        )

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps == (
        AttachmentDocumentGap(
            attachment_id=attachment_id,
            reason=AttachmentDocumentGapReason.MALFORMED_BINDING,
            detail_code="malformed_record",
        ),
    )
    assert docstore.mounted_docs(session_id) == []


def test_visual_only_png_mounts_without_inventing_ocr_text() -> None:
    session_id = _create_session()
    attachment = accept_upload(
        session_id=session_id,
        raw_name="chart.png",
        declared_media_type="image/png",
        payload=_png_bytes(),
        store=store,
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="bridge-visual-only",
        source="runtime_test",
        user_text="Analyze the chart.",
        attachment_ids=(attachment.attachment_id,),
        lease_owner="attachment-document-authority-test",
    )

    result = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=str(accepted["turn"]["turn_id"]),
    )

    assert result.mounts[0].processing_status == "partial"
    # OCR 真跑过并确认空白时会再补一条 page_empty（detail=ocr_confirmed_blank）。
    # 那恰恰是"没有编造文字"的证据，所以它允许出现；断言只管住"必须要求视觉"和
    # "不能冒出别的码"，别把码的条数写死——那会让这条用例随机器上有没有 OCR 引擎
    # 而时好时坏。
    codes = result.mounts[0].processing_diagnostic_codes
    assert "page_needs_vision" in codes
    assert set(codes) <= {"page_needs_vision", "page_empty"}
    snapshot = docstore.get_mounted_current_document_resource_snapshot(
        result.mounts[0].document_id,
        session_id=session_id,
        maximum_chunks=1,
    )
    assert snapshot is not None
    assert snapshot.total_chunk_count == 0
    assert snapshot.chunks == ()
    page_authority = docstore.get_current_document_page_authority(
        result.mounts[0].document_id,
        expected_version_id=result.mounts[0].document_version_id,
        session_id=session_id,
    )
    assert page_authority is not None
    page_gap_codes = {
        diagnostic.code.value
        for page in page_authority.page_manifest.pages
        for diagnostic in page.diagnostics
    }
    assert "page_needs_vision" in page_gap_codes
    assert page_gap_codes & {"page_empty", "ocr_engine_failed"}
    planning = freeze_mounted_document_planning_authority(
        session_id=session_id
    )
    assert planning.bindings[0].resource.resource_format is PlanningResourceFormat.PNG
    assert len(planning.visual_bindings) == 1
    assert planning.visual_bindings[0].parent_document_alias == "mounted_document_01"


def test_scanned_pdf_mounts_as_visual_authority_instead_of_disappearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, _attachment_id = _accepted_turn_with_upload(
        filename="scan.pdf",
        payload=_scanned_pdf_bytes(),
    )

    result = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert result.mounts[0].processing_status == "partial"
    planning = freeze_mounted_document_planning_authority(
        session_id=session_id
    )
    assert planning.bindings[0].resource.resource_format is PlanningResourceFormat.PDF
    assert planning.bindings[0].resource.coverage is PlanningResourceCoverage.PARTIAL
    assert planning.visual_bindings
    assert all(
        binding.resource.resource_format is PlanningResourceFormat.PDF
        for binding in planning.visual_bindings
    )








def test_partial_reader_coverage_and_gap_codes_survive_the_mount_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, _attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    path = _PROJECT_ROOT / str(record["stored_rel_path"])
    from personagraph.input_processing.documents import prepare_document_path

    prepared = prepare_document_path(str(path))
    assert not hasattr(prepared, "reason")
    partial = replace(
        prepared,
        processing_status="partial",
        processing_diagnostics=(
            {
                "code": "page_needs_vision",
                "at": "p1",
                "detail": "synthetic unresolved figure",
            },
        ),
    )

    result = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
        prepare=lambda _path: partial,
    )

    assert result.mounts[0].processing_status == "partial"
    assert result.mounts[0].processing_diagnostic_codes == (
        "page_needs_vision",
    )
    planning = freeze_mounted_document_planning_authority(
        session_id=session_id
    )
    assert (
        planning.bindings[0].resource.coverage
        is PlanningResourceCoverage.PARTIAL
    )


def test_attachment_document_bridge_is_idempotent_for_http_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, _attachment_id = _accepted_turn_with_upload()

    first = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
    )
    def must_not_reparse(_path: str):
        pytest.fail("exact replay must reuse mounted authority without reparsing")

    second = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=turn_id,
        prepare=must_not_reparse,
    )

    assert second == first
    assert len(docstore.mounted_docs(session_id)) == 1
    assert len(docstore.list_document_versions(first.mounts[0].document_id)) == 1


def test_managed_attachment_freeze_isolated_by_exact_task_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id = _create_session()
    mounts = []
    for ordinal in (1, 2):
        attachment = accept_upload(
            session_id=session_id,
            raw_name=f"task-{ordinal}.pdf",
            declared_media_type="application/pdf",
            payload=_text_pdf_bytes(f"Evidence for task {ordinal}"),
            store=store,
        )
        accepted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id=f"task-scope-upload-{ordinal}",
            source="runtime_test",
            user_text=f"Create task {ordinal}.",
            attachment_ids=(attachment.attachment_id,),
            lease_owner="attachment-document-authority-test",
        )
        turn_id = str(accepted["turn"]["turn_id"])
        bridge = mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )
        mounts.append(bridge.mounts[0])
        window = store.get_turn_execution_window(session_id)
        assert window is not None
        interrupted = store.mark_turn_execution_interrupted(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=int(window["state_version"]),
            stage="RESPONSE",
            interruption_reason="TEST_NEXT_ATTACHMENT",
        )
        store.settle_interrupted_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=int(interrupted["state_version"]),
            end_reason="host_stopped",
            error_code="TEST_NEXT_ATTACHMENT",
        )

    task_a = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id="task_a",
        allowed_managed_document_ids=(mounts[0].document_id,),
    )
    task_b = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id="task_b",
        allowed_managed_document_ids=(mounts[1].document_id,),
    )

    assert [item.resource.resource_id for item in task_a.bindings] == [
        mounts[0].document_id
    ]
    assert [item.resource.resource_id for item in task_b.bindings] == [
        mounts[1].document_id
    ]
    assert task_a.scope_snapshot_sha256 != task_b.scope_snapshot_sha256
    with pytest.raises(
        MountedDocumentPlanningAuthorityError,
        match="Task-scoped mounted Document authority is unavailable",
    ):
        freeze_mounted_document_planning_authority(
            session_id=session_id,
            task_id="task_a",
            allowed_managed_document_ids=("unknown_document",),
        )


def test_sensitive_attachment_name_is_a_typed_gap_before_document_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload(
        filename="board-secret.pdf"
    )

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert (
        caught.value.gaps[0].reason
        is AttachmentDocumentGapReason.SENSITIVE_PATH_DENIED
    )
    assert docstore.mounted_docs(session_id) == []


def test_tampered_bound_attachment_never_crosses_its_upload_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    path = _PROJECT_ROOT / str(record["stored_rel_path"])
    path.write_bytes(_text_pdf_bytes("different source after acceptance"))

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert (
        caught.value.gaps[0].reason
        is AttachmentDocumentGapReason.CONTENT_RECEIPT_MISMATCH
    )
    assert docstore.mounted_docs(session_id) == []


def test_internal_symlink_cannot_substitute_the_turn_bound_upload_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    path = _PROJECT_ROOT / str(record["stored_rel_path"])
    substitute = path.with_name("same-bytes-substitute.pdf")
    substitute.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(substitute)

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert (
        caught.value.gaps[0].reason
        is AttachmentDocumentGapReason.PATH_OUTSIDE_SESSION_INPUT
    )
    assert docstore.mounted_docs(session_id) == []


def test_oversized_tamper_is_rejected_before_an_unbounded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    path = _PROJECT_ROOT / str(record["stored_rel_path"])
    with path.open("wb") as handle:
        handle.truncate(storage.MAX_ATTACHMENT_BYTES + 1)

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert caught.value.gaps[0].reason is AttachmentDocumentGapReason.TOO_LARGE
    assert docstore.mounted_docs(session_id) == []


def test_corrupt_targeted_pdf_is_an_explicit_processing_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload(
        payload=b"%PDF-1.7\ncorrupt body without a catalog\n%%EOF\n"
    )

    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert (
        caught.value.gaps[0].reason
        is AttachmentDocumentGapReason.PROCESSING_INCOMPLETE
    )
    assert caught.value.gaps[0].detail_code
    assert docstore.mounted_docs(session_id) == []


def test_multi_attachment_production_commit_rolls_back_as_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id = _create_session()
    first = accept_upload(
        session_id=session_id,
        raw_name="first.pdf",
        declared_media_type="application/pdf",
        payload=_text_pdf_bytes("first exact source"),
        store=store,
    )
    second = accept_upload(
        session_id=session_id,
        raw_name="second.pdf",
        declared_media_type="application/pdf",
        payload=_text_pdf_bytes("second exact source"),
        store=store,
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="bridge-atomic-batch",
        source="runtime_test",
        user_text="Compare both attachments.",
        attachment_ids=(first.attachment_id, second.attachment_id),
        lease_owner="attachment-document-authority-test",
    )
    real_ingest = docstore.ingest
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second commit failure")
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(docstore, "ingest", fail_second)
    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=str(accepted["turn"]["turn_id"]),
        )

    assert caught.value.gaps[0].attachment_id == second.attachment_id
    assert caught.value.gaps[0].reason is AttachmentDocumentGapReason.COMMIT_FAILED
    assert docstore.mounted_docs(session_id) == []
    assert docstore.list_documents() == []


def test_source_change_after_prepare_rolls_back_the_authority_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload()
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    path = _PROJECT_ROOT / str(record["stored_rel_path"])
    from personagraph.workspace.documents.admission import turn_inputs as bridge_module

    real_commit = bridge_module._commit_prepared_document

    def commit_then_replace(*args, **kwargs):
        stored = real_commit(*args, **kwargs)
        path.write_bytes(_text_pdf_bytes("replacement after document preparation"))
        return stored

    monkeypatch.setattr(
        bridge_module,
        "_commit_prepared_document",
        commit_then_replace,
    )
    with pytest.raises(AttachmentDocumentAuthorityError) as caught:
        mount_turn_document_attachments(
            session_id=session_id,
            turn_id=turn_id,
        )

    assert caught.value.gaps[0].attachment_id == attachment_id
    assert caught.value.gaps[0].reason is AttachmentDocumentGapReason.SOURCE_CHANGED
    assert docstore.mounted_docs(session_id) == []
    assert docstore.list_documents() == []


def test_unlinked_turn_never_reaches_file_candidate_preparation() -> None:
    session_id = _create_session()

    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id="unknown_turn",
            task_id="unknown_task",
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert (
        result.reason_code
        == "attachment_authority_turn_not_authorized_for_task"
    )


def test_relevance_only_turn_cannot_drive_or_mount_for_a_task() -> None:
    session_id, task_id, _attachment_id = _seed_task_shell()
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="bridge-relevance-only",
        source="runtime_test",
        user_text="Just discuss why this task exists.",
        lease_owner="attachment-document-authority-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="bridge-relevance-only-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": "discuss why this task exists",
                        "execute_current": False,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=int(window["state_version"]),
    )
    result = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
        ),
    )

    assert result.status is AuxiliaryApplicationStatus.FAILED
    assert (
        result.reason_code
        == "attachment_authority_turn_not_authorized_for_task"
    )


@pytest.mark.parametrize(
    ("filename", "media_type", "payload", "resource_format"),
    (
        (
            "notes.txt",
            "text/plain",
            b"Project Cedar reached 71 percent accuracy.\n",
            PlanningResourceFormat.TXT,
        ),
        (
            "notes.md",
            "text/markdown",
            b"# Result\n\nProject Cedar reached 71 percent accuracy.\n",
            PlanningResourceFormat.MD,
        ),
    ),
)
def test_bound_plain_text_attachment_becomes_exact_mounted_evidence(
    filename: str,
    media_type: str,
    payload: bytes,
    resource_format: PlanningResourceFormat,
) -> None:
    session_id = _create_session()
    attachment = accept_upload(
        session_id=session_id,
        raw_name=filename,
        declared_media_type=media_type,
        payload=payload,
        store=store,
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"bridge-text-{resource_format.value}",
        source="runtime_test",
        user_text="Read it.",
        attachment_ids=(attachment.attachment_id,),
        lease_owner="attachment-document-authority-test",
    )

    result = mount_turn_document_attachments(
        session_id=session_id,
        turn_id=str(accepted["turn"]["turn_id"]),
    )

    assert result.ignored_attachment_ids == ()
    assert len(result.mounts) == 1
    mount = result.mounts[0]
    assert mount.attachment_id == attachment.attachment_id
    assert mount.processing_status == "complete"

    planning = freeze_mounted_document_planning_authority(
        session_id=session_id,
        allowed_managed_document_ids=(mount.document_id,),
    )
    assert planning.visual_bindings == ()
    assert len(planning.bindings) == 1
    resource = planning.bindings[0].resource
    assert resource.resource_format is resource_format
    assert resource.resource_version == mount.document_version_id
    assert resource.content_sha256 == mount.source_sha256

    outcome = MountedDocumentPlanningResourceReadPort(
        (planning.bindings[0].frozen_document,)
    ).read_frozen_resource(PlanningResourceReadRequest(resource=resource))
    assert outcome.status is PlanningObservationStatus.SUCCESS
    assert outcome.observed_resource_version == mount.document_version_id
    assert outcome.observed_content_sha256 == mount.source_sha256
    assert "Project Cedar reached 71 percent accuracy." in "\n".join(
        item.statement for item in outcome.evidence
    )
    assert all(filename not in item.locator for item in outcome.evidence)

    catalog = build_auxiliary_planning_capability_catalog(planning)
    document_capability = next(
        item
        for item in catalog.capabilities
        if item.capability_alias == "mounted_document_read"
    )
    assert document_capability.available is True
    assert document_capability.supported_resource_kinds == (resource_format.value,)

    cognition_capability = next(
        item
        for item in catalog.capabilities
        if item.capability_alias == MOUNTED_DOCUMENT_COGNITION_CAPABILITY
    )
    assert cognition_capability.available is True
    assert "iterative_chunk_reads" in cognition_capability.supported_operations

    cognition_runtime = build_session_mounted_document_cognition_runtime(
        planning
    )
    assert cognition_runtime is not None
    assert set(cognition_runtime.tool_ids) == {
        "inspect_mounted_document",
        "search_mounted_document",
        "read_mounted_document_chunks",
    }
    execution = build_auxiliary_execution_capability_catalogs(
        mounted_document_runtime=cognition_runtime
    )
    assert execution[MOUNTED_DOCUMENT_COGNITION_CAPABILITY] is (
        cognition_runtime.catalog_snapshot
    )
