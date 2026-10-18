from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import io
from pathlib import Path
import zlib

import pytest

from personagraph.session.attachments.application import accept_upload
from personagraph.workspace.files import attachments as storage
from personagraph.input_processing.documents import readers
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.documents.admission.turn_inputs import (
    AttachmentDocumentAuthorityError,
    AttachmentDocumentGapReason,
    AttachmentDocumentGap,
    mount_turn_document_attachments as _mount_turn_document_attachments,
    prepare_attachment_document,
)
from personagraph.retrieval.profile import durable_chunking_profile
from personagraph.session import store


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


def test_authorized_attachment_name_does_not_imply_a_private_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload(
        filename="board-secret.pdf"
    )

    result = mount_turn_document_attachments(session_id=session_id, turn_id=turn_id)

    assert [mount.attachment_id for mount in result.mounts] == [attachment_id]
    assert len(docstore.mounted_docs(session_id)) == 1


@pytest.mark.parametrize("private_path", ["API_TOKEN_PATH", "LOCAL_CONFIG_DIR", "STATE_DIR"])
def test_private_host_attachment_is_a_typed_gap_before_document_commit(
    monkeypatch: pytest.MonkeyPatch,
    private_path: str,
) -> None:
    from personagraph.configuration import paths

    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    session_id, turn_id, attachment_id = _accepted_turn_with_upload(
        filename="ordinary.pdf"
    )
    record = store.list_turn_attachments(session_id, turn_id)[0]
    assert _PROJECT_ROOT is not None
    attachment_path = _PROJECT_ROOT / str(record["stored_rel_path"])
    monkeypatch.setattr(
        paths,
        private_path,
        attachment_path if private_path == "API_TOKEN_PATH" else attachment_path.parent,
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
