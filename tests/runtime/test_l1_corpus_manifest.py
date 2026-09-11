from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from personagraph.runtime.entry.context.attachments import (
    AttachmentAccess,
    AttachmentKind,
    AttachmentProjectionItem,
    AttachmentProjection,
)
from personagraph.runtime.l1 import corpus_manifest
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.corpus_contracts import (
    L1CorpusManifestError,
    derive_l1_turn_run_id,
    load_l1_corpus_manifest,
)
from personagraph.runtime.l1.tool_runtime import L1WorkspaceCorpusAuthority


def _attachment_source(
    *,
    file_version_id: str = "file-version-one",
) -> SimpleNamespace:
    return SimpleNamespace(
        attachment_id="attachment-on-demand",
        ordinal=0,
        input_message_id="input-message-one",
        project_id="project-one",
        file_id="file-one",
        file_version_id=file_version_id,
        relative_path="input/private/attachment-on-demand/evidence.pdf",
        original_name="evidence.pdf",
        media_type="application/pdf",
        size_bytes=4096,
        kind=AttachmentKind.DOCUMENT,
        content_sha256="c" * 64,
    )


def _tool_runtime(
    *,
    file_version_id: str = "file-version-one",
    include_attachment: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        catalog_snapshot_sha256='4' * 64,
        workspace_corpus_authority=L1WorkspaceCorpusAuthority(
            boundary_fingerprint='1' * 64,
            scope_snapshot_sha256='2' * 64,
            local_file_authorization_receipt_id='receipt',
        ),
        turn_attachment_sources=(
            (_attachment_source(file_version_id=file_version_id),)
            if include_attachment
            else ()
        ),
    )


def _attachment_projection() -> AttachmentProjection:
    return AttachmentProjection(
        items=(
            AttachmentProjectionItem(
                attachment_id="attachment-on-demand",
                name="evidence.pdf",
                media_type="application/pdf",
                size_bytes=4096,
                kind=AttachmentKind.DOCUMENT,
                access=AttachmentAccess.ON_DEMAND,
                stored_rel_path="input/private/attachment-on-demand/evidence.pdf",
                content_hash="c" * 64,
            ),
        )
    )


def test_manifest_is_deterministic_metadata_only_and_private_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-corpus"
    turn_id = "turn-corpus"
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)

    first = corpus_manifest.freeze_l1_corpus_manifest(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=run_id,
        tool_runtime=_tool_runtime(),
        attachments=_attachment_projection(),
    )
    second = corpus_manifest.freeze_l1_corpus_manifest(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=run_id,
        tool_runtime=_tool_runtime(),
        attachments=_attachment_projection(),
    )

    assert first == second
    assert "input/private" not in first.manifest_json
    assert "/Users/" not in first.manifest_json
    assert first.manifest.workspace is not None
    assert not hasattr(first.manifest.workspace, "candidates")
    attachment = first.manifest.attachments[0]
    assert attachment.access == "on_demand"
    assert attachment.project_id == "project-one"
    assert attachment.file_id == "file-one"
    assert attachment.file_version_id == "file-version-one"
    assert attachment.content_sha256 == "c" * 64
    assert not hasattr(attachment, "projected_text_sha256")
    assert not hasattr(attachment, "mounted_document_alias")

    assert load_l1_corpus_manifest(
        first.manifest_json,
        first.manifest_sha256,
        expected_session_id=session_id,
        expected_turn_id=turn_id,
        expected_l1_turn_run_id=run_id,
    ) == first


def test_manifest_identity_changes_when_attachment_file_version_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-attachment-identity"
    turn_id = "turn-attachment-identity"
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    first = corpus_manifest.freeze_l1_corpus_manifest(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=run_id,
        tool_runtime=_tool_runtime(),
        attachments=_attachment_projection(),
    )
    second = corpus_manifest.freeze_l1_corpus_manifest(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=run_id,
        tool_runtime=_tool_runtime(file_version_id="file-version-two"),
        attachments=_attachment_projection(),
    )

    assert first.manifest_sha256 != second.manifest_sha256
    assert (
        first.manifest.attachments[0].projection_identity_sha256
        != second.manifest.attachments[0].projection_identity_sha256
    )


def test_manifest_rejects_attachment_projection_that_differs_from_file_source() -> None:
    session_id = "session-attachment-mismatch"
    turn_id = "turn-attachment-mismatch"
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    changed_item = _attachment_projection().items[0].model_copy(
        update={"content_hash": "d" * 64}
    )

    with pytest.raises(
        L1CorpusManifestError,
        match="projection and FileVersion authority differ",
    ):
        corpus_manifest.freeze_l1_corpus_manifest(
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=run_id,
            tool_runtime=_tool_runtime(),
            attachments=AttachmentProjection(items=(changed_item,)),
        )


def test_manifest_authentication_rejects_tamper_and_cross_turn_replay() -> None:
    session_id = "session-corpus"
    turn_id = "turn-corpus"
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    frozen = corpus_manifest.freeze_l1_corpus_manifest(
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=run_id,
        tool_runtime=_tool_runtime(include_attachment=False),
        attachments=AttachmentProjection(),
    )
    tampered = json.loads(frozen.manifest_json)
    tampered["attachment_count"] = 1

    with pytest.raises(L1CorpusManifestError, match="hash changed"):
        load_l1_corpus_manifest(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")),
            frozen.manifest_sha256,
            expected_session_id=session_id,
            expected_turn_id=turn_id,
            expected_l1_turn_run_id=run_id,
        )
    with pytest.raises(L1CorpusManifestError, match="ownership changed"):
        load_l1_corpus_manifest(
            frozen.manifest_json,
            frozen.manifest_sha256,
            expected_session_id=session_id,
            expected_turn_id="another-turn",
            expected_l1_turn_run_id=run_id,
        )

    malformed = json.loads(frozen.manifest_json)
    malformed["unexpected"] = True
    with pytest.raises(L1CorpusManifestError, match="invalid"):
        load_l1_corpus_manifest(
            canonical_json(malformed),
            sha256_json(malformed),
            expected_session_id=session_id,
            expected_turn_id=turn_id,
            expected_l1_turn_run_id=run_id,
        )
