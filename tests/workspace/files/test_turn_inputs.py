from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pytest

from personagraph.session.attachments.contracts import AttachmentKind
from personagraph.session.attachments.application import accept_upload
from personagraph.workspace.files.turn_inputs import (
    TurnInputFileAuthorityError,
    resolve_turn_input_files,
)
from personagraph.workspace.storage.context import current
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.workspace.files.attachments import MAX_ATTACHMENT_BYTES
from personagraph.session import store


@dataclass(frozen=True, slots=True)
class _AcceptedTurn:
    session_id: str
    turn_id: str
    project_root: Path
    records: tuple[dict[str, Any], ...]


@pytest.fixture
def accepted_turn(tmp_path: Path, partitioned_project_state: Path):
    del partitioned_project_state
    project_root = tmp_path / "project"
    project_root.mkdir()
    session_id = store.create_session(
        "Entelecheia",
        title="turn file authority",
        working_dir=str(project_root),
    )
    with store.session_database_scope(session_id):
        uploads = tuple(
            accept_upload(
                session_id=session_id,
                raw_name=name,
                declared_media_type="text/plain",
                payload=payload,
                store=store,
            )
            for name, payload in (
                ("first.txt", b"first accepted upload"),
                ("second.txt", b"second accepted upload"),
            )
        )
        accepted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id="turn-file-authority-fixture",
            source="runtime_test",
            user_text="Read both uploads.",
            attachment_ids=tuple(item.attachment_id for item in uploads),
            lease_owner="turn-file-authority-test",
        )
        turn_id = str(accepted["turn"]["turn_id"])
        yield _AcceptedTurn(
            session_id=session_id,
            turn_id=turn_id,
            project_root=project_root,
            records=tuple(store.list_turn_attachments(session_id, turn_id)),
        )


def _resolve(
    turn: _AcceptedTurn,
    records: tuple[dict[str, Any], ...] | None = None,
):
    return resolve_turn_input_files(
        turn.records if records is None else records,
        session_id=turn.session_id,
        turn_id=turn.turn_id,
    )


def _changed_record(
    turn: _AcceptedTurn,
    index: int,
    **changes: object,
) -> tuple[dict[str, Any], ...]:
    records = [dict(item) for item in turn.records]
    records[index].update(changes)
    return tuple(records)


def test_resolves_complete_turn_in_persisted_ordinal_order(
    accepted_turn: _AcceptedTurn,
) -> None:
    resolved = _resolve(accepted_turn)

    assert tuple(item.ordinal for item in resolved) == (0, 1)
    assert tuple(item.attachment_id for item in resolved) == tuple(
        str(item["attachment_id"]) for item in accepted_turn.records
    )
    assert len({item.input_message_id for item in resolved}) == 1
    for item, record in zip(resolved, accepted_turn.records, strict=True):
        assert item.project_id == record["project_id"] == record["input_project_id"]
        assert item.file_id == record["file_id"] == record["input_file_id"]
        assert (
            item.file_version_id
            == record["file_version_id"]
            == record["input_file_version_id"]
        )
        assert item.kind is AttachmentKind.TEXT
        assert item.relative_path == item.project_file.relative_path
        assert item.authority_path == accepted_turn.project_root / item.relative_path
        assert item.canonical_path == item.authority_path.resolve(strict=True)
        assert item.content_sha256 == record["content_hash"]
        assert item.size_bytes == record["size_bytes"]


def test_resolves_unchanged_attachment_with_updated_file_observation(
    accepted_turn: _AcceptedTurn,
) -> None:
    database = current()
    assert database is not None
    original = _resolve(accepted_turn)[0]
    modified_ns = original.source_mtime_ns + 1_000_000_000
    os.utime(original.canonical_path, ns=(modified_ns, modified_ns))
    registration = WorkspaceFileAuthority(database).ensure_current_path(
        original.relative_path,
        source=FileSource.USER_UPLOAD,
    )

    resolved = _resolve(accepted_turn)[0]

    assert resolved.file_version_id == original.file_version_id
    assert registration.version == original.project_file_version
    assert resolved.project_file_version == original.project_file_version
    assert resolved.source_mtime_ns == modified_ns
    assert resolved.content_sha256 == original.content_sha256


def test_resolves_attachment_after_mtime_only_change_without_version_advance(
    accepted_turn: _AcceptedTurn,
) -> None:
    original = _resolve(accepted_turn)[0]
    modified_ns = original.source_mtime_ns + 1_000_000_000
    os.utime(original.canonical_path, ns=(modified_ns, modified_ns))

    resolved = _resolve(accepted_turn)[0]

    assert resolved.file_version_id == original.file_version_id
    assert resolved.content_sha256 == original.content_sha256
    assert resolved.size_bytes == original.size_bytes
    assert resolved.source_mtime_ns == original.source_mtime_ns


def test_rejects_changed_attachment_bytes_without_version_advance(
    accepted_turn: _AcceptedTurn,
) -> None:
    original = _resolve(accepted_turn)[0]
    replacement = b"FIRST accepted upload"
    assert len(replacement) == original.size_bytes
    original.canonical_path.write_bytes(replacement)

    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn)

    assert raised.value.reason == "file_content_changed"
    assert str(accepted_turn.project_root) not in str(raised.value)


def test_rejects_oversized_attachment_before_reading_its_body(
    accepted_turn: _AcceptedTurn,
) -> None:
    original = _resolve(accepted_turn)[0]
    with original.canonical_path.open("wb") as stream:
        stream.truncate(MAX_ATTACHMENT_BYTES + 1)

    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn)

    assert raised.value.reason == "file_too_large"
    assert str(accepted_turn.project_root) not in str(raised.value)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"input_file_id": None}, "malformed_record"),
        ({"input_file_ordinal": 1}, "binding_order_mismatch"),
        ({"input_file_version_id": "filever_unrelated"}, "input_file_ref_mismatch"),
        ({"input_project_id": "project_other"}, "input_file_ref_mismatch"),
        ({"stored_rel_path": "elsewhere.txt"}, "registered_path_mismatch"),
        ({"size_bytes": 999}, "session_receipt_mismatch"),
        ({"content_hash": "0" * 64}, "session_receipt_mismatch"),
    ],
)
def test_rejects_incomplete_or_conflicting_session_read_model(
    accepted_turn: _AcceptedTurn,
    changes: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn, _changed_record(accepted_turn, 0, **changes))

    assert raised.value.reason == reason
    assert str(accepted_turn.project_root) not in str(raised.value)


def test_rejects_reordered_or_partial_turn_bindings(
    accepted_turn: _AcceptedTurn,
) -> None:
    with pytest.raises(TurnInputFileAuthorityError) as reordered:
        _resolve(accepted_turn, tuple(reversed(accepted_turn.records)))
    assert reordered.value.reason == "binding_order_mismatch"

    with pytest.raises(TurnInputFileAuthorityError) as partial:
        _resolve(accepted_turn, accepted_turn.records[1:])
    assert partial.value.reason == "binding_order_mismatch"


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("files", "origin"),
        ("file_versions", "producer"),
    ],
)
def test_rejects_non_upload_project_provenance(
    accepted_turn: _AcceptedTurn,
    table: str,
    column: str,
) -> None:
    database = current()
    assert database is not None
    file_id = str(accepted_turn.records[0]["file_id"])
    file_version_id = str(accepted_turn.records[0]["file_version_id"])
    with database.connect() as connection:
        if table == "files":
            connection.execute(
                "UPDATE files SET origin=? WHERE id=?",
                (FileSource.WORKSPACE_EXISTING.value, file_id),
            )
        else:
            connection.execute(
                "UPDATE file_versions SET producer=? WHERE id=?",
                (FileSource.WORKSPACE_EXISTING.value, file_version_id),
            )

    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn)
    assert raised.value.reason == "project_file_source_mismatch"
    assert str(accepted_turn.project_root) not in repr(raised.value)


def test_rejects_a_bound_version_after_project_file_advances(
    accepted_turn: _AcceptedTurn,
) -> None:
    database = current()
    assert database is not None
    first = accepted_turn.records[0]
    target = accepted_turn.project_root / str(first["stored_rel_path"])
    target.write_bytes(b"a later version of the same project upload")
    registered = WorkspaceFileAuthority(database).register_path(
        str(first["stored_rel_path"]),
        source=FileSource.USER_UPLOAD,
        file_id=str(first["file_id"]),
        media_type=str(first["media_type"]),
    )
    assert registered.version.file_version_id != first["file_version_id"]

    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn)
    assert raised.value.reason == "bound_version_not_current"


def test_rejects_a_symlink_at_the_registered_project_path(
    accepted_turn: _AcceptedTurn,
) -> None:
    first = accepted_turn.records[0]
    target = accepted_turn.project_root / str(first["stored_rel_path"])
    replacement = accepted_turn.project_root / "replacement.txt"
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement)

    with pytest.raises(TurnInputFileAuthorityError) as raised:
        _resolve(accepted_turn)
    assert raised.value.reason == "unsafe_project_path"
    assert str(accepted_turn.project_root) not in str(raised.value)
