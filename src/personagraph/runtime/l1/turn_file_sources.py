"""Project FileVersion sources accepted for one L1 Turn."""

from __future__ import annotations

from personagraph.session import store
from personagraph.workspace.files.turn_inputs import (
    ResolvedTurnInputFile,
    resolve_turn_input_files,
)


def freeze_turn_attachment_sources(
    *, session_id: str, turn_id: str,
) -> tuple[ResolvedTurnInputFile, ...]:
    """Join exact Turn receipts to current Project file versions."""

    with store.session_database_scope(session_id):
        records = store.list_turn_attachments(session_id, turn_id)
        if not records:
            return ()
        return resolve_turn_input_files(
            records, session_id=session_id, turn_id=turn_id,
        )


def attachment_file_catalog(
    sources: tuple[ResolvedTurnInputFile, ...],
) -> tuple[dict[str, object], ...]:
    """Expose real identities and public metadata, never private storage paths."""

    return tuple(
        {
            "file_id": source.file_id,
            "file_version_id": source.file_version_id,
            "name": source.original_name,
            "relative_path": source.relative_path,
            "media_type": source.media_type,
            "size_bytes": source.size_bytes,
            "kind": source.kind.value,
            "origin": "user_upload",
        }
        for source in sources
    )
