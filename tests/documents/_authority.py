"""Project-scoped document authority helpers shared by migration tests."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from personagraph.workspace.documents.admission import ingest_document_path
from personagraph.input_processing.documents.chunking import ChunkingProfile
from personagraph.retrieval.profile import (
    durable_chunking_profile,
)
from personagraph.input_processing.files import fingerprint_file
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
    RegisteredProjectFileVersion,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.storage import DocumentDatabase
from personagraph.workspace.storage.context import bind, current
from personagraph.retrieval.sources.events import SqliteDocumentIndexPort
from personagraph.session import store as session_store


@dataclass(frozen=True, slots=True)
class ProjectDocumentAuthority:
    """Own a real project root plus its single ``documents.sqlite`` authority."""

    database: DocumentDatabase
    document_index_port: SqliteDocumentIndexPort = field(
        default_factory=SqliteDocumentIndexPort
    )

    @property
    def root(self) -> Path:
        return self.database.project_root

    def write(
        self,
        relative_path: str,
        payload: str | bytes,
        *,
        source: FileSource = FileSource.WORKSPACE_EXISTING,
        media_type: str | None = None,
    ) -> tuple[Path, RegisteredProjectFileVersion]:
        """Materialize and register the exact immutable file version under test."""

        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        if not path.exists() or path.read_bytes() != encoded:
            path.write_bytes(encoded)
        registration = WorkspaceFileAuthority(self.database).ensure_current_path(
            relative_path,
            source=source,
            media_type=media_type,
        )
        return path.resolve(), registration

    def ingest(
        self,
        relative_path: str,
        title: str,
        mime: str,
        elements: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        """Register a source and pass its exact file/version identity to docstore."""

        from personagraph.workspace.documents import application as documents

        payload = kwargs.pop(
            "source_payload",
            "\n".join(str(item.get("content") or "") for item in elements),
        )
        if not isinstance(payload, (str, bytes)):
            raise TypeError("source_payload must be str or bytes")
        path, registration = self.write(
            relative_path,
            payload,
            media_type=mime,
        )
        session_id = kwargs.get("session_id")
        session_scope = (
            self.session(str(session_id)) if session_id else nullcontext()
        )
        with session_scope:
            return documents.ingest(
                str(path),
                title,
                mime,
                elements,
                file_id=registration.file.file_id,
                file_version_id=registration.version.file_version_id,
                source_fingerprint=fingerprint_file(path),
                document_index_port=self.document_index_port,
                **kwargs,
            )

    @contextmanager
    def session(self, session_id: str) -> Iterator[None]:
        """Enter the session database router used by mount authority operations."""

        with session_store.session_database_scope(session_id):
            yield


def ingest_registered_document_fixture(
    path: str | Path,
    *,
    session_id: str | None = None,
    chunking_profile: ChunkingProfile | None = None,
) -> dict[str, Any]:
    """Register source identity, then exercise the lower-level commit service.

    Production intake is owned by the durable ingest-job state machine. Parser,
    provenance and document-store tests use this narrower fixture so they do not
    recreate its lease and retrieval-publication orchestration.
    """

    database = current()
    if database is None:
        raise RuntimeError("document fixture requires a bound project database")
    canonical_path = Path(path).expanduser().resolve(strict=False)
    try:
        relative_path = canonical_path.relative_to(database.project_root)
    except ValueError:
        return {"ok": False, "reason": "outside_project_root"}
    files = WorkspaceFileAuthority(database)
    existing = files.get_file_by_relative_path(relative_path)
    registration = files.ensure_current_path(
        relative_path,
        source=(
            existing.source
            if existing is not None
            else FileSource.WORKSPACE_EXISTING
        ),
    )
    return ingest_document_path(
        str(canonical_path),
        session_id=session_id,
        with_summary=False,
        document_store=docstore,
        chunking_profile=chunking_profile or durable_chunking_profile(),
        file_id=registration.file.file_id,
        file_version_id=registration.version.file_version_id,
    )


@contextmanager
def bound_project_document_authority(
    tmp_path: Path,
    *,
    project_root: Path | None = None,
) -> Iterator[ProjectDocumentAuthority]:
    root = project_root or (tmp_path / "project")
    root.mkdir(parents=True, exist_ok=True)
    database = DocumentDatabase(
        "test-project",
        root,
        tmp_path / "var" / "projects" / "test-project" / "documents.sqlite",
    )
    authority = ProjectDocumentAuthority(database)
    with bind(database), session_store.document_mount_port_scope():
        yield authority
