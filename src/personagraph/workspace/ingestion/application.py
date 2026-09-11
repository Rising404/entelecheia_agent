"""按文件能力分派显式检查和准备；Host 与工具共用同一服务。"""

from collections.abc import Callable
from dataclasses import dataclass

from personagraph.input_processing.documents import ChunkingProfile
from personagraph.workspace.files.access import AuthorizedFileSource, FileAccess
from .contracts import FilePreparationResult, FilePreparationStatus
from .execution import DocumentIngestExecutionOwner
from .images import check_image_file, prepare_image_file
from .indexing_ports import IngestionGenerationIdentity
from .preparation import check_preparation_replay_source, prepare_file
from .state import resolve_file_preparation_state


@dataclass(frozen=True, slots=True)
class FileIngestionService:
    session_id: str
    access: FileAccess
    generation_identity: IngestionGenerationIdentity
    chunking_profile: ChunkingProfile
    validate_source: Callable[[str, str], bool]
    resolve_owner: Callable[[], DocumentIngestExecutionOwner]

    def check(self, source: AuthorizedFileSource) -> FilePreparationResult:
        if (source.media_type or "").startswith("image/"):
            return check_image_file(source=source, revalidate_source=self.access.revalidate)
        return resolve_file_preparation_state(
            session_id=self.session_id, canonical_path=source.canonical_path,
            frozen_fingerprint=source.fingerprint, generation_identity=self.generation_identity,
            chunking_profile=self.chunking_profile, validate_source_authority=self.validate_source,
        )

    def prepare(
        self, source: AuthorizedFileSource, *, pending_wait_seconds: float = 0.0,
        request_id: str | None = None, checkpoint: Callable[[], None] | None = None,
    ) -> FilePreparationResult:
        if checkpoint is not None:
            checkpoint()
        if (source.media_type or "").startswith("image/"):
            if request_id is not None:
                refusal = check_preparation_replay_source(
                    session_id=self.session_id, request_id=request_id,
                    canonical_path=source.canonical_path, frozen_fingerprint=source.fingerprint,
                )
                if refusal is not None:
                    return refusal
            return prepare_image_file(
                source=source, database=self.access.database, revalidate_source=self.access.revalidate,
            )
        owner = self.resolve_owner()
        if owner.generation_identity != self.generation_identity:
            return FilePreparationResult(
                status=FilePreparationStatus.STALE, reason_code="retrieval_generation_changed",
            )
        return prepare_file(
            session_id=self.session_id, canonical_path=source.canonical_path,
            frozen_fingerprint=source.fingerprint, ingest_owner=owner,
            chunking_profile=self.chunking_profile, validate_source_authority=self.validate_source,
            pending_wait_seconds=pending_wait_seconds,
            request_id=request_id, checkpoint=checkpoint,
        )
