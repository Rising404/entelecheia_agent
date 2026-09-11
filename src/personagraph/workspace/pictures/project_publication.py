"""Atomic Project publication for already-produced picture semantics.

This application boundary deliberately starts after pixel preparation and provider I/O.
It owns only Project authority validation, picture/unit admission, append-only observation
state and the neutral same-transaction publication port.  It has no dependency on vision
providers, tools, sessions or retrieval implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ..files.contracts import WorkspaceDatabasePort
from ..files.storage import repository as file_repository
from .admission import ensure_picture_in_transaction, ensure_picture_unit_in_transaction
from .contracts import PictureSourceKind, PictureSourceLocator, PictureUnitLocator
from .observations import (
    PictureObservationCommitResult,
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationPublicationPort,
    PictureObservationPublicationResult,
    PictureObservationRepository,
    PictureObservationService,
    PictureObservationStructuredPayload,
    normalize_picture_observation_question,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProjectPicturePublicationError(RuntimeError):
    """The semantic result cannot bind the current Project authority."""


class ProjectPictureBindingStale(ProjectPicturePublicationError):
    """The exact file/version/hash/media tuple is no longer current."""


@dataclass(frozen=True, slots=True)
class ProjectPictureBinding:
    project_id: str
    file_id: str
    file_version_id: str
    file_content_sha256: str
    file_media_type: str

    def __post_init__(self) -> None:
        for name in ("project_id", "file_id", "file_version_id"):
            _require_text(getattr(self, name), name)
        _require_sha256(self.file_content_sha256, "file_content_sha256")
        _require_media_type(self.file_media_type, "file_media_type")


@dataclass(frozen=True, slots=True)
class ProjectPictureUnitSpec:
    source_locator: PictureSourceLocator
    source_content_sha256: str
    source_media_type: str
    unit_locator: PictureUnitLocator
    producer_fingerprint: str
    parent_picture_unit_id: str | None
    pixel_sha256: str
    media_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_locator, PictureSourceLocator):
            raise TypeError("source_locator must be PictureSourceLocator")
        if not isinstance(self.unit_locator, PictureUnitLocator):
            raise TypeError("unit_locator must be PictureUnitLocator")
        _require_sha256(self.source_content_sha256, "source_content_sha256")
        _require_media_type(self.source_media_type, "source_media_type")
        _require_text(self.producer_fingerprint, "producer_fingerprint")
        if self.parent_picture_unit_id is not None:
            _require_text(self.parent_picture_unit_id, "parent_picture_unit_id")
        _require_sha256(self.pixel_sha256, "pixel_sha256")
        _require_media_type(self.media_type, "media_type")
        if not self.media_type.startswith("image/"):
            raise ValueError("picture unit media_type must be an image media type")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ProjectPictureObservationSpec:
    request_ordinal: int
    purpose: str
    kind: str
    text: str
    uncertainty: float | None
    processor_fingerprint: str
    prompt_fingerprint: str | None
    structured_payload: PictureObservationStructuredPayload | None
    question: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.request_ordinal, bool)
            or not isinstance(self.request_ordinal, int)
            or self.request_ordinal < 0
        ):
            raise ValueError("request_ordinal must be non-negative")
        for name in ("purpose", "kind", "text", "processor_fingerprint"):
            _require_text(getattr(self, name), name)
        if self.prompt_fingerprint is not None:
            _require_text(self.prompt_fingerprint, "prompt_fingerprint")
        object.__setattr__(
            self,
            "question",
            normalize_picture_observation_question(
                purpose=self.purpose,
                question=self.question,
            ),
        )
        if self.structured_payload is not None and not isinstance(
            self.structured_payload,
            PictureObservationStructuredPayload,
        ):
            raise TypeError("structured_payload has an unsupported type")


@dataclass(frozen=True, slots=True)
class ProjectPicturePublicationCommand:
    binding: ProjectPictureBinding
    unit: ProjectPictureUnitSpec
    logical_invocation_id: str
    observations: tuple[ProjectPictureObservationSpec, ...]
    occurred_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ProjectPictureBinding):
            raise TypeError("binding must be ProjectPictureBinding")
        if not isinstance(self.unit, ProjectPictureUnitSpec):
            raise TypeError("unit must be ProjectPictureUnitSpec")
        _require_text(self.logical_invocation_id, "logical_invocation_id")
        if not isinstance(self.observations, tuple) or len(self.observations) != 1:
            raise ValueError(
                "one logical picture analysis call must append exactly one observation"
            )
        if any(
            not isinstance(item, ProjectPictureObservationSpec)
            for item in self.observations
        ):
            raise TypeError("observations contain an unsupported value")
        ordinals = tuple(item.request_ordinal for item in self.observations)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("observation request ordinals must be unique")
        _require_text(self.occurred_at, "occurred_at")


@dataclass(frozen=True, slots=True)
class ProjectPicturePublicationResult:
    picture_id: str
    picture_unit_id: str
    observation_commits: tuple[PictureObservationCommitResult, ...]
    outbox_publications: tuple[PictureObservationPublicationResult, ...]

    def __post_init__(self) -> None:
        _require_text(self.picture_id, "picture_id")
        _require_text(self.picture_unit_id, "picture_unit_id")
        if len(self.observation_commits) != len(self.outbox_publications):
            raise ValueError("every observation commit requires a publication result")


class ProjectPicturePublicationService:
    """Commit one logical analysis call as one FIFO observation transaction."""

    def __init__(
        self,
        *,
        database: WorkspaceDatabasePort,
        publication_port: PictureObservationPublicationPort,
        observation_service: PictureObservationService | None = None,
    ) -> None:
        if not isinstance(database, WorkspaceDatabasePort):
            raise TypeError("database must implement WorkspaceDatabasePort")
        if not callable(getattr(publication_port, "publish_in_transaction", None)):
            raise TypeError("publication_port must implement publish_in_transaction")
        resolved_observation_service = observation_service or PictureObservationService(
            PictureObservationRepository()
        )
        if not isinstance(resolved_observation_service, PictureObservationService):
            raise TypeError("observation_service must be PictureObservationService")
        self._database = database
        self._publication_port = publication_port
        self._observation_service = resolved_observation_service

    @property
    def project_id(self) -> str:
        return self._database.project_id

    def publish(
        self,
        command: ProjectPicturePublicationCommand,
    ) -> ProjectPicturePublicationResult:
        if not isinstance(command, ProjectPicturePublicationCommand):
            raise TypeError("command must be ProjectPicturePublicationCommand")
        if command.binding.project_id != self._database.project_id:
            raise ProjectPictureBindingStale(
                "picture publication crossed its bound Project"
            )
        _require_file_backed_unit_matches_binding(command)

        with self._database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _require_exact_current_binding(conn, command.binding)
            picture = ensure_picture_in_transaction(
                conn,
                file_id=command.binding.file_id,
                file_version_id=command.binding.file_version_id,
                source_locator=command.unit.source_locator,
                source_content_sha256=command.unit.source_content_sha256,
                source_media_type=command.unit.source_media_type,
                created_at=command.occurred_at,
            ).picture
            unit = ensure_picture_unit_in_transaction(
                conn,
                picture_id=picture.picture_id,
                locator=command.unit.unit_locator,
                producer_fingerprint=command.unit.producer_fingerprint,
                parent_picture_unit_id=command.unit.parent_picture_unit_id,
                pixel_sha256=command.unit.pixel_sha256,
                media_type=command.unit.media_type,
                width=command.unit.width,
                height=command.unit.height,
                created_at=command.occurred_at,
            ).unit

            commits: list[PictureObservationCommitResult] = []
            publications: list[PictureObservationPublicationResult] = []
            for observation in command.observations:
                commit = self._observation_service.commit_in_transaction(
                    conn,
                    PictureObservationDraft(
                        picture_id=picture.picture_id,
                        picture_unit_id=unit.picture_unit_id,
                        logical_invocation_id=command.logical_invocation_id,
                        request_ordinal=observation.request_ordinal,
                        modality=PictureObservationModality.VLM,
                        purpose=observation.purpose,
                        kind=observation.kind,
                        text=observation.text,
                        uncertainty=observation.uncertainty,
                        processor_fingerprint=observation.processor_fingerprint,
                        prompt_fingerprint=observation.prompt_fingerprint,
                        question=observation.question,
                        structured_payload=observation.structured_payload,
                    ),
                    created_at=command.occurred_at,
                )
                publication = self._publication_port.publish_in_transaction(
                    conn,
                    commit,
                    occurred_at=command.occurred_at,
                )
                commits.append(commit)
                publications.append(publication)

            return ProjectPicturePublicationResult(
                picture_id=picture.picture_id,
                picture_unit_id=unit.picture_unit_id,
                observation_commits=tuple(commits),
                outbox_publications=tuple(publications),
            )


def _require_exact_current_binding(
    conn,
    binding: ProjectPictureBinding,
) -> None:
    bound = file_repository.get_file_with_version(
        conn,
        project_id=binding.project_id,
        file_id=binding.file_id,
        file_version_id=binding.file_version_id,
    )
    if bound is None:
        raise ProjectPictureBindingStale("picture file/version binding is unavailable")
    file, version = bound
    if (
        file.current_version_id != binding.file_version_id
        or version.content_sha256 != binding.file_content_sha256
        or file.media_type != binding.file_media_type
    ):
        raise ProjectPictureBindingStale(
            "picture file/version/hash/media binding is no longer current"
        )


def _require_file_backed_unit_matches_binding(
    command: ProjectPicturePublicationCommand,
) -> None:
    if command.unit.source_locator.kind not in {
        PictureSourceKind.WHOLE_FILE,
        PictureSourceKind.DOCUMENT_SURFACE,
    }:
        return
    if (
        command.unit.source_content_sha256
        != command.binding.file_content_sha256
        or command.unit.source_media_type != command.binding.file_media_type
    ):
        raise ProjectPictureBindingStale(
            "file-backed picture source does not match its frozen file binding"
        )


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 65_536:
        raise ValueError(f"{field} must be bounded non-empty text")


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 digest")


def _require_media_type(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip().lower()
        or ";" in value
        or value.count("/") != 1
        or any(not part for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a canonical media type")


__all__ = [
    "ProjectPictureBinding",
    "ProjectPictureBindingStale",
    "ProjectPictureObservationSpec",
    "ProjectPicturePublicationCommand",
    "ProjectPicturePublicationError",
    "ProjectPicturePublicationResult",
    "ProjectPicturePublicationService",
    "ProjectPictureUnitSpec",
]
