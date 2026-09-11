"""Stable picture-observation domain facade."""

from .contracts import (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY,
    PICTURE_OBSERVATION_OUTPUT_CONTRACT,
    PICTURE_OBSERVATION_PAYLOAD_CONTRACT,
    PICTURE_OBSERVATION_QUESTION_REQUEST_CONTRACT,
    PICTURE_OBSERVATION_REQUEST_CONTRACT,
    PictureObservationActiveWindow,
    PictureObservationAppendResult,
    PictureObservationCommitResult,
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationRecord,
    PictureObservationStructuredPayload,
    PictureObservationWindowPolicy,
    normalize_picture_observation_question,
    picture_observation_output_sha256,
    picture_observation_request_sha256,
)
from .projection import project_picture_observation_window
from .publication import (
    PictureObservationPublicationPort,
    PictureObservationPublicationResult,
)
from .repository import (
    PictureObservationForeignKeysRequired,
    PictureObservationIdempotencyConflict,
    PictureObservationPersistenceConflict,
    PictureObservationRepository,
    PictureObservationRepositoryError,
    PictureObservationTransactionRequired,
    PictureObservationUnitNotFound,
)
from .service import PictureObservationService


__all__ = [
    "DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY",
    "PICTURE_OBSERVATION_OUTPUT_CONTRACT",
    "PICTURE_OBSERVATION_PAYLOAD_CONTRACT",
    "PICTURE_OBSERVATION_QUESTION_REQUEST_CONTRACT",
    "PICTURE_OBSERVATION_REQUEST_CONTRACT",
    "PictureObservationActiveWindow",
    "PictureObservationAppendResult",
    "PictureObservationCommitResult",
    "PictureObservationDraft",
    "PictureObservationForeignKeysRequired",
    "PictureObservationIdempotencyConflict",
    "PictureObservationModality",
    "PictureObservationPersistenceConflict",
    "PictureObservationPublicationPort",
    "PictureObservationPublicationResult",
    "PictureObservationRecord",
    "PictureObservationRepository",
    "PictureObservationRepositoryError",
    "PictureObservationService",
    "PictureObservationStructuredPayload",
    "PictureObservationTransactionRequired",
    "PictureObservationUnitNotFound",
    "PictureObservationWindowPolicy",
    "normalize_picture_observation_question",
    "picture_observation_output_sha256",
    "picture_observation_request_sha256",
    "project_picture_observation_window",
]
