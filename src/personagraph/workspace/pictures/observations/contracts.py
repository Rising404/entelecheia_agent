"""Immutable contracts for OCR and visual-model picture observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import TypeAlias


PICTURE_OBSERVATION_REQUEST_CONTRACT = "picture-observation-request-v1"
PICTURE_OBSERVATION_QUESTION_REQUEST_CONTRACT = "picture-observation-request-v2"
PICTURE_OBSERVATION_OUTPUT_CONTRACT = "picture-observation-output-v1"
PICTURE_OBSERVATION_PAYLOAD_CONTRACT = "picture-observation-payload-v1"

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STRUCTURED_CONTRACT_PATTERN = re.compile(
    r"[a-z][a-z0-9._-]{0,126}-v[1-9][0-9]*"
)
_MAX_IDENTIFIER_LENGTH = 512
_MAX_LABEL_LENGTH = 1024
_MAX_OBSERVATION_TEXT_LENGTH = 65_536
_MAX_QUESTION_LENGTH = 4_000
_MAX_STRUCTURED_PAYLOAD_BYTES = 65_536


class PictureObservationModality(StrEnum):
    """Closed producer class used by the picture observation ledger."""

    OCR = "ocr"
    VLM = "vlm"


@dataclass(frozen=True, slots=True)
class PictureObservationStructuredPayload:
    """A bounded canonical envelope for modality-owned structured evidence.

    The picture domain owns only the envelope and canonical JSON invariants. OCR/VLM
    adapters own the versioned payload schema named by ``contract``.
    """

    contract: str
    canonical_json: str

    def __post_init__(self) -> None:
        _structured_contract(self.contract)
        parsed = _load_structured_envelope(self.canonical_json)
        if parsed["contract"] != self.contract:
            raise ValueError("structured payload contract does not match its envelope")
        normalized = _canonical_structured_payload(
            contract=self.contract,
            payload=parsed["payload"],
        )
        if normalized != self.canonical_json:
            raise ValueError("structured payload JSON is not canonical")

    @classmethod
    def from_payload(
        cls,
        *,
        contract: str,
        payload: Mapping[str, object],
    ) -> PictureObservationStructuredPayload:
        return cls(
            contract=contract,
            canonical_json=_canonical_structured_payload(
                contract=contract,
                payload=payload,
            ),
        )

    @classmethod
    def from_canonical_json(
        cls,
        canonical_json: str,
    ) -> PictureObservationStructuredPayload:
        parsed = _load_structured_envelope(canonical_json)
        return cls(
            contract=str(parsed["contract"]),
            canonical_json=canonical_json,
        )

    @property
    def payload(self) -> dict[str, JsonValue]:
        return dict(_load_structured_envelope(self.canonical_json)["payload"])


@dataclass(frozen=True, slots=True)
class PictureObservationDraft:
    """A validated single-unit observation before global sequence allocation."""

    picture_id: str
    picture_unit_id: str
    logical_invocation_id: str
    request_ordinal: int
    modality: PictureObservationModality
    purpose: str
    kind: str
    text: str
    uncertainty: float | None
    processor_fingerprint: str
    prompt_fingerprint: str | None
    question: str | None = None
    structured_payload: PictureObservationStructuredPayload | None = None

    def __post_init__(self) -> None:
        _identifier(self.picture_id, "picture_id")
        _identifier(self.picture_unit_id, "picture_unit_id")
        _identifier(self.logical_invocation_id, "logical_invocation_id")
        if (
            isinstance(self.request_ordinal, bool)
            or not isinstance(self.request_ordinal, int)
            or self.request_ordinal < 0
        ):
            raise ValueError("request_ordinal must be a non-negative integer")
        if not isinstance(self.modality, PictureObservationModality):
            raise ValueError("modality must be a PictureObservationModality")
        _label(self.purpose, "purpose")
        _label(self.kind, "kind")
        _bounded_text(self.text, "text")
        _optional_uncertainty(self.uncertainty)
        if self.uncertainty is not None:
            normalized_uncertainty = float(self.uncertainty)
            object.__setattr__(
                self,
                "uncertainty",
                0.0 if normalized_uncertainty == 0.0 else normalized_uncertainty,
            )
        _label(self.processor_fingerprint, "processor_fingerprint")
        if self.prompt_fingerprint is not None:
            _label(self.prompt_fingerprint, "prompt_fingerprint")
        normalized_question = normalize_picture_observation_question(
            purpose=self.purpose,
            question=self.question,
        )
        if (
            normalized_question is not None
            and self.modality is not PictureObservationModality.VLM
        ):
            raise ValueError("question observations must use the VLM modality")
        object.__setattr__(self, "question", normalized_question)
        if self.structured_payload is not None and not isinstance(
            self.structured_payload,
            PictureObservationStructuredPayload,
        ):
            raise TypeError(
                "structured_payload must be PictureObservationStructuredPayload or None"
            )

    @property
    def request_sha256(self) -> str:
        return picture_observation_request_sha256(
            picture_id=self.picture_id,
            picture_unit_id=self.picture_unit_id,
            modality=self.modality,
            purpose=self.purpose,
            kind=self.kind,
            processor_fingerprint=self.processor_fingerprint,
            prompt_fingerprint=self.prompt_fingerprint,
            question=self.question,
        )

    @property
    def output_sha256(self) -> str:
        return picture_observation_output_sha256(
            text=self.text,
            uncertainty=self.uncertainty,
            structured_payload=self.structured_payload,
        )

    @property
    def canonical_payload_json(self) -> str:
        return json.dumps(
            {
                "contract": PICTURE_OBSERVATION_PAYLOAD_CONTRACT,
                "output_sha256": self.output_sha256,
                "request_sha256": self.request_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PictureObservationRecord:
    """One append-only result in a picture-level FIFO stream."""

    observation_id: str
    sequence: int
    draft: PictureObservationDraft
    request_sha256: str
    output_sha256: str
    payload_sha256: str
    created_at: str

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.draft, PictureObservationDraft):
            raise TypeError("draft must be PictureObservationDraft")
        for name in ("request_sha256", "output_sha256", "payload_sha256"):
            _sha256(getattr(self, name), name)
        if self.request_sha256 != self.draft.request_sha256:
            raise ValueError("request_sha256 does not match the observation draft")
        if self.output_sha256 != self.draft.output_sha256:
            raise ValueError("output_sha256 does not match the observation draft")
        if self.payload_sha256 != self.draft.payload_sha256:
            raise ValueError("payload_sha256 does not match the observation draft")
        _timestamp(self.created_at, "created_at")

    @property
    def picture_id(self) -> str:
        return self.draft.picture_id

    @property
    def picture_unit_id(self) -> str:
        return self.draft.picture_unit_id


@dataclass(frozen=True, slots=True)
class PictureObservationActiveWindow:
    """A derived immutable FIFO view; it is never persisted as another authority."""

    picture_id: str
    observations: tuple[PictureObservationRecord, ...]
    content: str
    content_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.picture_id, "picture_id")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("observations must be a non-empty tuple")
        if any(
            observation.picture_id != self.picture_id
            for observation in self.observations
        ):
            raise ValueError("all observations must belong to picture_id")
        sequences = tuple(observation.sequence for observation in self.observations)
        if (
            sequences != tuple(sorted(sequences))
            or len(set(sequences)) != len(sequences)
        ):
            raise ValueError("observations must be uniquely ordered by sequence")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("content must be a non-empty string")
        _sha256(self.content_sha256, "content_sha256")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise ValueError("content_sha256 does not match content")

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(observation.observation_id for observation in self.observations)

    @property
    def first_sequence(self) -> int:
        return self.observations[0].sequence

    @property
    def last_sequence(self) -> int:
        return self.observations[-1].sequence


@dataclass(frozen=True, slots=True)
class PictureObservationWindowPolicy:
    """Maximum number of active observation entries retained per picture."""

    max_active_entries: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_active_entries, bool)
            or not isinstance(self.max_active_entries, int)
            or not 1 <= self.max_active_entries <= 256
        ):
            raise ValueError("max_active_entries must be between 1 and 256")


DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY = PictureObservationWindowPolicy(
    max_active_entries=8
)


@dataclass(frozen=True, slots=True)
class PictureObservationAppendResult:
    observation: PictureObservationRecord
    inserted: bool


@dataclass(frozen=True, slots=True)
class PictureObservationCommitResult:
    observation: PictureObservationRecord
    active_window: PictureObservationActiveWindow
    window_policy: PictureObservationWindowPolicy
    inserted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation, PictureObservationRecord):
            raise TypeError("observation must be PictureObservationRecord")
        if not isinstance(self.active_window, PictureObservationActiveWindow):
            raise TypeError("active_window must be PictureObservationActiveWindow")
        if not isinstance(self.window_policy, PictureObservationWindowPolicy):
            raise TypeError("window_policy must be PictureObservationWindowPolicy")
        if not isinstance(self.inserted, bool):
            raise TypeError("inserted must be a bool")
        if self.observation.picture_id != self.active_window.picture_id:
            raise ValueError("observation and active_window must belong to one picture")
        if self.inserted and (
            self.observation.observation_id not in self.active_window.observation_ids
        ):
            raise ValueError("committed observation must belong to active_window")
        if len(self.active_window.observations) > self.window_policy.max_active_entries:
            raise ValueError("active_window exceeds window_policy")


def picture_observation_request_sha256(
    *,
    picture_id: str,
    picture_unit_id: str,
    modality: PictureObservationModality,
    purpose: str,
    kind: str,
    processor_fingerprint: str,
    prompt_fingerprint: str | None,
    question: str | None = None,
) -> str:
    """Digest immutable request semantics, excluding invocation key and result."""

    _identifier(picture_id, "picture_id")
    _identifier(picture_unit_id, "picture_unit_id")
    if not isinstance(modality, PictureObservationModality):
        raise ValueError("modality must be a PictureObservationModality")
    _label(purpose, "purpose")
    _label(kind, "kind")
    _label(processor_fingerprint, "processor_fingerprint")
    if prompt_fingerprint is not None:
        _label(prompt_fingerprint, "prompt_fingerprint")
    normalized_question = normalize_picture_observation_question(
        purpose=purpose,
        question=question,
    )
    if normalized_question is None:
        contract = PICTURE_OBSERVATION_REQUEST_CONTRACT
        question_fields: dict[str, str] = {}
    else:
        contract = PICTURE_OBSERVATION_QUESTION_REQUEST_CONTRACT
        question_fields = {"question": normalized_question}
    encoded = json.dumps(
        {
            "contract": contract,
            "kind": kind,
            "modality": modality.value,
            "picture_id": picture_id,
            "picture_unit_id": picture_unit_id,
            "processor_fingerprint": processor_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
            "purpose": purpose,
            **question_fields,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_picture_observation_question(
    *,
    purpose: str,
    question: object,
) -> str | None:
    """Normalize the optional free-form question stored with one VLM answer."""

    _label(purpose, "purpose")
    if purpose != "question":
        if question is not None:
            raise ValueError("question is only valid when purpose is question")
        return None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required when purpose is question")
    normalized = question.strip()
    if len(normalized) > _MAX_QUESTION_LENGTH or "\x00" in normalized:
        raise ValueError("question must be NUL-free and at most 4000 characters")
    return normalized


def picture_observation_output_sha256(
    *,
    text: str,
    uncertainty: float | None,
    structured_payload: PictureObservationStructuredPayload | None,
) -> str:
    """Digest the complete normalized result while keeping readable text separate."""

    _bounded_text(text, "text")
    _optional_uncertainty(uncertainty)
    if structured_payload is not None and not isinstance(
        structured_payload,
        PictureObservationStructuredPayload,
    ):
        raise TypeError(
            "structured_payload must be PictureObservationStructuredPayload or None"
        )
    structured_value = (
        None
        if structured_payload is None
        else _load_structured_envelope(structured_payload.canonical_json)
    )
    normalized_uncertainty = (
        None
        if uncertainty is None
        else (0.0 if float(uncertainty) == 0.0 else float(uncertainty))
    )
    encoded = json.dumps(
        {
            "contract": PICTURE_OBSERVATION_OUTPUT_CONTRACT,
            "structured_payload": structured_value,
            "text": text,
            "uncertainty": normalized_uncertainty,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_timestamp(value: str, *, name: str = "timestamp") -> str:
    """Validate and normalize an aware ISO-8601 timestamp without reading a clock."""

    _timestamp(value, name)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.isoformat()


def _canonical_structured_payload(
    *,
    contract: str,
    payload: Mapping[str, object],
) -> str:
    _structured_contract(contract)
    if not isinstance(payload, Mapping):
        raise ValueError("structured payload must be an object")
    normalized = _normalize_json_value(payload)
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        {"contract": contract, "payload": normalized},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        byte_count = len(encoded.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("structured payload must contain valid UTF-8 text") from exc
    if byte_count > _MAX_STRUCTURED_PAYLOAD_BYTES:
        raise ValueError("structured payload exceeds 65536 UTF-8 bytes")
    return encoded


def _load_structured_envelope(encoded: str) -> dict[str, JsonValue]:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("structured payload must be non-empty canonical JSON")
    try:
        byte_count = len(encoded.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("structured payload must contain valid UTF-8 text") from exc
    if byte_count > _MAX_STRUCTURED_PAYLOAD_BYTES:
        raise ValueError("structured payload exceeds 65536 UTF-8 bytes")
    try:
        parsed = json.loads(encoded, parse_constant=_reject_nonfinite_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("structured payload is not valid JSON") from exc
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"contract", "payload"}
        or not isinstance(parsed["contract"], str)
        or not isinstance(parsed["payload"], dict)
    ):
        raise ValueError(
            "structured payload must contain only contract and object payload"
        )
    return parsed


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("structured payload must contain valid UTF-8 text") from exc
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("structured payload numbers must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 256
                or any(ord(character) < 32 for character in key)
            ):
                raise ValueError(
                    "structured payload object keys must be bounded strings"
                )
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_normalize_json_value(item) for item in value]
    raise ValueError(
        f"unsupported structured payload JSON value: {type(value).__name__}"
    )


def _reject_nonfinite_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {constant}")


def _structured_contract(value: str) -> None:
    if (
        not isinstance(value, str)
        or _STRUCTURED_CONTRACT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "structured payload contract must be a bounded versioned discriminator"
        )


def _identifier(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded non-empty identifier")


def _label(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_LABEL_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded printable string")


def _bounded_text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_OBSERVATION_TEXT_LENGTH
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be NUL-free and bounded")


def _optional_uncertainty(value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("uncertainty must be finite and between 0 and 1")


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase sha256")


def _timestamp(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")


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
    "PictureObservationModality",
    "PictureObservationRecord",
    "PictureObservationStructuredPayload",
    "PictureObservationWindowPolicy",
    "canonical_timestamp",
    "normalize_picture_observation_question",
    "picture_observation_output_sha256",
    "picture_observation_request_sha256",
]
