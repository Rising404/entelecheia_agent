"""Pure derivation of a bounded picture-level FIFO observation window."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json

from .contracts import (
    PictureObservationActiveWindow,
    PictureObservationRecord,
    PictureObservationWindowPolicy,
)


def project_picture_observation_window(
    observations: Iterable[PictureObservationRecord],
    *,
    policy: PictureObservationWindowPolicy,
) -> PictureObservationActiveWindow:
    """Select the latest N observations and present them in FIFO order."""

    if not isinstance(policy, PictureObservationWindowPolicy):
        raise TypeError("policy must be PictureObservationWindowPolicy")
    candidates = tuple(observations)
    if not candidates:
        raise ValueError("at least one observation is required")
    if any(not isinstance(item, PictureObservationRecord) for item in candidates):
        raise TypeError("observations must contain PictureObservationRecord values")

    picture_ids = {item.picture_id for item in candidates}
    if len(picture_ids) != 1:
        raise ValueError("an active window cannot mix pictures")
    sequences = [item.sequence for item in candidates]
    observation_ids = [item.observation_id for item in candidates]
    if len(set(sequences)) != len(sequences) or len(set(observation_ids)) != len(observation_ids):
        raise ValueError("an active window cannot contain duplicate observations")

    ordered = tuple(sorted(candidates, key=lambda item: item.sequence))
    active = ordered[-policy.max_active_entries :]
    picture_id = active[0].picture_id
    content = _render_active_window(picture_id=picture_id, observations=active)
    return PictureObservationActiveWindow(
        picture_id=picture_id,
        observations=active,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _render_active_window(
    *,
    picture_id: str,
    observations: tuple[PictureObservationRecord, ...],
) -> str:
    sections = [
        json.dumps(
            {
                "active_observation_count": len(observations),
                "picture_id": picture_id,
                "type": "picture_observation_window",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    for observation in observations:
        draft = observation.draft
        metadata = {
            "evidence_picture_unit_id": draft.picture_unit_id,
            "kind": draft.kind,
            "modality": draft.modality.value,
            "observation_id": observation.observation_id,
            "purpose": draft.purpose,
            "sequence": observation.sequence,
            "type": "picture_observation",
            "uncertainty": draft.uncertainty,
        }
        if draft.question is not None:
            metadata["question"] = draft.question
        sections.append(
            "\n".join(
                (
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    draft.text,
                )
            )
        )
    return "\n\n".join(sections)


__all__ = ["project_picture_observation_window"]
