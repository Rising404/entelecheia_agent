"""Atomic Project publication for provider-produced picture observations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from personagraph.retrieval.sources.picture_publication import (
    PictureObservationOutboxPublisher,
)
from personagraph.workspace.pictures import (
    PictureSourceLocator,
    PictureUnitLocator,
)
from personagraph.workspace.pictures.project_publication import (
    ProjectPictureBinding,
    ProjectPictureBindingStale,
    ProjectPictureObservationSpec,
    ProjectPicturePublicationCommand,
    ProjectPicturePublicationService,
    ProjectPictureUnitSpec,
)
from personagraph.workspace.pictures.observations import (
    PictureObservationStructuredPayload,
)
from personagraph.workspace.storage.database import DocumentDatabase


NOW = "2026-09-04T12:00:00+00:00"
FILE_SHA = "a" * 64
PIXEL_SHA = "b" * 64


def _database(tmp_path: Path) -> DocumentDatabase:
    root = tmp_path / "project"
    root.mkdir()
    database = DocumentDatabase(
        "project-1",
        root,
        tmp_path / "documents.sqlite",
    )
    database.initialize()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO files "
            "(id, project_id, relative_path, origin, media_type, "
            "current_version_id, created_at, updated_at) "
            "VALUES ('file-1', 'project-1', 'chart.png', 'user_upload', "
            "'image/png', NULL, ?, ?)",
            (NOW, NOW),
        )
        conn.execute(
            "INSERT INTO file_versions "
            "(id, file_id, version_number, producer, content_sha256, size_bytes, "
            "source_mtime_ns, created_at) "
            "VALUES ('version-1', 'file-1', 1, 'user_upload', ?, 128, 1, ?)",
            (FILE_SHA, NOW),
        )
        conn.execute(
            "UPDATE files SET current_version_id='version-1' WHERE id='file-1'"
        )
        conn.execute(
            "INSERT INTO retrieval_data_versions "
            "(id, fingerprint, role, state, created_at, activated_at) "
            "VALUES ('retrieval-v1', 'fingerprint-v1', 'active', 'ready', ?, ?)",
            (NOW, NOW),
        )
    return database


def _service(database: DocumentDatabase) -> ProjectPicturePublicationService:
    return ProjectPicturePublicationService(
        database=database,
        publication_port=PictureObservationOutboxPublisher(
            retrieval_data_version_resolver=lambda conn: (
                row[0]
                if (
                    row := conn.execute(
                        "SELECT id FROM retrieval_data_versions "
                        "WHERE role='active' AND state='ready'"
                    ).fetchone()
                )
                else None
            )
        ),
    )


def _observation(ordinal: int, text: str) -> ProjectPictureObservationSpec:
    return ProjectPictureObservationSpec(
        request_ordinal=ordinal,
        purpose="chart",
        kind="chart",
        text=text,
        uncertainty=0.1 + ordinal / 100,
        processor_fingerprint="provider-processor-v1",
        prompt_fingerprint="vision-purpose-v1",
        structured_payload=PictureObservationStructuredPayload.from_payload(
            contract="vlm-picture-observation-v1",
            payload={
                "provider": "provider",
                "model": "model",
                "endpoint_identity": "endpoint",
                "result_sha256": "c" * 64,
                "output_sha256": "d" * 64,
                "provider_observation_id": f"provider-observation-{ordinal}",
                "warnings": [],
                "unresolved_gap_refs": [],
            },
        ),
    )


def _command() -> ProjectPicturePublicationCommand:
    return ProjectPicturePublicationCommand(
        binding=ProjectPictureBinding(
            project_id="project-1",
            file_id="file-1",
            file_version_id="version-1",
            file_content_sha256=FILE_SHA,
            file_media_type="image/png",
        ),
        unit=ProjectPictureUnitSpec(
            source_locator=PictureSourceLocator.whole_file(),
            source_content_sha256=FILE_SHA,
            source_media_type="image/png",
            unit_locator=PictureUnitLocator.full(),
            producer_fingerprint="prepared-artifact-v1",
            parent_picture_unit_id=None,
            pixel_sha256=PIXEL_SHA,
            media_type="image/png",
            width=24,
            height=16,
        ),
        logical_invocation_id="mvc_" + "e" * 64,
        observations=(_observation(0, "one analysis-call observation"),),
        occurred_at=NOW,
    )


def test_one_call_publication_is_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    service = _service(database)

    first = service.publish(_command())
    replay = service.publish(_command())

    assert [item.observation.draft.text for item in first.observation_commits] == [
        "one analysis-call observation",
    ]
    assert [item.observation.draft.request_ordinal for item in first.observation_commits] == [
        0,
    ]
    assert all(item.inserted for item in first.observation_commits)
    assert all(not item.inserted for item in replay.observation_commits)
    assert replay.picture_id == first.picture_id
    assert replay.picture_unit_id == first.picture_unit_id
    with database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM pictures").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM picture_units").fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM picture_observations"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM retrieval_update_outbox "
            "WHERE source_type='picture' AND kind='upsert'"
        ).fetchone()[0] == 1


def test_question_publication_persists_the_question_with_the_answer(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    command = replace(
        _command(),
        observations=(
            replace(
                _observation(0, "There are three blue bars."),
                purpose="question",
                question="How many blue bars are shown?",
            ),
        ),
    )

    committed = _service(database).publish(command)

    assert committed.observation_commits[0].observation.draft.question == (
        "How many blue bars are shown?"
    )
    with database.connect() as conn:
        stored = conn.execute(
            "SELECT question, text FROM picture_observations"
        ).fetchone()
    assert tuple(stored) == (
        "How many blue bars are shown?",
        "There are three blue bars.",
    )


def test_one_logical_call_cannot_consume_multiple_fifo_entries() -> None:
    with pytest.raises(ValueError, match="exactly one observation"):
        replace(
            _command(),
            observations=(
                _observation(0, "first"),
                _observation(1, "second"),
            ),
        )


@pytest.mark.parametrize(
    "binding",
    [
        replace(_command().binding, project_id="other-project"),
        replace(_command().binding, file_version_id="version-2"),
        replace(_command().binding, file_content_sha256="f" * 64),
        replace(_command().binding, file_media_type="image/jpeg"),
    ],
    ids=("project", "version", "hash", "media"),
)
def test_wrong_project_or_file_binding_fails_without_partial_state(
    tmp_path: Path,
    binding: ProjectPictureBinding,
) -> None:
    database = _database(tmp_path)

    with pytest.raises(ProjectPictureBindingStale):
        _service(database).publish(replace(_command(), binding=binding))

    with database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM pictures").fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM picture_observations"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM retrieval_update_outbox"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "unit",
    [
        replace(_command().unit, source_content_sha256="f" * 64),
        replace(_command().unit, source_media_type="image/jpeg"),
    ],
    ids=("source-hash", "source-media"),
)
def test_file_backed_unit_must_match_frozen_binding(
    tmp_path: Path,
    unit: ProjectPictureUnitSpec,
) -> None:
    database = _database(tmp_path)

    with pytest.raises(ProjectPictureBindingStale):
        _service(database).publish(replace(_command(), unit=unit))

    with database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM pictures").fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM picture_observations"
        ).fetchone()[0] == 0
