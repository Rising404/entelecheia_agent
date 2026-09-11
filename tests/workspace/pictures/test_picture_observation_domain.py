from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from personagraph.workspace.pictures.observations import (
    PictureObservationDraft,
    PictureObservationIdempotencyConflict,
    PictureObservationModality,
    PictureObservationPersistenceConflict,
    PictureObservationRepository,
    PictureObservationService,
    PictureObservationStructuredPayload,
    PictureObservationTransactionRequired,
    PictureObservationUnitNotFound,
    PictureObservationWindowPolicy,
    picture_observation_output_sha256,
    picture_observation_request_sha256,
    project_picture_observation_window,
)
from personagraph.workspace.pictures.observations import repository as observation_repository
from personagraph.workspace.pictures.storage.schema import initialize_picture_schema


_NOW = "2026-09-04T12:00:00+00:00"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, current_version_id TEXT)"
    )
    conn.execute(
        "CREATE TABLE file_versions ("
        "id TEXT PRIMARY KEY, file_id TEXT NOT NULL, content_sha256 TEXT NOT NULL, "
        "UNIQUE(id, file_id))"
    )
    conn.execute("BEGIN IMMEDIATE")
    initialize_picture_schema(conn)
    conn.commit()
    return conn


def _seed_picture(
    conn: sqlite3.Connection,
    *,
    picture_id: str,
    unit_ids: tuple[str, ...],
) -> None:
    file_id = f"file-{picture_id}"
    version_id = f"version-{picture_id}"
    file_hash = _sha256(version_id)
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES (?, ?)",
        (file_id, version_id),
    )
    conn.execute(
        "INSERT INTO file_versions (id, file_id, content_sha256) VALUES (?, ?, ?)",
        (version_id, file_id, file_hash),
    )
    source_json = json.dumps(
        {"kind": "embedded_asset", "payload": {"asset": picture_id}},
        sort_keys=True,
        separators=(",", ":"),
    )
    conn.execute(
        "INSERT INTO pictures "
        "(picture_id, file_id, file_version_id, source_kind, source_locator_json, "
        "source_locator_sha256, source_content_sha256, source_media_type, created_at) "
        "VALUES (?, ?, ?, 'embedded_asset', ?, ?, ?, 'image/png', ?)",
        (
            picture_id,
            file_id,
            version_id,
            source_json,
            _sha256(source_json),
            file_hash,
            _NOW,
        ),
    )
    for ordinal, unit_id in enumerate(unit_ids):
        if ordinal == 0:
            unit_kind = "full"
            parent_unit_id = None
            payload: dict[str, int] = {}
        else:
            unit_kind = "tile"
            parent_unit_id = unit_ids[0]
            payload = {"height": 50, "tile": ordinal, "width": 50, "x": ordinal * 50, "y": 0}
        locator_json = json.dumps(
            {"kind": unit_kind, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "INSERT INTO picture_units "
            "(picture_unit_id, picture_id, unit_kind, unit_locator_json, "
            "unit_locator_sha256, producer_fingerprint, parent_picture_unit_id, "
            "pixel_sha256, media_type, width, height, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'test-renderer@1', ?, ?, 'image/png', 100, 100, ?)",
            (
                unit_id,
                picture_id,
                unit_kind,
                locator_json,
                _sha256(locator_json),
                parent_unit_id,
                _sha256(unit_id),
                _NOW,
            ),
        )


def _prepared_connection() -> sqlite3.Connection:
    conn = _connection()
    conn.execute("BEGIN IMMEDIATE")
    _seed_picture(
        conn,
        picture_id="picture-a",
        unit_ids=("unit-a-full", "unit-a-tile"),
    )
    _seed_picture(conn, picture_id="picture-b", unit_ids=("unit-b-full",))
    conn.commit()
    return conn


def _draft(
    *,
    picture_id: str = "picture-a",
    unit_id: str = "unit-a-full",
    invocation_id: str = "invocation-1",
    request_ordinal: int = 0,
    text: str = "a visual description",
    modality: PictureObservationModality = PictureObservationModality.VLM,
    purpose: str = "semantic-description",
    question: str | None = None,
    structured_payload: PictureObservationStructuredPayload | None = None,
) -> PictureObservationDraft:
    return PictureObservationDraft(
        picture_id=picture_id,
        picture_unit_id=unit_id,
        logical_invocation_id=invocation_id,
        request_ordinal=request_ordinal,
        modality=modality,
        purpose=purpose,
        kind="caption",
        text=text,
        uncertainty=0.1,
        processor_fingerprint="test-processor@1",
        prompt_fingerprint="test-prompt@1" if modality is PictureObservationModality.VLM else None,
        question=question,
        structured_payload=structured_payload,
    )


def _service(*, window_size: int = 8) -> PictureObservationService:
    return PictureObservationService(
        repository=PictureObservationRepository(),
        policy=PictureObservationWindowPolicy(max_active_entries=window_size),
    )

def _commit(
    conn: sqlite3.Connection,
    service: PictureObservationService,
    draft: PictureObservationDraft,
):
    conn.execute("BEGIN IMMEDIATE")
    result = service.commit_in_transaction(conn, draft, created_at=_NOW)
    assert conn.in_transaction is True
    conn.commit()
    return result


def _ocr_structured_payload() -> PictureObservationStructuredPayload:
    return PictureObservationStructuredPayload.from_payload(
        contract="ocr-lines-v1",
        payload={
            "units": [
                {
                    "coordinate_space": {
                        "height": 100,
                        "kind": "oriented_unit_pixels",
                        "width": 100,
                    },
                    "lines": [
                        {
                            "bbox": [1.25, 2.5, 80.75, 20.0],
                            "bbox_norm": [0.0125, 0.025, 0.8075, 0.2],
                            "confidence": 0.975,
                            "ordinal": 0,
                            "text": "recognized text",
                        }
                    ],
                    "picture_unit_id": "unit-a-full",
                    "status": "success",
                }
            ]
        },
    )


def test_observation_digests_are_derived_and_empty_ocr_text_is_valid() -> None:
    empty_ocr = replace(
        _draft(modality=PictureObservationModality.OCR),
        text="",
        purpose="text-extraction",
        kind="transcription",
        uncertainty=None,
        prompt_fingerprint=None,
    )

    assert empty_ocr.output_sha256 == picture_observation_output_sha256(
        text="",
        uncertainty=None,
        structured_payload=None,
    )
    assert empty_ocr.request_sha256 == picture_observation_request_sha256(
        picture_id="picture-a",
        picture_unit_id="unit-a-full",
        modality=PictureObservationModality.OCR,
        purpose="text-extraction",
        kind="transcription",
        processor_fingerprint="test-processor@1",
        prompt_fingerprint=None,
    )
    assert len(empty_ocr.payload_sha256) == 64
    committed = _commit(_prepared_connection(), _service(), empty_ocr)
    assert committed.observation.draft.text == ""
    assert committed.active_window.content
    assert committed.observation.observation_id in committed.active_window.content


def test_observation_request_digest_binds_one_exact_picture_unit() -> None:
    draft = _draft(unit_id="unit-a-full")

    assert draft.request_sha256 == (
        "8560359591f8158107b83ef4519e68291f706ff27b87aee2a5ba08a584925763"
    )
    assert draft.request_sha256 != replace(
        draft,
        picture_unit_id="unit-a-tile",
    ).request_sha256
    with pytest.raises(ValueError, match="picture_unit_id"):
        replace(_draft(), picture_unit_id="")
    with pytest.raises(ValueError, match="PictureObservationModality"):
        replace(_draft(), modality="vlm")  # type: ignore[arg-type]


def test_question_is_normalized_persisted_and_bound_to_request_identity() -> None:
    question = _draft(
        invocation_id="question-call",
        purpose="question",
        question="  图中有多少个圆？  ",
        text="图中有三个圆。",
    )
    same_answer_different_question = replace(question, question="图中有多少条线？")

    assert question.question == "图中有多少个圆？"
    assert question.request_sha256 != same_answer_different_question.request_sha256
    with pytest.raises(ValueError, match="question is required"):
        replace(question, question=None)
    with pytest.raises(ValueError, match="only valid"):
        replace(_draft(), question="不应存在")
    with pytest.raises(ValueError, match="VLM modality"):
        replace(question, modality=PictureObservationModality.OCR)

    conn = _prepared_connection()
    first = _commit(conn, _service(), question)
    loaded = PictureObservationRepository().get_by_id(
        conn,
        observation_id=first.observation.observation_id,
    )
    assert loaded is not None
    assert loaded.draft.question == "图中有多少个圆？"

    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(PictureObservationIdempotencyConflict):
        _service().commit_in_transaction(
            conn,
            same_answer_different_question,
            created_at=_NOW,
        )
    conn.rollback()


def test_structured_payload_round_trips_ocr_lines_geometry_and_unit_identity() -> None:
    structured = _ocr_structured_payload()
    reparsed = PictureObservationStructuredPayload.from_canonical_json(
        structured.canonical_json
    )
    draft = _draft(
        text="recognized text",
        modality=PictureObservationModality.OCR,
        structured_payload=structured,
    )

    assert reparsed == structured
    assert reparsed.payload["units"][0]["picture_unit_id"] == "unit-a-full"
    assert reparsed.payload["units"][0]["coordinate_space"]["kind"] == (
        "oriented_unit_pixels"
    )
    assert reparsed.payload["units"][0]["lines"][0]["bbox"] == [
        1.25,
        2.5,
        80.75,
        20.0,
    ]
    assert draft.output_sha256 != replace(
        draft,
        structured_payload=None,
    ).output_sha256

    conn = _prepared_connection()
    committed = _commit(conn, _service(), draft)
    loaded = PictureObservationRepository().get_by_id(
        conn,
        observation_id=committed.observation.observation_id,
    )
    assert loaded is not None
    assert loaded.draft.structured_payload == structured


def test_structured_payload_rejects_noncanonical_or_unbounded_json_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        PictureObservationStructuredPayload.from_payload(
            contract="ocr-lines-v1",
            payload={"confidence": float("nan")},
        )
    with pytest.raises(ValueError, match="bounded strings"):
        PictureObservationStructuredPayload.from_payload(
            contract="ocr-lines-v1",
            payload={1: "not a JSON object key"},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="65536 UTF-8 bytes"):
        PictureObservationStructuredPayload.from_payload(
            contract="ocr-lines-v1",
            payload={"text": "文" * 65_536},
        )
    with pytest.raises(ValueError, match="not canonical"):
        PictureObservationStructuredPayload.from_canonical_json(
            '{"payload": {}, "contract": "ocr-lines-v1"}'
        )
    with pytest.raises(TypeError, match="PictureObservationStructuredPayload"):
        replace(
            _draft(),
            structured_payload={"contract": "ocr-lines-v1", "payload": {}},
        )


def test_single_unit_observation_replay_is_idempotent() -> None:
    conn = _prepared_connection()
    service = _service()
    draft = _draft(unit_id="unit-a-tile")

    first = _commit(conn, service, draft)
    replayed = _commit(conn, service, draft)

    assert first.inserted is True
    assert replayed.inserted is False
    assert replayed.observation == first.observation
    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 1
    bound_unit = conn.execute(
        "SELECT picture_unit_id FROM picture_observations"
    ).fetchone()[0]
    assert bound_unit == "unit-a-tile"


def test_replay_returns_first_output_and_only_request_mismatch_conflicts() -> None:
    conn = _prepared_connection()
    service = _service()
    original = _draft()
    first = _commit(conn, service, original)

    different_output = replace(
        original,
        text="different candidate result",
        uncertainty=0.9,
        structured_payload=PictureObservationStructuredPayload.from_payload(
            contract="vision-result-v1",
            payload={"candidate": 2},
        ),
    )
    replayed = _commit(conn, service, different_output)
    assert replayed.inserted is False
    assert replayed.observation == first.observation
    assert replayed.observation.draft.text == original.text

    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(PictureObservationIdempotencyConflict):
        service.commit_in_transaction(
            conn,
            replace(original, purpose="different-request-purpose"),
            created_at=_NOW,
        )
    conn.rollback()

    other_picture = _draft(
        picture_id="picture-b",
        unit_id="unit-b-full",
        invocation_id=original.logical_invocation_id,
        request_ordinal=original.request_ordinal,
    )
    accepted = _commit(conn, service, other_picture)
    assert accepted.inserted is True


def test_non_idempotency_unique_constraint_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _prepared_connection()
    service = _service()
    original = _commit(conn, service, _draft())
    monkeypatch.setattr(
        observation_repository,
        "_observation_id",
        lambda _draft: original.observation.observation_id,
    )

    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(PictureObservationPersistenceConflict):
        service.commit_in_transaction(
            conn,
            _draft(
                picture_id="picture-b",
                unit_id="unit-b-full",
                invocation_id="another-invocation",
            ),
            created_at=_NOW,
        )
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 1


def test_fifo_window_is_per_picture_and_global_sequence_gaps_are_harmless() -> None:
    conn = _prepared_connection()
    service = _service(window_size=2)

    first_a = _commit(conn, service, _draft(invocation_id="a-1", text="a one"))
    _commit(
        conn,
        service,
        _draft(
            picture_id="picture-b",
            unit_id="unit-b-full",
            invocation_id="b-1",
            text="b one",
        ),
    )
    second_a = _commit(conn, service, _draft(invocation_id="a-2", text="a two"))
    third_a = _commit(conn, service, _draft(invocation_id="a-3", text="a three"))

    assert first_a.observation.sequence < second_a.observation.sequence
    assert third_a.active_window.observation_ids == (
        second_a.observation.observation_id,
        third_a.observation.observation_id,
    )
    assert [item.draft.text for item in third_a.active_window.observations] == [
        "a two",
        "a three",
    ]
    assert "b one" not in third_a.active_window.content
    assert "a one" not in third_a.active_window.content


def test_projection_canonicalizes_fifo_order_without_persisting_another_authority() -> None:
    conn = _prepared_connection()
    service = _service(window_size=3)
    one = _commit(conn, service, _draft(invocation_id="one", text="one")).observation
    two = _commit(conn, service, _draft(invocation_id="two", text="two")).observation

    projected = project_picture_observation_window(
        (two, one),
        policy=PictureObservationWindowPolicy(max_active_entries=2),
    )

    assert projected.observations == (one, two)
    assert projected.first_sequence == one.sequence
    assert projected.last_sequence == two.sequence
    assert projected.content_sha256 == _sha256(projected.content)
    assert '"evidence_picture_unit_id":"unit-a-full"' in projected.content
    assert "evidence_picture_unit_ids" not in projected.content
    application_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert "picture_observation_streams" not in application_tables
    assert "picture_observation_projections" not in application_tables


def test_unknown_or_cross_picture_evidence_unit_is_rejected() -> None:
    conn = _prepared_connection()
    service = _service()

    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(PictureObservationUnitNotFound):
        service.commit_in_transaction(
            conn,
            _draft(unit_id="unit-b-full"),
            created_at=_NOW,
        )
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0


def test_caller_owns_transaction_and_can_roll_back_complete_append() -> None:
    conn = _prepared_connection()
    service = _service()

    with pytest.raises(PictureObservationTransactionRequired):
        service.commit_in_transaction(conn, _draft(), created_at=_NOW)

    conn.execute("BEGIN IMMEDIATE")
    result = service.commit_in_transaction(conn, _draft(), created_at=_NOW)
    assert result.inserted is True
    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 1
    conn.rollback()

    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0


def test_append_savepoint_keeps_outer_transaction_after_insert_failure() -> None:
    conn = _prepared_connection()
    conn.execute(
        "CREATE TEMP TRIGGER fail_picture_observation "
        "BEFORE INSERT ON picture_observations "
        "BEGIN SELECT RAISE(ABORT, 'injected observation failure'); END"
    )
    conn.execute("BEGIN IMMEDIATE")

    with pytest.raises(PictureObservationPersistenceConflict):
        _service().commit_in_transaction(
            conn,
            _draft(),
            created_at=_NOW,
        )

    assert conn.in_transaction is True
    assert conn.execute("SELECT count(*) FROM picture_observations").fetchone()[0] == 0
    conn.execute(
        "INSERT INTO files (id, current_version_id) VALUES ('outer-still-live', NULL)"
    )
    conn.rollback()


def test_observation_rows_are_append_only() -> None:
    conn = _prepared_connection()
    result = _commit(conn, _service(), _draft())

    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE picture_observations SET text='mutated' WHERE observation_id=?",
            (result.observation.observation_id,),
        )
    conn.rollback()


def test_observation_domain_has_no_schema_network_tool_runtime_or_retrieval_dependency() -> None:
    source_root = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "personagraph"
        / "workspace"
        / "pictures"
        / "observations"
    )
    banned_roots = {
        "httpx",
        "requests",
        "socket",
        "urllib",
        "personagraph.model_io",
        "personagraph.retrieval",
        "personagraph.runtime",
        "personagraph.tools",
    }
    forbidden_connection_calls = {"commit", "rollback", "executescript"}

    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "CREATE TABLE" not in source.upper()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                assert not any(
                    name == banned or name.startswith(f"{banned}.")
                    for name in imported
                    for banned in banned_roots
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module == banned or node.module.startswith(f"{banned}.")
                    for banned in banned_roots
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_connection_calls
