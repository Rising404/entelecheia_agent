from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from personagraph.trajectory import (
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
    TrajectoryReadError,
    TrajectorySchemaError,
    TrajectoryStore,
    export_trajectory,
    render_trajectory,
    text_blob,
)


def _step(
    step_id: str,
    *,
    occurred_at: str,
    parts: tuple[Part, ...] = (),
    outcome: StepOutcome = StepOutcome.OK,
    reason_code: str | None = None,
) -> Step:
    return Step(
        step_id=step_id,
        kind=StepKind.MODEL_CALL,
        occurred_at=occurred_at,
        parts=parts,
        session_id="session",
        turn_id="turn",
        model_call_id=f"call-{step_id}",
        purpose="test",
        duration_ms=7,
        outcome=outcome,
        reason_code=reason_code,
        metrics={"input_tokens": 11},
    )


def test_read_all_preserves_order_and_content_addressed_deduplication(
    tmp_path,
) -> None:
    database = tmp_path / "trajectory.sqlite"
    store = TrajectoryStore(database)
    shared = text_blob("shared prompt")
    store.record(
        _step(
            "second",
            occurred_at="2026-09-06T00:00:02+00:00",
            parts=(Part(PartRole.USER, shared),),
            outcome=StepOutcome.REJECTED,
            reason_code="rejected",
        )
    )
    store.record(
        _step(
            "first",
            occurred_at="2026-09-06T00:00:01+00:00",
            parts=(
                Part(PartRole.SYSTEM, shared),
                Part(PartRole.ASSISTANT, text_blob("answer")),
            ),
        )
    )
    unreferenced = text_blob("unreferenced diagnostic")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO trajectory_blobs"
            " (sha256, byte_count, text, truncated, first_seen_at)"
            " VALUES (?,?,?,?,?)",
            (
                unreferenced.sha256,
                unreferenced.byte_count,
                unreferenced.text,
                0,
                "2026-09-06T00:00:03+00:00",
            ),
        )

    snapshot = store.read_all()

    assert set(snapshot) == {"format", "steps", "blobs", "integrity"}
    assert snapshot["format"] == "personagraph.trajectory"
    assert [step["step_id"] for step in snapshot["steps"]] == [
        "first",
        "second",
    ]
    assert snapshot["steps"][0] == {
        "step_id": "first",
        "kind": "model_call",
        "occurred_at": "2026-09-06T00:00:01+00:00",
        "session_id": "session",
        "turn_id": "turn",
        "model_call_id": "call-first",
        "purpose": "test",
        "duration_ms": 7,
        "outcome": "ok",
        "reason_code": None,
        "metrics": {"input_tokens": 11},
        "parts": [
            {"seq": 0, "role": "system", "blob_sha256": shared.sha256},
            {
                "seq": 1,
                "role": "assistant",
                "blob_sha256": text_blob("answer").sha256,
            },
        ],
    }
    assert snapshot["blobs"][shared.sha256]["text"] == "shared prompt"
    assert snapshot["blobs"][unreferenced.sha256]["text"] == (
        "unreferenced diagnostic"
    )
    assert snapshot["integrity"] == {
        "step_count": 2,
        "part_count": 3,
        "blob_count": 3,
        "truncated_blob_count": 0,
        "recording_failure_count": 0,
    }


def test_read_all_has_no_two_hundred_step_ceiling(tmp_path) -> None:
    store = TrajectoryStore(tmp_path / "trajectory.sqlite")
    for index in range(205):
        store.record(
            _step(
                f"step-{index:03d}",
                occurred_at=f"2026-09-06T00:{index // 60:02d}:{index % 60:02d}+00:00",
            )
        )

    assert len(store.read_all()["steps"]) == 205


def test_export_is_deterministic_and_returns_the_file_identity(tmp_path) -> None:
    database = tmp_path / "trajectory.sqlite"
    store = TrajectoryStore(database)
    store.record(
        _step(
            "step",
            occurred_at="2026-09-06T00:00:00+00:00",
            parts=(Part(PartRole.USER, text_blob("question")),),
        )
    )

    first = export_trajectory(database, tmp_path / "first.json")
    second = export_trajectory(database, tmp_path / "second.json")

    assert first.sha256 == second.sha256
    assert first.byte_count == second.byte_count
    assert first.snapshot == second.snapshot
    assert first.path.read_bytes() == second.path.read_bytes()


def test_readable_export_groups_rejection_with_call_and_links_actual_repair(tmp_path) -> None:
    database = tmp_path / "trajectory.sqlite"
    store = TrajectoryStore(database)
    digest = text_blob("rejected draft").sha256
    shared = text_blob(' {"question":"readable question"}')
    store.record(replace(_step(
        "first", occurred_at="2026-09-06T00:00:01+00:00",
        parts=(Part(PartRole.USER, shared), Part(PartRole.ASSISTANT, text_blob("draft"))),
    ), model_call_id="logical:physical:1"))
    store.record(replace(_step(
        "rejection", occurred_at="2026-09-06T00:00:02+00:00",
        parts=(Part(PartRole.REJECTED_OUTPUT, text_blob(json.dumps({
            "schema_version": "runtime-model-output-rejection-observation-v1",
            "rejected_response_sha256": digest, "repair_scheduled": True,
            "issues": [{"paths": ["/note"], "code": "schema.missing", "safe_explanation": "需要公开笔记。"}],
        }))),), outcome=StepOutcome.REJECTED, reason_code="schema.missing",
    ), model_call_id="logical:physical:1"))
    store.record(replace(_step(
        "repair", occurred_at="2026-09-06T00:00:03+00:00",
        parts=(Part(PartRole.USER, shared), Part(PartRole.ASSISTANT, text_blob(json.dumps({
            "reference_kind": "trajectory-redacted-rejected-model-output",
            "response_sha256": digest, "byte_count": 14, "body_owner": "runtime_model_rejected_output",
        }))), Part(PartRole.ASSISTANT, text_blob("final reply"))),
    ), model_call_id="logical:physical:2"))

    snapshot = store.read_all()
    original = json.dumps(snapshot, sort_keys=True)
    readable = render_trajectory(snapshot)

    assert readable.count("## 1. 模型调用") == readable.count("## 2. 模型调用") == 1
    assert "## 3." not in readable
    assert "需要公开笔记。" in readable and "已安排后续修复。" in readable
    assert "修复关系：[此前被拒的调用](#call-1)" in readable
    assert readable.count("readable question") == 1
    assert "[与前文相同](#part-1-1-0)" in readable
    assert "不保证该正文存在可读取的副本" in readable
    assert "final reply" in readable
    assert json.dumps(snapshot, sort_keys=True) == original

    artifact = export_trajectory(database, tmp_path / "readable.md", format="markdown")
    assert artifact.path.read_text() == readable
    assert artifact.snapshot == snapshot
    assert artifact.byte_count == len(readable.encode("utf-8"))
    assert store.read_all() == snapshot


def test_readable_export_marks_truncation_and_keeps_markdown_inside_content(tmp_path) -> None:
    from personagraph.trajectory import Blob

    store = TrajectoryStore(tmp_path / "trajectory.sqlite")
    text = "正文\n```\n<script>not executable</script>\n```"
    source = text_blob(text)
    store.record(_step(
        "part", occurred_at="2026-09-06T00:00:01+00:00",
        parts=(Part(PartRole.ASSISTANT, Blob(source.sha256, 900, text, truncated=True)),),
    ))
    readable = render_trajectory(store.read_all())
    assert f"````\n{text}\n````" in readable
    assert "仅保存前缀；原文共 900 字节" in readable


def test_unknown_export_format_does_not_read_or_create_files(tmp_path) -> None:
    existing = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="format"):
        export_trajectory(tmp_path / "missing.sqlite", tmp_path / "result", format="html")
    assert set(tmp_path.iterdir()) == existing


def test_read_all_never_creates_a_missing_database(tmp_path) -> None:
    database = tmp_path / "missing.sqlite"

    with pytest.raises(TrajectoryReadError, match="does not exist"):
        TrajectoryStore(database).read_all()

    assert not database.exists()


def test_read_all_does_not_repair_an_incomplete_database(tmp_path) -> None:
    database = tmp_path / "incomplete.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(TrajectoryReadError, match="schema is incomplete"):
        TrajectoryStore(database).read_all()

    with sqlite3.connect(database) as connection:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert names == {"unrelated"}


def test_record_rejects_an_unknown_schema_without_creating_current_tables(
    tmp_path,
) -> None:
    database = tmp_path / "unknown.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE trajectory_unknown (value TEXT)")

    with pytest.raises(TrajectorySchemaError, match="schema is not supported"):
        TrajectoryStore(database).record(
            _step("step", occurred_at="2026-09-06T00:00:00+00:00")
        )

    with sqlite3.connect(database) as connection:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "trajectory_blobs" not in names
    assert "trajectory_steps" not in names
    assert "trajectory_parts" not in names


@pytest.mark.parametrize(
    ("corrupt", "message"),
    (
        (
            "UPDATE trajectory_steps SET metrics_json='not-json'",
            "metrics_json",
        ),
        (
            "UPDATE trajectory_parts SET seq=2 WHERE step_id='step' AND seq=0",
            "part sequence",
        ),
        (
            "DELETE FROM trajectory_blobs",
            "missing blob",
        ),
        (
            "UPDATE trajectory_parts SET step_id='missing-step'",
            "orphan trajectory part",
        ),
        (
            "UPDATE trajectory_steps SET kind='not-a-kind'",
            "step kind",
        ),
        (
            "UPDATE trajectory_steps SET duration_ms=-1",
            "duration_ms",
        ),
    ),
)
def test_read_all_rejects_incomplete_or_invalid_storage(
    tmp_path,
    corrupt: str,
    message: str,
) -> None:
    database = tmp_path / "trajectory.sqlite"
    store = TrajectoryStore(database)
    store.record(
        _step(
            "step",
            occurred_at="2026-09-06T00:00:00+00:00",
            parts=(Part(PartRole.USER, text_blob("question")),),
        )
    )
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(corrupt)

    with pytest.raises(TrajectoryReadError, match=message):
        store.read_all()
