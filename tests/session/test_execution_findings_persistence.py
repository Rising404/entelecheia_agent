from __future__ import annotations

import hashlib
import sqlite3

import pytest

from personagraph.persistent_turn_content.findings import (
    ExecutionFindingKind,
    ExecutionFindingSourceRef,
    ExecutionFindingStatus,
    ExecutionFindingsMutationCommand,
    ExecutionFindingsOwnerKind,
    RecordExecutionFinding,
    RetractExecutionFinding,
    SupersedeExecutionFinding,
    canonical_json,
    derive_execution_findings_mutation_id,
)
from personagraph.persistent_turn_content.plan import (
    L1Acceptance,
    L1MessageSource,
    L1Plan,
)
from personagraph.session.persistence import execution_findings as records
from personagraph.session.persistence import schema
from personagraph.session.persistence.current_schema import CURRENT_SCHEMA_SQL
from personagraph.session.persistence.deps import StoreDeps
from personagraph.output_protocol.l1_persistence import legacy_l1_tool_result_id


L1_RUN_ID = "l1-run"
L1_STEP_ID = "l1-step"


@pytest.fixture
def findings_deps(tmp_path) -> StoreDeps:
    db_path = tmp_path / "findings.sqlite"

    def connect() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE runtime_turns (
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                PRIMARY KEY(turn_id),
                UNIQUE(session_id, turn_id)
            );
            CREATE TABLE l1_turn_runs (
                l1_turn_run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(session_id, turn_id, l1_turn_run_id)
            );
            CREATE TABLE l1_turn_run_states (
                l1_turn_run_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                plan_json TEXT
            );
            CREATE TABLE l1_turn_tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                l1_turn_run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                execution_class TEXT NOT NULL DEFAULT 'read_only',
                status TEXT NOT NULL,
                outcome_json TEXT,
                outcome_hash TEXT
            );
            """
        )
        for statement in schema._sql_statements(CURRENT_SCHEMA_SQL):
            if "execution_finding" in statement.splitlines()[0]:
                conn.execute(statement)
        conn.executemany(
            "INSERT INTO runtime_turns(session_id, turn_id) VALUES ('session', ?)",
            (("turn-l1",),),
        )
        conn.execute(
            "INSERT INTO l1_turn_runs "
            "(l1_turn_run_id, session_id, turn_id, status) "
            "VALUES (?, 'session', 'turn-l1', 'running')",
            (L1_RUN_ID,),
        )
        user_text = "summarize the document and identify risks"
        source = L1MessageSource(
            message_id="message-l1", content_sha256=hashlib.sha256(user_text.encode()).hexdigest(),
        )
        l1_plan = L1Plan(
            objective="summarize",
            acceptances=(
                L1Acceptance(
                    acceptance_id="summary", criterion="summarize the document", source=source,
                ),
                L1Acceptance(
                    acceptance_id="risks", criterion="identify risks", source=source,
                ),
            ),
        )
        conn.execute(
            "INSERT INTO l1_turn_run_states(l1_turn_run_id, stage, plan_json) "
            "VALUES (?, 'tool', ?)",
            (L1_RUN_ID, canonical_json(l1_plan)),
        )
        l1_outcome = {
            "status": "succeeded",
            "result": _document_output(),
            "error": None,
            "metadata": {},
        }
        l1_outcome_json = canonical_json(l1_outcome)
        conn.execute(
            "INSERT INTO l1_turn_tool_calls "
            "(tool_call_id, l1_turn_run_id, step_id, tool_id, status, "
            "outcome_json, outcome_hash) VALUES "
            "('l1-source-call', ?, ?, 'mounted_document_read', 'succeeded', ?, ?)",
            (
                L1_RUN_ID,
                L1_STEP_ID,
                l1_outcome_json,
                hashlib.sha256(l1_outcome_json.encode("utf-8")).hexdigest(),
            ),
        )
        _insert_l1_writer(conn, "l1-record-call", "record_execution_findings")


    return StoreDeps(
        init_db=lambda: None,
        connect=connect,
        now=lambda: "2026-08-26T00:00:00+00:00",
        new_id=lambda: "unused",
    )


def test_owner_companion_fails_closed_when_current_schema_is_missing() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA user_version = {schema.SCHEMA_VERSION}")

    with pytest.raises(
        records.ExecutionFindingsPersistenceError,
        match="schema is missing",
    ):
        records.create_execution_findings_owner_companion_in_transaction(
            conn,
            session_id="session",
            owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
            execution_owner_id="current-l1-run",
            now="2026-08-26T00:00:00+00:00",
        )


def test_l1_mutations_are_idempotent_and_preserve_revision_history(
    findings_deps: StoreDeps,
) -> None:
    created = records.create_execution_findings_ledger(
        findings_deps,
        session_id="session",
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )
    replayed_create = records.create_execution_findings_ledger(
        findings_deps,
        session_id="session",
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )
    assert created.replayed is False
    assert replayed_create.replayed is True

    source_ref = _l1_chunk_ref(findings_deps)
    record_command = _record_command(
        ledger_id=created.ledger.ledger_id,
        writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-record-call",
        expected_revision=0,
        item=RecordExecutionFinding(
            kind=ExecutionFindingKind.FINDING,
            claim="The first chunk contains the executive summary.",
            source_refs=(source_ref,),
            scope_keys=("summary",),
        ),
    )
    recorded = records.apply_execution_findings_mutation(
        findings_deps,
        command=record_command,
    )
    replayed = records.apply_execution_findings_mutation(
        findings_deps,
        command=record_command,
    )
    assert recorded.ledger.revision == 1
    assert replayed.replayed is True
    assert replayed.receipt == recorded.receipt
    entry_id = recorded.ledger.entry_revisions[0].entry_id

    _insert_l1_writer_with_deps(
        findings_deps,
        "l1-revise-call",
        "revise_execution_finding",
    )
    superseded = records.apply_execution_findings_mutation(
        findings_deps,
        command=ExecutionFindingsMutationCommand(
            ledger_id=created.ledger.ledger_id,
            mutation_id=derive_execution_findings_mutation_id(
                writer_tool_call_id="l1-revise-call"
            ),
            expected_ledger_revision=1,
            writer_unit_id=L1_STEP_ID,
            writer_tool_call_id="l1-revise-call",
            items=(
                SupersedeExecutionFinding(
                    entry_id=entry_id,
                    kind=ExecutionFindingKind.FINDING,
                    claim="The first chunk contains the scope and executive summary.",
                    source_refs=(source_ref,),
                    scope_keys=("summary", "risks"),
                ),
            ),
        ),
    )
    assert [item.status for item in superseded.ledger.entry_revisions] == [
        ExecutionFindingStatus.SUPERSEDED,
        ExecutionFindingStatus.ACTIVE,
    ]

    _insert_l1_writer_with_deps(
        findings_deps,
        "l1-retract-call",
        "revise_execution_finding",
    )
    retracted = records.apply_execution_findings_mutation(
        findings_deps,
        command=ExecutionFindingsMutationCommand(
            ledger_id=created.ledger.ledger_id,
            mutation_id=derive_execution_findings_mutation_id(
                writer_tool_call_id="l1-retract-call"
            ),
            expected_ledger_revision=2,
            writer_unit_id=L1_STEP_ID,
            writer_tool_call_id="l1-retract-call",
            items=(
                RetractExecutionFinding(
                    entry_id=entry_id,
                    reason="The cited chunk was interpreted too broadly.",
                ),
            ),
        ),
    )
    assert retracted.ledger.revision == 3
    assert retracted.ledger.entry_revisions[-1].status is (
        ExecutionFindingStatus.RETRACTED
    )
    assert retracted.active_projection.active_entries == ()
    assert len(retracted.ledger.entry_revisions) == 3


def test_persisted_claims_with_frozen_quota_remain_replayable_and_retractable(
    findings_deps: StoreDeps,
) -> None:
    created = records.create_execution_findings_ledger(
        findings_deps,
        session_id="session",
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )
    first_command = _record_command(
        ledger_id=created.ledger.ledger_id,
        writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-record-call",
        expected_revision=0,
        item=RecordExecutionFinding(
            kind=ExecutionFindingKind.FINDING,
            claim="Originally admitted under this ledger's frozen quota.",
            scope_keys=("summary",),
        ),
    )
    first = records.apply_execution_findings_mutation(
        findings_deps,
        command=first_command,
    )
    legacy_claim = "旧" * 1_500
    legacy_quota = created.ledger.quota.model_copy(
        update={"max_claim_characters": 1_500}
    )
    legacy_quota_json = canonical_json(legacy_quota)
    with findings_deps.connect() as conn:
        conn.execute(
            "UPDATE execution_findings_ledgers SET quota_json=?, quota_hash=? "
            "WHERE ledger_id=?",
            (
                legacy_quota_json,
                hashlib.sha256(legacy_quota_json.encode("utf-8")).hexdigest(),
                created.ledger.ledger_id,
            ),
        )
        conn.execute(
            "UPDATE execution_finding_entry_revisions SET claim=? "
            "WHERE entry_revision_id=?",
            (legacy_claim, first.ledger.entry_revisions[0].entry_revision_id),
        )

    _insert_l1_writer_with_deps(
        findings_deps,
        "l1-after-legacy-call",
        "record_execution_findings",
    )
    second_command = _record_command(
        ledger_id=created.ledger.ledger_id,
        writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-after-legacy-call",
        expected_revision=1,
        item=RecordExecutionFinding(
            kind=ExecutionFindingKind.GAP,
            claim="This is a current-size finding written after the upgrade.",
            scope_keys=("risks",),
        ),
    )
    second = records.apply_execution_findings_mutation(
        findings_deps,
        command=second_command,
    )
    assert second.ledger.entry_revisions[0].claim == legacy_claim

    replayed = records.apply_execution_findings_mutation(
        findings_deps,
        command=second_command,
    )
    assert replayed.replayed is True
    assert replayed.ledger.entry_revisions[0].claim == legacy_claim

    _insert_l1_writer_with_deps(
        findings_deps,
        "l1-retract-legacy-call",
        "revise_execution_finding",
    )
    retracted = records.apply_execution_findings_mutation(
        findings_deps,
        command=ExecutionFindingsMutationCommand(
            ledger_id=created.ledger.ledger_id,
            mutation_id=derive_execution_findings_mutation_id(
                writer_tool_call_id="l1-retract-legacy-call"
            ),
            expected_ledger_revision=2,
            writer_unit_id=L1_STEP_ID,
            writer_tool_call_id="l1-retract-legacy-call",
            items=(
                RetractExecutionFinding(
                    entry_id=first.ledger.entry_revisions[0].entry_id,
                    reason="Retire the legacy-sized finding.",
                ),
            ),
        ),
    )
    assert retracted.ledger.entry_revisions[-1].claim == legacy_claim
    assert retracted.ledger.entry_revisions[-1].status is (
        ExecutionFindingStatus.RETRACTED
    )


def test_projection_reducer_failure_is_stored_authority_corruption(
    findings_deps: StoreDeps,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = records.create_execution_findings_ledger(
        findings_deps,
        session_id="session",
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )

    def reject_corrupt_history(*_args: object, **_kwargs: object) -> object:
        raise ValueError("corrupt queue history")

    monkeypatch.setattr(
        records,
        "reduce_execution_findings_active_queue",
        reject_corrupt_history,
    )
    with pytest.raises(
        records.ExecutionFindingsStoredAuthorityCorrupt,
        match="queue history is invalid",
    ):
        records.get_execution_findings_ledger(
            findings_deps,
            ledger_id=created.ledger.ledger_id,
        )


def _document_output() -> dict[str, object]:
    return {
        "file_id": "file-a",
        "file_version_id": "version-a",
        "chunks": [
            {
                "chunk_id": "chunk-a",
                "sequence": 0,
                "content": "chunk zero",
            }
        ],
    }


def _record_command(
    *,
    ledger_id: str,
    writer_unit_id: str,
    writer_tool_call_id: str,
    expected_revision: int,
    item: RecordExecutionFinding,
) -> ExecutionFindingsMutationCommand:
    return ExecutionFindingsMutationCommand(
        ledger_id=ledger_id,
        mutation_id=derive_execution_findings_mutation_id(
            writer_tool_call_id=writer_tool_call_id
        ),
        expected_ledger_revision=expected_revision,
        writer_unit_id=writer_unit_id,
        writer_tool_call_id=writer_tool_call_id,
        items=(item,),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_l1_tool_result_identity_is_preserved_and_replayed(findings_deps, legacy):
    created = records.create_execution_findings_ledger(
        findings_deps, session_id="session", owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )
    ref = _l1_chunk_ref(findings_deps).model_copy(update={"chunk_id": None})
    if legacy:
        ref = ExecutionFindingSourceRef(tool_result_id=legacy_l1_tool_result_id(
            tool_call_id=ref.tool_result_id, result_sha256=ref.result_sha256,
        ))
    command = _record_command(
        ledger_id=created.ledger.ledger_id, writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-record-call", expected_revision=0,
        item=RecordExecutionFinding(
            kind=ExecutionFindingKind.FINDING, claim="The document supports this finding.",
            scope_keys=("summary",), source_refs=(ref,),
        ),
    )
    applied = records.apply_execution_findings_mutation(findings_deps, command=command)
    assert applied.ledger.entry_revisions[0].source_refs[0] == ref
    replayed = records.apply_execution_findings_mutation(findings_deps, command=command)
    assert replayed.replayed is True
    assert replayed.ledger.revision == 1
    assert len(replayed.ledger.entry_revisions) == 1


@pytest.mark.parametrize("corruption", ["result_id", "hash", "missing_digest", "wrong_digest", "digest_and_body", "call", "scope", "failed", "findings", "history_list", "history_read"])
def test_l1_tool_result_identity_does_not_weaken_source_authority(findings_deps, corruption):
    created = records.create_execution_findings_ledger(
        findings_deps, session_id="session", owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=L1_RUN_ID,
    )
    ref = _l1_chunk_ref(findings_deps).model_dump()
    if corruption == "result_id":
        ref["tool_result_id"] = "another-call"
    elif corruption == "hash":
        with findings_deps.connect() as conn:
            conn.execute("UPDATE l1_turn_tool_calls SET outcome_json='{}' WHERE tool_call_id='l1-source-call'")
    elif corruption == "missing_digest":
        ref.pop("result_sha256")
    elif corruption == "wrong_digest":
        ref["result_sha256"] = "0" * 64
    elif corruption == "digest_and_body":
        replacement = canonical_json({"status": "succeeded", "result": {"text": "different"}})
        with findings_deps.connect() as conn:
            conn.execute(
                "UPDATE l1_turn_tool_calls SET outcome_json=?, outcome_hash=? WHERE tool_call_id='l1-source-call'",
                (replacement, hashlib.sha256(replacement.encode()).hexdigest()),
            )
    else:
        field, value = {
            "call": ("tool_call_id", "other-call"),
            "scope": ("l1_turn_run_id", "another-l1-run"),
            "failed": ("status", "failed"),
            "findings": ("tool_id", "record_execution_findings"),
            "history_list": ("tool_id", "list_tool_results"),
            "history_read": ("tool_id", "read_tool_result"),
        }[corruption]
        with findings_deps.connect() as conn:
            conn.execute(f"UPDATE l1_turn_tool_calls SET {field}=? WHERE tool_call_id='l1-source-call'", (value,))
    command = _record_command(
        ledger_id=created.ledger.ledger_id, writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-record-call", expected_revision=0,
        item=RecordExecutionFinding(
            kind=ExecutionFindingKind.FINDING, claim="This source must be rejected.",
            scope_keys=("summary",), source_refs=(ExecutionFindingSourceRef(**ref),),
        ),
    )
    error_type = (records.ExecutionFindingsStoredAuthorityCorrupt if corruption == "hash"
                  else records.ExecutionFindingsSourceReferenceInvalid)
    with pytest.raises(error_type):
        records.apply_execution_findings_mutation(findings_deps, command=command)
    unchanged = records.get_execution_findings_ledger(findings_deps, ledger_id=created.ledger.ledger_id)
    assert unchanged.ledger.revision == 0
    assert unchanged.ledger.entry_revisions == ()


def _l1_chunk_ref(findings_deps: StoreDeps) -> ExecutionFindingSourceRef:
    with findings_deps.connect() as conn:
        outcome_hash = str(
            conn.execute(
                "SELECT outcome_hash FROM l1_turn_tool_calls "
                "WHERE tool_call_id='l1-source-call'"
            ).fetchone()[0]
        )
    return ExecutionFindingSourceRef(
        tool_result_id="l1-source-call",
        result_sha256=outcome_hash,
        chunk_id="chunk-a",
    )


@pytest.mark.parametrize("shape,accepted", [
    ("batch", True), ("pointer_only", False), ("other_chunk", False), ("ambiguous", False),
])
def test_finding_chunk_must_belong_to_the_cited_result(findings_deps, shape, accepted):
    if shape == "batch":
        output = {"files": [_document_output()]}
    elif shape == "pointer_only":
        output = {"head_chunk_id": "chunk-a", "tail_chunk_id": "chunk-a"}
    elif shape == "other_chunk":
        output = {"chunks": [{"chunk_id": "chunk-b", "content": "not cited"}]}
    else:
        output = {"chunks": [
            {"chunk_id": "chunk-a", "content": "first"},
            {"chunk_id": "chunk-a", "content": "conflicting second"},
        ]}
    outcome = canonical_json({"status": "succeeded", "result": output})
    with findings_deps.connect() as conn:
        conn.execute(
            "UPDATE l1_turn_tool_calls SET outcome_json=?, outcome_hash=? "
            "WHERE tool_call_id='l1-source-call'",
            (outcome, hashlib.sha256(outcome.encode()).hexdigest()),
        )
    ledger = records.create_execution_findings_ledger(
        findings_deps, session_id="session", owner_kind="l1_turn_run", execution_owner_id=L1_RUN_ID,
    ).ledger
    command = _record_command(
        ledger_id=ledger.ledger_id, writer_unit_id=L1_STEP_ID,
        writer_tool_call_id="l1-record-call", expected_revision=0,
        item=RecordExecutionFinding(
            kind="finding", claim="Native chunk reference", source_refs=(_l1_chunk_ref(findings_deps),),
        ),
    )
    if accepted:
        result = records.apply_execution_findings_mutation(findings_deps, command=command)
        assert result.ledger.entry_revisions[0].source_refs[0].chunk_id == "chunk-a"
    else:
        with pytest.raises(records.ExecutionFindingsSourceReferenceInvalid, match="chunk_"):
            records.apply_execution_findings_mutation(findings_deps, command=command)
        assert records.get_execution_findings_ledger(
            findings_deps, ledger_id=ledger.ledger_id,
        ).ledger.revision == 0


def _insert_l1_writer(
    conn: sqlite3.Connection,
    tool_call_id: str,
    tool_id: str,
) -> None:
    conn.execute(
        "INSERT INTO l1_turn_tool_calls "
        "(tool_call_id, l1_turn_run_id, step_id, tool_id, execution_class, status) "
        "VALUES (?, ?, ?, ?, 'runtime_state', 'pending')",
        (tool_call_id, L1_RUN_ID, L1_STEP_ID, tool_id),
    )


def _insert_l1_writer_with_deps(
    deps: StoreDeps,
    tool_call_id: str,
    tool_id: str,
) -> None:
    with deps.connect() as conn:
        _insert_l1_writer(conn, tool_call_id, tool_id)
