from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.persistent_turn_content.findings import (
    EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    ExecutionFindingEntry,
    ExecutionFindingKind,
    ExecutionFindingRevisionOperation,
    ExecutionFindingStatus,
    ExecutionFindingsLedgerStatus,
    ExecutionFindingsLedger,
    ExecutionFindingsMutationCommand,
    ExecutionFindingsOwnerKind,
    ExecutionFindingsQuota,
    RecordExecutionFinding,
    RetractExecutionFinding,
    SupersedeExecutionFinding,
    execution_findings_ledger_sha256,
    reduce_execution_findings_active_queue,
    validate_persisted_execution_finding_entry,
    validate_persisted_execution_findings_quota_json,
    canonical_json,
)


def test_finding_may_remain_an_unbound_model_candidate() -> None:
    finding = RecordExecutionFinding(
        kind=ExecutionFindingKind.FINDING,
        claim="This candidate has not yet been bound to durable evidence.",
        scope_keys=("summary",),
    )
    assert finding.source_refs == ()


@pytest.mark.parametrize("chunk_id", [None, "chunk-a"])
def test_finding_source_uses_only_native_result_and_optional_chunk_id(chunk_id):
    finding = RecordExecutionFinding.model_validate({
        "kind": "finding", "claim": "Read source",
        "source_refs": [{"tool_result_id": "result-a", "chunk_id": chunk_id}],
    })

    assert finding.source_refs[0].model_dump() == {
        "tool_result_id": "result-a", "chunk_id": chunk_id,
    }
    assert finding.scope_keys == ()


def test_optional_l1_result_pin_preserves_old_nested_wire_hash_input():
    original = {"operation": "record", "kind": "finding", "claim": "Read source",
                "source_refs": [{"tool_result_id": "result-a", "chunk_id": None}], "scope_keys": []}
    finding = RecordExecutionFinding.model_validate(original)
    assert canonical_json(finding) == canonical_json(original)
    pinned = RecordExecutionFinding.model_validate({
        **original, "source_refs": [{"tool_result_id": "call-a", "result_sha256": "a" * 64}],
    })
    assert pinned.source_refs[0].result_sha256 == "a" * 64
    assert pinned.model_dump(mode="json")["source_refs"][0]["result_sha256"] == "a" * 64


@pytest.mark.parametrize("source", [
    {}, {"tool_result_id": None},
    {"ref_type": "tool_result", "tool_call_id": "call-a", "result_sha256": "a" * 64},
    {"tool_result_id": "result-a", "document_alias": "alias-a"},
])
def test_finding_source_rejects_missing_identity_and_retired_reference_fields(source):
    with pytest.raises(ValidationError):
        RecordExecutionFinding.model_validate({
            "kind": "finding", "claim": "Read source", "source_refs": [source],
        })


def test_current_findings_quota_has_one_claim_limit_and_64_active_entries() -> None:
    quota = ExecutionFindingsQuota()

    assert EXECUTION_FINDING_CLAIM_MAX_CHARACTERS == 4_096
    assert quota.max_claim_characters == EXECUTION_FINDING_CLAIM_MAX_CHARACTERS
    assert quota.max_active_entries == 64
    with pytest.raises(ValidationError, match="less than or equal to 4096"):
        ExecutionFindingsQuota(max_claim_characters=4_097)


def test_default_active_queue_keeps_exactly_the_newest_64_entries() -> None:
    entries = tuple(
        ExecutionFindingEntry(
            ledger_id="ledger-1",
            entry_id=f"entry-{sequence}",
            entry_revision_id=f"entry-revision-{sequence}",
            entry_revision=1,
            sequence=sequence,
            operation=ExecutionFindingRevisionOperation.RECORD,
            kind=ExecutionFindingKind.FINDING,
            claim=f"claim {sequence}",
            scope_keys=("summary",),
            status=ExecutionFindingStatus.ACTIVE,
            writer_unit_id=f"attempt-{sequence}",
            writer_tool_call_id=f"call-{sequence}",
            mutation_id=f"mutation-{sequence}",
            created_at="2026-09-04T00:00:00+00:00",
        )
        for sequence in range(1, 66)
    )

    active = reduce_execution_findings_active_queue(
        entries,
        capacity=ExecutionFindingsQuota().max_active_entries,
    )

    assert len(active) == 64
    assert active[0].entry_id == "entry-2"
    assert active[-1].entry_id == "entry-65"


@pytest.mark.parametrize("contract", ("record", "supersede"))
def test_current_finding_claim_limit_accepts_4096_and_rejects_4097(
    contract: str,
) -> None:
    def build(claim: str) -> object:
        if contract == "record":
            return RecordExecutionFinding(
                kind=ExecutionFindingKind.FINDING,
                claim=claim,
                scope_keys=("summary",),
            )
        if contract == "supersede":
            return SupersedeExecutionFinding(
                entry_id="entry-1",
                kind=ExecutionFindingKind.FINDING,
                claim=claim,
                scope_keys=("summary",),
            )
    admitted = build("x" * EXECUTION_FINDING_CLAIM_MAX_CHARACTERS)
    assert getattr(admitted, "claim") == (
        "x" * EXECUTION_FINDING_CLAIM_MAX_CHARACTERS
    )

    with pytest.raises(ValidationError, match="String should have at most 4096"):
        build("x" * (EXECUTION_FINDING_CLAIM_MAX_CHARACTERS + 1))


@pytest.mark.parametrize("claim_length", [1_500, 4_096])
def test_current_and_persisted_findings_share_limit_and_respect_frozen_quota(
    claim_length: int,
) -> None:
    payload = {
        "ledger_id": "ledger-1",
        "entry_id": "entry-1",
        "entry_revision_id": "entry-revision-1",
        "entry_revision": 1,
        "sequence": 1,
        "operation": ExecutionFindingRevisionOperation.RECORD,
        "kind": ExecutionFindingKind.FINDING,
        "claim": "x" * claim_length,
        "scope_keys": ("summary",),
        "status": ExecutionFindingStatus.ACTIVE,
        "writer_unit_id": "attempt-1",
        "writer_tool_call_id": "call-1",
        "mutation_id": "mutation-1",
        "created_at": "2026-09-04T00:00:00+00:00",
    }
    restored = validate_persisted_execution_finding_entry(payload)
    assert restored == ExecutionFindingEntry.model_validate(payload)
    assert len(restored.claim) == claim_length
    for validate in (
        ExecutionFindingEntry.model_validate, validate_persisted_execution_finding_entry,
    ):
        with pytest.raises(ValidationError, match="String should have at most 4096"):
            validate({**payload, "claim": "x" * 4_097})

    ledger_payload = {
        "ledger_id": "ledger-1",
        "owner_kind": ExecutionFindingsOwnerKind.L1_TURN_RUN,
        "execution_owner_id": "run-1",
        "session_id": "session-1",
        "originating_turn_id": "turn-1",
        "status": ExecutionFindingsLedgerStatus.OPEN,
        "revision": 1,
        "quota": ExecutionFindingsQuota(max_claim_characters=1_024),
        "entry_revisions": (restored,),
        "mutation_count": 1,
        "created_at": "2026-09-04T00:00:00+00:00",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "closed_at": None,
        "ledger_sha256": "0" * 64,
    }
    provisional = ExecutionFindingsLedger.model_construct(**ledger_payload)
    ledger_payload["ledger_sha256"] = execution_findings_ledger_sha256(
        provisional
    )
    with pytest.raises(ValidationError, match="configured claim quota"):
        ExecutionFindingsLedger.model_validate(ledger_payload)

    frozen_quota_json = ExecutionFindingsQuota(
        max_claim_characters=1_500
    ).model_dump_json()
    assert ExecutionFindingsQuota.model_validate_json(frozen_quota_json).max_claim_characters == 1_500
    assert (
        validate_persisted_execution_findings_quota_json(
            frozen_quota_json
        ).max_claim_characters
        == 1_500
    )


def test_one_mutation_cannot_revise_the_same_entry_twice() -> None:
    with pytest.raises(ValidationError, match="same entry twice"):
        ExecutionFindingsMutationCommand(
            ledger_id="ledger-1",
            mutation_id="mutation-1",
            expected_ledger_revision=1,
            writer_unit_id="step-1",
            writer_tool_call_id="call-1",
            items=(
                RetractExecutionFinding(
                    entry_id="entry-1",
                    reason="The source was misread.",
                ),
                RetractExecutionFinding(
                    entry_id="entry-1",
                    reason="The same target must not appear twice.",
                ),
            ),
        )


def test_ledger_hash_covers_owner_quota_and_revision() -> None:
    payload = {
        "ledger_id": "ledger-1",
        "owner_kind": ExecutionFindingsOwnerKind.L1_TURN_RUN,
        "execution_owner_id": "run-1",
        "session_id": "session-1",
        "originating_turn_id": "turn-1",
        "status": ExecutionFindingsLedgerStatus.OPEN,
        "revision": 0,
        "quota": ExecutionFindingsQuota(),
        "entry_revisions": (),
        "mutation_count": 0,
        "created_at": "2026-08-26T00:00:00+00:00",
        "updated_at": "2026-08-26T00:00:00+00:00",
        "closed_at": None,
        "ledger_sha256": "0" * 64,
    }
    provisional = ExecutionFindingsLedger.model_construct(**payload)
    payload["ledger_sha256"] = execution_findings_ledger_sha256(provisional)
    admitted = ExecutionFindingsLedger.model_validate(payload)
    assert admitted.revision == 0

    with pytest.raises(ValidationError, match="ledger hash"):
        ExecutionFindingsLedger.model_validate({**payload, "revision": 1})
