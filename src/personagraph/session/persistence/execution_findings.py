"""执行作用域模型发现项的持久 Host 权威源。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
import re
import sqlite3

from ...persistent_turn_content.findings import (
    ExecutionFindingEntry,
    ExecutionFindingSourceRef,
    ExecutionFindingRevisionOperation,
    ExecutionFindingStatus,
    ExecutionFindingsActiveProjection,
    ExecutionFindingsLedgerCreateResult,
    ExecutionFindingsLedgerStatus,
    ExecutionFindingsLedger,
    ExecutionFindingsMutationCommand,
    ExecutionFindingsMutationIdentityCollision,
    ExecutionFindingsMutationReceipt,
    ExecutionFindingsMutationResult,
    ExecutionFindingsOwnerClosed,
    ExecutionFindingsOwnerKind,
    ExecutionFindingsPersistenceError,
    ExecutionFindingsQuota,
    ExecutionFindingsQuotaExceeded,
    ExecutionFindingsRevisionConflict,
    ExecutionFindingsScopeInvalid,
    ExecutionFindingsSnapshot,
    ExecutionFindingsSourceReferenceInvalid,
    ExecutionFindingsStoredAuthorityCorrupt,
    RecordExecutionFinding,
    RetractExecutionFinding,
    SupersedeExecutionFinding,
    canonical_json,
    derive_execution_finding_entry_id,
    derive_execution_findings_ledger_id,
    derive_execution_findings_mutation_id,
    execution_findings_ledger_sha256,
    execution_findings_projection_sha256,
    l1_execution_note_writer_id,
    reduce_execution_findings_active_queue,
    sha256_json,
    validate_persisted_execution_finding_entry,
    validate_persisted_execution_findings_mutation_result,
    validate_persisted_execution_findings_mutation_result_json,
    validate_persisted_execution_findings_quota_json,
)
from ...persistent_turn_content.evidence import l1_tool_result_id
from ...persistent_turn_content.tool_results import (
    ToolHistoryError,
    select_tool_result_chunk,
)
from ...output_protocol.l1 import (
    L1AttemptDecisionProposal,
)
from ...tools.findings.contracts import (
    EXECUTION_FINDINGS_TOOL_IDS,
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
    REVISE_EXECUTION_FINDING_TOOL_ID,
)
from .deps import StoreDeps


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_TERMINAL_L1_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_WORK_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


def create_execution_findings_ledger(
    deps: StoreDeps,
    *,
    session_id: str,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
    quota: ExecutionFindingsQuota | None = None,
) -> ExecutionFindingsLedgerCreateResult:
    """创建或重放由 L1 TurnRun 或 L2 WorkRun 所有的唯一账本。"""

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return create_execution_findings_ledger_in_transaction(
            conn,
            session_id=session_id,
            owner_kind=owner_kind,
            execution_owner_id=execution_owner_id,
            quota=quota,
            now=now,
        )


def create_execution_findings_ledger_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
    quota: ExecutionFindingsQuota | None = None,
    now: str,
) -> ExecutionFindingsLedgerCreateResult:
    """在执行所有者的打开事务内创建或重放账本。

    所有者持久化在插入 L1 TurnRun 或 L2 WorkRun 行后立即调用此函数。因此，之后任何
    所有者初始化失败都会让伴随账本与所有者一同回滚，而不会留下分裂提交。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("execution_owner_id", execution_owner_id)
    if not isinstance(now, str) or not now:
        raise ValueError("now must be a non-empty timestamp")
    parsed_owner_kind = ExecutionFindingsOwnerKind(owner_kind)
    ledger_id = derive_execution_findings_ledger_id(
        owner_kind=parsed_owner_kind,
        execution_owner_id=execution_owner_id,
    )
    existing = conn.execute(
        "SELECT * FROM execution_findings_ledgers WHERE ledger_id=? OR "
        "(owner_kind=? AND execution_owner_id=?)",
        (ledger_id, parsed_owner_kind.value, execution_owner_id),
    ).fetchall()
    if existing:
        if len(existing) != 1:
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "findings owner resolves to more than one ledger"
            )
        row = existing[0]
        # Quota is frozen with the execution owner.  A caller that omits it while
        # replaying must inherit the persisted value rather than reinterpret the
        # owner through whatever defaults this process happens to have now.
        admitted_quota = _load_quota(row) if quota is None else quota
        quota_json = canonical_json(admitted_quota)
        quota_hash = sha256_json(admitted_quota)
        expected = {
            "ledger_id": ledger_id,
            "session_id": session_id,
            "owner_kind": parsed_owner_kind.value,
            "execution_owner_id": execution_owner_id,
            "quota_json": quota_json,
            "quota_hash": quota_hash,
        }
        if any(str(row[key]) != value for key, value in expected.items()):
            raise ExecutionFindingsPersistenceError(
                "findings ledger replay facts changed"
            )
        snapshot = _build_snapshot(conn, row=row)
        return ExecutionFindingsLedgerCreateResult(
            ledger=snapshot.ledger,
            active_projection=snapshot.active_projection,
            replayed=True,
        )

    if quota is None:
        admitted_quota = ExecutionFindingsQuota()
    else:
        # ``model_copy(update=...)`` intentionally skips Pydantic validation.
        # Re-admit explicit facts for a new ledger so copied values cannot bypass
        # the shared findings quota contract.
        admitted_quota = ExecutionFindingsQuota.model_validate(
            quota.model_dump(mode="json")
        )
    quota_json = canonical_json(admitted_quota)
    quota_hash = sha256_json(admitted_quota)
    originating_turn_id = _require_create_owner(
        conn,
        session_id=session_id,
        owner_kind=parsed_owner_kind,
        execution_owner_id=execution_owner_id,
    )
    l1_turn_run_id = (
        execution_owner_id
        if parsed_owner_kind is ExecutionFindingsOwnerKind.L1_TURN_RUN
        else None
    )
    work_run_id = (
        execution_owner_id
        if parsed_owner_kind is ExecutionFindingsOwnerKind.WORK_RUN
        else None
    )
    try:
        conn.execute(
            "INSERT INTO execution_findings_ledgers "
            "(ledger_id, session_id, originating_turn_id, owner_kind, "
            "execution_owner_id, l1_turn_run_id, work_run_id, status, "
            "revision, quota_json, quota_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, ?, ?, ?, ?)",
            (
                ledger_id,
                session_id,
                originating_turn_id,
                parsed_owner_kind.value,
                execution_owner_id,
                l1_turn_run_id,
                work_run_id,
                quota_json,
                quota_hash,
                now,
                now,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ExecutionFindingsPersistenceError(
            "findings ledger conflicts with its execution owner"
        ) from exc
    row = _require_ledger_row(conn, ledger_id)
    snapshot = _build_snapshot(conn, row=row)
    return ExecutionFindingsLedgerCreateResult(
        ledger=snapshot.ledger,
        active_projection=snapshot.active_projection,
        replayed=False,
    )


def create_execution_findings_owner_companion_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
    quota: ExecutionFindingsQuota | None = None,
    now: str,
) -> ExecutionFindingsLedgerCreateResult:
    """创建所有者伴随记录；其表缺失时以关闭方式失败。"""

    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='execution_findings_ledgers'"
    ).fetchone()
    if table_exists is None:
        raise ExecutionFindingsPersistenceError(
            "execution findings schema is missing from the session database"
        )
    return create_execution_findings_ledger_in_transaction(
        conn,
        session_id=session_id,
        owner_kind=owner_kind,
        execution_owner_id=execution_owner_id,
        quota=quota,
        now=now,
    )


def get_execution_findings_ledger(
    deps: StoreDeps,
    *,
    ledger_id: str,
) -> ExecutionFindingsSnapshot | None:
    """加载完整审计历史和一个有界 FIFO 活动投影。"""

    _require_identifier("ledger_id", ledger_id)
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT * FROM execution_findings_ledgers WHERE ledger_id=?",
            (ledger_id,),
        ).fetchone()
        if row is None:
            return None
        return _build_snapshot(conn, row=row)


def get_execution_findings_ledger_for_owner(
    deps: StoreDeps,
    *,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
) -> ExecutionFindingsSnapshot | None:
    """根据执行所有者解析稳定账本身份。"""

    parsed_owner_kind = ExecutionFindingsOwnerKind(owner_kind)
    _require_identifier("execution_owner_id", execution_owner_id)
    ledger_id = derive_execution_findings_ledger_id(
        owner_kind=parsed_owner_kind,
        execution_owner_id=execution_owner_id,
    )
    return get_execution_findings_ledger(
        deps,
        ledger_id=ledger_id,
    )


def apply_execution_findings_mutation(
    deps: StoreDeps,
    *,
    command: ExecutionFindingsMutationCommand,
) -> ExecutionFindingsMutationResult:
    """以 CAS 应用一个内部有副作用且幂等的发现项变更。"""

    expected_mutation_id = derive_execution_findings_mutation_id(
        writer_tool_call_id=command.writer_tool_call_id
    )
    if command.mutation_id != expected_mutation_id:
        raise ExecutionFindingsMutationIdentityCollision(
            "mutation_id is not derived from the durable writer ToolCall"
        )
    request_json = canonical_json(command)
    request_hash = sha256_json(command)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        ledger_row = _require_ledger_row(conn, command.ledger_id)
        replay = conn.execute(
            "SELECT request_hash, result_json, result_hash "
            "FROM execution_findings_mutations "
            "WHERE ledger_id=? AND mutation_id=?",
            (command.ledger_id, command.mutation_id),
        ).fetchone()
        if replay is not None:
            if str(replay["request_hash"]) != request_hash:
                raise ExecutionFindingsMutationIdentityCollision(
                    "findings mutation replay facts changed"
                )
            result_json = str(replay["result_json"])
            if (
                hashlib.sha256(result_json.encode("utf-8")).hexdigest()
                != str(replay["result_hash"])
            ):
                raise ExecutionFindingsStoredAuthorityCorrupt(
                    "stored findings mutation receipt hash changed"
                )
            try:
                stored = validate_persisted_execution_findings_mutation_result_json(
                    result_json
                )
            except ValueError as exc:
                raise ExecutionFindingsStoredAuthorityCorrupt(
                    "stored findings mutation result is invalid"
                ) from exc
            return validate_persisted_execution_findings_mutation_result(
                {**stored.model_dump(mode="json"), "replayed": True}
            )

        if str(ledger_row["status"]) != ExecutionFindingsLedgerStatus.OPEN:
            raise ExecutionFindingsOwnerClosed("findings ledger is closed")
        actual_revision = int(ledger_row["revision"])
        if command.expected_ledger_revision != actual_revision:
            raise ExecutionFindingsRevisionConflict(
                expected=command.expected_ledger_revision,
                actual=actual_revision,
            )
        quota = _load_quota(ledger_row)
        _validate_mutation_quota(command, quota=quota)
        tool_id = _require_writer_authority(
            conn,
            ledger_row=ledger_row,
            writer_unit_id=command.writer_unit_id,
            writer_tool_call_id=command.writer_tool_call_id,
            command=command,
        )
        _validate_writer_operation(tool_id=tool_id, command=command)
        authoritative_scope_keys = _load_authoritative_scope_keys(
            conn,
            ledger_row=ledger_row,
        )
        requested_scope_keys = {
            key
            for item in command.items
            if not isinstance(item, RetractExecutionFinding)
            for key in item.scope_keys
        }
        if not requested_scope_keys <= authoritative_scope_keys:
            missing = sorted(requested_scope_keys - authoritative_scope_keys)
            raise ExecutionFindingsScopeInvalid(
                f"finding scope is not current owner authority: {missing}"
            )
        for item in command.items:
            if isinstance(item, RetractExecutionFinding):
                continue
            for source_ref in item.source_refs:
                _validate_source_ref(
                    conn,
                    ledger_row=ledger_row,
                    source_ref=source_ref,
                )

        existing_entries = _load_entry_revisions(conn, command.ledger_id)
        if len(existing_entries) + len(command.items) > (
            quota.max_durable_entry_revisions
        ):
            raise ExecutionFindingsQuotaExceeded(
                "durable findings revision quota is exhausted"
            )
        latest_by_entry = _latest_entries(existing_entries)
        next_sequence = (
            max((entry.sequence for entry in existing_entries), default=0) + 1
        )
        affected_entry_ids: list[str] = []
        for ordinal, item in enumerate(command.items, start=1):
            if isinstance(item, RecordExecutionFinding):
                entry_id = derive_execution_finding_entry_id(
                    ledger_id=command.ledger_id,
                    mutation_id=command.mutation_id,
                    item_ordinal=ordinal,
                )
                entry_revision = 1
                predecessor = None
                operation = ExecutionFindingRevisionOperation.RECORD
                kind = item.kind
                claim = item.claim
                source_refs = item.source_refs
                scope_keys = item.scope_keys
                status = ExecutionFindingStatus.ACTIVE
                reason = None
            else:
                current = latest_by_entry.get(item.entry_id)
                if current is None or current.status is not ExecutionFindingStatus.ACTIVE:
                    raise ExecutionFindingsPersistenceError(
                        "findings revision target is missing or no longer active"
                    )
                entry_id = current.entry_id
                entry_revision = current.entry_revision + 1
                predecessor = current.entry_revision_id
                if isinstance(item, SupersedeExecutionFinding):
                    operation = ExecutionFindingRevisionOperation.SUPERSEDE
                    kind = item.kind
                    claim = item.claim
                    source_refs = item.source_refs
                    scope_keys = item.scope_keys
                    status = ExecutionFindingStatus.ACTIVE
                    reason = None
                else:
                    operation = ExecutionFindingRevisionOperation.RETRACT
                    kind = current.kind
                    claim = current.claim
                    source_refs = current.source_refs
                    scope_keys = current.scope_keys
                    status = ExecutionFindingStatus.RETRACTED
                    reason = item.reason
            entry_revision_id = _stable_id(
                "efrevision",
                {
                    "ledger_id": command.ledger_id,
                    "entry_id": entry_id,
                    "entry_revision": entry_revision,
                    "mutation_id": command.mutation_id,
                },
            )
            entry_payload = {
                "ledger_id": command.ledger_id,
                "entry_id": entry_id,
                "entry_revision_id": entry_revision_id,
                "entry_revision": entry_revision,
                "sequence": next_sequence,
                "operation": operation,
                "kind": kind,
                "claim": claim,
                "source_refs": source_refs,
                "scope_keys": scope_keys,
                "status": status,
                "writer_unit_id": command.writer_unit_id,
                "writer_tool_call_id": command.writer_tool_call_id,
                "mutation_id": command.mutation_id,
                "supersedes_entry_revision_id": predecessor,
                "revision_reason": reason,
                "created_at": now,
            }
            entry = ExecutionFindingEntry.model_validate(entry_payload)
            _insert_entry_revision(conn, entry)
            latest_by_entry[entry_id] = entry
            affected_entry_ids.append(entry_id)
            next_sequence += 1

        updated_entries = _load_entry_revisions(conn, command.ledger_id)
        durable_bytes = _entry_revisions_utf8_bytes(updated_entries)
        if durable_bytes > quota.max_durable_utf8_bytes:
            raise ExecutionFindingsQuotaExceeded(
                "durable findings byte quota is exhausted"
            )
        applied_revision = actual_revision + 1
        if conn.execute(
            "UPDATE execution_findings_ledgers SET revision=?, updated_at=? "
            "WHERE ledger_id=? AND status='open' AND revision=?",
            (applied_revision, now, command.ledger_id, actual_revision),
        ).rowcount != 1:
            raise ExecutionFindingsRevisionConflict(
                expected=actual_revision,
                actual=int(_require_ledger_row(conn, command.ledger_id)["revision"]),
            )
        updated_row = _require_ledger_row(conn, command.ledger_id)
        prior_mutation_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM execution_findings_mutations "
                "WHERE ledger_id=?",
                (command.ledger_id,),
            ).fetchone()[0]
        )
        snapshot = _build_snapshot(
            conn,
            row=updated_row,
            mutation_count_override=prior_mutation_count + 1,
        )
        receipt = ExecutionFindingsMutationReceipt(
            ledger_id=command.ledger_id,
            mutation_id=command.mutation_id,
            request_sha256=request_hash,
            previous_ledger_revision=actual_revision,
            applied_ledger_revision=applied_revision,
            affected_entry_ids=tuple(affected_entry_ids),
            active_projection_sha256=(
                snapshot.active_projection.projection_sha256
            ),
            created_at=now,
        )
        result = ExecutionFindingsMutationResult(
            receipt=receipt,
            ledger=snapshot.ledger,
            active_projection=snapshot.active_projection,
            replayed=False,
        )
        result_json = canonical_json(result)
        try:
            conn.execute(
                "INSERT INTO execution_findings_mutations "
                "(ledger_id, mutation_id, request_json, request_hash, "
                "expected_revision, applied_revision, result_json, result_hash, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    command.ledger_id,
                    command.mutation_id,
                    request_json,
                    request_hash,
                    actual_revision,
                    applied_revision,
                    result_json,
                    hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ExecutionFindingsMutationIdentityCollision(
                "findings mutation identity conflicts with stored authority"
            ) from exc
        return result


def close_execution_findings_ledger(
    deps: StoreDeps,
    *,
    ledger_id: str,
    expected_ledger_revision: int,
) -> ExecutionFindingsSnapshot:
    """在所有者结算时冻结账本，不删除其历史。"""

    _require_identifier("ledger_id", ledger_id)
    if expected_ledger_revision < 0:
        raise ValueError("expected_ledger_revision must be non-negative")
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _require_ledger_row(conn, ledger_id)
        actual_revision = int(row["revision"])
        if actual_revision != expected_ledger_revision:
            raise ExecutionFindingsRevisionConflict(
                expected=expected_ledger_revision,
                actual=actual_revision,
            )
        if str(row["status"]) == ExecutionFindingsLedgerStatus.CLOSED:
            return _build_snapshot(conn, row=row)
        if conn.execute(
            "UPDATE execution_findings_ledgers "
            "SET status='closed', closed_at=?, updated_at=? "
            "WHERE ledger_id=? AND status='open' AND revision=?",
            (now, now, ledger_id, expected_ledger_revision),
        ).rowcount != 1:
            raise ExecutionFindingsPersistenceError(
                "findings ledger changed during close"
            )
        return _build_snapshot(conn, row=_require_ledger_row(conn, ledger_id))


def _require_create_owner(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    owner_kind: ExecutionFindingsOwnerKind,
    execution_owner_id: str,
) -> str:
    if owner_kind is ExecutionFindingsOwnerKind.L1_TURN_RUN:
        row = conn.execute(
            "SELECT session_id, turn_id, status FROM l1_turn_runs "
            "WHERE l1_turn_run_id=?",
            (execution_owner_id,),
        ).fetchone()
        if row is None or str(row["session_id"]) != session_id:
            raise ExecutionFindingsPersistenceError(
                "findings ledger has no owning L1 TurnRun"
            )
        if str(row["status"]) in _TERMINAL_L1_STATUSES:
            raise ExecutionFindingsOwnerClosed("owning L1 TurnRun is terminal")
        return str(row["turn_id"])

    row = conn.execute(
        "SELECT session_id, created_turn_id, status FROM insession_work_runs "
        "WHERE work_run_id=?",
        (execution_owner_id,),
    ).fetchone()
    if row is None or str(row["session_id"]) != session_id:
        raise ExecutionFindingsPersistenceError(
            "findings ledger has no owning WorkRun"
        )
    if str(row["status"]) in _TERMINAL_WORK_RUN_STATUSES:
        raise ExecutionFindingsOwnerClosed("owning WorkRun is terminal")
    return str(row["created_turn_id"])


def _require_writer_authority(
    conn: sqlite3.Connection,
    *,
    ledger_row: sqlite3.Row,
    writer_unit_id: str,
    writer_tool_call_id: str,
    command: ExecutionFindingsMutationCommand,
) -> str:
    owner_kind = ExecutionFindingsOwnerKind(str(ledger_row["owner_kind"]))
    owner_id = str(ledger_row["execution_owner_id"])
    if owner_kind is ExecutionFindingsOwnerKind.L1_TURN_RUN:
        if writer_tool_call_id == l1_execution_note_writer_id(writer_unit_id):
            _require_committed_l1_note_authority(
                conn, owner_id=owner_id, command=command,
            )
            return RECORD_EXECUTION_FINDINGS_TOOL_ID
        row = conn.execute(
            "SELECT call.step_id, call.tool_id, call.execution_class, "
            "call.status AS call_status, "
            "run.status AS run_status, state.stage AS run_stage "
            "FROM l1_turn_tool_calls AS call "
            "JOIN l1_turn_runs AS run "
            "ON run.l1_turn_run_id=call.l1_turn_run_id "
            "JOIN l1_turn_run_states AS state "
            "ON state.l1_turn_run_id=call.l1_turn_run_id "
            "WHERE call.tool_call_id=? AND call.l1_turn_run_id=?",
            (writer_tool_call_id, owner_id),
        ).fetchone()
        if (
            row is None
            or str(row["step_id"]) != writer_unit_id
            or str(row["execution_class"]) != "runtime_state"
            or str(row["call_status"]) != "pending"
            or str(row["run_status"]) != "running"
            or str(row["run_stage"]) != "tool"
        ):
            raise ExecutionFindingsPersistenceError(
                "findings mutation has no current L1 ToolCall authority"
            )
        tool_id = str(row["tool_id"])
    else:
        row = conn.execute(
            "SELECT call.attempt_id, call.tool_id, attempt.status AS attempt_status, "
            "run.status AS run_status, run.current_attempt_id "
            "FROM insession_work_run_tool_calls AS call "
            "JOIN insession_work_run_attempts AS attempt "
            "ON attempt.work_run_id=call.work_run_id "
            "AND attempt.attempt_id=call.attempt_id "
            "JOIN insession_work_runs AS run "
            "ON run.work_run_id=call.work_run_id "
            "WHERE call.tool_call_id=? AND call.work_run_id=?",
            (writer_tool_call_id, owner_id),
        ).fetchone()
        if (
            row is None
            or str(row["attempt_id"]) != writer_unit_id
            or str(row["attempt_status"]) != "active"
            or str(row["run_status"]) != "active"
            or str(row["current_attempt_id"] or "") != writer_unit_id
        ):
            raise ExecutionFindingsPersistenceError(
                "findings mutation has no current WorkRun ToolCall authority"
            )
        tool_id = str(row["tool_id"])
    if tool_id not in EXECUTION_FINDINGS_TOOL_IDS:
        raise ExecutionFindingsPersistenceError(
            "writer ToolCall is not an execution-findings tool"
        )
    return tool_id


def _require_committed_l1_note_authority(
    conn: sqlite3.Connection, *, owner_id: str, command: ExecutionFindingsMutationCommand,
) -> None:
    """固定 Host 调用只能逐字保存本轮最新已提交决定中的笔记。"""
    row = conn.execute(
        "SELECT step.request_json, step.request_hash, step.decision_json, step.decision_hash "
        "FROM l1_turn_steps AS step JOIN l1_turn_runs AS run "
        "ON run.l1_turn_run_id=step.l1_turn_run_id "
        "WHERE step.step_id=? AND step.l1_turn_run_id=? AND run.status='running' "
        "AND step.status IN ('decided','observed','final_answer') "
        "AND step.ordinal=(SELECT MAX(ordinal) FROM l1_turn_steps WHERE l1_turn_run_id=?)",
        (command.writer_unit_id, owner_id, owner_id),
    ).fetchone()
    if row is None:
        raise ExecutionFindingsPersistenceError("Host notes require the current committed L1 decision")
    for field in ("request", "decision"):
        raw = row[f"{field}_json"]
        if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row[f"{field}_hash"]:
            raise ExecutionFindingsStoredAuthorityCorrupt("L1 note authority hash mismatch")
    try:
        request = json.loads(row["request_json"])
        decision = json.loads(row["decision_json"])
        if request.get("execution_notes_required") is not True:
            raise ValueError("the frozen request does not authorize fixed notes")
        proposal = L1AttemptDecisionProposal.model_validate(decision)
        expected = (RecordExecutionFinding(kind="decision", claim=proposal.note),)
        revision = request["execution_findings"]["ledger_revision"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionFindingsStoredAuthorityCorrupt("L1 note authority is malformed") from exc
    if command.items != expected or command.expected_ledger_revision != revision:
        raise ExecutionFindingsPersistenceError("Host note content differs from the committed decision")


def _validate_writer_operation(
    *,
    tool_id: str,
    command: ExecutionFindingsMutationCommand,
) -> None:
    if tool_id == RECORD_EXECUTION_FINDINGS_TOOL_ID:
        if any(not isinstance(item, RecordExecutionFinding) for item in command.items):
            raise ExecutionFindingsPersistenceError(
                "record_execution_findings may only append new entries"
            )
        return
    if tool_id == REVISE_EXECUTION_FINDING_TOOL_ID and any(
        isinstance(item, RecordExecutionFinding) for item in command.items
    ):
        raise ExecutionFindingsPersistenceError(
            "revise_execution_finding may only supersede or retract entries"
        )


def _load_authoritative_scope_keys(
    conn: sqlite3.Connection,
    *,
    ledger_row: sqlite3.Row,
) -> set[str]:
    owner_kind = ExecutionFindingsOwnerKind(str(ledger_row["owner_kind"]))
    owner_id = str(ledger_row["execution_owner_id"])
    if owner_kind is ExecutionFindingsOwnerKind.L1_TURN_RUN:
        row = conn.execute(
            "SELECT plan_json FROM l1_turn_run_states WHERE l1_turn_run_id=?",
            (owner_id,),
        ).fetchone()
        if row is None or row["plan_json"] is None:
            raise ExecutionFindingsScopeInvalid(
                "L1 findings require an accepted current L1 Plan"
            )
        payload = _canonical_stored_json(
            str(row["plan_json"]),
            label="L1 Plan",
        )
        acceptances = (
            payload.get("acceptances") if isinstance(payload, dict) else None
        )
        if not isinstance(acceptances, list):
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "stored L1 Plan has no Acceptances"
            )
        values = {
            str(item.get("acceptance_id"))
            for item in acceptances
            if isinstance(item, dict)
            and isinstance(item.get("acceptance_id"), str)
        }
    else:
        row = conn.execute(
            "SELECT snapshot_json FROM insession_work_run_acceptance_progress "
            "WHERE work_run_id=?",
            (owner_id,),
        ).fetchone()
        if row is None:
            raise ExecutionFindingsScopeInvalid(
                "WorkRun findings require Acceptance progress"
            )
        payload = _canonical_stored_json(
            str(row["snapshot_json"]),
            label="WorkRun Acceptance progress",
        )
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "stored Acceptance progress has no items"
            )
        values = {
            str(item.get("acceptance_id"))
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("acceptance_id"), str)
        }
    if not values:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "execution owner has no authoritative scope keys"
        )
    return values


def _validate_source_ref(
    conn: sqlite3.Connection,
    *,
    ledger_row: sqlite3.Row,
    source_ref: ExecutionFindingSourceRef,
) -> None:
    owner_kind = ExecutionFindingsOwnerKind(str(ledger_row["owner_kind"]))
    owner_id = str(ledger_row["execution_owner_id"])
    if owner_kind is ExecutionFindingsOwnerKind.L1_TURN_RUN:
        # 原生结果 ID 从已存调用和 outcome hash 派生，不新增别名或第二份结果表。
        identities = conn.execute(
            "SELECT tool_call_id, outcome_hash FROM l1_turn_tool_calls "
            "WHERE l1_turn_run_id=? AND status='succeeded' AND outcome_hash IS NOT NULL",
            (owner_id,),
        ).fetchall()
        try:
            tool_call_id = next((
                str(identity["tool_call_id"]) for identity in identities
                if l1_tool_result_id(
                    tool_call_id=str(identity["tool_call_id"]),
                    result_sha256=str(identity["outcome_hash"]),
                ) == source_ref.tool_result_id
            ), None)
        except ValueError as exc:
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "stored L1 ToolResult identity is invalid"
            ) from exc
        row = conn.execute(
            "SELECT tool_id, status, outcome_json, outcome_hash "
            "FROM l1_turn_tool_calls "
            "WHERE l1_turn_run_id=? AND tool_call_id=?",
            (owner_id, tool_call_id),
        ).fetchone()
        if row is None or str(row["status"]) != "succeeded":
            raise ExecutionFindingsSourceReferenceInvalid(
                "L1 source ToolCall is absent or not successful"
            )
        from ...tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS

        if str(row["tool_id"]) in TOOL_HISTORY_TOOL_IDS:
            raise ExecutionFindingsSourceReferenceInvalid(
                "a tool-history receipt cannot support a finding; cite its original source"
            )
        if str(row["tool_id"]) in EXECUTION_FINDINGS_TOOL_IDS:
            raise ExecutionFindingsSourceReferenceInvalid(
                "an execution-findings result cannot support another finding"
            )
        result_json = str(row["outcome_json"] or "")
        result_hash = str(row["outcome_hash"] or "")
        payload = _canonical_stored_json(result_json, label="L1 ToolCall outcome")
        if hashlib.sha256(result_json.encode("utf-8")).hexdigest() != result_hash:
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "stored L1 ToolCall outcome hash changed"
            )
        output = payload.get("result") if isinstance(payload, dict) else None
    else:
        row = conn.execute(
            "SELECT result.tool_result_id, result.status, result.result_json, "
            "call.tool_id "
            "FROM insession_work_run_tool_results AS result "
            "JOIN insession_work_run_tool_calls AS call "
            "ON call.work_run_id=result.work_run_id "
            "AND call.tool_call_id=result.tool_call_id "
            "WHERE result.work_run_id=? AND result.tool_result_id=?",
            (owner_id, source_ref.tool_result_id),
        ).fetchone()
        if row is None or str(row["status"]) != "succeeded":
            raise ExecutionFindingsSourceReferenceInvalid(
                "WorkRun source ToolResult is absent or not successful"
            )
        if str(row["tool_id"]) in EXECUTION_FINDINGS_TOOL_IDS:
            raise ExecutionFindingsSourceReferenceInvalid(
                "an execution-findings result cannot support another finding"
            )
        result_json = str(row["result_json"])
        payload = _canonical_stored_json(result_json, label="WorkRun ToolResult")
        output = payload.get("output") if isinstance(payload, dict) else None

    if not isinstance(payload, dict) or payload.get("status") != "succeeded":
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored source outcome disagrees with its successful status"
        )
    if source_ref.chunk_id is not None:
        try:
            select_tool_result_chunk(output, source_ref.chunk_id)
        except ToolHistoryError as exc:
            raise ExecutionFindingsSourceReferenceInvalid(str(exc)) from exc


def _validate_mutation_quota(
    command: ExecutionFindingsMutationCommand,
    *,
    quota: ExecutionFindingsQuota,
) -> None:
    if len(command.items) > quota.max_mutation_items:
        raise ExecutionFindingsQuotaExceeded(
            "one findings write contains too many mutation items"
        )
    for item in command.items:
        if isinstance(item, RetractExecutionFinding):
            continue
        if len(item.claim) > quota.max_claim_characters:
            raise ExecutionFindingsQuotaExceeded(
                "one findings claim exceeds the configured character quota"
            )
        if len(item.source_refs) > quota.max_source_refs_per_entry:
            raise ExecutionFindingsQuotaExceeded(
                "one finding contains too many source references"
            )
        if len(item.scope_keys) > quota.max_scope_keys_per_entry:
            raise ExecutionFindingsQuotaExceeded(
                "one finding contains too many scope keys"
            )


def _insert_entry_revision(
    conn: sqlite3.Connection,
    entry: ExecutionFindingEntry,
) -> None:
    entry_state = (
        "retracted"
        if entry.status is ExecutionFindingStatus.RETRACTED
        else "active"
    )
    try:
        conn.execute(
            "INSERT INTO execution_finding_entry_revisions "
            "(entry_revision_id, ledger_id, entry_id, entry_revision, sequence, "
            "operation, kind, claim, source_refs_json, scope_keys_json, "
            "entry_state, writer_unit_id, writer_tool_call_id, mutation_id, "
            "supersedes_entry_revision_id, revision_reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.entry_revision_id,
                entry.ledger_id,
                entry.entry_id,
                entry.entry_revision,
                entry.sequence,
                entry.operation.value,
                entry.kind.value,
                entry.claim,
                canonical_json(
                    [item.model_dump(mode="json") for item in entry.source_refs]
                ),
                canonical_json(list(entry.scope_keys)),
                entry_state,
                entry.writer_unit_id,
                entry.writer_tool_call_id,
                entry.mutation_id,
                entry.supersedes_entry_revision_id,
                entry.revision_reason,
                entry.created_at,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ExecutionFindingsPersistenceError(
            "finding revision conflicts with stored history"
        ) from exc


def _load_entry_revisions(
    conn: sqlite3.Connection,
    ledger_id: str,
) -> tuple[ExecutionFindingEntry, ...]:
    rows = conn.execute(
        "SELECT * FROM execution_finding_entry_revisions "
        "WHERE ledger_id=? ORDER BY sequence",
        (ledger_id,),
    ).fetchall()
    latest_revision_by_entry: dict[str, int] = {}
    for row in rows:
        entry_id = str(row["entry_id"])
        latest_revision_by_entry[entry_id] = max(
            latest_revision_by_entry.get(entry_id, 0),
            int(row["entry_revision"]),
        )
    entries: list[ExecutionFindingEntry] = []
    for row in rows:
        try:
            source_refs = json.loads(str(row["source_refs_json"]))
            scope_keys = json.loads(str(row["scope_keys_json"]))
            is_latest = int(row["entry_revision"]) == latest_revision_by_entry[
                str(row["entry_id"])
            ]
            if not is_latest:
                status = ExecutionFindingStatus.SUPERSEDED
            else:
                status = ExecutionFindingStatus(str(row["entry_state"]))
            entry = validate_persisted_execution_finding_entry(
                {
                    "ledger_id": str(row["ledger_id"]),
                    "entry_id": str(row["entry_id"]),
                    "entry_revision_id": str(row["entry_revision_id"]),
                    "entry_revision": int(row["entry_revision"]),
                    "sequence": int(row["sequence"]),
                    "operation": str(row["operation"]),
                    "kind": str(row["kind"]),
                    "claim": str(row["claim"]),
                    "source_refs": source_refs,
                    "scope_keys": scope_keys,
                    "status": status.value,
                    "writer_unit_id": str(row["writer_unit_id"]),
                    "writer_tool_call_id": str(row["writer_tool_call_id"]),
                    "mutation_id": str(row["mutation_id"]),
                    "supersedes_entry_revision_id": row[
                        "supersedes_entry_revision_id"
                    ],
                    "revision_reason": row["revision_reason"],
                    "created_at": str(row["created_at"]),
                }
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionFindingsStoredAuthorityCorrupt(
                "stored finding revision is invalid"
            ) from exc
        entries.append(entry)
    return tuple(entries)


def _latest_entries(
    entries: Iterable[ExecutionFindingEntry],
) -> dict[str, ExecutionFindingEntry]:
    latest: dict[str, ExecutionFindingEntry] = {}
    for entry in entries:
        current = latest.get(entry.entry_id)
        if current is None or entry.entry_revision > current.entry_revision:
            latest[entry.entry_id] = entry
    return latest


def _build_snapshot(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    mutation_count_override: int | None = None,
) -> ExecutionFindingsSnapshot:
    entries = _load_entry_revisions(conn, str(row["ledger_id"]))
    mutation_count = (
        mutation_count_override
        if mutation_count_override is not None
        else int(
            conn.execute(
                "SELECT COUNT(*) FROM execution_findings_mutations "
                "WHERE ledger_id=?",
                (str(row["ledger_id"]),),
            ).fetchone()[0]
        )
    )
    payload = {
        "ledger_id": str(row["ledger_id"]),
        "owner_kind": ExecutionFindingsOwnerKind(str(row["owner_kind"])),
        "execution_owner_id": str(row["execution_owner_id"]),
        "session_id": str(row["session_id"]),
        "originating_turn_id": str(row["originating_turn_id"]),
        "status": ExecutionFindingsLedgerStatus(str(row["status"])),
        "revision": int(row["revision"]),
        "quota": _load_quota(row),
        "entry_revisions": entries,
        "mutation_count": mutation_count,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "closed_at": row["closed_at"],
        "ledger_sha256": "0" * 64,
    }
    provisional_ledger = ExecutionFindingsLedger.model_construct(**payload)
    payload["ledger_sha256"] = execution_findings_ledger_sha256(
        provisional_ledger
    )
    try:
        ledger = ExecutionFindingsLedger.model_validate(payload)
    except ValueError as exc:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings ledger is invalid"
        ) from exc
    projection = _build_active_projection(ledger)
    return ExecutionFindingsSnapshot(
        ledger=ledger,
        active_projection=projection,
    )


def _build_active_projection(
    ledger: ExecutionFindingsLedger,
) -> ExecutionFindingsActiveProjection:
    mutation_ids = tuple(
        dict.fromkeys(entry.mutation_id for entry in ledger.entry_revisions)
    )
    if (
        len(mutation_ids) != ledger.revision
        or ledger.mutation_count != ledger.revision
    ):
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings mutation history disagrees with ledger revision"
        )
    all_active = tuple(
        entry
        for entry in _latest_entries(ledger.entry_revisions).values()
        if entry.status is ExecutionFindingStatus.ACTIVE
    )

    def projection_utf8_bytes(
        selected: tuple[ExecutionFindingEntry, ...],
        active: tuple[ExecutionFindingEntry, ...],
        prefix_revisions: tuple[ExecutionFindingEntry, ...],
        ledger_revision: int,
    ) -> int:
        selected_ids = {entry.entry_id for entry in selected}
        historical_ledger = ledger.model_copy(
            update={
                "revision": ledger_revision,
                "entry_revisions": prefix_revisions,
                "mutation_count": ledger_revision,
            }
        )
        projection = _materialize_projection(
            historical_ledger,
            selected=selected,
            omitted_entry_ids=tuple(
                entry.entry_id
                for entry in active
                if entry.entry_id not in selected_ids
            ),
        )
        return len(canonical_json(projection).encode("utf-8"))

    try:
        queued = reduce_execution_findings_active_queue(
            ledger.entry_revisions,
            capacity=ledger.quota.max_active_entries,
            max_projection_utf8_bytes=(
                ledger.quota.max_active_projection_utf8_bytes
            ),
            projection_utf8_bytes=projection_utf8_bytes,
        )
    except ValueError as exc:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings queue history is invalid"
        ) from exc
    selected_ids = {entry.entry_id for entry in queued}
    omitted_ids = [
        entry.entry_id
        for entry in all_active
        if entry.entry_id not in selected_ids
    ]
    projection = _materialize_projection(
        ledger,
        selected=queued,
        omitted_entry_ids=omitted_ids,
    )
    if (
        len(canonical_json(projection).encode("utf-8"))
        > ledger.quota.max_active_projection_utf8_bytes
    ):
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "configured findings projection quota cannot hold its envelope"
        )
    return projection


def _materialize_projection(
    ledger: ExecutionFindingsLedger,
    *,
    selected: Sequence[ExecutionFindingEntry],
    omitted_entry_ids: Sequence[str],
) -> ExecutionFindingsActiveProjection:
    durable_bytes = _entry_revisions_utf8_bytes(ledger.entry_revisions)
    payload = {
        "ledger_id": ledger.ledger_id,
        "ledger_revision": ledger.revision,
        "active_entries": tuple(sorted(selected, key=lambda item: item.sequence)),
        "omitted_active_count": len(omitted_entry_ids),
        "omitted_entry_ids_sha256": (
            sha256_json(sorted(omitted_entry_ids)) if omitted_entry_ids else None
        ),
        "remaining_durable_revisions": max(
            0,
            ledger.quota.max_durable_entry_revisions
            - len(ledger.entry_revisions),
        ),
        "remaining_durable_utf8_bytes": max(
            0,
            ledger.quota.max_durable_utf8_bytes - durable_bytes,
        ),
        "projection_sha256": "0" * 64,
    }
    provisional = ExecutionFindingsActiveProjection.model_construct(**payload)
    payload["projection_sha256"] = execution_findings_projection_sha256(
        provisional
    )
    return ExecutionFindingsActiveProjection.model_validate(payload)


def _entry_revisions_utf8_bytes(
    entries: Iterable[ExecutionFindingEntry],
) -> int:
    return sum(len(canonical_json(entry).encode("utf-8")) for entry in entries)


def _load_quota(row: sqlite3.Row) -> ExecutionFindingsQuota:
    quota_json = str(row["quota_json"])
    if hashlib.sha256(quota_json.encode("utf-8")).hexdigest() != str(
        row["quota_hash"]
    ):
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings quota hash changed"
        )
    try:
        quota = validate_persisted_execution_findings_quota_json(quota_json)
    except ValueError as exc:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings quota is invalid"
        ) from exc
    if canonical_json(quota) != quota_json:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            "stored findings quota is not canonical"
        )
    return quota


def _canonical_stored_json(value: str, *, label: str) -> object:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            f"stored {label} is not JSON"
        ) from exc
    if canonical_json(payload) != value:
        raise ExecutionFindingsStoredAuthorityCorrupt(
            f"stored {label} is not canonical"
        )
    return payload


def _require_ledger_row(
    conn: sqlite3.Connection,
    ledger_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM execution_findings_ledgers WHERE ledger_id=?",
        (ledger_id,),
    ).fetchone()
    if row is None:
        raise ExecutionFindingsPersistenceError("findings ledger does not exist")
    return row


def _require_identifier(name: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} is not a bounded identifier")


def _stable_id(prefix: str, payload: object) -> str:
    return f"{prefix}_{sha256_json(payload)}"


__all__ = [
    "ExecutionFindingsMutationIdentityCollision",
    "ExecutionFindingsOwnerClosed",
    "ExecutionFindingsPersistenceError",
    "ExecutionFindingsQuotaExceeded",
    "ExecutionFindingsRevisionConflict",
    "ExecutionFindingsScopeInvalid",
    "ExecutionFindingsSourceReferenceInvalid",
    "ExecutionFindingsStoredAuthorityCorrupt",
    "apply_execution_findings_mutation",
    "close_execution_findings_ledger",
    "create_execution_findings_ledger",
    "create_execution_findings_ledger_in_transaction",
    "create_execution_findings_owner_companion_in_transaction",
    "get_execution_findings_ledger",
    "get_execution_findings_ledger_for_owner",
]
