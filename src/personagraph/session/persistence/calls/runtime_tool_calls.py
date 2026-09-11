"""崩溃安全 Runtime 工具调用分派的 SQLite 权威源。

现有 ``insession_work_run_tool_calls`` 行仍是 WorkRun 的语义 ToolCall 权威。此账本位于
该不可变调用之下：一个已认证逻辑请求、零次或多次物理分派尝试，以及每次物理尝试最多
一份不可变结算回执。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ....runtime.tool_calls import (
    RuntimeToolEffectClass,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
)
from ..deps import StoreDeps


class RuntimeToolCallPersistenceError(RuntimeError):
    """持久工具调用权威不变量以关闭方式失败。"""


class RuntimeToolCallIdentityCollision(RuntimeToolCallPersistenceError):
    """持久 ID 或键被复用于不同不可变权威。"""


class RuntimeToolCallWaitingExternalState(RuntimeToolCallPersistenceError):
    """待处理或不确定的物理分派需要协调。"""


class RuntimeToolCallTerminalState(RuntimeToolCallPersistenceError):
    """此逻辑调用下不得打开新的物理尝试。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredRuntimeToolPhysicalAttempt(_Record):
    request: RuntimeToolPhysicalAttemptRequest
    settlement: RuntimeToolPhysicalAttemptSettlement | None = None


class StoredRuntimeToolLogicalCall(_Record):
    request: RuntimeToolLogicalRequest
    physical_attempts: tuple[StoredRuntimeToolPhysicalAttempt, ...] = ()


class RuntimeToolLedgerMutationResult(_Record):
    status: Literal["applied", "replayed"]
    logical_call: StoredRuntimeToolLogicalCall
    physical_attempt_id: str | None = None
    settlement_id: str | None = None


def reserve_runtime_tool_logical_call(
    deps: StoreDeps,
    *,
    request: RuntimeToolLogicalRequest,
) -> RuntimeToolLedgerMutationResult:
    """在任何物理分派前预留一个精确物化 ToolCall。"""

    if not isinstance(request, RuntimeToolLogicalRequest):
        raise TypeError("request must be RuntimeToolLogicalRequest")
    deps.init_db()
    now = deps.now()
    request_json = _model_json(request)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _load_logical_call(conn, request.logical_tool_call_id)
        if existing is not None:
            if existing.request != request:
                raise RuntimeToolCallIdentityCollision(
                    "logical tool call ID crossed immutable request authority"
                )
            return RuntimeToolLedgerMutationResult(
                status="replayed",
                logical_call=existing,
            )

        _require_materialized_call_authority(
            conn,
            request,
            require_active=True,
            require_unsettled_materialized_call=True,
        )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_tool_logical_calls "
                "(logical_tool_call_id, session_id, work_run_id, attempt_id, "
                "call_ordinal, invocation_turn_id, catalog_snapshot_sha256, "
                "tool_id, contract_version, implementation_version, "
                "provider_identity_sha256, effect_profile_sha256, effect_class, "
                "retry_authority, arguments_sha256, result_contract, "
                "max_physical_attempts, state_guard_sha256, binding_sha256, "
                "request_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request.logical_tool_call_id,
                    request.session_id,
                    request.work_run_id,
                    request.attempt_id,
                    request.call_ordinal,
                    request.invocation_turn_id,
                    request.catalog_snapshot_sha256,
                    request.tool_id,
                    request.contract_version,
                    request.implementation_version,
                    request.provider_identity_sha256,
                    request.effect_profile_sha256,
                    request.effect_class.value,
                    request.retry_authority.value,
                    request.arguments_sha256,
                    request.result_contract,
                    request.max_physical_attempts,
                    request.state_guard_sha256,
                    request.binding_sha256,
                    request_json,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeToolCallPersistenceError(
                "logical tool request conflicts with Session/Turn/WorkRun/Attempt authority"
            ) from exc
        return RuntimeToolLedgerMutationResult(
            status="applied",
            logical_call=_require_logical_call(
                conn,
                request.logical_tool_call_id,
            ),
        )


def append_runtime_tool_physical_attempt(
    deps: StoreDeps,
    *,
    request: RuntimeToolPhysicalAttemptRequest,
) -> RuntimeToolLedgerMutationResult:
    """在工具 I/O 前追加恰好一次已授权物理分派。"""

    if not isinstance(request, RuntimeToolPhysicalAttemptRequest):
        raise TypeError("request must be RuntimeToolPhysicalAttemptRequest")
    deps.init_db()
    now = deps.now()
    request_json = _model_json(request)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        logical = _require_logical_call(conn, request.logical_tool_call_id)
        _require_physical_binding(logical.request, request)

        existing = _load_physical_attempt_by_identity(
            conn,
            physical_attempt_id=request.physical_attempt_id,
            physical_attempt_key=request.physical_attempt_key,
        )
        if existing is not None:
            if existing.request != request:
                raise RuntimeToolCallIdentityCollision(
                    "physical tool attempt ID/key crossed immutable request authority"
                )
            return RuntimeToolLedgerMutationResult(
                status="replayed",
                logical_call=_require_logical_call(
                    conn,
                    request.logical_tool_call_id,
                ),
                physical_attempt_id=request.physical_attempt_id,
            )

        attempts = logical.physical_attempts
        expected_ordinal = len(attempts) + 1
        if request.physical_ordinal != expected_ordinal:
            raise RuntimeToolCallPersistenceError(
                "physical tool attempt ordinal is not the exact next ordinal"
            )
        if request.physical_ordinal > logical.request.max_physical_attempts:
            raise RuntimeToolCallTerminalState(
                "logical tool call physical-attempt limit is exhausted"
            )
        if attempts:
            _require_retry_grant(logical.request, attempts[-1])
            _require_stable_provider_idempotency_key(
                logical.request,
                attempts,
                request,
            )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_tool_physical_attempts "
                "(physical_attempt_id, physical_attempt_key, session_id, "
                "work_run_id, attempt_id, logical_tool_call_id, "
                "logical_request_binding_sha256, physical_ordinal, "
                "started_turn_id, provider_identity_sha256, effect_class, "
                "retry_authority, result_contract, provider_idempotency_key, "
                "dispatch_authority_sha256, binding_sha256, "
                "physical_request_json, status, settlement_apply_id, "
                "started_at, settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)",
                (
                    request.physical_attempt_id,
                    request.physical_attempt_key,
                    logical.request.session_id,
                    logical.request.work_run_id,
                    logical.request.attempt_id,
                    request.logical_tool_call_id,
                    request.logical_request_binding_sha256,
                    request.physical_ordinal,
                    request.started_turn_id,
                    logical.request.provider_identity_sha256,
                    logical.request.effect_class.value,
                    request.retry_authority.value,
                    logical.request.result_contract,
                    request.provider_idempotency_key,
                    request.dispatch_authority_sha256,
                    request.binding_sha256,
                    request_json,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeToolCallPersistenceError(
                "physical tool request conflicts with logical/Turn authority"
            ) from exc
        return RuntimeToolLedgerMutationResult(
            status="applied",
            logical_call=_require_logical_call(
                conn,
                request.logical_tool_call_id,
            ),
            physical_attempt_id=request.physical_attempt_id,
        )


def settle_runtime_tool_physical_attempt(
    deps: StoreDeps,
    *,
    settlement: RuntimeToolPhysicalAttemptSettlement,
) -> RuntimeToolLedgerMutationResult:
    """结算一个待处理分派，并原子持久化其精确回执。"""

    if not isinstance(settlement, RuntimeToolPhysicalAttemptSettlement):
        raise TypeError(
            "settlement must be RuntimeToolPhysicalAttemptSettlement"
        )
    deps.init_db()
    now = deps.now()
    settlement_json = _model_json(settlement)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT settlement_json FROM "
            "insession_runtime_tool_call_settlement_receipts "
            "WHERE settle_apply_id=?",
            (settlement.settle_apply_id,),
        ).fetchone()
        if replay is not None:
            if str(replay["settlement_json"]) != settlement_json:
                raise RuntimeToolCallIdentityCollision(
                    "tool settlement apply ID crossed immutable receipt authority"
                )
            return RuntimeToolLedgerMutationResult(
                status="replayed",
                logical_call=_require_logical_call(
                    conn,
                    settlement.logical_tool_call_id,
                ),
                physical_attempt_id=settlement.physical_attempt_id,
                settlement_id=settlement.settlement_id,
            )
        settlement_identity = conn.execute(
            "SELECT settle_apply_id FROM "
            "insession_runtime_tool_call_settlement_receipts "
            "WHERE settlement_id=?",
            (settlement.settlement_id,),
        ).fetchone()
        if settlement_identity is not None:
            raise RuntimeToolCallIdentityCollision(
                "tool settlement ID crossed immutable receipt authority"
            )

        logical = _require_logical_call(
            conn,
            settlement.logical_tool_call_id,
        )
        attempt = next(
            (
                item
                for item in logical.physical_attempts
                if item.request.physical_attempt_id
                == settlement.physical_attempt_id
            ),
            None,
        )
        if attempt is None:
            raise RuntimeToolCallPersistenceError(
                "tool settlement references an unknown physical attempt"
            )
        if attempt.settlement is not None:
            raise RuntimeToolCallIdentityCollision(
                "physical tool attempt is already settled by another receipt"
            )
        if (
            settlement.physical_request_binding_sha256
            != attempt.request.binding_sha256
            or settlement.physical_ordinal != attempt.request.physical_ordinal
        ):
            raise RuntimeToolCallPersistenceError(
                "tool settlement crossed physical request authority"
            )
        if (
            settlement.typed_result is not None
            and settlement.typed_result.result_contract
            != logical.request.result_contract
        ):
            raise RuntimeToolCallPersistenceError(
                "typed tool result contract differs from its logical request"
            )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_tool_call_settlement_receipts "
                "(settle_apply_id, settlement_id, session_id, work_run_id, "
                "attempt_id, logical_tool_call_id, physical_attempt_id, "
                "physical_request_binding_sha256, physical_ordinal, outcome, "
                "receipt_sha256, settlement_json, settled_turn_id, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    settlement.settle_apply_id,
                    settlement.settlement_id,
                    logical.request.session_id,
                    logical.request.work_run_id,
                    logical.request.attempt_id,
                    settlement.logical_tool_call_id,
                    settlement.physical_attempt_id,
                    settlement.physical_request_binding_sha256,
                    settlement.physical_ordinal,
                    settlement.outcome.value,
                    settlement.receipt_sha256,
                    settlement_json,
                    settlement.settled_turn_id,
                    now,
                ),
            )
            updated = conn.execute(
                "UPDATE insession_runtime_tool_physical_attempts SET "
                "status=?, settlement_apply_id=?, settled_at=? "
                "WHERE physical_attempt_id=? AND logical_tool_call_id=? "
                "AND binding_sha256=? AND status='pending' "
                "AND settlement_apply_id IS NULL",
                (
                    settlement.outcome.value,
                    settlement.settle_apply_id,
                    now,
                    settlement.physical_attempt_id,
                    settlement.logical_tool_call_id,
                    settlement.physical_request_binding_sha256,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeToolCallPersistenceError(
                    "physical tool attempt changed during settlement"
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeToolCallPersistenceError(
                "tool settlement conflicts with physical/Turn authority"
            ) from exc
        return RuntimeToolLedgerMutationResult(
            status="applied",
            logical_call=_require_logical_call(
                conn,
                settlement.logical_tool_call_id,
            ),
            physical_attempt_id=settlement.physical_attempt_id,
            settlement_id=settlement.settlement_id,
        )


def get_runtime_tool_logical_call(
    deps: StoreDeps,
    *,
    session_id: str,
    logical_tool_call_id: str,
) -> StoredRuntimeToolLogicalCall | None:
    deps.init_db()
    with deps.connect() as conn:
        stored = _load_logical_call(conn, logical_tool_call_id)
    if stored is None or stored.request.session_id != session_id:
        return None
    return stored


def _require_retry_grant(
    logical: RuntimeToolLogicalRequest,
    previous: StoredRuntimeToolPhysicalAttempt,
) -> None:
    settlement = previous.settlement
    if settlement is None:
        raise RuntimeToolCallWaitingExternalState(
            "pending physical tool dispatch requires reconciliation"
        )
    if settlement.outcome is RuntimeToolPhysicalOutcome.UNCERTAIN:
        raise RuntimeToolCallWaitingExternalState(
            "uncertain physical tool outcome requires reconciliation"
        )
    if settlement.outcome is not RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE:
        raise RuntimeToolCallTerminalState(
            "logical tool call already has a terminal physical outcome"
        )
    if logical.retry_authority not in {
        RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
        RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY,
    }:
        raise RuntimeToolCallTerminalState(
            "retryable tool outcome has no frozen retry authority"
        )


def _require_stable_provider_idempotency_key(
    logical: RuntimeToolLogicalRequest,
    previous_attempts: tuple[StoredRuntimeToolPhysicalAttempt, ...],
    request: RuntimeToolPhysicalAttemptRequest,
) -> None:
    if (
        logical.retry_authority
        is not RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
    ):
        return
    first_key = previous_attempts[0].request.provider_idempotency_key
    if request.provider_idempotency_key != first_key:
        raise RuntimeToolCallPersistenceError(
            "provider-idempotent retries must reuse the exact frozen key"
        )


def _require_physical_binding(
    logical: RuntimeToolLogicalRequest,
    physical: RuntimeToolPhysicalAttemptRequest,
) -> None:
    if (
        physical.logical_request_binding_sha256 != logical.binding_sha256
        or physical.retry_authority is not logical.retry_authority
    ):
        raise RuntimeToolCallPersistenceError(
            "physical tool request differs from logical request authority"
        )


def _require_materialized_call_authority(
    conn: sqlite3.Connection,
    request: RuntimeToolLogicalRequest,
    *,
    require_active: bool,
    require_unsettled_materialized_call: bool,
) -> None:
    row = conn.execute(
        "SELECT run.session_id, attempt.input_turn_id AS invocation_turn_id, "
        "attempt.status AS attempt_status, "
        "attempt.catalog_snapshot_hash, call.ordinal, call.tool_id, "
        "call.tool_version, call.modifies_environment, call.arguments_hash, "
        "call.arguments_json, result.tool_result_id "
        "FROM insession_work_run_tool_calls AS call "
        "JOIN insession_work_runs AS run ON run.work_run_id=call.work_run_id "
        "JOIN insession_work_run_attempts AS attempt "
        "ON attempt.work_run_id=call.work_run_id "
        "AND attempt.attempt_id=call.attempt_id "
        "LEFT JOIN insession_work_run_tool_results AS result "
        "ON result.work_run_id=call.work_run_id "
        "AND result.attempt_id=call.attempt_id "
        "AND result.tool_call_id=call.tool_call_id "
        "WHERE call.work_run_id=? AND call.attempt_id=? "
        "AND call.tool_call_id=?",
        (
            request.work_run_id,
            request.attempt_id,
            request.logical_tool_call_id,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeToolCallPersistenceError(
            "logical tool request has no exact materialized WorkRun ToolCall"
        )
    expected_modifies_environment = int(
        request.effect_class is RuntimeToolEffectClass.PROTECTED_EFFECT
    )
    expected = {
        "session_id": request.session_id,
# 搁置 Attempt 重新绑定时，``attempt.turn_id`` 刻意保持可变。``input_turn_id`` 仍是逻辑
# ToolCall 的不可变来源，因此与账本绑定匹配。
        "invocation_turn_id": request.invocation_turn_id,
        "catalog_snapshot_hash": request.catalog_snapshot_sha256,
        "ordinal": request.call_ordinal,
        "tool_id": request.tool_id,
        "tool_version": request.implementation_version,
        "modifies_environment": expected_modifies_environment,
        "arguments_hash": request.arguments_sha256,
        "arguments_json": request.arguments_json,
    }
    _require_row_projection(
        row,
        expected,
        label="materialized WorkRun ToolCall",
    )
    if require_active and str(row["attempt_status"]) != "active":
        raise RuntimeToolCallPersistenceError(
            "a new logical tool request requires its active decided Attempt"
        )
    if require_unsettled_materialized_call and row["tool_result_id"] is not None:
        raise RuntimeToolCallPersistenceError(
            "a settled ToolResult cannot be adopted after tool dispatch"
        )


def _require_logical_call(
    conn: sqlite3.Connection,
    logical_tool_call_id: str,
) -> StoredRuntimeToolLogicalCall:
    stored = _load_logical_call(conn, logical_tool_call_id)
    if stored is None:
        raise RuntimeToolCallPersistenceError("logical tool call is missing")
    return stored


def _load_logical_call(
    conn: sqlite3.Connection,
    logical_tool_call_id: str,
) -> StoredRuntimeToolLogicalCall | None:
    row = conn.execute(
        "SELECT * FROM insession_runtime_tool_logical_calls "
        "WHERE logical_tool_call_id=?",
        (logical_tool_call_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        request = RuntimeToolLogicalRequest.model_validate_json(
            str(row["request_json"])
        )
    except Exception as exc:
        raise RuntimeToolCallPersistenceError(
            "stored logical tool request is corrupt"
        ) from exc
    _require_logical_row_projection(row, request)
    _require_materialized_call_authority(
        conn,
        request,
        require_active=False,
        require_unsettled_materialized_call=False,
    )

    attempts: list[StoredRuntimeToolPhysicalAttempt] = []
    rows = conn.execute(
        "SELECT * FROM insession_runtime_tool_physical_attempts "
        "WHERE logical_tool_call_id=? ORDER BY physical_ordinal",
        (logical_tool_call_id,),
    ).fetchall()
    for expected_ordinal, physical_row in enumerate(rows, start=1):
        try:
            physical = RuntimeToolPhysicalAttemptRequest.model_validate_json(
                str(physical_row["physical_request_json"])
            )
        except Exception as exc:
            raise RuntimeToolCallPersistenceError(
                "stored physical tool request is corrupt"
            ) from exc
        _require_physical_row_projection(physical_row, request, physical)
        if physical.physical_ordinal != expected_ordinal:
            raise RuntimeToolCallPersistenceError(
                "stored physical tool attempt ordinals are not contiguous"
            )
        _require_physical_binding(request, physical)

        settlement_row = conn.execute(
            "SELECT * FROM insession_runtime_tool_call_settlement_receipts "
            "WHERE physical_attempt_id=?",
            (physical.physical_attempt_id,),
        ).fetchone()
        settlement = None
        status = str(physical_row["status"])
        if settlement_row is None:
            if status != "pending" or physical_row["settlement_apply_id"] is not None:
                raise RuntimeToolCallPersistenceError(
                    "settled physical tool row has no immutable receipt"
                )
        else:
            try:
                settlement = RuntimeToolPhysicalAttemptSettlement.model_validate_json(
                    str(settlement_row["settlement_json"])
                )
            except Exception as exc:
                raise RuntimeToolCallPersistenceError(
                    "stored physical tool settlement is corrupt"
                ) from exc
            _require_settlement_row_projection(
                settlement_row,
                request,
                settlement,
            )
            if (
                status != settlement.outcome.value
                or str(physical_row["settlement_apply_id"])
                != settlement.settle_apply_id
                or settlement.physical_attempt_id != physical.physical_attempt_id
                or settlement.physical_request_binding_sha256
                != physical.binding_sha256
                or settlement.physical_ordinal != physical.physical_ordinal
            ):
                raise RuntimeToolCallPersistenceError(
                    "physical tool row and settlement receipt disagree"
                )
            if (
                settlement.typed_result is not None
                and settlement.typed_result.result_contract
                != request.result_contract
            ):
                raise RuntimeToolCallPersistenceError(
                    "stored typed tool result contract is cross-bound"
                )
        attempts.append(
            StoredRuntimeToolPhysicalAttempt(
                request=physical,
                settlement=settlement,
            )
        )

    if len(attempts) > request.max_physical_attempts:
        raise RuntimeToolCallPersistenceError(
            "stored logical tool call exceeds its physical-attempt limit"
        )
    for predecessor in attempts[:-1]:
        try:
            _require_retry_grant(request, predecessor)
        except (
            RuntimeToolCallWaitingExternalState,
            RuntimeToolCallTerminalState,
        ) as exc:
            raise RuntimeToolCallPersistenceError(
                "stored tool attempt has an unauthorized successor"
            ) from exc
    if (
        request.retry_authority
        is RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
        and len(attempts) > 1
    ):
        expected_key = attempts[0].request.provider_idempotency_key
        if any(
            item.request.provider_idempotency_key != expected_key
            for item in attempts[1:]
        ):
            raise RuntimeToolCallPersistenceError(
                "stored provider-idempotent attempts changed their key"
            )
    return StoredRuntimeToolLogicalCall(
        request=request,
        physical_attempts=tuple(attempts),
    )


def _load_physical_attempt_by_identity(
    conn: sqlite3.Connection,
    *,
    physical_attempt_id: str,
    physical_attempt_key: str,
) -> StoredRuntimeToolPhysicalAttempt | None:
    rows = conn.execute(
        "SELECT logical_tool_call_id "
        "FROM insession_runtime_tool_physical_attempts "
        "WHERE physical_attempt_id=? OR physical_attempt_key=?",
        (physical_attempt_id, physical_attempt_key),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeToolCallIdentityCollision(
            "physical tool ID and key resolve to different attempts"
        )
    logical = _require_logical_call(
        conn,
        str(rows[0]["logical_tool_call_id"]),
    )
    return next(
        item
        for item in logical.physical_attempts
        if item.request.physical_attempt_id == physical_attempt_id
        or item.request.physical_attempt_key == physical_attempt_key
    )


def _require_logical_row_projection(
    row: sqlite3.Row,
    request: RuntimeToolLogicalRequest,
) -> None:
    expected = {
        "logical_tool_call_id": request.logical_tool_call_id,
        "session_id": request.session_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "call_ordinal": request.call_ordinal,
        "invocation_turn_id": request.invocation_turn_id,
        "catalog_snapshot_sha256": request.catalog_snapshot_sha256,
        "tool_id": request.tool_id,
        "contract_version": request.contract_version,
        "implementation_version": request.implementation_version,
        "provider_identity_sha256": request.provider_identity_sha256,
        "effect_profile_sha256": request.effect_profile_sha256,
        "effect_class": request.effect_class.value,
        "retry_authority": request.retry_authority.value,
        "arguments_sha256": request.arguments_sha256,
        "result_contract": request.result_contract,
        "max_physical_attempts": request.max_physical_attempts,
        "state_guard_sha256": request.state_guard_sha256,
        "binding_sha256": request.binding_sha256,
    }
    _require_row_projection(row, expected, label="logical tool request")


def _require_physical_row_projection(
    row: sqlite3.Row,
    logical: RuntimeToolLogicalRequest,
    request: RuntimeToolPhysicalAttemptRequest,
) -> None:
    expected = {
        "physical_attempt_id": request.physical_attempt_id,
        "physical_attempt_key": request.physical_attempt_key,
        "session_id": logical.session_id,
        "work_run_id": logical.work_run_id,
        "attempt_id": logical.attempt_id,
        "logical_tool_call_id": request.logical_tool_call_id,
        "logical_request_binding_sha256": request.logical_request_binding_sha256,
        "physical_ordinal": request.physical_ordinal,
        "started_turn_id": request.started_turn_id,
        "provider_identity_sha256": logical.provider_identity_sha256,
        "effect_class": logical.effect_class.value,
        "retry_authority": request.retry_authority.value,
        "result_contract": logical.result_contract,
        "provider_idempotency_key": request.provider_idempotency_key,
        "dispatch_authority_sha256": request.dispatch_authority_sha256,
        "binding_sha256": request.binding_sha256,
    }
    _require_row_projection(row, expected, label="physical tool request")


def _require_settlement_row_projection(
    row: sqlite3.Row,
    logical: RuntimeToolLogicalRequest,
    settlement: RuntimeToolPhysicalAttemptSettlement,
) -> None:
    expected = {
        "settle_apply_id": settlement.settle_apply_id,
        "settlement_id": settlement.settlement_id,
        "session_id": logical.session_id,
        "work_run_id": logical.work_run_id,
        "attempt_id": logical.attempt_id,
        "logical_tool_call_id": settlement.logical_tool_call_id,
        "physical_attempt_id": settlement.physical_attempt_id,
        "physical_request_binding_sha256": (
            settlement.physical_request_binding_sha256
        ),
        "physical_ordinal": settlement.physical_ordinal,
        "outcome": settlement.outcome.value,
        "receipt_sha256": settlement.receipt_sha256,
        "settled_turn_id": settlement.settled_turn_id,
    }
    _require_row_projection(row, expected, label="physical tool settlement")


def _require_row_projection(
    row: sqlite3.Row,
    expected: dict[str, object],
    *,
    label: str,
) -> None:
    for column, value in expected.items():
        if row[column] != value:
            raise RuntimeToolCallPersistenceError(
                f"stored {label} indexed projection is corrupt: {column}"
            )


def _model_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "RuntimeToolCallIdentityCollision",
    "RuntimeToolCallPersistenceError",
    "RuntimeToolCallTerminalState",
    "RuntimeToolCallWaitingExternalState",
    'RuntimeToolLedgerMutationResult',
    'StoredRuntimeToolLogicalCall',
    'StoredRuntimeToolPhysicalAttempt',
    "append_runtime_tool_physical_attempt",
    "get_runtime_tool_logical_call",
    "reserve_runtime_tool_logical_call",
    "settle_runtime_tool_physical_attempt",
]
