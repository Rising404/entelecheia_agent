"""与 provider 无关的 Runtime 模型调用账本 SQLite 命令。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ....model_io.output_repair_contracts import RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES
from ....runtime.model_calls.contracts import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeModelPhysicalOutcome,
)
from ..deps import StoreDeps


_REJECTED_OUTPUT_RECORD_CONTRACT = "runtime-model-rejected-output-record-v1"
class RuntimeModelCallPersistenceError(RuntimeError):
    """通用模型账本权威不变量以关闭方式失败。"""


class RuntimeModelCallIdentityCollision(RuntimeModelCallPersistenceError):
    """持久 ID 或键被复用于不同不可变权威。"""


class RuntimeModelCallTerminalState(RuntimeModelCallPersistenceError):
    """此逻辑调用下不得打开新的物理尝试。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredRuntimeModelPhysicalAttempt(_Record):
    request: RuntimeModelPhysicalAttemptRequest
    settlement: RuntimeModelPhysicalAttemptSettlement | None = None


class StoredRuntimeModelLogicalCall(_Record):
    request: RuntimeModelLogicalRequest
    physical_attempts: tuple[StoredRuntimeModelPhysicalAttempt, ...] = ()


class StoredRuntimeModelRejectedOutput(_Record):
    """经过认证的被拒 Provider 正文，仅用于重建修复输入。"""

    schema_version: Literal["runtime-model-rejected-output-record-v1"] = (
        _REJECTED_OUTPUT_RECORD_CONTRACT
    )
    session_id: str
    logical_call_id: str
    physical_attempt_id: str
    physical_ordinal: int
    response_sha256: str
    byte_count: int
    response_text: str
    record_sha256: str


class RuntimeModelLedgerMutationResult(_Record):
    status: Literal["applied", "replayed"]
    logical_call: StoredRuntimeModelLogicalCall
    physical_attempt_id: str | None = None
    settlement_id: str | None = None


def reserve_runtime_model_logical_call(
    deps: StoreDeps,
    *,
    request: RuntimeModelLogicalRequest,
) -> RuntimeModelLedgerMutationResult:
    """在任何 Provider 分派前持久化一个完整逻辑请求。"""

    if not isinstance(request, RuntimeModelLogicalRequest):
        raise TypeError("request must be RuntimeModelLogicalRequest")
    deps.init_db()
    now = deps.now()
    request_json = _model_json(request)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _load_logical_call(conn, request.logical_call_id)
        if existing is not None:
            if existing.request != request:
                raise RuntimeModelCallIdentityCollision(
                    "logical model call ID crossed immutable request authority"
                )
            return RuntimeModelLedgerMutationResult(
                status="replayed",
                logical_call=existing,
            )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_model_logical_calls "
                "(logical_call_id, session_id, insession_task_id, "
                "auxiliary_graph_id, goal_id, execution_subject_id, "
                "invocation_turn_id, call_kind, purpose, provider, model, "
                "endpoint_fingerprint, request_contract, request_sha256, "
                "typed_result_contract, max_physical_attempts, "
                "state_guard_sha256, binding_sha256, request_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request.logical_call_id,
                    request.session_id,
                    request.task_id,
                    request.auxiliary_graph_id,
                    request.goal_id,
                    request.execution_subject_id,
                    request.invocation_turn_id,
                    request.call_kind,
                    request.purpose,
                    request.provider,
                    request.model,
                    request.endpoint_fingerprint,
                    request.request_contract,
                    request.request_sha256,
                    request.typed_result_contract,
                    request.max_physical_attempts,
                    request.state_guard_sha256,
                    request.binding_sha256,
                    request_json,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeModelCallPersistenceError(
                "logical model request conflicts with Session/Task/goal/subject authority"
            ) from exc
        stored = _require_logical_call(conn, request.logical_call_id)
        return RuntimeModelLedgerMutationResult(
            status="applied",
            logical_call=stored,
        )


def append_runtime_model_physical_attempt(
    deps: StoreDeps,
    *,
    request: RuntimeModelPhysicalAttemptRequest,
) -> RuntimeModelLedgerMutationResult:
    """在精确逻辑请求下追加一次物理 Provider 分派。"""

    if not isinstance(request, RuntimeModelPhysicalAttemptRequest):
        raise TypeError("request must be RuntimeModelPhysicalAttemptRequest")
    deps.init_db()
    now = deps.now()
    request_json = _model_json(request)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        logical = _require_logical_call(conn, request.logical_call_id)
        _require_physical_binding(logical.request, request)

        existing = _load_physical_attempt_by_identity(
            conn,
            physical_attempt_id=request.physical_attempt_id,
            physical_attempt_key=request.physical_attempt_key,
        )
        if existing is not None:
            if existing.request != request:
                raise RuntimeModelCallIdentityCollision(
                    "physical model attempt ID/key crossed immutable request authority"
                )
            return RuntimeModelLedgerMutationResult(
                status="replayed",
                logical_call=_require_logical_call(conn, request.logical_call_id),
                physical_attempt_id=request.physical_attempt_id,
            )

        attempts = logical.physical_attempts
        expected_ordinal = len(attempts) + 1
        if request.physical_ordinal != expected_ordinal:
            raise RuntimeModelCallPersistenceError(
                "physical model attempt ordinal is not the exact next ordinal"
            )
        if request.physical_ordinal > logical.request.max_physical_attempts:
            raise RuntimeModelCallTerminalState(
                "logical model call physical-attempt limit is exhausted"
            )
        if attempts:
            last = attempts[-1]
            if last.settlement is None:
                raise RuntimeModelCallTerminalState(
                    "a pending physical Provider request requires reconciliation"
                )
            if (
                last.settlement.outcome
                is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
            ):
                raise RuntimeModelCallTerminalState(
                    "logical model call already has a terminal physical outcome"
                )
        _require_physical_retry_chain(
            prior_attempts=attempts,
            request=request,
        )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_model_physical_attempts "
                "(physical_attempt_id, physical_attempt_key, session_id, "
                "logical_call_id, logical_request_binding_sha256, "
                "physical_ordinal, started_turn_id, provider, model, "
                "endpoint_fingerprint, request_sha256, provider_idempotency_key, "
                "dispatch_authority_sha256, binding_sha256, physical_request_json, "
                "status, settlement_apply_id, started_at, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "'pending', NULL, ?, NULL)",
                (
                    request.physical_attempt_id,
                    request.physical_attempt_key,
                    logical.request.session_id,
                    request.logical_call_id,
                    request.logical_request_binding_sha256,
                    request.physical_ordinal,
                    request.started_turn_id,
                    request.provider,
                    request.model,
                    request.endpoint_fingerprint,
                    request.request_sha256,
                    request.provider_idempotency_key,
                    request.dispatch_authority_sha256,
                    request.binding_sha256,
                    request_json,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeModelCallPersistenceError(
                "physical model request conflicts with logical/Turn authority"
            ) from exc
        return RuntimeModelLedgerMutationResult(
            status="applied",
            logical_call=_require_logical_call(conn, request.logical_call_id),
            physical_attempt_id=request.physical_attempt_id,
        )


def settle_runtime_model_physical_attempt(
    deps: StoreDeps,
    *,
    settlement: RuntimeModelPhysicalAttemptSettlement,
    rejected_response_text: str | None = None,
) -> RuntimeModelLedgerMutationResult:
    """结算一次尝试，并原子记录可选被拒正文检查点。"""

    if not isinstance(settlement, RuntimeModelPhysicalAttemptSettlement):
        raise TypeError(
            "settlement must be RuntimeModelPhysicalAttemptSettlement"
        )
    deps.init_db()
    now = deps.now()
    settlement_json = _model_json(settlement)
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT settlement_json FROM "
            "insession_runtime_model_call_settlement_receipts "
            "WHERE settle_apply_id=?",
            (settlement.settle_apply_id,),
        ).fetchone()
        if replay is not None:
            if str(replay["settlement_json"]) != settlement_json:
                raise RuntimeModelCallIdentityCollision(
                    "model settlement apply ID crossed immutable receipt authority"
                )
            logical = _require_logical_call(conn, settlement.logical_call_id)
            attempt = _require_settlement_attempt(
                logical=logical,
                settlement=settlement,
            )
            rejected_output = _prepare_rejected_output_record(
                session_id=logical.request.session_id,
                physical=attempt.request,
                settlement=settlement,
                rejected_response_text=rejected_response_text,
            )
            _require_rejected_output_replay(
                conn,
                physical_attempt_id=settlement.physical_attempt_id,
                expected=rejected_output,
            )
            return RuntimeModelLedgerMutationResult(
                status="replayed",
                logical_call=logical,
                physical_attempt_id=settlement.physical_attempt_id,
                settlement_id=settlement.settlement_id,
            )
        settlement_identity = conn.execute(
            "SELECT settle_apply_id FROM "
            "insession_runtime_model_call_settlement_receipts "
            "WHERE settlement_id=?",
            (settlement.settlement_id,),
        ).fetchone()
        if settlement_identity is not None:
            raise RuntimeModelCallIdentityCollision(
                "model settlement ID crossed immutable receipt authority"
            )

        logical = _require_logical_call(conn, settlement.logical_call_id)
        attempt = _require_settlement_attempt(
            logical=logical,
            settlement=settlement,
        )
        if attempt.settlement is not None:
            raise RuntimeModelCallIdentityCollision(
                "physical model attempt is already settled by another receipt"
            )
        if (
            settlement.physical_request_binding_sha256
            != attempt.request.binding_sha256
            or settlement.physical_ordinal != attempt.request.physical_ordinal
        ):
            raise RuntimeModelCallPersistenceError(
                "model settlement crossed physical request authority"
            )
        if (
            settlement.typed_result is not None
            and settlement.typed_result.result_contract
            != logical.request.typed_result_contract
        ):
            raise RuntimeModelCallPersistenceError(
                "typed model result contract differs from its logical request"
            )
        _require_output_repair_settlement_binding(
            logical=logical.request,
            physical=attempt.request,
            settlement=settlement,
        )
        rejected_output = _prepare_rejected_output_record(
            session_id=logical.request.session_id,
            physical=attempt.request,
            settlement=settlement,
            rejected_response_text=rejected_response_text,
        )
        try:
            conn.execute(
                "INSERT INTO insession_runtime_model_call_settlement_receipts "
                "(settle_apply_id, settlement_id, session_id, logical_call_id, "
                "physical_attempt_id, physical_request_binding_sha256, "
                "physical_ordinal, outcome, receipt_sha256, settlement_json, "
                "settled_turn_id, settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    settlement.settle_apply_id,
                    settlement.settlement_id,
                    logical.request.session_id,
                    settlement.logical_call_id,
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
            if rejected_output is not None:
                _insert_rejected_output_record(
                    conn,
                    record=rejected_output,
                )
            updated = conn.execute(
                "UPDATE insession_runtime_model_physical_attempts SET "
                "status=?, settlement_apply_id=?, settled_at=? "
                "WHERE physical_attempt_id=? AND logical_call_id=? "
                "AND binding_sha256=? AND status='pending' "
                "AND settlement_apply_id IS NULL",
                (
                    settlement.outcome.value,
                    settlement.settle_apply_id,
                    now,
                    settlement.physical_attempt_id,
                    settlement.logical_call_id,
                    settlement.physical_request_binding_sha256,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeModelCallPersistenceError(
                    "physical model attempt changed during settlement"
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeModelCallPersistenceError(
                "model settlement conflicts with physical/Turn authority"
            ) from exc
        return RuntimeModelLedgerMutationResult(
            status="applied",
            logical_call=_require_logical_call(conn, settlement.logical_call_id),
            physical_attempt_id=settlement.physical_attempt_id,
            settlement_id=settlement.settlement_id,
        )


def get_runtime_model_logical_call(
    deps: StoreDeps,
    *,
    session_id: str,
    logical_call_id: str,
) -> StoredRuntimeModelLogicalCall | None:
    deps.init_db()
    with deps.connect() as conn:
        stored = _load_logical_call(conn, logical_call_id)
    if stored is None or stored.request.session_id != session_id:
        return None
    return stored


def get_runtime_model_rejected_output(
    deps: StoreDeps,
    *,
    session_id: str,
    logical_call_id: str,
    rejected_physical_ordinal: int,
    rejected_response_sha256: str,
) -> StoredRuntimeModelRejectedOutput | None:
    """加载并认证修复反馈指定的被拒正文。"""

    deps.init_db()
    with deps.connect() as conn:
        logical = _load_logical_call(conn, logical_call_id)
        if logical is None or logical.request.session_id != session_id:
            return None
        row = conn.execute(
            "SELECT * FROM insession_runtime_model_rejected_outputs "
            "WHERE session_id=? AND logical_call_id=? "
            "AND physical_ordinal=? AND response_sha256=?",
            (
                session_id,
                logical_call_id,
                rejected_physical_ordinal,
                rejected_response_sha256,
            ),
        ).fetchone()
        if row is None:
            return None
        record = _validate_rejected_output_row(row)
        if not 1 <= rejected_physical_ordinal <= len(logical.physical_attempts):
            raise RuntimeModelCallPersistenceError(
                "stored rejected model output has no physical attempt authority"
            )
        attempt = logical.physical_attempts[rejected_physical_ordinal - 1]
        settlement = attempt.settlement
        feedback = (
            None
            if settlement is None
            else settlement.next_output_repair_feedback
        )
        if (
            record.physical_attempt_id != attempt.request.physical_attempt_id
            or settlement is None
            or settlement.outcome
            is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
            or settlement.error_code != "MODEL_BAD_RESPONSE"
            or feedback is None
            or feedback.rejected_physical_ordinal
            != rejected_physical_ordinal
            or feedback.rejected_response_sha256
            != rejected_response_sha256
        ):
            raise RuntimeModelCallPersistenceError(
                "stored rejected model output crossed repair-feedback authority"
            )
        return record


def _require_settlement_attempt(
    *,
    logical: StoredRuntimeModelLogicalCall,
    settlement: RuntimeModelPhysicalAttemptSettlement,
) -> StoredRuntimeModelPhysicalAttempt:
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
        raise RuntimeModelCallPersistenceError(
            "model settlement references an unknown physical attempt"
        )
    return attempt


def _prepare_rejected_output_record(
    *,
    session_id: str,
    physical: RuntimeModelPhysicalAttemptRequest,
    settlement: RuntimeModelPhysicalAttemptSettlement,
    rejected_response_text: str | None,
) -> StoredRuntimeModelRejectedOutput | None:
    feedback = settlement.next_output_repair_feedback
    if feedback is None:
        if rejected_response_text is not None:
            raise RuntimeModelCallPersistenceError(
                "rejected response requires output-repair feedback"
            )
        return None
    if rejected_response_text is None:
        raise RuntimeModelCallPersistenceError(
            "output-repair settlement requires the exact rejected response"
        )
    if not isinstance(rejected_response_text, str):
        raise TypeError("rejected_response_text must be str or None")
    if (
        feedback is None
        or settlement.outcome
        is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
        or settlement.error_code != "MODEL_BAD_RESPONSE"
        or feedback.rejected_physical_ordinal != settlement.physical_ordinal
        or settlement.physical_ordinal != physical.physical_ordinal
    ):
        raise RuntimeModelCallPersistenceError(
            "rejected response requires matching MODEL_BAD_RESPONSE feedback"
        )
    try:
        response_bytes = rejected_response_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeModelCallPersistenceError(
            "rejected model response is not valid UTF-8 text"
        ) from exc
    byte_count = len(response_bytes)
    if byte_count > RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES:
        raise RuntimeModelCallPersistenceError(
            "rejected model response exceeds its durable UTF-8 byte limit"
        )
    response_sha256 = hashlib.sha256(response_bytes).hexdigest()
    if response_sha256 != feedback.rejected_response_sha256:
        raise RuntimeModelCallPersistenceError(
            "rejected model response hash differs from repair feedback"
        )
    values: dict[str, object] = {
        "session_id": session_id,
        "logical_call_id": settlement.logical_call_id,
        "physical_attempt_id": settlement.physical_attempt_id,
        "physical_ordinal": settlement.physical_ordinal,
        "response_sha256": response_sha256,
        "byte_count": byte_count,
        "response_text": rejected_response_text,
    }
    return StoredRuntimeModelRejectedOutput(
        **values,
        record_sha256=_rejected_output_record_sha256(values),
    )


def _insert_rejected_output_record(
    conn: sqlite3.Connection,
    *,
    record: StoredRuntimeModelRejectedOutput,
) -> None:
    conn.execute(
        "INSERT INTO insession_runtime_model_rejected_outputs "
        "(session_id, logical_call_id, physical_attempt_id, "
        "physical_ordinal, response_sha256, byte_count, response_text, "
        "record_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.session_id,
            record.logical_call_id,
            record.physical_attempt_id,
            record.physical_ordinal,
            record.response_sha256,
            record.byte_count,
            record.response_text,
            record.record_sha256,
        ),
    )


def _require_rejected_output_replay(
    conn: sqlite3.Connection,
    *,
    physical_attempt_id: str,
    expected: StoredRuntimeModelRejectedOutput | None,
) -> None:
    if expected is None:
        return
    row = conn.execute(
        "SELECT * FROM insession_runtime_model_rejected_outputs "
        "WHERE physical_attempt_id=?",
        (physical_attempt_id,),
    ).fetchone()
    if row is None:
        raise RuntimeModelCallPersistenceError(
            "replayed model rejection has no durable rejected response"
        )
    if _validate_rejected_output_row(row) != expected:
        raise RuntimeModelCallIdentityCollision(
            "model settlement replay changed its rejected response authority"
        )


def _validate_rejected_output_row(
    row: sqlite3.Row,
) -> StoredRuntimeModelRejectedOutput:
    try:
        response_text = str(row["response_text"])
        response_bytes = response_text.encode("utf-8")
        values: dict[str, object] = {
            "session_id": str(row["session_id"]),
            "logical_call_id": str(row["logical_call_id"]),
            "physical_attempt_id": str(row["physical_attempt_id"]),
            "physical_ordinal": int(row["physical_ordinal"]),
            "response_sha256": str(row["response_sha256"]),
            "byte_count": int(row["byte_count"]),
            "response_text": response_text,
        }
        record = StoredRuntimeModelRejectedOutput(
            **values,
            record_sha256=str(row["record_sha256"]),
        )
    except Exception as exc:
        raise RuntimeModelCallPersistenceError(
            "stored rejected model output is corrupt"
        ) from exc
    if (
        record.byte_count != len(response_bytes)
        or record.byte_count > RUNTIME_MODEL_REJECTED_OUTPUT_MAX_UTF8_BYTES
        or record.response_sha256
        != hashlib.sha256(response_bytes).hexdigest()
        or record.record_sha256 != _rejected_output_record_sha256(values)
    ):
        raise RuntimeModelCallPersistenceError(
            "stored rejected model output is corrupt"
        )
    return record


def _rejected_output_record_sha256(values: dict[str, object]) -> str:
    payload = {
        "schema_version": _REJECTED_OUTPUT_RECORD_CONTRACT,
        **values,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_physical_binding(
    logical: RuntimeModelLogicalRequest,
    physical: RuntimeModelPhysicalAttemptRequest,
) -> None:
    if (
        physical.logical_request_binding_sha256 != logical.binding_sha256
        or physical.provider != logical.provider
        or physical.model != logical.model
        or physical.endpoint_fingerprint != logical.endpoint_fingerprint
        or physical.request_sha256 != logical.request_sha256
    ):
        raise RuntimeModelCallPersistenceError(
            "physical model request differs from its logical request authority"
        )
    if physical.output_repair_enabled != (
        logical.output_repair_protocol is not None
    ):
        raise RuntimeModelCallPersistenceError(
            "physical output-repair mode differs from its logical request authority"
        )
    feedback = physical.output_repair_feedback
    if (
        feedback is not None
        and feedback.target_contract != logical.output_repair_target_contract
    ):
        raise RuntimeModelCallPersistenceError(
            "physical output-repair target differs from its logical result contract"
        )


def _require_logical_call(
    conn: sqlite3.Connection,
    logical_call_id: str,
) -> StoredRuntimeModelLogicalCall:
    stored = _load_logical_call(conn, logical_call_id)
    if stored is None:
        raise RuntimeModelCallPersistenceError("logical model call is missing")
    return stored


def _load_logical_call(
    conn: sqlite3.Connection,
    logical_call_id: str,
) -> StoredRuntimeModelLogicalCall | None:
    row = conn.execute(
        "SELECT * FROM insession_runtime_model_logical_calls "
        "WHERE logical_call_id=?",
        (logical_call_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        request = RuntimeModelLogicalRequest.model_validate_json(
            str(row["request_json"])
        )
    except Exception as exc:
        raise RuntimeModelCallPersistenceError(
            "stored logical model request is corrupt"
        ) from exc
    _require_logical_row_projection(row, request)

    attempts: list[StoredRuntimeModelPhysicalAttempt] = []
    rows = conn.execute(
        "SELECT * FROM insession_runtime_model_physical_attempts "
        "WHERE logical_call_id=? ORDER BY physical_ordinal",
        (logical_call_id,),
    ).fetchall()
    for expected_ordinal, physical_row in enumerate(rows, start=1):
        try:
            physical = RuntimeModelPhysicalAttemptRequest.model_validate_json(
                str(physical_row["physical_request_json"])
            )
        except Exception as exc:
            raise RuntimeModelCallPersistenceError(
                "stored physical model request is corrupt"
            ) from exc
        _require_physical_row_projection(physical_row, physical)
        if physical.physical_ordinal != expected_ordinal:
            raise RuntimeModelCallPersistenceError(
                "stored physical model attempt ordinals are not contiguous"
            )
        _require_physical_binding(request, physical)

        settlement_row = conn.execute(
            "SELECT * FROM insession_runtime_model_call_settlement_receipts "
            "WHERE physical_attempt_id=?",
            (physical.physical_attempt_id,),
        ).fetchone()
        settlement = None
        status = str(physical_row["status"])
        if settlement_row is None:
            if status != "pending" or physical_row["settlement_apply_id"] is not None:
                raise RuntimeModelCallPersistenceError(
                    "settled physical model row has no immutable receipt"
                )
        else:
            try:
                settlement = RuntimeModelPhysicalAttemptSettlement.model_validate_json(
                    str(settlement_row["settlement_json"])
                )
            except Exception as exc:
                raise RuntimeModelCallPersistenceError(
                    "stored physical model settlement is corrupt"
                ) from exc
            _require_settlement_row_projection(settlement_row, settlement)
            if (
                status != settlement.outcome.value
                or str(physical_row["settlement_apply_id"])
                != settlement.settle_apply_id
                or settlement.physical_attempt_id != physical.physical_attempt_id
                or settlement.physical_request_binding_sha256
                != physical.binding_sha256
                or settlement.physical_ordinal != physical.physical_ordinal
            ):
                raise RuntimeModelCallPersistenceError(
                    "physical model row and settlement receipt disagree"
                )
            if (
                settlement.typed_result is not None
                and settlement.typed_result.result_contract
                != request.typed_result_contract
            ):
                raise RuntimeModelCallPersistenceError(
                    "stored typed model result contract is cross-bound"
                )
            _require_output_repair_settlement_binding(
                logical=request,
                physical=physical,
                settlement=settlement,
            )
            _require_settled_rejected_output_binding(
                conn,
                session_id=request.session_id,
                physical=physical,
                settlement=settlement,
            )
        _require_physical_retry_chain(
            prior_attempts=tuple(attempts),
            request=physical,
        )
        attempts.append(
            StoredRuntimeModelPhysicalAttempt(
                request=physical,
                settlement=settlement,
            )
        )
    if len(attempts) > request.max_physical_attempts:
        raise RuntimeModelCallPersistenceError(
            "stored logical model call exceeds its physical-attempt limit"
        )
    if any(item.settlement is None for item in attempts[:-1]):
        raise RuntimeModelCallPersistenceError(
            "only the final physical model attempt may remain pending"
        )
    return StoredRuntimeModelLogicalCall(
        request=request,
        physical_attempts=tuple(attempts),
    )


def _load_physical_attempt_by_identity(
    conn: sqlite3.Connection,
    *,
    physical_attempt_id: str,
    physical_attempt_key: str,
) -> StoredRuntimeModelPhysicalAttempt | None:
    rows = conn.execute(
        "SELECT logical_call_id FROM insession_runtime_model_physical_attempts "
        "WHERE physical_attempt_id=? OR physical_attempt_key=?",
        (physical_attempt_id, physical_attempt_key),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeModelCallIdentityCollision(
            "physical model ID and key resolve to different attempts"
        )
    logical = _require_logical_call(conn, str(rows[0]["logical_call_id"]))
    return next(
        item
        for item in logical.physical_attempts
        if item.request.physical_attempt_id == physical_attempt_id
        or item.request.physical_attempt_key == physical_attempt_key
    )


def _require_logical_row_projection(
    row: sqlite3.Row,
    request: RuntimeModelLogicalRequest,
) -> None:
    expected = {
        "logical_call_id": request.logical_call_id,
        "session_id": request.session_id,
        "insession_task_id": request.task_id,
        "auxiliary_graph_id": request.auxiliary_graph_id,
        "goal_id": request.goal_id,
        "execution_subject_id": request.execution_subject_id,
        "invocation_turn_id": request.invocation_turn_id,
        "call_kind": request.call_kind,
        "purpose": request.purpose,
        "provider": request.provider,
        "model": request.model,
        "endpoint_fingerprint": request.endpoint_fingerprint,
        "request_contract": request.request_contract,
        "request_sha256": request.request_sha256,
        "typed_result_contract": request.typed_result_contract,
        "max_physical_attempts": request.max_physical_attempts,
        "state_guard_sha256": request.state_guard_sha256,
        "binding_sha256": request.binding_sha256,
    }
    _require_row_projection(row, expected, label="logical model request")


def _require_physical_row_projection(
    row: sqlite3.Row,
    request: RuntimeModelPhysicalAttemptRequest,
) -> None:
    expected = {
        "physical_attempt_id": request.physical_attempt_id,
        "physical_attempt_key": request.physical_attempt_key,
        "logical_call_id": request.logical_call_id,
        "logical_request_binding_sha256": request.logical_request_binding_sha256,
        "physical_ordinal": request.physical_ordinal,
        "started_turn_id": request.started_turn_id,
        "provider": request.provider,
        "model": request.model,
        "endpoint_fingerprint": request.endpoint_fingerprint,
        "request_sha256": request.request_sha256,
        "provider_idempotency_key": request.provider_idempotency_key,
        "dispatch_authority_sha256": request.dispatch_authority_sha256,
        "binding_sha256": request.binding_sha256,
    }
    _require_row_projection(row, expected, label="physical model request")


def _require_output_repair_settlement_binding(
    *,
    logical: RuntimeModelLogicalRequest,
    physical: RuntimeModelPhysicalAttemptRequest,
    settlement: RuntimeModelPhysicalAttemptSettlement,
) -> None:
    feedback = settlement.next_output_repair_feedback
    repair_bad_response = (
        settlement.outcome is RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
        and settlement.error_code == "MODEL_BAD_RESPONSE"
    )
    if feedback is not None and not physical.output_repair_enabled:
        raise RuntimeModelCallPersistenceError(
            "model settlement repair feedback crossed physical request authority"
        )
    if feedback is not None and feedback.target_contract != logical.output_repair_target_contract:
        raise RuntimeModelCallPersistenceError(
            "model settlement repair target differs from its logical result contract"
        )
    if physical.output_repair_enabled and repair_bad_response and feedback is None:
        raise RuntimeModelCallPersistenceError(
            "repair-enabled model rejection has no durable feedback checkpoint"
        )


def _require_settled_rejected_output_binding(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    physical: RuntimeModelPhysicalAttemptRequest,
    settlement: RuntimeModelPhysicalAttemptSettlement,
) -> None:
    """根据结算信息认证可选正文检查点。"""

    feedback = settlement.next_output_repair_feedback
    row = conn.execute(
        "SELECT * FROM insession_runtime_model_rejected_outputs "
        "WHERE physical_attempt_id=?",
        (physical.physical_attempt_id,),
    ).fetchone()
    if row is None:
        if feedback is not None:
            raise RuntimeModelCallPersistenceError(
                "model rejection has no durable rejected response"
            )
        return
    record = _validate_rejected_output_row(row)
    if (
        feedback is None
        or settlement.outcome
        is not RuntimeModelPhysicalOutcome.RETRYABLE_FAILURE
        or settlement.error_code != "MODEL_BAD_RESPONSE"
        or record.session_id != session_id
        or record.logical_call_id != physical.logical_call_id
        or record.physical_attempt_id != physical.physical_attempt_id
        or record.physical_ordinal != physical.physical_ordinal
        or feedback.rejected_physical_ordinal != physical.physical_ordinal
        or record.response_sha256 != feedback.rejected_response_sha256
    ):
        raise RuntimeModelCallPersistenceError(
            "stored rejected model output crossed settlement authority"
        )


def _require_physical_retry_chain(
    *,
    prior_attempts: tuple[StoredRuntimeModelPhysicalAttempt, ...],
    request: RuntimeModelPhysicalAttemptRequest,
) -> None:
    """将每个物理 Prompt 变体绑定到其前序不可变回执。"""

    if request.output_repair_enabled and request.provider_idempotency_key is not None:
        raise RuntimeModelCallPersistenceError(
            "repair-enabled model request cannot reuse one Provider idempotency key"
        )
    if not prior_attempts:
        if request.output_repair_feedback is not None:
            raise RuntimeModelCallPersistenceError(
                "first physical model attempt cannot carry repair feedback"
            )
        return

    previous = prior_attempts[-1]
    settlement = previous.settlement
    if settlement is None:
        raise RuntimeModelCallPersistenceError(
            "physical retry has no preceding immutable settlement"
        )
    expected_feedback = (
        settlement.next_output_repair_feedback
        if settlement.error_code == "MODEL_BAD_RESPONSE"
        else previous.request.output_repair_feedback
    )
    if request.output_repair_feedback != expected_feedback:
        raise RuntimeModelCallPersistenceError(
            "physical retry changed its authorized output-repair feedback"
    )
    previous_repair_enabled = bool(previous.request.output_repair_enabled)
    current_repair_enabled = bool(request.output_repair_enabled)
    if previous_repair_enabled != current_repair_enabled:
        raise RuntimeModelCallPersistenceError(
            "physical retry changed its durable output-repair mode"
        )
    if (
        request.output_repair_feedback is not None
        and request.output_repair_feedback.rejected_physical_ordinal
        >= request.physical_ordinal
    ):
        raise RuntimeModelCallPersistenceError(
            "physical retry feedback does not reference an earlier attempt"
        )


def _require_settlement_row_projection(
    row: sqlite3.Row,
    settlement: RuntimeModelPhysicalAttemptSettlement,
) -> None:
    expected = {
        "settle_apply_id": settlement.settle_apply_id,
        "settlement_id": settlement.settlement_id,
        "logical_call_id": settlement.logical_call_id,
        "physical_attempt_id": settlement.physical_attempt_id,
        "physical_request_binding_sha256": (
            settlement.physical_request_binding_sha256
        ),
        "physical_ordinal": settlement.physical_ordinal,
        "outcome": settlement.outcome.value,
        "receipt_sha256": settlement.receipt_sha256,
        "settled_turn_id": settlement.settled_turn_id,
    }
    _require_row_projection(row, expected, label="physical model settlement")


def _require_row_projection(
    row: sqlite3.Row,
    expected: dict[str, object],
    *,
    label: str,
) -> None:
    for column, value in expected.items():
        if row[column] != value:
            raise RuntimeModelCallPersistenceError(
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
    "RuntimeModelCallIdentityCollision",
    "RuntimeModelCallPersistenceError",
    "RuntimeModelCallTerminalState",
    'RuntimeModelLedgerMutationResult',
    'StoredRuntimeModelLogicalCall',
    'StoredRuntimeModelPhysicalAttempt',
    'StoredRuntimeModelRejectedOutput',
    "append_runtime_model_physical_attempt",
    "get_runtime_model_logical_call",
    "get_runtime_model_rejected_output",
    "reserve_runtime_model_logical_call",
    "settle_runtime_model_physical_attempt",
]
