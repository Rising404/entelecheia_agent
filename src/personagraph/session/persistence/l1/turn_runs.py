"""由一个 Turn 持有的有界 L1 执行聚合的持久化记录。"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import sqlite3

from personagraph.persistent_turn_content.delivery import build_l1_terminal_notification
from personagraph.runtime.l1.corpus_contracts import (
    L1_CORPUS_MANIFEST_CONTRACT_VERSION,
    load_l1_corpus_manifest,
)
from personagraph.runtime.l1.semantic_contracts import (
    L1_SEMANTIC_RESULT_CONTRACT,
    L1_SEMANTIC_VERIFICATION_FEATURE,
    L1_SEMANTIC_VERIFICATION_RECEIPT_VERSION,
    L1SemanticVerificationReceipt,
    derive_l1_semantic_verification_trigger,
    parse_l1_semantic_verification_mode,
)
from ..deps import StoreDeps
from ..execution_findings import (
    ExecutionFindingsPersistenceError,
    create_execution_findings_owner_companion_in_transaction,
)
from ...turn_execution_contracts import TurnExecutionWindowRevisionConflict


class L1TurnRunPersistenceError(RuntimeError):
    """L1 聚合无法绑定到其 AcceptedTurn 权威状态。"""


_L1_VERIFICATION_CONTRACT_VERSION = "l1-final-reply-verification"


def create_l1_turn_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    routing_policy_snapshot_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None = None,
    l1_turn_run_id: str | None = None,
    deadline_at: str | None = None,
    max_attempts: int | None = None,
    max_tool_calls_per_attempt: int | None = None,
    catalog_snapshot_json: str | None = None,
    catalog_snapshot_hash: str | None = None,
    execution_config_json: str | None = None,
    execution_config_hash: str | None = None,
    corpus_manifest_contract_version: str | None = None,
    corpus_manifest_json: str | None = None,
    corpus_manifest_hash: str | None = None,
) -> dict[str, object]:
    """创建或重放一个 L1 聚合，并可选择以原子方式初始化。"""

    max_semantic_steps = max_attempts
    max_tool_calls_per_step = max_tool_calls_per_attempt

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if len(routing_policy_snapshot_hash) != 64:
        raise ValueError("routing_policy_snapshot_hash must be sha256")
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)
    if l1_turn_run_id is not None:
        _require_identifier("l1_turn_run_id", l1_turn_run_id)
    initialization = (
        deadline_at,
        max_semantic_steps,
        max_tool_calls_per_step,
        catalog_snapshot_json,
        catalog_snapshot_hash,
        execution_config_json,
        execution_config_hash,
        corpus_manifest_contract_version,
        corpus_manifest_json,
        corpus_manifest_hash,
    )
    initialize = any(value is not None for value in initialization)
    if initialize and not all(value is not None for value in initialization):
        raise ValueError("L1 atomic initialization fields must be supplied together")
    if initialize:
        assert deadline_at is not None
        assert max_semantic_steps is not None
        assert max_tool_calls_per_step is not None
        assert catalog_snapshot_json is not None
        assert catalog_snapshot_hash is not None
        assert execution_config_json is not None
        assert execution_config_hash is not None
        assert corpus_manifest_contract_version is not None
        assert corpus_manifest_json is not None
        assert corpus_manifest_hash is not None
        if l1_turn_run_id is None:
            raise ValueError(
                "atomic L1 initialization requires a replay-stable l1_turn_run_id"
            )
        _validate_execution_limits(
            deadline_at=deadline_at,
            max_semantic_steps=max_semantic_steps,
            max_tool_calls_per_step=max_tool_calls_per_step,
        )
        _validate_canonical_json_hash(
            catalog_snapshot_json,
            catalog_snapshot_hash,
            label="L1 catalog snapshot",
        )
        _validate_canonical_json_hash(
            execution_config_json,
            execution_config_hash,
            label="L1 execution configuration",
        )
        _validate_l1_corpus_manifest(
            contract_version=corpus_manifest_contract_version,
            payload=corpus_manifest_json,
            payload_hash=corpus_manifest_hash,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
        )

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        turn = conn.execute(
            "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise L1TurnRunPersistenceError("L1 TurnRun has no owning Runtime Turn")
        if str(turn["status"]) != "running":
            raise L1TurnRunPersistenceError("L1 TurnRun requires a running Turn")
        snapshot = conn.execute(
            "SELECT snapshot_hash FROM runtime_turn_routing_policy_snapshots "
            "WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if (
            snapshot is None
            or str(snapshot["snapshot_hash"]) != routing_policy_snapshot_hash
        ):
            raise L1TurnRunPersistenceError(
                "L1 TurnRun routing-policy authority is missing or stale"
            )

        existing = conn.execute(
            "SELECT * FROM l1_turn_runs WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        window = conn.execute(
            "SELECT * FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if window is None or str(window["turn_id"] or "") != turn_id:
            raise L1TurnRunPersistenceError("Turn does not own the active Window")
        actual_revision = int(window["state_version"])
        if str(window["window_state"]) != "active":
            raise L1TurnRunPersistenceError("L1 TurnRun requires an active Window")
        if (
            expected_lease_owner is not None
            and str(window["lease_owner"] or "") != expected_lease_owner
        ):
            raise L1TurnRunPersistenceError(
                "L1 TurnRun requires the current Window lease owner"
            )

        if existing is not None:
            if (
                str(existing["session_id"]) != session_id
                or str(existing["routing_policy_snapshot_hash"])
                != routing_policy_snapshot_hash
            ):
                raise L1TurnRunPersistenceError("L1 TurnRun replay facts changed")
            if (
                l1_turn_run_id is not None
                and str(existing["l1_turn_run_id"]) != l1_turn_run_id
            ):
                raise L1TurnRunPersistenceError("L1 TurnRun replay identity changed")
            if str(window["current_l1_turn_run_id"] or "") != str(
                existing["l1_turn_run_id"]
            ):
                raise L1TurnRunPersistenceError(
                    "L1 TurnRun replay lost its Window binding"
                )
            if actual_revision not in {
                expected_window_revision,
                expected_window_revision + 1,
            }:
                raise TurnExecutionWindowRevisionConflict(
                    expected=expected_window_revision,
                    actual=actual_revision,
                )
            state = conn.execute(
                "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
                (str(existing["l1_turn_run_id"]),),
            ).fetchone()
            if initialize:
                assert deadline_at is not None
                assert max_semantic_steps is not None
                assert max_tool_calls_per_step is not None
                assert catalog_snapshot_json is not None
                assert catalog_snapshot_hash is not None
                assert execution_config_json is not None
                assert execution_config_hash is not None
                assert corpus_manifest_contract_version is not None
                assert corpus_manifest_json is not None
                assert corpus_manifest_hash is not None
                if state is None:
                    if str(existing["status"]) != "created":
                        raise L1TurnRunPersistenceError(
                            "running L1 TurnRun replay has no executable state"
                        )
                    conn.execute(
                        "INSERT INTO l1_turn_run_states "
                        "(l1_turn_run_id, session_id, turn_id, stage, deadline_at, "
                        "max_semantic_steps, max_tool_calls_per_step, "
                        "semantic_steps_used, catalog_snapshot_json, "
                        "catalog_snapshot_hash, execution_config_json, "
                        "execution_config_hash, corpus_manifest_contract_version, "
                        "corpus_manifest_json, corpus_manifest_hash, updated_at) "
                        "VALUES (?, ?, ?, 'bootstrap', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(existing["l1_turn_run_id"]),
                            session_id,
                            turn_id,
                            deadline_at,
                            max_semantic_steps,
                            max_tool_calls_per_step,
                            catalog_snapshot_json,
                            catalog_snapshot_hash,
                            execution_config_json,
                            execution_config_hash,
                            corpus_manifest_contract_version,
                            corpus_manifest_json,
                            corpus_manifest_hash,
                            now,
                        ),
                    )
                    conn.execute(
                        "UPDATE l1_turn_runs SET status='running', "
                        "run_revision=run_revision+1, updated_at=? "
                        "WHERE l1_turn_run_id=? AND status='created'",
                        (now, str(existing["l1_turn_run_id"])),
                    )
                    next_revision = actual_revision + 1
                    _advance_l1_window(
                        conn,
                        session_id=session_id,
                        turn_id=turn_id,
                        l1_turn_run_id=str(existing["l1_turn_run_id"]),
                        current_l1_attempt_id=None,
                        stage="L1_BOOTSTRAP",
                        expected_window_revision=actual_revision,
                        next_window_revision=next_revision,
                        expected_lease_owner=expected_lease_owner,
                        now=now,
                    )
                    return _execution_projection(
                        conn,
                        run=_require_run(conn, str(existing["l1_turn_run_id"])),
                        state=_require_state(conn, str(existing["l1_turn_run_id"])),
                        window=_require_window(conn, session_id, turn_id),
                        replayed=False,
                    )
                expected_initialization = {
                    "deadline_at": deadline_at,
                    "max_semantic_steps": max_semantic_steps,
                    "max_tool_calls_per_step": max_tool_calls_per_step,
                    "catalog_snapshot_json": catalog_snapshot_json,
                    "catalog_snapshot_hash": catalog_snapshot_hash,
                    "execution_config_json": execution_config_json,
                    "execution_config_hash": execution_config_hash,
                    "corpus_manifest_contract_version": (
                        corpus_manifest_contract_version
                    ),
                    "corpus_manifest_json": corpus_manifest_json,
                    "corpus_manifest_hash": corpus_manifest_hash,
                }
                if any(
                    state[key] != value
                    for key, value in expected_initialization.items()
                ):
                    raise L1TurnRunPersistenceError(
                        "L1 TurnRun replay changed immutable initialization facts"
                    )
            return {
                "replayed": True,
                "run": _run_projection(existing),
                "window": dict(window),
                "state": (
                    _state_projection(state) if state is not None else None
                ),
            }

        if actual_revision != expected_window_revision:
            raise TurnExecutionWindowRevisionConflict(
                expected=expected_window_revision,
                actual=actual_revision,
            )

        if any(
            window[field] is not None
            for field in (
                "current_work_run_id",
                "current_attempt_id",
                "current_l1_turn_run_id",
                "current_l1_attempt_id",
            )
        ):
            raise L1TurnRunPersistenceError(
                "Turn Window is already bound to another execution aggregate"
            )
        assigned_l1_turn_run_id = l1_turn_run_id or f"l1run_{deps.new_id()}"
        if initialize:
            conn.execute(
                "INSERT INTO l1_turn_runs "
                "(l1_turn_run_id, session_id, turn_id, status, "
                "routing_policy_snapshot_hash, run_revision, created_at, "
                "updated_at) VALUES (?, ?, ?, 'running', ?, 1, ?, ?)",
                (
                    assigned_l1_turn_run_id,
                    session_id,
                    turn_id,
                    routing_policy_snapshot_hash,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO l1_turn_run_states "
                "(l1_turn_run_id, session_id, turn_id, stage, deadline_at, "
                "max_semantic_steps, max_tool_calls_per_step, "
                "semantic_steps_used, catalog_snapshot_json, "
                "catalog_snapshot_hash, execution_config_json, "
                "execution_config_hash, corpus_manifest_contract_version, "
                "corpus_manifest_json, corpus_manifest_hash, updated_at) "
                "VALUES (?, ?, ?, 'bootstrap', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    assigned_l1_turn_run_id,
                    session_id,
                    turn_id,
                    deadline_at,
                    max_semantic_steps,
                    max_tool_calls_per_step,
                    catalog_snapshot_json,
                    catalog_snapshot_hash,
                    execution_config_json,
                    execution_config_hash,
                    corpus_manifest_contract_version,
                    corpus_manifest_json,
                    corpus_manifest_hash,
                    now,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO l1_turn_runs "
                "(l1_turn_run_id, session_id, turn_id, status, "
                "routing_policy_snapshot_hash, run_revision, created_at, "
                "updated_at) VALUES (?, ?, ?, 'created', ?, 0, ?, ?)",
                (
                    assigned_l1_turn_run_id,
                    session_id,
                    turn_id,
                    routing_policy_snapshot_hash,
                    now,
                    now,
                ),
            )
        try:
            create_execution_findings_owner_companion_in_transaction(
                conn,
                session_id=session_id,
                owner_kind="l1_turn_run",
                execution_owner_id=assigned_l1_turn_run_id,
                now=now,
            )
        except ExecutionFindingsPersistenceError as exc:
            raise L1TurnRunPersistenceError(
                "L1 TurnRun findings companion could not be created"
            ) from exc
        next_revision = actual_revision + 1
        updated = conn.execute(
            "UPDATE turn_execution_windows SET current_l1_turn_run_id=?, "
            "current_l1_attempt_id=NULL, stage='L1_BOOTSTRAP', state_version=?, "
            "heartbeat_at=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND state_version=? "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND current_l1_turn_run_id IS NULL AND current_l1_attempt_id IS NULL",
            (
                assigned_l1_turn_run_id,
                next_revision,
                now,
                now,
                session_id,
                turn_id,
                actual_revision,
            ),
        ).rowcount
        if not updated:
            raise L1TurnRunPersistenceError("Turn Window changed during L1 binding")
        state = conn.execute(
            "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
            (assigned_l1_turn_run_id,),
        ).fetchone()
        run = _require_run(conn, assigned_l1_turn_run_id)
        bound_window = _require_window(conn, session_id, turn_id)
        return {
            "replayed": False,
            "run": _run_projection(run),
            "window": dict(bound_window),
            "state": (
                _state_projection(state) if state is not None else None
            ),
        }


def get_l1_turn_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object] | None:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT * FROM l1_turn_runs WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
    return _run_projection(row) if row is not None else None


def claim_l1_turn_run_resume(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    lease_owner: str,
    lease_seconds: int,
) -> dict[str, object]:
    """认领尚未完成交付的精确 L1 lane，包括失败后的通知补发。

    request-id 重放和进程内 Session 守卫位于此边界之上。存储层只证明持久化标识、
    lane 排他性和租约新鲜度。租约转移有意不推进 Window 状态版本：已准备的模型调用
    守卫包含该语义 revision，而更改执行所有者是正交的隔离操作。
    失败运行只返回通知交付权，不改变失败状态、重置预算或授权重新执行模型。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("lease_owner", lease_owner)
    if isinstance(lease_seconds, bool) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        window = conn.execute(
            "SELECT * FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        ).fetchone()

        def not_resumable(reason: str) -> dict[str, object]:
            return {
                "status": "not_resumable",
                "reason": reason,
                "window": dict(window) if window is not None else None,
            }

        if window is None or str(window["turn_id"] or "") != turn_id:
            return not_resumable("turn_does_not_own_window")
        if (
            str(window["window_state"] or "") != "active"
            or window["interruption_reason"] is not None
        ):
            return not_resumable("window_is_not_uninterrupted_active")
        if any(
            window[field] is not None
            for field in (
                "current_work_run_id",
                "current_attempt_id",
                "latest_checkpoint_id",
                "pending_operation_id",
            )
        ):
            return not_resumable("window_is_bound_to_another_execution_lane")

        l1_turn_run_id = str(window["current_l1_turn_run_id"] or "")
        if not l1_turn_run_id:
            return not_resumable("window_has_no_l1_run")
        turn = conn.execute(
            "SELECT * FROM runtime_turns WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if turn is None or str(turn["status"] or "") != "running":
            return not_resumable("turn_is_not_running")
        if str(turn["processing_level"] or "") not in {"", "L1"}:
            return not_resumable("turn_processing_level_conflicts_with_l1")

        run = conn.execute(
            "SELECT * FROM l1_turn_runs WHERE session_id=? AND turn_id=? "
            "AND l1_turn_run_id=?",
            (session_id, turn_id, l1_turn_run_id),
        ).fetchone()
        if run is None or str(run["status"] or "") not in {"created", "running", "failed"}:
            return not_resumable("l1_run_is_not_resumable")
        snapshot = conn.execute(
            "SELECT snapshot_hash FROM runtime_turn_routing_policy_snapshots "
            "WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if (
            snapshot is None
            or str(snapshot["snapshot_hash"] or "")
            != str(run["routing_policy_snapshot_hash"] or "")
        ):
            return not_resumable("l1_routing_authority_changed")

        state = conn.execute(
            "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
            (l1_turn_run_id,),
        ).fetchone()
        terminal_failure_code: str | None = None
        if str(run["status"]) == "failed":
            if state is None or str(state["stage"] or "") != "failed":
                return not_resumable("l1_failed_run_has_no_terminal_state")
            failure_code = str(state["failure_code"] or "")
            if build_l1_terminal_notification(failure_code) is None:
                return not_resumable("l1_failure_has_no_terminal_notification")
            terminal_failure_code = failure_code
        elif state is None:
            if str(run["status"]) != "created":
                return not_resumable("running_l1_run_has_no_state")
        elif (
            str(run["status"]) != "running"
            or str(state["stage"] or "")
            not in {"bootstrap", "model", "tool", "observation", "finalizing"}
        ):
            return not_resumable("l1_state_is_not_resumable")

        current_owner = _optional_text(window["lease_owner"])
        if current_owner not in {None, lease_owner} and _heartbeat_is_fresh(
            _optional_text(window["heartbeat_at"]),
            now=now,
            lease_seconds=lease_seconds,
        ):
            return {
                "status": "busy",
                "reason": "fresh_lease_owner",
                "window": dict(window),
                "turn": dict(turn),
                "run": dict(run),
            }

        updated = conn.execute(
            "UPDATE turn_execution_windows SET lease_owner=?, heartbeat_at=?, "
            "claimed_at=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND state_version=? "
            "AND current_l1_turn_run_id=? "
            "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
            "AND interruption_reason IS NULL",
            (
                lease_owner,
                now,
                now,
                now,
                session_id,
                turn_id,
                int(window["state_version"]),
                l1_turn_run_id,
            ),
        ).rowcount
        if updated != 1:
            raise L1TurnRunPersistenceError(
                "L1 TurnRun Window changed during resume claim"
            )
        pending_tool_call = conn.execute(
            "SELECT 1 FROM l1_turn_tool_calls WHERE l1_turn_run_id=? "
            "AND status='pending' LIMIT 1",
            (l1_turn_run_id,),
        ).fetchone()
        return {
            "status": "applied",
            "window": dict(
                conn.execute(
                    "SELECT * FROM turn_execution_windows WHERE session_id=?",
                    (session_id,),
                ).fetchone()
            ),
            "turn": dict(turn),
            "run": dict(run),
            "state": dict(state) if state is not None else None,
            "has_unconfirmed_tool_call": pending_tool_call is not None,
            "terminal_failure_code": terminal_failure_code,
        }


def renew_l1_turn_run_resume_lease(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    lease_owner: str,
) -> bool:
    """为一个精确 L1 重放所有者续约心跳，而不改变语义状态。"""

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    _require_identifier("l1_turn_run_id", l1_turn_run_id)
    _require_identifier("lease_owner", lease_owner)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return bool(
            conn.execute(
                "UPDATE turn_execution_windows SET heartbeat_at=?, updated_at=? "
                "WHERE session_id=? AND turn_id=? AND window_state='active' "
                "AND current_l1_turn_run_id=? AND lease_owner=? "
                "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
                "AND interruption_reason IS NULL",
                (
                    now,
                    now,
                    session_id,
                    turn_id,
                    l1_turn_run_id,
                    lease_owner,
                ),
            ).rowcount
        )


def initialize_l1_turn_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    deadline_at: str,
    max_attempts: int,
    max_tool_calls_per_attempt: int,
    catalog_snapshot_json: str,
    catalog_snapshot_hash: str,
    execution_config_json: str,
    execution_config_hash: str,
    corpus_manifest_contract_version: str,
    corpus_manifest_json: str,
    corpus_manifest_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> dict[str, object]:
    """初始化或重放 P1 run 标识之下的可执行状态。"""

    max_semantic_steps = max_attempts
    max_tool_calls_per_step = max_tool_calls_per_attempt

    _validate_execution_limits(
        deadline_at=deadline_at,
        max_semantic_steps=max_semantic_steps,
        max_tool_calls_per_step=max_tool_calls_per_step,
    )
    _validate_canonical_json_hash(
        catalog_snapshot_json,
        catalog_snapshot_hash,
        label="L1 catalog snapshot",
    )
    _validate_canonical_json_hash(
        execution_config_json,
        execution_config_hash,
        label="L1 execution configuration",
    )
    _validate_l1_corpus_manifest(
        contract_version=corpus_manifest_contract_version,
        payload=corpus_manifest_json,
        payload_hash=corpus_manifest_hash,
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=l1_turn_run_id,
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        existing = conn.execute(
            "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
            (l1_turn_run_id,),
        ).fetchone()
        if existing is not None:
            expected = {
                "session_id": session_id,
                "turn_id": turn_id,
                "deadline_at": deadline_at,
                "max_semantic_steps": max_semantic_steps,
                "max_tool_calls_per_step": max_tool_calls_per_step,
                "catalog_snapshot_json": catalog_snapshot_json,
                "catalog_snapshot_hash": catalog_snapshot_hash,
                "execution_config_json": execution_config_json,
                "execution_config_hash": execution_config_hash,
                "corpus_manifest_contract_version": (
                    corpus_manifest_contract_version
                ),
                "corpus_manifest_json": corpus_manifest_json,
                "corpus_manifest_hash": corpus_manifest_hash,
            }
            if any(existing[key] != value for key, value in expected.items()):
                raise L1TurnRunPersistenceError(
                    "L1 executable-state replay changed immutable initialization facts"
                )
            return _execution_projection(
                conn,
                run=run,
                state=existing,
                window=window,
                replayed=True,
            )
        if str(run["status"]) != "created":
            raise L1TurnRunPersistenceError(
                "only a created L1 TurnRun can initialize executable state"
            )
        conn.execute(
            "INSERT INTO l1_turn_run_states "
            "(l1_turn_run_id, session_id, turn_id, stage, deadline_at, "
            "max_semantic_steps, max_tool_calls_per_step, semantic_steps_used, "
            "catalog_snapshot_json, catalog_snapshot_hash, execution_config_json, "
            "execution_config_hash, corpus_manifest_contract_version, "
            "corpus_manifest_json, corpus_manifest_hash, updated_at) "
            "VALUES (?, ?, ?, 'bootstrap', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                l1_turn_run_id,
                session_id,
                turn_id,
                deadline_at,
                max_semantic_steps,
                max_tool_calls_per_step,
                catalog_snapshot_json,
                catalog_snapshot_hash,
                execution_config_json,
                execution_config_hash,
                corpus_manifest_contract_version,
                corpus_manifest_json,
                corpus_manifest_hash,
                now,
            ),
        )
        conn.execute(
            "UPDATE l1_turn_runs SET status='running', run_revision=run_revision+1, "
            "updated_at=? WHERE l1_turn_run_id=? AND status='created'",
            (now, l1_turn_run_id),
        )
        next_window_revision = int(window["state_version"]) + 1
        _advance_l1_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            current_l1_attempt_id=None,
            stage="L1_BOOTSTRAP",
            expected_window_revision=int(window["state_version"]),
            next_window_revision=next_window_revision,
            expected_lease_owner=expected_lease_owner,
            now=now,
        )
        return _execution_projection(
            conn,
            run=_require_run(conn, l1_turn_run_id),
            state=_require_state(conn, l1_turn_run_id),
            window=_require_window(conn, session_id, turn_id),
            replayed=False,
        )


def start_l1_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    request_json: str,
    request_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> dict[str, object]:
    """启动一次 Attempt，并在 I/O 前冻结含稳定步骤身份的精确请求。

    调用方提交尚未绑定步骤身份的候选 model view。Store 在持有运行事务时以
    ``run + ordinal`` 派生 ID，将 ID/序号写入同一份冻结 request，再计算最终 hash。
    因而 request hash 不参与自身 ID 的生成；响应丢失重放仍逐字比较同一权威。
    """

    request_candidate = _validate_canonical_json_hash(
        request_json,
        request_hash,
        label="L1 model request",
    )
    if not isinstance(request_candidate, dict):
        raise ValueError("L1 model request must be a JSON object")
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        state = _require_state(conn, l1_turn_run_id)
        if str(run["status"]) != "running" or str(state["stage"]) in {
            "finalizing",
            "completed",
            "failed",
            "cancelled",
        }:
            raise L1TurnRunPersistenceError("L1 TurnRun cannot open another Attempt")
        semantic_steps_used = int(state["semantic_steps_used"])
        if semantic_steps_used >= int(state["max_semantic_steps"]):
            raise L1TurnRunPersistenceError("L1 Attempt limit is exhausted")
        ordinal = semantic_steps_used + 1
        existing = conn.execute(
            "SELECT * FROM l1_turn_steps WHERE l1_turn_run_id=? AND ordinal=?",
            (l1_turn_run_id, ordinal),
        ).fetchone()
        attempt_id = _stable_id(
            "l1attempt",
            {
                "l1_turn_run_id": l1_turn_run_id,
                "ordinal": ordinal,
            },
        )
        frozen_request = dict(request_candidate)
        supplied_attempt_id = frozen_request.get("attempt_id")
        supplied_attempt_ordinal = frozen_request.get("attempt_ordinal")
        if supplied_attempt_id is not None and supplied_attempt_id != attempt_id:
            raise L1TurnRunPersistenceError(
                "L1 model request carries another Attempt identity"
            )
        if (
            supplied_attempt_ordinal is not None
            and supplied_attempt_ordinal != ordinal
        ):
            raise L1TurnRunPersistenceError(
                "L1 model request carries another Attempt ordinal"
            )
        frozen_request.update(
            attempt_id=attempt_id,
            attempt_ordinal=ordinal,
        )
        frozen_request_json = _canonical_json(frozen_request)
        frozen_request_hash = hashlib.sha256(
            frozen_request_json.encode("utf-8")
        ).hexdigest()
        if existing is not None:
            # 旧冻结行没有显式步骤字段；只允许其原始候选逐字重放，不升级或改写。
            legacy_exact_replay = (
                str(existing["request_json"]) == request_json
                and str(existing["request_hash"]) == request_hash
            )
            if (
                str(existing["status"]) != "prepared"
                or not (
                    legacy_exact_replay
                    or (
                        str(existing["step_id"]) == attempt_id
                        and str(existing["request_json"]) == frozen_request_json
                        and str(existing["request_hash"]) == frozen_request_hash
                    )
                )
            ):
                raise L1TurnRunPersistenceError(
                    "L1 Attempt ordinal already has different authority"
                )
            return {
                **_execution_projection(
                    conn,
                    run=run,
                    state=state,
                    window=window,
                    replayed=True,
                ),
                "attempt": _attempt_projection(existing),
            }
        request_json = frozen_request_json
        request_hash = frozen_request_hash
        logical_model_call_id = _stable_id(
            "l1model",
            {"attempt_id": attempt_id, "request_hash": request_hash},
        )
        next_run_revision = int(run["run_revision"]) + 1
        next_window_revision = int(window["state_version"]) + 1
        findings_ledger_id, findings_revision = _l1_findings_guard_facts(
            conn,
            l1_turn_run_id=l1_turn_run_id,
        )
        state_guard_hash = _l1_state_guard_hash(
            l1_turn_run_id=l1_turn_run_id,
            step_id=attempt_id,
            ordinal=ordinal,
            run_revision=next_run_revision,
            window_revision=next_window_revision,
            semantic_steps_used=semantic_steps_used,
            plan_hash=_optional_text(state["plan_hash"]),
            observation_hash=_optional_text(state["latest_observation_hash"]),
            catalog_snapshot_hash=str(state["catalog_snapshot_hash"]),
            execution_config_hash=str(state["execution_config_hash"]),
            corpus_manifest_hash=_optional_text(state["corpus_manifest_hash"]),
            execution_findings_ledger_id=findings_ledger_id,
            execution_findings_revision=findings_revision,
            request_hash=request_hash,
        )
        conn.execute(
            "INSERT INTO l1_turn_steps "
            "(step_id, l1_turn_run_id, session_id, turn_id, ordinal, status, "
            "logical_model_call_id, request_json, request_hash, state_guard_hash, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                l1_turn_run_id,
                session_id,
                turn_id,
                ordinal,
                logical_model_call_id,
                request_json,
                request_hash,
                state_guard_hash,
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE l1_turn_runs SET run_revision=?, updated_at=? "
            "WHERE l1_turn_run_id=? AND run_revision=?",
            (next_run_revision, now, l1_turn_run_id, int(run["run_revision"])),
        )
        _advance_l1_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            current_l1_attempt_id=attempt_id,
            stage="L1_BOOTSTRAP",
            expected_window_revision=int(window["state_version"]),
            next_window_revision=next_window_revision,
            expected_lease_owner=expected_lease_owner,
            now=now,
        )
        return {
            **_execution_projection(
                conn,
                run=_require_run(conn, l1_turn_run_id),
                state=_require_state(conn, l1_turn_run_id),
                window=_require_window(conn, session_id, turn_id),
                replayed=False,
            ),
            "attempt": _attempt_projection(_require_attempt(conn, attempt_id)),
        }


def get_l1_attempt_state_guard(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
) -> str:
    """围绕一次模型调用重新推导精确的决策前权威状态。"""

    deps.init_db()
    with deps.connect() as conn:
        run = _require_run(conn, l1_turn_run_id)
        state = _require_state(conn, l1_turn_run_id)
        attempt = _require_attempt(conn, attempt_id)
        window = _require_window(conn, session_id, turn_id)
        if (
            str(run["session_id"]) != session_id
            or str(run["turn_id"]) != turn_id
            or str(run["status"]) != "running"
            or str(attempt["l1_turn_run_id"]) != l1_turn_run_id
            or str(attempt["status"]) != "prepared"
            or str(window["current_l1_turn_run_id"] or "") != l1_turn_run_id
            or str(window["current_l1_attempt_id"] or "") != attempt_id
            or str(window["window_state"]) != "active"
        ):
            raise L1TurnRunPersistenceError("L1 model state guard is no longer current")
        findings_ledger_id, findings_revision = _l1_findings_guard_facts(
            conn,
            l1_turn_run_id=l1_turn_run_id,
        )
        actual = _l1_state_guard_hash(
            l1_turn_run_id=l1_turn_run_id,
            step_id=attempt_id,
            ordinal=int(attempt["ordinal"]),
            run_revision=int(run["run_revision"]),
            window_revision=int(window["state_version"]),
            semantic_steps_used=int(state["semantic_steps_used"]),
            plan_hash=_optional_text(state["plan_hash"]),
            observation_hash=_optional_text(state["latest_observation_hash"]),
            catalog_snapshot_hash=str(state["catalog_snapshot_hash"]),
            execution_config_hash=str(state["execution_config_hash"]),
            corpus_manifest_hash=_optional_text(state["corpus_manifest_hash"]),
            execution_findings_ledger_id=findings_ledger_id,
            execution_findings_revision=findings_revision,
            request_hash=str(attempt["request_hash"]),
        )
        if actual != str(attempt["state_guard_hash"]):
            raise L1TurnRunPersistenceError("L1 model state guard hash changed")
        return actual


def commit_l1_attempt_decision(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    state_guard_hash: str,
    decision_json: str,
    decision_hash: str,
    action_kind: str,
    tool_call_count: int,
    plan_json: str | None,
    plan_hash: str | None,
    final_reply: str | None,
    verification_contract_version: str | None,
    verification_report_json: str | None,
    verification_report_hash: str | None,
    semantic_verification_contract_version: str | None,
    semantic_verification_report_json: str | None,
    semantic_verification_report_hash: str | None,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> dict[str, object]:
    """接受一个已验证的 AttemptDecision，并推进持久化游标。"""

    # SQLite 仍以 final_answer 表示已接受的终态 Attempt；对上层只投影
    # 当前的 submit_final_reply 动作名，不把物理枚举泄漏回 Runtime。
    step_id = attempt_id
    physical_action_kind = (
        "final_answer"
        if action_kind == "submit_final_reply"
        else action_kind
    )

    if action_kind not in {"call_tools", "submit_final_reply"}:
        raise ValueError("unsupported L1 action_kind")
    if isinstance(tool_call_count, bool) or not 0 <= tool_call_count <= 16:
        raise ValueError("tool_call_count must be within 0..16")
    if action_kind == "call_tools" and tool_call_count < 1:
        raise ValueError("call_tools requires at least one call")
    if action_kind == "submit_final_reply" and tool_call_count != 0:
        raise ValueError("submit_final_reply cannot contain ToolCalls")
    decision = _validate_canonical_json_hash(
        decision_json,
        decision_hash,
        label="L1 decision",
    )
    if (
        not isinstance(decision, dict)
        or not isinstance(decision.get("action"), dict)
        or decision["action"].get("kind") != action_kind
    ):
        raise ValueError("L1 decision action.kind does not match action_kind")
    _validate_optional_json_pair(plan_json, plan_hash, label="L1 Plan")
    _validate_optional_json_pair(
        verification_report_json,
        verification_report_hash,
        label="L1 verification report",
    )
    _validate_optional_json_pair(
        semantic_verification_report_json,
        semantic_verification_report_hash,
        label="L1 semantic verification report",
    )
    verification_report: dict[str, object] | None = None
    semantic_verification_report: L1SemanticVerificationReceipt | None = None
    if action_kind == "submit_final_reply":
        if not isinstance(final_reply, str) or not final_reply.strip():
            raise ValueError("submit_final_reply requires a non-empty reply")
        if decision["action"].get("reply") != final_reply:
            raise ValueError(
                "submitted final_reply does not match the canonical decision"
            )
        if (
            not isinstance(verification_contract_version, str)
            or not verification_contract_version.strip()
            or verification_report_json is None
        ):
            raise ValueError(
                "submit_final_reply requires a verification receipt"
            )
        raw_verification_report = json.loads(verification_report_json)
        if (
            not isinstance(raw_verification_report, dict)
            or verification_contract_version
            != _L1_VERIFICATION_CONTRACT_VERSION
            or raw_verification_report.get("contract_version")
            != verification_contract_version
            or raw_verification_report.get("passed") is not True
            or raw_verification_report.get("issues") != []
        ):
            raise ValueError("verification receipt is not an accepted pass")
        verification_report = raw_verification_report
        if (
            semantic_verification_contract_version
            != L1_SEMANTIC_VERIFICATION_RECEIPT_VERSION
            or semantic_verification_report_json is None
        ):
            raise ValueError(
                "submit_final_reply requires a semantic verification receipt"
            )
        try:
            semantic_verification_report = (
                L1SemanticVerificationReceipt.model_validate_json(
                    semantic_verification_report_json
                )
            )
        except ValueError as exc:
            raise ValueError(
                "semantic verification receipt has the wrong contract"
            ) from exc
        if (
            semantic_verification_report.contract_version
            != semantic_verification_contract_version
            or _canonical_json(
                semantic_verification_report.model_dump(mode="json")
            )
            != semantic_verification_report_json
        ):
            raise ValueError("semantic verification receipt is not canonical")
    elif any(
        value is not None
        for value in (
            final_reply,
            verification_contract_version,
            verification_report_json,
            semantic_verification_contract_version,
            semantic_verification_report_json,
        )
    ):
        raise ValueError("call_tools cannot persist final delivery fields")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        state = _require_state(conn, l1_turn_run_id)
        step = _require_attempt(conn, step_id)
        if str(step["status"]) != "prepared":
            if (
                str(step["decision_hash"] or "") == decision_hash
                and str(step["action_kind"] or "") == physical_action_kind
            ):
                return {
                    **_execution_projection(
                        conn,
                        run=run,
                        state=state,
                        window=window,
                        replayed=True,
                    ),
                    "attempt": _attempt_projection(step),
                }
            raise L1TurnRunPersistenceError(
                "L1 Attempt already committed another decision"
            )
        if str(step["state_guard_hash"]) != state_guard_hash:
            raise L1TurnRunPersistenceError("L1 decision crossed its state guard")
        actual_guard = get_l1_attempt_state_guard_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            step=step,
            run=run,
            state=state,
            window=window,
        )
        if actual_guard != state_guard_hash:
            raise L1TurnRunPersistenceError("L1 decision state guard is stale")
        used = int(state["semantic_steps_used"])
        if int(step["ordinal"]) != used + 1:
            raise L1TurnRunPersistenceError("L1 Attempt ordinal changed")
        if used >= int(state["max_semantic_steps"]):
            raise L1TurnRunPersistenceError("L1 Attempt limit is exhausted")
        current_plan_revision = _plan_revision(state["plan_json"])
        next_plan_revision = _plan_revision(plan_json)
        if used == 0:
            if plan_json is None or next_plan_revision != 1:
                raise L1TurnRunPersistenceError(
                    "first L1 AttemptDecision requires Plan revision 1"
                )
            if current_plan_revision is not None:
                raise L1TurnRunPersistenceError(
                    "first L1 AttemptDecision found an existing Plan"
                )
        elif current_plan_revision is None:
            raise L1TurnRunPersistenceError(
                "later L1 AttemptDecision requires an existing Plan"
            )
        elif plan_json is not None and next_plan_revision != current_plan_revision + 1:
            raise L1TurnRunPersistenceError(
                "L1 Plan revision must advance exactly once"
            )
        if action_kind == "submit_final_reply":
            assert verification_report is not None
            assert semantic_verification_report is not None
            active_plan = _plan_payload(
                plan_json if plan_json is not None else state["plan_json"]
            )
            acceptances = active_plan.get("acceptances")
            checked_acceptances = verification_report.get(
                "checked_acceptances"
            )
            checked_tool_results = verification_report.get(
                "checked_tool_results"
            )
            durable_tool_call_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM l1_turn_tool_calls "
                    "WHERE l1_turn_run_id=?",
                    (l1_turn_run_id,),
                ).fetchone()[0]
            )
            if (
                not isinstance(acceptances, list)
                or isinstance(checked_acceptances, bool)
                or not isinstance(checked_acceptances, int)
                or checked_acceptances != len(acceptances)
                or isinstance(checked_tool_results, bool)
                or not isinstance(checked_tool_results, int)
                or checked_tool_results != durable_tool_call_count
            ):
                raise L1TurnRunPersistenceError(
                    "verification receipt does not cover current execution facts"
                )
            active_plan_hash = str(
                plan_hash if plan_hash is not None else state["plan_hash"] or ""
            )
            _validate_semantic_receipt_in_transaction(
                conn,
                receipt=semantic_verification_report,
                session_id=session_id,
                turn_id=turn_id,
                decision_hash=decision_hash,
                active_plan_hash=active_plan_hash,
                mechanical_verification_hash=str(
                    verification_report_hash or ""
                ),
                state_guard_hash=state_guard_hash,
                acceptance_count=len(acceptances),
                tool_result_count=durable_tool_call_count,
                execution_config_json=state["execution_config_json"],
            )

        step_status = (
            "decided" if action_kind == "call_tools" else "final_answer"
        )
        conn.execute(
            "UPDATE l1_turn_steps SET status=?, decision_json=?, decision_hash=?, "
            "action_kind=?, tool_call_count=?, updated_at=? "
            "WHERE step_id=? AND status='prepared'",
            (
                step_status,
                decision_json,
                decision_hash,
                physical_action_kind,
                tool_call_count,
                now,
                step_id,
            ),
        )
        state_stage = "tool" if action_kind == "call_tools" else "finalizing"
        conn.execute(
            "UPDATE l1_turn_run_states SET stage=?, semantic_steps_used=?, "
            "plan_json=COALESCE(?, plan_json), plan_hash=COALESCE(?, plan_hash), "
            "final_reply=?, "
            "verification_contract_version=?, verification_report_json=?, "
            "verification_report_hash=?, "
            "semantic_verification_contract_version=?, "
            "semantic_verification_report_json=?, "
            "semantic_verification_report_hash=?, "
            "updated_at=? WHERE l1_turn_run_id=?",
            (
                state_stage,
                used + 1,
                plan_json,
                plan_hash,
                final_reply,
                verification_contract_version,
                verification_report_json,
                verification_report_hash,
                semantic_verification_contract_version,
                semantic_verification_report_json,
                semantic_verification_report_hash,
                now,
                l1_turn_run_id,
            ),
        )
        if plan_json is not None and plan_hash is not None:
            assert next_plan_revision is not None
            conn.execute(
                "INSERT INTO l1_turn_plan_revisions "
                "(l1_turn_run_id, revision, session_id, turn_id, "
                "accepted_step_id, plan_json, plan_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    l1_turn_run_id,
                    next_plan_revision,
                    session_id,
                    turn_id,
                    step_id,
                    plan_json,
                    plan_hash,
                    now,
                ),
            )
        conn.execute(
            "UPDATE l1_turn_runs SET run_revision=run_revision+1, updated_at=? "
            "WHERE l1_turn_run_id=? AND status='running'",
            (now, l1_turn_run_id),
        )
        next_window_revision = int(window["state_version"]) + 1
        _advance_l1_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            current_l1_attempt_id=step_id,
            stage="TOOL" if action_kind == "call_tools" else "RESPONSE",
            expected_window_revision=int(window["state_version"]),
            next_window_revision=next_window_revision,
            expected_lease_owner=expected_lease_owner,
            now=now,
        )
        return {
            **_execution_projection(
                conn,
                run=_require_run(conn, l1_turn_run_id),
                state=_require_state(conn, l1_turn_run_id),
                window=_require_window(conn, session_id, turn_id),
                replayed=False,
            ),
            "attempt": _attempt_projection(_require_attempt(conn, step_id)),
        }


def reject_l1_final_reply_candidate(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    state_guard_hash: str,
    decision_json: str,
    decision_hash: str,
    plan_json: str | None,
    plan_hash: str | None,
    verification_feedback_json: str,
    verification_feedback_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> dict[str, object]:
    """为下一次 L1 Attempt 持久化 verifier 反馈。

    被语义拒绝的交付是已关闭的 submit/review Attempt，而不是格式错误的 provider
    响应。保留其 VerificationFeedback，可让恢复过程精确重放审查结果，也让模型
    决定是收集更多证据、修改 Plan，还是重新提交。
    """

    step_id = attempt_id
    observation_json = verification_feedback_json
    observation_hash = verification_feedback_hash

    _validate_canonical_json_hash(decision_json, decision_hash, label="L1 decision")
    _validate_optional_json_pair(plan_json, plan_hash, label="L1 Plan")
    _validate_canonical_json_hash(
        observation_json,
        observation_hash,
        label="L1 VerificationFeedback",
    )
    decision = json.loads(decision_json)
    verification_feedback = json.loads(observation_json)
    if (
        not isinstance(decision, dict)
        or not isinstance(decision.get("action"), dict)
        or decision["action"].get("kind") != "submit_final_reply"
    ):
        raise ValueError(
            "verification rejection requires a submit_final_reply decision"
        )
    if (
        not isinstance(verification_feedback, dict)
        or verification_feedback.get("kind") != "verification_rejected"
        or verification_feedback.get("candidate_decision_hash") != decision_hash
    ):
        raise ValueError("VerificationFeedback is not bound to its candidate")

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        state = _require_state(conn, l1_turn_run_id)
        step = _require_attempt(conn, step_id)
        if str(step["status"]) == "observed":
            if (
                str(step["decision_hash"] or "") == decision_hash
                and str(step["observation_hash"] or "") == observation_hash
            ):
                return {
                    **_execution_projection(
                        conn,
                        run=run,
                        state=state,
                        window=window,
                        replayed=True,
                    ),
                    "attempt": _attempt_projection(step),
                }
            raise L1TurnRunPersistenceError(
                "L1 verification rejection replay changed"
            )
        if str(step["status"]) != "prepared":
            raise L1TurnRunPersistenceError(
                "L1 Attempt cannot accept verifier feedback"
            )
        if str(step["state_guard_hash"]) != state_guard_hash:
            raise L1TurnRunPersistenceError(
                "L1 verifier feedback crossed its state guard"
            )
        actual_guard = get_l1_attempt_state_guard_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            step=step,
            run=run,
            state=state,
            window=window,
        )
        if actual_guard != state_guard_hash:
            raise L1TurnRunPersistenceError(
                "L1 verifier feedback state guard is stale"
            )
        used = int(state["semantic_steps_used"])
        if int(step["ordinal"]) != used + 1:
            raise L1TurnRunPersistenceError("L1 Attempt ordinal changed")
        if used >= int(state["max_semantic_steps"]):
            raise L1TurnRunPersistenceError("L1 Attempt limit is exhausted")
        current_plan_revision = _plan_revision(state["plan_json"])
        next_plan_revision = _plan_revision(plan_json)
        if used == 0:
            if plan_json is None or next_plan_revision != 1:
                raise L1TurnRunPersistenceError(
                    "first rejected L1 candidate requires Plan revision 1"
                )
            if current_plan_revision is not None:
                raise L1TurnRunPersistenceError(
                    "first rejected L1 candidate found an existing Plan"
                )
        elif current_plan_revision is None:
            raise L1TurnRunPersistenceError(
                "later rejected L1 candidate requires an existing Plan"
            )
        elif plan_json is not None and next_plan_revision != current_plan_revision + 1:
            raise L1TurnRunPersistenceError(
                "L1 Plan revision must advance exactly once"
            )

        conn.execute(
            "UPDATE l1_turn_steps SET status='observed', decision_json=?, "
            "decision_hash=?, action_kind='final_answer', tool_call_count=0, "
            "observation_json=?, observation_hash=?, updated_at=? "
            "WHERE step_id=? AND status='prepared'",
            (
                decision_json,
                decision_hash,
                observation_json,
                observation_hash,
                now,
                step_id,
            ),
        )
        conn.execute(
            "UPDATE l1_turn_run_states SET stage='observation', "
            "semantic_steps_used=?, plan_json=COALESCE(?, plan_json), "
            "plan_hash=COALESCE(?, plan_hash), latest_observation_json=?, "
            "latest_observation_hash=?, updated_at=? WHERE l1_turn_run_id=?",
            (
                used + 1,
                plan_json,
                plan_hash,
                observation_json,
                observation_hash,
                now,
                l1_turn_run_id,
            ),
        )
        if plan_json is not None and plan_hash is not None:
            assert next_plan_revision is not None
            conn.execute(
                "INSERT INTO l1_turn_plan_revisions "
                "(l1_turn_run_id, revision, session_id, turn_id, "
                "accepted_step_id, plan_json, plan_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    l1_turn_run_id,
                    next_plan_revision,
                    session_id,
                    turn_id,
                    step_id,
                    plan_json,
                    plan_hash,
                    now,
                ),
            )
        conn.execute(
            "UPDATE l1_turn_runs SET run_revision=run_revision+1, updated_at=? "
            "WHERE l1_turn_run_id=? AND status='running'",
            (now, l1_turn_run_id),
        )
        next_window_revision = int(window["state_version"]) + 1
        _advance_l1_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            current_l1_attempt_id=step_id,
            stage="OBSERVATION",
            expected_window_revision=int(window["state_version"]),
            next_window_revision=next_window_revision,
            expected_lease_owner=expected_lease_owner,
            now=now,
        )
        return {
            **_execution_projection(
                conn,
                run=_require_run(conn, l1_turn_run_id),
                state=_require_state(conn, l1_turn_run_id),
                window=_require_window(conn, session_id, turn_id),
                replayed=False,
            ),
            "attempt": _attempt_projection(_require_attempt(conn, step_id)),
        }


def reserve_l1_tool_call(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    call_ordinal: int,
    tool_id: str,
    contract_version: str,
    implementation_version: str,
    arguments_json: str,
    arguments_hash: str,
    policy_json: str,
    execution_class: str = "read_only",
    effect_profile_sha256: str | None = None,
    provider_identity_sha256: str | None = None,
    approval_receipt_ids_json: str = "[]",
    approval_receipts_sha256: str | None = None,
    expected_window_revision: int | None = None,
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """在物理执行前预留一个精确 L1 ToolCall。"""

    step_id = attempt_id

    if isinstance(call_ordinal, bool) or not 1 <= call_ordinal <= 16:
        raise ValueError("call_ordinal must be within 1..16")
    for name, value in (
        ("tool_id", tool_id),
        ("contract_version", contract_version),
        ("implementation_version", implementation_version),
    ):
        _require_identifier(name, value)
    _validate_canonical_json_hash(arguments_json, arguments_hash, label="tool arguments")
    supports_obligation_keys_json = "[]"
    policy = _validate_canonical_json(policy_json, label="tool policy")
    if (
        not isinstance(policy, dict)
        or policy.get("disposition")
        not in {
            "allow",
            "deny",
            "authorization_required",
            "approval_required",
            "defer",
        }
    ):
        raise ValueError("tool policy must contain a supported disposition")
    if execution_class not in {
        "read_only",
        "runtime_state",
        "protected_effect",
    }:
        raise ValueError("L1 ToolCall execution class is unsupported")
    receipt_ids = _validate_l1_approval_receipt_ids(approval_receipt_ids_json)
    if expected_window_revision is not None and expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)
    if execution_class == "read_only":
        if (
            effect_profile_sha256 is not None
            or provider_identity_sha256 is not None
            or receipt_ids
            or approval_receipts_sha256 is not None
        ):
            raise ValueError("read-only L1 ToolCalls cannot carry protected facts")
    elif execution_class == "runtime_state":
        if policy.get("disposition") != "allow":
            raise ValueError("runtime-state L1 ToolCalls require an allowed policy")
        if expected_window_revision is None:
            raise ValueError(
                "runtime-state L1 ToolCalls require the current Window revision"
            )
        _require_sha256("effect_profile_sha256", effect_profile_sha256)
        if (
            provider_identity_sha256 is not None
            or receipt_ids
            or approval_receipts_sha256 is not None
        ):
            raise ValueError(
                "runtime-state L1 ToolCalls cannot carry external-effect authority"
            )
    else:
        if policy.get("disposition") != "allow":
            raise ValueError("protected L1 ToolCalls require an allowed policy")
        if expected_window_revision is None:
            raise ValueError(
                "protected L1 ToolCalls require the current Window revision"
            )
        _require_sha256("effect_profile_sha256", effect_profile_sha256)
        _require_sha256("provider_identity_sha256", provider_identity_sha256)
        if not receipt_ids:
            raise ValueError("protected L1 ToolCalls require approval receipts")
        _require_sha256("approval_receipts_sha256", approval_receipts_sha256)
        assert approval_receipts_sha256 is not None
        if (
            hashlib.sha256(approval_receipt_ids_json.encode("utf-8")).hexdigest()
            != approval_receipts_sha256
        ):
            raise ValueError("approval receipt hash does not match its identifiers")
    tool_call_id = _stable_id(
        "l1tool",
        {
            "l1_turn_run_id": l1_turn_run_id,
            "step_id": step_id,
            "call_ordinal": call_ordinal,
            "tool_id": tool_id,
            "contract_version": contract_version,
            "arguments_hash": arguments_hash,
        },
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        step = _require_attempt(conn, step_id)
        state = _require_state(conn, l1_turn_run_id)
        run = _require_run(conn, l1_turn_run_id)
        if (
            str(step["session_id"]) != session_id
            or str(step["turn_id"]) != turn_id
            or str(step["l1_turn_run_id"]) != l1_turn_run_id
            or str(step["status"]) != "decided"
            or str(step["action_kind"]) != "call_tools"
            or str(state["stage"]) != "tool"
            or call_ordinal > int(step["tool_call_count"])
        ):
            raise L1TurnRunPersistenceError("L1 ToolCall has no current decision authority")
        protected_state_guard_sha256: str | None = None
        protected_operation_binding_sha256: str | None = None
        protected_phase = "not_required"
        if execution_class == "runtime_state":
            assert expected_window_revision is not None
            _require_active_run_window(
                conn,
                session_id=session_id,
                turn_id=turn_id,
                l1_turn_run_id=l1_turn_run_id,
                expected_window_revision=expected_window_revision,
                expected_lease_owner=expected_lease_owner,
            )
        elif execution_class == "protected_effect":
            assert expected_window_revision is not None
            _, window = _require_active_run_window(
                conn,
                session_id=session_id,
                turn_id=turn_id,
                l1_turn_run_id=l1_turn_run_id,
                expected_window_revision=expected_window_revision,
                expected_lease_owner=expected_lease_owner,
            )
            _require_l1_protected_tool_dispatch_state(
                run=run,
                state=state,
                step=step,
                window=window,
                l1_turn_run_id=l1_turn_run_id,
            )
            protected_state_guard_sha256 = _l1_protected_tool_state_guard_hash(
                run=run,
                state=state,
                step=step,
                window=window,
                tool_call_id=tool_call_id,
                call_ordinal=call_ordinal,
                tool_id=tool_id,
                contract_version=contract_version,
                implementation_version=implementation_version,
                arguments_hash=arguments_hash,
                policy_json=policy_json,
                effect_profile_sha256=effect_profile_sha256,
                provider_identity_sha256=provider_identity_sha256,
                approval_receipts_sha256=approval_receipts_sha256,
            )
            protected_operation_binding_sha256 = (
                _l1_protected_operation_binding_hash(
                    session_id=session_id,
                    turn_id=turn_id,
                    l1_turn_run_id=l1_turn_run_id,
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    call_ordinal=call_ordinal,
                    tool_id=tool_id,
                    contract_version=contract_version,
                    implementation_version=implementation_version,
                    arguments_hash=arguments_hash,
                    policy_json=policy_json,
                    effect_profile_sha256=effect_profile_sha256,
                    provider_identity_sha256=provider_identity_sha256,
                    approval_receipts_sha256=approval_receipts_sha256,
                    protected_state_guard_sha256=protected_state_guard_sha256,
                )
            )
            protected_phase = "ready"
        existing = conn.execute(
            "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=? OR "
            "(l1_turn_run_id=? AND step_id=? AND call_ordinal=?)",
            (tool_call_id, l1_turn_run_id, step_id, call_ordinal),
        ).fetchall()
        if existing:
            if len(existing) != 1:
                raise L1TurnRunPersistenceError("L1 ToolCall identity is ambiguous")
            row = existing[0]
            expected = {
                "tool_call_id": tool_call_id,
                "tool_id": tool_id,
                "contract_version": contract_version,
                "implementation_version": implementation_version,
                "supports_obligation_keys_json": supports_obligation_keys_json,
                "arguments_json": arguments_json,
                "arguments_hash": arguments_hash,
                "policy_json": policy_json,
                "execution_class": execution_class,
                "effect_profile_sha256": effect_profile_sha256,
                "provider_identity_sha256": provider_identity_sha256,
                "approval_receipt_ids_json": approval_receipt_ids_json,
                "approval_receipts_sha256": approval_receipts_sha256,
                "protected_operation_binding_sha256": (
                    protected_operation_binding_sha256
                ),
                "protected_state_guard_sha256": protected_state_guard_sha256,
            }
            if any(row[key] != value for key, value in expected.items()):
                raise L1TurnRunPersistenceError("L1 ToolCall replay facts changed")
            return {"replayed": True, "tool_call": _tool_call_projection(row)}
        conn.execute(
            "INSERT INTO l1_turn_tool_calls "
            "(tool_call_id, l1_turn_run_id, step_id, session_id, turn_id, "
            "call_ordinal, tool_id, contract_version, implementation_version, "
            "supports_obligation_keys_json, arguments_json, arguments_hash, "
            "policy_json, execution_class, effect_profile_sha256, "
            "provider_identity_sha256, approval_receipt_ids_json, "
            "approval_receipts_sha256, protected_operation_binding_sha256, "
            "protected_state_guard_sha256, protected_phase, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "'pending', ?)",
            (
                tool_call_id,
                l1_turn_run_id,
                step_id,
                session_id,
                turn_id,
                call_ordinal,
                tool_id,
                contract_version,
                implementation_version,
                supports_obligation_keys_json,
                arguments_json,
                arguments_hash,
                policy_json,
                execution_class,
                effect_profile_sha256,
                provider_identity_sha256,
                approval_receipt_ids_json,
                approval_receipts_sha256,
                protected_operation_binding_sha256,
                protected_state_guard_sha256,
                protected_phase,
                now,
            ),
        )
        return {
            "replayed": False,
            "tool_call": _tool_call_projection(
                conn.execute(
                    "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
                    (tool_call_id,),
                ).fetchone()
            ),
        }


def settle_l1_tool_call(
    deps: StoreDeps,
    *,
    tool_call_id: str,
    outcome_status: str,
    outcome_json: str,
    outcome_hash: str,
) -> dict[str, object]:
    """对一个已预留读取调用执行且仅执行一次结算。"""

    _validate_l1_tool_outcome(
        outcome_status=outcome_status,
        outcome_json=outcome_json,
        outcome_hash=outcome_hash,
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
            (tool_call_id,),
        ).fetchone()
        if row is None:
            raise L1TurnRunPersistenceError("L1 ToolCall does not exist")
        if str(row["execution_class"] or "read_only") not in {
            "read_only",
            "runtime_state",
        }:
            raise L1TurnRunPersistenceError(
                "protected L1 ToolCalls require protected dispatch settlement"
            )
        if str(row["status"]) != "pending":
            if (
                str(row["status"]) == outcome_status
                and str(row["outcome_hash"] or "") == outcome_hash
            ):
                return {
                    "replayed": True,
                    "tool_call": _tool_call_projection(row),
                }
            raise L1TurnRunPersistenceError("L1 ToolCall already has another outcome")
        conn.execute(
            "UPDATE l1_turn_tool_calls SET status=?, outcome_json=?, outcome_hash=?, "
            "settled_at=? WHERE tool_call_id=? AND status='pending'",
            (outcome_status, outcome_json, outcome_hash, now, tool_call_id),
        )
        return {
            "replayed": False,
            "tool_call": _tool_call_projection(
                conn.execute(
                    "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
                    (tool_call_id,),
                ).fetchone()
            ),
        }


def begin_l1_protected_tool_dispatch(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    tool_call_id: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
    protected_operation_binding_sha256: str,
    physical_attempt_id: str,
) -> dict[str, object]:
    """持久地将一个 L1 effect 从 ``ready`` 移至 ``dispatching``。

    对于已处于 dispatching 状态的匹配操作，返回的 ``started`` 标志有意为 false。
    其先前进程可能已经越过 I/O 边界，因此调用方必须将其结算为 unconfirmed，
    而不是再次发送。
    """

    step_id = attempt_id
    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("l1_turn_run_id", l1_turn_run_id),
        ("attempt_id", attempt_id),
        ("tool_call_id", tool_call_id),
        ("physical_attempt_id", physical_attempt_id),
    ):
        _require_identifier(name, value)
    if expected_window_revision < 1:
        raise ValueError("expected_window_revision must be positive")
    if expected_lease_owner is not None:
        _require_identifier("expected_lease_owner", expected_lease_owner)
    _require_sha256(
        "protected_operation_binding_sha256",
        protected_operation_binding_sha256,
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
            (tool_call_id,),
        ).fetchone()
        if row is None:
            raise L1TurnRunPersistenceError("L1 protected ToolCall does not exist")
        if (
            str(row["session_id"]) != session_id
            or str(row["turn_id"]) != turn_id
            or str(row["l1_turn_run_id"]) != l1_turn_run_id
            or str(row["step_id"]) != step_id
            or str(row["execution_class"] or "") != "protected_effect"
            or str(row["protected_operation_binding_sha256"] or "")
            != protected_operation_binding_sha256
        ):
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall authority changed"
            )
        phase = str(row["protected_phase"] or "")
        if phase == "dispatching":
            if (
                str(row["status"]) == "pending"
                and str(row["physical_attempt_id"] or "") == physical_attempt_id
            ):
                return {
                    "started": False,
                    "tool_call": _tool_call_projection(row),
                }
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall was already dispatched"
            )
        if str(row["status"]) != "pending" or phase != "ready":
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall is not ready for dispatch"
            )
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        state = _require_state(conn, l1_turn_run_id)
        step = _require_attempt(conn, step_id)
        _require_l1_protected_tool_dispatch_state(
            run=run,
            state=state,
            step=step,
            window=window,
            l1_turn_run_id=l1_turn_run_id,
        )
        actual_state_guard = _l1_protected_tool_state_guard_from_row(
            run=run,
            state=state,
            step=step,
            window=window,
            row=row,
        )
        if actual_state_guard != str(row["protected_state_guard_sha256"] or ""):
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall state guard changed"
            )
        updated = conn.execute(
            "UPDATE l1_turn_tool_calls SET protected_phase='dispatching', "
            "physical_attempt_id=?, protected_started_at=? "
            "WHERE tool_call_id=? AND status='pending' "
            "AND protected_phase='ready' "
            "AND protected_operation_binding_sha256=?",
            (
                physical_attempt_id,
                now,
                tool_call_id,
                protected_operation_binding_sha256,
            ),
        ).rowcount
        if updated != 1:
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall changed during dispatch begin"
            )
        return {
            "started": True,
            "tool_call": _tool_call_projection(
                conn.execute(
                    "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
                    (tool_call_id,),
                ).fetchone()
            ),
        }


def settle_l1_protected_tool_dispatch(
    deps: StoreDeps,
    *,
    tool_call_id: str,
    protected_operation_binding_sha256: str,
    physical_attempt_id: str,
    outcome_status: str,
    outcome_json: str,
    outcome_hash: str,
) -> dict[str, object]:
    """以原子方式存储 L1 受保护 receipt 及其外层 ToolCall 结果。"""

    _require_identifier("tool_call_id", tool_call_id)
    _require_identifier("physical_attempt_id", physical_attempt_id)
    _require_sha256(
        "protected_operation_binding_sha256",
        protected_operation_binding_sha256,
    )
    _validate_l1_tool_outcome(
        outcome_status=outcome_status,
        outcome_json=outcome_json,
        outcome_hash=outcome_hash,
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
            (tool_call_id,),
        ).fetchone()
        if row is None:
            raise L1TurnRunPersistenceError("L1 protected ToolCall does not exist")
        if (
            str(row["execution_class"] or "") != "protected_effect"
            or str(row["protected_operation_binding_sha256"] or "")
            != protected_operation_binding_sha256
            or str(row["physical_attempt_id"] or "") != physical_attempt_id
        ):
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall settlement crossed authority"
            )
        if str(row["status"]) != "pending":
            if (
                str(row["status"]) == outcome_status
                and str(row["outcome_hash"] or "") == outcome_hash
                and str(row["protected_receipt_sha256"] or "")
                == _l1_protected_dispatch_receipt_hash(
                    tool_call_id=tool_call_id,
                    protected_operation_binding_sha256=(
                        protected_operation_binding_sha256
                    ),
                    physical_attempt_id=physical_attempt_id,
                    outcome_status=outcome_status,
                    outcome_hash=outcome_hash,
                )
            ):
                return {
                    "replayed": True,
                    "tool_call": _tool_call_projection(row),
                }
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall already has another outcome"
            )
        if str(row["protected_phase"] or "") != "dispatching":
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall was not durably dispatched"
            )
        protected_phase = _l1_protected_terminal_phase(outcome_status)
        receipt_json = _canonical_json(
            {
                "schema_version": "l1-protected-tool-dispatch-receipt-v1",
                "tool_call_id": tool_call_id,
                "operation_binding_sha256": protected_operation_binding_sha256,
                "physical_attempt_id": physical_attempt_id,
                "outcome_status": outcome_status,
                "outcome_hash": outcome_hash,
            }
        )
        receipt_hash = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()
        updated = conn.execute(
            "UPDATE l1_turn_tool_calls SET status=?, outcome_json=?, outcome_hash=?, "
            "settled_at=?, protected_phase=?, protected_receipt_json=?, "
            "protected_receipt_sha256=?, protected_settled_at=? "
            "WHERE tool_call_id=? AND status='pending' "
            "AND protected_phase='dispatching' "
            "AND protected_operation_binding_sha256=? "
            "AND physical_attempt_id=?",
            (
                outcome_status,
                outcome_json,
                outcome_hash,
                now,
                protected_phase,
                receipt_json,
                receipt_hash,
                now,
                tool_call_id,
                protected_operation_binding_sha256,
                physical_attempt_id,
            ),
        ).rowcount
        if updated != 1:
            raise L1TurnRunPersistenceError(
                "L1 protected ToolCall changed during settlement"
            )
        return {
            "replayed": False,
            "tool_call": _tool_call_projection(
                conn.execute(
                    "SELECT * FROM l1_turn_tool_calls WHERE tool_call_id=?",
                    (tool_call_id,),
                ).fetchone()
            ),
        }


def close_l1_attempt(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    attempt_id: str,
    tool_results_json: str,
    tool_results_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> dict[str, object]:
    """为下一次 Attempt 检查点保存已完全结算的工具批次。"""

    step_id = attempt_id
    observation_json = tool_results_json
    observation_hash = tool_results_hash

    _validate_canonical_json_hash(
        observation_json,
        observation_hash,
        label="L1 ToolResults",
    )
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run, window = _require_active_run_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            expected_window_revision=expected_window_revision,
            expected_lease_owner=expected_lease_owner,
        )
        state = _require_state(conn, l1_turn_run_id)
        step = _require_attempt(conn, step_id)
        if str(step["status"]) == "observed":
            if str(step["observation_hash"] or "") != observation_hash:
                raise L1TurnRunPersistenceError("L1 ToolResults replay changed")
            return {
                **_execution_projection(
                    conn,
                    run=run,
                    state=state,
                    window=window,
                    replayed=True,
                ),
                "attempt": _attempt_projection(step),
            }
        if str(step["status"]) != "decided" or str(step["action_kind"]) != "call_tools":
            raise L1TurnRunPersistenceError(
                "L1 Attempt is not awaiting ToolResults"
            )
        calls = conn.execute(
            "SELECT * FROM l1_turn_tool_calls WHERE step_id=? ORDER BY call_ordinal",
            (step_id,),
        ).fetchall()
        if len(calls) != int(step["tool_call_count"]) or any(
            str(call["status"]) == "pending" for call in calls
        ):
            raise L1TurnRunPersistenceError("L1 tool batch is not fully settled")
        conn.execute(
            "UPDATE l1_turn_steps SET status='observed', observation_json=?, "
            "observation_hash=?, updated_at=? WHERE step_id=? AND status='decided'",
            (observation_json, observation_hash, now, step_id),
        )
        conn.execute(
            "UPDATE l1_turn_run_states SET stage='observation', "
            "latest_observation_json=?, latest_observation_hash=?, updated_at=? "
            "WHERE l1_turn_run_id=?",
            (observation_json, observation_hash, now, l1_turn_run_id),
        )
        conn.execute(
            "UPDATE l1_turn_runs SET run_revision=run_revision+1, updated_at=? "
            "WHERE l1_turn_run_id=? AND status='running'",
            (now, l1_turn_run_id),
        )
        next_window_revision = int(window["state_version"]) + 1
        _advance_l1_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            l1_turn_run_id=l1_turn_run_id,
            current_l1_attempt_id=step_id,
            stage="OBSERVATION",
            expected_window_revision=int(window["state_version"]),
            next_window_revision=next_window_revision,
            expected_lease_owner=expected_lease_owner,
            now=now,
        )
        return {
            **_execution_projection(
                conn,
                run=_require_run(conn, l1_turn_run_id),
                state=_require_state(conn, l1_turn_run_id),
                window=_require_window(conn, session_id, turn_id),
                replayed=False,
            ),
            "attempt": _attempt_projection(_require_attempt(conn, step_id)),
        }


def get_l1_turn_execution(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object] | None:
    deps.init_db()
    with deps.connect() as conn:
        run = conn.execute(
            "SELECT * FROM l1_turn_runs WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if run is None:
            return None
        state = conn.execute(
            "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
            (str(run["l1_turn_run_id"]),),
        ).fetchone()
        attempts = conn.execute(
            "SELECT * FROM l1_turn_steps WHERE l1_turn_run_id=? ORDER BY ordinal",
            (str(run["l1_turn_run_id"]),),
        ).fetchall()
        calls = conn.execute(
            # Attempt IDs are opaque; every execution consumer receives chronological calls.
            "SELECT c.* FROM l1_turn_tool_calls c "
            "JOIN l1_turn_steps s ON s.step_id=c.step_id "
            "WHERE c.l1_turn_run_id=? ORDER BY s.ordinal,c.call_ordinal",
            (str(run["l1_turn_run_id"]),),
        ).fetchall()
        plan_revisions = conn.execute(
            "SELECT * FROM l1_turn_plan_revisions WHERE l1_turn_run_id=? "
            "ORDER BY revision",
            (str(run["l1_turn_run_id"]),),
        ).fetchall()
        return {
            "run": _run_projection(run),
            "state": _state_projection(state) if state is not None else None,
            "attempts": [_attempt_projection(row) for row in attempts],
            "tool_calls": [_tool_call_projection(row) for row in calls],
            "plan_revisions": [
                _plan_revision_projection(row)
                for row in plan_revisions
            ],
        }


def fail_l1_turn_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    failure_code: str,
) -> None:
    """记录有类型的内部 L1 失败，但不结算外层 Turn。"""

    _require_identifier("failure_code", failure_code)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run = _require_run(conn, l1_turn_run_id)
        if str(run["session_id"]) != session_id or str(run["turn_id"]) != turn_id:
            raise L1TurnRunPersistenceError("L1 failure crossed run ownership")
        status = str(run["status"])
        if status == "failed":
            state = conn.execute(
                "SELECT failure_code FROM l1_turn_run_states "
                "WHERE l1_turn_run_id=?",
                (l1_turn_run_id,),
            ).fetchone()
            if state is None or str(state["failure_code"] or "") == failure_code:
                return
            raise L1TurnRunPersistenceError(
                "failed L1 run already records another failure code"
            )
        if status in {"completed", "cancelled"}:
            raise L1TurnRunPersistenceError(
                f"{status} L1 run cannot fail"
            )
        conn.execute(
            "UPDATE l1_turn_runs SET status='failed', run_revision=run_revision+1, "
            "updated_at=? WHERE l1_turn_run_id=? AND status IN ('created', 'running')",
            (now, l1_turn_run_id),
        )
        conn.execute(
            "UPDATE l1_turn_run_states SET stage='failed', failure_code=?, "
            "updated_at=? WHERE l1_turn_run_id=?",
            (failure_code, now, l1_turn_run_id),
        )
        _close_l1_findings_in_transaction(
            conn,
            l1_turn_run_id=l1_turn_run_id,
            closed_at=now,
        )


def complete_l1_run_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    assistant_content: str,
    completed_at: str,
) -> None:
    """在正式 assistant 消息旁以原子方式封存 L1 聚合。"""

    window = _require_window(conn, session_id, turn_id)
    run_id = str(window["current_l1_turn_run_id"] or "")
    if not run_id:
        raise L1TurnRunPersistenceError("L1 finalization has no Window-bound run")
    run = _require_run(conn, run_id)
    state = _require_state(conn, run_id)
    if (
        str(run["session_id"]) != session_id
        or str(run["turn_id"]) != turn_id
        or str(run["status"]) != "running"
        or str(state["stage"]) != "finalizing"
        or str(state["final_reply"] or "") != assistant_content
        or state["semantic_verification_report_json"] is None
    ):
        raise L1TurnRunPersistenceError("L1 formal reply crossed finalizing authority")
    conn.execute(
        "UPDATE l1_turn_runs SET status='completed', run_revision=run_revision+1, "
        "updated_at=?, completed_at=? WHERE l1_turn_run_id=? AND status='running'",
        (completed_at, completed_at, run_id),
    )
    conn.execute(
        "UPDATE l1_turn_run_states SET stage='completed', updated_at=? "
        "WHERE l1_turn_run_id=? AND stage='finalizing'",
        (completed_at, run_id),
    )
    _close_l1_findings_in_transaction(
        conn,
        l1_turn_run_id=run_id,
        closed_at=completed_at,
    )


def _validate_semantic_receipt_in_transaction(
    conn: sqlite3.Connection,
    *,
    receipt: L1SemanticVerificationReceipt,
    session_id: str,
    turn_id: str,
    decision_hash: str,
    active_plan_hash: str,
    mechanical_verification_hash: str,
    state_guard_hash: str,
    acceptance_count: int,
    tool_result_count: int,
    execution_config_json: object,
) -> None:
    """在最终化过程中重新推导 policy 并认证一次语义通过。"""

    execution_config = _validate_canonical_json(
        str(execution_config_json or ""),
        label="L1 execution configuration",
    )
    if not isinstance(execution_config, dict) or not isinstance(
        execution_config.get("features"),
        dict,
    ):
        raise L1TurnRunPersistenceError(
            "L1 execution configuration omitted its features"
        )
    try:
        mode = parse_l1_semantic_verification_mode(
            execution_config["features"].get(
                L1_SEMANTIC_VERIFICATION_FEATURE
            )
        )
        expected_trigger = derive_l1_semantic_verification_trigger(
            mode=mode,
            acceptance_count=acceptance_count,
            tool_result_count=tool_result_count,
        )
    except ValueError as exc:
        raise L1TurnRunPersistenceError(
            "L1 semantic verification policy is invalid"
        ) from exc
    expected = {
        "trigger": expected_trigger,
        "decision_hash": decision_hash,
        "plan_hash": active_plan_hash,
        "mechanical_verification_hash": mechanical_verification_hash,
        "state_guard_hash": state_guard_hash,
        "checked_acceptances": acceptance_count,
        "checked_tool_results": tool_result_count,
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise L1TurnRunPersistenceError(
            "semantic verification receipt crossed current delivery facts"
        )
    if expected_trigger.required:
        _authenticate_semantic_model_pass(
            conn,
            receipt=receipt,
            session_id=session_id,
            turn_id=turn_id,
            expected_candidate_binding={
                "decision_hash": decision_hash,
                "plan_hash": active_plan_hash,
                "mechanical_verification_hash": mechanical_verification_hash,
                "state_guard_hash": state_guard_hash,
            },
        )


def _authenticate_semantic_model_pass(
    conn: sqlite3.Connection,
    *,
    receipt: L1SemanticVerificationReceipt,
    session_id: str,
    turn_id: str,
    expected_candidate_binding: dict[str, str],
) -> None:
    logical_call_id = receipt.reviewer_logical_call_id
    result_hash = receipt.reviewer_result_hash
    result = receipt.reviewer_result
    if logical_call_id is None or result_hash is None or result is None:
        raise L1TurnRunPersistenceError(
            "triggered semantic receipt omitted its reviewer"
        )
    logical = conn.execute(
        "SELECT invocation_turn_id, call_kind, purpose, request_contract, "
        "typed_result_contract, state_guard_sha256, request_json "
        "FROM insession_runtime_model_logical_calls "
        "WHERE session_id=? AND logical_call_id=?",
        (session_id, logical_call_id),
    ).fetchone()
    if (
        logical is None
        or str(logical["invocation_turn_id"]) != turn_id
        or str(logical["call_kind"]) != "l1_semantic_verifier"
        or str(logical["purpose"]) != "runtime_l1_semantic_verifier"
        or str(logical["request_contract"])
        != "l1-semantic-verifier-model-protocol"
        or str(logical["typed_result_contract"])
        != L1_SEMANTIC_RESULT_CONTRACT
        or str(logical["state_guard_sha256"])
        != expected_candidate_binding["state_guard_hash"]
    ):
        raise L1TurnRunPersistenceError(
            "semantic verifier has no matching durable model authority"
        )
    logical_envelope = _validate_canonical_json(
        str(logical["request_json"]),
        label="L1 semantic logical request",
    )
    nested_request_json = (
        logical_envelope.get("request_json")
        if isinstance(logical_envelope, dict)
        else None
    )
    request_payload = _validate_canonical_json(
        str(nested_request_json or ""),
        label="L1 semantic request payload",
    )
    if (
        not isinstance(request_payload, dict)
        or request_payload.get("candidate_binding")
        != expected_candidate_binding
    ):
        raise L1TurnRunPersistenceError(
            "semantic verifier request crossed candidate authority"
        )
    settlement = conn.execute(
        "SELECT settlement_json FROM "
        "insession_runtime_model_call_settlement_receipts "
        "WHERE session_id=? AND logical_call_id=? AND outcome='succeeded' "
        "ORDER BY physical_ordinal DESC LIMIT 1",
        (session_id, logical_call_id),
    ).fetchone()
    if settlement is None:
        raise L1TurnRunPersistenceError(
            "semantic verifier has no durable success settlement"
        )
    settlement_payload = _validate_canonical_json(
        str(settlement["settlement_json"]),
        label="L1 semantic model settlement",
    )
    typed = (
        settlement_payload.get("typed_result")
        if isinstance(settlement_payload, dict)
        else None
    )
    expected_result_json = _canonical_json(result.model_dump(mode="json"))
    if (
        not isinstance(typed, dict)
        or typed.get("result_contract")
        != L1_SEMANTIC_RESULT_CONTRACT
        or typed.get("result_sha256") != result_hash
        or typed.get("result_json") != expected_result_json
        or hashlib.sha256(expected_result_json.encode("utf-8")).hexdigest()
        != result_hash
    ):
        raise L1TurnRunPersistenceError(
            "semantic verifier receipt differs from its durable typed result"
        )


def get_l1_attempt_state_guard_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    step: sqlite3.Row,
    run: sqlite3.Row,
    state: sqlite3.Row,
    window: sqlite3.Row,
) -> str:
    if (
        str(run["session_id"]) != session_id
        or str(run["turn_id"]) != turn_id
        or str(run["status"]) != "running"
        or str(step["l1_turn_run_id"]) != l1_turn_run_id
        or str(step["status"]) != "prepared"
        or str(window["current_l1_turn_run_id"] or "") != l1_turn_run_id
        or str(window["current_l1_attempt_id"] or "") != str(step["step_id"])
        or str(window["window_state"]) != "active"
    ):
        raise L1TurnRunPersistenceError("L1 model state guard is no longer current")
    findings_ledger_id, findings_revision = _l1_findings_guard_facts(
        conn,
        l1_turn_run_id=l1_turn_run_id,
    )
    return _l1_state_guard_hash(
        l1_turn_run_id=l1_turn_run_id,
        step_id=str(step["step_id"]),
        ordinal=int(step["ordinal"]),
        run_revision=int(run["run_revision"]),
        window_revision=int(window["state_version"]),
        semantic_steps_used=int(state["semantic_steps_used"]),
        plan_hash=_optional_text(state["plan_hash"]),
        observation_hash=_optional_text(state["latest_observation_hash"]),
        catalog_snapshot_hash=str(state["catalog_snapshot_hash"]),
        execution_config_hash=str(state["execution_config_hash"]),
        corpus_manifest_hash=_optional_text(state["corpus_manifest_hash"]),
        execution_findings_ledger_id=findings_ledger_id,
        execution_findings_revision=findings_revision,
        request_hash=str(step["request_hash"]),
    )


def _require_active_run_window(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    expected_window_revision: int,
    expected_lease_owner: str | None,
) -> tuple[sqlite3.Row, sqlite3.Row]:
    run = _require_run(conn, l1_turn_run_id)
    if str(run["session_id"]) != session_id or str(run["turn_id"]) != turn_id:
        raise L1TurnRunPersistenceError("L1 run ownership changed")
    window = _require_window(conn, session_id, turn_id)
    actual_revision = int(window["state_version"])
    if actual_revision != expected_window_revision:
        raise TurnExecutionWindowRevisionConflict(
            expected=expected_window_revision,
            actual=actual_revision,
        )
    if str(window["window_state"]) != "active":
        raise L1TurnRunPersistenceError("L1 run requires an active Window")
    if str(window["current_l1_turn_run_id"] or "") != l1_turn_run_id:
        raise L1TurnRunPersistenceError("L1 run lost its Window binding")
    if (
        expected_lease_owner is not None
        and str(window["lease_owner"] or "") != expected_lease_owner
    ):
        raise L1TurnRunPersistenceError("L1 run requires the current Window lease owner")
    return run, window


def _advance_l1_window(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    current_l1_attempt_id: str | None,
    stage: str,
    expected_window_revision: int,
    next_window_revision: int,
    expected_lease_owner: str | None,
    now: str,
) -> None:
    updated = conn.execute(
        "UPDATE turn_execution_windows SET current_l1_attempt_id=?, stage=?, "
        "state_version=?, heartbeat_at=?, updated_at=? "
        "WHERE session_id=? AND turn_id=? AND window_state='active' "
        "AND current_l1_turn_run_id=? AND state_version=? "
        "AND current_work_run_id IS NULL AND current_attempt_id IS NULL "
        "AND (? IS NULL OR lease_owner=?)",
        (
            current_l1_attempt_id,
            stage,
            next_window_revision,
            now,
            now,
            session_id,
            turn_id,
            l1_turn_run_id,
            expected_window_revision,
            expected_lease_owner,
            expected_lease_owner,
        ),
    ).rowcount
    if updated != 1:
        raise L1TurnRunPersistenceError("L1 Window changed during checkpoint commit")


def _execution_projection(
    conn: sqlite3.Connection,
    *,
    run: sqlite3.Row,
    state: sqlite3.Row,
    window: sqlite3.Row,
    replayed: bool,
) -> dict[str, object]:
    return {
        "replayed": replayed,
        "run": _run_projection(run),
        "state": _state_projection(state),
        "window": dict(window),
    }


def _run_projection(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["revision"] = value.pop("run_revision")
    if value.get("status") in {"created", "running"}:
        value["status"] = "active"
    return value


def _state_projection(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["max_attempts"] = value.pop("max_semantic_steps")
    value["max_tool_calls_per_attempt"] = value.pop(
        "max_tool_calls_per_step"
    )
    value["attempts_started"] = value.pop("semantic_steps_used")
    # 已退役的逐项自评物理列不再对外投影，也不用于当前交付准入。
    value.pop("completion_report_json")
    value.pop("completion_report_hash")
    latest_json = value.pop("latest_observation_json")
    latest_hash = value.pop("latest_observation_hash")
    feedback = None
    if isinstance(latest_json, str):
        try:
            candidate = json.loads(latest_json)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("kind") == "verification_rejected":
            feedback = latest_json
    value["verification_feedback_json"] = feedback
    value["verification_feedback_hash"] = latest_hash if feedback else None
    return value


def _attempt_projection(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["attempt_id"] = value.pop("step_id")
    if value.get("action_kind") == "final_answer":
        value["action_kind"] = "submit_final_reply"
    value["status"] = (
        "active"
        if value.get("status") in {"prepared", "decided"}
        else "closed"
    )
    result_json = value.pop("observation_json")
    result_hash = value.pop("observation_hash")
    if value.get("action_kind") == "call_tools":
        value["tool_results_json"] = result_json
        value["tool_results_hash"] = result_hash
    elif result_json is not None:
        value["verification_feedback_json"] = result_json
        value["verification_feedback_hash"] = result_hash
    return value


def _tool_call_projection(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["attempt_id"] = value.pop("step_id")
    value.pop("supports_obligation_keys_json", None)
    return value


def _plan_revision_projection(row: sqlite3.Row) -> dict[str, object]:
    value = dict(row)
    value["accepted_attempt_id"] = value.pop("accepted_step_id")
    return value


def _require_run(conn: sqlite3.Connection, l1_turn_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM l1_turn_runs WHERE l1_turn_run_id=?",
        (l1_turn_run_id,),
    ).fetchone()
    if row is None:
        raise L1TurnRunPersistenceError("L1 TurnRun does not exist")
    return row


def _require_state(conn: sqlite3.Connection, l1_turn_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM l1_turn_run_states WHERE l1_turn_run_id=?",
        (l1_turn_run_id,),
    ).fetchone()
    if row is None:
        raise L1TurnRunPersistenceError("L1 executable state does not exist")
    return row


def _require_attempt(conn: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM l1_turn_steps WHERE step_id=?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        raise L1TurnRunPersistenceError("L1 Attempt does not exist")
    return row


def _require_window(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM turn_execution_windows WHERE session_id=? AND turn_id=?",
        (session_id, turn_id),
    ).fetchone()
    if row is None:
        raise L1TurnRunPersistenceError("L1 Turn does not own the Window")
    return row


def _validate_execution_limits(
    *,
    deadline_at: str,
    max_semantic_steps: int,
    max_tool_calls_per_step: int,
) -> None:
    try:
        parsed = datetime.fromisoformat(deadline_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadline_at must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("deadline_at must include a timezone")
    if isinstance(max_semantic_steps, bool) or not 1 <= max_semantic_steps <= 64:
        raise ValueError("max_semantic_steps must be within 1..64")
    if (
        isinstance(max_tool_calls_per_step, bool)
        or not 1 <= max_tool_calls_per_step <= 16
    ):
        raise ValueError("max_tool_calls_per_step must be within 1..16")


def _validate_l1_corpus_manifest(
    *,
    contract_version: str,
    payload: str,
    payload_hash: str,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
) -> None:
    if contract_version != L1_CORPUS_MANIFEST_CONTRACT_VERSION:
        raise ValueError("L1 corpus manifest contract version is unsupported")
    load_l1_corpus_manifest(
        payload,
        payload_hash,
        expected_session_id=session_id,
        expected_turn_id=turn_id,
        expected_l1_turn_run_id=l1_turn_run_id,
    )


def _validate_optional_json_pair(
    payload: str | None,
    payload_hash: str | None,
    *,
    label: str,
) -> None:
    if payload is None and payload_hash is None:
        return
    if payload is None or payload_hash is None:
        raise ValueError(f"{label} JSON and hash must be supplied together")
    _validate_canonical_json_hash(payload, payload_hash, label=label)


def _plan_revision(payload: object) -> int | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise L1TurnRunPersistenceError("L1 Plan has the wrong representation")
    parsed = _plan_payload(payload)
    revision = parsed.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise L1TurnRunPersistenceError("L1 Plan revision must be an integer")
    if not 1 <= revision <= 64:
        raise L1TurnRunPersistenceError("L1 Plan revision is out of range")
    return revision


def _plan_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, str):
        raise L1TurnRunPersistenceError("L1 Plan has the wrong representation")
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise L1TurnRunPersistenceError("L1 Plan is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise L1TurnRunPersistenceError("L1 Plan must be a JSON object")
    return parsed


def _validate_canonical_json_hash(
    payload: str,
    expected_hash: str,
    *,
    label: str,
) -> object:
    parsed = _validate_canonical_json(payload, label=label)
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
        or hashlib.sha256(payload.encode("utf-8")).hexdigest() != expected_hash
    ):
        raise ValueError(f"{label} hash does not match")
    return parsed


def _validate_canonical_json(payload: str, *, label: str) -> object:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{label} must be non-empty canonical JSON")
    parsed = json.loads(payload)
    if _canonical_json(parsed) != payload:
        raise ValueError(f"{label} must use canonical JSON")
    return parsed


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()}"


def _l1_state_guard_hash(**facts: object) -> str:
    return hashlib.sha256(
        _canonical_json({"contract": "l1-model-state-guard-v1", **facts}).encode(
            "utf-8"
        )
    ).hexdigest()


def _l1_findings_guard_facts(
    conn: sqlite3.Connection,
    *,
    l1_turn_run_id: str,
) -> tuple[str | None, int | None]:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='execution_findings_ledgers'"
    ).fetchone()
    if table is None:
        return None, None
    rows = conn.execute(
        "SELECT ledger_id, revision FROM execution_findings_ledgers "
        "WHERE owner_kind='l1_turn_run' AND l1_turn_run_id=?",
        (l1_turn_run_id,),
    ).fetchall()
    if not rows:
        return None, None
    if len(rows) != 1:
        raise L1TurnRunPersistenceError(
            "L1 TurnRun has ambiguous execution-findings authority"
        )
    return str(rows[0]["ledger_id"]), int(rows[0]["revision"])


def _close_l1_findings_in_transaction(
    conn: sqlite3.Connection,
    *,
    l1_turn_run_id: str,
    closed_at: str,
) -> None:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='execution_findings_ledgers'"
    ).fetchone()
    if table is None:
        return
    rows = conn.execute(
        "SELECT ledger_id, status FROM execution_findings_ledgers "
        "WHERE owner_kind='l1_turn_run' AND l1_turn_run_id=?",
        (l1_turn_run_id,),
    ).fetchall()
    if not rows:
        return
    if len(rows) != 1:
        raise L1TurnRunPersistenceError(
            "L1 TurnRun has ambiguous execution-findings authority"
        )
    if str(rows[0]["status"]) == "closed":
        return
    if conn.execute(
        "UPDATE execution_findings_ledgers "
        "SET status='closed', closed_at=?, updated_at=? "
        "WHERE ledger_id=? AND status='open'",
        (closed_at, closed_at, str(rows[0]["ledger_id"])),
    ).rowcount != 1:
        raise L1TurnRunPersistenceError(
            "L1 findings ledger changed during terminal settlement"
        )


def _validate_l1_tool_outcome(
    *,
    outcome_status: str,
    outcome_json: str,
    outcome_hash: str,
) -> None:
    allowed = {
        "succeeded",
        "rejected",
        "failed",
        "timed_out",
        "cancelled",
        "completion_unconfirmed",
    }
    if outcome_status not in allowed:
        raise ValueError("unsupported L1 tool outcome status")
    outcome = _validate_canonical_json_hash(
        outcome_json,
        outcome_hash,
        label="tool outcome",
    )
    if not isinstance(outcome, dict) or outcome.get("status") != outcome_status:
        raise ValueError("tool outcome status does not match the settlement")
    if outcome_status == "succeeded":
        if (
            not isinstance(outcome.get("result"), dict)
            or outcome.get("error") is not None
        ):
            raise ValueError("successful tool outcome has invalid result/error fields")
    elif not isinstance(outcome.get("error"), dict) or outcome.get("result") is not None:
        raise ValueError("unsuccessful tool outcome has invalid result/error fields")


def _validate_l1_approval_receipt_ids(payload: str) -> tuple[str, ...]:
    parsed = _validate_canonical_json(payload, label="approval receipt identifiers")
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item.strip() or len(item) > 200
        for item in parsed
    ):
        raise ValueError("approval receipt identifiers must be a string list")
    identifiers = tuple(parsed)
    if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("approval receipt identifiers must be sorted and unique")
    return identifiers


def _require_sha256(name: str, value: str | None) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a canonical sha256")


def _require_l1_protected_tool_dispatch_state(
    *,
    run: sqlite3.Row,
    state: sqlite3.Row,
    step: sqlite3.Row,
    window: sqlite3.Row,
    l1_turn_run_id: str,
) -> None:
    if (
        str(run["status"]) != "running"
        or str(state["stage"]) != "tool"
        or str(step["l1_turn_run_id"]) != l1_turn_run_id
        or str(step["status"]) != "decided"
        or str(step["action_kind"]) != "call_tools"
        or str(window["current_l1_attempt_id"] or "")
        != str(step["step_id"])
    ):
        raise L1TurnRunPersistenceError(
            "L1 protected ToolCall has no current dispatch authority"
        )


def _l1_protected_tool_state_guard_hash(
    *,
    run: sqlite3.Row,
    state: sqlite3.Row,
    step: sqlite3.Row,
    window: sqlite3.Row,
    tool_call_id: str,
    call_ordinal: int,
    tool_id: str,
    contract_version: str,
    implementation_version: str,
    arguments_hash: str | None,
    policy_json: str,
    effect_profile_sha256: str | None,
    provider_identity_sha256: str | None,
    approval_receipts_sha256: str | None,
) -> str:
    for name, value in (
        ("arguments_hash", arguments_hash),
        ("effect_profile_sha256", effect_profile_sha256),
        ("provider_identity_sha256", provider_identity_sha256),
        ("approval_receipts_sha256", approval_receipts_sha256),
    ):
        _require_sha256(name, value)
    return hashlib.sha256(
        _canonical_json(
            {
                "contract": "l1-protected-tool-state-guard-v1",
                "session_id": str(run["session_id"]),
                "turn_id": str(run["turn_id"]),
                "l1_turn_run_id": str(run["l1_turn_run_id"]),
                "step_id": str(step["step_id"]),
                "tool_call_id": tool_call_id,
                "call_ordinal": call_ordinal,
                "tool_id": tool_id,
                "contract_version": contract_version,
                "implementation_version": implementation_version,
                "run_revision": int(run["run_revision"]),
                "window_revision": int(window["state_version"]),
                "decision_hash": str(step["decision_hash"] or ""),
                "tool_call_count": int(step["tool_call_count"]),
                "semantic_steps_used": int(state["semantic_steps_used"]),
                "plan_hash": _optional_text(state["plan_hash"]),
                "observation_hash": _optional_text(
                    state["latest_observation_hash"]
                ),
                "catalog_snapshot_hash": str(state["catalog_snapshot_hash"]),
                "execution_config_hash": str(state["execution_config_hash"]),
                "corpus_manifest_hash": _optional_text(
                    state["corpus_manifest_hash"]
                ),
                "arguments_hash": arguments_hash,
                "policy_hash": hashlib.sha256(
                    policy_json.encode("utf-8")
                ).hexdigest(),
                "effect_profile_sha256": effect_profile_sha256,
                "provider_identity_sha256": provider_identity_sha256,
                "approval_receipts_sha256": approval_receipts_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()


def _l1_protected_tool_state_guard_from_row(
    *,
    run: sqlite3.Row,
    state: sqlite3.Row,
    step: sqlite3.Row,
    window: sqlite3.Row,
    row: sqlite3.Row,
) -> str:
    return _l1_protected_tool_state_guard_hash(
        run=run,
        state=state,
        step=step,
        window=window,
        tool_call_id=str(row["tool_call_id"]),
        call_ordinal=int(row["call_ordinal"]),
        tool_id=str(row["tool_id"]),
        contract_version=str(row["contract_version"]),
        implementation_version=str(row["implementation_version"]),
        arguments_hash=_optional_text(row["arguments_hash"]),
        policy_json=str(row["policy_json"]),
        effect_profile_sha256=_optional_text(row["effect_profile_sha256"]),
        provider_identity_sha256=_optional_text(row["provider_identity_sha256"]),
        approval_receipts_sha256=_optional_text(row["approval_receipts_sha256"]),
    )


def _l1_protected_operation_binding_hash(
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    step_id: str,
    tool_call_id: str,
    call_ordinal: int,
    tool_id: str,
    contract_version: str,
    implementation_version: str,
    arguments_hash: str,
    policy_json: str,
    effect_profile_sha256: str | None,
    provider_identity_sha256: str | None,
    approval_receipts_sha256: str | None,
    protected_state_guard_sha256: str,
) -> str:
    for name, value in (
        ("arguments_hash", arguments_hash),
        ("effect_profile_sha256", effect_profile_sha256),
        ("provider_identity_sha256", provider_identity_sha256),
        ("approval_receipts_sha256", approval_receipts_sha256),
        ("protected_state_guard_sha256", protected_state_guard_sha256),
    ):
        _require_sha256(name, value)
    return hashlib.sha256(
        _canonical_json(
            {
                "contract": "l1-protected-tool-operation-binding-v1",
                "session_id": session_id,
                "turn_id": turn_id,
                "l1_turn_run_id": l1_turn_run_id,
                "step_id": step_id,
                "tool_call_id": tool_call_id,
                "call_ordinal": call_ordinal,
                "tool_id": tool_id,
                "contract_version": contract_version,
                "implementation_version": implementation_version,
                "arguments_hash": arguments_hash,
                "policy_hash": hashlib.sha256(
                    policy_json.encode("utf-8")
                ).hexdigest(),
                "effect_profile_sha256": effect_profile_sha256,
                "provider_identity_sha256": provider_identity_sha256,
                "approval_receipts_sha256": approval_receipts_sha256,
                "protected_state_guard_sha256": protected_state_guard_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()


def _l1_protected_terminal_phase(outcome_status: str) -> str:
    if outcome_status == "succeeded":
        return "succeeded"
    if outcome_status in {"completion_unconfirmed", "timed_out", "cancelled"}:
        return "uncertain"
    return "terminal_failure"


def _l1_protected_dispatch_receipt_hash(
    *,
    tool_call_id: str,
    protected_operation_binding_sha256: str,
    physical_attempt_id: str,
    outcome_status: str,
    outcome_hash: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "l1-protected-tool-dispatch-receipt-v1",
                "tool_call_id": tool_call_id,
                "operation_binding_sha256": protected_operation_binding_sha256,
                "physical_attempt_id": physical_attempt_id,
                "outcome_status": outcome_status,
                "outcome_hash": outcome_hash,
            }
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _heartbeat_is_fresh(
    heartbeat_at: str | None,
    *,
    now: str,
    lease_seconds: int,
) -> bool:
    if heartbeat_at is None:
        return False
    try:
        heartbeat = datetime.fromisoformat(heartbeat_at)
        observed = datetime.fromisoformat(now)
    except ValueError:
        return False
    if heartbeat.tzinfo is None or observed.tzinfo is None:
        return False
    return observed - heartbeat <= timedelta(seconds=lease_seconds)


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be 1..200 non-whitespace characters")


__all__ = [
    "L1TurnRunPersistenceError",
    "begin_l1_protected_tool_dispatch",
    "close_l1_attempt",
    "commit_l1_attempt_decision",
    "complete_l1_run_in_transaction",
    "create_l1_turn_run",
    "claim_l1_turn_run_resume",
    "fail_l1_turn_run",
    "get_l1_attempt_state_guard",
    "get_l1_turn_execution",
    "get_l1_turn_run",
    "initialize_l1_turn_run",
    "reject_l1_final_reply_candidate",
    "renew_l1_turn_run_resume_lease",
    "reserve_l1_tool_call",
    "settle_l1_protected_tool_dispatch",
    "settle_l1_tool_call",
    "start_l1_attempt",
]
