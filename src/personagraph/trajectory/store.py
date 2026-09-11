"""限定在会话作用域内的诊断轨迹。

生产环境中的轨迹与所属对话一同保存在对应会话的 ``session.sqlite`` 中。
测试和离线诊断工具仍可显式指定路径；当没有活动会话作用域时，
该路径绝不会充当生产环境的回退位置。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    Blob,
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectory_blobs (
    sha256        TEXT PRIMARY KEY,
    byte_count    INTEGER NOT NULL,
    text          TEXT NOT NULL,
    truncated     INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    step_id         TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    session_id      TEXT,
    turn_id         TEXT,
    model_call_id   TEXT,
    purpose         TEXT,
    duration_ms     INTEGER,
    outcome         TEXT NOT NULL,
    reason_code     TEXT,
    metrics_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trajectory_parts (
    step_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL,
    PRIMARY KEY (step_id, seq),
    FOREIGN KEY (step_id) REFERENCES trajectory_steps(step_id) ON DELETE CASCADE,
    FOREIGN KEY (blob_sha256) REFERENCES trajectory_blobs(sha256)
);

CREATE INDEX IF NOT EXISTS idx_trajectory_step_turn
    ON trajectory_steps(turn_id, occurred_at, step_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_step_session
    ON trajectory_steps(session_id, occurred_at, step_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_step_call
    ON trajectory_steps(model_call_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_part_blob
    ON trajectory_parts(blob_sha256);
"""

_CURRENT_TABLES = frozenset(
    {
        "trajectory_blobs",
        "trajectory_steps",
        "trajectory_parts",
    }
)


class TrajectorySchemaError(RuntimeError):
    """轨迹数据库不符合当前唯一受支持的存储结构。"""


class TrajectoryReadError(TrajectorySchemaError):
    """无法在不修改数据库的前提下完整读取轨迹。"""


def _trajectory_table_names(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type='table' AND name GLOB 'trajectory_*'"
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _required_text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TrajectoryReadError(f"invalid trajectory {field}")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrajectoryReadError(f"invalid trajectory {field}")
    return value


def _required_integer(
    value: object,
    *,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrajectoryReadError(f"invalid trajectory {field}")
    return value


def _is_before(occurred_at: str, cutoff: datetime) -> bool:
    """先解析再比较时间戳，绝不直接按字符串比较。

    记录的时间戳虽然都采用 ISO-8601 格式，字节形态却并不相同——以 ``Z``
    结尾和采用 ``+00:00`` 偏移的值具有不同的排序结果，因此按字典序设置截止点
    会仅因写法不同而误删或漏删行。无法解析的时间戳视为尚未到期：多保留一行
    尚可补救，误删未到期的行则无法恢复。
    """

    try:
        parsed = datetime.fromisoformat(occurred_at)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed < cutoff


# 成功步骤的保留时长。该值由用户于 2026-08-22 选定，是一项明确决策，而非
# 看似安全的默认值；因此以具名策略保存在此处，而不是内联到某个调用点。
DEFAULT_RETENTION_DAYS = 30


class TrajectoryStore:
    """追加已记录的步骤，并以完整步骤为单位读回。"""

    def __init__(self, path: Path | str | None = None) -> None:
        # 默认门面每次操作都解析当前 Session 作用域；测试、离线读取器和迁移工具
        # 可以显式固定一个独立路径。
        self._explicit_path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        """每次访问都解析当前 Session 数据库；显式 path 仅供独立测试/离线使用。

        默认实例不缓存某个 Session 的路径，也不在缺少 scope 时退回全局 trajectory 库。
        """

        if self._explicit_path is not None:
            return self._explicit_path
        from ..session.store import current_session_database_path

        return current_session_database_path()

    @property
    def is_session_scoped(self) -> bool:
        """此门面是否通过活动会话绑定来解析路径。"""

        return self._explicit_path is None

    def _connect(self) -> sqlite3.Connection:
        # 先解析路径，确保缺少作用域的生产调用在初始化或创建目录前就失败。
        path = self.path
        if self.is_session_scoped:
            # 这些表归完整的会话基线所有。在此初始化基线，可避免仅含轨迹表的
            # 不完整模式污染新建的会话数据库。
            from ..session.store import init_db as init_session_db

            init_session_db()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if not self.is_session_scoped:
                table_names = _trajectory_table_names(connection)
                if table_names and table_names != _CURRENT_TABLES:
                    raise TrajectorySchemaError(
                        "trajectory database schema is not supported"
                    )
                with connection:
                    connection.executescript(_SCHEMA)
            return connection
        except Exception:
            connection.close()
            raise

    def _connect_read_only(self) -> sqlite3.Connection:
        try:
            path = self.path
        except Exception as exc:
            raise TrajectoryReadError(
                "trajectory database path is unavailable"
            ) from exc
        if not path.exists():
            raise TrajectoryReadError("trajectory database does not exist")
        if not path.is_file():
            raise TrajectoryReadError("trajectory database path is not a file")

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA query_only = ON")
            return connection
        except (OSError, sqlite3.Error) as exc:
            if connection is not None:
                connection.close()
            raise TrajectoryReadError("trajectory database is unreadable") from exc

    # --- 写入 -------------------------------------------------------------

    def record(self, step: Step) -> bool:
        """存储一个步骤及其引用的全部内容；已存在时返回 ``False``。

        重复记录同一步骤属于重放而非错误：重试写入不得重复增加记录，也不得
        扰动已有内容。

        一个事务内保存 Step、按顺序排列的 Parts 和按 SHA-256 去重的 Blobs。
        去重键是调用方生成的 step_id；这里不判断 Attempt 是否执行过，不承担重试或恢复
        authority。读取整条轨迹只能还原“记录投影”，不能由省略正文的 hash 自动恢复原文。
        """

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM trajectory_steps WHERE step_id=?",
                (step.step_id,),
            ).fetchone()
            if existing is not None:
                return False
            for part in step.parts:
                self._insert_blob(connection, part.blob, step.occurred_at)
            connection.execute(
                "INSERT INTO trajectory_steps (step_id, kind, occurred_at, session_id,"
                " turn_id, model_call_id, purpose, duration_ms, outcome, reason_code,"
                " metrics_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    step.step_id,
                    step.kind.value,
                    step.occurred_at,
                    step.session_id,
                    step.turn_id,
                    step.model_call_id,
                    step.purpose,
                    step.duration_ms,
                    step.outcome.value,
                    step.reason_code,
                    json.dumps(step.metrics, sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.executemany(
                "INSERT INTO trajectory_parts (step_id, seq, role, blob_sha256)"
                " VALUES (?,?,?,?)",
                [
                    (step.step_id, seq, part.role.value, part.blob.sha256)
                    for seq, part in enumerate(step.parts)
                ],
            )
        return True

    @staticmethod
    def _insert_blob(connection: sqlite3.Connection, blob: Blob, now: str) -> None:
        """每段内容仅保存一次；重复正是去重目的，并非冲突。"""

        connection.execute(
            "INSERT OR IGNORE INTO trajectory_blobs"
            " (sha256, byte_count, text, truncated, first_seen_at)"
            " VALUES (?,?,?,?,?)",
            (
                blob.sha256,
                blob.byte_count,
                blob.text,
                1 if blob.truncated else 0,
                now,
            ),
        )

    # --- 读取 -------------------------------------------------------------

    def get(self, step_id: str) -> Step | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trajectory_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if row is None:
                return None
            return self._step_from_row(connection, row)

    def steps_for_turn(self, turn_id: str) -> tuple[Step, ...]:
        return self._query("WHERE turn_id=? ORDER BY occurred_at, step_id", (turn_id,))

    def steps_for_session(self, session_id: str, *, limit: int = 200) -> tuple[Step, ...]:
        return self._query(
            "WHERE session_id=? ORDER BY occurred_at DESC, step_id DESC LIMIT ?",
            (session_id, limit),
        )

    def steps_for_model_call(self, model_call_id: str) -> tuple[Step, ...]:
        return self._query(
            "WHERE model_call_id=? ORDER BY occurred_at, step_id", (model_call_id,)
        )

    def read_all(self) -> dict[str, Any]:
        """在同一只读快照中导出数据库保存的全部轨迹内容。

        该入口不创建数据库、不补表、不迁移数据，也不采用交互界面读取时的行数
        上限。任何会破坏完整导出的结构或标量异常都会使整个读取失败。
        """

        connection = self._connect_read_only()
        try:
            connection.execute("BEGIN")
            table_names = _trajectory_table_names(connection)
            missing_tables = _CURRENT_TABLES - table_names
            if missing_tables:
                raise TrajectoryReadError("trajectory database schema is incomplete")
            if table_names != _CURRENT_TABLES:
                raise TrajectoryReadError(
                    "trajectory database schema is not supported"
                )

            step_rows = connection.execute(
                "SELECT step_id, kind, occurred_at, session_id, turn_id,"
                " model_call_id, purpose, duration_ms, outcome, reason_code,"
                " metrics_json"
                " FROM trajectory_steps ORDER BY occurred_at, step_id"
            ).fetchall()
            part_rows = connection.execute(
                "SELECT step_id, seq, role, blob_sha256"
                " FROM trajectory_parts ORDER BY step_id, seq"
            ).fetchall()
            blob_rows = connection.execute(
                "SELECT sha256, byte_count, text, truncated, first_seen_at"
                " FROM trajectory_blobs ORDER BY sha256"
            ).fetchall()
            snapshot = _snapshot_from_rows(
                step_rows=step_rows,
                part_rows=part_rows,
                blob_rows=blob_rows,
            )
            connection.commit()
            return snapshot
        except TrajectoryReadError:
            connection.rollback()
            raise
        except (json.JSONDecodeError, sqlite3.Error, TypeError, ValueError) as exc:
            connection.rollback()
            raise TrajectoryReadError("trajectory database content is invalid") from exc
        finally:
            connection.close()

    def _query(self, clause: str, params: Sequence[object]) -> tuple[Step, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM trajectory_steps {clause}", tuple(params)
            ).fetchall()
            return tuple(self._step_from_row(connection, row) for row in rows)

    def _step_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Step:
        parts = connection.execute(
            "SELECT p.role, b.sha256, b.byte_count, b.text, b.truncated"
            " FROM trajectory_parts AS p"
            " JOIN trajectory_blobs AS b ON b.sha256=p.blob_sha256"
            " WHERE p.step_id=? ORDER BY p.seq",
            (row["step_id"],),
        ).fetchall()
        return Step(
            step_id=str(row["step_id"]),
            kind=StepKind(str(row["kind"])),
            occurred_at=str(row["occurred_at"]),
            parts=tuple(
                Part(
                    role=PartRole(str(part["role"])),
                    blob=Blob(
                        sha256=str(part["sha256"]),
                        byte_count=int(part["byte_count"]),
                        text=str(part["text"]),
                        truncated=bool(part["truncated"]),
                    ),
                )
                for part in parts
            ),
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            model_call_id=row["model_call_id"],
            purpose=row["purpose"],
            duration_ms=row["duration_ms"],
            outcome=StepOutcome(str(row["outcome"])),
            reason_code=row["reason_code"],
            metrics=json.loads(str(row["metrics_json"] or "{}")),
        )

    # --- 维护 -------------------------------------------------------------

    def delete_session(self, session_id: str) -> int:
        """删除一个会话的步骤；其与其他会话共享的内容块仍会保留。"""

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM trajectory_steps WHERE session_id=?", (session_id,)
            )
            return int(cursor.rowcount or 0)

    def prune(self, *, older_than_days: int, keep_failures: bool = True) -> dict[str, int]:
        """删除超过保留期限的步骤，再回收已无引用的内容。

        ``keep_failures`` 会使被拒绝和失败的步骤不受期限规则影响。两项规则共同
        保证操作安全：如果永久保留失败记录，却没有近期成功记录可供对照，就
        无法判断异常；通常只有先知道“正常时是什么样”，才能认出当前并不正常。

        这里刻意不提供默认期限。悄然生效的保留期等于替用户作出他们从未做过的
        决策。
        """

        if isinstance(older_than_days, bool) or not isinstance(older_than_days, int):
            raise TypeError("older_than_days must be an integer")
        if older_than_days < 0:
            raise ValueError("older_than_days must not be negative")

        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT step_id, occurred_at, outcome FROM trajectory_steps"
            ).fetchall()
            doomed = [
                str(row["step_id"])
                for row in rows
                if not (keep_failures and str(row["outcome"]) != StepOutcome.OK.value)
                and _is_before(str(row["occurred_at"]), cutoff)
            ]
            connection.executemany(
                "DELETE FROM trajectory_steps WHERE step_id=?",
                [(step_id,) for step_id in doomed],
            )
        return {
            "removed_steps": len(doomed),
            "removed_blobs": self.collect_unreferenced_blobs(),
        }

    def collect_unreferenced_blobs(self) -> int:
        """删除已不再被任何步骤引用的内容块。

        此操作与步骤删除分离，而不采用级联删除：内容块在设计上可被共享，
        因此它何时不再需要取决于整个存储，而非碰巧最后被删除的某个步骤。
        """

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM trajectory_blobs WHERE sha256 NOT IN"
                " (SELECT blob_sha256 FROM trajectory_parts)"
            )
            return int(cursor.rowcount or 0)

    def usage(self) -> dict[str, int]:
        """返回行数和字节数，用于判断是否需要制定保留策略。

        ``stored_bytes`` 表示实际落盘大小；``referenced_bytes`` 表示同一内容若按
        每次引用各写一份所需的大小。两者之差就是去重节省的空间，也是随着片段
        变长而值得持续关注的指标。
        """

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS blobs,"
                # 对 TEXT 使用 LENGTH() 得到字符数；转换为 BLOB 后才得到字节数，
                # 只有后者能与 byte_count 比较。
                " COALESCE(SUM(LENGTH(CAST(text AS BLOB))),0) AS stored_bytes,"
                " COALESCE(SUM(byte_count),0) AS logical_bytes"
                " FROM trajectory_blobs"
            ).fetchone()
            steps = connection.execute(
                "SELECT COUNT(*) FROM trajectory_steps"
            ).fetchone()[0]
            refs = connection.execute(
                "SELECT COUNT(*) FROM trajectory_parts"
            ).fetchone()[0]
            referenced = connection.execute(
                "SELECT COALESCE(SUM(b.byte_count),0) FROM trajectory_parts AS p"
                " JOIN trajectory_blobs AS b ON b.sha256=p.blob_sha256"
            ).fetchone()[0]
            return {
                "steps": int(steps),
                "parts": int(refs),
                "blobs": int(row["blobs"]),
                "stored_bytes": int(row["stored_bytes"]),
                "referenced_bytes": int(referenced),
            }


def _snapshot_from_rows(
    *,
    step_rows: Sequence[sqlite3.Row],
    part_rows: Sequence[sqlite3.Row],
    blob_rows: Sequence[sqlite3.Row],
) -> dict[str, Any]:
    blobs = _export_blobs(blob_rows)
    step_ids: set[str] = set()
    steps: list[dict[str, Any]] = []
    for row in step_rows:
        step = _export_step(row)
        step_id = step["step_id"]
        if step_id in step_ids:
            raise TrajectoryReadError("duplicate trajectory step_id")
        step_ids.add(step_id)
        steps.append(step)

    parts_by_step: dict[str, list[dict[str, Any]]] = {
        step_id: [] for step_id in step_ids
    }
    for row in part_rows:
        step_id = _required_text(row["step_id"], field="part step_id")
        if step_id not in step_ids:
            raise TrajectoryReadError("orphan trajectory part")
        sequence = _required_integer(row["seq"], field="part sequence")
        expected_sequence = len(parts_by_step[step_id])
        if sequence != expected_sequence:
            raise TrajectoryReadError("trajectory part sequence is not contiguous")
        role_text = _required_text(row["role"], field="part role")
        try:
            role = PartRole(role_text)
        except ValueError as exc:
            raise TrajectoryReadError("invalid trajectory part role") from exc
        blob_sha256 = _required_text(
            row["blob_sha256"],
            field="part blob_sha256",
        )
        if blob_sha256 not in blobs:
            raise TrajectoryReadError("trajectory part references a missing blob")
        parts_by_step[step_id].append(
            {
                "seq": sequence,
                "role": role.value,
                "blob_sha256": blob_sha256,
            }
        )

    for step in steps:
        step["parts"] = parts_by_step[step["step_id"]]

    return {
        "format": "personagraph.trajectory",
        "steps": steps,
        "blobs": blobs,
        "integrity": {
            "step_count": len(steps),
            "part_count": len(part_rows),
            "blob_count": len(blobs),
            "truncated_blob_count": sum(
                1 for blob in blobs.values() if blob["truncated"]
            ),
            "recording_failure_count": sum(
                1
                for step in steps
                if step["kind"] == StepKind.RECORDING_FAILURE.value
            ),
        },
    }


def _export_blobs(
    rows: Sequence[sqlite3.Row],
) -> dict[str, dict[str, Any]]:
    exported: dict[str, dict[str, Any]] = {}
    for row in rows:
        sha256 = _required_text(row["sha256"], field="blob sha256")
        byte_count = _required_integer(row["byte_count"], field="blob byte_count")
        text = _required_text(row["text"], field="blob text", allow_empty=True)
        truncated_value = _required_integer(row["truncated"], field="blob truncated")
        if truncated_value not in {0, 1}:
            raise TrajectoryReadError("invalid trajectory blob truncated flag")
        truncated = bool(truncated_value)
        first_seen_at = _required_text(
            row["first_seen_at"],
            field="blob first_seen_at",
        )
        try:
            Blob(
                sha256=sha256,
                byte_count=byte_count,
                text=text,
                truncated=truncated,
            )
        except (TypeError, ValueError) as exc:
            raise TrajectoryReadError("invalid trajectory blob scalar") from exc
        encoded_text = text.encode("utf-8")
        stored_byte_count = len(encoded_text)
        if stored_byte_count > byte_count:
            raise TrajectoryReadError("invalid trajectory blob byte_count")
        if not truncated:
            if stored_byte_count != byte_count:
                raise TrajectoryReadError("invalid trajectory blob byte_count")
            if hashlib.sha256(encoded_text).hexdigest() != sha256:
                raise TrajectoryReadError("invalid trajectory blob digest")
        if sha256 in exported:
            raise TrajectoryReadError("duplicate trajectory blob sha256")
        exported[sha256] = {
            "byte_count": byte_count,
            "text": text,
            "truncated": truncated,
            "first_seen_at": first_seen_at,
        }
    return exported


def _export_step(row: sqlite3.Row) -> dict[str, Any]:
    step_id = _required_text(row["step_id"], field="step_id")
    kind_text = _required_text(row["kind"], field="step kind")
    try:
        kind = StepKind(kind_text)
    except ValueError as exc:
        raise TrajectoryReadError("invalid trajectory step kind") from exc
    outcome_text = _required_text(row["outcome"], field="step outcome")
    try:
        outcome = StepOutcome(outcome_text)
    except ValueError as exc:
        raise TrajectoryReadError("invalid trajectory step outcome") from exc

    duration_value = row["duration_ms"]
    duration_ms = (
        None
        if duration_value is None
        else _required_integer(duration_value, field="step duration_ms")
    )
    reason_code = _optional_text(row["reason_code"], field="step reason_code")
    if outcome is not StepOutcome.OK and not reason_code:
        raise TrajectoryReadError(
            "invalid trajectory step reason_code for unsuccessful outcome"
        )

    return {
        "step_id": step_id,
        "kind": kind.value,
        "occurred_at": _required_text(
            row["occurred_at"],
            field="step occurred_at",
        ),
        "session_id": _optional_text(row["session_id"], field="step session_id"),
        "turn_id": _optional_text(row["turn_id"], field="step turn_id"),
        "model_call_id": _optional_text(
            row["model_call_id"],
            field="step model_call_id",
        ),
        "purpose": _optional_text(row["purpose"], field="step purpose"),
        "duration_ms": duration_ms,
        "outcome": outcome.value,
        "reason_code": reason_code,
        "metrics": _metrics_from_json(row["metrics_json"]),
        "parts": [],
    }


def _metrics_from_json(value: object) -> dict[str, int]:
    if not isinstance(value, str):
        raise TrajectoryReadError("invalid trajectory metrics_json scalar")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TrajectoryReadError("invalid trajectory metrics_json") from exc
    if not isinstance(parsed, dict):
        raise TrajectoryReadError("invalid trajectory metrics_json object")
    if any(
        not isinstance(key, str)
        or isinstance(metric, bool)
        or not isinstance(metric, int)
        for key, metric in parsed.items()
    ):
        raise TrajectoryReadError("invalid trajectory metrics_json values")
    return dict(parsed)


_ACTIVE: TrajectoryStore | None = None


def active_store() -> TrajectoryStore:
    """返回可复用的门面；其生产路径按会话作用域动态解析。"""

    global _ACTIVE

    if _ACTIVE is None:
        _ACTIVE = TrajectoryStore()
    return _ACTIVE
