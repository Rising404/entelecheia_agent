"""模型供应商配额 scope 的跨进程准入队列。

队列被刻意设为独立于供应商和 Runtime 请求代码。它只回答一个问题：*现在可以开始一次物理
请求吗？* 调用方持久化自身逻辑/物理模型调用 authority，为每次物理尝试排入一张 ticket，
取得 lease、发送请求并结算 lease。

一个 SQLite 数据库协调使用同一配额 scope 的所有进程。凭据绝不会跨越该持久化边界；只存储
确定性 scope hash。重试是一张新 ticket（通常是下一个物理序号），因此会排到已经等待的
首次尝试之后。延迟 ticket 在 ``eligible_at`` 前会被跳过，故不会阻塞已就绪工作。

RPM 与 TPM 使用滚动 60 秒窗口。每周 token 限制采用保守的滚动七天窗口，因为供应商没有
公开统一的日历重置契约。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Iterator, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4


MINUTE_SECONDS = 60.0
WEEK_SECONDS = 7.0 * 24.0 * 60.0 * 60.0
_SCHEMA_VERSION = 1
_SCOPE_HASH_RE = re.compile(r"[0-9a-f]{64}")


class QuotaQueueError(RuntimeError):
    """配额队列基础错误。"""


class QuotaScopeNotRegistered(QuotaQueueError):
    """请求的 scope 没有持久配额配置。"""


class QuotaScopeConfigurationConflict(QuotaQueueError):
    """两个进程为同一 scope hash 提供了不同限制。"""


class QuotaTicketConflict(QuotaQueueError):
    """逻辑调用/物理序号身份被以不同方式复用。"""


class QuotaReservationImpossible(QuotaQueueError):
    """一张 ticket 永远无法满足已配置 token 限制。"""


class QuotaLeaseStale(QuotaQueueError):
    """lease 已过期、已恢复，或不再拥有其 ticket。"""


class QuotaTicketNotAcquirable(QuotaQueueError):
    """请求的 ticket 已终止或当前由其他方拥有。"""


class QuotaLeasePhaseError(QuotaQueueError):
    """在 lease 的有效分派阶段之前/之后尝试了 lease 操作。"""


class QuotaWaitReason(StrEnum):
    """一次非阻塞获取尝试未授予 lease 的原因。"""

    EMPTY = "empty"
    TICKET_TERMINAL = "ticket_terminal"
    NOT_YET_ELIGIBLE = "not_yet_eligible"
    FIFO_PREDECESSOR = "fifo_predecessor"
    SCOPE_COOLDOWN = "scope_cooldown"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_WEEK = "tokens_per_week"
    MAX_IN_FLIGHT = "max_in_flight"


class QuotaDispatchBlocked(QuotaQueueError):
    """分派前 lease 不再满足当前 scope 策略。

    可以确定尚未调用 Provider，且 lease 仍处于 reserved 阶段。因此调用方可以安全地把该
    lease 传给 :meth:`ModelApiQuotaQueue.abandon_before_dispatch`，并选择 ``requeue`` 或
    ``cancel`` disposition。
    """

    def __init__(
        self,
        *,
        blocked_by: tuple[QuotaWaitReason, ...],
        retry_at: float | None,
    ) -> None:
        super().__init__("quota policy blocked the pre-dispatch transition")
        self.blocked_by = blocked_by
        self.retry_at = retry_at


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    """映射到同一配额 scope 的所有模型 profile 共享的限制。"""

    requests_per_minute: int | None
    tokens_per_minute: int | None
    tokens_per_week: int | None
    max_in_flight: int | None

    def __post_init__(self) -> None:
        for name in (
            "requests_per_minute",
            "tokens_per_minute",
            "tokens_per_week",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_in_flight is not None and (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or self.max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class QuotaTicket:
    """一次已排队的物理 Provider 尝试。"""

    ticket_id: str
    scope_hash: str
    logical_call_id: str
    physical_ordinal: int | None
    eligible_at: float
    deadline: float
    token_reservation: int
    queue_sequence: int
    queue_owner: str
    queue_heartbeat_until: float
    queue_heartbeat_ttl: float
    status: Literal[
        "queued",
        "leased",
        "settled",
        "cancelled",
        "expired",
        "reconciliation_required",
    ]


@dataclass(frozen=True, slots=True)
class QuotaLease:
    """分派一张 ticket 的排他且会过期的 authority。"""

    lease_id: str
    lease_owner: str
    lease_until: float
    ticket: QuotaTicket


@dataclass(frozen=True, slots=True)
class QuotaAcquireResult:
    """一次尝试 lease 下一张就绪 ticket 的非阻塞结果。"""

    lease: QuotaLease | None
    blocked_by: tuple[QuotaWaitReason, ...] = ()
    retry_at: float | None = None

    @property
    def acquired(self) -> bool:
        return self.lease is not None


@dataclass(frozen=True, slots=True)
class QuotaUsageSnapshot:
    """在某一时刻投影的当前持久用量。"""

    requests_last_minute: int
    tokens_last_minute: int
    tokens_last_week: int
    in_flight: int
    admitted_not_dispatched: int
    cooldown_until: float


Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def derive_quota_scope_hash(
    *,
    base_url: str,
    credential: str,
    quota_group: str | None = None,
) -> str:
    """返回稳定 scope hash，同时不公开或保留凭据。

    默认情况下，不同凭据属于不同 scope。提供显式 ``quota_group`` 会有意让同一 base URL
    下的凭据共享一个 scope，以覆盖账户级配额由多个 key 共享的供应商。
    """

    normalized_url = _normalize_base_url(base_url)
    group = (quota_group or "").strip()
    if group:
        identity = {"kind": "quota_group", "value": group}
    else:
        if not credential:
            raise ValueError("credential is required when quota_group is absent")
        identity = {
            "kind": "credential_fingerprint",
            "value": sha256(credential.encode("utf-8")).hexdigest(),
        }
    material = json.dumps(
        {"base_url": normalized_url, "identity": identity},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(material).hexdigest()


class ModelApiQuotaQueue:
    """由 SQLite 支持、按供应商配额 scope 分组的 FIFO 准入队列。"""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Clock = time.time,
        sleep: Sleeper = time.sleep,
        in_flight_poll_seconds: float = 0.25,
        queued_ticket_ttl_seconds: float = 15.0,
        sqlite_timeout_seconds: float = 30.0,
    ) -> None:
        if in_flight_poll_seconds <= 0 or not math.isfinite(in_flight_poll_seconds):
            raise ValueError("in_flight_poll_seconds must be finite and positive")
        if sqlite_timeout_seconds <= 0 or not math.isfinite(sqlite_timeout_seconds):
            raise ValueError("sqlite_timeout_seconds must be finite and positive")
        if queued_ticket_ttl_seconds <= 0 or not math.isfinite(
            queued_ticket_ttl_seconds
        ):
            raise ValueError("queued_ticket_ttl_seconds must be finite and positive")
        self.database_path = Path(database_path)
        self._clock = clock
        self._sleep = sleep
        self._in_flight_poll_seconds = float(in_flight_poll_seconds)
        self._queued_ticket_ttl_seconds = float(queued_ticket_ttl_seconds)
        self._sqlite_timeout_seconds = float(sqlite_timeout_seconds)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_scope(self, scope_hash: str, limits: QuotaLimits) -> None:
        """创建一个配额 scope，或精确验证其现有限制。"""

        _require_scope_hash(scope_hash)
        if not isinstance(limits, QuotaLimits):
            raise TypeError("limits must be QuotaLimits")
        now = self._now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM model_api_quota_scopes WHERE scope_hash = ?",
                (scope_hash,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO model_api_quota_scopes "
                    "(scope_hash, requests_per_minute, tokens_per_minute, "
                    "tokens_per_week, max_in_flight, cooldown_until, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?)",
                    (
                        scope_hash,
                        limits.requests_per_minute,
                        limits.tokens_per_minute,
                        limits.tokens_per_week,
                        limits.max_in_flight,
                        now,
                    ),
                )
                return
            if _limits_from_row(row) != limits:
                raise QuotaScopeConfigurationConflict(
                    "quota scope already exists with different limits"
                )

    def configure_scope(self, scope_hash: str, limits: QuotaLimits) -> None:
        """原子创建或替换一个 scope 的当前策略。

        与 :meth:`register_scope` 不同，这是显式可变策略 API。历史分派保持完整，并立即按
        新滚动窗口限制评估。更新时保留 cooldown 状态。
        """

        _require_scope_hash(scope_hash)
        if not isinstance(limits, QuotaLimits):
            raise TypeError("limits must be QuotaLimits")
        now = self._now()
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO model_api_quota_scopes "
                "(scope_hash, requests_per_minute, tokens_per_minute, "
                "tokens_per_week, max_in_flight, cooldown_until, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(scope_hash) DO UPDATE SET "
                "requests_per_minute = excluded.requests_per_minute, "
                "tokens_per_minute = excluded.tokens_per_minute, "
                "tokens_per_week = excluded.tokens_per_week, "
                "max_in_flight = excluded.max_in_flight",
                (
                    scope_hash,
                    limits.requests_per_minute,
                    limits.tokens_per_minute,
                    limits.tokens_per_week,
                    limits.max_in_flight,
                    now,
                ),
            )

    def enqueue(
        self,
        *,
        scope_hash: str,
        logical_call_id: str,
        physical_ordinal: int | None,
        token_reservation: int,
        queue_owner: str,
        eligible_at: float | None = None,
        deadline: float,
        ticket_id: str | None = None,
    ) -> QuotaTicket:
        """将一次物理尝试追加到其 scope 队列。

        ``ticket_id`` 是唯一幂等身份。``physical_ordinal`` 是可选可观测性元数据：持久
        Runtime 恢复可能要到配额准入后才知道最终序号，且不得与更早已结算 ticket 冲突。
        重试通常会获得新 ticket ID，因而也获得新队列序列。
        """

        _require_scope_hash(scope_hash)
        logical_call_id = _require_nonempty(logical_call_id, "logical_call_id")
        queue_owner = _require_nonempty(queue_owner, "queue_owner")
        if physical_ordinal is not None and (
            not isinstance(physical_ordinal, int)
            or isinstance(physical_ordinal, bool)
            or physical_ordinal <= 0
        ):
            raise ValueError("physical_ordinal must be a positive integer or None")
        if (
            not isinstance(token_reservation, int)
            or isinstance(token_reservation, bool)
            or token_reservation < 0
        ):
            raise ValueError("token_reservation must be a non-negative integer")
        now = self._now()
        ready = now if eligible_at is None else _finite_time(eligible_at, "eligible_at")
        expires = _finite_time(deadline, "deadline")
        if expires <= ready:
            raise ValueError("deadline must be later than eligible_at")
        heartbeat_until = min(expires, now + self._queued_ticket_ttl_seconds)
        requested_ticket_id = (
            _require_nonempty(ticket_id, "ticket_id") if ticket_id else uuid4().hex
        )

        with self._transaction() as conn:
            scope = self._require_scope(conn, scope_hash)
            limits = _limits_from_row(scope)
            if (
                limits.tokens_per_minute is not None
                and token_reservation > limits.tokens_per_minute
            ):
                raise QuotaReservationImpossible(
                    "ticket reservation exceeds the scope TPM limit"
                )
            if (
                limits.tokens_per_week is not None
                and token_reservation > limits.tokens_per_week
            ):
                raise QuotaReservationImpossible(
                    "ticket reservation exceeds the scope weekly token limit"
                )
            existing = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (requested_ticket_id,),
            ).fetchone()
            if ticket_id is not None and existing is not None:
                if (
                    existing["scope_hash"] != scope_hash
                    or existing["logical_call_id"] != logical_call_id
                    or existing["physical_ordinal"] != physical_ordinal
                    or float(existing["eligible_at"]) != ready
                    or float(existing["deadline"]) != expires
                    or int(existing["token_reservation"]) != token_reservation
                    or existing["queue_owner"] != queue_owner
                ):
                    raise QuotaTicketConflict(
                        "ticket_id crossed immutable queue fields"
                    )
                return _ticket_from_row(existing)

            collision = conn.execute(
                "SELECT 1 FROM model_api_quota_tickets WHERE ticket_id = ?",
                (requested_ticket_id,),
            ).fetchone()
            if collision is not None:
                raise QuotaTicketConflict("ticket_id is already in use")
            queue_sequence = self._next_queue_sequence(conn)
            conn.execute(
                "INSERT INTO model_api_quota_tickets "
                "(ticket_id, scope_hash, logical_call_id, physical_ordinal, "
                "eligible_at, deadline, token_reservation, queue_sequence, "
                "queue_owner, queue_heartbeat_until, queue_heartbeat_ttl, "
                "status, lease_id, lease_owner, lease_until, settled_outcome, "
                "created_at, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, "
                "NULL, NULL, NULL, ?, NULL)",
                (
                    requested_ticket_id,
                    scope_hash,
                    logical_call_id,
                    physical_ordinal,
                    ready,
                    expires,
                    token_reservation,
                    queue_sequence,
                    queue_owner,
                    heartbeat_until,
                    self._queued_ticket_ttl_seconds,
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (requested_ticket_id,),
            ).fetchone()
            assert stored is not None
            return _ticket_from_row(stored)

    def heartbeat_ticket(
        self,
        ticket_id: str,
        *,
        queue_owner: str,
    ) -> QuotaTicket:
        """让调用方所有的已排队 ticket 在等待时保持有效。

        heartbeat 无法复活超过已存 TTL 或任务 deadline 的 ticket。在
        :meth:`wait_for_ticket` 外执行工作的调用方可以显式调用；阻塞等待方法会自动续期。
        """

        ticket_id = _require_nonempty(ticket_id, "ticket_id")
        owner = _require_nonempty(queue_owner, "queue_owner")
        now = self._now()
        expired = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None or row["status"] != "queued":
                raise QuotaTicketNotAcquirable("ticket is not queued")
            if row["queue_owner"] != owner:
                raise QuotaTicketNotAcquirable(
                    "ticket belongs to a different queue owner"
                )
            if (
                float(row["queue_heartbeat_until"]) <= now
                or float(row["deadline"]) <= now
            ):
                conn.execute(
                    "UPDATE model_api_quota_tickets SET status = 'expired' "
                    "WHERE ticket_id = ? AND status = 'queued'",
                    (ticket_id,),
                )
                expired = True
            else:
                heartbeat_until = min(
                    float(row["deadline"]),
                    now + float(row["queue_heartbeat_ttl"]),
                )
                conn.execute(
                    "UPDATE model_api_quota_tickets SET queue_heartbeat_until = ? "
                    "WHERE ticket_id = ? AND status = 'queued'",
                    (heartbeat_until, ticket_id),
                )
                updated = conn.execute(
                    "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                    (ticket_id,),
                ).fetchone()
                assert updated is not None
                return _ticket_from_row(updated)
        assert expired
        raise QuotaTicketNotAcquirable("queued ticket heartbeat expired")

    def try_acquire_ticket(
        self,
        ticket_id: str,
        *,
        lease_owner: str,
        lease_seconds: float,
    ) -> QuotaAcquireResult:
        """尝试一次，精确 lease 调用方所有的 ticket。

        只有当该 ticket 是其 scope 中当前合格且最早排队的 ticket 时才会授予。这里刻意不采用
        ``acquire_next(scope)`` API：请求字节和回调保留在调用方本地，把其他人的 ticket
        交给该调用方会把错误 payload 绑定到 lease。
        """

        ticket_id = _require_nonempty(ticket_id, "ticket_id")
        owner = _require_nonempty(lease_owner, "lease_owner")
        duration = _finite_time(lease_seconds, "lease_seconds")
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        now = self._now()

        with self._transaction() as conn:
            original = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if original is None:
                raise QuotaTicketNotAcquirable("ticket does not exist")
            if original["queue_owner"] != owner:
                raise QuotaTicketNotAcquirable(
                    "ticket belongs to a different queue owner"
                )
            scope_hash = str(original["scope_hash"])
            scope = self._require_scope(conn, scope_hash)
            self._recover_expired_leases(conn, scope_hash=scope_hash, now=now)
            conn.execute(
                "UPDATE model_api_quota_tickets SET status = 'expired' "
                "WHERE scope_hash = ? AND status = 'queued' "
                "AND (deadline <= ? OR queue_heartbeat_until <= ?)",
                (scope_hash, now, now),
            )
            candidate = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            assert candidate is not None
            if candidate["status"] == "leased":
                if candidate["lease_owner"] != owner:
                    raise QuotaTicketNotAcquirable(
                        "ticket is leased by a different owner"
                    )
                return QuotaAcquireResult(
                    lease=QuotaLease(
                        lease_id=str(candidate["lease_id"]),
                        lease_owner=owner,
                        lease_until=float(candidate["lease_until"]),
                        ticket=_ticket_from_row(candidate),
                    )
                )
            if candidate["status"] != "queued":
                return QuotaAcquireResult(
                    lease=None,
                    blocked_by=(QuotaWaitReason.TICKET_TERMINAL,),
                    retry_at=None,
                )
            if float(candidate["eligible_at"]) > now:
                retry_at = float(candidate["eligible_at"])
                reasons = [QuotaWaitReason.NOT_YET_ELIGIBLE]
                cooldown_until = float(scope["cooldown_until"])
                if cooldown_until > retry_at:
                    retry_at = cooldown_until
                    reasons.append(QuotaWaitReason.SCOPE_COOLDOWN)
                return QuotaAcquireResult(
                    lease=None,
                    blocked_by=tuple(reasons),
                    retry_at=retry_at,
                )

            head = conn.execute(
                "SELECT ticket_id FROM model_api_quota_tickets "
                "WHERE scope_hash = ? AND status = 'queued' "
                "AND eligible_at <= ? AND deadline > ? "
                "ORDER BY queue_sequence ASC LIMIT 1",
                (scope_hash, now, now),
            ).fetchone()
            if head is None or head["ticket_id"] != ticket_id:
                return QuotaAcquireResult(
                    lease=None,
                    blocked_by=(QuotaWaitReason.FIFO_PREDECESSOR,),
                    retry_at=now + self._in_flight_poll_seconds,
                )

            limits = _limits_from_row(scope)
            reservation = int(candidate["token_reservation"])
            if (
                limits.tokens_per_minute is not None
                and reservation > limits.tokens_per_minute
            ) or (
                limits.tokens_per_week is not None
                and reservation > limits.tokens_per_week
            ):
                conn.execute(
                    "UPDATE model_api_quota_tickets SET status = 'expired' "
                    "WHERE ticket_id = ? AND status = 'queued'",
                    (ticket_id,),
                )
                return QuotaAcquireResult(
                    lease=None,
                    blocked_by=(QuotaWaitReason.TICKET_TERMINAL,),
                    retry_at=None,
                )

            constraints = self._quota_constraints(
                conn,
                scope=scope,
                reservation=reservation,
                now=now,
            )
            if constraints:
                retry_at = max(item[1] for item in constraints)
                if retry_at >= float(candidate["deadline"]):
                    conn.execute(
                        "UPDATE model_api_quota_tickets SET status = 'expired' "
                        "WHERE ticket_id = ? AND status = 'queued'",
                        (ticket_id,),
                    )
                    return QuotaAcquireResult(
                        lease=None,
                        blocked_by=(QuotaWaitReason.TICKET_TERMINAL,),
                        retry_at=None,
                    )
                return QuotaAcquireResult(
                    lease=None,
                    blocked_by=tuple(item[0] for item in constraints),
                    retry_at=retry_at,
                )

            lease_id = uuid4().hex
                # Provider 分派前，lease 只是准入 authority；ticket 发送 deadline 后它绝不能
                # 继续可用。只有发送转换提交后，mark_dispatched 才能将其延长到该 deadline 后。
            lease_until = min(float(candidate["deadline"]), now + duration)
            conn.execute(
                "UPDATE model_api_quota_tickets "
                "SET status = 'leased', lease_id = ?, lease_owner = ?, "
                "lease_until = ? WHERE ticket_id = ? AND status = 'queued'",
                (lease_id, owner, lease_until, ticket_id),
            )
            conn.execute(
                "INSERT INTO model_api_quota_dispatches "
                "(scope_hash, ticket_id, lease_id, phase, reserved_at, "
                "started_at, charged_tokens, actual_tokens, settled_at) "
                "VALUES (?, ?, ?, 'reserved', ?, NULL, ?, NULL, NULL)",
                (
                    scope_hash,
                    ticket_id,
                    lease_id,
                    now,
                    int(candidate["token_reservation"]),
                ),
            )
            leased_row = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            assert leased_row is not None
            return QuotaAcquireResult(
                lease=QuotaLease(
                    lease_id=lease_id,
                    lease_owner=owner,
                    lease_until=lease_until,
                    ticket=_ticket_from_row(leased_row),
                )
            )

    def wait_for_ticket(
        self,
        ticket_id: str,
        *,
        lease_owner: str,
        lease_seconds: float,
        stop_at: float | None = None,
    ) -> QuotaLease | None:
        """等待直到恰好由 ``ticket_id`` 获得其 FIFO lease。

        ``stop_at`` 限制调度器等待，但不改变 ticket 自身 deadline。FIFO 与 max-in-flight
        等待会轮询，以便及时发现其他调用方的结算。配额时间窗口则直接休眠到下一个确定边界。
        """

        stop = None if stop_at is None else _finite_time(stop_at, "stop_at")
        while True:
            heartbeat = self.heartbeat_ticket(
                ticket_id,
                queue_owner=lease_owner,
            )
            result = self.try_acquire_ticket(
                ticket_id,
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
            )
            if result.lease is not None:
                return result.lease
            if result.blocked_by == (QuotaWaitReason.TICKET_TERMINAL,):
                return None
            now = self._now()
            if stop is not None and now >= stop:
                return None
            retry_at = result.retry_at or (now + self._in_flight_poll_seconds)
            delay = max(0.0, retry_at - now)
            delay = min(delay, heartbeat.queue_heartbeat_ttl / 3.0)
            if (
                QuotaWaitReason.MAX_IN_FLIGHT in result.blocked_by
                or QuotaWaitReason.FIFO_PREDECESSOR in result.blocked_by
            ):
                delay = min(delay, self._in_flight_poll_seconds)
            if stop is not None:
                delay = min(delay, max(0.0, stop - now))
            if delay <= 0:
                delay = min(
                    self._in_flight_poll_seconds,
                    max(0.0, (stop - now) if stop is not None else 0.001),
                )
            self._sleep(delay)

    def mark_dispatched(
        self,
        lease: QuotaLease,
        *,
        lease_seconds: float | None = None,
    ) -> QuotaLease:
        """在 Provider 调用前立即提交已准入 lease。

        调用方必须先持久化其物理尝试/``STARTED`` authority。该转换提交后，lease 过期会被
        视为不确定 Provider 结果，且绝不能自动重新 lease。
        """

        duration = (
            None
            if lease_seconds is None
            else _finite_time(lease_seconds, "lease_seconds")
        )
        if duration is not None and duration <= 0:
            raise ValueError("lease_seconds must be positive")
        now = self._now()
        with self._transaction() as conn:
            self._recover_expired_leases(
                conn,
                scope_hash=lease.ticket.scope_hash,
                now=now,
            )
            ticket_row = self._require_live_lease(conn, lease)
            deadline = float(ticket_row["deadline"])
            if deadline <= now:
                raise QuotaDispatchBlocked(
                    blocked_by=(QuotaWaitReason.TICKET_TERMINAL,),
                    retry_at=None,
                )
            dispatch = conn.execute(
                "SELECT phase FROM model_api_quota_dispatches WHERE lease_id = ?",
                (lease.lease_id,),
            ).fetchone()
            if dispatch is None or dispatch["phase"] != "reserved":
                raise QuotaLeasePhaseError(
                    "lease is not in the pre-dispatch reserved phase"
                )
            scope = self._require_scope(conn, lease.ticket.scope_hash)
            limits = _limits_from_row(scope)
            reservation = int(ticket_row["token_reservation"])
            impossible: list[QuotaWaitReason] = []
            if (
                limits.tokens_per_minute is not None
                and reservation > limits.tokens_per_minute
            ):
                impossible.append(QuotaWaitReason.TOKENS_PER_MINUTE)
            if (
                limits.tokens_per_week is not None
                and reservation > limits.tokens_per_week
            ):
                impossible.append(QuotaWaitReason.TOKENS_PER_WEEK)
            if impossible:
                raise QuotaDispatchBlocked(
                    blocked_by=tuple(impossible),
                    retry_at=None,
                )
            constraints = self._quota_constraints(
                conn,
                scope=scope,
                reservation=reservation,
                now=now,
                excluded_lease_id=lease.lease_id,
            )
            if constraints:
                raise QuotaDispatchBlocked(
                    blocked_by=tuple(reason for reason, _ in constraints),
                    retry_at=max(retry_at for _, retry_at in constraints),
                )
            conn.execute(
                "UPDATE model_api_quota_dispatches "
                "SET phase = 'dispatched', started_at = ? WHERE lease_id = ?",
                (now, lease.lease_id),
            )
            lease_until = float(ticket_row["lease_until"])
            if duration is not None:
                lease_until = max(lease_until, now + duration)
                conn.execute(
                    "UPDATE model_api_quota_tickets SET lease_until = ? "
                    "WHERE ticket_id = ? AND lease_id = ?",
                    (lease_until, lease.ticket.ticket_id, lease.lease_id),
                )
            stored = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (lease.ticket.ticket_id,),
            ).fetchone()
            assert stored is not None
            return QuotaLease(
                lease_id=lease.lease_id,
                lease_owner=lease.lease_owner,
                lease_until=lease_until,
                ticket=_ticket_from_row(stored),
            )

    def renew_lease(
        self,
        lease: QuotaLease,
        *,
        lease_seconds: float,
    ) -> QuotaLease:
        """延长由精确 lease token 持有且尚未过期的 lease。"""

        duration = _finite_time(lease_seconds, "lease_seconds")
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        now = self._now()
        with self._transaction() as conn:
            self._recover_expired_leases(
                conn,
                scope_hash=lease.ticket.scope_hash,
                now=now,
            )
            row = self._require_live_lease(conn, lease)
            lease_until = max(float(row["lease_until"]), now + duration)
            conn.execute(
                "UPDATE model_api_quota_tickets SET lease_until = ? "
                "WHERE ticket_id = ? AND lease_id = ?",
                (lease_until, lease.ticket.ticket_id, lease.lease_id),
            )
            updated = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (lease.ticket.ticket_id,),
            ).fetchone()
            assert updated is not None
            return QuotaLease(
                lease_id=lease.lease_id,
                lease_owner=lease.lease_owner,
                lease_until=lease_until,
                ticket=_ticket_from_row(updated),
            )

    def abandon_before_dispatch(
        self,
        lease: QuotaLease,
        *,
        disposition: Literal["requeue", "cancel"],
    ) -> QuotaTicket:
        """可以确定尚未调用 Provider 时撤销准入。

        这会关闭配额准入与实际 HTTP 调用之间的 gap。deadline/状态复检失败可以取消 ticket；
        Runtime ``STARTED`` 事件持久化失败可将其重新排到队尾。临时分派记录会被原子删除，
        因此从未到达 Provider 的工作既不计入 RPM，也不计入 token 用量。
        """

        if disposition not in {"requeue", "cancel"}:
            raise ValueError("disposition must be 'requeue' or 'cancel'")
        now = self._now()
        with self._transaction() as conn:
            self._recover_expired_leases(
                conn,
                scope_hash=lease.ticket.scope_hash,
                now=now,
            )
            row = self._require_live_lease(conn, lease)
            dispatch = conn.execute(
                "SELECT phase FROM model_api_quota_dispatches WHERE lease_id = ?",
                (lease.lease_id,),
            ).fetchone()
            if dispatch is None or dispatch["phase"] != "reserved":
                raise QuotaLeasePhaseError(
                    "only a pre-dispatch reservation may be abandoned"
                )
            conn.execute(
                "DELETE FROM model_api_quota_dispatches WHERE lease_id = ?",
                (lease.lease_id,),
            )
            if disposition == "cancel":
                status = "cancelled"
                queue_sequence = int(row["queue_sequence"])
            elif float(row["deadline"]) <= now:
                status = "expired"
                queue_sequence = int(row["queue_sequence"])
            else:
                status = "queued"
                queue_sequence = self._next_queue_sequence(conn)
            heartbeat_until = min(
                float(row["deadline"]),
                now + float(row["queue_heartbeat_ttl"]),
            )
            conn.execute(
                "UPDATE model_api_quota_tickets "
                "SET status = ?, queue_sequence = ?, lease_id = NULL, "
                "lease_owner = NULL, lease_until = NULL, "
                "queue_heartbeat_until = ? "
                "WHERE ticket_id = ?",
                (
                    status,
                    queue_sequence,
                    heartbeat_until,
                    lease.ticket.ticket_id,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (lease.ticket.ticket_id,),
            ).fetchone()
            assert stored is not None
            return _ticket_from_row(stored)

    def settle(
        self,
        lease: QuotaLease,
        *,
        outcome: Literal["succeeded", "failed"],
        actual_tokens: int | None,
    ) -> QuotaTicket:
        """释放 lease，并根据实际用量校正其预留。

        ``actual_tokens=None`` 会保留保守预留，适用于网络失败导致供应商用量未知的情况。
        重试被刻意分开：调用方结算当前 ticket，并在所需 backoff 后排入下一物理序号。
        """

        if outcome not in {"succeeded", "failed"}:
            raise ValueError("outcome must be 'succeeded' or 'failed'")
        if actual_tokens is not None and (
            not isinstance(actual_tokens, int)
            or isinstance(actual_tokens, bool)
            or actual_tokens < 0
        ):
            raise ValueError("actual_tokens must be a non-negative integer or None")
        now = self._now()
        with self._transaction() as conn:
            self._recover_expired_leases(
                conn,
                scope_hash=lease.ticket.scope_hash,
                now=now,
            )
            self._require_live_lease(conn, lease)
            dispatch = conn.execute(
                "SELECT phase FROM model_api_quota_dispatches WHERE lease_id = ?",
                (lease.lease_id,),
            ).fetchone()
            if dispatch is None or dispatch["phase"] != "dispatched":
                raise QuotaLeasePhaseError(
                    "lease must be marked dispatched before settlement"
                )
            if actual_tokens is not None:
                conn.execute(
                    "UPDATE model_api_quota_dispatches "
                    "SET charged_tokens = ?, actual_tokens = ?, settled_at = ? "
                    "WHERE lease_id = ?",
                    (actual_tokens, actual_tokens, now, lease.lease_id),
                )
            else:
                conn.execute(
                    "UPDATE model_api_quota_dispatches SET settled_at = ? "
                    "WHERE lease_id = ?",
                    (now, lease.lease_id),
                )
            conn.execute(
                "UPDATE model_api_quota_tickets "
                "SET status = 'settled', settled_outcome = ?, settled_at = ? "
                "WHERE ticket_id = ? AND lease_id = ?",
                (outcome, now, lease.ticket.ticket_id, lease.lease_id),
            )
            stored = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (lease.ticket.ticket_id,),
            ).fetchone()
            assert stored is not None
            return _ticket_from_row(stored)

    def reconcile_uncertain_dispatch(
        self,
        ticket_id: str,
        *,
        queue_owner: str,
        outcome: Literal["succeeded", "failed"],
        actual_tokens: int | None,
    ) -> QuotaTicket:
        """结算在正常结算前过期的分派后 lease。

        Runtime 仍负责根据持久 attempt ledger 判断 Provider 结果。本方法只关闭对应配额记录；
        用量未知时保留保守 token 预留。
        """

        ticket_id = _require_nonempty(ticket_id, "ticket_id")
        owner = _require_nonempty(queue_owner, "queue_owner")
        if outcome not in {"succeeded", "failed"}:
            raise ValueError("outcome must be 'succeeded' or 'failed'")
        if actual_tokens is not None and (
            not isinstance(actual_tokens, int)
            or isinstance(actual_tokens, bool)
            or actual_tokens < 0
        ):
            raise ValueError("actual_tokens must be a non-negative integer or None")
        now = self._now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "reconciliation_required"
                or row["queue_owner"] != owner
            ):
                raise QuotaTicketNotAcquirable(
                    "ticket has no caller-owned uncertain dispatch to reconcile"
                )
            dispatch = conn.execute(
                "SELECT * FROM model_api_quota_dispatches "
                "WHERE ticket_id = ? AND phase = 'dispatched' "
                "ORDER BY dispatch_sequence DESC LIMIT 1",
                (ticket_id,),
            ).fetchone()
            if dispatch is None or dispatch["settled_at"] is not None:
                raise QuotaLeasePhaseError(
                    "uncertain ticket has no unsettled dispatched quota record"
                )
            if actual_tokens is None:
                conn.execute(
                    "UPDATE model_api_quota_dispatches SET settled_at = ? "
                    "WHERE dispatch_sequence = ?",
                    (now, dispatch["dispatch_sequence"]),
                )
            else:
                conn.execute(
                    "UPDATE model_api_quota_dispatches "
                    "SET charged_tokens = ?, actual_tokens = ?, settled_at = ? "
                    "WHERE dispatch_sequence = ?",
                    (
                        actual_tokens,
                        actual_tokens,
                        now,
                        dispatch["dispatch_sequence"],
                    ),
                )
            conn.execute(
                "UPDATE model_api_quota_tickets "
                "SET status = 'settled', settled_outcome = ?, settled_at = ? "
                "WHERE ticket_id = ?",
                (outcome, now, ticket_id),
            )
            stored = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            assert stored is not None
            return _ticket_from_row(stored)

    def apply_rate_limit_cooldown(
        self,
        scope_hash: str,
        *,
        retry_after_seconds: float,
    ) -> float:
        """对整个 scope 应用 429 cooldown，并保留最长时限。"""

        _require_scope_hash(scope_hash)
        delay = _finite_time(retry_after_seconds, "retry_after_seconds")
        if delay < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        cooldown_until = self._now() + delay
        with self._transaction() as conn:
            scope = self._require_scope(conn, scope_hash)
            effective = max(float(scope["cooldown_until"]), cooldown_until)
            conn.execute(
                "UPDATE model_api_quota_scopes SET cooldown_until = ? "
                "WHERE scope_hash = ?",
                (effective, scope_hash),
            )
            return effective

    def cancel(self, ticket_id: str) -> bool:
        """取消已排队工作；已 lease 或已终止 ticket 保持不动。"""

        ticket_id = _require_nonempty(ticket_id, "ticket_id")
        with self._transaction() as conn:
            changed = conn.execute(
                "UPDATE model_api_quota_tickets SET status = 'cancelled' "
                "WHERE ticket_id = ? AND status = 'queued'",
                (ticket_id,),
            ).rowcount
            return changed == 1

    def get_ticket(self, ticket_id: str) -> QuotaTicket | None:
        ticket_id = _require_nonempty(ticket_id, "ticket_id")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            return None if row is None else _ticket_from_row(row)
        finally:
            conn.close()

    def usage_snapshot(self, scope_hash: str) -> QuotaUsageSnapshot:
        """回收所有已过期 in-flight lease 后读取用量。"""

        _require_scope_hash(scope_hash)
        now = self._now()
        with self._transaction() as conn:
            scope = self._require_scope(conn, scope_hash)
            self._recover_expired_leases(conn, scope_hash=scope_hash, now=now)
            requests = self._requests_in_window(
                conn, scope_hash=scope_hash, now=now, window=MINUTE_SECONDS
            )
            minute_tokens = self._tokens_in_window(
                conn, scope_hash=scope_hash, now=now, window=MINUTE_SECONDS
            )
            week_tokens = self._tokens_in_window(
                conn, scope_hash=scope_hash, now=now, window=WEEK_SECONDS
            )
            in_flight = int(
                conn.execute(
                    "SELECT COUNT(*) FROM model_api_quota_tickets "
                    "WHERE scope_hash = ? AND status = 'leased' AND lease_until > ?",
                    (scope_hash, now),
                ).fetchone()[0]
            )
            admitted_not_dispatched = int(
                conn.execute(
                    "SELECT COUNT(*) FROM model_api_quota_dispatches AS d "
                    "JOIN model_api_quota_tickets AS t ON t.lease_id = d.lease_id "
                    "WHERE d.scope_hash = ? AND d.phase = 'reserved' "
                    "AND t.status = 'leased' AND t.lease_until > ?",
                    (scope_hash, now),
                ).fetchone()[0]
            )
            return QuotaUsageSnapshot(
                requests_last_minute=requests,
                tokens_last_minute=minute_tokens,
                tokens_last_week=week_tokens,
                in_flight=in_flight,
                admitted_not_dispatched=admitted_not_dispatched,
                cooldown_until=float(scope["cooldown_until"]),
            )

    def _quota_constraints(
        self,
        conn: sqlite3.Connection,
        *,
        scope: sqlite3.Row,
        reservation: int,
        now: float,
        excluded_lease_id: str | None = None,
    ) -> list[tuple[QuotaWaitReason, float]]:
        scope_hash = str(scope["scope_hash"])
        limits = _limits_from_row(scope)
        blocked: list[tuple[QuotaWaitReason, float]] = []
        cooldown_until = float(scope["cooldown_until"])
        if cooldown_until > now:
            blocked.append((QuotaWaitReason.SCOPE_COOLDOWN, cooldown_until))

        if limits.max_in_flight is not None:
            active = conn.execute(
                "SELECT lease_until FROM model_api_quota_tickets "
                "WHERE scope_hash = ? AND status = 'leased' AND lease_until > ? "
                "AND (? IS NULL OR lease_id != ?) "
                "ORDER BY lease_until ASC",
                (scope_hash, now, excluded_lease_id, excluded_lease_id),
            ).fetchall()
            if len(active) >= limits.max_in_flight:
                release_index = len(active) - limits.max_in_flight
                blocked.append(
                    (
                        QuotaWaitReason.MAX_IN_FLIGHT,
                        float(active[release_index]["lease_until"]),
                    )
                )

        if limits.requests_per_minute is not None:
            entries = self._quota_capacity_entries(
                conn,
                scope_hash=scope_hash,
                now=now,
                window=MINUTE_SECONDS,
                excluded_lease_id=excluded_lease_id,
            )
            if len(entries) >= limits.requests_per_minute:
                release_index = len(entries) - limits.requests_per_minute
                blocked.append(
                    (
                        QuotaWaitReason.REQUESTS_PER_MINUTE,
                        entries[release_index][0],
                    )
                )

        if limits.tokens_per_minute is not None:
            entries = self._quota_capacity_entries(
                conn,
                scope_hash=scope_hash,
                now=now,
                window=MINUTE_SECONDS,
                excluded_lease_id=excluded_lease_id,
            )
            release = _capacity_token_release_time(
                entries,
                reservation=reservation,
                limit=limits.tokens_per_minute,
            )
            if release is not None:
                blocked.append((QuotaWaitReason.TOKENS_PER_MINUTE, release))

        if limits.tokens_per_week is not None:
            entries = self._quota_capacity_entries(
                conn,
                scope_hash=scope_hash,
                now=now,
                window=WEEK_SECONDS,
                excluded_lease_id=excluded_lease_id,
            )
            release = _capacity_token_release_time(
                entries,
                reservation=reservation,
                limit=limits.tokens_per_week,
            )
            if release is not None:
                blocked.append((QuotaWaitReason.TOKENS_PER_WEEK, release))
        return blocked

    def _recover_expired_leases(
        self,
        conn: sqlite3.Connection,
        *,
        scope_hash: str,
        now: float,
    ) -> int:
        rows = conn.execute(
            "SELECT t.ticket_id, t.deadline, t.lease_id, d.phase "
            "FROM model_api_quota_tickets AS t "
            "JOIN model_api_quota_dispatches AS d ON d.lease_id = t.lease_id "
            "WHERE t.scope_hash = ? AND t.status = 'leased' AND t.lease_until <= ? "
            "ORDER BY t.queue_sequence ASC",
            (scope_hash, now),
        ).fetchall()
        for row in rows:
            if row["phase"] == "dispatched":
                conn.execute(
                    "UPDATE model_api_quota_tickets "
                    "SET status = 'reconciliation_required', lease_id = NULL, "
                    "lease_owner = NULL, lease_until = NULL WHERE ticket_id = ?",
                    (row["ticket_id"],),
                )
            elif float(row["deadline"]) <= now:
                conn.execute(
                    "DELETE FROM model_api_quota_dispatches WHERE lease_id = ?",
                    (row["lease_id"],),
                )
                conn.execute(
                    "UPDATE model_api_quota_tickets "
                    "SET status = 'expired', lease_id = NULL, lease_owner = NULL, "
                    "lease_until = NULL WHERE ticket_id = ?",
                    (row["ticket_id"],),
                )
            else:
                conn.execute(
                    "DELETE FROM model_api_quota_dispatches WHERE lease_id = ?",
                    (row["lease_id"],),
                )
                conn.execute(
                    "UPDATE model_api_quota_tickets "
                    "SET status = 'queued', queue_sequence = ?, "
                    "lease_id = NULL, lease_owner = NULL, lease_until = NULL, "
                    "queue_heartbeat_until = MIN(deadline, ? + queue_heartbeat_ttl) "
                    "WHERE ticket_id = ?",
                    (self._next_queue_sequence(conn), now, row["ticket_id"]),
                )
        return len(rows)

    def _require_live_lease(
        self,
        conn: sqlite3.Connection,
        lease: QuotaLease,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM model_api_quota_tickets WHERE ticket_id = ?",
            (lease.ticket.ticket_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "leased"
            or row["lease_id"] != lease.lease_id
            or row["lease_owner"] != lease.lease_owner
        ):
            raise QuotaLeaseStale("lease no longer owns this ticket")
        return row

    def _require_scope(
        self,
        conn: sqlite3.Connection,
        scope_hash: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM model_api_quota_scopes WHERE scope_hash = ?",
            (scope_hash,),
        ).fetchone()
        if row is None:
            raise QuotaScopeNotRegistered("quota scope is not registered")
        return row

    @staticmethod
    def _next_queue_sequence(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(queue_sequence), 0) + 1 "
            "FROM model_api_quota_tickets"
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _quota_capacity_entries(
        conn: sqlite3.Connection,
        *,
        scope_hash: str,
        now: float,
        window: float,
        excluded_lease_id: str | None = None,
    ) -> list[tuple[float, int]]:
        """返回已用/已预留配额的（保证释放时间，token 数）。"""

        dispatched = conn.execute(
            "SELECT started_at, charged_tokens "
            "FROM model_api_quota_dispatches "
            "WHERE scope_hash = ? AND phase = 'dispatched' AND started_at > ? "
            "AND (? IS NULL OR lease_id != ?)",
            (
                scope_hash,
                now - window,
                excluded_lease_id,
                excluded_lease_id,
            ),
        ).fetchall()
        reserved = conn.execute(
            "SELECT t.lease_until, d.charged_tokens "
            "FROM model_api_quota_dispatches AS d "
            "JOIN model_api_quota_tickets AS t ON t.lease_id = d.lease_id "
            "WHERE d.scope_hash = ? AND d.phase = 'reserved' "
            "AND t.status = 'leased' AND t.lease_until > ? "
            "AND (? IS NULL OR d.lease_id != ?)",
            (scope_hash, now, excluded_lease_id, excluded_lease_id),
        ).fetchall()
        entries = [
            (float(row["started_at"]) + window, int(row["charged_tokens"]))
            for row in dispatched
        ]
        entries.extend(
            (float(row["lease_until"]), int(row["charged_tokens"]))
            for row in reserved
        )
        entries.sort(key=lambda item: item[0])
        return entries

    @staticmethod
    def _dispatch_usage_rows(
        conn: sqlite3.Connection,
        *,
        scope_hash: str,
        now: float,
        window: float,
    ) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT started_at, charged_tokens "
            "FROM model_api_quota_dispatches "
            "WHERE scope_hash = ? AND phase = 'dispatched' AND started_at > ? "
            "ORDER BY started_at ASC, dispatch_sequence ASC",
            (scope_hash, now - window),
        ).fetchall()

    def _requests_in_window(
        self,
        conn: sqlite3.Connection,
        *,
        scope_hash: str,
        now: float,
        window: float,
    ) -> int:
        return len(
            self._dispatch_usage_rows(
                conn, scope_hash=scope_hash, now=now, window=window
            )
        )

    def _tokens_in_window(
        self,
        conn: sqlite3.Connection,
        *,
        scope_hash: str,
        now: float,
        window: float,
    ) -> int:
        return sum(
            int(row["charged_tokens"])
            for row in self._dispatch_usage_rows(
                conn, scope_hash=scope_hash, now=now, window=window
            )
        )

    def _now(self) -> float:
        return _finite_time(self._clock(), "clock result")

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_api_quota_schema (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_api_quota_scopes (
                    scope_hash TEXT PRIMARY KEY,
                    requests_per_minute INTEGER,
                    tokens_per_minute INTEGER,
                    tokens_per_week INTEGER,
                    max_in_flight INTEGER,
                    cooldown_until REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_api_quota_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    scope_hash TEXT NOT NULL REFERENCES model_api_quota_scopes(scope_hash),
                    logical_call_id TEXT NOT NULL,
                    physical_ordinal INTEGER,
                    eligible_at REAL NOT NULL,
                    deadline REAL NOT NULL,
                    token_reservation INTEGER NOT NULL,
                    queue_sequence INTEGER NOT NULL UNIQUE,
                    queue_owner TEXT NOT NULL,
                    queue_heartbeat_until REAL NOT NULL,
                    queue_heartbeat_ttl REAL NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'queued', 'leased', 'settled', 'cancelled', 'expired',
                            'reconciliation_required'
                        )
                    ),
                    lease_id TEXT UNIQUE,
                    lease_owner TEXT,
                    lease_until REAL,
                    settled_outcome TEXT CHECK (
                        settled_outcome IS NULL OR settled_outcome IN ('succeeded', 'failed')
                    ),
                    created_at REAL NOT NULL,
                    settled_at REAL
                );
                CREATE TABLE IF NOT EXISTS model_api_quota_dispatches (
                    dispatch_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_hash TEXT NOT NULL REFERENCES model_api_quota_scopes(scope_hash),
                    ticket_id TEXT NOT NULL REFERENCES model_api_quota_tickets(ticket_id),
                    lease_id TEXT NOT NULL UNIQUE,
                    phase TEXT NOT NULL CHECK (phase IN ('reserved', 'dispatched')),
                    reserved_at REAL NOT NULL,
                    started_at REAL,
                    charged_tokens INTEGER NOT NULL,
                    actual_tokens INTEGER,
                    settled_at REAL
                );
                CREATE INDEX IF NOT EXISTS model_api_quota_ready_idx
                    ON model_api_quota_tickets (
                        scope_hash, status, eligible_at, queue_sequence
                    );
                CREATE INDEX IF NOT EXISTS model_api_quota_lease_idx
                    ON model_api_quota_tickets (scope_hash, status, lease_until);
                CREATE INDEX IF NOT EXISTS model_api_quota_usage_idx
                    ON model_api_quota_dispatches (
                        scope_hash, started_at, dispatch_sequence
                    );
                """
            )
            row = conn.execute(
                "SELECT version FROM model_api_quota_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT OR IGNORE INTO model_api_quota_schema "
                    "(singleton, version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
                row = conn.execute(
                    "SELECT version FROM model_api_quota_schema WHERE singleton = 1"
                ).fetchone()
            if row is None or int(row["version"]) != _SCHEMA_VERSION:
                raise QuotaQueueError("unsupported model API quota schema version")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.database_path,
            timeout=self._sqlite_timeout_seconds,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        busy_timeout_ms = int(self._sqlite_timeout_seconds * 1000)
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except QuotaLeaseStale:
            # lease 操作会在验证 token 前回收过期项。即使过期调用方必须失败，也要保留该恢复
            # 转换；否则每个迟到调用方都会把 ticket 回滚到恢复前的 leased 状态。
            conn.commit()
            raise
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def _capacity_token_release_time(
    entries: list[tuple[float, int]],
    *,
    reservation: int,
    limit: int,
) -> float | None:
    current = sum(tokens for _, tokens in entries)
    excess = current + reservation - limit
    if excess <= 0:
        return None
    released = 0
    for release_at, tokens in entries:
        released += tokens
        if released >= excess:
            return release_at
    raise AssertionError("reservation fit was checked before token release calculation")


def _ticket_from_row(row: sqlite3.Row) -> QuotaTicket:
    return QuotaTicket(
        ticket_id=str(row["ticket_id"]),
        scope_hash=str(row["scope_hash"]),
        logical_call_id=str(row["logical_call_id"]),
        physical_ordinal=(
            None if row["physical_ordinal"] is None else int(row["physical_ordinal"])
        ),
        eligible_at=float(row["eligible_at"]),
        deadline=float(row["deadline"]),
        token_reservation=int(row["token_reservation"]),
        queue_sequence=int(row["queue_sequence"]),
        queue_owner=str(row["queue_owner"]),
        queue_heartbeat_until=float(row["queue_heartbeat_until"]),
        queue_heartbeat_ttl=float(row["queue_heartbeat_ttl"]),
        status=str(row["status"]),  # type: ignore[arg-type]
    )


def _limits_from_row(row: sqlite3.Row) -> QuotaLimits:
    return QuotaLimits(
        requests_per_minute=(
            None
            if row["requests_per_minute"] is None
            else int(row["requests_per_minute"])
        ),
        tokens_per_minute=(
            None if row["tokens_per_minute"] is None else int(row["tokens_per_minute"])
        ),
        tokens_per_week=(
            None if row["tokens_per_week"] is None else int(row["tokens_per_week"])
        ),
        max_in_flight=(
            None if row["max_in_flight"] is None else int(row["max_in_flight"])
        ),
    )


def _normalize_base_url(base_url: str) -> str:
    raw = _require_nonempty(base_url, "base_url").rstrip("/")
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("base_url must be an absolute URL")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def _require_scope_hash(scope_hash: str) -> str:
    if not isinstance(scope_hash, str) or _SCOPE_HASH_RE.fullmatch(scope_hash) is None:
        raise ValueError("scope_hash must be a lowercase SHA-256 hex digest")
    return scope_hash


def _require_nonempty(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_time(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "MINUTE_SECONDS",
    "WEEK_SECONDS",
    "ModelApiQuotaQueue",
    "QuotaAcquireResult",
    "QuotaDispatchBlocked",
    "QuotaLease",
    "QuotaLeasePhaseError",
    "QuotaLeaseStale",
    "QuotaLimits",
    "QuotaQueueError",
    "QuotaReservationImpossible",
    "QuotaScopeConfigurationConflict",
    "QuotaScopeNotRegistered",
    "QuotaTicket",
    "QuotaTicketConflict",
    "QuotaTicketNotAcquirable",
    "QuotaUsageSnapshot",
    "QuotaWaitReason",
    "derive_quota_scope_hash",
]
