"""共享模型 API 配额队列的已准备请求集成。

本模块是模型 Profile 与 SQLite 队列之间的窄桥梁。准备完成后，它刻意不保留端点 URL 或
凭据：已准备请求只携带不透明 scope hash、数值限制及其保守 token 预留。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal
from uuid import uuid4

from ..configuration.paths import STATE_DIR, resolve_absolute_environment_path
from .endpoint_profiles import ModelProfileQuota, normalize_model_profile_quota
from .api_quota_queue import (
    ModelApiQuotaQueue,
    QuotaDispatchBlocked,
    QuotaLease,
    QuotaLimits,
    QuotaQueueError,
    QuotaTicketNotAcquirable,
    QuotaWaitReason,
    derive_quota_scope_hash,
)


# 产品当前整 Turn 上限为 20 分 48 秒。不持有 TurnDeadline 的直接网关调用方仍需要有界队列
# 等待，而不能无限同步挂起。
DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS = 20 * 60 + 48
DEFAULT_API_QUOTA_PREDISPATCH_LEASE_SECONDS = 30.0
DEFAULT_API_QUOTA_LEASE_GRACE_SECONDS = 30.0
API_QUOTA_DATABASE_PATH = STATE_DIR / "model_api_quota.sqlite3"
API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE = (
    "PERSONAGRAPH_MODEL_API_QUOTA_DATABASE_PATH"
)
MODEL_API_QUOTA_ENVIRONMENT_VARIABLE = "PERSONAGRAPH_MODEL_API_QUOTA"
_MODEL_API_QUOTA_ENVIRONMENT_SCHEMA_VERSION = 1


class ApiQuotaAdmissionError(RuntimeError):
    """供应商请求无法安全进入或结算共享队列。"""


class ApiQuotaWaitTimeout(ApiQuotaAdmissionError):
    """请求在 Provider 分派前到达队列等待 deadline。"""


class ApiQuotaDispatchDeferred(ApiQuotaAdmissionError):
    """策略变化阻止了已知尚未分派的请求。"""

    def __init__(
        self,
        *,
        blocked_by: tuple[QuotaWaitReason, ...],
        retry_at: float | None,
    ) -> None:
        super().__init__("model API quota policy changed before dispatch")
        self.blocked_by = blocked_by
        self.retry_at = retry_at


def model_profile_quota_environment_value(quota: ModelProfileQuota) -> str:
    """序列化一个不含端点和凭据的严格配额环境快照。"""

    if not isinstance(quota, ModelProfileQuota):
        raise TypeError("quota must be ModelProfileQuota")
    return json.dumps(
        {
            "schema_version": _MODEL_API_QUOTA_ENVIRONMENT_SCHEMA_VERSION,
            **quota.to_dict(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def model_profile_quota_from_environment(
    environment: Mapping[str, str] | None = None,
) -> ModelProfileQuota | None:
    """读取隔离进程的非密钥配额快照；未配置时返回 ``None``。"""

    source = os.environ if environment is None else environment
    raw = str(source.get(MODEL_API_QUOTA_ENVIRONMENT_VARIABLE) or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("model API quota environment is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("model API quota environment must be an object")
    expected = {
        "schema_version",
        "requests_per_minute",
        "tokens_per_minute",
        "tokens_per_week",
        "max_in_flight",
        "quota_group",
    }
    if set(payload) != expected:
        raise ValueError("model API quota environment has the wrong fields")
    if payload.pop("schema_version") != _MODEL_API_QUOTA_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("model API quota environment has an unsupported schema")
    return normalize_model_profile_quota(payload)


def resolve_api_quota_database_path(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """解析跨进程共享队列路径，显式覆盖必须是仓库外绝对路径。"""

    source = os.environ if environment is None else environment
    return resolve_absolute_environment_path(
        API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE,
        API_QUOTA_DATABASE_PATH,
        environment=source,
    )


@dataclass(frozen=True, slots=True)
class PreparedApiQuotaRequest:
    """冻结在精确已准入线上字节旁、不含内容的配额 authority。"""

    scope_hash: str
    limits: QuotaLimits
    token_reservation: int
    provider_timeout_seconds: float
    database_path: str
    request_identity: str = field(repr=False)

    def acquire(
        self,
        *,
        logical_call_id: str,
        wait_timeout_seconds: float,
    ) -> "ApiQuotaPermit":
        wait = _positive_finite(wait_timeout_seconds, "wait_timeout_seconds")
        provider_timeout = _positive_finite(
            self.provider_timeout_seconds,
            "provider_timeout_seconds",
        )
        queue = _default_queue(self.database_path)
        now = time.time()
        stop_at = now + wait
        # 准入本身只获得短暂的分派前 lease。原子 mark-dispatched 转换会把它延长到足以覆盖
        # 最长剩余 Provider 调用，同时避免发送前崩溃占用整个 Turn 的 in-flight 槽位。
        dispatch_lease_seconds = (
            wait + provider_timeout + DEFAULT_API_QUOTA_LEASE_GRACE_SECONDS
        )
        ticket_id = f"quota_{uuid4().hex}"
        queued_ticket_id = ticket_id
        owner = f"pid-{os.getpid()}:thread-{threading.get_ident()}:{uuid4().hex}"
        try:
        # Profile 更新是显式可变策略变化。历史分派记录仍留在同一 scope，并按新配置的滚动
        # 限制评估。
            queue.configure_scope(self.scope_hash, self.limits)
            ticket = queue.enqueue(
                scope_hash=self.scope_hash,
                logical_call_id=logical_call_id,
                physical_ordinal=None,
                token_reservation=self.token_reservation,
                queue_owner=owner,
                deadline=stop_at,
                ticket_id=ticket_id,
            )
            queued_ticket_id = ticket.ticket_id
            lease = queue.wait_for_ticket(
                ticket.ticket_id,
                lease_owner=owner,
                lease_seconds=DEFAULT_API_QUOTA_PREDISPATCH_LEASE_SECONDS,
                stop_at=stop_at,
            )
        except QuotaTicketNotAcquirable as exc:
            _cancel_unsent_ticket(queue, queued_ticket_id)
            raise ApiQuotaWaitTimeout(
                "model API quota ticket could not run before its deadline"
            ) from exc
        except QuotaQueueError as exc:
            _cancel_unsent_ticket(queue, queued_ticket_id)
            raise ApiQuotaAdmissionError(
                "model API quota admission failed"
            ) from exc
        except BaseException:
            # 尽力取消只在授予 lease 前安全；队列方法本身会拒绝取消已 lease/已分派工作。
            _cancel_unsent_ticket(queue, queued_ticket_id)
            raise
        if lease is None:
            queue.cancel(ticket.ticket_id)
            raise ApiQuotaWaitTimeout(
                "model API quota wait reached its deadline"
            )
        return ApiQuotaPermit(
            queue=queue,
            lease=lease,
            dispatch_lease_seconds=dispatch_lease_seconds,
            request_identity=self.request_identity,
        )


@dataclass(slots=True)
class ApiQuotaPermit:
    """一个调用方所有、恰好覆盖一次 Provider 请求的队列 lease。"""

    queue: ModelApiQuotaQueue = field(repr=False)
    lease: QuotaLease
    dispatch_lease_seconds: float = field(repr=False)
    request_identity: str = field(repr=False)
    _phase: Literal["reserved", "dispatched", "closed"] = field(
        default="reserved",
        init=False,
        repr=False,
    )

    @property
    def scope_hash(self) -> str:
        return self.lease.ticket.scope_hash

    def mark_dispatched(self) -> None:
        if self._phase != "reserved":
            raise ApiQuotaAdmissionError(
                "quota permit is not awaiting Provider dispatch"
            )
        try:
            self.lease = self.queue.mark_dispatched(
                self.lease,
                lease_seconds=self.dispatch_lease_seconds,
            )
        except QuotaDispatchBlocked as exc:
            raise ApiQuotaDispatchDeferred(
                blocked_by=exc.blocked_by,
                retry_at=exc.retry_at,
            ) from exc
        except QuotaQueueError as exc:
            raise ApiQuotaAdmissionError(
                "model API quota dispatch transition failed"
            ) from exc
        self._phase = "dispatched"

    def abandon_before_dispatch(
        self,
        *,
        disposition: Literal["requeue", "cancel"] = "cancel",
    ) -> None:
        if self._phase == "closed":
            return
        if self._phase != "reserved":
            raise ApiQuotaAdmissionError(
                "a dispatched quota permit cannot be abandoned"
            )
        try:
            self.queue.abandon_before_dispatch(
                self.lease,
                disposition=disposition,
            )
        except QuotaQueueError as exc:
            raise ApiQuotaAdmissionError(
                "model API quota reservation could not be abandoned"
            ) from exc
        self._phase = "closed"

    def apply_rate_limit_cooldown(self, retry_after_seconds: float) -> None:
        delay = float(retry_after_seconds)
        if not math.isfinite(delay) or delay < 0:
            delay = 1.0
        try:
            self.queue.apply_rate_limit_cooldown(
                self.scope_hash,
                retry_after_seconds=delay,
            )
        except QuotaQueueError as exc:
            raise ApiQuotaAdmissionError(
                "model API rate-limit cooldown could not be recorded"
            ) from exc

    def settle(
        self,
        *,
        outcome: Literal["succeeded", "failed"],
        actual_tokens: int | None,
    ) -> None:
        if self._phase != "dispatched":
            raise ApiQuotaAdmissionError(
                "quota permit must be dispatched before settlement"
            )
        try:
            self.queue.settle(
                self.lease,
                outcome=outcome,
                actual_tokens=actual_tokens,
            )
        except QuotaQueueError as exc:
            raise ApiQuotaAdmissionError(
                "model API quota settlement requires reconciliation"
            ) from exc
        self._phase = "closed"


def prepare_api_quota_request(
    *,
    base_url: str,
    credential: str,
    quota: ModelProfileQuota,
    estimated_input_tokens: int,
    output_token_limit: int,
    provider_timeout_seconds: float,
    queue_database_path: str | Path | None = None,
) -> PreparedApiQuotaRequest | None:
    """冻结配额请求；无限额 Profile 返回 ``None``。"""

    if not isinstance(quota, ModelProfileQuota):
        raise TypeError("quota must be ModelProfileQuota")
    numeric_limits = (
        quota.requests_per_minute,
        quota.tokens_per_minute,
        quota.tokens_per_week,
        quota.max_in_flight,
    )
    if all(value is None for value in numeric_limits):
        return None
    if (
        isinstance(estimated_input_tokens, bool)
        or not isinstance(estimated_input_tokens, int)
        or estimated_input_tokens <= 0
    ):
        raise ValueError("estimated_input_tokens must be a positive integer")
    if (
        isinstance(output_token_limit, bool)
        or not isinstance(output_token_limit, int)
        or output_token_limit <= 0
    ):
        raise ValueError("output_token_limit must be a positive integer")
    return PreparedApiQuotaRequest(
        scope_hash=derive_quota_scope_hash(
            base_url=base_url,
            credential=credential,
            quota_group=quota.quota_group,
        ),
        limits=QuotaLimits(
            requests_per_minute=quota.requests_per_minute,
            tokens_per_minute=quota.tokens_per_minute,
            tokens_per_week=quota.tokens_per_week,
            max_in_flight=quota.max_in_flight,
        ),
        token_reservation=estimated_input_tokens + output_token_limit,
        provider_timeout_seconds=_positive_finite(
            provider_timeout_seconds,
            "provider_timeout_seconds",
        ),
        database_path=str(
            resolve_api_quota_database_path(
                (
                    {
                        API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE: str(
                            queue_database_path
                        )
                    }
                    if queue_database_path is not None
                    else None
                )
            )
        ),
        request_identity=f"quota_request_{uuid4().hex}",
    )


def actual_model_tokens(result: Any) -> int | None:
    """返回已知可计费用量；信息不完整时保留预留量。"""

    input_tokens = getattr(result, "input_tokens", None)
    output_tokens = getattr(result, "output_tokens", None)
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        return None
    extra = 0
    for field_name in ("cache_read_tokens", "cache_write_tokens"):
        value = getattr(result, field_name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        extra += value
    return input_tokens + output_tokens + extra


@lru_cache(maxsize=8)
def _default_queue(database_path: str) -> ModelApiQuotaQueue:
    return ModelApiQuotaQueue(database_path)


def clear_api_quota_queue_cache() -> None:
    """测试/配置 hook；生产调用方通常不需要它。"""

    _default_queue.cache_clear()


def _cancel_unsent_ticket(queue: ModelApiQuotaQueue, ticket_id: str) -> None:
    """无法取消已 lease 或已分派工作的尽力清理。"""

    try:
        queue.cancel(ticket_id)
    except Exception:
        pass


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


__all__ = [
    "API_QUOTA_DATABASE_PATH",
    "API_QUOTA_DATABASE_PATH_ENVIRONMENT_VARIABLE",
    "DEFAULT_API_QUOTA_PREDISPATCH_LEASE_SECONDS",
    "DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS",
    "ApiQuotaAdmissionError",
    "ApiQuotaDispatchDeferred",
    "ApiQuotaPermit",
    "ApiQuotaWaitTimeout",
    "MODEL_API_QUOTA_ENVIRONMENT_VARIABLE",
    "PreparedApiQuotaRequest",
    "actual_model_tokens",
    "clear_api_quota_queue_cache",
    "model_profile_quota_environment_value",
    "model_profile_quota_from_environment",
    "prepare_api_quota_request",
    "resolve_api_quota_database_path",
]
