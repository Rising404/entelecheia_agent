"""模型网关结果、已准备调用与物理分派约束。"""

from __future__ import annotations

import math
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

from . import endpoint_profiles as model_profiles
from .api_quota_controller import (
    DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
    ApiQuotaAdmissionError,
    ApiQuotaDispatchDeferred,
    ApiQuotaPermit,
    ApiQuotaWaitTimeout,
    PreparedApiQuotaRequest,
    actual_model_tokens,
    prepare_api_quota_request,
)
from .context_budgeting import (
    PreparedAdmittedProviderRequest,
    PreparedProviderEnvelopeMetadata,
)
from .contracts import ModelResult
from .tier_bindings import ModelTierBinding
from .trajectory_projection import (
    _TRAJECTORY_REDACTED_INPUT_SHA256S,
    _TRAJECTORY_REDACT_PROVIDER_REPLY,
)


@dataclass(frozen=True, slots=True)
class PreparedModelCall:
    """一个已通过上下文准入、尚未执行 I/O 的供应商请求。

    准备阶段持有唯一序列化的请求体。因此 Runtime 编排可以在预留物理尝试前执行此阶段，
    并仅在预留成功后把持久调用 id 传入 :meth:`dispatch`。
    """

    _dispatch: Callable[[str | None], ModelResult] = field(repr=False)
    context_budget: PreparedAdmittedProviderRequest | None = field(
        default=None,
        repr=False,
    )
    api_quota: PreparedApiQuotaRequest | None = field(default=None, repr=False)
    trajectory_redacted_input_sha256s: tuple[str, ...] = field(
        default=(),
        repr=False,
    )
    trajectory_redact_reply: bool = field(default=False, repr=False)

    @property
    def budget_metadata(self) -> PreparedProviderEnvelopeMetadata | None:
        return (
            self.context_budget.metadata
            if self.context_budget is not None
            else None
        )

    def dispatch(self, *, model_call_id: str | None = None) -> ModelResult:
        """便捷单次分派入口：取得本请求的 API quota permit，再进入带 permit 的物理交接。

        Runtime 的共享 wrapper 可提前取得 permit，再走 dispatch_with_api_quota，
        以便在真正 Provider I/O 前建立持久 physical attempt；这里不创建 L1 决策轮。
        """

        resolved_call_id = model_call_id or str(uuid.uuid4())
        permit = self.acquire_api_quota(
            model_call_id=resolved_call_id,
            wait_timeout_seconds=DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
        )
        return self.dispatch_with_api_quota(
            model_call_id=resolved_call_id,
            permit=permit,
        )

    def acquire_api_quota(
        self,
        *,
        model_call_id: str,
        wait_timeout_seconds: float,
    ) -> ApiQuotaPermit | None:
        if self.api_quota is None:
            return None
        try:
            return self.api_quota.acquire(
                logical_call_id=model_call_id,
                wait_timeout_seconds=wait_timeout_seconds,
            )
        except ApiQuotaWaitTimeout as exc:
            raise ModelGatewayError(
                "MODEL_QUOTA_WAIT_TIMEOUT",
                "Model API quota wait reached its deadline.",
                retryable=False,
                details={"reason": "quota_wait_timeout"},
            ) from exc
        except ApiQuotaAdmissionError as exc:
            raise ModelGatewayError(
                "MODEL_QUOTA_ADMISSION_FAILED",
                "Model API quota admission failed.",
                retryable=False,
                details={"reason": "quota_admission_failed"},
            ) from exc

    def abandon_api_quota(
        self,
        permit: ApiQuotaPermit | None,
        *,
        disposition: Literal["requeue", "cancel"] = "cancel",
    ) -> None:
        if permit is None:
            return
        try:
            permit.abandon_before_dispatch(disposition=disposition)
        except ApiQuotaAdmissionError as exc:
            raise ModelGatewayError(
                "MODEL_QUOTA_RECONCILIATION_REQUIRED",
                "Model API quota reservation could not be closed safely.",
                retryable=False,
                details={"reason": "quota_reservation_close_failed"},
            ) from exc

    def dispatch_with_api_quota(
        self,
        *,
        model_call_id: str,
        permit: ApiQuotaPermit | None,
    ) -> ModelResult:
        """以匹配此 prepared request 的 quota permit 分派，并结算本次配额使用。

        身份/范围不匹配直接拒绝；有 quota 时先标 dispatched，再执行 Provider，之后
        按真实用量或失败结算。quota 结算与 Runtime 模型账本是不同职责，任一无法确认
        都不能宣称本次完整成功。无 quota 的请求仍经过同一 trajectory 投影 scope。
        """

        if self.api_quota is None:
            if permit is not None:
                raise TypeError("an unlimited request cannot consume a quota permit")
            return self._dispatch_with_trajectory_redaction(model_call_id)
        if permit is None:
            raise TypeError("a quota-managed request requires its own permit")
        if (
            permit.request_identity != self.api_quota.request_identity
            or permit.scope_hash != self.api_quota.scope_hash
        ):
            raise TypeError("quota permit belongs to a different prepared request")
        try:
            permit.mark_dispatched()
        except ApiQuotaDispatchDeferred as exc:
            try:
                permit.abandon_before_dispatch(disposition="cancel")
            except ApiQuotaAdmissionError as abandon_exc:
                raise ModelGatewayError(
                    "MODEL_QUOTA_RECONCILIATION_REQUIRED",
                    "Model API quota reservation could not be closed safely.",
                    retryable=False,
                    details={"reason": "quota_reservation_close_failed"},
                ) from abandon_exc
            retry_after_seconds = (
                None
                if exc.retry_at is None
                else max(0.0, exc.retry_at - time.time())
            )
            raise ModelGatewayError(
                "MODEL_QUOTA_DISPATCH_DEFERRED",
                "Model API quota policy changed before Provider dispatch.",
                retryable=True,
                details={
                    "reason": "quota_policy_changed",
                    "blocked_by": [reason.value for reason in exc.blocked_by],
                    "retry_after_seconds": retry_after_seconds,
                },
            ) from exc
        except ApiQuotaAdmissionError as exc:
            try:
                permit.abandon_before_dispatch(disposition="cancel")
            except ApiQuotaAdmissionError:
                pass
            raise ModelGatewayError(
                "MODEL_QUOTA_RECONCILIATION_REQUIRED",
                "Model API quota dispatch could not be recorded safely.",
                retryable=False,
                details={"reason": "quota_dispatch_transition_failed"},
            ) from exc

        result: ModelResult | None = None
        provider_error: BaseException | None = None
        cooldown_error: BaseException | None = None
        try:
            result = self._dispatch_with_trajectory_redaction(model_call_id)
        except BaseException as exc:
            provider_error = exc
            if _is_provider_rate_limit(exc):
                try:
                    permit.apply_rate_limit_cooldown(
                        _provider_retry_after_seconds(exc),
                    )
                except ApiQuotaAdmissionError as cooldown_exc:
                    cooldown_error = cooldown_exc
        try:
            permit.settle(
                outcome="succeeded" if provider_error is None else "failed",
                actual_tokens=(
                    actual_model_tokens(result) if result is not None else None
                ),
            )
        except ApiQuotaAdmissionError as exc:
            raise ModelGatewayError(
                "MODEL_QUOTA_RECONCILIATION_REQUIRED",
                "Model API quota settlement requires reconciliation.",
                retryable=False,
                details={"reason": "quota_settlement_failed"},
            ) from exc
        if cooldown_error is not None:
            raise ModelGatewayError(
                "MODEL_QUOTA_RECONCILIATION_REQUIRED",
                "Provider rate-limit cooldown could not be recorded safely.",
                retryable=False,
                details={"reason": "quota_cooldown_failed"},
            ) from cooldown_error
        if provider_error is not None:
            raise provider_error
        assert result is not None
        return result

    def _dispatch_with_trajectory_redaction(
        self,
        model_call_id: str | None,
    ) -> ModelResult:
        """仅在本次 Provider dispatch 期间绑定 trajectory 的正文投影规则，finally 还原。

        _dispatch 闭包发送已准入 wire bytes；ContextVar 只影响 recorder 看到的副本，
        不能用于改写已冻结的请求或把另一调用的被拒正文一起隐藏。
        """

        input_token = _TRAJECTORY_REDACTED_INPUT_SHA256S.set(
            frozenset(self.trajectory_redacted_input_sha256s)
        )
        reply_token = _TRAJECTORY_REDACT_PROVIDER_REPLY.set(
            self.trajectory_redact_reply
        )
        try:
            return self._dispatch(model_call_id)
        finally:
            _TRAJECTORY_REDACT_PROVIDER_REPLY.reset(reply_token)
            _TRAJECTORY_REDACTED_INPUT_SHA256S.reset(input_token)


class ModelGatewayError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


def _is_provider_rate_limit(error: BaseException) -> bool:
    return (
        isinstance(error, ModelGatewayError)
        and error.details.get("status_code") == 429
    )


def _provider_retry_after_seconds(error: BaseException) -> float:
    if isinstance(error, ModelGatewayError):
        value = error.details.get("retry_after_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delay = float(value)
            if math.isfinite(delay) and delay >= 0:
                return min(delay, 7 * 24 * 60 * 60)
    # 即使响应头缺失或格式错误，也会让整个账户短暂冷却。共享重试包装器还会叠加常规的
    # 指数抖动。
    return 1.0


def _quota_for_endpoint(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    binding: ModelTierBinding | None,
) -> model_profiles.ModelProfileQuota:
    if binding is not None:
        quota_profile_id = binding.quota_profile_id or binding.profile_id
        current = (
            model_profiles.resolve_profile(quota_profile_id)
            if quota_profile_id
            else None
        )
        if (
            current is not None
            and current.kind == "model"
            and current.provider == provider
            and current.base_url.rstrip("/") == base_url.rstrip("/")
            and current.api_key == api_key
        ):
            # 配额策略属于运行策略而非语义模型身份：即使持久端点绑定保持不变，管理员收紧
            # 共享账户限制也必须影响后续重试。
            return current.quota
        return binding.quota
    active = model_profiles.resolve_active_profile("model")
    if (
        active is not None
        and active.provider == provider
        and active.base_url.rstrip("/") == base_url.rstrip("/")
        and active.api_key == api_key
    ):
        return active.quota
    return model_profiles.ModelProfileQuota()


def _prepare_model_api_quota(
    *,
    context_budget: PreparedAdmittedProviderRequest,
    provider: str,
    base_url: str,
    api_key: str,
    timeout_s: float | None,
    binding: ModelTierBinding | None,
) -> PreparedApiQuotaRequest | None:
    # Quota authority comes from the immutable receipt sealed to the exact wire
    # bytes.  Metadata remains an observability projection and must not become a
    # second mutable/copyable admission authority.
    receipt = context_budget.admitted_request.receipt
    return prepare_api_quota_request(
        base_url=base_url,
        credential=api_key,
        quota=_quota_for_endpoint(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            binding=binding,
        ),
        estimated_input_tokens=receipt.quota_input_token_estimate,
        output_token_limit=receipt.reserved_output_tokens,
        provider_timeout_seconds=_effective_model_timeout_s(timeout_s),
    )

_MODEL_HTTP_TIMEOUT_CEILING_S: ContextVar[float | None] = ContextVar(
    "personagraph_model_http_timeout_ceiling_s", default=None
)
DEFAULT_MODEL_TIMEOUT_S = 60.0


@contextmanager
def model_http_timeout_ceiling(timeout_s: float) -> Iterator[None]:
    """临时限制一次物理 Runtime 调用等待供应商 HTTP 响应的时长。"""

    normalized = float(timeout_s)
    if normalized <= 0.0:
        raise ValueError("model HTTP timeout ceiling must be positive")
    token = _MODEL_HTTP_TIMEOUT_CEILING_S.set(normalized)
    try:
        yield
    finally:
        _MODEL_HTTP_TIMEOUT_CEILING_S.reset(token)


def _effective_model_timeout_s(timeout_s: float | None) -> float:
    configured = (
        float(timeout_s)
        if timeout_s is not None
        else DEFAULT_MODEL_TIMEOUT_S
    )
    ceiling = _MODEL_HTTP_TIMEOUT_CEILING_S.get()
    return configured if ceiling is None else min(configured, ceiling)
