"""对规范提供商请求信封执行最终准入的门。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import math
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

from .contracts import (
    CanonicalEnvelopeMeasurement,
    ContextBudgetPressure,
    ContextBudgetReceipt,
    ContextBudgetViolation,
    ContextWindowProfile,
    EnvelopeTokenComponent,
    ProviderEnvelopeMeterResult,
    ProviderEnvelopeTokenMeter,
    TextTokenCounter,
    TokenEstimateKind,
)
from .errors import (
    ContextBudgetEncodingError,
    ContextBudgetControlMismatch,
    ContextBudgetEnvelopeMismatch,
    ContextBudgetExceeded,
    ContextBudgetInputError,
    ContextBudgetMeasurementError,
    ContextBudgetRouteMismatch,
)
from .estimation import token_counter_fingerprint


def canonical_json_bytes(envelope: Any) -> bytes:
    """确定性地序列化精确 JSON 请求，否则采用失败关闭策略。"""

    try:
        rendered = json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        return rendered.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ContextBudgetEncodingError(
            "provider envelope is not supported canonical UTF-8 JSON"
        ) from exc


def _count_text(counter: TextTokenCounter, text: str) -> int:
    try:
        value = counter.count_text(text)
    except Exception as exc:
        raise ContextBudgetMeasurementError(
            "token counter failed while measuring a provider envelope"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextBudgetMeasurementError(
            "token counter must return a non-negative integer"
        )
    return value


def _provider_meter_route(
    meter: ProviderEnvelopeTokenMeter,
) -> tuple[str, str, str]:
    """从可信计量器标识读取一个非空物理路由。"""

    try:
        route = (
            str(meter.provider or "").strip(),
            str(meter.model or "").strip(),
            str(meter.dialect or "").strip(),
        )
    except Exception as exc:
        raise ContextBudgetMeasurementError(
            "provider envelope meter has no valid physical route"
        ) from exc
    if not all(route):
        raise ContextBudgetMeasurementError(
            "provider envelope meter has no valid physical route"
        )
    return route


def _require_json_object_bytes(
    serialized_envelope: bytes,
    *,
    label: str = "provider envelope",
) -> None:
    try:
        decoded = serialized_envelope.decode("utf-8")
        parsed = json.loads(
            decoded,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextBudgetEncodingError(
            f"{label} bytes are not a valid UTF-8 JSON object"
        ) from exc
    if not isinstance(parsed, dict):
        raise ContextBudgetEncodingError(
            f"{label} bytes must encode one JSON object"
        )


def measure_precounted_envelope_bytes(
    serialized_envelope: bytes,
    *,
    provider: str,
    model: str,
    dialect: str,
    counter: TextTokenCounter,
    components: Iterable[EnvelopeTokenComponent],
    output_token_limit: int,
    shared_reasoning_reserve_tokens: int,
    request_controls: Any,
) -> CanonicalEnvelopeMeasurement:
    """将提供商专属词元账单绑定到精确的 UTF-8 线上字节。"""

    if not isinstance(serialized_envelope, bytes):
        raise TypeError("serialized_envelope must be bytes")
    _require_json_object_bytes(serialized_envelope)
    controls_bytes = canonical_json_bytes(request_controls)
    _require_json_object_bytes(controls_bytes, label="request controls")
    component_tuple = tuple(sorted(components, key=lambda item: item.component_id))
    return CanonicalEnvelopeMeasurement(
        provider=provider,
        model=model,
        dialect=dialect,
        request_sha256=sha256(serialized_envelope).hexdigest(),
        serialized_request_utf8_bytes=len(serialized_envelope),
        output_token_limit=output_token_limit,
        shared_reasoning_reserve_tokens=shared_reasoning_reserve_tokens,
        request_controls_sha256=sha256(controls_bytes).hexdigest(),
        input_tokens=sum(item.tokens for item in component_tuple),
        quota_input_token_estimate=sum(item.tokens for item in component_tuple),
        token_counter=token_counter_fingerprint(counter),
        components=component_tuple,
    )


def measure_envelope_with_meter(
    serialized_envelope: bytes,
    *,
    meter: ProviderEnvelopeTokenMeter,
) -> CanonicalEnvelopeMeasurement:
    """针对精确不可变线上字节运行一个绑定路由的提供商计量器。"""

    if not isinstance(serialized_envelope, bytes):
        raise TypeError("serialized_envelope must be bytes")
    _require_json_object_bytes(serialized_envelope)
    route = _provider_meter_route(meter)
    counter = token_counter_fingerprint(meter)
    try:
        result = meter.measure(serialized_envelope)
    except ContextBudgetInputError:
        raise
    except Exception as exc:
        raise ContextBudgetMeasurementError(
            "provider envelope meter failed"
        ) from exc
    if not isinstance(result, ProviderEnvelopeMeterResult):
        raise ContextBudgetMeasurementError(
            "provider envelope meter returned the wrong contract"
        )
    if (
        _provider_meter_route(meter) != route
        or token_counter_fingerprint(meter) != counter
    ):
        raise ContextBudgetMeasurementError(
            "provider envelope meter identity changed during measurement"
        )
    components = tuple(sorted(result.components, key=lambda item: item.component_id))
    return CanonicalEnvelopeMeasurement(
        provider=route[0],
        model=route[1],
        dialect=route[2],
        request_sha256=sha256(serialized_envelope).hexdigest(),
        serialized_request_utf8_bytes=len(serialized_envelope),
        output_token_limit=result.output_token_limit,
        shared_reasoning_reserve_tokens=(
            result.shared_reasoning_reserve_tokens
        ),
        request_controls_sha256=result.request_controls_sha256,
        input_tokens=result.input_tokens,
        quota_input_token_estimate=result.quota_input_token_estimate,
        token_counter=counter,
        components=components,
    )


def measure_precounted_envelope(
    envelope: Any,
    *,
    provider: str,
    model: str,
    dialect: str,
    counter: TextTokenCounter,
    components: Iterable[EnvelopeTokenComponent],
    output_token_limit: int,
    shared_reasoning_reserve_tokens: int,
    request_controls: Any,
) -> CanonicalEnvelopeMeasurement:
    """依据精确请求字节冻结提供商专属词元账单。

    提供商适配器应在应用其聊天模板、原生工具、响应模式、成帧及视觉规则后使用此接缝。
    核心会重新计算请求哈希和正文大小，但只信任类型化的组件计数。
    """

    return measure_precounted_envelope_bytes(
        canonical_json_bytes(envelope),
        provider=provider,
        model=model,
        dialect=dialect,
        counter=counter,
        components=components,
        output_token_limit=output_token_limit,
        shared_reasoning_reserve_tokens=shared_reasoning_reserve_tokens,
        request_controls=request_controls,
    )


def measure_canonical_json_envelope(
    envelope: Any,
    *,
    provider: str,
    model: str,
    dialect: str,
    counter: TextTokenCounter,
    estimate_kind: TokenEstimateKind,
    output_token_limit: int,
    shared_reasoning_reserve_tokens: int,
    request_controls: Any,
    extra_token_components: Mapping[str, int] | None = None,
) -> CanonicalEnvelopeMeasurement:
    """在没有适配器计量器时保守计量规范 JSON。

    此回退会将整个序列化 JSON 作为文本计数。若聊天成帧、原生图像或工具模式具有模型专属
    词元规则，提供商适配器应优先使用 :func:`measure_precounted_envelope`。
    """

    if estimate_kind not in {
        TokenEstimateKind.HEURISTIC,
        TokenEstimateKind.TOKENIZER,
    }:
        raise ContextBudgetInputError(
            "canonical JSON fallback cannot claim provider-derived token counts"
        )
    encoded = canonical_json_bytes(envelope)
    _require_json_object_bytes(encoded)
    controls_bytes = canonical_json_bytes(request_controls)
    _require_json_object_bytes(controls_bytes, label="request controls")
    components = [
        EnvelopeTokenComponent(
            component_id="canonical_json",
            tokens=_count_text(counter, encoded.decode("utf-8")),
            estimate_kind=estimate_kind,
        )
    ]
    for component_id, tokens in sorted((extra_token_components or {}).items()):
        if component_id == "canonical_json":
            raise ContextBudgetInputError(
                "extra token component cannot replace canonical_json"
            )
        components.append(
            EnvelopeTokenComponent(
                component_id=component_id,
                tokens=tokens,
                estimate_kind=estimate_kind,
            )
        )
    component_tuple = tuple(components)
    return CanonicalEnvelopeMeasurement(
        provider=provider,
        model=model,
        dialect=dialect,
        request_sha256=sha256(encoded).hexdigest(),
        serialized_request_utf8_bytes=len(encoded),
        output_token_limit=output_token_limit,
        shared_reasoning_reserve_tokens=shared_reasoning_reserve_tokens,
        request_controls_sha256=sha256(controls_bytes).hexdigest(),
        input_tokens=sum(item.tokens for item in component_tuple),
        quota_input_token_estimate=sum(item.tokens for item in component_tuple),
        token_counter=token_counter_fingerprint(counter),
        components=component_tuple,
    )


def check_context_budget(
    measurement: CanonicalEnvelopeMeasurement,
    *,
    profile: ContextWindowProfile,
    purpose: str,
    projection_epoch: str,
    projection_generation: int,
) -> ContextBudgetReceipt:
    """返回一个物理请求的不含内容准入决策。"""

    measured_route = (measurement.provider, measurement.model, measurement.dialect)
    profile_route = (profile.provider, profile.model, profile.dialect)
    if measured_route != profile_route:
        raise ContextBudgetRouteMismatch(
            "canonical envelope route does not match its context-window profile"
        )
    if (
        measurement.output_token_limit != profile.reserved_output_tokens
        or measurement.shared_reasoning_reserve_tokens
        != profile.shared_reasoning_reserve_tokens
    ):
        raise ContextBudgetControlMismatch(
            "measured output/reasoning controls do not match their frozen reserves"
        )

    input_budget = profile.input_budget_tokens
    soft_limit = max(1, math.ceil(input_budget * profile.soft_pressure_ratio))
    violations: list[ContextBudgetViolation] = []
    if measurement.input_tokens > input_budget:
        violations.append(ContextBudgetViolation.INPUT_TOKENS)
    if (
        profile.max_serialized_request_utf8_bytes is not None
        and measurement.serialized_request_utf8_bytes
        > profile.max_serialized_request_utf8_bytes
    ):
        violations.append(ContextBudgetViolation.REQUEST_BODY_BYTES)

    if violations:
        pressure = ContextBudgetPressure.HARD_LIMIT_EXCEEDED
    elif measurement.input_tokens >= soft_limit:
        pressure = ContextBudgetPressure.SOFT_LIMIT_EXCEEDED
    else:
        pressure = ContextBudgetPressure.NORMAL

    return ContextBudgetReceipt(
        profile_id=profile.profile_id,
        provider=profile.provider,
        model=profile.model,
        dialect=profile.dialect,
        purpose=purpose,
        projection_epoch=projection_epoch,
        projection_generation=projection_generation,
        request_sha256=measurement.request_sha256,
        context_window_tokens=profile.context_window_tokens,
        configured_input_limit_tokens=profile.configured_input_limit_tokens,
        provider_input_capacity_tokens=profile.provider_input_capacity_tokens,
        reserved_output_tokens=profile.reserved_output_tokens,
        shared_reasoning_reserve_tokens=profile.shared_reasoning_reserve_tokens,
        safety_margin_tokens=profile.safety_margin_tokens,
        request_controls_sha256=measurement.request_controls_sha256,
        input_budget_tokens=input_budget,
        soft_input_limit_tokens=soft_limit,
        input_tokens=measurement.input_tokens,
        quota_input_token_estimate=measurement.quota_input_token_estimate,
        serialized_request_utf8_bytes=measurement.serialized_request_utf8_bytes,
        max_serialized_request_utf8_bytes=(
            profile.max_serialized_request_utf8_bytes
        ),
        soft_pressure_ratio=profile.soft_pressure_ratio,
        pressure_ratio=measurement.input_tokens / input_budget,
        pressure=pressure,
        admitted=not violations,
        violations=tuple(violations),
        token_counter=measurement.token_counter,
        components=measurement.components,
    )


def require_context_budget(
    measurement: CanonicalEnvelopeMeasurement,
    *,
    profile: ContextWindowProfile,
    purpose: str,
    projection_epoch: str,
    projection_generation: int,
) -> ContextBudgetReceipt:
    """返回已准入回执，否则在提供商 I/O 前失败。"""

    receipt = check_context_budget(
        measurement,
        profile=profile,
        purpose=purpose,
        projection_epoch=projection_epoch,
        projection_generation=projection_generation,
    )
    if not receipt.admitted:
        raise ContextBudgetExceeded(receipt)
    return receipt


class AdmittedContextRequest:
    """与不含内容回执配对的瞬态不可变字节。"""

    __slots__ = ("_receipt", "_serialized_envelope", "__weakref__")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "AdmittedContextRequest can only be issued by the final context gate"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("AdmittedContextRequest is immutable")

    def __repr__(self) -> str:
        return f"AdmittedContextRequest(receipt={self._receipt!r})"

    @property
    def receipt(self) -> ContextBudgetReceipt:
        return self._receipt

    def _require_gate_authority(self) -> None:
        with _ADMISSION_REGISTRY_LOCK:
            record = _ADMISSION_REGISTRY.get(self)
        if record is None:
            raise ContextBudgetEnvelopeMismatch(
                "dispatch request was not issued by the final context gate"
            )
        try:
            current_record = (
                sha256(
                    canonical_json_bytes(self._receipt.model_dump(mode="json"))
                ).hexdigest(),
                sha256(self._serialized_envelope).hexdigest(),
                len(self._serialized_envelope),
            )
        except Exception as exc:
            raise ContextBudgetEnvelopeMismatch(
                "dispatch request authority could not be verified"
            ) from exc
        if current_record != record:
            raise ContextBudgetEnvelopeMismatch(
                "dispatch request changed after final context admission"
            )

    def body_for_dispatch(self) -> bytes:
        """返回最终门准入的精确不可变字节。"""

        self.require_dispatch_body(self._serialized_envelope)
        return self._serialized_envelope

    def require_dispatch_body(self, body: bytes) -> None:
        """如果调用方尝试派发替换后的请求正文，则失败。"""

        self._require_gate_authority()
        if not isinstance(body, bytes):
            raise TypeError("dispatch body must be bytes")
        if (
            len(body) != self.receipt.serialized_request_utf8_bytes
            or sha256(body).hexdigest() != self.receipt.request_sha256
        ):
            raise ContextBudgetEnvelopeMismatch(
                "dispatch body differs from the admitted provider envelope"
            )


_ADMISSION_REGISTRY: WeakKeyDictionary[
    AdmittedContextRequest,
    tuple[str, str, int],
] = WeakKeyDictionary()
_ADMISSION_REGISTRY_LOCK = RLock()


def _issue_admitted_context_request(
    *,
    receipt: ContextBudgetReceipt,
    serialized_envelope: bytes,
) -> AdmittedContextRequest:
    if not receipt.admitted:
        raise ValueError("AdmittedContextRequest requires an admitted receipt")
    request = object.__new__(AdmittedContextRequest)
    object.__setattr__(request, "_receipt", receipt)
    object.__setattr__(request, "_serialized_envelope", serialized_envelope)
    with _ADMISSION_REGISTRY_LOCK:
        _ADMISSION_REGISTRY[request] = (
            sha256(
                canonical_json_bytes(receipt.model_dump(mode="json"))
            ).hexdigest(),
            sha256(serialized_envelope).hexdigest(),
            len(serialized_envelope),
        )
    request.require_dispatch_body(serialized_envelope)
    return request


def admit_context_request(
    serialized_envelope: bytes,
    *,
    meter: ProviderEnvelopeTokenMeter,
    profile: ContextWindowProfile,
    purpose: str,
    projection_epoch: str,
    projection_generation: int,
) -> AdmittedContextRequest:
    """计量并准入唯一可派发的精确字节。"""

    if not isinstance(serialized_envelope, bytes):
        raise TypeError("serialized_envelope must be bytes")
    measurement = measure_envelope_with_meter(
        serialized_envelope,
        meter=meter,
    )
    receipt = require_context_budget(
        measurement,
        profile=profile,
        purpose=purpose,
        projection_epoch=projection_epoch,
        projection_generation=projection_generation,
    )
    return _issue_admitted_context_request(
        receipt=receipt,
        serialized_envelope=serialized_envelope,
    )


__all__ = [
    "AdmittedContextRequest",
    "admit_context_request",
    "canonical_json_bytes",
    "check_context_budget",
    "measure_canonical_json_envelope",
    "measure_envelope_with_meter",
    "measure_precounted_envelope",
    "measure_precounted_envelope_bytes",
    "require_context_budget",
]
