"""最终上下文预算准入 gate 的供应商侧准备。

本模块被刻意设计为 :mod:`personagraph.context_budget` 周围的适配器：核心层无需了解模型
profile、请求方言或旧 token 计数器。此处没有函数执行 HTTP I/O。调用方只能分派
``AdmittedContextRequest.body_for_dispatch()`` 返回的字节。
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Any
import warnings

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from ..context_budget.token_counter import TokenCounter, get_token_counter
from ..context_budget import (
    AdmittedContextRequest,
    ContextBudgetControlMismatch,
    ContextBudgetEncodingError,
    ContextBudgetInputError,
    ContextBudgetMeasurementError,
    ContextBudgetRouteMismatch,
    ContextWindowProfile,
    EnvelopeTokenComponent,
    ProviderEnvelopeMeterResult,
    TokenEstimateKind,
    admit_context_request,
    canonical_json_bytes,
)
from .dialects import RequestDialect
from .capability_profiles import ModelProfile, profile_for_model


# 为供应商侧聊天 framing 和小幅 tokenizer 漂移留出空间。它被刻意设为独立于从线上请求
# 读取的输出 token 预留。
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 1_024

# 即使粗略 token 估算碰巧可容纳，也要在本地拒绝病态 JSON 正文。8 MiB 是保守 Host 上限，
# 不是对任何供应商当前 HTTP 限制的声明；后续供应商 profile 可以安全收紧它。
DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES = 8 * 1024 * 1024
DEFAULT_CONTEXT_SOFT_PRESSURE_RATIO = 0.90

# 历史 DeepSeek live settlement 中，raw heuristic 对真实 provider input 的最高低估约
# 19%。该方言的 TPM 使用 25% 明示余量；没有校准证据的其他 heuristic 路由继续使用
# context hard bound，直到它们拥有精确 tokenizer 或各自的结算校准。
DEEPSEEK_QUOTA_HEURISTIC_HEADROOM_RATIO = 1.25

_TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
_CONTENT_BEARING_FIELDS = frozenset({"messages", "system", "tools"})
_OPENAI_DIALECTS = frozenset(
    {
        RequestDialect.DEEPSEEK_OPENAI,
        RequestDialect.OPENAI_NATIVE,
        RequestDialect.GENERIC_OPENAI,
    }
)
_ANTHROPIC_DIALECTS = frozenset(
    {
        RequestDialect.DEEPSEEK_ANTHROPIC,
        RequestDialect.ANTHROPIC_NATIVE,
        RequestDialect.GENERIC_ANTHROPIC,
    }
)


class PreparedProviderEnvelopeMetadata(BaseModel):
    """一个已准入供应商 payload 的无内容身份与计量单。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=500)
    dialect: str = Field(min_length=1, max_length=200)
    profile_id: str = Field(min_length=1, max_length=200)
    stream: bool
    output_token_field: str = Field(pattern=r"^max_(?:completion_)?tokens$")
    output_token_limit: int = Field(gt=0)
    context_window_tokens: int = Field(gt=0)
    input_budget_tokens: int = Field(gt=0)
    soft_input_limit_tokens: int = Field(gt=0)
    measured_input_tokens: int = Field(gt=0)
    quota_input_token_estimate: int = Field(gt=0)
    safety_margin_tokens: int = Field(ge=0)
    pressure: str = Field(pattern=r"^(?:normal|soft_limit_exceeded)$")
    pressure_ratio: float = Field(ge=0.0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    serialized_request_utf8_bytes: int = Field(gt=0)
    request_controls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    meter_name: str = Field(min_length=1, max_length=200)
    meter_kind: str = Field(pattern=r"^(?:heuristic|tokenizer)$")


@dataclass(frozen=True, slots=True)
class PreparedAdmittedProviderRequest:
    """精确分派 authority 及不保留 prompt 内容的元数据。"""

    admitted_request: AdmittedContextRequest = field(repr=False)
    metadata: PreparedProviderEnvelopeMetadata


@dataclass(frozen=True, slots=True, init=False)
class RouteBoundJsonEnvelopeMeter:
    """根据精确线上字节保守计量一条具体 chat-JSON 路由。

    当前仓库没有供应商精确 count-tokens 客户端。因此本适配器报告旧计数器真实的
    ``heuristic`` 或 ``tokenizer`` provenance。它计算完整文本/控制 JSON 形状，但移除内联
    base64 图像传输文本，并通过模型 profile 中显式启发式视觉组件计量每张图像。将路由和
    控制语法计入输入是刻意的保守做法。
    """

    provider: str
    model: str
    dialect: str
    name: str
    kind: str
    requested: str
    fallback_reason: str | None
    _counter: TokenCounter = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dialect: str | RequestDialect,
        counter: TokenCounter | None = None,
    ) -> None:
        normalized_provider, normalized_model, normalized_dialect = _normalize_route(
            provider=provider,
            model=model,
            dialect=dialect,
        )
        resolved_counter = counter if counter is not None else get_token_counter()
        try:
            counter_name = str(resolved_counter.name or "").strip()
            counter_kind = str(resolved_counter.kind or "").strip()
            counter_requested = str(resolved_counter.requested or "").strip()
            fallback_reason = resolved_counter.fallback_reason
        except Exception as exc:
            raise ContextBudgetMeasurementError(
                "configured token counter has no stable identity"
            ) from exc
        if not counter_name or not counter_requested:
            raise ContextBudgetMeasurementError(
                "configured token counter has no stable identity"
            )
        if counter_kind not in {"heuristic", "tokenizer"}:
            raise ContextBudgetMeasurementError(
                "JSON envelope meter requires a heuristic or tokenizer counter"
            )
        if fallback_reason is not None and not isinstance(fallback_reason, str):
            raise ContextBudgetMeasurementError(
                "configured token counter fallback reason must be text"
            )
        object.__setattr__(self, "provider", normalized_provider)
        object.__setattr__(self, "model", normalized_model)
        object.__setattr__(self, "dialect", normalized_dialect.value)
        object.__setattr__(self, "name", f"route-json/{counter_name}"[:200])
        object.__setattr__(self, "kind", counter_kind)
        object.__setattr__(self, "requested", counter_requested[:200])
        object.__setattr__(self, "fallback_reason", fallback_reason)
        object.__setattr__(self, "_counter", resolved_counter)

    def measure(self, serialized_envelope: bytes) -> ProviderEnvelopeMeterResult:
        _decoded, payload = _decode_json_object(serialized_envelope)
        dialect = RequestDialect(self.dialect)
        inspection = _inspect_wire_payload(
            payload,
            provider=self.provider,
            model=self.model,
            dialect=dialect,
        )
        image_components = _strip_inline_images_for_meter(
            payload,
            dialect=dialect,
            model_profile=profile_for_model(self.model, provider=self.provider),
        )
        metered_json = _canonical_meter_json_text(payload)
        try:
            text_token_count = self._counter.count_text(metered_json)
        except Exception as exc:
            raise ContextBudgetMeasurementError(
                "configured token counter failed on the provider envelope"
            ) from exc
        if (
            isinstance(text_token_count, bool)
            or not isinstance(text_token_count, int)
            or text_token_count <= 0
        ):
            raise ContextBudgetMeasurementError(
                "provider envelope token count must be a positive integer"
            )
        nominal_text_token_count = text_token_count
        if self.kind == "heuristic":
            # 旧 char/4 回退适合 packing，但对 emoji、高熵 ASCII、假名或陌生
            # tokenizer 并非安全最终边界。每 UTF-8 字节一个 token 是刻意悲观的
            # 估算，可约束普通字节/字符 tokenizer，同时保持一级投影估算器低成本。
            text_token_count = max(
                text_token_count,
                len(metered_json.encode("utf-8")),
            )
        estimate_kind = (
            TokenEstimateKind.TOKENIZER
            if self.kind == "tokenizer"
            else TokenEstimateKind.HEURISTIC
        )
        controls = {
            key: value
            for key, value in payload.items()
            if key not in _CONTENT_BEARING_FIELDS
        }
        controls_sha256 = sha256(canonical_json_bytes(controls)).hexdigest()
        text_component = EnvelopeTokenComponent(
            component_id="wire_json_without_inline_image_data",
            tokens=text_token_count,
            estimate_kind=estimate_kind,
        )
        components = (text_component, *image_components)
        hard_input_tokens = sum(item.tokens for item in components)
        unadjusted_input_token_estimate = nominal_text_token_count + sum(
            item.tokens for item in image_components
        )
        quota_input_token_estimate = (
            min(
                hard_input_tokens,
                math.ceil(
                    unadjusted_input_token_estimate
                    * DEEPSEEK_QUOTA_HEURISTIC_HEADROOM_RATIO
                ),
            )
            if self.kind == "heuristic"
            and dialect
            in {
                RequestDialect.DEEPSEEK_OPENAI,
                RequestDialect.DEEPSEEK_ANTHROPIC,
            }
            else hard_input_tokens
        )
        return ProviderEnvelopeMeterResult(
            output_token_limit=inspection.output_token_limit,
            # 当前线上控制使用一个共享 completion 信封：隐藏推理消耗相同的
            # max_tokens/max_completion_tokens 上限。
            shared_reasoning_reserve_tokens=0,
            request_controls_sha256=controls_sha256,
            input_tokens=hard_input_tokens,
            quota_input_token_estimate=quota_input_token_estimate,
            components=components,
        )


@dataclass(frozen=True, slots=True)
class _WireInspection:
    output_token_field: str
    output_token_limit: int
    stream: bool


@dataclass(frozen=True, slots=True)
class _InlineImageMeasurement:
    size_bytes: int
    media_type: str
    width: int
    height: int


def _strip_inline_images_for_meter(
    payload: dict[str, Any],
    *,
    dialect: RequestDialect,
    model_profile: ModelProfile,
) -> tuple[EnvelopeTokenComponent, ...]:
    """移除 base64 传输文本，并添加显式启发式视觉成本。

    精确原始字节在 ``AdmittedContextRequest`` 中保持不变，并仍计入 HTTP 正文上限。只有此
    已解析计量副本会被脱敏，从而防止 base64 传输膨胀伪装成 prompt 文本 token。
    """

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    openai_shape = dialect in _OPENAI_DIALECTS
    images: list[_InlineImageMeasurement] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if openai_shape and block_type == "image":
                raise ContextBudgetEncodingError(
                    "OpenAI-shaped provider payload contains an Anthropic image block"
                )
            if not openai_shape and block_type == "image_url":
                raise ContextBudgetEncodingError(
                    "Anthropic-shaped provider payload contains an OpenAI image block"
                )
            is_inline_image = (
                (openai_shape and block_type == "image_url")
                or (not openai_shape and block_type == "image")
            )
            if is_inline_image and not model_profile.accepts_images:
        # 无须触及攻击者控制的图像字节即可知晓路由能力。在此拒绝可避免不必要的
        # base64/Pillow 工作，并向调用方提供可操作的稳定能力错误。
                raise ContextBudgetInputError(
                    "selected model profile does not admit inline images"
                )
            parsed = (
                _strip_openai_image_block(block)
                if openai_shape and block_type == "image_url"
                else _strip_anthropic_image_block(block)
                if not openai_shape and block_type == "image"
                else None
            )
            if parsed is not None:
                images.append(parsed)
    if not images:
        return ()
    if len(images) > model_profile.max_images_per_call:
        raise ContextBudgetInputError(
            "provider payload exceeds the model profile's image-count limit"
        )
    components: list[EnvelopeTokenComponent] = []
    for ordinal, image in enumerate(images, start=1):
        if image.size_bytes > model_profile.max_image_bytes:
            raise ContextBudgetInputError(
                "provider payload exceeds the model profile's per-image byte limit"
            )
        # 基于字节的 profile 估算对传输开销大的图像仍有用；像素项则避免高度可压缩的超大 PNG
        # 看起来几乎免费。28px 网格是刻意保守的跨供应商上界估算，而非供应商精确用量。
        pixel_tokens = math.ceil(image.width / 28) * math.ceil(image.height / 28)
        components.append(
            EnvelopeTokenComponent(
                component_id=f"vision_image_{ordinal:04d}",
                tokens=max(
                    model_profile.image_tokens(image.size_bytes),
                    pixel_tokens,
                ),
                estimate_kind=TokenEstimateKind.HEURISTIC,
            )
        )
    return tuple(components)


def _strip_anthropic_image_block(
    block: dict[str, Any],
) -> _InlineImageMeasurement:
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        raise ContextBudgetEncodingError(
            "Anthropic image block requires one inline base64 source"
        )
    media_type = _require_image_media_type(source.get("media_type"))
    measurement = _inspect_decoded_image(source.get("data"), media_type=media_type)
    source["data"] = "<inline-image>"
    return measurement


def _strip_openai_image_block(block: dict[str, Any]) -> _InlineImageMeasurement:
    image_url = block.get("image_url")
    if not isinstance(image_url, dict):
        raise ContextBudgetEncodingError(
            "OpenAI image block requires an image_url object"
        )
    url = image_url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ContextBudgetEncodingError(
            "OpenAI image block requires one inline base64 data URL"
        )
    header, separator, encoded = url.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ContextBudgetEncodingError(
            "OpenAI image block requires one inline base64 data URL"
        )
    media_type = _require_image_media_type(header[5:-7])
    measurement = _inspect_decoded_image(encoded, media_type=media_type)
    image_url["url"] = f"data:{media_type};base64,<inline-image>"
    return measurement


def _require_image_media_type(value: object) -> str:
    media_type = str(value or "").strip().lower()
    if (
        not media_type.startswith("image/")
        or len(media_type) > 100
        or any(character.isspace() for character in media_type)
    ):
        raise ContextBudgetEncodingError(
            "inline image requires a bounded image media type"
        )
    return media_type


def _inspect_decoded_image(
    value: object,
    *,
    media_type: str,
) -> _InlineImageMeasurement:
    if not isinstance(value, str) or not value:
        raise ContextBudgetEncodingError("inline image base64 data is missing")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContextBudgetEncodingError(
            "inline image contains invalid base64 data"
        ) from exc
    if not decoded:
        raise ContextBudgetEncodingError("inline image payload is empty")
    expected_format = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
    }.get(media_type)
    if expected_format is None:
        raise ContextBudgetEncodingError(
            "inline image media type is not supported by the final meter"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(decoded)) as opened:
                width, height = opened.size
                actual_format = opened.format
                frame_count = int(getattr(opened, "n_frames", 1))
                opened.verify()
    except (ImportError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ContextBudgetEncodingError(
            "inline image bytes could not be safely inspected"
        ) from exc
    except Image.DecompressionBombWarning as exc:
        raise ContextBudgetInputError(
            "inline image exceeds the safe decompression bound"
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ContextBudgetInputError(
            "inline image exceeds the safe decompression bound"
        ) from exc
    if actual_format != expected_format:
        raise ContextBudgetEncodingError(
            "inline image media type does not match its encoded bytes"
        )
    if width < 1 or height < 1 or width > 16_384 or height > 16_384:
        raise ContextBudgetInputError(
            "inline image dimensions exceed the final meter's safe bound"
        )
    if width * height > 40_000_000:
        raise ContextBudgetInputError(
            "inline image pixel count exceeds the final meter's safe bound"
        )
    if frame_count != 1:
        raise ContextBudgetInputError(
            "multi-frame inline images are not supported"
        )
    return _InlineImageMeasurement(
        size_bytes=len(decoded),
        media_type=media_type,
        width=width,
        height=height,
    )


def _canonical_meter_json_text(payload: dict[str, Any]) -> str:
    """序列化脱敏计量副本，绝不再次序列化分派信封。"""

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContextBudgetEncodingError(
            "redacted provider meter payload is not finite canonical JSON"
        ) from exc


def build_context_window_profile(
    *,
    provider: str,
    model: str,
    dialect: str | RequestDialect,
    output_token_limit: int,
    configured_input_limit_tokens: int | None = None,
) -> ContextWindowProfile:
    """为一个具体线上请求冻结保守容量事实。"""

    normalized_provider, normalized_model, normalized_dialect = _normalize_route(
        provider=provider,
        model=model,
        dialect=dialect,
    )
    if (
        isinstance(output_token_limit, bool)
        or not isinstance(output_token_limit, int)
        or output_token_limit <= 0
    ):
        raise ContextBudgetControlMismatch(
            "provider output-token control must be a positive integer"
        )
    if configured_input_limit_tokens is not None and (
        isinstance(configured_input_limit_tokens, bool)
        or not isinstance(configured_input_limit_tokens, int)
        or configured_input_limit_tokens <= 0
    ):
        raise ContextBudgetInputError(
            "configured input-token limit must be a positive integer"
        )
    family_profile = profile_for_model(
        normalized_model,
        provider=normalized_provider,
    )
    profile_preimage = "\x1f".join(
        (
            normalized_provider,
            normalized_model,
            normalized_dialect.value,
            str(family_profile.context_window),
            str(output_token_limit),
            str(configured_input_limit_tokens or "provider"),
            str(DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS),
            str(DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES),
        )
    )
    profile_id = "context-window:" + sha256(profile_preimage.encode("utf-8")).hexdigest()[:24]
    try:
        return ContextWindowProfile(
            profile_id=profile_id,
            provider=normalized_provider,
            model=normalized_model,
            dialect=normalized_dialect.value,
            context_window_tokens=family_profile.context_window,
            configured_input_limit_tokens=configured_input_limit_tokens,
            reserved_output_tokens=output_token_limit,
    # 当前供应商控制公开共享 completion 上限，而非第二个可独立叠加的推理额度。
            shared_reasoning_reserve_tokens=0,
            safety_margin_tokens=DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
            soft_pressure_ratio=DEFAULT_CONTEXT_SOFT_PRESSURE_RATIO,
            max_serialized_request_utf8_bytes=(
                DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES
            ),
        )
    except ValueError as exc:
        raise ContextBudgetInputError(
            "provider route reserves leave no safe input context budget"
        ) from exc


def prepare_and_admit_provider_request(
    payload: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    dialect: str | RequestDialect,
    stream: bool,
    purpose: str,
    projection_epoch: str,
    projection_generation: int,
    configured_input_limit_tokens: int | None = None,
    counter: TokenCounter | None = None,
) -> PreparedAdmittedProviderRequest:
    """只序列化一次，随后计量并准入最终供应商 JSON payload。

    ``stream`` 会在序列化前插入，若现有值冲突则拒绝。返回对象不保留 payload 映射；其中的
    已准入请求是为后续分派所选精确字节的唯一所有者。

    这是完整 wire envelope 的预算门：已包含 Provider 方言、工具 schema、stream 和
    输出上限控制，不能用上游 Entry/L1 的估算代替。这里只得到准入凭据，不代表
    quota、durable physical attempt 或网络发送已经发生。
    """

    normalized_provider, normalized_model, normalized_dialect = _normalize_route(
        provider=provider,
        model=model,
        dialect=dialect,
    )
    if not isinstance(payload, Mapping):
        raise ContextBudgetInputError("provider payload must be a JSON object")
    if not all(isinstance(key, str) for key in payload):
        raise ContextBudgetInputError("provider payload keys must be strings")
    if not isinstance(stream, bool):
        raise ContextBudgetControlMismatch("provider stream control must be boolean")
    wire_payload = dict(payload)
    existing_stream = wire_payload.get("stream")
    if "stream" in wire_payload and existing_stream is not stream:
        raise ContextBudgetControlMismatch(
            "provider payload stream control conflicts with the selected transport"
        )
    wire_payload["stream"] = stream
    inspection = _inspect_wire_payload(
        wire_payload,
        provider=normalized_provider,
        model=normalized_model,
        dialect=normalized_dialect,
    )
    # 这是完整供应商 payload 的唯一序列化。meter 会解析并哈希这些字节；分派稍后必须复用。
    serialized_envelope = canonical_json_bytes(wire_payload)
    meter = RouteBoundJsonEnvelopeMeter(
        provider=normalized_provider,
        model=normalized_model,
        dialect=normalized_dialect,
        counter=counter,
    )
    profile = build_context_window_profile(
        provider=normalized_provider,
        model=normalized_model,
        dialect=normalized_dialect,
        output_token_limit=inspection.output_token_limit,
        configured_input_limit_tokens=configured_input_limit_tokens,
    )
    admitted = admit_context_request(
        serialized_envelope,
        meter=meter,
        profile=profile,
        purpose=purpose,
        projection_epoch=projection_epoch,
        projection_generation=projection_generation,
    )
    receipt = admitted.receipt
    return PreparedAdmittedProviderRequest(
        admitted_request=admitted,
        metadata=PreparedProviderEnvelopeMetadata(
            provider=normalized_provider,
            model=normalized_model,
            dialect=normalized_dialect.value,
            profile_id=profile.profile_id,
            stream=inspection.stream,
            output_token_field=inspection.output_token_field,
            output_token_limit=inspection.output_token_limit,
            context_window_tokens=receipt.context_window_tokens,
            input_budget_tokens=receipt.input_budget_tokens,
            soft_input_limit_tokens=receipt.soft_input_limit_tokens,
            measured_input_tokens=receipt.input_tokens,
            quota_input_token_estimate=receipt.quota_input_token_estimate,
            safety_margin_tokens=receipt.safety_margin_tokens,
            pressure=receipt.pressure.value,
            pressure_ratio=receipt.pressure_ratio,
            request_sha256=receipt.request_sha256,
            serialized_request_utf8_bytes=receipt.serialized_request_utf8_bytes,
            request_controls_sha256=receipt.request_controls_sha256,
            meter_name=receipt.token_counter.name,
            meter_kind=receipt.token_counter.kind,
        ),
    )


def _normalize_route(
    *,
    provider: object,
    model: object,
    dialect: object,
) -> tuple[str, str, RequestDialect]:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    try:
        normalized_dialect = RequestDialect(
            str(getattr(dialect, "value", dialect) or "").strip().lower()
        )
    except ValueError as exc:
        raise ContextBudgetRouteMismatch(
            "provider context meter requires a concrete supported request dialect"
        ) from exc
    if not normalized_provider or not normalized_model:
        raise ContextBudgetRouteMismatch(
            "provider context meter requires a non-empty physical route"
        )
    allowed = (
        _OPENAI_DIALECTS
        if normalized_provider == "openai-compatible"
        else _ANTHROPIC_DIALECTS
        if normalized_provider == "anthropic-compatible"
        else frozenset()
    )
    if normalized_dialect not in allowed:
        raise ContextBudgetRouteMismatch(
            "provider protocol and request dialect do not identify the same route"
        )
    return normalized_provider, normalized_model, normalized_dialect


def _decode_json_object(serialized_envelope: bytes) -> tuple[str, dict[str, Any]]:
    if not isinstance(serialized_envelope, bytes):
        raise TypeError("serialized_envelope must be bytes")
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
            "provider envelope bytes are not a finite UTF-8 JSON object"
        ) from exc
    if not isinstance(parsed, dict):
        raise ContextBudgetEncodingError(
            "provider envelope bytes must encode one JSON object"
        )
    return decoded, parsed


def _inspect_wire_payload(
    payload: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    dialect: RequestDialect,
) -> _WireInspection:
    if payload.get("model") != model:
        raise ContextBudgetRouteMismatch(
            "provider payload model does not match the route-bound context meter"
        )
    stream = payload.get("stream")
    if not isinstance(stream, bool):
        raise ContextBudgetControlMismatch(
            "final provider payload must contain one boolean stream control"
        )
    present_fields = tuple(field for field in _TOKEN_LIMIT_FIELDS if field in payload)
    if len(present_fields) != 1:
        raise ContextBudgetControlMismatch(
            "final provider payload must contain exactly one output-token control"
        )
    expected_field = (
        "max_completion_tokens"
        if dialect is RequestDialect.OPENAI_NATIVE
        else "max_tokens"
    )
    output_field = present_fields[0]
    if output_field != expected_field:
        raise ContextBudgetControlMismatch(
            "provider output-token field conflicts with the request dialect"
        )
    output_limit = payload[output_field]
    if (
        isinstance(output_limit, bool)
        or not isinstance(output_limit, int)
        or output_limit <= 0
    ):
        raise ContextBudgetControlMismatch(
            "provider output-token control must be a positive integer"
        )
    # 此处也要重新验证协议所有权：meter.measure() 是公开方法，即使调用方绕过 prepare helper，
    # 也必须保守失败。
    _normalize_route(provider=provider, model=model, dialect=dialect)
    return _WireInspection(
        output_token_field=output_field,
        output_token_limit=output_limit,
        stream=stream,
    )


__all__ = [
    "DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_CONTEXT_SOFT_PRESSURE_RATIO",
    "DEFAULT_MAX_SERIALIZED_REQUEST_UTF8_BYTES",
    "DEEPSEEK_QUOTA_HEURISTIC_HEADROOM_RATIO",
    "PreparedAdmittedProviderRequest",
    'PreparedProviderEnvelopeMetadata',
    "RouteBoundJsonEnvelopeMeter",
    "build_context_window_profile",
    "prepare_and_admit_provider_request",
]
