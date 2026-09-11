"""一次真实 Provider 结构化输出修复的可选冒烟测试。

该测试会被常规测试套件收集，但只有显式设置
``PERSONAGRAPH_RUN_LIVE_REPAIR_SMOKE=1`` 才会执行提供方 I/O。凭据和端点
仅在主动启用的测试内部读取，且绝不会被记录。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)
from personagraph.model_io.gateway import ModelResult, complete_structured
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
)
from personagraph.runtime.model_calls import (
    request_model_with_retry,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.model_io.prepared_structured_provider import (
    prepare_structured_repair_request,
)
from personagraph.runtime.turn_events import RuntimeStage, TurnEvent


_RUN_ENV = "PERSONAGRAPH_RUN_LIVE_REPAIR_SMOKE"
_PURPOSE = "live_structured_repair_smoke"
_TARGET_CONTRACT = "live-repair-smoke-target-v1"
_SYSTEM_PROMPT = """你是结构化输出修复链路的端点 smoke 测试模型。
只输出一个 JSON object，不要输出 Markdown、解释或额外字段。
顶层必须且只能包含 schema_version、status、repaired：
- schema_version 固定为 live-repair-smoke-v1；
- status 只能是 initial 或 repaired；
- repaired 必须是 JSON boolean。
严格服从当前最后一条 Host 用户消息给出的目标值。"""
_USER_CONTENT = json.dumps(
    {
        "instruction": "首轮输出初始候选。",
        "target": {
            "schema_version": "live-repair-smoke-v1",
            "status": "initial",
            "repaired": False,
        },
    },
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
_INITIAL_MOCK_PAYLOAD = {
    "schema_version": "live-repair-smoke-v1",
    "status": "initial",
    "repaired": False,
}


pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_ENV) != "1",
    reason=f"set {_RUN_ENV}=1 to run the real Provider repair smoke",
)


class _RepairTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["live-repair-smoke-v1"]
    status: Literal["repaired"]
    repaired: Literal[True]


@dataclass(frozen=True, slots=True)
class _AttemptMetrics:
    provider: str
    model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None


class _ObservedPreparedRequest:
    """只记录一次已准备分发中的非敏感结果元数据。"""

    def __init__(self, prepared: object, metrics: list[_AttemptMetrics]) -> None:
        self._prepared = prepared
        self._metrics = metrics

    def dispatch(self, *, model_call_id: str) -> ModelResult:
        dispatch = getattr(self._prepared, "dispatch", None)
        if not callable(dispatch):
            raise TypeError("complete_structured.prepare returned no dispatch seam")
        result = dispatch(model_call_id=model_call_id)
        if not isinstance(result, ModelResult):
            raise TypeError("prepared Provider dispatch returned the wrong contract")
        self._metrics.append(
            _AttemptMetrics(
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        )
        return result


class _ObservedPreparedProvider:
    """委托给生产准备器时只保留角色，绝不保留正文。"""

    def __init__(
        self,
        *,
        metrics: list[_AttemptMetrics],
        message_roles: list[tuple[str, ...]],
    ) -> None:
        self._metrics = metrics
        self._message_roles = message_roles

    def prepare(
        self,
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        **kwargs: object,
    ) -> _ObservedPreparedRequest:
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is None:
            roles = ("system", "user")
        else:
            if not isinstance(repair_messages, list):
                raise TypeError("repair_messages must be a list")
            roles = tuple(
                str(message.get("role"))
                for message in repair_messages
                if isinstance(message, dict)
            )
            if len(roles) != len(repair_messages):
                raise TypeError("every repair message must be an object")
        self._message_roles.append(roles)
        prepared = complete_structured.prepare(  # type: ignore[attr-defined]
            system_prompt,
            user_content,
            purpose=purpose,
            **kwargs,
        )
        return _ObservedPreparedRequest(prepared, self._metrics)


def _required_live_setting(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        pytest.fail(
            f"{_RUN_ENV}=1 requires non-empty {name}",
            pytrace=False,
        )
    return value


def _record_optional_total(
    record_property: Callable[[str, object], None],
    name: str,
    values: tuple[int | None, ...],
) -> None:
    record_property(
        name,
        sum(value for value in values if value is not None)
        if all(value is not None for value in values)
        else "unreported",
    )


def test_live_openai_compatible_repair_smoke(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """在本地拒绝第一次尝试，随后只接受修复后的目标 JSON。"""

    # 成功的提供方调用和有意拒绝的输出都不得进入轨迹存储。该冒烟测试不拥有
    # 任何持久化或会话存储。
    monkeypatch.setenv("PERSONAGRAPH_TRAJECTORY", "off")
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url=_required_live_setting("PERSONAGRAPH_BASE_URL"),
        model=_required_live_setting("PERSONAGRAPH_MODEL"),
        api_key=_required_live_setting("PERSONAGRAPH_API_KEY"),
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        profile_id="live-repair-smoke",
        profile_name="Live repair smoke",
        request_dialect=(
            os.environ.get("PERSONAGRAPH_REQUEST_DIALECT") or "auto"
        ).strip(),
    )
    metrics: list[_AttemptMetrics] = []
    message_roles: list[tuple[str, ...]] = []
    provider = _ObservedPreparedProvider(
        metrics=metrics,
        message_roles=message_roles,
    )
    def prepare_kwargs():
        return {
            "mock_payload": _INITIAL_MOCK_PAYLOAD,
            "max_tokens": 512,
            "timeout_s": 120.0,
            "json_mode": True,
            "binding": binding,
        }
    prepare_repair = prepare_structured_repair_request(
        provider,
        system_prompt=_SYSTEM_PROMPT,
        user_content=_USER_CONTENT,
        purpose=_PURPOSE,
        prepare_kwargs=prepare_kwargs,
    )
    validation_calls = 0

    def validate(result: ModelResult) -> _RepairTarget:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise ModelOutputValidationError(
                "intentional first-attempt rejection for live repair smoke",
                repair_code="live_repair_smoke.target_required",
                safe_repair_reason=(
                    "Regenerate the complete target object with status=repaired "
                    "and repaired=true."
                ),
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category=(
                            RuntimeModelOutputRepairIssueCategory.HOST_GUARD
                        ),
                        code="host_guard.live_repair_smoke.target_required",
                        paths=("/repaired", "/status"),
                        safe_explanation=(
                            "输出必须将 status 设为 repaired，并将 repaired 设为 true。"
                        ),
                    ),
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.COMPLETE
                ),
            )
        try:
            return _RepairTarget.model_validate_json(result.reply)
        except ValidationError:
            # 不要串联解析器错误：Pydantic 的异常呈现可能包含被拒绝的响应片段。
            raise ModelOutputValidationError(
                "live smoke repair did not return the target contract",
                retryable=False,
                repair_code="live_repair_smoke.target_invalid",
                safe_repair_reason=(
                    "The repaired response must exactly satisfy the smoke target."
                ),
            ) from None

    events: list[TurnEvent] = []
    requested = request_model_with_retry(
        turn_id="live-repair-smoke-turn",
        session_id=None,
        purpose=_PURPOSE,
        stage=RuntimeStage.VERIFICATION,
        prepare_request=lambda: provider.prepare(
            _SYSTEM_PROMPT,
            _USER_CONTENT,
            purpose=_PURPOSE,
            **prepare_kwargs(),
        ),
        prepare_repair_request=prepare_repair,
        repair_target_contract=_TARGET_CONTRACT,
        validate=validate,
        emit=events.append,
        max_attempts=2,
    )

    assert requested.value == _RepairTarget(
        schema_version="live-repair-smoke-v1",
        status="repaired",
        repaired=True,
    )
    assert requested.attempts == 2
    assert validation_calls == 2
    assert message_roles == [
        ("system", "user"),
        ("system", "user", "assistant", "user"),
    ]
    assert len(metrics) == 2
    assert all(item.provider == "openai-compatible" for item in metrics)

    record_property("live_repair.provider", metrics[-1].provider)
    record_property("live_repair.model", metrics[-1].model)
    record_property("live_repair.attempts", requested.attempts)
    record_property(
        "live_repair.latency_ms_total",
        sum(item.latency_ms for item in metrics),
    )
    _record_optional_total(
        record_property,
        "live_repair.input_tokens_total",
        tuple(item.input_tokens for item in metrics),
    )
    _record_optional_total(
        record_property,
        "live_repair.output_tokens_total",
        tuple(item.output_tokens for item in metrics),
    )
