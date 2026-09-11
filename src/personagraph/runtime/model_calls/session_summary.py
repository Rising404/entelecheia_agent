"""Session summary 的有界模型调用适配器。"""

from __future__ import annotations

from collections.abc import Callable

from personagraph.configuration.app_settings import get_setting
from personagraph.context_budget.token_counter import block_cap
from personagraph.model_io.gateway import (
    DEFAULT_MODEL_TIMEOUT_S,
    ModelGatewayError,
    ModelResult,
    PreparedModelCall,
    anthropic_compatible_chat,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.session.session_summary import (
    SESSION_SUMMARY_SYSTEM_PROMPT,
    SessionSummaryGenerationError,
    SessionSummaryState,
    SessionSummaryTurnPair,
    mock_session_summary,
    render_session_summary_input,
)

from .requests import request_model_with_retry
from ..turn_events import RuntimeStage, TurnEvent


def generate_session_summary(
    state: SessionSummaryState | None,
    pairs: tuple[SessionSummaryTurnPair, ...],
    turn_id: str,
    job_id: str,
    emit: Callable[[TurnEvent], int | None] | None = None,
) -> str:
    """为一个已声明的摘要作业运行有界提供方调用。"""

    messages = [
        {"role": "system", "content": SESSION_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": render_session_summary_input(state, pairs)},
    ]
    provider = (get_setting("provider", "mock") or "mock").strip().lower()
    max_tokens = max(256, min(2048, block_cap("summary") * 2))
    summary_provider = anthropic_compatible_chat
    prepared_summary_provider = getattr(summary_provider, "prepare", None)

    def dispatch_summary(model_call_id: str | None) -> ModelResult:
        if model_call_id is None:
            raise ValueError("prepared session summary requires model_call_id")
        if provider in {"", "mock"}:
            return ModelResult(
                reply=mock_session_summary(state, pairs),
                provider="mock",
                model="mock-session-summary",
                latency_ms=0,
                model_call_id=model_call_id,
                purpose="session_summary",
            )
        return summary_provider(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_s=DEFAULT_MODEL_TIMEOUT_S,
            model_call_id=model_call_id,
            purpose="session_summary",
        )

    def prepare_request() -> PreparedModelCall:
        if provider not in {"", "mock"} and callable(prepared_summary_provider):
            return prepared_summary_provider(
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_s=DEFAULT_MODEL_TIMEOUT_S,
                purpose="session_summary",
            )
        return PreparedModelCall(_dispatch=dispatch_summary)

    def validate(result: ModelResult) -> str:
        if not isinstance(result.reply, str) or not result.reply.strip():
            raise ModelOutputValidationError("empty session summary")
        return result.reply.strip()

    try:
        return request_model_with_retry(
            turn_id=turn_id,
            session_id=None,
            purpose="session_summary",
            stage=RuntimeStage.PERSIST,
            prepare_request=prepare_request,
            validate=validate,
            emit=emit if emit is not None else _discard_summary_trace,
        ).value
    except ModelGatewayError as exc:
        raise SessionSummaryGenerationError(
            _summary_model_failure_code(exc),
            "The session summary model call failed.",
            retryable=False,
        ) from exc


def _discard_summary_trace(_event: TurnEvent) -> None:
    """隐藏派生摘要的模型活动不进入公开 Runtime 事件轨道。"""


def _summary_model_failure_code(error: ModelGatewayError) -> str:
    if error.code == "MODEL_CALL_TIMEOUT":
        return "SUMMARY_MODEL_TIMEOUT"
    if error.code in {"MODEL_BAD_RESPONSE", "SUMMARY_EMPTY_OUTPUT"}:
        return "SUMMARY_MODEL_OUTPUT_INVALID"
    if not error.retryable and error.details.get("physical_error_retryable") is not True:
        return "SUMMARY_MODEL_CONFIGURATION_FAILURE"
    return "SUMMARY_MODEL_TRANSPORT_FAILURE"


__all__ = ["generate_session_summary"]
