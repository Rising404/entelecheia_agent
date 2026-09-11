"""Small prepared-provider adapter for model-call unit tests."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from typing import Any

from personagraph.model_io.gateway import ModelResult, PreparedModelCall


def prepare_test_model_request(
    dispatch: Callable[[str], ModelResult],
) -> Callable[[], PreparedModelCall]:
    """Adapt a direct unit-test dispatch fake to the prepared-only runtime seam."""

    def prepare() -> PreparedModelCall:
        def prepared_dispatch(model_call_id: str | None) -> ModelResult:
            if model_call_id is None:
                raise ValueError("prepared test dispatch requires model_call_id")
            return dispatch(model_call_id)

        return PreparedModelCall(_dispatch=prepared_dispatch)

    return prepare


def as_prepared_test_provider(
    provider: Callable[..., ModelResult],
    *,
    add_l1_notes: bool = False,
) -> Callable[..., ModelResult]:
    """Give a legacy-shaped test fake the production ``prepare`` surface.

    Runtime callers are prepared-only. This adapter keeps individual tests focused on
    their parser or guard while still exercising the real prepared dispatch path.
    ``add_l1_notes`` 必须由各测试显式开启；只补缺失的 L1 工作笔记，不修复
    空笔记、无效笔记或其它模型输出错误，以保留协议拒绝测试的真实性。
    适配属性只写在独立包装器上，不能修改共享 Provider 或已有包装器的 prepare。
    """

    def adapt(
        result: ModelResult, *, user_content: str, purpose: object
    ) -> ModelResult:
        return _with_l1_notes(result, purpose=purpose) if add_l1_notes else result

    existing_prepare = getattr(provider, "prepare", None)
    if callable(existing_prepare) and not add_l1_notes:
        return provider

    @wraps(provider)
    def adapted_provider(*args: object, **kwargs: object) -> ModelResult:
        return provider(*args, **kwargs)

    if callable(existing_prepare):

        def prepare_with_notes(system_prompt: str, user_content: str, **kwargs: object):
            prepared = existing_prepare(system_prompt, user_content, **kwargs)
            dispatch = prepared._dispatch
            return replace(
                prepared,
                _dispatch=lambda call_id: adapt(
                    dispatch(call_id),
                    user_content=user_content,
                    purpose=kwargs.get("purpose"),
                ),
            )

        adapted_provider.prepare = prepare_with_notes  # type: ignore[attr-defined]
        return adapted_provider

    parameters = inspect.signature(provider).parameters
    accepts_extra_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def prepare(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> PreparedModelCall:
        call_kwargs: dict[str, Any] = dict(kwargs)
        if repair_messages is not None:
            call_kwargs["repair_messages"] = repair_messages
        if not accepts_extra_keywords:
            call_kwargs = {
                key: value for key, value in call_kwargs.items() if key in parameters
            }

        def dispatch(model_call_id: str | None) -> ModelResult:
            if model_call_id is None:
                raise ValueError("prepared test dispatch requires model_call_id")
            result = provider(
                system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
                **call_kwargs,
            )
            return adapt(result, user_content=user_content, purpose=purpose)

        return PreparedModelCall(_dispatch=dispatch)

    adapted_provider.prepare = prepare  # type: ignore[attr-defined]
    return adapted_provider


def _with_l1_notes(result: ModelResult, *, purpose: object) -> ModelResult:
    """非笔记场景显式填默认 note；无效/缺少其它字段不纠错。"""
    if purpose != "runtime_l1_attempt":
        return result
    try:
        decision = json.loads(result.reply)
    except (TypeError, ValueError):
        return result
    if not isinstance(decision, dict) or "note" in decision:
        return result
    decision["note"] = "测试公开笔记：执行本次选择的动作。"
    return replace(result, reply=json.dumps(decision, ensure_ascii=False))


def repair_feedback_from_provider_kwargs(
    kwargs: dict[str, object],
) -> dict[str, object] | None:
    """Decode canonical repair feedback presented to a prepared test provider."""

    repair_messages = kwargs.get("repair_messages")
    if repair_messages is None:
        return None
    if not isinstance(repair_messages, list) or len(repair_messages) != 4:
        raise AssertionError("repair_messages must contain the canonical four messages")
    instruction = repair_messages[3]
    if not isinstance(instruction, dict):
        raise AssertionError("repair instruction must be a message object")
    content = instruction.get("content")
    if not isinstance(content, str) or "Host 修复清单：" not in content:
        raise AssertionError(
            "repair instruction omitted the canonical feedback envelope"
        )
    feedback = json.loads(content.split("Host 修复清单：", 1)[1])
    if not isinstance(feedback, dict):
        raise AssertionError("repair feedback envelope must be a JSON object")
    return feedback


__all__ = [
    "as_prepared_test_provider",
    "prepare_test_model_request",
    "repair_feedback_from_provider_kwargs",
]
