"""结构化 Provider 请求的准备边界。

结构化 Runtime 调用必须经过 Provider ``prepare`` 准入后才能分派。缺少该
能力的 Provider 在组合时直接失败，不再通过 ``None`` 隐式切换到另一条请求路径。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from .output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssueCoverage,
    RuntimeModelStructuredPrompt,
)
from .structured_calls import build_structured_repair_messages


_REPAIR_USER_TEMPLATE = (
    "上一份输出未通过校验。修正下列问题，返回符合原请求的完整 JSON；"
    "不要返回 patch、Markdown 或修复清单。\n"
    "{coverage_note}Host 修复清单：{canonical_feedback}"
)


def durable_structured_provider_prompt(
    durable_call: object | None,
    *,
    system_prompt: str,
    user_content: str,
) -> tuple[str, str]:
    """返回持久结构化调用所冻结的精确前两条消息。

    每个持久逻辑请求必须携带精确提示词快照。没有持久权威信息的调用继续使用当前渲染器。
    """

    if durable_call is None:
        return system_prompt, user_content
    logical_request = getattr(durable_call, "logical_request", None)
    snapshot = getattr(logical_request, "structured_prompt", None)
    if not isinstance(snapshot, RuntimeModelStructuredPrompt):
        terminal_state_error = getattr(
            durable_call,
            "terminal_state_error",
            None,
        )
        if callable(terminal_state_error):
            raise terminal_state_error(
                "durable structured prompt has the wrong contract"
            )
        raise TypeError("durable structured prompt is missing or invalid")
    return snapshot.system_prompt, snapshot.user_content


def prepare_structured_request(
    provider: object,
    *,
    system_prompt: str,
    user_content: str,
    purpose: str,
    prepare_kwargs: Callable[[], dict[str, object]] | None = None,
) -> Callable[[], object]:
    """构造交给 request_model_with_retry 的零参数 prepare 工厂，本次构造不发送请求。

    每个物理 attempt 调用工厂时才取 prepare_kwargs 并执行 provider.prepare，
    得到已准入但未 dispatch 的对象。purpose / repair_messages 不允许被额外参数覆盖，
    避免初始请求与修复请求在组合层串用身份或消息形状。
    """

    prepare = getattr(provider, "prepare", None)
    if not callable(prepare):
        raise TypeError("structured provider must expose prepare")

    def prepare_request() -> object:
        extra_kwargs = {} if prepare_kwargs is None else prepare_kwargs()
        if not isinstance(extra_kwargs, dict):
            raise TypeError("prepare_kwargs must return a dict")
        if set(extra_kwargs) & {"purpose", "repair_messages"}:
            raise ValueError(
                "prepare_kwargs cannot replace purpose or repair_messages"
            )
        return prepare(
            system_prompt,
            user_content,
            purpose=purpose,
            **extra_kwargs,
        )

    return prepare_request


def prepare_structured_repair_request(
    provider: object,
    *,
    system_prompt: str,
    user_content: str,
    purpose: str,
    prepare_kwargs: Callable[[], dict[str, object]] | None = None,
) -> Callable[[RuntimeModelOutputRepairFeedback, str], object]:
    """返回四消息整段响应修复准备器。

    回调接收持久修复信封及单独恢复的被拒响应。它先通过 SHA-256 绑定两者，再构造精确四消息对话。
    调用 Provider 的普通 ``prepare`` 入口，可让完整修复正文再次通过上下文准入。
    模型只看到问题路径、原因和必要定位；版本、hash、物理序号和诊断分类留在 Host 信封。
    """

    prepare = getattr(provider, "prepare", None)
    if not callable(prepare):
        raise TypeError("structured repair provider must expose prepare")

    def prepare_repair(
        feedback: RuntimeModelOutputRepairFeedback,
        rejected_response_text: str,
    ) -> object:
        if not isinstance(feedback, RuntimeModelOutputRepairFeedback):
            raise TypeError("feedback must use the output-repair contract")
        if not isinstance(rejected_response_text, str):
            raise TypeError("rejected response must be text")
        response_sha256 = hashlib.sha256(
            rejected_response_text.encode("utf-8")
        ).hexdigest()
        if response_sha256 != feedback.rejected_response_sha256:
            raise ValueError(
                "rejected response SHA-256 does not match the repair envelope"
            )

        displayed_feedback = {
            "current_issues": [
                {
                    "paths": list(issue.paths),
                    "safe_explanation": issue.safe_explanation,
                    **(
                        {"json_line": issue.json_line, "json_column": issue.json_column}
                        if issue.json_line is not None else {}
                    ),
                }
                for issue in feedback.current_issues
            ],
        }
        canonical_feedback = json.dumps(
            displayed_feedback,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        repair_user_content = _REPAIR_USER_TEMPLATE.format(
            coverage_note=(
                "" if feedback.issue_coverage is RuntimeModelOutputRepairIssueCoverage.COMPLETE
                else "当前清单可能不完整，请同时检查其余字段。\n"
            ),
            canonical_feedback=canonical_feedback,
        )
        repair_messages = build_structured_repair_messages(
            system_prompt,
            user_content,
            rejected_response_text=rejected_response_text,
            repair_user_content=repair_user_content,
        )
        extra_kwargs = {} if prepare_kwargs is None else prepare_kwargs()
        if not isinstance(extra_kwargs, dict):
            raise TypeError("prepare_kwargs must return a dict")
        if set(extra_kwargs) & {"purpose", "repair_messages"}:
            raise ValueError(
                "prepare_kwargs cannot replace purpose or repair_messages"
            )
        return prepare(
            system_prompt,
            user_content,
            purpose=purpose,
            repair_messages=repair_messages,
            **extra_kwargs,
        )

    return prepare_repair


__all__ = [
    "durable_structured_provider_prompt",
    "prepare_structured_request",
    "prepare_structured_repair_request",
]
