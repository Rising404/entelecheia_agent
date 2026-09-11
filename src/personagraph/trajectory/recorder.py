"""将一次网关调用转换为一条记录步骤。

该模块位于模型网关旁边而非内部：网关的职责是获得答案，记录绝不能妨碍它。
Recorder 自身失败会降级为最小轨迹标记或结构化进程日志，不会静默消失。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .contracts import (
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
    text_blob,
)
from .scope import current_turn_linkage, resolve_turn_linkage


RECORDING_ENV = "PERSONAGRAPH_TRAJECTORY"
_LOG = logging.getLogger(__name__)


class _TrajectoryStoreWriteError(RuntimeError):
    """区分投影失败与承载该投影的存储自身失败。"""


def _persist_step(step: Step, store: Any | None) -> None:
    from .store import active_store

    try:
        (store or active_store()).record(step)
    except Exception as exc:
        raise _TrajectoryStoreWriteError from exc


def _record_recording_failure(
    *,
    operation: str,
    error: BaseException,
    session_id: str | None,
    turn_id: str | None,
    model_call_id: str | None = None,
    store: Any | None,
) -> None:
    """非递归地报告一次 Recorder 失败。

    投影或序列化失败时，使用同一 Store 写入一个最小标记。若失败点正是 Store，
    不能再次调用它来证明自己失败，只写结构化进程日志。确定性 ID 将同一 Turn、
    同一入口的重复故障压成一条，避免观测故障形成记录风暴。
    """

    scoped = current_turn_linkage()
    resolved_session_id = scoped.session_id if scoped is not None else session_id
    resolved_turn_id = scoped.turn_id if scoped is not None else turn_id
    has_recording_scope = store is not None or resolved_turn_id is not None
    if not has_recording_scope:
        # 独立模型调用和单元测试可以不绑定 Session；这不是生产 Turn 的轨迹丢失。
        return

    root_error = error.__cause__ if error.__cause__ is not None else error
    error_type = type(root_error).__name__
    _LOG.error(
        "trajectory recording failed operation=%s session_id=%s turn_id=%s error_type=%s",
        operation,
        resolved_session_id,
        resolved_turn_id,
        error_type,
    )
    if isinstance(error, _TrajectoryStoreWriteError):
        return

    identity = "\x1f".join(
        (
            resolved_session_id or "unscoped",
            resolved_turn_id or "unscoped",
            operation,
        )
    )
    step_id = f"trf_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    try:
        _persist_step(
            Step(
                step_id=step_id,
                kind=StepKind.RECORDING_FAILURE,
                occurred_at=datetime.now(UTC).isoformat(),
                session_id=resolved_session_id,
                turn_id=resolved_turn_id,
                model_call_id=model_call_id,
                purpose=operation,
                outcome=StepOutcome.FAILED,
                reason_code=f"trajectory_recording_failed:{error_type}",
            ),
            store,
        )
    except Exception as marker_error:
        marker_root = (
            marker_error.__cause__
            if marker_error.__cause__ is not None
            else marker_error
        )
        _LOG.error(
            "trajectory failure marker could not be persisted operation=%s "
            "session_id=%s turn_id=%s error_type=%s",
            operation,
            resolved_session_id,
            resolved_turn_id,
            type(marker_root).__name__,
        )


def recording_enabled() -> bool:
    """除非关闭，否则保持开启。

    默认开启，因为其价值完全在于出错时数据已经存在；需要事先记得开启的记录器，
    往往恰在需要时处于关闭状态。
    """

    return (os.getenv(RECORDING_ENV, "on") or "on").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
    }


def record_model_call(
    *,
    model_call_id: str,
    purpose: str | None,
    provider: str,
    model: str,
    payload: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    reply: str,
    duration_ms: int,
    session_id: str | None = None,
    turn_id: str | None = None,
    store: Any | None = None,
) -> None:
    """记录一次已返回的网关调用；观测故障不改变模型调用结果。

    接收 Provider adapter 已投影的 payload / response / reply；结构化 reply/thinking
    可能只是 hash 标记。这里的 outcome=OK 表示网关返回，后续 Runtime 合同校验仍
    可能另记 rejected。记录失败交给 _record_recording_failure 写最小标记或进程日志。
    """

    if not recording_enabled():
        return
    try:
        _record_model_call(
            model_call_id=model_call_id,
            purpose=purpose,
            provider=provider,
            model=model,
            payload=payload,
            response=response,
            reply=reply,
            duration_ms=duration_ms,
            session_id=session_id,
            turn_id=turn_id,
            store=store,
        )
    except Exception as exc:
        _record_recording_failure(
            operation="record_model_call",
            error=exc,
            session_id=session_id,
            turn_id=turn_id,
            model_call_id=model_call_id,
            store=store,
        )
        # 观测失败仍不得把已经成功的模型调用变成失败。
        return


def _record_model_call(
    *,
    model_call_id: str,
    purpose: str | None,
    provider: str,
    model: str,
    payload: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    reply: str,
    duration_ms: int,
    session_id: str | None,
    turn_id: str | None,
    store: Any | None,
) -> None:
    """将网关记录视图拆为有序 message Parts、用量指标和一个 MODEL_CALL Step。

    不保存完整 HTTP envelope；缺失 token 指标不补零。reply/thinking 是否保留正文
    已由上游 projection 决定，此处只把收到的文本转成内容寻址 blob。
    相同 model_call_id 下再按 Parts 内容生成 step_id，供 Store 去重。
    """

    session_id, turn_id = resolve_turn_linkage(
        session_id=session_id,
        turn_id=turn_id,
    )

    blocks = list((response or {}).get("content") or [])
    thinking = "\n".join(
        str(block.get("thinking") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "thinking"
    ).strip()

    parts: list[Part] = []
    system = payload.get("system")
    if isinstance(system, str) and system:
        parts.append(Part(PartRole.SYSTEM, text_blob(_strip_evidence_bodies(system))))
    for message in payload.get("messages") or []:
        if isinstance(message, Mapping):
            parts.extend(_message_parts(message))
    if thinking:
        parts.append(Part(PartRole.THINKING, text_blob(thinking)))
    if reply:
        parts.append(Part(PartRole.ASSISTANT, text_blob(reply)))

    usage = (response or {}).get("usage") or {}
    metrics = {
        name: int(usage[key])
        for name, key in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_tokens", "cache_read_input_tokens"),
            ("cache_write_tokens", "cache_creation_input_tokens"),
        )
        if isinstance(usage.get(key), int)
    }
    if thinking:
        metrics["thinking_bytes"] = len(thinking.encode("utf-8"))

    _persist_step(
        Step(
            step_id=_attempt_step_id(model_call_id, parts),
            kind=StepKind.MODEL_CALL,
            occurred_at=datetime.now(UTC).isoformat(),
            parts=tuple(parts),
            session_id=session_id,
            turn_id=turn_id,
            model_call_id=model_call_id,
            purpose=f"{provider}:{model}:{purpose or 'unknown'}",
            duration_ms=duration_ms,
            outcome=StepOutcome.OK,
            metrics=metrics,
        ),
        store,
    )


def record_model_request_failure(
    *,
    model_call_id: str,
    purpose: str,
    reason_code: str,
    attempts: int,
    duration_ms: int,
    session_id: str | None = None,
    turn_id: str | None = None,
    store: Any | None = None,
) -> None:
    """记录有界请求最终失败的汇总 Step，不代表一次额外的 Provider 调用。

    没有 prompt/response Parts，只有安全 reason_code、累计耗时和 attempts 计数。
    requests 的正常重试失败出口调用它；未经过该出口的异常不会自动出现在这里。
    因此 MODEL_CALL 行数混有成功返回、拒绝诊断和终态摘要，不能直接当作调用次数。
    """

    if not recording_enabled():
        return
    try:
        session_id, turn_id = resolve_turn_linkage(
            session_id=session_id,
            turn_id=turn_id,
        )
        identity = "\x1f".join(
            (turn_id or "unscoped", model_call_id, purpose, reason_code)
        )
        _persist_step(
            Step(
                step_id=(
                    "mcf_"
                    f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
                ),
                kind=StepKind.MODEL_CALL,
                occurred_at=datetime.now(UTC).isoformat(),
                session_id=session_id,
                turn_id=turn_id,
                model_call_id=model_call_id,
                purpose=purpose,
                duration_ms=max(0, int(duration_ms)),
                outcome=StepOutcome.FAILED,
                reason_code=reason_code,
                metrics={"attempts": max(1, int(attempts))},
            ),
            store,
        )
    except Exception as exc:
        _record_recording_failure(
            operation="record_model_request_failure",
            error=exc,
            session_id=session_id,
            turn_id=turn_id,
            model_call_id=model_call_id,
            store=store,
        )


def _attempt_step_id(model_call_id: str, parts: Sequence[Part]) -> str:
    """按 model_call_id 与有序 Parts 的内容身份派生去重键。

    同一调用 ID 下不同请求/响应内容可保留为不同记录；完全相同的 ID 和 Parts
    会合并。它不编码物理 ordinal，因此不能保证一条 Step 对应一次真实 HTTP 请求。
    """

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.role.value.encode("utf-8"))
        digest.update(part.blob.sha256.encode("utf-8"))
    return f"mc_{model_call_id}_{digest.hexdigest()[:12]}"


# 检索证据由 retrieval/prompt_context 包装在消息文本内送达模型。匹配该形态
# 确实形成耦合；其可接受原因及固定方式见 _strip_evidence_bodies。
_EVIDENCE_BLOCK = re.compile(
    r"(<retrieved_item\b[^>]*>)\n(.*?)\n(</retrieved_item>)", re.DOTALL
)


def _strip_evidence_bodies(text: str) -> str:
    """用大小说明替换检索证据正文。

    正文是文档存储中已有分块的逐字副本，可通过标签自身 ``citation`` 属性中仍然
    存在的 ID 寻址。保留第二份副本会占据轨迹主体：一个分块可达数十 KB，而回复
    只有数百字节；同时还会创建可能与权威不一致的副本。

    只移除正文，标签及其属性保留，因此步骤仍精确记录展示了哪些证据及其顺序。

    这里匹配 ``retrieval/prompt_context`` 拥有的格式。该处变化会静默阻止剥离，
    让文件再次增长；因此相关测试会提供该模块的真实输出，而非手写仿制品。
    """

    def _replace(match: re.Match[str]) -> str:
        body_bytes = len(match.group(2).encode("utf-8"))
        return (
            f"{match.group(1)}\n"
            f"«正文 {body_bytes}B 未记录：见本标签 citation 中的来源 id»\n"
            f"{match.group(3)}"
        )

    return _EVIDENCE_BLOCK.sub(_replace, text)


    # 一种协议把系统提示放在顶层字段，另一种放在首条消息。若只读取第一种形态，
    # 会把系统提示记录成用户输入；这是轨迹中绝不能与其他内容混淆的部分。
_MESSAGE_ROLES = {
    "system": PartRole.SYSTEM,
    "assistant": PartRole.ASSISTANT,
    "user": PartRole.USER,
}


def _message_parts(message: Mapping[str, Any]) -> list[Part]:
    """只投影消息文本：支持字符串和 text blocks，其余 Provider block 不猜测转换。

    图片读取事实在视觉工具轨迹 / Vision ledger；这里只处理特定 retrieved_item
    标签的证据正文省略，不递归清洗任意 JSON 文本内的证据字段。
    """

    role = _MESSAGE_ROLES.get(str(message.get("role") or ""), PartRole.USER)
    content = message.get("content")
    if isinstance(content, str):
        stripped = _strip_evidence_bodies(content)
        return [Part(role, text_blob(stripped))] if stripped else []
    if not isinstance(content, Sequence):
        return []

    parts: list[Part] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        # 模型消息轨迹只投影自身能够如实持有的文本。图片使用事实由
        # read_file_visuals 的工具轨迹与专用 Vision ledger 持有；不得从
        # provider wire 的 base64/image_url 形态伪造来源引用。
        if block.get("type") == "text":
            text = _strip_evidence_bodies(str(block.get("text") or ""))
            if text:
                parts.append(Part(role, text_blob(text)))
    return parts


def record_tool_call(
    *,
    tool_id: str,
    arguments: Mapping[str, Any],
    status: str,
    result: Mapping[str, Any] | None,
    error_code: str | None,
    duration_ms: int,
    error: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    step_id: str | None = None,
    store: Any | None = None,
) -> None:
    """记录一次工具调用，包括被拒绝的调用。

    此处拒绝比成功更重要。事件流已经说明工具调用失败并携带错误代码，却无法说明
    实际请求了什么，而这正是解释原因的唯一信息。绝不抛出异常。

    覆盖范围取决于调用点：只有进入此 recorder 的结果才有 TOOL_CALL Step，
    Host 在 ToolExecutor 之前拒绝或专用 dispatcher 旁路时不能据此推断“没有调用”。
    参数保留为 TOOL_ARGUMENTS，结果经工具专用投影；持久执行状态仍以 ToolCall 为准。
    调用方可传入安全的结构化 error，将完整约束与结果一并保存在 TOOL_RESULT；
    未传 error 时保持既有结果形状。稳定 step_id 可让持久结果重放不重复记账。
    """

    if not recording_enabled():
        return
    try:
        session_id, turn_id = resolve_turn_linkage(
            session_id=session_id,
            turn_id=turn_id,
        )

        parts = [Part(PartRole.TOOL_ARGUMENTS, text_blob(_as_text(arguments)))]
        tool_result = (
            _tool_result_for_trajectory(tool_id, result) if result is not None else None
        )
        if error is not None:
            tool_result = {"result": tool_result, "error": dict(error)}
        if tool_result is not None:
            parts.append(
                Part(
                    PartRole.TOOL_RESULT,
                    text_blob(_as_text(tool_result)),
                )
            )
        _persist_step(
            Step(
                step_id=step_id or f"tc_{uuid.uuid4()}",
                kind=StepKind.TOOL_CALL,
                occurred_at=datetime.now(UTC).isoformat(),
                parts=tuple(parts),
                session_id=session_id,
                turn_id=turn_id,
                purpose=tool_id,
                duration_ms=duration_ms,
                outcome=_tool_outcome(status),
                reason_code=error_code or (None if status == "succeeded" else status),
            ),
            store,
        )
    except Exception as exc:
        _record_recording_failure(
            operation="record_tool_call",
            error=exc,
            session_id=session_id,
            turn_id=turn_id,
            store=store,
        )
        return


_RETRIEVAL_TOOL_IDS = frozenset(
    {"retrieve_files", "retrieve_history"}
)
_RETRIEVAL_BODY_FIELDS = frozenset(
    {"content", "locator", "preview", "snippet", "text"}
)


def _tool_result_for_trajectory(
    tool_id: str,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """对 retrieve_files / retrieve_history 的 evidence 条目省略正文和 locator 等字段。

    其他工具结果直接保留；这里不是通用递归脱敏器。只修改浅复制后的 evidence
    投影，不改变工具实际返回给 Runtime 的结果。
    """

    if tool_id not in _RETRIEVAL_TOOL_IDS:
        return result
    projected = dict(result)
    evidence = result.get("evidence")
    if isinstance(evidence, (list, tuple)):
        projected["evidence"] = [
            {
                key: value
                for key, value in item.items()
                if key not in _RETRIEVAL_BODY_FIELDS
            }
            for item in evidence
            if isinstance(item, Mapping)
        ]
    return projected


def record_retrieval(
    *,
    queries: Sequence[str],
    source_scope: str,
    evidence: Sequence[Mapping[str, Any]],
    method_outcomes: Sequence[Mapping[str, Any]] = (),
    reranker_outcomes: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[Mapping[str, Any]] = (),
    fusion: Mapping[str, Any] | None = None,
    outcome: str,
    reason_code: str | None,
    duration_ms: int,
    session_id: str | None = None,
    turn_id: str | None = None,
    step_id: str | None = None,
    store: Any | None = None,
) -> None:
    """记录一次检索：请求、无正文执行审计与最终引用。

    查询正是重点。生成器提示经过长期调优，却无人能看到它在生产中实际生成什么；
    这里填补了该缺口。调用方只能传入方法、融合、重排和最终引用的安全投影；文档
    正文仍由权威 Source 持有，不在 trajectory 中复制。诊断同样只能是 Host 生成的
    安全分类与无正文身份投影。绝不抛出异常。

    按传入序列分别落 QUERY / RETRIEVAL_METHOD / FUSION / RERANKER / DIAGNOSTIC /
    EVIDENCE Parts。无正文承诺依赖调用方提供安全投影；本函数不会重新运行检索或
    递归删除任意传入字典里的正文。文件检索应在最终权限/来源投影之后调用 audit。
    """

    if not recording_enabled():
        return
    try:
        session_id, turn_id = resolve_turn_linkage(
            session_id=session_id,
            turn_id=turn_id,
        )

        parts = [Part(PartRole.QUERY, text_blob(query)) for query in queries if query]
        for item in method_outcomes:
            parts.append(
                Part(PartRole.RETRIEVAL_METHOD, text_blob(_as_text(item)))
            )
        if fusion is not None:
            parts.append(Part(PartRole.FUSION, text_blob(_as_text(fusion))))
        for item in reranker_outcomes:
            parts.append(Part(PartRole.RERANKER, text_blob(_as_text(item))))
        for item in diagnostics:
            parts.append(
                Part(PartRole.RETRIEVAL_DIAGNOSTIC, text_blob(_as_text(item)))
            )
        for item in evidence:
            parts.append(Part(PartRole.EVIDENCE, text_blob(_as_text(item))))
        _persist_step(
            Step(
                step_id=step_id or f"rt_{uuid.uuid4()}",
                kind=StepKind.RETRIEVAL,
                occurred_at=datetime.now(UTC).isoformat(),
                parts=tuple(parts),
                session_id=session_id,
                turn_id=turn_id,
                purpose=source_scope,
                duration_ms=duration_ms,
                outcome=_retrieval_outcome(outcome),
                reason_code=reason_code
                or (
                    None
                    if outcome in {"ok", "complete", "matched"}
                    else outcome
                ),
                metrics={
                    "queries": len(queries),
                    "items": len(evidence),
                    **(
                        {"method_runs": len(method_outcomes)}
                        if method_outcomes
                        else {}
                    ),
                    **(
                        {"reranker_runs": len(reranker_outcomes)}
                        if reranker_outcomes
                        else {}
                    ),
                    **(
                        {"diagnostics": len(diagnostics)}
                        if diagnostics
                        else {}
                    ),
                },
            ),
            store,
        )
    except Exception as exc:
        _record_recording_failure(
            operation="record_retrieval",
            error=exc,
            session_id=session_id,
            turn_id=turn_id,
            store=store,
        )
        return


def record_rejected_output(
    *,
    stage: str,
    rejected: str,
    reason_code: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    model_call_id: str | None = None,
    step_id: str | None = None,
    store: Any | None = None,
) -> None:
    """记录调用守卫提供的被拒输出诊断。

    旧调用方可以提供有界的被拒片段。运行时结构化输出修复则提供安全的主机生成
    JSON 投影（哈希、契约、问题代码与路径），使精确被拒响应只保留一个持久所有者，
    不会复制到轨迹存储。绝不抛出异常。

    kind 仍为 MODEL_CALL，outcome 为 REJECTED；这是一条验证诊断，不能重复计为
    新的模型请求。rejected 的安全性由调用守卫投影负责，本函数按传入文本记录。
    """

    if not recording_enabled():
        return
    try:
        session_id, turn_id = resolve_turn_linkage(
            session_id=session_id,
            turn_id=turn_id,
        )

        _persist_step(
            Step(
                step_id=step_id or f"rj_{uuid.uuid4()}",
                kind=StepKind.MODEL_CALL,
                occurred_at=datetime.now(UTC).isoformat(),
                parts=(Part(PartRole.REJECTED_OUTPUT, text_blob(rejected)),),
                session_id=session_id,
                turn_id=turn_id,
                model_call_id=model_call_id,
                purpose=stage,
                outcome=StepOutcome.REJECTED,
                reason_code=reason_code,
            ),
            store,
        )
    except Exception as exc:
        _record_recording_failure(
            operation="record_rejected_output",
            error=exc,
            session_id=session_id,
            turn_id=turn_id,
            model_call_id=model_call_id,
            store=store,
        )
        return


def _tool_outcome(status: str) -> StepOutcome:
    if status == "succeeded":
        return StepOutcome.OK
    if status == "rejected":
        return StepOutcome.REJECTED
    return StepOutcome.FAILED


def _retrieval_outcome(outcome: str) -> StepOutcome:
    if outcome in {"ok", "complete", "partial", "matched", "no_match"}:
        return StepOutcome.OK
    return StepOutcome.FAILED


def _as_text(value: Any) -> str:
    """呈现结构化载荷，同时不让异常对象破坏记录。"""

    try:
        # MappingProxyType is a JSON object, not a display string. Let the JSON
        # encoder recurse; malformed/circular inputs stay inside this log guard.
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            default=lambda item: dict(item) if isinstance(item, Mapping) else str(item),
        )
    except Exception:
        return repr(value)


def new_model_call_id() -> str:
    return str(uuid.uuid4())
