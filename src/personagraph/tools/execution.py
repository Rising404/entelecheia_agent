"""Tool Platform 的单次调用执行器。

它会验证输入和输出，在处理器模式允许时强制执行期限，并返回结构化结果。
它特意不包含 Operation ID、重试循环、回执存储或批准副作用。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..trajectory import record_tool_call
from .contracts import ExecutionOutcome, ExecutionStatus, ToolError
from .effects import EffectAction
from .execution_context import (
    ToolExecutionContext,
    ToolInvocationCancelled,
    tool_execution_scope,
)
from .registration import ExecutableToolRegistration, ExecutionMode
from .schema_validation import SchemaValidationError, ToolSchemaCompiler


Clock = Callable[[], float]
_NON_MODIFYING_ACTIONS = frozenset({EffectAction.READ, EffectAction.SEARCH})
_KNOWN_RETRYABLE_FAILED_CODES = frozenset(
    {
        "tool_exception",
        "invalid_tool_output",
        "read_only_response_unavailable",
        "tool_output_too_large",
    }
)


class ToolBusinessFailure(RuntimeError):
    """透明重试不得再次执行的已知终态处理器失败。

    只有在再次物理调用无法改变结果时，处理器才使用它（例如持久损坏的本地数据文件）。
    传输、超时及其他可能为瞬态的异常仍属于普通 ``tool_exception`` 结果，
    并保留有界重试策略。
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.error = ToolError(code, message, details or {})
        super().__init__(message)


def is_known_retryable_technical_failure(outcome: ExecutionOutcome) -> bool:
    """返回一个终态结果能否透明重试。

    ``TIMED_OUT`` 已是 Tool Platform 对完成状态已知的证明；结果不确定的有副作用执行使用
    ``COMPLETION_UNCONFIRMED``。其他失败只有命中此封闭技术代码允许列表时才符合条件，
    从而将拒绝、取消及业务失败排除在重试循环之外。
    """

    if outcome.status is ExecutionStatus.TIMED_OUT:
        return True
    return (
        outcome.status is ExecutionStatus.FAILED
        and outcome.error is not None
        and outcome.error.code in _KNOWN_RETRYABLE_FAILED_CODES
    )


@dataclass(frozen=True)
class ResolvedInvocation:
    registration: ExecutableToolRegistration
    arguments: Mapping[str, Any]
    deadline_monotonic: float | None = None
    cancellation_event: threading.Event | None = None
    logical_tool_call_id: str | None = None
    continuation_check: Callable[[], bool] | None = None


class ToolExecutor:
    def __init__(self, *, schemas: ToolSchemaCompiler | None = None, clock: Clock = time.monotonic) -> None:
        self._schemas = schemas or ToolSchemaCompiler()
        self._clock = clock

    def execute(self, invocation: ResolvedInvocation) -> ExecutionOutcome:
        """运行一个工具，并记录请求内容及返回结果。

        记录逻辑采用外层包装，而不是贯穿内部传递，因为最重要的路径是提前返回——已取消、
        期限已过、参数被拒。这些恰恰是仅检测成功路径会漏掉的情况，也正是参数能够解释结果的情况。
        """

        started = self._clock()
        outcome = self._execute(invocation)
        # Immutable tool values are not directly JSON serializable; the same wire
        # projection used by consumers prevents repr strings in trajectory.
        recorded = outcome.to_dict()
        record_tool_call(
            tool_id=invocation.registration.tool_id,
            arguments=invocation.arguments,
            status=outcome.status.value,
            result=recorded["result"],
            error_code=outcome.error.code if outcome.error else None,
            error=recorded["error"],
            duration_ms=max(0, int((self._clock() - started) * 1000)),
        )
        return outcome

    def _execute(self, invocation: ResolvedInvocation) -> ExecutionOutcome:
        """单次 handler 执行：取消/deadline/schema 前置检查 → 同步或异步分派 → 输出校验。

        这里不做 L1 policy、durable reservation 或自动重试；调用方须先完成这些准入。
        handler 正常返回也要通过 output schema，才能成为 SUCCEEDED 的工具结果。
        """

        registration = invocation.registration
        metadata = {
            "tool_id": registration.tool_id,
            "contract_version": registration.contract_version,
            "implementation_version": registration.implementation_version,
            "execution_mode": registration.execution_profile.execution_mode.value,
        }
        if invocation.cancellation_event and invocation.cancellation_event.is_set():
            return ExecutionOutcome(
                ExecutionStatus.CANCELLED,
                error=ToolError("execution_cancelled", "Tool invocation was cancelled before execution."),
                metadata=metadata,
            )
        timeout_s = self._effective_timeout(registration, invocation.deadline_monotonic)
        if timeout_s is not None and timeout_s <= 0:
            return ExecutionOutcome(
                ExecutionStatus.TIMED_OUT,
                error=ToolError("deadline_exceeded", "Tool deadline elapsed before execution."),
                metadata=metadata,
            )
        try:
            arguments = self._schemas.validate_input(registration.spec.input_schema, invocation.arguments)
        except SchemaValidationError as exc:
            return ExecutionOutcome.rejected(exc.to_tool_error(code="invalid_tool_input"), metadata=metadata)

        effective_deadline = self._clock() + timeout_s if timeout_s is not None else None
        if invocation.deadline_monotonic is not None:
            effective_deadline = (
                invocation.deadline_monotonic if effective_deadline is None
                else min(effective_deadline, invocation.deadline_monotonic)
            )
        control = ToolExecutionContext(
            deadline_monotonic=effective_deadline,
            parent_cancellation_event=invocation.cancellation_event,
            clock=self._clock,
            logical_tool_call_id=invocation.logical_tool_call_id,
            continuation_check=invocation.continuation_check,
        )
        if registration.execution_profile.execution_mode is ExecutionMode.ASYNC or inspect.iscoroutinefunction(registration.handler):
            outcome = self._execute_async(registration, arguments, control, metadata)
        else:
            outcome = self._execute_sync(registration, arguments, control, metadata)
        if outcome.status is ExecutionStatus.SUCCEEDED:
            outcome = self._validate_output(registration, outcome.result or {}, metadata)
            reason = control.interruption_code()
            if reason is not None:
                outcome = self._interrupted_outcome(
                    registration, reason, "Tool deadline or cancellation preceded result settlement.",
                    {**metadata, **control.snapshot()},
                )
        control.settle(outcome.status.value)
        return outcome

    def _effective_timeout(
        self,
        registration: ExecutableToolRegistration,
        deadline: float | None,
    ) -> float | None:
        profile = registration.execution_profile
        candidates = [value for value in (profile.default_timeout_s, profile.hard_timeout_s) if value is not None]
        if deadline is not None:
            candidates.append(deadline - self._clock())
        return min(candidates) if candidates else None

    def _execute_sync(
        self,
        registration: ExecutableToolRegistration,
        arguments: dict[str, Any],
        control: ToolExecutionContext,
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        """在线程中执行同步 handler，并观察取消和剩余时间。

        copy_context 保留 Session / Turn trajectory 归属。Future.cancel 无法强杀已运行
        Python 线程；只读调用可返回 TIMED_OUT / CANCELLED，有修改 effect 的调用返回
        COMPLETION_UNCONFIRMED，不能据此推断副作用已停止或直接重试。
        """

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="personagraph-tool")
        inherited_context = copy_context()

        def run_handler() -> Any:
            with tool_execution_scope(control):
                try:
                    control.checkpoint()
                    result = registration.handler(arguments)
                    control.checkpoint()
                    return result
                finally:
                    control.finish_handler()

        future = pool.submit(inherited_context.run, run_handler)
        try:
            while True:
                reason = control.interruption_code()
                if reason is not None:
                    future.cancel()
                    return self._interrupted_outcome(
                        registration,
                        reason,
                        "Tool invocation stopped accepting results; a running native call may still be finishing.",
                        {**metadata, **control.snapshot()},
                    )
                try:
                    result = future.result(timeout=0.01)
                except FutureTimeoutError:
                    continue
                control.checkpoint()
                return self._success_or_failure(result, metadata)
        except ToolInvocationCancelled as exc:
            return self._interrupted_outcome(
                registration, exc.reason, "Tool handler acknowledged cancellation.",
                {**metadata, **control.snapshot()},
            )
        except ToolBusinessFailure as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=exc.error,
                metadata=metadata,
            )
        except Exception as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_exception", str(exc), {"exception_type": type(exc).__name__}),
                metadata=metadata,
            )
        finally:
            # Future.cancel 只能撤销未开始的工作，不能强杀运行中的 Python / Torch。
            pool.shutdown(wait=False, cancel_futures=True)

    def _execute_async(
        self,
        registration: ExecutableToolRegistration,
        arguments: dict[str, Any],
        control: ToolExecutionContext,
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with tool_execution_scope(control):
                return asyncio.run(self._await_handler(registration, arguments, control, metadata))
        return ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError("executor_event_loop_conflict", "Use the asynchronous adapter from an active event loop."),
            metadata=metadata,
        )

    async def _await_handler(
        self,
        registration: ExecutableToolRegistration,
        arguments: dict[str, Any],
        control: ToolExecutionContext,
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        try:
            control.checkpoint()
            result = registration.handler(arguments)
            if not inspect.isawaitable(result):
                control.checkpoint()
                return self._success_or_failure(result, metadata)
            task = asyncio.create_task(result)
            while not task.done():
                reason = control.interruption_code()
                if reason is not None:
                    task.cancel()
                    return await self._finish_cancelled_task(
                        registration,
                        task,
                        reason,
                        metadata,
                    )
                await asyncio.sleep(0.005)
            control.checkpoint()
            return self._success_or_failure(task.result(), metadata)
        except ToolInvocationCancelled as exc:
            return self._interrupted_outcome(
                registration, exc.reason, "Tool handler acknowledged cancellation.",
                {**metadata, **control.snapshot()},
            )
        except asyncio.CancelledError:
            return ExecutionOutcome(
                ExecutionStatus.CANCELLED,
                error=ToolError("execution_cancelled", "Async tool handler acknowledged cancellation."),
                metadata=metadata,
            )
        except ToolBusinessFailure as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=exc.error,
                metadata=metadata,
            )
        except Exception as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_exception", str(exc), {"exception_type": type(exc).__name__}),
                metadata=metadata,
            )
        finally:
            control.finish_handler()

    async def _finish_cancelled_task(
        self,
        registration: ExecutableToolRegistration,
        task: asyncio.Task[Any],
        code: str,
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        try:
            await task
        except asyncio.CancelledError:
            status = (
                ExecutionStatus.CANCELLED
                if code == "execution_cancelled"
                else ExecutionStatus.TIMED_OUT
            )
            return ExecutionOutcome(
                status,
                error=ToolError(
                    code,
                    "Async tool handler acknowledged cancellation.",
                ),
                metadata=metadata,
            )
        except ToolBusinessFailure as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=exc.error,
                metadata=metadata,
            )
        except Exception as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_exception", str(exc), {"exception_type": type(exc).__name__}),
                metadata=metadata,
            )
        return self._interrupted_outcome(
            registration,
            code,
            "Async handler completed after cancellation was requested.",
            metadata,
        )

    @staticmethod
    def _success_or_failure(result: Any, metadata: dict[str, Any]) -> ExecutionOutcome:
        if not isinstance(result, Mapping):
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("invalid_tool_output", "Tool handler must return a JSON object."),
                metadata=metadata,
            )
        return ExecutionOutcome.succeeded(dict(result), metadata=metadata)

    @staticmethod
    def _interrupted_outcome(
        registration: ExecutableToolRegistration,
        code: str,
        message: str,
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        read_only = all(
            descriptor.action in _NON_MODIFYING_ACTIONS
            for descriptor in registration.effect_profile.effects
        )
        if read_only:
            status = (
                ExecutionStatus.CANCELLED
                if code == "execution_cancelled"
                else ExecutionStatus.TIMED_OUT
            )
            return ExecutionOutcome(
                status,
                error=ToolError(code, message),
                metadata=metadata,
            )
        return ExecutionOutcome(
            ExecutionStatus.COMPLETION_UNCONFIRMED,
            error=ToolError(code, message),
            metadata=metadata,
        )

    def _validate_output(
        self,
        registration: ExecutableToolRegistration,
        result: Mapping[str, Any],
        metadata: dict[str, Any],
    ) -> ExecutionOutcome:
        try:
            normalized = self._schemas.validate_output(registration.spec.output_schema, result)
            encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except SchemaValidationError as exc:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=exc.to_tool_error(code="invalid_tool_output"),
                metadata=metadata,
            )
        except (TypeError, ValueError):
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("invalid_tool_output", "Tool output is not JSON serializable."),
                metadata=metadata,
            )
        if len(encoded) > registration.execution_profile.max_output_bytes:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError(
                    "tool_output_too_large",
                    "Tool output exceeds the registration output bound.",
                    {"max_output_bytes": registration.execution_profile.max_output_bytes, "actual_bytes": len(encoded)},
                ),
                metadata=metadata,
            )
        return ExecutionOutcome.succeeded(normalized, metadata=metadata)
