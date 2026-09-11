"""模型可见检索工具与统一 Retrieval service 之间的数据平面 facade。

本模块只负责校验组合、选择 File/History 执行器，并确保 trajectory 审计只记录一次。
来源范围校验、证据投影和实际查询分别由同包内的窄模块持有。
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from ...contracts import CorpusKey
from ...execution import checkpoint, current_execution
from ...ports import RetrievalCancelled
from ..contracts import (
    RetrievalCorpus,
    RetrievalToolPort,
    RetrievalToolRequest,
    RetrievalToolResult,
)
from .audit import (
    _DeferredQueryAudit,
    _QueryAudit,
    _record_query_audit,
    _safe_retrieval_exception_code,
)
from .file import retrieve_file_corpus
from .history import retrieve_history
from .ports import FileRetrievalReadinessPort, RetrievalFoundationReadPort
from .projection import _blocked, _require_foundation


MAX_FILE_SERVICE_CALLS = 64


@dataclass(frozen=True, slots=True)
class RetrievalServiceToolPort(RetrievalToolPort):
    """将冻结工具读取映射到独立 File/History 基础。"""

    file_foundation: RetrievalFoundationReadPort | None = None
    history_foundation: RetrievalFoundationReadPort | None = None
    file_readiness: FileRetrievalReadinessPort | None = None
    file_readonly: bool = False
    trajectory_turn_id: str | None = None
    max_file_service_calls: int = MAX_FILE_SERVICE_CALLS

    def __post_init__(self) -> None:
        if self.file_foundation is not None:
            _require_foundation(self.file_foundation, CorpusKey.FILE)
        if self.history_foundation is not None:
            _require_foundation(self.history_foundation, CorpusKey.HISTORY)
        if not isinstance(self.file_readonly, bool):
            raise TypeError("file_readonly must be a bool")
        if self.trajectory_turn_id is not None and (
            not isinstance(self.trajectory_turn_id, str)
            or not self.trajectory_turn_id.strip()
        ):
            raise ValueError("trajectory_turn_id must be non-empty when provided")
        if (
            self.file_readonly
            and self.file_foundation is not None
            and not callable(
                getattr(
                    getattr(self.file_foundation, "service", None),
                    "retrieve_context_readonly",
                    None,
                )
            )
        ):
            raise TypeError(
                "file_readonly requires a service with retrieve_context_readonly"
            )
        if (
            isinstance(self.max_file_service_calls, bool)
            or not isinstance(self.max_file_service_calls, int)
            or not 1 <= self.max_file_service_calls <= MAX_FILE_SERVICE_CALLS
        ):
            raise ValueError(
                f"max_file_service_calls must be from 1 to {MAX_FILE_SERVICE_CALLS}"
            )

    def retrieve(
        self,
        request: RetrievalToolRequest,
    ) -> RetrievalToolResult:
        if not isinstance(request, RetrievalToolRequest):
            raise TypeError("request must be RetrievalToolRequest")
        if request.corpus is RetrievalCorpus.FILES:
            if not self.file_readonly:
                return retrieve_file_corpus(
                    request,
                    foundation=self.file_foundation,
                    file_readiness=self.file_readiness,
                    file_readonly=self.file_readonly,
                    max_file_service_calls=self.max_file_service_calls,
                )
            result, deferred_audit = self.retrieve_file_readonly_unrecorded(request)
            deferred_audit.record()
            return result
        result, deferred_audit = self.retrieve_history_unrecorded(request)
        deferred_audit.record()
        return result

    def retrieve_file_readonly_unrecorded(
        self,
        request: RetrievalToolRequest,
    ) -> tuple[RetrievalToolResult, "_DeferredQueryAudit"]:
        """执行一次只读 File 查询，并把唯一落盘时机交给最外层投影。

        普通调用方仍使用 :meth:`retrieve`，其行为不变。File 检索需要等当前 chunk
        authority 校验完成后才能准确说明哪些证据实际进入模型上下文，因此只在该内部
        组合边界延迟记录。
        """

        if not isinstance(request, RetrievalToolRequest):
            raise TypeError("request must be RetrievalToolRequest")
        if request.corpus is not RetrievalCorpus.FILES or not self.file_readonly:
            raise ValueError(
                "deferred audit is only available for readonly File queries"
            )
        started = time.monotonic()
        audit = _QueryAudit(execution=current_execution())
        try:
            result = retrieve_file_corpus(
                request,
                foundation=self.file_foundation,
                file_readiness=self.file_readiness,
                file_readonly=self.file_readonly,
                max_file_service_calls=self.max_file_service_calls,
                audit=audit,
            )
            checkpoint()
        except RetrievalCancelled:
            def record_cancelled(_status: str) -> None:
                try:
                    _record_query_audit(
                        request=request,
                        result=_blocked(request, "retrieval_cancelled"),
                        audit=audit,
                        turn_id=self.trajectory_turn_id,
                        duration_ms=max(0, int((time.monotonic() - started) * 1_000)),
                        final_projection={
                            "status": "blocked", "evidence": [],
                            "gaps": [{"code": "retrieval_cancelled", "blocking": True}],
                        },
                        diagnostics=({
                            "stage": "dataplane", "code": "retrieval_cancelled",
                            "status": "cancelled", "blocking": True, "known_count": 1,
                        },),
                    )
                except Exception:
                    pass

            if audit.execution is None:
                record_cancelled("cancelled")
            else:
                audit.execution.control.on_settled(record_cancelled)
            raise
        except Exception as exc:
            duration_ms = max(0, int((time.monotonic() - started) * 1_000))
            failure_result = _blocked(request, "retrieval_backend_unavailable")
            try:
                _record_query_audit(
                    request=request,
                    result=failure_result,
                    audit=audit,
                    turn_id=self.trajectory_turn_id,
                    duration_ms=duration_ms,
                    diagnostics=(
                        {
                            "stage": "dataplane",
                            "code": _safe_retrieval_exception_code(exc),
                            "status": "failed",
                            "blocking": True,
                            "known_count": 1,
                        },
                    ),
                )
            except Exception:
                pass
            raise
        return result, _DeferredQueryAudit(
            request=request,
            result=result,
            audit=audit,
            turn_id=self.trajectory_turn_id,
            duration_ms=max(0, int((time.monotonic() - started) * 1_000)),
        )

    def retrieve_history_unrecorded(
        self,
        request: RetrievalToolRequest,
    ) -> tuple[RetrievalToolResult, "_DeferredQueryAudit"]:
        """执行 History 查询，把唯一记录时机交给最外层公共投影。"""

        if not isinstance(request, RetrievalToolRequest):
            raise TypeError("request must be RetrievalToolRequest")
        if request.corpus is not RetrievalCorpus.HISTORY:
            raise ValueError("deferred History audit requires a History request")
        started = time.monotonic()
        audit = _QueryAudit()
        try:
            result = retrieve_history(
                request,
                foundation=self.history_foundation,
                audit=audit,
            )
        except Exception as exc:
            duration_ms = max(0, int((time.monotonic() - started) * 1_000))
            failure_result = _blocked(request, "retrieval_backend_unavailable")
            try:
                _record_query_audit(
                    request=request,
                    result=failure_result,
                    audit=audit,
                    turn_id=self.trajectory_turn_id,
                    duration_ms=duration_ms,
                    diagnostics=(
                        {
                            "stage": "dataplane",
                            "code": _safe_retrieval_exception_code(exc),
                            "status": "failed",
                            "blocking": True,
                            "known_count": 1,
                        },
                    ),
                )
            except Exception:
                pass
            raise
        return result, _DeferredQueryAudit(
            request=request,
            result=result,
            audit=audit,
            turn_id=self.trajectory_turn_id,
            duration_ms=max(0, int((time.monotonic() - started) * 1_000)),
        )


__all__ = [
    "FileRetrievalReadinessPort",
    "MAX_FILE_SERVICE_CALLS",
    "RetrievalFoundationReadPort",
    "RetrievalServiceToolPort",
]
