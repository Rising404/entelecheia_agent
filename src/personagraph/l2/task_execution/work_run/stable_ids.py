"""TaskNode WorkRun 的纯稳定持久 ID 推导。

该计划刻意保持轻量导入：构造标识符不得加载 Session Store、泳道运行器或 Tool Runtime。
唯一的桥接计划方法只会在桥接器实际请求结果值时才导入它们。
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from ..tool_bridge.attempt_contracts import AttemptToolBridgeRequest
    from ..tool_bridge.persistence_contracts import ToolBridgePersistencePlan


class _StableIdContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class WorkRunTurnStableIdPlan(_StableIdContract):
    """每项持久 TaskNode WorkRun 操作的确定性标识。"""

    namespace: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def _reject_blank_namespace(self) -> 'WorkRunTurnStableIdPlan':
        if not self.namespace.strip():
            raise ValueError("stable ID namespace must not be blank")
        return self

    @property
    def work_run_id(self) -> str:
        return self._id("workrun")

    @property
    def create_apply_id(self) -> str:
        return self._id("create")

    def attempt_id(self, ordinal: int) -> str:
        return self._id("attempt", ordinal)

    def start_attempt_apply_id(self, ordinal: int) -> str:
        return self._id("start-attempt", ordinal)

    def resume_active_attempt_apply_id(self, ordinal: int, turn_id: str) -> str:
        """为每次物理 Turn 移交提供稳定且有界的收据标识。"""

        turn_token = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:16]
        return f"{self._id('resume-attempt', ordinal)}:{turn_token}"

    def resume_idle_work_run_apply_id(self, ordinal: int, turn_id: str) -> str:
        """绑定已关闭的只读检查点，并启动 Attempt 序号 + 1。"""

        turn_token = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:16]
        return f"{self._id('resume-idle', ordinal)}:{turn_token}"

    def continue_waiting_user_apply_id(self, question_attempt_ordinal: int) -> str:
        return self._id("continue-waiting-user", question_attempt_ordinal)

    def attempt_action_apply_id(self, ordinal: int) -> str:
        return self._id("attempt-action", ordinal)

    def attempt_model_call_id(self, ordinal: int) -> str:
        return self._id("attempt-model", ordinal)

    def tool_call_id(self, attempt_id: str, ordinal: int) -> str:
        return self._id("tool-call", _attempt_ordinal(attempt_id), ordinal)

    def verification_request_id(self, submitted_attempt_ordinal: int) -> str:
        return self._id("verification", submitted_attempt_ordinal)

    def verification_model_call_id(self, submitted_attempt_ordinal: int) -> str:
        return self._id("verification-model", submitted_attempt_ordinal)

    def verification_prepare_apply_id(self, submitted_attempt_ordinal: int) -> str:
        return self._id("verification-prepare", submitted_attempt_ordinal)

    def verification_commit_apply_id(
        self,
        submitted_attempt_ordinal: int,
        prepared_request_revision: int = 1,
    ) -> str:
        return self._id(
            "verification-commit",
            submitted_attempt_ordinal,
            prepared_request_revision,
        )

    def verification_interrupt_apply_id(
        self,
        submitted_attempt_ordinal: int,
        prepared_request_revision: int = 1,
    ) -> str:
        return self._id(
            "verification-interrupt",
            submitted_attempt_ordinal,
            prepared_request_revision,
        )

    def verification_recovery_apply_id(
        self,
        submitted_attempt_ordinal: int,
        expected_request_revision: int,
        turn_id: str,
    ) -> str:
        turn_token = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:16]
        prefix = self._id(
            "verification-recovery",
            submitted_attempt_ordinal,
            expected_request_revision,
        )
        return f"{prefix}:{turn_token}"

    def delivery_id(self, submitted_attempt_ordinal: int) -> str:
        return self._id("delivery", submitted_attempt_ordinal)

    def tool_bridge_persistence_plan(
        self,
        request: AttemptToolBridgeRequest,
    ) -> ToolBridgePersistencePlan:
        """构建与此控制器调用 ID 匹配的桥接计划。"""

        from ..tool_bridge.persistence_contracts import (
            ToolBridgeCallPersistence,
            ToolBridgePersistencePlan,
        )

        return ToolBridgePersistencePlan(
            decision_apply_id=request.apply_id,
            close_apply_id=self._id("tool-close", _attempt_ordinal(request.attempt_id)),
            calls=tuple(
                ToolBridgeCallPersistence(
                    tool_call_id=call.tool_call_id,
                    tool_result_id=self._id(
                        "tool-result",
                        _attempt_ordinal(request.attempt_id),
                        ordinal,
                    ),
                    result_apply_id=self._id(
                        "tool-result-apply",
                        _attempt_ordinal(request.attempt_id),
                        ordinal,
                    ),
                )
                for ordinal, call in enumerate(
                    request.decision.action.calls,
                    start=1,
                )
            ),
        )

    def _id(self, kind: str, *ordinals: int) -> str:
        suffix = ":".join(str(item) for item in ordinals)
        value = f"{self.namespace}:{kind}" + (f":{suffix}" if suffix else "")
        if len(value) > 200:
            raise ValueError("stable ID exceeds the Store identifier limit")
        return value


def _attempt_ordinal(attempt_id: str) -> int:
    try:
        ordinal = int(attempt_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Attempt ID is outside the stable ID plan") from exc
    if ordinal < 1:
        raise ValueError("Attempt ordinal must be positive")
    return ordinal


__all__ = ['WorkRunTurnStableIdPlan']
