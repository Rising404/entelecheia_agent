"""一次 Tool Bridge 分派所用的纯 Host 持久化计划契约。

此处的值声明持久标识与可选单调截止时间；它们既不检查 WorkRun，也不执行 Tool。将其与具体
桥接器分离，使 Runtime 组装无需加载 Store、Catalog、策略或执行器代码即可推导稳定计划。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _PersistenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolBridgeCallPersistence(_PersistenceContract):
    """一个提案序号所用的 Host 标识与截止时间。

    ID 由所属 Controller 提供，而不是在桥接器内隐式分配。这样，精确应用重放会使用相同的调用、
    结果与收据标识。
    """

    tool_call_id: str = Field(min_length=1)
    tool_result_id: str = Field(min_length=1)
    result_apply_id: str = Field(min_length=1)
    deadline_monotonic: float | None = None


class ToolBridgePersistencePlan(_PersistenceContract):
    decision_apply_id: str = Field(min_length=1)
    close_apply_id: str = Field(min_length=1)
    calls: tuple[ToolBridgeCallPersistence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_host_identities(self) -> 'ToolBridgePersistencePlan':
        call_ids = [item.tool_call_id for item in self.calls]
        result_ids = [item.tool_result_id for item in self.calls]
        apply_ids = [
            self.decision_apply_id,
            *(item.result_apply_id for item in self.calls),
            self.close_apply_id,
        ]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool_call_id values must be unique")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("tool_result_id values must be unique")
        if len(apply_ids) != len(set(apply_ids)):
            raise ValueError("bridge apply_id values must be unique")
        return self


__all__ = ['ToolBridgeCallPersistence', 'ToolBridgePersistencePlan']
