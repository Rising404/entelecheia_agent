"""Host 接受的本轮计划；消息来源不是精确片段引用或任务完成证明。"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

L1_ACCEPTANCE_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class L1MessageSource(_Contract):
    message_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class L1Acceptance(_Contract):
    acceptance_id: str = Field(pattern=L1_ACCEPTANCE_ID_PATTERN)
    criterion: str = Field(min_length=1, max_length=1200)
    source: L1MessageSource


class L1Plan(_Contract):
    schema_version: Literal["l1-plan-v3"] = "l1-plan-v3"
    revision: int = Field(default=1, ge=1, le=64)
    objective: str = Field(min_length=1, max_length=2000)
    acceptances: tuple[L1Acceptance, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def _unique_ids(self) -> L1Plan:
        ids = [item.acceptance_id for item in self.acceptances]
        if len(ids) != len(set(ids)):
            raise ValueError("materialized L1 acceptances must be unique")
        return self


__all__ = ["L1_ACCEPTANCE_ID_PATTERN", "L1Acceptance", "L1MessageSource", "L1Plan"]
