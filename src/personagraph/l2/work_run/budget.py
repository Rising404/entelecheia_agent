"""由 Host 持有的纯 WorkRun 活跃时间预算转换。

Host 会在持久安全检查点显式提供已经过的活跃时间。本模块特意不持有时钟：
因此，等待用户输入、授权或外部结果所耗费的墙上时间，绝不会由持久化层推断或计费。
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import WorkRunBudget


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkRunBudgetDisposition(StrEnum):
    WITHIN_LIMIT = "within_limit"
    SOFT_LIMIT_REACHED = "soft_limit_reached"
    HARD_LIMIT_REACHED = "hard_limit_reached"


class WorkRunBudgetTransition(_Contract):
    """对 WorkRun 预算执行的一次确定性活跃时间计费。"""

    active_seconds_delta: float = Field(gt=0)
    budget_before: WorkRunBudget
    budget_after: WorkRunBudget
    disposition: WorkRunBudgetDisposition

    @field_validator("active_seconds_delta", mode="before")
    @classmethod
    def _require_finite_delta(cls, value: object) -> object:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("active-time charge must be a finite positive number")
        return value

    @model_validator(mode="after")
    def _require_consistent_transition(self) -> 'WorkRunBudgetTransition':
        before = self.budget_before
        after = self.budget_after
        if (
            after.max_attempts != before.max_attempts
            or after.soft_active_seconds != before.soft_active_seconds
            or after.hard_active_seconds != before.hard_active_seconds
            or after.attempts_started != before.attempts_started
            or after.active_seconds_consumed
            != before.active_seconds_consumed + self.active_seconds_delta
        ):
            raise ValueError("active-time transition budgets are inconsistent")
        if self.disposition is not _classify_budget(after):
            raise ValueError("active-time transition disposition is inconsistent")
        return self


def _classify_budget(budget: WorkRunBudget) -> WorkRunBudgetDisposition:
    if budget.active_seconds_consumed >= budget.hard_active_seconds:
        return WorkRunBudgetDisposition.HARD_LIMIT_REACHED
    if budget.active_seconds_consumed >= budget.soft_active_seconds:
        return WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
    return WorkRunBudgetDisposition.WITHIN_LIMIT


def charge_work_run_active_seconds(
    budget: WorkRunBudget,
    *,
    active_seconds_delta: float,
) -> WorkRunBudgetTransition:
    """加入一段由 Host 测得的活跃区间，并对其精确边界分类。"""

    if not isinstance(budget, WorkRunBudget):
        raise TypeError("budget must be a WorkRunBudget")
    if (
        isinstance(active_seconds_delta, bool)
        or not isinstance(active_seconds_delta, int | float)
        or not math.isfinite(float(active_seconds_delta))
        or float(active_seconds_delta) <= 0
    ):
        raise ValueError("active_seconds_delta must be a finite positive number")

    delta = float(active_seconds_delta)
    budget_after = WorkRunBudget(
        max_attempts=budget.max_attempts,
        soft_active_seconds=budget.soft_active_seconds,
        hard_active_seconds=budget.hard_active_seconds,
        attempts_started=budget.attempts_started,
        active_seconds_consumed=budget.active_seconds_consumed + delta,
    )
    disposition = _classify_budget(budget_after)
    return WorkRunBudgetTransition(
        active_seconds_delta=delta,
        budget_before=budget,
        budget_after=budget_after,
        disposition=disposition,
    )
