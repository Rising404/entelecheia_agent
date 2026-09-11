"""确定性的准入规则，用于替换 L1 计划快照。"""

from __future__ import annotations

from collections.abc import Iterable

from ...model_io.output_repair_contracts import RuntimeModelOutputRepairIssue
from ...persistent_turn_content import L1Plan
from .identity import canonical_json


class L1PlanRevisionError(ValueError):
    """由模型作者撰写的计划修订将破坏已接受的 L1 历史。"""

    def __init__(
        self,
        message: str,
        *,
        repair_issue: RuntimeModelOutputRepairIssue | None = None,
    ) -> None:
        self.repair_issue = repair_issue
        super().__init__(message)


class L1PlanUnchangedError(L1PlanRevisionError):
    """完整的替换会完全重复当前的计划。"""


def validate_l1_plan_revision(
    previous: L1Plan,
    candidate: L1Plan,
    *,
    protected_acceptance_ids: Iterable[str] = (),
) -> None:
    """验证一次完整的 Plan 替换，不推断模型意图。"""

    if candidate.revision != previous.revision + 1:
        raise L1PlanRevisionError("Plan revision must advance exactly once")

    previous_by_id = {
        item.acceptance_id: item for item in previous.acceptances
    }
    candidate_by_id = {
        item.acceptance_id: item for item in candidate.acceptances
    }
    protected = frozenset(str(value) for value in protected_acceptance_ids)
    unknown_protected = protected - previous_by_id.keys()
    if unknown_protected:
        raise L1PlanRevisionError(
            "persisted execution history references an unknown Acceptance"
        )
    removed = protected - candidate_by_id.keys()
    if removed:
        raise L1PlanRevisionError(
            "Plan revision cannot remove an Acceptance referenced by "
            "execution history",
            repair_issue=RuntimeModelOutputRepairIssue(
                category="host_guard",
                code="host_guard.l1_protected_acceptance_removed",
                paths=("/plan/acceptances",),
                safe_explanation=(
                    "计划修订不能删除已被执行历史引用的 Acceptance；"
                    "请在完整替换计划中保留这些项目及其原 acceptance_id。"
                    "不修改计划时使用 plan=null。"
                ),
            ),
        )

    for index, item in enumerate(candidate.acceptances):
        acceptance_id = item.acceptance_id
        if acceptance_id not in previous_by_id:
            continue
        if (
            previous_by_id[acceptance_id].source
            != candidate_by_id[acceptance_id].source
        ):
            raise L1PlanRevisionError(
                "Plan revision cannot rebind an existing acceptance_id "
                "to another user message",
                repair_issue=RuntimeModelOutputRepairIssue(
                    category="host_guard",
                    code="host_guard.l1_acceptance_source_rebound",
                    paths=(f"/plan/acceptances/{index}/acceptance_id",),
                    safe_explanation=(
                        "已有 acceptance_id 不能改绑到另一条用户消息；"
                        "修订时保留原项目身份，新增项目省略 acceptance_id。"
                    ),
                ),
            )

    previous_semantics = previous.model_dump(mode="json", exclude={"revision"})
    candidate_semantics = candidate.model_dump(mode="json", exclude={"revision"})
    if canonical_json(previous_semantics) == canonical_json(candidate_semantics):
        raise L1PlanUnchangedError(
            "Plan revision must change the objective or Acceptances"
        )


__all__ = [
    "L1PlanRevisionError",
    "L1PlanUnchangedError",
    "validate_l1_plan_revision",
]
