"""纯 Acceptance 进度初始化与失败关闭合并规则。"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .contracts import (
    AcceptanceProgressItem,
    AcceptanceProgressSnapshot,
    AcceptanceUpdate,
    ExecutionSubject,
    HostMaterializedOutputWindowAction,
    HostMaterializedSubmitOutputWindowAction,
    HostMaterializedWriteOutputWindowAction,
    OutputWindow,
    SubmitOutputWindowAction,
    WriteOutputWindowAction,
)


class _ProgressContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceProgressErrorCode(StrEnum):
    """适用于未来 Store 命令结果的稳定拒绝代码。"""

    PROGRESS_REVISION_CONFLICT = "progress_revision_conflict"
    WORK_RUN_REVISION_CONFLICT = "work_run_revision_conflict"
    DUPLICATE_ACCEPTANCE_UPDATE = "duplicate_acceptance_update"
    UNKNOWN_ACCEPTANCE_ID = "unknown_acceptance_id"
    FALSE_UPDATE_HAS_SUPPORTING_RESULTS = "false_update_has_supporting_results"
    FALSE_UPDATE_HAS_EMPTY_SUPPORT_JUSTIFICATION = (
        "false_update_has_empty_support_justification"
    )
    SUPPORTING_RESULTS_HAVE_EMPTY_SUPPORT_JUSTIFICATION = (
        "supporting_results_have_empty_support_justification"
    )
    INVALID_SUPPORTING_TOOL_RESULT_ID = "invalid_supporting_tool_result_id"
    DUPLICATE_SUPPORTING_TOOL_RESULT_ID = "duplicate_supporting_tool_result_id"
    UNKNOWN_SUPPORTING_TOOL_RESULT_ID = "unknown_supporting_tool_result_id"
    EVALUATED_OUTPUT_REVISION_CONFLICT = "evaluated_output_revision_conflict"
    SUBMIT_REQUIRES_ALL_ACCEPTANCES_SATISFIED = (
        "submit_requires_all_acceptances_satisfied"
    )


class AcceptanceProgressIssue(_ProgressContract):
    code: AcceptanceProgressErrorCode
    acceptance_id: str | None = None
    tool_result_id: str | None = None


class AcceptanceProgressMergeResult(_ProgressContract):
    status: Literal["applied", "rejected"]
    snapshot: AcceptanceProgressSnapshot
    changed: bool
    expected_work_run_revision: int
    issues: tuple[AcceptanceProgressIssue, ...] = ()
    error_codes: tuple[AcceptanceProgressErrorCode, ...] = ()

    @model_validator(mode="after")
    def _validate_result_shape(self) -> 'AcceptanceProgressMergeResult':
        derived_codes = tuple(dict.fromkeys(issue.code for issue in self.issues))
        if self.status == "applied":
            if self.issues or self.error_codes:
                raise ValueError("an applied progress merge cannot contain errors")
            return self
        if self.changed or not self.issues or self.error_codes != derived_codes:
            raise ValueError("a rejected progress merge requires matching errors and no change")
        return self


class OutputWindowActionApplyResult(_ProgressContract):
    """一次整窗操作及其进度增量的原子纯结果。"""

    status: Literal["applied", "rejected"]
    output_window: OutputWindow
    output_changed: bool
    progress_merge: AcceptanceProgressMergeResult
    materialized_action: HostMaterializedOutputWindowAction | None = None

    @model_validator(mode="after")
    def _validate_result_shape(self) -> 'OutputWindowActionApplyResult':
        if self.status != self.progress_merge.status:
            raise ValueError("window and progress results must have the same status")
        if self.status == "rejected":
            if (
                self.output_changed
                or self.materialized_action is not None
            ):
                raise ValueError("a rejected window action cannot materialize a change")
            return self
        if self.materialized_action is None:
            raise ValueError("an applied window action requires a materialized reference")
        if (
            self.materialized_action.work_run_id != self.output_window.work_run_id
            or self.materialized_action.output_revision
            != self.output_window.output_revision
            or self.materialized_action.format != self.output_window.format
            or self.materialized_action.size_bytes
            != len(self.output_window.content.encode("utf-8"))
        ):
            raise ValueError("materialized OutputWindow reference does not match the window")
        return self


def initialize_acceptance_progress(
    *,
    work_run_id: str,
    subject: ExecutionSubject,
    acceptance_ids: Sequence[str],
    evaluated_output_revision: int = 1,
) -> AcceptanceProgressSnapshot:
    """创建修订一，其中每个 Acceptance 均明确为假且没有支持证据。"""

    ids = tuple(acceptance_ids)
    if not ids:
        raise ValueError("at least one Acceptance is required")
    if any(not acceptance_id.strip() for acceptance_id in ids):
        raise ValueError("Acceptance IDs must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("Acceptance IDs must be unique")
    return AcceptanceProgressSnapshot(
        work_run_id=work_run_id,
        subject=subject,
        revision=1,
        evaluated_output_revision=evaluated_output_revision,
        items=tuple(
            AcceptanceProgressItem(
                acceptance_id=acceptance_id,
                model_claimed_satisfied=False,
                supporting_tool_result_ids=(),
                empty_support_justification=None,
            )
            for acceptance_id in ids
        ),
    )


def merge_acceptance_progress(
    snapshot: AcceptanceProgressSnapshot,
    updates: Sequence[AcceptanceUpdate],
    *,
    known_historical_tool_result_ids: Collection[str],
    expected_progress_revision: int,
    current_work_run_revision: int,
    expected_work_run_revision: int,
) -> AcceptanceProgressMergeResult:
    """验证并原子合并一批完整替换更新。

    ``known_historical_tool_result_ids`` 是由 Host 提供的允许列表，来源为此 WorkRun
    中此前的 Attempt，特意不含当前 Attempt 的结果。拒绝时始终返回原始快照；
    调用方不得部分应用任何条目。
    """

    return _merge_acceptance_progress(
        snapshot,
        updates,
        known_historical_tool_result_ids=known_historical_tool_result_ids,
        expected_progress_revision=expected_progress_revision,
        current_work_run_revision=current_work_run_revision,
        expected_work_run_revision=expected_work_run_revision,
        evaluated_output_revision=snapshot.evaluated_output_revision,
        reset_for_output_change=False,
    )


def apply_output_window_action(
    current_output_window: OutputWindow,
    snapshot: AcceptanceProgressSnapshot,
    action: (
        WriteOutputWindowAction
        | SubmitOutputWindowAction
    ),
    *,
    acceptance_updates: Sequence[AcceptanceUpdate] = (),
    updated_turn_id: str,
    updated_attempt_id: str,
    known_historical_tool_result_ids: Collection[str],
    expected_progress_revision: int,
    current_work_run_revision: int,
    expected_work_run_revision: int,
) -> OutputWindowActionApplyResult:
    """完整替换一个 WorkRun 窗口，并原子地重新评估进度。

    只有 ``format + content`` 完全相等才属于窗口空操作。实际窗口变更会先重置每个
    Acceptance，再应用同一模型决策中的更新；即使生成的条目值恰好与上一快照相同，
    也会强制推进进度修订。
    """

    if current_output_window.work_run_id != snapshot.work_run_id:
        raise ValueError("OutputWindow and AcceptanceProgress must share a WorkRun")

    if snapshot.evaluated_output_revision != current_output_window.output_revision:
        rejected = _rejected(
            snapshot,
            expected_work_run_revision,
            (
                AcceptanceProgressIssue(
                    code=AcceptanceProgressErrorCode.EVALUATED_OUTPUT_REVISION_CONFLICT
                ),
            ),
        )
        return OutputWindowActionApplyResult(
            status="rejected",
            output_window=current_output_window,
            output_changed=False,
            progress_merge=rejected,
        )

    action_format = action.format
    action_content = action.content

    output_changed = not (
        current_output_window.format == action_format
        and current_output_window.content == action_content
    )
    candidate_window = (
        OutputWindow(
            work_run_id=current_output_window.work_run_id,
            output_revision=current_output_window.output_revision + 1,
            format=action_format,
            content=action_content,
            updated_turn_id=updated_turn_id,
            updated_attempt_id=updated_attempt_id,
        )
        if output_changed
        else current_output_window
    )

    merged = _merge_acceptance_progress(
        snapshot,
        acceptance_updates,
        known_historical_tool_result_ids=known_historical_tool_result_ids,
        expected_progress_revision=expected_progress_revision,
        current_work_run_revision=current_work_run_revision,
        expected_work_run_revision=expected_work_run_revision,
        evaluated_output_revision=candidate_window.output_revision,
        reset_for_output_change=output_changed,
    )
    if merged.status == "rejected":
        return OutputWindowActionApplyResult(
            status="rejected",
            output_window=current_output_window,
            output_changed=False,
            progress_merge=merged,
        )

    if isinstance(action, SubmitOutputWindowAction) and not all(
        item.model_claimed_satisfied for item in merged.snapshot.items
    ):
        rejected = _rejected(
            snapshot,
            expected_work_run_revision,
            (
                AcceptanceProgressIssue(
                    code=(
                        AcceptanceProgressErrorCode.SUBMIT_REQUIRES_ALL_ACCEPTANCES_SATISFIED
                    )
                ),
            ),
        )
        return OutputWindowActionApplyResult(
            status="rejected",
            output_window=current_output_window,
            output_changed=False,
            progress_merge=rejected,
        )

    if isinstance(action, SubmitOutputWindowAction):
        materialized: HostMaterializedOutputWindowAction = HostMaterializedSubmitOutputWindowAction(
            work_run_id=candidate_window.work_run_id,
            output_revision=candidate_window.output_revision,
            format=candidate_window.format,
            size_bytes=len(candidate_window.content.encode("utf-8")),
        )
    else:
        materialized = HostMaterializedWriteOutputWindowAction(
            work_run_id=candidate_window.work_run_id,
            output_revision=candidate_window.output_revision,
            format=candidate_window.format,
            size_bytes=len(candidate_window.content.encode("utf-8")),
        )
    return OutputWindowActionApplyResult(
        status="applied",
        output_window=candidate_window,
        output_changed=output_changed,
        progress_merge=merged,
        materialized_action=materialized,
    )


def _merge_acceptance_progress(
    snapshot: AcceptanceProgressSnapshot,
    updates: Sequence[AcceptanceUpdate],
    *,
    known_historical_tool_result_ids: Collection[str],
    expected_progress_revision: int,
    current_work_run_revision: int,
    expected_work_run_revision: int,
    evaluated_output_revision: int,
    reset_for_output_change: bool,
) -> AcceptanceProgressMergeResult:
    revision_issues: list[AcceptanceProgressIssue] = []
    if expected_progress_revision != snapshot.revision:
        revision_issues.append(
            AcceptanceProgressIssue(
                code=AcceptanceProgressErrorCode.PROGRESS_REVISION_CONFLICT
            )
        )
    if expected_work_run_revision != current_work_run_revision:
        revision_issues.append(
            AcceptanceProgressIssue(
                code=AcceptanceProgressErrorCode.WORK_RUN_REVISION_CONFLICT
            )
        )
    if revision_issues:
        return _rejected(snapshot, expected_work_run_revision, revision_issues)

    base_items = (
        tuple(
            AcceptanceProgressItem(
                acceptance_id=item.acceptance_id,
                model_claimed_satisfied=False,
                supporting_tool_result_ids=(),
                empty_support_justification=None,
            )
            for item in snapshot.items
        )
        if reset_for_output_change
        else snapshot.items
    )
    current_by_id = {item.acceptance_id: item for item in base_items}
    allowed_result_ids = frozenset(known_historical_tool_result_ids)
    issues: list[AcceptanceProgressIssue] = []
    seen_acceptance_ids: set[str] = set()

    for update in updates:
        acceptance_id = update.acceptance_id
        if acceptance_id in seen_acceptance_ids:
            issues.append(
                AcceptanceProgressIssue(
                    code=AcceptanceProgressErrorCode.DUPLICATE_ACCEPTANCE_UPDATE,
                    acceptance_id=acceptance_id,
                )
            )
        seen_acceptance_ids.add(acceptance_id)

        if acceptance_id not in current_by_id:
            issues.append(
                AcceptanceProgressIssue(
                    code=AcceptanceProgressErrorCode.UNKNOWN_ACCEPTANCE_ID,
                    acceptance_id=acceptance_id,
                )
            )

        result_ids = update.supporting_tool_result_ids
        if not update.model_claimed_satisfied and result_ids:
            issues.append(
                AcceptanceProgressIssue(
                    code=AcceptanceProgressErrorCode.FALSE_UPDATE_HAS_SUPPORTING_RESULTS,
                    acceptance_id=acceptance_id,
                )
            )
        if (
            not update.model_claimed_satisfied
            and update.empty_support_justification is not None
        ):
            issues.append(
                AcceptanceProgressIssue(
                    code=(
                        AcceptanceProgressErrorCode.FALSE_UPDATE_HAS_EMPTY_SUPPORT_JUSTIFICATION
                    ),
                    acceptance_id=acceptance_id,
                )
            )
        if result_ids and update.empty_support_justification is not None:
            issues.append(
                AcceptanceProgressIssue(
                    code=(
                        AcceptanceProgressErrorCode.SUPPORTING_RESULTS_HAVE_EMPTY_SUPPORT_JUSTIFICATION
                    ),
                    acceptance_id=acceptance_id,
                )
            )

        seen_result_ids: set[str] = set()
        for result_id in result_ids:
            if not result_id.strip():
                issues.append(
                    AcceptanceProgressIssue(
                        code=AcceptanceProgressErrorCode.INVALID_SUPPORTING_TOOL_RESULT_ID,
                        acceptance_id=acceptance_id,
                        tool_result_id=result_id,
                    )
                )
            if result_id in seen_result_ids:
                issues.append(
                    AcceptanceProgressIssue(
                        code=AcceptanceProgressErrorCode.DUPLICATE_SUPPORTING_TOOL_RESULT_ID,
                        acceptance_id=acceptance_id,
                        tool_result_id=result_id,
                    )
                )
            seen_result_ids.add(result_id)
            if result_id not in allowed_result_ids:
                issues.append(
                    AcceptanceProgressIssue(
                        code=AcceptanceProgressErrorCode.UNKNOWN_SUPPORTING_TOOL_RESULT_ID,
                        acceptance_id=acceptance_id,
                        tool_result_id=result_id,
                    )
                )

    if issues:
        return _rejected(snapshot, expected_work_run_revision, issues)

    replacements = {
        update.acceptance_id: AcceptanceProgressItem(
            acceptance_id=update.acceptance_id,
            model_claimed_satisfied=update.model_claimed_satisfied,
            supporting_tool_result_ids=(
                update.supporting_tool_result_ids
                if update.model_claimed_satisfied
                else ()
            ),
            empty_support_justification=(
                update.empty_support_justification
                if update.model_claimed_satisfied
                else None
            ),
        )
        for update in updates
    }
    merged_items = tuple(
        replacements.get(item.acceptance_id, item) for item in base_items
    )
    changed = (
        reset_for_output_change
        or merged_items != snapshot.items
        or evaluated_output_revision != snapshot.evaluated_output_revision
    )
    merged_snapshot = (
        AcceptanceProgressSnapshot(
            work_run_id=snapshot.work_run_id,
            subject=snapshot.subject,
            revision=snapshot.revision + 1,
            evaluated_output_revision=evaluated_output_revision,
            items=merged_items,
        )
        if changed
        else snapshot
    )
    return AcceptanceProgressMergeResult(
        status="applied",
        snapshot=merged_snapshot,
        changed=changed,
        expected_work_run_revision=expected_work_run_revision,
    )


def _rejected(
    snapshot: AcceptanceProgressSnapshot,
    expected_work_run_revision: int,
    issues: Sequence[AcceptanceProgressIssue],
) -> AcceptanceProgressMergeResult:
    ordered_codes = tuple(dict.fromkeys(issue.code for issue in issues))
    return AcceptanceProgressMergeResult(
        status="rejected",
        snapshot=snapshot,
        changed=False,
        expected_work_run_revision=expected_work_run_revision,
        issues=tuple(issues),
        error_codes=ordered_codes,
    )
