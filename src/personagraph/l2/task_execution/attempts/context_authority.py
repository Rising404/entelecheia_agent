"""对一个已启动的 Attempt 执行基于 Store 的权威状态检查。

该组件重新读取持久化的 WorkRun、Turn、目录、验证器和节点事实，以核验内存中的 :class:`AttemptDecisionContext` 绑定。它不请求模型决策、不分配 Tool ID，也不分发动作。
"""

from __future__ import annotations

import hashlib
import json

from ...task_graph.contracts import InSessionTaskAcceptanceProposal
from ....session import store as session_store
from ....session.l2_store import task_graph as task_graph_store
from ....session.l2_store import work_run as work_run_store
from ....session.l2_store.work_run import StoredAttempt, StoredWorkRun
from ....tools.contracts import ToolSpec
from ...work_run import (
    AttemptStatus,
    RequestUserInputAction,
    TaskNodeSubject,
    WorkRunStatus,
)
from .input_projection import (
    AttemptDecisionContext,
    AttemptVerificationFeedback,
    AttemptUserInput,
    PriorToolResultProjection,
    mandatory_prior_tool_result_ids,
    select_bounded_prior_tool_results,
)
class AttemptControllerStateConflict(RuntimeError):
    """可信的决策上下文不再匹配持久化的权威状态。"""


def require_fresh_attempt_context(
    context: AttemptDecisionContext,
    *,
    expected_window_revision: int,
) -> StoredWorkRun:
    """无需重建语义节点输入，重新验证 Store 拥有的绑定。"""

    stored = work_run_store.get_work_run(
        session_id=context.session_id,
        work_run_id=context.work_run_id,
    )
    run = stored.work_run
    if stored.session_id != context.session_id:
        _stale("session")
    if run.work_run_id != context.work_run_id:
        _stale("work_run_id")
    if run.revision != context.work_run_revision:
        _stale("work_run_revision")
    if run.subject != context.subject:
        _stale("subject")
    if run.status is not WorkRunStatus.ACTIVE or run.reason is not None:
        _stale("work_run_phase")
    if stored.acceptance_progress != context.acceptance_progress:
        _stale("acceptance_progress")
    if stored.output_window != context.output_window:
        _stale("output_window")
    if stored.current_attempt_id != context.attempt_id:
        _stale("current_attempt_id")
    if stored.pending_user_question is not None:
        _stale("pending_user_question")

    _require_authoritative_node_semantics(context)
    matching_attempts = tuple(
        item
        for item in stored.attempts
        if item.attempt.attempt_id == context.attempt_id
    )
    if len(matching_attempts) != 1:
        _stale("attempt_identity")
    current = matching_attempts[0]
    if current.attempt.work_run_id != context.work_run_id:
        _stale("attempt_work_run")
    if current.attempt.ordinal != context.attempt_ordinal:
        _stale("attempt_ordinal")
    if current.turn_id != context.turn_id:
        _stale("attempt_turn")
    if current.attempt.status is not AttemptStatus.ACTIVE:
        _stale("attempt_status")
    if (
        current.action is not None
        or current.decision is not None
        or current.committed_output_revision is not None
        or current.attempt.submitted_output_revision is not None
    ):
        _stale("attempt_decision")
    if context.user_input != project_authoritative_attempt_user_input(
        stored=stored,
        current_attempt=current,
    ):
        _stale("user_input_authority")
    if current.input_output_revision != context.output_window.output_revision:
        _stale("attempt_input_output")
    if current.input_output_hash != _output_window_hash(context):
        _stale("attempt_input_output_hash")
    _require_authoritative_verification_feedback(
        context,
        current_attempt=current,
    )
    if any(
        item.attempt_id == context.attempt_id for item in stored.tool_calls
    ) or any(
        item.attempt_id == context.attempt_id for item in stored.tool_results
    ):
        _stale("attempt_execution_history")

    _require_authoritative_allowed_tools(
        context,
        catalog_snapshot=current.catalog_snapshot,
    )

    _require_authorized_prior_tool_results(
        context,
        stored=stored,
        current_attempt_ordinal=current.attempt.ordinal,
    )
    _require_current_window_binding(
        context,
        expected_window_revision=expected_window_revision,
    )
    return stored


def project_authoritative_attempt_user_input(
    *,
    stored: StoredWorkRun,
    current_attempt: StoredAttempt,
) -> AttemptUserInput:
    """读取精确的不可变的 Turn 输入和已消费的等待问题。

    Attempt ``turn_id`` 是执行归属，并且在未决的 Attempt 恢复时可能会重新绑定。提示意义遵循不可变的 ``input_turn_id``。先前的问题仅通过 Store 的精确前驱引用进行投影；序号邻近绝不会用作推断依据。
    """

    try:
        turn_input = session_store.get_turn_execution_input(
            session_id=stored.session_id,
            turn_id=current_attempt.input_turn_id,
        )
    except Exception as exc:
        raise AttemptControllerStateConflict(
            "authoritative Attempt user input is unavailable"
        ) from exc
    if (
        str(turn_input.get("session_id") or "") != stored.session_id
        or str(turn_input.get("turn_id") or "") != current_attempt.input_turn_id
    ):
        _stale("attempt_input_turn")
    content = turn_input.get("content")
    if not isinstance(content, str) or not content:
        _stale("attempt_user_input")

    question: str | None = None
    predecessor_id = current_attempt.predecessor_question_attempt_id
    if predecessor_id is not None:
        predecessors = tuple(
            item
            for item in stored.attempts
            if item.attempt.attempt_id == predecessor_id
        )
        if len(predecessors) != 1:
            _stale("predecessor_question_attempt")
        predecessor = predecessors[0]
        if (
            predecessor.attempt.status is not AttemptStatus.CLOSED
            or predecessor.attempt.ordinal >= current_attempt.attempt.ordinal
            or predecessor.action != "request_user_input"
            or predecessor.decision is None
            or not isinstance(predecessor.decision.action, RequestUserInputAction)
        ):
            _stale("predecessor_question_attempt")
        question = predecessor.decision.action.question

    return AttemptUserInput(
        content=content,
        prior_waiting_user_question=question,
    )


def _require_authoritative_verification_feedback(
    context: AttemptDecisionContext,
    *,
    current_attempt: StoredAttempt,
) -> None:
    """将下一个 Attempt 反馈绑定到经 Store 联结得到的验证器结果。"""

    attempt = current_attempt.attempt
    request_id = current_attempt.input_verification_request_id
    result = current_attempt.input_verification_result
    if (request_id is None) is not (result is None):
        _stale("attempt_verification_binding")
    if attempt.ordinal == 1 and (request_id is not None or result is not None):
        _stale("first_attempt_verification_feedback")
    if result is None:
        if context.verification_feedback is not None:
            _stale("verification_feedback_authority")
        return

    if (
        result.all_pass
        or result.verification_request_id != request_id
        or result.work_run_id != context.work_run_id
        or result.subject != context.subject
        or result.submitted_attempt_id == context.attempt_id
        or result.output_revision != context.output_window.output_revision
        or result.output_revision != current_attempt.input_output_revision
        or result.locked_work_run_revision >= context.work_run_revision
        or tuple(item.acceptance_id for item in result.acceptance_results)
        != tuple(item.acceptance_id for item in context.acceptances)
    ):
        _stale("attempt_verification_binding")

    authoritative = AttemptVerificationFeedback(
        submitted_output_revision=result.output_revision,
        acceptance_results=result.acceptance_results,
        downstream_results=result.downstream_results,
    )
    if context.verification_feedback != authoritative:
        _stale("verification_feedback_authority")


def _require_authoritative_allowed_tools(
    context: AttemptDecisionContext,
    *,
    catalog_snapshot: dict[str, object],
) -> None:
    """从冻结的 Attempt 目录中投影精确的已暴露的 ToolSpecs。"""

    entries = catalog_snapshot.get("entries")
    if not isinstance(entries, list | tuple):
        _stale("attempt_catalog_snapshot")
    authoritative: list[ToolSpec] = []
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                _stale("attempt_catalog_entry")
            status = str(entry.get("status") or "")
            if status not in {"active", "deprecated"}:
                continue
            registration = entry.get("registration")
            if not isinstance(registration, dict):
                _stale("attempt_catalog_registration")
            raw_spec = registration.get("spec")
            if not isinstance(raw_spec, dict):
                _stale("attempt_catalog_tool_spec")
            authoritative.append(
                ToolSpec(
                    tool_id=raw_spec["tool_id"],
                    contract_version=raw_spec["contract_version"],
                    name=raw_spec["name"],
                    description=raw_spec["description"],
                    input_schema=raw_spec["input_schema"],
                    output_schema=raw_spec["output_schema"],
                    catalog_tags=tuple(raw_spec.get("catalog_tags", ())),
                )
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise AttemptControllerStateConflict(
            "frozen Attempt catalog cannot produce authoritative ToolSpecs"
        ) from exc
    if tuple(item.to_dict() for item in authoritative) != tuple(
        item.to_dict() for item in context.allowed_tools
    ):
        _stale("allowed_tools")


def _require_authoritative_node_semantics(
    context: AttemptDecisionContext,
) -> None:
    """将提示文本和 Acceptances 绑定到精确标记的当前节点。"""

    if not isinstance(context.subject, TaskNodeSubject):
        _stale("subject_kind")

    details = task_graph_store.get_insession_task_details(
        context.session_id,
        context.subject.task_id,
    )
    if details is None:
        _stale("task_details")
    assert details is not None
    if (
        details.session_id != context.session_id
        or details.insession_task_id != context.subject.task_id
        or details.current_graph_revision != context.subject.graph_revision
    ):
        _stale("task_graph_revision")

    matches = tuple(
        node
        for node in details.nodes
        if str(node.get("insession_task_node_id") or "")
        == context.subject.node_id
        and int(node.get("node_revision") or 0) == context.subject.node_revision
    )
    if len(matches) != 1:
        _stale("task_node_revision")
    node = matches[0]
    if str(node.get("title") or "") != context.node_title:
        _stale("node_title")
    if str(node.get("objective") or "") != context.node_objective:
        _stale("node_objective")
    try:
        authoritative_acceptances = tuple(
            InSessionTaskAcceptanceProposal.model_validate(item)
            for item in node.get("acceptance_criteria", ())
        )
    except (TypeError, ValueError) as exc:
        raise AttemptControllerStateConflict(
            "authoritative TaskNode Acceptance projection is invalid"
        ) from exc
    if authoritative_acceptances != context.acceptances:
        _stale("node_acceptances")


def _require_authorized_prior_tool_results(
    context: AttemptDecisionContext,
    *,
    stored: StoredWorkRun,
    current_attempt_ordinal: int,
) -> None:
    attempts = {
        item.attempt.attempt_id: item.attempt for item in stored.attempts
    }
    calls = {item.call.tool_call_id: item for item in stored.tool_calls}
    authorized: list[PriorToolResultProjection] = []
    for result in stored.tool_results:
        attempt = attempts.get(result.attempt_id)
        if (
            attempt is None
            or attempt.status is not AttemptStatus.CLOSED
            or attempt.ordinal >= current_attempt_ordinal
        ):
            continue
        call = calls.get(result.tool_call_id)
        if call is None or call.attempt_id != result.attempt_id:
            _stale("prior_tool_result_call")
        authorized.append(
            PriorToolResultProjection(
                tool_id=call.call.tool_id,
                tool_version=call.call.tool_version,
                result=result,
            )
        )

    authorized_by_id = {
        item.result.tool_result_id: (position, item)
        for position, item in enumerate(authorized)
    }
    positions: list[int] = []
    for projected in context.prior_tool_results.items:
        matched = authorized_by_id.get(projected.result.tool_result_id)
        if matched is None or matched[1] != projected:
            _stale("prior_tool_results")
        positions.append(matched[0])
    if positions != sorted(positions):
        _stale("prior_tool_result_order")
    if (
        not context.prior_tool_results.truncated
        and context.prior_tool_results.items != tuple(authorized)
    ):
        _stale("prior_tool_result_completeness")
    projected_ids = {
        item.result.tool_result_id for item in context.prior_tool_results.items
    }
    supporting_ids = {
        result_id
        for item in context.acceptance_progress.items
        for result_id in item.supporting_tool_result_ids
    }
    required_ids = mandatory_prior_tool_result_ids(
        tuple(authorized),
        supporting_result_ids=supporting_ids,
    )
    if not supporting_ids.issubset(projected_ids):
        _stale("supporting_tool_result_projection")
    if not required_ids.issubset(projected_ids):
        _stale("mandatory_tool_result_projection")
    expected_projection = select_bounded_prior_tool_results(
        tuple(authorized),
        required_result_ids=required_ids,
        limits=context.input_limits,
    )
    if context.prior_tool_results != expected_projection:
        _stale("prior_tool_result_bounded_projection")


def _require_current_window_binding(
    context: AttemptDecisionContext,
    *,
    expected_window_revision: int,
) -> None:
    window = session_store.get_turn_execution_window(context.session_id)
    if window is None:
        _stale("turn_execution_window")
    assert window is not None
    if str(window.get("turn_id") or "") != context.turn_id:
        _stale("window_turn")
    if str(window.get("window_state") or "") != "active":
        _stale("window_state")
    if str(window.get("current_work_run_id") or "") != context.work_run_id:
        _stale("window_work_run")
    if str(window.get("current_attempt_id") or "") != context.attempt_id:
        _stale("window_attempt")
    if int(window.get("state_version") or 0) != expected_window_revision:
        _stale("window_revision")


def _output_window_hash(context: AttemptDecisionContext) -> str:
    canonical = json.dumps(
        context.output_window.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stale(binding: str) -> None:
    raise AttemptControllerStateConflict(
        f"trusted Attempt context is stale or misbound: {binding}"
    )


__all__ = [
    "AttemptControllerStateConflict",
    "project_authoritative_attempt_user_input",
    "require_fresh_attempt_context",
]
