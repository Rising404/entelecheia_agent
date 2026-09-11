"""已验证 WorkRun 轮次请求契约的边界覆盖。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.task_execution.work_run import turn_request_contracts as contracts
from personagraph.l2.task_execution.task_node.input_limits import (
    AttemptDecisionInputLimits,
    NodeVerificationInputLimits,
    TaskNodeDependencyInputLimits,
)
from personagraph.l2.work_run import TaskNodeSubject
from personagraph.l2.work_run.pending_user_question_contracts import (
    PendingUserQuestion,
)


def _subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=1,
        node_id="node-1",
        node_revision=1,
    )


def _limits() -> tuple[AttemptDecisionInputLimits, NodeVerificationInputLimits]:
    dependency = TaskNodeDependencyInputLimits(
        profile_id="request-contract-dependencies",
        max_items=2,
        max_serialized_utf8_bytes=10_000,
    )
    return (
        AttemptDecisionInputLimits(
            profile_id="request-contract-attempt",
            max_prior_tool_result_items=2,
            max_prior_tool_results_serialized_utf8_bytes=10_000,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=100_000,
        ),
        NodeVerificationInputLimits(
            profile_id="request-contract-verification",
            max_acceptance_items=8,
            max_supporting_tool_result_items=0,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=100_000,
        ),
    )


def test_waiting_user_request_keeps_stale_fence_validation() -> None:
    subject = _subject()
    attempt_limits, verification_limits = _limits()
    question = PendingUserQuestion(
        session_id="session-1",
        work_run_id="work-run-1",
        work_run_revision=2,
        subject=subject,
        question_attempt_id="attempt-1",
        question_attempt_ordinal=1,
        question_turn_id="turn-1",
        question="Continue?",
        task_state_version=1,
        node_state_version=1,
        acceptance_progress_revision=1,
        output_window_revision=1,
    )

    with pytest.raises(ValidationError, match="belongs to another Session"):
        contracts.WorkRunTurnWaitingUserContinuationRequest(
            session_id="session-2",
            turn_id="turn-2",
            pending_question=question,
            expected_window_revision=1,
            attempt_input_limits=attempt_limits,
            verification_input_limits=verification_limits,
        )
    with pytest.raises(ValidationError, match="must follow the question Turn"):
        contracts.WorkRunTurnWaitingUserContinuationRequest(
            session_id="session-1",
            turn_id="turn-1",
            pending_question=question,
            expected_window_revision=1,
            attempt_input_limits=attempt_limits,
            verification_input_limits=verification_limits,
        )
