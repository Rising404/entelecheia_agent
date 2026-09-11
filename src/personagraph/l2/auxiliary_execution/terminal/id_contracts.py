"""冷的不可变终端-ID 合约用于 AuxiliaryGraph。

这些值源自已权威的图和终端候选事实，从而衍生出重启稳定的回执。它们不读取 Session 状态，不调用模型，不封存提案，也不提交 TaskGraph 的修订。
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.auxiliary_graph.contracts import TaskGraphSemanticTerminalCandidateBinding


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class _IdContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryTerminalIdPlan(_IdContract):
    reviewer_request_ids: tuple[str, str]
    reviewer_logical_call_ids: tuple[str, str]
    reviewer_result_ids: tuple[str, str]
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    terminal_seal_apply_id: str = Field(pattern=_ID_PATTERN)
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    task_graph_commit_apply_id: str = Field(pattern=_ID_PATTERN)


class AuxiliaryTerminalCandidateSemanticIdPlan(_IdContract):
    """Attempt 范围内的语义 ID 在完成前后共享。"""

    reviewer_request_ids: tuple[str, str]
    reviewer_logical_call_ids: tuple[str, str]
    reviewer_result_ids: tuple[str, str]
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    candidate_review_id: str = Field(pattern=_ID_PATTERN)


def derive_auxiliary_terminal_ids(
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    structure_sha256: str,
) -> AuxiliaryTerminalIdPlan:
    """从不可变当前图的执行归属衍生出重启稳定的标识。"""

    digest = _sha256_value(
        {
            "schema_version": "auxiliary-v2-terminal-stable-ids-v1",
            "session_id": session_id,
            "task_id": task_id,
            "auxiliary_graph_id": auxiliary_graph_id,
            "goal_id": goal_id,
            "auxiliary_graph_revision": auxiliary_graph_revision,
            "structure_sha256": structure_sha256,
        }
    )[:32]
    prefix = f"auxv2terminal-{digest}"
    return AuxiliaryTerminalIdPlan(
        reviewer_request_ids=(
            f"{prefix}:semantic-request-1",
            f"{prefix}:semantic-request-2",
        ),
        reviewer_logical_call_ids=(
            f"{prefix}:semantic-call-1",
            f"{prefix}:semantic-call-2",
        ),
        reviewer_result_ids=(
            f"{prefix}:semantic-result-1",
            f"{prefix}:semantic-result-2",
        ),
        semantic_settlement_id=f"{prefix}:semantic-settlement",
        terminal_seal_apply_id=f"{prefix}:seal",
        finish_gate_receipt_id=f"{prefix}:finish-gate",
        terminal_proposal_receipt_id=f"{prefix}:terminal-proposal",
        task_graph_commit_apply_id=f"{prefix}:task-graph-commit",
    )


def derive_auxiliary_terminal_candidate_semantic_ids(
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    structure_sha256: str,
    candidate_binding: TaskGraphSemanticTerminalCandidateBinding,
) -> AuxiliaryTerminalCandidateSemanticIdPlan:
    """将审阅者身份绑定到一个确切的 WorkRun Attempt 输出。"""

    digest = _sha256_value(
        {
            "schema_version": "auxiliary-v2-terminal-candidate-semantic-ids-v1",
            "session_id": session_id,
            "task_id": task_id,
            "auxiliary_graph_id": auxiliary_graph_id,
            "goal_id": goal_id,
            "auxiliary_graph_revision": auxiliary_graph_revision,
            "structure_sha256": structure_sha256,
            "candidate_binding": candidate_binding.model_dump(mode="json"),
        }
    )[:32]
    prefix = f"auxv2candidate-{digest}"
    return AuxiliaryTerminalCandidateSemanticIdPlan(
        reviewer_request_ids=tuple(
            f"{prefix}:semantic-request-{ordinal}" for ordinal in (1, 2)
        ),
        reviewer_logical_call_ids=tuple(
            f"{prefix}:semantic-call-{ordinal}" for ordinal in (1, 2)
        ),
        reviewer_result_ids=tuple(
            f"{prefix}:semantic-result-{ordinal}" for ordinal in (1, 2)
        ),
        semantic_settlement_id=f"{prefix}:semantic-settlement",
        candidate_review_id=f"{prefix}:candidate-review",
    )


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuxiliaryTerminalCandidateSemanticIdPlan",
    "AuxiliaryTerminalIdPlan",
    "derive_auxiliary_terminal_candidate_semantic_ids",
    "derive_auxiliary_terminal_ids",
]
