"""L1 唯一公开提案：每步笔记、可选计划与原生结果引用、一个动作。"""

from __future__ import annotations
import hashlib
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ..persistent_turn_content.plan import (
    L1_ACCEPTANCE_ID_PATTERN,
    L1Acceptance,
    L1MessageSource,
    L1Plan,
)
from .actions import CallToolsAction
from ..persistent_turn_content.findings import (
    EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    validate_execution_finding_text,
)

L1_ATTEMPT_PROTOCOL_VERSION = "l1-attempt-model-view-v5"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SubmitFinalReplyAction(_Contract):
    kind: Literal["submit_final_reply"] = "submit_final_reply"
    reply: str = Field(min_length=1)

    @field_validator("reply")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a submitted L1 final reply must not be empty")
        return value


class L1ResultReference(_Contract):
    """引用已经执行的真实结果；可选块 ID 必须属于该结果。"""

    tool_result_id: str = Field(min_length=1, max_length=200)
    chunk_id: str | None = Field(default=None, min_length=1, max_length=200)


class L1AcceptanceProposal(_Contract):
    criterion: str = Field(min_length=1, max_length=1200)
    acceptance_id: str | None = Field(
        default=None,
        pattern=L1_ACCEPTANCE_ID_PATTERN,
        description="Only use an existing plan item ID when revising it; omit for a new item.",
    )


class L1PlanProposal(_Contract):
    objective: str = Field(min_length=1, max_length=2000)
    acceptances: tuple[L1AcceptanceProposal, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def _unique_ids(self) -> L1PlanProposal:
        ids = [
            item.acceptance_id
            for item in self.acceptances
            if item.acceptance_id is not None
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("L1 acceptance IDs must be unique")
        return self


L1AttemptActionProposal = Annotated[
    CallToolsAction | SubmitFinalReplyAction,
    Field(discriminator="kind"),
]


class L1AttemptDecisionProposal(_Contract):
    note: str = Field(min_length=1, max_length=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS)
    action: L1AttemptActionProposal
    plan: L1PlanProposal | None = None
    references: tuple[L1ResultReference, ...] = Field(default=(), max_length=24)

    @field_validator("note")
    @classmethod
    def _public_note(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a public execution note must not be empty")
        return validate_execution_finding_text(value.strip())


def materialize_l1_plan(
    proposal: L1PlanProposal,
    *,
    input_message_id: str,
    user_text: str,
    revision: int = 1,
    previous: L1Plan | None = None,
) -> L1Plan:
    """Host 分配新项身份；修改既有项必须明确使用其 ID，不按位置猜测。"""
    source = L1MessageSource(
        message_id=input_message_id,
        content_sha256=hashlib.sha256(user_text.encode()).hexdigest(),
    )
    known = (
        {item.acceptance_id: item for item in previous.acceptances} if previous else {}
    )
    acceptances = []
    for index, item in enumerate(proposal.acceptances):
        if item.acceptance_id is not None:
            if item.acceptance_id not in known:
                raise ValueError(
                    "acceptance_id must identify an existing plan item; omit for new items"
                )
            identity = item.acceptance_id
            item_source = known[identity].source
        else:
            seed = f"{input_message_id}:{revision}:{index}"
            identity = "a_" + hashlib.sha256(seed.encode()).hexdigest()[:16]
            item_source = source
        acceptances.append(
            L1Acceptance(
                acceptance_id=identity,
                criterion=item.criterion,
                source=item_source,
            )
        )
    return L1Plan(
        revision=revision, objective=proposal.objective, acceptances=tuple(acceptances)
    )


__all__ = [
    "L1_ATTEMPT_PROTOCOL_VERSION",
    "L1AcceptanceProposal",
    "L1AttemptActionProposal",
    "L1AttemptDecisionProposal",
    "L1PlanProposal",
    "L1ResultReference",
    "SubmitFinalReplyAction",
    "materialize_l1_plan",
]
