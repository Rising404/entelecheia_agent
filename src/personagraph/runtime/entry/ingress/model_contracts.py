"""Entry 路由模型的已验证输出契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ...turn.contracts import ProcessingLevel


class EntryExistingRootTaskReference(BaseModel):
    """Read-only task reference allowed on an L0 route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_type: Literal["existing_root"] = "existing_root"
    insession_task_id: str = Field(min_length=1, max_length=128)
    source_excerpt: str = Field(min_length=1, max_length=4_000)
    execute_current: bool = False


class EntryClassification(BaseModel):
    """来自 entry classifier、已验证但非权威的提案。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    processing_level: ProcessingLevel
    task_matches: tuple[Any, ...] = Field(default=(), max_length=24)

    @field_validator("task_matches", mode="before")
    @classmethod
    def _validate_task_matches(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("task_matches must be an array")
        matches = tuple(value)
        if not matches:
            return ()
        processing_level = info.data.get("processing_level")
        if processing_level == "L1":
            raise ValueError("L1 task_matches must be empty")
        if processing_level == "L0":
            if not all(_task_match_type(match) == "existing_root" for match in matches):
                raise ValueError(
                    "L0 task_matches only support existing_root references"
                )
            return tuple(
                EntryExistingRootTaskReference.model_validate(
                    _task_match_payload(match)
                )
                for match in matches
            )
        # Mutating proposals are an L2 concern. Keep their schema and class identity
        # behind the route that can actually consume them, so importing or selecting
        # L1 never initializes personagraph.l2.
        from personagraph.l2.task_graph.task_matching import (
            InSessionTaskMatchesProposal,
        )

        return InSessionTaskMatchesProposal.model_validate(
            {"task_matches": [_task_match_payload(match) for match in matches]}
        ).task_matches

    def task_matches_proposal(self) -> object:
        """返回 Guard 与 Store 边界消费的领域批次。"""

        from personagraph.l2.task_graph.task_matching import (
            InSessionTaskMatchesProposal,
        )

        return InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    _task_match_payload(match) for match in self.task_matches
                ]
            }
        )


def _task_match_type(value: object) -> object:
    if isinstance(value, dict):
        return value.get("match_type")
    return getattr(value, "match_type", None)


def _task_match_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


__all__ = ["EntryClassification", "EntryExistingRootTaskReference"]
