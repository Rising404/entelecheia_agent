"""语义编译的冷候选项与结果契约。

这些不可变 Pydantic DTO 描述不可信模型候选项，以及归约器和持久化层使用的已验证结果。
它们不构造可信编译器输入、不解析模型回复、不检查锚点文本，也不决定语义覆盖范围。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnchorSource(StrEnum):
    RAW_TURN = "raw_turn"
    TYPED_EVENT = "typed_event"


class SemanticAttribution(StrEnum):
    USER_INSTRUCTION = "user_instruction"
    USER_CONTEXT = "user_context"
    QUOTED_EXTERNAL = "quoted_external"
    ATTACHMENT = "attachment"
    TOOL_OBSERVATION = "tool_observation"
    HYPOTHETICAL = "hypothetical"


class CommunicativeFunction(StrEnum):
    INFORM = "inform"
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    CONFIRM = "confirm"
    CORRECT = "correct"
    FEEDBACK = "feedback"


class SemanticContribution(StrEnum):
    BACKGROUND = "background"
    DETAIL = "detail"
    EVIDENCE = "evidence"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    DELIVERABLE = "deliverable"
    CRITERION = "criterion"


class SemanticModality(StrEnum):
    ASSERTED = "asserted"
    DESIRED = "desired"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"


class Explicitness(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    SPECULATIVE = "speculative"


class SegmentRelationType(StrEnum):
    ELABORATES = "elaborates"
    SUPPORTS = "supports"
    CONSTRAINS = "constrains"
    CONTRADICTS = "contradicts"
    CORRECTS = "corrects"
    DELIVERABLE_OF = "deliverable_of"
    VALIDATES = "validates"
    DEPENDS_ON = "depends_on"
    SHARES_CONTEXT = "shares_context"
    SCOPED_TO = "scoped_to"


class GoalCandidateStatus(StrEnum):
    EXPLICIT = "explicit"
    SPECULATIVE = "speculative"


class ObligationKind(StrEnum):
    ACKNOWLEDGE = "acknowledge"
    ANSWER = "answer"
    ANALYZE = "analyze"
    PROPOSE = "propose"
    PRODUCE_ARTIFACT = "produce_artifact"
    CLARIFY = "clarify"
    RETRIEVE = "retrieve"
    VERIFY = "verify"
    STAGE_SIDE_EFFECT = "stage_side_effect"
    REFUSE_OR_BOUND = "refuse_or_bound"


class ObligationProvenance(StrEnum):
    USER_EXPLICIT = "user_explicit"
    USER_CONTEXTUAL = "user_contextual"
    DIALOGUE_POLICY = "dialogue_policy"
    RUNTIME_SAFETY = "runtime_safety"
    JOURNEY_STATE = "journey_state"
    TOOL_RESULT = "tool_result"
    SPECULATIVE = "speculative"


class ObligationAuthorityCeiling(StrEnum):
    DIRECT_RESPONSE = "direct_response"
    READ_ONLY_ACTION = "read_only_action"
    APPROVAL_REQUIRED = "approval_required"


class ObligationDuration(StrEnum):
    ONE_OFF = "one_off"
    PERSISTENT = "persistent"
    CONTINUOUS = "continuous"


class ObligationCriticality(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class StateUpdateCommandType(StrEnum):
    """R6.0 特意保持精简且仅适用于候选项的命令白名单。

    生命周期命令（supersede、satisfy、defer、cancel）需要持久状态归约器，
    因此在此首版契约中有意不予接受。
    """

    ADD_GOAL = "add_goal"
    ATTACH_CONSTRAINT = "attach_constraint"
    ATTACH_PREFERENCE = "attach_preference"
    OPEN_QUESTION = "open_question"
    CREATE_OBLIGATION = "create_obligation"


class SemanticValidationCode(StrEnum):
    SCHEMA_INVALID = "schema_invalid"
    RAW_TURN_MISMATCH = "raw_turn_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    UNKNOWN_ANCHOR_SOURCE = "unknown_anchor_source"
    INVALID_ANCHOR_RANGE = "invalid_anchor_range"
    ANCHOR_EXCERPT_MISMATCH = "anchor_excerpt_mismatch"
    TRUSTED_ATTRIBUTION_MISMATCH = "trusted_attribution_mismatch"
    UNKNOWN_SEGMENT_REFERENCE = "unknown_segment_reference"
    UNKNOWN_GOAL_REFERENCE = "unknown_goal_reference"
    UNKNOWN_OBLIGATION_REFERENCE = "unknown_obligation_reference"
    EXPLICIT_GOAL_WITHOUT_USER_DEMAND = "explicit_goal_without_user_demand"
    SPECULATIVE_GOAL_ESCALATION = "speculative_goal_escalation"
    ORPHAN_EXPLICIT_DEMAND = "orphan_explicit_demand"
    UNATTACHED_CONSTRAINT = "unattached_constraint"
    GOAL_WITHOUT_OBLIGATION = "goal_without_obligation"
    OBLIGATION_DEPENDENCY_CYCLE = "obligation_dependency_cycle"
    INVALID_STATE_COMMAND = "invalid_state_command"


def _require_unique_ids(values: list[str] | tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class SemanticAnchor(_Contract):
    source: AnchorSource
    source_id: str = Field(pattern=_ID_PATTERN)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def _require_nonempty_range(self) -> 'SemanticAnchor':
        if self.end <= self.start:
            raise ValueError("anchor end must exceed start")
        return self


class FunctionalSegment(_Contract):
    segment_id: str = Field(pattern=_ID_PATTERN)
    anchors: tuple[SemanticAnchor, ...] = Field(min_length=1, max_length=4)
    communicative_functions: tuple[CommunicativeFunction, ...] = Field(min_length=1, max_length=4)
    semantic_contributions: tuple[SemanticContribution, ...] = Field(min_length=1, max_length=5)
    modalities: tuple[SemanticModality, ...] = Field(min_length=1, max_length=4)
    attribution: SemanticAttribution
    explicitness: Explicitness
    normalized_content: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("communicative_functions", "semantic_contributions", "modalities")
    @classmethod
    def _require_unique_dimensions(cls, values: tuple[StrEnum, ...]) -> tuple[StrEnum, ...]:
        if len(values) != len(set(values)):
            raise ValueError("semantic dimensions must not contain duplicates")
        return values


class SegmentRelation(_Contract):
    from_segment_id: str = Field(pattern=_ID_PATTERN)
    to_segment_id: str = Field(pattern=_ID_PATTERN)
    relation: SegmentRelationType
    evidence_segment_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("evidence_segment_ids")
    @classmethod
    def _require_unique_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique_ids(values, "relation evidence_segment_ids")
        return values

    @model_validator(mode="after")
    def _forbid_self_relation(self) -> 'SegmentRelation':
        if self.from_segment_id == self.to_segment_id:
            raise ValueError("segment relations cannot self-reference")
        return self


class GoalFrameCandidate(_Contract):
    goal_id: str = Field(pattern=_ID_PATTERN)
    outcome: str = Field(min_length=1, max_length=500)
    status: GoalCandidateStatus
    request_segment_ids: tuple[str, ...] = Field(default=(), max_length=8)
    question_segment_ids: tuple[str, ...] = Field(default=(), max_length=8)
    support_segment_ids: tuple[str, ...] = Field(default=(), max_length=12)
    preference_segment_ids: tuple[str, ...] = Field(default=(), max_length=8)
    constraint_segment_ids: tuple[str, ...] = Field(default=(), max_length=8)
    deliverables: tuple[str, ...] = Field(default=(), max_length=8)
    completion_criteria: tuple[str, ...] = Field(default=(), max_length=8)

    @field_validator(
        "request_segment_ids",
        "question_segment_ids",
        "support_segment_ids",
        "preference_segment_ids",
        "constraint_segment_ids",
    )
    @classmethod
    def _require_unique_segment_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique_ids(values, "goal segment ids")
        return values

    @model_validator(mode="after")
    def _explicit_goal_requires_demand_reference(self) -> 'GoalFrameCandidate':
        if self.status == GoalCandidateStatus.EXPLICIT and not (
            self.request_segment_ids or self.question_segment_ids
        ):
            raise ValueError("explicit goals require a request or question segment")
        return self


class ObligationCandidate(_Contract):
    obligation_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    kind: ObligationKind
    provenance: ObligationProvenance
    criticality: ObligationCriticality
    duration: ObligationDuration
    source_segment_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    satisfaction_criteria: str = Field(min_length=1, max_length=500)
    depends_on_obligation_ids: tuple[str, ...] = Field(default=(), max_length=8)
    authority_ceiling: ObligationAuthorityCeiling

    @field_validator("source_segment_ids", "depends_on_obligation_ids")
    @classmethod
    def _require_unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique_ids(values, "obligation reference ids")
        return values

    @model_validator(mode="after")
    def _forbid_self_dependency(self) -> 'ObligationCandidate':
        if self.obligation_id in self.depends_on_obligation_ids:
            raise ValueError("an obligation cannot depend on itself")
        return self


class StateUpdateCommand(_Contract):
    command: StateUpdateCommandType
    candidate_goal_id: str = Field(pattern=_ID_PATTERN)
    candidate_obligation_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    evidence_segment_ids: tuple[str, ...] = Field(min_length=1, max_length=12)

    @field_validator("evidence_segment_ids")
    @classmethod
    def _require_unique_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique_ids(values, "state command evidence ids")
        return values

    @model_validator(mode="after")
    def _validate_target_shape(self) -> 'StateUpdateCommand':
        requires_obligation = self.command == StateUpdateCommandType.CREATE_OBLIGATION
        if requires_obligation != (self.candidate_obligation_id is not None):
            raise ValueError("only create_obligation commands may carry candidate_obligation_id")
        return self


class SemanticCompilation(_Contract):
    """不可信且 Turn 局部的模型候选项。它不是持久状态。"""

    schema_version: Literal["semantic-compilation-v1"]
    raw_turn_id: str = Field(pattern=_ID_PATTERN)
    expected_revision: int = Field(ge=0)
    functional_segments: tuple[FunctionalSegment, ...] = Field(min_length=1, max_length=32)
    segment_relations: tuple[SegmentRelation, ...] = Field(default=(), max_length=48)
    goal_frames: tuple[GoalFrameCandidate, ...] = Field(default=(), max_length=16)
    obligations: tuple[ObligationCandidate, ...] = Field(default=(), max_length=24)
    state_update_commands: tuple[StateUpdateCommand, ...] = Field(default=(), max_length=40)

    @model_validator(mode="after")
    def _require_unique_object_ids(self) -> 'SemanticCompilation':
        _require_unique_ids([item.segment_id for item in self.functional_segments], "segment ids")
        _require_unique_ids([item.goal_id for item in self.goal_frames], "goal ids")
        _require_unique_ids([item.obligation_id for item in self.obligations], "obligation ids")
        return self


class SemanticCoverageReport(_Contract):
    explicit_demand_coverage: float = Field(ge=0.0, le=1.0)
    orphan_explicit_segment_ids: tuple[str, ...] = ()
    unsupported_goal_ids: tuple[str, ...] = ()
    unattached_constraint_segment_ids: tuple[str, ...] = ()
    goals_without_obligation_ids: tuple[str, ...] = ()


class SemanticCompilationValidationResult(_Contract):
    status: Literal["accepted", "rejected"]
    compilation: SemanticCompilation | None = None
    error_codes: tuple[SemanticValidationCode, ...] = ()
    coverage_report: SemanticCoverageReport


__all__ = [
    "AnchorSource",
    "CommunicativeFunction",
    "Explicitness",
    'FunctionalSegment',
    "GoalCandidateStatus",
    'GoalFrameCandidate',
    "ObligationAuthorityCeiling",
    'ObligationCandidate',
    "ObligationCriticality",
    "ObligationDuration",
    "ObligationKind",
    "ObligationProvenance",
    "SegmentRelationType",
    'SegmentRelation',
    'SemanticAnchor',
    "SemanticAttribution",
    'SemanticCompilation',
    'SemanticCompilationValidationResult',
    "SemanticContribution",
    'SemanticCoverageReport',
    "SemanticModality",
    "SemanticValidationCode",
    "StateUpdateCommandType",
    'StateUpdateCommand',
]
