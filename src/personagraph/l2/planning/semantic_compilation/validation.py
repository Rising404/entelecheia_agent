"""L2 语义编译的可信输入与失败关闭验证策略。

候选项/结果 DTO 位于 ``contracts``，使归约器和持久化层无需加载
此裁决策略即可依赖它们。本模块特意止步于模型调用、持久化、操作选择或图集成之前。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, model_validator

from .contracts import (
    AnchorSource,
    CommunicativeFunction,
    Explicitness,
    FunctionalSegment,
    GoalCandidateStatus,
    GoalFrameCandidate,
    ObligationCandidate,
    ObligationProvenance,
    SemanticAnchor,
    SemanticAttribution,
    SemanticCompilation,
    SemanticCompilationValidationResult,
    SemanticContribution,
    SemanticCoverageReport,
    SemanticValidationCode,
    StateUpdateCommandType,
    _Contract,
    _ID_PATTERN,
)


class VerifiedTypedEvent(_Contract):
    """未来编译器可引用的经 Host 验证类型化事件。"""

    event_id: str = Field(pattern=_ID_PATTERN)
    text: str = Field(min_length=1, max_length=4_000)
    attribution: SemanticAttribution


class TrustedAttributionSpan(_Contract):
    """锚点来源中 Host 已知的非用户指令材料。"""

    source: AnchorSource
    source_id: str = Field(pattern=_ID_PATTERN)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    attribution: Literal[
        SemanticAttribution.QUOTED_EXTERNAL,
        SemanticAttribution.ATTACHMENT,
        SemanticAttribution.TOOL_OBSERVATION,
        SemanticAttribution.HYPOTHETICAL,
    ]

    @model_validator(mode="after")
    def _require_nonempty_range(self) -> 'TrustedAttributionSpan':
        if self.end <= self.start:
            raise ValueError("trusted attribution span end must exceed start")
        return self


class SemanticCompilationInput(_Contract):
    """可信编译器输入，特意排除人格与记忆。"""

    schema_version: Literal[1] = 1
    raw_turn_id: str = Field(pattern=_ID_PATTERN)
    raw_turn_text: str = Field(min_length=1, max_length=24_000)
    expected_revision: int = Field(ge=0)
    verified_typed_events: tuple[VerifiedTypedEvent, ...] = Field(default=(), max_length=16)
    trusted_attribution_spans: tuple[TrustedAttributionSpan, ...] = Field(
        default=(), max_length=64
    )

    @model_validator(mode="after")
    def _validate_sources_and_spans(self) -> 'SemanticCompilationInput':
        event_ids = [event.event_id for event in self.verified_typed_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("verified typed event ids must be unique")

        source_texts = _input_source_texts(self)
        ranges_by_source: dict[tuple[AnchorSource, str], list[tuple[int, int]]] = {}
        for span in self.trusted_attribution_spans:
            text = source_texts.get((span.source, span.source_id))
            if text is None:
                raise ValueError("trusted attribution span references an unknown source")
            if span.end > len(text):
                raise ValueError("trusted attribution span exceeds source text")
            ranges_by_source.setdefault((span.source, span.source_id), []).append(
                (span.start, span.end)
            )
        for ranges in ranges_by_source.values():
            ranges.sort()
            if any(previous[1] > current[0] for previous, current in zip(ranges, ranges[1:])):
                raise ValueError("trusted attribution spans must not overlap")
        return self


def parse_and_validate_semantic_compilation(
    candidate: object,
    *,
    compile_input: SemanticCompilationInput,
) -> SemanticCompilationValidationResult:
    """解析不可信候选项，然后强制执行 Host 持有的语义检查。"""
    try:
        parsed = SemanticCompilation.model_validate(candidate)
    except ValidationError:
        return SemanticCompilationValidationResult(
            status="rejected",
            error_codes=(SemanticValidationCode.SCHEMA_INVALID,),
            coverage_report=_empty_coverage_report(),
        )
    return validate_semantic_compilation(parsed, compile_input=compile_input)


def validate_semantic_compilation(
    compilation: SemanticCompilation,
    *,
    compile_input: SemanticCompilationInput,
) -> SemanticCompilationValidationResult:
    """验证已解析候选项，但不调用模型或变更状态。"""
    errors: set[SemanticValidationCode] = set()
    if compilation.raw_turn_id != compile_input.raw_turn_id:
        errors.add(SemanticValidationCode.RAW_TURN_MISMATCH)
    if compilation.expected_revision != compile_input.expected_revision:
        errors.add(SemanticValidationCode.REVISION_MISMATCH)

    source_texts = _input_source_texts(compile_input)
    typed_event_attributions = {
        event.event_id: event.attribution for event in compile_input.verified_typed_events
    }
    segments = {segment.segment_id: segment for segment in compilation.functional_segments}
    for segment in segments.values():
        for anchor in segment.anchors:
            text = source_texts.get((anchor.source, anchor.source_id))
            if text is None:
                errors.add(SemanticValidationCode.UNKNOWN_ANCHOR_SOURCE)
                continue
            if anchor.end > len(text):
                errors.add(SemanticValidationCode.INVALID_ANCHOR_RANGE)
                continue
            if text[anchor.start:anchor.end] != anchor.excerpt:
                errors.add(SemanticValidationCode.ANCHOR_EXCERPT_MISMATCH)
            if anchor.source == AnchorSource.TYPED_EVENT:
                if typed_event_attributions[anchor.source_id] != segment.attribution:
                    errors.add(SemanticValidationCode.TRUSTED_ATTRIBUTION_MISMATCH)
            elif _overlaps_mismatched_trusted_span(anchor, segment.attribution, compile_input):
                errors.add(SemanticValidationCode.TRUSTED_ATTRIBUTION_MISMATCH)

    _validate_relations(compilation, segments, errors)
    coverage = _coverage_report(compilation, segments)
    if coverage.orphan_explicit_segment_ids:
        errors.add(SemanticValidationCode.ORPHAN_EXPLICIT_DEMAND)
    if coverage.unsupported_goal_ids:
        errors.add(SemanticValidationCode.EXPLICIT_GOAL_WITHOUT_USER_DEMAND)
    if coverage.unattached_constraint_segment_ids:
        errors.add(SemanticValidationCode.UNATTACHED_CONSTRAINT)
    if coverage.goals_without_obligation_ids:
        errors.add(SemanticValidationCode.GOAL_WITHOUT_OBLIGATION)

    _validate_goals(compilation, segments, errors)
    _validate_obligations(compilation, segments, errors)
    _validate_state_commands(compilation, segments, errors)

    ordered_errors = tuple(sorted(errors, key=str))
    return SemanticCompilationValidationResult(
        status="accepted" if not ordered_errors else "rejected",
        compilation=compilation if not ordered_errors else None,
        error_codes=ordered_errors,
        coverage_report=coverage,
    )


def _validate_relations(
    compilation: SemanticCompilation,
    segments: dict[str, FunctionalSegment],
    errors: set[SemanticValidationCode],
) -> None:
    for relation in compilation.segment_relations:
        references = (relation.from_segment_id, relation.to_segment_id, *relation.evidence_segment_ids)
        if any(reference not in segments for reference in references):
            errors.add(SemanticValidationCode.UNKNOWN_SEGMENT_REFERENCE)


def _validate_goals(
    compilation: SemanticCompilation,
    segments: dict[str, FunctionalSegment],
    errors: set[SemanticValidationCode],
) -> None:
    for goal in compilation.goal_frames:
        references = _goal_segment_ids(goal)
        if any(reference not in segments for reference in references):
            errors.add(SemanticValidationCode.UNKNOWN_SEGMENT_REFERENCE)
            continue
        if goal.status != GoalCandidateStatus.EXPLICIT:
            continue
        demand_ids = (*goal.request_segment_ids, *goal.question_segment_ids)
        if not any(_is_user_demand(segments[segment_id]) for segment_id in demand_ids):
            errors.add(SemanticValidationCode.EXPLICIT_GOAL_WITHOUT_USER_DEMAND)


def _validate_obligations(
    compilation: SemanticCompilation,
    segments: dict[str, FunctionalSegment],
    errors: set[SemanticValidationCode],
) -> None:
    goals = {goal.goal_id: goal for goal in compilation.goal_frames}
    obligations = {obligation.obligation_id: obligation for obligation in compilation.obligations}
    for obligation in obligations.values():
        goal = goals.get(obligation.goal_id)
        if goal is None:
            errors.add(SemanticValidationCode.UNKNOWN_GOAL_REFERENCE)
        elif goal.status == GoalCandidateStatus.SPECULATIVE:
            errors.add(SemanticValidationCode.SPECULATIVE_GOAL_ESCALATION)
        if any(segment_id not in segments for segment_id in obligation.source_segment_ids):
            errors.add(SemanticValidationCode.UNKNOWN_SEGMENT_REFERENCE)
        if any(dependency not in obligations for dependency in obligation.depends_on_obligation_ids):
            errors.add(SemanticValidationCode.UNKNOWN_OBLIGATION_REFERENCE)
        if obligation.provenance == ObligationProvenance.USER_EXPLICIT and not any(
            segment_id in segments and _is_user_demand(segments[segment_id])
            for segment_id in obligation.source_segment_ids
        ):
            errors.add(SemanticValidationCode.EXPLICIT_GOAL_WITHOUT_USER_DEMAND)
    if _has_obligation_cycle(obligations):
        errors.add(SemanticValidationCode.OBLIGATION_DEPENDENCY_CYCLE)


def _validate_state_commands(
    compilation: SemanticCompilation,
    segments: dict[str, FunctionalSegment],
    errors: set[SemanticValidationCode],
) -> None:
    goals = {goal.goal_id: goal for goal in compilation.goal_frames}
    obligations = {obligation.obligation_id: obligation for obligation in compilation.obligations}
    for command in compilation.state_update_commands:
        goal = goals.get(command.candidate_goal_id)
        if goal is None:
            errors.add(SemanticValidationCode.UNKNOWN_GOAL_REFERENCE)
        elif goal.status == GoalCandidateStatus.SPECULATIVE:
            errors.add(SemanticValidationCode.SPECULATIVE_GOAL_ESCALATION)
        if any(segment_id not in segments for segment_id in command.evidence_segment_ids):
            errors.add(SemanticValidationCode.UNKNOWN_SEGMENT_REFERENCE)
        if command.command == StateUpdateCommandType.CREATE_OBLIGATION:
            obligation = obligations.get(command.candidate_obligation_id or "")
            if obligation is None:
                errors.add(SemanticValidationCode.UNKNOWN_OBLIGATION_REFERENCE)
            elif obligation.goal_id != command.candidate_goal_id:
                errors.add(SemanticValidationCode.INVALID_STATE_COMMAND)
        elif command.candidate_obligation_id is not None:
            errors.add(SemanticValidationCode.INVALID_STATE_COMMAND)


def _coverage_report(
    compilation: SemanticCompilation,
    segments: dict[str, FunctionalSegment],
) -> SemanticCoverageReport:
    explicit_demands = {
        segment_id for segment_id, segment in segments.items() if _is_user_demand(segment)
    }
    covered = {
        segment_id
        for goal in compilation.goal_frames
        for segment_id in _goal_segment_ids(goal)
    }
    orphan = tuple(sorted(explicit_demands - covered))
    unsupported = tuple(sorted(
        goal.goal_id
        for goal in compilation.goal_frames
        if goal.status == GoalCandidateStatus.EXPLICIT
        and not any(
            segment_id in segments and _is_user_demand(segments[segment_id])
            for segment_id in (*goal.request_segment_ids, *goal.question_segment_ids)
        )
    ))
    attached_constraints = {
        segment_id
        for goal in compilation.goal_frames
        for segment_id in goal.constraint_segment_ids
    }
    unattached_constraints = tuple(sorted(
        segment_id
        for segment_id, segment in segments.items()
        if segment.attribution == SemanticAttribution.USER_INSTRUCTION
        and segment.explicitness == Explicitness.EXPLICIT
        and SemanticContribution.CONSTRAINT in segment.semantic_contributions
        and segment_id not in attached_constraints
    ))
    obligation_goal_ids = {obligation.goal_id for obligation in compilation.obligations}
    without_obligation = tuple(sorted(
        goal.goal_id
        for goal in compilation.goal_frames
        if goal.status == GoalCandidateStatus.EXPLICIT and goal.goal_id not in obligation_goal_ids
    ))
    coverage = 1.0 if not explicit_demands else len(explicit_demands & covered) / len(explicit_demands)
    return SemanticCoverageReport(
        explicit_demand_coverage=coverage,
        orphan_explicit_segment_ids=orphan,
        unsupported_goal_ids=unsupported,
        unattached_constraint_segment_ids=unattached_constraints,
        goals_without_obligation_ids=without_obligation,
    )


def _goal_segment_ids(goal: GoalFrameCandidate) -> tuple[str, ...]:
    return (
        *goal.request_segment_ids,
        *goal.question_segment_ids,
        *goal.support_segment_ids,
        *goal.preference_segment_ids,
        *goal.constraint_segment_ids,
    )


def _is_user_demand(segment: FunctionalSegment) -> bool:
    return (
        segment.attribution == SemanticAttribution.USER_INSTRUCTION
        and segment.explicitness == Explicitness.EXPLICIT
        and bool(
            set(segment.communicative_functions)
            & {
                CommunicativeFunction.QUESTION,
                CommunicativeFunction.REQUEST,
                CommunicativeFunction.COMMAND,
            }
        )
    )


def _overlaps_mismatched_trusted_span(
    anchor: SemanticAnchor,
    attribution: SemanticAttribution,
    compile_input: SemanticCompilationInput,
) -> bool:
    for span in compile_input.trusted_attribution_spans:
        if (span.source, span.source_id) != (anchor.source, anchor.source_id):
            continue
        overlaps = anchor.start < span.end and span.start < anchor.end
        if overlaps and (
            attribution != span.attribution
            or anchor.start < span.start
            or anchor.end > span.end
        ):
            return True
    return False


def _has_obligation_cycle(obligations: dict[str, ObligationCandidate]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(obligation_id: str) -> bool:
        if obligation_id in visiting:
            return True
        if obligation_id in visited:
            return False
        visiting.add(obligation_id)
        for dependency in obligations[obligation_id].depends_on_obligation_ids:
            if dependency in obligations and visit(dependency):
                return True
        visiting.remove(obligation_id)
        visited.add(obligation_id)
        return False

    return any(visit(obligation_id) for obligation_id in obligations)


def _input_source_texts(
    compile_input: SemanticCompilationInput,
) -> dict[tuple[AnchorSource, str], str]:
    return {
        (AnchorSource.RAW_TURN, compile_input.raw_turn_id): compile_input.raw_turn_text,
        **{
            (AnchorSource.TYPED_EVENT, event.event_id): event.text
            for event in compile_input.verified_typed_events
        },
    }


def _empty_coverage_report() -> SemanticCoverageReport:
    return SemanticCoverageReport(explicit_demand_coverage=0.0)
