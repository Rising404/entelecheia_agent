"""冷态语义编译候选契约的边界覆盖。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.planning.semantic_compilation import (
    contracts as candidate_contracts,
)


def test_candidate_contracts_keep_schema_invariants_and_frozen_results() -> None:
    with pytest.raises(ValidationError, match="explicit goals require"):
        candidate_contracts.GoalFrameCandidate(
            goal_id="goal_1",
            outcome="Provide an analysis",
            status=candidate_contracts.GoalCandidateStatus.EXPLICIT,
        )
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        candidate_contracts.ObligationCandidate(
            obligation_id="obligation_1",
            goal_id="goal_1",
            kind=candidate_contracts.ObligationKind.ANALYZE,
            provenance=candidate_contracts.ObligationProvenance.USER_EXPLICIT,
            criticality=candidate_contracts.ObligationCriticality.NORMAL,
            duration=candidate_contracts.ObligationDuration.ONE_OFF,
            source_segment_ids=("segment_1",),
            satisfaction_criteria="Explain the analysis",
            depends_on_obligation_ids=("obligation_1",),
            authority_ceiling=candidate_contracts.ObligationAuthorityCeiling.DIRECT_RESPONSE,
        )
    with pytest.raises(ValidationError, match="only create_obligation"):
        candidate_contracts.StateUpdateCommand(
            command=candidate_contracts.StateUpdateCommandType.ADD_GOAL,
            candidate_goal_id="goal_1",
            candidate_obligation_id="obligation_1",
            evidence_segment_ids=("segment_1",),
        )

    result = candidate_contracts.SemanticCompilationValidationResult(
        status="rejected",
        error_codes=(candidate_contracts.SemanticValidationCode.SCHEMA_INVALID,),
        coverage_report=candidate_contracts.SemanticCoverageReport(
            explicit_demand_coverage=0.0,
        ),
    )
    with pytest.raises(ValidationError):
        result.status = "accepted"  # type: ignore[misc]
