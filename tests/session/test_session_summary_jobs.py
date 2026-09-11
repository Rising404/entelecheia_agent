from __future__ import annotations

from personagraph.session.session_summary import SessionSummaryGenerationError
from personagraph.session.session_summary_jobs import summary_generation_failure_code


def test_summary_generation_errors_keep_their_stable_persistence_categories() -> None:
    expected_categories = {
        "SUMMARY_BOUNDARY_UNAVAILABLE": "SUMMARY_BOUNDARY_UNAVAILABLE",
        "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE": (
            "SUMMARY_NO_FRESH_SOURCE_FOR_STALE_STATE"
        ),
        "SUMMARY_INPUT_OVER_BUDGET": "SUMMARY_INPUT_OVER_BUDGET",
        "SUMMARY_EMPTY_OUTPUT": "SUMMARY_MODEL_OUTPUT_INVALID",
        "SUMMARY_MODEL_TIMEOUT": "SUMMARY_MODEL_TIMEOUT",
        "SUMMARY_MODEL_CONFIGURATION_FAILURE": (
            "SUMMARY_MODEL_CONFIGURATION_FAILURE"
        ),
    }

    for code, expected in expected_categories.items():
        error = SessionSummaryGenerationError(
            code,
            "safe test message",
            retryable=(code == "SUMMARY_EMPTY_OUTPUT"),
        )
        assert summary_generation_failure_code(error) == expected
