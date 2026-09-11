"""Stable identities for the execution-findings Tool family.

The Runtime owns the findings ledger lifecycle and mutation authority.  The Tool
package owns the model-visible capability identities, so importing a Tool contract
never requires loading Runtime state machinery.
"""

from __future__ import annotations


RECORD_EXECUTION_FINDINGS_TOOL_ID = "record_execution_findings"
REVISE_EXECUTION_FINDING_TOOL_ID = "revise_execution_finding"
EXECUTION_FINDINGS_TOOL_IDS = frozenset(
    {
        RECORD_EXECUTION_FINDINGS_TOOL_ID,
        REVISE_EXECUTION_FINDING_TOOL_ID,
    }
)


__all__ = [
    "EXECUTION_FINDINGS_TOOL_IDS",
    "RECORD_EXECUTION_FINDINGS_TOOL_ID",
    "REVISE_EXECUTION_FINDING_TOOL_ID",
]
