"""Safe model projection for execution-findings mutation receipts."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_TOOL_RESULT_SCHEMA = "execution-findings-tool-result-v1"
_COMPACTED_TOOL_RESULT_SCHEMA = (
    "execution-findings-tool-result-reference-v1"
)
_COMPACTED_PROJECTION_SCHEMA = (
    "execution-findings-active-projection-reference-v1"
)
_ENTRY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


def project_execution_findings_tool_output_for_model(
    output: object,
) -> object:
    """Remove the duplicated findings snapshot from one model-facing receipt.

    The durable ToolResult keeps the exact mutation outcome.  The next model
    request receives the current active projection through its dedicated
    ``execution_findings`` field, so replaying every historical snapshot would
    bypass FIFO eviction and grow the prompt quadratically.
    """

    if output is None:
        return output
    if not isinstance(output, Mapping):
        return {
            "schema_version": _COMPACTED_TOOL_RESULT_SCHEMA,
            "compacted": True,
        }

    active_projection = output.get("active_projection")
    projection_sha256 = (
        active_projection.get("projection_sha256")
        if isinstance(active_projection, Mapping)
        else None
    )
    compacted: dict[str, Any] = {
        "schema_version": _COMPACTED_PROJECTION_SCHEMA,
        "compacted": True,
    }
    if (
        isinstance(projection_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", projection_sha256)
    ):
        compacted["projection_sha256"] = projection_sha256

    # This is deliberately an allowlist.  A malformed historical result, or a
    # future result with an additional ledger/claim field, must not bypass the
    # bounded current projection merely because the field is unfamiliar here.
    projected: dict[str, Any] = {
        "schema_version": _COMPACTED_TOOL_RESULT_SCHEMA,
        "compacted": True,
        "active_projection": compacted,
    }
    if output.get("schema_version") == _TOOL_RESULT_SCHEMA:
        projected["source_schema_version"] = _TOOL_RESULT_SCHEMA
    if output.get("status") == "applied":
        projected["status"] = "applied"
    ledger_revision = output.get("ledger_revision")
    if (
        isinstance(ledger_revision, int)
        and not isinstance(ledger_revision, bool)
        and ledger_revision >= 1
    ):
        projected["ledger_revision"] = ledger_revision
    affected_entry_ids = output.get("affected_entry_ids")
    if (
        isinstance(affected_entry_ids, (list, tuple))
        and 1 <= len(affected_entry_ids) <= 4
        and all(
            isinstance(item, str) and _ENTRY_ID.fullmatch(item)
            for item in affected_entry_ids
        )
        and len(set(affected_entry_ids)) == len(affected_entry_ids)
    ):
        projected["affected_entry_ids"] = list(affected_entry_ids)
    if isinstance(output.get("replayed"), bool):
        projected["replayed"] = output["replayed"]
    return projected


__all__ = ["project_execution_findings_tool_output_for_model"]
