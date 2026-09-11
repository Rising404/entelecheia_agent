"""执行本地 findings ToolCall 的 Host dispatcher。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from ...persistent_turn_content.findings import (
    ExecutionFindingsMutationCommand,
    ExecutionFindingsMutationIdentityCollision,
    ExecutionFindingsMutationResult,
    ExecutionFindingsOwnerClosed,
    ExecutionFindingsPersistenceError,
    ExecutionFindingsQuotaExceeded,
    ExecutionFindingsRevisionConflict,
    ExecutionFindingsScopeInvalid,
    ExecutionFindingsSourceReferenceInvalid,
    ExecutionFindingsStoredAuthorityCorrupt,
    RecordExecutionFinding,
    RetractExecutionFinding,
    SupersedeExecutionFinding,
    derive_execution_findings_mutation_id,
)
from ..contracts import ExecutionOutcome, ExecutionStatus, ToolError
from .contracts import (
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
    REVISE_EXECUTION_FINDING_TOOL_ID,
)


class ExecutionFindingsMutationStore(Protocol):
    def apply_execution_findings_mutation(
        self,
        *,
        command: ExecutionFindingsMutationCommand,
    ) -> ExecutionFindingsMutationResult: ...


def execute_execution_findings_tool(
    *,
    store: ExecutionFindingsMutationStore,
    tool_id: str,
    normalized_arguments: Mapping[str, Any],
    ledger_id: str,
    writer_unit_id: str,
    writer_tool_call_id: str,
) -> ExecutionOutcome:
    """验证模型字段、注入 Host 标识，并以原子方式变更账本。"""

    try:
        expected_revision = normalized_arguments["expected_ledger_revision"]
        raw_items = normalized_arguments["items"]
        if not isinstance(raw_items, list):
            raise ValueError("items must be a list")
        if tool_id == RECORD_EXECUTION_FINDINGS_TOOL_ID:
            items = tuple(
                RecordExecutionFinding.model_validate(
                    {"operation": "record", **_mapping(item)}
                )
                for item in raw_items
            )
        elif tool_id == REVISE_EXECUTION_FINDING_TOOL_ID:
            parsed: list[
                SupersedeExecutionFinding | RetractExecutionFinding
            ] = []
            for item in raw_items:
                payload = _mapping(item)
                operation = payload.get("operation")
                if operation == "supersede":
                    parsed.append(
                        SupersedeExecutionFinding.model_validate(payload)
                    )
                elif operation == "retract":
                    parsed.append(RetractExecutionFinding.model_validate(payload))
                else:
                    raise ValueError("revision operation is unsupported")
            items = tuple(parsed)
        else:
            raise ValueError("tool is not an execution-findings operation")
        command = ExecutionFindingsMutationCommand(
            ledger_id=ledger_id,
            mutation_id=derive_execution_findings_mutation_id(
                writer_tool_call_id=writer_tool_call_id
            ),
            expected_ledger_revision=expected_revision,
            writer_unit_id=writer_unit_id,
            writer_tool_call_id=writer_tool_call_id,
            items=items,
        )
        result = store.apply_execution_findings_mutation(command=command)
    except ExecutionFindingsRevisionConflict as exc:
        return _rejected(
            "execution_findings_revision_conflict",
            "The findings ledger changed; retry from the latest active projection.",
            {"expected_revision": exc.expected, "actual_revision": exc.actual},
        )
    except ExecutionFindingsQuotaExceeded as exc:
        return _rejected("execution_findings_quota_exceeded", str(exc))
    except ExecutionFindingsSourceReferenceInvalid as exc:
        return _rejected("execution_findings_source_invalid", str(exc))
    except ExecutionFindingsScopeInvalid as exc:
        return _rejected("execution_findings_scope_invalid", str(exc))
    except ExecutionFindingsOwnerClosed as exc:
        return _rejected("execution_findings_owner_closed", str(exc))
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        return _rejected(
            "execution_findings_input_invalid",
            "The findings mutation does not satisfy its typed contract.",
            {"reason": str(exc)[:500]},
        )
    except (
        ExecutionFindingsStoredAuthorityCorrupt,
        ExecutionFindingsMutationIdentityCollision,
        ExecutionFindingsPersistenceError,
    ) as exc:
        return ExecutionOutcome(
            ExecutionStatus.FAILED,
            error=ToolError(
                code="execution_findings_internal_failure",
                message="The Host could not safely mutate execution findings.",
                details={"reason": str(exc)[:500]},
            ),
        )
    return ExecutionOutcome.succeeded(
        {
            "schema_version": "execution-findings-tool-result-v1",
            "status": "applied",
            "ledger_revision": result.ledger.revision,
            "affected_entry_ids": list(result.receipt.affected_entry_ids),
            "active_projection": result.active_projection.model_dump(mode="json"),
            "replayed": result.replayed,
        }
    )


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("findings mutation item must be an object")
    return {str(key): item for key, item in value.items()}


def _rejected(
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> ExecutionOutcome:
    return ExecutionOutcome.rejected(
        ToolError(code=code, message=message, details=details or {})
    )


__all__ = [
    'ExecutionFindingsMutationStore',
    "execute_execution_findings_tool",
]
