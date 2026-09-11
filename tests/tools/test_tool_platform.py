from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from personagraph.tools.catalog import CatalogConflictError, CatalogStatus, ToolCatalog, ToolKey, ToolResolutionError
from personagraph.tools.contracts import (
    ExecutionOutcome,
    ExecutionStatus,
    ToolError,
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.execution import (
    ResolvedInvocation,
    ToolExecutor,
    is_known_retryable_technical_failure,
)
from personagraph.tools.policy import (
    TOOL_POLICY_VERSION,
    AuthorityFacts,
    BudgetFacts,
    PolicyDisposition,
    PolicyRequest,
    ScopeGrant,
    ToolPolicyCore,
)
from personagraph.tools.registration import ExecutionMode, ToolExecutionProfile, ToolRegistration
from personagraph.tools.schema_validation import SchemaCompilationError


def _registration(
    tool_id: str = "tool",
    *,
    handler=None,
    input_schema=None,
    output_schema=None,
    effects=None,
    source: ToolSourceDescriptor | None = None,
    execution: ToolExecutionProfile | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="1.0.0",
            name=tool_id,
            description="A test tool.",
            input_schema=input_schema or {"type": "object", "properties": {"value": {"type": "string"}}},
            output_schema=output_schema or {"type": "object", "required": ["value"], "properties": {"value": {"type": "string"}}},
            catalog_tags=("read",),
        ),
        implementation_version="impl-1",
        source=source or ToolSourceDescriptor(ToolSourceKind.LOCAL, "test"),
        handler=handler or (lambda payload: {"value": payload.get("value", "ok")}),
        effect_profile=effects
        or ToolEffectProfile((EffectDescriptor(EffectResource.MEMORY, EffectAction.READ, EffectScopeKind.LOCAL),)),
        execution_profile=execution or ToolExecutionProfile(),
    )


def test_tool_contract_rejects_unknown_tags_and_legacy_fields():
    with pytest.raises(ValueError, match="unknown catalog tags"):
        ToolSpec("tool", "1.0.0", "name", "description", {}, {}, ("arbitrary",))
    with pytest.raises(TypeError):
        ToolSpec(  # type: ignore[call-arg]
            "tool", "1.0.0", "name", "description", {}, {}, risk_level="low"
        )


def test_registration_compiles_complete_json_schema_at_registration_time():
    with pytest.raises(SchemaCompilationError):
        _registration(input_schema={"type": 123})


def test_catalog_snapshot_keeps_old_registration_after_explicit_replace():
    first = _registration(handler=lambda payload: {"value": "first"})
    replacement = _registration(handler=lambda payload: {"value": "replacement"})
    catalog = ToolCatalog()
    catalog.register(first)
    snapshot = catalog.snapshot()
    catalog.register(replacement, replace=True, expected_revision=catalog.revision)

    assert snapshot.resolve("tool").handler({})["value"] == "first"
    assert catalog.snapshot().resolve("tool").handler({})["value"] == "replacement"
    assert catalog.audit_log()[-1].action == "replace"


def test_catalog_cas_and_lifecycle_prevent_new_disabled_resolution():
    catalog = ToolCatalog()
    catalog.register(_registration())
    key = ToolKey("tool", "1.0.0")
    with pytest.raises(CatalogConflictError):
        catalog.set_status(key, CatalogStatus.DISABLED, expected_revision=0)
    catalog.set_status(key, CatalogStatus.DISABLED, expected_revision=catalog.revision)
    with pytest.raises(ToolResolutionError):
        catalog.snapshot().resolve("tool")
    catalog.register(_registration(handler=lambda payload: {"value": "new"}), replace=True, expected_revision=catalog.revision)
    with pytest.raises(ToolResolutionError):
        catalog.snapshot().resolve("tool")
    catalog.set_status(key, CatalogStatus.RETIRED, expected_revision=catalog.revision)
    with pytest.raises(Exception, match="historical reference"):
        catalog.purge_retired(key, reference_guard=lambda entry: True, expected_revision=catalog.revision)
    catalog.purge_retired(key, reference_guard=lambda entry: False, expected_revision=catalog.revision)


def test_executor_validates_output_and_deadline_before_handler():
    called = []
    invalid = _registration(handler=lambda payload: {"wrong": "shape"})
    out = ToolExecutor().execute(ResolvedInvocation(invalid, {"value": "x"}))
    assert out.status is ExecutionStatus.FAILED
    assert out.error and out.error.code == "invalid_tool_output"

    expired = _registration(handler=lambda payload: called.append(payload) or {"value": "ok"})
    out = ToolExecutor(clock=lambda: 50.0).execute(ResolvedInvocation(expired, {"value": "x"}, deadline_monotonic=49.0))
    assert out.status is ExecutionStatus.TIMED_OUT
    assert called == []


def test_executor_accepts_valid_json_output_and_freezes_only_the_validated_result():
    registration = _registration(
        handler=lambda payload: {
            "value": payload["value"],
            "nested": {"items": ["one", "two"]},
        }
    )

    out = ToolExecutor().execute(ResolvedInvocation(registration, {"value": "ok"}))

    assert out.status is ExecutionStatus.SUCCEEDED
    assert out.to_dict()["result"] == {
        "value": "ok",
        "nested": {"items": ["one", "two"]},
    }
    assert out.result is not None
    with pytest.raises(TypeError):
        out.result["value"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        out.result["nested"]["items"][0] = "changed"  # type: ignore[index]


def test_executor_reports_non_json_handler_output_as_output_error():
    registration = _registration(handler=lambda payload: {"value": object()})

    out = ToolExecutor().execute(ResolvedInvocation(registration, {"value": "ok"}))

    assert out.status is ExecutionStatus.FAILED
    assert out.error and out.error.code == "invalid_tool_output"
    assert out.error.message == "Tool output does not satisfy its JSON Schema."


def test_executor_timeout_signals_only_its_own_call_and_handler_acknowledges():
    from personagraph.tools.execution_context import (
        ToolInvocationCancelled,
        current_tool_execution,
    )

    parent_cancel = threading.Event()
    acknowledged = threading.Event()
    captured = []
    now = [100.0]

    def controlled(_payload):
        control = current_tool_execution()
        assert control is not None
        captured.append(control)
        now[0] = 102.0
        assert control.cancellation_event.wait(timeout=1.0)
        try:
            control.checkpoint()
        except ToolInvocationCancelled:
            acknowledged.set()
            raise

    registration = _registration(
        handler=controlled,
        execution=ToolExecutionProfile(default_timeout_s=1.0),
    )
    outcome = ToolExecutor(clock=lambda: now[0]).execute(
        ResolvedInvocation(registration, {}, cancellation_event=parent_cancel)
    )

    assert outcome.status is ExecutionStatus.TIMED_OUT
    assert acknowledged.wait(timeout=1.0)
    assert not parent_cancel.is_set()
    assert captured[0].deadline_monotonic == 101.0
    assert captured[0].snapshot()["cancellation_acknowledged"] is True
    assert current_tool_execution() is None


def test_executor_does_not_publish_a_result_completed_after_its_deadline():
    now = [100.0]

    def late(_payload):
        now[0] = 102.0
        return {"value": "late"}

    outcome = ToolExecutor(clock=lambda: now[0]).execute(
        ResolvedInvocation(
            _registration(
                handler=late,
                execution=ToolExecutionProfile(default_timeout_s=1.0),
            ),
            {},
        )
    )

    assert outcome.status is ExecutionStatus.TIMED_OUT
    assert outcome.result is None


def test_input_validation_time_never_extends_the_callers_absolute_deadline():
    from personagraph.tools.execution_context import current_tool_execution
    from personagraph.tools.schema_validation import ToolSchemaCompiler

    now = [100.0]
    deadlines = []

    class AdvancingSchemas(ToolSchemaCompiler):
        def validate_input(self, schema, arguments):
            now[0] = 102.0
            return super().validate_input(schema, arguments)

    def handler(_payload):
        deadlines.append(current_tool_execution().deadline_monotonic)
        return {"value": "ok"}

    outcome = ToolExecutor(schemas=AdvancingSchemas(), clock=lambda: now[0]).execute(
        ResolvedInvocation(_registration(handler=handler), {}, deadline_monotonic=103.0)
    )
    assert outcome.status is ExecutionStatus.SUCCEEDED
    assert deadlines == [103.0]


def test_executor_nested_call_does_not_inherit_parent_local_cancellation():
    from personagraph.tools.execution_context import current_tool_execution

    controls = []

    def inner(_payload):
        controls.append(current_tool_execution())
        return {"value": "inner"}

    def outer(_payload):
        controls.append(current_tool_execution())
        result = ToolExecutor().execute(ResolvedInvocation(_registration(handler=inner), {}))
        assert result.status is ExecutionStatus.SUCCEEDED
        return {"value": "outer"}

    result = ToolExecutor().execute(ResolvedInvocation(_registration(handler=outer), {}))
    assert result.status is ExecutionStatus.SUCCEEDED
    assert all(control is not None for control in controls)
    assert controls[0] is not controls[1]
    assert controls[0].cancellation_event is not controls[1].cancellation_event


def test_executor_passes_host_call_identity_without_exposing_it_as_arguments():
    from personagraph.tools.execution_context import current_tool_execution

    captured = []

    def handler(payload):
        captured.append((current_tool_execution().logical_tool_call_id, payload))
        return {"value": "ok"}

    outcome = ToolExecutor().execute(ResolvedInvocation(
        _registration(handler=handler),
        {"value": "input"},
        logical_tool_call_id="host-tool-call-1",
    ))

    assert outcome.status is ExecutionStatus.SUCCEEDED
    assert captured == [("host-tool-call-1", {"value": "input"})]
    assert current_tool_execution() is None


@pytest.mark.parametrize("identity", ["", " ", 123])
def test_tool_execution_identity_rejects_invalid_host_values(identity):
    from personagraph.tools.execution_context import ToolExecutionContext

    with pytest.raises(ValueError, match="logical_tool_call_id"):
        ToolExecutionContext(deadline_monotonic=None, logical_tool_call_id=identity)


def test_tool_settlement_observer_sees_schema_failure_not_handler_success():
    from personagraph.tools.execution_context import current_tool_execution

    settled = []

    def handler(_payload):
        current_tool_execution().on_settled(settled.append)
        assert settled == []
        return {"wrong": "output"}

    outcome = ToolExecutor().execute(ResolvedInvocation(_registration(handler=handler), {}))
    assert outcome.status is ExecutionStatus.FAILED
    assert settled == ["failed"]


def test_cancelled_tool_audit_waits_for_worker_cleanup_without_blocking_return():
    from personagraph.tools.execution_context import current_tool_execution

    now = [100.0]
    release = threading.Event()
    recorded = threading.Event()
    settled = []

    def handler(_payload):
        control = current_tool_execution()
        control.on_settled(lambda status: (settled.append((status, control.snapshot())), recorded.set()))
        now[0] = 102.0
        assert release.wait(timeout=1.0)
        return {"value": "late"}

    try:
        outcome = ToolExecutor(clock=lambda: now[0]).execute(ResolvedInvocation(
            _registration(handler=handler, execution=ToolExecutionProfile(default_timeout_s=1.0)), {},
        ))
        assert outcome.status is ExecutionStatus.TIMED_OUT
        assert settled == []
    finally:
        release.set()
    assert recorded.wait(timeout=1.0)
    assert len(settled) == 1
    assert settled[0][0] == "timed_out"
    assert settled[0][1]["cancellation_acknowledged"] is True
    assert settled[0][1]["handler_finished"] is True


@pytest.mark.parametrize("action", (EffectAction.READ, EffectAction.SEARCH))
def test_executor_reports_known_timeout_for_uncancellable_read_only_sync_call(
    action,
):
    started = threading.Event()

    def slow(payload):
        started.set()
        time.sleep(0.08)
        return {"value": "late"}

    registration = _registration(
        handler=slow,
        effects=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.NETWORK,
                    action,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution=ToolExecutionProfile(default_timeout_s=0.01),
    )
    out = ToolExecutor().execute(ResolvedInvocation(registration, {"value": "x"}))

    assert started.is_set()
    assert out.status is ExecutionStatus.TIMED_OUT
    assert "outcome_known" not in out.to_dict()
    assert out.error and out.error.code == "execution_timeout"


def test_executor_reserves_completion_unconfirmed_for_effectful_sync_call():
    started = threading.Event()

    def slow(payload):
        started.set()
        time.sleep(0.08)
        return {"value": "late"}

    registration = _registration(
        handler=slow,
        effects=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.FILESYSTEM,
                    EffectAction.UPDATE,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution=ToolExecutionProfile(default_timeout_s=0.01),
    )
    out = ToolExecutor().execute(
        ResolvedInvocation(registration, {"value": "x"})
    )

    assert started.is_set()
    assert out.status is ExecutionStatus.COMPLETION_UNCONFIRMED
    assert "outcome_known" not in out.to_dict()
    assert out.error and out.error.code == "execution_timeout"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_exception", "technical"),
            ),
            True,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("invalid_tool_output", "technical"),
            ),
            True,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("tool_output_too_large", "technical"),
            ),
            True,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.TIMED_OUT,
                error=ToolError("execution_timeout", "known timeout"),
            ),
            True,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.REJECTED,
                error=ToolError("invalid_tool_input", "rejected"),
            ),
            False,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.CANCELLED,
                error=ToolError("execution_cancelled", "cancelled"),
            ),
            False,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.COMPLETION_UNCONFIRMED,
                error=ToolError("execution_timeout", "uncertain"),
            ),
            False,
        ),
        (
            ExecutionOutcome(
                ExecutionStatus.FAILED,
                error=ToolError("business_rule_failed", "business"),
            ),
            False,
        ),
    ],
)
def test_known_retryable_technical_failure_uses_closed_status_and_code_allowlist(
    outcome,
    expected,
):
    assert is_known_retryable_technical_failure(outcome) is expected


def test_executor_cancels_async_handler_and_proves_acknowledgement():
    async def slow(payload):
        await asyncio.sleep(1)
        return {"value": "late"}

    registration = _registration(
        handler=slow,
        execution=ToolExecutionProfile(default_timeout_s=0.01, execution_mode=ExecutionMode.ASYNC),
    )
    out = ToolExecutor().execute(ResolvedInvocation(registration, {"value": "x"}))

    assert out.status is ExecutionStatus.TIMED_OUT
    assert "outcome_known" not in out.to_dict()


def _request_for(effect: EffectDescriptor, *, arguments=None, authority=None, budget=None) -> PolicyRequest:
    registration = _registration(effects=ToolEffectProfile((effect,)))
    return PolicyRequest.from_registration(
        registration,
        arguments or {},
        authority=authority,
        budget=budget,
    )


def test_policy_version_is_owned_by_the_policy_core() -> None:
    request = PolicyRequest(normalized_arguments={}, effects=None)

    assert request.policy_version == TOOL_POLICY_VERSION
    with pytest.raises(ValueError, match="unsupported Tool policy version"):
        PolicyRequest(
            normalized_arguments={},
            effects=None,
            policy_version="runtime-selected-version",
        )


@pytest.mark.parametrize("scope_kind", [EffectScopeKind.SESSION, EffectScopeKind.EXECUTION])
def test_internal_result_updates_do_not_need_human_approval(scope_kind):
    request = _request_for(EffectDescriptor(
        EffectResource.RUNTIME_STATE, EffectAction.UPDATE, scope_kind,
        default_scope="current-execution", data_egress=DataEgress.NONE,
    ))

    assert ToolPolicyCore().evaluate(request).disposition is PolicyDisposition.ALLOW


def test_policy_core_uses_injected_effect_authority_and_budget_facts():
    policy = ToolPolicyCore()
    assert policy.evaluate(PolicyRequest(normalized_arguments={}, effects=None)).disposition is PolicyDisposition.DENY
    local_read = _request_for(EffectDescriptor(EffectResource.MEMORY, EffectAction.READ, EffectScopeKind.LOCAL))
    assert policy.evaluate(local_read).disposition is PolicyDisposition.ALLOW

    workspace_read = _request_for(
        EffectDescriptor(EffectResource.FILESYSTEM, EffectAction.READ, EffectScopeKind.WORKSPACE, default_scope="workspace:a")
    )
    decision = policy.evaluate(workspace_read)
    assert decision.disposition is PolicyDisposition.AUTHORIZATION_REQUIRED

    granted = _request_for(
        EffectDescriptor(EffectResource.FILESYSTEM, EffectAction.READ, EffectScopeKind.WORKSPACE, default_scope="workspace:a"),
        authority=AuthorityFacts((ScopeGrant(EffectResource.FILESYSTEM, EffectAction.READ, EffectScopeKind.WORKSPACE, "workspace:a"),)),
    )
    assert policy.evaluate(granted).disposition is PolicyDisposition.ALLOW

    write = _request_for(EffectDescriptor(EffectResource.FILESYSTEM, EffectAction.UPDATE, EffectScopeKind.LOCAL))
    assert policy.evaluate(write).disposition is PolicyDisposition.APPROVAL_REQUIRED

    sensitive_egress = _request_for(
        EffectDescriptor(
            EffectResource.NETWORK,
            EffectAction.TRANSMIT,
            EffectScopeKind.REMOTE_DOMAIN,
            data_egress=DataEgress.CONTENT,
            egress_arguments=("query",),
        ),
        arguments={"query": "api_key=sk-abcdefghijklmnop"},
    )
    decision = policy.evaluate(sensitive_egress)
    assert decision.disposition is PolicyDisposition.ALLOW
    assert decision.effects[0].sensitive_egress is True
    assert not decision.approval_required

    exhausted = _request_for(
        EffectDescriptor(EffectResource.MEMORY, EffectAction.READ, EffectScopeKind.LOCAL),
        budget=BudgetFacts(remaining_tool_calls=0),
    )
    assert policy.evaluate(exhausted).disposition is PolicyDisposition.DENY


def test_tool_platform_and_catalog_have_one_way_dependencies() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "personagraph" / "tools"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "contracts.py",
        "effects.py",
        "execution.py",
        "execution_context.py",
        "policy.py",
        "registration.py",
        "schema_validation.py",
    }
    files = sorted(root.glob("*.py")) + sorted((root / "catalog").rglob("*.py"))
    forbidden = {"runtime", "langgraph", "session", "insession_task"}
    for path in files:
        module = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = "." * node.level + (node.module or "")
                imported.update(
                    f"{prefix}.{alias.name}" if prefix else alias.name
                    for alias in node.names
                )
        reverse_dependencies = {
            name
            for name in imported
            if set(name.split(".")) & forbidden
        }
        assert not reverse_dependencies, (path, reverse_dependencies)


def test_core_and_live_catalog_imports_do_not_load_catalog_control_plane() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    probe = """
import json
import sys

import personagraph.tools.contracts
after_contracts = sorted(sys.modules)
import personagraph.tools.catalog
after_catalog = sorted(sys.modules)
print(json.dumps({"after_contracts": after_contracts, "after_catalog": after_catalog}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(completed.stdout)
    forbidden = (
        "personagraph.tools.catalog.default_profile",
        "personagraph.tools.catalog.materialization",
        "personagraph.tools.catalog.persistence",
        "personagraph.tools.catalog.revocation",
        "personagraph.tools.catalog.snapshots",
        "personagraph.tools.catalog.trusted_factories",
    )
    for stage in ("after_contracts", "after_catalog"):
        assert not {
            module_name
            for module_name in loaded[stage]
            if module_name.startswith(forbidden)
        }, stage
