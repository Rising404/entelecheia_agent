from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from personagraph.persistent_turn_content.findings import (
    EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    ExecutionFindingSourceRef,
    ExecutionFindingsQuota,
)
from personagraph.l2.task_execution.tool_bridge.execution_findings_catalog import (
    augment_execution_findings_tool_runtime,
    catalog_exposes_execution_findings_tools,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.findings import (
    ExecutionFindingsBindingFacts,
    build_execution_findings_tool_bindings,
    build_execution_findings_tool_definition_manifest,
    derive_execution_findings_source_fingerprint,
)
from personagraph.tools.findings.execution_findings_tools import (
    EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION,
    build_execution_findings_tool_registrations,
)


class _CustomBridgeWithoutCatalogRebind:
    supports_protected_recovery = False


def test_findings_catalog_uses_native_refs_and_optional_host_revision_and_scopes():
    record, revise = build_execution_findings_tool_registrations()
    for spec in (record.spec, revise.spec):
        schema = spec.to_dict()["input_schema"]
        assert "expected_ledger_revision" not in schema["required"]
        source_schema = dict(schema["$defs"]["sourceRef"])
        assert source_schema == ExecutionFindingSourceRef.model_json_schema()
        assert "result_sha256" in source_schema["properties"]
        assert "result_sha256" not in source_schema["required"]
        for field in ("document_alias", "ref_type", "producing_tool_call_id"):
            assert field not in json.dumps(schema)
    Draft202012Validator(record.spec.to_dict()["input_schema"]).validate({
        "items": [{
            "kind": "finding", "claim": "Native reference",
            "source_refs": [{"tool_result_id": "result-a", "chunk_id": "chunk-a"}],
        }],
    })
    Draft202012Validator(revise.spec.to_dict()["input_schema"]).validate({
        "items": [{"operation": "supersede", "entry_id": "entry-a", "kind": "decision", "claim": "Updated"}],
    })


def test_findings_tool_contract_cold_import_does_not_load_runtime() -> None:
    code = """
import json
import sys
import personagraph.tools.findings.execution_findings_tools
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith('personagraph.runtime')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_findings_catalog_can_be_enabled_and_removed_symmetrically() -> None:
    empty = CatalogSnapshot(revision=1, entries=())

    enabled, bridge = augment_execution_findings_tool_runtime(
        catalog_snapshot=empty,
        tool_bridge=None,
        enabled=True,
        strict_bridge_rebind=True,
    )

    assert catalog_exposes_execution_findings_tools(enabled) is True
    assert isinstance(bridge, SqliteWorkRunToolBridge)

    disabled, rebound = augment_execution_findings_tool_runtime(
        catalog_snapshot=enabled,
        tool_bridge=bridge,
        enabled=False,
        strict_bridge_rebind=True,
    )

    assert disabled.exposed() == ()
    assert catalog_exposes_execution_findings_tools(disabled) is False
    assert isinstance(rebound, SqliteWorkRunToolBridge)
    assert rebound is not bridge


def test_custom_bridge_must_explicitly_opt_in_to_catalog_rebind() -> None:
    empty = CatalogSnapshot(revision=1, entries=())
    bridge = _CustomBridgeWithoutCatalogRebind()

    unchanged_snapshot, unchanged_bridge = augment_execution_findings_tool_runtime(
        catalog_snapshot=empty,
        tool_bridge=bridge,  # type: ignore[arg-type]
        enabled=True,
        strict_bridge_rebind=False,
    )
    assert unchanged_snapshot is empty
    assert unchanged_bridge is bridge

    with pytest.raises(TypeError, match="cannot bind"):
        augment_execution_findings_tool_runtime(
            catalog_snapshot=empty,
            tool_bridge=bridge,  # type: ignore[arg-type]
            enabled=True,
            strict_bridge_rebind=True,
        )

    installed, installed_bridge = augment_execution_findings_tool_runtime(
        catalog_snapshot=empty,
        tool_bridge=None,
        enabled=True,
        strict_bridge_rebind=True,
    )
    assert installed_bridge is not None
    with pytest.raises(TypeError, match="explicitly rebindable"):
        augment_execution_findings_tool_runtime(
            catalog_snapshot=installed,
            tool_bridge=bridge,  # type: ignore[arg-type]
            enabled=True,
            strict_bridge_rebind=False,
        )


def test_partial_findings_tool_surface_is_never_treated_as_enabled() -> None:
    catalog = ToolCatalog()
    catalog.register(build_execution_findings_tool_registrations()[0])

    with pytest.raises(ValueError, match="partially exposed"):
        catalog_exposes_execution_findings_tools(catalog.snapshot())


def test_current_findings_schema_allows_an_unbound_finding() -> None:
    record, revise = build_execution_findings_tool_registrations()
    schema = record.spec.input_schema
    quota = ExecutionFindingsQuota()

    assert record.contract_version == EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION
    assert record.contract_version == "2.0.0"
    assert "findingRequiresSource" not in schema["$defs"]
    assert "allOf" not in schema["$defs"]["recordItem"]
    assert record.spec.name == "向任务台账追加记录"
    assert "模型自己认为有价值的发现与观察" in record.spec.description
    assert "依据允许为空" in record.spec.description
    assert "台账内容不代表绝对事实" in record.spec.description
    assert revise.spec.name == "修改或撤回任务台账记录"
    assert "使用 supersede 提交完整的新内容" in revise.spec.description
    assert "使用 retract 并说明原因" in revise.spec.description

    record_properties = schema["$defs"]["recordItem"]["properties"]
    assert record_properties["claim"]["maxLength"] == 4_096
    assert record_properties["claim"]["maxLength"] == quota.max_claim_characters
    assert (
        record_properties["claim"]["maxLength"]
        == EXECUTION_FINDING_CLAIM_MAX_CHARACTERS
    )
    assert (
        record_properties["source_refs"]["maxItems"]
        == quota.max_source_refs_per_entry
    )
    assert (
        schema["$defs"]["scopeKeys"]["maxItems"]
        == quota.max_scope_keys_per_entry
    )
    supersede_properties = revise.spec.input_schema["$defs"]["supersedeItem"][
        "properties"
    ]
    assert (
        supersede_properties["claim"]["maxLength"]
        == quota.max_claim_characters
    )
    assert (
        supersede_properties["source_refs"]["maxItems"]
        == quota.max_source_refs_per_entry
    )


def test_findings_definition_and_contextual_binding_are_exact() -> None:
    scope = "session:test/turn:test/attempt:test/findings"

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    facts = ExecutionFindingsBindingFacts(
        effect_scope_sha256=digest(scope),
        ledger_identity_sha256=digest("ledger"),
        scope_key_authority_sha256=digest("scope-key-authority"),
        dispatcher_identity_sha256=digest("dispatcher"),
        mutation_store_identity_sha256=digest("mutation-store"),
    )
    fingerprint = derive_execution_findings_source_fingerprint(facts)
    registrations = build_execution_findings_tool_registrations(
        effect_scope=scope,
        source_fingerprint=fingerprint,
    )
    manifest = build_execution_findings_tool_definition_manifest()
    bindings = build_execution_findings_tool_bindings(
        registrations,
        facts=facts,
    )

    assert tuple(item.definition.identity.tool_id for item in manifest) == (
        "record_execution_findings",
        "revise_execution_finding",
    )
    assert tuple(item.declared_behavior_revision for item in manifest) == (
        "record-execution-findings-handler-3",
        "revise-execution-finding-handler-3",
    )
    assert all(
        item.definition.effect_template.effects[0].default_scope == "*"
        for item in manifest
    )
    assert all(binding.source.fingerprint == fingerprint for binding in bindings)
    assert tuple(
        BoundToolRegistration(item.definition, binding).descriptor()
        for item, binding in zip(manifest, bindings, strict=True)
    ) == tuple(registration.descriptor() for registration in registrations)
