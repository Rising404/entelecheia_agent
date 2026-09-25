"""L1 工具目录快照的冻结、投影与恢复边界。

本模块只管理 L1 已选工具定义及其本轮绑定的持久化形状。工具定义与绑定的
生产归 ``personagraph.tools`` 所有；L1 如何冻结并恢复它们则归运行链路所有。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Mapping

from ...tools.catalog.binding import BoundToolRegistration, ToolDefinition
from ...persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from .identity import canonical_json, sha256_json

if TYPE_CHECKING:
    from ...tools.composition.default_catalog import (
        ProductionDefaultDefinitionProfile,
    )


class L1CatalogSnapshotRestoreError(RuntimeError):
    """A persisted L1 catalog snapshot cannot be safely rebound."""


@dataclass(frozen=True, slots=True)
class L1ReconciledCatalogSnapshot:
    """一次恢复边界最终采用的 L1 工具目录及模型投影。"""

    definitions: tuple[ToolDefinition, ...]
    registrations_by_tool_id: dict[str, BoundToolRegistration]
    unavailable_reason_by_tool_id: dict[str, str]
    disabled_tool_ids: frozenset[str]
    snapshot_json: str
    snapshot_sha256: str
    model_catalog_json: str


def _l1_tool_catalog_descriptor(
    *,
    profile: ProductionDefaultDefinitionProfile,
    registrations_by_tool_id: Mapping[str, BoundToolRegistration],
    unavailable_reason_by_tool_id: Mapping[str, str],
    disabled_tool_ids: frozenset[str],
) -> dict[str, object]:
    """构造 L1 持久工具表的规范快照描述。"""

    tools: list[dict[str, object]] = []
    for item in profile.items:
        definition = item.definition
        tool_id = definition.identity.tool_id
        registration = registrations_by_tool_id.get(tool_id)
        unavailable_reason = unavailable_reason_by_tool_id.get(tool_id)
        tools.append(
            {
                "ordinal": item.ordinal,
                "availability": item.availability.value,
                "definition": definition.descriptor(),
                "definition_digest": definition.digest,
                "binding_status": (
                    "bound" if registration is not None else "unavailable"
                ),
                "binding": (
                    registration.binding.descriptor()
                    if registration is not None
                    else None
                ),
                "binding_digest": (
                    registration.binding_digest
                    if registration is not None
                    else None
                ),
                "unavailable_reason": unavailable_reason,
                "host_execution_status": (
                    "disabled" if tool_id in disabled_tool_ids else "enabled"
                ),
            }
        )
    return {
        "schema_version": "l1-tool-catalog-snapshot-v2",
        "catalog_revision": profile.catalog_revision,
        "catalog_digest": profile.catalog_digest,
        "profile_catalog_revision": profile.profile_catalog_revision,
        "profile_catalog_digest": profile.profile_catalog_digest,
        "profile_revision": profile.profile_revision,
        "profile_digest": profile.profile_digest,
        "tools": tools,
    }


def _model_catalog_items_from_definitions(
    definitions: tuple[ToolDefinition, ...],
) -> list[dict[str, object]]:
    """把完整工具定义投影成紧凑的模型可见目录。"""

    return [_model_tool_descriptor(definition) for definition in definitions]


def _model_tool_descriptor(
    definition: ToolDefinition,
) -> dict[str, object]:
    """省略只供 Host 校验的返回 Schema，不复制第二份工具定义。"""

    descriptor = definition.spec.to_dict()
    descriptor.pop("output_schema")
    descriptor["effects"] = [
        effect.to_dict() for effect in definition.effect_template.effects
    ]
    return descriptor


def _reconcile_persisted_l1_catalog_snapshot(
    *,
    expected_json: str,
    expected_sha256: str,
    current_json: str,
    current_sha256: str,
    definitions: tuple[ToolDefinition, ...],
    registrations_by_tool_id: Mapping[str, BoundToolRegistration],
    unavailable_reason_by_tool_id: Mapping[str, str],
    disabled_tool_ids: frozenset[str],
    current_model_catalog_json: str,
) -> L1ReconciledCatalogSnapshot:
    """Decode one old snapshot at the recovery boundary; never write it anew."""

    try:
        persisted = json.loads(expected_json)
    except (TypeError, ValueError) as exc:
        raise L1CatalogSnapshotRestoreError(
            "persisted L1 Tool Catalog is not valid JSON"
        ) from exc
    if (
        not isinstance(persisted, dict)
        or canonical_json(persisted) != expected_json
        or sha256_json(persisted) != expected_sha256
    ):
        raise L1CatalogSnapshotRestoreError(
            "persisted L1 Tool Catalog failed integrity checks"
        )

    if persisted.get("schema_version") == "l1-tool-catalog-snapshot-v2":
        if expected_json != current_json or expected_sha256 != current_sha256:
            raise L1CatalogSnapshotRestoreError(
                "persisted L1 Tool Catalog changed after Turn bootstrap"
            )
        return L1ReconciledCatalogSnapshot(
            definitions=definitions,
            registrations_by_tool_id=dict(registrations_by_tool_id),
            unavailable_reason_by_tool_id=dict(
                unavailable_reason_by_tool_id
            ),
            disabled_tool_ids=disabled_tool_ids,
            snapshot_json=expected_json,
            snapshot_sha256=expected_sha256,
            model_catalog_json=current_model_catalog_json,
        )

    if set(persisted) != {"revision", "entries"}:
        raise L1CatalogSnapshotRestoreError(
            "persisted L1 Tool Catalog schema is unsupported"
        )
    revision = persisted.get("revision")
    entries = persisted.get("entries")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(entries, list)
    ):
        raise L1CatalogSnapshotRestoreError(
            "legacy L1 Tool Catalog header is invalid"
        )

    definition_by_tool_id = {
        definition.identity.tool_id: definition for definition in definitions
    }
    selected_definitions: list[ToolDefinition] = []
    selected_registrations: dict[str, BoundToolRegistration] = {}
    legacy_model_catalog_items: list[dict[str, object]] = []
    created_revisions: set[int] = set()
    previous_key: tuple[str, str] | None = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "tool_id",
            "contract_version",
            "status",
            "created_revision",
            "updated_revision",
            "registration",
        }:
            raise L1CatalogSnapshotRestoreError(
                "legacy L1 Tool Catalog entry is invalid"
            )
        tool_id = entry.get("tool_id")
        contract_version = entry.get("contract_version")
        status = entry.get("status")
        created_revision = entry.get("created_revision")
        updated_revision = entry.get("updated_revision")
        if (
            not isinstance(tool_id, str)
            or not tool_id
            or not isinstance(contract_version, str)
            or not contract_version
            or status != "active"
            or isinstance(created_revision, bool)
            or not isinstance(created_revision, int)
            or created_revision < 1
            or isinstance(updated_revision, bool)
            or not isinstance(updated_revision, int)
            or updated_revision != created_revision
            or updated_revision > revision
        ):
            raise L1CatalogSnapshotRestoreError(
                "legacy L1 Tool Catalog entry metadata is invalid"
            )
        key = (tool_id, contract_version)
        if previous_key is not None and key <= previous_key:
            raise L1CatalogSnapshotRestoreError(
                "legacy L1 Tool Catalog entries are duplicated or unordered"
            )
        previous_key = key
        created_revisions.add(created_revision)
        registration = registrations_by_tool_id.get(tool_id)
        definition = definition_by_tool_id.get(tool_id)
        legacy_registration = entry.get("registration")
        if (
            registration is None
            or definition is None
            or registration.contract_version != contract_version
            or registration.definition != definition
            or not _legacy_registration_can_rebind(
                tool_id=tool_id,
                legacy=legacy_registration,
                current=registration.descriptor(),
            )
        ):
            raise L1CatalogSnapshotRestoreError(
                f"legacy L1 Tool binding cannot be reconstructed: {tool_id!r}"
            )
        assert isinstance(legacy_registration, dict)
        spec = legacy_registration.get("spec")
        effects = legacy_registration.get("effects")
        if not isinstance(spec, dict) or not isinstance(effects, list):
            raise L1CatalogSnapshotRestoreError(
                "legacy L1 model tool projection is invalid"
            )
        selected_definitions.append(definition)
        selected_registrations[tool_id] = registration
        legacy_model_catalog_items.append({**spec, "effects": effects})

    if (
        not entries
        or revision != len(entries)
        or created_revisions != set(range(1, revision + 1))
    ):
        raise L1CatalogSnapshotRestoreError(
            "legacy L1 Tool Catalog revision history is invalid"
        )

    return L1ReconciledCatalogSnapshot(
        definitions=tuple(selected_definitions),
        registrations_by_tool_id=selected_registrations,
        unavailable_reason_by_tool_id={},
        disabled_tool_ids=frozenset(
            disabled_tool_ids.intersection(selected_registrations)
        ),
        snapshot_json=expected_json,
        snapshot_sha256=expected_sha256,
        model_catalog_json=canonical_json(legacy_model_catalog_items),
    )


def _legacy_registration_can_rebind(
    *,
    tool_id: str,
    legacy: object,
    current: dict[str, Any],
) -> bool:
    """Allow only reviewed provenance upgrades at the legacy read boundary."""

    if not isinstance(legacy, dict) or set(legacy) != set(current):
        return False
    if legacy == current:
        return True
    if tool_id == "web_search":
        legacy_without_source = dict(legacy)
        current_without_source = dict(current)
        legacy_source = legacy_without_source.pop("source", None)
        current_source = current_without_source.pop("source", None)
        return (
            legacy_without_source == current_without_source
            and legacy_source
            == {
                "kind": "provider",
                "source_id": "personagraph.web.v2",
                "fingerprint": "web-v2-2026-08-26",
                "display_name": "Bounded public web access",
            }
            and isinstance(current_source, dict)
            and current_source.get("kind") == "provider"
            and current_source.get("source_id")
            == "personagraph.web.v2.search-provider-pipeline"
            and current_source.get("display_name")
            == "Bounded public web access"
            and _is_sha256(current_source.get("fingerprint"))
        )
    if tool_id not in EXECUTION_FINDINGS_TOOL_IDS:
        return False

    legacy_parts = dict(legacy)
    current_parts = dict(current)
    legacy_source = legacy_parts.pop("source", None)
    current_source = current_parts.pop("source", None)
    legacy_effects = legacy_parts.pop("effects", None)
    current_effects = current_parts.pop("effects", None)
    if (
        legacy_parts != current_parts
        or legacy_source
        != {
            "kind": "local",
            "source_id": "personagraph.execution_findings",
            "fingerprint": None,
            "display_name": "Execution Findings Ledger",
        }
        or not isinstance(current_source, dict)
        or current_source.get("kind") != "local"
        or current_source.get("source_id")
        != "personagraph.execution_findings"
        or current_source.get("display_name") != "Execution Findings Ledger"
        or not _is_sha256(current_source.get("fingerprint"))
        or not isinstance(legacy_effects, list)
        or not isinstance(current_effects, list)
        or len(legacy_effects) != 1
        or len(current_effects) != 1
        or not isinstance(legacy_effects[0], dict)
        or not isinstance(current_effects[0], dict)
    ):
        return False
    legacy_effect = dict(legacy_effects[0])
    current_effect = dict(current_effects[0])
    legacy_scope = legacy_effect.pop("default_scope", None)
    current_scope = current_effect.pop("default_scope", None)
    return (
        legacy_scope == "current_execution_findings"
        and isinstance(current_scope, str)
        and current_scope.startswith("l1_turn_run:")
        and legacy_effect == current_effect
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "L1CatalogSnapshotRestoreError",
    "L1ReconciledCatalogSnapshot",
]
