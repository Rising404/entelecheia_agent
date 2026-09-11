"""WorkRun runtime 上 execution-findings 工具的 Catalog 组合。"""

from __future__ import annotations

from ....tools.catalog import CatalogEntry, CatalogSnapshot, ToolCatalog
from ....tools.findings.execution_findings_tools import (
    EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION,
    build_execution_findings_tool_registrations,
)
from ....tools.registration import ToolRegistration
from personagraph.l2.task_execution.tool_bridge.contracts import (
    AttemptToolBridge,
    bridge_supports_catalog_rebind,
)
from personagraph.l2.task_execution.work_run.execution_findings import (
    execution_findings_tool_persistence_plan,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge


def augment_execution_findings_tool_runtime(
    *,
    catalog_snapshot: CatalogSnapshot,
    tool_bridge: AttemptToolBridge | None,
    enabled: bool = True,
    strict_bridge_rebind: bool = False,
) -> tuple[CatalogSnapshot, AttemptToolBridge | None]:
    """添加精确工具对，并将 bridge 绑定到所得快照。

    生产 SQLite bridge 可重新绑定，而不会丢失其权威、policy、executor、受保护
    dispatcher 或 persistence-plan factory。自定义注入 bridge 必须显式实现
    ``with_catalog_snapshot`` 才能选择加入。非严格调用方会保留未修改的自定义
    runtime，而不会暴露其 bridge 无法执行的工具。
    """

    if not isinstance(enabled, bool):
        raise TypeError("execution-findings enabled flag must be boolean")
    registrations = build_execution_findings_tool_registrations()
    expected_by_id = {item.tool_id: item for item in registrations}
    existing = tuple(
        entry
        for entry in catalog_snapshot.entries
        if entry.key.tool_id in expected_by_id
    )
    if existing:
        _require_known_execution_findings_pair(existing)

    if not enabled:
        retained = tuple(
            entry
            for entry in catalog_snapshot.entries
            if entry.key.tool_id not in expected_by_id
        )
        if len(retained) == len(catalog_snapshot.entries):
            return catalog_snapshot, tool_bridge
        if tool_bridge is not None and not bridge_supports_catalog_rebind(
            tool_bridge
        ):
            raise TypeError(
                "custom Tool Bridge cannot remove the disabled findings catalog"
            )
        stripped = _snapshot_from_entries(retained)
        if tool_bridge is None:
            return stripped, None
        return stripped, tool_bridge.with_catalog_snapshot(stripped)  # type: ignore[attr-defined]

    if existing:
        if catalog_exposes_execution_findings_tools(catalog_snapshot) and (
            tool_bridge is None
            or not bridge_supports_catalog_rebind(tool_bridge)
        ):
            raise TypeError(
                "execution-findings tools require an explicitly rebindable bridge"
            )
        return catalog_snapshot, tool_bridge

    missing = registrations

    if tool_bridge is not None and not bridge_supports_catalog_rebind(tool_bridge):
        if strict_bridge_rebind:
            raise TypeError(
                "custom Tool Bridge cannot bind the augmented findings catalog"
            )
        return catalog_snapshot, tool_bridge

    augmented = _snapshot_from_entries(
        catalog_snapshot.entries,
        appended=missing,
    )

    if tool_bridge is None:
        if catalog_snapshot.exposed():
            raise ValueError(
                "an exposed base catalog cannot be augmented without its bridge"
            )
        rebound: AttemptToolBridge = SqliteWorkRunToolBridge(
            catalog_snapshot=augmented,
            persistence_plan_factory=execution_findings_tool_persistence_plan,
        )
    else:
        rebound = tool_bridge.with_catalog_snapshot(augmented)  # type: ignore[attr-defined]
    return augmented, rebound


def catalog_exposes_execution_findings_tools(
    catalog_snapshot: CatalogSnapshot,
) -> bool:
    """返回精确完整 findings 工具对是否对模型可见。"""

    tool_ids = {
        item.tool_id for item in build_execution_findings_tool_registrations()
    }
    exposed = tuple(
        entry
        for entry in catalog_snapshot.exposed()
        if entry.key.tool_id in tool_ids
    )
    if not exposed:
        return False
    _require_known_execution_findings_pair(exposed)
    return True


def _require_known_execution_findings_pair(
    entries: tuple[CatalogEntry, ...],
) -> str:
    """要求完整的当前 execution-findings 工具对。"""

    tool_ids = {
        item.tool_id for item in build_execution_findings_tool_registrations()
    }
    if len(entries) != len(tool_ids) or {
        entry.key.tool_id for entry in entries
    } != tool_ids:
        raise ValueError("execution-findings tools are only partially exposed")
    versions = {entry.key.contract_version for entry in entries}
    if len(versions) != 1:
        raise ValueError("execution-findings tool versions cannot be mixed")
    version = next(iter(versions))
    if version != EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION:
        raise ValueError(
            "execution-findings tool ID conflicts with another registration"
        )
    expected = {
        item.tool_id: item.descriptor()
        for item in build_execution_findings_tool_registrations()
    }
    if any(
        entry.registration.descriptor() != expected[entry.key.tool_id]
        for entry in entries
    ):
        raise ValueError("execution-findings tool exposure changed its contract")
    return version


def _snapshot_from_entries(
    entries: tuple[CatalogEntry, ...],
    *,
    appended: tuple[ToolRegistration, ...] = (),
) -> CatalogSnapshot:
    catalog = ToolCatalog()
    for entry in entries:
        catalog.register(entry.registration, status=entry.status)
    for registration in appended:
        catalog.register(registration)
    return catalog.snapshot()


__all__ = [
    "augment_execution_findings_tool_runtime",
    "catalog_exposes_execution_findings_tools",
]
