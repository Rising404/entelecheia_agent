"""普通 TaskGraph WorkRun 所用的 Task 范围文档 Tool Runtime。

Auxiliary 规划会在提交 TaskGraph 前冻结已挂载文档。普通 TaskGraph 执行器必须先复现该精确权威，
节点才能看到文档工具；Session 范围挂载并不足以构成权威。

本模块提供惰性节点运行时工厂。其构造过程不执行 Store 或文档 I/O。在某个图修订版本的第一个
主体到来时，它会重新加载已提交的规划权威、复现精确挂载代次、与已冻结的 Session 工作区
Catalog 合并，并在同一组合 CatalogSnapshot 上创建一个桥接器。随后，该修订版本中的每个节点
复用此不可变捆绑。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.catalog.binding import ToolBinding
from personagraph.tools.policy import AuthorityFacts, ScopeGrant
from personagraph.l2.work_run import TaskNodeSubject
from personagraph.l2.task_execution.tool_bridge.attempt_contracts import AttemptToolBridgeRequest
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    MountedDocumentPlanningAuthorityError,
    build_mounted_document_authority_projection,
    freeze_mounted_document_planning_authority,
    recover_task_scoped_managed_document_ids,
)
from personagraph.l2.auxiliary_execution.planning.task_document_scope import AuxiliaryTaskDocumentScope
from personagraph.l2.auxiliary_execution.planning.mounted_visual_resource import (
    MountedVisualPlanningAuthorityError,
)
from personagraph.l2.task_execution.tool_bridge.execution_findings_catalog import (
    augment_execution_findings_tool_runtime,
)
from personagraph.l2.task_execution.tool_bridge.contracts import AttemptToolBridge
from personagraph.l2.task_execution.tool_bridge.mounted_document_adapter import (
    SessionMountedDocumentCognitionRuntime,
    build_session_mounted_document_cognition_runtime,
)
from personagraph.l2.task_execution.tool_bridge.mounted_visual_adapter import (
    MountedVisualToolRuntimeError,
    SessionMountedVisualToolRuntime,
    build_session_mounted_visual_tool_runtime,
)
from personagraph.tools.policy import (
    ProtectedToolExecutionAuthority,
)
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    build_runtime_protected_tool_dispatcher,
)
from personagraph.runtime.tool_calls import (
    RuntimeToolLedgerStore,
)
from personagraph.l2.task_execution.task_node.tool_runtime_contracts import (
    TaskNodeToolRuntimeFactory,
    TaskNodeToolRuntime,
)
from personagraph.l2.task_execution.tool_bridge.persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge
from personagraph.tools.workspace.session_read_source import (
    SessionWorkspaceReadonlyRuntime,
)


class TaskNodeDocumentToolRuntimeError(RuntimeError):
    """当前 Task 无法安全暴露其文档 Tool 表面。"""


@dataclass(frozen=True, slots=True)
class _TaskScopedDocumentRuntimeRevision:
    graph_revision: int
    subjects: frozenset[TaskNodeSubject]
    runtime: TaskNodeToolRuntime

    def __post_init__(self) -> None:
        if self.graph_revision < 1:
            raise ValueError("graph_revision must be positive")
        if not self.subjects:
            raise ValueError("a TaskGraph runtime revision requires nodes")
        if any(
            subject.graph_revision != self.graph_revision
            for subject in self.subjects
        ):
            raise ValueError("TaskGraph runtime subjects crossed revisions")


def build_task_scoped_document_node_runtime_factory(
    *,
    session_id: str,
    task_id: str,
    workspace_runtime: SessionWorkspaceReadonlyRuntime | None,
    workspace_tool_bridge: AttemptToolBridge | None,
    ledger_store: RuntimeToolLedgerStore,
    features: Mapping[str, Any] | None = None,
    task_document_scope: AuxiliaryTaskDocumentScope | None = None,
    file_retrieval_data_version: str | None = None,
) -> TaskNodeToolRuntimeFactory:
    """返回一个绑定到精确 Session 与 Task、且不执行 I/O 的工厂。

    ``run_task_graph_work_runs`` 会在变更第一个 WorkRun 前预检每个当前节点。因此，该闭包对每个
    图修订版本至多构建一个不可变运行时，并拒绝 Store 所投影当前节点集合之外的任何主体。
    """

    for name, value in (("session_id", session_id), ("task_id", task_id)):
        if not isinstance(value, str) or not value or len(value) > 200:
            raise ValueError(f"{name} must be a bounded durable identity")
    if (
        workspace_runtime is not None
        and workspace_runtime.session_id != session_id
    ):
        raise ValueError("workspace Runtime belongs to another Session")
    if (workspace_runtime is None) != (workspace_tool_bridge is None):
        raise ValueError(
            "workspace Runtime and WorkRun bridge must be composed together"
        )
    if task_document_scope is not None and (
        task_document_scope.session_id != session_id
        or task_document_scope.task_id != task_id
    ):
        raise ValueError("Task file candidate scope crossed Session/Task authority")

    revisions: dict[int, _TaskScopedDocumentRuntimeRevision] = {}

    def runtime_for(subject: TaskNodeSubject) -> TaskNodeToolRuntime:
        if not isinstance(subject, TaskNodeSubject):
            raise TypeError("node Tool runtime requires TaskNodeSubject")
        if subject.task_id != task_id:
            raise TaskNodeDocumentToolRuntimeError(
                "Task node crossed the document runtime Task scope"
            )
        revision = revisions.get(subject.graph_revision)
        if revision is None:
            revision = _build_task_scoped_document_runtime_revision(
                session_id=session_id,
                task_id=task_id,
                subject=subject,
                workspace_runtime=workspace_runtime,
                workspace_tool_bridge=workspace_tool_bridge,
                ledger_store=ledger_store,
            )
            revisions[subject.graph_revision] = revision
        if subject not in revision.subjects:
            raise TaskNodeDocumentToolRuntimeError(
                "Task node is absent from the frozen document runtime revision"
            )
        return revision.runtime

    return runtime_for


def compose_task_node_document_tool_runtime(
    *,
    session_id: str,
    workspace_runtime: SessionWorkspaceReadonlyRuntime | None,
    workspace_tool_bridge: AttemptToolBridge | None,
    mounted_runtime: SessionMountedDocumentCognitionRuntime | None,
    ledger_store: RuntimeToolLedgerStore,
    mounted_visual_runtime: SessionMountedVisualToolRuntime | None = None,
) -> TaskNodeToolRuntime:
    """在一个桥接器下组合工作区、挂载文本与挂载视觉内容。"""

    for runtime in (workspace_runtime, mounted_runtime):
        if runtime is not None and runtime.session_id != session_id:
            raise ValueError("document Tool Runtime crossed Session authority")
    if (workspace_runtime is None) != (workspace_tool_bridge is None):
        raise ValueError(
            "workspace Runtime and WorkRun bridge must be composed together"
        )
    if (
        mounted_visual_runtime is not None
        and mounted_visual_runtime.session_id != session_id
    ):
        raise ValueError("mounted visual Tool source crossed Session authority")
    if (
        workspace_runtime is None
        and mounted_runtime is None
        and mounted_visual_runtime is None
    ):
        snapshot, bridge = augment_execution_findings_tool_runtime(
            catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
            tool_bridge=None,
            strict_bridge_rebind=True,
        )
        return TaskNodeToolRuntime(
            catalog_snapshot=snapshot,
            tool_bridge=bridge,
        )
    if mounted_runtime is None and mounted_visual_runtime is None:
        assert workspace_runtime is not None
        snapshot, bridge = augment_execution_findings_tool_runtime(
            catalog_snapshot=workspace_runtime.catalog_snapshot,
            tool_bridge=workspace_tool_bridge,
            strict_bridge_rebind=False,
        )
        return TaskNodeToolRuntime(
            catalog_snapshot=snapshot,
            tool_bridge=bridge,
        )
    if workspace_runtime is None and mounted_visual_runtime is None:
        contextual_bindings = _mounted_document_contextual_bindings(
            mounted_runtime
        )
        snapshot, bridge = augment_execution_findings_tool_runtime(
            catalog_snapshot=mounted_runtime.catalog_snapshot,
            tool_bridge=mounted_runtime.tool_bridge,
            strict_bridge_rebind=False,
        )
        return TaskNodeToolRuntime(
            catalog_snapshot=snapshot,
            tool_bridge=bridge,
            contextual_bindings=contextual_bindings,
        )

    catalog = ToolCatalog()
    exposed_tool_ids: set[str] = set()
    for runtime in (workspace_runtime, mounted_runtime):
        if runtime is None:
            continue
        snapshot = runtime.catalog_snapshot
        for entry in snapshot.entries:
            if entry.key.tool_id in exposed_tool_ids:
                raise ValueError(
                    "composed document Tool IDs must be disjoint"
                )
            exposed_tool_ids.add(entry.key.tool_id)
            catalog.register(entry.registration, status=entry.status)
    if mounted_visual_runtime is not None:
        visual_registration = mounted_visual_runtime.registration
        if visual_registration.tool_id in exposed_tool_ids:
            raise ValueError("composed document Tool IDs must be disjoint")
        exposed_tool_ids.add(visual_registration.tool_id)
        catalog.register(visual_registration)
    snapshot = catalog.snapshot()
    authority = _merge_authority(
        *(
            runtime.authority
            for runtime in (workspace_runtime, mounted_runtime)
            if runtime is not None
        ),
        *(
            (mounted_visual_runtime.authority,)
            if mounted_visual_runtime is not None
            else ()
        ),
    )
    protected_authority_by_key = _merge_protected_authority_maps(
        (
            getattr(workspace_runtime, "protected_authority_by_key", {})
            if workspace_runtime is not None
            else None
        ),
        (
            mounted_visual_runtime.protected_authority_by_key
            if mounted_visual_runtime is not None
            else None
        ),
    )
    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=_task_node_document_tool_persistence_plan,
        authority=authority,
        protected_dispatcher=build_runtime_protected_tool_dispatcher(
            protected_authority_by_key,
            ledger_store=ledger_store,
        ),
        protected_authority_by_key=protected_authority_by_key,
    )
    augmented_snapshot, augmented_bridge = augment_execution_findings_tool_runtime(
        catalog_snapshot=snapshot,
        tool_bridge=bridge,
        strict_bridge_rebind=True,
    )
    return TaskNodeToolRuntime(
        catalog_snapshot=augmented_snapshot,
        tool_bridge=augmented_bridge,
        contextual_bindings=_mounted_document_contextual_bindings(
            mounted_runtime
        ),
    )


def _mounted_document_contextual_bindings(
    runtime: SessionMountedDocumentCognitionRuntime | None,
) -> tuple[ToolBinding, ...]:
    """投影已冻结来源的候选绑定，不赋予其任何 profile 顺序语义。"""

    if runtime is None:
        return ()
    bindings = runtime.contextual_bindings
    if not isinstance(bindings, tuple) or any(
        not isinstance(binding, ToolBinding) for binding in bindings
    ):
        raise TypeError(
            "mounted document contextual bindings must be typed and immutable"
        )
    identities = tuple(binding.identity for binding in bindings)
    if len(identities) != len(set(identities)):
        raise ValueError("mounted document contextual bindings must be unique")
    return bindings


def _build_task_scoped_document_runtime_revision(
    *,
    session_id: str,
    task_id: str,
    subject: TaskNodeSubject,
    workspace_runtime: SessionWorkspaceReadonlyRuntime | None,
    workspace_tool_bridge: AttemptToolBridge | None,
    ledger_store: RuntimeToolLedgerStore,
) -> _TaskScopedDocumentRuntimeRevision:
    try:
        auxiliary = auxiliary_graph_store.get_auxiliary_graph_for_task(
            session_id=session_id,
            insession_task_id=task_id,
        )
        task = task_graph_store.get_insession_task_details(session_id, task_id)
    except Exception as exc:
        raise TaskNodeDocumentToolRuntimeError(
            "Task document authority could not be loaded"
        ) from exc
    if (
        auxiliary is None
        or auxiliary.authority_snapshot is None
        or auxiliary.goal_status != "committed"
        or auxiliary.revision_status != "committed"
        or auxiliary.target_task_graph_revision != subject.graph_revision
    ):
        raise TaskNodeDocumentToolRuntimeError(
            "Task document authority is not the committed graph revision"
        )
    if task is None or task.current_graph_revision != subject.graph_revision:
        raise TaskNodeDocumentToolRuntimeError(
            "TaskGraph changed before document runtime preflight"
        )
    subjects = _current_task_subjects(task, task_id=task_id)
    if subject not in subjects:
        raise TaskNodeDocumentToolRuntimeError(
            "Task node is absent from the current TaskGraph"
        )

    try:
        allowed_managed_document_ids = recover_task_scoped_managed_document_ids(
            session_id=session_id,
            task_id=task_id,
            authority_snapshot=auxiliary.authority_snapshot,
        )
        mounted_authority = freeze_mounted_document_planning_authority(
            session_id=session_id,
            task_id=task_id,
            allowed_managed_document_ids=allowed_managed_document_ids,
        )
        creation_source = task_graph_store.get_insession_task_creation_source(
            session_id=session_id,
            insession_task_id=task_id,
        )
        build_mounted_document_authority_projection(
            authority_snapshot=auxiliary.authority_snapshot,
            task_creation_source=creation_source,
            mounted_authority=mounted_authority,
        )
        mounted_runtime = build_session_mounted_document_cognition_runtime(
            mounted_authority
        )
        mounted_visual_runtime = build_session_mounted_visual_tool_runtime(
            mounted_authority
        )
    except (
        MountedDocumentPlanningAuthorityError,
        MountedVisualPlanningAuthorityError,
        MountedVisualToolRuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise TaskNodeDocumentToolRuntimeError(
            "Task mounted-document authority could not be reproduced"
        ) from exc

    return _TaskScopedDocumentRuntimeRevision(
        graph_revision=subject.graph_revision,
        subjects=subjects,
        runtime=compose_task_node_document_tool_runtime(
            session_id=session_id,
            workspace_runtime=workspace_runtime,
            workspace_tool_bridge=workspace_tool_bridge,
            mounted_runtime=mounted_runtime,
            ledger_store=ledger_store,
            mounted_visual_runtime=mounted_visual_runtime,
        ),
    )


def _current_task_subjects(task, *, task_id: str) -> frozenset[TaskNodeSubject]:
    revision = task.current_graph_revision
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise TaskNodeDocumentToolRuntimeError("TaskGraph revision is malformed")
    subjects: set[TaskNodeSubject] = set()
    for node in task.nodes:
        node_id = node.get("insession_task_node_id")
        node_revision = node.get("node_revision")
        if (
            not isinstance(node_id, str)
            or not node_id
            or isinstance(node_revision, bool)
            or not isinstance(node_revision, int)
            or node_revision < 1
        ):
            raise TaskNodeDocumentToolRuntimeError(
                "TaskGraph node identity is malformed"
            )
        subjects.add(
            TaskNodeSubject(
                task_id=task_id,
                graph_revision=revision,
                node_id=node_id,
                node_revision=node_revision,
            )
        )
    if len(subjects) != len(task.nodes):
        raise TaskNodeDocumentToolRuntimeError(
            "TaskGraph node subjects are not unique"
        )
    return frozenset(subjects)


def _merge_authority(*values: AuthorityFacts) -> AuthorityFacts:
    grants: list[ScopeGrant] = []
    approval_grants: list[ScopeGrant] = []
    for value in values:
        for grant in value.grants:
            if grant not in grants:
                grants.append(grant)
        for grant in value.approval_grants:
            if grant not in approval_grants:
                approval_grants.append(grant)
    return AuthorityFacts(
        grants=tuple(grants),
        approval_grants=tuple(approval_grants),
        allow_local_read=all(value.allow_local_read for value in values),
    )


def _merge_protected_authority_maps(
    *values: Mapping[
        tuple[str, str], ProtectedToolExecutionAuthority
    ]
    | None,
) -> dict[tuple[str, str], ProtectedToolExecutionAuthority]:
    present = tuple(value for value in values if value is not None)
    combined = {}
    for value in present:
        for key, authority in value.items():
            previous = combined.get(key)
            if previous is not None and previous != authority:
                raise ValueError(
                    "protected tool authority conflicts during composition"
                )
            combined[key] = authority
    return combined


def _task_node_document_tool_persistence_plan(
    request: AttemptToolBridgeRequest,
) -> ToolBridgePersistencePlan:
    common = {
        "schema_version": "task-node-document-tool-persistence-v1",
        "session_id": request.session_id,
        "work_run_id": request.work_run_id,
        "attempt_id": request.attempt_id,
        "decision_apply_id": request.apply_id,
    }
    return ToolBridgePersistencePlan(
        decision_apply_id=request.apply_id,
        close_apply_id=_stable_id("task-node-document-tool-close", common),
        calls=tuple(
            ToolBridgeCallPersistence(
                tool_call_id=call.tool_call_id,
                tool_result_id=_stable_id(
                    "task-node-document-tool-result",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
                result_apply_id=_stable_id(
                    "task-node-document-tool-result-apply",
                    {**common, "tool_call_id": call.tool_call_id},
                ),
            )
            for call in request.decision.action.calls
        ),
    )


def _stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "TaskNodeDocumentToolRuntimeError",
    "build_task_scoped_document_node_runtime_factory",
    "compose_task_node_document_tool_runtime",
]
