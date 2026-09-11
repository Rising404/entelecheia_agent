"""AuxiliaryGraph 规划中的延迟生产投影。

此模块不含 Store 或提供者输入输出。它将已认证的辅助图状态和冻结的已安装文档范围转换为 Architect 和执行组合根所需的确切别名值。
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningContextArtifactProjection,
    PlanningGoalPromptContext,
    TaskGraphSemanticBaseSnapshot,
)
from personagraph.session.l2_store.auxiliary_graph import (
    StoredAuxiliaryGraphDetails,
)
from personagraph.l2.task_graph import TaskGraphRevisionTrigger
from personagraph.tools.catalog import CatalogSnapshot
from .architect import (
    AuxiliaryGraphArchitectPrompt,
    AuxiliaryGraphArchitectReplanTrigger,
    AuxiliaryGraphArchitectRequest,
    AuxiliaryGraphProtectedCapabilityGrant,
    AuxiliaryGraphCurrentRevisionProjection,
    TaskGraphRevisionRouteAuthority,
)
from personagraph.l2.auxiliary_execution.work_run.contracts import (
    KNOWLEDGE_COGNITION_CAPABILITY,
)
from .mounted_document_authority import (
    FrozenMountedDocumentPlanningAuthority,
    MOUNTED_DOCUMENT_READ_CAPABILITY,
)
from .mounted_visual_resource import MOUNTED_VISUAL_READ_CAPABILITY
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_CAPABILITY,
)
from personagraph.l2.task_execution.tool_bridge.mounted_document_adapter import (
    SessionMountedDocumentCognitionRuntime,
)
from personagraph.tools.workspace.session_read_source import (
    SessionWorkspaceReadonlyRuntime,
)


MODEL_ANALYSIS_CAPABILITY = "model_analysis"
WORKSPACE_READONLY_CAPABILITY = "workspace_readonly"
AUXILIARY_ARCHITECT_PROFILE_ID = "auxiliary_graph_architect_v2_production"


class AuxiliaryPlanningProjectionError(RuntimeError):
    """没有权威状态，无法投影存储中的规划状态。"""


def build_auxiliary_planning_capability_catalog(
    mounted_authority: FrozenMountedDocumentPlanningAuthority,
    *,
    workspace_runtime: SessionWorkspaceReadonlyRuntime | None = None,
    knowledge_cognition_history_enabled: bool = False,
) -> PlanningCapabilityCatalogProjection:
    """冻结暴露给 Architect 的有限能力表面。"""

    if not isinstance(
        mounted_authority, FrozenMountedDocumentPlanningAuthority
    ):
        raise TypeError(
            "mounted_authority must be FrozenMountedDocumentPlanningAuthority"
        )
    if not isinstance(knowledge_cognition_history_enabled, bool):
        raise TypeError("knowledge history switch must be boolean")
    document_kinds = tuple(
        sorted(
            {
                binding.resource.resource_format.value
                for binding in mounted_authority.bindings
            }
        )
    )
    visual_kinds = tuple(
        sorted(
            {
                binding.resource.resource_format.value
                for binding in mounted_authority.visual_bindings
            }
        )
    )
    capabilities = (
        PlanningCapabilityDescriptor(
            capability_alias=MODEL_ANALYSIS_CAPABILITY,
            label="Bounded model analysis",
            description=(
                "Analyze only the frozen goal and verified dependency outputs; "
                "this capability has no direct file or tool access."
            ),
            available=True,
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=("analyze_verified_dependencies",),
            supported_resource_kinds=("planning_context_artifact",),
            limitations=("no_direct_resource_access", "no_effectful_tools"),
        ),
        PlanningCapabilityDescriptor(
            capability_alias=KNOWLEDGE_COGNITION_CAPABILITY,
            label="Node-scoped history retrieval",
            description="Retrieve authorized history through one exact model WorkRun scope.",
            available=knowledge_cognition_history_enabled,
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=("refine_retrieval_query", "retrieve_authorized_history"),
            supported_resource_kinds=("current_session_history", "long_term_task_memory", "long_term_user_memory"),
            limitations=("host_frozen_history_scope_only", "model_work_run_only", "read_only"),
        ),
        PlanningCapabilityDescriptor(
            capability_alias=MOUNTED_DOCUMENT_READ_CAPABILITY,
            label="Bounded mounted document preview",
            description=(
                "Read the bounded first window of exactly one Host-frozen "
                "Session-mounted plain-text, "
                "Markdown, PDF, image, Word, or PowerPoint resource and produce "
                "a verified context artifact. Prefer mounted_document_cognition "
                "when the task requires searching or reading a complete document."
            ),
            available=bool(mounted_authority.bindings),
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=("read_one_mounted_document",),
            supported_resource_kinds=document_kinds,
            limitations=(
                "bounded_first_window_only",
                "one_resource_alias_per_node",
                "read_only",
            ),
        ),
        PlanningCapabilityDescriptor(
            capability_alias=MOUNTED_DOCUMENT_COGNITION_CAPABILITY,
            label="Iterative mounted document cognition",
            description=(
                "Use a model_work_run to inspect total chunk coverage, search the "
                "entire frozen content of Session-mounted documents, and continue "
                "through exact chunk windows until the required evidence or full "
                "document coverage is reached."
            ),
            available=bool(mounted_authority.bindings),
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=(
                "inspect_document_coverage",
                "iterative_chunk_reads",
                "search_complete_document",
            ),
            supported_resource_kinds=document_kinds,
            limitations=(
                "model_work_run_only",
                "read_only",
                "requires_explicit_coverage_for_exhaustive_tasks",
                "task_authorized_mounted_documents_only",
            ),
        ),
        PlanningCapabilityDescriptor(
            capability_alias=MOUNTED_VISUAL_READ_CAPABILITY,
            label="Mounted visual perception",
            description=(
                "Resolve exactly one Host-frozen visual unit from a mounted "
                "PDF or image and produce a verified context artifact. The "
                "model receives only an opaque alias; the Host retains paths "
                "and enforces any required disclosure grant."
            ),
            available=bool(mounted_authority.visual_bindings),
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=("read_one_mounted_visual_unit",),
            supported_resource_kinds=visual_kinds,
            limitations=tuple(
                sorted(
                    {
                        "external_provider_requires_disclosure",
                        "one_visual_alias_per_node",
                        "ooxml_embedded_pixels_not_indexed",
                        "private_path_host_only",
                        *(
                            ("complete_visual_projection_unavailable",)
                            if mounted_authority.visual_projection_limit is not None
                            else ()
                        ),
                    }
                )
            ),
        ),
        PlanningCapabilityDescriptor(
            capability_alias=WORKSPACE_READONLY_CAPABILITY,
            label="Iterative workspace document cognition",
            description=(
                "Use a model_work_run to inspect the Session-bound working "
                "directory, locate likely files from vague descriptions, choose "
                "a format-specific reader or renderer, search document content, "
                "and adjust tool parameters after each observed result."
            ),
            available=workspace_runtime is not None,
            effect=PlanningCapabilityEffect.READ_ONLY,
            supported_operations=(
                "discover_directory_structure",
                "find_files_by_name_or_description",
                "inspect_file_metadata",
                "iterative_model_tool_calls",
                "read_document_or_image",
                "search_document_content",
            ),
            supported_resource_kinds=(
                "doc",
                "docx",
                "image",
                "markdown",
                "pdf",
                "png",
                "ppt",
                "pptx",
                "text",
            ),
            limitations=tuple(
                sorted(
                    {
                        "model_work_run_only",
                        "read_only",
                        "results_may_be_partial_and_report_coverage",
                        "session_working_directory_only",
                        *(
                            ()
                            if workspace_runtime is None
                            or workspace_runtime.visual_analysis_available
                            else ("visual_semantics_requires_protected_operation",)
                        ),
                    }
                )
            ),
        ),
    )
    capabilities = tuple(
        sorted(capabilities, key=lambda item: item.capability_alias)
    )
    private_snapshot = {
        "schema_version": "auxiliary-v2-production-capability-snapshot-v1",
        "mounted_scope_snapshot_sha256": (
            mounted_authority.scope_snapshot_sha256
        ),
        "workspace_scope_snapshot_sha256": (
            None
            if workspace_runtime is None
            else workspace_runtime.scope_snapshot_sha256
        ),
        "workspace_tool_ids": (
            [] if workspace_runtime is None else list(workspace_runtime.tool_ids)
        ),
        "workspace_visual_analysis_available": (
            False
            if workspace_runtime is None
            else workspace_runtime.visual_analysis_available
        ),
        "workspace_visual_analysis_reason": (
            None
            if workspace_runtime is None
            else workspace_runtime.visual_analysis_reason
        ),
        "capabilities": [
            item.model_dump(mode="json") for item in capabilities
        ],
    }
    snapshot_sha256 = _sha256_value(private_snapshot)
    return PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id=(
            "auxv2-capabilities-v1-" + snapshot_sha256[:32]
        ),
        capability_catalog_snapshot_sha256=snapshot_sha256,
        capabilities=capabilities,
    )


def build_auxiliary_execution_capability_catalogs(
    *,
    workspace_runtime: SessionWorkspaceReadonlyRuntime | None = None,
    mounted_document_runtime: (
        SessionMountedDocumentCognitionRuntime | None
    ) = None,
) -> Mapping[str, CatalogSnapshot]:
    """返回与提示能力 ID 相对应的模型节点目录。"""

    catalogs: dict[str, CatalogSnapshot] = {
        MODEL_ANALYSIS_CAPABILITY: CatalogSnapshot(revision=1, entries=())
    }
    if workspace_runtime is not None:
        catalogs[WORKSPACE_READONLY_CAPABILITY] = (
            workspace_runtime.catalog_snapshot
        )
    if mounted_document_runtime is not None:
        catalogs[MOUNTED_DOCUMENT_COGNITION_CAPABILITY] = (
            mounted_document_runtime.catalog_snapshot
        )
    return catalogs


def build_auxiliary_graph_current_revision_projection(
    details: StoredAuxiliaryGraphDetails,
) -> AuxiliaryGraphCurrentRevisionProjection:
    """将 Store 身份转换回修订本地的 Architect 命名空间。"""

    if not isinstance(details, StoredAuxiliaryGraphDetails):
        raise TypeError("details must be StoredAuxiliaryGraphDetails")
    node_key_by_id = {
        node.auxiliary_node_id: node.local_node_key for node in details.nodes
    }
    if len(node_key_by_id) != len(details.nodes):
        raise AuxiliaryPlanningProjectionError(
            "stored AuxiliaryGraph node identities are ambiguous"
        )
    try:
        structure = AuxiliaryGraphStructureProposal(
            terminal_node_key=node_key_by_id[
                details.terminal_auxiliary_node_id
            ],
            nodes=tuple(
                AuxiliaryNodeProposal(
                    node_key=node.local_node_key,
                    node_kind=AuxiliaryNodeKind(node.node_kind),
                    executor_kind=AuxiliaryNodeExecutorKind(
                        node.executor_kind
                    ),
                    title=node.title,
                    objective=node.objective,
                    acceptance_criteria=node.acceptance_criteria,
                    capability_profile_id=node.capability_profile_id,
                    input_resource_aliases=node.input_resource_aliases,
                    source_anchor_ids=node.source_anchor_ids,
                    output_contract=node.output_contract,
                    required=node.required,
                    # 投影仅暴露当前别名作为合法标识。
                    # 下一次提案的起源。历史起源链接
                    # 保持私有 Store 权威状态 并且不会递归
                    # 投影到这个完整当前快照中。
                    origin_node_alias=None,
                )
                for node in details.nodes
            ),
            edges=tuple(
                AuxiliaryGraphEdgeProposal(
                    source_node_key=node_key_by_id[
                        edge.dependency_auxiliary_node_id
                    ],
                    target_node_key=node_key_by_id[
                        edge.consumer_auxiliary_node_id
                    ],
                    required=True,
                )
                for edge in details.edges
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuxiliaryPlanningProjectionError(
            "stored AuxiliaryGraph cannot become an alias-only projection"
        ) from exc
    return AuxiliaryGraphCurrentRevisionProjection.create(
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        base_task_graph_revision=details.base_task_graph_revision,
        structure=structure,
        source_structure_sha256=details.structure_sha256,
    )


def build_auxiliary_architect_request(
    *,
    details: StoredAuxiliaryGraphDetails,
    authority: PlanningAuthorityProjection,
    capabilities: PlanningCapabilityCatalogProjection,
    objective: str,
    desired_output: str,
    context_artifacts: tuple[PlanningContextArtifactProjection, ...] = (),
    task_graph_semantic_base: TaskGraphSemanticBaseSnapshot | None = None,
    task_graph_revision_trigger: TaskGraphRevisionTrigger | None = None,
    task_graph_revision_route: TaskGraphRevisionRouteAuthority | None = None,
    replan_trigger: AuxiliaryGraphArchitectReplanTrigger | None = None,
) -> AuxiliaryGraphArchitectRequest:
    """构建一个稳定的持久化 Architect 请求，基于精确的 Store 投影。"""

    if (
        authority.authority_snapshot_id != details.authority_snapshot_id
        or authority.authority_snapshot_sha256
        != details.authority_snapshot_sha256
        or details.budget.goal_id != details.goal_id
    ):
        raise AuxiliaryPlanningProjectionError(
            "Architect projections differ from current Store authority"
        )
    authorization_aliases = tuple(
        card.alias
        for card in authority.cards
        if card.authority_class is PlanningAuthorityClass.AUTHORIZATION
    )
    protected_capability_grants: tuple[
        AuxiliaryGraphProtectedCapabilityGrant, ...
    ] = ()
    current = build_auxiliary_graph_current_revision_projection(details)
    prompt = AuxiliaryGraphArchitectPrompt.create(
        goal=PlanningGoalPromptContext(
            goal_id=details.goal_id,
            objective=objective,
            desired_output=desired_output,
            authorization_aliases=authorization_aliases,
        ),
        authority=authority,
        context_artifacts=context_artifacts,
        capabilities=capabilities,
        protected_capability_grants=protected_capability_grants,
        budget=details.budget,
        current_revision=current,
        task_graph_semantic_base=task_graph_semantic_base,
        task_graph_revision_trigger=task_graph_revision_trigger,
        task_graph_revision_route=task_graph_revision_route,
        replan_trigger=replan_trigger,
    )
    identity = _sha256_value(
        {
            "schema_version": "auxiliary-v2-architect-request-identity-v1",
            "session_id": details.session_id,
            "task_id": details.task_id,
            "auxiliary_graph_id": details.auxiliary_graph_id,
            "goal_id": details.goal_id,
            "auxiliary_graph_revision": details.auxiliary_graph_revision,
            "prompt_payload_sha256": prompt.payload_sha256,
        }
    )[:32]
    return AuxiliaryGraphArchitectRequest.create(
        architect_request_id=f"auxv2arch-{identity}:request",
        logical_call_id=f"auxv2arch-{identity}:model",
        architect_profile_id=AUXILIARY_ARCHITECT_PROFILE_ID,
        goal=details.goal,
        prompt_payload=prompt,
    )


def canonical_auxiliary_architect_state_guard(
    request: AuxiliaryGraphArchitectRequest,
) -> str:
    """对 Host-密封的 Architect 权威状态进行哈希，用于持久化分发。"""

    if not isinstance(request, AuxiliaryGraphArchitectRequest):
        raise TypeError("request must be AuxiliaryGraphArchitectRequest")
    # ``RuntimeModelLogicalRequest.state_guard_sha256`` 被冻结到
    # Architect 请求绑定自身。返回那个确切值保持了
    # 单一的纯投影辅助和持久化模型调用的权威状态
    # 合同上；将其包装在另一个哈希中会使每次实际分发
    # 失败其首次当前状态检查。
    return request.binding_sha256


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AUXILIARY_ARCHITECT_PROFILE_ID",
    "MODEL_ANALYSIS_CAPABILITY",
    "MOUNTED_DOCUMENT_COGNITION_CAPABILITY",
    "MOUNTED_DOCUMENT_READ_CAPABILITY",
    "MOUNTED_VISUAL_READ_CAPABILITY",
    "WORKSPACE_READONLY_CAPABILITY",
    "AuxiliaryPlanningProjectionError",
    "build_auxiliary_graph_current_revision_projection",
    "build_auxiliary_architect_request",
    "build_auxiliary_execution_capability_catalogs",
    "build_auxiliary_planning_capability_catalog",
    "canonical_auxiliary_architect_state_guard",
]
