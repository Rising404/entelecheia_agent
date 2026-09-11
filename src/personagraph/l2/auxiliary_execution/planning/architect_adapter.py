"""纯边界区分 Architect 决策与 AuxiliaryGraph 修订持久化 DTO。

Architect 仅拥有提案局部结构。持久化图、目标、权威和预算标识符保持 Host/Store 的权威状态，因此此模块不进行任何 I/O 操作，也从不调用 Store 的界面。它的唯一任务是接受精确的创建/修订决策，而不默默地丢弃任何结构字段。
"""

from __future__ import annotations

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdgeProposal,
    AuxiliaryGraphRevisionProposalDisposition,
    AuxiliaryGraphRevisionReason,
    AuxiliaryGraphStructureProposal,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeProposal,
)
from personagraph.l2.task_graph import InSessionTaskAcceptanceProposal
from personagraph.session.l2_store.auxiliary_graph import (
    AuxiliaryGraphEdgeProposalRecord,
    AuxiliaryGraphNodeProposalRecord,
    AuxiliaryGraphRevisionProposalRecord,
)
from .architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
)


_TASK_GRAPH_PROPOSAL_CONTRACT = "task_graph_revision_proposal_v2"

_STRUCTURE_FIELDS = frozenset({"terminal_node_key", "nodes", "edges"})
_ARCHITECT_NODE_FIELDS = frozenset(
    {
        "node_key",
        "node_kind",
        "executor_kind",
        "title",
        "objective",
        "acceptance_criteria",
        "capability_profile_id",
        "input_resource_aliases",
        "source_anchor_ids",
        "output_contract",
        "required",
        "origin_node_alias",
    }
)
_PERSISTENCE_NODE_FIELDS = frozenset(
    (_ARCHITECT_NODE_FIELDS - {"node_key"}) | {"local_node_key"}
)
_ARCHITECT_EDGE_FIELDS = frozenset(
    {"source_node_key", "target_node_key", "required"}
)
_PERSISTENCE_EDGE_FIELDS = frozenset(
    {"dependency_node_key", "consumer_node_key", "required"}
)
_PERSISTENCE_REVISION_FIELDS = frozenset(
    {"revision_reason", "terminal_node_key", "nodes", "edges"}
)


class AuxiliaryArchitectAdapterError(ValueError):
    """Architect 的决策不能安全地成为持久化提案。"""

    code = "auxiliary_v2_architect_decision_not_committable"


def auxiliary_graph_revision_proposal_from_architect_decision(
    decision: AuxiliaryGraphArchitectDecision,
    *,
    expected_current_auxiliary_graph_revision: int | None,
    required_revision_reason: AuxiliaryGraphRevisionReason | None = None,
) -> AuxiliaryGraphRevisionProposalRecord:
    """将一个精确的创建/修订决策转换为 Store 的类型化的 DTO。

    持久化提案本身不携带预期当前 CAS。因此，调用者必须提供用于其最终提交的 Store 当前指针；此适配器证明它精确匹配密封的 Architect 决策，然后返回任何可提交的结构。
    """

    admitted = _revalidate_decision(decision)
    _require_current_revision_value(expected_current_auxiliary_graph_revision)
    if required_revision_reason is not None and not isinstance(
        required_revision_reason,
        AuxiliaryGraphRevisionReason,
    ):
        raise AuxiliaryArchitectAdapterError(
            "required revision reason must be AuxiliaryGraphRevisionReason"
        )
    proposal = admitted.proposal

    if admitted.action not in {
        AuxiliaryGraphArchitectAction.CREATE_REVISION,
        AuxiliaryGraphArchitectAction.REVISE_REVISION,
    }:
        raise AuxiliaryArchitectAdapterError(
            f"Architect action {admitted.action.value} is not committable as a structure"
        )
    if proposal.requested_user_question is not None:
        raise AuxiliaryArchitectAdapterError(
            "request_user_input is not committable as a structure"
        )

    if admitted.action is AuxiliaryGraphArchitectAction.CREATE_REVISION:
        if (
            proposal.disposition
            is not AuxiliaryGraphRevisionProposalDisposition.CREATE_REVISION
        ):
            raise AuxiliaryArchitectAdapterError(
                "create action does not carry a create_revision proposal"
            )
        if (
            expected_current_auxiliary_graph_revision is not None
            or proposal.expected_current_auxiliary_graph_revision is not None
        ):
            raise AuxiliaryArchitectAdapterError(
                "create_revision requires an exact null current revision"
            )
        if proposal.revision_reason is not AuxiliaryGraphRevisionReason.INITIAL:
            raise AuxiliaryArchitectAdapterError(
                "create_revision requires the initial revision reason"
            )
    else:
        if (
            proposal.disposition
            is not AuxiliaryGraphRevisionProposalDisposition.REVISE_REVISION
        ):
            raise AuxiliaryArchitectAdapterError(
                "revise action does not carry a revise_revision proposal"
            )
        if (
            expected_current_auxiliary_graph_revision is None
            or proposal.expected_current_auxiliary_graph_revision
            != expected_current_auxiliary_graph_revision
        ):
            raise AuxiliaryArchitectAdapterError(
                "revise_revision does not match the exact Host current revision"
            )
        if proposal.revision_reason in (
            None,
            AuxiliaryGraphRevisionReason.INITIAL,
        ):
            raise AuxiliaryArchitectAdapterError(
                "revise_revision requires a non-initial revision reason"
            )

    if proposal.structure is None or proposal.revision_reason is None:
        raise AuxiliaryArchitectAdapterError(
            "committable Architect decision requires a complete structure and reason"
        )
    if (
        required_revision_reason is not None
        and proposal.revision_reason is not required_revision_reason
    ):
        raise AuxiliaryArchitectAdapterError(
            "Architect revision does not match the Host-required reason"
        )
    return _convert_structure(
        proposal.structure,
        revision_reason=proposal.revision_reason,
    )


def build_terminal_only_auxiliary_graph_bootstrap_proposal(
    *,
    terminal_node_key: str,
    title: str,
    objective: str,
    source_anchor_ids: tuple[str, ...],
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...],
    input_resource_aliases: tuple[str, ...] = (),
    revision_reason: AuxiliaryGraphRevisionReason = (
        AuxiliaryGraphRevisionReason.INITIAL
    ),
) -> AuxiliaryGraphRevisionProposalRecord:
    """构建一个非执行版本，用于确立规划权威状态。

    ``initial`` 仅对新容器的修订版一有效。后续规划目标提供其 Host 绑定的非初始谱系原因。任一壳体必须在执行前由 Architect 替换。
    """

    if not isinstance(source_anchor_ids, tuple):
        raise AuxiliaryArchitectAdapterError(
            "terminal-only bootstrap source anchors must be a typed tuple"
        )
    if not isinstance(input_resource_aliases, tuple):
        raise AuxiliaryArchitectAdapterError(
            "terminal-only bootstrap resource aliases must be a typed tuple"
        )
    if not isinstance(acceptance_criteria, tuple) or any(
        not isinstance(item, InSessionTaskAcceptanceProposal)
        for item in acceptance_criteria
    ):
        raise AuxiliaryArchitectAdapterError(
            "terminal-only bootstrap Acceptance criteria must be typed"
        )
    if not isinstance(revision_reason, AuxiliaryGraphRevisionReason):
        raise AuxiliaryArchitectAdapterError(
            "terminal-only bootstrap revision reason must be typed"
        )

    try:
        admitted_acceptance = tuple(
            InSessionTaskAcceptanceProposal.model_validate_json(
                item.model_dump_json()
            )
            for item in acceptance_criteria
        )
        structure = AuxiliaryGraphStructureProposal(
            terminal_node_key=terminal_node_key,
            nodes=(
                AuxiliaryNodeProposal(
                    node_key=terminal_node_key,
                    node_kind=AuxiliaryNodeKind.SYNTHESIZE,
                    executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
                    title=title,
                    objective=objective,
                    acceptance_criteria=admitted_acceptance,
                    capability_profile_id=None,
                    input_resource_aliases=input_resource_aliases,
                    source_anchor_ids=source_anchor_ids,
                    output_contract=_TASK_GRAPH_PROPOSAL_CONTRACT,
                    required=True,
                    origin_node_alias=None,
                ),
            ),
            edges=(),
        )
        return _convert_structure(
            structure,
            revision_reason=revision_reason,
        )
    except AuxiliaryArchitectAdapterError:
        raise
    except Exception as exc:
        raise AuxiliaryArchitectAdapterError(
            "terminal-only bootstrap proposal failed contract validation"
        ) from exc


def _revalidate_decision(
    decision: AuxiliaryGraphArchitectDecision,
) -> AuxiliaryGraphArchitectDecision:
    if not isinstance(decision, AuxiliaryGraphArchitectDecision):
        raise AuxiliaryArchitectAdapterError(
            "decision must be an AuxiliaryGraphArchitectDecision"
        )
    try:
        return AuxiliaryGraphArchitectDecision.model_validate_json(
            decision.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryArchitectAdapterError(
            "Architect decision failed fresh contract validation"
        ) from exc


def _require_current_revision_value(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise AuxiliaryArchitectAdapterError(
            "expected current revision must be null or a positive integer"
        )


def _require_exact_contract_shapes() -> None:
    shapes = (
        (AuxiliaryGraphStructureProposal, _STRUCTURE_FIELDS),
        (AuxiliaryNodeProposal, _ARCHITECT_NODE_FIELDS),
        (AuxiliaryGraphEdgeProposal, _ARCHITECT_EDGE_FIELDS),
        (AuxiliaryGraphNodeProposalRecord, _PERSISTENCE_NODE_FIELDS),
        (AuxiliaryGraphEdgeProposalRecord, _PERSISTENCE_EDGE_FIELDS),
        (AuxiliaryGraphRevisionProposalRecord, _PERSISTENCE_REVISION_FIELDS),
    )
    for contract, expected_fields in shapes:
        if frozenset(contract.model_fields) != expected_fields:
            raise AuxiliaryArchitectAdapterError(
                f"{contract.__name__} changed; lossless adapter update is required"
            )


def _convert_structure(
    structure: AuxiliaryGraphStructureProposal,
    *,
    revision_reason: AuxiliaryGraphRevisionReason,
) -> AuxiliaryGraphRevisionProposalRecord:
    _require_exact_contract_shapes()
    try:
        admitted = AuxiliaryGraphStructureProposal.model_validate_json(
            structure.model_dump_json()
        )
        converted = AuxiliaryGraphRevisionProposalRecord(
            revision_reason=revision_reason,
            terminal_node_key=admitted.terminal_node_key,
            nodes=tuple(
                AuxiliaryGraphNodeProposalRecord(
                    local_node_key=node.node_key,
                    node_kind=node.node_kind,
                    executor_kind=node.executor_kind,
                    title=node.title,
                    objective=node.objective,
                    acceptance_criteria=node.acceptance_criteria,
                    capability_profile_id=node.capability_profile_id,
                    input_resource_aliases=node.input_resource_aliases,
                    source_anchor_ids=node.source_anchor_ids,
                    output_contract=node.output_contract,
                    required=node.required,
                    origin_node_alias=node.origin_node_alias,
                )
                for node in admitted.nodes
            ),
            edges=tuple(
                AuxiliaryGraphEdgeProposalRecord(
                    dependency_node_key=edge.source_node_key,
                    consumer_node_key=edge.target_node_key,
                    required=edge.required,
                )
                for edge in admitted.edges
            ),
        )
    except Exception as exc:
        raise AuxiliaryArchitectAdapterError(
            "Architect structure cannot be represented by the persistence contract"
        ) from exc

    expected = {
        "revision_reason": revision_reason.value,
        "terminal_node_key": admitted.terminal_node_key,
        "nodes": [
            {
                "local_node_key": node.node_key,
                "node_kind": node.node_kind.value,
                "executor_kind": node.executor_kind.value,
                "title": node.title,
                "objective": node.objective,
                "source_anchor_ids": list(node.source_anchor_ids),
                "acceptance_criteria": [
                    item.model_dump(mode="json")
                    for item in node.acceptance_criteria
                ],
                "output_contract": node.output_contract,
                "capability_profile_id": node.capability_profile_id,
                "input_resource_aliases": list(node.input_resource_aliases),
                "required": node.required,
                "origin_node_alias": node.origin_node_alias,
            }
            for node in admitted.nodes
        ],
        "edges": [
            {
                "dependency_node_key": edge.source_node_key,
                "consumer_node_key": edge.target_node_key,
                "required": edge.required,
            }
            for edge in admitted.edges
        ],
    }
    if converted.model_dump(mode="json") != expected:
        raise AuxiliaryArchitectAdapterError(
            "Architect structure changed during persistence conversion"
        )
    return converted


__all__ = [
    "AuxiliaryArchitectAdapterError",
    "auxiliary_graph_revision_proposal_from_architect_decision",
    "build_terminal_only_auxiliary_graph_bootstrap_proposal",
]
