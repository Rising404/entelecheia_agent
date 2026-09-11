"""对不可信 L2 TaskGraph 提案执行确定性验证。"""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskGraphRevisionValidationResult,
    InSessionTaskGraphValidationCode,
    InSessionTaskGraphValidationContext,
    InSessionTaskGraphValidationResult,
    InSessionTaskNodeKind,
    InSessionTaskNodeProposal,
    InSessionTaskRootGraphProposal,
    NewInSessionTaskGraphsProposal,
)


def validate_new_insession_task_graphs(
    proposal: NewInSessionTaskGraphsProposal,
    *,
    context: InSessionTaskGraphValidationContext,
) -> InSessionTaskGraphValidationResult:
    """只接受绑定到来源、结构完整且未超出 Host 资源限制的树。

    自然语言分组仍属于模型质量问题。此函数特意只验证 Host 能够证明的事实：
    来源引用、节点及 Acceptance 的来源覆盖、树结构和有界大小。
    """

    errors: set[InSessionTaskGraphValidationCode] = set()
    if proposal.source_turn_id != context.source_turn_id:
        errors.add(InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH)
    if not proposal.roots:
        errors.add(InSessionTaskGraphValidationCode.EMPTY_BATCH)
    if len(proposal.roots) > context.limits.max_root_tasks:
        errors.add(InSessionTaskGraphValidationCode.ROOT_LIMIT_EXCEEDED)

    anchor_ids = {anchor.anchor_id for anchor in context.source_anchors}
    authorization_anchor_ids = set(context.authorization_anchor_ids)
    if not authorization_anchor_ids:
        errors.add(InSessionTaskGraphValidationCode.MISSING_AUTHORIZATION_ANCHOR)
    if not authorization_anchor_ids.issubset(anchor_ids):
        errors.add(InSessionTaskGraphValidationCode.UNKNOWN_AUTHORIZATION_ANCHOR)
    _validate_authorization_anchor_kinds(
        context,
        authorization_anchor_ids,
        errors,
    )
    mapped_required_anchor_ids: set[str] = set()
    for root in proposal.roots:
        _validate_root_graph(
            root,
            anchor_ids,
            authorization_anchor_ids,
            mapped_required_anchor_ids,
            context,
            errors,
        )

    if not set(context.required_anchor_ids).issubset(mapped_required_anchor_ids):
        errors.add(InSessionTaskGraphValidationCode.UNMAPPED_REQUIRED_ANCHOR)

    ordered_errors = tuple(sorted(errors, key=str))
    return InSessionTaskGraphValidationResult(
        status="accepted" if not ordered_errors else "rejected",
        proposal=proposal if not ordered_errors else None,
        trusted_context=context if not ordered_errors else None,
        error_codes=ordered_errors,
    )


def validate_insession_task_graph_revision(
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    context: InSessionTaskGraphRevisionValidationContext,
) -> InSessionTaskGraphRevisionValidationResult:
    """依据 Host 持有的权威事实验证一个完整图快照。

    此纯边界既支持空壳初始化（预期当前修订为 ``None``），也支持基于正数基础修订且结构有效的
    完整快照。对于后者，它特意不证明节点标识或节点修订连续性。在相应协议就绪前，
    P1.1 持久化边界必须拒绝非空基础修订。

    ``previously_authorized_task_state`` 在此可以授权范围，唯一原因是上下文由 Host 提供。
    接受并不等于写入权威：Store 必须重新加载目标 Task、基础修订、来源 Turn 和精确锚点材料，
    并在其事务中重新运行相关验证。
    """

    errors: set[InSessionTaskGraphValidationCode] = set()
    anchor_ids = {anchor.anchor_id for anchor in context.source_anchors}
    authorization_anchor_ids = set(context.authorization_anchor_ids)
    if not authorization_anchor_ids:
        errors.add(InSessionTaskGraphValidationCode.MISSING_AUTHORIZATION_ANCHOR)
    if not authorization_anchor_ids.issubset(anchor_ids):
        errors.add(InSessionTaskGraphValidationCode.UNKNOWN_AUTHORIZATION_ANCHOR)
    _validate_revision_anchor_turns(context, errors)
    _validate_revision_authorization_anchor_kinds(
        context,
        authorization_anchor_ids,
        errors,
    )

    mapped_required_anchor_ids: set[str] = set()
    _validate_root_graph(
        proposal.root,
        anchor_ids,
        authorization_anchor_ids,
        mapped_required_anchor_ids,
        context,
        errors,
    )
    if not set(context.required_anchor_ids).issubset(mapped_required_anchor_ids):
        errors.add(InSessionTaskGraphValidationCode.UNMAPPED_REQUIRED_ANCHOR)

    ordered_errors = tuple(sorted(errors, key=str))
    return InSessionTaskGraphRevisionValidationResult(
        status="accepted" if not ordered_errors else "rejected",
        proposal=proposal if not ordered_errors else None,
        trusted_context=context if not ordered_errors else None,
        error_codes=ordered_errors,
    )


def bind_used_evidence_to_required_acceptance_coverage(
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    context: InSessionTaskGraphRevisionValidationContext,
) -> InSessionTaskGraphRevisionValidationContext:
    """将每个被引用的证据来源设为必需的 Acceptance 义务。

    只出现在节点上的证据引用不是可审计的完成条件。终态封存已经强制执行此规则；
    在 WorkRun 和 Store 预检前应用同样的纯收紧措施，可让所有边界保持确定性。
    """

    if not isinstance(proposal, InSessionTaskGraphRevisionProposal):
        raise TypeError("proposal must be an InSessionTaskGraphRevisionProposal")
    if not isinstance(context, InSessionTaskGraphRevisionValidationContext):
        raise TypeError(
            "context must be an InSessionTaskGraphRevisionValidationContext"
        )
    used_ids = {
        source_id
        for node in proposal.root.nodes
        for source_id in (
            *node.source_anchor_ids,
            *(
                source_id
                for acceptance in node.acceptance_criteria
                for source_id in acceptance.source_anchor_ids
            ),
        )
    }
    required_ids = set(context.required_anchor_ids) | (
        used_ids - set(context.authorization_anchor_ids)
    )
    known_ids = tuple(anchor.anchor_id for anchor in context.source_anchors)
    known_set = set(known_ids)
    ordered_required = tuple(
        anchor_id for anchor_id in known_ids if anchor_id in required_ids
    ) + tuple(
        anchor_id
        for anchor_id in context.required_anchor_ids
        if anchor_id not in known_set
    )
    return context.model_copy(update={"required_anchor_ids": ordered_required})


def validate_current_user_anchor_spans(
    context: InSessionTaskGraphValidationContext,
    *,
    user_text: str,
) -> tuple[InSessionTaskGraphValidationCode, ...]:
    """依据一个权威输入字符串验证当前用户锚点。

    对于类型为当前用户材料的锚点，``start`` 和 ``end`` 使用 Python/Unicode 码位偏移量。
    其他来源类型特意不由此辅助函数处理：其资源标识及偏移方案属于后续类型化的
    TaskGraphContext 契约。
    """

    errors: set[InSessionTaskGraphValidationCode] = set()
    current_user_kinds = {"current_user_instruction", "current_user_context"}
    for anchor in context.source_anchors:
        if anchor.source_kind not in current_user_kinds:
            continue
        if anchor.source_turn_id != context.source_turn_id:
            errors.add(InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH)
            continue
        if (
            anchor.end > len(user_text)
            or user_text[anchor.start : anchor.end] != anchor.excerpt
        ):
            errors.add(InSessionTaskGraphValidationCode.CURRENT_USER_ANCHOR_MISMATCH)
    return tuple(sorted(errors, key=str))


def _validate_root_graph(
    root: InSessionTaskRootGraphProposal,
    anchor_ids: set[str],
    authorization_anchor_ids: set[str],
    mapped_required_anchor_ids: set[str],
    context: InSessionTaskGraphValidationContext,
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    nodes_by_key = _nodes_by_key(root.nodes, errors)
    root_node = nodes_by_key.get(root.root_key)
    if root_node is None:
        errors.add(InSessionTaskGraphValidationCode.ROOT_NODE_MISSING)
    elif root_node.node_kind is not InSessionTaskNodeKind.ROOT or root_node.parent_node_key is not None:
        errors.add(InSessionTaskGraphValidationCode.ROOT_NODE_INVALID)

    if len(root.nodes) > context.limits.max_nodes_per_task:
        errors.add(InSessionTaskGraphValidationCode.NODE_LIMIT_EXCEEDED)
    _validate_parent_references(nodes_by_key, root.root_key, errors)
    _validate_tree_depth(nodes_by_key, root.root_key, context.limits.max_depth, errors)
    _validate_nodes_source_and_coverage(
        root.nodes,
        anchor_ids,
        authorization_anchor_ids,
        mapped_required_anchor_ids,
        errors,
    )


def _nodes_by_key(
    nodes: Iterable[InSessionTaskNodeProposal],
    errors: set[InSessionTaskGraphValidationCode],
) -> dict[str, InSessionTaskNodeProposal]:
    nodes_by_key: dict[str, InSessionTaskNodeProposal] = {}
    for node in nodes:
        if node.node_key in nodes_by_key:
            errors.add(InSessionTaskGraphValidationCode.DUPLICATE_NODE_KEY)
            continue
        nodes_by_key[node.node_key] = node
    return nodes_by_key


def _validate_parent_references(
    nodes_by_key: dict[str, InSessionTaskNodeProposal],
    root_key: str,
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    for node in nodes_by_key.values():
        parent_key = node.parent_node_key
        if node.node_key == root_key:
            continue
        if parent_key is None or parent_key not in nodes_by_key:
            errors.add(InSessionTaskGraphValidationCode.INVALID_PARENT_REFERENCE)
            continue
        if parent_key == node.node_key:
            errors.add(InSessionTaskGraphValidationCode.CYCLE_DETECTED)
            continue
        parent = nodes_by_key[parent_key]
        if parent.node_kind not in {InSessionTaskNodeKind.ROOT, InSessionTaskNodeKind.SUBTASK}:
            errors.add(InSessionTaskGraphValidationCode.INVALID_PARENT_KIND)


def _validate_tree_depth(
    nodes_by_key: dict[str, InSessionTaskNodeProposal],
    root_key: str,
    max_depth: int,
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    if root_key not in nodes_by_key:
        return
    if _has_parent_cycle(nodes_by_key):
        errors.add(InSessionTaskGraphValidationCode.CYCLE_DETECTED)
    children: dict[str, list[str]] = {key: [] for key in nodes_by_key}
    for node in nodes_by_key.values():
        if node.parent_node_key in children:
            children[node.parent_node_key].append(node.node_key)

    seen: set[str] = set()
    stack: list[tuple[str, int, frozenset[str]]] = [(root_key, 1, frozenset())]
    while stack:
        key, depth, lineage = stack.pop()
        if key in lineage:
            errors.add(InSessionTaskGraphValidationCode.CYCLE_DETECTED)
            continue
        if depth > max_depth:
            errors.add(InSessionTaskGraphValidationCode.DEPTH_LIMIT_EXCEEDED)
        seen.add(key)
        next_lineage = lineage | {key}
        for child_key in children.get(key, []):
            stack.append((child_key, depth + 1, next_lineage))

    if set(nodes_by_key) - seen:
        errors.add(InSessionTaskGraphValidationCode.ORPHAN_NODE)


def _has_parent_cycle(
    nodes_by_key: dict[str, InSessionTaskNodeProposal],
) -> bool:
    """即使环状分量与根节点断开，也能检测出环。"""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_key: str) -> bool:
        if node_key in visiting:
            return True
        if node_key in visited:
            return False
        visiting.add(node_key)
        parent_key = nodes_by_key[node_key].parent_node_key
        if parent_key in nodes_by_key and visit(parent_key):
            return True
        visiting.remove(node_key)
        visited.add(node_key)
        return False

    return any(visit(node_key) for node_key in nodes_by_key)


def _validate_nodes_source_and_coverage(
    nodes: Iterable[InSessionTaskNodeProposal],
    anchor_ids: set[str],
    authorization_anchor_ids: set[str],
    mapped_required_anchor_ids: set[str],
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    for node in nodes:
        _validate_source_refs(node.source_anchor_ids, anchor_ids, errors)
        node_authorization_anchor_ids = (
            set(node.source_anchor_ids) & authorization_anchor_ids
        )
        if not node_authorization_anchor_ids:
            errors.add(InSessionTaskGraphValidationCode.NODE_WITHOUT_AUTHORIZED_SOURCE)
        # A4a 没有类型化且绑定来源的约束对象。持久化模型直接生成的约束会静默扩大用户任务，
        # 因此在 A4b 定义其来源契约前采用失败关闭策略。
        if node.constraints:
            errors.add(InSessionTaskGraphValidationCode.UNSOURCED_CONSTRAINT)

        acceptance_ids: set[str] = set()
        for acceptance in node.acceptance_criteria:
            if acceptance.acceptance_id in acceptance_ids:
                errors.add(InSessionTaskGraphValidationCode.DUPLICATE_ACCEPTANCE_ID)
            acceptance_ids.add(acceptance.acceptance_id)
            _validate_source_refs(acceptance.source_anchor_ids, anchor_ids, errors)
            if not set(acceptance.source_anchor_ids) & authorization_anchor_ids:
                errors.add(
                    InSessionTaskGraphValidationCode.ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE
                )
            mapped_required_anchor_ids.update(acceptance.source_anchor_ids)


def _validate_authorization_anchor_kinds(
    context: InSessionTaskGraphValidationContext,
    authorization_anchor_ids: set[str],
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    """让检索/记忆事实只充当证据；它们不能单独创建任务。

    A4a 尚未为 ``previously_authorized_task_state`` 定义带修订限定的引用。在此之前，
    只有当前用户持有的材料能够授权新图。其他来源类型仍可作为上下文引用，
    但每个节点和 Acceptance 都必须同时带有一个此类授权锚点。
    """

    anchors_by_id = {anchor.anchor_id: anchor for anchor in context.source_anchors}
    allowed_kinds = {"current_user_instruction", "current_user_context"}
    for anchor_id in authorization_anchor_ids:
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            continue
        if anchor.source_kind not in allowed_kinds:
            errors.add(InSessionTaskGraphValidationCode.UNAUTHORIZED_AUTHORIZATION_SOURCE)
        if anchor.source_turn_id != context.source_turn_id:
            errors.add(InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH)


def _validate_revision_anchor_turns(
    context: InSessionTaskGraphRevisionValidationContext,
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    """将当前用户材料绑定到请求此次提交的 Turn。"""

    current_user_kinds = {"current_user_instruction", "current_user_context"}
    if any(
        anchor.source_kind in current_user_kinds
        and anchor.source_turn_id != context.source_turn_id
        for anchor in context.source_anchors
    ):
        errors.add(InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH)


def _validate_revision_authorization_anchor_kinds(
    context: InSessionTaskGraphRevisionValidationContext,
    authorization_anchor_ids: set[str],
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    """只允许当前用户或经 Host 验证的既有 Task 状态授予范围。"""

    anchors_by_id = {anchor.anchor_id: anchor for anchor in context.source_anchors}
    allowed_kinds = {
        "current_user_instruction",
        "current_user_context",
        "previously_authorized_task_state",
    }
    for anchor_id in authorization_anchor_ids:
        anchor = anchors_by_id.get(anchor_id)
        if anchor is not None and anchor.source_kind not in allowed_kinds:
            errors.add(InSessionTaskGraphValidationCode.UNAUTHORIZED_AUTHORIZATION_SOURCE)


def _validate_source_refs(
    source_anchor_ids: Iterable[str],
    known_anchor_ids: set[str],
    errors: set[InSessionTaskGraphValidationCode],
) -> None:
    refs = tuple(source_anchor_ids)
    if not refs:
        errors.add(InSessionTaskGraphValidationCode.MISSING_SOURCE_ANCHOR)
        return
    if not set(refs).issubset(known_anchor_ids):
        errors.add(InSessionTaskGraphValidationCode.UNKNOWN_SOURCE_ANCHOR)
