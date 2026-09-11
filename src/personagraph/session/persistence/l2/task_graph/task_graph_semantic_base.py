"""Store 所有的一个当前正 base TaskGraph Prompt 投影。

只有 :class:`TaskGraphSemanticBaseSnapshot` 是 Prompt 安全的。相邻别名绑定刻意保留在
Host 侧，由正 base 提交守卫消费，以恢复持久节点身份。
"""

from __future__ import annotations

from enum import StrEnum
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from personagraph.l2.auxiliary_graph import (
    TaskGraphSemanticBaseNode,
    TaskGraphSemanticBaseSnapshot,
)
from personagraph.l2.task_graph import InSessionTaskNodeRevisionDefinition
from ..auxiliary_graph import auxiliary_task_graph_commit as task_graph_commit_records
from . import insession_tasks as task_records
from ..work_run import work_verification as verification_records
from ...deps import StoreDeps


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_NODES = 512


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGraphSemanticBaseProjectionFailureCode(StrEnum):
    UNKNOWN_TASK = "unknown_task"
    REVISION_NOT_CURRENT = "revision_not_current"
    SOURCE_SNAPSHOT_CORRUPT = "source_snapshot_corrupt"
    GRAPH_AUTHORITY_CORRUPT = "graph_authority_corrupt"
    DELIVERY_AUTHORITY_MISSING = "delivery_authority_missing"
    DELIVERY_AUTHORITY_AMBIGUOUS = "delivery_authority_ambiguous"
    DELIVERY_AUTHORITY_CORRUPT = "delivery_authority_corrupt"
    CARRY_AUTHORITY_CORRUPT = "carry_authority_corrupt"
    PROJECTION_INVALID = "projection_invalid"


class TaskGraphSemanticBaseProjectionError(RuntimeError):
    def __init__(
        self,
        code: TaskGraphSemanticBaseProjectionFailureCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class TaskGraphSemanticBaseProjection(_Record):
    """Prompt 快照及精确 Host 私有别名命名空间。"""

    schema_version: Literal["task-graph-semantic-base-projection-v1"] = (
        "task-graph-semantic-base-projection-v1"
    )
    snapshot: TaskGraphSemanticBaseSnapshot
    base_node_alias_bindings: tuple[
        task_graph_commit_records.AuxiliaryBaseNodeAliasBinding, ...
    ] = Field(min_length=1, max_length=_MAX_NODES)

    @model_validator(mode="after")
    def _validate_exact_alias_namespace(self) -> 'TaskGraphSemanticBaseProjection':
        aliases = tuple(item.node_alias for item in self.base_node_alias_bindings)
        if aliases != tuple(item.node_alias for item in self.snapshot.nodes):
            raise ValueError("base alias bindings must exactly cover snapshot nodes")
        if len({item.insession_task_node_id for item in self.base_node_alias_bindings}) != len(
            self.base_node_alias_bindings
        ):
            raise ValueError("base alias bindings must name unique durable nodes")
        return self


class _CompletedDeliveryAuthority(_Record):
    delivery_id: str
    source_graph_revision: int = Field(ge=1)
    content: str


def project_task_graph_semantic_base(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> TaskGraphSemanticBaseProjection:
    """根据精确当前 Store 状态重新派生有界语义 base。

    只有解析出已完成节点的唯一冻结 PASS Delivery 后才会纳入。来自旧图 revision 的
    Delivery 还需要进入 ``graph_revision`` 的精确不可变结转回执。权威信息缺失、含糊或
    漂移都会中止整个投影。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    if not isinstance(graph_revision, int) or isinstance(graph_revision, bool):
        raise TypeError("graph_revision must be an integer")
    if graph_revision < 1:
        raise ValueError("graph_revision must be positive")

    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task = conn.execute(
            "SELECT session_id, current_graph_revision FROM insession_tasks "
            "WHERE insession_task_id=?",
            (task_id,),
        ).fetchone()
        if task is None or str(task["session_id"]) != session_id:
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.UNKNOWN_TASK,
                "Task is unknown in this Session",
            )
        current = task["current_graph_revision"]
        if current is None or int(current) != graph_revision:
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.REVISION_NOT_CURRENT,
                "requested TaskGraph revision is not current",
            )

        revision = conn.execute(
            "SELECT proposal_hash FROM insession_task_graph_revisions "
            "WHERE insession_task_id=? AND graph_revision=?",
            (task_id, graph_revision),
        ).fetchone()
        if revision is None or not _is_sha256(revision["proposal_hash"]):
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.SOURCE_SNAPSHOT_CORRUPT,
                "current TaskGraph revision proposal hash is missing or corrupt",
            )
        source_snapshot_sha256 = str(revision["proposal_hash"])

        rows = conn.execute(
            "SELECT node.*, edge.parent_insession_task_node_id, state.status, "
            "state.state_version "
            "FROM insession_task_graph_nodes AS node "
            "LEFT JOIN insession_task_graph_edges AS edge "
            "ON edge.insession_task_id=node.insession_task_id "
            "AND edge.graph_revision=node.graph_revision "
            "AND edge.child_insession_task_node_id=node.insession_task_node_id "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "ORDER BY node.ordinal, node.insession_task_node_id",
            (task_id, graph_revision),
        ).fetchall()
        if not rows or len(rows) > _MAX_NODES:
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.GRAPH_AUTHORITY_CORRUPT,
                "current TaskGraph node set is empty or exceeds the projection bound",
            )
        ordinals = tuple(int(row["ordinal"]) for row in rows)
        if ordinals != tuple(range(len(rows))):
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.GRAPH_AUTHORITY_CORRUPT,
                "current TaskGraph node ordinals are not canonical",
            )
        node_ids = tuple(str(row["insession_task_node_id"]) for row in rows)
        if len(node_ids) != len(set(node_ids)):
            _fail(
                TaskGraphSemanticBaseProjectionFailureCode.GRAPH_AUTHORITY_CORRUPT,
                "current TaskGraph contains duplicate durable nodes",
            )
        aliases = tuple(_node_alias(ordinal) for ordinal in ordinals)
        alias_by_node_id = dict(zip(node_ids, aliases, strict=True))

        delivery_by_node_id: dict[str, _CompletedDeliveryAuthority] = {}
        for row in rows:
            if str(row["status"]) != "completed":
                continue
            node_id = str(row["insession_task_node_id"])
            delivery_by_node_id[node_id] = _load_completed_delivery_authority(
                conn,
                session_id=session_id,
                task_id=task_id,
                graph_revision=graph_revision,
                node_id=node_id,
                node_revision=int(row["node_revision"]),
            )

        rows_by_id = {str(row["insession_task_node_id"]): row for row in rows}
        children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for row in rows:
            parent_id = row["parent_insession_task_node_id"]
            if parent_id is not None:
                parent = str(parent_id)
                if parent not in children_by_parent:
                    _fail(
                        TaskGraphSemanticBaseProjectionFailureCode.GRAPH_AUTHORITY_CORRUPT,
                        "current TaskGraph edge references an unknown parent",
                    )
                children_by_parent[parent].append(str(row["insession_task_node_id"]))

        for node_id, delivery in delivery_by_node_id.items():
            child_ids = tuple(children_by_parent[node_id])
            if any(child_id not in delivery_by_node_id for child_id in child_ids):
                _fail(
                    TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_CORRUPT,
                    "completed TaskNode has an incomplete Delivery dependency closure",
                )
            dependency_delivery_ids = tuple(
                delivery_by_node_id[child_id].delivery_id for child_id in child_ids
            )
            if delivery.source_graph_revision == graph_revision:
                _require_no_current_carry_receipt(
                    conn,
                    task_id=task_id,
                    graph_revision=graph_revision,
                    node_id=node_id,
                    node_revision=int(rows_by_id[node_id]["node_revision"]),
                )
            else:
                _authenticate_current_carry_receipt(
                    conn,
                    session_id=session_id,
                    task_id=task_id,
                    graph_revision=graph_revision,
                    row=rows_by_id[node_id],
                    delivery=delivery,
                    dependency_delivery_ids=dependency_delivery_ids,
                )

        try:
            projected_nodes = tuple(
                _project_node(
                    row,
                    alias=alias_by_node_id[str(row["insession_task_node_id"])],
                    parent_alias=(
                        None
                        if row["parent_insession_task_node_id"] is None
                        else alias_by_node_id[str(row["parent_insession_task_node_id"])]
                    ),
                    delivery=delivery_by_node_id.get(
                        str(row["insession_task_node_id"])
                    ),
                )
                for row in rows
            )
            roots = tuple(
                node.node_alias
                for node in projected_nodes
                if node.parent_node_alias is None
            )
            if len(roots) != 1:
                raise ValueError("current TaskGraph must have exactly one root")
            snapshot = TaskGraphSemanticBaseSnapshot.create(
                base_task_graph_revision=graph_revision,
                root_node_alias=roots[0],
                nodes=projected_nodes,
                source_snapshot_sha256=source_snapshot_sha256,
            )
            bindings = tuple(
                task_graph_commit_records.AuxiliaryBaseNodeAliasBinding(
                    node_alias=alias,
                    insession_task_node_id=node_id,
                )
                for alias, node_id in zip(aliases, node_ids, strict=True)
            )
            result = TaskGraphSemanticBaseProjection(
                snapshot=snapshot,
                base_node_alias_bindings=bindings,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            _fail_from(
                TaskGraphSemanticBaseProjectionFailureCode.PROJECTION_INVALID,
                "current TaskGraph cannot form a canonical semantic base projection",
                exc,
            )
        conn.commit()
        return result
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _project_node(
    row: object,
    *,
    alias: str,
    parent_alias: str | None,
    delivery: _CompletedDeliveryAuthority | None,
) -> TaskGraphSemanticBaseNode:
    values = task_records._node_projection(row)  # type: ignore[arg-type]
    completed = str(values["status"]) == "completed"
    if completed != (delivery is not None):
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_MISSING,
            "completed TaskNode does not have exact Delivery authority",
        )
    source_aliases = tuple(
        sorted(str(item) for item in values["source_anchor_ids"])
    )
    return TaskGraphSemanticBaseNode(
        node_alias=alias,
        node_revision=int(values["node_revision"]),
        node_kind=str(values["node_kind"]),
        parent_node_alias=parent_alias,
        title=str(values["title"]),
        objective=str(values["objective"]),
        source_anchor_aliases=source_aliases,
        acceptance_criteria=tuple(values["acceptance_criteria"]),
        constraints=tuple(str(item) for item in values["constraints"]),
        status=str(values["status"]),
        completed_delivery_summary=(None if delivery is None else delivery.content),
        completed_delivery_coverage=(None if delivery is None else "full"),
        delivery_authority_aliases=(
            () if delivery is None else source_aliases
        ),
    )


def _load_completed_delivery_authority(
    conn: object,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    node_id: str,
    node_revision: int,
) -> _CompletedDeliveryAuthority:
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT delivery_id FROM insession_task_node_deliveries "
        "WHERE session_id=? AND insession_task_id=? "
        "AND insession_task_node_id=? AND node_revision=? "
        "ORDER BY graph_revision, delivery_id",
        (session_id, task_id, node_id, node_revision),
    ).fetchall()
    if not rows:
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_MISSING,
            "completed TaskNode has no Delivery authority",
        )
    if len(rows) != 1:
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_AMBIGUOUS,
            "completed TaskNode has ambiguous Delivery authority",
        )
    delivery_id = str(rows[0]["delivery_id"])
    try:
        resolved = verification_records._load_task_node_delivery(
            conn,
            session_id=session_id,
            delivery_id=delivery_id,
        )
    except (
        verification_records.WorkExecutionPersistenceError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        _fail_from(
            TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_CORRUPT,
            "completed TaskNode Delivery authority is corrupt",
            exc,
        )
    subject = resolved.delivery.subject
    if (
        subject.task_id != task_id
        or subject.node_id != node_id
        or subject.node_revision != node_revision
        or subject.graph_revision > graph_revision
    ):
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.DELIVERY_AUTHORITY_CORRUPT,
            "completed TaskNode Delivery names an invalid graph subject",
        )
    return _CompletedDeliveryAuthority(
        delivery_id=delivery_id,
        source_graph_revision=subject.graph_revision,
        content=resolved.output_window.content,
    )


def _require_no_current_carry_receipt(
    conn: object,
    *,
    task_id: str,
    graph_revision: int,
    node_id: str,
    node_revision: int,
) -> None:
    row = conn.execute(  # type: ignore[attr-defined]
        "SELECT carry_receipt_id FROM "
        "insession_auxiliary_v2_task_graph_node_carry_receipts "
        "WHERE insession_task_id=? AND target_task_graph_revision=? "
        "AND insession_task_node_id=? AND node_revision=?",
        (task_id, graph_revision, node_id, node_revision),
    ).fetchone()
    if row is not None:
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "direct current Delivery has an unexpected carry receipt",
        )


def _authenticate_current_carry_receipt(
    conn: object,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    row: object,
    delivery: _CompletedDeliveryAuthority,
    dependency_delivery_ids: tuple[str, ...],
) -> None:
    node_id = str(row["insession_task_node_id"])  # type: ignore[index]
    node_revision = int(row["node_revision"])  # type: ignore[index]
    carry_rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT * FROM insession_auxiliary_v2_task_graph_node_carry_receipts "
        "WHERE session_id=? AND insession_task_id=? "
        "AND target_task_graph_revision=? AND insession_task_node_id=? "
        "AND node_revision=?",
        (session_id, task_id, graph_revision, node_id, node_revision),
    ).fetchall()
    if len(carry_rows) != 1:
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "carried completed TaskNode has no unique current carry receipt",
        )
    carry_row = carry_rows[0]
    receipt_json = str(carry_row["receipt_json"])
    try:
        receipt = (
            task_graph_commit_records.AuxiliaryTaskGraphNodeCarryReceipt.model_validate_json(
                receipt_json
            )
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_from(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "current completed-node carry receipt is not typed",
            exc,
        )

    definition = _definition_from_row(row)
    definition_sha256 = _sha256_value(definition.model_dump(mode="json"))
    scalar_bindings = {
        "carry_receipt_id": receipt.carry_receipt_id,
        "apply_id": receipt.apply_id,
        "session_id": receipt.session_id,
        "insession_task_id": receipt.task_id,
        "base_task_graph_revision": receipt.base_task_graph_revision,
        "target_task_graph_revision": receipt.target_task_graph_revision,
        "insession_task_node_id": receipt.node_id,
        "node_revision": receipt.node_revision,
        "source_delivery_id": receipt.source_delivery_id,
        "definition_sha256": receipt.definition_sha256,
        "dependency_closure_sha256": receipt.dependency_closure_sha256,
        "source_authority_sha256": receipt.source_authority_sha256,
        "capability_catalog_sha256": receipt.capability_catalog_sha256,
        "freshness_authority_sha256": receipt.freshness_authority_sha256,
    }
    if (
        _model_json(receipt) != receipt_json
        or _sha256_text(receipt_json) != str(carry_row["receipt_sha256"])
        or any(str(carry_row[column]) != str(value) for column, value in scalar_bindings.items())
        or _canonical_json(list(receipt.dependency_delivery_ids))
        != str(carry_row["dependency_delivery_ids_json"])
        or receipt.session_id != session_id
        or receipt.task_id != task_id
        or receipt.base_task_graph_revision != graph_revision - 1
        or receipt.target_task_graph_revision != graph_revision
        or receipt.node_id != node_id
        or receipt.node_revision != node_revision
        or receipt.source_delivery_id != delivery.delivery_id
        or receipt.definition_sha256 != definition_sha256
        or receipt.dependency_delivery_ids != dependency_delivery_ids
    ):
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "current completed-node carry receipt does not bind the current graph",
        )
    _authenticate_carry_commit_receipt(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        carry_receipt_id=receipt.carry_receipt_id,
        apply_id=receipt.apply_id,
    )


def _authenticate_carry_commit_receipt(
    conn: object,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    carry_receipt_id: str,
    apply_id: str,
) -> None:
    row = conn.execute(  # type: ignore[attr-defined]
        "SELECT * FROM insession_auxiliary_v2_task_graph_commit_receipts "
        "WHERE apply_id=?",
        (apply_id,),
    ).fetchone()
    if row is None:
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "completed-node carry lost its TaskGraph commit receipt",
        )
    result_json = str(row["result_json"])
    try:
        result = task_graph_commit_records.AuxiliaryTaskGraphCommitResult.model_validate_json(
            result_json
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_from(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "completed-node carry TaskGraph commit receipt is not typed",
            exc,
        )
    if (
        _model_json(result) != result_json
        or _sha256_text(result_json) != str(row["result_sha256"])
        or str(row["session_id"]) != session_id
        or str(row["insession_task_id"]) != task_id
        or int(row["base_task_graph_revision"]) != graph_revision - 1
        or int(row["committed_task_graph_revision"]) != graph_revision
        or result.apply_id != apply_id
        or result.task_id != task_id
        or result.previous_graph_revision != graph_revision - 1
        or result.committed_graph_revision != graph_revision
        or carry_receipt_id not in result.carry_receipt_ids
    ):
        _fail(
            TaskGraphSemanticBaseProjectionFailureCode.CARRY_AUTHORITY_CORRUPT,
            "completed-node carry TaskGraph commit receipt is corrupt",
        )


def _definition_from_row(row: object) -> InSessionTaskNodeRevisionDefinition:
    values = task_records._node_projection(row)  # type: ignore[arg-type]
    return InSessionTaskNodeRevisionDefinition(
        node_kind=str(values["node_kind"]),
        parent_node_id=(
            None
            if values["parent_insession_task_node_id"] is None
            else str(values["parent_insession_task_node_id"])
        ),
        title=str(values["title"]),
        objective=str(values["objective"]),
        source_anchor_ids=tuple(str(item) for item in values["source_anchor_ids"]),
        acceptance_criteria=tuple(values["acceptance_criteria"]),
        constraints=tuple(str(item) for item in values["constraints"]),
    )


def _node_alias(ordinal: int) -> str:
    return f"base_node_{ordinal:03d}"


def _require_identifier(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a valid identifier")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _model_json(value: BaseModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _fail(
    code: TaskGraphSemanticBaseProjectionFailureCode,
    message: str,
) -> None:
    raise TaskGraphSemanticBaseProjectionError(code, message)


def _fail_from(
    code: TaskGraphSemanticBaseProjectionFailureCode,
    message: str,
    cause: BaseException,
) -> None:
    raise TaskGraphSemanticBaseProjectionError(code, message) from cause


__all__ = [
    "TaskGraphSemanticBaseProjectionError",
    'TaskGraphSemanticBaseProjectionFailureCode',
    'TaskGraphSemanticBaseProjection',
    "project_task_graph_semantic_base",
]
