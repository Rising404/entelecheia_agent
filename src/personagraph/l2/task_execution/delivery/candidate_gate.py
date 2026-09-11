"""对一个规范根候选项执行冻结前整 Task 审查。

节点验证器仍持有节点 Acceptance。此门只在该验证器通过后运行，并且仅当当前规范根仍可变、
每个非根节点均已完成且不存在根 Delivery 时运行。因此，其结果可以将仅根节点的执行缺陷
路由回普通同 WorkRun Attempt 循环，而绝不冻结候选项。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.model_io.tier_bindings import ModelTierBinding, ModelTier, resolve_tier
from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskNodeKind,
    TaskDeliveryValidationChildDelivery,
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationNode,
    TaskDeliveryValidationNodeVerificationAttestation,
    TaskDeliveryValidationPrompt,
    TaskDeliveryValidationRequest,
    TaskDeliveryValidationResult,
    require_root_only_task_delivery_execution_retry,
)
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.l2.work_run import (
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    NodeVerificationResult,
    TaskNodeSubject,
)
from personagraph.runtime.model_calls.contracts import (
    DurableLogicalModelCallAuthority,
    DurableModelCallStateGuardRejected,
)
from personagraph.runtime.model_calls.contracts import (
    RuntimeModelLedgerStore,
    RuntimeModelLogicalRequest,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls.policy import MAX_MODEL_ATTEMPTS
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.l2.task_execution.verification.controller import (
    NodeDownstreamVerificationGate,
    NodeDownstreamVerificationRouteRequired,
    NodeVerificationApplicationRequest,
    NodeVerificationResumeRequest,
    PreparedNodeVerification,
)
from personagraph.runtime.model_calls.authority import (
    RuntimeLogicalModelCallAuthority,
)
from .validation import request_task_delivery_validation
from .model_contracts import (
    PURPOSE,
    REQUEST_CONTRACT,
    RESULT_CONTRACT,
    TaskDeliveryValidationStructuredProvider,
    _SYSTEM_PROMPT,
    task_delivery_validation_model_payload,
)
from .model_provider import (
    build_task_delivery_validation_structured_provider,
)
from personagraph.runtime.turn_events import TurnEvent
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskDeliveryCandidateAuthority(_Contract):
    """绑定到一个模型请求的精确当前根候选项事实。"""

    schema_version: Literal["task-delivery-candidate-authority-v1"] = (
        "task-delivery-candidate-authority-v1"
    )
    session_id: str = Field(min_length=1, max_length=200)
    invocation_turn_id: str = Field(min_length=1, max_length=200)
    subject: TaskNodeSubject
    task_state_version: int = Field(ge=1)
    work_run_id: str = Field(min_length=1, max_length=200)
    submitted_attempt_id: str = Field(min_length=1, max_length=200)
    verification_request_id: str = Field(min_length=1, max_length=200)
    verification_request_revision: int = Field(ge=1)
    output_revision: int = Field(ge=1)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_delivery_id: str = Field(min_length=1, max_length=200)
    review_request: TaskDeliveryValidationRequest
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_bindings(self) -> "TaskDeliveryCandidateAuthority":
        prompt = self.review_request.prompt
        if (
            self.session_id != prompt.session_id
            or self.invocation_turn_id
            != self.review_request.invocation_turn_id
            or self.subject.task_id != prompt.task_id
            or self.subject.graph_revision != prompt.graph_revision
            or self.subject.node_id != prompt.task_id
            or self.task_state_version != prompt.task_state_version
            or self.candidate_delivery_id != prompt.root_delivery_id
            or self.output_sha256
            != hashlib.sha256(
                prompt.root_output_body.encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError("candidate review authority crossed frozen facts")
        expected = _sha256(
            self.model_dump(mode="json", exclude={"authority_sha256"})
        )
        if self.authority_sha256 != expected:
            raise ValueError("candidate review authority hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> "TaskDeliveryCandidateAuthority":
        values = dict(values)
        values["authority_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["authority_sha256"] = _sha256(
            provisional.model_dump(mode="json", exclude={"authority_sha256"})
        )
        return cls.model_validate(values)


class TaskDeliveryCandidateUnsupportedRoute(
    NodeDownstreamVerificationRouteRequired
):
    """要求调用方应用专用候选结算路由。"""

    def __init__(
        self,
        *,
        authority: TaskDeliveryCandidateAuthority,
        result: TaskDeliveryValidationResult,
    ) -> None:
        self.disposition = result.disposition
        self.result_id = result.verification_result_id
        self.result_sha256 = result.result_sha256
        super().__init__(
            settlement_intent=_settlement_intent(authority, result)
        )


class TaskDeliveryCandidateModelAuthorityFactory(Protocol):
    def __call__(
        self,
        authority: TaskDeliveryCandidateAuthority,
        *,
        rederive_state_guard_sha256: Callable[[], str],
    ) -> DurableLogicalModelCallAuthority: ...


NodeDownstreamVerificationGateFactory = Callable[
    [NodeVerificationApplicationRequest | NodeVerificationResumeRequest],
    NodeDownstreamVerificationGate,
]


def create_task_delivery_candidate_model_call_authority(
    authority: TaskDeliveryCandidateAuthority,
    *,
    rederive_state_guard_sha256: Callable[[], str],
    ledger_store: RuntimeModelLedgerStore,
    model_binding: ModelTierBinding | None = None,
) -> RuntimeLogicalModelCallAuthority:
    """为根候选项预留一次提供商中立的持久调用。"""

    if not isinstance(authority, TaskDeliveryCandidateAuthority):
        raise TypeError("candidate model authority requires its exact request")
    selected_model_binding = model_binding or resolve_tier(ModelTier.FINAL_GATE)
    if (
        not isinstance(selected_model_binding, ModelTierBinding)
        or selected_model_binding.tier is not ModelTier.FINAL_GATE
    ):
        raise TypeError("candidate validation requires the final-gate model tier")
    identity = configured_structured_model_endpoint_identity(
        selected_model_binding
    )
    structured_prompt = RuntimeModelStructuredPrompt.create(
        system_prompt=_SYSTEM_PROMPT,
        user_content=_canonical_json(
            authority.review_request.prompt.model_dump(mode="json")
        ),
    )
    logical = RuntimeModelLogicalRequest.create(
        logical_call_id=authority.review_request.logical_call_id,
        session_id=authority.session_id,
        task_id=authority.subject.task_id,
        auxiliary_graph_id=None,
        goal_id=None,
    # 通用账本的 execution_subject_id 是指向单独物化主体权威行的可选外键。
    # 精确 TaskNode 已冻结在此候选项权威信息中，因此不要仅根据节点标识伪造该无关外键。
        execution_subject_id=None,
        invocation_turn_id=authority.invocation_turn_id,
        call_kind="task_delivery_candidate_validation",
        purpose=PURPOSE,
        provider=identity.provider,
        model=identity.model,
        endpoint_fingerprint=identity.endpoint_fingerprint,
        request_contract=REQUEST_CONTRACT,
        request_payload=authority.model_dump(mode="json"),
        output_repair_protocol=(
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        ),
        structured_prompt=structured_prompt,
        typed_result_contract=RESULT_CONTRACT,
        max_physical_attempts=MAX_MODEL_ATTEMPTS,
        state_guard_sha256=authority.authority_sha256,
    )

    def typed_replay_payload(_model_result: object, value: object) -> object:
        if not isinstance(value, TaskDeliveryValidationResult):
            raise TypeError("candidate replay requires a validation result")
        return task_delivery_validation_model_payload(value)

    return RuntimeLogicalModelCallAuthority(
        logical_request=logical,
        state_guard_sha256=rederive_state_guard_sha256,
        typed_replay_payload_builder=typed_replay_payload,
        model_binding=selected_model_binding,
        store=ledger_store,
    )


def build_task_delivery_candidate_gate_factory(
    *,
    provider: TaskDeliveryValidationStructuredProvider | None,
    emit: Callable[[TurnEvent], object],
    model_call_authority_factory: TaskDeliveryCandidateModelAuthorityFactory,
    deadline: TurnDeadline | None = None,
) -> NodeDownstreamVerificationGateFactory:
    """绑定物理审查端口，同时逐 Attempt 延迟候选项 ID。"""

    if not callable(emit):
        raise TypeError("candidate review event sink must be callable")
    if not callable(model_call_authority_factory):
        raise TypeError("candidate model authority factory must be callable")
    selected_provider = (
        provider or build_task_delivery_validation_structured_provider()
    )

    def factory(
        request: NodeVerificationApplicationRequest
        | NodeVerificationResumeRequest,
    ) -> NodeDownstreamVerificationGate:
        if not isinstance(
            request,
            (NodeVerificationApplicationRequest, NodeVerificationResumeRequest),
        ):
            raise TypeError("candidate gate requires a node verification request")
        return _TaskDeliveryCandidateGate(
            candidate_delivery_id=request.delivery_id,
            provider=selected_provider,
            emit=emit,
            authority_factory=model_call_authority_factory,
            deadline=deadline,
        )

    return factory


@dataclass(frozen=True, slots=True)
class _TaskDeliveryCandidateGate:
    candidate_delivery_id: str
    provider: TaskDeliveryValidationStructuredProvider
    emit: Callable[[TurnEvent], object]
    authority_factory: TaskDeliveryCandidateModelAuthorityFactory
    deadline: TurnDeadline | None

    def __call__(
        self,
        prepared: PreparedNodeVerification,
        node_result: NodeVerificationResult,
    ) -> tuple[DownstreamVerificationFeedback, ...]:
        if not node_result.all_pass:
            return ()
        authority = _project_candidate_authority(
            prepared=prepared,
            node_result=node_result,
            candidate_delivery_id=self.candidate_delivery_id,
        )
        if authority is None:
            return ()

        def rederive_guard() -> str:
            current = _rederive_candidate_authority(
                authority=authority,
            )
            if current != authority:
                raise DurableModelCallStateGuardRejected(
                    "root candidate changed before model settlement"
                )
            return current.authority_sha256

        durable_call = self.authority_factory(
            authority,
            rederive_state_guard_sha256=rederive_guard,
        )
        requested = request_task_delivery_validation(
            authority.review_request,
            provider=self.provider,
            emit=self.emit,
            deadline=self.deadline,
            durable_call=durable_call,
        )
        result = requested.value
        if result.disposition is TaskDeliveryValidationDisposition.PASS:
        # PASS 也在根 NodeVerification 事务中结算，因此发布绝不会派发第二个整 Task 审查器。
            raise NodeDownstreamVerificationRouteRequired(
                settlement_intent=_settlement_intent(authority, result)
            )
        if (
            result.disposition
            is TaskDeliveryValidationDisposition.RETRY_EXECUTION
        ):
            require_root_only_task_delivery_execution_retry(
                result=result,
                root_node_id=authority.subject.node_id,
            )
            return (_feedback(result, disposition="retry_attempt"),)
        raise TaskDeliveryCandidateUnsupportedRoute(
            authority=authority,
            result=result,
        )


def _settlement_intent(
    authority: TaskDeliveryCandidateAuthority,
    result: TaskDeliveryValidationResult,
) -> task_delivery_store.TaskDeliveryCandidateSettlementIntent:
    return task_delivery_store.TaskDeliveryCandidateSettlementIntent(
        schema_version="task-delivery-candidate-settlement-intent-v1",
        **authority.model_dump(
            mode="python",
            exclude={"schema_version"},
        ),
        result=result,
    )


def _project_candidate_authority(
    *,
    prepared: PreparedNodeVerification,
    node_result: NodeVerificationResult,
    candidate_delivery_id: str,
) -> TaskDeliveryCandidateAuthority | None:
    context = prepared.context
    subject = context.subject
    if not isinstance(subject, TaskNodeSubject):
        return None
    root_candidate_attestation = _root_candidate_attestation(
        prepared=prepared,
        node_result=node_result,
        candidate_delivery_id=candidate_delivery_id,
    )
    return _authority_from_material(
        session_id=context.session_id,
        invocation_turn_id=context.invocation_turn_id,
        subject=subject,
        work_run_id=context.work_run_id,
        submitted_attempt_id=context.submitted_attempt_id,
        verification_request_id=prepared.verification_request_id,
        verification_request_revision=(
            prepared.verification_request_revision
        ),
        output_revision=context.locked_output_window.output_revision,
        output_format=context.locked_output_window.format.value,
        output_body=context.locked_output_window.content,
        candidate_delivery_id=candidate_delivery_id,
        root_candidate_attestation=root_candidate_attestation,
    )


def _rederive_candidate_authority(
    *,
    authority: TaskDeliveryCandidateAuthority,
) -> TaskDeliveryCandidateAuthority:
    persisted = verification_store.get_prepared_task_node_verification(
        session_id=authority.session_id,
        invocation_turn_id=authority.invocation_turn_id,
        verification_request_id=authority.verification_request_id,
    )
    request = persisted.record.request
    subject = request.subject
    if not isinstance(subject, TaskNodeSubject):
        raise DurableModelCallStateGuardRejected(
            "candidate is no longer a TaskNode"
        )
    attestations = authority.review_request.prompt.node_verification_attestations
    root_candidate_attestation = (
        next(
            item
            for item in attestations
            if item.node_id == authority.subject.node_id
        )
        if attestations
        else None
    )
    current = _authority_from_material(
        session_id=request.session_id,
        invocation_turn_id=persisted.invocation_turn_id,
        subject=subject,
        work_run_id=request.work_run_id,
        submitted_attempt_id=request.submitted_attempt_id,
        verification_request_id=request.verification_request_id,
        verification_request_revision=request.revision,
        output_revision=persisted.locked_output_window.output_revision,
        output_format=persisted.locked_output_window.format.value,
        output_body=persisted.locked_output_window.content,
        candidate_delivery_id=authority.candidate_delivery_id,
        include_child_delivery_projection=(
            authority.review_request.prompt.child_delivery_projection is not None
        ),
        root_candidate_attestation=root_candidate_attestation,
    )
    if current is None:
        raise DurableModelCallStateGuardRejected(
            "candidate is no longer the eligible canonical root"
        )
    return current


def _authority_from_material(
    *,
    session_id: str,
    invocation_turn_id: str,
    subject: TaskNodeSubject,
    work_run_id: str,
    submitted_attempt_id: str,
    verification_request_id: str,
    verification_request_revision: int,
    output_revision: int,
    output_format: str,
    output_body: str,
    candidate_delivery_id: str,
    include_child_delivery_projection: bool = True,
    root_candidate_attestation: (
        TaskDeliveryValidationNodeVerificationAttestation | None
    ) = None,
) -> TaskDeliveryCandidateAuthority | None:
    if subject.node_id != subject.task_id:
        return None
    details = task_graph_store.get_insession_task_details(
        session_id,
        subject.task_id,
    )
    if (
        details is None
        or details.status.value != "active"
        or details.current_graph_revision != subject.graph_revision
    ):
        return None
    roots = tuple(
        item
        for item in details.nodes
        if item.get("node_kind") == InSessionTaskNodeKind.ROOT.value
    )
    if (
        len(roots) != 1
        or roots[0].get("insession_task_node_id") != subject.task_id
        or roots[0].get("node_revision") != subject.node_revision
        or roots[0].get("status") != "active"
        or any(
            item.get("status") != "completed"
            for item in details.nodes
            if item.get("insession_task_node_id") != subject.task_id
        )
    ):
        return None
    nodes = tuple(_validation_node(item) for item in details.nodes)
    child_delivery_projection = (
        task_delivery_store.project_task_delivery_validation_child_deliveries(
            session_id=session_id,
            task_id=subject.task_id,
            graph_revision=subject.graph_revision,
        )
        if include_child_delivery_projection
        else None
    )
    node_verification_attestations = ()
    if root_candidate_attestation is not None:
        _require_current_root_candidate_attestation(
            attestation=root_candidate_attestation,
            session_id=session_id,
            subject=subject,
            work_run_id=work_run_id,
            submitted_attempt_id=submitted_attempt_id,
            verification_request_id=verification_request_id,
            verification_request_revision=verification_request_revision,
            output_revision=output_revision,
            candidate_delivery_id=candidate_delivery_id,
        )
        child_attestations = tuple(
            _verified_child_delivery_attestation(
                session_id=session_id,
                child=child,
            )
            for child in (
                child_delivery_projection.deliveries
                if child_delivery_projection is not None
                else ()
            )
        )
        by_node = {
            item.node_id: item
            for item in (root_candidate_attestation, *child_attestations)
        }
        node_verification_attestations = tuple(
            by_node[item.node_id]
            for item in sorted(nodes, key=lambda item: (item.ordinal, item.node_id))
        )
    prompt = TaskDeliveryValidationPrompt.create(
        session_id=session_id,
        task_id=subject.task_id,
        graph_revision=subject.graph_revision,
        task_state_version=details.task_state_version,
        title=details.title,
        objective=details.objective,
        source_anchors=details.source_anchors,
        nodes=nodes,
        root_delivery_id=candidate_delivery_id,
        root_output_format=output_format,
        root_output_body=output_body,
        child_delivery_projection=child_delivery_projection,
        node_verification_attestations=node_verification_attestations,
    )
    identity_payload: dict[str, object] = {
        "contract": "task-delivery-candidate-id-plan-v1",
        "session_id": session_id,
        "invocation_turn_id": invocation_turn_id,
        "subject": subject.model_dump(mode="json"),
        "task_state_version": details.task_state_version,
        "work_run_id": work_run_id,
        "submitted_attempt_id": submitted_attempt_id,
        "verification_request_id": verification_request_id,
        "verification_request_revision": verification_request_revision,
        "output_revision": output_revision,
        "output_sha256": hashlib.sha256(
            output_body.encode("utf-8")
        ).hexdigest(),
        "candidate_delivery_id": candidate_delivery_id,
    }
    if include_child_delivery_projection:
        identity_payload.update(
            {
                "contract": "task-delivery-candidate-id-plan-v2",
                "prompt_payload_sha256": prompt.payload_sha256,
            }
        )
    identity = _sha256(identity_payload)[:40]
    review_request = TaskDeliveryValidationRequest.create(
        verification_request_id=f"tdcv2_req_{identity}",
        verification_result_id=f"tdcv2_result_{identity}",
        logical_call_id=f"tdcv2_call_{identity}",
        verification_profile_id="whole-task-root-candidate-v2",
        invocation_turn_id=invocation_turn_id,
        prompt=prompt,
    )
    return TaskDeliveryCandidateAuthority.create(
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        subject=subject,
        task_state_version=details.task_state_version,
        work_run_id=work_run_id,
        submitted_attempt_id=submitted_attempt_id,
        verification_request_id=verification_request_id,
        verification_request_revision=verification_request_revision,
        output_revision=output_revision,
        output_sha256=hashlib.sha256(output_body.encode("utf-8")).hexdigest(),
        candidate_delivery_id=candidate_delivery_id,
        review_request=review_request,
    )


def _root_candidate_attestation(
    *,
    prepared: PreparedNodeVerification,
    node_result: NodeVerificationResult,
    candidate_delivery_id: str,
) -> TaskDeliveryValidationNodeVerificationAttestation:
    """绑定进行中的根语义 PASS，但不声明冻结 Delivery。"""

    context = prepared.context
    subject = context.subject
    if not isinstance(subject, TaskNodeSubject) or not node_result.all_pass:
        raise DurableModelCallStateGuardRejected(
            "root candidate attestation requires an exact node semantic PASS"
        )
    record = verification_store.get_task_node_verification_record(
        session_id=context.session_id,
        verification_request_id=prepared.verification_request_id,
    )
    request = record.request
    if (
        record.result is not None
        or request.status.value != "pending"
        or request.subject != subject
        or request.work_run_id != node_result.work_run_id
        or request.submitted_attempt_id != node_result.submitted_attempt_id
        or request.output_revision != node_result.output_revision
        or request.revision != node_result.verification_request_revision
        or tuple(
            item.acceptance_id for item in node_result.acceptance_results
        )
        != request.acceptance_ids
    ):
        raise DurableModelCallStateGuardRejected(
            "root candidate node PASS crossed its durable verification request"
        )
    return TaskDeliveryValidationNodeVerificationAttestation.create(
        node_id=subject.node_id,
        node_revision=subject.node_revision,
        verification_source="root_candidate",
        delivery_id=candidate_delivery_id,
        verified_subject_graph_revision=subject.graph_revision,
        verification_request_id=node_result.verification_request_id,
        verification_request_revision=(
            node_result.verification_request_revision
        ),
        work_run_id=node_result.work_run_id,
        submitted_attempt_id=node_result.submitted_attempt_id,
        output_revision=node_result.output_revision,
        acceptance_ids=request.acceptance_ids,
        supporting_tool_result_ids=request.supporting_tool_result_ids,
        all_pass=True,
        verification_result_sha256=_sha256(
            node_result.model_dump(mode="json")
        ),
    )


def _require_current_root_candidate_attestation(
    *,
    attestation: TaskDeliveryValidationNodeVerificationAttestation,
    session_id: str,
    subject: TaskNodeSubject,
    work_run_id: str,
    submitted_attempt_id: str,
    verification_request_id: str,
    verification_request_revision: int,
    output_revision: int,
    candidate_delivery_id: str,
) -> None:
    """重新检查冻结根证明使用的每个持久请求绑定。"""

    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=verification_request_id,
    )
    request = record.request
    if (
        attestation.verification_source != "root_candidate"
        or attestation.node_id != subject.node_id
        or attestation.node_revision != subject.node_revision
        or attestation.delivery_id != candidate_delivery_id
        or attestation.verified_subject_graph_revision != subject.graph_revision
        or attestation.verification_request_id != verification_request_id
        or attestation.verification_request_revision
        != verification_request_revision
        or attestation.work_run_id != work_run_id
        or attestation.submitted_attempt_id != submitted_attempt_id
        or attestation.output_revision != output_revision
        or request.status.value != "pending"
        or record.result is not None
        or request.subject != subject
        or request.work_run_id != work_run_id
        or request.submitted_attempt_id != submitted_attempt_id
        or request.output_revision != output_revision
        or request.revision != verification_request_revision
        or attestation.acceptance_ids != request.acceptance_ids
        or attestation.supporting_tool_result_ids
        != request.supporting_tool_result_ids
    ):
        raise DurableModelCallStateGuardRejected(
            "root candidate attestation no longer matches its durable request"
        )


def _verified_child_delivery_attestation(
    *,
    session_id: str,
    child: TaskDeliveryValidationChildDelivery,
) -> TaskDeliveryValidationNodeVerificationAttestation:
    """投影一个已冻结子节点 PASS，但不复制证据正文。"""

    resolved = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=child.delivery_id,
    )
    delivery = resolved.delivery
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=delivery.verification_request_id,
    )
    result = record.result
    if (
        result is None
        or not result.all_pass
        or delivery.subject.node_id != child.node_id
        or delivery.subject.node_revision != child.node_revision
        or delivery.subject.graph_revision != child.source_graph_revision
        or result.subject != delivery.subject
        or result.work_run_id != delivery.work_run_id
        or result.submitted_attempt_id != delivery.submitted_attempt_id
        or result.output_revision != delivery.output_revision
    ):
        raise DurableModelCallStateGuardRejected(
            "child Delivery attestation lost its persisted node PASS"
        )
    return TaskDeliveryValidationNodeVerificationAttestation.create(
        node_id=child.node_id,
        node_revision=child.node_revision,
        verification_source=f"{child.resolution_kind}_delivery",
        delivery_id=child.delivery_id,
        verified_subject_graph_revision=delivery.subject.graph_revision,
        verification_request_id=result.verification_request_id,
        verification_request_revision=result.verification_request_revision,
        work_run_id=result.work_run_id,
        submitted_attempt_id=result.submitted_attempt_id,
        output_revision=result.output_revision,
        acceptance_ids=record.request.acceptance_ids,
        supporting_tool_result_ids=record.request.supporting_tool_result_ids,
        all_pass=True,
        verification_result_sha256=_sha256(
            result.model_dump(mode="json")
        ),
    )


def _validation_node(raw: dict[str, object]) -> TaskDeliveryValidationNode:
    return TaskDeliveryValidationNode(
        node_id=str(raw["insession_task_node_id"]),
        node_revision=int(raw["node_revision"]),
        node_kind=InSessionTaskNodeKind(str(raw["node_kind"])),
        parent_node_id=(
            str(raw["parent_insession_task_node_id"])
            if raw.get("parent_insession_task_node_id") is not None
            else None
        ),
        ordinal=int(raw["ordinal"]),
        title=str(raw["title"]),
        objective=str(raw["objective"]),
        source_anchor_ids=tuple(str(item) for item in raw["source_anchor_ids"]),
        acceptance_criteria=tuple(
            InSessionTaskAcceptanceProposal.model_validate(item)
            for item in raw["acceptance_criteria"]
        ),
        constraints=tuple(str(item) for item in raw["constraints"]),
    )


def _feedback(
    result: TaskDeliveryValidationResult,
    *,
    disposition: str,
) -> DownstreamVerificationFeedback:
    if disposition == "pass":
        return DownstreamVerificationFeedback(
            gate_id="whole_task_delivery_v2",
            disposition=DownstreamVerificationDisposition.PASS,
            finding=result.summary,
            source_result_id=result.verification_result_id,
            source_result_sha256=result.result_sha256,
        )
    return DownstreamVerificationFeedback(
        gate_id="whole_task_delivery_v2",
        disposition=DownstreamVerificationDisposition.RETRY_ATTEMPT,
        finding=result.summary,
        repair_objective=result.execution_repair_objective,
        source_result_id=result.verification_result_id,
        source_result_sha256=result.result_sha256,
        affected_subject_ids=tuple(
            sorted(
                {
                    node_id
                    for finding in result.findings
                    for node_id in finding.affected_node_ids
                }
            )
        ),
    )


__all__ = [
    "NodeDownstreamVerificationGateFactory",
    "TaskDeliveryCandidateAuthority",
    "TaskDeliveryCandidateModelAuthorityFactory",
    "TaskDeliveryCandidateUnsupportedRoute",
    "build_task_delivery_candidate_gate_factory",
    "create_task_delivery_candidate_model_call_authority",
]
