"""延迟恢复控制器用于 AuxiliaryGraph Host 原语。

确定性图驱动程序选择一个``host_primitive`` 节点。然后，该控制器请求应用程序实现一个已授权的资源感知请求，冻结并预留该请求以供 I/O 使用，在每次调度时执行匹配的读取适配器，并原子性地密封其结果。重启必须重新生成相同的请求，并恢复预留的逻辑调用，而不是创建另一个身份。持久化合同正好是一个逻辑调用和结算。有限资源读取故意只读，并且在物理过程丢失后可以物理重试；它不被误认为是物理恰好一次 I/O。新鲜预留重放从不授予第二次调度。

请求工厂故意不是一个模型/工具循环。它只接收当前 Store 认证的调度上下文和稳定的标识符。应用程序负责解析冻结文件资源，而无需在工厂内部执行读取操作。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    PlanningEpisodeBudgetDisposition,
    PlanningObservationStatus,
)
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.l2.work_run import AuxiliaryNodeSubject
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    canonical_auxiliary_graph_driver_state_guard,
    decide_auxiliary_graph_driver_step,
)
from .mounted_document_resource_read_port import (
    MountedDocumentPlanningResourceReadPort,
)
from personagraph.l2.planning.invocation_contracts import (
    FrozenPlanningContextArtifactBinding,
    FrozenPlanningContextPrimitiveInvocation,
    PlanningContextPrimitiveKind,
)
from personagraph.l2.planning.resource_perception import (
    PlanningResourcePerceptionRequest,
    PlanningResourcePerceptionResult,
    PlanningResourceReadPort,
    freeze_planning_resource_perception_invocation,
    run_planning_resource_perception,
)


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
AuxiliaryHostPrimitiveRequestValue = PlanningResourcePerceptionRequest
AuxiliaryHostPrimitiveResultValue = PlanningResourcePerceptionResult


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AuxiliaryHostPrimitiveControllerError(RuntimeError):
    """选定的 Host 原语无法保留精确的持久化权威。"""


class AuxiliaryHostPrimitiveConfigurationError(
    AuxiliaryHostPrimitiveControllerError
):
    """应用程序未暴露选定的有限原语。"""


class AuxiliaryHostPrimitiveStateConflict(
    AuxiliaryHostPrimitiveControllerError
):
    """请求、选定节点或预留调用发生了变化。"""


class AuxiliaryHostPrimitiveWaitingExternal(RuntimeError):
    """一个预留的 Host 读取需要外部授权或可恢复条件。"""

    def __init__(self, *, reason_code: str) -> None:
        if re.fullmatch(_ID_PATTERN, reason_code) is None:
            raise ValueError("Host waiting reason must be a bounded stable code")
        super().__init__(reason_code)
        self.reason_code = reason_code


class AuxiliaryHostPrimitiveIdPlan(_Contract):
    primitive_call_id: str = Field(pattern=_ID_PATTERN)
    artifact_id: str = Field(pattern=_ID_PATTERN)
    verification_receipt_id: str = Field(pattern=_ID_PATTERN)
    seal_apply_id: str = Field(pattern=_ID_PATTERN)


class AuxiliaryHostPrimitiveControllerRequest(_Contract):
    session_id: str = Field(pattern=_ID_PATTERN)
    turn_id: str = Field(pattern=_ID_PATTERN)
    subject: AuxiliaryNodeSubject
    initial_driver_state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)


class AuxiliaryHostPrimitiveDispatchContext(_Contract):
    """Store 认证的输入可供有限请求工厂使用。"""

    session_id: str = Field(pattern=_ID_PATTERN)
    turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    subject: AuxiliaryNodeSubject
    local_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    capability_profile_id: str = Field(min_length=1, max_length=200)
    authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    id_plan: AuxiliaryHostPrimitiveIdPlan
    recovering_reserved_call: bool

    def artifact_binding(
        self,
        *,
        scope_snapshot_sha256: str,
        alias_prefix: str,
        artifact_alias: str,
        affected_obligations: tuple[str, ...] = ("task_graph_proposal",),
    ) -> FrozenPlanningContextArtifactBinding:
        """构建此选定节点唯一被接受的绑定。"""

        return FrozenPlanningContextArtifactBinding(
            session_id=self.session_id,
            task_id=self.task_id,
            auxiliary_graph_id=self.auxiliary_graph_id,
            goal_id=self.goal_id,
            producer_auxiliary_node=self.subject,
            primitive_call_id=self.id_plan.primitive_call_id,
            artifact_id=self.id_plan.artifact_id,
            verification_receipt_id=self.id_plan.verification_receipt_id,
            authority_snapshot_id=self.authority_snapshot_id,
            scope_snapshot_sha256=scope_snapshot_sha256,
            alias_prefix=alias_prefix,
            artifact_alias=artifact_alias,
            producer_node_alias=self.local_node_key,
            affected_obligations=affected_obligations,
        )


class AuxiliaryHostPrimitiveRequestFactory(Protocol):
    def __call__(
        self,
        context: AuxiliaryHostPrimitiveDispatchContext,
    ) -> AuxiliaryHostPrimitiveRequestValue: ...


class AuxiliaryHostPrimitiveControllerStatus(StrEnum):
    COMPLETED = "completed"
    WAITING_EXTERNAL = "waiting_external"


class AuxiliaryHostPrimitiveControllerResult(_Contract):
    status: AuxiliaryHostPrimitiveControllerStatus
    reason_code: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    primitive_kind: PlanningContextPrimitiveKind
    primitive_call_id: str = Field(pattern=_ID_PATTERN)
    artifact_id: str = Field(pattern=_ID_PATTERN)
    verification_receipt_id: str = Field(pattern=_ID_PATTERN)
    observation_status: PlanningObservationStatus | None = None
    seal_status: str | None = Field(default=None, pattern=r"^(applied|replayed)$")
    budget_disposition: PlanningEpisodeBudgetDisposition | None = None
    task_state_version: int | None = Field(default=None, ge=1)
    node_state_version: int | None = Field(default=None, ge=1)
    budget_state_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_terminal_shape(self) -> "AuxiliaryHostPrimitiveControllerResult":
        terminal_values = (
            self.observation_status,
            self.seal_status,
            self.budget_disposition,
            self.task_state_version,
            self.node_state_version,
            self.budget_state_version,
        )
        completed = (
            self.status is AuxiliaryHostPrimitiveControllerStatus.COMPLETED
        )
        if completed and not all(value is not None for value in terminal_values):
            raise ValueError("only completed Host primitives carry settlement state")
        if not completed and any(value is not None for value in terminal_values):
            raise ValueError("waiting Host primitives cannot carry settlement state")
        return self


def derive_auxiliary_host_primitive_ids(
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
) -> AuxiliaryHostPrimitiveIdPlan:
    """从一个节点版本推导出有界且重启稳定的标识。"""

    if re.fullmatch(_ID_PATTERN, session_id) is None:
        raise ValueError("session_id must be a bounded durable identity")
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise TypeError("subject must be an AuxiliaryNodeSubject")
    payload = {
        "schema_version": "auxiliary-v2-host-primitive-stable-ids-v1",
        "session_id": session_id,
        "task_id": subject.task_id,
        "auxiliary_graph_id": subject.auxiliary_graph_id,
        "auxiliary_graph_revision": subject.auxiliary_graph_revision,
        "node_id": subject.node_id,
        "node_revision": subject.node_revision,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    namespace = f"auxv2hp-{digest}"
    return AuxiliaryHostPrimitiveIdPlan(
        primitive_call_id=f"{namespace}:call",
        artifact_id=f"{namespace}:artifact",
        verification_receipt_id=f"{namespace}:verification",
        seal_apply_id=f"{namespace}:seal",
    )


def run_auxiliary_host_primitive(
    request: AuxiliaryHostPrimitiveControllerRequest,
    *,
    request_factory: AuxiliaryHostPrimitiveRequestFactory,
    primitive_kinds_by_capability_profile: Mapping[
        str, PlanningContextPrimitiveKind
    ],
    monotonic_clock: Callable[[], float],
    resource_read_port: PlanningResourceReadPort | None = None,
) -> AuxiliaryHostPrimitiveControllerResult:
    """驱动一个新鲜的或预留的 Host 原语通过原子结算。"""

    if not isinstance(request, AuxiliaryHostPrimitiveControllerRequest):
        raise TypeError(
            "request must be AuxiliaryHostPrimitiveControllerRequest"
        )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=request.session_id,
        turn_id=request.turn_id,
        insession_task_id=request.subject.task_id,
    )
    if (
        canonical_auxiliary_graph_driver_state_guard(frontier)
        != request.initial_driver_state_guard_sha256
    ):
        raise AuxiliaryHostPrimitiveStateConflict(
            "AuxiliaryGraph driver state changed before Host dispatch"
        )
    decision = decide_auxiliary_graph_driver_step(frontier)
    recovering = decision.action is AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE
    if decision.action not in {
        AuxiliaryGraphDriverAction.RUN_HOST_PRIMITIVE,
        AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE,
    }:
        raise AuxiliaryHostPrimitiveStateConflict(
            "current driver action is not a Host primitive dispatch"
        )
    if decision.subject != request.subject:
        raise AuxiliaryHostPrimitiveStateConflict(
            "Host primitive request is not the driver-selected subject"
        )

    if recovering:
        candidate = frontier.recoverable_primitive[0]
        if candidate.invocation_turn_id != request.turn_id:
            raise AuxiliaryHostPrimitiveStateConflict(
                "reserved Host primitive belongs to another Turn"
            )
        node_state_version = candidate.node_state_version
        local_node_key = candidate.local_node_key
        capability_profile_id = candidate.capability_profile_id
    else:
        candidate = frontier.ready_fresh[0]
        if candidate.executor_kind is not AuxiliaryNodeExecutorKind.HOST_PRIMITIVE:
            raise AuxiliaryHostPrimitiveStateConflict(
                "driver selected a non-Host execution candidate"
            )
        if candidate.capability_profile_id is None:
            raise AuxiliaryHostPrimitiveConfigurationError(
                "Host primitive node has no capability profile"
            )
        node_state_version = candidate.node_state_version
        local_node_key = candidate.local_node_key
        capability_profile_id = candidate.capability_profile_id

    configured_kind = primitive_kinds_by_capability_profile.get(
        capability_profile_id
    )
    if not isinstance(configured_kind, PlanningContextPrimitiveKind):
        raise AuxiliaryHostPrimitiveConfigurationError(
            "selected capability profile has no exact Host primitive kind"
        )
    if configured_kind is not PlanningContextPrimitiveKind.RESOURCE_PERCEPTION:
        raise AuxiliaryHostPrimitiveConfigurationError(
            "selected capability profile is not a resource-perception primitive"
        )
    if recovering and candidate.primitive_kind != configured_kind.value:
        raise AuxiliaryHostPrimitiveStateConflict(
            "reserved primitive kind differs from its capability profile"
        )
    id_plan = derive_auxiliary_host_primitive_ids(
        session_id=request.session_id,
        subject=request.subject,
    )
    if recovering and decision.primitive_call_id != id_plan.primitive_call_id:
        raise AuxiliaryHostPrimitiveStateConflict(
            "reserved primitive identity differs from the stable node identity"
        )
    context = AuxiliaryHostPrimitiveDispatchContext(
        session_id=request.session_id,
        turn_id=request.turn_id,
        task_id=request.subject.task_id,
        auxiliary_graph_id=frontier.auxiliary_graph_id,
        goal_id=frontier.goal_id,
        auxiliary_graph_revision=frontier.auxiliary_graph_revision,
        subject=request.subject,
        local_node_key=local_node_key,
        capability_profile_id=capability_profile_id,
        authority_snapshot_id=frontier.authority_snapshot_id,
        authority_snapshot_sha256=frontier.authority_snapshot_sha256,
        structure_sha256=frontier.structure_sha256,
        budget_snapshot_sha256=frontier.budget_snapshot_sha256,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        expected_budget_state_version=frontier.budget_state_version,
        id_plan=id_plan,
        recovering_reserved_call=recovering,
    )
    primitive_request = request_factory(context)
    primitive_kind = _primitive_kind(primitive_request)
    if primitive_kind is not configured_kind:
        raise AuxiliaryHostPrimitiveConfigurationError(
            "request factory returned a primitive outside the capability profile"
        )
    _require_exact_binding(primitive_request.binding, context=context)
    effective_resource_read_port = resource_read_port
    if effective_resource_read_port is None:
        effective_resource_read_port = MountedDocumentPlanningResourceReadPort()
    invocation = _freeze(primitive_request, context=context)

    if recovering:
        stored = planning_store.get_planning_primitive_invocation(
            session_id=request.session_id,
            primitive_call_id=id_plan.primitive_call_id,
        )
        if (
            stored is None
            or stored.status != "reserved"
            or stored.invocation != invocation
        ):
            raise AuxiliaryHostPrimitiveStateConflict(
                "request factory did not reproduce the reserved invocation"
            )
    else:
        reservation = planning_store.reserve_planning_primitive_invocation(
            invocation=invocation
        )
        if reservation.status != "applied":
            return _waiting_external_result(
                request=request,
                primitive_kind=primitive_kind,
                id_plan=id_plan,
                reason_code="host_primitive_dispatch_already_reserved",
            )
    planning_store.require_planning_primitive_invocation_current(
        invocation=invocation
    )

    started_at = float(monotonic_clock())
    try:
        result = _execute(
            primitive_request,
            resource_read_port=effective_resource_read_port,
        )
    except AuxiliaryHostPrimitiveWaitingExternal as exc:
        return _waiting_external_result(
            request=request,
            primitive_kind=primitive_kind,
            id_plan=id_plan,
            reason_code=exc.reason_code,
        )
    active_seconds_delta = max(0.0, float(monotonic_clock()) - started_at)
    sealed = planning_store.seal_auxiliary_host_primitive_result(
        command=planning_store.SealAuxiliaryHostPrimitiveResultCommand(
            apply_id=id_plan.seal_apply_id,
            session_id=request.session_id,
            invocation_turn_id=request.turn_id,
            task_id=request.subject.task_id,
            auxiliary_graph_id=frontier.auxiliary_graph_id,
            goal_id=frontier.goal_id,
            auxiliary_graph_revision=frontier.auxiliary_graph_revision,
            auxiliary_node_id=request.subject.node_id,
            node_revision=request.subject.node_revision,
            primitive_kind=primitive_kind.value,
            primitive_call_id=id_plan.primitive_call_id,
            expected_artifact_id=id_plan.artifact_id,
            expected_verification_receipt_id=id_plan.verification_receipt_id,
            expected_scope_snapshot_sha256=(
                primitive_request.binding.scope_snapshot_sha256
            ),
            expected_base_task_graph_revision=frontier.base_task_graph_revision,
            expected_task_state_version=invocation.expected_task_state_version,
            expected_node_state_version=invocation.expected_node_state_version,
            expected_control_state_version=(
                invocation.expected_control_state_version
            ),
            expected_goal_state_version=invocation.expected_goal_state_version,
            expected_revision_state_version=(
                invocation.expected_revision_state_version
            ),
            expected_budget_state_version=(
                invocation.expected_budget_state_version
            ),
            expected_authority_snapshot_id=frontier.authority_snapshot_id,
            expected_authority_snapshot_sha256=(
                invocation.authority_snapshot_sha256
            ),
            expected_structure_sha256=invocation.structure_sha256,
            expected_budget_snapshot_sha256=invocation.budget_snapshot_sha256,
            logical_request_json=invocation.logical_request_json,
            logical_request_sha256=invocation.logical_request_sha256,
            state_guard_sha256=invocation.state_guard_sha256,
            active_seconds_delta=active_seconds_delta,
        ),
        result=result,
    )
    return AuxiliaryHostPrimitiveControllerResult(
        status=AuxiliaryHostPrimitiveControllerStatus.COMPLETED,
        reason_code="host_primitive_atomically_settled",
        subject=request.subject,
        primitive_kind=primitive_kind,
        primitive_call_id=id_plan.primitive_call_id,
        artifact_id=id_plan.artifact_id,
        verification_receipt_id=id_plan.verification_receipt_id,
        observation_status=result.observation_status,
        seal_status=sealed.status,
        budget_disposition=sealed.budget_disposition,
        task_state_version=sealed.task_state_version,
        node_state_version=sealed.node_state_version,
        budget_state_version=sealed.budget_state_version,
    )


def _primitive_kind(
    request: AuxiliaryHostPrimitiveRequestValue,
) -> PlanningContextPrimitiveKind:
    if isinstance(request, PlanningResourcePerceptionRequest):
        return PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    raise TypeError("request factory returned an unsupported Host primitive")


def _require_exact_binding(
    binding: FrozenPlanningContextArtifactBinding,
    *,
    context: AuxiliaryHostPrimitiveDispatchContext,
) -> None:
    expected = context.id_plan
    if (
        binding.session_id != context.session_id
        or binding.task_id != context.task_id
        or binding.auxiliary_graph_id != context.auxiliary_graph_id
        or binding.goal_id != context.goal_id
        or binding.producer_auxiliary_node != context.subject
        or binding.primitive_call_id != expected.primitive_call_id
        or binding.artifact_id != expected.artifact_id
        or binding.verification_receipt_id != expected.verification_receipt_id
        or binding.authority_snapshot_id != context.authority_snapshot_id
        or binding.producer_node_alias != context.local_node_key
    ):
        raise AuxiliaryHostPrimitiveStateConflict(
            "Host primitive binding differs from the selected durable node"
        )


def _freeze(
    request: AuxiliaryHostPrimitiveRequestValue,
    *,
    context: AuxiliaryHostPrimitiveDispatchContext,
) -> FrozenPlanningContextPrimitiveInvocation:
    kwargs = {
        "invocation_turn_id": context.turn_id,
        "expected_task_state_version": context.expected_task_state_version,
        "expected_node_state_version": context.expected_node_state_version,
        "expected_control_state_version": context.expected_control_state_version,
        "expected_goal_state_version": context.expected_goal_state_version,
        "expected_revision_state_version": context.expected_revision_state_version,
        "expected_budget_state_version": context.expected_budget_state_version,
        "authority_snapshot_sha256": context.authority_snapshot_sha256,
        "structure_sha256": context.structure_sha256,
        "budget_snapshot_sha256": context.budget_snapshot_sha256,
    }
    return freeze_planning_resource_perception_invocation(request, **kwargs)


def _execute(
    request: AuxiliaryHostPrimitiveRequestValue,
    *,
    resource_read_port: PlanningResourceReadPort,
) -> AuxiliaryHostPrimitiveResultValue:
    return run_planning_resource_perception(
        request,
        read_port=resource_read_port,
    )


def _waiting_external_result(
    *,
    request: AuxiliaryHostPrimitiveControllerRequest,
    primitive_kind: PlanningContextPrimitiveKind,
    id_plan: AuxiliaryHostPrimitiveIdPlan,
    reason_code: str,
) -> AuxiliaryHostPrimitiveControllerResult:
    return AuxiliaryHostPrimitiveControllerResult(
        status=AuxiliaryHostPrimitiveControllerStatus.WAITING_EXTERNAL,
        reason_code=reason_code,
        subject=request.subject,
        primitive_kind=primitive_kind,
        primitive_call_id=id_plan.primitive_call_id,
        artifact_id=id_plan.artifact_id,
        verification_receipt_id=id_plan.verification_receipt_id,
    )


__all__ = [
    "AuxiliaryHostPrimitiveConfigurationError",
    "AuxiliaryHostPrimitiveControllerError",
    "AuxiliaryHostPrimitiveControllerRequest",
    "AuxiliaryHostPrimitiveControllerResult",
    "AuxiliaryHostPrimitiveDispatchContext",
    "AuxiliaryHostPrimitiveRequestFactory",
    "AuxiliaryHostPrimitiveIdPlan",
    "AuxiliaryHostPrimitiveStateConflict",
    "AuxiliaryHostPrimitiveWaitingExternal",
    "derive_auxiliary_host_primitive_ids",
    "run_auxiliary_host_primitive",
]
