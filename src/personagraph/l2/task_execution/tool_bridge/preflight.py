"""一个模型所提 Tool Bridge 批次的纯 Catalog/策略预检。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ....tools.catalog import CatalogSnapshot
from ....tools.contracts import ToolSpec
from ....tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
    derive_effect_facts,
)
from ....tools.policy import (
    AuthorityFacts,
    BudgetFacts,
    PolicyDisposition,
    PolicyRequest,
    ToolPolicyCore,
)
from ....tools.registration import ToolRegistration
from ....tools.schema_validation import SchemaValidationError, ToolSchemaCompiler
from ...work_run import (
    AttemptDecision,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
)
from .preflight_contracts import (
    ToolBridgeMaterialized,
    ToolBridgePreflightResult,
    ToolBridgeRejected,
    ToolBridgeRejectionCode,
)
from ....tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS


@dataclass(frozen=True)
class _PreparedCall:
    ordinal: int
    registration: ToolRegistration
    arguments: dict[str, Any]
    modifies_environment: bool
    tool_call_id: str


_NON_MODIFYING_ACTIONS = frozenset({EffectAction.READ, EffectAction.SEARCH})


def preflight_and_materialize_call_tools_decision(
    decision: AttemptDecision,
    *,
    catalog_snapshot: CatalogSnapshot,
    stored_catalog_snapshot: Mapping[str, Any],
    allowed_tools: tuple[ToolSpec, ...],
    tool_call_ids: tuple[str, ...],
    authority: AuthorityFacts | None = None,
    budget: BudgetFacts | None = None,
    schemas: ToolSchemaCompiler | None = None,
    policy: ToolPolicyCore | None = None,
    allow_protected_effects: bool = False,
) -> ToolBridgePreflightResult:
    """纯粹守卫一个模型 ``call_tools`` 决定，并由 Host 将其具体化。

    控制器必须在 ``request_model_with_retry`` 输出验证器*内部*调用此可调用对象。类型化拒绝表示
    该物理响应可在同一逻辑模型请求内修复；它不构成在同一 Attempt 中请求第二次逻辑决定的许可。
    本函数不读取 Store、不执行处理器，也没有副作用。
    """

    if not isinstance(decision.action, CallToolsAction):
        return _rejected(
            ToolBridgeRejectionCode.NOT_CALL_TOOLS,
            "Tool preflight accepts only a call_tools AttemptDecision.",
        )
    if dict(stored_catalog_snapshot) != catalog_snapshot.to_descriptor():
        return _rejected(
            ToolBridgeRejectionCode.CATALOG_SNAPSHOT_MISMATCH,
            "The live CatalogSnapshot does not match the descriptor frozen for this Attempt.",
        )
    exposed_tools = tuple(
        entry.registration.spec for entry in catalog_snapshot.exposed()
    )
    if tuple(tool.to_dict() for tool in allowed_tools) != tuple(
        tool.to_dict() for tool in exposed_tools
    ):
        return _rejected(
            ToolBridgeRejectionCode.EXPOSED_TOOL_SET_MISMATCH,
            "The model allowed_tools projection is not the exact frozen exposed catalog.",
        )
    if len(decision.action.calls) != len(tool_call_ids) or any(
        not item.strip() for item in tool_call_ids
    ):
        return _rejected(
            ToolBridgeRejectionCode.ID_PLAN_MISMATCH,
            "Host tool_call_ids must cover every proposed call exactly once.",
        )
    if len(tool_call_ids) != len(set(tool_call_ids)):
        return _rejected(
            ToolBridgeRejectionCode.ID_PLAN_MISMATCH,
            "Host tool_call_ids must be unique within the Attempt.",
        )
    prepared_or_rejection = _prepare_calls(
        decision.action,
        tool_call_ids=tool_call_ids,
        snapshot=catalog_snapshot,
        schemas=schemas or ToolSchemaCompiler(),
        policy=policy or ToolPolicyCore(),
        authority=authority or AuthorityFacts(),
        budget=budget or BudgetFacts(),
        allow_protected_effects=allow_protected_effects,
    )
    if isinstance(prepared_or_rejection, ToolBridgeRejected):
        return prepared_or_rejection
    prepared = prepared_or_rejection
    return ToolBridgeMaterialized(
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=decision.acceptance_updates,
            action=HostMaterializedCallToolsAction(
                calls=tuple(
                    HostMaterializedToolCall(
                        tool_call_id=item.tool_call_id,
                        tool_id=item.registration.tool_id,
        # 契约版本冻结在 Catalog 快照中。单个持久字段标识可执行实现权威。
                        tool_version=item.registration.implementation_version,
                        arguments=item.arguments,
                        modifies_environment=item.modifies_environment,
                    )
                    for item in prepared
                )
            ),
        )
    )


def _prepare_calls(
    action: CallToolsAction,
    *,
    tool_call_ids: tuple[str, ...],
    snapshot: CatalogSnapshot,
    schemas: ToolSchemaCompiler,
    policy: ToolPolicyCore,
    authority: AuthorityFacts,
    budget: BudgetFacts,
    allow_protected_effects: bool = False,
) -> tuple[_PreparedCall, ...] | ToolBridgeRejected:
    exposed_by_id: dict[str, list[ToolRegistration]] = {}
    for entry in snapshot.exposed():
        exposed_by_id.setdefault(entry.key.tool_id, []).append(entry.registration)

    prepared: list[_PreparedCall] = []
    for ordinal, (proposal, tool_call_id) in enumerate(
        zip(action.calls, tool_call_ids, strict=True),
        start=1,
    ):
        registrations = exposed_by_id.get(proposal.tool_id, [])
        if not registrations:
            return _rejected(
                ToolBridgeRejectionCode.TOOL_NOT_EXPOSED,
                "The proposal references a tool_id that was not exposed in the frozen catalog.",
                call_ordinal=ordinal,
                tool_id=proposal.tool_id,
            )
        if len(registrations) != 1:
            return _rejected(
                ToolBridgeRejectionCode.AMBIGUOUS_EXPOSED_TOOL,
                "The exposed tool_id does not resolve to one exact contract version.",
                call_ordinal=ordinal,
                tool_id=proposal.tool_id,
                details={
                    "contract_versions": sorted(
                        registration.contract_version
                        for registration in registrations
                    )
                },
            )
        registration = registrations[0]
        try:
            arguments = schemas.validate_input(
                registration.spec.input_schema,
                proposal.arguments,
            )
        except SchemaValidationError as exc:
            return _rejected(
                ToolBridgeRejectionCode.INVALID_TOOL_INPUT,
                "The proposal arguments do not satisfy the exposed input schema.",
                call_ordinal=ordinal,
                tool_id=proposal.tool_id,
                details={
                    "violations": [item.to_dict() for item in exc.violations]
                },
            )
        effects = derive_effect_facts(registration.effect_profile, arguments)
        internal_runtime_state = _is_execution_findings_runtime_state(
            registration,
            effects=effects,
        )
        modifies_environment = (
            any(effect.action not in _NON_MODIFYING_ACTIONS for effect in effects)
            and not internal_runtime_state
        )
        prepared.append(
            _PreparedCall(
                ordinal=ordinal,
                registration=registration,
                arguments=arguments,
                modifies_environment=modifies_environment,
                tool_call_id=tool_call_id,
            )
        )

    if sum(item.modifies_environment for item in prepared) > 1:
        return _rejected(
            ToolBridgeRejectionCode.TOO_MANY_MODIFYING_CALLS,
            "One Attempt may contain at most one environment-modifying call.",
        )
    if sum(
        _is_execution_findings_runtime_state(item.registration)
        for item in prepared
    ) > 1:
        return _rejected(
            ToolBridgeRejectionCode.TOO_MANY_MODIFYING_CALLS,
            "One Attempt batch may contain at most one execution-findings mutation.",
        )
    modifying = next(
        (item for item in prepared if item.modifies_environment),
        None,
    )
    if modifying is not None and not allow_protected_effects:
        return _rejected(
            ToolBridgeRejectionCode.MODIFYING_CALL_UNSUPPORTED,
            "The first Tool Bridge slice executes read-only calls only; modifying calls require a durable Operation boundary.",
            call_ordinal=modifying.ordinal,
            tool_id=modifying.registration.tool_id,
        )

    for item in prepared:
        policy_decision = policy.evaluate(
            PolicyRequest.from_registration(
                item.registration,
                item.arguments,
                authority=authority,
                budget=budget,
            )
        )
        if policy_decision.disposition is not PolicyDisposition.ALLOW:
            return _rejected(
                ToolBridgeRejectionCode.POLICY_NOT_ALLOWED,
                "The Host policy did not allow this invocation in the current slice.",
                call_ordinal=item.ordinal,
                tool_id=item.registration.tool_id,
                details={
                    "disposition": policy_decision.disposition.value,
                    "reason_codes": list(policy_decision.reason_codes),
                },
            )
    return tuple(prepared)


def _is_execution_findings_runtime_state(
    registration: ToolRegistration,
    *,
    effects: tuple[object, ...] | None = None,
) -> bool:
    """识别精确的 Host 内部状态工具，绝不只依据 action。"""

    if registration.tool_id not in EXECUTION_FINDINGS_TOOL_IDS:
        return False
    resolved = (
        derive_effect_facts(registration.effect_profile, {})
        if effects is None
        else effects
    )
    return bool(resolved) and all(
        getattr(effect, "resource", None) is EffectResource.RUNTIME_STATE
        and getattr(effect, "action", None) is EffectAction.UPDATE
        and getattr(effect, "scope_kind", None) is EffectScopeKind.EXECUTION
        and getattr(effect, "data_egress", None) is DataEgress.NONE
        for effect in resolved
    )


def _rejected(
    code: ToolBridgeRejectionCode,
    message: str,
    *,
    call_ordinal: int | None = None,
    tool_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> ToolBridgeRejected:
    return ToolBridgeRejected(
        code=code,
        message=message,
        call_ordinal=call_ordinal,
        tool_id=tool_id,
        details=details or {},
    )


__all__ = ["preflight_and_materialize_call_tools_decision"]
