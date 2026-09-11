"""一个有界 L1 TurnRun 的 Tool Platform 组合。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from types import MappingProxyType
from typing import Any, Mapping

from ...tools.catalog import CatalogError, ToolResolutionError
from ...tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
)
from ...tools.catalog.persistence import (
    CatalogPersistenceError,
    EmergencyRevocationTarget,
    ToolCatalogRepository,
)
from ...tools.catalog.revocation import (
    EmergencyRevocationGuard,
    EmergencyRevocationGuardStage,
)
from ...tools.contracts import ExecutionOutcome, ToolError
from ...tools.effects import ToolEffectProfile
from ...tools.execution import ResolvedInvocation, ToolBusinessFailure, ToolExecutor
from ...tools.findings.execution_findings_tools import (
    build_execution_findings_tool_registrations,
)
from ...tools.policy import (
    AuthorityFacts,
    BudgetFacts,
    PolicyRequest,
    ScopeGrant,
    ToolPolicyCore,
    ToolInvocationAuthorityResolver,
)
from ...tools.schema_validation import SchemaValidationError, ToolSchemaCompiler
from .attachment_contracts import L1TurnAttachmentToolRuntimeError
from ...tools.policy import ProtectedToolExecutionAuthority
from .turn_file_sources import (
    attachment_file_catalog,
    freeze_turn_attachment_sources,
)
from personagraph.workspace.files.turn_inputs import (
    ResolvedTurnInputFile,
    TurnInputFileAuthorityError,
)
from .identity import canonical_json, sha256_json
from ...tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from .tool_catalog_snapshot import (
    L1CatalogSnapshotRestoreError,
    _l1_tool_catalog_descriptor,
    _model_catalog_items_from_definitions,
    _reconcile_persisted_l1_catalog_snapshot,
)


@dataclass(frozen=True, slots=True)
class L1ToolExecution:
    tool_id: str
    contract_version: str
    implementation_version: str
    normalized_arguments: dict[str, Any]
    policy: dict[str, Any]
    outcome: ExecutionOutcome


@dataclass(frozen=True, slots=True)
class L1PreparedToolCall:
    definition: ToolDefinition
    registration: BoundToolRegistration | None
    normalized_arguments: dict[str, Any]
    policy: dict[str, Any]
    rejected_outcome: ExecutionOutcome | None = None
    protected_authority: ProtectedToolExecutionAuthority | None = None

    @property
    def requires_protected_dispatch(self) -> bool:
        """此调用是否必须跨越 L1 的持久化 effect 边界。"""

        return self.protected_authority is not None

    @property
    def tool_id(self) -> str:
        return self.definition.identity.tool_id

    @property
    def contract_version(self) -> str:
        return self.definition.identity.contract_version

    @property
    def implementation_version(self) -> str:
        return self.definition.identity.implementation_version

    @property
    def effect_profile(self) -> ToolEffectProfile:
        registration = self.registration
        return (
            registration.effect_profile
            if registration is not None
            else self.definition.effect_template
        )


class L1WorkspaceUnavailableError(RuntimeError):
    """The L1 lane cannot prove its mandatory fixed workspace authority."""


class L1ToolCatalogUnavailableError(RuntimeError):
    """The persistent L1 tool table cannot be materialized safely."""


@dataclass(frozen=True, slots=True)
class L1WorkspaceCorpusAuthority:
    """L1 workspace 工具背后的私有不可变来源清单。"""

    boundary_fingerprint: str
    scope_snapshot_sha256: str
    local_file_authorization_receipt_id: str


@dataclass(frozen=True, slots=True)
class L1ToolRuntime:
    """L1 的稳定模型工具面与本 Turn 可执行绑定。"""

    definitions: tuple[ToolDefinition, ...]
    registrations_by_tool_id: Mapping[str, BoundToolRegistration]
    unavailable_reason_by_tool_id: Mapping[str, str]
    disabled_tool_ids: frozenset[str]
    model_catalog_json: str
    revocation_guard: EmergencyRevocationGuard
    authority: AuthorityFacts
    catalog_snapshot_json: str
    catalog_snapshot_sha256: str
    workspace_corpus_authority: L1WorkspaceCorpusAuthority | None = None
    turn_attachment_sources: tuple[ResolvedTurnInputFile, ...] = ()
    attachment_file_catalog: tuple[dict[str, object], ...] = ()
    protected_authority_by_key: Mapping[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = MappingProxyType({})
    invocation_authority_resolver_by_key: Mapping[
        tuple[str, str], ToolInvocationAuthorityResolver
    ] = MappingProxyType({})

    def model_catalog(self) -> tuple[dict[str, object], ...]:
        """从冻结 JSON 返回模型可见的 Tool Catalog 副本，不重新发现或绑定工具。

        仅投影本轮绑定且启用的定义；全局持久总表与执行时权限校验不变。
        """

        decoded = json.loads(self.model_catalog_json)
        if not isinstance(decoded, list) or any(
            not isinstance(item, dict) for item in decoded
        ):
            raise RuntimeError("frozen L1 model catalog is invalid")
        return tuple(item for item in decoded
                     if item.get("tool_id") in self.registrations_by_tool_id
                     and item.get("tool_id") not in self.disabled_tool_ids)

    def knows_tool(self, tool_id: str) -> bool:
        """判断工具 ID 是否属于冻结定义集；True 不代表有 binding 或已获执行许可。"""

        return any(
            definition.identity.tool_id == tool_id
            for definition in self.definitions
        )

    @staticmethod
    def is_execution_findings_tool(tool_id: str) -> bool:
        return tool_id in EXECUTION_FINDINGS_TOOL_IDS

    def prepare(
        self,
        *,
        tool_id: str,
        arguments: Mapping[str, Any],
        remaining_tool_calls: int,
    ) -> L1PreparedToolCall:
        """为一次 ToolCall 做执行前准入，返回参数、policy 与可选拒绝结果。

        按 schema → Host 开关 → binding → 撤权 → effect/budget policy 检查。
        不调用 handler，也不预留持久 ToolCall；controller 随后记录准入事实。
        即使获准，真正 dispatch 前仍需复查撤权，避免 prepare 到 I/O 之间的权限变化。
        """

        definition = self._definition(tool_id)
        registration = self.registrations_by_tool_id.get(tool_id)
        schemas = ToolSchemaCompiler()
        try:
            normalized = schemas.validate_input(
                definition.spec.input_schema,
                arguments,
            )
        except SchemaValidationError as exc:
            outcome = ExecutionOutcome.rejected(
                exc.to_tool_error(code="invalid_tool_input"),
                metadata={
                    "tool_id": tool_id,
                    "contract_version": definition.identity.contract_version,
                },
            )
            return L1PreparedToolCall(
                definition=definition,
                registration=registration,
                normalized_arguments=dict(arguments),
                policy={"disposition": "deny", "reason_codes": ["invalid_tool_input"]},
                rejected_outcome=outcome,
            )

        if tool_id in self.disabled_tool_ids:
            return L1PreparedToolCall(
                definition=definition,
                registration=registration,
                normalized_arguments=normalized,
                policy={
                    "disposition": "deny",
                    "reason_codes": ["tool_disabled_by_host"],
                },
                rejected_outcome=ExecutionOutcome.rejected(
                    ToolError(
                        code="tool_disabled_by_host",
                        message="The Host disabled this tool for the current run.",
                    ),
                    metadata={
                        "tool_id": tool_id,
                        "contract_version": definition.identity.contract_version,
                    },
                ),
            )

        unavailable_reason = self.unavailable_reason_by_tool_id.get(tool_id)
        if registration is None:
            reason = unavailable_reason or "runtime_binding_unavailable"
            return L1PreparedToolCall(
                definition=definition,
                registration=None,
                normalized_arguments=normalized,
                policy={
                    "disposition": "deny",
                    "reason_codes": ["tool_unavailable", reason],
                },
                rejected_outcome=ExecutionOutcome.rejected(
                    ToolError(
                        code="tool_unavailable",
                        message=(
                            "The tool is known, but the Host cannot resolve an "
                            "executable binding in this Turn."
                        ),
                        details={"reason": reason},
                    ),
                    metadata={
                        "tool_id": tool_id,
                        "contract_version": definition.identity.contract_version,
                    },
                ),
            )

        revocation = self._check_revocation(
            registration,
            stage=EmergencyRevocationGuardStage.RESOLVE,
        )
        if revocation is not None:
            return L1PreparedToolCall(
                definition=definition,
                registration=registration,
                normalized_arguments=normalized,
                policy={
                    "disposition": "deny",
                    "reason_codes": [revocation.error.code],
                },
                rejected_outcome=revocation,
            )

        key = (registration.tool_id, registration.contract_version)
        authority = self.authority
        protected_authority = self.protected_authority_by_key.get(key)
        resolve_authority = self.invocation_authority_resolver_by_key.get(key)
        if resolve_authority is not None:
            try:
                invocation_authority = resolve_authority(normalized)
            except ToolBusinessFailure as exc:
                return L1PreparedToolCall(
                    definition=definition,
                    registration=registration,
                    normalized_arguments=normalized,
                    policy={"disposition": "deny", "reason_codes": [exc.error.code]},
                    rejected_outcome=ExecutionOutcome.rejected(exc.error),
                )
            authority = _merge_authority(authority, invocation_authority.authority)
            protected_authority = invocation_authority.protected_authority

        policy = ToolPolicyCore().evaluate(
            PolicyRequest.from_registration(
                registration,
                normalized,
                authority=authority,
                budget=BudgetFacts(remaining_tool_calls=remaining_tool_calls),
            )
        )
        if not policy.allowed:
            outcome = ExecutionOutcome.rejected(
                ToolError(
                    code=f"tool_policy_{policy.disposition.value}",
                    message="Tool invocation was rejected by the Host policy.",
                    details={"reason_codes": list(policy.reason_codes)},
                ),
                metadata={
                    "tool_id": registration.tool_id,
                    "contract_version": registration.contract_version,
                },
            )
        else:
            outcome = None
        if outcome is not None:
            protected_authority = None
        return L1PreparedToolCall(
            definition=definition,
            registration=registration,
            normalized_arguments=normalized,
            policy=policy.to_dict(),
            rejected_outcome=outcome,
            protected_authority=protected_authority,
        )

    def execute_prepared(
        self,
        prepared: L1PreparedToolCall,
        *,
        deadline_monotonic: float,
        logical_tool_call_id: str | None = None,
    ) -> L1ToolExecution:
        """执行已准备的普通工具调用，并保留明确的拒绝 outcome。

        protected effect 必须交给 L1 durable dispatcher；本路径再次检查撤权，之后才
        进入 ToolExecutor。提前返回的拒绝不会经过 ToolExecutor 的 trajectory 包装，
        但 controller 仍须把这个 outcome 结算到持久 ToolCall。
        """

        outcome = prepared.rejected_outcome
        if outcome is None and prepared.requires_protected_dispatch:
            outcome = ExecutionOutcome.rejected(
                ToolError(
                    code="protected_tool_dispatch_required",
                    message=(
                        "This tool requires the L1 durable protected-operation "
                        "dispatcher."
                    ),
                ),
                metadata={
                    "tool_id": prepared.tool_id,
                    "contract_version": prepared.contract_version,
                },
            )
        if outcome is None:
            registration = prepared.registration
            if registration is None:
                raise RuntimeError("prepared executable tool omitted its binding")
            outcome = self._check_revocation(
                registration,
                stage=EmergencyRevocationGuardStage.EXECUTE,
            )
        if outcome is None:
            registration = prepared.registration
            assert registration is not None
            outcome = ToolExecutor().execute(
                ResolvedInvocation(
                    registration=registration,
                    arguments=prepared.normalized_arguments,
                    deadline_monotonic=deadline_monotonic,
                    logical_tool_call_id=logical_tool_call_id,
                )
            )
        return L1ToolExecution(
            tool_id=prepared.tool_id,
            contract_version=prepared.contract_version,
            implementation_version=prepared.implementation_version,
            normalized_arguments=prepared.normalized_arguments,
            policy=prepared.policy,
            outcome=outcome,
        )

    def authorize_external_dispatch(
        self,
        prepared: L1PreparedToolCall,
    ) -> ExecutionOutcome | None:
        """在 findings / protected dispatcher 交接前复查紧急撤权（dispatch guard）。

        返回 None 表示该检查未拒绝，不替代此前的 policy、租约或专用 dispatcher 权威校验。
        """

        registration = prepared.registration
        if registration is None:
            return prepared.rejected_outcome
        return self._check_revocation(
            registration,
            stage=EmergencyRevocationGuardStage.EXECUTE,
        )

    def _definition(self, tool_id: str) -> ToolDefinition:
        for definition in self.definitions:
            if definition.identity.tool_id == tool_id:
                return definition
        raise ToolResolutionError(f"unknown L1 tool: {tool_id!r}")

    def _check_revocation(
        self,
        registration: BoundToolRegistration,
        *,
        stage: EmergencyRevocationGuardStage,
    ) -> ExecutionOutcome | None:
        try:
            target = EmergencyRevocationTarget.from_bound_registration(registration)
        except (TypeError, ValueError):
            decision = None
        else:
            decision = self.revocation_guard.check_current(
                stage=stage,
                target=target,
            )
        if decision is not None and decision.allowed:
            return None
        code = (
            decision.code
            if decision is not None
            else "emergency_revocation_target_invalid"
        )
        details = (
            decision.to_dict()
            if decision is not None
            else {"stage": stage.value}
        )
        return ExecutionOutcome.rejected(
            ToolError(
                code=code,
                message="The Host denied this tool transition.",
                details=details,
            ),
            metadata={
                "tool_id": registration.tool_id,
                "contract_version": registration.contract_version,
            },
        )


def build_l1_tool_runtime(
    session_id: str,
    *,
    turn_id: str | None = None,
    execution_findings_enabled: bool = True,
    execution_features: Mapping[str, Any] | None = None,
    file_retrieval_data_version: str | None = None,
    session_retrieval_data_version: str | None = None,
    session_retrieval_assistant_turn_cutoff: int | None = None,
    expected_catalog_snapshot_json: str | None = None,
    expected_catalog_snapshot_sha256: str | None = None,
) -> L1ToolRuntime:
    """冻结 Tool Catalog，并组合当前 Turn 的可执行资源 binding。

    工作区、附件、findings 与受 feature gate 控制的文件/历史检索分别提供 binding，
    再由持久默认 Definition 表统一物化。模型目录保留完整定义；禁用、不可用与
    protected authority 单独保存，不能把“模型可见”当作“可以执行”。
    恢复时按 expected catalog snapshot 对账；本函数可能读取/初始化目录和资源状态，
    不是纯 prompt builder，但不会执行模型提出的 ToolCall。
    """

    if not isinstance(execution_findings_enabled, bool):
        raise TypeError("execution_findings_enabled must be boolean")
    if (expected_catalog_snapshot_json is None) is not (
        expected_catalog_snapshot_sha256 is None
    ):
        raise ValueError(
            "expected L1 catalog snapshot JSON and SHA-256 must be supplied together"
        )

    from ...tools.web.web_tools import WEB_FETCH_TOOL_ID, WEB_SEARCH_TOOL_ID
    from ...tools.workspace.workspace_write_adapter import (
        build_session_workspace_action_tool_source,
    )
    from ...tools.workspace.session_read_source import (
        build_session_workspace_readonly_runtime,
    )
    from ...session import store as session_store
    from ...workspace.discovery import (
        RipgrepFailed,
        RipgrepTimeout,
        RipgrepUnavailable,
    )

    features = dict(execution_features or {})
    external_web_tools_enabled = features.get(
        "l1_external_web_tools_enabled",
        True,
    )
    if not isinstance(external_web_tools_enabled, bool):
        raise TypeError("l1_external_web_tools_enabled must be boolean")
    file_tools_enabled = bool(
        features.get("l1_retrieval_tools_enabled", False) is True
        and features.get("file_retrieval_read_enabled", False) is True
    )
    external_web_tool_ids = frozenset({WEB_SEARCH_TOOL_ID, WEB_FETCH_TOOL_ID})
    disabled_tool_ids: set[str] = set()
    if not external_web_tools_enabled:
        disabled_tool_ids.update(external_web_tool_ids)
    contextual_bindings: list[ToolBinding] = []
    if execution_findings_enabled and turn_id is not None:
        from ...tools.findings import (
            ExecutionFindingsBindingFacts,
            build_execution_findings_tool_bindings,
            derive_execution_findings_source_fingerprint,
        )
        from .corpus_contracts import derive_l1_turn_run_id

        l1_turn_run_id = derive_l1_turn_run_id(
            session_id=session_id,
            turn_id=turn_id,
        )
        findings_effect_scope = f"l1_turn_run:{l1_turn_run_id}"
        findings_facts = ExecutionFindingsBindingFacts(
            effect_scope_sha256=_sha256_text(findings_effect_scope),
            ledger_identity_sha256=sha256_json(
                {
                    "schema_version": "l1-findings-ledger-route-v1",
                    "l1_turn_run_id": l1_turn_run_id,
                }
            ),
            scope_key_authority_sha256=sha256_json(
                {
                    "schema_version": "l1-findings-scope-authority-v1",
                    "owner_kind": "l1_turn_run",
                    "scope_key_sources": [
                        "current_plan_acceptance",
                        "current_tool_result",
                        "current_document_chunk",
                    ],
                }
            ),
            dispatcher_identity_sha256=sha256_json(
                {
                    "schema_version": "l1-findings-dispatcher-v1",
                    "callable": (
                        "personagraph.tools.findings.dispatcher."
                        "execute_execution_findings_tool"
                    ),
                }
            ),
            mutation_store_identity_sha256=sha256_json(
                {
                    "schema_version": "l1-findings-store-port-v1",
                    "port": "personagraph.runtime.l1.ports.L1StorePort",
                }
            ),
        )
        findings_registrations = build_execution_findings_tool_registrations(
            effect_scope=findings_effect_scope,
            source_fingerprint=(
                derive_execution_findings_source_fingerprint(findings_facts)
            ),
        )
        contextual_bindings.extend(
            build_execution_findings_tool_bindings(
                findings_registrations,
                facts=findings_facts,
            )
        )
    elif not execution_findings_enabled:
        disabled_tool_ids.update(EXECUTION_FINDINGS_TOOL_IDS)
    authority = AuthorityFacts()
    if turn_id is not None:
        from .tool_context.history_binding import bind_l1_tool_history

        history_runtime = bind_l1_tool_history(session_id=session_id, turn_id=turn_id)
        contextual_bindings.extend(history_runtime.bindings)
        authority = _merge_authority(authority, history_runtime.authority)
    workspace_corpus_authority = None
    attachment_sources = ()
    protected_authority_by_key: dict[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = {}
    invocation_authority_resolver_by_key: dict[
        tuple[str, str], ToolInvocationAuthorityResolver
    ] = {}
    if turn_id is not None:
        try:
            attachment_sources = freeze_turn_attachment_sources(
                session_id=session_id, turn_id=turn_id,
            )
        except (OSError, TurnInputFileAuthorityError, TypeError, ValueError) as exc:
            raise L1TurnAttachmentToolRuntimeError() from exc
    try:
        workspace = build_session_workspace_readonly_runtime(session_id)
    except (
        OSError,
        sqlite3.Error,
        session_store.SessionCatalogError,
        session_store.SessionStoreError,
        RipgrepFailed,
        RipgrepTimeout,
        RipgrepUnavailable,
    ) as exc:
        # 这里仅归一化由持久化、固定根目录或目录发现后端报告的可预期
        # 运行时失败。TypeError、AssertionError 等契约/编程错误必须继续
        # 冒泡，避免把代码缺陷伪装成“用户工作区暂不可用”。
        raise L1WorkspaceUnavailableError(
            "L1 fixed workspace could not be materialized"
        ) from exc
    except ValueError as exc:
        # workspace source 目前用这个精确值表达“Session 已不存在”。其余
        # ValueError 包括跨边界等内部不变量破坏，不能归入普通工作区故障。
        if str(exc) != "session does not exist":
            raise
        raise L1WorkspaceUnavailableError(
            "L1 fixed workspace could not be materialized"
        ) from exc
    if workspace is None:
        raise L1WorkspaceUnavailableError(
            "L1 requires the Session fixed workspace to be readable"
        )
    if workspace is not None:
        authority = workspace.authority
        workspace_corpus_authority = L1WorkspaceCorpusAuthority(
            boundary_fingerprint=workspace.boundary_fingerprint,
            scope_snapshot_sha256=workspace.scope_snapshot_sha256,
            local_file_authorization_receipt_id=(
                workspace.local_file_authorization_receipt_id
            ),
        )
        # Workspace source 已完成边界与授权冻结；L1 只把
        # contextual Binding 交给持久默认 Definition 表统一解析。
        contextual_bindings.extend(workspace.workspace_discovery_bindings)
        contextual_bindings.extend(workspace.format_observation_bindings)
        contextual_bindings.extend(
            workspace.external_visual_analysis_bindings
        )
        protected_authority_by_key.update(
            workspace.protected_authority_by_key
        )
        invocation_authority_resolver_by_key.update(
            workspace.invocation_authority_resolver_by_key
        )
        from .protected_tool_dispatch import (
            l1_protected_dispatch_identity_sha256,
            l1_protected_operation_ledger_identity_sha256,
        )

        action_source = build_session_workspace_action_tool_source(
            session_id=session_id,
            boundary=workspace.boundary,
            boundary_sha256=workspace.boundary_fingerprint,
            protected_dispatch_sha256=(
                l1_protected_dispatch_identity_sha256()
            ),
            operation_ledger_sha256=(
                l1_protected_operation_ledger_identity_sha256()
            ),
        )
        contextual_bindings.extend(action_source.workspace_write_bindings)
        authority = _merge_authority(authority, action_source.authority)
        protected_authority_by_key.update(
            action_source.protected_authority_by_key
        )
        from ...tools.workspace.output_file_tool import build_session_output_file_tool_source

        output_source = build_session_output_file_tool_source(
            session_id=session_id, boundary=workspace.boundary,
            workspace_grant_id=workspace.local_file_authorization_receipt_id,
            protected_dispatch_sha256=l1_protected_dispatch_identity_sha256(),
            operation_ledger_sha256=l1_protected_operation_ledger_identity_sha256(),
        )
        if output_source is not None:
            contextual_bindings.append(output_source.binding)
            authority = _merge_authority(authority, output_source.authority)
            output_identity = output_source.binding.identity
            protected_authority_by_key[
                (output_identity.tool_id, output_identity.contract_version)
            ] = output_source.protected_authority

    if file_tools_enabled:
        from ...tools.composition.file_tools import build_file_tool_source

        file_source = build_file_tool_source(
            session_id=session_id,
            turn_id=turn_id,
            workspace_boundary=workspace.boundary,
            workspace_grant_id=workspace.local_file_authorization_receipt_id,
            file_retrieval_data_version=file_retrieval_data_version,
            attachment_sources=attachment_sources,
        )
        contextual_bindings.extend(file_source.bindings)
        authority = _merge_authority(authority, file_source.authority)
        protected_authority_by_key.update(file_source.protected_authority_by_key)
        invocation_authority_resolver_by_key.update(
            file_source.invocation_authority_resolver_by_key
        )

    history_retrieval_enabled = bool(
        features.get("l1_retrieval_tools_enabled", False) is True
        and features.get("history_retrieval_read_enabled", False) is True
    )
    if history_retrieval_enabled:
        from .history_retrieval_composition import (
            build_l1_history_retrieval_tool_source,
        )

        history_source = build_l1_history_retrieval_tool_source(
            session_id=session_id,
            turn_id=turn_id,
            session_retrieval_data_version=session_retrieval_data_version,
            session_retrieval_assistant_turn_cutoff=(
                session_retrieval_assistant_turn_cutoff
            ),
        )
        if history_source is not None:
            contextual_bindings.extend(
                history_source.history_retrieval_bindings
            )
            authority = _merge_authority(authority, history_source.authority)

    from ...tools.catalog.default_profile import (
        materialize_contextual_default_catalog,
    )
    from ...tools.catalog.trusted_factories import (
        TrustedCatalogResolutionError,
    )
    from ...tools.composition.default_catalog import (
        ProductionDefaultProfileResolutionError,
        bootstrap_production_default_catalog,
        build_production_default_factory_registry,
        resolve_production_default_profile,
    )

    repository = ToolCatalogRepository()
    try:
        bootstrap_production_default_catalog(repository)
        profile = resolve_production_default_profile(repository)
        materialized = materialize_contextual_default_catalog(
            repository,
            build_production_default_factory_registry(),
            contextual_bindings=tuple(contextual_bindings),
        )
    except (
        CatalogError,
        CatalogPersistenceError,
        ProductionDefaultProfileResolutionError,
        TrustedCatalogResolutionError,
    ) as exc:
        raise L1ToolCatalogUnavailableError(
            "the persistent L1 tool table is unavailable or inconsistent"
        ) from exc
    if (
        materialized.profile_revision != profile.profile_revision
        or materialized.profile_digest != profile.profile_digest
        or materialized.catalog_revision != profile.catalog_revision
        or materialized.catalog_digest != profile.catalog_digest
        or materialized.profile_catalog_revision
        != profile.profile_catalog_revision
    ):
        raise L1ToolCatalogUnavailableError(
            "L1 Tool Catalog changed while its Definition profile was bound"
        )

    definitions = profile.definitions
    registrations_by_tool_id = _index_l1_registrations(
        materialized.registrations,
        definitions=definitions,
    )
    unavailable_reason_by_tool_id = {
        item.identity.tool_id: item.reason
        for item in materialized.unavailable
    }
    descriptor = _l1_tool_catalog_descriptor(
        profile=profile,
        registrations_by_tool_id=registrations_by_tool_id,
        unavailable_reason_by_tool_id=unavailable_reason_by_tool_id,
        disabled_tool_ids=frozenset(disabled_tool_ids),
    )
    snapshot_json = canonical_json(descriptor)
    snapshot_sha256 = sha256_json(descriptor)
    model_catalog_json = canonical_json(
        _model_catalog_items_from_definitions(definitions)
    )
    if expected_catalog_snapshot_json is not None:
        assert expected_catalog_snapshot_sha256 is not None
        reconciled_snapshot = _reconcile_persisted_l1_catalog_snapshot(
            expected_json=expected_catalog_snapshot_json,
            expected_sha256=expected_catalog_snapshot_sha256,
            current_json=snapshot_json,
            current_sha256=snapshot_sha256,
            definitions=definitions,
            registrations_by_tool_id=registrations_by_tool_id,
            unavailable_reason_by_tool_id=unavailable_reason_by_tool_id,
            disabled_tool_ids=frozenset(disabled_tool_ids),
            current_model_catalog_json=model_catalog_json,
        )
        definitions = reconciled_snapshot.definitions
        registrations_by_tool_id = (
            reconciled_snapshot.registrations_by_tool_id
        )
        unavailable_reason_by_tool_id = (
            reconciled_snapshot.unavailable_reason_by_tool_id
        )
        disabled_tool_ids = set(reconciled_snapshot.disabled_tool_ids)
        snapshot_json = reconciled_snapshot.snapshot_json
        snapshot_sha256 = reconciled_snapshot.snapshot_sha256
        model_catalog_json = reconciled_snapshot.model_catalog_json
    return L1ToolRuntime(
        definitions=definitions,
        registrations_by_tool_id=MappingProxyType(
            dict(registrations_by_tool_id)
        ),
        unavailable_reason_by_tool_id=MappingProxyType(
            dict(unavailable_reason_by_tool_id)
        ),
        disabled_tool_ids=frozenset(disabled_tool_ids),
        model_catalog_json=model_catalog_json,
        revocation_guard=EmergencyRevocationGuard(repository),
        authority=authority,
        catalog_snapshot_json=snapshot_json,
        catalog_snapshot_sha256=snapshot_sha256,
        workspace_corpus_authority=workspace_corpus_authority,
        turn_attachment_sources=attachment_sources,
        attachment_file_catalog=attachment_file_catalog(attachment_sources),
        protected_authority_by_key=MappingProxyType(
            dict(protected_authority_by_key)
        ),
        invocation_authority_resolver_by_key=MappingProxyType(
            dict(invocation_authority_resolver_by_key)
        ),
    )


def _index_l1_registrations(
    registrations: tuple[BoundToolRegistration, ...],
    *,
    definitions: tuple[ToolDefinition, ...],
) -> dict[str, BoundToolRegistration]:
    definition_by_tool_id: dict[str, ToolDefinition] = {}
    for definition in definitions:
        tool_id = definition.identity.tool_id
        if tool_id in definition_by_tool_id:
            raise RuntimeError(
                f"L1 default profile contains duplicate tool_id {tool_id!r}"
            )
        definition_by_tool_id[tool_id] = definition

    indexed: dict[str, BoundToolRegistration] = {}
    for registration in registrations:
        tool_id = registration.tool_id
        expected = definition_by_tool_id.get(tool_id)
        if expected is None or registration.definition != expected:
            raise RuntimeError(
                f"L1 binding escaped its persisted Definition: {tool_id!r}"
            )
        if tool_id in indexed:
            raise RuntimeError(f"duplicate L1 binding for {tool_id!r}")
        indexed[tool_id] = registration
    return indexed


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


__all__ = [
    "L1CatalogSnapshotRestoreError",
    "L1ToolExecution",
    "L1ToolCatalogUnavailableError",
    "L1PreparedToolCall",
    "L1ToolRuntime",
    "L1WorkspaceUnavailableError",
    "L1WorkspaceCorpusAuthority",
    "build_l1_tool_runtime",
]
