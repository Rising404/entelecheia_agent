"""用于工作区认知、绑定会话的 Tool Platform 组合。

每项注册都闭包同一个由宿主解析的工作目录边界。本地读取由一份持久根绑定回执
覆盖；获批的外部视觉读取还会经过受保护的操作账本，因此恢复过程绝不会把
丢失的响应变成第二次传输。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ...input_processing.vision.providers import (
    vision_adapter_transmits_externally,
)
from ...input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
)
from ...workspace.storage.context import require_current
from ..catalog import CatalogSnapshot, ToolCatalog
from ..catalog.binding import ToolBinding
from ..documents.format_observation_catalog import (
    FormatObservationBindingFacts,
    build_format_observation_tool_bindings,
    build_format_observation_tool_definition_manifest,
)
from ..documents.external_visual_analysis_catalog import (
    ExternalVisualAnalysisBindingFacts,
    build_external_visual_analysis_tool_bindings,
)
from ..documents.external_visual_authority import (
    build_external_visual_authority_resolver,
)
from ..effects import (
    EffectAction,
    EffectResource,
    EffectScopeKind,
)
from ..execution import ToolBusinessFailure
from ..documents.format_observation_tools import (
    EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS,
    FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
    build_external_visual_analysis_tool_registrations,
    build_format_observation_tool_registrations,
)
from ..policy import AuthorityFacts, ScopeGrant
from ..visual.visual_tools import default_vision_adapter
from ..visual.project_observation_publication import VisualObservationPublisher
from ..visual.path_publication_source import build_visual_path_resolver
from ..files.file_authority import SessionFileToolAuthority
from .workspace_discovery_catalog import (
    WorkspaceDiscoveryBindingFacts,
    build_workspace_discovery_tool_bindings,
)
from .workspace_tools import (
    FrozenWorkspaceToolBoundary,
    WORKSPACE_DISCOVERY_TOOL_IDS,
    build_workspace_discovery_tool_registrations,
)
from ...runtime.model_calls.vision import (
    DurableMountedVisionAdapter,
    SqliteMountedVisualCallLedger,
)
from ..policy import ProtectedToolExecutionAuthority, ToolInvocationAuthorityResolver
from ...session.local_file_authority import (
    LocalFileAction,
    SqliteSessionFileAuthority,
)

_LOCAL_VISUAL_ADAPTER_UNSUPPORTED = "local_visual_adapter_not_composed"


@dataclass(frozen=True, slots=True)
class SessionWorkspaceReadonlyRuntime:
    """面向一个会话根目录的不可变目录与执行桥接器。"""

    session_id: str
    boundary: FrozenWorkspaceToolBoundary
    catalog_snapshot: CatalogSnapshot
    workspace_discovery_bindings: tuple[
        ToolBinding,
        ToolBinding,
        ToolBinding,
        ToolBinding,
        ToolBinding,
    ]
    format_observation_bindings: tuple[ToolBinding, ...]
    external_visual_analysis_bindings: tuple[ToolBinding, ...]
    scope_snapshot_sha256: str
    tool_ids: tuple[str, ...]
    boundary_fingerprint: str
    local_file_authorization_receipt_id: str
    visual_analysis_available: bool
    visual_analysis_reason: str | None
    authority: AuthorityFacts
    protected_authority_by_key: dict[tuple[str, str], ProtectedToolExecutionAuthority]
    invocation_authority_resolver_by_key: dict[
        tuple[str, str], ToolInvocationAuthorityResolver
    ]


def build_session_workspace_readonly_runtime(
    session_id: str,
) -> SessionWorkspaceReadonlyRuntime | None:
    """冻结当前会话根目录，并组合所有受支持的读取工具。

    ``None`` 表示会话没有可用的已绑定工作目录。这种缺失会作为不可用能力投影
    给规划器，绝不会用进程当前目录替代。

    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")
    from ...session import store as session_store

    session = session_store.get_session(session_id)
    if session is None:
        raise ValueError("session does not exist")
    working_dir = session.get("working_dir")
    if not isinstance(working_dir, str) or not working_dir.strip():
        return None
    try:
        boundary = FrozenWorkspaceToolBoundary(
            session_id=session_id,
            root=working_dir,
        )
    except ValueError:
        return None

    file_authority = SqliteSessionFileAuthority()
    local_read = file_authority.authorize(
        session_id=session_id,
        candidate=boundary.root,
        action=LocalFileAction.READ,
        working_root=boundary.root,
    )
    local_search = file_authority.authorize(
        session_id=session_id,
        candidate=boundary.root,
        action=LocalFileAction.SEARCH,
        working_root=boundary.root,
    )
    if (
        not local_read.allowed
        or not local_search.allowed
        or not local_read.grant_id
        or local_search.grant_id != local_read.grant_id
    ):
        return None
    frozen_file_authority = SqliteSessionFileAuthority(file_authority.path)
    read_tool_boundary = replace(
        boundary,
        freshness_check=_workspace_read_freshness_check(
            authority=frozen_file_authority,
            session_id=session_id,
            root=boundary.root,
            expected_grant_id=local_read.grant_id,
        ),
    )

    configured_vision_adapter = default_vision_adapter()
    configured_external_vision = vision_adapter_transmits_externally(
        configured_vision_adapter
    )
    visual_call_ledger = SqliteMountedVisualCallLedger()
    durable_vision_adapter = (
        DurableMountedVisionAdapter(
            configured_vision_adapter,
            session_id=session_id,
            ledger=visual_call_ledger,
        )
        if configured_external_vision
        else None
    )
    runtime_vision_adapter = durable_vision_adapter or configured_vision_adapter
    # 封装器保留一个不可变快照，供外发、提供方绑定与注册指纹共同使用。
    vision_capabilities = runtime_vision_adapter.capabilities()
    external_vision = bool(vision_capabilities.available and configured_external_vision)
    # 外发默认允许；具体来源的路径、版本与用途在每次调用参数确定后校验。
    visual_analysis_available = external_vision
    if not vision_capabilities.available:
        visual_analysis_reason = vision_capabilities.reason_code
    elif not external_vision:
        visual_analysis_reason = _LOCAL_VISUAL_ADAPTER_UNSUPPORTED
    else:
        visual_analysis_reason = None

    resolve_visual_file = None
    visual_publisher = None
    if visual_analysis_available:
        with session_store.session_database_scope(session_id):
            database = require_current()
        if database.project_root != read_tool_boundary.root:
            raise ValueError("visual publication crossed the bound Project root")
        visual_file_authority = SessionFileToolAuthority(
            session_id,
            read_tool_boundary,
            local_read.grant_id,
        )
        resolve_visual_file = build_visual_path_resolver(
            database=database,
            validate_path=visual_file_authority.validate_path,
        )
        visual_publisher = VisualObservationPublisher(
            session_id=session_id,
            call_ledger=visual_call_ledger,
        )

    external_visual_registrations = (
        build_external_visual_analysis_tool_registrations(
            read_tool_boundary,
            vision_adapter=runtime_vision_adapter,
            resolve_visual_file=resolve_visual_file,
            visual_publisher=visual_publisher,
        )
        if visual_analysis_available
        else ()
    )

    boundary_fingerprint = _canonical_sha256(
        {
            "schema_version": "session-workspace-tool-boundary-v1",
            "session_id": session_id,
            "resolved_root": str(boundary.root),
            "root_device": boundary.root_device,
            "root_inode": boundary.root_inode,
            "respect_ignore": boundary.respect_ignore,
            "hidden": boundary.hidden,
        }
    )
    workspace_discovery_registrations = build_workspace_discovery_tool_registrations(
        read_tool_boundary,
    )
    format_observation_registrations = build_format_observation_tool_registrations(
        read_tool_boundary,
    )
    raw_registrations = (
        *workspace_discovery_registrations,
        *format_observation_registrations,
        *external_visual_registrations,
    )
    registrations = tuple(
        replace(
            registration,
            source=replace(
                registration.source,
                fingerprint=_canonical_sha256(
                    {
                        "schema_version": "workspace-registration-scope-v1",
                        "root_boundary": boundary_fingerprint,
                        "source": registration.source.to_dict(),
                        "tool_id": registration.tool_id,
                    }
                ),
            ),
        )
        for registration in raw_registrations
    )
    tool_ids = tuple(registration.tool_id for registration in registrations)
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("workspace tool IDs must be unique")
    _require_supported_workspace_effects(
        registrations,
        external_visual_enabled=visual_analysis_available,
    )
    registrations_by_id = {
        registration.tool_id: registration for registration in registrations
    }
    read_authority_sha256 = _canonical_sha256(
        {
            "schema_version": "workspace-read-authority-binding-v1",
            "session_id": session_id,
            "grant_id": local_read.grant_id,
        }
    )
    workspace_discovery_bindings = build_workspace_discovery_tool_bindings(
        tuple(registrations_by_id[tool_id] for tool_id in WORKSPACE_DISCOVERY_TOOL_IDS),
        facts=WorkspaceDiscoveryBindingFacts(
            boundary_sha256=boundary_fingerprint,
            read_authority_sha256=read_authority_sha256,
        ),
    )
    format_observation_bindings = build_format_observation_tool_bindings(
        tuple(
            registrations_by_id[tool_id]
            for tool_id in FORMAT_OBSERVATION_LOCAL_TOOL_IDS
        ),
        facts=FormatObservationBindingFacts(
            boundary_sha256=boundary_fingerprint,
            read_authority_sha256=read_authority_sha256,
            reader_stack_sha256=_canonical_sha256(
                {
                    "schema_version": "format-observation-reader-stack-v1",
                    "definition_digests": [
                        item.definition.digest
                        for item in (
                            build_format_observation_tool_definition_manifest()
                        )
                    ],
                }
            ),
        ),
    )
    external_visual_analysis_bindings: tuple[ToolBinding, ...] = ()
    if external_visual_registrations:
        external_visual_analysis_bindings = (
            build_external_visual_analysis_tool_bindings(
                tuple(
                    registrations_by_id[tool_id]
                    for tool_id in EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS
                ),
                facts=ExternalVisualAnalysisBindingFacts(
                    boundary_sha256=boundary_fingerprint,
                    read_authority_sha256=read_authority_sha256,
                    provider_identity_sha256=(
                        _vision_provider_identity_sha256(vision_capabilities)
                    ),
                    capability_snapshot_sha256=_canonical_sha256(
                        {
                            "schema_version": (
                                "workspace-vision-capability-binding-v1"
                            ),
                            "available": vision_capabilities.available,
                            "provider": vision_capabilities.provider,
                            "model": vision_capabilities.model,
                            "endpoint_identity": (
                                vision_capabilities.endpoint_identity
                            ),
                            "processor_fingerprint": (
                                vision_capabilities.processor_fingerprint
                            ),
                            "supported_purposes": [
                                purpose.value
                                for purpose in (vision_capabilities.supported_purposes)
                            ],
                            "reason_code": vision_capabilities.reason_code,
                        }
                    ),
                    egress_policy_sha256=_canonical_sha256(
                        {
                            "schema_version": (
                                "workspace-vision-disclosure-binding-v1"
                            ),
                            "session_id": session_id,
                            "policy": "default-egress-with-exact-source-binding",
                        }
                    ),
                    physical_call_ledger_sha256=_canonical_sha256(
                        {
                            "schema_version": (
                                "workspace-vision-physical-ledger-binding-v1"
                            ),
                            "session_id": session_id,
                            "path": str(
                                visual_call_ledger.path_for(session_id).resolve()
                            ),
                        }
                    ),
                ),
            )
        )

    catalog = ToolCatalog()
    for registration in registrations:
        catalog.register(registration)
    snapshot = catalog.snapshot()
    scope_snapshot_sha256 = _canonical_sha256(
        {
            "schema_version": "session-workspace-tool-scope-v2",
            "session_id": session_id,
            "resolved_root": str(boundary.root),
            "root_device": boundary.root_device,
            "root_inode": boundary.root_inode,
            "boundary_fingerprint": boundary_fingerprint,
            "local_file_authorization_receipt_id": local_read.grant_id,
            "catalog": snapshot.to_descriptor(),
        }
    )
    authority = AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.WORKSPACE,
                str(boundary.root),
            ),
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.SEARCH,
                EffectScopeKind.WORKSPACE,
                str(boundary.root),
            ),
        ),
    )
    protected_authority_by_key: dict[
        tuple[str, str], ProtectedToolExecutionAuthority
    ] = {}
    invocation_authority_resolver_by_key: dict[
        tuple[str, str], ToolInvocationAuthorityResolver
    ] = {}
    if external_visual_registrations:
        provider_identity = _vision_provider_identity_sha256(vision_capabilities)
        for registration in external_visual_registrations:
            invocation_authority_resolver_by_key[
                (registration.tool_id, registration.contract_version)
            ] = build_external_visual_authority_resolver(
                boundary=read_tool_boundary,
                tool_id=registration.tool_id,
                capabilities=vision_capabilities,
                backend_identity_sha256=provider_identity,
            )
    return SessionWorkspaceReadonlyRuntime(
        session_id=session_id,
        boundary=boundary,
        catalog_snapshot=snapshot,
        workspace_discovery_bindings=workspace_discovery_bindings,
        format_observation_bindings=format_observation_bindings,
        external_visual_analysis_bindings=(external_visual_analysis_bindings),
        scope_snapshot_sha256=scope_snapshot_sha256,
        tool_ids=tool_ids,
        boundary_fingerprint=boundary_fingerprint,
        local_file_authorization_receipt_id=local_read.grant_id,
        visual_analysis_available=visual_analysis_available,
        visual_analysis_reason=visual_analysis_reason,
        authority=authority,
        protected_authority_by_key=protected_authority_by_key,
        invocation_authority_resolver_by_key=invocation_authority_resolver_by_key,
    )


def _workspace_read_freshness_check(
    *,
    authority: SqliteSessionFileAuthority,
    session_id: str,
    root: Path,
    expected_grant_id: str,
) -> Callable[[], None]:
    """Freeze one exact read grant check into every local read handler."""

    def require_current() -> None:
        decision = authority.authorize(
            session_id=session_id,
            candidate=root,
            action=LocalFileAction.READ,
            working_root=root,
        )
        if not decision.allowed or decision.grant_id != expected_grant_id:
            raise ToolBusinessFailure(
                "workspace_authority_revoked",
                "The frozen workspace read authorization is no longer active.",
            )

    return require_current


def _vision_provider_identity_sha256(
    capabilities: VisionCapabilitySnapshot,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "workspace-vision-provider-identity-v1",
            "provider": capabilities.provider,
            "model": capabilities.model,
            "endpoint_identity": capabilities.endpoint_identity,
            "processor_fingerprint": capabilities.processor_fingerprint,
        }
    )


def _require_supported_workspace_effects(
    registrations,
    *,
    external_visual_enabled: bool,
) -> None:
    external_tool_ids = {"analyze_image", "analyze_pdf_page"}
    for registration in registrations:
        for descriptor in registration.effect_profile.effects:
            if descriptor.action in {EffectAction.READ, EffectAction.SEARCH}:
                if (
                    descriptor.resource is not EffectResource.FILESYSTEM
                    or descriptor.scope_kind is not EffectScopeKind.WORKSPACE
                ):
                    raise ValueError(
                        "workspace local tools require exact filesystem scope"
                    )
                continue
            if (
                external_visual_enabled
                and registration.tool_id in external_tool_ids
                and (descriptor.resource, descriptor.action)
                in {
                    (EffectResource.NETWORK, EffectAction.TRANSMIT),
                    (EffectResource.RUNTIME_STATE, EffectAction.UPDATE),
                }
                and descriptor.scope_kind is EffectScopeKind.SESSION
            ):
                continue
            raise ValueError("workspace runtime contains an unsupported effect")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SessionWorkspaceReadonlyRuntime",
    "build_session_workspace_readonly_runtime",
]
