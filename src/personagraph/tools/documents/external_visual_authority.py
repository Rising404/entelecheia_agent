"""格式视觉工具的执行来源绑定：具体来源、用途和提供方共同确定回执。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...input_processing.vision.contracts import VisionCapabilitySnapshot, VisionPurpose
from ..visual.egress_policy import auto_visual_egress_receipt
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..execution import ToolBusinessFailure
from ..policy import (
    AuthorityFacts, ProtectedToolExecutionAuthority, ScopeGrant,
    ToolInvocationAuthority, ToolInvocationAuthorityResolver,
)
from ..workspace.workspace_tools import FrozenWorkspaceToolBoundary
from .format_observation_source_authority import _resolve_file, _source_identity
from .format_observation_tools import IMAGE_SUFFIXES, PDF_SUFFIXES


def build_external_visual_authority_resolver(
    *,
    boundary: FrozenWorkspaceToolBoundary,
    tool_id: str,
    capabilities: VisionCapabilitySnapshot,
    backend_identity_sha256: str,
) -> ToolInvocationAuthorityResolver:
    """模型选定文件后冻结可读来源、用途和提供方；外发默认同意。"""

    suffixes = {
        "analyze_image": IMAGE_SUFFIXES,
        "analyze_pdf_page": PDF_SUFFIXES,
    }[tool_id]

    def prepare(arguments: Mapping[str, Any]) -> ToolInvocationAuthority:
        arguments = dict(arguments)
        target, _ = _resolve_file(boundary, arguments, suffixes)
        source = _source_identity(target)
        purpose = VisionPurpose(arguments["purpose"])
        if purpose not in capabilities.supported_purposes:
            raise ToolBusinessFailure(
                "vision_purpose_unavailable", "当前视觉服务不支持此用途。"
            )

        receipt_id = auto_visual_egress_receipt(
            session_id=boundary.session_id,
            source_sha256=source["sha256"],
            endpoint_identity=capabilities.endpoint_identity,
            model=capabilities.model,
            purpose=purpose,
        )

        def revalidate() -> bool:
            try:
                current_target, _ = _resolve_file(boundary, arguments, suffixes)
                return bool(
                    current_target == target
                    and _source_identity(current_target) == source
                )
            except (OSError, RuntimeError, TypeError, ValueError, ToolBusinessFailure):
                return False

        return ToolInvocationAuthority(
            authority=AuthorityFacts(grants=(
                ScopeGrant(
                    EffectResource.NETWORK, EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION, boundary.session_id,
                ),
                # 视觉操作可保存其派生问答，但不授予修改工作区文件的能力。
                ScopeGrant(
                    EffectResource.RUNTIME_STATE, EffectAction.UPDATE,
                    EffectScopeKind.SESSION, boundary.session_id,
                ),
            )),
            protected_authority=ProtectedToolExecutionAuthority(
                approval_receipt_ids=(receipt_id,),
                execution_backend_identity_sha256=backend_identity_sha256,
                revalidate=revalidate,
            ),
        )

    return prepare
