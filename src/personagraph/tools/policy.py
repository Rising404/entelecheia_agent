"""Tool Platform 的纯效果感知策略核心。

调用方提供范围授予和预算事实。本模块既不读取 Session/Runtime 状态，也不渲染批准 UI。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .effects import (
    DataEgress,
    EffectAction,
    EffectFact,
    EffectResource,
    EffectScopeKind,
    derive_effect_facts,
)
from .registration import ExecutableToolRegistration


TOOL_POLICY_VERSION = "1.1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PolicyDisposition(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    AUTHORIZATION_REQUIRED = "authorization_required"
    APPROVAL_REQUIRED = "approval_required"
    DEFER = "defer"


class PolicySeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ScopeGrant:
    resource: EffectResource
    action: EffectAction
    scope_kind: EffectScopeKind
    scope: str = "*"

    def covers(self, fact: EffectFact) -> bool:
        return (
            self.resource is fact.resource
            and self.action is fact.action
            and self.scope_kind is fact.scope_kind
            and (self.scope == "*" or self.scope == fact.scope)
        )


@dataclass(frozen=True)
class AuthorityFacts:
    grants: tuple[ScopeGrant, ...] = ()
    approval_grants: tuple[ScopeGrant, ...] = ()
    allow_local_read: bool = True

    def authorizes(self, fact: EffectFact) -> bool:
        return any(grant.covers(fact) for grant in self.grants)

    def approves(self, fact: EffectFact) -> bool:
        """一个精确的受保护效果是否已有 Host 回执。

        授权回答工具可在“何处”操作；批准保留给文件修改等受保护效果。
        产品已默认同意外发，不再用这一组事实对网络传输逐次询问。
        """

        return any(grant.covers(fact) for grant in self.approval_grants)


@dataclass(frozen=True, slots=True)
class ProtectedToolExecutionAuthority:
    """物理效果前重验来源/后端的证明；持久 receipt 字段不等于人工审批。"""

    approval_receipt_ids: tuple[str, ...]
    execution_backend_identity_sha256: str
    revalidate: Callable[[], bool] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        receipts = self.approval_receipt_ids
        if not isinstance(receipts, tuple) or not receipts:
            raise ValueError("protected tool authority requires approval receipts")
        if len(receipts) != len(set(receipts)):
            raise ValueError("protected tool approval receipts must be unique")
        if any(
            not isinstance(receipt_id, str)
            or not receipt_id
            or len(receipt_id) > 200
            for receipt_id in receipts
        ):
            raise ValueError("protected tool approval receipt is invalid")
        if not _SHA256.fullmatch(self.execution_backend_identity_sha256):
            raise ValueError(
                "execution_backend_identity_sha256 must be a canonical hash"
            )
        if not callable(self.revalidate):
            raise TypeError("protected tool revalidate must be callable")

    @property
    def binding_sha256(self) -> str:
        """将语义回执集合绑定到精确物理后端。"""

        encoded = json.dumps(
            {
                "schema_version": "protected-tool-execution-authority-v1",
                "approval_receipt_ids": sorted(self.approval_receipt_ids),
                "execution_backend_identity_sha256": (
                    self.execution_backend_identity_sha256
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolInvocationAuthority:
    """参数确定后取得的单次授权事实，不扩大后续调用的权限。

    视觉读取等工具必须先知道具体来源和用途，才能生成有效回执。组合层提供
    解析函数，运行层在 schema 校验之后调用，再交给本模块的纯 policy 检查。
    """

    authority: AuthorityFacts
    protected_authority: ProtectedToolExecutionAuthority


ToolInvocationAuthorityResolver = Callable[
    [Mapping[str, Any]], ToolInvocationAuthority
]


@dataclass(frozen=True)
class BudgetFacts:
    remaining_tool_calls: int | None = None
    remaining_output_bytes: int | None = None


@dataclass(frozen=True)
class PolicyRequest:
    normalized_arguments: Mapping[str, Any]
    effects: tuple[EffectFact, ...] | None
    authority: AuthorityFacts = field(default_factory=AuthorityFacts)
    budget: BudgetFacts = field(default_factory=BudgetFacts)
    policy_version: str = TOOL_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != TOOL_POLICY_VERSION:
            raise ValueError("unsupported Tool policy version")

    @classmethod
    def from_registration(
        cls,
        registration: ExecutableToolRegistration,
        normalized_arguments: Mapping[str, Any],
        *,
        authority: AuthorityFacts | None = None,
        budget: BudgetFacts | None = None,
    ) -> "PolicyRequest":
        return cls(
            normalized_arguments=dict(normalized_arguments),
            effects=derive_effect_facts(registration.effect_profile, normalized_arguments),
            authority=authority or AuthorityFacts(),
            budget=budget or BudgetFacts(),
        )


@dataclass(frozen=True)
class ToolPolicyDecision:
    disposition: PolicyDisposition
    severity: PolicySeverity
    reason_codes: tuple[str, ...]
    effects: tuple[EffectFact, ...]
    required_authorization: tuple[EffectFact, ...] = ()
    approval_required: bool = False
    policy_version: str = TOOL_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != TOOL_POLICY_VERSION:
            raise ValueError("unsupported Tool policy version")

    @property
    def allowed(self) -> bool:
        return self.disposition is PolicyDisposition.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "severity": self.severity.value,
            "reason_codes": list(self.reason_codes),
            "approval_required": self.approval_required,
            "policy_version": self.policy_version,
            "effects": [
                {
                    "resource": effect.resource.value,
                    "action": effect.action.value,
                    "scope_kind": effect.scope_kind.value,
                    "scope": effect.scope,
                    "resource_id": effect.resource_id,
                    "data_egress": effect.data_egress.value,
                    "sensitive_egress": effect.sensitive_egress,
                }
                for effect in self.effects
            ],
        }


class ToolPolicyCore:
    """保守的确定性策略。批准和授权是事实，而非布尔值。"""

    def evaluate(self, request: PolicyRequest) -> ToolPolicyDecision:
        effects = request.effects
        if not effects:
            return ToolPolicyDecision(
                PolicyDisposition.DENY,
                PolicySeverity.HIGH,
                ("missing_effect_profile",),
                (),
                policy_version=request.policy_version,
            )
        if request.budget.remaining_tool_calls is not None and request.budget.remaining_tool_calls <= 0:
            return ToolPolicyDecision(
                PolicyDisposition.DENY,
                PolicySeverity.MEDIUM,
                ("tool_call_budget_exhausted",),
                effects,
                policy_version=request.policy_version,
            )

        authorization_needed = tuple(effect for effect in effects if self._needs_authorization(effect, request.authority))
        if authorization_needed:
            return ToolPolicyDecision(
                PolicyDisposition.AUTHORIZATION_REQUIRED,
                self._severity(effects),
                ("scope_authorization_required",),
                effects,
                required_authorization=authorization_needed,
                policy_version=request.policy_version,
            )

        if any(
            self._needs_approval(effect)
            and not request.authority.approves(effect)
            for effect in effects
        ):
            return ToolPolicyDecision(
                PolicyDisposition.APPROVAL_REQUIRED,
                self._severity(effects),
                ("protected_effect_approval_required",),
                effects,
                approval_required=True,
                policy_version=request.policy_version,
            )

        return ToolPolicyDecision(
            PolicyDisposition.ALLOW,
            self._severity(effects),
            ("effects_allowed",),
            effects,
            policy_version=request.policy_version,
        )

    @staticmethod
    def _needs_authorization(effect: EffectFact, authority: AuthorityFacts) -> bool:
        if authority.authorizes(effect):
            return False
        if effect.scope_kind is EffectScopeKind.LOCAL and effect.action in {EffectAction.READ, EffectAction.SEARCH}:
            return not authority.allow_local_read
        return effect.scope_kind in {
            EffectScopeKind.WORKSPACE,
            EffectScopeKind.CONFIGURED_ROOT,
            EffectScopeKind.ACCOUNT,
        }

    @staticmethod
    def _needs_approval(effect: EffectFact) -> bool:
        if (
            effect.resource is EffectResource.RUNTIME_STATE
            and effect.scope_kind in {EffectScopeKind.EXECUTION, EffectScopeKind.SESSION}
            and effect.action is EffectAction.UPDATE
            and effect.data_egress is DataEgress.NONE
        ):
            # 计划、分析结果等内部状态不是用户文件写入，不需要单独人工批准。
            return False
        # 外发由用户统一同意。敏感内容分类仅用于审计，不再构成人工批准门槛；
        # 文件修改和执行等独立效果仍须满足自己的权限，不能借外发默认值提权。
        return effect.action in {
            EffectAction.CREATE, EffectAction.UPDATE,
            EffectAction.DELETE, EffectAction.EXECUTE,
        }

    @staticmethod
    def _severity(effects: tuple[EffectFact, ...]) -> PolicySeverity:
        if all(
            effect.resource is EffectResource.RUNTIME_STATE
            and effect.scope_kind is EffectScopeKind.EXECUTION
            and effect.data_egress is DataEgress.NONE
            for effect in effects
        ):
            return PolicySeverity.INFO
        if any(effect.sensitive_egress for effect in effects):
            return PolicySeverity.CRITICAL
        if any(effect.action is EffectAction.DELETE for effect in effects):
            return PolicySeverity.HIGH
        if any(effect.action in {EffectAction.CREATE, EffectAction.UPDATE, EffectAction.EXECUTE, EffectAction.TRANSMIT} for effect in effects):
            return PolicySeverity.MEDIUM
        if any(effect.resource is EffectResource.NETWORK for effect in effects):
            return PolicySeverity.LOW
        return PolicySeverity.INFO
