"""Auxiliary 语义验证权威的窄持久化门面。

``session.store`` 仍是稳定公开兼容接口。此端口向相邻语义验证记录传递显式
:class:`StoreDeps`，用于持久材料投影、能力冻结、审查请求与结果提交，以及法定人数读取
或结算。它刻意排除辅助重规划触发器、通用执行重规划请求、Runtime/API 策略和 schema
所有权。
"""

from __future__ import annotations

from . import auxiliary_semantic_verification
from ...deps import StoreDeps


AuxiliarySemanticVerificationPersistenceError = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationPersistenceError
)
AuxiliarySemanticVerificationIdentityCollision = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationIdentityCollision
)
AuxiliarySemanticVerificationStaleAuthority = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationStaleAuthority
)
AuxiliarySemanticVerificationStoredAuthorityCorrupt = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationStoredAuthorityCorrupt
)
FreezeAuxiliarySemanticCapabilityCatalogCommand = (
    auxiliary_semantic_verification.FreezeAuxiliarySemanticCapabilityCatalogCommand
)
CommitAuxiliarySemanticVerificationRequestCommand = (
    auxiliary_semantic_verification.CommitAuxiliarySemanticVerificationRequestCommand
)
CommitAuxiliarySemanticVerificationResultCommand = (
    auxiliary_semantic_verification.CommitAuxiliarySemanticVerificationResultCommand
)
SettleAuxiliarySemanticVerificationQuorumCommand = (
    auxiliary_semantic_verification.SettleAuxiliarySemanticVerificationQuorumCommand
)
StoredAuxiliarySemanticCapabilityCatalog = (
    auxiliary_semantic_verification.StoredAuxiliarySemanticCapabilityCatalog
)
StoredAuxiliarySemanticVerificationRequest = (
    auxiliary_semantic_verification.StoredAuxiliarySemanticVerificationRequest
)
StoredAuxiliarySemanticVerificationResult = (
    auxiliary_semantic_verification.StoredAuxiliarySemanticVerificationResult
)
StoredAuxiliarySemanticQuorumSettlement = (
    auxiliary_semantic_verification.StoredAuxiliarySemanticQuorumSettlement
)
AuxiliarySemanticCapabilityCatalogMutationResult = (
    auxiliary_semantic_verification.AuxiliarySemanticCapabilityCatalogMutationResult
)
AuxiliarySemanticVerificationRequestMutationResult = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationRequestMutationResult
)
AuxiliarySemanticVerificationResultMutationResult = (
    auxiliary_semantic_verification.AuxiliarySemanticVerificationResultMutationResult
)
AuxiliarySemanticQuorumSettlementMutationResult = (
    auxiliary_semantic_verification.AuxiliarySemanticQuorumSettlementMutationResult
)
AuxiliarySemanticMaterialProjection = (
    auxiliary_semantic_verification.AuxiliarySemanticMaterialProjection
)


def derive_auxiliary_semantic_review_policy(
    *,
    prompt_payload,
):
    return auxiliary_semantic_verification.derive_auxiliary_semantic_review_policy(
        prompt_payload=prompt_payload,
    )


def project_auxiliary_semantic_material(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> AuxiliarySemanticMaterialProjection:
    return auxiliary_semantic_verification.project_auxiliary_semantic_material(
        deps,
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )


def freeze_auxiliary_semantic_capability_catalog(
    deps: StoreDeps,
    *,
    command: FreezeAuxiliarySemanticCapabilityCatalogCommand,
) -> AuxiliarySemanticCapabilityCatalogMutationResult:
    return auxiliary_semantic_verification.freeze_auxiliary_semantic_capability_catalog(
        deps,
        command=command,
    )


def get_auxiliary_semantic_capability_catalog(
    deps: StoreDeps,
    *,
    session_id: str,
    capability_catalog_snapshot_id: str,
    projection_sha256: str,
) -> StoredAuxiliarySemanticCapabilityCatalog | None:
    return auxiliary_semantic_verification.get_auxiliary_semantic_capability_catalog(
        deps,
        session_id=session_id,
        capability_catalog_snapshot_id=capability_catalog_snapshot_id,
        projection_sha256=projection_sha256,
    )


def commit_auxiliary_semantic_verification_request(
    deps: StoreDeps,
    *,
    command: CommitAuxiliarySemanticVerificationRequestCommand,
) -> AuxiliarySemanticVerificationRequestMutationResult:
    return auxiliary_semantic_verification.commit_auxiliary_semantic_verification_request(
        deps,
        command=command,
    )


def commit_auxiliary_semantic_verification_result(
    deps: StoreDeps,
    *,
    command: CommitAuxiliarySemanticVerificationResultCommand,
) -> AuxiliarySemanticVerificationResultMutationResult:
    return auxiliary_semantic_verification.commit_auxiliary_semantic_verification_result(
        deps,
        command=command,
    )


def settle_auxiliary_semantic_verification_quorum(
    deps: StoreDeps,
    *,
    command: SettleAuxiliarySemanticVerificationQuorumCommand,
) -> AuxiliarySemanticQuorumSettlementMutationResult:
    return auxiliary_semantic_verification.settle_auxiliary_semantic_verification_quorum(
        deps,
        command=command,
    )


def get_auxiliary_semantic_verification_request(
    deps: StoreDeps,
    *,
    session_id: str,
    verification_request_id: str,
) -> StoredAuxiliarySemanticVerificationRequest | None:
    return auxiliary_semantic_verification.get_auxiliary_semantic_verification_request(
        deps,
        session_id=session_id,
        verification_request_id=verification_request_id,
    )


def get_auxiliary_semantic_verification_result(
    deps: StoreDeps,
    *,
    session_id: str,
    verification_result_id: str,
) -> StoredAuxiliarySemanticVerificationResult | None:
    return auxiliary_semantic_verification.get_auxiliary_semantic_verification_result(
        deps,
        session_id=session_id,
        verification_result_id=verification_result_id,
    )


def get_auxiliary_semantic_quorum_settlement(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    frozen_prompt_payload_sha256: str,
) -> StoredAuxiliarySemanticQuorumSettlement | None:
    return auxiliary_semantic_verification.get_auxiliary_semantic_quorum_settlement(
        deps,
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=auxiliary_graph_id,
        goal_id=goal_id,
        auxiliary_graph_revision=auxiliary_graph_revision,
        frozen_prompt_payload_sha256=frozen_prompt_payload_sha256,
    )
