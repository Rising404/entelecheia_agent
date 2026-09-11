"""L2 AuxiliaryGraph semantic-verification persistence facade.

Every persistence call resolves a fresh dependency bundle from ``session.store``
so the active Session database route is honored without making L2 own schema or
transaction lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable

from .. import store as session_store
from ..persistence.l2.delivery import auxiliary_semantic_verification_port as semantic_records
from ..persistence.deps import StoreDeps


AuxiliarySemanticVerificationPersistenceError = (
    semantic_records.AuxiliarySemanticVerificationPersistenceError
)
AuxiliarySemanticVerificationIdentityCollision = (
    semantic_records.AuxiliarySemanticVerificationIdentityCollision
)
AuxiliarySemanticVerificationStaleAuthority = (
    semantic_records.AuxiliarySemanticVerificationStaleAuthority
)
AuxiliarySemanticVerificationStoredAuthorityCorrupt = (
    semantic_records.AuxiliarySemanticVerificationStoredAuthorityCorrupt
)
FreezeAuxiliarySemanticCapabilityCatalogCommand = (
    semantic_records.FreezeAuxiliarySemanticCapabilityCatalogCommand
)
CommitAuxiliarySemanticVerificationRequestCommand = (
    semantic_records.CommitAuxiliarySemanticVerificationRequestCommand
)
CommitAuxiliarySemanticVerificationResultCommand = (
    semantic_records.CommitAuxiliarySemanticVerificationResultCommand
)
SettleAuxiliarySemanticVerificationQuorumCommand = (
    semantic_records.SettleAuxiliarySemanticVerificationQuorumCommand
)
StoredAuxiliarySemanticCapabilityCatalog = (
    semantic_records.StoredAuxiliarySemanticCapabilityCatalog
)
StoredAuxiliarySemanticVerificationRequest = (
    semantic_records.StoredAuxiliarySemanticVerificationRequest
)
StoredAuxiliarySemanticVerificationResult = (
    semantic_records.StoredAuxiliarySemanticVerificationResult
)
StoredAuxiliarySemanticQuorumSettlement = (
    semantic_records.StoredAuxiliarySemanticQuorumSettlement
)
AuxiliarySemanticCapabilityCatalogMutationResult = (
    semantic_records.AuxiliarySemanticCapabilityCatalogMutationResult
)
AuxiliarySemanticVerificationRequestMutationResult = (
    semantic_records.AuxiliarySemanticVerificationRequestMutationResult
)
AuxiliarySemanticVerificationResultMutationResult = (
    semantic_records.AuxiliarySemanticVerificationResultMutationResult
)
AuxiliarySemanticQuorumSettlementMutationResult = (
    semantic_records.AuxiliarySemanticQuorumSettlementMutationResult
)
AuxiliarySemanticMaterialProjection = (
    semantic_records.AuxiliarySemanticMaterialProjection
)


class _AuxiliarySemanticStoreFacade:
    def __init__(self, *, deps_factory: Callable[[], StoreDeps]) -> None:
        self._deps_factory = deps_factory

    def derive_auxiliary_semantic_review_policy(self, *, prompt_payload):
        return semantic_records.derive_auxiliary_semantic_review_policy(
            prompt_payload=prompt_payload,
        )

    def project_auxiliary_semantic_material(
        self,
        *,
        session_id: str,
        turn_id: str,
        task_id: str,
    ) -> AuxiliarySemanticMaterialProjection:
        return semantic_records.project_auxiliary_semantic_material(
            self._deps_factory(),
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
        )

    def freeze_auxiliary_semantic_capability_catalog(
        self,
        *,
        command: FreezeAuxiliarySemanticCapabilityCatalogCommand,
    ) -> AuxiliarySemanticCapabilityCatalogMutationResult:
        return semantic_records.freeze_auxiliary_semantic_capability_catalog(
            self._deps_factory(),
            command=command,
        )

    def get_auxiliary_semantic_capability_catalog(
        self,
        *,
        session_id: str,
        capability_catalog_snapshot_id: str,
        projection_sha256: str,
    ) -> StoredAuxiliarySemanticCapabilityCatalog | None:
        return semantic_records.get_auxiliary_semantic_capability_catalog(
            self._deps_factory(),
            session_id=session_id,
            capability_catalog_snapshot_id=capability_catalog_snapshot_id,
            projection_sha256=projection_sha256,
        )

    def commit_auxiliary_semantic_verification_request(
        self,
        *,
        command: CommitAuxiliarySemanticVerificationRequestCommand,
    ) -> AuxiliarySemanticVerificationRequestMutationResult:
        return semantic_records.commit_auxiliary_semantic_verification_request(
            self._deps_factory(),
            command=command,
        )

    def commit_auxiliary_semantic_verification_result(
        self,
        *,
        command: CommitAuxiliarySemanticVerificationResultCommand,
    ) -> AuxiliarySemanticVerificationResultMutationResult:
        return semantic_records.commit_auxiliary_semantic_verification_result(
            self._deps_factory(),
            command=command,
        )

    def settle_auxiliary_semantic_verification_quorum(
        self,
        *,
        command: SettleAuxiliarySemanticVerificationQuorumCommand,
    ) -> AuxiliarySemanticQuorumSettlementMutationResult:
        return semantic_records.settle_auxiliary_semantic_verification_quorum(
            self._deps_factory(),
            command=command,
        )

    def get_auxiliary_semantic_verification_request(
        self,
        *,
        session_id: str,
        verification_request_id: str,
    ) -> StoredAuxiliarySemanticVerificationRequest | None:
        return semantic_records.get_auxiliary_semantic_verification_request(
            self._deps_factory(),
            session_id=session_id,
            verification_request_id=verification_request_id,
        )

    def get_auxiliary_semantic_verification_result(
        self,
        *,
        session_id: str,
        verification_result_id: str,
    ) -> StoredAuxiliarySemanticVerificationResult | None:
        return semantic_records.get_auxiliary_semantic_verification_result(
            self._deps_factory(),
            session_id=session_id,
            verification_result_id=verification_result_id,
        )

    def get_auxiliary_semantic_quorum_settlement(
        self,
        *,
        session_id: str,
        task_id: str,
        auxiliary_graph_id: str,
        goal_id: str,
        auxiliary_graph_revision: int,
        frozen_prompt_payload_sha256: str,
    ) -> StoredAuxiliarySemanticQuorumSettlement | None:
        return semantic_records.get_auxiliary_semantic_quorum_settlement(
            self._deps_factory(),
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            goal_id=goal_id,
            auxiliary_graph_revision=auxiliary_graph_revision,
            frozen_prompt_payload_sha256=frozen_prompt_payload_sha256,
        )


_FACADE = _AuxiliarySemanticStoreFacade(
    deps_factory=lambda: session_store.current_store_deps(),
)

derive_auxiliary_semantic_review_policy = (
    _FACADE.derive_auxiliary_semantic_review_policy
)
project_auxiliary_semantic_material = (
    _FACADE.project_auxiliary_semantic_material
)
freeze_auxiliary_semantic_capability_catalog = (
    _FACADE.freeze_auxiliary_semantic_capability_catalog
)
get_auxiliary_semantic_capability_catalog = (
    _FACADE.get_auxiliary_semantic_capability_catalog
)
commit_auxiliary_semantic_verification_request = (
    _FACADE.commit_auxiliary_semantic_verification_request
)
commit_auxiliary_semantic_verification_result = (
    _FACADE.commit_auxiliary_semantic_verification_result
)
settle_auxiliary_semantic_verification_quorum = (
    _FACADE.settle_auxiliary_semantic_verification_quorum
)
get_auxiliary_semantic_verification_request = (
    _FACADE.get_auxiliary_semantic_verification_request
)
get_auxiliary_semantic_verification_result = (
    _FACADE.get_auxiliary_semantic_verification_result
)
get_auxiliary_semantic_quorum_settlement = (
    _FACADE.get_auxiliary_semantic_quorum_settlement
)


__all__ = [
    "AuxiliarySemanticCapabilityCatalogMutationResult",
    "AuxiliarySemanticQuorumSettlementMutationResult",
    "AuxiliarySemanticVerificationIdentityCollision",
    "AuxiliarySemanticVerificationPersistenceError",
    "AuxiliarySemanticVerificationRequestMutationResult",
    "AuxiliarySemanticVerificationResultMutationResult",
    "AuxiliarySemanticVerificationStaleAuthority",
    "AuxiliarySemanticVerificationStoredAuthorityCorrupt",
    "AuxiliarySemanticMaterialProjection",
    "CommitAuxiliarySemanticVerificationRequestCommand",
    "CommitAuxiliarySemanticVerificationResultCommand",
    "FreezeAuxiliarySemanticCapabilityCatalogCommand",
    "SettleAuxiliarySemanticVerificationQuorumCommand",
    "StoredAuxiliarySemanticCapabilityCatalog",
    "StoredAuxiliarySemanticQuorumSettlement",
    "StoredAuxiliarySemanticVerificationRequest",
    "StoredAuxiliarySemanticVerificationResult",
    "commit_auxiliary_semantic_verification_request",
    "commit_auxiliary_semantic_verification_result",
    "derive_auxiliary_semantic_review_policy",
    "freeze_auxiliary_semantic_capability_catalog",
    "get_auxiliary_semantic_capability_catalog",
    "get_auxiliary_semantic_quorum_settlement",
    "get_auxiliary_semantic_verification_request",
    "get_auxiliary_semantic_verification_result",
    "project_auxiliary_semantic_material",
    "settle_auxiliary_semantic_verification_quorum",
]
