"""L2 Auxiliary terminal persistence facade.

Validation projections, terminal sealing, and TaskGraph commit retain their
existing persistence owners and transaction boundaries.  This facade only
resolves the current Session database route for complete operations.
"""

from __future__ import annotations

from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionValidationContext,
)

from .. import store as session_store
from ..persistence.l2.auxiliary_graph import auxiliary_task_graph_commit as commit_records
from ..persistence.l2.auxiliary_graph import auxiliary_terminal_seal as seal_records
from ..persistence.l2.auxiliary_graph import auxiliary_terminal_validation as validation_records


AuxiliaryTerminalSemanticSupport = (
    validation_records.AuxiliaryTerminalSemanticSupport
)
AuxiliaryTerminalSealPersistenceError = (
    seal_records.AuxiliaryTerminalSealPersistenceError
)
AuxiliaryTerminalSealIdentityCollision = (
    seal_records.AuxiliaryTerminalSealIdentityCollision
)
SealAuxiliaryTerminalProposalCommand = (
    seal_records.SealAuxiliaryTerminalProposalCommand
)
AuxiliaryTerminalSealResult = seal_records.AuxiliaryTerminalSealResult
AuxiliaryTerminalProposalReceipt = (
    seal_records.AuxiliaryTerminalProposalReceipt
)
AuxiliaryTaskGraphCommitFailureCode = (
    commit_records.AuxiliaryTaskGraphCommitFailureCode
)
AuxiliaryTaskGraphCommitPersistenceError = (
    commit_records.AuxiliaryTaskGraphCommitPersistenceError
)
AuxiliaryTaskGraphCommitIdentityCollision = (
    commit_records.AuxiliaryTaskGraphCommitIdentityCollision
)
AuxiliaryBaseNodeAliasBinding = (
    commit_records.AuxiliaryBaseNodeAliasBinding
)
CommitAuxiliaryTaskGraphProposalCommand = (
    commit_records.CommitAuxiliaryTaskGraphProposalCommand
)
AuxiliaryTaskGraphCommitResult = (
    commit_records.AuxiliaryTaskGraphCommitResult
)


def build_auxiliary_terminal_task_graph_validation_context(
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
) -> InSessionTaskGraphRevisionValidationContext:
    return validation_records.build_auxiliary_terminal_task_graph_validation_context(
        session_store.current_store_deps(),
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
    )


def project_auxiliary_terminal_semantic_support(
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
) -> AuxiliaryTerminalSemanticSupport:
    return validation_records.project_auxiliary_terminal_semantic_support(
        session_store.current_store_deps(),
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
    )


def seal_auxiliary_terminal_proposal(
    *,
    command: SealAuxiliaryTerminalProposalCommand,
) -> AuxiliaryTerminalSealResult:
    return seal_records.seal_auxiliary_terminal_proposal(
        session_store.current_store_deps(),
        command=command,
    )


def get_auxiliary_terminal_proposal_receipt(
    *,
    session_id: str,
    terminal_proposal_receipt_id: str,
) -> AuxiliaryTerminalProposalReceipt:
    return seal_records.get_auxiliary_terminal_proposal_receipt(
        session_store.current_store_deps(),
        session_id=session_id,
        terminal_proposal_receipt_id=terminal_proposal_receipt_id,
    )


def commit_auxiliary_task_graph_proposal(
    *,
    command: CommitAuxiliaryTaskGraphProposalCommand,
) -> AuxiliaryTaskGraphCommitResult:
    return commit_records.commit_auxiliary_task_graph_proposal(
        session_store.current_store_deps(),
        command=command,
    )


__all__ = [
    "AuxiliaryBaseNodeAliasBinding",
    "AuxiliaryTaskGraphCommitFailureCode",
    "AuxiliaryTaskGraphCommitIdentityCollision",
    "AuxiliaryTaskGraphCommitPersistenceError",
    "AuxiliaryTaskGraphCommitResult",
    "AuxiliaryTerminalProposalReceipt",
    "AuxiliaryTerminalSealIdentityCollision",
    "AuxiliaryTerminalSealPersistenceError",
    "AuxiliaryTerminalSealResult",
    "AuxiliaryTerminalSemanticSupport",
    "CommitAuxiliaryTaskGraphProposalCommand",
    'InSessionTaskGraphRevisionValidationContext',
    "SealAuxiliaryTerminalProposalCommand",
    "build_auxiliary_terminal_task_graph_validation_context",
    "commit_auxiliary_task_graph_proposal",
    "get_auxiliary_terminal_proposal_receipt",
    "project_auxiliary_terminal_semantic_support",
    "seal_auxiliary_terminal_proposal",
]
