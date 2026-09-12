"""Explicit Project document authority for Auxiliary controller integration tests."""

from dataclasses import replace

import pytest

from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.task_document_scope import (
    AuxiliaryTaskDocumentScope,
    prepare_auxiliary_task_document_scope,
)

from tests.documents._authority import bound_project_document_authority


@pytest.fixture(autouse=True)
def auxiliary_project_authority(tmp_path):
    """Keep synthetic source files and their current document database together."""
    with bound_project_document_authority(tmp_path, project_root=tmp_path) as authority:
        yield authority


def authorized_auxiliary_documents(
    *, session_id: str, turn_id: str, task_id: str, document_ids: tuple[str, ...]
) -> AuxiliaryTaskDocumentScope:
    """Supply exactly the synthetic documents authorized for this test's Task.

    The default application scope only recovers already persisted Task authority.
    Tests entering the application below Runtime Entry must explicitly provide the
    initial document scope, just as they provide the accepted Task execution lane.
    """
    allowed_ids = tuple(sorted(set(document_ids)))
    scope = prepare_auxiliary_task_document_scope(
        session_id=session_id, turn_id=turn_id, task_id=task_id
    )
    return replace(
        scope,
        allowed_managed_document_ids=allowed_ids,
        mounted_authority=freeze_mounted_document_planning_authority(
            session_id=session_id,
            task_id=task_id,
            allowed_managed_document_ids=allowed_ids,
        ),
    )
