"""保留下来的工作区文档控制器保持其 API 门面完整。"""

from __future__ import annotations

import pytest

from personagraph.api import service
from personagraph.api.service import workspace_documents


_DOMAIN_OPERATIONS = (
    (
        workspace_documents,
        (
            "list_documents",
            "get_document",
            "patch_document",
            "delete_document",
            "detach_document",
        ),
    ),
)


@pytest.mark.parametrize(
    ("controller", "operation"),
    [
        (controller, operation)
        for controller, operations in _DOMAIN_OPERATIONS
        for operation in operations
    ],
)
def test_facades_reexport_workspace_domain_operations(controller, operation):
    implementation = getattr(controller, operation)
    assert operation in service.__all__
    assert getattr(service, operation) is implementation
