from __future__ import annotations

import pytest

from personagraph.retrieval.contracts import SourceType
from personagraph.retrieval.orchestration.selection import (
    SourceQuotaPool,
    SourceSelectionQuotas,
    ordered_quota_pools,
    pool_for_source,
)


def test_source_quota_shape_has_three_pools_and_shares_user_task_memory():
    assert pool_for_source(SourceType.LONG_TERM_USER) is SourceQuotaPool.LONG_TERM_MEMORY
    assert pool_for_source(SourceType.LONG_TERM_TASK) is SourceQuotaPool.LONG_TERM_MEMORY
    assert ordered_quota_pools(
        (
            SourceType.CURRENT_SESSION,
            SourceType.LONG_TERM_USER,
            SourceType.LONG_TERM_TASK,
            SourceType.DOCUMENT,
        )
    ) == (
        SourceQuotaPool.CURRENT_SESSION,
        SourceQuotaPool.LONG_TERM_MEMORY,
        SourceQuotaPool.DOCUMENT,
    )


def test_source_selection_quotas_reject_zero_and_normalise_serialized_pool_names():
    quotas = SourceSelectionQuotas({"document": 2})
    assert quotas.enabled is True
    assert quotas.item_limit(SourceQuotaPool.DOCUMENT) == 2
    with pytest.raises(ValueError, match="greater than zero"):
        SourceSelectionQuotas({SourceQuotaPool.DOCUMENT: 0})
