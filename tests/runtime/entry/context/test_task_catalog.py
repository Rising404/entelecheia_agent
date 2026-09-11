from __future__ import annotations

import pytest

from personagraph.context_budget.token_counter import estimate_tokens
from personagraph.runtime.entry.context.task_catalog import (
    pack_insession_task_catalog,
    validate_catalog_task_ids,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskCatalogItem,
    InSessionTaskStatus,
)


def _item(task_id: str, summary: str) -> InSessionTaskCatalogItem:
    return InSessionTaskCatalogItem(
        insession_task_id=task_id,
        goal_summary=summary,
        status=InSessionTaskStatus.PROPOSED,
        current_graph_revision=1,
    )


def test_catalog_packer_keeps_order_and_makes_omission_explicit():
    first = _item("insession_task_one", "第一项任务")
    second = _item("insession_task_two", "第二项任务")
    budget = estimate_tokens(
        "insession_task_id=insession_task_one\n"
        "status=proposed\n"
        "graph_revision=1\n"
        "goal_summary=第一项任务"
    )

    catalog = pack_insession_task_catalog((first, second), token_budget=budget)

    assert catalog.items == (first,)
    assert catalog.truncated is True


def test_catalog_id_validator_accepts_only_distinct_visible_ids():
    first = _item("insession_task_one", "第一项任务")
    catalog = pack_insession_task_catalog((first,), token_budget=10_000)

    assert validate_catalog_task_ids(("insession_task_one",), catalog) == (
        "insession_task_one",
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_catalog_task_ids(("insession_task_one", "insession_task_one"), catalog)
    with pytest.raises(ValueError, match="outside"):
        validate_catalog_task_ids(("insession_task_missing",), catalog)


def test_catalog_packer_supports_a_root_shell_without_a_graph_revision():
    shell = InSessionTaskCatalogItem(
        insession_task_id="insession_task_shell",
        goal_summary="刚由入口识别、尚未建立任务图",
        status=InSessionTaskStatus.PROPOSED,
        current_graph_revision=None,
    )
    expected_text = (
        "insession_task_id=insession_task_shell\n"
        "status=proposed\n"
        "graph_revision=none\n"
        "goal_summary=刚由入口识别、尚未建立任务图"
    )

    catalog = pack_insession_task_catalog(
        (shell,), token_budget=estimate_tokens(expected_text)
    )

    assert catalog.items == (shell,)
    assert catalog.truncated is False


def test_catalog_packer_counts_pending_question_in_the_same_budget():
    item = InSessionTaskCatalogItem(
        insession_task_id="insession_task_waiting",
        goal_summary="安排旅行",
        status=InSessionTaskStatus.AWAITING_USER,
        current_graph_revision=1,
        pending_user_question="你希望哪一天出发？",
    )
    without_question = (
        "insession_task_id=insession_task_waiting\n"
        "status=awaiting_user\n"
        "graph_revision=1\n"
        "goal_summary=安排旅行"
    )

    catalog = pack_insession_task_catalog(
        (item,), token_budget=estimate_tokens(without_question)
    )

    assert catalog.items == ()
    assert catalog.truncated is True
