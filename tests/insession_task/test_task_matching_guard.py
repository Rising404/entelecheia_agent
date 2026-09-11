from __future__ import annotations

from hashlib import sha256

from personagraph.l2.task_graph import (
    InSessionTaskCatalogItem,
    InSessionTaskCatalog,
    InSessionTaskMatchGuardCode,
    InSessionTaskMatchesProposal,
    InSessionTaskMatchingLimits,
    guard_insession_task_matches,
)


def _catalog(*task_ids: str) -> InSessionTaskCatalog:
    return InSessionTaskCatalog(
        items=tuple(
            InSessionTaskCatalogItem(
                insession_task_id=task_id,
                goal_summary=f"{task_id} 的目标",
                status="active",
                current_graph_revision=1,
            )
            for task_id in task_ids
        )
    )


def _proposal(*matches: dict[str, object]) -> InSessionTaskMatchesProposal:
    return InSessionTaskMatchesProposal.model_validate(
        {"task_matches": list(matches)}
    )


def test_accepts_an_empty_task_match_batch():
    result = guard_insession_task_matches(
        InSessionTaskMatchesProposal(),
        authoritative_user_text="今天天气真好。",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "accepted"
    assert result.accepted_task_matches == ()
    assert result.error_codes == ()


def test_accepts_all_three_match_shapes_and_binds_exact_source_spans():
    user_text = "请新建上海旅行计划。继续论文摘要。并在论文任务下补充局限分析。"
    proposal = _proposal(
        {
            "match_type": "new_root",
            "local_key": "shanghai_trip",
            "title": "上海旅行计划",
            "objective": "制定上海旅行计划",
            "source_excerpt": "请新建上海旅行计划",
        },
        {
            "match_type": "existing_root",
            "insession_task_id": "task_paper",
            "source_excerpt": "继续论文摘要",
        },
        {
            "match_type": "existing_root_branch",
            "insession_task_id": "task_paper",
            "branch_key": "limitations",
            "branch_summary": "补充论文局限分析",
            "source_excerpt": "在论文任务下补充局限分析",
        },
    )

    result = guard_insession_task_matches(
        proposal,
        authoritative_user_text=user_text,
        trusted_root_catalog=_catalog("task_paper"),
    )

    assert result.status == "accepted"
    assert [item.proposal.match_type for item in result.accepted_task_matches] == [
        "new_root",
        "existing_root",
        "existing_root_branch",
    ]
    for item in result.accepted_task_matches:
        excerpt = item.proposal.source_excerpt
        span = item.source_span
        assert user_text[span.start : span.end] == excerpt
        assert span.text_sha256 == sha256(excerpt.encode("utf-8")).hexdigest()


def test_rejects_an_existing_task_id_outside_the_trusted_catalog():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "existing_root",
                "insession_task_id": "task_unknown",
                "source_excerpt": "继续之前的任务",
            }
        ),
        authoritative_user_text="继续之前的任务",
        trusted_root_catalog=_catalog("task_known"),
    )

    assert result.status == "rejected"
    assert result.accepted_task_matches == ()
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.UNKNOWN_INSESSION_TASK_ID,
    )


def test_accepts_explicit_existing_root_target_change_with_exact_source_binding():
    user_text = "请把论文任务的目标改成只比较实验设计与消融结果"
    excerpt = "目标改成只比较实验设计与消融结果"
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "existing_root_target_change",
                "insession_task_id": "task_paper",
                "replacement_objective": "比较论文的实验设计与消融结果",
                "source_excerpt": excerpt,
                "execute_current": True,
            }
        ),
        authoritative_user_text=user_text,
        trusted_root_catalog=_catalog("task_paper"),
    )

    assert result.status == "accepted"
    accepted = result.accepted_task_matches[0]
    assert accepted.proposal.match_type == "existing_root_target_change"
    assert accepted.proposal.replacement_objective == "比较论文的实验设计与消融结果"
    assert user_text[accepted.source_span.start : accepted.source_span.end] == excerpt
    assert accepted.source_span.text_sha256 == sha256(excerpt.encode()).hexdigest()


def test_rejects_target_change_for_task_outside_trusted_catalog():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "existing_root_target_change",
                "insession_task_id": "task_unknown",
                "replacement_objective": "替换后的目标",
                "source_excerpt": "把目标改成替换后的目标",
                "execute_current": True,
            }
        ),
        authoritative_user_text="把目标改成替换后的目标",
        trusted_root_catalog=_catalog("task_known"),
    )

    assert result.status == "rejected"
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.UNKNOWN_INSESSION_TASK_ID,
    )


def test_rejects_duplicate_local_keys_and_exact_duplicate_matches():
    duplicate = {
        "match_type": "new_root",
        "local_key": "trip",
        "title": "旅行",
        "objective": "制定旅行计划",
        "source_excerpt": "制定旅行计划",
    }
    result = guard_insession_task_matches(
        _proposal(duplicate, duplicate),
        authoritative_user_text="制定旅行计划",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "rejected"
    assert set(result.error_codes) == {
        InSessionTaskMatchGuardCode.DUPLICATE_LOCAL_KEY,
        InSessionTaskMatchGuardCode.DUPLICATE_TASK_MATCH,
    }


def test_temporary_keys_cannot_disguise_duplicate_root_or_branch_content():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "new_root",
                "local_key": "trip_one",
                "title": "旅行",
                "objective": "制定旅行计划",
                "source_excerpt": "制定旅行计划",
            },
            {
                "match_type": "new_root",
                "local_key": "trip_two",
                "title": "旅行",
                "objective": "制定旅行计划",
                "source_excerpt": "制定旅行计划",
            },
            {
                "match_type": "existing_root_branch",
                "insession_task_id": "task_paper",
                "branch_key": "limits_one",
                "branch_summary": "补充局限",
                "source_excerpt": "补充局限",
            },
            {
                "match_type": "existing_root_branch",
                "insession_task_id": "task_paper",
                "branch_key": "limits_two",
                "branch_summary": "补充局限",
                "source_excerpt": "补充局限",
            },
        ),
        authoritative_user_text="制定旅行计划，补充局限",
        trusted_root_catalog=_catalog("task_paper"),
    )

    assert result.status == "rejected"
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.DUPLICATE_TASK_MATCH,
    )


def test_rejects_more_new_roots_than_the_host_limit():
    matches = tuple(
        {
            "match_type": "new_root",
            "local_key": f"task_{index}",
            "title": f"任务 {index}",
            "objective": f"完成任务 {index}",
            "source_excerpt": f"任务{index}",
        }
        for index in range(4)
    )
    result = guard_insession_task_matches(
        _proposal(*matches),
        authoritative_user_text="任务0；任务1；任务2；任务3",
        trusted_root_catalog=_catalog(),
        limits=InSessionTaskMatchingLimits(max_new_root_tasks=3),
    )

    assert result.status == "rejected"
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.NEW_ROOT_LIMIT_EXCEEDED,
    )


def test_rejects_a_source_excerpt_missing_from_authoritative_user_text():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "new_root",
                "local_key": "trip",
                "title": "旅行",
                "objective": "制定旅行计划",
                "source_excerpt": "并不存在的原文",
            }
        ),
        authoritative_user_text="请制定旅行计划",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "rejected"
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.SOURCE_EXCERPT_NOT_FOUND,
    )


def test_rejects_an_excerpt_that_occurs_more_than_once_including_overlaps():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "new_root",
                "local_key": "letters",
                "title": "字母",
                "objective": "处理字母",
                "source_excerpt": "aa",
            }
        ),
        authoritative_user_text="aaa",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "rejected"
    assert result.error_codes == (
        InSessionTaskMatchGuardCode.SOURCE_EXCERPT_AMBIGUOUS,
    )


def test_rejects_whitespace_only_human_text_fields():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "new_root",
                "local_key": "blank_task",
                "title": "   ",
                "objective": "\t",
                "source_excerpt": " \n ",
            }
        ),
        authoritative_user_text=" \n ",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "rejected"
    assert set(result.error_codes) == {
        InSessionTaskMatchGuardCode.BLANK_OBJECTIVE,
        InSessionTaskMatchGuardCode.BLANK_SOURCE_EXCERPT,
        InSessionTaskMatchGuardCode.BLANK_TITLE,
    }


def test_rejects_blank_existing_id_and_branch_summary_without_calling_them_unknown():
    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "existing_root_branch",
                "insession_task_id": "   ",
                "branch_key": "new_branch",
                "branch_summary": "\t",
                "source_excerpt": "调整这个分支",
            }
        ),
        authoritative_user_text="调整这个分支",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "rejected"
    assert set(result.error_codes) == {
        InSessionTaskMatchGuardCode.BLANK_BRANCH_SUMMARY,
        InSessionTaskMatchGuardCode.BLANK_INSESSION_TASK_ID,
    }
    assert InSessionTaskMatchGuardCode.UNKNOWN_INSESSION_TASK_ID not in result.error_codes


def test_does_not_pretend_to_validate_natural_language_semantics():
    """语义分歧有意留给模型/评测层处理。"""

    result = guard_insession_task_matches(
        _proposal(
            {
                "match_type": "new_root",
                "local_key": "database_refactor",
                "title": "重构数据库",
                "objective": "重构数据库并迁移全部表",
                "source_excerpt": "今天天气真好",
            }
        ),
        authoritative_user_text="今天天气真好",
        trusted_root_catalog=_catalog(),
    )

    assert result.status == "accepted"
    assert result.error_codes == ()
