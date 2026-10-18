from __future__ import annotations

from personagraph.runtime.l1 import model as l1_model
from personagraph.runtime.l1 import semantic_verification as l1_verification


_WORKSPACE_TOOL_IDS = (
    "workspace_overview",
    "list_workspace_directory",
    "find_files",
    "search_text_files",
    "inspect_file",
    "read_text",
    "read_pdf_text",
    "read_word",
    "read_slides",
    "inspect_image",
)
_MOUNTED_DOCUMENT_TOOL_IDS = (
    "inspect_mounted_document",
    "search_mounted_document",
    "read_mounted_document_chunks",
    "read_mounted_visuals",
)
_FILE_TOOL_IDS = (
    "check_files_state",
    "prepare_files",
    "retrieve_files",
    "read_file_chunks",
    "list_file_visuals",
    "read_file_visuals",
)


def test_l1_guidance_is_a_neutral_assistant_role_not_an_internal_runtime_role() -> None:
    guidance = l1_model._L1_GUIDANCE

    assert l1_model._l1_system_prompt() == l1_model._L1_SYSTEM_PROMPT
    assert guidance.strip().startswith("你是负责完成当前用户请求的助手")
    # 角色约束只检查指导文字；公开 Schema 的类型名可以包含 Attempt。
    for internal_term in (
        "PersonaGraph",
        "TurnRun",
    ):
        assert internal_term not in guidance
    assert "tool_catalog" in guidance
    for tool_id in (*_WORKSPACE_TOOL_IDS, *_MOUNTED_DOCUMENT_TOOL_IDS, *_FILE_TOOL_IDS):
        assert tool_id not in guidance


def test_l1_prompt_uses_short_references_without_retired_model_bookkeeping() -> None:
    prompt = l1_model._L1_SYSTEM_PROMPT

    assert "call_ref" in prompt and "chunk_id" in prompt
    assert "acceptance_id" in prompt
    assert '"call_ref":"c1.1"' in prompt
    assert "JSON 顶层" in prompt
    assert "本次执行内固定" in prompt
    assert "其他会话或历史 Turn 中的同名短编号" in prompt
    for retired_field in (
        "tool_result_id",
        "tool_call_id",
        "result_sha256",
        "acceptance_progress",
        "scope_keys",
        "execution_notes",
        "result_ref",
        "supporting_tool_result_ids",
    ):
        assert retired_field not in prompt


def test_l1_guidance_keeps_execution_and_source_boundaries_explicit() -> None:
    guidance = l1_model._L1_GUIDANCE

    assert "finalization_required=true" in guidance and "submit_final_reply" in guidance
    assert "max_tool_calls_this_attempt" in guidance
    assert "每步必须写 note" in guidance and "Host" in guidance
    assert "attachments" in guidance and "不提供正文" in guidance
    assert "_terminal" in guidance and "原样重试" in guidance
    assert "历史回读" in guidance and "不必重新执行" in guidance


def test_l1_guidance_separates_file_readiness_from_reading_and_reported_failure() -> None:
    # Prompt wording is a regression boundary, not evidence of live model compliance.
    guidance = l1_model._L1_GUIDANCE
    assert "status=ready 不等于正文已读" in guidance
    assert "未尝试读取不等于权限拒绝" in guidance
    assert "对应文件或能力的工具错误" in guidance
    assert "笔记不是事实证据" in guidance
    assert "尚未取得足够正文" in guidance


def test_l1_guidance_explains_working_notes_and_result_eviction() -> None:
    # 验证真实投影策略与提示一致，不把文字断言当成模型必然遵从的证明。
    guidance = l1_model._L1_GUIDANCE
    assert "只展示紧邻前一步" in guidance
    assert "具体待办问题" in guidance
    assert "已就绪的文件/能力" in guidance
    assert "call_ref" in guidance
    assert "未确认结果结构时从 /result" in guidance


def test_l1_prompts_do_not_include_benchmark_answers_or_task_specific_strategies() -> (
    None
):
    prompts = (
        l1_model._L1_SYSTEM_PROMPT,
        l1_verification._L1_SEMANTIC_SYSTEM_PROMPT,
    )

    for prompt in prompts:
        for sample_strategy in (
            "Oracle KGLM",
            "65%",
            "X/Y",
            "numerator/denominator",
            "@1/@5",
            "对象内条目的数量",
            "穷举正文中每一个数字",
            "补读相邻文本",
        ):
            assert sample_strategy not in prompt
