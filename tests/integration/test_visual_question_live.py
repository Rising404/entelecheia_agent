"""显式启用的真实视觉工具问答 smoke；默认不读取本机密钥、不联网。

只检查注册工具→真实 VLM→Project 问答记录的执行闭环，不把它称为自主 L1 或
DocBench 评分。PDF/图片及具体问题由调用者指定，不内置评测答案。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import pytest


@pytest.mark.skipif(
    os.environ.get("PERSONAGRAPH_RUN_VISUAL_QUESTION_LIVE") != "1",
    reason="explicit visual API opt-in required",
)
def test_live_visual_question_is_published_without_cross_call_reuse(
    monkeypatch, tmp_path, bound_partitioned_session,
):
    from personagraph.configuration import app_settings
    from personagraph.input_processing.vision.contracts import (
        VisionPurpose,
        normalize_vision_question,
    )
    from personagraph.input_processing.vision.providers.http import (
        HttpVisionModelAdapter,
        _post_json,
        load_provider_config,
    )
    from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
    from personagraph.tools.contracts import ExecutionStatus
    from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
    from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
    from personagraph.workspace.storage.context import require_current

    config_path = Path(os.environ["PERSONAGRAPH_VISUAL_QUESTION_LIVE_CONFIG"])
    source_path = Path(os.environ["PERSONAGRAPH_VISUAL_QUESTION_LIVE_SOURCE"])
    question = normalize_vision_question(
        VisionPurpose.QUESTION,
        os.environ["PERSONAGRAPH_VISUAL_QUESTION_LIVE_QUESTION"],
    )
    assert config_path.is_absolute() and config_path.is_file()
    assert source_path.is_absolute() and source_path.is_file()
    assert source_path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
    # 仅加载明确指定的视觉配置；创建测试 Session 时仍使用隔离配置，避免写入产品设置。
    with monkeypatch.context() as config_scope:
        config_scope.setattr(app_settings, "CONFIG_PATH", config_path)
        config = load_provider_config()
    if config is None:
        pytest.fail("explicit config has no complete vision provider settings")

    dispatches = []

    def transport(provider_config, body):
        text = body["messages"][0]["content"][0]["text"]
        assert question in text
        dispatches.append(time.monotonic())
        return _post_json(provider_config, body)

    adapter = HttpVisionModelAdapter(config, transport=transport)
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: adapter,
    )
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / source_path.name
    shutil.copyfile(source_path, target)
    session_id = bound_partitioned_session(working_dir=root)
    database = require_current()
    is_pdf = target.suffix.lower() == ".pdf"
    media_type = (
        "application/pdf" if is_pdf else
        "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    )
    WorkspaceFileAuthority(database).register_path(
        target.name, source=FileSource.USER_UPLOAD, media_type=media_type,
    )
    runtime = build_l1_tool_runtime(session_id)
    arguments = {
        "path": target.name,
        "purpose": "question",
        "question": question,
        "detail": "high",
        "region": "page",
    }
    if is_pdf:
        arguments["pages"] = [1]
    prepared = runtime.prepare(
        tool_id="analyze_pdf_page" if is_pdf else "analyze_image",
        arguments=arguments,
        remaining_tool_calls=3,
    )
    assert prepared.rejected_outcome is None
    assert prepared.protected_authority is not None
    assert prepared.protected_authority.revalidate()

    replies = []
    for call_id in ("live-question-1", "live-question-2", "live-question-1"):
        assert prepared.protected_authority.revalidate()
        result = ToolExecutor().execute(ResolvedInvocation(
            registration=prepared.registration,
            arguments=prepared.normalized_arguments,
            deadline_monotonic=time.monotonic() + 180,
            logical_tool_call_id=call_id,
        ))
        if result.status is not ExecutionStatus.SUCCEEDED:
            pytest.fail(f"visual tool failed: {result.error.code if result.error else result.status.value}")
        observation = result.result["observations"][0]
        assert observation["question"] == question
        assert observation["status"] == "completed"
        assert observation["observation"].strip()
        replies.append(observation["observation"])

    # 两次新调用均外发；第三次仅恢复第一次调用，不跨调用复用。
    assert len(dispatches) == 2
    with database.connect() as conn:
        rows = conn.execute(
            "SELECT question, text FROM picture_observations ORDER BY sequence"
        ).fetchall()
    assert len(rows) == 2
    assert [(row[0], row[1]) for row in rows] == [(question, reply) for reply in replies[:2]]
    assert replies[2] == replies[0]
    print(json.dumps({
        "question": question,
        "answers": replies[:2],
        "provider_dispatches": len(dispatches),
        "persisted_pairs": len(rows),
        "same_call_replay_without_dispatch": True,
    }, ensure_ascii=False))
